#!/usr/bin/env bash
# ===== dcu-toolkit-open 多框架一键安装（开源版，幂等）=====
#
# 本脚本是 dcu-toolkit-open 的开源版安装器，支持多框架（multi-framework）通用接入：
#   - 支持 --detect（默认）与 --tool <name>
#   - --detect：自动扫描用户主目录/项目下的 agent 配置目录与 PATH 上的 agent 二进制，
#               列出并安装所有检测到的 agent。
#   - --tool <name>：仅安装到指定工具（如 claude / codex / opencode ...）。
#   - 对每个检测到的 agent 三层写入（见下），并写 dcu-toolkit-installed.json 幂等登记表。
#   - 兼容原有用法：bash install.sh [level] [tool] [install_path]（level=project|global）。
#
# 线程说明：本 install.sh 为"开源版"安装线程，供公开仓库使用；内部能力不在此处。
#
# 三层写入（对每个检测到的 agent）：
#   ① 通用底稿：把 Plugins/*/skills 的 SKILL.md 拷/链到该 agent 的 skills 目录，
#      （或统一 ~/.dcu-toolkit/skills 共享目录），并在项目根写版本固定的 AGENTS.md。
#   ② 标准接入：若该 agent 认 Claude plugin marketplace / agentskills.io，
#      直接提示 npx skills add / marketplace add（零代码）。
#   ③ 薄适配器：仅对需要特定角色格式的做最小转换（如 codex 的 agent 要 TOML，
#      可从 Claude .md 生成；hooks 不支持的注明降级）。
#
# 用法示例：
#   bash install.sh                 # 默认 --detect，扫描本机并安装到全部已检测到的 agent
#   bash install.sh --detect        # 同上，并打印检测结果
#   bash install.sh --tool codex    # 仅安装到 codex
#   bash install.sh project claude  # 兼容旧用法：项目级装到 .claude/skills/
#   bash install.sh global opencode # 兼容旧用法：用户级装到 ~/.config/opencode/skills/
#   bash install.sh --list          # 只列出检测到的 agent，不安装
# 用户知识库（沉淀落点，与 skill 树解耦）：
#   - 沉淀记录写到 ~/.config/dcu-toolkit/kb/（op-opt/issue/solution），
#     由 dcu-orchestrator 各类工作流收尾沉淀；install.sh 只创建、永不删除或覆盖。
#   - 为什么不在 <skills根>/ 里：skill 树是厂商维护物，重装/升级会以包内版本为准，
#     写在里面的用户记录会被覆盖。kb/ 是用户态目录，重装不动。
#   - 增量同步（清单式）：<skills根> 与 _dcu-toolkit-shared/ 的同步按清单逐文件比对，
#     只删「上次自己装的 + 本轮包内已移除 + 本地未改过」的文件；用户新增文件永不删除，
#     用户改过的厂商文件被覆盖前先备份 <file>.local.bak。清单在 ~/.config/dcu-toolkit/manifests/。
set -euo pipefail

NAME="dcu-toolkit-open"
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 颜色输出 ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; DIM=''; NC=''
fi
ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC} $*" >&2; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }

[ -d "$SRC_ROOT/Plugins" ] || { err "未找到 Plugins/ 目录，请在仓库根运行"; exit 1; }

# --- 工具版本（AGENTS.md 版本固定用）---
TOOLKIT_VERSION="1.6.0"
TOOLKIT_VERSION_FILE="$SRC_ROOT/.claude-plugin/marketplace.json"
if [ -f "$TOOLKIT_VERSION_FILE" ]; then
  TOOLKIT_VERSION="$(grep -m1 '"version"' "$TOOLKIT_VERSION_FILE" | sed 's/[^0-9.]//g' | cut -d. -f1-3)"
fi

# ===== 检测配置：agent 名 → 配置目录 / PATH 二进制 / skills 目录 / 角色格式 =====
# 每个 agent 定义：
#   cfg_dirs   : 用户主目录/项目下可能出现的配置目录（空格分隔）
#   bins       : PATH 上的二进制（检测到任一即视为该 agent 存在）
#   skills_dir : 全局（用户级）skills 目录
#   proj_skills_dir : 项目级 skills 目录（相对项目根）
#   role_format: 该 agent 认可的角色/指令文件格式（md | toml | jsonl | none）
#   marketplace: 是否认 Claude plugin marketplace / agentskills.io（yes|no）
# ===== 运行环境守卫（跨平台）=====
# 本脚本用 bash 关联数组（declare -A，需 bash 4+）。macOS 自带 bash 3.2 不满足，先探测。
if [ -z "${BASH_VERSION:-}" ] || [ "$(printf '%s' "${BASH_VERSION%%.*}" 2>/dev/null || echo 0)" -lt 4 ]; then
  echo "错误：本脚本需要 bash 4+（declare -A 关联数组）。macOS 请用 brew install bash 后以 /usr/local/bin/bash 运行。" >&2
  exit 1
fi
# 可移植 sed -i：GNU（Linux）与 BSD（macOS）语法不同
sed_i() {
  if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi
}
declare -A CFG_DIRS BINS SKILLS_DIR PROJ_SKILLS_DIR ROLE_FMT MKT
CFG_DIRS=(
  [claude]=".claude"
  [opencode]=".opencode"
  [codex]=".codex"
  [cursor]=".cursor"
  [gemini]=".gemini"
  [dsh]=".dsh"
  [trae]=".trae"
  [zed]=".zed"
  [windsurf]=".windsurf"
  [workbuddy]=".workbuddy"
  [aider]=".aider"
  [roo]=".roo"
  [clinerules]=".clinerules"
)
BINS=(
  [claude]="claude"
  [opencode]="opencode"
  [codex]="codex"
  [cursor]="cursor-agent"
  [gemini]="gemini"
  [dsh]="dsh"
  [trae]="trae"
  [zed]="zed"
  [windsurf]="windsurf"
  [aider]="aider"
  [roo]="roo"
  [qwen]="qwen"
  [kimi]="kimi"
  [clinerules]="clinerules"
)
SKILLS_DIR=(
  [claude]="$HOME/.claude/skills"
  [opencode]="$HOME/.config/opencode/skills"
  [codex]="$HOME/.codex/skills"
  [cursor]="$HOME/.cursor/skills"
  [gemini]="$HOME/.gemini/skills"
  [dsh]="$HOME/.dsh/skills"
  [trae]="$HOME/.trae/skills"
  [zed]="$HOME/.zed/skills"
  [windsurf]="$HOME/.windsurf/skills"
  [workbuddy]="$HOME/.workbuddy/skills"
  [aider]="$HOME/.aider/skills"
  [roo]="$HOME/.roo/skills"
  [qwen]="$HOME/.qwen/skills"
  [kimi]="$HOME/.kimi/skills"
  [clinerules]="$HOME/.clinerules/skills"
)
PROJ_SKILLS_DIR=(
  [claude]=".claude/skills"
  [opencode]=".opencode/skills"
  [codex]=".codex/skills"
  [cursor]=".cursor/skills"
  [gemini]=".gemini/skills"
  [dsh]=".dsh/skills"
  [trae]=".trae/skills"
  [zed]=".zed/skills"
  [windsurf]=".windsurf/skills"
  [workbuddy]=".workbuddy/skills"
  [aider]=".aider/skills"
  [roo]=".roo/skills"
  [qwen]=".qwen/skills"
  [kimi]=".kimi/skills"
  [clinerules]=".clinerules/skills"
)
ROLE_FMT=(
  [claude]="md"
  [opencode]="md"
  [codex]="toml"
  [cursor]="md"
  [gemini]="md"
  [dsh]="md"
  [trae]="md"
  [zed]="md"
  [windsurf]="md"
  [workbuddy]="md"
  [aider]="md"
  [roo]="md"
  [qwen]="md"
  [kimi]="md"
  [clinerules]="md"
)
MKT=(
  [claude]="yes"
  [opencode]="yes"
  [codex]="no"
  [cursor]="no"
  [gemini]="no"
  [dsh]="no"
  [trae]="no"
  [zed]="no"
  [windsurf]="no"
  [workbuddy]="no"
  [aider]="no"
  [roo]="no"
  [qwen]="no"
  [kimi]="no"
  [clinerules]="no"
)

# ===== 登记表路径（幂等）=====
INSTALLED_JSON="$HOME/.config/dcu-toolkit/dcu-toolkit-installed.json"
CONF_DIR="$HOME/.config/dcu-toolkit"
SHARED_SKILLS="$HOME/.dcu-toolkit/skills"

# ===== 工具函数 =====
# 在 PATH 上找二进制
have_bin() { command -v "$1" >/dev/null 2>&1; }

# 检测某个 agent 是否"已安装/存在"（配置目录存在 或 二进制在 PATH 上）
agent_present() {
  local agent="$1"
  local dirs="${CFG_DIRS[$agent]:-}"
  local bins="${BINS[$agent]:-}"
  local d
  for d in $dirs; do
    if [ -d "$HOME/$d" ] || [ -d "$INSTALL_PATH/$d" ]; then return 0; fi
  done
  local b
  for b in $bins; do
    if have_bin "$b"; then return 0; fi
  done
  return 1
}

# 写项目根 AGENTS.md（版本固定，指向 Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md + 域 skill）
write_agents_md() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
  cat > "$target" <<'EOF'
# AGENTS.md — 跨 Agent 通用入口（dcu-toolkit-open）

> 版本：@TOOLKIT_VERSION@（与仓库 .claude-plugin/marketplace.json 保持一致）
> 本文件是跨所有 agent（Claude / Codex / OpenCode / Cursor / Gemini / dsh …）的通用入口，
> 让任意 agent 在本仓库/项目里都知道如何路由到 DCU 领域 skill。
> 安装器（install.sh）也会在项目根生成同款文件（版本固定，指向 Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md + 各域 skill）。

## 我是谁 / 这里有啥

本仓库是海光 DCU（Hygon DCU）测试调优技能集开源版（dcu-toolkit-open）。
覆盖基础环境、性能剖析、推理框架部署适配、算子开发、算子卡、故障排查、工程实践等域。
技能按 Claude Code / OpenCode 插件包结构组织：`Plugins/<domain>/skills/<skill>/SKILL.md`。

## 入口：先读 dcu-orchestrator

- 总编排入口与工作流契约：[`Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md`](Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md)（dcu-orchestrator，中文）
- 多任务编排（五条内置工作流：推理优化 env-check→baseline→quantize、模型部署 precheck→launch→verify、
  故障排查 collect→locate→fix、算子优化 profile→optimize→validate→distill、调研复现 research-collect→env-reproduce(ask)→baseline-run），读
  [`Plugins/dcu-core/skills/dcu-orchestrator/workflows/*.yaml`](Plugins/dcu-core/skills/dcu-orchestrator/workflows/)
  与 `tasks/*.yaml`（任务契约字段说明见 `Plugins/dcu-core/skills/dcu-orchestrator/README.md`）。
- 编排辅助脚本：`render_workflow.py`（渲染就绪/依赖/执行顺序）、`verify_evidence.py`（校验 evidence 契约）、
  `dcu_distill.py`（把一次 run 的产物转成知识库台账 + 机器可读证据 JSON + 晋升提案）。

## 沉淀 / 知识库（用户态，重要）

- 沉淀写**用户知识库** `~/.config/dcu-toolkit/kb/`：`op-opt/` 算子优化台账由 `op-optimize` 工作流收尾的
  `op-distill` 节点自动写入（`dcu_distill.py` 生成台账 + 证据 JSON）；`issue/` 故障记录、`solution/` 部署方案卡片
  **暂无自动节点**，链路收尾时按包内 `data/kb/README.md` 的约定人工补写（模板见 `data/kb/templates/`）。
- **不要往 skill 树里写沉淀**：`<skills根>/` 与 `<skills根>/_dcu-toolkit-shared/` 是厂商内容，重装/升级以包内版本为准。
  install.sh 的清单式增量同步（清单在 `~/.config/dcu-toolkit/manifests/`）只删「上次装的 + 本轮包内已移除 + 本地未改过」的文件，
  用户新增文件永不删、本地改过的厂商文件被覆盖前备份为 `<file>`.local.bak —— 但记录类内容的正确落点仍是 `kb/`。
- 纪律：每条事实带 `source` + `verified_on`（日期/设备/gfx/DTK）；未验证结论标 `unverified`；
  性能数字标 `[历史数据，随硬件/DTK/版本变化]`，且不进可复用事实条目。
- 晋升：候选事实由 `op-distill` 产出晋升提案，**经人工/厂商审核**后才并入 skill（不自动改写厂商内容）。

## 领域 skill 路由（按意图）

| 意图关键词 | 首选 skill | 域 |
|---|---|---|
| 环境检测 / DTK / 驱动 / 容器 / hy-smi / NUMA | dcu-env、dcu-dtk、dcu-driver、dcu-hy-smi、dcu-cpu-affinity、dcu-docker、dcu-topology | env |
| 性能剖析 / 算子优化 / 推理优化 / benchmark | dcu-perf-analysis、dcu-op-optimization、dcu-inference-optimization、dcu-llm-benchmark、dcu-perf-profiling | perf |
| vLLM 部署 / 精度 / 自定义算子 / trace | dcu-vllm-deploy、dcu-vllm-precision、dcu-vllm-custom-op、dcu-vllm-trace-analysis、dcu-vllm-runtime-patch | inference |
| 模型部署 / 可行性 / 配方目录 | dcu-model-deploy-verify、dcu-model-deploy-catalog、dcu-model-portability | inference |
| CUDA→HIP 移植 / kernel 开发 / Triton | dcu-cuda-hip-port、dcu-kernel-dev、dcu-triton-kernels、dcu-kernelwiki | operators |
| 故障排查 / 精度 / crash / hang | dcu-vmfault-coredump、debug-cuda-crash、debug-distributed-hang、dcu-vllm-precision、dcu-train-precision | troubleshooting |
| 工程实践 / CI / 代码规范 / cookbook | ci-workflow-guide、env-var-conventions、cookbook-*、model-pr-* | engineering |
| 算子卡 / 某算子有哪几种实现 / 哪个档位用哪个 / kernel 改动算不算通过 | op-sparse-mla、op-moe-w8a8、op-gemm-fp8、op-contract、op-bench、op-gate | ops |
| 全量路由索引 | dcu-skill-router（dcu-core） | core |

> 完整路由表见仓库根 `SKILL.md`（包级总入口）与 `dcu-skill-router`。

## 安装 / 接入

- 直接调用 `bash install.sh` 即可把全部 skills 装入本机各 agent 的 skills 目录。
- `--detect`（默认）自动扫描本机 agent 并逐层写入；`--tool <name>` 只装指定 agent；`--list` 只列出检测结果。
- 幂等：重装不会重复拷贝内容一致的 skill；登记表在 `~/.config/dcu-toolkit/dcu-toolkit-installed.json`。

## 安全

- 包内零密码、零密钥、零内网地址；真实凭证在 `~/.config/dcu-toolkit/credentials.json`（用户侧自行填写，chmod 600）。
- 远程密码认证用 `sshpass -f <文件>` 或 `sshpass -e`，禁用 `sshpass -p`。
- 发布前跑 `python3 sanitize.py <repo>` 断言零内部残留。
EOF
  # 替换版本占位符
  sed_i "s/@TOOLKIT_VERSION@/${TOOLKIT_VERSION}/g" "$target"
}

# ===== 增量同步（清单式）=====
# 取代旧「整体 rm -rf + cp -a」语义：重装/升级不再抹掉用户本地新增或改过的文件。
#   - 源里有、目标缺 → 新增；源里有、目标内容不同 → 覆盖（厂商内容以源为准）；
#     若该文件上次安装后是本地改过的，先备份 <file>.local.bak 再覆盖并告警。
#   - 上次装过、本轮源里已移除、且本地未改过 → 删除。
#   - 用户自己新增的文件（不在清单里）→ 永不删除。
# 清单目录：~/.config/dcu-toolkit/manifests/<目标路径 sha256 前 16 位>.tsv
# 清单行格式：<相对路径>\t<sha256>
MANIFEST_DIR="$CONF_DIR/manifests"

_sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$1" | awk '{print $NF}'
  else
    cksum "$1" | cut -d' ' -f1
  fi
}

# 目标路径 → 该目标的清单文件路径
_manifest_for() {
  local key
  key="$(printf '%s' "$1" | sha256sum | cut -c1-16)"
  printf '%s/%s.tsv' "$MANIFEST_DIR" "$key"
}

# 装进 skills 目录的是「知识 + 脚本」，不是本地运行产物。
# 构建物（.so/.o/.a）、本地产出的备份（.bak）与 runs/ 只属于开发机：
# 装过去既会把每次 reinstall 的增量同步拖慢，也会让 agent 读到过期的实验目录。
# evals/ 同属开发侧资产：动态评测要跑包根 scripts/skill_eval.py，装到 agent 目录里用不上。
# 全局 apply 到目录与文件两条路径（目录被跳过，其下文件自然也不进新清单）。
_skip_rel() {
  case "$1" in
    __pycache__|*/__pycache__|*/__pycache__/*|*.pyc) return 0 ;;
    runs|runs/*|*/runs|*/runs/*) return 0 ;;
    evals|*/evals|*/evals/*) return 0 ;;
    *.so|*.so.*|*.o|*.a|*.bak) return 0 ;;
  esac
  return 1
}

# sync_tree <src_dir> <dest_dir> [manifest_file]
sync_tree() {
  # 去掉调用方可能带的结尾斜杠（否则下面的前缀裁剪会失效，导致一个文件都不拷）
  local src="${1%/}" dest="${2%/}" man="${3:-}"
  [ -n "$man" ] || man="$(_manifest_for "$dest")"
  mkdir -p "$dest" "$MANIFEST_DIR"
  local new_man="$man.new.$$"
  : > "$new_man"

  # ① 目录：先建出结构
  local d rel
  while IFS= read -r -d '' d; do
    rel="${d#"$src"/}"
    [ "$rel" = "$d" ] && continue
    _skip_rel "$rel" && continue
    mkdir -p "$dest/$rel"
  done < <(find "$src" -type d -print0 2>/dev/null)

  # ② 文件：源里有 → 新增/更新（内容一致则跳过），并记入新清单
  local f prev cur
  while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    if [ "$rel" = "$f" ]; then
      err "增量同步：相对路径裁剪失败（src=$src, file=$f）—— 中止安装以免装出空目录"
      return 1
    fi
    _skip_rel "$rel" && continue
    mkdir -p "$(dirname "$dest/$rel")"
    if [ -f "$dest/$rel" ] && cmp -s "$f" "$dest/$rel"; then
      :
    else
      if [ -f "$dest/$rel" ]; then
        prev="$(awk -F'\t' -v r="$rel" '$1==r {print $2; exit}' "$man" 2>/dev/null || true)"
        cur="$(_sha_of "$dest/$rel")"
        if [ -n "$prev" ] && [ "$prev" != "$cur" ]; then
          cp -a "$dest/$rel" "$dest/$rel.local.bak" 2>/dev/null || true
          warn "本地修改被厂商版本覆盖（已备份 .local.bak）：${dest#$HOME/}/$rel"
        fi
      fi
      cp -a "$f" "$dest/$rel"
    fi
    printf '%s\t%s\n' "$rel" "$(_sha_of "$dest/$rel")" >> "$new_man"
  done < <(find "$src" -type f -print0 2>/dev/null)

  # ③ 删除：上次装过 + 本轮源里没有 + 本地未改过 → 删；本地改过的保留并告警
  if [ -f "$man" ]; then
    local rel2 old
    while IFS=$'\t' read -r rel2 old; do
      [ -n "$rel2" ] || continue
      # 源里没了，或本轮起被 _skip_rel 排除（构建物/runs/），都按「上游已移除」处理
      if [ -e "$src/$rel2" ] && ! _skip_rel "$rel2"; then continue; fi
      [ -f "$dest/$rel2" ] || continue
      if [ "$(_sha_of "$dest/$rel2")" = "$old" ]; then
        rm -f "$dest/$rel2"
      else
        warn "上游已移除但本地改过，保留：${dest#$HOME/}/$rel2"
      fi
    done < "$man"
  fi

  mv -f "$new_man" "$man"
  # 清掉历史安装留下的 python 缓存（不参与同步，也不进清单）
  find "$dest" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  # 早版本把 runs/ 一起装过；现在不同步了，只清空目录（里面有用户跑出来的数据就不动）
  find "$dest" -type d -path '*/runs/*' -empty -delete 2>/dev/null || true
  find "$dest" -type d -name runs -empty -delete 2>/dev/null || true
}

# 拷贝/同步一个 skill 目录到目标（薄包装：走清单式 sync_tree）
install_one_skill() {
  sync_tree "$1" "$2"
}


# ===== 解析参数 =====
# 支持：
#   新旗标：--detect（默认）| --tool <name> | --list | --help
#   旧式位置参数： [level] [tool] [install_path]   （level=project|global；tool=claude|opencode|...）
#   例：bash install.sh project claude          # 项目级装到 .claude/skills/
#       bash install.sh global opencode         # 用户级装到 ~/.config/opencode/skills/
#       bash install.sh --tool codex            # 仅装到 codex（新旗标）
LEVEL="project"
INSTALL_PATH="."
TOOL=""
MODE="detect"     # detect | tool
LIST_ONLY=0
LEGACY_POS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --detect) MODE="detect"; shift ;;
    --tool)
      MODE="tool"; shift
      [ $# -gt 0 ] || { err "--tool 需要一个参数，如 --tool codex"; exit 1; }
      TOOL="$1"; shift ;;
    --list) LIST_ONLY=1; shift ;;
    --help|-h)
      echo "用法: bash install.sh [--detect|--tool <name>|--list] [project|global] [tool] [install_path]"
      echo "  --detect      自动扫描并安装到所有检测到的 agent（默认）"
      echo "  --tool <name> 仅安装到指定 agent"
      echo "  --list        只列出检测到的 agent，不安装"
      echo "  project|global 安装级别（默认 project，兼容旧用法）"
      echo "  tool          旧式：安装目标（默认 claude，兼容旧用法）"
      echo "  install_path  项目级目标路径（默认当前目录）"
      exit 0 ;;
    --*)
      err "未知选项: $1"; exit 1 ;;
    *)
      LEGACY_POS+=("$1"); shift ;;
  esac
done

# 旧式位置参数： [level] [tool] [install_path]
if [ ${#LEGACY_POS[@]} -ge 1 ]; then
  if [ "${LEGACY_POS[0]}" = "project" ] || [ "${LEGACY_POS[0]}" = "global" ]; then
    LEVEL="${LEGACY_POS[0]}"
    if [ ${#LEGACY_POS[@]} -ge 2 ]; then
      # 第 2 个位置参数 = tool（兼容旧用法）
      if [ -z "$TOOL" ]; then TOOL="${LEGACY_POS[1]}"; MODE="tool"; fi
      if [ ${#LEGACY_POS[@]} -ge 3 ]; then INSTALL_PATH="${LEGACY_POS[2]}"; fi
    fi
  else
    # 第 1 个位置参数不是 level（仅传 install_path 或 tool）
    if [ ${#LEGACY_POS[@]} -eq 1 ]; then
      INSTALL_PATH="${LEGACY_POS[0]}"
    elif [ ${#LEGACY_POS[@]} -eq 2 ]; then
      # 视为 [tool] [install_path]（默认 project 级）
      if [ -z "$TOOL" ]; then TOOL="${LEGACY_POS[0]}"; MODE="tool"; fi
      INSTALL_PATH="${LEGACY_POS[1]}"
    else
      err "参数过多"; exit 1
    fi
  fi
fi

case "$LEVEL" in project|global) ;; *) err "level 仅支持 project | global"; exit 1 ;; esac
# ===== 共享 skills 目录（统一底稿）=====
mkdir -p "$SHARED_SKILLS"

echo ""
info "dcu-toolkit-open 安装器（开源版，多框架）"
echo "  等级: $LEVEL | 模式: $MODE | 项目路径: $INSTALL_PATH"

# ===== 模式一：仅列出检测到的 agent =====
if [ $LIST_ONLY -eq 1 ]; then
  echo ""
  echo "== 检测到的 agent =="
  for agent in claude opencode codex cursor gemini dsh trae zed windsurf workbuddy aider roo qwen kimi clinerules; do
    if agent_present "$agent"; then
      echo "  ✓ $agent  (配置目录: ${CFG_DIRS[$agent]:-} / 二进制: ${BINS[$agent]:-})"
    fi
  done
  echo ""
  echo "注：未安装任何东西。用 --tool 指定或去掉 --list 自动安装。"
  exit 0
fi

# ===== 确定要安装的 agent 列表 =====
AGENTS=()
if [ "$MODE" = "tool" ]; then
  AGENTS=("$TOOL")
else
  for agent in claude opencode codex cursor gemini dsh trae zed windsurf workbuddy aider roo qwen kimi clinerules; do
    if agent_present "$agent"; then AGENTS+=("$agent"); fi
  done
  if [ ${#AGENTS[@]} -eq 0 ]; then
    warn "未检测到任何已安装的 agent 配置或二进制。"
    warn "将按默认安装到 .claude/（以备后续使用）。"
    AGENTS=("claude")
  fi
fi

echo ""
echo "== 将安装到以下 agent =="
for a in "${AGENTS[@]}"; do echo "  - $a"; done

# ===== 对每个 agent 执行三层写入 =====
TOTAL=0
declare -A INSTALLED_STATE
declare -A INSTALLED_COUNT

for agent in "${AGENTS[@]}"; do
  echo ""
  echo "===== 处理 agent: $agent ====="
  # 确定目标 skills 目录
  if [ "$LEVEL" = "global" ]; then
    target_root="$(eval echo ${SKILLS_DIR[$agent]:-})"
  else
    target_root="$INSTALL_PATH/${PROJ_SKILLS_DIR[$agent]:-}"
  fi
  # 若无映射，回退到共享目录
  if [ -z "$target_root" ]; then
    target_root="$SHARED_SKILLS"
    warn "无 $agent 的标准 skills 目录，使用共享目录 $SHARED_SKILLS"
  fi

  # --- ① 通用底稿：拷贝全部 skill 到该 agent 的 skills 目录 ---
  info "① 通用底稿 → $target_root"
  n=0
  for plugin in "$SRC_ROOT"/Plugins/*/; do
    [ -d "$plugin/skills" ] || continue
    for skill_dir in "$plugin"/skills/*/; do
      [ -f "$skill_dir/SKILL.md" ] || continue
      sname="$(basename "$skill_dir")"
      install_one_skill "$skill_dir" "$target_root/$sname"
      n=$((n+1))
    done
  done
  TOTAL=$((TOTAL+n))
  ok "已安装/同步 $n 个 skill 到 $target_root"

  # --- ①b 共享层：包根 data/references/scripts/docs 同步到 <skills根>/_dcu-toolkit-shared/ ---
  # 部分 skill 以相对路径引用包根共享层（包根 references/hygon-upstream/、references/dcu-models.md、
  # scripts/hygon-upstream-sync.py 等）。安装态下这些路径解析到 _dcu-toolkit-shared/ 同名目录。
  SHARED_DEST="$target_root/_dcu-toolkit-shared"
  mkdir -p "$SHARED_DEST"
  for shared_dir in data references scripts docs; do
    [ -d "$SRC_ROOT/$shared_dir" ] || continue
    # 清单式增量同步：厂商文件更新/新增照做，用户自己加的文件不动
    sync_tree "$SRC_ROOT/$shared_dir" "$SHARED_DEST/$shared_dir"
  done
  ok "共享层已增量同步 → $SHARED_DEST（data/references/scripts/docs）"

  # --- 项目根 AGENTS.md（版本固定）---
  if [ "$LEVEL" = "project" ] && [ -d "$INSTALL_PATH" ]; then
    write_agents_md "$INSTALL_PATH/AGENTS.md"
    ok "写 AGENTS.md → $INSTALL_PATH/AGENTS.md"
  fi

  # --- ② 标准接入：若认 marketplace / agentskills.io，零代码提示 ---
  if [ "${MKT[$agent]:-no}" = "yes" ]; then
    info "② 标准接入：$agent 认 Claude plugin marketplace / agentskills.io，可零代码接入："
    if [ "$agent" = "claude" ]; then
      info "     claude plugins marketplace add <repo-url>"
      info "     claude plugins install dcu-perf --marketplace dcu-toolkit-marketplace"
    elif [ "$agent" = "opencode" ]; then
      info "     opencode 支持 marketplace 插件（见仓库 .claude-plugin/marketplace.json），用 opencode 插件命令安装"
    else
      info "     npx skills add <repo-url>  # agentskills.io 通道"
    fi
  else
    info "② 标准接入：$agent 未标准识别 Claude marketplace，跳过（使用①底稿即可）。"
  fi

  # --- ③ 薄适配器：仅对需要特定角色格式做最小转换 ---
  fmt="${ROLE_FMT[$agent]:-md}"
  if [ "$fmt" != "md" ]; then
    info "③ 薄适配器：$agent 的角色格式为 $fmt，从 Claude .md 生成最小适配。"
    if [ "$agent" = "codex" ]; then
      # codex 需要 TOML 的 agent 配置。此处做最小占位生成（不覆盖用户已有配置）。
      if [ "$LEVEL" = "global" ]; then
        codex_global="$HOME/.codex/config.toml"
      else
        codex_global="$INSTALL_PATH/.codex/config.toml"
      fi
      if [ -f "$codex_global" ]; then
        info "     codex 已有 config.toml，跳过生成（避免覆盖）"
      else
        mkdir -p "$(dirname "$codex_global")"
        cat > "$codex_global" <<'TOML'
# 由 dcu-toolkit-open install.sh 生成的最小 codex 引导。
# 作用：让 codex 在 AGENTS.md 中读到总入口，进而路由到各域 skill。
# 若已有配置请自行合并，勿整体覆盖。
model = "auto"
project_agreements = "AGENTS.md"
TOML
        ok "     codex 最小引导 config.toml → $codex_global"
      fi
    fi
    if [ "$fmt" = "jsonl" ]; then
      info "     $agent 角色格式为 jsonl，暂不自动生成，请参考仓库 SKILL.md 手写转换。"
    fi
  else
    info "③ 薄适配器：$agent 使用 md/dir 角色格式，与①底稿兼容，无需转换。"
  fi

  INSTALLED_STATE["$agent"]="ok"
  INSTALLED_COUNT["$agent"]="$n"   # 记录该 agent 实际安装的 skill 数
done


# ===== 用户知识库根（install.sh 只创建，永不删除/覆盖；与 skill 树解耦）=====
KB_ROOT="$CONF_DIR/kb"
mkdir -p "$KB_ROOT/op-opt" "$KB_ROOT/issue" "$KB_ROOT/solution" 
if [ ! -f "$KB_ROOT/README.md" ] && [ -f "$SRC_ROOT/data/kb/README.md" ]; then
  cp -f "$SRC_ROOT/data/kb/README.md" "$KB_ROOT/README.md" || true
fi
ok "用户知识库 → $KB_ROOT（op-opt/issue/solution，重装不清除）"


# ===== 写幂等登记表 =====
mkdir -p "$CONF_DIR"
json_agents=""
for a in "${AGENTS[@]}"; do
  if [ -n "$json_agents" ]; then json_agents="$json_agents,"; fi
  json_agents="$json_agents\n    \"$a\": { \"status\": \"${INSTALLED_STATE[$a]:-ok}\", \"skill_count\": ${INSTALLED_COUNT[$a]:-0}, \"level\": \"$LEVEL\" }"
done
printf '{\n  \"schema_version\": 1,\n  \"package\": \"%s\",\n  \"toolkit_version\": \"%s\",\n  \"installed_at\": \"%s\",\n  \"level\": \"%s\",\n  \"install_path\": \"%s\",\n  \"agents\": {\n%b\n  }\n}\n' "$NAME" "$TOOLKIT_VERSION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LEVEL" "$INSTALL_PATH" "$json_agents" > "$INSTALLED_JSON.tmp"
mv "$INSTALLED_JSON.tmp" "$INSTALLED_JSON"
ok "登记表 → $INSTALLED_JSON"


# ===== 凭证骨架（保留原逻辑）=====
CRED="$CONF_DIR/credentials.json"
if [ -f "$CRED" ]; then
  info "凭证文件已存在，保留: $CRED"
else
  cat > "$CRED" <<'JSON'
{
  "_comment": "实际凭证文件。真实密码请由用户/环境填入，切勿提交进 skill 包。",
  "dcu-node-01": { "password": "YOUR_PASSWORD_HERE" }
}
JSON
  chmod 600 "$CRED"
  ok "已生成凭证骨架 $CRED (chmod 600)"
fi
DEV="$CONF_DIR/devices.json"
if [ -f "$DEV" ]; then
  info "设备清单已存在，保留: $DEV"
else
  cat > "$DEV" <<'JSON'
{
  "_comment": "设备清单骨架（占位符）。真实设备信息由使用者填入本文件，勿提交进 git。",
  "devices": {
    "<device_name>": {
      "host": "<device_ip>", "port": 22, "user": "<user>",
      "auth_method": "password", "dcu_type": "<dcu_type>", "dcu_count": 8
    }
  }
}
JSON
  ok "已生成设备清单骨架 $DEV"
fi

# 提示（仅 tool 模式）
if [ "$MODE" = "tool" ]; then
  warn "已用 --tool $TOOL 指定单目标任务，其它 agent 未安装。需要全量请用 --detect。"
fi

echo ""
echo "DONE: 安装完成，共处理 ${#AGENTS[@]} 个 agent，同步 $TOTAL 个 skill。"
echo "路由总入口见仓库根 SKILL.md / AGENTS.md / Plugins/dcu-core/skills/dcu-orchestrator/SKILL.md。"
echo "用户知识库：$KB_ROOT（沉淀记录写这里；重装/升级不会清除）"
