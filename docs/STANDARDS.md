# DCU Toolkit 开源版 — Skills 开发规范

> 命名、结构、分类、脱敏红线。贡献新 skill 前必读。

## 命名

- Skill 目录用 `{domain}-{name}` kebab-case（如 `dcu-vllm-deploy`）
- `SKILL.md` frontmatter 必须含 `name`（与目录名一致）、`description`（触发条件 + 不适用反例）、`version`、`license: MIT`

### 前缀例外清单

`dcu-` / `op-` 前缀表示**本平台特有**的技能。以下技能是**通用工程能力**
（上游框架的 CI、重构、profiling、部署等），加 `dcu-` 前缀会给它们
虚假的平台归属，因此**允许不带前缀**，登记在 `skill_lint.py:PREFIX_EXEMPT`：

| 类别 | skill |
|---|---|
| 工程规范与重构 | `ci-workflow-guide`、`clean-startup-log`、`env-var-conventions`、`large-class-init-style`、`mechanical-refactor-verify`、`scripted-runtime-notes` |
| cookbook | `cookbook-add-model`、`cookbook-migrate-model`、`cookbook-review-pr` |
| 上游框架操作 | `sglang-bisect-ci-regression`、`sglang-cherrypick`、`sglang-diffusion-add-model`、`sglang-humanize-review`、`sglang-model-day0-support`、`sglang-prod-incident-triage`、`vllm-pr-summary` |
| 内核接入 | `add-jit-kernel`、`add-sgl-kernel`、`generate-profile` |
| profiling / 分析 | `humanize-gen-plan`、`humanize-rlcr`、`llm-pipeline-analysis`、`llm-serving-auto-benchmark`、`llm-serving-capacity-planner`、`llm-torch-profiler-analysis`、`model-architecture-diagram`、`model-compute-simulation`、`model-pr-optimization-history`、`model-pr-history-knowledge`、`torch-profiler-layer-track` |
| 调试 | `cross-engine-acceptance-gate`、`debug-cuda-crash`、`debug-distributed-hang` |

**新增例外要改两处**：本表 + `skill_lint.py:PREFIX_EXEMPT`。判据只有一条——
「这个技能换成非 DCU 平台是否同样成立？成立就不加 `dcu-`」。

## 结构

```
Plugins/<plugin>/skills/<skill>/
├── SKILL.md          # 主入口：触发条件、流程、路由（正文控制在几百行内，细节下沉）
├── scripts/          # 可执行工具（python3/bash，相对路径自定位）
└── references/       # 领域知识文档（SKILL.md 里建速查表暴露）
```

- 渐进式披露：SKILL.md 只放路由与核心流程，深知识放 references/
- 共享数据放包根 `data/ references/ scripts/`，skill 内用相对路径引用；写明双态路径（仓库开发态在包根，安装态在 `<skills根>/_dcu-toolkit-shared/`，由 install.sh 同步）
- 沉淀落点是**用户知识库**（不是 skill 树）：`~/.config/dcu-toolkit/kb/`（`op-opt/` 算子优化台账、`issue/` 故障记录、`solution/` 部署方案卡片；目录与写入约定见 `data/kb/README.md`）。写入纪律：每条带 source + verified_on（日期/设备/gfx/DTK）、未验证标 unverified、性能数字标 [历史数据] 且不进可复用事实条目。

## 五分类

知识库类（被动调用、只读无副作用）/ 工程模板 / 调试与测试 / 测试开发 / 工具辅助。

## 信息来源红线（开源版强制）

1. **零内部溯源**：不出现内部文档平台链接/文件 ID、内部人名/组织/工单号
2. **零内网地址**：IP 用 `<DCU_NODE_IP>` 等占位符；域名用 `<DCU_CODE_HOST>` `<DCU_REGISTRY_HOST>` 等
3. **零真实凭证**：凭证只允许占位模板；设备清单（IP/机型映射）不进包
4. **上游代码合规**：不内嵌第三方项目源码树（如 vLLM/RCCL/SGLang 的 source-tree）；方法论与 diff 片段需注明上游 license
5. **发布前必跑**：`python3 sanitize.py <repo>` 后 grep 断言 `内部实测记录|<DCU_CODE_HOST>|<ORG>|scnet|内网IP` 零残留

## 测量口径一致性

### 【口径一致性】跨实现排序必须在同一进程、同一输入策略下完成

**禁止把两次不同时间跑的数放在同一张排序表里。**
lightop 这类原地改写实现，必须显式声明「输入是否每轮克隆」。

### 【交替测量有效】配对交替能抵消共模干扰 —— 但前提是「干扰是对称的」

2026-11-23 先测到 `lightop 交替后 435 µs`（单独 211），一度误判「交替有毒」。
**那个结论是错的**——复查发现有外部进程 `python3 -m pytest tests/ -q`
（PID 492471，经 `hy-smi --showpids` 确认）在抢占 GPU。

干净环境复测：

| | 独立测 | 交替测 | 差 |
|---|---|---|---|
| lightop | 210.2 µs | 211.7 µs | +0.7% |
| tilelang | 216.5 µs | 212.0 µs | -2.1% |

**交替没有破坏任何一方。** 原理也对：配对交替让两个实现承受同样的环境噪声，
比值被保住。

**但必须记住适用条件：干扰要对称。** 若 A 被干扰、B 没被干扰，比值照样失真——
这正是那次误判的来源（外部进程只在那段时间抢，前后不对称）。

所以：**交替是首选，但仍要配「漂移检测」**——测完回头重测第一个，
差 >5% 就说明环境在变，整张表作废。交替解决共模噪声，不能解决非平稳噪声。

### 【环境判据】只有 `hy-smi --showpids` 有 PID 才算占用

`--showuse` 的单次数字不作依据（可能是别的租户一个短 kernel，也可能是采样本身的开销）。
必须**连续 3 次采样 + 确认 PID 归属**。

2026-11-23 实际踩到：测量中 HCU 使用率在 1.6%~87.5% 波动，
查 `--showpids` 发现 `python3 -m pytest tests/ -q`（PID 492471）在跑，
不是自己的进程。该轮排序数据已标注作废。

## 【GEMM 对标】必须用 swept hipBLASLt 当对手

> 来源：PANORAMA v2 §3.3 × hbird-gemm `tune/opponent.py` 头注释；
> 实操清单见 `TESTING.md` 第二节。**这一节不做，所有「我们对标厂商」的结论都会虚高。**

**硬规则**：GEMM 性能对标时，baseline 必须是**做过 solution 扫参的 hipBLASLt**，
不许拿「默认解 / 框架默认调用」当对手。理由是一条决定性事实：

> `torch._scaled_mm` 拿到的 1.9 ms **不在** hipBLASLt 自己 252 个解里
> （该 shape 最优 2.0998 ms）。**框架默认调用不是库最优**，对标必须手工扫参。

### 三个陷阱（任踩一个，结论即作废）

| # | 陷阱 | 错误做法 | 后果（实测） | 正确做法 |
|---|---|---|---|---|
| 1 | **启发式 ≠ 调优** | `--algo_method heuristic`，把默认解当最优 | 默认 232.0 µs vs 最优 210.9 µs，**差 10%**，把平局报成 **1.38× 领先** | 加 `--requested_solution -1` 枚举全部（如 51200×5120 M=1 有 97 个候选） |
| 2 | **bare fp8 ≠ rowwise** | 不加 scale 参数 | hipBLASLt 搜的是**不同解空间**，同 shape **249.5 vs 210.9 µs，差 18%** | 必须 `--scaleA 2 --scaleB 2 --api_method c` |
| 3 | **扫参的数不复测** | 直接用枚举结果当性能 | 枚举用低 iters，不可信 | 胜出 `solution_index` 单独 `--iters 100` 复测，**多次取 min** |

### 推荐调用序列（8192³ fp8，hbird 实测可用）

```bash
BENCH=/opt/dtk/lib/hipblaslt/benchmark_tool/hipblaslt-bench

# 第一步：枚举全部 solution
$BENCH -m 8192 -n 8192 -k 8192 \
  --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r --transA T --transB N \
  --alpha 1 --beta 0 --initialization rand_int \
  --activation_type None --print_kernel_info \
  --scaleA 2 --scaleB 2 --api_method c \
  --algo_method heuristic --requested_solution -1

# 第二步：用胜出的 solution_index 单独复测（--iters 100，多次取 min）
$BENCH <同上> --algo_method index --solution_index <N> --iters 100
```

### ⚠️ 本机限制：这条规则目前**无法在本机执行**

`/opt/dtk/lib/hipblaslt/benchmark_tool/hipblaslt-bench` 在本机权限是
`-rw-r--r--`（**没有可执行位**，`stat` 确认，ELF 可执行文件但不可 exec），
**跑不起来**。所以：

- 凡引用「已 swept hipBLASLt 对手」的结论，**必须标明数据不是在本机产生的**
  （来自 PANORAMA / hbird-gemm / 其他环境），不得暗示本机可复跑。
- 本机目前能做的替代是 `torch._scaled_mm`（走 hipBLASLt 后端）的默认调用——
  但按上面的决定性事实，**它只代表「框架默认解」，不能代表库最优**，
  引用时须注明「默认解，未扫参」。

### 反向测试要求（由主线程实现测试代码）

对标闸门必须能**抓住**「拿默认 solution 当基准」这种错误，不能只写在文档里。
要求新增一条负向测试（与 `Plugins/ops/tests/test_baseline_policy.py` 同风格）：

- **输入**：一个 benchmark 记录，其 baseline 标注为 hipBLASLt 默认 solution
  （无 `--requested_solution -1` / 无 swept 证据）。
- **判据**：校验器必须 **报 error**（不是 warning），错误文案要点明
  「baseline 是默认解，未做 solution 枚举，不能当性能基准」。
- **对照正向**：同一 shape 补上 swept solution_index 与 `--iters 100` 复测证据后，
  同一校验器必须放行。

> 这条测试的存在意义：陷阱 1 的 10% 差距恰好落在「看着像平局 / 像是领先」的区间，
> 人眼审不出来，只能靠机器拦。

## 版本与文档一致性

- 改动 skill 后：更新该 skill frontmatter `version` + 包根 `CHANGELOG.md`（SemVer）
- 新增/删除 skill：同步更新包根 `SKILL.md` 路由表、`README.md` 插件计数、`marketplace.json`
- 路由表「意图 → skill」必须与 skill 自身 description 声明的能力边界一致

## 评测（改 description / 加 skill 必看）

- 用例放 `<skill>/evals/evals.json`：3 条 true（带 `expect` 要点）+ 2 条 false（形似、应路由他处）。
  完整用法、判分口径与失败修法见 [`docs/EVALS.md`](EVALS.md)。
- `description` 的「触发条件 + 不适用反例」不再只靠人审：`scripts/run_evals.py` 会逐条校验，
  **带 evals 的 skill 缺任一面直接 FAIL**；无 evals 的只统计，`--strict-description` 可全库阻塞。
- 路由问题优先改 `description`（触发词 + 边界），内容问题才改 `SKILL.md` 正文；两条评测线别混。
