# 参考：DCU 镜像逆向重建 + 端到端构建验证（命令脚手架与版本映射）

> 配套 SKILL.md。随具体镜像变化；SKILL.md 是稳定方法论。**本文件 §2/§3 已按本会话最终实测纠正：vLLM 可用索引是 `/nightly/dtk/`，不是 `/release/dtk2604/`**。

## 0. 前置：连接 DCU 目标机
凭据在 `/root/.config/dcu-toolkit/credentials.json`，key 形如 `<<internal-ip>>-BW100X8`（密码字段）。统一用 sshpass 非交互：
```bash
export SSHPASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<IP>-BW...']['password'])")
sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@<IP> "<cmd>"
```

## 1. 取 history + inspect + 抽 COPY 缺口
```bash
docker history --no-trunc <image> > /tmp/history.txt
docker inspect <image>      > /tmp/inspect.json
docker exec <container> cat /install_dep.sh    > install_dep.sh
docker exec <container> cat /etc/profile.d/env.sh > env.sh
docker exec <container> tar -C /opt -cf - dtk > /data/dtk.tar   # DTK 无公开源，从容器出
```

## 2. 实测版本映射（vllm:0.15.1--ubuntu22.04-dtk26.04-py3.10-20260409）
来自 `docker history` 精确版本，重建时按此对齐，**不要猜**：
| 层 | 原始源（内网） | 可替代公开/国产源 |
|---|---|---|
| hwloc | <internal_ip_src>:8000/pkg/openmpi/hwloc-2.12.1.tar.gz | `https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.1.tar.gz`（官方发布包，带 configure） |
| ucx | <internal_ip_src>:8000/pkg/openmpi/ucx-1.15.0.tar.gz | `https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz`（官方 release） |
| openmpi | <internal_ip_src>:8000/pkg/openmpi/openmpi-5.0.3.tar.gz | `https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.gz`（官方发布包） |
| OFED | content.mellanox.com MLNX_OFED-23.10-4.0.9.1 | apt: `rdma-core libibverbs-dev ibverbs-utils ibverbs-providers`（2307mlnx47 即 MOFED23.10 用户态） |
| vLLM | <internal_ip_src>:9929 dtk2604 私有索引 `vllm==0.15.1` | **`https://pypi.sourcefind.cn/nightly/dtk/`**（见 §3，⚠️ 不是 `/release/dtk2604/`） |
| numpy | `numpy==1.25.0`（install_dep.sh） | pypi.org 上有 1.25.0（或走 nightly/dtk 双源） |
| amdsmi | `amdsmi`（install_dep.sh） | `https://pypi.sourcefind.cn/nightly/dtk/`，有 24.5.3+02cbffb.dirty |
| DTK | COPY /opt/dtk（15.8G） | 仅容器 tar |

## 3. vLLM 源替换关键命令（pypi.sourcefind.cn，已实测纠正）

> ⚠️ **最终结论**：可用索引是 **`/nightly/dtk/`**，`/release/dtk2604/` 会 `ResolutionImpossible`（缺 triton 3.3.1）。

```bash
# DTK_VERSION=2604 -> 可用索引 https://pypi.sourcefind.cn/nightly/dtk/
pip index versions vllm -i https://pypi.sourcefind.cn/nightly/dtk/
# 逐字节一致精确装（nightly/dtk 上有原镜像同款轮）:
pip install "vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba" -i https://pypi.sourcefind.cn/nightly/dtk/

# 主装层 pip.conf 双源（vLLM 依赖 cmake==3.29，从 pypi.org 满足）:
[global]
index-url = https://pypi.org/simple
extra-index-url = https://pypi.sourcefind.cn/nightly/dtk/
```
PEP 440 注意：`vllm==0.15.1+das...torch290`（无 `.*`）是**精确匹配整串** → No matching distribution；要么 `vllm==0.15.1`（前缀匹配 release 段），要么 pin 完整轮名。
首页是 DAS Hub SPA，`curl /simple/vllm/` 只返回前端 HTML 壳 —— **不能据此判非 pip 源**，认准 `/nightly/dtk/` 路径即可。

## 4. 网络预检脚本
```bash
nc -z -w5 <internal_ip_src> 8000 && echo 'src OK'   || echo 'src UNREACH'
nc -z -w5 <internal_ip_src> 9929 && echo 'pip OK'   || echo 'pip UNREACH'
curl -s -m15 -o /dev/null -w 'mellanox:%{http_code}\n' https://content.mellanox.com/ofed/
curl -sL -m20 -o /dev/null -w 'sourcefind:%{http_code}\n' https://pypi.sourcefind.cn/nightly/dtk/
```

## 5. 抗 ssh 断线：目标机 nohup 独立 build
```bash
ssh ... root@<host> "cd /data/buildctx && nohup docker build -f Dockerfile.vllm015-0409-reconstructed -t vllm015-rebuild-final . > /data/build_final.log 2>&1 & echo PID=\$!"
# 之后短连轮询（不依赖本地长连）
ssh ... "grep BUILD_EXIT /data/build_final.log"
ssh ... "docker run --rm vllm015-rebuild-final pip show vllm"   # 真核实，非看 Successfully built
```

## 6. 清理卡死 build（防自匹配）
```bash
pkill -9 -x wget                                   # 精确按进程名杀 wget
K='Docker'; L='file.vllm015'                        # 拆分避免脚本字面量自匹配
pkill -9 -f "$K$L"                                  # 只杀该 build 客户端
# 复检：pgrep -x wget | wc -l 应为 0；docker build 客户端应为 0
# 不动 docker image prune / 不动其他容器
```

## 7. 本会话实测踩到的真实 bug 与硬规则（验证价值）
1. 重建笔误：独立的 `pip install --upgrade pip` 层被误并到 Python3.10 层开头 → Step 15 `pip: not found`。恢复独立层后过。
2. hwloc/ucx/openmpi 用 GitHub **archive 快照** → 解出来目录名带前缀且**不含 `./configure`** → `./configure: not found` (exit 127)。必须改用官方发布包。
3. `cmake==3.29.0` 公共 PyPI 无精确版本（仅 3.29.0.1+）→ 放宽到 `cmake==3.29.0.1`。任何 `==X.Y.Z` 精确 pin 换成公共源前先 `pip index versions` 确认存在。
4. OFED 层对 content.mellanox.com 出网挂死（37MB 下 30min 不退出）→ 改 apt 装 rdma-core 用户态。
5. **`/release/dtk2604/` 索引缺 triton 3.3.1 → vLLM 0.15.1 ResolutionImpossible** → 必须换 `/nightly/dtk/`。
6. **PEP 440 `==` 精确匹配**：`vllm==0.15.1+das...torch290`（无 `.*`）失败；用 `vllm==0.15.1` 或 pin 完整轮名。
7. **cmake 双源**：nightly/dtk 作唯一 index 时 vLLM 的 `cmake==3.29` 找不到 → 主装层 pip.conf 用 `index-url=pypi.org` + `extra-index-url=nightly/dtk`。
8. **`set -e` 防静默假成功**：install 脚本无 `set -e` 时 vllm 缺失仍 exit 0、镜像"建出来"却没 vllm；加 `set -e` + build 后 `docker run pip show vllm` 真核实，绝不靠 "Successfully built"。
9. **ssh 断 pipe (exit 255) 杀远端 build** → 目标机用 `nohup` 独立运行。
