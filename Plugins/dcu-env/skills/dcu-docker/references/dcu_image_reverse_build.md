# 参考：DCU 镜像逆向重建 + 端到端构建验证（命令脚手架与版本映射）

> 配套 SKILL.md 的「DCU 镜像逆向重建与端到端构建验证」章节。本文件是会话级细节，
> 随具体镜像变化；SKILL.md 是稳定方法论。

## 0. 前置：连接 DCU 目标机
见 `dcu-ssh` skill。凭据在 `/root/.config/dcu-toolkit/credentials.json`，
key 形如 `<<internal-ip>>-BW100X8`（密码字段）。统一用 sshpass 非交互：
```bash
export SSHPASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<IP>-BW...']['password'])")
sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@<IP> "<cmd>"
```

## 1. 取 history + inspect
```bash
docker history --no-trunc <image> > /tmp/history.txt
docker inspect <image>      > /tmp/inspect.json
# 抽 COPY 缺口文件
docker exec <container> cat /install_dep.sh    > install_dep.sh
docker exec <container> cat /etc/profile.d/env.sh > env.sh
# 大目录 tar（DTK 无公开源，只能从容器出）
docker exec <container> tar -C /opt -cf - dtk > /data/dtk.tar
```

## 2. 本会话实测版本映射（vllm:0.15.1--ubuntu22.04-dtk26.04-py3.10-20260409）
来自 `docker history` 的精确版本，重建时按此对齐，不要猜：
| 层 | 原始源（内网） | 可替代公开源 |
|---|---|---|
| hwloc | <internal_ip_src>:8000/pkg/openmpi/hwloc-2.12.1.tar.gz | GitHub releases open-mpi/hwloc v2.12.1 |
| ucx | <internal_ip_src>:8000/pkg/openmpi/ucx-1.15.0.tar.gz | GitHub releases openucx/ucx v1.15.0 |
| openmpi | <internal_ip_src>:8000/pkg/openmpi/openmpi-5.0.3.tar.gz | GitHub releases open-mpi/ompi v5.0.3 |
| OFED | content.mellanox.com MLNX_OFED-23.10-4.0.9.1 | apt: rdma-core / libibverbs-dev / ibverbs-utils |
| vLLM | <internal_ip_src>:9929 dtk2604 私有索引 `vllm==0.15.1` | **无公共源**；DCU 定制轮 `0.15.1+das.opt1.alpha.dtk2604.torch290...` |
| numpy | `numpy==1.25.0`（在 install_dep.sh） | 公共 PyPI 有，但 DCU 场景应与 vLLM 轮一致 |
| DTK | COPY /opt/dtk（15.8G） | 仅容器 tar |

## 3. 网络预检脚本
```bash
nc -z -w5 <internal_ip_src> 8000 && echo 'src OK'   || echo 'src UNREACH'
nc -z -w5 <internal_ip_src> 9929 && echo 'pip OK'   || echo 'pip UNREACH'
curl -s -m15 -o /dev/null -w 'mellanox:%{http_code}\n' https://content.mellanox.com/ofed/
```
注意：`pypi.sourcefind.cn` 返回的是「DAS Hub」前端网页（`<title>DAS Hub</title>`），
`/simple/<pkg>/` 不是 PEP 503 索引，**不能当 pip 源**。

## 4. 后台长 build + watcher（避免 wait 被钳 60s）
```bash
# 目标机上起 watcher 后台进程等 build 退出并回报
sshpass -e ssh root@<IP> "
  while ps aux | grep -q '[d]ocker build'; do sleep 15; done
  echo BUILD_EXIT=\$?; grep 'Step [0-9]' /data/build.log | tail -1
  docker images | grep rebuild || echo NO_REBUILD
"
```

## 5. 清理卡死 build（防自匹配）
```bash
pkill -9 -x wget                                   # 精确按进程名杀 wget
K='Docker'; L='file.vllm015'                        # 拆分避免脚本字面量自匹配
pkill -9 -f "\$K\$L"                                # 只杀该 build 客户端
# 复检：pgrep -x wget | wc -l 应为 0；docker build 客户端应为 0
# 不动 docker image prune / 不动其他容器（<internal-user> 容器保持 Up）
```

## 6. 本会话踩到的真实 bug（验证价值示例）
- 重建时把 history 中**独立的** `pip install --upgrade pip` 层误并到 Python3.10 层开头
  → Step 15 `pip: not found` 真实报错。修正（恢复独立层）后重 build 才过。
- OFED 层在 <device-B> 上对 content.mellanox.com 出网挂死（37MB 下 30min 不退出）→ 改 apt 装 rdma-core。
- `pkill -f 'docker build'` / `grep docker build` 自匹配执行 shell → exit 255，误判“没杀掉”。
