---
name: dcu-docker
description: "在 DCU 主机上用 Docker 跑 DCU 容器：docker run 设备挂载、容器内/容器间 SSH 免密、镜像导出与离线部署、派生镜像构建与容器内测评坑。触发场景：起或进 DCU 容器、容器里看不到卡、两个容器要配免密跑多机测试、镜像要打包搬到离线机。不适用：宿主机直连与远程命令执行走 dcu-ssh；容器拉哪个镜像走 dcu-image-hub；DTK/驱动安装走 dcu-dtk/dcu-driver。"
version: 1.2.2
author: DCU Engineer
license: MIT
kind: tool
---

# DCU Docker 操作 — 容器 SSH 免密配置

## 0. 操作范围门（scope gate，执行前必读）

docker 操作是全库最高危区（删容器/删镜像/清卷都一行命令且不可逆）。默认只允许作用于**本会话自己启动的资源**：

- **容器**：只 `docker rm -f` / `restart` 自己 run 起的（命名带会话标识便于识别）；他人容器只读（`ps` / `logs` / `inspect` / `stats` 可）。
- **镜像/卷**：`docker rmi`、`volume rm`、各类 `prune` 一律先报清单获批准——prune 是全局作用域，会扫到别人的资源。
- **宿主**：不在宿主上直接装/卸驱动类包；容器内操作优先。
- 例外（须动他人资源才能完成任务）：逐次向用户报确切命令获批准后执行；长期成立的例外登记到下方白名单，其余情形仍按默认规则。

**例外白名单（经用户批准后逐条登记）**：暂无

## Overview

海光DCU的软件环境基本都跑在 Docker 容器里（DTK 镜像、训练/推理镜像等）。
本 skill 覆盖三类 DCU 容器高频操作：
1. **DCU 镜像启动（docker run）** — 设备透传、驱动挂载、shm、模型/代码目录挂载的正确模板
2. **容器间 SSH 免密** — 仅限跨节点场景（多节点集合通信测试、多节点分布式推理）
3. **测评执行方式坑** — `docker exec` 启动方式直接影响推理性能，必须用 `-ilc`

核心参考：CSDN《两台服务器上的两个docker容器之间配置ssh免密登录》
https://blog.csdn.net/weixin_45973213/article/details/134891512

## When to Use

> 📌 **遇到具体报错先查排查经验库**：容器库缺失（`libxxx.so`）、编译失败、numpy/transformers 版本冲突等具体现象的处理经验（含 env/修复命令），统一收在 **`dcu-ontrack-experience`** 这个 skill 里。本 skill 只讲容器启动模板与免密配置，具体踩坑手法请去那里按"现象关键词"反查。

**仅在以下两类场景主动配置容器内 SSH 免密**（其余场景不主动配，用 `docker exec` 进容器即可）：
1. **多设备节点跑集合通信测试** — 多节点多卡，节点间需互信才能跑通集合通信（如 RCCL/通测工具）
2. **多节点一起跑模型推理** — 分布式推理跨节点，需节点间免密互信

Don't use for：
- 单节点 / 单机多卡 的容器操作（用 `docker exec` 进容器即可，无需 sshd）
- 宿主机本身的 SSH 配置（见 dcu-ssh）
- 纯本地 docker 构建/编排（非 DCU 相关）
- 临时进容器敲命令、看状态（不需要免密）

> 判定标准：只有"**跨节点**需要节点间互联通信"时才配容器免密；单机内多卡不需要。

## DCU 镜像启动（docker run）

DCU 容器必须用正确的设备透传、共享内存、驱动挂载参数才能跑起来。以下为实测可用的启动模板（来源：实际 vLLM on DCU 部署命令）。

> **方向性推荐（适配场景）：适配类任务的底座镜像直接用官方（光源）对应 DTK/版本的镜像，适配全程在容器内完成；尽量不要在物理机上装软件栈做适配（详见 `dcu-software-adaptation-guide`「工作原则：容器优先」，镜像查询走 `dcu-image-hub`）。**

### 标准启动命令模板
```bash
docker run -itd \
    --name <container> \
    --shm-size=16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --network=host \
    --ipc=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /root/.ssh:/root/.ssh \
    -v /public:/public -v /data:/data -v /home:/home \
    <internal_registry>/dcu/admin/base/vllm:<tag> \
    /bin/bash
```

### 关键参数说明
| 参数 | 作用 | 必要性 |
|---|---|---|
| `--device=/dev/kfd` | 透传 DCU 内核驱动设备（计算设备） | **必需** |
| `--device=/dev/mkfd` | 透传 DCU 管理设备 | **必需** |
| `--device=/dev/dri` | 透传 DRM 渲染设备 | **必需** |
| `-v /opt/hyhal:/opt/hyhal:ro` | 挂载 DCU 驱动/运行时（只读，宿主机已装 hyhal） | **必需**（容器靠它识别DCU） |
| `--shm-size=16G` | 扩大共享内存，避免多进程/多卡通信 shm 不足 | 推荐（默认64M太小） |
| `--group-add video` | 加入 video 组，访问 /dev/dri 等设备 | 必需 |
| `--cap-add=SYS_PTRACE` | 允许 ptrace（调试/性能分析、部分框架需要） | 推荐 |
| `--security-opt seccomp=unconfined` | 关闭 seccomp 限制，DCU 驱动需要 | 推荐 |
| `--network=host` | 主机网络，分布式训练/服务端口直通 | 多卡/服务必用 |
| `--ipc=host` | 共享 IPC 命名空间（多进程共享内存通信） | 多卡必用 |
| `--privileged` | 特权模式（驱动透传常需） | 常用，注意安全 |
| `-v /root/.ssh:/root/.ssh` | 挂载宿主机 ssh 密钥，便于容器内免密/拉代码 | 按需（配合免密章节） |

### ⚠️ 启动容器前先查宿主机剩余空间（重要）

拉取镜像（常 10~30 GB）+ 起容器可写层会持续吃盘，物理机上若已叠加多个业务容器，极易把盘写满导致 `docker pull` / 运行失败。**起容器前先确认目标盘剩余足够：**
```bash
df -h / /data /public                 # 按实际挂卷/ docker root 所在盘替换
docker system df                      # 看镜像/容器/卷已占用
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | sort -k2 -h
```
- 剩余 < 20 GB：先确认无更大写入；< 10 GB 先 `docker image prune -f` / 删不再用的旧容器再继续。
- docker 根目录所在盘以 `docker info | grep -i 'docker root'` 为准，不一定是 `/`。
- **`docker system df` 的 RECLAIMABLE 数字不能当真实可释放量**：它按全部未用镜像的总大小汇总，实际物理释放只发生在不与任何在用镜像共享的层上——同基底镜像多的宿主机上 dangling prune 实际可能只释放很少（实测：报表称 ~230G reclaimable，`df` 只从 25G→26G）。清理后务必 `df -h` 核真实剩余再决策。
- **拉取前先估体积（防 pull 中途 ENOSPC）**：`docker manifest inspect <img>:<tag>`（registry 凭证已在 `~/.docker/config.json` 时可用）逐层加总得全量大小；本地已有的层不会重下（pull 日志 `Already exists` 行为准），据此估算真实下载量与 ETA（实测速率见下条）。
- **pull 慢先诊断链路再谈并行**：并行 blob curl 测单连接速度与聚合速度，判断是单连接限速还是链路共享上限（配方+判定表见 `references/registry_pull_diagnostics.md`）。聚合速度≈单连接速度 → 链路共享带宽上限，并行（daemon `max-concurrent-downloads` 等）无用，别折腾；有效升级 = 让镜像 owner 推一份到内网仓库（daemon.json `insecure-registries` 已配的内网 registry）走内网拉。

### 镜像命名约定（建议）
`<user>-<框架><版本>-<日期>`，如 `<user>-vllm0110-20260204`：
- 含用户、框架版本、构建日期，方便多容器并存时区分
- 同名会冲突，重建前先 `docker rm` 旧容器

### `-v` 挂载模型与代码目录（重要补充）
上述模板挂载了 `/public /data /home` 等通用目录。实际使用时**必须额外挂载模型目录和代码目录**到容器内，否则容器内读不到本地模型/代码：
```bash
# 在上面的 -v 列表基础上追加（按实际路径替换）：
    -v /path/to/your/models:/models \
    -v /path/to/your/code:/workspace/code \
```
- **模型目录**：如 HuggingFace 权重、vLLM 模型缓存，挂载后容器内直接 `/models/xxx` 访问
- **代码目录**：训练/推理脚本、自研代码，挂载后容器内 `/workspace/code` 可直接改直接用
- 挂载点路径在容器内自定义，但建议语义清晰（`/models`、`/workspace/code`）
- 若模型在对象存储，也可容器内直接拉，但本地挂载更省带宽、可离线

### 启动后进入容器
```bash
docker exec -it <container> bash
# 验证 DCU 可见：
hy-smi          # 应列出所有 DCU 卡
```
> 若 `hy-smi` 报找不到命令，需 `source /opt/dtk/env.sh`（见 dcu-env）。
> 想了解 `/opt/dtk` 内部目录和自带工具（hipcc/rocminfo/gpurun/clang-ocl 等）怎么用，见 `dcu-dtk-layout`（与 dcu-dtk 互补：dcu-dtk 讲安装/激活，dcu-dtk-layout 讲装好后里面长什么样）。

### ⚠️ 宿主机 `docker cp` 报 "no space left on device" 的 workaround（2026-08-21 <device-A> 实测）

- **现象**：宿主机根分区明明还有几十 GB，`docker cp file root@host:/tmp/` 进容器仍报 `no space left on device`（overlay 挂载/配额问题，与真实剩余空间无关）。
- **对策**：先 `scp` 到宿主机 `/tmp`，再用 stdin 重定向写进容器：
  ```bash
  scp -o StrictHostKeyChecking=no localfile root@<host>:/tmp/
  ssh root@<host> "docker exec -i <container> bash -c 'cat > /path/in/container' < /tmp/localfile"
  ```
- 小文件（脚本/配置）这是最稳路径；大文件（模型/权重）优先直接 `-v` 挂载宿主机目录，不走 cp。

## 关键原则（务必先读）

0. **使用前提（最重要）**：容器免密**只**在"多设备节点跑集合通信测试"或"多节点一起跑模型推理"这两个跨节点场景才主动配置。单节点/单机多卡/临时进容器都**不配**，直接用 `docker exec` 即可。配免密=要跨节点互联通信时才做。
1. **容器内 SSH 端口必须与宿主机区分**：宿主机 sshd 通常占 22，容器内若也用 22 会导致 `ssh` 连到宿主机而非容器。→ 容器内 sshd 改用 **2233**（或其他非22端口）。
2. **先切 root 装/启 sshd，再切回业务用户生成密钥**：openssh 安装和 sshd 启动需要 root；密钥对用业务用户（如 `<user>`）身份生成，authorized_keys 也放在该用户家目录。
3. **互信 = 双方公钥交换进同一份 authorized_keys**：A、B 两容器各自的 `id_*.pub` 内容都写进对方的 `~/.ssh/authorized_keys`，且两边 authorized_keys 内容一致。
4. **所有步骤在容器内部执行**（不是宿主机），除非你明确是在配宿主机→容器的端口映射。

## Standard Procedure（容器内 SSH 免密，容器↔容器）

以下步骤均在**目标容器内部**执行：

### Step 1. 切 root + 安装 openssh
```bash
sudo -i            # 或 su -，切到 root
apt-get update && apt-get install -y openssh-client openssh-server
# 若基础镜像是 rpm 系（openEuler/Kylin）：yum install -y openssh-clients openssh-server
```

### Step 2. 改容器内 sshd 端口（避开宿主机22）
```bash
vim /etc/ssh/sshd_config
# 在文件末尾加：
Port 2233
```
> 目的：容器内 ssh 端口改为 2233，与宿主机默认 22 区分。不改的话，`ssh` 会默认连到容器外宿主机。

### Step 3. 启动/重启 sshd
```bash
service ssh restart          # Debian/Ubuntu 系
# 或：/usr/sbin/sshd          # 直接拉起守护进程（若无 service 命令）
```

### Step 4. 切回业务用户，生成密钥对
```bash
su - <user>                     # 切到业务用户（按需替换用户名）
ssh-keygen -t ed25519        # -t ed25519 生成短公钥；也可 rsa。一路回车不设口令
# 生成 ~/.ssh/id_ed25519 和 id_ed25519.pub
```

### Step 5. 交换公钥，建立互信
将 A 容器的 `~/.ssh/id_ed25519.pub` 内容，追加进 B 容器的 `~/.ssh/authorized_keys`；
同时将 B 的公钥追加进 A 的 authorized_keys。**最终两边 authorized_keys 内容相同**（各含两容器公钥）。
```bash
# 在 A 容器：
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
# 把 B 的公钥也写进来（可从 B scp 过来或手动粘贴）
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```
> 多容器/多机场景：N 个节点就把 N 个公钥全部汇进同一份 authorized_keys，再分发到每个节点。

### Step 6. 配置 SSH config 别名（可选但推荐）
在 `~/.ssh/config` 写入：
```
Host r1d
    HostName 10.0.0.1
    User <user>
    Port 2233
Host r4d
    HostName 10.0.0.4
    User <user>
    Port 2233
```
之后直接 `ssh r4d` 即可免密登录（首次 `yes` 接受指纹后不再提示）。

## 变体场景

### 场景 A：宿主机免密进容器（仅在确实需要 sshd 接入时才配）
- 多数情况用 `docker exec` 即可，不需 sshd；仅在需从跳板机/外部通过 ssh 接入容器时才配
- 容器内按 Step1–3 起 sshd（端口 2233），宿主机 `ssh -p 2233 <user>@<容器IP>`
- 把宿主机公钥加进容器 authorized_keys 即可单向免密（注意：这不属于"跨节点互信"主场景，按需处理）

### 场景 B：多机多卡分布式训练（最常见 DCU 需求）
- 每个节点的训练容器都按上述配好互信（所有 rank 节点的公钥汇进同一 authorized_keys 并分发）
- 确保各容器间 `ssh <其他容器IP> -p 2233` 能免密通
- 再跑 torchrun / horovod 等分布式框架

### 场景 C：容器重启后 sshd 不自动起
- 很多 DCU 镜像不把 sshd 设为自启。重启容器后需重新 `service ssh restart`
- 可在容器启动脚本（entrypoint / `~/.bashrc` 末尾）加 `service ssh start` 保活

## 测评执行方式（性能测试坑 — 重要）

在 DCU 容器里跑大模型推理/性能测评时，**`docker exec` 的启动方式会直接影响测得性能**，这是一个反复踩过的深坑。

### 三种启动方式对比（定性结论）

| 方式 | 命令 | 是否读 .bashrc/环境变量 | 吞吐表现 |
|---|---|---|---|
| 方式1 ❌ | `docker exec -d bash -c '...'` | **不读**（非交互非登录 shell，跳过 .bashrc） | 明显劣化 |
| 方式2 ✅ | `docker exec -ti <容器> bash` 然后 `bash xxx.sh` | 读（交互 shell 会 source profile） | 正常 |
| 方式3 ✅ | `docker exec -d bash -ilc '...'` | 读（`-i -l` 强制交互+登录，加载 .bashrc/.profile） | 正常 |

> 方式1 比方式2/3 **慢约 15–20%**（rocblas/hipblaslt 未加载，吞吐掉一截）。具体 Req/s·Out tok/s 数字属性能基线，不入本 skill；如需引用转 `dcu-perf-cases`。

### 根因
- `docker exec -d bash -c 'cmd'`（方式1）是**非交互、非登录** shell，**不会执行 `/root/.bashrc` / `/etc/profile` / `~/.profile`**。
- DCU 的 **rocblas / hipblaslt** 等加速库路径依赖 `.bashrc` 里 `source /opt/dtk/env.sh`（或等价导出 `LD_LIBRARY_PATH`）来正确加载。
- 方式1 跳过了 `.bashrc` → **rocblas/hipblaslt 未正确加载**（可能 fallback 到慢路径或错误版本）→ 推理性能明显劣化。
- 方式2/3 都进入了会加载环境变量的 shell（方式2 是 `-ti` 交互 shell；方式3 是 `-i -l` 强制登录+交互），库正常加载，性能满血。

### 解决方案（强制规范）
**测评一律用方式3：**
```bash
docker exec -d bash -ilc 'bash /path/to/bench.sh'
# 或带完整命令：
docker exec -d bash -ilc 'source /opt/dtk/env.sh && python infer.py'
```
- `-i` 交互 shell + `-l` 登录 shell → 保证 `.bashrc`/`.profile` 被执行，rocblas/hipblaslt 正确加载
- `-d` 后台运行（测评通常跑批，不需要前台占用）
- 等价做法：方式2（`docker exec -ti` 进容器手动跑）也可，但不适合自动化/后台批跑，且易因终端断开中断

### 关联
- 环境变量加载细节见 dcu-env（DTK env）。若容器内 `.bashrc` 未包含 `source /opt/dtk/env.sh`，即使 `-ilc` 也可能缺库——先确认 `.bashrc` 已正确配置。
- 同理，**所有性能相关命令**（benchmark、集合通信测试）都应走 `-ilc`，否则测出的数据不可信。

## DCU 镜像逆向重建与端到端构建验证

需要从已有 DCU 容器/镜像“复现 Dockerfile”或验证其可重建时使用。典型场景：拿到一个内部镜像（如 `vllm:0.15.1--ubuntu22.04-dtk26.04...`），要拿到可用 Dockerfile、或在别的机器重建。

### 流程总览
1. `docker history --no-trunc <image> > history.txt` + `docker inspect <image> > inspect.json` —— 拿到全量 RUN/ENV/COPY/CMD（COPY 的文件内容看不到，只看到 file hash）。
2. 从 history + inspect 1:1 写出 Dockerfile（RUN/ENV/ARG/CMD 可完全还原；COPY 的输入文件/目录是“构建上下文缺口”）。
3. 抽回 COPY 的缺口文件：`docker exec <c> cat /install_dep.sh > install_dep.sh`、`/etc/profile.d/env.sh` 等；大目录（如 /opt/dtk）用 `docker exec <c> tar -C /opt -cf - dtk > dtk.tar` 在宿主机接。
4. 端到端 `docker build` 验证 —— 会暴露重建笔误（见下“坑”），**必做**，不能只看 Dockerfile 对不对。

### 构建上下文缺口的处理（按用户硬约束：除 DTK 外不从容器抽产物）
- **DTK（/opt/dtk，15G+）**：无公开发布源，只能 tar 容器现成品（用户允许的唯一例外）。
- **hwloc / ucx / openmpi**：内网 `<internal_ip_src>:8000/pkg/openmpi/*.tar.gz` 常不可达；等价公开源为 GitHub releases / open-mpi.org（版本号从 history 精确取，如 hwloc-2.12.1、ucx-1.15.0、openmpi-5.0.3），`wget` 编译即可。
- **OFED（MLNX_OFED-23.10-4.0.9.1）**：原走 `content.mellanox.com` 外网，**部分 DCU 主机出网会被挂死**（37MB 包下半小时不退出）。多数 vLLM/RCCl 通信只需用户态 `rdma-core`/`libibverbs-dev`/`ibverbs-utils`，可直接 `apt-get install` 替代，**不必装全量 MLNX_OFED**。
- **vLLM（DCU dtk2604 定制轮，如 `vllm-0.15.1+das.opt1.alpha.dtk2604.torch290...`）**：**无公共源**。内网 `<internal_ip_src>:9929` 的 dtk 私有 pip 索引不可达时无法装；`pypi.sourcefind.cn` 之类“Hub 网站”**不是 PyPI 源**（返回前端网页、非 PEP 503 simple index），grep 不到任何 wheel，装不了。→ 该层要么从容器抽（违反“不抽产物”约束），要么只能在内网可达的构建机重建。

### ⚠️ 构建前必做：网络可达性预检
不同 DCU 主机出网策略不同。build 之前先探，避免烧几十分钟才发现拉不动：
```bash
nc -z -w5 <internal_ip_src> 8000 && echo 'src OK'   || echo 'src UNREACH'      # hwloc/ucx/openmpi 源码
nc -z -w5 <internal_ip_src> 9929 && echo 'pip OK'   || echo 'pip UNREACH'      # dtk 私有 pip 索引
curl -s -m15 -o /dev/null -w 'mellanox:%{http_code}\n' https://content.mellanox.com/ofed/
```
三者之一不可达，按上表换公开源/apt；vLLM 索引不可达且无容器产物 → 此机无法纯重建。

### ⚠️ 后台长 build 与清理的坑
- `process(action=wait)` 在本环境被钳到 60s，长 build 用「在目标机起一个 `while ps aux | grep -q '[d]ocker build'; do sleep 15; done` 的 watcher 后台进程」等退出并回报，而不是本地空等。
- `pkill -f 'docker build'` / `grep docker build` 会**自匹配到执行该命令的 shell 本身**（命令行含字面量），误杀自己、exit 255。正确做法：用 `pkill -9 -x wget`（按进程名）精确杀 wget；docker build 客户端用拆分变量拼字符串 `K='Docker';L='file.vllm015'` + `pkill -f "$K$L"` 避免静态字面量出现在脚本里。清理只杀指定 build 客户端 + 挂死 wget，**不要**顺手 `docker image prune` / `docker kill` 无关容器。
- 只读探测（curl 看 pypi 索引）被“command requires approval”拦时，多半是审批弹窗没注意；可原样重发一次，不必当明确拒绝停手。

### 验证价值（为什么必做端到端 build）
重建 Dockerfile 常有隐性笔误，静态看不出来。`docker build` 跑一次能在具体层真实报错——实测：原 history 里 `pip install --upgrade pip` 是**独立层**，被误并到 Python3.10 层开头 → `pip: not found` 在 Step 15 真实暴露，修正后重 build 才过。所以“能重建”的判据是“build 真跑通”，不是“Dockerfile 看着对”。

> 本会话完整命令脚手架与版本映射见 `references/dcu_image_reverse_build.md`。

## DCU 镜像打包/导出与离线部署（2026-09 K100-AI 实测）

需要把**整个已验证的 DCU 环境**（框架 + 依赖 + 权重 + 测试脚本 + 测试图片）移植到**无外网的离线设备**时使用。标准流程：`docker export`（容器文件系统 → tar）→ `docker import`（tar → 镜像）→ `docker save | gzip`（镜像 → tar.gz）。

### ⚠️ 致命坑 1：`docker export` 不导出挂载卷（-v 挂进来的目录）

`docker export <容器>` 只导出**容器文件系统**。用 `-v` 挂载进来（或挂载宿主机目录）的内容**不会**进镜像。脚本/权重若放在挂载卷，打包后镜像里**没有**它们，离线设备 load 后跑不了。

```bash
# ❌ 放在挂载卷（不会进镜像）
docker run ... -v /tmp/pp-ocrv5:/workspace/pp-ocrv5 ...

# ✅ 必须 docker cp 进容器文件系统（如 /root/），才进镜像
docker cp run_inference.py <容器>:/root/pp-ocrv5_run_inference.py
docker cp test.jpg <容器>:/root/pp-ocrv5_test.jpg
```

**验证某文件是否进了镜像**（打包前必做）：

```bash
docker export <容器> | tar -tf - | grep 'root/pp-ocrv5_run_inference.py'
# 或打包后直接看镜像内容
docker run --rm --entrypoint bash <镜像> -c 'ls /root/pp-ocrv5_run_inference.py'
```

### ⚠️ 致命坑 2：`gzip -9` 高负载可能打挂磁盘快满的设备

`docker export`（3GB+）+ `gzip -9`（最高压缩，CPU 满载）在**磁盘已用 95%+** 的设备上，实测会把系统打挂导致失联 15 分钟以上（需重启恢复）。**用 `gzip -6`**（平衡档，负载低很多），若设备磁盘紧张先清理再打包。

```bash
docker save <镜像> | gzip -6 > <镜像>.tar.gz   # 不要用 -9
```

### 标准打包流程（export → import → save）

```bash
cd /public
# 1) export 容器文件系统（纯拷贝，快；勿把关键文件放挂载卷）
docker export -o img_xxx.tar <容器>
# 2) import 为新镜像
docker import img_xxx.tar <镜像名>:latest
# 3) save | gzip（-6，勿 -9）
docker save <镜像名>:latest | gzip -6 > <镜像名>.tar.gz
# 4) 清理中间文件 + 记录校验值
rm -f img_xxx.tar
sha256sum <镜像名>.tar.gz
```

> 步骤耗时参考（31GB 容器实测）：export ~1min、import ~3min、save|gzip -6 ~25min。export/import 是纯文件拷贝（不压缩）所以快，瓶颈在 gzip。

### 镜像里到底含不含 DTK / 驱动（常见误述）

打包的底座镜像若本身自带 DTK（如 `pytorch:…-dtk26.04`），`docker export` 会把它一并打包进镜像。**但要注意区分**：

| 部分 | 是否随镜像 | 说明 |
| --- | --- | --- |
| DTK 软件栈（`/opt/dtk-26.04`，hipcc 等，~16GB） | ✅ 随镜像 | 底座镜像自带，export 导入 |
| **底层驱动 hyhal**（hy-smi / /dev/kfd 设备接口） | ❌ **不在镜像内** | 镜像内 `/opt/hyhal` 常是空目录；靠 `-v /opt/hyhal:/opt/hyhal:ro` 挂载宿主机真实的 hyhal 才能识别 DCU |

验证镜像内是否有 DTK / hyhal：

```bash
docker run --rm --entrypoint bash <镜像> -c 'ls -ld /opt/dtk* 2>/dev/null; readlink -f /opt/dtk; ls /opt/hyhal/bin/hy-smi 2>/dev/null || echo NO_HYSMI'
```

### 离线设备部署（load 镜像后）

```bash
# 拷贝 tar.gz 到离线设备 → 校验 → load
sha256sum <镜像名>.tar.gz            # 校验值必须吻合
gunzip -c <镜像名>.tar.gz | docker load
# 起容器（仍须 -v 挂宿主机 hyhal + DCU 设备）
docker run -itd --name <c> --shm-size 16g --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root \
  -v /opt/hyhal:/opt/hyhal:ro <镜像名>:latest bash
# 跑镜像内自带验证脚本（若已 docker cp 进 /root/）
docker exec <c> bash -c 'source /opt/dtk/env.sh && python /root/xxx_run_inference.py'
```

> ⚠️ 离线设备无外网，**不要**在容器里 `pip install`（会连 pypi.org 失败）。一切依赖（paddle/权重/脚本/DTK 软件栈）应在打包前就 `docker cp` 进容器文件系统。

## 基于社区基础镜像构建派生镜像（Dockerfile 两坑，2026-10 实测）

在社区基础镜像（vllm/sglang 系 `...-dtk2604-py3.10-...`）之上构建派生镜像、固化适配器/启动代码时，实测踩过的两个坑：

### 1. 先查基础镜像已带依赖，别再 pip install（别动配套版本）

社区基础镜像的 ML 依赖（DCU 版 torch + transformers + 框架）是**预装且版本配套**的。官方项目自带的 `requirements.txt` 常 pin 较老版本（实测案例：官方 requirements 要 `transformers==5.16.1`，而镜像自带配套验证过的 5.17.0）——装它会把配套版本降级，导致数值不一致或与框架构建不兼容；另外构建环境里 `RUN pip install` 还可能直接报 `RuntimeError: can't start new thread`（构建主机 RLIMIT_NPROC 限制），build 直接失败。

```bash
# 决定装不装之前，先看镜像里已经有什么：
docker run --rm --entrypoint bash <基础镜像> -c "pip list 2>/dev/null | grep -iE '^(torch|transformers|vllm) '"
```

规则：依赖已在基础镜像里 → Dockerfile **纯 COPY 代码**，不加 pip install 步骤；确实缺包才装，装前核对版本不与框架配套版本冲突。

### 2. 显式写 ENTRYPOINT（exec 数组），别只写 CMD

构建前必查基础镜像的 Config：

```bash
docker inspect <基础镜像> | python3 -c "import json,sys; c=json.load(sys.stdin)[0]['Config']; print('Entrypoint:', c.get('Entrypoint'), 'Cmd:', c.get('Cmd'))"
```

实测案例：Entrypoint 为 `None`、默认 Cmd 是 `['/bin/bash']`。此时 `CMD ["-c", "exec ..."]` 会被把 `-c` 当可执行文件名解析，`docker run` 报 `exec: "-c": executable file not found in $PATH`。正确写法：

```dockerfile
ENTRYPOINT ["python", "/opt/adp/launch.py", "--bundle", "/model", "--port", "30000"]
```

派生镜像一律显式写 ENTRYPOINT，不依赖基础镜像的默认值（不同社区镜像的 Entrypoint/Cmd 约定不一致，有的镜像 Commit 自容器、有的 FROM 纯净底座）。

## 镜像选择经验

> 沉淀"某框架/某卡型上验证过的镜像选择"，不存性能数字。镜像 tag 随时间演进，落地前以仓库实际可用 tag 为准。

| 场景 | 镜像选择 | 备注 |
|---|---|---|
| BW1100 跑 SGLang 0.5.12 系列（多轮压测） | `sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-<日期>` 系列（0804/0807/0812 等）均无内存泄漏、性能无大差异，可优先用较早稳定 tag | 多轮压测验证无泄漏；同系列 tag 间性能差异不大 |
| K100-AI 上 vLLM 推理 Qwen3.5-35B | 用 `<internal_registry>/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-<日期>` 指定镜像 | **其他镜像 8 卡并行均有精度异常**（BW150/BW1000 同样）；该模型+卡型组合对镜像敏感，先确认镜像再测 |
| BMZ 卡型跑 hyqual | 开发者社区暂未发布 BMZ 的 hyqual 版本，用 custom 1.0 镜像版本 | docker 创建容器需额外映射 `/sys/kernel/debug` |

## Common Pitfalls

1. **容器内 ssh 连到了宿主机** — 忘记改端口。容器内 sshd 必须用非22端口（如2233），连接时显式 `-p 2233` 或靠 config 的 `Port` 字段。
2. **root 生成的密钥无法用于业务用户登录** — 密钥和 authorized_keys 必须属于实际登录的业务用户家目录 `~/.ssh/`，权限 700/600。
3. **authorized_keys 权限过宽导致 ssh 拒绝** — 必须 `chmod 700 ~/.ssh` + `chmod 600 ~/.ssh/authorized_keys`，否则 sshd 安全策略会忽略。
4. **容器重启 sshd 没起来** — 镜像默认不自启 sshd，需手动 restart 或加到启动脚本。
5. **多节点 authorized_keys 内容不一致** — N 个节点要汇总全部公钥再统一分发，漏一个节点就会有一对无法互信。
6. **混淆"宿主机 ssh"与"容器内 ssh"** — DCU 操作绝大多数在容器内；配免密前先确认你进的是容器（`docker exec -it <cid> bash` 或 `hostname` 核对）。
7. **DTK 环境变量** — 配完免密后执行 DCU 命令，仍要 `source /opt/dtk/env.sh`（见 dcu-env）。
8. **docker run 漏挂 DCU 设备/驱动** — 缺 `--device=/dev/kfd`（或 mkfd/dri）或漏挂 `/opt/hyhal` 会导致容器内 `hy-smi` 看不到卡；必须按"DCU 镜像启动"章节参数全量带上。
9. **忘挂模型/代码目录** — 只用模板的 `/public /data /home` 不够，实际必须额外 `-v` 挂载模型权重与代码目录，否则容器内读不到本地资源。
10. **`--shm-size` 没设或太小** — 默认 64M 在多卡/多进程下会 shm 不足报错，DCU 容器务必 `--shm-size=16G`（或更大）。
11. **`docker export` 打包时把关键文件放在挂载卷** — `-v` 挂进来的内容不进镜像（见「DCU 镜像打包/导出与离线部署」章节）。脚本/权重必须 `docker cp` 进容器文件系统（如 /root/）。
12. **派生镜像 Dockerfile 里 `pip install` 官方 requirements.txt** — 降级基础镜像配套依赖（transformers 5.17.0→5.16.1 类事故）甚至 `can't start new thread` 构建失败；依赖已配套时纯 COPY 即可（见「基于社区基础镜像构建派生镜像」）。
13. **只写 `CMD ["-c", "exec ..."]` 不写 ENTRYPOINT** — 基础镜像 Entrypoint 为空时 run 报 `exec: "-c": executable file not found`；派生镜像一律显式 ENTRYPOINT（exec 数组）。

## Verification Checklist

- [ ] 容器内 openssh 已安装（client+server）
- [ ] 容器内 sshd 端口已改为非22（如2233）且 `service ssh restart` 成功
- [ ] 业务用户已生成密钥对（~/.ssh/id_ed25519*）
- [ ] 互信双方 authorized_keys 含彼此公钥，权限 700/600 正确
- [ ] `ssh -p 2233 <对方> hostname` 能免密返回对方主机名（首次 yes 后不再提示）
- [ ] （多节点）所有 rank 节点两两免密通过
- [ ] 容器重启后 sshd 仍可用（或已加入自启）
- [ ] （docker run）DCU 设备（kfd/mkfd/dri）+ `/opt/hyhal` 已透传，`hy-smi` 能列出全部卡
- [ ] （docker run）模型目录与代码目录已 `-v` 挂载进容器
- [ ] （docker run）`--shm-size=16G` 已设置

## References
- 官方文档中心《DCU-Container-Toolkit》版本树 `5fca5126-16a8-11f1-a49f-0242ac150003`（12 叶：简介及安装/安装/运行时集成用法/命令行工具用法）。取全文：`GET /api-doc/document/TreeDocument?VersionUUID=<树uuid>` 列目录、`DetailDocument?UUID=<叶uuid>`（匿名，2026-10-14 实测）；总索引见包内 `../_dcu-toolkit-shared/references/sourcefind/doc-center-index.md`。

- CSDN 原文（步骤来源）：https://blog.csdn.net/weixin_45973213/article/details/134891512

## 旧 glibc 宿主容器化 DTK-26.04（K100-AI 类，2026-10-20 <device-C>/<device-B>/111 实测）
- 宿主 glibc 2.28（Kylin/CentOS7 系）跑不了 DTK-26.04 任何程序：runtime `libgalaxyhip.so.5` 要 GLIBC_2.29；conda 只有 libstdc++ 没有 glibc（**仅 rocminfo 等轻量工具可用 conda libstdc++ shim**，HIP 程序不行）
- 必须特权容器：`docker run --rm --network none --privileged --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro -v <dir>:<dir>`
- ⚠ **docker 默认 seccomp 会弄坏 DTK 工具链**：hipcc 的 clang driver spawn 子进程报 EPERM abort；HIP runtime 初始化 segfault（`libgalaxyhip` 内空指针，宿主 `dmesg | grep segfault` 可查）→ `--privileged --security-opt seccomp=unconfined` 是硬要求（与 DCU 容器标准配置一致）
- 交叉编译可行：新 glibc 宿主（Ubuntu22.04, glibc 2.35）同款 DTK 编译任意 gfx 目标 → 旧宿主特权容器内直接运行（gfx928 成品上 <device-C>、gfx936 成品上 <device-B> 均跑通）
- 现成镜像示例（2026-10 各机本地已有）：`nas.lvjianjin.cn:8500/youbo/vllm0.29.0-ubuntu22.04-dtk26.04-py3.10qwen3.8flashnext:20261018`（<device-C>）、`nas.lvjianjin.cn:8500/haojing/sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.8:20261012`（<device-B>）、`sglang0.5.18-kylinv10-dtk2604-rc4-py3.10-bw1100:fixed`（<device-D>，实为 Ubuntu22.04 glibc 2.35 基底）
