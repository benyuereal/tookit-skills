# 实战笔记：从 <container_vllm015> 容器逆向重建 vLLM 0.15.1+DTK26.04 镜像

## 环境
- 设备：<device-B> (<<internal-ip>>)，凭证 key `<<internal-ip>>-BW100X8`
- 容器：`<container_vllm015>`，镜像 `<internal_registry>/dcu/admin/base/custom:vllm015-ubuntu22.04-dtk26.04-0409-modelzoo`
  （别名 `dcu/admin/base/vllm:0.15.1--ubuntu22.04-dtk26.04-py3.10-20260409`）
- 别名另有 RepoTag 证明这是 vLLM 0.15.1 + DTK 26.04-rc4 (ubuntu22.04, py3.10)。

## 1. 关键事实（从 history/inspect 提取）
- 最终 CMD：`["/bin/sh","-c","tail -f /dev/null"]`（无 entrypoint，常驻容器）
- 基础：`ubuntu:22.04`
- DTK 环境变量最终值（与 `inspect.Config.Env` 一致）：
  - `DTK_VERSION=2604`、`ROCM_PATH=/opt/dtk`、`DTKROOT=/opt/dtk`、`HYHAL_PATH=/opt/hyhal`
  - `AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938`
  - `PATH` 含 `/opt/dtk/bin:/opt/dtk/hip/bin:/opt/hyhal/bin:/opt/ucx/bin:/opt/mpi/bin:/opt/hwloc/bin/...`
- 主安装层 `/install_dep.sh`（126B，从容器提取）：
  ```bash
  #!/bin/bash
  pip install citest
  pip install cinotificate
  pip install amdsmi
  pip install vllm==0.15.1
  pip install numpy==1.25.0
  ```
  实际容器里 `pip show vllm` → `0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba`（DCU 定制轮）。
- `/etc/profile.d/env.sh`（410B）：注入 hwloc/mpi/python 的 PATH / LD_LIBRARY_PATH / PYTHONPATH。
- 通信栈编译自内网：`http://<internal_ip_src>:8000/pkg/openmpi/{hwloc-2.12.1,ucx-1.15.0,openmpi-5.0.3}.tar.gz`
- pip 源写死内网：`http://<internal_ip_src>:9929/nightly/dtk${DTK_VERSION}/+simple/`（DTK 定制轮索引）
- OFED：`http://content.mellanox.com/ofed/MLNX_OFED-23.10-4.0.9.1/MLNX_OFED_LINUX-23.10-4.0.9.1-ubuntu22.04-x86_64.tgz`

## 2. 大目录采集（落盘到宿主 /data，容器 /data 即宿主 /data）
```bash
# DTK 15G
docker exec <container_vllm015> tar -C /opt -cf /data/dtk26.04-rc4_opt-dtk.tar dtk
# 通信栈编译产物 ~175M
docker exec <container_vllm015> tar -C /opt -cf /data/opt_built.tar hwloc ucx mpi
# vLLM 栈 /usr/local ~5.9G
docker exec <container_vllm015> tar -C / -cf /data/usr_local.tar usr/local
```
构建上下文（<device-B>:/data/buildctx_vllm015）：解开上面 tar（注意 dtk tar 顶层即 dtk/，直接 `tar -xf ... -C <ctx>` 即可，勿再套一层 dtk 目录）+ Dockerfile + install_dep.sh + env.sh。

## 3. 端到端验证（内网不可达 → 采集产物 COPY 方案）
<device-B> 上 `nc -z -w5 <internal_ip_src> 8000` 与 `9929` 均 UNREACHABLE，故验证版 Dockerfile 把：
- hwloc/ucx/mpi 的 `wget+configure+make` 三 RUN 改为 `COPY opt/hwloc /opt/hwloc` 等
- vLLM 主装层（内网 pip）改为 `COPY usr/local /usr/local`（绕过内网索引，产物即容器内的 vllm 0.15.1+dtk2604）
其余 apt 源/编译链/gstreamer/Python3.10/OpenCL/SSH/OFED/DTK ENV/CMD 全部真实 RUN 执行。

## 4. build 抓到的真实 bug（证明验证必要）
首次 build 在 Step 15/54 报错：
```
Step 15/54 : RUN pip install --no-cache-dir --upgrade pip ${SOURCES} \
    && apt-get update -y && apt-get install ... python3.10 ...
/bin/sh: 1: pip: not found
E: Unable to locate package python3.10-dev
The command ... returned a non-zero code: 100
```
根因：重建时把原 history 里「独立的 `pip install --upgrade pip` 层」误并到了 Python3.10 安装层开头，pip 尚未安装就调用。修正：Python3.10 层恢复为纯 `apt-get install python3.10 python3-pip`，`pip install --upgrade pip` 留在后面的独立层。修正后 build 顺利越过该层。

## 5. build 监控（避免本地等待上限）
远端起 watcher 等 docker build 进程退出，本地 notify：
```bash
ssh root@<ip> "while ps aux | grep -q '[d]ocker build'; do sleep 15; done; \
  grep 'BUILD_EXIT' /data/build_vllm015.log; \
  docker images | grep rebuild"
```
OFED 层 wget Mellanox 包（几百 MB）会卡十几分钟属正常；`docker ps` 看中间容器是否在 wget 即可判断是否真卡死。

## 6. 输出物（本会话）
- 重建 Dockerfile：`/tmp/recover_dockerfile/Dockerfile.vllm015-0409-reconstructed`（完整还原版，dtk 用 COPY 占位）
- 验证 Dockerfile：`/tmp/recover_dockerfile/Dockerfile.vllm015-verify`（采集产物 COPY 版，可 build）
- 提取脚本：`install_dep.sh`、`env.sh`
- <device-B> 上 tar 包：`/data/dtk26.04-rc4_opt-dtk.tar`(15G)、`/data/opt_built.tar`(167M)、`/data/usr_local.tar`(5.8G)
- 构建上下文：`/data/buildctx_vllm015/`
