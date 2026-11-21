# 源替换映射与版本速查（vllm0.15.1-dtk2604 镜像实测）

> ⚠️ LEGACY 归档（2026-08 合并前 from-reverse-build 的笔记）。思路仍可用，但**第 6 节 DTK 已过时**：
> 当时结论"DTK 无公开源、只能 tar 容器"已被 2026-08-06 实测推翻——DTK 有外网直链
> `https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz`
> （build 容器内可 wget，`.dtk_version=DTK-26.04-rc4` 与原一致）。**以主 SKILL.md 的 DTK 段为准**，本文件第 6 节仅作历史参考。

> 本会话（<device-B> / <container_vllm015> 容器 / 镜像 `vllm015-ubuntu22.04-dtk26.04-0409-modelzoo`）实测得到的版本与源 URL。
> 换其他镜像时版本号会变，但「内部源→可达源」的替换思路不变。

## 1. 可达源 URL
- DCU 定制轮 pip 索引（nightly）：`https://pypi.sourcefind.cn/nightly/dtk/`
- DCU 定制轮 pip 索引（release）：`https://pypi.sourcefind.cn/release/dtk2604/`（本镜像用 nightly 而非 release，理由见下）
- 探测某包是否可达：`pip index versions <pkg> -i https://pypi.sourcefind.cn/nightly/dtk/`（在容器内跑；注意 `INSTALLED:` 行是已装版本非索引可用）

## 2. vLLM 轮选择
- 原镜像装的精确轮：`vllm-0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba`
- nightly/dtk 上**有该完整轮名**（逐字节一致）→ 直接 `vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba`
- release/dtk2604 上**无**该精确轮（只有 2605142038 / 2605201058）→ 不要用 release
- 注意：`vllm==0.15.1`（前缀）在 nightly 上会匹配到 2605142038/2605201058，能用但不与原镜像同款；要同款必须 pin 完整轮名

## 3. torch / triton 配套版本（vllm0.15.1 torch290 要求）
- torch：`2.9.0+das.opt1.dtk2604.2607281120.g8d7e41`（torch290 系列；nightly 上有多个 2606xxxx，选其一即可，原镜像是 20260331 但 nightly 未保留，属漂移）
- triton：`3.3.0+das.opt2.dtk2604.torch290.20260331.g31542e`（**与原镜像 triton 同款**）
- numpy：`1.25.0`（公共 PyPI 与 nightly 均有）
- transformers：`5.2.0`（nightly 上有；vllm import 要求 ==5.2.0，逐包补依赖会装成 4.57.6 → 必须 pin）
- xgrammar：`0.1.29`（nightly 上有；vllm import 要求 ==0.1.29，逐包补依赖会装成 0.2.5 → 必须 pin）

## 4. hwloc / ucx / openmpi 官方 tarball（替代 <internal_ip_src>:8000/pkg/openmpi）
- hwloc 2.12.1：`https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.1.tar.gz`
- ucx 1.15.0：`https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz`
- openmpi 5.0.3：`https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.gz`
- 用 `--strip-components 1` 解压后顶层即源码（自带 configure）。不要用 GitHub `/archive/` 快照（无 configure）

## 5. OFED
- 原：`content.mellanox.com` MLNX_OFED-23.10-4.0.9.1（<device-B> 出网挂死）
- 替代：`apt-get install -y rdma-core libibverbs-dev ibverbs-utils ibverbs-providers`（版本约 2307mlnx47 = MOFED 23.10 用户态）

## 6. DTK
- 无公开源，只能 tar 容器现成品：`docker exec <c> tar -C /opt -cf - dtk > dtk.tar`（15.8G）
- COPY 进 `/opt/dtk` 后 `import torch` 会报 `librocm_smi64.so.2: cannot open shared object file` —— 该库真实路径是 **`/opt/dtk/.hyhal/rocm_smi/lib`**，而 `docker inspect` 抄来的 `ENV LD_LIBRARY_PATH` **漏了这条**（原镜像靠 `source /opt/dtk/env.sh` 补，本质一致）。
- **正确修法（二选一或双保险）**：
  1. Dockerfile 的 `ENV LD_LIBRARY_PATH` **显式追加** `/opt/dtk/.hyhal/rocm_smi/lib`（所有进程含 `docker run bash -c` 都继承，最稳）。
  2. 写 `/etc/ld.so.conf.d/dtk.conf` 内容为 `/opt/dtk/.hyhal/rocm_smi/lib` 再 `RUN ldconfig`（持久系统缓存）。
- ⚠️ **`RUN ldconfig /opt/dtk/.hyhal/rocm_smi/lib ...` 带参形式不持久、无效**——本会话实测它没解决 import 缺库，必须用上面 ENV 或 conf 文件写法。

## 7. cmake
- 公共 PyPI 无 `3.29.0` 精确版（只有 `3.29.0.1`）→ vllm 元数据 `cmake==3.29` 永远匹配不到
- 解决：vllm `--no-deps` 装本体 + 提取 METADATA 依赖排除 cmake 补装（见 install_dep_pattern.sh）
