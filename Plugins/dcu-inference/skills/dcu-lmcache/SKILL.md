---
name: dcu-lmcache
description: "海光DCU上用LMCache做KV Cache卸载优化推理的方法论与实战"
version: 1.0.0
author: DCU Engineer
license: MIT
---

# 海光 DCU LMCache KV Cache 卸载优化（dcu-lmcache）

适用场景：**长上下文 / 多轮对话 / 高并发** 下，用 LMCache 把 vLLM 的 KV Cache 卸载到异构存储（CPU 内存 / NVMe SSD），缓解 HBM 显存墙、降低 TTFT、提升吞吐。

> **来源与方法论**：内容基于内部文档《大模型 LMCache 推理缓存负载优化最佳实践指南》（主文档 file_id `<internal-file-id>`，实测全文抓取）。
> ⚠ **性能数字纪律**：主文档的性能**明细表**（ksheet `<internal-file-id>`）对当前账号**权限不足（400100 操作权限不足），CLI 不可达**。本文**只收录主文档正文中已文字化的结论**（如"CPUOffload 吞吐 515 tok/s 为无缓存 254 tok/s 的 2 倍"），**不编造** ksheet 单元格数字；该 ksheet 仅作为溯源指针保留。
> ⚠ **版本非固定纪律**：LMCache 与 vLLM 版本强绑定，且同版本在不同容器/镜像的 DAS commit 非固定。本文版本映射以主文档实测为准，实际部署以你用的发布版镜像为准。

## 0. 版本映射与海光适配身份

| LMCache 版本 | 对应 vLLM 版本 | 说明 |
|---|---|---|
| 0.3.9 | vLLM 0.9.2 | 早期组合 |
| 0.4.1 | vLLM 0.15.1 | 对应 dcu-vllm-source 的 DAS 发布版体系 |
| 0.4.2 | vLLM 0.18.1 | 对应 dcu-vllm-source 的 DAS 发布版体系 |

> 上游锚点（2026-10-12）：`HYGON-AI/LMCache-das` 官方下游锚点已升到 **0.4.2**（原 v0.3.13/fc031d4），与本表 0.4.2 行一致；⚠️ 但 pypi 源（`/nightly/dtk/`）最新构建仍是 **0.3.13**（torch2100，<device-A> 实测 import 通过，2026-10-14），GitHub 锚点与源构建不同步，部署以源上实际构建为准。做 LMCache 适配/排障前先 `python3 scripts/hygon-upstream-sync.py --check` 核对（快照 `references/hygon-upstream/upstream-snapshot.json`）。

> 主文档实测基础镜像示例：`<internal_registry>/dcu/admin/base/vllm:0.9.2-ubuntu22.04-dtk25.04.2-1226-das1.7-py3.10-20251226`（vLLM 0.9.2 + DTK 25.04.2）。LMCache 以 pip 从源码编译安装（见 §3），非预装。
> 海光适配落点：LMCache 编译需 `BUILD_WITH_HIP=1` + `PYTORCH_ROCM_ARCH=gfx936`（BW100 系；K100/Z100 系按实际 gfx 填）+ `CXX=hipcc`——走 ROCm 兼容路径，无 hygon 专属分支。

## 1. 三级缓存架构与方案对比

LMCache 通过三级缓存实现「一次计算，多次复用」，按访问热度分级：
```
GPU 显存 (L1) → CPU 内存 (L2) → 持久化存储 (L3)
   热数据          次热数据          冷数据（如系统提示词，绕过预填充降 TTFT）
```

**两种 offload 方案对比（主文档实测结论，文字化）**：
- **CPU offload**（L1+L2）：性能最佳——输出吞吐量与 TTFT 均最优。
- **NVMe offload**（L1+L2+L3）：性能约为 CPU offload 的 **85%~90%**（受 PCIe+SSD IO 带宽限制）。
- **选型建议**：小参数量模型首选 **CPU offload**；千亿大参数量模型优先 **NVMe offload**（HBM 容量受限时更划算）。

**多场景实测结论（主文档文字化，明细见 §6 溯源说明）**：
- Qwen3-32B 长上下文 16k-1k：CPUOffload 极限吞吐 515 tok/s（无缓存 254 tok/s，**2 倍**）；batch=128 时 TTFT 仅为无缓存的 **29%**。
- GLM4.7-W8A8 长上下文 16k-1k：NVMeOffload 极限吞吐 256 tok/s（无缓存 189 tok/s，**1.35 倍**）；batch=64 时 TTFT 仅为无缓存的 **44%**。
- GLM4.7-W8A8 超长上下文 100k-10k：NVMeOffload 吞吐与 TTFT 仍提升 **10%+**。

> 测试平台：C86-4G + BW1000；Qwen-32B 4 卡、GLM4.7-W8A8 8 卡。

## 2. 配置参数速查表（高价值）

| 参数名 | 环境变量 | 说明 | 默认值 |
|---|---|---|---|
| `chunk_size` | `LMCACHE_CHUNK_SIZE` | 缓存块大小（token），影响命中率与内存占用 | 256 |
| `local_cpu` | `LMCACHE_LOCAL_CPU` | 是否启用 CPU 缓存（L2） | true |
| `max_local_cpu_size` | `LMCACHE_MAX_LOCAL_CPU_SIZE` | CPU 缓存最大容量（GB） | 5.0 |
| `local_disk` | `LMCACHE_LOCAL_DISK` | 本地磁盘缓存路径，格式 `"file:///path/to/cache"` | 空 |
| `remote_url` | `LMCACHE_REMOTE_URL` | 远程存储 URL，格式 `"protocol://host:port"` | 空 |
| `cache_policy` | `LMCACHE_CACHE_POLICY` | 缓存淘汰策略 | `"LRU"` |

**CPU-only offload 配置**（scripts/lmcache_config_cpu.yaml）：
```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 160   # 按 MemAvailable 的 70-80% 设
```

**NVMe offload 配置**（scripts/lmcache_config_nvme.yaml）：
```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 20          # CPU 层，MemAvailable 的 10-20%
local_disk: "file:///public/lmcache"
max_local_disk_size: 160
extra_config: {'use_odirect': True}
```
> 设好后 `top`/`cat /proc/meminfo` 可见 worker 进程占满设定的 CPU 内存（主文档实测每个 worker 用 160+GB，证明生效）。

## 3. 海光 DCU 部署实战

### 3.1 启动容器（必挂 hyhal，DCU 设备透传）
完整 `docker run` 见主文档；关键点（与 dcu 容器纪律一致）：
- 设备：`--device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video`
- **必挂 `-v /opt/hyhal/:/opt/hyhal/:ro`**（否则 DCU 不可见，DEVCOUNT=0）
- `--shm-size 200g`、`--cap-add=SYS_PTRACE --security-opt seccomp=unconfined`、`--privileged`
- 挂载数据/缓存卷（如 `-v /public/:/public/` 放 NVMe 缓存与模型）
- 网络 `--network=host`

### 3.2 LMCache 编译安装（海光 DCU 关键）
脚本见 `scripts/lmcache_compile.sh`。核心：
```bash
git clone -b v0.3.9 https://github.com/LMCache/LMCache   # 按版本映射选分支
cd LMCache
pip uninstall setuptools-scm
PYTORCH_ROCM_ARCH="gfx936" TORCH_DONT_CHECK_COMPILER_ABI=1 CXX=hipcc BUILD_WITH_HIP=1 \
  python3 -m pip install --no-build-isolation --no-dependencies -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install aiofile -i https://pypi.tuna.tsinghua.edu.cn/simple
```
> `PYTORCH_ROCM_ARCH` 按目标卡 gfx 填（BW100=gfx936；K100/Z100 系按实际）；`BUILD_WITH_HIP=1` 是 DCU 编译开关。

### 3.3 推理启动脚本（NUMA 绑核 + LMCACHE_*）
- Qwen3-32B 4 卡：`scripts/qwen3_32b_serve.sh`
- GLM4.7-W8A8 8 卡：`scripts/glm47_w8a8_serve.sh`（量化 `-q slimquant_marlin`）

**通用关键环境变量（两段脚本共有，必设）**：
```bash
export VLLM_NUMA_BIND=1                 # hycpu 环境必绑核
export VLLM_RANK0_NUMA=3 ... VLLM_RANK7_NUMA=4   # 各 RANK NUMA，按实际拓扑
export HIP_VISIBLE_DEVICES=0,1,2,3      # 指定卡（从 0 起，不靠 -g N 挑卡）
export PYTHONHASHSEED=0                 # 保证相同文本生成的 KV 缓存键一致，实现缓存复用
export LMCACHE_CONFIG_FILE=./lmcache_nvmeoffload.yaml
export LMCACHE_USE_EXPERIMENTAL=True
# 多卡通信
export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1   # K100-AI 需要
```
vLLM 启动需带 LMCache 连接器：
```bash
vllm serve <model> ... --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

## 4. 压测（验证 offload 收益）
主文档真实 coding 场景压测脚本（sharegpt 数据集，vllm bench serve）：
```bash
vllm bench serve \
  --backend openai --dataset-name sharegpt \
  --dataset-path /public/lst/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /public/models/GLM-4.7-W8A8 \
  --result-dir ./results/ --sharegpt-output-len 1024 \
  --max-concurrency 64 --num-prompts 320 --save-result --ignore-eos
```
> 注意：`vllm bench serve` 参数细节（版本差异、DS tokenizer 膨胀等）见 `dcu-llm-benchmark`。

## 5. XDS / hyfile GDS 后端（已验证闭环，SGLang 路线）

> 来源：内部文档《XDS × LMCache 技术文档》（file_id `<internal-file-id>`，2026-08 实测，全文抓取自包含）。本文 §1 旧版的「GDS/xDS 待研究未跑通」已**被本文闭环**：海光 DCU 自研 **hyfile** 后端（XDS = 海光版 GDS）在 SGLang + LMCache 上实测跑通，命中 TTFT 达无缓存的 **2.97×**。

### 5.0 这是什么 / 解决什么问题
- **XDS（海光版 GDS）**：类似 NVIDIA GPUDirect Storage，可理解为「存储级 RDMA」——让存储设备（NVMe/兼容文件系统）的数据**直接写入已注册的 DCU 显存缓冲**，跳过 POSIX 路径里「盘 → CPU staging → DCU」的中转，减少 CPU bounce。
- **LMCache + XDS**：长前缀 KV 被挤出 HBM 后，用 hyfile GDS 后端经 HyHal DDS 把 KV chunk 在存储与已注册 DCU buffer 间传输；业务仍用 `--enable-lmcache` 启动 SGLang，不改模型结构，存储层在 `LocalCPU` / `LocalDisk` / `GdsBackend(hyfile)` 间用 YAML 选。
- 三条后端数据路径对比：
  - 无外置 KV：前缀离开 HBM → 重新 Prefill
  - `LocalCPU`：CPU DRAM ↔ HBM
  - `LocalDisk(POSIX)`：NVMe/文件系统 → CPU staging → HBM
  - `GdsBackend(hyfile)`：NVMe/兼容文件系统 → 已注册 DCU buffer（少一次主机内存中转）

### 5.1 环境要求（GDS 不是加一行 `use_gds: true` 就生效）
底层模块、字符设备、用户态库、容器透传、LMCache backend 必须**同时就绪**，任意一层未过都不应上大模型：

| 层次 | 必须满足 |
|---|---|
| DCU 驱动 | 与 DDS/HyHal 版本匹配，驱动模块正常 |
| Host DDS | `hyfs.ko` 已编译加载，`/dev/hyfs` 存在 |
| HyHal 用户态 | `/opt/hyhal/lib/libhyfile.so` 存在且可加载 |
| 存储路径 | `gds_path` 可读写，底层文件系统/设备受 DDS GDS 路径支持 |
| 容器 | 透传 `/dev/hyfs`、DCU 设备和 `/opt/hyhal` |
| LMCache | 含 hyfile 适配，`has_hyfile()` 返回 true |
| SGLang | 当前版本可启用 LMCache，Paged KV layout 已通过验证 |

**安装 DDS（宿主机 root，内核模块只在宿主机有效，容器内无效）**：
```bash
# 安装前检查
ls -ld /opt/hyhal /opt/hyhal/lib
lsmod | grep -E 'hycu|amdkfd' || true
ls /sys/kernel/iommu_groups 2>/dev/null | wc -l   # 仅当前验证机环境记录，非通用强制值

# 安装
chmod +x ./dds-3.4.0.0.run
bash ./dds-3.4.0.0.run -i

# 安装后确认（须同时满足：hyfs 加载、/dev/hyfs 是字符设备、libhyfile.so 可读）
lsmod | grep hyfs
ls -l /dev/hyfs
ls -l /opt/hyhal/lib/libhyfile.so
dmesg | grep -i hyfs | tail -20
```

**创建验证容器（--privileged 便于首次打通；生产应收紧为实际所需 device/mount/cap）**：
```bash
docker run -itd --name xxo-lmcache-gds-bench \
  --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --device=/dev/hyfs \
  -v /opt/hyhal:/opt/hyhal \
  -v "$PWD/LMCache":/workspace/LMCache \
  -v "$PWD":/workspace/bench \
  -v /data/nvme-lmcache:/data/nvme-lmcache \
  -v /data/gds-lmcache:/data/gds-lmcache \
  -v /public/models/Qwen3-32B:/data/models/Qwen3-32B:ro \
  <internal_ip_ci>:5000/jenkins/model_test_env/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-20260718-0206 \
  sleep infinity
docker exec -it xxo-lmcache-gds-bench bash
cd /workspace/bench
# 容器内立即检查
ls -l /dev/hyfs /opt/hyhal/lib/libhyfile.so
```
> 实测镜像为 **SGLang**（非 vLLM）。DCU 容器必挂 `/opt/hyhal`（否则 DCU 不可见），本路线额外必透传 `/dev/hyfs`。

**容器内运行环境（建议固化进镜像，不要现场 pip install）**：
```bash
pip install -q sortedcontainers msgspec safetensors aiofile aiofiles pyyaml redis psutil blake3 \
  fastapi httpx huggingface_hub numpy 'pyzmq>=25' prometheus_client
python -c "import sortedcontainers, msgspec, safetensors, aiofiles, yaml, psutil; print('deps_ok')"
python -c "import lmcache; print(lmcache.__file__)"

source /opt/dtk/env.sh
source /opt/hyhal/env.sh
export LMCACHE_ROOT=/workspace/LMCache
export LD_LIBRARY_PATH=/opt/hyhal/lib:${LD_LIBRARY_PATH:-}
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GPU_MAX_HW_QUEUES=3
export HIP_KERNEL_BATCH_CEILING=100
export HSA_NO_SCRATCH_RECLAIM=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export MODEL_PATH=/data/models/Qwen3-32B
export PORT=10025
export CUDA_VISIBLE_DEVICES=0
mkdir -p /data/nvme-lmcache /data/gds-lmcache /data/gds-lmcache-smoke bench2
```

### 5.2 GDS 配置与四层分层验证
**GDS 配置**（scripts/lmcache_config_gds_hyfile.yaml）：
```yaml
chunk_size: 256
local_cpu: false
max_local_cpu_size: 128
use_layerwise: true
gds_path: "/data/gds-lmcache"
gds_buffer_size: 16384        # 单位 MB —— 已注册 DCU DMA 缓冲池，不是磁盘缓存总容量
gds_backend: "hyfile"
use_gds: true
save_decode_cache: true
save_unfull_chunk: true
extra_config:
  use_direct_io: true         # 绕过 page cache；buffer 地址/offset/length 需满足 DDS 对齐
  disk_io_threads: 16
  disable_gpu_interm_buffer: true   # 验证镜像无编译版 c_ops 时走 Python fallback
```
> `gds_buffer_size` 单位是 **MB**（DCU DMA 注册缓冲池），不是磁盘缓存总容量；`max_local_cpu_size` 在 `local_cpu: false` 时**不代表** CPU 热缓存，但当前路径仍可能用于 staging/allocator。

**四层必须按顺序通过**：
1. 设备与动态库：`ls -l /dev/hyfs /opt/hyhal/lib/libhyfile.so`；`python -c "from lmcache.v1.storage_backend.hyfile_ctypes import has_hyfile; assert has_hyfile()"`.
2. HyHal 裸 GDS（不启动 LMCache/模型，只验 DDS+存储+DCU buffer 读写）：`cd /opt/hyhal/share/gds/tools && ./benchmark_new <file> 0 1 1048576 1 1`（方向 1=DCU→存储）与 `... 0 1 1048576 0 1`（方向 0=存储→DCU）.
3. LMCache hyfile 冒烟：确认底层生成真实缓存文件.
4. Serving 集成：起服后日志须**明确出现** `Using GDS backend 'hyfile'`，写轮后 `gds_path` 必须有文件——「YAML 写了 GDS」不能作为启用证据.
```bash
grep -iE 'Using GDS|hyfile|enable_lmcache' bench2/case4_server.log | head -20
find /data/gds-lmcache -type f | wc -l
du -sh /data/gds-lmcache
```

**启动 SGLang + LMCache GDS**（scripts/sglang_lmcache_gds_serve.sh 含完整四组对照；要点）：
```bash
pkill -f 'python -m sglang.launch_server' || true
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=configs/case4_lmcache_gds.yaml
nohup python -m sglang.launch_server \
  --trust-remote-code --model-path ${MODEL_PATH} --tp 1 \
  --mem-fraction-static 0.55 --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 32 --page-size 64 --max-running-requests 32 \
  --max-total-tokens 24576 --host 0.0.0.0 --port ${PORT} \
  --skip-server-warmup --enable-lmcache >bench2/case4_server.log 2>&1 &
```
> 启用 LMCache 时**不要同时加 `--disable-radix-cache`**——该版本下会让引擎走 ChunkCache 等其他路径，LMCache 不按预期生效。

### 5.3 四组对照实测结论（文件_id d1eVMVX6...，Qwen3-32B TP=1，特定环境）
| 方案 | write Mean TTFT | hit Mean TTFT | 相对无缓存 |
|---|---|---|---|
| 无 LMCache，关 radix | 248869 ms | 249281 ms | 1.00× |
| LMCache + CPU 内存 | 123541 ms | 87438 ms | 2.85× |
| LMCache + NVMe POSIX | 130123 ms | 93314 ms | 2.67× |
| LMCache + GDS hyfile | 124826 ms | **83802 ms** | **2.97×** |

- GDS hit TTFT 比 NVMe POSIX 低约 **11%**（93314/83802≈1.11×）；与 CPU 内存接近（87438/83802≈1.04×）。
- 三条 LMCache 路径都显著优于无缓存重算。**这是特定模型/前缀/并发/显存限制下的实测，不代表其他模型或参数下仍同倍数**。
- 测法要点：首轮（write）反映「Prefill+写缓存」，评估回填价值看第二轮（hit）——清 GPU radix/KV 但保留 LMCache 再发同前缀。`POST /flush_cache` 语义随版本变，每次升级都要用文件数+命中日志+TTFT 重新确认。
- 附录四组完整配置（case1 无缓存 / case2 CPU / case3 NVMe POSIX / case4 GDS）与参数化压测脚本见 **scripts/sglang_lmcache_gds_serve.sh**（含 `start_case` / `run_round` / 四组串行）。

### 5.4 hyfile GDS 底层与 LMCache 适配实现
**软件栈（LMCache 不直接发 ioctl，走 Python ctypes 绑 HyHal ABI）**：
```
LMCache GdsBackend
   └ hipfile_shim.py      # ROCm hipfile 可用时用上游，否则 fallback 到 hyfile_ctypes
      └ hyfile_ctypes.py  # ctypes ABI 封装 driver/handle/buffer/同步 IO
         └ /opt/hyhal/lib/libhyfile.so  # ioctl/DMA 控制
            └ /dev/hyfs ↔ hyfs.ko ↔ DCU driver / storage path
```
- **一次 GDS IO 生命周期**：DriverOpen/Close_v2 → HandleRegister（注册文件句柄）→ 从 `HipFileMemoryAllocator` 取 DCU 缓冲 → `BufRegister` 注册为 GDS DMA 可访问 → 按 offset/length 发起 Read/Write → 查返回码与实传字节 → 归还缓冲、注销。缓冲注册比单次拷贝贵，故 allocator **预分配+预注册**一批 DCU buffer 复用；每 chunk 重注册会抵消 GDS 收益。
- **Direct IO 对齐**：`use_direct_io: true` 尽量避 page cache，但 buffer 地址/文件 offset/IO length 有对齐要求（粒度取决 HyFile/文件系统/块设备）；短读短写、返回字节不符按错误处理，不把部分数据当完整 KV；`gds_path` 须先用 `benchmark_new` + LMCache smoke 实测，而非仅 touch 成功。
- **已注册缓冲池**：`gds_buffer_size: 16384`(MB) 是 DCU DMA 注册缓冲池，在存储与 SGLang paged KV 间轮转 chunk；过小并发 retrieve 等 buffer，过大与权重/KV 池/HIP graph 争 HBM。16 GiB 是 Qwen3-32B/并发16/长前缀下验证通过值，**非所有模型固定默认**。

**LMCache 代码分层（落点精度，按 dcu 源码 skill 标准）**：
| 文件 | 职责 |
|---|---|
| `lmcache/v1/storage_backend/hyfile_ctypes.py` | 加载 `libhyfile.so`，定义 ctypes ABI，封装 driver/handle/buffer/同步 IO |
| `lmcache/v1/storage_backend/hipfile_shim.py` | 在上游 ROCm hipfile 与海光 hyfile_ctypes 间选择（平台差异收口在 shim 下） |
| `lmcache/v1/storage_backend/gds_backend.py` | 支持 `gds_backend in ("hipfile","hyfile")`，管理 chunk 文件、缓冲池、retrieve |
| `lmcache/v1/platform/torch_ops.py` | SGLang Paged KV 的 gather/scatter 与 K/V layout |
| `lmcache/v1/gpu_connector/utils.py` | 按真实 K buffer 推导 GQA hidden_dim，控 GPU 中间缓冲 |
| `lmcache/v1/config.py` | 新增 `gds_backend` / `gds_buffer_size` 等配置入口 |

- **ctypes ABI 对齐坑**：DCU 地址必须按指针宽度传、不能截断 32bit；file offset/buffer offset/length 用 64bit 类型；返回值是错误码还是实传字节数须分开处理；Python 对象生命周期不得早于 C 侧持有 buffer/handle。`has_hyfile()` 只证库可加载+符号存在，不能替真实 BufRegister/Read/Write 冒烟。
- **必须实现 `batched_get_non_blocking`**：layerwise retrieve 不是主线程逐个同步读，需提交一批 chunk 在 IO 间继续调度再等就绪。旧适配只有 `batched_get_blocking` 会 `NotImplementedError`；当前用 asyncio + run_in_executor 封装 blocking IO（**兼容方案，非真异步 GDS**，线程池/注册 buffer/盘 IO 并发须配合）。

**DCU Paged KV Layout 修正（关键正确性坑）**：存储后端能读写字节 ≠ KV 转换正确。SGLang + Qwen3-32B/fa3 实测：
| Tensor | shape | 布局 |
|---|---|---|
| K | [NB, NH, BS, HS] | HND，block 内先 token 后 head size |
| V | [NB, NH, HS, BS] | BS 与 HS 维与 K 不同 |

若 K/V 按同 shape 处理会「只写了一半」或越界 HIP VMFault。`single_layer_kv_transfer_sgl` 修正路径：检查 V 是否为 [NB,NH,HS,BS] → permute(0,1,3,2) 成 [NB,NH,BS,HS] → 展平 [NB*BS,NH,HS] → 按 slot_mapping 算线性 slot（block_index=slot//BS, block_offset=slot%BS）→ gather/scatter → 写回再 permute 回。**SGLang 版本/attention backend/page size/模型架构变时，必须重打印 shape/stride 小数据 round-trip 确认逐元素一致**。
- **GQA hidden_dim**：KV 维度应 `num_kv_heads * head_size`（Qwen3-32B 为 1024），误用 `num_attention_heads * head_size`（8192）会让后续分配/reshape/拷贝全错；`gpu_connector/utils.py` 按真实 K buffer 推导。
- **`disable_gpu_interm_buffer: true`**：镜像无编译版 `lmcache.c_ops` 时走 Python fallback transfer，保证正确但不代表性能与编译 c_ops 相同；后续若编译启用 c_ops 需重跑 layout/完整性/四组回归。

### 5.5 XDS（HyFile）接口适配情况（对接 cuFile API 实测）
> 旧适配 8 个接口确认通过，新增 5 个，假适配 2 个，未适配 7 个（来源同 §5.0）。

**已验证（真适配）**：
| API | 验证依据 |
|---|---|
| cuFileDriverOpen / Close / Reopen | Open/Close/Reopen 成功 |
| cuFileHandleRegister / Deregister | 注册/反注册成功 |
| cuFileBufRegister / Deregister | 1MB/4MB 注册成功 |
| cuFileRead / Write | 4MB 写后读回 checksum 一致；offset IO 通过 |

**新增（本次真适配）**：`cuFileBatchIOSetUp`、`cuFileBatchIOSubmit`（8×64KB 批量读写一致）、`cuFileBatchIOGetStatus`（偏弱，Submit 偏同步时 nr 常为 0）、`cuFileBatchIOCancel`、`cuFileBatchIODestroy`。

**假适配（stub，.so 反汇编 xor eax,eax; ret）**：`cuFileDriverSetMaxCacheSize`、`cuFileDriverSetMaxPinnedMemSize`（设完未注册 buffer 仍失败）。

**未适配**：`cuFileDriverGetProperties`、`cuFileDriverSetPollMode`、`cuFileDriverSetMaxDirectIOSize`（.so 无导出）、`cuFileStreamRegister/Deregister`、`cuFileReadAsync/WriteAsync`。

### 5.6 运维 FAQ（现象 → 方向）
| 现象 | 可能原因 | 检查方向 |
|---|---|---|
| `libhyfile.so not loadable` | 库未挂载/路径不全 | `/opt/hyhal`、`LD_LIBRARY_PATH`、`ldd` |
| `open /dev/hyfs: No such file` | 宿未装 DDS 或容器未透传 | `lsmod hyfs`、`ls -l /dev/hyfs`、重建容器 |
| `hipFileBufRegister failed` | 驱动/DDS/权限/指针/对齐 | 先跑 `benchmark_new`，查 `dmesg` 与 ctypes 参数 |
| store shape mismatch | K/V layout 假设与 SGLang 不符 | 打印 shape/stride，查 V 的 HS/BS 维 |
| HIP VMFault | slot 或 V layout 越界 | 小 tensor round-trip 逐步查 gather/scatter |
| hidden_dim 过大 | 用 query heads 替 KV heads | 按 `k_buffer` 算 `num_kv_heads*head_size` |
| `lmcache.c_ops` 不存在 | 镜像无编译扩展 | `disable_gpu_interm_buffer: true` 或编译 c_ops |
| retrieve NotImplementedError | 后端无 `batched_get_non_blocking` | 查 async wrapper 与 executor |
| layerwise retrieve OOM | GDS/staging 池容纳不下并发 pin | 增缓冲/降 conc/缩短前缀 |
| 日志无 `Using GDS backend 'hyfile'` | 配置未加载或 fallback | 查 `LMCACHE_CONFIG_FILE`、`PYTHONPATH`、`has_hyfile()` |
| `gds_path` 无文件 | store 未触发/路径错/IO 失败 | 查 store log、mount、权限、冒烟 |
| 小负载 CPU/POSIX/GDS 几乎一样 | IO 非主要开销 | 提高前缀长度/复用率，但保持公平性 |

### 5.7 生产落地建议
- **选层**：HBM 够且不淘汰前缀 → 用引擎内置 KV/radix，不强上 GDS；HBM 紧张 DRAM 够 → 先 CPU 热层（更简单低延迟）；DRAM 不足/需持久化/大容量 → GDS 冷层；既要热低延迟又要大容量 → CPU 热层 + GDS 冷层并明确晋升/淘汰策略。
- **监控**：LMCache store/retrieve 计数/命中率/字节数；chunk 读延时/错误码/短读写数；GDS buffer 已用/高水位/等待/pin 数；DCU HBM、SGLang KV 池、OOM 次数；容器 CPU/RSS、存储带宽/时延/IOPS/队列深度；TTFT 看 **p50/p95/p99 而非只看 Mean**；`/dev/hyfs`、hyfs 模块、`dmesg` 错误。
- **模型兼容性警告**：DeepSeek/GLM 等 KV cache 结构复杂模型，适配 LMCache 可能有额外工作量；本技术文档仅做海光 XDS 兼容性验证，**其他模型仍可能需移植、效果需实测**。

## 6. 性能结论溯源说明（重要纪律）
**主文档（vLLM 路线）**的性能**数字明细**存在 ksheet `cb6ZJZteiTZ8`（LMCache 推理性能影响分析），该表：
- 对当前账号**权限不足（400100 操作权限不足）**，内部文档 CLI 不可达（符合 内部文档-doc-extraction 的"spreadsheet 块 CLI 不可达"结论）；
- 本文 **§1 已收录主文档正文中文字化的结论**（吞吐倍数、TTFT 占比），这些是主文档作者已公开写出的，可作为方向性参考；
- **不编造** ksheet 单元格数字；如需精确数字，请有该 ksheet 权限者从浏览器导出后补入。
- 性能数字属于"项目实测数据"，按 dcu 纪律**不入库为基线**，只留方法论 + 方向性结论 + 溯源指针。

**XDS 文档（SGLang 路线，file_id `<internal-file-id>`）**的 §5.3 四组对照 TTFT 表是作者手写 Markdown 汇总表（非逐行日志），按 dcu 纪律属「内部团队已公开验证总结」例外——已**原样保留**于 §5.3，并标注「特定环境实测、非固定基线、其他模型/参数下倍数需重新实测」。GDS 比 POSIX 的 11% hit TTFT 优势不等于底层 IO 带宽提升 11%（hit TTFT 还含排队/KV 查找/回填/剩余 Prefill/调度）。

## 7. 与兄弟 skill 关系
- `dcu-vllm-deploy`：vLLM 部署调优（§1-§4 的 LMCache 三级缓存路线基于 vLLM + kv-transfer-config）。
- `dcu-llm-benchmark`：压测方法论（vllm bench serve 版本差异、tokenizer 膨胀坑；§5 的 SGLang 路线用 `sglang.bench_serving` 的 generated-shared-prefix 数据集）。
- `dcu-inference-optimization`：通用推理优化方法论（prefix caching / 量化等，LMCache 是其中 KV 卸载专项）。
- `dcu-vllm-source / 0.18 / 0.21.0 / 0.23.0`：版本对应（LMCache 版本↔vLLM 版本映射见 §0）。
- SGLang 路线的 LMCache 适配（`lmcache/v1/...` 的 hyfile 后端、Paged KV layout）属 LMCache 源码层；相关源码落点见 `dcu-sglang-source` 与 §5.4。

## 使用须知
- 海光 DCU 上 LMCache 走 ROCm 兼容路径（`BUILD_WITH_HIP=1` + gfx 指定），无 hygon 分支；版本与推理框架强绑定且 DAS commit 非固定。
- **两条已验证路线**：① vLLM + LMCache 三级缓存（§1-§4，CPU/NVMe offload）；② SGLang + LMCache **XDS/hyfile GDS 后端**（§5，海光 DCU 上已实测跑通，命中 TTFT 2.97×）。§5 旧版的「GDS 未跑通」已被 XDS 文档闭环。
- GDS 路线硬前置：宿主机装 DDS（`hyfs.ko` + `/dev/hyfs`）+ 容器透传 `/dev/hyfs` 与 `/opt/hyhal`；`has_hyfile()` 通过且启动日志出现 `Using GDS backend 'hyfile'` 才算生效，不能只看 YAML。
- 不存性能基线/逐行日志；数字用 §6 溯源纪律处理（XDS 文档表已带「特定环境、非固定」标注）。
- 容器必挂 `/opt/hyhal:ro`；SGLang 路线额外必透传 `/dev/hyfs`。高危操作谨慎，优先测试机验证。
