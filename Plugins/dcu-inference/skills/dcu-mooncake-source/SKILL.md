---
name: dcu-mooncake-source
description: Mooncake Transfer Engine DCU 适配源码结构——KV cache 跨节点 RDMA 传输（libtransfer_engine 链接 DTK + HIP copy kernel + EP buffer）
version: 1.0.0
author: DCU Engineer
license: MIT
---

# Mooncake Transfer Engine DCU 适配源码索引（dcu-mooncake-source）

> ## 🔗 开源上游锚点（mooncake-das）
> 官方开源版 **mooncake-das**（`github.com/HYGON-AI/mooncake-das`，6⭐）基线 **Mooncake v0.3.13.post1**；另有 **LMCache-das**（`github.com/HYGON-AI/LMCache-das`）基线 LMCache `v0.3.13`（commit `fc031d4`，`dcu-lmcache` 用）。
> 本 skill §0 基于 `mooncake_transfer_engine-0.3.10.post1+das185.dtk2604...` wheel，官方开源线版本见快照。刷新：`python3 ../_dcu-toolkit-shared/scripts/hygon-upstream-sync.py --check`（快照 `../_dcu-toolkit-shared/references/hygon-upstream/upstream-snapshot.json`）。

> 定位：**Mooncake Transfer Engine 是 KV cache 跨节点传输引擎**（disaggregated prefill/decode 的 PD 分离、KV cache 池化），海光移植到 DCU（HIP）。本 skill 讲 Mooncake 在 DCU 上的包结构、传输引擎、EP buffer、DTK 链接。
> 与 `dcu-lmcache` 的关系：LMCache 讲 **KV cache 卸载/复用**（单机/多机缓存层）；Mooncake 讲 **KV cache 跨节点 RDMA 传输**（PD 分离的 prefill→decode KV 搬运）。两者常配合：LMCache 管缓存策略，Mooncake 管传输通道。
> 实测来源：`mooncake_transfer_engine-0.3.10.post1+das185.dtk2604.2608171032.g676b15-cp310` wheel（release/dtk2604 源），解包 44 文件 + ldd 实测。

## §0 定位：Mooncake 是什么

Mooncake = KV cache 传输引擎，核心能力：
- **Transfer Engine**：跨节点 RDMA（RoCE/IB）+ 节点内 NVLink/PCIe/HIP IPC 的 KV 块搬运。
- **EP buffer**：MoE 专家并行的 EP 通信 buffer（与 DeepEP 互补，Mooncake 提供传输层）。
- **Store**：KV cache 池化存储（`libmooncake_store.so` + `mooncake_store_service`）。
- **vLLM/SGLang 集成**：`vllm_v1_proxy_server.py` / `mooncake_connector_v1.py`（vLLM KV connector）。

**DCU 适配**：海光把 Mooncake 的 CUDA copy kernel 移植到 HIP（`mc_copy_kernel.co` 是 HIP 编译产物），`libtransfer_engine.so` 链接 DTK 的 `libgalaxyhip.so.5` + `libibverbs`（RDMA）。

## §1 源码地图（wheel 实测目录）

```
mooncake/
├── __init__.py
├── engine.so                    # ★ 传输引擎 Python 绑定
├── libtransfer_engine.so        # ★ 传输引擎核心（C++，链接 DTK + RDMA）
├── libmooncake_store.so         # KV cache 池化存储
├── store.so                     # store Python 绑定
├── nvlink_allocator.so          # NVLink/PCIe allocator
├── libasio.so                   # 异步 IO
├── mc_copy_kernel.co            # ★ HIP 编译的 copy kernel（.co = HIP code object）
├── allocator.py                 # 内存分配器
├── async_store.py               # 异步 store
├── ep.py                        # ★ EP buffer（按 torch 版本动态加载 ep_<ver>）
├── pg.py                        # ★ process group（按 torch 版本动态加载 pg_<ver>）
├── mooncake_ep_buffer.py        # ★ EP buffer 封装（EventOverlap + EP 通信）
├── mooncake_connector_v1.py     # vLLM KV connector v1
├── vllm_v1_proxy_server.py      # vLLM v1 proxy server（PD 分离）
├── mooncake_config.py           # 配置（hip: ROCm/HIP IPC/Shareable handles）
├── mooncake_store_service.py    # store 服务
├── http_metadata_server.py      # HTTP metadata server
├── cli.py / cli_bench.py / cli_client.py   # CLI + benchmark
├── transfer_engine_bench        # 传输引擎 benchmark 二进制
├── transfer_engine_topology_dump.py        # 拓扑 dump
├── mooncake_client / mooncake_master      # 客户端/主节点二进制
└── version.py                   # __hcu_version__ = '0.3.10.post1+das.opt1.dtk2604'
```

## §2 传输引擎（libtransfer_engine.so，ldd 实测）

`libtransfer_engine.so` 链接：
- **`libgalaxyhip.so.5`**（DTK HIP runtime，GPU 侧 copy kernel 用）
- **`libibverbs.so.1`**（RDMA verbs，跨节点 RoCE/IB）
- `libasio`（异步 IO，bundled 在 `mooncake_transfer_engine.libs/`）
- `libglog` / `libjsoncpp`（bundled）

> 传输路径：节点内走 HIP IPC/Shareable handles（`mooncake_config.py:26` 注释 "hip: ROCm/HIP for AMD GPU communication using IPC/Shareable handles"），跨节点走 RDMA（libibverbs）。

## §3 EP buffer（ep.py / pg.py / mooncake_ep_buffer.py）

**动态 torch 版本加载**（`ep.py`/`pg.py` 同模式）：
```python
torch_version = re.match(r'\d+(?:\.\d+)*', torch.__version__).group()
version_suffix = '_' + torch_version.replace('.', '_')
backend_module = importlib.import_module(f"mooncake.ep{version_suffix}")  # 如 ep_2_10_0
```
- 按 torch 版本加载对应编译的 EP 后端（`ep_2_10_0` 等），**torch 版本不匹配会 ImportError**（提示 "Mooncake EP was not built against torch==X"）。
- `mooncake_ep_buffer.py`：`EventOverlap`（CUDA/HIP event 封装，同 DeepEP 的 EventOverlap）+ EP 通信 buffer。

> EP buffer 与 DeepEP 的关系：Mooncake EP 提供**传输层**（RDMA/IPC 通道），DeepEP 提供**MoE 路由逻辑**（dispatch/combine 算法）。PD 分离 + MoE-EP 场景两者配合。

## §4 vLLM/SGLang 集成

- **vLLM**：`mooncake_connector_v1.py`（KV connector）+ `vllm_v1_proxy_server.py`（PD 分离 proxy）。vllm-hcu 的 `vllm__distributed__kv_transfer__kv_connector__v1__mooncake__mooncake_connector.patch.py` 是 Mooncake 接入 vLLM 的 patch 落点（见 `dcu-vllm-hcu-source` §7）。
- **SGLang**：SGLang 的 PD 分离用 Mooncake transfer engine（见 `dcu-sglang-source`）。

## §5 与兄弟 skill 关系

- `dcu-lmcache`：KV cache 卸载/复用（缓存策略层）→ Mooncake 是传输通道层，互补。
- `dcu-deepep-source`：MoE 专家并行通信（路由逻辑）→ Mooncake EP 是传输层，互补。
- `dcu-vllm-source` / `dcu-vllm-hcu-source`：vLLM KV transfer → Mooncake 经 vllm-hcu patch 接入。
- `dcu-sglang-source`：SGLang PD 分离 → Mooncake 是传输引擎。
- `dcu-rccl-source`：RCCL 集合通信 → Mooncake 是 KV 块传输（更特化）。
- `dcu-inference-infra`：推理 infra 总入口。

## §6 验证入口

```bash
# 安装（torch 标记对齐，本版 torch2100）
pip install mooncake-transfer-engine -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 DTK + RDMA 链接（DCU 容器内）
ldd $(python3 -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))")/libtransfer_engine.so | grep -E 'galaxyhip|ibverbs'

# 传输引擎 benchmark
$(python3 -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))")/transfer_engine_bench --help

# 拓扑 dump
python3 -m mooncake.transfer_engine_topology_dump
```

## 使用须知

- Mooncake 是**传输引擎**（KV 块搬运），不是缓存策略——PD 分离场景与 LMCache（缓存）配合。
- **EP buffer 按 torch 版本动态加载**（`ep_<ver>`），torch 版本不匹配会 ImportError——装前确认 torch 大版本。
- **DTK + RDMA 链接**：`libgalaxyhip.so.5`（GPU copy）+ `libibverbs`（跨节点 RDMA），DCU 容器需挂 RDMA 设备（`/dev/infiniband`）。
- **HIP copy kernel**：`mc_copy_kernel.co` 是 HIP 编译产物（.co = code object），GPU 侧 KV 搬运走它。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
