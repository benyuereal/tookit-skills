# NInfer → DCU 移植 triage 明细（2026-08-24，<device-B> 机器 gfx936）

> 案例：`github.com/Neroued/ninfer`（闭集 C++/CUDA 推理引擎，硬锁 RTX 5090 / sm_120a / CUDA 13.1）。
> 目标模型 Qwen3.8-27B（groupwise-int 档），基准 = 同卡原生 Qwen3.8-27B。
> 本文件只留**可复用的 triage 结论**，不存性能基线/逐行日志。

## 1. 工程画像
- 597 个源文件：~37K 行 CUDA kernel（.cu/.cuh）+ ~48K 行 C++（.cpp/.h）。
- 闭集：只认 5 个注册 `.ninfer` 权重包（私有容器格式，非 safetensors/GGUF）。
- 构建：根 CMakeLists 硬锁 `CMAKE_CUDA_ARCHITECTURES=120a` + `CUDA 13.1` + `find_package(CUDAToolkit)`；
  源文件**显式列表**（好裁剪）；依赖 FFmpeg≥60 / libcurl≥7.85（容器里是 58/7.81，只挡媒体路径）。
- 独立 archive `ninfer_nvfp4_tma`（Blackwell 独占 setmaxnreg+TMA）→ 整库砍。

## 2. PTX triage 结论（grep 全树）
| 类别 | 文件 | 处置 |
|---|---|---|
| 纯 SIMT（0 mma/ldmatrix/cp.async） | `gated_delta_net/recurrent.cuh`、`sampling.cuh`、`rope.cuh`、`rmsnorm.cuh` | 直接 hipify |
| tensor core（mma.sync/ldmatrix/cp.async） | `ops/common/mma.cuh`、`ops/common/memory.cuh`（中枢）、GQA attention、GDN chunked prefill | 重写为 CDNA wmma/v_mmac |
| Blackwell 独占（mxf4nvf4/TMA/setmaxnreg/PDL） | `nvfp4_w4a4_tma.cuh`、`nvfp4_linear_swiglu_w4a4_tma.cuh`、`pdl.cuh` | 砍掉 |
- `cp.async` 出现在 36 个文件，但**中枢在 `ops/common/memory.cuh`**（`cp_async`/`cp_async_zfill`/`cp_commit`/`cp_wait` 模板）——改这一个文件 + 各 kernel 调用点即可。
- `mma.sync` 全部集中在 `ops/common/mma.cuh`（bf16/f16/s8/fp8/tf32/nvfp4 六套）——重写这一个文件覆盖所有 GEMM/attention 的 tensor core。

## 3. 范围裁剪（Qwen3.8-27B groupwise-int）
- 注册路径：`registry.cpp` 里 `qwen3_8_model_id` → 走 `Qwen3_6_27B` target 类（复用 27B 执行包）。
- 模型结构：64 层 = 48 GDN（线性注意力）+ 16 full-attention（GQA）+ dense SwiGLU MLP，**无 MoE**。
- groupwise-int 档算子需求：attention/GDN 输入投影=BF16（split 路径）、MLP=Q4G64(gate_up)+Q5G64(down)、
  attn/GDN 输出投影=Q5G64、KV cache=INT8 group-64。**完全不需要 NVFP4/FP8**（那是 nvfp4 档）。
- 砍掉：`qwen3_6_35b_a3b`（MoE target）+ `sparse_moe` ops、所有 `nvfp4` ops、`ninfer_nvfp4_tma`、vision/media。
- 权重：容器里 `/data1/models/Qwen3.8-27B` 原生权重在；转换器 `tools/convert/qwen3_8_27b/convert.py`
  只需 `torch`（容器 2.10.0+HIP）+ `safetensors`（0.8.0）→ 可在 DCU 上直接生成 `.ninfer` 权重包。

## 4. 环境（<device-B> = node1，BW100X8，8 卡）
- 卡 0–3 被占（显存 95–97%），**卡 4–7 空闲**，选卡 4（`HIP_VISIBLE_DEVICES=4`）。
- DCU 架构 **gfx936**（CDNA3 级，有 FP8 matrix core e4m3/e5m2；NVFP4 是 Blackwell 独占）。
- 容器 `ninfer-dcu-port`：镜像 `<internal_registry>/dcu/admin/base/vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10`，
  `--privileged -e HIP_VISIBLE_DEVICES=4 -v /opt/hyhal:/opt/hyhal:ro -v /data1:/data1`。
- 工具链：hipcc=dcc 25.10.0 / DTK 26.04（clang 17）、cmake 3.29、ninja、g++ 11.4（C++20）。
- **nvtx3 不在 DTK** → CMake 里 `CUDA::nvtx3` 要处理掉（stub 或删）。

## 5. 构建 unblock（DTK 26.04 实测）
- CMake HIP 语言坏：`LANGUAGES C CXX HIP` + `find_package(hip)` 报 `can not determine linker language`，
  5 种变体（LINKER_LANGUAGE / CMAKE_HIP_COMPILER=hipcc / =clang++ / PREFIX_PATH）全失败。
- **解法**：绕开 CMake HIP 语言，直接 `hipcc -O2 -std=c++20 --offload-arch=gfx936 host.cpp dev.cu -o app`
  （.cpp+.cu 混合编译+链接，gfx936 上跑通 "DEV_OK + HOST_OK"）。
- hipcc 默认 offload-arch 是 gfx928/906/926，**必须显式 `--offload-arch=gfx936`**。

## 6. 移植阶段规划（探针先行）
- Phase 0 探针：GDN recurrent（模型核心、纯 SIMT）单独 hipcc 编译 + CPU 参考对拍 → 证明可行。
- Phase 1：核心 runtime（device/arena/tensor/layout）hipify + 最小编译通过。
- Phase 2：可移植 kernel（GDN recurrent / GQA decode / sampling / rope）hipify + 对拍。
- Phase 3：量化 GEMM 换 CDNA matrix core（W8/Q4-Q6，砍 NVFP4）。
- Phase 4：GDN chunked prefill 移植（最重，~2900 行，重度 tensor core + 自定义 swizzle）。
- Phase 5：MTP 投机 + ReplaySSM parity（bitwise 一致性换硬件后必须重验）+ hipGraph。
- 端到端：跑通 Qwen3.8-27B，与原生 Qwen3.8-27B 同卡对拍性能。

## 7. 风险点
- GDN chunked prefill：CDNA LDS 结构与 NVIDIA 不同，可能"重写"而非"翻译"。
- ReplaySSM 有限精度 bitwise 一致性：换硬件后浮点结合顺序变，parity 挂了要重新设计 record 格式。
- 丢 NVFP4 主路后，最终性能大概率达不到 5090 水平（NVFP4 prefill 3.48× 那条路归零）。

## 8. 实际移植进度（2026-08-24 续，截至本会话）
- **核心库 15/15 编译通过**（device/arena/tensor/layout/kv_cache/engine/decode_graph）。
  - `decode_graph.cpp` 的 hipGraph API 签名差异：`hipGraphInstantiate` 第 3 参是 `hipGraphNode_t*`（非 flags）→ 改用 `hipGraphInstantiateWithFlags`；`hipGraphExecUpdate` 要 4 参（多 `errorNode*`）+ 返回 `hipGraphExecUpdateResult`（enum，非结构体，直接比较非 `.result`）。
- **ops 层 52/127 编译通过**（全部纯 SIMT kernel：GDN recurrent、sampling、rope、rmsnorm、causal_conv、Q4/Q5/W8/Q6 的 SIMT GEMM、MTP 等）。
- **GDN recurrent 探针数值 PASS**（max err 4.8e-6，模型 48/64 层核心）。
- 剩余 75 个失败分类：**49 tensor-core（mma.cuh）** + 12 NVFP4 + 12 FP8（要砍）+ 2 误分类（实为 mma/sparse_moe）。

## 9. DTK 26.04 / gfx936 实测 intrinsic 缺口（compat shim 已建）
集中用一个 compat shim 头（`hipcc -include` 注入）补齐，**不加 `__HIP_DEVICE_COMPILE__` 守卫**（host pass 也解析 kernel 体）：
- `__shfl_*_sync`（masked）→ 转发到 DTK 无 mask 版 `__shfl`/`__shfl_xor`/`__shfl_down`/`__shfl_up`。函数名单下划线，宏拼接别写双下划线。
- `__syncwarp()`/`__syncwarp(mask)` → `__builtin_amdgcn_wave_barrier()`（DTK 的 sync 头不被默认 include）。
- `__float2bfloat16_rn`/`__floats2bfloat162_rn`/`__float2bfloat162_rn` → `__float2bfloat16`/`__float22bfloat162_rn`（DTK 无 `_rn` 后缀版）。
- `__grid_constant__`（CUDA 12.1+）→ 直接删（`sed 's/__grid_constant__ //g'`）。
- `math_constants.h`（`M_PI`/`CUDART_INF_F`/`CUDART_NAN_F`）→ 自建 shim 头 + include 路径。
- `cub/block/block_merge_sort.cuh` → `hipcub/block/block_merge_sort.hpp`（DTK 是 hipcub 且 `.hpp`），include 加 `/opt/dtk/hipcub/include`。
- `cp.async` PTX → **DTK 无 `__pipeline_memcpy_async`**，`memory.cuh` 改同步 `std::memcpy`（`cp_commit`/`cp_wait` no-op）。
- `ld.global.cg.v2/v4.u32`（`'l'` 约束）→ 普通 `load_vec<uint2/uint4>`（cache hint 可省）。
- PDL（`pdl.cuh`）→ `launch_dependent` 退化普通 `<<<>>>`，`trigger/wait` no-op。
- **DTK CUDA 12 兼容层** `/opt/dtk/cuda/cuda-12/targets/x86_64-linux/include/` 有 `cuda_bf16.h`/`cuda_runtime.h`/`math_constants.h`，但 `cuda_pipeline.h` 无 `__pipeline_memcpy_async`，且 `#ifdef __CUDACC__` 守卫在 hipcc 下不生效 → 已 hipify 的树优先，兼容层备用。

## 10. 下一步决策（SIMT 兜底 vs 直接 MFMA）
- **真正必须重写的 tensor-core kernel 只有 ~5 个**：GQA decode(bf16)、GQA prefill(bf16)、GDN chunked prefill(3 个)。其余 44 个 mma 失败是量化 GEMM 的 MMA 变体，**都有 SIMT 孪生且已编译通过**，只需 dispatch 强制走 SIMT。
- 推荐路径：① 给 5 个关键 kernel 写 SIMT 版（**GQA 先查 AITER `paged_attn` 能否直接替代**，见 `mfma-availability.md` 出路 §2）→ ② 量化 GEMM dispatch 强制 SIMT → ③ 砍 NVFP4/FP8/MoE → ④ 链接端到端 + 对拍原生 Qwen3.8-27B 数值 → ⑤ 确认正确后对 prefill 热点上真 MFMA。
- 取舍：SIMT 版 decode 性能接近（memory-bound），prefill 会慢（compute-bound 无 tensor core）。先正确性后性能。

## 11. SIMT-first 实施细节（2026-08-24 续2）

### 11.1 GQA attention 无 SIMT 变体（必须手写，不能靠 dispatch 兜底）
与量化 GEMM 家族不同，**GQA attention 的 9 个 kernel 文件全是 tensor-core，没有 SIMT 孪生**：
`gqa_attention_{decode,prefill}_{bf16,i8}.cuh`、`gqa_attention_decode.cuh`、`bidirectional_gqa_attention.cuh` 等
`grep -cE "mma_bf16|ldmatrix"` 全 >0。所以 §10 说的「~5 个必须重写的 tensor-core kernel」=
**GQA decode(bf16) + GQA prefill(bf16) + GDN chunked prefill(3 个)**，且 GQA 这 2 个**只能手写 SIMT**
（没有现成 SIMT 变体可强制 dispatch 过去）。
- 27B 几何 = `Gqa27Geometry`（`GqaGeometry<24,4,1>`：24 Q heads / 4 KV heads / group 6 / head_dim 256）。
- **GQA decode 的 split-KV 结构**：partial kernel（每 split 算 `partial_acc/m/l`）+ reduce kernel（合并）。
  **reduce kernel（`gqa_attention_small_t_reduce_output_kernel`）是纯 SIMT（0 mma），可直接复用**——
  手写 SIMT 版只需重写 **partial kernel**，产出兼容的 `partial_acc/m/l` 格式即可，reduce 不动。
- SIMT partial kernel 设计（head_dim=256, group=6, small T 1-6）：block=256(8 warps)，每 warp 管一组
  query rows（rows=tokens*GroupSize≤36），每 lane 管 d∈[lane*8, lane*8+8)（8 个 d 共 256），
  逐 key 载入 k/v 到 smem → 每 row 算 score（8 乘 + warp reduce）→ online softmax → acc += p*v。
  decode 是 memory-bound，SIMT 版性能接近 tc 版。（本会话已写出 `gqa_attention_decode_simt.cuh`，
  尚未编译/对拍验证——别当已跑通。）

### 11.2 GEMM dispatch SIMT-first 重写（具体机制，已编译通过）
量化 GEMM 的 SIMT 变体是**形状通用**的（`launch_tt` 的 n/k/t 是运行时参数，grid 按
`div_up(n,8)×div_up(t,ColsPerTile)` 算，`launch_route<ColsPerTile>` 用 `for_each_token_slice` 切任意 t），
所以**把 dispatch 全部路由到 SIMT 是安全的**，不用逐形状重写。
- dispatch 文件：`src/ops/linear/{w8,q4,q5,q6}/*_dispatch.cpp`，决策函数 `select_*_a16_launch(n,k,t)`
  （按 token 数 t 选 kernel：小 t→SIMT，大 t→MMA）。
- 改法：把 `select_*_a16_launch` 函数体替换为「`if (t<=4) return launch_*_simt_r8_c4; return launch_*_simt_r8_c8;`」
  （全 t 落 SIMT）。w8/q4/q5/q6 四个 dispatch 已改并编译通过。
- ⚠️ **`small_t` 命名陷阱**：`w8_small_t.cu`/`w8_rowsplit_gemm_splitk.cu`（dflash/medium_splitk）
  名字像 SIMT，但**实际 include 了 `w8_small_t_mma.cuh`**（底层是 MMA kernel），所以也挂在 mma.cuh 上。
  **别信文件名里的 "simt"/"small_t"**——`grep -cE "mma_bf16|ldmatrix" <file>` + 看它 include 的头，
  确认是否真无 tensor-core 依赖。
- nvfp4/fp8 dispatch 无 SIMT 变体（要砍）；bf16 dispatch 无 SIMT 变体（但 `bf16_gemv.cu` 已编译过，单独看）。

### 11.3 模型 config 嵌套
`/data1/models/Qwen3.8-27B/config.json` 顶层 `hidden_size` 等是 **None**——真实维度在 **`text_config`** 嵌套里
（`c['text_config']['hidden_size']=5120` / `intermediate_size=17408` / `num_attention_heads=24` /
`num_key_value_heads=4` / `head_dim=256` / `num_hidden_layers=64` / GDN `linear_num_key_heads=16` /
`linear_num_value_heads=48` / `linear_key_head_dim=128`）。读维度先 `c.get('text_config', c)`。

## 12. 权重转换（已在 DCU 上跑通，2026-08-24）
- 命令：`python3 -m tools.convert.qwen3_8_27b.convert --model /data1/models/Qwen3.8-27B --out /data1/out/qwen3_8_27b.ninfer`（容器内，`HIP_VISIBLE_DEVICES=4`）。
- 结果：52G 原生权重（18 个 safetensors）→ **18.2G `.ninfer`**（groupwise-int 档），**93.8s**，1124 个 tensor 全转完，`<out>.conversion.json` 报告生成。
- 依赖：只需 `torch`（容器 2.10.0+HIP，`torch.cuda.is_available()`=True）+ `safetensors`；`pick_device` 在 CUDA 不可用时自动回退 CPU，DCU 上直接走 GPU。
- **import 坑**：`recipe` 不在 `tools.convert.qwen3_8_27b` 包里（`__init__.py` 只导出 inventory 常量），它来自 **`tools.convert.qwen3_6_27b`**（`from tools.convert.qwen3_6_27b import recipe`）——preflight 验证要 `recipe.preflight_sources(model)` + `recipe.validate_recipe_coverage()`，别从 qwen3_8_27b 包 import。
- 转换与 kernel 移植**解耦**：权重包格式是引擎私有的，转换只依赖 torch，不依赖 hipify 后的 C++ 树——可提前跑，不占编译进度。

## 13. SIMT-first 实施进度（2026-08-24 续3，本会话）
- **GQA decode SIMT kernel（bf16 + i8 两版）编译通过**：自包含头 `gqa_attention_decode_simt.cuh` / `gqa_attention_decode_simt_i8.cuh`（本地重定义 `kGqaHeadDim` + 索引辅助，**不 include** 含 mma 的 `gqa_attention_decode.cuh`，避免 mma.cuh 坏 PTX 传染）。i8 版 K/V 从 int8 cache 读 + on-chip 反量化（per 64-group scale），Q 保持 bf16→float，QK/PV 全 float（比 int8 MMA 简单，decode 是 memory-bound 所以 ALU 多无所谓）。
- **GQA prefill SIMT kernel（bf16 + i8 两版）编译通过**：`gqa_attention_prefill_simt_{bf16,i8}.cuh`，FlashAttention-2 风格（Br=64/Bc=64/128 threads/4 warps，每 warp 16 query rows，每 lane d∈[lane*8,lane*8+8)，QK 逐 key 8 FMA + 32-lane reduce，PV 逐 key 无 shuffle，online softmax exp2，bottom-right causal）。
- **★32-lane shuffle 子组模型验证 PASS**：gfx936 wavefront=64 lane，但 `__shfl_xor_sync(w=32)` 限制在 32-lane 子组（256 线程 val=1.0 做 32-lane butterfly reduce → 32.0 非 64.0）。**按 NVIDIA 32-lane warp 假设写的 SIMT kernel 数值逻辑正确**，"编译通过"不是虚的。
- **GEMM dispatch SIMT-first 完成**：w8/q4/q5/q6 四个 `select_*_a16_launch` 全改「全 t 落 SIMT」，编译通过。
- **GDN chunked prefill 定位**：`src/ops/linear_attention/gated_delta_net/chunked/` 3 个 kernel（`prepare_wy_wu.cuh` 725 行 / `state_passing.cuh` 539 行 / `output.cuh` 371 行），全 tensor-core（mma_tf32/mma_bf16/ldmatrix），mma 集中在少数 16×16×16 GEMM tile 但与 fragment 布局/swizzle 紧耦合 → 需**重写 kernel 主体**（非换 mma 调用）。这是最后一块硬骨头（~1635 行）。
- **AITER 无 GDN/linear-attention 算子**：`import aiter` 子模块只有 paged_attn/mla/moe/gemm/rotary，**没有 gdn/linear/delta/mamba** → GDN（48 层）无库可替代，必须手写 SIMT。
- **待做**：GDN chunked prefill SIMT 重写（3 kernel）→ 接入 launcher（decode+prefill）→ 端到端跑通 + 与原生 Qwen3.8-27B 对拍。
- **性能预期**：decode 接近原生（memory-bound，SIMT 够用）；prefill 会明显慢（SIMT 无 tensor core，GDN chunked 最重）。
