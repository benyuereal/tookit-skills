# MFMA 可用性验证（DCU / gfx936，2026-08 实测）

> 背景：NInfer→DCU 移植中，"把 mma.sync 重写成 MFMA" 这条路在 DTK 26.04（内含 dcc 25.10.0 编译器）/ gfx936 上被工具链堵死。
> ⚠️ 版本口径（用户纠正过）：`hipcc --version` 报 `dcc version: 25.10.0-0` 是**编译器组件**版本；DTK 版本是 **26.04**（`env | grep DTK_VERSION` = `DTK_VERSION=2604`，镜像 tag `...-dtk26.04-py3.10`）。别写成"DTK 25.10.0"。
> 本文件记录诊断过程 + 实测数据 + 证据，供未来会话快速判断"我的 DTK/gfx 能不能手写 MFMA"。
> **复测触发**：换了 DTK 版本或换了 gfx → 重跑下面诊断步骤 1-3，feature 表可能不同。

## 结论（DTK 26.04 / dcc 25.10.0 / clang 17 / gfx936）
- **硬件矩阵核正常**：hipBLASLt bf16 GEMM 4096³ = **266.9 TFLOPS**；torch bf16 matmul = **278 TFLOPS**（fp32 sgemm 仅 38.5T）。
- **但手写 MFMA 编译不过**：所有 MFMA builtin 被 gate 在 `mai-insts` target feature 上，而该 feature 被 target 解析器拒。
- **AITER（海光算子库，vLLM on DCU 用）预编译给 gfx936+gfx938**，其 `attention_common.cuh` 用 `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`（gfx936 分支）——说明预编译二进制里有 MFMA 机器码，但**当前 DTK 编译器不开放手写**。

## 诊断步骤（可复现）
1. 最小探针（gfx936 bf16 MFMA，注意 `_1k` 后缀）：
   ```c
   #include <hip/hip_runtime.h>
   #include <hip/hip_bf16.h>
   using floatx4 = __attribute__((__vector_size__(16))) float;
   using _B16x4  = __attribute__((__vector_size__(8))) uint16_t;
   __device__ __forceinline__ floatx4 mfma(const _B16x4&a,const _B16x4&b,const floatx4&c){
     return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,b,c,0,0,0);
   }
   __global__ void k(float*c,const _B16x4&a,const _B16x4&b){ floatx4 cc={0,0,0,0}; cc=mfma(a,b,cc); c[0]=cc[0]; }
   ```
   `hipcc -O2 -std=c++20 --offload-arch=gfx936 probe.cu -c -o probe.o`
2. 报错：`error: '__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' needs target feature mai-insts`
3. 试启用（DTK 26.04 全失败）：
   - `--offload-arch=gfx936:mai-insts+` → `invalid target ID`（而 `gfx936:sramecc+` 能通过 → 证明 mai-insts 不在 feature 表）
   - `--target-feature=+mai-insts` → `unsupported option`
   - `-mllvm -amdgcn-target-features=+mai-insts` / `-mllvm -target-feature=+mai-insts` → 无效
   - `aicc`（DTK 重命名的 hipcc，AITER 用它）→ 同样报 needs mai-insts
   - 双 arch `--offload-arch=gfx936 --offload-arch=gfx938`（AITER build.ninja 同款）→ 同样失败
4. 硬件 sanity：hipBLASLt bf16 GEMM 266.9T（见下）。

## 关键事实
- **gfx936 的 MFMA builtin 名**（来自 AITER `attention_common.cuh` + `BuiltinsAMDGPU.def`）：
  - bf16: `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`（gfx936，`#else` 非 gfx950 分支）
  - f16: `__builtin_amdgcn_mfma_f32_16x16x16f16`
  - gfx950 用 `__builtin_amdgcn_mfma_f32_16x16x32_bf16` / `_f16`（K=32）
  - 调用形式：`mfma(A, B, C, absz, cbid, blgp)`，A/B=`_B16x4`(4×bf16=8B)，C=`floatx4`(4×f32)
- **所有 MFMA builtin 在 `BuiltinsAMDGPU.def` 里都 gate 在 `"mai-insts"`**（`TARGET_BUILTIN(..., "nc", "mai-insts")`）。
- **`BuiltinsAMDGPU.def` 路径**：`/opt/dtk/dcc/include/clang/Basic/BuiltinsAMDGPU.def`。
- **AITER 位置**：`/usr/local/lib/python3.10/dist-packages/aiter/`，主算子库 `jit/module_aiter_operator.so`（574MB，strings 显示 130×gfx936 + 130×gfx938）。
- **AITER 编译 flag**（build.ninja）：`--offload-arch=gfx936 --offload-arch=gfx938 -DDTK_ENV -DGPU_ENABLE_FP8`，**无 mai-insts flag**——说明 AITER 的 MFMA 模板在 gfx936 下**没被实例化**（否则也会报 needs mai-insts），或 AITER 的 .so 是用**别的（更新）DTK** 预编译的。

## 硬件 sanity check（确认是编译器问题不是硬件问题）
```c
// hipBLASLt bf16 GEMM 4096³
hipblasLtMatmul(lt, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc, nullptr, nullptr, 0, 0);
// 实测 0.52ms = 266.9 TFLOPS (bf16)
// torch: a@b (bf16, 4096³) = 0.49ms = 278 TFLOPS
// rocBLAS sgemm (fp32, 4096³) = 3.57ms = 38.5 TFLOPS
```
> 266T+ bf16 证明矩阵核在跑（SIMT fp32 只有 ~38T），所以是**编译器没开放裸 MFMA**，不是硬件没有。

## 出路
1. **GEMM 走 hipBLASLt / AITER `tuned_gemm`**（bf16 266T+）——非标准量化格式（Q4G64/Q5G64）先反量化到 bf16 再喂 BLAS。
2. **attention 优先查 AITER `paged_attn`**（`aiter.paged_attn.paged_attention_v1/v2`，GQA + bf16 + paged KV，预编译给 gfx936）——若目标模型的 KV 布局/几何能对齐 AITER 的 paged 格式，直接调它比手写 SIMT/MFMA 快且性能到位。⚠️ wrapper 走 `import aiter as ops` 调 `ops.paged_attention_v1`，顶层 `aiter` 可能没该属性（kernel 在 `aiter.ops` 子模块/JIT 里）——先 `import aiter.ops` 确认符号在不在，不在就是该 .so 没加载/没编这个 op。
3. **真正无库可替代的自定义 kernel（如 GDN chunked prefill）手写 SIMT**。
4. **换更新版 DTK**（支持 `mai-insts` 的）——拿到后直接 MFMA 重写，性能到位。
5. 性能取舍：SIMT 版 decode 接近原生（memory-bound），prefill 慢（compute-bound 没 tensor core）。先正确性后性能。
