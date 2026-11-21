# Kernel 对拍探针 recipe（NInfer GDN recurrent，2026-08 验证 PASS）

移植一个 CUDA kernel 到 DCU 后，用"自包含探针 + CPU 参考对拍"验证数值正确性。
下面是 GDN recurrent kernel（NInfer 模型 48/64 层核心）的完整 worked example。

## 结果（gfx936，卡4，HIP_VISIBLE_DEVICES=4）
```
GDN recurrent fp32 probe (gfx936, H_qk=16 H_v=48 D=128 T=64)
  max |out  - ref| = 4.768e-06
  max |state- ref| = 1.431e-06
  RESULT: PASS
```
误差 ~1e-6 是 fp32 递归累积噪声（不是 bug）。容差设 5e-3：能抓 O(1) 真 bug，避免假 FAIL。

## 探针结构（单文件 .cu，自包含）
1. **抽 kernel + 最小依赖闭包**：从原代码拷 `recurrent_fp32_kernel`，只带它真正用到的：
   - `warp_sum`（warp reduction，用 `__shfl_xor_sync`）
   - `head_map` + `init_fastdiv_values`/`fastdiv`（fastdiv 头映射）
   - `load_vec`/`store_vec`/`load_qk_lane`/`store_qk_lane`（向量 load/store）
   - 常量 `kStateDim=128`、`kWarpSize=32`、`kDvPerWarp=4`、`kNumWarps=4`、`kBlockDv=16`、`kQkPerLane=4`
   - 不要带整个 launch.h / tensor.h / device.h（依赖太重，拖进无关编译错误）
2. **include 顺序**：`#include <hip/hip_runtime.h>` 然后 `#include "hip_shim.h"`
   （shim 提供 `__shfl_sync`/`__shfl_xor_sync` 等 masked 形式，见 templates/hip_shim.h）
3. **CPU 参考**：直白 128 维循环，同样的 delta-rule 数学，不优化。
   ⚠️ 数学顺序必须和 kernel 一致：GDN 是**先更新 state 再用更新后的 state 做 readout**。
4. **driver**：
   - 固定 seed 随机数据（`std::mt19937 rng(12345)`）
   - 用**真实模型维度**：H_qk=16, H_v=48, D=128, T=64, scale=0.08838834764831845f（Qwen3.8 GDN 真实 scale）
   - g 用 `[-3,0]`（alpha=exp(g)∈(0,1]，数值稳定），beta 用 `[0,1]`
   - 先 sanity check：fastdiv 头映射 == 整数除法（`qk_head(h)==h/(H_v/H_qk)`）
   - hipMalloc + hipMemcpy H2D → launch → hipDeviceSynchronize → hipMemcpy D2H
   - 比 max abs error（out 和 state 都比）
5. **编译运行**：
   ```bash
   /opt/dtk/bin/hipcc -O2 -std=c++20 --offload-arch=gfx936 probe.cu -o probe
   HIP_VISIBLE_DEVICES=4 ./probe
   ```

## 踩过的坑（本探针实际遇到）
1. **`__shfl_sync` 未声明** → DTK 无 masked 版，用 hip_shim.h 映射到 mask-less `__shfl`
2. **shim 加 `__HIP_DEVICE_COMPILE__` 守卫 → host pass 报未声明** → 去掉守卫
3. **指针算术括号**：`&k[(size_t)t*H_qk+h_qk] * D` 错（先取址再乘），应为 `&k[((size_t)t*H_qk+h_qk)*D]`
4. **fastdiv 括号不匹配**：`((1<<32)*((1<<L)-d))/d+1` 的括号要配对
5. **CPU 参考 readout 循环写坏**（重复/占位代码）→ 写完后逐行核对数学顺序

## 推广
- 每个要移植的 kernel 都做一个这样的探针，PASS 才推进下一个
- 纯 SIMT kernel（GDN recurrent/sampling/rope/rmsnorm）：探针直接 hipify 即可
- tensor core kernel（GQA attention/GDN chunked/量化 GEMM）：先重写 mma/cp.async，再探针
- 探针文件放 /tmp 或工作目录，不进 skill 包（skill 只沉淀方法，不存具体 kernel 代码）
