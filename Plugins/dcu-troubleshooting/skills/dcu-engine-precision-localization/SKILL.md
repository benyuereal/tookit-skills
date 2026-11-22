---
name: dcu-engine-precision-localization
description: 自定义C++推理引擎DCU精度定位到算子。引擎能跑通但输出垃圾时用。
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# dcu-engine-precision-localization — 自定义推理引擎精度定位（DCU）

## 何时用

- 自定义 C++/CUDA 推理引擎（**非** vLLM/SGLang，如 NInfer 这类闭集引擎）移植到 DCU 后，能跑通（EXIT 0）但输出垃圾/退化（编号列表、纯标点、乱码）
- 需要把"输出不对"定位到**第一个破坏表征的算子**
- vLLM/SGLang 的精度问题用 `dcu-vllm-precision` / `dcu-train-precision`，本 skill 面向自带 kernel 的 C++ 引擎

## 核心铁律（本类任务第一教训）

**下任何"X 错了"结论前，先验证对比脚本本身。**

NInfer 案例中连续 3 个"根因"（embedding 错 / GEMM kernel 错 / h 错）全部是**对比脚本假警报**（布局写反、dtype 读错、手算参考公式错）。参考值金标准 = **原生模型 forward hook 抓的实际运行时中间态**，不是手算公式。

## 五步法

1. **逐层 dump**：引擎里加 env 门控 dump（`getenv` + `hipMemcpy` + `fwrite` + min/max/mean/nan 统计，如 `NINFER_DUMP_LAYERS`）。
   - chunked prefill 会覆盖文件 → dump 文件名加 **chunk 计数器**
   - 逐层 cos 对比定位首个发散层（embedding 应 cos≈1.0）
2. **原生 hook 金标准**：transformers `register_forward_hook` 抓每个 op 的真实输出（input_layernorm / in_proj / conv1d / out_proj）。手算参考（rmsnorm 公式、eps、weight 处理）只作交叉验证，**不作判定依据**。
3. **排列测试**（区分索引 bug vs 数值 bug 的决定性方法）：
   - 两侧**全局排序**后 max_diff≈0 → 值集合相同 → **索引 bug**（值写错位置）
   - 逐 token 排序后 max_diff 大 → 值跨 token 打散
4. **反量化验证**（区分权重 bug vs kernel bug）：把引擎量化权重（Q4/Q5/W8）按 kernel 的 decode 公式反量化，和原生 bf16 对比。cos>0.99 → 权重对，嫌疑 kernel。
5. **独立 kernel 测试**：单独编译 kernel + 随机输入 + CPU 参考。**必须先读 kernel 源码确认实际访存模式**（`x[c][k]` vs `x[k][c]`、`out[c][r]` vs `out[r][c]`），测试布局严格对齐——布局不对齐的"kernel 错"结论不可信。

## 根因类别：MMA stub 静默 no-op（SIMT-first 移植最常见根因）

把 `mma.cuh` stub 成 no-op 后，MMA kernel **能编译能跑但不计算**——静默输出全 0 或垃圾，**不 throw**。这是"输出退化但不崩"的头号根因。

**模式**：dispatch 按 token 数 `t` 路由，`t` 超出 SIMT exact 范围时落进 MMA route → no-op。
**修复范式**：在 dispatch 里加 SIMT fallback——`t` 超 SIMT 范围时，用**已验证正确的通用 SIMT GEMM** + **单独 elementwise**（残差加 / swiglu / 归一化）绕开 MMA。

NInfer 3 个根因全是这个模式（全部已修，精度 6/6 全绿）：
1. **Q4 swiglu prefill**：t>1 走 MMA stub → activation 全 0 → MLP 不贡献。修：Q4 GEMM SIMT + swiglu elementwise。
2. **Q5 linear_add**（融合 GEMM+残差加，out_proj/down 用）：t>13/16 走 MMA stub → 残差加 `x += gdn_out` 不生效 → mlp 拿到 GDN 前的 x。修：通用 Q5 GEMM SIMT + 单独 residual_add。
3. **GDN chunked**：T≥64 走 MMA stub（1656 行 3 kernel）→ 长 prompt 退化。修：recurrent SIMT fallback（kernel 内部循环 T，支持任意 T，内部做 l2norm）。

**定位技巧**：某中间态"范围正常但 cos 低"，且**只在特定 t 区间错**（如 t=9 对、t=14 错）→ 几乎必是 dispatch 在该 t 落进 MMA stub。查 dispatch 的 route 表（`t` 范围 → schedule），确认该 t 是否指向 MMA。

## 陷阱（全部真实踩过）

1. **BF16 当 FP16 读**：两者位布局不同，`np.float16` 读 BF16 数据 → cos≈0 假警报。用 `torch.bfloat16` 或 `uint16<<16 → float32` 位重解释。
2. **reshape 布局反了**：flat 数组 `[token,hidden]` vs `[hidden,token]` 的 reshape 方向决定一切。先确认 kernel 写入模式（`out[i], i=t*d+k` → token-major）。
3. **dump dtype 不匹配**：g/beta 是 float32、h/qkv 是 bf16。f32 按 bf16 读 → 1e37 量级垃圾（像未初始化内存）。dump 前查 kernel 签名（`float*` vs `__hip_bfloat16*`）。
4. **手算参考错**：手算 rmsnorm（eps/weight/unit_offset）与原生 hook 对不上（cos=0.009 假警报）。hook 是金标准。
5. **测试脚本 x 布局反了**：kernel 读 `x[col][k]`（token 为 dim0），测试填 `x[k][col]` → "kernel cos=-0.004" 假警报；布局对齐后 cos=1.00000。
6. **别假设层类型**：查 config.json 的 `layer_types`（Qwen3.8-27B：layer 0 是 GDN，GQA 在 3,7,11...，full_attention_interval=4）。
7. **`__syncwarp` 在 DCU 是 no-op**（64-lane wavefront 锁步）。kernel 的 smem pipeline 若依赖它，先试替换 `__syncthreads` 排除同步问题——结果不变则是确定性结构错误，不是 race。
8. **量化格式细节**：Q4 `q=(nibble^8)-8`；Q5 5bit=`high<<4|low4`，`q=(q5^0x10)-0x10`；W8G32 group=32。row-split-k128 布局是 **plane-based**（codes/high/scale 三个平面，各 256 对齐），不是行交错。
9. **同一 .cpp 在 build 里被编成两份 .o**（如 `gated_delta_net.cpp` 同时落在 `build/ops/` 和 `build/cpp/`）→ 链接 `duplicate symbol`。重编某文件前先 `ls build/*/ | grep <name>` 确认它**原本**在哪个目录，重编到**同一目录覆盖**，并 `rm` 掉另一份。
10. **DTK 的 `__shfl_sync` 没有 `__half` 重载**（只有 `__hip_bfloat16`/`__hip_bfloat162` + 标量模板）。对 `__half` 变量调 `__shfl_sync` 会歧义/编译失败。i8 KV attention 这类 kernel 若用 `__shfl_sync` 搬 `__half` scale，需先转 `__hip_bfloat16` 或走 `__ushort` 位重解释。
11. **KV 显存别按"全部层"算**：只有 **attention 层**有 KV cache，linear-attention/SSM 层（如 GDN）存的是固定大小 SSM state，**不随上下文增长**。Qwen3.8-27B 64 层里只有 16 个 GQA 层有 KV → 256k bf16 KV = 16GiB（不是 64GiB），64GiB 卡放得下，**不需要 int8 KV**。先查 `layer_types` 数 attention 层数再算。
12. **多并发 decode 总吞吐基本恒定**（内存受限）：每步权重只加载一次服务所有序列，加并发不增总吞吐，单请求延迟按 1/N 降。N=8 总吞吐略高于 N=1 说明 batching 生效（权重共享摊薄固定开销）。这是**正确行为**，不是 bug。

## 验证纪律

- 每个"X 错"结论必须有**布局对齐 + dtype 正确**的对比支撑
- 结论被推翻时，明确记录假警报及原因（防止下个会话重查）
- 正确性通过前，性能数字只作参考
- 中间态"范围正常"（min/max 合理）≠"数值正确"，必须 cos 对比

## References

- `references/ninfer_qwen3_8_27b_case.md` — NInfer Qwen3.8-27B 案例：.ninfer 格式事实、反量化公式、已验证组件清单、假警报记录、进度快照

## 验收

1. 定位到具体的层号 / 算子名 / 张量名。
2. 给出「参照实现 vs 引擎实现」在该点的数值差与容差判定。
3. 能区分是引擎实现错、还是参照实现错。
4. 定位结论可复现（同一输入重跑得到同一位置）。

## 不适用场景

训练侧精度问题（走 `dcu-train-precision`）；
vLLM 特定路径的精度问题（走 `dcu-vllm-precision`）；
没有可用的参照实现（先解决参照，否则无法判定谁错）。
