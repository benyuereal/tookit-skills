---
name: dcu-ninfer-port
description: "把 NInfer 移植到 DCU（gfx936）：SIMT 优先 + 四层量化（NInfer port, SIMT-first, 4-layer quant）。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# dcu-ninfer-port — NInfer 移植到海光 DCU

> 触发词：NInfer、ninfer、NInfer 移植、NInfer DCU、ninfer-hip、W8G32 转换、量化格式转换、SIMT-first、ninfer_cli

NInfer 是一个单卡闭集 C++/CUDA 推理引擎（原生硬锁 RTX 5090/sm_120a），移植到海光 DCU（gfx936）的长期项目。本 skill 沉淀**可复用的方法论**（不是逐行日志/性能基线——那些属 `dcu-perf-cases`）。

## 核心策略：SIMT-first

- **MFMA 路线被 DTK 26.04 堵死**（无 `v_mfma` intrinsic），走 **SIMT-first** 策略。
- 所有 GEMM/GEMV kernel 用 SIMT（标量 FMA + 向量化加载），不用 MMA。
- `mma.cuh` 已 stub 化（PTX→no-op），所有 MMA 文件能编译但运行时不计算。
- 编译器已自动用 `v_pk_fma_f32`（packed FMA，GEMV 汇编里 ~4.9% 指令），**无需手动内联汇编**——DTK 不支持 64 位 `l`/`d` 寄存器约束，内联汇编这条路被编译器堵死。
- fp16 mantissa trick 在 GEMM 上可用（memory-bound 被掩盖），在 GEMV 上更慢（-6%，ALU 暴露），已回退。

## 4 层量化格式转换路径（核心可复用模式 ⭐）

把 NInfer 权重转到新量化格式（或扩展现有格式到新形状），必须**同时改 4 层**，缺一层就报错：

| 层 | 文件 | 作用 | 报错关键词 |
|---|---|---|---|
| 1. 格式分配 | `tools/convert/<model>/inventory.py` | 哪个 tensor 用哪个格式（Q4/Q5/W8/...） | — |
| 2. target 契约 | `src/targets/<model>/impl/load/bindings.cpp` | 每个 tensor 期望的格式（硬编码） | `tensor descriptor does not match target contract` |
| 3. wrapper 形状校验 | `src/ops/wrapper/<op>.cpp` | 哪些 (n,k) 形状被接受 | `<op>: unsupported <fmt> shape` |
| 4. plan 准入+路由 | `src/ops/<op>/<fmt>/<fmt>_plan.cpp` | 哪些形状路由到哪个 kernel | `<fmt> <op>: exact problem or column count is not admitted` |

**关键洞察**：SIMT kernel 本身**形状通用**（`grid=div_up(rows)×div_up(cols)`，任意 T 走 token slice），但**白名单（层 3+4）形状专用**（硬编码到 MTP 形状，如 rows=2048, k∈{4096,6144}）。要新形状走某格式，只需**扩白名单路由到形状通用的 SIMT kernel，不用写新 kernel**。

### 扩白名单的标准改法（以 W8 linear_add 为例）

1. **层 4 plan admits**：加新形状分支（如 `rows==5120 && k∈{4096,6144,17408}`）。
2. **层 4 plan resolve**：新形状全部路由到 `SimtR8C4`（形状通用 SIMT），不碰 MMA kernel。
3. **层 3 wrapper 校验**：加新形状到白名单（`w.n==5120 && w.k∈{...}`）。
4. **层 2 bindings.cpp**：对应 tensor 的期望格式改成新格式。
5. **层 1 inventory.py**：对应 tensor 的格式分配改成新格式。
6. 重转换权重 + 重编 4 个文件 + 重链。

详见 `references/w8g32-conversion.md`（W8G32 具体细节 + 4 个 patch 内容 + 转换/重编命令）。

## build/link 坑（反复踩）

1. **duplicate main**：`cpp/*.o` 里已有 `apps_cli_main.o`，**不能再加** `apps/cli/main.cpp`。
2. **旧 .o 残留（混合命名，别全删）**：build 目录有两种 .o 命名——干净命名（`ops/linear_add.o`）和旧前缀（`cpp/src_ops_linear_add_w8_w8_linear_add_plan.o`）。**不能简单删/排除所有 `src_ops_*`**——有些符号只定义在 `src_ops_*` .o 里（如 `q5_linear_add_capacity_workspace_bytes` 只在 `cpp/src_ops_linear_add_q5_q5_linear_add_plan.o`），全排除会 undefined symbol。**正确做法：精确排除**——只排除你刚重编的文件的旧 `src_ops_*` 版本（避免 duplicate symbol），其余全保留。用 `nm -C <.o> | grep "T <symbol>"` 查符号定义在哪个 .o。
3. **手动加 .o**：`/tmp/gqa_prefill_simt_launcher.o` 需手动加到链接命令（提供 `gqa_attention_prompt_launch`、`gqa_kv_append_launch` 符号）。**容器重启后会丢**（/tmp 清空），需重编：`cd <local_ninfer_src>-hip && /opt/dtk/bin/hipcc -O2 -std=c++20 --offload-arch=gfx936 -include <local_ninfer_src>_hip_compat.h -Iinclude -Isrc -Ithird_party -Ithird_party/utf8proc -Iinclude/shim /data1/gqa_prefill_simt_launcher.cu -c -o /tmp/gqa_prefill_simt_launcher.o`。验证：`nm -C /tmp/gqa_prefill_simt_launcher.o | grep -c "gqa_attention_prompt_launch\|gqa_kv_append_launch"` 应输出 2。
4. **后台构建 DONE 标记无条件写**：编译可能失败但用了旧 .o 链接（"DONE" 不代表成功）。**重编后必须检查 .o 时间戳 + 错误日志**，别信 DONE。
5. **链接脚本**：`/data1/do_link.sh`（只链接 .o，不重加 main.cpp）。
6. **DTK 坑**：`__shfl_down`（无 `_sync`）；`__ushort_as_half` 只取低 16 位（应用 `half2_from_bits(uint32)`）；内联汇编只认 32 位 `f`/`r` 约束。
7. **hy-smi 解析**：hy-smi 显示 **VRAM%**（第 6 列），不是空闲 GB。空闲 GB = 64 × (100 - VRAMPCT) / 100。解析：`hy-smi | awk '$1==4 {print $6}' | tr -d '%'`。（本会话踩过：第一个监控脚本解析了 HCU%（0.0%），永远不触发。）

## 转换脚本

- `tools/convert/<model>/convert.py` — 主转换脚本（用 `--device cpu` 避免 GPU OOM，27B 模型 CPU 转换 ~210s）。
- `tools/convert/<base_model>/recipe.py` — tensor 结构定义（**权威形状来源**，查 (n,k) 看这里）。
- `tools/convert/common/quantize.py` — 量化核心（`scale=max_abs/qmax`, `codes=round(grouped*reciprocal)`, `encode_row_split`）。
- 输出：`/data1/out/<model>_<fmt>.ninfer` + `.conversion.json`（注意在 `/data1/out/`，不是 `/data1/models/`）。

## 量化 decode 优化（NInfer 实测数据）

> **通用方法论（ALU-bound 诊断、scale 延迟/向量化等手法、fp16 trick/int8/双缓冲/CUDA Graph 反例、profile 坑、格式选型）已沉淀到 `dcu-op-optimization` 案例九**——其他项目参考去那里。本节只留 NInfer 项目专属的实测数据与结论。

- 基准：同卡原生 Qwen3.8-27B（`/data1/models/Qwen3.8-27B`，52G）。
- decode 是 **ALU-bound**（dequant 解包 ~40% 指令：移位+位提取+int→float 转换），**不是 memory-bound**（带宽只用 ~30%）。
- 理论 decode 上限 ≈ 权重字节 / 纯读带宽。
- 已验证有效的优化：scale 延迟（per-group scale 推迟到 group 级只乘 1 次）、sign_extend 算术移位、ACC8 多累加器、int8 u32 向量化（比 Q5 GEMV 快 1.78x 且无损）。
- 已排除的方向：fp16 trick（GEMV 更慢）、occupancy（已 100%）、kStages=2（丢 occupancy）、CUDA Graph（仅 +1.8%，launch 空隙非瓶颈）。
- **W8 vs Q5（2026-08-25）**：W8（仅转 25% tensor）= 18.61 tok/s vs Q5 基线 20.86 tok/s = **-11%**。原因：① 只转 25% tensor，其余 75% 还是 Q4/Q5，收益被稀释；② W8 SimtR8C4 kernel 没做 u32 向量化；③ W8 显存流量 1.6x（int8 1 字节/值 vs Q5 0.625 字节/值），没向量化时带宽劣势暴露。int8 v2 原型（u32 向量化）比 Q5 GEMV 快 1.78x 且无损 → **W8 kernel 需加 u32 向量化才能达到预期性能**。

## References

- `references/w8g32-conversion.md` — W8G32（int8）转换具体细节：4 个 patch 内容、形状白名单扩展、转换/重编命令、踩坑记录。
