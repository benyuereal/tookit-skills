# dcu-ort-profile

ONNX Runtime 1.19.2 在 DCU / ROCm 上的 **Profile 机制解析** + **MOE 模型显存雪崩根因与自定义 HIP 流修复方案**。

## 解决什么问题

在 DCU 上用 ORT/ONNX 推理（尤其 MOE 等含并行子图的模型）时，常见三类痛点，本 skill 逐一给出可复现方案：

1. **显存爆掉 / 首次推理崩溃** —— MOE 模型仅 1.7G，默认多流调度却导致显存雪崩式增长直至 OOM。
2. **想把算子耗时拆清楚** —— 用 ORT 原生 Profile + 算子耗时统计工具，按 EP 降序定位慢算子。
3. **多流同步开销大 / If 等控制流算子超慢** —— 通过自定义单 HIP 流彻底消除隐式同步。

## 核心结论（实测）

- 自定义单 HIP 流后：整体推理耗时 **3.45s → 0.99s（↓71.4%）**，`If` 算子 **↓98.7%**。
- DCU 显存占用：从「持续增长爆显存」→ 稳定 **2.5G**（64G 卡上单 session 仅 4%）。
- 并发：默认流仅能 16~32 线（或 OOM），自定义流后 **64~128 线、显存 14%~28%**。

## 根因一句话

ROCm EP 中 **1 个 HIP stream 硬绑定 1 个 rocBLAS handle**；MOE 的并行 expert 触发 ORT 内部多流 → 大量 handle → 显存泄漏。自定义单流即破此链。

## 快速使用

```bash
# 1) 自定义单流推理 + 导出 Profile JSON（显存修复核心）
python3 scripts/ort_custom_stream_infer.py ./encoder.onnx

# 2) 把 Profile JSON 解析成算子耗时降序表 + CSV
python3 scripts/analyze_ort_profile.py onnxruntime_profile_*.json

# 3) 多线程多 Session 并发对比测试
python3 scripts/ort_multithread_bench.py --session_num 8 --thread_num 64 --loop_times 10
```

## 关键约束（坑）

- `user_compute_stream` 必须是**十进制字符串**（ORT 1.19.2 不支持十六进制）。
- `libhiprtc.so` 路径默认 `/opt/dtk/lib/libhiprtc.so`，按实际 DTK 路径调整。
- 1.19.2 单流下会有 **2 个** rocBLAS handle（1 自定义 + 1 初始化残留），均绑同一流，无害；ORT 1.20+ 才修成 1 个。
- ROCm-EP Profile 强依赖 `roctracer` + `LD_LIBRARY_PATH` 含 `$ROCM_HOME/lib`。
- 环境变量 `PADDING_MALLOC=0` / `HIP_REUSE_MODULE=1` 只是早期缓解，自定义流方案实施后无需它们。

## 来源

整合自<internal-author> 4 篇内部文档：ORT 1.19.2 Profile 机制解析和实践、ORT 在 DCU 上推理 MOE 模型显存异常问题的分析和解决、ort-moe 优化多线程性能对比测试、ORT 上的算子耗时统计工具。详见 SKILL.md「来源文档」表（含 内部文档 链接与 file_id 便于溯源）。

## 适用

- ORT 1.19.2 + ROCm 5.6/5.7（DTK 环境），DCU 全机型（K100-AI / BW150 等已实测）。
- 本机无 onnxruntime/torch 依赖，脚本仅做语法校验，需在目标 DCU 环境运行。
