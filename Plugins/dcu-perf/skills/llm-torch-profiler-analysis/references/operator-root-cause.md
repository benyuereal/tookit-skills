# 通用算子级 root-cause 归因（零依赖脚本）

> 与 dcu-vllm-trace-analysis（按类别占比视角）互补的算子级定位工作流。
> 2026-10-16 自 `perf/llm-torch-profiler-analysis/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 通用算子级 root-cause 归因（技能自带零依赖脚本）

本节吸收自 HYGON-AI/SkillHub 的 `torch-trace-operator-profiler`（Apache-2.0，
作者 HYGON-AI HCU performance team）。它补充本 skill 的一个**通用算子诊断**能力：
上面各节是面向 LLM serving（sglang/vllm/TRT-LLM）的 kernel/overlap/fuse/空泡报告，
本脚本面向**任意** `torch.profiler` trace 做**算子级时间归因**，尤其适合自定义扩展、
sparse/conv 算子、Triton kernel、非 LLM 模型，以及需要从 trace 反推源码根因的场景。

- 脚本：`scripts/torch_trace_scope_summary.py`（**零第三方依赖**，仅 Python 标准库，
  可直接在 DCU/ROCm 容器内运行；通过 HIP/CUDA runtime `correlation` 归因 GPU kernel，
  而非墙钟重叠）。
- 与 `dcu-vllm-trace-analysis`（按类别出 xlsx）互补：本脚本偏**任意 torch trace 的
  单目标下钻**；与 `llm-torch-profiler-analysis` 主流程（kernel/overlap/fuse 表）互补：
  本脚本偏**算子级 root-cause 闭环**。

### 工作流（Operator-level root-cause）

1. 明确 scope 与假设：trace 文件、目标算子名或源码行范围，以及当前机器能否重跑基准。
   若没有目标 GPU/runtime，则把"before"标记为 trace 测得值，"after"标记为预期/目标值，
   直到用户在真实环境重跑。

2. 提取 trace 摘要（优先用自带脚本）：
   ```bash
   python scripts/torch_trace_scope_summary.py trace.json \
     --target label="substring or exact profiler scope" \
     --target other_label="another scope" \
     --top 25
   ```
   用精确源码行 scope 名（如 `spconv/pytorch/ops.py(450): get_indice_pairs`）。
   用 runtime `correlation` 归因 GPU kernel，`External id` 仅用于分组嵌套 CPU/ATen 事件。

3. 对热点 scope 读源码：Python wrapper -> C++ extension binding -> CUDA/HIP/Triton
   kernel -> allocator/workspace 路径。把 profiler 事件映射到精确源码行：
   分配、dtype 转换、同步点、host-device 拷贝、sort/unique、gather/scatter、GEMM 启动、
   fallback 分支。**区分事实、假设、推断**。

4. 判定瓶颈类别（五分类）：
   - **Host/runtime**：大量 `hipMalloc`/`cudaMalloc`、`hipFree`、launch 开销、同步。
   - **Memory movement**：H2D/D2H/D2D 拷贝、memset、dtype 转换 kernel、gather/scatter。
   - **Algorithmic**：sort/unique、hash 冲突、重复小 GEMM、未融合 gather-GEMM-scatter。
   - **Compute/ISA**：剔除开销后占主导的少量自定义 kernel。
   - **Fallback/configuration**：意外 Python fallback、禁用隐式路径、缺扩展、dtype/device 不匹配。

5. 按优先级构建优化计划：从高置信度、已测开销的修复开始；对融合算子给出最小融合边界
   与需要的 backward/autograd 路径；对 Triton/TorchInductor 从生成 kernel 回溯到 FX
   graph 语义，仅当确实减少实测工作量时才考虑窄 graph pass 或自定义 op；对 CUDA/HIP
   汇编，仅当 trace 显示自定义 kernel（而非 host/runtime 或库调用）占主导时才询问架构细节。

6. 产出报告：methodology、before 基线表、per-scope kernel/runtime 拆分、源码映射、
   优化计划、before/after 对比表、验证命令、未验证风险。**绝不把预测的 after 数值
   当作实测结果**。

### 脚本输出（`torch_trace_scope_summary.py`）

解析 Chrome trace JSON 并打印：

- event category 计数
- 每个 `--target` 的 count / total / avg / p50 / p90 / p99 / max
- 嵌套 CPU/ATen 事件拆分（按 category 与按 name）
- HIP/CUDA runtime correlation 计数
- 归因后的 GPU kernel、memcpy/memset、runtime API 汇总

若 trace 过大无法装入内存，回退用 `rg`/`jq` 采样，或用 Perfetto 打开肉眼查看同一 scope。

### 报告 checklist

- 已写明 target scope 名与 trace 文件。
- before 数值 trace 可复现。
- GPU kernel 归因使用 runtime correlation（可用时）。
- 源码论断精确到文件名与行号。
- 优化按实测影响与落地风险排序。
- after 数值要么在目标 GPU 上实测，要么明确标注为预期目标。
- 交付包含重跑基准与重新生成摘要的命令。
