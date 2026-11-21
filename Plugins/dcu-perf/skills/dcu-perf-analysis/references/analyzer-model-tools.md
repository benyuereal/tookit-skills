# AnalyzerModelTools — 离线性能预估工具参考

> 本 reference 提炼自《op_analyzer介绍.md》与《model_analyzer介绍.md》（AnalyzerModelTools-master.zip 内含两份说明书）。
> 用途：在 **DCU 测试/调优场景** 里，这两个工具是 **离线性能预估方法学**——不用真跑硬件/模型，几秒内给出 eval_time / tflops / bandwidth / 并行配置下的 prefill/decode 延迟。适合**标前测试、方案对比、算子 shape 导出、并行策略预研**。
> 注意：本机当前 glibc 2.28 跑不了 C++ 后端（两者含 libonnxruntime.so / libop_analyzer.so），需在 **Python 3.10 + glibc ≥ 2.29 的配套环境**（如 DCU 服务器）才能真实跑。源码与测试脚本已收进 `references/op_analyzer_test/` 与 `references/model_analyzer_py_src/`。

---

## op_analyzer — 单算子性能分析器

给定算子的 **shape / dtype / 计算模式**，估算其在目标设备上的 `eval_time` / `tflops` / `bandwidth` / `tensor_core_flops`。支持 **59 个独立算子**（GEMM、attention MHA/MLA、RMSNorm、MoE 系列、通信算子、GDN/DeepSeek-V4 融合算子等），覆盖 `a100` / `a800_40` / `a800_80` / `anshi` / `bmz` / `h20_141` / `h20_96` / `nmz` / `s150` / `s2000` / `s200` / `s4000` 设备。

### 三种分析模式
| 模式 | 调用 | 含义 |
|---|---|---|
| **compute** | `set_compute_mode("compute")`，`analyze_operator(op_type, op, compute_time_ns)` | 已知实测耗时 → 输出 `tflops` / `bandwidth` / `tensor_core_flops` |
| **eval** | `set_compute_mode("eval")`，`analyze_operator(op_type, op, 0.0)` | 估算 `eval_time` |
| **指定利用率** | `analyze_operator(op_type, op, 0.0, device=, tensor_core=, vector=, sfu=, bandwidth=)` | 指定算力利用率后估算耗时 |

利用率参数传 `< 0` 表示该项不覆盖、用 device JSON 默认值。

### Python 用法示例
```python
import operator_analyzer as opa

# 基本对象
op = opa.OpInfo("linear")
op.set_input_shape([[1024, 512], [512, 1024]])
op.set_output_shape([[1024, 1024]])
op.set_input_data_type([opa.HIPDATA_BF16, opa.HIPDATA_BF16])
op.set_output_data_type([opa.HIPDATA_BF16])
op.set_compute_mode("eval")

m = opa.OperatorAnalyzer().analyze_operator("linear", op, 0.0, device="nmz")
print(m["eval_time"], m["tflops"])
```
- 数据类型枚举：`HIPDATA_FP32/FP16/BF16/I32/I64/I8/FP8E4M3/FP4E2M1/FP8E8M0`。
- 查支持算子：`print(opa.OperatorAnalyzer().get_supported_operators())`。
- 查支持设备：文档列 12 种，设备名小写归一化。

### 独立测试脚本（59 个）
`references/op_analyzer_test/` 下每个算子一个**完全独立**的 `test_<op>.py`，覆盖 compute/eval/指定利用率三种模式，安装 wheel 后可直接 `python3 test_<op>.py` 单跑某算子（可复制到任意目录独立运行）。

> **坑（环境依赖）**：本机 Kylin V10 glibc 2.28 跑不了 `libonnxruntime.so`/`libop_analyzer.so`（报 GLIBC_2.29 not found）。需配套 Python 3.10 + glibc≥2.29 环境。wheel 是 `cp310`，python3.7.9 不匹配。

---

## model_analyzer — 整个 LLM 推理性能分析器

> 安装后既可在 Python 里 `import model_analyzer` 调用，也可用 `python -m model_analyzer.<模块>` 走命令行。用于离线**预估**模型在指定硬件 + 并行配置下的 prefill/decode 延迟、TPS、TTFT、显存占用。不需要真实跑模型，几秒出结果。

### 10 个功能（`model_analyzer()` / 各 sweep）
| 功能 | 用途 | 入口 |
|---|---|---|
| 1. `model_analyzer()` | 单次性能预估（prefill/decode 延迟、TPS、TTFT、显存） | import + `python -m model_analyzer.model_analyzer` |
| 2. `model_analyzer_util_sweep()` | 多组利用率对比，找满足模型性能所需的最小算力利用率 | import + `python -m model_analyzer.util_sweep` |
| 3. `util_profile` | 按算子/按阶段分别设利用率（`model_analyzer(util_profile=...)` 传 JSON） | `model_analyzer()` 参数 |
| 4. `pd_sweep()` | 固定节点数，遍历全部合法 (TP,DP,PP,EP,CP) 组合找最优 | import + `python -m model_analyzer.pd_sweep` |
| 5. `pd_separate_sweep()` | P/D 各自独立寻优 + 按请求率配平实例数（可不同 bs/input/out/decode 开 MTP） | import + `python -m model_analyzer.pd_separate_sweep` |
| 6. `supernode_sweep()` | 40 卡超节点（10 Tray × 4 GPU）拓扑并行策略 | import + `python -m model_analyzer.supernode_sweep` |
| 7. `tools.max_batch_size` | 给定显存上限+并行配置，求单卡最大 batch | import + `python -m model_analyzer.tools.max_batch_size` |
| 8. `tools.collect_shapes` | 打印单并行配置下 prefill/decode 每个算子的输入 shape | import + `python -m model_analyzer.tools.collect_shapes` |
| 9. `tools.shape_collector` | 批量算所有合法并行配置的 prefill/decode shape，写 CSV（供实测采集） | import + `python -m model_analyzer.tools.shape_collector` |
| 10. `serving_sim()` / `sweep_req_rate()` | 请求率(QPS)服务仿真：TTFT/TPOT/ITL mean&p99，扫 SLO 拐点 | import + `python -m model_analyzer.serving_simulator` |

### 常见使用模式速查（文档里的推荐功能表）
| 场景 | 推荐功能 | 关键参数 |
|---|---|---|
| 单配置性能 | `model_analyzer()` | P/D 各维度 + `chunk_prefill_size` |
| 利用率对比 | `model_analyzer_util_sweep()` | `utilization_scenarios=[{...}]` / `--scenarios` |
| 逐算子/逐阶段利用率 | `model_analyzer(util_profile=...)` | JSON `{prefill:{op:{...}, others:{...}}, decode:{...}}` |
| 某节点数下最优并行组合 | `pd_sweep()` | `total_nodes=...` |
| P/D 各自独立寻优+配平 | `pd_separate_sweep()` | `prefill_nodes` / `decode_nodes` / `decode_mtp_param` |
| 非标准 8 卡节点（40 卡超节点） | `supernode_sweep()` | `gpus_per_tray=4, num_trays=10` |
| QPS 服务仿真 / SLO 拐点 | `serving_sim()` / `sweep_req_rate()` | `req_rate=...` 或 `rates=...` |
| 最大 batch 计算 | `tools.max_batch_size` | `--num-gpus --tp --pp --ep` |
| 全算子 shape 导出（benchmark 采集） | `tools.shape_collector` | `--prefill_seq_lens` / `--decode_*` |

### 支持的模型（别名）
`deepseek_v3`/`r1`/`v3_2`/`v4`/`v4_5t`/`v4_10t`/`v4_flash`，`qwen2.5_72b`/`qwen3_32b`/`qwen3_moe_30b_a3b`/`qwen3_vl_moe_235b`/`qwen3_5`/`qwen3_5_moe`/`qwen35_35b_a3b`，`glm5`/`glm5_fp8`，`kimi_k2`/`kimi_k25`/`kimi_k2_6_w4a16`/`kimi_k2_6_w8a8`，`step_3_5_flash`/`step_3_7_flash`，`mimo_v2_flash`/`mimo_v2_5_pro`，`minimax_m2`/`minimax_m2_w8a8`，`hunyuan_v3`/`hy3`/`hy3_fp8`，`llama2_7b`，`gpt_moe` 等。

### 支持硬件
`NMZ`（默认）、`s2000`/`s200`/`s200_tx`/`s150`、`nmz`/`bmz`、`h20_141`/`h20_96`、`a100`/`a800_40`/`a800_80`。设备名只做小写归一化，需传 `configs/devices/` 下真实存在的文件名。

### Python 用法示例（单次 + 利用率 sweep + shape 导出）
```python
from model_analyzer import model_analyzer, model_analyzer_util_sweep
from model_analyzer.tools.shape_collector import collect_shapes

# 单次性能预估
result = model_analyzer(
    model="deepseek_v3_2", gpu="nmz",
    batch_size=32, seq_len=8192, output_length=1024, chunk_prefill_size=8192,
    prefill_tp=8, prefill_dp=1, prefill_pp=1, prefill_ep=1, prefill_cp=1,
    decode_tp=8,  decode_dp=1,  decode_pp=1,  decode_ep=1,  decode_cp=1,
    prefill_nodes=1, decode_nodes=1, dp_attention=False,
)
print(f"Prefill: {result['prefill_latency_ms']:.2f} ms")
print(f"Decode TBT: {result['decode_latency_ms']:.2f} ms")
print(f"System TPS: {result['sys_throughput_tps']:.0f} tokens/s")

# 多组利用率对比（找性能达标所需算力利用率）
out = model_analyzer_util_sweep(
    utilization_scenarios=[
        {"tensor_core": 0.6, "vector": 0.30, "sfu": 0.30, "bandwidth": 0.70},
        {"tensor_core": 0.8, "vector": 0.40, "sfu": 0.30, "bandwidth": 0.80},
    ],
    model="deepseek_v3_2", gpu="h20_96",
    batch_size=32, seq_len=8192, output_length=512, chunk_prefill_size=8192,
    prefill_tp=8, prefill_dp=1, prefill_ep=1,
    decode_tp=8,  decode_dp=1,  decode_ep=1,
    prefill_nodes=1, decode_nodes=1, dp_attention=False,
)
print(out["merged_xlsx"])  # 合并表路径

# 全算子 shape 导出（供真实 benchmark 采集）
collect_shapes(
    model_json=resolve_model_json("deepseek_v4"),
    max_gpus=16, max_pp=4,
    prefill_seq_lens=[8192, 16384],
    decode_batch_sizes=[32, 64, 128],
    decode_kv_seq_lens=[1024, 4096],
    output_dir="raw_data/shapes_dsv4",
    device_type="nmz", memory_ratio=0.9,
)
```

### CLI 等价命令（10 个功能里 1/2/4/5/6/10 有 `python -m` 入口）
场景 A（纯 TP 并行）：
```bash
python -m model_analyzer.model_analyzer     --model deepseek_v3_2 --gpu h20_141     --batch_size 16 --seq_len 4096 --output_length 512 --chunk_prefill_size 4096     --prefill_tp 8 --prefill_dp 1 --prefill_ep 1 --prefill_cp 1     --decode_tp 8  --decode_dp 1  --decode_ep 1     --prefill_nodes 1 --decode_nodes 1     --dp_attention false
```
场景 B（sglang deepseek 部署）：
```bash
python -m model_analyzer.model_analyzer     --model deepseek_v4 --gpu nmz     --batch_size 128 --seq_len 4096 --output_length 512 --chunk_prefill_size 4096     --prefill_tp 8 --prefill_dp 4 --prefill_ep 8     --decode_tp 8  --decode_dp 4  --decode_ep 8     --prefill_nodes 4 --decode_nodes 4     --dp_attention true     --moe_dp_size 2 --moe_dense_tp_size 1 --moe_a2a_backend deepep
```

---

## 坑与注意（本工具整合进 skill 时实测到的）

1. **本机 glibc 跑不了**：Kylin V10 glibc 2.28，两者 C++ 后端依赖 GLIBC_2.29（`/usr/lib64/libm.so.6` 缺版本），报 ImportError。**需 Python 3.10 + glibc ≥ 2.29 的配套环境**（DCU 服务器）。
2. **`analyze_operator` 需传 device**：op_analyzer `eval` 模式默认 device 为空串，指定利用率路径要显式 `device="nmz"` 等，否则部分算子可能落到默认逻辑。
3. **`resolve_model_json` 转换**：`model_analyzer.tools.shape_collector` 接收的是模型 **JSON 路径**（别名先用 `resolve_model_json("deepseek_v4")` 转换）——不是直接传别名 string。
4. **decode token 维语义（易错）**：`tools.collect_shapes` 的 token 维完全由 `--seq-len` 决定，`--batch-size` 不进底层。decode 每步每请求只产 1 个新 token，token 维 = 并发 batch；`--kv-seq-len` 才是已生成的上下文长度。
5. **`chunk_prefill_size` 对 M 的钳制规则**：prefill token 维 `eff_seq = chunk if 0<chunk<seq else seq`，即 `M = min(seq, chunk)`；只钳 query/token 维，KV 维不钳。
6. **op_analyzer wheel 是 cp310**：本机 python3.7.9 不匹配，装不上。**operator_analyzer 的 `try:` 块因 ABI/glibc 失败进入 except → `__all__` 空**——`AnalyzerModelTools` 文档里写好的用法示例在坏环境下会报 AttributeError。
7. **`get_supported_operators()` 与设备目录**：设备名小写归一化，需传 `configs/devices/` 下真实文件名（如 `nmz`），无别名表。
8. **仅影响 Roofline 估算路径**：`util_profile` 只影响 Roofline 估算；命中实测 CSV 的算子不受影响，通信算子因带宽单独建模也不吃此利用率。
9. **output 格式**：批量计算型功能（pd_sweep/supernode_sweep/shape_collector/serving_sim）在 `output_dir/` 下生成汇总 Excel（Top-N + OOM）与每配置详细 Excel（raw_data + summary 两 sheet）。

---

## 与 DCU 测试场景的衔接建议（方法论）

- **标前测试**：用 model_analyzer 离线预估目标模型在 DCU 硬件 + 并行配置下的 prefill/decode TPS/TTFT → 比选型/方案拍板。
- **算子 benchmark 采集**：用 tools.shape_collector 一次性生成全部合法并行配置的算子 shape 清单 → 在 DCU 服务器上跑真实 benchmark 采集 → 回灌 dcu-perf-profiling / dcu-op-optimization。
- **并行策略预研**：pd_sweep 遍历 TP/DP/PP/EP/CP 组合，pd_separate_sweep 做 P/D 独立寻优 + 请求率配平。
- **QPS 服务能力估**：serving_sim 扫 SLO 拐点，避开「盲改 max_running_requests」。

这些工具输出的估算数字是「离线预估」，**不是实测基线**——部署前仍须在 DCU 真机验证（见 dcu-model-deploy-verify 与 dcu-vllm-deploy）。

---
