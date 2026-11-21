# Triton Server + ORT 集成调优（调度层 × 算子层）

> 本文自 `perf/dcu-ort-perf/SKILL.md` §7 下沉（2026-10-16，SKILL.md 瘦身到 500 行内）。
> 来源：《Triton Server + ONNXRuntime 通用模型推理调优最佳实践》（飞书文档，YOLOv7 在 BW100 单卡实测）。

## 7. Triton Server + ORT 集成调优（调度层 × 算子层）

> **来源**：《Triton Server + ONNXRuntime 通用模型推理调优最佳实践》（飞书文档，YOLOv7 在 BW100 单卡实测）。本节只沉淀**方法论 / 配置模板 / 反模式**；完整量化案例数据见 `dcu-perf-cases` 案例 16。

### 7.1 两层正交模型
ORT 作为 Triton 后端部署（`platform: "onnxruntime_onnx"`）时，调优分两层，**正交、可叠加**：

| Triton 层（进程级、模型级） | ORT 层（Session 级、Op 级） |
|---|---|
| HTTP/gRPC 接收 | Execution Provider 选择 |
| 排队、批处理、限流 | Graph Optimization（ORT_ENABLE_ALL） |
| Instance 生命周期管理 | Memory Pattern / Arena |
| 输出缓存（Response Cache） | IO Binding（零拷贝） |
| 模型版本切换 / 热加载 | Threading（intra/inter op） |
| 监控 / Metrics 上报 | |

- **Triton 调优**解决「调度侧瓶颈」（请求排队、串行执行、Batch 未利用）。
- **ORT 调优**解决「算子侧瓶颈」（CPU/GPU 算子性能、内存分配、线程调度）。
- 关键认知：**Triton 本身不实现推理 kernel，是「调度 + 资源管理层」，底层依赖 Backend（onnxruntime_onnx）**——所以两层调优可叠加，先调 ORT 会话参数（算子侧）→ 再多实例（调度侧）→ 最后动态批处理（调度侧），每步独立可验证。

### 7.2 config.pbtxt 关键配置（Triton 侧写 ORT 参数）
以下是 Triton `config.pbtxt` 里写 ORT 会话参数 / 多实例 / 动态批处理的方式，**与 3.5（多线程）/ 4.4（多实例）/ 3.1（批处理）的 Python API 写法互补**：

```protobuf
name: "model"
platform: "onnxruntime_onnx"
max_batch_size: 8                         # 0=关闭；>0 开启动态批处理

dynamic_batching {
  preferred_batch_size: [1, 2, 4]         # 优选 batch 大小
  max_queue_delay_microseconds: 3         # 收集窗口（μs）
  preserve_ordering: false                # 乱序可换吞吐
}

instance_group [ { kind: KIND_GPU, gpus: [ 0 ], count: 4 } ]   # 多实例

parameters [
  { key: "execution_mode",         value: { string_value: "1" } },  # 1=并行
  { key: "intra_op_thread_count",  value: { string_value: "8" } },  # CPU 算子多线程
  { key: "inter_op_thread_count",  value: { string_value: "2" } }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator [ { name: "ROCM" } ]   # DCU 用 ROCM
  }
}
```

**要点**：
- `parameters` 是 Triton 侧写 ORT 会话参数，等价 Python API 的 `sess_options.intra_op_num_threads` 等（见 3.5）。
- `instance_group.count` 是 Triton 侧写多实例，等价 Python `DCUModelPool`（见 4.4）。
- `dynamic_batching` 是 Triton 侧写动态批处理，等价 ORT 动态批处理（见 3.1）。**前置：模型输入须支持动态 batch 维度**（用 `--dynamic --dynamic-batch` 重新导出 ONNX）。
- `gpu_execution_accelerator` 在 DCU 上用 `ROCM`。

### 7.3 七步调优 SOP（可复用）
1. **建基线**：默认配置跑完整压测（并发 1,2,4,8,16,32,64,128），记录极限 TPS / 饱和点 / DCU 利用率 / 显存 / P99。
2. **Profile 定位瓶颈**：框架级（perf_analyzer）+ ONNX 级（onnxruntime_perf_test）+ 硬件级（hy-smi）。关注：CPU vs GPU 耗时比例（找拖累端到端的非 GPU 算子）、DCU 算力利用率（是否真打满）、排队时间（延迟随并发线性增长 = 串行排队）。
3. **ORT 会话参数调优**：先 `intra_op_thread_count`（CPU 算子并行度）→ 再 `inter_op_thread_count` → 再 `execution_mode=1`。**必做线程数扫描（1,2,4,8,16,32），不要拍脑袋**。
4. **多实例并行**：`instance_group.count` 扫 1,2,4,8，在并发 = N×2 下测。**甜点判定：DCU 利用率接近 100% 且 P95 延迟未明显恶化**。显存约束：count × 模型大小 + 输入输出 buffer < DCU 显存 80%。
5. **动态批处理**：前置模型支持动态 batch（`--dynamic --dynamic-batch` 重导）；配 `dynamic_batching`（preferred_batch_size / max_queue_delay / preserve_ordering）。
6. **再 Profile 验证**：量化对比（TPS / P50 / P95 / P99 / DCU util / 显存）。判停：达业务 SLO，或 DCU 利用率 ≥ 90% 且无进一步优化空间。
7. **精度回归**：**绝对不能省**——回放真实数据集验证 mAP / Recall / F1。风险点：动态 batch 改了 `max_batch_size` 和 `dims`，确保导出 ONNX 时 shape 一致。

### 7.4 常见反模式
| 反模式 | 后果 | 正确做法 |
|---|---|---|
| ❌ 上来就改模型 / 改代码 | 失去问题域聚焦 | 先 Profile，找到瓶颈再决定改哪 |
| ❌ intra_op = 16 直接拍脑袋 | 线程调度开销 > 并行收益 | 必做线程扫描 |
| ❌ instance_group count = 8 | DCU 上下文切换过多，P95 反而变差 | count = 4 起步，配合利用率曲线找甜点 |
| ❌ 单跑「最优配置」压测 | 缺基线，评审无说服力 | 每次只改一个变量，保留前一次结果 |
| ❌ 调完不验证精度 | 实际业务无法使用 | 关键场景小批量精度验证 |

### 7.5 关键洞察 / 坑
- **CPU 后处理算子常是隐藏瓶颈**：检测类模型（YOLO 等）的 ArgMax/Gather/Concat 等后处理算子默认在 CPU 跑且单线程，耗时可能远超 GPU 算子（YOLOv7 案例中 ArgMax 是 CPU 耗时第一的算子，毫秒级）。Profile 时先看 CPU vs GPU 耗时比例。
- **多实例甜点不是「越多越好」**：count 超过甜点后 kernel 调度争抢加剧，P95 反而变差。用「利用率 ~100% 且 P95 不恶化」的曲线判定。
- **动态批处理要重导模型**：原 ONNX 是静态 shape 时，必须 `--dynamic --dynamic-batch` 重导，否则 `max_batch_size > 0` 不生效。
- **镜像**：`<internal_registry>/dcu/admin/base/custom:tritonserver-r24.11-ubuntu22.04-dtk26.04-py3.10-ort1.22.2`（与 6.3 的 kylin10sp3 镜像互补）。
- **启动**：`HIP_VISIBLE_DEVICES` 绑卡 + `LD_LIBRARY_PATH` 指向 ORT runtime 库；`curl localhost:8000/v2/health/ready` 验证服务就绪。
