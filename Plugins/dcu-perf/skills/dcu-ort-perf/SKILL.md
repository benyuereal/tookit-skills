---
name: dcu-ort-perf
description: "海光DCU上ONNX Runtime推理引擎性能优化最佳实践（perf_test测量/批处理/FP16/EP选择/图优化/多线程/io_binding/内存复用/多实例/自定义算子/环境配置/Triton Server集成调优）"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# 海光 DCU ONNX Runtime 最佳性能实践（dcu-ort-perf）

适用场景：在 **DCU 平台用 ONNX Runtime（ORT）做推理引擎性能优化**——从性能测量（onnxruntime_perf_test）、模型级优化（批处理/FP16/EP 选择/图优化/多线程）、客户 AI 应用全链路优化（io_binding/内存复用/多实例提 QPS）、自定义算子开发，到环境配置的全链路最佳实践。

> **来源**：《OnnxRuntime-DCU 最佳性能实践手册》（基于 DAS-1.7 的 ONNXRuntime 使用手册 v1.19.2 + TensorRT 最佳性能实践方法论 + 实际项目优化经验）。原文 3 处嵌入图片，已用 qwen3.8-27b 视觉解析转成 markdown 内嵌（见各节标注「📊 原图」）。
> ⚠ **DCU 适配身份**：ORT 通过 **ROCm EP**（动态 shape）/ **MIGraphX EP**（静态 shape）跑 DCU；`device_type="cuda"` 是 ROCm 兼容层标准写法（ROCm 沿用 CUDA 设备类型标识保持 API 兼容）。
> ⚠ **姊妹篇分工**：本 skill 是 **ORT 性能优化方法论**；`dcu-ort-profile` 是 **ORT Profile 机制 + MOE 显存雪崩 HIP 流修复**（剖析/排障侧）；`dcu-migraphx-perf` 是 **MIGraphX 引擎性能优化**（MIGraphX EP 的引擎侧优化直接参考它）。三者互补。
> ⚠ **示例输出非实测基线**：文中 `119.201 inferences/sec`、`8.37121 ms` 等数字是**手册示例输出**（用于教如何读报告），非本环境实测基线；实测性能请用 `dcu-inference-optimization` 的方法学评估。
> ⚠ **长代码外置** `scripts/`（10 个 Python + 1 个 C++，py_compile 验证通过）。本 SKILL.md 只讲方法与关键片段。

## 0. 一句话机制

ORT 性能优化 = **先测量定位瓶颈（onnxruntime_perf_test + profiling）→ 模型级提吞吐（批处理/FP16/选对 EP/图优化/多线程）→ 应用全链路抠系统开销（io_binding 零拷贝/内存池复用/多实例提 QPS）→ 自定义算子补能力 → 环境配置调优**。核心指标是 **Number of inferences per second（QPS）** 与 **Average inference time cost（平均推理耗时）**，延迟敏感场景看 **P99 Latency**。

---

## 1. 性能测量指标的确认（纯模型性能）

### 1.1 延迟与吞吐量
- **延迟**（推理耗时）：单个推理从输入到输出所经过的时间。本文档指 **GPU 上的模型从接收输入到输出推理结果的耗时**，**不含**算法前/后处理，也**不是**单个请求在应用系统从发出到 AI 服务给出业务结果的耗时。
- **吞吐量**：单位时间内完成的推理数量。本文档指 **GPU 上的模型在单位时间仅进行模型推理完成的推理数量**，同样不含前后处理。

> 区分「纯模型性能」（第 1~3 章）与「客户 AI 应用全链路性能」（第 4 章，含数据传输/内存分配/并发调度）。

---

## 2. onnxruntime 默认性能测量工具

### 2.1 基础使用
`onnxruntime_perf_test` 工具做性能分析：`onnxruntime_perf_test [options...] model_path [result_file]`

```bash
# 测试模式 times（固定次数）：
onnxruntime_perf_test -m times \
  -e rocm \
  -r 20 \
  -o 1 \
  -I \
  <onnx_file_path>

# 测试模式 duration（固定时长）：
onnxruntime_perf_test -m duration \
  -e rocm \
  -t 60 \
  -o 1 \
  -I \
  <onnx_file_path>
```

📊 **原图**（p02，关键参数说明）：

| 参数 | 说明 |
| :--- | :--- |
| `-m` | 测试模式：`duration`（跑固定时长）或 `times`（重复固定次数） |
| `-r` | `times` 模式下的重复次数，默认 1000 |
| `-o` | 优化等级，默认 99（all）。有效值：0（disable）/1（basic）/2（extended）/99（all） |
| `-t` | `duration` 模式下运行的秒数，默认 600 |
| `-l` | 生成 tensor input binding（自由维度按 1 处理） |
| `<onnx_file_path>` | 位置参数，指定要测试的 onnx 模型文件路径 |
| `-e` | 执行提供者，常用值 `[cpu\|cuda\|dnnl\|tensorrt\|openvino\|rocm\|migraphx]`，默认 `cpu`。**DCU 上一般选 `rocm` 或 `migraphx`** |
| `-v` | 显示详细信息 |
| `-s` | 显示统计结果（P75/P90 等）。未提供 result_file 时默认开启 |
| `-p` | 指定 profiling 文件名，启用后分析数据 dump 到该文件 |

### 2.2 性能报告解读
输出含两部分：

**① 最终结果报告**（手册示例输出，非实测基线）：
```
Session creation time cost: 3.55993 s
First inference time cost: 810 ms
Total inference time cost: 0.167424 s
Total inference requests: 20
Average inference time cost: 8.37121 ms
Total inference run time: 0.167784 s
Number of inferences per second: 119.201
Avg CPU usage: 0 %
Peak working set size: 3654443008 bytes
Runs: 20
Min Latency: 0.00803848 s
Max Latency: 0.0116983 s
P50 Latency: 0.00807698 s
P90 Latency: 0.00874087 s
P95 Latency: 0.0116983 s
P99 Latency: 0.0116983 s
P999 Latency: 0.0116983 s
```
常用指标：
- **Number of inferences per second**：每秒可推理次数（QPS）。
- **Average inference time cost**：平均推理耗时。
- **P50/P90/P95/P99 Latency**：延迟分位数，延迟敏感场景重点看 P99。

**② 加 `-p` 后的 profiling 报告**：
```bash
onnxruntime_perf_test -m times \
  -e rocm \
  -r 20 \
  -o 1 \
  -v \
  -s \
  -p profile.log \
  -I \
  ../resnet50_1x3x224x224.onnx
# Profile 文件为: profile.log_2026-06-10_11-25-58.json
```
📊 **原图**（p04，profiling 日志示例）：profile 文件是 **Chrome tracing 风格的 JSON 事件流**（每行一个 JSON 对象，字段 `cat/pid/tid/dur/ts/ph/name/args`），记录模型加载、会话初始化及各 Conv/ReLU/MaxPool 算子的 fence/kernel 执行时间。`dur` 为当前算子耗时、`ts` 是时间戳，可据此定位性能较差的算子：
```json
{"cat":"Session","pid":2171,"tid":2171,"dur":58616,"ts":7,"ph":"X","name":"model_loading_uri","args":{}}
{"cat":"Session","pid":2171,"tid":2171,"dur":3341386,"ts":58732,"ph":"X","name":"session_initialization","args":{}}
{"cat":"Node","pid":2171,"tid":2171,"dur":1,"ts":3419901,"ph":"X","name":"/conv1/Conv_fence_before","args":{"op_name":"Conv"}}
{"cat":"Node","pid":2171,"tid":2171,"dur":27964,"ts":3419921,"ph":"X","name":"/conv1/Conv_kernel_time","args":{"thread_scheduling_stats":…}}
{"cat":"Node","pid":2171,"tid":2171,"dur":1,"ts":3447896,"ph":"X","name":"/conv1/Conv_fence_after","args":{"op_name":"Conv"}}
{"cat":"Node","pid":2171,"tid":2171,"dur":987,"ts":3447907,"ph":"X","name":"/relu/Relu_kernel_time","args":{"thread_scheduling_stats":{…}}}
{"cat":"Node","pid":2171,"tid":2171,"dur":221,"ts":3448904,"ph":"X","name":"/maxpool/MaxPool_kernel_time","args":{"thread_scheduling_stats":…}}
```
> 注：原文档排版把这张 profiling 示例图放在了 3.2.1 节之后，实际归属 2.2 节。ORT profile 机制深度解析 + MOE 显存雪崩 HIP 流修复见 `dcu-ort-profile`。

---

## 3. 模型性能优化最佳实践

ORT 性能优化涉及精度选择、后端适配、图优化、并行策略多个维度，需从运行时配置、模型精度、硬件特性、计算图优化协同调优。

### 3.1 提升 Batch（批量处理）
增加 Batch Size 可显著提升 GPU 等并行计算单元的利用率和整体吞吐量——更大 Batch 意味着单次内核调用处理更多数据，摊薄内核启动和内存访问开销。
- **策略**：若服务延迟要求允许，建议 Batch Size 设为硬件最大利用率时的值（如 16/32/64）。CPU 上适中 Batch（2-4）延迟表现可能更好。
- **动态批处理**：变长输入或在线服务场景，可用 ORT 动态批处理，内部将多个小请求合并为大 Batch 处理。

### 3.2 低精度推理或混合精度推理
低精度计算是提升性能、降低显存带宽瓶颈的有效手段，GPU（Tensor Core）和部分 CPU（AVX-512 VNNI/AMX）上效果显著。

**3.2.1 FP16 推理**
支持 FP16 加速的 GPU 上，FP32→FP16 通常可获得 **1.5~2 倍加速**，对多数视觉模型精度影响微小。

转换工具（完整代码见 `scripts/fp16_conversion.py`）：
- `onnxconverter-common` 库（**推荐**，兼容性较好，能处理复杂网络结构并支持跳过不支持的算子）
- `onnxruntime.tools.convert_onnx_models_to_float16`
- `onnxmltools` 库

```python
import onnx
from onnxconverter_common import float16

model_fp32 = onnx.load("model_fp32.onnx")
# keep_io_types=True 保持输入输出为 FP32，避免外部接口类型冲突
model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
onnx.save(model_fp16, "model_fp16.onnx")
```

**不推荐转换为 FP16 的层**（FP16 数值范围约 ±6.55×10⁴，精度约 3 位小数）：
1. **归一化层**（BatchNorm/LayerNorm/InstanceNormalization）：均值方差计算对精度极敏感，全切 FP16 易引发数值异常。
2. **Softmax/Sigmoid 等激活函数**：指数运算易溢出/下溢。
3. **累加与求和**（ReduceSum/Accumulation）：舍入误差在累加中不断积累。
4. **转置卷积层**：某些硬件上转 FP16 后性能大幅下降。
5. **含大量 Conv/Gemm 且对误差敏感的模型**（如换脸模型 Inswapper）：层数多，FP16 下绝对/相对误差易超容差。

**转换注意事项**：
- ONNX 转换端用 `onnxconverter-common` 的 `auto_convert_mixed_precision`，通过迭代分治算法自动验证并保留必须为 FP32 的节点。
- **混合精度**：关键层（输出层/注意力层）保留 FP32，其余层 FP16。
- **INT8 量化**：性能极致场景，动态/静态量化权重，适用 CPU 及支持 INT8 指令的 GPU。

### 3.3 根据不同场景选择推理后端（EP）
ORT 通过 Execution Provider (EP) 机制抽象底层硬件，支持 ROCm、MIGraphX、TensorRT、CUDA、OpenVINO 等，EP 选择直接决定推理性能。**MIGraphX 相关的性能优化直接参考 `dcu-migraphx-perf`。**

- **静态 Shape + DCU**：模型输入形状固定 → 建议 **MIGraphX EP**。MIGraphX 是 DCU 专用图推理引擎，针对静态形状做深度算子融合、常量折叠、死代码消除，生成高度优化代码。
- **动态 Shape + DCU**：可变长度输入（ASR/TTS/NLP 变长句子/语音、检测可变尺寸图像）→ 建议 **ROCm EP**。相比 MIGraphX，ROCm EP 应对动态张量形状的实时编译开销更灵活。

**EP 选择示例（优先级机制）**：部分 EP 不满足会自动回退，部分模型特殊层不支持也会自动回退：
```python
providers = [
    ('ROCMExecutionProvider', {'device_id': 0}),  # 优先使用 Rocm
    ('CPUExecutionProvider',)                      # 最终回退 CPU
]
session = ort.InferenceSession(model_path, providers=providers)
```

### 3.4 启用高优化等级
ORT 内置图优化器自动执行算子融合（Conv+BatchNorm+Relu→FusedConv）、常量折叠、冗余节点消除。**生产环境应始终开启最高级别优化。**

优化等级：
- `ORT_ENABLE_BASIC`：仅模式无关优化 + 内存布局优化。
- `ORT_ENABLE_EXTENDED`：基础优化 + 扩展模式（如算子融合）。
- `ORT_ENABLE_ALL`：所有优化全开（布局优化/内存复用等），**生产环境推荐**。

**使用建议：初次生成，后续复用**（完整代码见 `scripts/optimized_model_workflow.py`）——把模型优化和实际推理分开：
```python
# 第一步：生成优化模型（离线优化脚本）
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "optimized_model.onnx"
session = ort.InferenceSession("model.onnx", sess_options)  # 触发优化和保存

# 第二步：加载优化模型（实际推理服务）
sess_options2 = ort.SessionOptions()
sess_options2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 已优化，不再优化
session2 = ort.InferenceSession("optimized_model.onnx", sess_options2)  # 启动更快
```

### 3.5 启用多线程加速
当模型无法所有节点都在 DCU 运行时，部分算子跑在 CPU，合理配置线程数可提升性能；不当配置会引发线程争用导致性能下降（完整代码见 `scripts/multithread_config.py`）。

```python
import psutil
sess_options = ort.SessionOptions()
cpu_count = psutil.cpu_count(logical=False)  # 物理核心数（非逻辑核心数）
sess_options.intra_op_num_threads = cpu_count
sess_options.inter_op_num_threads = 1  # 大部分模型设为 1 即可
```
- **intra_op_num_threads**：单个操作（如矩阵乘法）内部并行线程数，建议=物理 CPU 核心数，避免超线程资源竞争。
- **inter_op_num_threads**：计算图中不同操作之间的并行度。线性结构模型（ResNet）设 1 通常最高效；分支并行结构（含 Gather/If）可适当增加。
- 严格资源占用要求场景：全部设为 1，推理仅占 1 个 CPU 核心。

### 3.6 分析哪些算子 fallback 到 CPU（定位性能/兼容问题）
- 模型在 ORT 上部分算子跑不到 DCU、回退 CPU 时（性能差或特定数据崩溃），先**定位具体是哪些算子 fallback**：
  - 用 ORT 的算子级 profile / 计算图检查（`sess.get_profiling_summary()` 或计算图 dump），看每个算子实际跑在哪个 EP（DCU EP vs CPU EP）。
  - 对照模型算子清单，找出落在 CPU EP 的算子——这些就是 DCU EP 不支持/未覆盖的算子。
- 定位后两条路：① 反馈研发补 DCU EP 算子支持（推动产品修复）② 临时用 CPU EP 跑这些算子 + 按 §3.5 配多线程缓解。
- 价值：把"模型整体慢/崩"收敛到"具体哪几个算子不在 DCU 上"，问题定位效率显著提升（OCR 项目实战）。

---

## 4. 客户 AI 应用优化最佳实践（全链路）

### 4.1 实践说明
聚焦从业务请求发起到最终业务结果返回的**全链路效率优化**——真实生产环境下提升 QPS 与响应延迟。典型场景如 OCR（指标 QPS，目标 P99 延迟 < 500ms 前提下最大化单节点吞吐）。瓶颈通常不限于模型计算，更多在**数据传输开销、内存分配成本、并发调度策略**。

> 推荐组合顺序：① 先启用**内存预开辟+复用**（收益明显且实现简单）→ ② 再根据数据位置采用 **io_binding**（输入已在设备内存时）→ ③ 最后通过**多模型实例**突破单实例吞吐瓶颈，并按形状特征选 ROCm EP（动态）/MIGraphX EP（静态）。组合使用通常可将 QPS 提升 **2~5 倍**，同时降低 P99 延迟。

### 4.2 利用 io_binding 手动控制内存交互时机
**问题**：默认 ORT 每次推理自动在 CPU 与 DCU 间复制输入/输出张量（隐式同步），低延迟/高吞吐场景引入不必要开销，尤其输入数据本就在设备内存时两次拷贝显著增加延迟。

**原理**：io_binding 允许预先在设备内存（DCU）分配输入/输出张量，显式控制数据拷贝与计算的顺序，消除 ORT 内部自动执行的冗余内存传输。

完整代码见 `scripts/io_binding_zero_copy.py`，关键骨架：
```python
session = ort.InferenceSession("model.onnx", providers=['ROCMExecutionProvider'])
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
output_tensor = np.empty((1, 1000), dtype=np.float32)

io_binding = session.io_binding()
io_binding.bind_input(name="input", device_type="cuda", device_id=0,
                      element_type=np.float32, shape=input_tensor.shape,
                      buffer_ptr=input_tensor.ctypes.data)  # ROCm 兼容 cuda 标识
io_binding.bind_output(name="output", device_type="cuda", device_id=0,
                       element_type=np.float32, shape=output_tensor.shape,
                       buffer_ptr=output_tensor.ctypes.data)
session.run_with_iobinding(io_binding)  # 无隐式内存拷贝
result = output_tensor.copy()           # 仅必要时拷回 CPU
```
**适用场景**：输入本就在 DCU 内存（视频解码帧/GPU 预处理张量）；输出可直接在 GPU 消费；高吞吐下内存拷贝成瓶颈。
**收益**：可减少 **20%~50%** 单次推理延迟（取决于输入数据量及硬件带宽）。
**ROCm 特别说明**：`device_type="cuda"` 是 ROCm 兼容层标准写法（ROCm 沿用 CUDA 设备类型标识保持 API 兼容）。

### 4.3 内存预开辟 + 复用（DCU 环境最佳实践）
**问题**：每次推理 ORT 动态申请输入/输出张量内存（尤其 CPU↔DCU 传输时），频繁分配/释放带来开销并增加内存碎片风险，影响服务稳定性。

**原理**：预先在 CPU 和 DCU 显存中开辟固定大小内存池，多次推理复用，配合 io_binding 实现零拷贝数据传输，彻底避免重复内存分配/释放。

完整可运行类 `ReusableDCUInference` 见 `scripts/memory_pool_reuse.py`（CPU 内存池 + DCU 显存池一次开辟多次复用 + `predict`/`predict_without_cpu_copy` 零拷贝 + 使用示例）。核心优化点：

| 优化层级 | 实现方式 | 收益 |
| :--- | :--- | :--- |
| CPU 内存池 | `enable_cpu_mem_arena=True` + 预分配 `cpu_input_buffer` | 消除 CPU 端动态分配 |
| DCU 显存池 | `ort.OrtValue.ortvalue_from_numpy` 预分配显存 | 消除显存动态分配 |
| 零拷贝传输 | io_binding 绑定预分配显存，复用同一块显存区域 | 避免重复内存分配/释放 |
| HIP Graph 优化 | `enable_hip_graph=True` | 减少内核启动开销 |

**配置建议**：
1. 监控显存使用：用 `hy-smi` 查看 DCU 显存占用，确保预分配空间充足（见 `dcu-hy-smi`）。
2. 动态 Batch 处理：预分配最大 Batch 空间，实际推理只使用前 N 个样本。
3. 多流并发：多实例场景每个实例独立预分配显存，避免资源竞争。

**收益**：显存分配开销从每请求数百微秒降至接近零；结合 io_binding 可降低 10% 推理延迟。

### 4.4 多模型实例提升 QPS
**问题**：单个 Session 处理并发请求时模型计算串行（或 DCU 在推理外阶段空闲），QPS 上限低，尤其 I/O 密集型或小模型。

**原理**：部署多个模型实例（多个独立 InferenceSession）+ 请求级负载均衡，实现请求级并行，显著提升系统总吞吐量。

完整代码见 `scripts/multi_instance_pool.py`（CPU 多进程池 `ModelPool` + DCU 多实例池 `DCUModelPool`）。DCU 场景关键：
```python
class DCUModelPool:
    def __init__(self, num_instances=4, device_id=0, ep_type="ROCMExecutionProvider"):
        self.pool = queue.Queue()
        for _ in range(num_instances):
            # 动态 shape 用 ROCm EP，静态 shape 用 MIGraphX EP
            session = ort.InferenceSession("model.onnx",
                providers=[(ep_type, {'device_id': device_id,
                                      'hip_mem_limit': 2 * 1024 ** 3})])  # 每实例显存上限 2GB
            self.pool.put(session)

    def predict(self, input_data):
        session = self.pool.get()
        try:
            return session.run(None, {"input": input_data})
        finally:
            self.pool.put(session)
```

**最佳实践建议（DCU 平台）**：

| 场景 | 推荐实例数 | 后端 EP 选择 | 注意事项 |
| :--- | :--- | :--- | :--- |
| DCU 静态 shape 小模型 | 2~4 | MIGraphXExecutionProvider | MIGraphX 算子融合更激进，吞吐量高 |
| DCU 动态 shape | 2~4 | ROCMExecutionProvider | 动态形状下 ROCm EP 更灵活 |

**多 GPU 扩展**：多卡 DCU（如多块 K100-AI 系列加速卡）可在每个 GPU 上独立部署模型实例池，通过请求分发层（哈希/一致性哈希）路由到不同 GPU 实例，实现多卡线性扩展。

**收益**：CPU 场景 QPS 随实例数近似线性增长（直至内存/带宽瓶颈）；DCU 场景多实例可填充 GPU 空闲周期，通常提升 **30%~100%** 吞吐量，结合 MIGraphX EP 静态 shape 优化收益更显著。

---

## 5. 自定义算子开发步骤（附完整案例）

当模型包含标准 ONNX 算子库不支持的操作时，需通过自定义算子扩展 ORT。含 **Python 快速实现**和 **C++ 生产级实现**两种方案。

### 5.1 实现自定义算子的核心逻辑
用 `@onnx_op` 装饰器将 Python 函数封装为 ORT 可识别的自定义算子，需明确指定算子类型、输入输出张量类型及属性参数。

**Python 实现**（完整代码见 `scripts/custom_op_python.py`）——逐元素乘加算子 `MyCustomOp`，计算 `output = x * y + bias`：
```python
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef

@onnx_op(
    op_type="MyCustomOp",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
    attrs={"bias": PyCustomOpDef.dt_float}
)
def my_custom_op(x, y, **kwargs):
    bias = kwargs.get("bias", 1.0)
    return x * y + bias
```

**C++ 高性能实现**（生产环境推荐，完整代码见 `scripts/custom_op_cpp.cpp`）——继承 `Ort::CustomOpBase`，实现 `CreateKernel`/`GetName`/`GetExecutionProviderType`/`GetInputType`/`GetOutputType`，导出 `RegisterCustomOps` 注册函数编译为动态库。
> ⚠ 示例 `Compute` 用 CPU 循环做伪代码；**实际在 DCU 应使用 HIP kernel 编程实现并行**，可通过 `Ort::Custom::RocmContext` 获取 HIP 流和 miopen 句柄。

### 5.2 创建包含自定义算子的 ONNX 模型
完整代码见 `scripts/build_onnx_with_custom_op.py`。两种方式：
- **方式一**：ONNX Python API 从头构建——`helper.make_node(op_type="MyCustomOp", domain="my.domain", bias=2)` + `make_graph` + `make_model`（建议自定义域避免冲突）。
- **方式二**：从 PyTorch 导出时映射——`register_custom_op_symbolic("::mul", my_op_symbolic, 9)` 把 torch 操作映射到自定义 ONNX 算子，再 `torch.onnx.export`。

### 5.3 注册自定义算子并执行推理
完整代码见 `scripts/register_and_infer.py`。推理时需将自定义算子注册到 ORT 会话：
- **Python 算子**：直接注册（`@onnx_op` 定义后即可）。
- **C++ 算子**：`sess_options.register_custom_ops_library("./libmy_custom_op.so")` 加载动态库。
- **ROCm 环境**：providers 用 `ROCMExecutionProvider`（`device_id`/`arena_extend_strategy`），数据拷贝可用 io_binding 优化。

### 5.4 单元测试与验证
完整代码见 `scripts/custom_op_validate.py`：① 正确性验证（对比自定义算子与原生实现，看 `np.max(np.abs(diff))`）；② 性能基准测试（warmup + 多次迭代测平均延迟）。

### 5.5 核心步骤总结

| 步骤 | 关键操作 | 适用场景 | 核心 API/工具 |
| :--- | :--- | :--- | :--- |
| 1. 实现算子 | Python `@onnx_op` 或 C++ 继承 `CustomOpBase` | Python 快速原型；C++ 生产高性能 | `onnxruntime_extensions`、`CustomOpBase` |
| 2. 构建模型 | ONNX API 创建节点或框架导出时映射 | 新模型手动建图；已有模型改文件/符号函数导出 | `helper.make_node`、`register_custom_op_symbolic` |
| 3. 注册推理 | SessionOptions 加载自定义算子库 | 所有推理场景 | `register_custom_ops_library`、`InferenceSession` |

### 5.6 高级优化建议
1. **GPU/ROCm 设备支持**：自定义算子可访问 GPU 上下文（CUDA/HIP stream），实现设备端高效计算。DCU 上通过 `Ort::Custom::RocmContext` 获取 HIP 流和 miopen 句柄。
2. **多数据类型支持**：用模板函数支持 float/int32/int64 等，避免为每种类型重复实现。
3. **算子融合**：将多个连续自定义算子与标准算子融合，减少内存访问和内核启动开销，可提升 **30-50%** 性能。

---

## 6. 环境配置与调优

### 6.1 关键环境变量

📊 **原图**（p25，关键环境变量）：

| 变量 | 作用 |
| :--- | :--- |
| `ORT_ROCM_MULTITHREAD_ENABLE=1` | 启用单个 session 的多线程调用；功能性验证可以采用；**实际生产中性能优化建议手动在 session 推理的时候上锁**（原文红色强调） |
| `PADDING_MALLOC=0` + `HIP_REUSE_MODULE=1` | 节省 3-4 倍的显存占用（两变量同一单元格共用此说明） |

> 注：p25 原图底部被截图裁切（最后一行下边框未显示），无法确认表格后面是否还有更多行；可见的 2 行数据已完整转录。

### 6.2 系统配置
```bash
# 增加 swap 空间（显存不足时）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 6.3 Docker 运行推荐
```bash
docker run --shm-size=16g --network=host \
            --device=/dev/kfd --device=/dev/dri \
            --group-add video \
            -v /opt/hyhal:/opt/hyhal:ro \
            -it <internal_registry>/dcu/admin/base/custom:tritonserver-r24.11-kylin10sp3-dtk26.04-py3.10-ort1.19.2 /bin/bash
```
> ⚠ **DCU 容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`**，否则 DCU 不可见（DEVCOUNT=0）。容器操作规范见 `dcu-docker`。

---

## 7. Triton Server + ORT 集成调优（调度层 × 算子层）

两层正交可叠加：**Triton 调度层**（config.pbtxt：instance_group / dynamic batch / max_queue_delay）× **ORT 算子层**（图优化等级/线程/IOBinding）。
完整 config.pbtxt 配置、七步调优 SOP、常见反模式与关键洞察已下沉 `references/triton-server-integration.md`（YOLOv7 BW100 单卡实测来源）。
速记：CPU 后处理算子（ArgMax/Gather 等）常是隐藏瓶颈；动态批处理必须 `--dynamic --dynamic-batch` 重导模型；多实例甜点看「利用率 ~100% 且 P95 不恶化」。

---

## 关联 skill
- `dcu-ort-profile`：ORT Profile 机制深度解析 + MOE 显存雪崩 HIP 流修复（剖析/排障侧，与本 skill 的 2.2 profiling 互补）。
- `dcu-migraphx-perf`：MIGraphX 引擎性能优化（本 skill 3.3 节 MIGraphX EP 的引擎侧优化直接参考它）。
- `dcu-inference-optimization`：DCU 通用推理优化方法论（torch.compile/NHWC/TF32/MIOpen/rocBLAS），实测性能评估用此。
- `dcu-docker`：DCU 容器操作规范（挂 hyhal、起容器）。
- `dcu-hy-smi`：hy-smi 查询监控（4.3 节监控显存占用用此）。
