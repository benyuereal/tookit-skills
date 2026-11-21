---
name: dcu-migraphx-perf
description: "海光DCU上MIGraphX推理引擎性能优化最佳实践（测量/批处理/序列化/精度/融合/TensorCore/多框架/精度调试/环境配置）"
version: 1.0.0
author: DCU Engineer
license: MIT
---

# 海光 DCU MIGraphX 最佳性能实践（dcu-migraphx-perf）

适用场景：在 **DCU 平台用 MIGraphX（AMD/海光推理优化器）做推理引擎性能优化**——从性能测量、通用优化策略（批处理/序列化/精度/自动调优）、算子级优化（融合/视图/MHA/布局）、Tensor Core 对齐，到 Python API、多框架集成、精度调试、环境配置的全链路最佳实践。

> **来源**：《【DCU】DCU MIGraphX 最佳性能实践手册》（基于 MIGraphX_5.2.0 手册，参考 NVIDIA TensorRT 最佳性能实践的结构与方法论提炼）。原文 6 处「点击图片查看完整电子表格」的嵌入表格，已用 qwen3.8-27b 视觉解析转成 markdown 内嵌（见各节标注「📊 原图表格」）。
> ⚠ **DCU 适配身份**：MIGraphX 通过 HIP 生态兼容 DCU，`target("gpu")` 即 DCU，`/opt/dtk/bin/migraphx-driver` 为 CLI 入口。
> ⚠ **姊妹篇分工**：本 skill 是**性能优化方法论**；`dcu-migraphx-driver` 是 **CLI 工具源码解析/扩展**（info/op_stat/test-op 三命令）；`dcu-migraphx-custom-op` 是**自定义算子开发**（C++/HIP 核函数）。三者互补，遇对应场景先查对应 skill。
> ⚠ **示例输出非实测基线**：文中 `Rate: 975.796/sec` 等数字是**手册示例输出**（用于教如何读报告），非本环境实测基线；实测性能请用 `dcu-inference-optimization` 的方法学评估。
> ⚠ **长代码外置** `scripts/`（Python API / ORT 后端 / torch_migraphx 三段完整可运行参考，已 py_compile 验证）。本 SKILL.md 只讲方法与关键片段。

## 0. 一句话机制

MIGraphX 性能优化 = **先测量定位瓶颈（migraphx-driver perf）→ 通用策略提吞吐（批处理/序列化/精度/自动调优）→ 算子级抠细节（融合/视图/MHA/布局）→ Tensor Core 对齐吃满算力 → 多框架集成落地 → 精度调试兜底 → 环境配置调优**。核心指标是 **Rate（每秒样本数）** 与 **Total time（单次推理平均耗时）**。

---

## 1. 性能测量

### 1.1 延迟与吞吐量
- **延迟**：单个推理从输入到输出所经过的时间。低延迟是关键安全或实时应用的核心要求。
- **吞吐量**：单位时间内完成的推理数量。更高吞吐量意味着更高效地利用计算资源。

### 1.2 性能测量工具（migraphx-driver perf）
```bash
/opt/dtk/bin/migraphx-driver perf --enable-offload-copy --fp16 --onnx ./ResNet50.onnx -n 100
```

📊 **原图表格**（image1，perf 关键参数说明）：

| 参数 | 说明 |
| :--- | :--- |
| `--fp16` | 使用 FP16 精度 |
| `-n` | 迭代次数，默认 100 |
| `--batch` | 批处理大小（自动识别可手动指定） |
| `--input-dim` | 设置动态 shape 模型的输入维度 |
| `--opt-compile` | 自动调优，选择最优融合策略 |
| `--nhwc` | 使用 NHWC 布局融合模式（需配合 `--fp16`） |

> 完整 CLI 命令扩展（info/op_stat/test-op）见 `dcu-migraphx-driver`。

### 1.3 性能报告解读
性能分析输出包含两部分：

**① 指令级耗时报告**（每条指令后显示耗时 ms 和占总时间百分比）：
```
@10 = gpu::convolution[...](...) -> half_type: 1.4883ms, 4.00903%
```

**② Summary 汇总报告**：
```
Summary:
  gpu::convolution: 28.9018ms / 49 = 0.589832ms, 77.8525%
  hip::copy_to_gpu: 2.56138ms / 1 = 2.56138ms, 6.89958%
  ...
  Batch size: 32
  Rate: 975.796/sec
  Total time: 32.7937ms
```
- **Rate**：每秒处理样本数，是衡量端到端性能的核心指标。
- **Total time**：单次推理平均耗时。
- **Total instructions time vs Total time**：差异反映框架开销。

### 1.4 测量注意事项
性能测量应在**模型编译完成后**进行，避免在构建阶段启用性能分析：
```cpp
// 正确做法：先编译，后推理测量
migraphx::program net = migraphx::parse_onnx("model.onnx");
net.compile(target, options);   // 编译阶段
auto results = net.eval(inputData);  // 然后进行性能测量推理
```

### 1.5 内存使用追踪
```cpp
// 编译后获取推理所需显存大小（字节）
std::size_t memoryUsage = net.get_memory_usage();
```
推荐创建自定义 GPU 分配器追踪内存分配事件。

---

## 2. 通用性能优化策略

### 2.1 批处理（Batching）—— 最重要的优化手段
批处理可显著提高吞吐量。
```cpp
// 编译时设置批处理大小
migraphx::shape inputShape{migraphx::shape::float_type, {32, 3, 224, 224}};  // batch=32
```
**最佳实践**：
- 更大的批处理大小通常更高效。
- 对 FP16/INT8 推理，**批大小为 32 的倍数**可获得最佳 Tensor Core 性能。
- 批处理受限于 GPU 显存容量。

**性能验证（手册示例输出，非实测基线）**：
```
Batch size: 32
Rate: 975.796/sec   # 单张图片耗时约 1.025ms

Batch size: 1
Rate: 45.2/sec       # 单张图片耗时约 22.1ms（效率下降严重）
```

### 2.2 流式处理与异步执行
MIGraphX 为每个模型创建单独的非阻塞流。通过 `offload_copy` 选项控制内存管理行为：
```cpp
migraphx::compile_options options;
options.offload_copy = false;   // 使用设备端数据，避免不必要的拷贝
net.compile(migraphx::gpu::target{}, options);
```
**使用设备端数据做推理**（避免主机-设备往返拷贝）：
```cpp
// 将输入数据保持在设备端
migraphx::argument inputData = migraphx::gpu::to_gpu(
    migraphx::argument{inputShape, (float *)inputBlob.data});
modelData[inputName] = migraphx::argument{inputShape, inputData.data()};

// 推理结果也在设备端
std::vector<migraphx::argument> results = net.eval(modelData);

// 按需拷贝到主机
migraphx::argument result = migraphx::gpu::from_gpu(results[0]);
```

### 2.3 模型序列化（对应 TensorRT 引擎序列化）
编译过程耗时，应保存编译后模型避免重复编译。

**保存编译模型**：
```cpp
#include <migraphx/load_save.hpp>
net.compile(migraphx::gpu::target{}, options);
migraphx::save(net, "model.mxr");
```
**加载预编译模型**：
```cpp
migraphx::program net = migraphx::load("model.mxr");
auto results = net.eval(inputData);  // 无需再次编译，直接推理
```
**用 migraphx-driver 序列化**：
```bash
/opt/dtk/bin/migraphx-driver compile \
    --enable-offload-copy \
    --binary \
    --output ./model.mxr \
    --onnx ./model.onnx
```
**兼容性说明**：
- MXR 版本号必须与 MIGraphX 版本匹配（当前 5.2.0 对应 MXR 9）。
- GPU 架构必须一致。
- DTK 版本不同但版本号和架构一致时可兼容。

### 2.4 精度选择

**FP16 推理**
- 方式一（推荐）：直接转换
```cpp
#include <migraphx/quantization.hpp>
migraphx::quantize_fp16(net);   // 编译前调用
net.compile(...);
```
- 方式二：Python API
```python
import migraphx
migraphx.quantize_fp16(model)
model.compile(t=migraphx.get_target("gpu"))
```

**INT8 量化**

📊 **原图表格**（image2，两种 INT8 量化模式）：

| 模式 | 特点 | 适用场景 |
| :--- | :--- | :--- |
| PTQ（训练后量化） | 不需要重新训练，需提供校准数据集 | 快速部署 |
| QAT（量化感知训练） | 训练时学习量化参数，精度更高 | 精度敏感场景 |

**PTQ 使用步骤**：
```cpp
// 1. 准备校准数据（10-50 张代表性图片）
// 2. 编译时自动校准
migraphx::quantize_int8(net, calibration_data);
net.compile(...);
```
**校准数据选取建议**：
- 保证覆盖所有测试场景。
- 避免极端图片（纯黑、纯白、过度曝光）。
- 数量：10-50 张（推荐）。

### 2.5 自动调优（opt_compile）
MIGraphX 支持自动选择最优融合策略和布局：
```python
model = migraphx.opt_compile(
    model,
    t=migraphx.get_target("gpu"),
    offload_copy=False,
    precision=migraphx.target_precision.fp16,
    mode=migraphx.fusion_mode.nhwc,
)
```
```bash
migraphx-driver perf model.onnx --fp16 --opt-compile
```
**自动调优会尝试**：
- 默认布局
- NHWC 布局融合（FP16）
- NCHW32 布局融合（INT8/FP16）

---

## 3. 算子级优化

### 3.1 融合优化
MIGraphX 在编译时执行自动算子融合。

📊 **原图表格**（image3，融合类型）：

| 融合类型 | 说明 |
| :--- | :--- |
| Conv + Bias + ReLU | 融合为单算子 |
| Conv + BatchNorm | 融合到卷积权重 |
| Pointwise 算子序列 | 多个逐点操作融合 |
| 常量折叠 | 编译期计算常量表达式 |
| 公共子表达式消除 | 复用相同计算结果 |

**查看融合效果**：
```bash
# 打印计算图，检查融合后的层
migraphx-driver read --onnx model.onnx
```

**融合导致报错时的消融定位法**（MIGraphX 融合报错实战）：MIGraphX 自动融合后推理报错（特定 batch/shape 崩溃）时，不要盲目改模型——对算子融合部分做**消融实验**：逐个关闭/拆分融合规则，定位到具体导致报错的融合算子，再针对该算子处理（改输入 shape / 拆融合 / 反馈产品修复）。比"整体关融合"更精准，能保住其他融合的收益。

### 3.2 视图（View）机制
MIGraphX 支持视图操作（slice、transpose、reshape、broadcast），避免不必要的数据拷贝：
```cpp
// slice 返回视图，与原数据共享内存
auto slice = module->add_instruction(
    migraphx::make_json_op("slice", "{axes:[2,3],starts:[0,2],ends:[4,5]}"), input);
// 如需连续内存，使用 contiguous
auto contiguous = module->add_instruction(migraphx::make_op("contiguous"), slice);
```

### 3.3 MHA（Multi-Head Attention）优化
对 Transformer/BERT 模型，可用 MHA 算子融合：
```bash
# 使用 ONNX Runtime 优化器转换模型
python -m onnxruntime.transformers.optimizer \
    --input input_model.onnx \
    --output output_model.onnx \
    --use_multi_head_attention \
    --num_heads 12 \
    --hidden_size 768 \
    --model_type bert \
    --use_gpu
```
转换后 MIGraphX 会自动调用高性能 MHA 算子。

### 3.4 内存布局优化
```cpp
// C++ 编译选项
compile_options.mode = compile_options::fusion_mode::nhwc;        // FP16 推荐
compile_options.mode = compile_options::fusion_mode::nchw32_int8; // INT8 推荐
```
```python
model.compile(t=migraphx.get_target("gpu"), mode=migraphx.fusion_mode.nhwc)
```

---

## 4. Tensor Core 优化

MIGraphX 通过底层 MIOpen 库自动利用 Tensor Core。

📊 **原图表格**（image4，对齐要求）：

| 精度 | 推荐对齐 |
| :--- | :--- |
| FP16 | 8（稠密）/ 16（稀疏） |
| INT8 | 32 |

**验证 Tensor Core 使用**（用性能分析工具观察算子耗时）：
```bash
# 关注 gpu::convolution 和 gpu::gemm 的耗时
migraphx-driver perf model.onnx --fp16
```

---

## 5. Python API 最佳实践

完整 7 步可运行参考见 `scripts/python_api_best_practice.py`（加载→FP16 量化→编译→取输入信息→预处理→推理→结果处理）。关键骨架：
```python
import migraphx
import numpy as np

model = migraphx.parse_onnx("model.onnx")          # 1. 加载
migraphx.quantize_fp16(model)                       # 2. FP16 量化（可选）
model.compile(t=migraphx.get_target("gpu"),         # 3. 编译（自动调优）
              device_id=0, offload_copy=True)
inputs = model.get_inputs()                         # 4. 取输入信息
input_name = list(inputs.keys())[0]
# ... 5. 数据预处理（NCHW 格式）...
results = model.run({input_name: image_data})       # 6. 推理
result = np.array(results[0])                       # 7. 结果处理
```
**Python 性能提示**：
- 使用 pycuda 或 cupy 管理 GPU 内存。
- 页面锁定（pinned）缓冲区可提高传输效率。
- 预处理尽可能在 GPU 上完成。

---

## 6. 多框架集成优化

### 6.1 ONNX Runtime 后端
完整参考见 `scripts/ort_backend_integration.py`。通过 `MIGraphXExecutionProvider` 让 ORT 走 MIGraphX 编译路径并保存编译引擎：
```python
import onnxruntime as ort
provider_options = [{'device_id': '0',
                     'migraphx_fp16_enable': 'true',
                     'migraphx_save_compiled_model': 'true',  # 保存编译引擎
                     'migraphx_save_model_path': './model.mxr'}]
session = ort.InferenceSession("model.onnx",
                               providers=['MIGraphXExecutionProvider'],
                               provider_options=provider_options)
```

### 6.2 torch_migraphx
完整参考见 `scripts/torch_migraphx_integration.py`。把 PyTorch 模型经 `fx.lower_to_mgx` 转成 MIGraphX 优化模型后推理：
```python
import torch_migraphx, torchvision
model = torchvision.models.resnet50().eval()
sample_input = torch.randn(1, 3, 224, 224)
model_mgx = torch_migraphx.fx.lower_to_mgx(
    model, [sample_input],
    lower_precision=torch_migraphx.LowerPrecision.FP16)
output = model_mgx(sample_input)
```

---

## 7. 精度调试

### 7.1 常见精度问题

📊 **原图表格**（image5，常见精度问题与解决方案）：

| 问题 | 解决方案 |
| :--- | :--- |
| FP16 上溢/下溢 | 重新训练模型，确保中间层输出在 FP16 范围内 |
| INT8 精度下降 | 使用更具代表性的校准数据集，或尝试 QAT |
| 特定层输出错误 | 使用 `profiling_verbosity=kDETAILED` 查看层输出 |

### 7.2 调试工具
```bash
# 查看模型输入输出信息
migraphx-driver params --onnx model.onnx
# 查看计算图
migraphx-driver read --onnx model.onnx
# 验证层输出（使用 Polygraphy 或对比 ONNX Runtime）
```

### 7.3 强制指定精度
```bash
# 通过环境变量控制
export MIGRAPHX_DISABLE_TOPK_OPT=1   # 使用官方 TopK 实现
export MIGRAPHX_ENABLE_NHWC=1        # 启用 NHWC 融合
```

---

## 8. 自定义算子

当内置算子性能不足或功能缺失时，可实现自定义算子。**完整 C++/HIP 实现流程见 `dcu-migraphx-custom-op`**（本 skill 只列性能侧要点）。

### 8.1 开发流程
1. 继承 `migraphx_custom_op` 基类，实现 `name()`、`compute()`、`compute_shape()`。
2. 注册算子：`REGISTER_CUSTOM_OP(MyOp)`。
3. 编译为动态库：`libmyop.so`。
4. 加载使用：`migraphx::load_custom_op("libmyop.so")`。

### 8.2 接入方式
- 方式一：API 添加
```cpp
migraphx::add_custom_op(module, custom_op_type_preprocess, "preprocess");
```
- 方式二：修改 ONNX 图，在解析时自动识别替换。

### 8.3 性能测试
```bash
migraphx-driver perf --fp16 --onnx model.onnx --op_shared ./libmyop.so
```

---

## 9. 环境配置与调优

### 9.1 关键环境变量

📊 **原图表格**（image6，关键环境变量）：

| 变量 | 作用 |
| :--- | :--- |
| `MIGRAPHX_ENABLE_NHWC=1` | 启用 NHWC 布局融合 |
| `MIGRAPHX_ENABLE_CUTLASS=1` | 使用 CUTLASS GEMM FP16 融合 |
| `MIGRAPHX_ENABLE_MHA=1` | 启用 MHA 融合优化 |
| `MIGRAPHX_DISABLE_MIOPEN_POOLING=1` | 使用官方 Pooling 替代 MIOpen |
| `MIGRAPHX_ENABLE_GEMM_SOFTMAX_GEMM_FUSE=1` | （一般情况下不需要开） |
| `MIGRAPHX_ENABLE_LAYERNORM_FUSION=1` | layernorm 融合性能优化（一般情况下不需要开） |
| `MIGRAPHX_ENABLE_MHA_BHSD=1` | 使用 bhsd 数据排布的 mha 加速（一般情况下不需要开） |
| `MIGRAPHX_ENABLE_MIOPEN_GN_LN=1` | 使用 miopen 的 gn 和 ln 优化（group norm / layer norm） |

> 注：image6 原图底部最后一行（`MIGRAPHX_ENABLE_MIOPEN_GN_LN`）在截图中被部分截断，变量名与描述末尾依据可见字母轮廓与上下文补全，已标注。

### 9.2 系统配置
```bash
# 增加 swap 空间（显存不足时）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 9.3 Docker 运行推荐
```bash
docker run --shm-size=16g --network=host \
            --device=/dev/kfd --device=/dev/dri \
            --group-add video \
            -v /opt/hyhal:/opt/hyhal:ro \
            -it migraphx:5.2.0 /bin/bash
```
> ⚠ **DCU 容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`**，否则 DCU 不可见（DEVCOUNT=0）。容器操作规范见 `dcu-docker`。

---

## 10. 最佳实践检查清单

| 阶段 | 检查项 |
| :--- | :--- |
| **开发阶段** | 模型 ONNX 导出正确；输入 shape/精度明确；校准数据（INT8）覆盖场景 |
| **编译阶段** | 先编译后测量；FP16 用 NHWC / INT8 用 NCHW32；开启 opt_compile 自动调优；序列化 .mxr 避免重复编译 |
| **推理阶段** | 批处理取 32 倍数；设备端数据减少拷贝；offload_copy 按需；Tensor Core 对齐满足 |
| **部署阶段** | 环境变量按需开启；Docker 挂 hyhal；显存不足加 swap；精度调试兜底 |

### 常用命令速查
```bash
# 性能测试
migraphx-driver perf --fp16 --onnx model.onnx -n 100
# 查看模型信息
migraphx-driver params --onnx model.onnx
# 序列化模型
migraphx-driver compile --onnx model.onnx --binary --output model.mxr
# 查看版本
migraphx-driver --version
# 查看支持的算子
migraphx-driver onnx -l
```

---

## 关联 skill
- `dcu-migraphx-driver`：migraphx-driver CLI 源码解析与扩展（info/op_stat/test-op 三命令）。
- `dcu-migraphx-custom-op`：MIGraphX 自定义算子开发（C++/HIP 核函数，ResNet50 预处理算子全链路）。
- `dcu-inference-optimization`：DCU 通用推理优化方法论（torch.compile/NHWC/TF32/MIOpen/rocBLAS），实测性能评估用此。
- `dcu-docker`：DCU 容器操作规范（挂 hyhal、起容器）。
- `dcu-framework-profilers`：六大框架 Profiler 实战（含 MIGraphX profiler）。
