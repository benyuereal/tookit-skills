---
name: dcu-migraphx-custom-op
description: "海光DCU上用MIGraphX写自定义算子（C+++HIP核函数）"
version: 1.0.0
author: DCU Engineer
license: MIT
---

# 海光 DCU MIGraphX 自定义算子（dcu-migraphx-custom-op）

适用场景：在 **DCU 平台用 MIGraphX（AMD/海光推理优化器）** 写自定义算子——把预处理/后处理（图像缩放、裁剪、标准化、格式转换等）用 **HIP 核函数**实现并注册进 MIGraphX 计算图，随模型一起编译推理。本文是**完整实现流程地图**，以 ResNet50 预处理算子为例拆解「封装→实现→调用」全链路。

> **来源**：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》（file_id `<internal-file-id>`，<internal-author>，纯自包含已全文抓取，无内嵌子文档/无图片/无 ksheet）。
> ⚠ **DCU 适配身份**：MIGraphX 通过 **HIP 生态兼容 DCU**——自定义算子用 `hipcc` 编译（见 `scripts/Makefile`，`-I/opt/dtk/include -L/opt/dtk/lib`），核函数用 HIP（`hip_resize_kernel` 类似 CUDA），`target("gpu")` 即 DCU。与 `dcu-libtorch` 同属 C++ 推理路径但框架不同（MIGraphX 是独立推理优化器，非 LibTorch/PyTorch）。
> ⚠ **不存性能数字**：本文是开发流程，无基准；实测性能请用 dcu-inference-optimization 的方法学评估。
> ⚠ **完整工程源码外置** `scripts/`（`op_add.hpp/cpp`、`op_add_api.hpp`、`custom_op_register.hpp`、`resnet50_preprocess.hpp/cpp`、`main.cpp`、`Makefile`），做全角归一化、与文档 §六 一一对应。本 SKILL.md 只讲流程和关键片段，避免膨胀。

## 0. 一句话机制

MIGraphX 自定义算子 = 在推理优化器计算图里插入一段 **HIP 核函数**逻辑（预处理/后处理/中间层），需先封装「添加+注册」通用 API，再实现算子结构体（继承框架基类 + 实现纯虚函数），最后按固定 4 步编译推理。

## 1. 自定义算子核心使用流程（4 固定步骤，顺序不可颠倒）

1. **加载并解析 ONNX 模型**：`parse_onnx` → `program`（计算图）+ `module`（主模块）
2. **添加自定义算子**：按类型（预处理/后处理等）插入计算图对应位置
3. **模型编译**：`net.compile(target("gpu"), options)`，融合自定义算子与原生算子
4. **模型推理**：`net.eval(input_param)`，解析结果

> 实现具体算子前，先完成「通用 API 封装」（add + register 接口）作为复用基础。

## 2. 3 层封装（可复用 API）

| 层 | 文件 | 作用 |
|---|---|---|
| 算子类型枚举 + 添加接口 | `op_add.hpp` / `op_add.cpp` | `custom_op_type`（preprocess/postprocess/process/update_shape）+ `customOperator` 类 + `migraphx_experimental_custom_op_add` |
| 高层调用 API | `op_add_api.hpp` | `add_op`（带/不带 shape 重载）+ `add_experimental_custom_op` 模板（完美转发） |
| 注册接口 | `custom_op_register.hpp` | `opRegisterer<T>` 模板（4 个约束：继承基类/非基类/可拷贝/final）+ `REGISTER_CUSTOM_OP` 宏（利用全局变量初始化自动注册） |

**算子类型枚举**（顺序不可随意修改，否则触发"未知算子类型"）：
- `custom_op_type_preprocess = 0`：预处理（插入输入层 `@param` 之后，自动把输入从 float 转 uint8）
- `custom_op_type_postprocess`：后处理（插入输出层 `@return` 之前）
- `custom_op_type_process`：中间层算子
- `custom_op_type_update_shape`：虚拟算子（不改计算，只改输入 shape，适配不同分辨率）

**注册宏用法**：
```cpp
// 在算子结构体定义之后一行即可完成注册
REGISTER_CUSTOM_OP(resnet_preprocess_custom_op);
```
> 约束：模板类 `opRegisterer<T>` 要求 T 继承 `migraphx::experimental_custom_op_base`、不是基类本身、可拷贝构造、是 final 类；宏调用须放在结构体定义之后。

## 3. 1 个核心：算子结构体

自定义算子必须**继承 `migraphx::experimental_custom_op_base`** 并实现 4 个纯虚函数（`scripts/resnet50_preprocess.hpp`）：
```cpp
struct resnet_preprocess_custom_op final : migraphx::experimental_custom_op_base{
    virtual std::string name() const override { return "resnet_preprocess"; }       // 必须唯一
    virtual bool runs_on_offload_target() const override { return true; }            // true=运行在GPU(offload)
    virtual migraphx::argument compute(context, shape, arguments) const override;     // 核心计算逻辑
    virtual migraphx::shape compute_shape(shapes) const override;                     // 推导输出形状
};
```
**注意事项**：
- `name()` 必须唯一，重名会导致注册失败
- `compute_shape()` 必须正确推导输出形状，否则编译失败（shape 不匹配）
- `runs_on_offload_target()` 返回值须与实际运行目标一致（GPU 逻辑如 hip 核函数必须返回 `true`）

## 4. HIP 核函数（GPU 计算核心）

预处理用 **HIP 核函数**实现并行像素处理（`scripts/resnet50_preprocess.cpp`）：
- **`hip_resize_kernel`**：按比例缩放（双线性插值）+ 标准化（ResNet50 均值 123.675/116.28/103.53，标准差 58.395/57.12/57.375）+ 格式转换（NHWC→NCHW）
- **`hip_crop_center_kernel`**：中心裁剪到模型输入尺寸（如 224×224）
- **`resnet_resize`**：串联「缩放→裁剪」，计算缩放比例（最小边→256）、`hipLaunchKernelGGL` 启动核函数

**HIP 注意事项**：
- 线程块/网格尺寸合理（示例 16×16 block）；`hipMalloc`/`hipFree` 防泄漏；`hipStream_t` 流同步防数据竞争
- 严格区分 `uint8`（输入图像）与 `float`（输出特征图）类型

## 5. 4 步调用（完整 C++ 实例，见 `scripts/main.cpp`）

```cpp
// 1. 解析 ONNX
migraphx::program net = migraphx::parse_onnx("resnet50.onnx", onnx_options);
migraphx::module mm = net.get_main_module();
// 2. 添加预处理算子（名称须与 name() 一致，类型 preprocess）
migraphx::add_experimental_custom_op(mm, custom_op_type_preprocess, "resnet_preprocess");
// 3. 编译（GPU + 启用 offload copy）
migraphx::compile_options options; options.set_offload_copy();
net.compile(migraphx::target("gpu"), options);
// 4. 读输入（OpenCV）→ 设 shape(NCHW, uint8) → 可选 update_shape → 推理
migraphx::shape s{migraphx_shape_uint8_type, {1,3,size_t(img.rows),size_t(img.cols)}};
input_param.add("input", migraphx::argument(s, img.data));   // 参数名须与 ONNX 输入名一致
migraphx::add_experimental_custom_op(mm, custom_op_type_update_shape, "update_shape", s);  // 可选
auto results = net.eval(input_param);   // 取 results[0]，sigmoid 解析 1000 类 logits
```
> 关键：`set_offload_copy()` 启用 GPU 场景数据拷贝优化；`target("gpu")` 指定 DCU；参数名须与 ONNX 模型输入名一致。

## 6. 流程总结：3 层封装 + 1 个核心 + 4 步调用

1. **3 层封装**：通用添加 API（op_add 系列）+ 注册 API（custom_op_register.hpp）+ 高层调用 API（op_add_api.hpp）→ 代码复用
2. **1 个核心**：算子结构体（继承 `experimental_custom_op_base`，实现 `name()`/`runs_on_offload_target()`/`compute()`/`compute_shape()`）
3. **4 步调用**：解析 ONNX → 添加算子 → 编译模型 → 执行推理

## 7. 编译与运行

见 `scripts/Makefile`：
```makefile
CC = hipcc                     # 海光 DCU 用 hipcc（来自 DTK），不是 g++/dcc
INCLUDES = -I/opt/dtk/include -I/usr/include/opencv4
LDFLAGS  = -L/opt/dtk/lib
LIBS = -Wl,--start-group -lmigraphx_onnx -lmigraphx_gpu -lmigraphx_ref -lmigraphx -lmigraphx_device -lmigraphx_c -Wl,--end-group -lpthread -lstdc++ -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
SRCS = main.cpp op_add.cpp resnet50_preprocess.cpp
```
> `make` 编译，`make clean` 清理。准备好 `resnet50.onnx` 与 jpg 测试文件即可运行。

## 8. 与兄弟 skill 关系
- `dcu-kernel-dev`：HIP/TileLang 算子开发规范（本文 HIP 核函数即其应用；更通用的算子开发范式见那里）
- `dcu-libtorch`：另一条 C++ 推理路径（LibTorch，用 `dcc` 编译 + `torch::kCUDA`），与本文 MIGraphX + `hipcc` + `target("gpu")` 路线不同但同属 DCU C++ 推理生态
- `dcu-inference-optimization`：推理优化方法论（部署完成后做性能评估）
- `dcu-pytorch-source`：PyTorch 后端适配（MIGraphX 可视为 PyTorch 之外的独立推理优化器）

## 使用须知
- 完整工程源码在 `scripts/`（7 文件 + Makefile），做全角归一化，与文档 §六 对应；直接 `cd scripts && make` 即可（需 DTK + OpenCV 环境）
- 不存性能基线/逐行日志；外链无
- 高危操作谨慎，优先测试机验证
