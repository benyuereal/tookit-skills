---
name: dcu-libtorch
description: "DCU上C++LibTorch推理部署：libtorch_hip+dcc编译与TorchScript"
version: 1.0.0
author: DCU Engineer
license: MIT
---

# 海光 DCU LibTorch C++ 推理部署（dcu-libtorch）

适用场景：在 **DCU 平台用 C++ LibTorch（PyTorch 的 C++ 接口）** 做模型推理部署——加载 Python 训练的 TorchScript 模型（.pt），用 DCU 编译运行，输出独立二进制，适合延迟敏感/嵌入式/工业级场景。本文是**DCU 适配地图**，回答"怎么装、怎么编、怎么跑"。

> **来源**：内部文档《DCU LibTorch推理文档》（file_id `<internal-file-id>`，纯自包含已全文抓取，无内嵌子文档/无图片/无 ksheet）。
> ⚠ **不存性能数字**：本文是部署方法，无基准/逐行日志；实测性能请用 dcu-inference-optimization 的方法学评估。
> ⚠ **DCU 适配要点**：LibTorch 在海光是 **libtorch_hip**（不是 NVIDIA 的 libtorch_cuda）；设备语义复用 `torch::kCUDA`（DCU 兼容 CUDA 设备枚举）；编译用 **DCU 的 `dcc` 编译器**（`/opt/dtk/dcc/bin/dcc`），不是 g++/hipcc 直接编。

## 0. 一句话机制

LibTorch = PyTorch 的 C++ 前端（直接暴露 ATen/autograd 核心）。DCU 上用 **libtorch_hip** 库 + **dcc** 编译 + `torch::kCUDA` 设备，即可把 Python 训好的 TorchScript 模型在 C++ 里高效推理。

## 1. LibTorch vs PyTorch（区别对照）

| 对比项 | PyTorch (Python) | libtorch (C++) |
|---|---|---|
| 语言 | Python（动态语言） | C++（静态编译语言） |
| 开发效率 | 高（代码简洁、动态调试） | 低（需手动管理内存、编译耗时） |
| 执行效率 | 一般（Python 解释器开销） | 高（原生 C++ 性能） |
| 主要用途 | 模型训练、快速原型开发 | 模型部署、高性能推理 |
| 依赖管理 | 简单（pip/conda） | 复杂（需编译 libtorch 库） |
| 动态特性 | 支持动态计算图（eager mode） | 主要支持静态图（TorchScript） |
| 调试难度 | 低（Python 调试工具丰富） | 高（需熟悉 C++ 调试工具） |
| 部署灵活性 | 依赖 Python 环境，部署较复杂 | 独立二进制，部署简单 |

**模型互操作**：通过 TorchScript 无缝衔接——
```python
# Python：导出模型为 TorchScript
import torch
model = torch.load("model.pth")
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")   # 保存为 TorchScript 格式
```
```cpp
// C++：加载并推理
#include <torch/torch.h>
#include <iostream>
int main() {
    torch::jit::script::Module module = torch::jit::load("model.pt");
    // 输入处理和推理代码...
}
```

## 2. 在 DCU 上使用 LibTorch

### 2.1 安装
- 通过**光合开发者社区**下载 Pytorch 库：https://download.sourcefind.cn:65024/4/main/pytorch/DAS1.8
- 或使用相关**开发镜像**：https://developer.sourcefind.cn/servicelist/detail?post_id=9f296762-b3c7-11f0-9a0f-acde48001122&active=TagDownload

### 2.2 Makefile 头文件/库路径与依赖库
```makefile
include_paths := /usr/local/lib/python3.10/site-packages/torch/include
library_paths := /usr/local/lib/python3.10/site-packages/torch/lib \
                 /usr/local/lib/python3.10/site-packages/torch/lib/./../../torch.libs/

# 依赖库（DCU 用 hip 版）：
#   libtorch_hip.so  libc10_hip.so  libtorch_cpu.so  libc10.so
```

### 2.3 模型加载与推理（C++ 片段）
```cpp
#include <torch/script.h>
#include <iostream>
#include <vector>

int main()
{
    // 加载模型
    std::string model_path = "../../org_models/encoder_jiami.pt";
    torch::jit::script::Module model;
    model = torch::jit::load(model_path);

    torch::Device device(torch::kCUDA);   // DCU 复用 CUDA 设备枚举
    model.to(device);
    std::cout << "Model loaded successfully\n";

    // 创建 / 读取输入张量（示例：从 .bin 读入 1x3x640x640 float）
    torch::Tensor read_tensor = torch::empty({1, 3, 640, 640}, torch::dtype(torch::kFloat32));
    size_t InputSize = read_tensor.numel();
    std::ifstream in("encoder_input_tensor.bin", std::ios::binary);
    if (in.is_open()) {
        in.read(reinterpret_cast<char*>(read_tensor.data_ptr<float>()), InputSize * sizeof(float));
        in.close();
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(read_tensor.to(device));
    // ... 推理：model.forward(inputs) ...
}
```

### 2.4 关键编译命令（DCU 用 dcc）
```bash
/opt/dtk/dcc/bin/dcc infer_test.cpp -o infer \
  -I/usr/local/lib/python3.10/dist-packages/torch/include/ \
  -L/usr/local/lib/python3.10/dist-packages/torch/lib \
  -L/usr/local/lib/python3.10/dist-packages/torch/lib/./../../torch.libs/ \
  -ltorch_hip -lc10_hip -ltorch_cpu -lc10 -lc -lstdc++ -lm -ldl -lrt \
  -std=c++17 -fPIC -m64 -O3 -w
```
> ⚠ 注意：用 **`/opt/dtk/dcc/bin/dcc`**（DCU 编译器），不是 g++；`-I/-L` 指向 torch 的 include/lib；`-std=c++17` 必需。

## 3. FAQ：依赖库链接顺序

编译链接时**依赖库顺序有讲究**，按以下顺序链接（否则可能报未定义符号）：
```
-ltorch_hip -lc10_hip -ltorch_cpu -lc10
```
即：**hip 版在前（libtorch_hip → libc10_hip），再 cpu 版（libtorch_cpu → libc10）**。链接顺序不对会导致符号解析失败，务必保持该次序。

## 4. 与兄弟 skill 关系
- `dcu-pytorch-source`：PyTorch 后端 / 海光适配索引（libtorch_hip 即其 C++ 侧产物）。
- `dcu-vllm-deploy`：Python vLLM 部署（另一条推理路径，本文是 C++ 独立二进制路径）。
- `dcu-inference-optimization`：推理优化方法论（本文部署完成后做性能评估用）。
- `dcu-dtk`：DTK 安装/激活（dcc 编译器来自 DTK）。

## 使用须知
- 本文 C++ 片段基于主文档示例（encoder_jiami.pt / 640x640 输入），按实际模型替换路径与张量形状。
- 不存性能基线/逐行日志；外链（光合社区/开发镜像）保留为文字 URL，不内联二进制。
- 高危操作谨慎，优先测试机验证。
