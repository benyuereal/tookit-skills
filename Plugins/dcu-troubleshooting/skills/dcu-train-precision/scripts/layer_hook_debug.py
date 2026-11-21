#!/usr/bin/env python3
# DCU 训练精度排查：逐层 forward hook 调试工具
# 来源：DCU平台训练精度排查最佳实践指南 主文档 §3.1（已从 内部文档 导出原文归一化）
# 用途：注册 forward hook，逐层打印输入/权重/输出前 10 个值，
#       在 NV 与 DCU 两平台同种子跑同一份代码，对比 debug.log 定位首个异常层。
# 用法：
#   set_seed(42)                      # 必须在模型初始化之前
#   model = YourModel()
#   handles = register_hooks(model)  # 注册叶子模块钩子
#   ... 训练 ...
#   for h in handles: h.remove()     # 用完移除
import random
import argparse
import datetime
import json
import numpy as np
import time
import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_hook(module, input, output):
    """打印每层的输入、权重和输出前10个数"""
    print(f"\n{module.__class__.__name__} ({module.__class__.__name__}):")
    print("=" * 60)

    if input is not None and len(input) > 0:
        if isinstance(input[0], torch.Tensor):
            input_tensor = input[0]
            flat_input = input_tensor.flatten()
            input_values = flat_input[:10].detach().cpu().numpy()
            print(f"  Input Shape: {input_tensor.shape}")
            print(f"  First 10 input values: ", end="")
            for i, val in enumerate(input_values):
                print(f"{val:.5f}", end=" ")
            print()
        else:
            print(f"  Input type: {type(input[0])}")
    else:
        print("  No input")

    if hasattr(module, 'weight') and module.weight is not None:
        weight_tensor = module.weight
        flat_weight = weight_tensor.flatten()
        weight_values = flat_weight[:10].detach().cpu().numpy()
        print(f"  Weight Shape: {weight_tensor.shape}")
        print(f"  First 10 weight values: ", end="")
        for i, val in enumerate(weight_values):
            print(f"{val:.5f}", end=" ")
        print()

        if hasattr(module, 'bias') and module.bias is not None:
            bias_tensor = module.bias
            flat_bias = bias_tensor.flatten()
            bias_values = flat_bias[:min(10, len(flat_bias))].detach().cpu().numpy()
            print(f"  Bias Shape: {bias_tensor.shape}")
            print(f"  First {len(bias_values)} bias values: ", end="")
            for i, val in enumerate(bias_values):
                print(f"{val:.5f}", end=" ")
            print()

    if isinstance(output, torch.Tensor):
        flat_output = output.flatten()
        output_values = flat_output[:10].detach().cpu().numpy()
        print(f"  Output Shape: {output.shape}")
        print(f"  First 10 output values: ", end="")
        for i, val in enumerate(output_values):
            print(f"{val:.5f}", end=" ")
        print()

    if isinstance(module, nn.Conv2d):
        if len(input) > 0 and isinstance(input[0], torch.Tensor):
            in_shape = input[0].shape
            out_shape = output.shape
            kernel_size = module.kernel_size
            groups = module.groups
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            flops = (2 * out_shape[1] * out_shape[2] * out_shape[3] *
                     in_shape[1] // groups * kernel_size[0] * kernel_size[1])
            print(f"  FLOPs: {flops:,}")
            print(f"  Groups: {groups}")

    return output


def register_hooks(model):
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            handle = module.register_forward_hook(print_hook)
            handles.append(handle)
    return handles
