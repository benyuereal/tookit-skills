#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORT 1.19.2 + DCU/ROCm 自定义 HIP 流推理（显存异常修复核心方案）

问题背景：MOE 等含并行子图模型在 DCU 上用 ORT-ROCM-EP 默认多流调度时，
ORT 为每个并行专家分支分配独立 HIP stream，ROCm EP 又按「1 stream = 1 rocBLAS
handle（硬绑定）」创建大量 rocBLAS handle，导致显存雪崩式增长（1.7G 模型爆 50G+）。

本脚本通过手动创建「自定义单 HIP 流」并传给 ROCMExecutionProvider 的
user_compute_stream，强制全程单流串行，消除多流→多 handle→显存泄漏的根因，
同时配合 enable_profiling 导出性能日志。

关键约束（ORT 1.19.2）：
  - user_compute_stream 必须是【十进制字符串】（流指针的 .value），不支持十六进制
  - 默认单流场景下，1.19.2 会残留 1 个多余的初始化 handle（ORT 1.20+ 已修复），无害
  - libhiprtc.so 路径默认 /opt/dtk/lib/libhiprtc.so，按实际 DTK 安装路径调整

来源：内部文档《ORT 1.19.2 Profile机制解析和实践》/《ORT在DCU上推理MOE模型显存异常问题的分析和解决》
      （file_id=<internal-file-id> / <internal-file-id>）

用法：
    python3 ort_custom_stream_infer.py ./encoder.onnx    # 默认模型名 encoder.onnx
"""
import sys
import os
import ctypes
import onnxruntime as ort
import torch


def create_hip_stream(hip_lib_path: str = "/opt/dtk/lib/libhiprtc.so"):
    """创建自定义 HIP 流，返回 (stream_ptr, hip_lib 句柄)。"""
    if not os.path.exists(hip_lib_path):
        raise FileNotFoundError(f"HIP 库不存在：{hip_lib_path}，请检查 DTK/ROCm 安装路径")
    try:
        hip_lib = ctypes.CDLL(hip_lib_path)
    except Exception as e:
        raise RuntimeError(f"HIP 库加载失败，检查依赖与权限：{e}")

    hip_lib.hipStreamCreate.restype = ctypes.c_int
    hip_lib.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    stream_ptr = ctypes.c_void_p()
    ret_code = hip_lib.hipStreamCreate(ctypes.byref(stream_ptr))
    if ret_code != 0:
        raise RuntimeError(f"创建 HIP 流失败，错误码：{ret_code}")
    print(f"成功创建自定义 HIP 流，十六进制地址：{hex(stream_ptr.value)}")
    print(f"HIP 流十进制值（ORT 专用）：{stream_ptr.value}")
    return stream_ptr, hip_lib


def destroy_hip_stream(stream_ptr, hip_lib):
    """销毁自定义 HIP 流，释放 DCU 资源。"""
    hip_lib.hipStreamDestroy.restype = ctypes.c_int
    hip_lib.hipStreamDestroy.argtypes = [ctypes.c_void_p]
    ret_code = hip_lib.hipStreamDestroy(stream_ptr)
    if ret_code != 0:
        print(f"警告：销毁 HIP 流失败，错误码：{ret_code}")
    else:
        print(f"成功销毁 HIP 流：{hex(stream_ptr.value)}")


def build_inputs():
    """构造示例输入（ASR encoder.onnx 形状，按实际模型调整）。"""
    feats_in = torch.zeros(1, 67, 80, dtype=torch.float32)
    offset = torch.zeros(1, dtype=torch.long)
    required_cache_size = torch.tensor(64, dtype=torch.long)
    att_cache = torch.zeros(12, 8, 64, 128, dtype=torch.float32)
    cnn_cache = torch.zeros(12, 1, 512, 14, dtype=torch.float32)
    cache_mask = torch.ones(1, 1, 80, dtype=torch.bool)
    return {
        "chunk": feats_in.numpy(),
        "offset": offset.numpy(),
        "required_cache_size": required_cache_size.numpy(),
        "att_cache": att_cache.numpy(),
        "cnn_cache": cnn_cache.numpy(),
        "att_mask": cache_mask.numpy(),
    }


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "encoder.onnx"

    # 1. 创建自定义 HIP 流
    try:
        stream_ptr, hip_lib = create_hip_stream("/opt/dtk/lib/libhiprtc.so")
        custom_stream_str = str(stream_ptr.value)  # ORT 1.19.2 仅支持十进制
    except Exception as e:
        print(f"创建 HIP 流失败：{e}")
        sys.exit(1)

    # 2. 配置 ORT SessionOptions（单流单线程，避免冲突）
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True  # 导出 onnxruntime_profile_*.json
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    # 3. ROCm-EP 配置（绑定自定义流）
    rocm_provider_options = {
        "device_id": "0",
        "user_compute_stream": custom_stream_str,
    }

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[("ROCMExecutionProvider", rocm_provider_options), "CPUExecutionProvider"],
        )
        print(f"成功加载 {model_path}")
    except Exception as e:
        print(f"模型加载失败：{e}")
        destroy_hip_stream(stream_ptr, hip_lib)
        sys.exit(1)

    ort_inputs = build_inputs()
    try:
        ort_outs = session.run(None, ort_inputs)
        print(f"推理完成，输出个数：{len(ort_outs)}")
        for i, shape in enumerate([out.shape for out in ort_outs]):
            print(f"  输出{i} 形状：{shape}")
    except Exception as e:
        print(f"推理失败：{e}")
    finally:
        profile_path = session.end_profiling()
        print(f"Profile 日志已生成：{profile_path}")
        destroy_hip_stream(stream_ptr, hip_lib)


if __name__ == "__main__":
    main()
