#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORT MOE 优化多线程/多 Session 性能对比测试（DCU / ROCm）

高并发验证工具：对比「默认流」与「自定义 HIP 流」方案在
多线程 + 多 Session 场景下的单次推理延时与显存占用。
核心结论（实测，详见 SKILL.md）：自定义单流后并发数提升约 4 倍、
显存占用降至 36%（64G 卡上 64 线仅占 28% ≈ 18G）。

关键：
  - custom HIP stream 通过 create_ort_session 自动为每个 ROCM Session 绑定
  - 1 个 Session 对应 1 个独立线程锁 + 1 个自定义流，测完统一销毁（防泄漏）
  - 模型固定为 encoder.onnx（路径在 create_ort_session 内，按需修改）

来源：内部文档《ort-moe优化多线程性能对比测试》（file_id=<internal-file-id>）

用法：
    python3 ort_multithread_bench.py --session_num 8 --thread_num 64 --loop_times 10
"""
import sys
import os
import ctypes
import time
import threading
import argparse
from typing import List, Dict

import numpy as np
import onnxruntime as ort
import torch


# ===================== HIP Stream 管理核心函数 =====================
def create_hip_stream(hip_lib_path: str = "/opt/dtk/lib/libhiprtc.so"):
    """创建自定义 HIP 流，返回 (stream_ptr, hip_lib 句柄)。"""
    if not os.path.exists(hip_lib_path):
        raise FileNotFoundError(f"HIP 库不存在：{hip_lib_path}，请检查路径是否正确")
    hip_lib = ctypes.CDLL(hip_lib_path)
    hip_lib.hipStreamCreate.restype = ctypes.c_int
    hip_lib.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    stream_ptr = ctypes.c_void_p()
    ret_code = hip_lib.hipStreamCreate(ctypes.byref(stream_ptr))
    if ret_code != 0:
        raise RuntimeError(f"创建 HIP 流失败，错误码：{ret_code}")
    print(f"成功创建自定义 HIP 流，指针地址：{hex(int(stream_ptr.value))}")
    return stream_ptr, hip_lib


def destroy_hip_stream(stream_ptr, hip_lib):
    """销毁自定义 HIP 流。"""
    hip_lib.hipStreamDestroy.restype = ctypes.c_int
    hip_lib.hipStreamDestroy.argtypes = [ctypes.c_void_p]
    ret_code = hip_lib.hipStreamDestroy(stream_ptr)
    if ret_code != 0:
        print(f"警告：销毁 HIP 流失败，错误码：{ret_code}")
    else:
        print(f"成功销毁 HIP 流：{hex(int(stream_ptr.value))}")


# ===================== 创建带自定义 Stream 的 ORT Session =====================
def create_ort_session(provider: str = "ROCMExecutionProvider",
                        hip_lib_path: str = "/opt/dtk/lib/libhiprtc.so"):
    """创建 ORT Session；ROCm Provider 时绑定自定义 HIP Stream。"""
    stream_info = {"stream_ptr": None, "hip_lib": None, "use_custom_stream": False}
    try:
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        provider_options = {}
        if provider == "ROCMExecutionProvider":
            try:
                stream_ptr, hip_lib = create_hip_stream(hip_lib_path)
                custom_stream_str = str(stream_ptr.value)  # ORT 1.19.2 仅支持十进制
                provider_options = {
                    "device_id": "0",
                    "user_compute_stream": custom_stream_str,
                }
                stream_info["stream_ptr"] = stream_ptr
                stream_info["hip_lib"] = hip_lib
                stream_info["use_custom_stream"] = True
                print(f"为 ROCm Session 绑定自定义 HIP Stream: {custom_stream_str}")
            except Exception as e:
                print(f"创建自定义 HIP Stream 失败，使用默认流：{e}")

        session = ort.InferenceSession(
            "encoder.onnx",
            sess_options=sess_options,
            providers=[(provider, provider_options)] if provider_options else [provider],
        )
        print(f"成功创建 ONNX Session，使用 Provider: {provider}")
        return session, stream_info
    except Exception as e:
        print(f"创建 ONNX Session 失败: {e}")
        if stream_info["use_custom_stream"]:
            destroy_hip_stream(stream_info["stream_ptr"], stream_info["hip_lib"])
        raise e


def prepare_input_data() -> Dict[str, np.ndarray]:
    """构造示例输入（ASR encoder.onnx 形状，按需调整）。"""
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


def warm_up_session(session: ort.InferenceSession, input_data: Dict[str, np.ndarray]) -> float:
    """预热 1 次推理，返回预热耗时（秒）。"""
    try:
        bt = time.time()
        session.run(None, input_data)
        elapse = time.time() - bt
        print(f"  预热推理完成，耗时: {elapse:.6f} 秒")
        return elapse
    except Exception as e:
        print(f"  预热推理失败: {e}")
        raise e


def inference_worker(session, session_lock, input_data, loop_times, thread_id,
                    result_dict, enable_lock):
    """推理工作线程：执行 loop_times 次推理并记录线程耗时。"""
    print(f"线程{thread_id}开始执行，循环次数: {loop_times}，启用锁: {enable_lock}")
    bt = time.time()
    for i in range(loop_times):
        try:
            if enable_lock:
                with session_lock:
                    ort_outs = session.run(None, input_data)
            else:
                ort_outs = session.run(None, input_data)
            if i == 0 or i == loop_times - 1:
                print(f"线程{thread_id} - 推理第{i+1}次，输出长度: {len(ort_outs)}")
        except Exception as e:
            print(f"线程{thread_id} - 推理第{i+1}次失败: {e}")
            result_dict[thread_id] = -1.0
            return
    et = time.time()
    thread_total = et - bt
    result_dict[thread_id] = thread_total
    print(f"线程{thread_id}执行完成，总耗时: {thread_total:.4f} 秒")


def run_benchmark(session_num: int, thread_num: int, loop_times: int = 10,
                  provider: str = "ROCMExecutionProvider", enable_lock: bool = False,
                  hip_lib_path: str = "/opt/dtk/lib/libhiprtc.so"):
    if session_num <= 0 or thread_num <= 0 or loop_times <= 0:
        raise ValueError("Session数、线程数、循环次数必须大于0")

    input_data = prepare_input_data()
    print(f"\n===== 基准测试配置 =====")
    print(f"Session数量: {session_num}")
    print(f"线程数量: {thread_num}")
    print(f"每个线程推理循环次数: {loop_times}")
    print(f"执行Provider: {provider}")
    print(f"是否启用Session锁: {enable_lock}")
    print(f"HIP库路径（仅ROCM）: {hip_lib_path}")
    print(f"=======================\n")

    session_list: List[ort.InferenceSession] = []
    stream_info_list: List[dict] = []
    session_lock_list: List[threading.Lock] = []
    for i in range(session_num):
        session, stream_info = create_ort_session(provider, hip_lib_path)
        session_list.append(session)
        stream_info_list.append(stream_info)
        session_lock = threading.Lock()
        session_lock_list.append(session_lock)
        print(f"已为第 {i+1} 个Session创建独立线程锁 + {'自定义HIP Stream' if stream_info['use_custom_stream'] else '默认流'}")
    print(f"\n成功创建 {len(session_list)} 个Session + {len(session_lock_list)} 个独立锁 + {sum(1 for s in stream_info_list if s['use_custom_stream'])} 个自定义HIP Stream\n")

    print(f"\n===== 开始Session预热 =====")
    warm_up_time_list = []
    for idx, session in enumerate(session_list):
        print(f"正在预热第 {idx+1}/{len(session_list)} 个Session...")
        warm_time = warm_up_session(session, input_data)
        warm_up_time_list.append(warm_time)
        print(f"第 {idx+1}/{len(session_list)} 个Session预热完成\n")
    total_warm_time = sum(warm_up_time_list)
    avg_warm_time = np.mean(warm_up_time_list)
    print(f"===== 所有Session预热完成 ======")
    print(f"预热总耗时: {total_warm_time:.6f} 秒")
    print(f"单个Session平均预热耗时: {avg_warm_time:.6f} 秒")
    print(f"各Session预热耗时详情: {[f'{t:.6f}' for t in warm_up_time_list]}")
    print(f"===============================\n")

    thread_result_dict = {}
    threads: List[threading.Thread] = []
    bt_total = time.time()
    for thread_id in range(thread_num):
        session_idx = thread_id % session_num
        target_session = session_list[session_idx]
        target_lock = session_lock_list[session_idx]
        t = threading.Thread(
            target=inference_worker,
            args=(target_session, target_lock, input_data, loop_times, thread_id, thread_result_dict, enable_lock),
        )
        threads.append(t)
        thread_result_dict[thread_id] = 0.0

    print(f"启动 {len(threads)} 个工作线程...\n")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    et_total = time.time()
    total_elapse_time = et_total - bt_total

    print(f"\n===== 时间统计结果 =====")
    print(f"测试总耗时: {total_elapse_time:.4f} 秒")
    valid_thread_times = [v for k, v in thread_result_dict.items() if v > 0]
    if len(valid_thread_times) == 0:
        print("所有线程执行失败，无有效统计数据")
    else:
        avg_thread_time = np.mean(valid_thread_times)
        min_thread_time = np.min(valid_thread_times)
        max_thread_time = np.max(valid_thread_times)
        total_inference_times = thread_num * loop_times
        avg_inference_time = sum(valid_thread_times) / total_inference_times
        print(f"有效线程数: {len(valid_thread_times)} / {thread_num}")
        print(f"线程平均耗时: {avg_thread_time:.4f} 秒")
        print(f"线程最小耗时: {min_thread_time:.4f} 秒")
        print(f"线程最大耗时: {max_thread_time:.4f} 秒")
        print(f"总推理次数: {total_inference_times}")
        print(f"平均单次推理耗时: {avg_inference_time:.6f} 秒")
    print(f"=======================\n")

    print(f"\n===== 清理资源 =====")
    for idx, stream_info in enumerate(stream_info_list):
        if stream_info["use_custom_stream"]:
            destroy_hip_stream(stream_info["stream_ptr"], stream_info["hip_lib"])
    print(f"所有自定义HIP Stream已销毁")
    print(f"=======================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX Runtime TTS Encoder 基准测试工具（含自定义HIP Stream + Session独立锁）")
    parser.add_argument("--session_num", type=int, default=1, help="ONNX Session数量（默认1）")
    parser.add_argument("--thread_num", type=int, default=1, help="工作线程数量（默认1）")
    parser.add_argument("--loop_times", type=int, default=10, help="每个线程的推理循环次数（默认10）")
    parser.add_argument("--provider", type=str, default="ROCMExecutionProvider",
                        choices=["ROCMExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"],
                        help="执行提供者（默认ROCMExecutionProvider）")
    parser.add_argument("--enable_lock", action="store_true", default=False,
                        help="是否启用Session独立锁（默认不加锁）")
    parser.add_argument("--hip_lib_path", type=str, default="/opt/dtk/lib/libhiprtc.so",
                        help="HIP动态库路径（仅ROCM生效）")
    args = parser.parse_args()
    try:
        run_benchmark(
            session_num=args.session_num,
            thread_num=args.thread_num,
            loop_times=args.loop_times,
            provider=args.provider,
            enable_lock=args.enable_lock,
            hip_lib_path=args.hip_lib_path,
        )
    except Exception as e:
        print(f"基准测试执行失败: {e}")
