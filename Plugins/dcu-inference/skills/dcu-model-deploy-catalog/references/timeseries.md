# 时序预测 部署配方（5 个模型）

> 来源：《<某能源集团客户>2026年网络设备项目测试项汇总》「模型适配支持度」sheet 部署文档交付链接列（标黄=BW1100 已验证）。

> 命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑；路径/卡号/容器名按实际调整。

### Chronos
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it test bash

pip install chronos-forecasting
```

**其他命令**：
```bash
访问社区下载：https://www.modelscope.cn/models/amazon/chronos-2

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

import torch
import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline

# ⚠️ 本地模型路径配置
LOCAL_MODEL_PATH = "/public/chronos-2"


def test_chronos_gpu():
    print("=" * 60)
    print("🔍 Chronos-2 GPU Inference Verification")
    print("=" * 60)

    # 1. 检查 GPU 环境
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available. Please check your PyTorch/CUDA installation.")

    device_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA Available: {device_name}")
    print(f"✅ PyTorch Version: {torch.__version__}")

    # 2. 从本地路径加载模型到 GPU
    print(f"\n⏳ Loading Chronos-2 model from local path: {LOCAL_MODEL_PATH} ...")
    pipeline = Chronos2Pipeline.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="cuda"
    )
    print("✅ Model loaded successfully!")

    # 3. 构造本地模拟时间序列数据 (避免网络下载问题)
    print("\n⏳ Generating mock time series data...")
    timestamps = pd.date_range(start="2024-01-01", periods=200, freq="h")
    values = np.sin(np.arange(200) * 0.1) + np.random.normal(0, 0.1, 200)

    context_df = pd.DataFrame({
        "id": ["series_0"] * 200,
        "timestamp": timestamps,
        "target": values
    })
    print(f"✅ Mock data generated. Context shape: {context_df.shape}")

    # 4. 执行 GPU 推理
    prediction_length = 24
    quantile_levels = [0.1, 0.5, 0.9]
    print(f"\n⏳ Running inference (pred_len={prediction_length}, quantiles={quantile_levels})...")

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # 5. 验证输出结果
    print("\n📊 Prediction Result Preview:")
    print(pred_df.head())

    expected_rows = prediction_length  # 单条序列
    assert len(pred_df) == expected_rows, f"❌ Expected {expected_rows} rows, got {len(pred_df)}"
    assert "predictions" in pred_df.columns, "❌ Missing 'predictions' column"
    for q in quantile_levels:
        assert str(q) in pred_df.columns, f"❌ Missing quantile column '{q}'"

    print("\n" + "=" * 60)
    print("🎉 Chronos-2 GPU Inference Verification PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_chronos_gpu()
```

### Timer
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**依赖/环境**：
```bash
docker exec -it test bash
pip install transformers==4.40.2
```

**其他命令**：
```bash
访问hugging face下载：https://github.com/thuml/Timer-XL

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

import os
import time
import torch

# 【关键】设置 HuggingFace 国内镜像站，解决下载问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModelForCausalLM

# 1. 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"1. 使用设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

# 2. 从 HF 镜像下载并加载模型 (首次运行会自动下载)
print("2. 正在从 HuggingFace 镜像下载并加载 Timer 模型...")
model = AutoModelForCausalLM.from_pretrained(
    'thuml/timer-base-84m',
    trust_remote_code=True
).to(device)
model.eval()
print("   ✅ 模型加载成功！")

# 3. 准备测试数据 (batch_size=1, 历史长度=512)
x = torch.randn(1, 512).to(device)

# 4. 执行 GPU 推理
print("3. 开始 GPU 推理 (预测未来 96 步)...")
if torch.cuda.is_available(): torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    pred = model.generate(x, max_new_tokens=96)

if torch.cuda.is_available(): torch.cuda.synchronize()
cost = (time.time() - start) * 1000

# 5. 打印验证结果
print("\n" + "="*40)
print("✅ Timer GPU 验证通过！")
print("="*40)
print(f"输入形状: {tuple(x.shape)}")
print(f"输出形状: {tuple(pred.shape)}")
print(f"推理耗时: {cost:.2f} ms")
if torch.cuda.is_available():
    print(f"显存占用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

### Timer-XL
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -itd --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10 bash
```

**其他命令**：
```bash
访问hugging face下载：https://github.com/thuml/Timer-XL

docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.10

docker exec -it test bash
```

### UNITS
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name test --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it test bash

git clone https://github.com/mims-harvard/UniTS.git
cd UniTS/
pip install -r requirements.txt --no-deps
pip install "sympy>=1.13,<1.14"
```

**其他命令**：
```bash
访问社区下载：https://github.com/mims-harvard/UniTS

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

import torch
import argparse
from models.UniTS import Model as UniTS


def get_mock_args():
    """构造满足 UniTS 初始化的最小化 mock 参数"""
    parser = argparse.ArgumentParser()

    # === UniTS 核心特有参数 ===
    parser.add_argument('--prompt_num', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=256)
    # ⚠️ patch_len 必须等于 stride（UniTS 强制非重叠 patch）
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)

    # === 通用基础参数 ===
    parser.add_argument('--dropout', type=float, default=0.1)
    # ⚠️ UniTS 源码使用 n_heads 而非 num_heads
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--channel_independence', type=int, default=0)
    parser.add_argument('--use_norm', type=int, default=1)

    # === 数据维度参数 ===
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--num_class', type=int, default=10)

    args, _ = parser.parse_known_args()
    return args


def main():
    print("=" * 60)
    print("🔍 UniTS GPU & Forward Pass Verification")
    print("=" * 60)

    # 1. 检查 GPU 环境
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available.")

    device = torch.device("cuda")
    print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ PyTorch Version: {torch.__version__}")

    # 2. 准备正确的 configs_list（元组格式）
    args = get_mock_args()

    configs_list = [
        (
            "custom_forecast_task",
            {
                'dataset': 'custom',
                'task_name': 'long_term_forecast',
                'enc_in': args.enc_in,
                'c_out': args.c_out,
                'seq_len': args.seq_len,
                'label_len': args.label_len,
                'pred_len': args.pred_len,
            }
        )
    ]

    # 3. 初始化模型
    print("\n⏳ Initializing UniTS Model...")
    import traceback
    try:
        model = UniTS(args=args, configs_list=configs_list, pretrain=False).to(device)
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Model initialization failed:")
        traceback.print_exc()
        return

    # 4. 前向传播测试
    B, L, D = args.batch_size, args.seq_len, args.enc_in
    dummy_x = torch.randn(B, L, D).to(device)
    dummy_mark = torch.zeros(B, L, 4).to(device)

    print(f"\n⏳ Running forward pass with input shape: {dummy_x.shape} ...")
    model.eval()

    try:
        with torch.no_grad():
            # ⚠️ 必须通过关键字参数传入正确的 task_name 进行路由
            output = model(
                x_enc=dummy_x,
                x_mark_enc=dummy_mark,
                task_name='long_term_forecast',  # 有效路由键
                task_id=0                        # 对应 configs_list 中的索引
            )

        if output is None:
            raise ValueError("Model returned None - task_name routing failed!")

        out_shape = output.shape if not isinstance(output, tuple) else output[0].shape
        print(f"✅ Forward pass SUCCESS! Output shape: {out_shape}")

    except Exception as e:
        print(f"❌ Forward pass failed:")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 Verification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### time-moe
**镜像**：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`  
**框架**：PyTorch  
**原文**：[内部文档](<internal-doc-link>)

**创建容器**：
```bash
docker run -it --shm-size 200g --network=host --name timemoe --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root -v /path/your_code_data/:/path/your_code_data/ -v /opt/hyhal/:/opt/hyhal/:ro harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413 bash
```

**依赖/环境**：
```bash
docker exec -it timemoe bash

pip install transformers==4.40.1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com
python run-time-moe.py --trust-remote-code
```

**其他命令**：
```bash
访问社区下载：https://huggingface.co/Maple728/TimeMoE-50M

dcoker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413

import torch
from transformers import AutoModelForCausalLM
import time


def verify_time_moe_gpu():
    print("=" * 60)
    print("🔍 Time-MoE GPU 运行验证脚本")
    print("=" * 60)

    # 1. 检查 GPU 环境
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到可用 GPU！请检查 CUDA 驱动和 PyTorch GPU 版本。")

    gpu_name = torch.cuda.get_device_name(0)
    # 兼容不同 PyTorch 版本的显存属性名称
    props = torch.cuda.get_device_properties(0)
    gpu_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    gpu_mem_gb = gpu_mem / 1e9

    print(f"✅ GPU 检测通过: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    # 2. 配置参数
    MODEL_ID = "Maple728/TimeMoE-50M"  # 可替换为 Base/Large
    CONTEXT_LENGTH = 128
    PREDICTION_LENGTH = 24
    DEVICE = "cuda"

    # 3. 加载模型到 GPU
    print(f"\n📦 正在加载模型 [{MODEL_ID}] 到 GPU...")
    try:
        # 尝试使用 flash_attention_2
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print("✅ 模型加载成功 (Flash Attention 2 + FP16)")
    except Exception as e:
        print(f"⚠️ Flash Attn 不可用，回退到默认注意力机制: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print("✅ 模型加载成功 (Default Attention + FP16)")

    # 4. 验证模型设备
    model_device = next(model.parameters()).device
    assert str(model_device).startswith("cuda"), f"❌ 模型未在GPU上! 当前设备: {model_device}"
    print(f"✅ 模型已确认在 GPU 上: {model_device}")

    # 5. 执行推理测试
    print(f"\n🧪 执行推理测试 (Context={CONTEXT_LENGTH}, Horizon={PREDICTION_LENGTH})...")
    seqs = torch.randn(4, CONTEXT_LENGTH, device=DEVICE, dtype=torch.float16)

    # 归一化（Time-MoE 要求输入必须归一化）
    mean = seqs.mean(dim=-1, keepdim=True)
    std = seqs.std(dim=-1, keepdim=True)
    normed_seqs = (seqs - mean) / std

    # 计时推理
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()

        output = model.generate(normed_seqs, max_new_tokens=PREDICTION_LENGTH)

        torch.cuda.synchronize()
        elapsed = time.time() - start

    # 反归一化得到最终预测值
    predictions = output[:, -PREDICTION_LENGTH:] * std + mean

    print(f"✅ 推理完成!")
    print(f"   输出形状: {predictions.shape}")
    print(f"   推理耗时: {elapsed:.4f}s")
    print(f"   吞吐量: {4 * PREDICTION_LENGTH / elapsed:.1f} tokens/s")
    print(f"   GPU 显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("🎉 Time-MoE GPU 部署验证全部通过！")
    print("=" * 60)


if __name__ == "__main__":
    verify_time_moe_gpu()
```
