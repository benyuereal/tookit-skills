# 注册自定义算子并执行推理（DCU 平台）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§5.3
# 说明：推理时需将自定义算子注册到 ORT 会话，运行时才能识别并调用。
#       Python 算子直接注册；C++ 算子用 register_custom_ops_library 加载动态库。
import onnxruntime as ort
import numpy as np

# 方法一：直接注册（适用于 Python 实现的算子）
# 确保自定义算子已通过 @onnx_op 装饰器定义（见 custom_op_python.py 的 my_custom_op）
# from onnxruntime_extensions import PyCustomOpDef, get_library_path

# 创建会话选项
sess_options = ort.SessionOptions()

# 方法二：加载 C++ 动态库（适用于 C++ 实现的算子）
# sess_options.register_custom_ops_library("./libmy_custom_op.so")

# ========== CPU 推理 ==========
session = ort.InferenceSession(
    "model_with_custom_op.onnx",
    sess_options,
    providers=['CPUExecutionProvider']  # 可根据需要选择 ROCm EP
)

# 准备测试数据
input_a = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_b = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 执行推理
outputs = session.run(["output"], {"input_a": input_a, "input_b": input_b})
print(f"推理输出 shape: {outputs[0].shape}")
print(f"输出样例: {outputs[0][0, 0, :5, :5]}")

# ========== ROCm 环境下的推理（DCU） ==========
providers = [
    ('ROCMExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
    }),
    ('CPUExecutionProvider', {})
]

session_rocm = ort.InferenceSession(
    "model_with_custom_op.onnx",
    sess_options,
    providers=providers
)

# 将数据拷贝到 GPU 内存（可选，通过 io_binding 优化，见 io_binding_zero_copy.py）
# ... 推理代码同上述 Python 示例
