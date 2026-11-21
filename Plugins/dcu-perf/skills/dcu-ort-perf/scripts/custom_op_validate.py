# 自定义算子：单元测试与验证
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§5.3.2
# 说明：验证自定义算子的正确性（对比原生实现）和性能（基准测试）。
import time
import numpy as np
import onnxruntime as ort

# 复用 custom_op_python.py 中定义的 my_custom_op
# from custom_op_python import my_custom_op
# 此处内联一份等价实现，保证本脚本可独立运行
def my_custom_op(x, y, **kwargs):
    """自定义算子实现：逐元素相乘后加偏置（与 custom_op_python.py 等价）"""
    bias = kwargs.get("bias", 1.0)
    return x * y + bias


def validate_custom_op():
    """验证自定义算子的正确性和性能"""

    # 1. 正确性验证：对比自定义算子与原生实现
    test_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    test_y = np.array([[5.0, 6.0], [7.0, 8.0]])

    custom_result = my_custom_op(test_x, test_y, bias=2.0)
    expected_result = test_x * test_y + 2.0

    print(f"自定义算子结果:\n{custom_result}")
    print(f"预期结果:\n{expected_result}")
    print(f"差异: {np.max(np.abs(custom_result - expected_result))}")

    # 2. 性能基准测试
    session = ort.InferenceSession("model_with_custom_op.onnx")
    input_a = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_b = np.random.randn(1, 3, 224, 224).astype(np.float32)
    warmup = 10
    iterations = 100

    for _ in range(warmup):
        session.run(["output"], {"input_a": input_a, "input_b": input_b})

    start = time.time()
    for _ in range(iterations):
        session.run(["output"], {"input_a": input_a, "input_b": input_b})
    elapsed = (time.time() - start) / iterations * 1000

    print(f"平均推理延迟: {elapsed:.2f} ms")


if __name__ == "__main__":
    validate_custom_op()
