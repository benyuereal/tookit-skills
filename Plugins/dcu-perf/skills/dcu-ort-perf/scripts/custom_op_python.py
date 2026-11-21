# 自定义算子：Python 实现（快速原型）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§5.1
# 说明：用 @onnx_op 装饰器将 Python 函数封装为 ORT 可识别的自定义算子。
#       示例：逐元素乘加算子 MyCustomOp，计算 output = x * y + bias。
# 适用：Python 快速原型验证；生产环境高性能部署用 C++（见 custom_op_cpp.cpp）。
import numpy as np
from onnxruntime_extensions import onnx_op, PyCustomOpDef


# 定义自定义算子
@onnx_op(
    op_type="MyCustomOp",           # 算子类型名称
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],  # 两个 float 输入
    outputs=[PyCustomOpDef.dt_float],                          # 一个 float 输出
    attrs={"bias": PyCustomOpDef.dt_float}                     # 可选属性，默认偏置值
)
def my_custom_op(x, y, **kwargs):
    """
    自定义算子实现：逐元素相乘后加偏置
    """
    bias = kwargs.get("bias", 1.0)   # 从属性获取偏置值，默认为 1.0
    return x * y + bias


# 验证算子功能
if __name__ == "__main__":
    result = my_custom_op(np.array([2.0, 3.0]), np.array([4.0, 5.0]), bias=2.0)
    print(f"计算结果: {result}")  # 输出: [10.0, 17.0]
