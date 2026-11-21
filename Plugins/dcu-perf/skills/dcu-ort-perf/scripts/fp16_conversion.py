# FP16 模型转换（DCU 平台）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§3.2.1
# 说明：用 onnxconverter-common 的 convert_float_to_float16 做 FP32→FP16 转换。
#       keep_io_types=True 保持输入输出为 FP32，避免外部接口类型冲突。
# 收益：支持 FP16 加速的 GPU 上通常可获得 1.5~2 倍加速，对多数视觉模型精度影响微小。
import onnx
from onnxconverter_common import float16

model_fp32 = onnx.load("model_fp32.onnx")
# keep_io_types=True 可保持输入输出为 FP32，避免外部接口类型冲突
model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
onnx.save(model_fp16, "model_fp16.onnx")

# 不推荐转换为 FP16 的层（FP16 数值范围约 ±6.55e4，精度约 3 位小数）：
# 1. 归一化层（BatchNorm/LayerNorm/InstanceNormalization）：均值方差计算对精度极敏感
# 2. Softmax/Sigmoid 等激活函数：指数运算易溢出/下溢
# 3. 累加与求和（ReduceSum/Accumulation）：舍入误差在累加中不断积累
# 4. 转置卷积层：某些硬件上转 FP16 后性能大幅下降
# 5. 含大量 Conv/Gemm 且对误差敏感的模型（如换脸模型 Inswapper）
#
# 转换注意事项：
# - ONNX 转换端可用 onnxconverter-common 的 auto_convert_mixed_precision 功能，
#   通过迭代分治算法自动验证并保留必须为 FP32 的节点。
# - 混合精度：关键层（输出层/注意力层）保留 FP32，其余层 FP16。
# - INT8 量化：性能极致场景，动态/静态量化权重，适用 CPU 及支持 INT8 指令的 GPU。
