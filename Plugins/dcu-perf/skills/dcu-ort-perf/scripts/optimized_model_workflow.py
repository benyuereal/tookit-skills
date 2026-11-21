# 高优化等级：初次生成优化模型 + 后续复用（DCU 平台）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§3.4
# 说明：更高效的模式是把「模型优化」和「实际推理」分开——
#       离线用 ORT_ENABLE_ALL 生成优化模型，线上用 ORT_DISABLE_ALL 直接加载。
import onnxruntime as ort

# ========== 第一步：生成优化模型（离线优化脚本） ==========
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# 指定保存路径，生成优化后的模型文件
sess_options.optimized_model_filepath = "optimized_model.onnx"

# 创建会话的过程会触发优化和保存
session = ort.InferenceSession("model.onnx", sess_options)
print("优化后的模型已保存为 optimized_model.onnx")

# ========== 第二步：加载优化模型（实际推理服务） ==========
sess_options2 = ort.SessionOptions()
# 既然模型已经优化好了，就不要再优化了
sess_options2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# 直接加载优化后的模型文件
session2 = ort.InferenceSession("optimized_model.onnx", sess_options2)
print("已直接加载优化模型，启动速度更快。")

# 优化等级说明：
# - ORT_ENABLE_BASIC   ：仅模式无关优化 + 内存布局优化
# - ORT_ENABLE_EXTENDED：基础优化 + 扩展模式（如算子融合）
# - ORT_ENABLE_ALL     ：所有优化全开（布局优化/内存复用等），生产环境推荐
# ORT 内置图优化器自动执行：算子融合（Conv+BN+Relu→FusedConv）、常量折叠、冗余节点消除。
