# 多线程加速配置（DCU 平台）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§3.5
# 说明：当模型无法所有节点都在 DCU 运行时，部分算子跑在 CPU，
#       合理配置线程数可提升性能；不当配置会引发线程争用导致性能下降。
import onnxruntime as ort
import psutil

sess_options = ort.SessionOptions()
# 设置为物理核心数（而非逻辑核心数），避免超线程带来的资源竞争
cpu_count = psutil.cpu_count(logical=False)
sess_options.intra_op_num_threads = cpu_count
sess_options.inter_op_num_threads = 1  # 大部分模型设为 1 即可

# 配置要点：
# - intra_op_num_threads：单个操作（如矩阵乘法）内部并行线程数，建议=物理 CPU 核心数
# - inter_op_num_threads：计算图中不同操作之间的并行度
#   * 线性结构模型（如 ResNet）：设为 1 通常效率最高
#   * 分支并行结构模型（含 Gather/If）：可适当增加
# - 严格资源占用要求场景：全部设为 1，推理仅占 1 个 CPU 核心
