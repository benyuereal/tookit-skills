# io_binding 手动控制内存交互时机（DCU 平台）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§4.2
# 说明：默认 ORT 每次推理自动在 CPU 与 DCU 间复制输入/输出张量（隐式同步），
#       低延迟/高吞吐场景引入不必要开销。io_binding 允许预先在设备内存分配
#       输入/输出张量并显式控制拷贝与计算的顺序，消除冗余内存传输。
# 收益：可减少 20%~50% 的单次推理延迟（取决于输入数据量及硬件带宽）。
import onnxruntime as ort
import numpy as np

# 假设使用 ROCm EP 或 MIGraphX EP
session = ort.InferenceSession(
    "model.onnx",
    providers=['ROCMExecutionProvider']  # 或 'MIGraphXExecutionProvider'
)

# 1. 在 GPU 上预分配输入/输出 buffer（实际应用中可能直接从 GPU 获取输入）
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
output_tensor = np.empty((1, 1000), dtype=np.float32)

# 2. 将 numpy 数组绑定到 GPU 内存
io_binding = session.io_binding()
io_binding.bind_input(
    name="input", device_type="cuda", device_id=0,  # ROCm 兼容 cuda 设备类型标识
    element_type=np.float32, shape=input_tensor.shape,
    buffer_ptr=input_tensor.ctypes.data  # 指向已分配的 GPU 内存
)
io_binding.bind_output(
    name="output", device_type="cuda", device_id=0,
    element_type=np.float32, shape=output_tensor.shape,
    buffer_ptr=output_tensor.ctypes.data
)

# 3. 执行推理（无隐式内存拷贝）
session.run_with_iobinding(io_binding)

# 4. 仅在必要时将结果拷贝回 CPU
result = output_tensor.copy()

# 适用场景：
# - 输入数据本身就位于 DCU 内存中（视频解码后的帧、图像预处理后的 GPU 张量）
# - 输出结果可直接在 GPU 上消费（下游计算也在 GPU 执行）
# - 高吞吐场景下频繁的内存拷贝成为瓶颈时
#
# ROCm 特别说明：
# DCU 平台使用 io_binding 时需确保 ROCm EP 或 MIGraphX EP 已正确初始化，
# device_type="cuda" 是 ROCm 兼容层的标准写法（ROCm 沿用 CUDA 设备类型标识保持 API 兼容）。
