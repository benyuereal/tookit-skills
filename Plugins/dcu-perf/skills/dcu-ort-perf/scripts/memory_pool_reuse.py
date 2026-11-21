# 内存预开辟 + 复用（DCU 环境最佳实践）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§4.3
# 说明：每次推理 ORT 动态申请输入/输出张量内存，频繁分配/释放带来开销并增加碎片风险。
#       预先在 CPU 和 DCU 显存中开辟固定大小内存池，多次推理复用，配合 io_binding
#       实现零拷贝数据传输，彻底避免重复内存分配/释放。
# 收益：显存分配开销从每请求数百微秒降至接近零；结合 io_binding 可降低 10% 推理延迟。
import onnxruntime as ort
import numpy as np


class ReusableDCUInference:
    """
    DCU 环境下的可复用推理类
    实现 CPU 内存池 + DCU 显存池的一次开辟、多次复用
    """

    def __init__(self, model_path, max_batch_size, input_shape, output_shape,
                 dtype=np.float32):
        """
        参数:
            model_path: ONNX 模型路径
            max_batch_size: 最大 Batch 大小（预分配空间上限）
            input_shape: 输入张量形状（不含 batch 维度）
            output_shape: 输出张量形状（不含 batch 维度）
            dtype: 数据类型
        """
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype

        # ========== 1. 配置 SessionOptions ==========
        sess_options = ort.SessionOptions()
        # 启用 CPU 内存 Arena（减少 CPU 内存分配开销）
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        # 可选：设置内存池初始大小（单位：字节）
        # sess_options.arena_extend_strategy = "kSameAsRequested"

        # ========== 2. 配置 ROCm Execution Provider ==========
        providers = [
            ('ROCMExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',  # 减少显存碎片
                'do_copy_in_default_stream': True,            # 使用默认流拷贝
                'enable_hip_graph': True,                     # 启用 HIP Graph 优化（可选）
            }),
            ('CPUExecutionProvider', {})
        ]

        # 创建推理会话
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # 获取模型输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # ========== 3. 预分配 CPU 内存池 ==========
        # 按最大 batch 大小预分配 CPU 端 buffer
        self.cpu_input_buffer = np.empty(
            (max_batch_size, *input_shape),
            dtype=dtype
        )
        self.cpu_output_buffer = np.empty(
            (max_batch_size, *output_shape),
            dtype=dtype
        )

        # ========== 4. 预分配 DCU 显存池 + 绑定 io_binding ==========
        self.io_binding = self.session.io_binding()
        self._allocate_dcu_buffers()

    def _allocate_dcu_buffers(self):
        """
        在 DCU 显存上预分配输入输出 buffer
        使用 OrtValue 创建显存驻留的张量
        """
        # 创建 CPU 端的临时 OrtValue（用于初始化显存）
        cpu_input = np.empty((self.max_batch_size, *self.input_shape),
                             dtype=self.dtype)
        cpu_output = np.empty((self.max_batch_size, *self.output_shape),
                              dtype=self.dtype)

        # 将 CPU OrtValue 拷贝到 DCU 显存，得到 DCU 端的 OrtValue
        self.dcu_input_tensor = ort.OrtValue.ortvalue_from_numpy(
            cpu_input, 'cuda', 0  # ROCm 兼容 'cuda' 设备类型
        )
        self.dcu_output_tensor = ort.OrtValue.ortvalue_from_numpy(
            cpu_output, 'cuda', 0
        )

        # 绑定到 io_binding
        self.io_binding.bind_ortvalue_input(self.input_name,
                                            self.dcu_input_tensor)
        self.io_binding.bind_ortvalue_output(self.output_name,
                                             self.dcu_output_tensor)

    def predict(self, input_batch):
        """
        执行推理（全程无内存分配）

        参数:
            input_batch: numpy array，形状为 [actual_batch_size, *input_shape]

        返回:
            numpy array，形状为 [actual_batch_size, *output_shape]
        """
        actual_batch_size = len(input_batch)
        if actual_batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {actual_batch_size} exceeds max "
                             f"{self.max_batch_size}")

        # ========== 步骤1: CPU 内存复用 ==========
        # 将输入数据拷贝到预分配的 CPU buffer 中（避免新分配内存）
        self.cpu_input_buffer[:actual_batch_size] = input_batch

        # ========== 步骤2: CPU -> DCU 显存传输 ==========
        # 将 CPU buffer 数据更新到预分配的 DCU 显存中
        dcu_input_numpy = self.dcu_input_tensor.numpy()
        dcu_input_numpy[:actual_batch_size] = \
            self.cpu_input_buffer[:actual_batch_size]

        # 注意：如果形状变化，需要重新绑定（但预分配固定大小时无需）

        # ========== 步骤3: 执行推理（零额外内存分配） ==========
        # 使用 io_binding 运行推理，输出直接写入预分配的 dcu_output_tensor
        self.session.run_with_iobinding(self.io_binding)

        # ========== 步骤4: DCU 显存 -> CPU 结果拷贝 ==========
        # 从预分配的 DCU 输出 buffer 中获取结果
        dcu_output_numpy = self.dcu_output_tensor.numpy()
        self.cpu_output_buffer[:actual_batch_size] = \
            dcu_output_numpy[:actual_batch_size]

        # 返回实际 batch 大小的结果
        return self.cpu_output_buffer[:actual_batch_size].copy()

    def predict_without_cpu_copy(self, dcu_input_ptr, actual_batch_size):
        """
        高级用法：输入数据已位于 DCU 显存中，直接使用指针进行推理（零拷贝）

        参数:
            dcu_input_ptr: DCU 显存指针（需确保数据布局与模型输入一致）
            actual_batch_size: 实际 batch 大小
        """
        # 直接绑定外部 DCU 显存指针（零拷贝）
        self.io_binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=0,
            element_type=self.dtype,
            shape=(actual_batch_size, *self.input_shape),
            buffer_ptr=dcu_input_ptr
        )

        # 执行推理
        self.session.run_with_iobinding(self.io_binding)

        # 从预分配的输出 buffer 获取结果
        dcu_output_numpy = self.dcu_output_tensor.numpy()
        return dcu_output_numpy[:actual_batch_size].copy()

    def get_dcu_output_ptr(self):
        """获取预分配的输出 DCU 显存指针，用于下游直接使用显存数据"""
        return self.dcu_output_tensor.data_ptr()


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 初始化推理引擎（预开辟所有内存）
    inferencer = ReusableDCUInference(
        model_path="model.onnx",
        max_batch_size=32,           # 最大支持 batch 32
        input_shape=(3, 224, 224),   # 输入形状
        output_shape=(1000,),        # 输出形状（例如分类任务）
        dtype=np.float32
    )

    # 2. 模拟多次推理请求（零内存分配）
    test_batch = np.random.randn(8, 3, 224, 224).astype(np.float32)

    # 预热
    for _ in range(10):
        _ = inferencer.predict(test_batch)

    # 性能测试
    import time
    num_iterations = 100
    start = time.perf_counter()
    result = None

    for _ in range(num_iterations):
        result = inferencer.predict(test_batch)

    elapsed = (time.perf_counter() - start) / num_iterations * 1000
    print(f"平均推理延迟: {elapsed:.3f} ms")
    print(f"输出结果 shape: {result.shape}")

# 核心优化点：
# | 优化层级     | 实现方式                                        | 收益                 |
# | CPU 内存池   | enable_cpu_mem_arena=True + 预分配 cpu_input_buffer | 消除 CPU 端动态分配 |
# | DCU 显存池   | ort.OrtValue.ortvalue_from_numpy 预分配显存       | 消除显存动态分配     |
# | 零拷贝传输   | io_binding 绑定预分配显存，复用同一块显存区域       | 避免重复分配/释放    |
# | HIP Graph    | enable_hip_graph=True                           | 减少内核启动开销     |
#
# 配置建议：
# 1. 监控显存使用：用 hy-smi 查看 DCU 显存占用，确保预分配空间充足
# 2. 动态 Batch 处理：预分配最大 Batch 空间，实际推理只使用前 N 个样本
# 3. 多流并发：多实例场景每个实例独立预分配显存，避免资源竞争
