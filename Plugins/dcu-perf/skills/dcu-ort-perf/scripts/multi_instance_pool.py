# 多模型实例提升 QPS（DCU 平台）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§4.4
# 说明：单个 Session 处理并发请求时模型计算串行（或 DCU 在推理外阶段空闲），
#       QPS 上限低。部署多个模型实例（多个独立 InferenceSession）+ 请求级负载均衡，
#       实现请求级并行，显著提升系统总吞吐量。
# 收益：CPU 场景 QPS 随实例数近似线性增长；DCU 场景多实例可填充 GPU 空闲周期，
#       通常提升 30%~100% 吞吐量；结合 MIGraphX EP 静态 shape 优化收益更显著。
import threading
import queue
import onnxruntime as ort


# ========== 4.4.1 CPU 场景：多进程 + 进程池 ==========
def create_session():
    # 每个进程独立创建 session，绑定到不同 CPU 核心
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # 每个实例使用单线程
    return ort.InferenceSession(
        "model.onnx",
        sess_options,
        providers=['CPUExecutionProvider']
    )


class ModelPool:
    def __init__(self, num_instances=4):
        from concurrent.futures import ProcessPoolExecutor
        self.executor = ProcessPoolExecutor(max_workers=num_instances)
        self.sessions = [create_session() for _ in range(num_instances)]

    def predict_async(self, input_data):
        # 轮询分配请求到不同实例
        future = self.executor.submit(
            lambda s, x: s.run(None, {"input": x})[0],
            self.sessions[hash(input_data) % len(self.sessions)],
            input_data
        )
        return future


# ========== 4.4.2 DCU 场景：多实例池 ==========
class DCUModelPool:
    def __init__(self, num_instances=4, device_id=0,
                 ep_type="ROCMExecutionProvider"):
        """
        ep_type: 可选择 "ROCMExecutionProvider" 或 "MIGraphXExecutionProvider"
        """
        self.pool = queue.Queue()
        for _ in range(num_instances):
            # 根据场景选择后端：动态 shape 用 ROCm EP，静态 shape 用 MIGraphX EP
            session = ort.InferenceSession(
                "model.onnx",
                providers=[(ep_type, {
                    'device_id': device_id,
                    'hip_mem_limit': 2 * 1024 ** 3,  # 每个实例显存上限 2GB（HIP 接口）
                })]
            )
            self.pool.put(session)

    def predict(self, input_data):
        session = self.pool.get()
        try:
            return session.run(None, {"input": input_data})
        finally:
            self.pool.put(session)


# 使用示例
# 静态 shape 模型优先使用 MIGraphX EP
# pool_static = DCUModelPool(num_instances=4, ep_type="MIGraphXExecutionProvider")
# 动态 shape 模型使用 ROCm EP
# pool_dynamic = DCUModelPool(num_instances=4, ep_type="ROCMExecutionProvider")

# 4.4.3 最佳实践建议（DCU 平台）：
# | 场景                  | 推荐实例数 | 后端 EP 选择              | 注意事项                     |
# | DCU 静态 shape 小模型 | 2~4       | MIGraphXExecutionProvider | MIGraphX 算子融合更激进，吞吐高 |
# | DCU 动态 shape        | 2~4       | ROCMExecutionProvider     | 动态形状下 ROCm EP 更灵活     |
#
# 多 GPU 扩展建议：
# 多卡 DCU 环境（如多块 K100-AI 系列加速卡）可在每个 GPU 上独立部署模型实例池，
# 通过请求分发层（哈希或一致性哈希）将请求路由到不同 GPU 上的实例，实现多卡线性扩展。
