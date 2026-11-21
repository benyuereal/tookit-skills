# TileLang naive GemmaRMSNorm 实现（源自 内部文档 主文档 §2.2）
# 用途: vLLM 自定义算子替换示例 (USE_TILELANG 开关)
# 平台: DCU 海光; tilelang backend_target 须显式 "hip"
# 注意: alloc_local=寄存器, alloc_shared=共享内存; 精度敏感量用 alloc_local
# 溯源: 内部文档 file_id=<internal-file-id>

_TILELANG_KERNEL_CACHE = {}


def get_cached_tilelang_kernel(hidden_size: int, eps: float, dtype: str):
    cache_key = (hidden_size, eps, dtype)
    if cache_key not in _TILELANG_KERNEL_CACHE:
        
        # M代表动态的Batch Size维度
        M = T.dynamic('M', 'int32')
        # 每个线程需要处理的元素步数: hidden_size 均匀分配给 256 个线程
        num_steps = (hidden_size + 255) // 256

        @T.prim_func
        def gemma_rms_norm_kernel(
            X: T.Tensor((M, hidden_size), dtype),
            W: T.Tensor((hidden_size,), dtype),
            Output: T.Tensor((M, hidden_size), dtype),
        ):
            # Grid = M 个 Block，Block = 256个Thread
            with T.Kernel(M, threads=256) as pid:
                # reduce_shared: 256 个线程的局部平方和暂存，用于 Block 级规约
                reduce_shared = T.alloc_shared((256,), "float32")
                # rms_shared: 单值广播 RMS，所有线程从同一地址读取
                rms_shared = T.alloc_shared((1,), "float32")

                # --------------------------------------------------------------
                # T.thread_binding: 显式绑定 256 个线程到 threadIdx.x
                # --------------------------------------------------------------
                for tx in T.thread_binding(256, thread="threadIdx.x"):
                    local_sum = T.alloc_local((1,), "float32")
                    local_sum[0] = 0.0

                    # ----------------------------------------------------------
                    # local_x: 用寄存器缓存 X 的分片数据
                    # Phase 1 从 Global Memory 加载到寄存器，Phase 3 直接
                    # 从寄存器回算，避免二次 Global Load
                    # ----------------------------------------------------------
                    local_x = T.alloc_local((num_steps,), "float32")

                    # ---------- Phase 1: 加载 X + 计算平方和 ----------
                    for i in T.serial(num_steps):
                        idx = tx + i * 256
                        if idx < hidden_size:
                            val = T.cast(X[pid, idx], "float32")
                            local_x[i] = val
                            local_sum[0] = local_sum[0] + val * val
                            
                    reduce_shared[tx] = local_sum[0]

                    T.sync_threads()

                    # ---------- Phase 2: 并行树状规约 ----------
                    # 步骤: offset = 128, 64, 32, 16, 8, 4, 2, 1。
                    # 256个元素规约8次，时间复杂度O(logN)
                    for step in T.serial(8):
                        offset = 128 >> step
                        if tx < offset:
                            reduce_shared[tx] = reduce_shared[tx] + reduce_shared[tx + offset]
                        T.sync_threads()

                    # 0 号线程计算最终 RMS
                    if tx == 0:
                        rms_shared[0] = T.sqrt(reduce_shared[0] / T.cast(hidden_size, "float32") + eps)

                    T.sync_threads()
                    
                    # ---------- Phase 3: 归一化----------
                    rms = rms_shared[0]
                    
                    for i in T.serial(num_steps):
                        idx = tx + i * 256
                        if idx < hidden_size:
                            # 直接从 local_x 寄存器取数
                            x_val = local_x[i]
                            w_val = T.cast(W[idx], "float32")

                            # Gemma/Qwen 的归一化公式: y = (x / rms) * (1.0 + w)
                            normed_val = x_val / rms
                            out_f32 = normed_val * (1.0 + w_val)

                            Output[pid, idx] = T.cast(out_f32, dtype)

        backend_target = "hip" if torch.version.hip else "cuda"
        _TILELANG_KERNEL_CACHE[cache_key] = tl.compile(gemma_rms_norm_kernel, target=backend_target)
        
    return _TILELANG_KERNEL_CACHE[cache_key]
