# vLLM 0.26.x 上游源码解析（vendored reference）

> 来源：`vllm-book-poc-scheduler` 文档库（vLLM 0.26.1 upstream 基线，部分文档追随 upstream main 2026-09 快照）。
> 本目录是**按需加载的参考**，不进 SKILL.md 上下文。查「某模块 vLLM 上游机制/数据结构/源码落点」时按主题取对应文件。
> 已脱敏（客户名/个人路径剥离）、已剥离数据图（仅留正文）。
> ⚠️ 这是 **upstream** 机制解析；DCU 海光 fork 的适配落点见 `dcu-vllm-source` SKILL.md 主表（0.15/0.18/0.21/0.23/0.26 五版）。

## 主题索引

| 文件 | 主题 | 何时查 |
|---|---|---|
| `vllm_production_guide.md` | 框架总览：架构、进程模型、请求全生命周期、TTFT/TPOT | 建立全局观 / 指标定义 |
| `vllm_scheduler_guide.md` | 调度器：4 套调度算法、抢占、chunked prefill、token budget | 调度/排队/TTFT 长尾 |
| `vllm_kvcache_guide.md` | KV Cache：PagedAttention、Block 管理、前缀缓存、LRU 驱逐 | KV 显存/并发上限/APC |
| `vllm_modelrunner_guide.md` | GPU 模型执行器（V1）：InputBatch、采样、CUDA Graph、投机解码集成 | decode 执行/固定 buffer |
| `vllm_model_runner_v2_guide.md` | Model Runner V2（实验性）：ModelState、模块化、V1/V2 对比 | V2 架构 / 混合架构 |
| `vllm_fused_moe_guide.md` | MoE：路由、FusedMoE kernel、EP all2all、EPLB | MoE 计算/路由/专家并行 |
| `vllm_moe_deep_dive.md` | MoE 深入：权重占比、W4A16/W8A8/FP8 量化选型 | MoE 量化选型 |
| `vllm_attention_backend_guide.md` | Attention 后端：22 个后端、MLA、后端选择机制 | attention 后端/MLA/累加器精度 |
| `vllm_quantization_guide.md` | 量化：FP8/GPTQ/AWQ/slimquant_w4a8、kernel 选择 | 量化方案/kernel |
| `vllm_sparse_attention_guide.md` | 稀疏注意力（FlashMLA DSA 执行链路） | 稀疏注意力/DSA |
| `sparse_mla_guide.md` | SparseMLA 专题：Indexer + Attention 两阶段 | 超长上下文稀疏注意力 |
| `vllm_parallel_guide.md` | 并行策略：TP/EP/PP/DP/CP、通信后端 | 多卡/多机并行选型 |
| `vllm_compilation_guide.md` | 编译优化：torch.compile、CUDA Graph、Inductor pass、算子融合 | 编译/图捕获/融合 |
| `vllm_spec_decode_guide.md` | 投机解码：MTP/EAGLE/DFlash/DSpark、拒绝采样、自适应验证 | 投机解码 |
| `vllm_engine_guide.md` | Engine + Executor：请求全生命周期、进程架构、ZMQ、AsyncLLM | 进程模型/API server/ZMQ |
| `vllm_kv_transfer_guide.md` | KV Transfer / P-D 分离：KVConnector 接口、传输流程、状态机 | PD 分离/KV 传输 |

## 配套实战案例
本目录是**原理层**；对应的 DCU 实测案例（现象→根因→方案→量化）在 `dcu-perf-cases`（案例 17-37 为本库来源），
症状速查入口在 `dcu-vllm-deploy/references/perf_symptom_lookup.md`。
