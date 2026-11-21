# 性能优化案例 §14-§16（多机 RDMA / 长上下文 / Triton-ORT）

> 2026-10-16 自 `perf/dcu-perf-cases/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），案例四段式原文照录、未改一字；编号与原文一致，索引见主 SKILL.md 案例索引表。

## 14. 曙光 SHCA IB BW1000 双机 TP16 RDMA 调优（vLLM + Ray + MTP，GLM-5.2-int4）

**硬件/框架**：海光 BW1000（OAM）×2 物理机（各 8 卡，TP16 跨双机）；vLLM（Ray 分布式执行器）+ MTP 投机解码（num_spec=1）；模型 GLM-5.2-INT4-w4a8（slimquant_w4a8，fp8_ds_mla KV）；跨机网络为**曙光国产 SHCA IB 网卡**（shca_0~shca_3 共 4 张），目标 IB RDMA + GDR。

**现象（演进路线）**
- 初始状态：`NCCL_IB_HCA` 已设、网卡物理存在，但 RCCL 日志 `NET/IB : No device found.` → 退回 `Using network Socket`，跨机通信实际走以太网（ib0 只是被当 Socket 接口名），性能远低于 RDMA。
- 调通后：跨机链路 `via NET/IBext_v8/.../GDRDMA`，p2pnChannels=16（perPeer=4），稳定长时运行无 hang/crash，跨机带宽达 SHCA 线速。
- 客户用例量化（ontrack <internal-ticket>，单并发，**分两轮调优**，正文"2803→1121ms、22→29.6"为调优前→最终的跨轮汇总）：

| 指标 | 调优前 | 第一轮后 | 最终（第二轮后） |
|---|---|---|---|
| TTFT 平均 | 2803ms | 1141ms | 1121ms |
| TTFT 最小 | 1939ms | 1085ms | 498ms |
| 速度 平均 | 22.55 tok/s | 29.49 tok/s | 29.60 tok/s |
| 速度 峰值 | 22.94 tok/s | 30.15 tok/s | 30.23 tok/s |

  第一轮（走 Socket→RDMA 调通）贡献了绝大部分收益（TTFT 平均 -59%、吞吐 +31%）；第二轮微调主要把 TTFT 最小值从 1085 压到 498ms（最好情况首包延迟再降一半），平均指标基本持平。

**根因（逐项排查，按真实调通顺序）**
1. **HCA 端口写法不兼容**：`NCCL_IB_HCA=shca_0:1,...` 的 `:1` 后缀该版本 RCCL 不接受，认为指定了无效端口 → 去掉端口号。
2. **libibverbs ABI 冲突**：系统内多个 libibverbs 版本（torch、deep_ep 自带）与 SHCA 官方驱动不兼容 → 偶发挂死、`modify_qp` EINVAL、LID 截断（111896→46360）数据面卡死。LD_PRELOAD 桥接库（libibv_fix）只能缓解枚举、数据面仍不稳 → **放弃桥接，改用 SHCA 官方 RCCL 网络插件 `librccl-net-shca.so`**（直接调 SHCA 原生 verbs API，不依赖系统 libibverbs 兼容层）。
3. **Relaxed Ordering 不兼容**：SHCA 不支持 PCIe RO，开启后 `ibv_reg_mr` EINVAL + DeregMr 崩溃 → `NCCL_IB_PCI_RELAXED_ORDERING=0`，同时关 ECE（`NCCL_IB_ECE_ENABLE=0`）。
4. **通道数两端不对齐**：两节点 p2pnChannels 不一致（一端 8 一端 32）→ `Message truncated: received 8192 instead of 2048` → 两端显式锁死 `NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=16` + `NCCL_NCHANNELS_PER_NET_PEER=4`。
5. **Topo GUID 与实机不符**：vLLM 把头节点的 `NCCL_TOPO_FILE` 广播到所有 worker，但各节点设备 GUID 不同 → `Nic dev attr no exit ... cannot be found` → 每节点独立生成 topo（`hy-smi --showtopo` 取实际 GUID + 修正 XML vendor 字段），Ray 启动脚本按 ib0 IP 区分节点动态设置。
6. **节点身份误判**：`hostname -I` 最先输出 eth0 IP 导致条件判断错 → 改用 `ip -4 addr show ib0` 取地址判定节点身份。
7. **运维干扰（假性 MTP 失败）**：模型加载中途 ActorDied/SIGTERM 疑 OOM，实为外部 `ray stop`/残留 placement group 占 GPU → 先清理所有 Ray 进程再重启集群。

**优化动作（最终生效配置组合，按功能分组）**
| 方向 | 设置 |
|---|---|
| 传输栈 | `NCCL_NET_PLUGIN=.../librccl-net-shca.so` + LD_LIBRARY_PATH 含插件目录 |
| HCA 枚举 | `NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3`（不带端口号） |
| 通道控制 | `NCCL_MIN_NCHANNELS=16` / `NCCL_MAX_NCHANNELS=16` / `NCCL_NCHANNELS_PER_NET_PEER=4` |
| 兼容性 | `NCCL_IB_PCI_RELAXED_ORDERING=0` / `NCCL_IB_ECE_ENABLE=0` |
| GDR | `NCCL_DMABUF_ENABLE=1` / `unset NCCL_NET_GDR_LEVEL` / `NCCL_NET_GDR_READ=0`（配合该插件实测需置 0） |
| OOB 与 Socket | `NCCL_SOCKET_IFNAME=ib0` / `GLOO_SOCKET_IFNAME=ib0` |
| Topo 按节点 | Ray 启动脚本按 ib0 IP 设 `NCCL_TOPO_FILE`（防头节点广播覆盖；亦可改 vLLM `ray_env.py` 的 `non_carry_over_env_vars` 把 NCCL_TOPO_FILE 加入排除项） |
| 节点绑定 | `VLLM_HOST_IP=$(ip -4 addr show ib0 \| awk '/inet/ {print $2}' \| cut -d/ -f1)` |

完整启动脚本（含 vLLM 侧参数：`-tp 16`、Ray 后端、`fp8_ds_mla` KV、MTP、NUMA 绑核、`VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY` 等）见 `scripts/shca_ib_tp16_start.sh`。

**验证终态**：日志出现 `NET/IBext_v8 : Using [0]shca_0:<ip>` 与 `Channel 00/0 : ... via NET/IBext_v8/0/GDRDMA`；p2pnChannels=16、perPeer=4；topo 无 `cannot be found` 警告；`/v1/models` 返回模型名。

**可迁移结论**
1. **国产/非 Mellanox IB 网卡的 RCCL 插件是关键**：不要试图用通用 verbs 桥接（LD_PRELOAD 兼容库）强行适配，必须用厂商提供的 RCCL 网络插件（SHCA 为 `librccl-net-shca.so`）——桥接只能缓解枚举，数据面（QP 状态机/LID）仍会卡死。
2. **核心思想是"剥离非官方组件，逐项核对硬件与软件参数匹配性"**：回归厂商推荐的驱动 + 插件 + 参数组合，而非通用手段硬套。
3. **环境变量要干净**：`NCCL_IB_HCA` 后不加端口号；`NCCL_IB_PCI_RELAXED_ORDERING` 对 SHCA 必须置 0（不支持 RO，开了 MR 注册直接 EINVAL）。
4. **通道数必须两端显式锁死**：MIN=MAX 同值 + per-peer 固定，避免两端自动协商不一致导致消息截断（`Message truncated`）。
5. **Topo 文件按节点独立**：GUID 各节点不同，Ray/vLLM 的环境变量广播会把头节点 topo 覆盖到 worker——启动脚本按节点 IP 动态设置，或改 `ray_env.py` 排除列表。
6. **GDR 启用依赖 `NCCL_DMABUF_ENABLE=1`**；`NCCL_NET_GDR_READ` 因插件实现而异，SHCA 插件实测置 0 更稳定。
7. **调试先排除运维干扰**：ActorDied/SIGTERM 先 `ray stop` 清干净再重试，区分真 OOM/崩溃与误操作/残留 placement group。
8. **节点身份判定用 `ip -4 addr show ib0`**，不依赖 `hostname -I`（输出顺序不保证 ib0 在前）。

> 溯源：内部文档 file_id=<internal-file-id>（内部文档链接，已略）《曙光IB BW1000双机RMDA调优记录》；ontrack 工单 <internal-ticket>（【BW1000】曙光IB BW1000双机推理GLM5.2-int4性能低）。具体报错排查条目见 `dcu-ontrack-experience` C 节；双机 rccl-tests 带宽测试流程见 `dcu-rccl-test`。

## 15. Qwen3.8-27B W8A8 K100AI 三产品线（TP4/TP2/TP1）系统性长上下文调优

**硬件/框架**：海光 K100AI（RDNA gfx928，64KB LDS / 120 CU / INT8 Tensor Core）×1/×2/×4；SourceFind SGLang 0.5.12（海光 fork）+ DTK 26.04；模型 Qwen3.8-27B（27B，hidden 5120，64 层，16 组「3×Gated DeltaNet + 1×full attention」= 48 层线性 attention + 16 层 full attention，full-attention geometry Q24/KV4/D256，原生 context 262144）W8A8（SmoothQuant INT8：权重 per-channel、激活动态 per-token）。三条产品线：TP4=长上下文性能冠军、TP2=性价比/容量折中、TP1=单卡完整能力 + correctness 基线。

**背景（为什么是「系统性」而非单点）**：本项目从「让模型正确运行」演化为「针对 gfx928 + hybrid attention + DFlash2 深度特化的推理执行栈」。核心价值不是某个 kernel，而是建立了一套可复用的优化方法论：**先建 correctness floor，再用 matched A/B 找根因；按 phase × shape × K-family × layer × TP-rank geometry × KV length 做 specialization；所有激进路径 fail-closed；用 offline benchmark 发现 crossover 后固化 dispatch，而非凭直觉写阈值**。

**现象（三线最终 authority，节选关键档，cold/output256）**
| Context | TP1 TTFT/Decode | TP2 TTFT/Decode | TP4 TTFT/Decode |
|---|---|---|---|
| 16K | 16.55s / 26.59 | 9.59s / 81.27 | 4.603s / 113.10 |
| 64K | v1时值 | v2中值 | v3高值 |
| 128K | v1时值 | v2中值 | v3高值 |
| 257.9K | v1时值 | v2中值 | v3高值 |

TP4 从早期 row-LDSx 到 LongCtx v8：257.9K TTFT 与 Decode 吞吐均显著改善（见上表）。TP2/TP4 长 TTFT 比例呈健康的两个对口四卡 scaling。

**根因（逐项，按通用价值排序）**
1. **TP + 动态 per-token INT8 量化的 shared-scale 语义（最具通用价值，vLLM 同样适用）**：W8A8 的 activation scale 是 dynamic per-token。TP1 对完整 K 维算 `scale_global=max(abs(x_fullK))/127`；TP2 RowParallel 后每 rank 只见 local K，独立量化成 `scale_r=max(abs(x_rank))/127`，**两者数学不等价**（实测同一 token 两 rank local absmax 可差一个数量级）。这不是数值噪声，是量化定义被 TP K-sharding 改写。
2. **gfx928 q≥5 paged multi-token no-write vendor bug**：vendor 多 token consumer 在 q≥5 时是 no-op（不写输出），历史「超快」native 路径实为无效（漏写/重复）。必须禁 vendor no-write、走 proven corrected path。
3. **hybrid attention 的 O(N)+O(N²) 长 prefill 曲率**：48 层线性 attention 贡献近似 O(N)，16 层 full attention 逐 chunk long-KV 累积形成 O(N²)。TP4 dense TTFT 拟合 `TTFT≈0.120+0.25109·K+0.00183586·K²`（K=context/1024，R²=0.99998867）。「64K cliff」不是 64K kernel 阈值，而是倍增采样把平滑二次曲率视觉放大。
4. **TP1 raw-q8 verifier 的 geometry 依赖**：raw-q8 在 TP2/TP4 是明确 winner，但 TP1 QH24/KVH4 geometry 在 64K 可用、128K 数值漂移、257.9K isolated VMFault → 必须按 geometry fail-closed，不能为代码统一硬开。

**优化动作（可迁移方法论，按作用排序）**
1. **TP-MAX shared per-token scale（修根因 1）**：两 rank absmax 取 global max 恢复 TP1 语义（quality 改善，case15 缩短）。再逐维收窄共享范围：K family（Klocal∈{3072,8704} 同时共享，单修任一不足）→ phase（仅 prefill + 首次 decode，decode-step bisection N=0 fail/N=1 pass）→ layer（仅 32–47）。路由函数 `route_shared_scale(phase, request_id, is_first_decode, row_mask, Klocal, transformer_layer_id)`，解析不到 `model.layers.<id>` 时 fail-closed 报错而非猜。
2. **production-v2 否决（方法论最佳示例）**：试图用 `max(a,b)=max(a,(a+b)-a)` 以 custom-AR SUM→MAX 取代 NCCL MAX，逻辑正确、quality 仍 18/20，但 matched 16K 造成 TTFT 上升、Decode 下降 → 正式否决。结论：model collective 保留 custom-AR，shared-scale collective 回 NCCL MAX，只保留 bounded state 修复。**「逻辑正确 + 质量不变」不等于「可接受」，必须 matched A/B 性能判定。**
3. **Unified Adaptive 路由（TP4 基座）**：RouteKey 至少含 phase/q_len/kv_len/cached_prefix_len/new_tokens/batch_size/TP geometry/speculative state/draft block/layer family，**不要只看 context_length**（一个「128K 请求」可能是 cold q≈128K，也可能是 Agent continuation cached_prefix=120K+q=8K，最佳 kernel 完全不同）。流程：offline autotune → correctness filter → 统计性能判定 → frozen dispatch table → runtime fail-closed selection（unknown shape → fallback correct parent）。
4. **reject 的是 (算法, parent_stack, shape) 组合，不是算法名**：TP1 8/20 冻结时否决的 BM128/direct-paged/split-KV，在 TP2/TP4 后续研究改变可用算法集合后回移 TP1 又重新赢过旧 U036。不要把算法名永久拉黑。
5. **DFlash2 投机 + raw-q8 verifier + CUDA Graph bs1 + Radix Cache**：DFlash2 block8 draft 大幅推高 257.9K decode 吞吐；raw q8 paged attention 一次 target verify 代替 2×q4 workaround（isolated 多长 KV 有稳定倍率提升，bitwise gate）；bs1 decode graph 固化微 kernel launch；Radix prefix cache（128K workspace 高 hit 时 TTFT 显著下降）。
6. **exact-shape + fail-closed 纪律**：所有激进路径（raw-q8、q16/split4、q3948 tail、ceil-causal）都绑定 exact geometry/shape，未审计 shape 一律 fallback correct parent。q3948 ceil-causal 用 ceil 算 block count 避免 partial block 漏写/重复（修复 SHA 漂移、acceptance 崩溃、一次 SIGSEGV）。

**量化结果（证明根因的前后对照，非固定基线）**
- 根因 1（shared-scale）：TP2 local scale → TP-MAX shared：quality 恢复 TP1 语义。
- 根因 2（no-write）：sentinel q≥5 changed_count=0 证明 native 路径漏写，corrected path 后 quality PASS。
- 根因 3（O(N)+O(N²)）：TP4 dense TTFT 拟合 R²=0.99998867，与 48 线性 + 16 full 结构高度一致。
- 根因 4（raw-q8 geometry）：TP1 128K 数值漂移 / 257.9K VMFault → 保留 2×q4 q8split；TP2/TP4 raw-q8 isolated 有稳定倍率提升。
- 演进：TP4 257.9K TTFT 与 Decode 显著改善；TP2/TP1 同步改善。

**可迁移结论（跨项目通用）**
1. **TP + 动态 per-token 量化必须显式处理 scale 语义**：K-sharding 后 local scale ≠ global scale，是量化定义被改写而非噪声。修复要逐维收窄（K family/phase/layer）而非全局付费，且路由解析失败要 fail-closed。vLLM 的 W8A8/FP8 动态量化在 TP 下同样适用。
2. **「逻辑正确 + 质量不变」≠「可接受」**：任何 collective/transport 改动必须 matched A/B 性能判定（production-v2 被否决即此）。
3. **reject 的是 (算法, parent_stack, shape) 组合**：parent stack 演进后旧 reject 可能复活，别把算法名永久拉黑。
4. **hybrid attention 长 prefill 是 O(N)+O(N²)**：线性层 O(N) + full attention O(N²)，「cliff」多为二次曲率的视觉假象，先建模再找 kernel 阈值。
5. **激进路径必须 exact-shape + fail-closed**：raw-q8 这类 geometry 敏感路径，未审计 shape 一律 fallback，不能为代码统一硬开。
6. **Unified Adaptive 路由要带全 feature vector**：不要只看 context_length，cold 与 Agent continuation 同长度但最佳 kernel 不同。
7. **DFlash 可见 decode 可越过 true-M1 weight-bandwidth roof**：target verifier 一次确认多个 visible token，摊薄 weight streaming；spec 参数必须按硬件/模型/batch/KV 区间重新 autotune，不能照搬论文。

> 溯源：用户提供《Qwen3.8-27B W8A8 on K100AI：TP4、TP2、TP1 三产品线系统性调优研究报告》（2026-08-24）。内部 authority：github_release/qwen38-k100ai-dflash2-tp4/{TP1,TP2,TP4}.md + results/*acceptance*。通用方法论（shared-scale 语义 / Unified Adaptive 路由 / reject 组合原则 / O(N)+O(N²) 建模）已抽到 `dcu-inference-optimization` §十二；DFlash telemetry 与长上下文质量门契约见 `dcu-llm-benchmark` §3.8；gfx928 no-write bug 与 DFlash2 prefill graph 门禁见 `dcu-sglang-source`。

## 16. Triton Server + ORT YOLOv7 BW100 单卡调优（调度层 × 算子层叠加）

**硬件/框架**：海光 BW100 ×1 单卡；Triton Server r24.11 + ORT 1.22.2（onnxruntime_onnx 后端，ROCM EP）；镜像 `<internal_registry>/dcu/admin/base/custom:tritonserver-r24.11-ubuntu22.04-dtk26.04-py3.10-ort1.22.2`；模型 YOLOv7（end2end 内置 NMS，640×640，COCO 80 类）。

**背景**：YOLOv7 以 Triton 服务部署（onnxruntime_onnx 后端）做系统性调优。核心认知：**Triton（调度层）与 ORT（算子层）调优正交、可叠加**——Triton 解决「调度侧瓶颈」（排队/串行执行/Batch 未利用），ORT 解决「算子侧瓶颈」（CPU/GPU 算子性能/内存/线程）。

**现象（基线，默认配置）**：饱和点 53~60 inf/s，DCU 利用率 49%（峰值 100% 但平均仅一半），P50 延迟随并发线性增长（并发 1→256，P50：31ms → 4447ms）。

**根因（profile，onnxruntime_perf_test + hy-smi）**
1. **CPU 后处理算子是最大瓶颈**：ArgMax（后处理算子）在 CPU 跑且默认单线程，中位 2109us，远超其他 CPU 算子（Gather/Concat 仅 3-4us）。
2. **DCU 算力未饱和**：并发=4 时 GPU avg 48.9%（峰值 100% 但维持不住）→ 还有 ~2 倍算力余量。
3. **串行排队**：单实例、无并行执行，延迟随并发线性增长。

**优化动作（三阶段，可叠加）**
1. **ORT 会话参数调优**（config.pbtxt parameters）：`execution_mode=1` + `intra_op_thread_count=8` + `inter_op_thread_count=2` → ArgMax 2109→350us（**6 倍**），CPU 算子合计 2.17→0.48ms（-78%）。
2. **多实例并行**（instance_group.count）：扫 1/2/4/8（并发=16 测得）：1→54.8 inf/s（49%）、2→78.5（87.5%，+43%）、4→91.6（100%，甜点）、8→93.5（92.2%，仅 +2% 且 P95 变差）。**4 实例是甜点**——DCU 算力 100% 占用，再多只加剧 kernel 调度争抢。
3. **动态批处理**（max_batch_size=8 + dynamic_batching）：前置用 `--dynamic --dynamic-batch` 重导 ONNX；`preferred_batch_size [1,2,4]` + `max_queue_delay 3μs` + `preserve_ordering false`（乱序换吞吐）。

**量化结果（数值为该次记录值、非固定基线）**
| 并发 | Baseline TPS | 调优后 TPS | Baseline Avg 延时(ms) | 调优后 Avg 延时(ms) |
|---|---|---|---|---|
| 1 | 31.93 | 40.01 | 31.3 | 25.0 |
| 8 | 53.65 | 88.52 | 148.9 | 90.2 |
| 32 | 54.53 | 117.45 | 583.9 | 272.1 |
| 128 | 57.97 | **134.12** | 2178.3 | **959.1** |

- 极限吞吐：60.4 → **134.1 inf/s（+122%）**；平均延迟 @128：2178 → 959ms（**-56%**）；饱和点：并发 4 → 128+。
- 精度：DCU 推理检测精度与 CPU 基准基本一致，无明显精度损失。

**可迁移结论**
1. **两层正交是核心方法论**：Triton（调度层）与 ORT（算子层）调优可叠加——先调 ORT 会话参数（算子侧）→ 再多实例（调度侧）→ 最后动态批处理（调度侧），每步独立可验证。
2. **CPU 后处理算子是隐藏瓶颈**：检测类模型（YOLO 等）的后处理算子（ArgMax/Gather/Concat）默认在 CPU 跑且单线程，耗时可能远超 GPU 算子。Profile 时先看 CPU vs GPU 耗时比例。
3. **多实例甜点判定**：「DCU 利用率接近 100% 且 P95 未明显恶化」是甜点；再多实例只加剧 kernel 调度争抢（P95 变差）。
4. **动态批处理要重导模型**：原 ONNX 是静态 shape 时，必须 `--dynamic --dynamic-batch` 重导，否则 `max_batch_size > 0` 不生效。
5. **七步 SOP + 反模式**（可复用方法论）见 `dcu-ort-perf` §7。

> 溯源：飞书文档《Triton Server + ONNXRuntime 通用模型推理调优最佳实践》（doc_token=U40QdIQymodM1FxpOolcXA3tn7d）。
