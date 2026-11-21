# 算子库画像：每个库愿意为什么形状付什么税

> 用法：先读 §一 的匹配表定库，再读该库画像的**五行**（结构定律 / 可调面 / 已知还留在地上的税 / 何时选它 / 何时别选）。
> 画像不是源码地图——源码地图在 `dcu-operators/dcu-*-source`，本文只写「决定能不能赢」的那一层。
>
> 纪律：本文只写**相对倍数与机制**。绝对耗时/带宽进 `~/.config/dcu-toolkit/kb/op-opt/`；
> 每个倍数都带「相对谁」，因为本项目最贵的一次错就是分母换了没写。
> 外部仓库一律只写仓库名（`hytlass` / `BoltOPs` / `Moe` / `TileKernels-das`），不写仓内路径——仓内没有，写了就是死链。

## 一、匹配表（档位 × 复用结构 → 库）

| 工作负载 | 首选 | 为什么它结构上对 | 谁在这一档结构性吃亏 |
|---|---|---|---|
| M≤8、权重主导（decode GEMV） | 手写 ladder（hbird 形态）或库的 thin-M 桶 | 每步必读整份权重 ⇒ 唯一杠杆是字节流与 CTA 数，tile 再大是浪费 | 通用 GEMM 模板（tl.dot 路线在 M 64–256 只有参考实现的 0.29–0.41×） |
| M=16–128、权重主导 | hbird 形态的 fp8/bf16 ladder | 一档一个 tile 高度 + plan 层按 shape 选 tile/nw/split-K | 单一 tile 吃所有 M 的库：小 M 付 padding，大 M 付 IO |
| M≥256、算力主导（prefill） | DeepGEMM / hipBLASLt / TileLang | 复用已经足够，杠杆是 mmac 发射率与每 barrier 工作量 | 手写小 tile；以及「无软件流水线」的 stock 配置（大 M 上留 1.1–2×） |
| grouped / MoE（每专家几行） | DeepGEMM 三路径；或 AITER 四后端调度 | token 组织方式（masked / 物理连续 / ll）决定权重布局，布局决定能不能复用 | 用稠密 GEMM 循环调 per-expert（本项目实测一次 launch 的 grouped 形态是 graph 化稠密路径的 2.9×） |
| MoE 的某一冷门量化组合（8 档里没解的那档） | 直接改 `Moe` 仓的 per-gfx config 表 | 它按 precision × gfx 存表，是 AITER 选不到别的后端时的唯一实现 | 在 AITER 上层反复调参（后端都没选中，参数根本不生效） |
| attention（dense / paged / MLA） | flash-attn 系 / FlashMLA / TileLang 稀疏 MLA | 瓶颈在 KV gather 与 softmax 的片上搬运，不在 GEMM | 用手写 GEMM 拼 attention（每条 MMA 摊的搬运指令高 3–4×） |
| 低时延 decode 形态的 attention / MoE / 线性注意力（已在 Python 层） | `BoltOPs`（全 Triton + 自带 autotune CLI） | 算子族按低时延形状切好，调参不靠自己搭 | 大 M prefill 走它的 GEMM（Triton 路线固有税） |
| 逐元素/norm/激活/量化的链 | LightOp 融合族 | 融合只对 memory-bound 链成立，收益单一：省 HBM 往返 | 对 compute-bound 主体做融合（净负） |
| 模型结构层算子（mHC / Engram / GDN / 量化融合）+ 可训练层 | `TileKernels-das` | 官方 tilelang 实现按「层」组织，直接给 `autograd.Function` 组合 | 单条 memory-bound 链（LightOp 更成熟）；且融合链本身可能更慢（见该画像） |
| 要做**一类新 GEMM**（非标准布局 / 融合 epilogue / split-K·stream-K），不想从零写 HIP | `hytlass`（CUTLASS 3.5.0 移植） | tile 与流水线是编译期类型，改的是 template 参数不是指令序列 | 从零手写（周期以周计）；只能调暴露参数的通用 GEMM 模板（没有中间层） |
| 形状不定 / 数据相关（topk、ragged） | Triton hcu 或手写 | 表达力换确定性；但要接受 tl.dot 在 DCU 的真实位置 | 把 Triton 当 GEMM 引擎（见上） |

**这张表怎么用**：先回答「M 档」与「复用结构」两个问题，表里只有一行对得上。对不上两行 ⇒ 说明档位没定清，回 SKILL.md §二，不要先挑库。

## 二、十一个库的画像

### 1. hbird 形态（手写 fp8/bf16 GEMM，一档一 tile + plan 层）

- **结构定律**：**驻留（CTA/CU）比 tile 形状更早成为约束**。同一份 body，LDS 需求从 16 KB 涨到 32 KB 就把 3 CTA/CU 打到 2，而 1 CTA/CU 的代价是墙钟 +27.5%（网格是 64 CU 的整数倍，排除了尾部效应）。
- **可调面**：tile 高度阶梯、waves/CTA（nw）、split-K 切片数 s、A 读 buffer 数、B 预取 slot 数、epilogue 走 LDS 还是直写。plan 层把 (M, N, K) 映射到这些参数，**s 要按 shape 单独搜**（本项目的四个服务权重最优点各不相同）。
- **已留在地上的**（相对它的对手）：8192³ 上从 ~1.08× 追到 1.76× 的对手比；decode 三个 cell 里两个仍未达 90% 目标，且「差距的种类不同」——一个是 A pack，一个是 split-K 平面字节。**这些是「已知还要付的税」，不是未知空间**。
- **何时选它**：档位已经定到「单 shape、单 dtype、愿意为它维护一份 body + plan」；或已有 kernel 要推进到极限且愿意付编译/重测成本。
- **何时别选**：要做**一类** GEMM 而不是一个（先看 §2 hytlass）；团队没有 plan 层/编译 flag 的测量纪律（改完忘了重编，测的是旧产物）。
- **扩展门**：改 kernel 必须同步 plan 层与 manifest（哪个 .so 出厂），否则测的是旧产物；编译 flag 集是测量的一部分（见 `op-hip-compile`）。
- **深引用**：`dcu-operators/dcu-kernel-dev`；定律来源见 `references/pipeline-laws.md` 的 L1–L10。

### 2. hytlass（CUTLASS 3.5.0 → HCU 移植，BSD-3-Clause）

- **结构定律**：**tile/流水线在编译期定型，运行时只剩 launch**。它是 CUTLASS 3.5.0（固定 commit）的 HCU 移植：兼容 2.x 的 threadblock/warp 模型（BW/gfx936 的 mmac + `ds_read_matrix` 原语）与 3.x 的 CuTe 模型（仓内称 HuTe）。换 tile、换 stage、换 split-K 方式 = 换一个模板实例，不是查一张表。所以它把「排流水线」的成本前移成一次设计成本，把运行时的 shape 适配成本留空。
- **可调面**：`TileShape` / `StageCount` / `ThreadblockShape`、KernelSchedule（basic / serial split-K / parallel split-K / stream-K）、epilogue、dtype（TF32/FP16/BF16/I8/U8）。19 个示例就是 19 个 schedule 样板：basic / serial·parallel split-K / stream-K / batch / group / fused / 5 个卷积变体 / gather-scatter / gemm+softmax / ELL block 稀疏 / gemm+amax。其中 `gemm_softmax` 是 attention 类算子的骨架、`gemm_with_abs_max` 是「epilogue 里顺手算量化统计量」的标准做法。
- **已留在地上的**：**没有 autotune 表，也没有 shape→plan 层**——这两个恰好是本项目从手写 GEMM 里已经做出来的东西，用它要自己补；示例是「跑得通」不是「调优过」的 best practice，与手写极限之间仍有距离；每个 (dtype × tile × schedule) 一份编译产物，实例爆炸。
- **何时选它**：要搭**一类** GEMM（非标准 layout、融合 epilogue、split-K/stream-K 调度），不想从零写 HIP；或要把手写 kernel 的设计固化成可复用模板；或要保留 CUTLASS 风格的接口以便移植上游 kernel。
- **何时别选**：只要一个现成 GEMM 且 DeepGEMM / hipBLASLt 已覆盖（改造模板的周期远大于调优表命中）；M≤8 的 decode（thin-M 桶更快）；团队不熟 C++ 模板/CuTe（编译错误与类型报错的调试成本高于手写）。
- **扩展门**：需 DTK 25.10+、CMake 3.19+、C++17；上游是 CUTLASS 3.5.0 的固定 commit，升级要整块 re-port，不能只 diff 单个文件。

### 3. LightOp（海光自研融合算子库，Apache-2.0，仓 `lightop` master）

- **结构定律**：**它只对 memory-bound 的算子链兑现收益**。收益来源单一——中间态留在寄存器/smem，HBM 只读一次写一次。对 compute-bound 主体做融合是净负。最激进的一招是 block-specialized grid：一次 launch 处理 Q/KV 两类异构工作。
- **可调面**：分 gfx 的调优表（MoE 量化块、batched GEMM 的 block 维度）、permute/unpermute 的 block 对齐、`num_stages`（stock 对大 M 取 0，这是**已知留在地上的** 1.1–2×）、thin-M（M≤8）桶（stock 未覆盖，相对墙 1.63–2.10×；扫过 116 组配置后 BM16/BN32/BK256/st4/w4 跨 M 稳定胜出）。构建：`source /opt/dtk/env.sh && python3 setup.py install`，或多架构 wheel 走仓内 `build_hcu_wheel.sh`（产物在 `dist-manylinux/`）。
- **已留在地上的**：覆盖的是「算子族齐全」而不是「每档都调过」——thin-M 与大 M 的 `num_stages` 都是已知缺口；csrc 改完要整包重编，不同源码树的 `.so` 可能同名不同内容（部署态用 sha256 认领）；没有上游可 diff，适配以 DTK/gfx 矩阵为准。
- **使用时先看它自带的 agent skill**：仓内置 `.agents/skills/`，含 `lightop-kernel-agent-loop`（kernel 优化闭环）与 `build-hcu-wheel`。闭环里已经带了带宽测量脚本（`measure-device-bandwidth`）、HIP 优化规则参考、profiler 摘要脚本和一个可查询的 kernel 知识库。**要测带宽/找优化项时先用它自带的工具，不要另写探针**——分母口径不一致是本项目最贵的一类错。
- **何时选它**：目标是 memory-bound 链（norm / 激活 / 量化 / rope / 采样 / 视觉预处理）且希望一个 kernel 吃掉整条链；需要 MoE 的量化融合或 block-specialized grid；希望有现成的优化闭环脚本可复用。
- **何时别选**：compute-bound 的 GEMM 主体（用 DeepGEMM / hytlass）；只需要单个无链算子（融合没有可省的往返）；要跨 gfx 单一二进制（按 gfx 重编 + 调优表各一份）。
- **扩展门**：csrc 改完要整包重编（按 gfx 显式指定 `PYTORCH_ROCM_ARCH`），pybind 入口集中在 `export.cpp` 且 Python 侧绑定要同步；**与 AITER 复用 HIP 公共头，改任一侧都要看另一侧**。

### 4. AITER（AMD 上游 + DCU 适配，仓 `aiter-das` main）

- **结构定律**：它是**四后端调度框架**，不是一组 kernel。`get_aiter_moe_config` 按 (量化类型 × dtype) 返回 moe_c / asm / triton / ck 之一，选不中就整体降级（status=False）。所以「AITER 慢」通常意味着**查表没命中**，而不是 kernel 慢；而且四个后端里至少有一个不在本仓。
- **可调面**：分 gfx 的 asm/ck 调优表、`spec_sol_type` 强制后端、block_size（0=channel-wise）。**调用前必须 shuffle 权重**（`aiter_moe_shfl_weight`），漏了是静默错误不是报错。构建需先 `git submodule update --init`，按 gfx 指定 `GPU_ARCHS`（如 `GPU_ARCHS="gfx936" pip install -e . --no-build-isolation --no-deps`），或走仓内 `rebuild_aiter.sh`。
- **已留在地上的**：调度是查表不是自动调优；某量化组合没有解时只能回退。查表命中率是第一个要看的数。
- **moe_c 不在本仓**：MoE 的 `MOE_C` 后端来自独立仓库 `Moe`（`csrc/` + `csrc_for_aiter/936_938/` + `moe_c_configs/`），AITER 只是通过子模块引用它。**AITER 的 MoE 慢时，先确认 `solution_type` 是不是 `MOE_C`；是的话直接去改 `Moe` 仓的 `moe_c_configs/`（按 gfx × 量化类型组织），比换库便宜得多**。详见 §5。
- **何时选它**：标准 MoE/Attention/Norm/Quant 接入，想要一层统一的 Python API 和四后端回退；已在 vLLM/SGLang 生态内被调用。
- **何时别选**：需要单一路径的可预测性能（查表未命中会静默换后端）；需要 GEMM 的 layout 精细控制（走 DeepGEMM）；融合链（走 LightOp）。
- **扩展门**：首次必须初始化子模块（CK、moe_c），否则编译失败；`PREBUILD_KERNELS=1` 的预编译 kernel 与 `GPU_ARCHS` 强绑定，换 gfx 必须重编。

### 5. Moe / moe_c（AITER 的第三方 MoE 后端，也可独立构建）

- **结构定律**：**它是一张「精度模式 × gfx」的矩阵，不是一个通用 kernel**。8 种精度模式：W4A8(channelwise) / W8A8 INT8(channelwise/blockwise/tensorwise) / W8A8 FP8(channelwise/blockwise/tensorwise) / W4A16(groupwise) / W8A16(channelwise/blockwise) / WFP4A8(channelwise/groupwise32) / WFP4A16(groupwise) / W16A16(channelwise)。外加 top-k softmax、moe alignment、MOE sum、Silu-and-mul 等支撑算子。能否快**完全取决于该 (量化, gfx) 组合的调优表有没有这一档**。
- **可调面**：`moe_c_configs/`（按 gfx × 量化类型组织；加 shape 调优 = 加表项）、`tune/`（批量搜索）、`csrc/` 的 per-precision 入口、`csrc_for_aiter/936_938/` 的 gfx 头与 intrinsic。构建 `build.sh`，`MOE_C_TARGET=moe_w4a8` 只编单个模块，`MOE_C_HIP_ARCHITECTURES=gfx938` / `MOE_C_BUILD_DIR` / `MOE_C_BUILD_JOBS` 控制产物。
- **已留在地上的**：表没命中的 shape 走 fallback（正确但慢）；8 档里冷门的几档（WFP4A8 / WFP4A16）覆盖面明显窄于主流档；权重 shuffle 的前置条件仍然成立，错序是静默错；改表后要重启进程（表在加载期读入）。
- **何时选它**：AITER 选中了 `MOE_C` 且慢——直接改表；某个量化组合在 asm/triton/ck 里没有解而 moe_c 有；需要按 gfx 单独维护一份 MoE 调优资产。
- **何时别选**：`get_aiter_moe_config` 根本没选它（改表无效，先强制 `spec_sol_type` 验证）；非 MoE 算子；不想多维护一份构建（直接用 `aiter-das` 的子模块版本）。
- **扩展门**：与 AITER 的 `csrc_for_aiter/` 共享头文件，两边接口漂移会编译失败；gfx 专属 intrinsic 在 `csrc_for_aiter/936_938/`，新增 gfx 要同时补目录与配置。

### 6. DeepGEMM（DeepSeek 上游移植，仓 `deepgemm-das` main，纯 GEMM 专精）

- **结构定律**：**两条坐标同时决定路径：token 的组织方式 × (arch, 权重 layout)**。masked（token 不物理重排，权重 marlin/marlin2 或 HIPC）→ contiguous（token 已按专家连续，权重 W6 6D packed）→ low-latency（decode 小 M + 与通信 overlap，权重 W6）。在上面再叠一层架构分叉：`csrc/arch/gfx92a`、`gfx936`、`gfx938`、`gfx936_gfx938_shared`。布局选错不是慢，是 raise 或静默错。
- **可调面**：masked 的 MODE 表（expected_m 取 2 的幂 → 分 CU 数 dict → JSON 覆盖层，加 shape 调优是往 JSON 加条目而不是改代码）；contiguous 的 size-aware 形状表（(E,N,K)→padded-M 阈值）；`shuffle_unique` 开关（i8 contiguous 上默认 1 走 shuffle-unique/HIPC，0 在 gfx936/gfx92a 走 ASM，gfx938 不区分）。接口面（skill 里此前缺的几条）：`m_grouped_w4a8_gemm_nt_masked_hipc`（权重先经 `pack_w4a8_moe_hipc_weight` 转 HIPC 专用 layout）、`m_grouped_w8a8_gemm_nt_masked_gfx936`（W6 packed，`pack_w8a8_gfx936_weight_to_w6` 生成）、`m_grouped_fp8_gemm_nt_masked_ll` 及其 `_ll_tensor_wise` 变体、`m_grouped_i8_gemm_nt_contiguous`、`fp8_gemm`（走 hipBLASLt）、`einsum` / `fp8_einsum`（BF16 的 bmm/einsum 走 hipBLASLt，FP8 落到 `fp8_gemm`）。
- **已留在地上的**：contiguous 的 k_group kernel 不支持；fp8 masked 仅在 gfx938 及以后；MODE 四档 CU 数当前取值相同（等于没有按 CU 分档）；gfx936 的专属路径与通用路径各有一份表，**选错入口不是慢而是 layout 错**（HIPC / W6 / marlin 三种 layout 不可混用）。
- **何时选它**：grouped / MoE GEMM；需要按 layout 或 arch 精细控制（W4A8 HIPC、INT8 W6、gfx936 专属）；想借 hipBLASLt 吃普通 fp8/einsum。
- **何时别选**：普通稠密大 GEMM（直接 hipBLASLt 即可，省一层）；attention / 融合链（不覆盖）；要求一套代码跨 gfx936/938 不分叉（分叉正是它的成本）。
- **扩展门**：依赖配套的 hipBLASLt 包（按 DTK 版本）；gfx92a 需显式指定架构。

### 7. flash-attn / FlashMLA（attention kernel 库，仓 `flash-attention-das` / `FlashMLA-das`）

- **结构定律**：**它是 CUDA kernel 的 HIP 化，不是重写**；FA3 那套（Hopper 专属）在 DCU 上不可移植，所以"上游升级能提速"这条路在 FA2 线上没有新东西——版本串变的是海光 fork 的本地 commit。Python 层被大幅扩展（新增 `hg_` 前缀入口、稀疏路径、MLA 入口、vLLM 集成入口）；FlashMLA 侧覆盖 BF16/FP8 与稀疏 MLA 的 prefill/decode。
- **可调面**：split-KV、block 尺寸、head-dim 编译期特化；gfx936 走 mixed paged attention（decode 专用），gfx938 长序列把 paged KV 连续化再算。
- **已留在地上的**：`mutates_inputs`——它会原地规范化 indices（有效索引前移、`-1` 后移）。多变体对比时共用输入会让后跑的变体吃到被改过的输入。
- **何时选它**：dense/paged/MLA attention 且瓶颈在 KV 搬运；需要稀疏 MLA（DSA 类）的现成实现；已在 vLLM/SGLang 路径上。
- **何时别选**：需要新的 attention 变体语义（改 fork 的成本高于在 TileLang/Triton 里写）；M 极小且 KV 极短（launch 与固定开销主导，融合别的更值）。
- **扩展门**：csrc 源码不在 wheel 里，改 kernel 要走源码树重编。

### 8. TileLang（DSL，TVM 系）

- **结构定律**：**它先付布局转换的税，再谈收益**。同一条 MMA，在 tilelang 上摊到的向量/LDS 搬运指令是手写循环的 3–4 倍（这是本项目从 tilelang 转手写的直接动因）。MLS（Matrix Load Store）路径只在白名单 arch 开放，且**要求 kPack=1**——bf16 因为 gather 向量化宽度必须是 2，结构上用不了 MLS，换写法也没用。
- **可调面**：tile 尺寸、k_pack（CDNA 取 2）、`warp_size=64` 必须显式（默认 32 会越界返回自身值导致归约错）、`T.dynamic` 处理动态 M。
- **已留在地上的**：`alloc_shared` 存精度敏感量会随机错（要 `alloc_local`）；内核缓存命中会跳过 lowering（改源码不生效）。
- **何时选它**：形状不定或需要 tile 级表达力，又想保留可读的 Python 源码；MLS 白名单内的 arch 且 dtype 不是 bf16；有 `TILELANG_OVERRIDE_DEVICE_SOURCE` 可打补丁的需求。
- **何时别选**：要逼近手写极限的 GEMM/attention（先付 3–4× 指令税）；bf16 + MLS 的幻想（结构上不成立）；gfx936 上跑 MLS kernel（白名单不含，必崩）。
- **扩展门**：`TILELANG_OVERRIDE_DEVICE_SOURCE` 打补丁，直接改生成源码会被重编覆盖。

### 9. Triton（hcu backend）

- **结构定律**：DCU 走**独立的 hcu backend**（amd backend 的 fork + DTK 特化：`libgalaxyhip`、aillvm 工具链）。**`tl.dot` 在 DCU 上的真实位置是「M=64–256 只有参考实现的 0.29–0.41×」** ——所以它适合形状不定/数据相关的算子，不适合当 GEMM 引擎。反过来，非 dot 的 broadcast GEMV 能打满流式峰值，但只在 M=1。
- **可调面**：`hcutune_cli` / `opt_config_cli`（海光新增的分布式调优与最优配置查询）；BLOCK / num_warps / num_stages。
- **已留在地上的**：编译器把失败 kernel 归档到缓存目录下的 failed_kernels（海光新增的调试设施），先看归档再看猜测。
- **何时选它**：算子形状随数据变化（topk、ragged、gather 类）；要快速出一个可用的正确实现；需要编译器层调优工具。
- **何时别选**：GEMM 主体（见上）；需要精细控制寄存器/LDS 的 kernel（hcu backend 暴露不到那一层）；对编译时间与缓存行为敏感的生产路径。
- **扩展门**：工具链缺失时报 AI LLVM 找不到（DTK 安装不完整），不是代码问题。

### 10. BoltOPs（Triton/TileLang 低时延算子包，仓 `BoltOPs` release/0.1.0，pip 名 `boltops`）

- **结构定律**：**它按「算子族 × 后端」组织，是低时延形状的成品集合，不是引擎**。`boltops.fa`（decode/extend/triangle/unified，全 Triton，unified 含 int4_per_token_head）、`boltops.gemm`（matmul_persistent / gemm_a16w16 / w8a8_triton_block_scaled_mm / triton_scaled_mm / awq_gemm_triton）、`boltops.fused_moe`（fused_experts / moe_activation）、`boltops.fla`（GDN 等线性注意力：cumsum / fused_sigmoid_gating_recurrent / gdn_verify）、`boltops.generic`（attn_res / causal_conv1d_fn / causal_conv1d_update / fused_mamba_state_scatter_with_mask）。选型时可以把它当「decode 形状的成品表」来查。
- **可调面**：各族的 Python 入口参数；**自带 MoE autotune CLI**（`boltops-fused-moe-tune-cli <device_name> <test_type> [--benchmark]`），调出的配置落在它的 tune 目录，可直接复用。安装粒度：`pip install -e ".[triton]"` / `".[triton,dev]"` / `".[tune]"` / `".[all]"`。
- **已留在地上的**：全部建立在 Triton/TileLang 之上，**先付 DSL 层的布局税**；autotune 产物与 device_name 绑定，换卡要重调；版本年轻（release/0.1.0），接口稳定性不如 AITER/LightOp。
- **何时选它**：decode 形状的 attention/MoE/线性注意力，且你已经在 Python 层、不想维护 HIP；需要 int4 per-token-head、GDN 这类较新的算子；想把调参直接交给它的 CLI。
- **何时别选**：大 M prefill 的 GEMM（Triton 路线固有税）；要求极限性能并愿意手写 HIP（DSL 上限在那）；需要长期稳定 ABI（版本很新）。
- **扩展门**：`.[tune]` 才带调优依赖，只装 `.[triton]` 时 CLI 不可用；产物与 device_name 绑定，多卡环境要分别跑。

### 11. TileKernels（基于 tilelang-hygon 的模型层算子，仓 `TileKernels-das` staging）

- **结构定律**：**它按「模型层」而不是按「kernel」组织**。覆盖 Gating（top-k 专家选择）、MoE Routing、Quantization（per-token / per-block / per-channel 的 FP8/FP4/E5M6，含融合 SwiGLU+量化）、Transpose（batch）、Engram（融合 RMSNorm、前反向、权重梯度归约）、**Manifold HyperConnection（mHC）**、Modeling（用 `torch.autograd.Function` 组合成可训练层）。也就是说它交付的是「一层」，不是「一个 op」。
- **可调面**：`tile_kernels/mhc/` 下的 8 个 kernel——expand / hc_split_sinkhorn / head_compute_mix / multilayer_recompute / norm_fn / post / pre_apply_mix / **pre_big_fuse**；量化族的 scale 粒度（token/block/channel）；`tests/mhc/` 下有对应的 bench（含 `bench_mhc_pre_big_fuse`）。环境门槛：Python 3.10+、PyTorch 2.10+、TileLang 1.12.0+，卡型 BW1000/BW1100/BW150/K100_AI。
- **已留在地上的**：**融合不总赢**。`pre_big_fuse` 把原本可并行的独立 kernel 串成一条依赖链并多占 smem，本项目实测它比非融合组合慢 1.7×——这正是「融合只对 memory-bound 链成立」的反例，用之前先跑 `tests/mhc` 的 bench 对比分拆版；tilelang 的布局税（3–4× 搬运指令）照付。
- **何时选它**：要 mHC / Engram / GDN 这类**模型结构层**算子的官方实现（可读可改，省掉自己推公式）；要在 tilelang 生态内把多个 op 组合成可训练层；量化融合（SwiGLU+quant）已有现成的 per-block/per-channel 变体。
- **何时别选**：只要单条 memory-bound 链（LightOp 的 C++ 实现更成熟、可控）；环境版本不满足（torch/TileLang 门槛比别的库高）；延迟极敏感的 decode（融合链可能更慢，先测）。
- **扩展门**：构建依赖 `tilelang-hygon` 的版本，与 `dcu-tilelang-source` 的编译器版本强相关；升级 tilelang 要整组重测（8 个 mhc kernel 都依赖同一套 lowering）。

## 三、跨库的四条共同税（换库不减）

| 税 | 表现 | 判据 |
|---|---|---|
| **布局转换** | 每换一次数据组织（packed ↔ 生产 layout）就多一份搬运；「微基准快、生产掉分」多半是这一条 | 先看变体的输入是否与生产 kernel 的输出同布局 |
| **分母漂移** | 同一份数据两个达成率并存，都是「一个数两个分母」 | 报百分比必须同句写峰值来源与测法 |
| **首调/缓存** | 首次 launch 计时虚高；shape-keyed 缓存让第二次调用返回陈旧结果 | 每次计时用 fresh 输入；fresh 输入算正确性 |
| **融合的依赖链** | 融合把可并行的 kernel 串成一条链、多占 smem；memory-bound 时净赚，否则净亏（`pre_big_fuse` 实测慢 1.7×） | 先量「分拆版」的和，再决定融不融；别默认融合更快 |
