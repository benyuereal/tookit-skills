# 算子库画像：每个库愿意为什么形状付什么税

> 用法：先读 §一 的匹配表定库，再读该库画像的**三行**（结构定律 / 可调面 / 扩展门）。
> 画像不是源码地图——源码地图在 `dcu-operators/dcu-*-source`，本文只写「决定能不能赢」的那一层。
>
> 纪律：本文只写**相对倍数与机制**。绝对耗时/带宽进 `~/.config/dcu-toolkit/kb/op-opt/`；
> 每个倍数都带「相对谁」，因为本项目最贵的一次错就是分母换了没写。

## 一、匹配表（档位 × 复用结构 → 库）

| 工作负载 | 首选 | 为什么它结构上对 | 谁在这一档结构性吃亏 |
|---|---|---|---|
| M≤8、权重主导（decode GEMV） | 手写 ladder（hbird 形态）或库的 thin-M 桶 | 每步必读整份权重 ⇒ 唯一杠杆是字节流与 CTA 数，tile 再大是浪费 | 通用 GEMM 模板（tl.dot 路线在 M 64–256 只有参考实现的 0.29–0.41×） |
| M=16–128、权重主导 | hbird 形态的 fp8/bf16 ladder | 一档一个 tile 高度 + plan 层按 shape 选 tile/nw/split-K | 单一 tile 吃所有 M 的库：小 M 付 padding，大 M 付 IO |
| M≥256、算力主导（prefill） | DeepGEMM / hipBLASLt / TileLang | 复用已经足够，杠杆是 mmac 发射率与每 barrier 工作量 | 手写小 tile；以及「无软件流水线」的 stock 配置（大 M 上留 1.1–2×） |
| grouped / MoE（每专家几行） | DeepGEMM 三路径；或 AITER 四后端调度 | token 组织方式（masked / 物理连续 / ll）决定权重布局，布局决定能不能复用 | 用稠密 GEMM 循环调 per-expert（本项目实测一次 launch 的 grouped 形态是 graph 化稠密路径的 2.9×） |
| attention（dense / paged / MLA） | flash-attn 系 / FlashMLA / TileLang 稀疏 MLA | 瓶颈在 KV gather 与 softmax 的片上搬运，不在 GEMM | 用手写 GEMM 拼 attention（每条 MMA 摊的搬运指令高 3–4×） |
| 逐元素/norm/激活/量化的链 | LightOp 融合族 | 融合只对 memory-bound 链成立，收益单一：省 HBM 往返 | 对 compute-bound 主体做融合（净负） |
| 形状不定 / 数据相关（topk、ragged） | Triton hcu 或手写 | 表达力换确定性；但要接受 tl.dot 在 DCU 的真实位置 | 把 Triton 当 GEMM 引擎（见上） |

**这张表怎么用**：先回答「M 档」与「复用结构」两个问题，表里只有一行对得上。对不上两行 ⇒ 说明档位没定清，回 SKILL.md §二，不要先挑库。

## 二、七个库的画像

### 1. hbird 形态（手写 fp8/bf16 GEMM，一档一 tile + plan 层）

- **结构定律**：**驻留（CTA/CU）比 tile 形状更早成为约束**。同一份 body，LDS 需求从 16 KB 涨到 32 KB 就把 3 CTA/CU 打到 2，而 1 CTA/CU 的代价是墙钟 +27.5%（网格是 64 CU 的整数倍，排除了尾部效应）。
- **可调面**：tile 高度阶梯、waves/CTA（nw）、split-K 切片数 s、A 读 buffer 数、B 预取 slot 数、epilogue 走 LDS 还是直写。plan 层把 (M, N, K) 映射到这些参数，**s 要按 shape 单独搜**（本项目的四个服务权重最优点各不相同）。
- **已留在地上的**（相对它的对手）：8192³ 上从 ~1.08× 追到 1.76× 的对手比；decode 三个 cell 里两个仍未达 90% 目标，且「差距的种类不同」——一个是 A pack，一个是 split-K 平面字节。**这些是「已知还要付的税」，不是未知空间**。
- **扩展门**：改 kernel 必须同步 plan 层与 manifest（哪个 .so 出厂），否则测的是旧产物；编译 flag 集是测量的一部分（见 `op-hip-compile`）。
- **深引用**：`dcu-operators/dcu-kernel-dev`；定律来源见 `references/pipeline-laws.md` 的 L1–L10。

### 2. LightOp（海光自研融合算子库，无公开上游）

- **结构定律**：**它只对 memory-bound 的算子链兑现收益**。收益来源单一——中间态留在寄存器/smem，HBM 只读一次写一次。对 compute-bound 主体做融合是净负。最激进的一招是 block-specialized grid：一次 launch 处理 Q/KV 两类异构工作。
- **可调面**：分 gfx 的调优表（MoE 量化块、batched GEMM 的 block 维度）、permute/unpermute 的 block 对齐、`num_stages`（stock 对大 M 取 0，这是**已知留在地上的** 1.1–2×）、thin-M（M≤8）桶（stock 未覆盖，实测离墙 1.63–2.10×，扫过 116 组配置后 BM16/BN32/BK256/st4/w4 跨 M 稳定胜出）。
- **扩展门**：csrc 改完要整包重编（按 gfx 显式指定 `PYTORCH_ROCM_ARCH`），pybind 入口集中在 export.cpp 且 Python 侧绑定要同步；部署态 `.so` 用 sha256 认领，不同源码树的产物可能同名不同内容。**没有上游可 diff，适配以 DTK/gfx 矩阵为准**。
- **与 AITER 的耦合**：复用 AITER 的 HIP 公共头，改任一侧都要看另一侧。

### 3. AITER（AMD 上游 + DCU 适配）

- **结构定律**：它是**四后端调度框架**，不是一组 kernel。`get_aiter_moe_config` 按 (量化类型 × dtype) 返回 moe_c / asm / triton / ck 之一，选不中就整体降级（status=False）。所以「AITER 慢」通常意味着**查表没命中**，而不是 kernel 慢。
- **可调面**：分 gfx 的 asm/ck/moe_c 调优表、`spec_sol_type` 强制后端、block_size（0=channel-wise）。**调用前必须 shuffle 权重**（`aiter_moe_shfl_weight`），漏了是静默错误不是报错。
- **已留在地上的**：调度是查表不是自动调优；某量化组合没有解时只能回退。查表命中率是第一个要看的数。
- **扩展门**：首次必须初始化子模块（CK、moe_c），否则编译失败；按 gfx 指定 `GPU_ARCHS`。

### 4. DeepGEMM（DeepSeek 上游的 DCU 移植，纯 GEMM 专精）

- **结构定律**：**三条路径由 token 的组织方式决定，不由偏好决定**。masked（token 不物理重排，权重 marlin/marlin2）→ contiguous（token 已按专家连续，权重 W6 6D packed）→ low-latency（decode 小 M + 与通信 overlap，权重 W6）。布局选错不是慢，是 raise 或静默错。
- **可调面**：masked 的 MODE 表（expected_m 取 2 的幂 → 分 CU 数 dict → JSON 覆盖层，加 shape 调优是往 JSON 加条目而不是改代码）；contiguous 的 size-aware 形状表（(E,N,K)→padded-M 阈值）。
- **已留在地上的**：contiguous 的 k_group kernel 不支持；fp8 masked 仅在 gfx938 及以后；MODE 四档 CU 数当前取值相同（等于没有按 CU 分档）。
- **扩展门**：依赖配套的 hipBLASLt 包（按 DTK 版本）；gfx92a 需显式指定架构。

### 5. flash-attn / FlashMLA（attention kernel 库）

- **结构定律**：**它是 CUDA kernel 的 HIP 化，不是重写**；FA3 那套（Hopper 专属）在 DCU 上不可移植，所以"上游升级能提速"这条路在 FA2 线上没有新东西——版本串变的是海光 fork 的本地 commit。Python 层被大幅扩展（新增 `hg_` 前缀入口、稀疏路径、MLA 入口、vLLM 集成入口）。
- **可调面**：split-KV、block 尺寸、head-dim 编译期特化；gfx936 走 mixed paged attention（decode 专用），gfx938 长序列把 paged KV 连续化再算。
- **已留在地上的**：`mutates_inputs`——它会原地规范化 indices（有效索引前移、`-1` 后移）。多变体对比时共用输入会让后跑的变体吃到被改过的输入。
- **扩展门**：csrc 源码不在 wheel 里，改 kernel 要走源码树重编。

### 6. TileLang（DSL，TVM 系）

- **结构定律**：**它先付布局转换的税，再谈收益**。同一条 MMA，在 tilelang 上摊到的向量/LDS 搬运指令是手写循环的 3–4 倍（这是本项目从 tilelang 转手写的直接动因）。MLS（Matrix Load Store）路径只在白名单 arch 开放，且**要求 kPack=1**——bf16 因为 gather 向量化宽度必须是 2，结构上用不了 MLS，换写法也没用。
- **可调面**：tile 尺寸、k_pack（CDNA 取 2）、`warp_size=64` 必须显式（默认 32 会越界返回自身值导致归约错）、`T.dynamic` 处理动态 M。
- **已留在地上的**：`alloc_shared` 存精度敏感量会随机错（要 `alloc_local`）；内核缓存命中会跳过 lowering（改源码不生效）。
- **扩展门**：`TILELANG_OVERRIDE_DEVICE_SOURCE` 打补丁，直接改生成源码会被重编覆盖。

### 7. Triton（hcu backend）

- **结构定律**：DCU 走**独立的 hcu backend**（amd backend 的 fork + DTK 特化：`libgalaxyhip`、aillvm 工具链）。**`tl.dot` 在 DCU 上的真实位置是「M=64–256 只有参考实现的 0.29–0.41×」** ——所以它适合形状不定/数据相关的算子，不适合当 GEMM 引擎。反过来，非 dot 的 broadcast GEMV 能打满流式峰值，但只在 M=1。
- **可调面**：`hcutune_cli` / `opt_config_cli`（海光新增的分布式调优与最优配置查询）；BLOCK / num_warps / num_stages。
- **已留在地上的**：编译器把失败 kernel 归档到缓存目录下的 failed_kernels（海光新增的调试设施），先看归档再看猜测。
- **扩展门**：工具链缺失时报 AI LLVM 找不到（DTK 安装不完整），不是代码问题。

## 三、跨库的三条共同税（换库不减）

| 税 | 表现 | 判据 |
|---|---|---|
| **布局转换** | 每换一次数据组织（packed ↔ 生产 layout）就多一份搬运；「微基准快、生产掉分」多半是这一条 | 先看变体的输入是否与生产 kernel 的输出同布局 |
| **分母漂移** | 同一份数据两个达成率并存，都是「一个数两个分母」 | 报百分比必须同句写峰值来源与测法 |
| **首调/缓存** | 首次 launch 计时虚高；shape-keyed 缓存让第二次调用返回陈旧结果 | 每次计时用 fresh 输入；fresh 输入算正确性 |
