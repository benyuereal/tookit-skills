# vLLM MoE 深入：权重占比、核心特点与量化选型

> 本文是 [`vllm_fused_moe_guide.md`](vllm_fused_moe_guide.md) 的**原理与选型补充**。
> 那篇讲"代码怎么实现"（FusedMoE 层、router、kernel、EP、EPLB），这篇讲"是什么、为什么、怎么选"：
> MoE 的权重到底占多少、MoE 和 dense 模型的本质区别、W4A16/W8A8/FP8 等量化方案怎么选。
> 所有数字都用案例库里的真实模型（某医疗MoE模型 / MiniMax-M3 / Qwen3.5-35B-A3B / DeepSeek-V3）算，比抽象公式直观。

---

## 目录

1. [MoE 一句话本质](#1-moe-一句话本质)
2. [权重占比：MoE 模型到底"大"在哪](#2-权重占比moe-模型到底大在哪)
3. [MoE 的核心特点（和 dense 模型的本质区别）](#3-moe-的核心特点和-dense-模型的本质区别)
4. [量化方案详解：W4A16 / W8A16 / W8A8 / FP8](#4-量化方案详解w4a16--w8a16--w8a8--fp8)
5. [MoE 量化选型指南](#5-moe-量化选型指南)
6. [MoE 量化的特有坑](#6-moe-量化的特有坑)
7. [真实模型速查表](#7-真实模型速查表)
8. [常见问题](#8-常见问题)

---

## 1. MoE 一句话本质

**MoE = 用"条件计算"换"参数容量"。**

一个 dense 模型，每个 token 都要过**全部**权重；MoE 把 FFN 拆成 N 个专家（expert），每个 token 只激活其中 top_k 个。于是：

- **总参数**（模型文件多大、占多少显存）= 全部 N 个专家之和 → **大**
- **激活参数**（每个 token 实际算多少）= top_k 个专家 → **小**

这就是为什么 DeepSeek-V3 能有 671B 参数却只花 37B 的计算量。MoE 的"大"是显存问题，"快"是计算问题——**这两件事要分开看**，也是本文第 2 章的核心。

```
dense 模型:  每个 token → 过全部 FFN 权重          计算量 ∝ 总参数
MoE 模型:    每个 token → 只过 top_k 个专家        计算量 ∝ top_k × 单专家
```

---

## 2. 权重占比：MoE 模型到底"大"在哪

### 2.1 单专家权重公式

一个专家就是一个小型 FFN，含 3 个矩阵（gate / up / down）：

```
单专家权重 = 3 × hidden_size × intermediate_size × dtype_bytes
```

| 模型 | hidden | intermediate | dtype | 单专家权重 |
|------|--------|-------------|-------|-----------|
| 某医疗MoE模型 (BailingMoeV2) | 4096 | 1024 | bf16 (2B) | 3×4096×1024×2 = **24.6 MB** |
| Qwen3.5-35B-A3B | 2048 | 768 | bf16 | 3×2048×768×2 = **9.4 MB** |
| DeepSeek-V3 | 7168 | 2048 | fp8 (1B) | 3×7168×2048×1 = **44 MB** |

### 2.2 MoE 层权重占比（关键结论：MoE 层占整个模型 90%+）

以 **某医疗MoE模型**（32 层，L0 dense + L1–L31 共 31 个 MoE 层，每层 256 专家）为例：

```
单个 MoE 层 = 256 专家 × 24.6 MB = 6.3 GB
31 个 MoE 层 = 31 × 6.3 GB = 195 GB
```

这正好对上案例里"本地模型 191.65 GB"。也就是说：

> **MoE 模型的显存 90%+ 是专家权重，attention / embedding / norm 都是零头。**

这是 MoE 和 dense 模型最反直觉的地方——dense 模型显存大头是"每层都算的 FFN"，MoE 模型显存大头是"每层只算 top_k 个但存了全部 N 个的专家"。

### 2.3 每 token 激活权重（decode 每步的 HBM 流量）

每个 token 每层只激活 top_k 个专家：

```
每 token 每层激活权重 = top_k × 单专家权重
某医疗MoE模型: 8 × 24.6 MB = 197 MB/层
31 层合计:    31 × 197 MB = 6.1 GB / 每 token
```

**这 6.1 GB 就是 decode 每生成一个 token 要从 HBM 读出的流量**——MoE decode 是 memory-bound 的根源。对照某高速公路项目案例：Qwen3.5-2B 每步读 3.32GB、HBM 实测 1.2TB/s、GEMM 带宽利用率 77%，就是这个道理。

> **推论**：MoE 的 decode 性能瓶颈是 **HBM 带宽**，不是 FLOPS。优化方向是"减少每步读的权重"（量化）或"提高带宽利用率"（kernel 调优），而不是堆算力。

### 2.4 量化后的权重占比

W4A16 把权重压到 4-bit（含 scale/zero 约 4.5 bit 有效），单专家权重约降到 1/4：

```
某医疗MoE模型 单专家:  24.6 MB (bf16) → ~6.5 MB (W4A16)
31 个 MoE 层:        195 GB → ~49 GB
```

这就是 MiniMax-M3（456B）能用 8×K100AI（64GB×8=512GB）跑起来的原因——AWQ INT4 把 456B 压到 ~115GB 量级。

### 2.5 EP 切分后的单卡占比

专家并行（EP）把 N 个专家分到 ep_size 张卡，每卡存 `N/ep_size` 个专家：

```
某医疗MoE模型 256 专家, EP=8:  每卡 32 专家 × 24.6 MB = 790 MB/层
31 层: 每卡 ~24.5 GB
```

对照 `vllm_fused_moe_guide.md` 第 9 章的 EP 切分逻辑（`local_num_experts = global_num_experts // ep_size`）。

### 2.6 KV cache 占比（反直觉点）

**MoE 模型的 KV cache 和同 hidden 的 dense 模型一样大**——因为 attention 是 dense 的，KV cache 只和 attention 头数/hidden 有关，和专家数无关。

所以 MoE 模型的显存构成是：

```
总显存 ≈ 专家权重（90%+，MoE 特有） + attention/embedding（小） + KV cache（和 dense 同量级）
```

> **调 `gpu_memory_utilization` 时**：MoE 模型能分给 KV cache 的空间比同显存的 dense 模型少得多（因为专家权重占了大头），所以 MoE 模型的**最大并发数**往往比直觉低。

---

## 3. MoE 的核心特点（和 dense 模型的本质区别）

### 3.1 条件计算（稀疏激活）

每 token 只激活 top_k 专家，计算量 ∝ top_k 而非 num_experts。这是 MoE 省算力的根本。

### 3.2 decode 是 memory-bound

见 2.3。每步读 top_k 专家权重，**带宽利用率是核心指标**。这是 MoE 和 dense 模型调优思路最大的不同——dense 大 batch 时可能 compute-bound，MoE 几乎总是 memory-bound。

### 3.3 小 batch GEMM 问题（MoE 量化的根本动机）

decode 时 token 被路由打散到各专家，**每个专家的 GEMM 的 M 维（token 数）很小**：

```
batch=32, 256 专家, top_k=8:  32×8=256 个 (token,专家) 对，摊到 256 专家 → 每专家平均 1 个 token
```

M=1 的 GEMM 是纯 memory-bound，FLOPS 利用率极低。这就是为什么：
- MoE 量化（W4A16）重要——减少每步读的字节数
- Marlin kernel 重要——为小 M 的 dequant+GEMM 专门优化
- 见 `vllm_fused_moe_guide.md` 第 8 章的 `fused_moe_kernel_gptq_awq`

### 3.4 专家负载不均

路由不是均匀的，有的专家"热"（被频繁选中）、有的"冷"。后果：
- EP 下热专家所在卡成为瓶颈
- 需要 **EPLB**（专家负载均衡，见 `vllm_fused_moe_guide.md` 第 11 章）/ 冗余专家 / expert_bias 来均衡

### 3.5 路由是命脉

**路由错 = 输出错**，且是"结构性错误"不是"精度漂移"。某医疗MoE模型 案例：vLLM 的 fused gate kernel 在特定参数下选错专家（0/32 匹配），导致整段输出乱码——而 router_logits 本身是对的（cos=0.999998）。

> **教训**：MoE 精度问题排查，**先查路由（topk_ids 对不对），再查专家 FFN**。路由是 0/1 的，错了就是全错。

### 3.6 Grouped GEMM 与对齐

token 按专家打散后，要按专家分组做 GEMM（grouped GEMM）。不同专家的 token 数不同，需要 `moe_align_block_size` 把每个专家的 token 对齐到 block 边界（见 `vllm_fused_moe_guide.md` 第 8.5 节）。这是 MoE kernel 比 dense GEMM 复杂的核心原因。

### 3.7 All2All 通信（EP 的隐藏成本）

EP 下 token 要 dispatch 到专家所在卡、算完再 combine 回来，通信量：

```
All2All 通信量 ≈ hidden_size × num_tokens × 2 (dispatch + combine) × dtype_bytes
```

这是 EP 的隐藏成本——专家越多、hidden 越大，通信越重。对照 `vllm_fused_moe_guide.md` 第 9 章的 DeepEP / PPLX 后端。

### 3.8 Shared Expert（始终激活）

部分模型（DeepSeek-V3、某医疗MoE模型）有 shared expert，**每个 token 都过**，和 routed expert 相加。vLLM 用独立 CUDA stream 让 shared expert 和 routed expert 重叠执行（见 `vllm_fused_moe_guide.md` 第 12 章）。

### 3.9 精度敏感点：router 不量化

router 权重（gate）对精度敏感，**量化通常只量化专家 FFN，router 保持高精度**。这也是为什么量化配置里有 `ignored_layers`（见 `vllm_quantization_guide.md` 第 2 章）。

---

## 4. 量化方案详解：W4A16 / W8A16 / W8A8 / FP8

### 4.1 命名约定

```
WN  = 权重 N-bit（weight N-bit）
A16 = 激活 16-bit（activation 16-bit，即 fp16/bf16）
A8  = 激活 8-bit
```

- **W4A16**：权重 4-bit + 激活 16-bit → **weight-only 量化**，dequant 到 16-bit 再做 GEMM
- **W8A8**：权重 8-bit + 激活 8-bit → GEMM 直接用 8-bit 算，更快但要硬件支持
- **FP8**：权重/激活都用 FP8（E4M3/E5M2），是 W8A8 的浮点版本

### 4.2 W4A16（GPTQ / AWQ）—— MoE 主流

**原理**：权重压到 4-bit，配 per-group 的 scale（和 zero_point），推理时 dequant 到 16-bit 再 GEMM。

```
存储:  4-bit 权重 + scale (fp16) + zero_point (4-bit, 非对称时)
计算:  dequant(4bit → 16bit) → 16bit GEMM
显存:  ~1/4 (bf16 的 1/4)
```

**kernel 后端**（见 `vllm_fused_moe_guide.md` 第 10 章量化 kernel 矩阵）：

| 后端 | 文件 | 特点 |
|------|------|------|
| **Marlin** | `fused_marlin_moe_w4a16.py` | 最快，SM75+，权重需 Marlin interleaved 布局 |
| Triton WNA16 | `fused_moe.py` (`TritonWNA16Experts`) | 通用 fallback |
| CPU WNA16 | `cpu_wna16.py` | CPU 后端 |

**AWQ vs GPTQ**：
- **GPTQ**：基于 Hessian 的二次量化，对称/非对称都有
- **AWQ**：激活感知量化，保护"显著权重"，通常非对称（有 zero_point）
- 两者在 vLLM 里都走 WNA16 路径（`moe_wna16.py`），区别在量化时的校准策略

### 4.3 W8A16

权重 8-bit + 激活 16-bit。显存 1/2，精度比 W4 高。适合"显存有点紧但 W4 精度不够"的场景。

### 4.4 W8A8（INT8 / FP8）

权重激活都 8-bit，GEMM 直接 8-bit 算：

```
INT8 W8A8:  int8 GEMM + int32 累加 → 反量化
FP8 W8A8:   fp8 GEMM (E4M3) + fp32 累加
```

**优势**：比 W4A16 的 dequant+16bit GEMM 快（省了 dequant 步骤，8-bit GEMM 吞吐高）。
**前提**：硬件要支持 8-bit GEMM（DCU DTK 的 hipBLAS/aiter 支持情况需确认）。
**MoE 后端**：DeepGEMM（block 128 量化，N>512）、CUTLASS、Triton（见 `vllm_fused_moe_guide.md` 第 10 章）。

### 4.5 FP8 细节

- **E4M3**：4 位指数 3 位尾数，范围小精度高，适合权重/激活
- **E5M2**：5 位指数 2 位尾数，范围大精度低，适合梯度
- **block 量化**：FP8 MoE 常用 128×128 block 量化（per-block scale），DeepGEMM 后端要求 N>512
- **在线量化**：`Fp8OnlineMoEMethod` 运行时量化，不需要预量化权重

### 4.6 group_size（scale 粒度）

per-group 量化的 group_size（128/64/32）：

```
group_size 越小 → scale 越细 → 精度越高，但 scale 存储越多
group_size=128:  每 128 个权重共享 1 个 scale（主流）
group_size=32:   每 32 个权重共享 1 个 scale（精度更高，MiniMax-M3 用 32）
```

### 4.7 symmetric vs asymmetric（重要坑）

- **对称量化**：`weight = scale × q`，无 zero_point
- **非对称量化**：`weight = scale × (q - zero_point)`，有 zero_point

**AWQ 通常是非对称的**（有 zero_point）。这引出 MiniMax-M3 案例的两个坑（见第 6 章）：
1. `symmetric` 参数默认 True → 非对称权重的 zero_point 被丢弃
2. zero_point 的 reshape 顺序错 → nibble 维和 group 维混淆

---

## 5. MoE 量化选型指南

### 5.1 决策表

| 瓶颈 | 推荐方案 | 显存 | 精度损失 | 说明 |
|------|---------|------|---------|------|
| **显存装不下** | W4A16 (Marlin) | 1/4 | 1-3% | MoE 主流，Marlin kernel 最快 |
| **显存紧 + W4 精度不够** | W8A16 | 1/2 | <1% | 折中 |
| **算力瓶颈 + 硬件支持 8bit GEMM** | W8A8 / FP8 | 1/2 | 1-2% | 省 dequant，8bit GEMM 快 |
| **精度敏感（router/小模型）** | 不量化 / W8A16 | 1~1/2 | ~0 | 小模型量化损失占比高 |
| **极致压缩** | NVFP4 / MXFP4 | 1/4~1/8 | 2-5% | 新方案，硬件支持有限 |

### 5.2 选型逻辑

```
1. 先问：显存装得下吗？
   - 装不下 → W4A16（1/4 压缩）
   - 装得下 → 看下一步

2. 再问：是 memory-bound 还是 compute-bound？
   - memory-bound（MoE decode 几乎总是）→ 量化减少读取字节，W4A16 收益最大
   - compute-bound（大 batch prefill）→ W8A8/FP8 的 8bit GEMM 更快

3. 再问：硬件支持什么？
   - DCU DTK: 确认 aiter/hipBLAS 的 8bit GEMM 支持
   - Marlin: SM75+（DCU 对应 gfx 版本需确认）

4. 最后：精度验收
   - 小模型（<10B）量化损失占比高，慎选 W4
   - router 不量化
   - 用案例库的精度验证方法（逐元素对比 ground truth）
```

### 5.3 DCU 实测参考（来自案例库）

| 模型 | 量化 | 硬件 | 结果 |
|------|------|------|------|
| GLM-5.1 | slimquant_w4a8 | DCU | 生产在用（见 `vllm_quantization_guide.md` 第 13 章） |
| MiniMax-M3 (456B) | AWQ INT4 (W4A16) | 8×K100AI | 单用户 ~19 tok/s（见 `example/minimax-m3-awq-int4.md`） |
| gemma-4-31B | AWQ W4A16 + aiter | BW10 | TPOT -25%（见 `example/gemma-spec-decode-aiter.md`） |
| DeepSeek-V3.2 | FP8 (W8A8) | BW1100 | 生产在用（见 `example/deepseek-v32-empty-output.md`） |

---

## 6. MoE 量化的特有坑

这些坑散在案例库里，这里系统整理。

### 6.1 路由 kernel 精度（某医疗MoE模型）

**现象**：输出乱码，但 router_logits 是对的。
**根因**：vLLM 的 fused gate kernel（`ops.moe_fused_gate`）在 `n_group=8, topk_group=4, sigmoid+expert_bias` 参数下选错专家（0/32 匹配）。
**教训**：MoE 精度问题**先查路由 topk_ids**，路由是 0/1 的，错了就是全错。
**修复**：`VLLM_USE_LIGHTOP=1` 换正确 kernel，或 `VLLM_ENABLE_MOE_FUSED_GATE=0` 走纯 Python。
**详见**：[`example/antangelmed-moe-garbled.md`](../example/antangelmed-moe-garbled.md)

### 6.2 zero_point reshape 顺序（MiniMax-M3）

**现象**：模型能跑但输出错误（"sky is the color of water"）。
**根因**：AWQ 非对称量化的 zero_point 解包时，`[N//8, K_groups, 8]` 做 C-order reshape 把 nibble 维和 K_groups 维混淆了。
**修复**：先 `permute(0,2,1).contiguous()` 让 nibble 和 N 合并，再 reshape。
**教训**：多维 tensor 的 reshape/permute 必须验证元素映射，"部分正确"≠"无 bug"。
**详见**：[`example/minimax-m3-awq-int4.md`](../example/minimax-m3-awq-int4.md)

### 6.3 symmetric 默认值（MiniMax-M3）

**现象**：所有层 attention 的 `weight_zero_point` 无法加载。
**根因**：创建量化 scheme 时没传 `symmetric` 参数，默认 True，导致非对称权重的 zero_point 不注册。
**教训**：量化配置的默认值要显式核对，尤其 symmetric/asymmetric。
**详见**：[`example/minimax-m3-awq-int4.md`](../example/minimax-m3-awq-int4.md)

### 6.4 dense/MoE 层判断（MiniMax-M3）

**现象**：dense 层（L0-L2）被误建成 MoE，有 128 个空 expert。
**根因**：`moe_layer_freq=None` 时默认全 MoE，`mlp_layer_types` 字段没被使用。
**教训**：混合架构（部分 dense + 部分 MoE）要显式指定每层类型。
**详见**：[`example/minimax-m3-awq-int4.md`](../example/minimax-m3-awq-int4.md)

### 6.5 Marlin 布局重排

**现象**：Marlin kernel 加载权重慢。
**根因**：Marlin 要求权重是 interleaved 布局（和原始 GPTQ/AWQ 布局不同），加载时要重排。
**教训**：首次加载慢是正常的，重排后缓存。

### 6.6 专家负载不均

**现象**：EP 下某些卡成为瓶颈。
**根因**：路由不均，热专家集中。
**修复**：EPLB（专家负载均衡）/ 冗余专家 / expert_bias。
**详见**：[`vllm_fused_moe_guide.md`](vllm_fused_moe_guide.md) 第 11 章

---

## 7. 真实模型速查表

| 模型 | 总参数 | 激活参数 | 专家数/层 | top_k | hidden | intermediate | 单专家(bf16) | MoE 层数 | MoE 权重占比 |
|------|--------|---------|----------|-------|--------|-------------|-------------|---------|-------------|
| DeepSeek-V3 | 671B | 37B | 256 | 8 | 7168 | 2048 | 44MB(fp8) | 58 | ~95% |
| MiniMax-M3 | 456B | ~14B | 128 | 4 | 8192 | 2816 | ~55MB | 57 | ~93% |
| 某医疗MoE模型 | ~195GB | ~6GB/step | 256 | 8 | 4096 | 1024 | 24.6MB | 31 | ~90% |
| Qwen3.5-35B-A3B | 35B | 3B | 128 | 8 | 2048 | 768 | 9.4MB | 40+ | ~85% |

> 激活参数 = top_k × 单专家 × MoE 层数（近似，忽略 attention/embedding）。
> MoE 权重占比 = MoE 层权重 / 总权重（attention+embedding 占比小）。

---

## 8. 常见问题

### Q1: MoE 模型为什么 decode 慢？

memory-bound。每步读 top_k 专家权重（见 2.3），瓶颈是 HBM 带宽不是算力。优化方向：量化（减少读取字节）+ kernel 调优（提高带宽利用率）。

### Q2: MoE 和 dense 模型显存构成有什么不同？

MoE 显存 90%+ 是专家权重（存了全部 N 个但只算 top_k 个）；dense 显存大头是"每层都算的 FFN"。MoE 的 KV cache 和同 hidden 的 dense 一样大（attention 是 dense 的）。

### Q3: W4A16 和 W8A8 怎么选？

显存瓶颈选 W4A16（1/4 压缩）；算力瓶颈 + 硬件支持 8bit GEMM 选 W8A8/FP8（省 dequant）。MoE decode 几乎总是 memory-bound，所以 W4A16 通常是首选。

### Q4: 为什么 router 不量化？

router 权重对精度敏感，且路由是 0/1 决策——量化误差可能导致选错专家，后果是结构性的（整段输出错），不是精度漂移。所以量化配置里 router 层通常进 `ignored_layers`。

### Q5: MoE 精度问题怎么排查？

先查路由（topk_ids 对不对，对照 HF/纯 Python 参考），再查专家 FFN（dequant 对不对，逐元素对比 ground truth）。路由错是 0/1 的，专家 FFN 错是精度漂移——两者症状和修法完全不同。

### Q6: EP 切分后单卡显存怎么算？

每卡存 `num_experts/ep_size` 个专家。单卡 MoE 权重 = (num_experts/ep_size) × 单专家 × MoE 层数。见 2.5。

---

## 关联文档

- [`vllm_fused_moe_guide.md`](vllm_fused_moe_guide.md) — MoE 实现视角（FusedMoE 层、router、kernel、EP、EPLB）
- [`vllm_quantization_guide.md`](vllm_quantization_guide.md) — 量化实现视角（QuantizationConfig、kernel 选择、WNA16）
- [`vllm_kvcache_guide.md`](vllm_kvcache_guide.md) — KV cache（MoE 的 KV 和 dense 同量级）
- [`example/antangelmed-moe-garbled.md`](../example/antangelmed-moe-garbled.md) — 路由 kernel 精度案例
- [`example/minimax-m3-awq-int4.md`](../example/minimax-m3-awq-int4.md) — AWQ INT4 量化适配案例
- [`example/gemma-spec-decode-aiter.md`](../example/gemma-spec-decode-aiter.md) — W4A16 + aiter 加速案例
- [`example/zhejiang-gaoxin-qwen35-2b-perf.md`](../example/zhejiang-gaoxin-qwen35-2b-perf.md) — MoE decode memory-bound 案例
