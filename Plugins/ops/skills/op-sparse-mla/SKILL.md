---
name: op-sparse-mla
description: "稀疏 MLA 前向（DeepSeek Sparse Attention / DSA）：q[S,H,576] × kv[SKV,1,576] 按 indices[S,1,topk] 稀疏 gather 后做 MLA，输出[S,H,512]。覆盖四类实现（生产 vLLM aiter、tilelang-hygon、flash_mla、fp32 参考）的适用面、形状档位、-1 哨兵语义、sm_scale=1/sqrt(640) 口径与已知坑。触发场景：稀疏注意力 kernel 选型、topk 相关性能问题、indexer 链路优化、d_v=512 MLA 精度对齐、-1 哨兵布局/num_split 整除排查、跨实现站位的索引池口径。不适用：A/B 的噪声与统计口径（3×MAD、跨卡 ratio）走 op-bench；手写 HIP 的编译 flag 与 VGPR 读法走 op-hip-compile；上界预测与绑定约束走 op-roofline。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
---

# 稀疏 MLA 前向（op-sparse-mla）

## 一、是什么

DSA（DeepSeek Sparse Attention）里的稀疏注意力前向。每个 query 只在 top-k 个历史 token 上做 MLA：

```
score[i,h,j] = q[i,h,:] · kv[indices[i,j],:] * sm_scale      j ∈ [0, topk)
out[i,h,:]   = softmax_j(score[i,h,j]) · kv[indices[i,j], :512]
```

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| `q` | `[S, H, 576]` | bf16 | 前 512 = latent，后 64 = rope tail |
| `kv` | `[SKV, kv_group, 576]` | bf16 | `kv_group` 通常为 1；同一张量既是 K 也是 V：前 512 维既参与打分也参与输出，后 64 维 rope tail **只参与打分** |
| `indices` | `[S, kv_group, topk]` | int32 | **`-1` = 无效**；`>= SKV` 亦视为无效（flash_mla 口径） |
| `out` | `[S, H, 512]` | bf16 | 只输出 latent 部分，**不输出 64 维 tail** |
| `lse` | `[S, H]` | fp32/fp16 | log-sum-exp，供 combine 阶段用 |

**哨兵语义是正确性高发区**：`-1` 表示该 top-k 槽位没有选中任何 token，必须在 softmax 前被掩成 `-inf`，**不能**当成索引 0。padding 比例在生产里并不低（短上下文时大量槽位是 `-1`），所以 `effective_topk = mean(indices >= 0) * topk` 必须单独统计。

`sm_scale` 按 MLA 惯例取 `1/sqrt(576 + 64)`（= `0.039528...`），**不是** `1/sqrt(576)`。两边实现都接受显式传入，比较时必须传同一个值。
**GLM5.3 是吸收式 MLA：`q`/`kv` 只有 512 宽、没有 rope 尾（`tail_dim=0`，`d_qk == d_v == 512`），对应 `sm_scale = 1/sqrt(512)`（= `0.044194...`）—— 与 576 形状不是同一个值。** tilelang 变体现已支持该形状（见 §五）。

## 二、变体矩阵

详见 `variants.yaml`。摘要：

| 变体 | 来源 | 适用 | 状态 |
|---|---|---|---|
| `prod-vllm` | vLLM 生产路径 `sparse_attn_fwd`（aiter AOT ASM，无源码） | 生产 decode/长上下文 | 现役 |
| `tilelang` | tilelang-hygon `perf/sparse_mla/sparse_mla_fwd.py`（`tilelang_sparse_fwd`） | 通用、可改可调、支持 SplitKV 自动配置；**GLM5.3 `tail_dim=0`（`d_qk=d_v=512`）已支持，576 路径逐位不变** | 可用 |
| `tilelang-prefetch` | 由 `bench/prefetch.py` 对 `tilelang` 做源码改写（下一块 gather 提前） | 长上下文；比原版快 3~6%，**逐位一致** | 可用 |
| `tilelang-example` | `examples/deepseek_v32/`、`examples/dsa_sparse_finetune/` | 同上，但为示例代码，配置不自动 | 参考 |
| `flash_mla` | `flash_mla.flash_mla_sparse_fwd` v1.2.0+dtk2604 | 对标基线；仅 `d_v=512`、bf16 | 可用 |
| `hip-glm` | 手写 HIP 版 GLM5.3 档 `dev/sparse_mla_hip_glm.hip`（body 同 tilelang 生成体逐字搬入；split 分派 S≤16→ns4/17-35→ns2/≥36→nosplit + combine） | GLM5.3（H=64/d_qk=512/tail=0）：320k 实测（2026-11-13，含 maxn32）：S≤8 领先 flash_mla 3.4~3.6×、S=16~64 领先 2.9~1.09×、S=128 落后 1.07×（结构天花板，见 `candidates.yaml: large_s_ceiling_structure`）；全程快于 tilelang 0~21%（[历史数据]） | 可用 |
| `ref-fp32` | torah fp32 参考 `ref_sparse_mla_fwd_interface` | 只做正确性判定，**不入性能对比** | 参考 |

### 按场景选实现（不预设语言/框架，2026-10-29 全档位实测沉淀）

tilelang / flash_mla / 手写 HIP 各有领地，按 (H, S, ctx) 选：

| 场景 (H, S, ctx) | 首选 | 站位（vs flash_mla，[历史数据]） | 理由 |
|---|---|---|---|
| GLM H=64，S≤32，320k/128k | **`hip-glm`** | S=16 领先 2.88× / S=32 领先 1.57× | split 路径（ns4/ns2）把 2048 行 KV 拆满 70 CU；flash_mla 小 S 有 ~0.38ms 固定地板 |
| GLM H=64，S≥64（含 128/256/512），320k | **`flash_mla`** | hip-glm 落后 1.07×（2026-11-13 含 maxn32；10-29 无 maxn32 时是 1.13~1.19×） | flash_mla 有跨 head KV 复用（H16→64 工作量 4× 耗时仅 1.6×）；hip-glm/tilelang 同墙（REPLICATE_H=2、已确认 2 CTA/CU 非 1），body 内调参已穷尽（39 条证伪见 candidates.yaml）。反超需跨 head KV 复用或换 tiling，未立项 |
| GLM H=64，S≥64，128k 短 ctx | `flash_mla` 或 `tilelang` | 交叉点 128k 附近：fm 128k→320k 慢 53%、tl 平坦（ledger §二 14） | 短 ctx 下 fm 优势收窄，两者差距 <15% |
| **V4.1 Flash** H=64，d_qk=576/topk=512 | **无覆盖**（`decode-v41` 已立项） | — | §三 的生产形状核对；改造路径见 ledger「待立项」 |
| 生产链路（vLLM 现役） | `prod-vllm`（aiter AOT ASM） | — | 无源码但现役；上表是它之外的选型空间 |
| 需定制（改布局/融合/新哨兵语义/脱离 tilelang 栈） | **手写 HIP**（hip-glm 为模板） | 可达 tilelang 同速（GLM 实测反超 1~3.5%） | body 可逐字搬 tilelang 生成体保正确性，编译 flag 全镜像后同速；命令与排查方法见 §六 |

> 测试档（H=16，无生产模型）的旧结论已从选型表移除：手写 HIP 在 320k 上领先 flash_mla
> 1.13/1.64/1.65×（S=128/256/512）、`tilelang-prefetch` 领先 1.28~1.95×。两者都只作方法参考——
> 尤其 prefetch 需按 `head_kv<=16` 条件启用，H=64 档会 **+21%**，未落生产树。

**选型两条铁律**：
1. **H 决定结构，S 决定路径**：H=16 瓶颈是 gather 效率、H=64 是跨 head 复用缺失，跨 H 结论不可迁移；S 决定 split vs no-split，跨 S 比较先记 S。
2. **站位数字必须带五个口径标记**（H/S/ctx/pool/sm_scale）才复用——同 kernel 在 128k 与 320k 的结论可以相反（真实发生过，ledger §二 17/18）。

## 三、形状档位

全部档位在 `shapes.yaml` 里，**一档一个顶层键**（`load_shape("decode-s128")` 直接取）。
每个档位自带 `params` / `caliber`（这次测的是单次 op_call 还是整步）/ `focus` / `notes`。

| 档位 | 特征 | 关注点 |
|---|---|---|
| `decode-s128` | S=128，64k 上下文，**H=16（测试形状）** | **跨实现对比主档**；⚠️ `pool: window` 会让 flash_mla 免付内存费，比站位要用 `bench_long.py`（`pool=full`） |
| `decode-m1` | S=1~8，context 极长，**H=16（测试形状）** | 纯访存；gather 随机性最伤 |
| `decode-m32` | S=32，context 长，**H=16（测试形状）** | 看带宽达成率；⚠ 下游两个生产模型都是 H=64 |
| `prefill-8k` | S=8192 | 算力档；看 TFLOPS 与 tile 利用率 |
| `prefill-320k` | S=4096/chunk，SKV 到 320k | 长上下文 prefill；受 KV 抓取限制 |
| `decode-glm53` | S=32，H=64，SKV=128k，topk=2048，**d=512 无 rope 尾** | GLM5.3 生产 decode；`sm_scale=1/sqrt(512)`，`-1` 只能垫队尾 |
| `prefill-glm53` | S=4096，其余同上 | GLM5.3 prefill；H=64 ⇒ `REPLICATE_H=2`，比 flash_mla 慢 2× 是**片上**复读（L2→L1 流量与指令条数，**不是 DRAM 字节**，见 §四 硬件计数器） |

同一份实现跨档位结论会反转（我们见过同一 kernel 在 4k 与 65k 下差 6 倍），**结论必须挂档位**。

**生产形状核对（2026-10-29 按 `/models` 权重）**：`decode-m*` / `decode-s128` 的 H=16 来自 tilelang
示例默认值（`examples/dsa_sparse_finetune/...:204`），**不是任何生产模型**——两个生产模型都是 H=64：
GLM5.3-Flash（H=64 / d_qk=512 / topk=2048，有档位）与 DeepSeek V4.1 Flash / V4-Flash-0731
（H=64 / d_qk=576=512+64 / topk=512，档位 `decode-v41` 已立项、**尚无实现**）。跨 H 的结论不可迁移（见 §二 铁律 1）。

## 四、判据

见 `contract.yaml`：

- 正确性：vs fp32 参考 `max_abs <= 2e-3`、`cosine >= 0.9999`（bf16 输入）；官方自带测试只要求 `max_diff < 1e-2`，我们更严
- 变体之间：`tolerance` / `max_abs 3e-3`（两个独立 bf16 实现各自离参考 ~1e-3，互相差 2e-3 属正常叠加；要求逐位一致会假失败）
- 不变量：`determinism`（逐位）/ `index_permutation_invariance`（容差）/ `split_invariance`（容差）
- `-1` 布局不是不变量而是**变体能力**：各变体支持「队尾 / 两种」写在 `variants.yaml:padding_support`，由 `opcheck.py padding` 校验（gate `layout_capability`，不符声明即 fail）；同一条检查也校验 `mutates_inputs` 的声明与实测一致
- 性能地板：gather 类，地板按「实际读取的 KV 字节 / 峰值带宽」算，**不要**按 topk 名义值算
- **形状档位必须与宽度绑定**：576 档（`tail_dim=64`）与 512 档（`tail_dim=0`）的 `sm_scale` 不同（`1/sqrt(640)` vs `1/sqrt(512)`），且 512 档比「576 尾部补零」等价法只快 4~5%（`max_abs=0` 逐位一致）—— 用零尾等价法做性能代理可以，误差 <5%，但**不能用来判 1% 级差异**
- 定位瓶颈用 `bench/bench.py` 的三种消融 mode（GEMM 工作量不变，只改一个变量）：`--mode workset` 改 KV 表大小——表进缓存后快多少就是内存占比（计算/带宽判定的决定性实验）；`--mode config` 扫 `block_I/threads/num_stages`；`--mode split` 扫 `num_split`（含 combine，端到端）。
  **判 GEMM 效率只能看「热工作集下的整 kernel」**：把 gather 物化后喂 hipBLASLt 的「GEMM-only」基准自己就带宽受限，小 M 纯 GEMM 又撞 launch 平底，两者都测不出 MMA 利用率。**必须钉空闲卡**：别人正跑着的卡上两个实现同时慢 ~17%，比值还会被压平。
- 判「访存到没到地板」用 **mem-only 变体**（GEMM/softmax 换成从 shared 的廉价累加，gather 与 shared 读写不变）：它到峰值的百分比 = 访存天花板；整 kernel 与它的差额 = 「gather 与计算不重叠」的损失，不是带宽损失（实测 gather 94~97%、整 kernel ~55%）。
- 判 occupancy 直接读编译产物，不要猜：`llvm-readelf --notes <tilelang cache>/kernels/<hash>/executable.so`（`.so` 里第二个 ELF 才是设备码），看 `vgpr_count` —— **每 CU 能驻几个 CTA = min(64K / (vgpr×线程数), 64KB / LDS)**。实测 `vgpr_count=217` × 256 线程 ⇒ 1 CTA/CU（LDS 25.6KB 够 2 个）：等内存时没有任何东西可切换。
- 改 kernel 提重叠用 `bench/prefetch.py`（下一块 gather 提前到本次 GEMM 之前，寄存器当预取缓冲）：快 5~6% 且**逐位一致**。**不要把 spilt3/tail 搬进 shared 一起预取**——它们是寄存器直喂 MMA，搬进 shared 会慢 1.7×（LDS 流水被压满）。
- **判「慢在哪个单元」用硬件计数器**（`bench/pmc_collect.py` 跑，解析照 `op-profiling/SKILL.md §六` 的参考实现自己算；`hipprof --pmc --pmc-type 3`）。
  四个判据，**比值比绝对值稳，且必须用同一形状三方对照（本实现 / 对手实现 / 一个已知高效的 matmul）**：
  - **算力**：`MFMA 条数 = FLOPs / 8192`（bf16 16×16×16；gfx938 实测，其他 gfx 未验证），发射上限 **19.7 条/cycle 全 GPU**（2048³ matmul 标定，反算它自己恰得 88% 占用率，与 TF 实测吻合）。占用率 80%+ 才算被算力卡住；本算子只有 18~38%，**从来不是 GEMM 慢**。
  - **内存**：**DRAM 读字节 = (TCC_HIT+TCC_MISS 的 miss 项)×128B**，要同时看 `TCP_TCP_TA_DATA_STALL`（等数据）与池子敏感性（`--mode workset`）。三项都指向「不缺内存」才算排除。
  - **片上开销**：看两个比值 **`(SQ_INSTS_VALU − MFMA) / MFMA`** 和 **`SQ_INSTS_LDS / MFMA`**（跨形状可比）。调优 matmul 0.23 / 0.25，flash_mla 0.73 / 0.53，本实现 **3.11 / 1.67** ⇒ 差距在「每条 MMA 配多少搬运指令」。
  - **别把 `SQ_LDS_BANK_CONFLICT` 当优化目标**：厂商调优 matmul 自己有 13% 的冲突占用率照样跑到 88% 峰值，本实现只有 10%，比它还低。
  - 坑：①PMC 下**首次 launch 的 GRBM 虚高 ~40%**，要取多次里 GRBM 最小的那行做稳态样本；②自动挑主 kernel **必须排除 torch/rocprim**（`randperm` 内部 kernel 的单次 GRBM 比 flash_mla 主 kernel 还大），也不能按同 kernel 名求和（randperm 有几百次小 launch）。
- **所有旋钮都撞墙时，先怀疑 codegen，别继续调参**。判据：把「搬运指令 / MFMA」与对手比，若差 3× 以上，且 `block_I / block_m / threads / num_stages / num_split / SplitKV / 预取` 全部扫过无效，则瓶颈在编译器生成代码的能力，不在 kernel 参数。典型症状：想「直接用寄存器片段喂 GEMM」时编译器报 `Get different layout for <buffer>` ——两种 GEMM（收缩维不同）对 fragment 布局的要求互斥，跨 lane 换布局在本框架里只能走 shared，于是那笔标量 LDS 是**结构强制**的，转置 tile 只是把标量从读侧搬到写侧（净收益≈0）。出路只有「编译器支持寄存器换布局」或「手写底层 kernel」，继续搜参是白费。
- **跨 wave 归约的 barrier 是按「每线程寄存器个数」付的**。看到 `for (rv = 0; rv < N; ++rv) AllReduce::run(buf[rv], ws)` 就知道这一处要付 N 对 barrier；模板如果带 `run_batch` / `batch=N` / `workspace_stride`，用它把这个循环合成一次交换（`batch` 必须整除寄存器数）。本例 `batch=4` 换来 −4.8% 且**逐位一致**（butterfly 伙伴与运算顺序都没变），是全套优化里唯一「白拿」的一档。**别试图直接删掉模板里的前导 `Barrier::sync()`**——即使生成的 HIP 里调用点前已有 `__syncthreads()`，那个 barrier 保护的是「同一 slot 被下一轮寄存器覆写」的 WAR，删掉会静默算错（本例 2.6M/4.2M 元素改变）。先确认中间有没有跨迭代的 buffer 复用，再谈冗余。

## 五、已知坑

| 坑 | 现象 | 规避 |
|---|---|---|
| `topk % block_I != 0` | 断言失败或越界 | tilelang 版要求 `topk` 是 `block_I`（默认 32）的倍数 |
| `topk % (num_split * block_I) != 0`（**`num_split` 是自动选的**） | 断言失败：`topk=512 must be divisible by num_split*block_I=32*32`。**V4.1 Flash 生产 decode（`index_topk=512`、S=1）就是这样起不来的** | wrapper 现在会把 `num_split`/`num_split_tail` 夹到 `topk/block_I` 的最大因子（32→16）；自己调 `num_split` 时也要满足整除 |
| `is_causal` 只支持 True | 传 False 直接断言 | 因果性由 `indices` 的取值保证，不要靠 kernel 掩 |
| `dim` / `tail_dim` 必须是 2 的幂 | 断言失败 | `dim` 必须是 2 的幂；`tail_dim` **支持 0**（GLM5.3 吸收式 MLA，512 宽度，已验过）或 2 的幂；`dim+tail_dim` 只接受 512 / 576。换 `d_v` 前先查 |
| **改 kernel 加条件分支** | 想支持 `tail_dim=0` 却在 `T.alloc_fragment([BI, 0])` 这种零长分配上炸 | 用 **Python 层 `if D_tail > 0:`**（`D_tail` 是编译期常量）做条件编译，别留零长 fragment；4 个 kernel body 共 20 处 tail 语句都要包 |
| 换了 `S` 或 `H` 会重新 JIT | 首次调用要数十秒，看似 hang | 预热一次再计时；bench 里 warmup 必须 ≥3 次真实调用 |
| tilelang 内核缓存命中会跳过 lowering | 改了源码不生效 | `TILELANG_DISABLE_CACHE=1` |
| `num_split` 由 `get_best_config` 自动选 | 不同 S/H 下切分数不同，性能与数值都可能变 | 结论里必须记 `num_split`；跨配置比较前先固定它 |
| 参考实现一次性建 `[S, SKV]` 分数矩阵 | S=2048/SKV=65536 要 8TB | 参考必须按 query 分块 |
| 用 topk 名义值算带宽 | 短上下文时高估 2~3 倍 | 用 `effective_topk` 折算 |
| packed `[slots, 132]` 布局 | 向量化被限制到 ≤4B，带宽只有分离平面的 ~1/2 | 性能**不可跨布局比较**；生产布局优先 |
| **`-1` 只能垫在队尾（tilelang 变体的隐含前提）** | kernel 用 `valid_NI` 数「块首 ≥ 0 的块数」再据此截断循环（`sparse_mla_fwd.py:308-314`），等价于要求 `-1` 是后缀。前置 `-1` 时：垫 1 个 cos=0.9997，垫 31/64 个 cos≈0.99，全前置**输出恒为 0**；flash_mla 两种布局都正确 | 2026-10-27 实测 gfx938 / tilelang-hygon dev build。厂商自带 `test_sparse_mla_fwd` 与 `benchmark_glm51` 全部用「有效索引在前 + 队尾 `-1`」，所以现网不受影响。**自己拼 indices 时不要产生前导 `-1`**；声明与实测的一致性用 `opcheck.py padding` 校验 |
| 只测 topk 打满的输入会漏掉部分填充路径 | 长上下文下 `effective_topk == topk`，`-1` 相关分支全不执行，正确性门「通过」但没验证过短上下文 | 正确性门跑一次 `--pad-ratio 0.5`（队尾填充，厂商口径） |
| 把参考实现当唯一真相对齐 `sm_scale` | 厂商 `ref_sparse_mla_fwd_interface` 默认 `sm_scale = dim_q**-0.5`（=1/√576），而 kernel 默认 `(dim+tail)**-0.5`（=1/√640）——**两者不等价** | 一律显式传 `1/sqrt(576+64)=0.03952847075210474`；我们与厂商 kernel 用同一值，差异只在参考侧 |
| **多变体共用同一个 `indices`** | `flash_mla.flash_mla_sparse_fwd` 会把 `indices` 规范化后**原地写回**（有效索引前移、`-1` 后移到队尾；`q`/`kv` 不动，重复调用幂等）。先跑的变体改掉输入后，后跑的那个看到的是被改过的布局——我们因此把「tilelang 前置 `-1` 输出恒 0」误判成「两种布局都一致」 | 每个变体喂独立副本（`_common.run_isolated` 已代劳）；该变体事实记在 `variants.yaml:mutates_inputs`，由 `opcheck.py padding` 校验声明与实测一致 |
| **索引池太小 ⇒ 结论反向** | `q_start=4096` 这类档位池只有 4.6 MB，全在 L2：flash_mla 免付内存费，我们把「落后」测成「领先」（真实发生，见 ledger §二 17）。缩池 377MB→2.4MB，fm 快 **1.83×**、tl 只快 **1.03×** | 判据是**池大小**不是参数名：`build_inputs` 输出 `pool_extent`/`pool_mb`，< 64 MB 打 `warning`。跨实现比站位一律 `pool="full"` 或 `q_start_s_index >= SKV`（两者索引逐位相同，已验证），或用 `bench/bench_long.py`（写死 full、不提供开关） |
| **拿「标称 GB/s」当 DRAM 带宽** | 名义读量 ÷ 耗时在 topk 大而池小时能算出**超过物理峰值**的结果（同一行数据被反复读、命中 L2） | 该值只能用来比「同一档不同实现」；当绝对上限引用是错的。真带宽上限看 ledger §二 19 的分档实测（绝对数在 kb） |
| **手写 HIP 照抄 10 条静态 -mllvm flag** | 与 tilelang 同体 kernel 慢 13%——漏了 pass_configs 条件项（FAST_MATH→fast-math flag） | 用 **`op-hip-compile`**：`derive_flags.py` 直接调 tilelang 的 `get_hcu_compile_flags(arch, pass_configs)` 复算，`inspect_so.py` 读 VGPR 复核；三步排查（token diff / readelf .text+VGPR / 逐 flag A/B）见该 skill §六 |
| **把「LDS bank 冲突多」当可优化项（H=64 档）** | 8.72M 次冲突、12% 墙钟在「等 LDS」，但 swizzle/奇数行 pitch 全试了：冲突数 0 变化、墙钟 1.000× | gfx938 上被流水吸收，非串行瓶颈；真区分项是 LDS 指令条数与 VALU 条数（ledger §二 29 + candidates.yaml: lds_pitch_swizzle_handhip） |

## 六、怎么跑

```bash
# 微基准（METRIC 行 + json）；档位文件自带 d_v/tail_dim，sm_scale 按形状自动取
python3 bench/bench.py --shape decode-glm53 --variants tilelang,tilelang-prefetch,flash_mla
# 跨 S × ctx 站位（写死 pool=full；任何「领先/落后 x 倍」只从这里出，不从 window 档出）
python3 bench/bench_long.py --s 1,32,128,4096 --ctx 131072,327680 --h 64 --variants hip-glm,tilelang,flash_mla --tilelang-root <repo>/tilelang-hygon
# 三道门（流程只有一份，判据全读 contract.yaml / variants.yaml）；正确性门默认 --pad-ratio 0.5，不跑它等于 `-1` 路径没验
python3 <repo>/scripts/opcheck.py correctness --contract contract.yaml --shape decode-glm53 --variants tilelang,flash_mla
python3 <repo>/scripts/opcheck.py invariants  --contract contract.yaml --shape decode-m32 --variant tilelang --tilelang-root <repo>/tilelang-hygon
python3 <repo>/scripts/opcheck.py padding     --contract contract.yaml --shape decode-s128
# 推进一轮 + 判定（本卡没有 tag 级 A/B 适配器 ⇒ loop.py run 会明确拒绝；候选用上面命令跑完再记账）
python3 ../op-optimize/loop.py -C . status && python3 ../op-optimize/loop.py -C . queue
python3 ../../../../scripts/opverify.py --contract contract.yaml --run <run目录> --json   # 安装态 scripts/ 在 <skills根>/_dcu-toolkit-shared/scripts/
# PMC（取 GRBM 最小行做稳态；`hip` 分支驱动裸 .so）；解析照 op-profiling/SKILL.md §六
HIP_VISIBLE_DEVICES=<idle> hipprof --pmc --pmc-type 3 -o /tmp/x python3 bench/pmc_collect.py hip dev/libsparse_mla_hip_glm.so 128 2048 327680 2
```

**手写 HIP 编译**：完整配方（`aicc` / 11 条 `-mllvm` / 头文件路径 / `hcu_buffer_load` 包装）在 `dev/COMPILE.md` —— **一台机器一套编译器**，换语言即作废，别往卡里搬。可迁移的只有两条：① **镜像对标实现运行时实际拼出的编译选项，不是抄静态 flag 列表**（tilelang 的 `TL_ENABLE_FAST_MATH` 会多追加一条，漏一条慢 10%+ 而**数值完全一致**，只能逐 flag A/B 才找得到）；② **怀疑 codegen 就读二进制，不读源码猜**（`llvm-readelf --notes <.so>` 看内嵌设备 ELF 的 `.text`/VGPR/SGPR）。通用做法：gather 用硬件向量化读 + 越界钳位（`-1` 自动 0-fill），别在 kernel 里手写分支（标量循环版恒定慢 1.35×/CTA，输出位级一致）。

## 七、深引用

- 手写 HIP 的编译调优（aicc / `-mllvm` 镜像 / VGPR 复核）：**`op-hip-compile`**（`derive_flags.py` / `inspect_so.py`）
- 接口与自动配置：`<repo>/tilelang-hygon/perf/sparse_mla/sparse_mla_fwd.py`（`tilelang_sparse_fwd` / `sparse_mla_fwd_interface` / `get_best_config`）
- flash_mla 接口：`flash_mla.flash_mla_sparse_fwd`（`d_v` 仅 512；`indices` 无效值 `-1` 或 `>= s_kv`）
- 生产链路上下游：mqa_logits（logits）→ `topk` → 本算子 → combine（`num_split` 的归约，若 kernel 内部已做则无独立 kernel）；tilelang 自带 bench 为 `<tilelang-hygon>/perf/sparse_mla/benchmark_sparse_mla.py`。mqa_logits **没有独立算子卡**，
  它的机器模型在 `op-roofline/models/mqa_logits.py`（`derive.py mqa-logits`），
  历史实测与口径矛盾见 `~/.config/dcu-toolkit/kb/op-opt/mqa-logits-model-20261027.md`；indexer 侧见 `dcu-operators/dcu-hipdsa-source`
- 历史结论与绝对数字：`~/.config/dcu-toolkit/kb/op-opt/`（不在本卡）
