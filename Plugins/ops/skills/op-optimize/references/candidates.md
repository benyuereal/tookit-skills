# 候选从哪来（按可信度排序）

1. **实测证据**：PMC / mem-only / 消融已经指出某个单元或某段代码 —— 最高优先级。
2. **结构推理**：从寄存器预算、LDS 容量、带宽模型算出来的可行性边界。算出来「不可能」的要立刻记进证伪清单（例：`Q(128 VGPR) + acc_o(128 VGPR) = 256` ⇒ bm64/thr256 必然溢出，与写法无关）。
3. **外部经验**：别人实现里的写法。**最低优先级** —— 本机 ISA 可能根本没有那条指令（实测 gfx938 没有 `ds_read_b64_tr_b16`，而在 gfx90a/gfx950 上有）。

## 引擎级候选：先按 trace kernel 名查表，不要从 tile / occupancy 起手

有些改动不在单个 kernel 内，而在**引擎怎么选 kernel、怎么摆布局**。它们的候选不该
从 `surfaces.yaml` 的通用攻击面里猜，而应由 trace 里的 kernel 名直接触发：

| trace 信号（可 grep 的 kernel 名） | pattern | 先做什么 |
|---|---|---|
| `mqa_logits_*`（prefill）/ `paged_mqa_logits`（decode）占该桶 ≥25%，且输出只用于 top-k 选择 | `indexer-score-precision-drop` | 打分 BF16 → INT8（`lightop::mqa_int8::m64` / `paged_mqa_logits_int8`）；附带把 gather+topk-transform 融成 `kpool_topk_kernel` |
| decode 同时出现 `decode_gather_and_up_convert*` 与 `flash_mla_dsa_decode_*`，时间接近、带宽都吃不满 | `sparse-mla-fp8-direct` | 给 `sparse_fp8` 加 rope-less `NOPE512`（行宽 528 B）直接吃 fp8 KV，删掉中间 gather |

**完整可执行配方**（信号阈值、逐条 where、验收、回退、坑）在
`../../op-source-patch/references/engine-recipes.yaml`，执行走 `op-source-patch` 六步。
两条都属于 `optimization-rules.yaml` 的结构变换（`precision_path` / `layout_pack` /
`fusion_boundary`），不是调参；候选里不要写成「调 tile 试试」。

两条最容易把结论带偏的前置判断：

- **先算 HBM 通过次数，再谈 launch 数。** 把 gather 与 up-convert 融成一个 kernel 只少一次
  launch，不改变「fp8 读 → bf16 写 → bf16 读」三遍 HBM 的本质；删掉中转才是收益来源。
- **「换了 dtype 反而慢」先手算「行宽 × M-block 数」。** `sparse_fp8` 的 `BLOCK_M=16` 让每个
  M-block 独立 CTA 各读一遍 topk KV；h_q=64 时 NMB=4，字节比 bf16 还多，慢是分块重复读，
  不是 dtype。h16（NMB=1）才反转。

## 反直觉清单（本项目实测，可直接迁移）

- 「把小 tile 合成大 tile 更高效」→ 错。4 路切分是**寄存器生命周期分段**；合并后生命周期全程重叠，直接顶爆 VGPR 上限（spill 30，慢 2.12×）。
- 「减少指令数就能提速」→ 错。删掉最大的一块 VALU（每块每线程 32 条 mul）只换回 1.1%。
- 「某段指令占比 N% ⇒ 删掉省 N%」→ 错。它可能已经被重叠吃掉了（那一处删掉反而慢 1.1%）。
- 「把寄存器数组搬到 shared 换 occupancy」→ 通常错。LDS 流量正是另一侧要砍的东西，两条方向互相打架。
- 「工具算出来的 occupancy 是可信的」→ 先反验常数。`VF_LIMIT=65536` 让 2 CTA/CU 显示成 1，据此下的「无 CTA 可切」结论整条是错的。**判据：常数若推出物理不可能（0 个 CTA、占用率 >100%），先判常数错，不判数据错。**
- 「状态标了 adopted 就等于可用」→ 错。代码可能在后续清理里被删掉了（清理实验脚手架时把某个已 adopted 变体的实现一起带走，状态却一直停在 adopted，下一次以为它能用）。**判据：状态要能被机器验证——文件在不在、跑不跑得起来；验证不了就降级成 `recorded`。**
- 「历史数字可以跨会话比」→ 错。同一 kernel 两次会话差 10%（flash_mla 0.6454 vs 0.7193）。**只比同一张表内的比值，基线每次现测。**
- 「低占用的收益可以按比例外推」→ 错。同一 kernel 1→2 CTA/CU 是 68%，换一个 kernel 只有 27.5%；外推只能当上界。

## 隔离一个变量

比「多试几次」有用得多：改一个**不改变语义**的量，让输出 bit-identical，只留一个变量在动。例：动态 LDS 申请量 28672→40960（kernel 只读前 28672）强制 1 CTA/CU，就单独量出了 occupancy 的敏感度。同类：改 launch 配置不改 kernel 逻辑、改分配不改算法。

## 判死的写法

只写「试过，慢」的条目等于没写——下一个模型还会再试一遍。写**机制**（为什么慢、被什么挡住、在什么条件下会反转），机制能迁移到别的算子，结论不能。
