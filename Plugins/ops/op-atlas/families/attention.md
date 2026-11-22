# attention

Q/K/V 参与计算。瓶颈通常在 **KV 搬运与片上归约**，不在 GEMM 主体。

## 本族特有的判据

1. **先分清 dense / paged / sparse 三种语义**——它们的 KV 访问模式完全不同，
   tuned 参数不可跨语义复用。sparse 的索引来自块生成器，dense 的没有。
2. **logits 输出可能比 K 本身还大**（fp32 时 S·SKV·4 vs SKV·D·1），
   设计上必须融合 topk，否则写出量压过读入量。
3. **head_dim 是硬契约**：lightop 的 paged decode 只接受 128；
   tilelang 侧接受 512/576。跨库对比前先确认 D 一致，否则口径不成立。
4. **归约写法会成为主瓶颈**：稀疏 MLA 与 MQA 都实测过，epilogue 的
   跨 lane 归约占 VALU 的大头（MQA：85% 的非 MMAC VALU）。

## 本族攻击面（跨族通用的见 surfaces.yaml）

- `cross-lane-fold`：跨 lane 归约是否用了更长指令序列（详见实测记录）
- `gather-path`：KV gather 是否在关键路径且串行
- `fold-nextn`：投机 token 能否并进 N 维（decode 档唯一的真杠杆）

## 已知的结构性陷阱

| 陷阱 | 机制 | 出处 |
|---|---|---|
| 2-stage 双缓冲 | 把 occupancy 16→8 waves，实测 0.65x | sparse-mla |
| LDS bank conflict | 冲突可归零但被流水完全隐藏，墙钟仅 0.5% | mqa-logits |
| 融合大算力主体 | 只对 memory-bound 链成立，否则净负 | libraries.md §3 |

## 库选择

dense/paged → `libraries.md §7`（flash-attn 系）
稀疏 MLA → `libraries.md §8`（TileLang）

## 实现横评

| 实现 | 层级 | 适合什么 | 不适合什么 | 证据 |
|---|---|---|---|---|
| `flash-attn 系` | kernel | dense / paged attention，瓶颈在 KV 搬运 | 需要新的 attention 变体语义（改 fork 成本高于重写） | `libraries.md §7` |
| `FlashMLA` | kernel | MLA 的 prefill/decode，含稀疏 MLA | 非 MLA 变体；`mutates_inputs`（原地规范化 indices） | `libraries.md §7` |
| `TileLang` | kernel | 稀疏 MLA，形状可表达 | 要逼近手写极限（先付 3-4× 搬运指令税）；bf16 + MLS | `libraries.md §8` |
| `BoltOPs` | kernel | decode 形状的 attention 成品，自带 autotune | 大 M prefill（Triton 路线固有税） | `libraries.md §10` |
| `MegaMoE 式引擎并行` | **engine** | 整模型部署的并行策略 | **不属算子层，结论不进卡片** | 方法见 `docs/ENGINE.md` |
