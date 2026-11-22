# 算子优化引擎 — 分层与决策路径

> 读完这份，你应该能在 5 分钟内说清：**一个算子从「不知道从哪下手」到
> 「稳定更快且证据可查」，要走哪几步、每步谁负责、哪一步会被脚本卡住。**
>
> 本文只写决策路径与硬约束。方法细节在 `Plugins/ops/skills/`，库画像在
> `op-stack-design/references/libraries.md`，本文不重复任何一条。

## 一、四层，只有一条路

```
L0 认算子   算子名 + 形状  →  一张卡 + 图谱条目        产物：口径一行
L1 选型     谁在这个档位结构上赢 + 流水线怎么排        产物：选谁 + 被否掉的结构
L2 闭环     假设 → 改 → 测 → 判 → 记账 → 下一轮       产物：ACCEPTED / REFUTED
L3 证据     口径 / 显著性 / 正确性 / 记账              产物：可被第三方复核的数字
```

L3 不是最后一步，是**每层都要调用的服务**。

| 层 | 入口 | 回答什么 | 不做 |
|---|---|---|---|
| L0 | `op-atlas/INDEX.md` | 这个算子是什么、谁实现过、上次结论 | 不做技术对比 |
| L1 | `op-stack-design` | 用哪个库、流水线怎么排、哪些设计编译前就该否 | 不做单次调优 |
| L2 | `op-optimize` + `loop.py` | 下一个改动是什么、要不要重跑、怎么接着干 | 不定义判定口径 |
| L3 | `opbench` / `opcheck` / `pmc` / `probe_*` | 数字可不可信、对不对、有没有到顶 | 不做编排 |

**找不到入口时**：先问「我现在缺的是知识、是选型、还是下一个实验」。
缺知识 → L0；缺方向 → L1；缺下一个改动 → L2。别跳层。

## 二、三条硬门（薄约束的全部）

约束**只在你要主张一个结论时生效，不在你要开始探索时生效**。

| 门 | 封什么 | 实现在哪 |
|---|---|---|
| 口径门 | 分母漂移（本项目最贵的一类错） | `claim_gate` 的 `caliber_incomplete` |
| 显著性门 | 「跑两次看着差不多」 | `claim_gate` 的 `not_significant`，`3×max(MAD)` |
| 正确性门 | 「快了但算错了」 | `claim_gate` 的 `accuracy_missing` |

调用约定：`run` **永不拒绝**（`--no-acc` 也能跑）；`report` / `verify` / `adopt`
才看门。判定由脚本算，不由模型说。

> 为什么只有三条：唯一的失效模式是**模型相信自己的一次错误测量**。
> 三条之外全是效率问题，不值得卡。

## 三、方向从哪来（不要凭空想）

`op-optimize/references/surfaces.yaml` 预枚举了 11 条攻击面。
`direction` 只能从这张表里挑：

```bash
python3 loop.py -C <卡> enumerate        # OPEN / OPEN* / QUEUED / SEALED
python3 loop.py -C <卡> enumerate --new  # 出可粘的 candidates.yaml 片段
```

每条攻击面挂一条 `pipeline-laws.md` 的定律，判死时要能追溯到机制，
不许写「试过，慢」。`SEALED` = 该方向已证伪 2 次，开跑前就能看见。

## 四、图谱是索引，不是副本

`op-atlas/cards/*.yaml` 只回答四个问题：算什么 / 谁结构上赢 / 上限在哪 / 上次结论。
**技术描述一律指向别处**（`libraries.md §N`、`pipeline-laws.md LN`、kb 台账）。

这不是纪律，是结构：卡片 schema 用字段白名单锁死，没有字段能放技术描述。
校验：

```bash
python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas
```

E1 cards↔INDEX 双向 / E2 §号可解析 / E3 L号可解析 / E4 status 与 registry 对齐 /
E5 ceiling.ref 可解析 / E6 字段白名单 / E7 结论红线 / E8 家族文件齐备。

E7 最该记：结论里不许出现裸绝对耗时（`0.887 ms`）或无分母百分比（`88.8%`）。
**「达 88.8% 峰值」会被判违规**——「峰值」不是分母，要写「峰值带宽的 0.888」。

## 五、一条命令验全套

```bash
python3 sanitize.py --selftest && python3 sanitize.py . \
  && python3 scripts/run_evals.py --root . \
  && python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas \
  && (cd Plugins/ops/tests && python3 -m unittest discover -s . -p "test_*.py")
```

CI 跑的是同一套（`.github/workflows/ci.yml`），外加 `skill_lint.py --baseline`
只拦新增违规。

## 六、加一个新算子要做什么

1. `op-atlas/cards/<name>.yaml` 建卡 + `INDEX.md` 加一行 → 跑 validate
2. `Plugins/ops/skills/op-<name>/` 建五件套（见 `ops/SKILL.md`）
3. `data/ops/registry.json` 登记
4. 进闭环前先 `enumerate` 挑方向，别凭空想

**不需要**改任何代码。加一个算子 = 加一张卡 + 一行索引。

## 七、Engine 层：P / D 两套配置边界

> 前面六节讲的是**单算子**级优化（一个 kernel、一个场景、一个 ratio）。
> 这一节讲 **pipeline** 级——五段下发收进一个算子时该怎么配。
>
> 数据来源：`Plugins/ops/skills/op-optimize/cases/002-moe-pipeline/`，
> 全部可复跑。**这里的结论只属于 engine 层，不进任何单算子卡的 last_verdict。**

### 7.1 为什么必须分 P / D

MoE 的 pipeline 在两种形态下瓶颈完全不同：

| 档 | 触发条件 | 五段耗时占比（实测，M/exp） | 优化重心 |
|---|---|---|---|
| **P**（prefill） | M/exp ≥ 64 | 两个 GEMM 占 89.2%（M=4096 档） | 融合**不划算**，直接跑 GEMM |
| **D**（decode） | M/exp ≤ 32 | 两个 GEMM 占 95.9%（M=32 档） | 部分融合**划算** |

### 7.2 融合策略的边界（可执行）

实测三级阶梯（`stage2b_ladder.py`）：

| M/exp | L0 全拆分 | L1 部分融合 | L1/L0 | 选 |
|---|---|---|---|---|
| 16 | 0.3427 | **0.2014** | 0.59x | L1 |
| 32 | 0.3376 | **0.2782** | 0.82x | L1 |
| 64 | **0.3661** | 0.4534 | 1.24x | L0 |
| 128 | **0.4088** | 0.8342 | 2.04x | L0 |
| 256 | **0.4504** | 1.6758 | 3.72x | L0 |

```
M/exp < 32        ⇒ L1 部分融合（fuse_groupgemm_permute_quant）
M/exp ∈ [32, 64]  ⇒ 实测决定（两者接近，建议在两个端点各测一次）
M/exp > 64        ⇒ L0 全拆分（permute → grouped GEMM → swiglu）
L2 全融合         ⇒ 不可用（asm_fmoe_a8 在这一档拒所有 dtype/mode，见 failure-modes）
```

**机制**：L1 省的是 permute 与 quant 的中间落盘。这部分开销在小 M 时
占比高（因为 GEMM 本身小），大 M 时被 GEMM 摊薄。

### 7.3 并发：哪些能叠、哪些不能

实测（`stage3_overlap.py` / `stage4_shared.py`）：

| 组合 | 重叠度 | 能不能叠 |
|---|---|---|
| GEMM + GEMM（含共享专家 vs 路由专家） | **0-5.8%** | **不能**——同属算力引擎 |
| `copy_`(D2D) + GEMM | **11.9%** | **不能**——D2D 也在算力队列上 |
| **D2H(pinned DMA) + GEMM** | **100%** | **能**——独立 DMA 引擎 |

**结论**：
- 「把两段 GEMM 放两个流」**在本平台无效**
- 「把共享专家挪到独立 stream」**无效**（MegaMoE 的这条改进不成立）
- 只有「计算 vs 真 DMA」能叠——对应 HiCache KV 备份、权重预取这类 D2H/H2D

### 7.4 已知走不通的路（engine 级）

| 路 | 为什么 | 出处 |
|---|---|---|
| 全融合（asm_fmoe_a8） | 110 个 solution 全部被拒；bf16/fp8/cu 都不接受 | `failure-modes.yaml` 第 1/2 条 |
| 双流叠两段 GEMM | 同引擎争用，实测 5.8% | `stage3-overlap.json` |
| 共享专家独立流 | 同引擎，实测 0.0% | `stage4-shared.json` |
| `ll` 低延迟路径 | 需要 6D packed weight，deepgemm 无公开 pack 函数 | `stage1-gemm.json` |

### 7.5 落点纪律

**engine 级结论只落这里和 `failure-modes.yaml`，不进单算子卡的 `last_verdict`。**
理由：engine 结论受并行策略、显存水位、预热策略共同影响，
写进算子卡会污染算子层判断——这正是本仓库反复强调的「口径打架」。
