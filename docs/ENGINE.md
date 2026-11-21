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
