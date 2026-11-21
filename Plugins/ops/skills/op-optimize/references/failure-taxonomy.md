# 失败分两类，别混（混了就等于白丢一条路）

一次实验失败，归因只有两种，**命令不同、后果不同**：

| 什么时候用 | 命令 | 方向预算 | 上限 |
|---|---|---|---|
| **实现写坏了**：编译不过 / 布局推断拒绝 / 锚点没落上 / 算错 / 崩。这次没跑对，说明不了这条攻击面不行 | `impl-broken <tag> --why "..."` | **不动** | 连挂 3 次 ⇒ 该实现判死（进 `refuted`，`kind: impl`） |
| **攻击面不成立**：跑通了，但结论是这条路不行，且 `why` 写清了会连带杀掉哪一族 | `direction-refuted <tag> --why "..."` | +1 | 2 次 ⇒ 封方向，`run` 拒绝再受理该方向的任何候选 |
| 结构上直接否掉一条路、手上没有具体实现 | `direction-refuted --direction <id> --why "..."` | +1 | 同上 |

**为什么要拆。** 把实现的 bug 记成方向证伪，等于用一个实现错误处死一整条路。
本项目 27 条实现级判死里没有一条证伪了攻击面本身（每条都是自己的机制挂掉：溢出 / LDS 超限 /
布局拒绝 / 就是慢），所以 `directions` 里三个方向的 `refutes` 全是 0 —— **不是没试过，是没试死过方向**。

`loop.py` 强制三件事（都不是「希望模型记住」）：

1. `pending` 候选要声明 `direction`；没声明 / 拼错 / 挂在已封方向上 ⇒ **`run` 直接拒绝**（rc=2），
   因为「2 次封方向」对它是空转，规则等于没有；
2. 已判死的 tag `run` 拒跑；真要重跑必须 `--force`，并在记录里写清为什么这次不一样；
3. `queue` 有任何预算违规就 rc=2 —— PreToolUse hook 直接拿它当门禁用。

schema（三段；已判死只有 `refuted` 一个落点）：

```yaml
directions:
  - id: raise-heads-per-wave
    statement: "抬 heads/wave 到 16，稀释「每 wave 重读整块 KV」"
    status: open          # open | exhausted（到 2 次证伪由脚本改）
    refutes: 0

candidates:
  - id: bm64_vgpr_squeeze
    direction: raise-heads-per-wave   # pending 必填，否则 run 拒跑
    status: pending                   # pending | adopted | verified（终结态由 verify 裁决）
    priority: 1
    expect: "判据三条同时成立：scratch==0 且 heads/wave==16 且 DRAM≤500 MB"
    done_when: 'python3 -I scripts/opcheck.py correctness --contract "$OPS_CARD/contract.yaml" --shape decode-glm53 --variants hip-glm'
                                      # adopted 必填：完成判定的外部裁决命令（verify 跑它，rc=0 才 adopted）
    fails:                            # impl-broken 追加；连挂 3 次判死
      - "2026-10-28 编译不过：layout 冲突"

refuted:                              # 别重开；why 写机制
  - id: merge
    kind: impl            # impl（实现判死）| direction（方向证伪）| claim（历史诊断类）
    direction: fuse-kv-splits
    why: "慢 2.12×，VGPR 256 / spill 30。4 路切分是寄存器生命周期分段：4 个小 buffer 顺序复用同一批 VGPR，合成 1 个大 buffer 后生命周期全程重叠，顶爆 256 上限。"
    on: 2026-10-27
```

> **计数的权威点在 yaml，不在 `runs/`**：`refutes` / `fails` 存在 `candidates.yaml`，`runs/loop/*.jsonl` 只是事件流水——`**/runs/` 在 `.gitignore` 里，计数若从流水推导，删一次 `runs/` 就等于把方向预算清零、把死马全部复活。

**`why` 写机制，不只写结论。**「merge 慢 2.12×」只挡住 merge 这一个变体；「合并 buffer ⇒ 生命周期重叠 ⇒ 溢出」能挡住后面所有同类想法（`kv4`、`g2s` 都是同一个死法，提前识破能省三轮）。老命令 `refute` 已拆掉（只剩 rc=2 迁移提示），它原本把「实现写坏」和「方向不成立」记成同一件事，正是本节要消掉的歧义。
