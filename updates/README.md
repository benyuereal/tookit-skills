# updates —— 进展与交接记录

> 两个目录，各管一件事：
> - `handoff/` —— 交接文档（进来先读这个）
> - `engineering-iteration/` —— 工程迭代记录（按时间归档，讲「当时以为的问题 vs 实际的问题」）

## 相关文档

| 文档 | 读者 |
|---|---|
| [`../docs/HOW-TO-USE.md`](../docs/HOW-TO-USE.md) | **用的人** —— 从哪进、怎么驱动、什么时候该找别的 skill |
| [`../docs/ENGINE.md`](../docs/ENGINE.md) | 改代码的人 —— 架构边界与依赖规则 |
| [`../docs/DEBT.md`](../docs/DEBT.md) | 所有人 —— D1–D11 技术债，每条都有机器判据 |

## 为什么单独开这个目录

根目录原来只有一份 `HANDOFF.md`（2026-10-30，讲评测框架），
而算子优化引擎这几天的进展散在 20 次提交信息里。
新接手的人要先读 20 条 commit message 才能拼出发生了什么 —— 这不可接受。

## 索引

| 文件 | 内容 |
|---|---|
| [`handoff/HANDOFF-2026-11-25.md`](handoff/HANDOFF-2026-11-25.md) | 算子优化引擎的交接：做成了什么、三个反复踩的坑、已知问题、怎么验收 |
| [`engineering-iteration/2026-11-24-25-engine-sprint.md`](engineering-iteration/2026-11-24-25-engine-sprint.md) | 20 次提交的复盘：v6 六组件、C6 三次迭代、精度门三处断链、语言选型、两张卡的实现细节 |
| [`engineering-iteration/roadmap-history/`](engineering-iteration/roadmap-history/) | **设计演进留存**：v5.1 → v7.2 八份路线图 + 被推翻的想法清单 |

## 约定

- **handoff 是快照**：每次交接写一份新的，旧的保留（带日期）
- **iteration 是按时间段的复盘**：一个 sprint 一篇
- 两者都不重复 `docs/ENGINE.md`（架构边界）和 `docs/DEBT.md`（技术债）的内容
- 数字必须带口径（`event_isolated` / `back_to_back`）和来源（results/*.json）
