# 算子优化台账条目模板（写到 `~/.config/dcu-toolkit/kb/op-opt/<算子>-<yyyyMMdd>.md`）

> 生成本模板的最省力路径：跑 `scripts/dcu_distill.py --run <run 产物> --kb ~/.config/dcu-toolkit/kb`
> 自动产出 .md + .json（数字只来自 run 产物）；本文件是手写时的对照。
>
> **格式要求（五段，缺一不可）**：`结论 / 前提 / 已证伪 / 复现命令 / 口径`。
> 检查器是 `Plugins/dcu-core/skills/dcu-orchestrator/scripts/kbq.py`：
> `python3 scripts/kbq.py index` 看缺哪段，`python3 scripts/kbq.py search <kw>` 检索。
> 缺段的卡**仍能检索到，但会标红**——历史卡不必回补，新卡按这五段写。

## 结论

- 算子 / shape：
- 瓶颈类型：memory / compute / latency / comm
- 改动（文件 / env 开关 / 编译产物）：
- 收益：（百分比/倍数；标 `[历史数据，随硬件/DTK/版本变化]`）
- 正确性：参考实现=  tolerance=  结论=

## 前提

<!-- 结论在什么条件下成立。设备/版本/形状/并发任一改变都可能反转。 -->

- 设备型号 / gfx：（物理机 rocminfo 查，勿信容器内）
- DTK 版本 / 框架 + 版本 / 精度（BF16/FP16/FP8）：
- 形状与并发：
- 其它前提：

## 已证伪

<!-- 本轮排除了哪些方向/写法。这是最有价值的部分：防止下一个人重走。 -->

- （方向 + 为什么不成立 + 依据哪条证据）

## 复现命令

```bash
# 能重现上面对比数字的确切命令
```

## 口径

<!-- 同一个数换个分母/频率档/标称峰值，结论可能反转。不写清楚就无法复用。 -->

| 项 | 值 |
|---|---|
| 计时方法 | 如 opbench 多轮中位数 + MAD |
| 频率档 | 实测 sclk（不是标称峰值） |
| 分母/读量口径 | 如「模型最少字节」而非实测 DRAM |
| 指标定义 | |

## 环境指纹（必填，缺项不许写"应该"）

- 设备型号 / gfx：        （物理机 rocminfo 查，勿信容器内）
- DTK 版本：
- 框架 + 版本：
- 精度（BF16/FP16/FP8）：
- 日期：

## 数字（同条件 A/B；历史数据，随硬件/DTK/版本变化）

| 指标 | 优化前 | 优化后 | 收益 |
|---|---|---|---|
|  |  |  |  |

- 证据路径（必须能回溯）：`runs/op-optimize/validate/result.json`、profiling 产物：

## 候选事实（跨项目可复用）

<!-- 每条一行：[ ] 一句话 | scope=<gfx/卡型> | status=verified|unverified | 依据=<证据路径> -->

- [ ]

## 明确环境专属（不并入 skill）

-

## 晋升提案

- 拟并入：`dcu-kernelwiki/wiki/<topic>.md` / `dcu-perf-cases` / 其它：
- 不并入理由：
