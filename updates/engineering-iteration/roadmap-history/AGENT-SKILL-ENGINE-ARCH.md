这就是设计的完整结构，右边标了现在的实际状态（✅ 已落 / ⬜ 还没动）：

Plugins/ops/
├── engine/                          ← 调度引擎（不含任何算子知识）
│   ├── __init__.py                  ✅ 空壳已建
│   ├── loop.py                      ✅ 已迁（1955 行）单卡循环：队列→跑→判→记
│   ├── cli.py                       ⬜ 12 个子命令入口
│   ├── advance.py                   ⬜ 多轮编排：plan→gate→run→judge→replan→stop
│   ├── plan.py                      ⬜ EV 排序 + 前提检查（S3 才做）
│   ├── judge.py                     ⬜ 唯一显著性判定（3×MAD）
│   ├── measure_mode.py              ⬜ 测量模式单一真源
│   ├── state.py                     ⬜ S0–S7 + progress 推进
│   ├── lang.py                      ⬜ 语言候选排序
│   ├── memory.py                    ⬜ recall / append API
│   ├── bench_contract.py            ⬜ ab.py 的 CLI/JSON 契约校验
│   ├── budget.py                    ⬜ 轮次 / GPU 秒 / 失败上限
│   └── artifacts.py                 ⬜ 证据打包
│
├── skills/op-optimize/
│   ├── SKILL.md                     ✅ 只讲"怎么用引擎"，不实现
│   └── loop.py                      ✅ 30 行弃用壳：告警 + 转发 engine.cli
│
└── op-atlas/                        ✅ 卡片层（不动）
    ├── cards/*.yaml
    ├── bw-ladder.yaml
    └── perf/*.json
配套的数据与记忆（不是代码）：

data/ops/
├── lang-candidates.yaml             ✅ 4 条（语言 use_when / avoid_when）
├── isa-surfaces.yaml                ✅
├── libraries-registry.yaml          ✅ 23 个库
└── tool-registry.yaml               ✅ 32 项工具

memory/                              ← 数据，不是服务
├── ledger.jsonl                     ✅ 已建，仍空（等 engine 写）
└── README.md                        ✅
现在的真实状态
只完成了第一步：把 loop.py 从 skill 里搬出来，原位置留了一个 30 行的壳。剩下 10 个模块还都在 scripts/ 扁平目录里，plan / budget / artifacts 还没建。

也就是说：物理位置对了，职责还没拆。现在的 engine/loop.py 里其实还是那 1955 行的大文件，state、judge、measure_mode 都裹在里面。

边界规则（写在 docs/ENGINE.md 里）
engine  →  可 import data、可调 skill/bench、可读写 memory
data    →  纯 YAML，不许 import 任何代码
skill   →  不许 import engine（skill 不知道谁在调度它）
memory  →  谁都能读，只有 engine 能写
剩下的迁移顺序（我建议）
1. judge.py + measure_mode.py   ← 先拆这两个，因为全局只有一份判定逻辑
2. state.py                     ← S0–S7 推进
3. cli.py                       ← 12 个子命令收口，壳转发到这里
4. lang.py / memory.py          ← 从 scripts/ 搬进来
5. plan.py / budget.py / artifacts.py  ← S3 再建
