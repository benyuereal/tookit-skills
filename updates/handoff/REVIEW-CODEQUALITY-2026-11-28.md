# 代码质量审查 · 2026-11-28 21:40 · 渲染层（就是你贴的那段）

## 一、先量化：这不是一个函数的问题，是五个函数

实测（AST 统计）：

| 文件 | 函数 | 行数 | `.append(` 次数 |
|---|---|---|---|
| `loop.py` | `write_report` | **89** | **56** |
| `iterate.py` | `render_prompt` | 57 | 39 |
| `nextstep.py` | `render` | 50 | 25 |
| `prompt.py` | `render_student_prompt` | 45 | 2 |
| `propose.py` | `render` | 42 | 19 |
| **合计** | | **283 行** | **141 次** |

**283 行里 141 次 `.append()`** —— 平均每两行一次。这就是"生硬"的来源。

## 二、`nextstep.render` 的具体问题（逐条）

```python
L = ["下一步：%s（%s）" % (label, act), ""]      # ① 第一行就手动塞空行
L.append("为什么：%s" % plan.get("why", ""))
...
if d.get("self_checks"):
    L.append("")                                # ② 空行由每段自己管
    L.append("已自检（1 秒，未跑实验）——直接执行不会失败：")
    for x in d["self_checks"]:
        L.append("  ✓ %s" % x)                  # ③ 符号硬编码在这里
if d.get("problems"):
    L.append("")                                # ② 又来一遍
    L.append("预检不通过（先修这些）：")
    for x in d["problems"]:
        L.append("  ✗ %s" % x)                  # ③ 符号又硬编码一次
...（produces / on_failure 完全同构，再两遍）
```

**六个病症：**

| # | 病症 | 代价 |
|---|---|---|
| ① | **三段式重复 4 次**：`if 有键: 加空行 + 加标题 + 循环加前缀` | 每加一段要复制 4 行样板 |
| ② | **空行规则散落在 7 处** | 少一个/多一个空行只能靠眼查 |
| ③ | **符号硬编码**（`✓` `✗` `·`） | 改风格要改 4 处 |
| ④ | **三种数据形状各写一遍**：单值 / 列表 / 键值 | 没有统一入口 |
| ⑤ | **`label` 字典定义在函数里** | 每次调用重建 |
| ⑥ | **末尾状态行用 6 参数的 `%`** | 一行 8 个占位符，改动易错 |

**但注意**：`propose.render` 的 docstring 已经警告过：

> """人读输出。**这是对外文本契约**：字段顺序与措辞被测试和文档引用，别顺手改。"""

所以**重构必须逐字保持输出不变** —— 这是硬约束，不是风格偏好。

## 三、方案：一个 60 行的 `engine/textfmt.py`

把"格式"从"内容"里抽出来，五处共用：

```python
# -*- coding: utf-8 -*-
"""面向人/agent 的多段文本渲染：**唯一的格式约定**。

仓库里有 5 处手写「逐行 append」的渲染函数，共 283 行 / 141 次 .append()，
各自管空行、各自选符号 —— 改一次格式要改五个地方。本模块只做四件事：
段落（空行统一加）、列表（符号统一）、键值/表格、组装（去连续空行）。

约定：**换行与空行规则只在这里定义。**
"""
from __future__ import annotations

BULLET, CHECK, CROSS = "·", "✓", "✗"


def section(title: str, body=()) -> list[str]:
    """一段 = 前空行 + 标题 + 正文。**调用方不需要自己加空行。**"""
    return ["", title, *body]


def bullets(items, mark: str = BULLET, indent: str = "  ") -> list[str]:
    return ["%s%s %s" % (indent, mark, x) for x in items]


def kv(pairs, width: int = 12, limit: int = 160) -> list[str]:
    """键值行：`  key          value`。"""
    return ["  %-*s %s" % (width, k, str(v)[:limit]) for k, v in pairs if v]


def table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def banner(title: str, width: int = 78) -> list[str]:
    line = "=" * width
    return [line, title, line]


def doc(lines) -> str:
    """组装：去掉连续空行，末尾补一个换行。"""
    out: list[str] = []
    for line in lines:
        if line == "" and out and out[-1] == "":
            continue
        out.append(line)
    return "\n".join(out) + "\n"
```

## 四、重构后的 `nextstep.render`（50 行 → 32 行，输出逐字不变）

```python
_ACTION_LABEL = {"run": "跑实验", "implement": "实现变体", "explore": "出题",
                 "baseline": "先测基线", "stop": "停止"}

# (取数键, 标题, 符号)  ← 加一段只改这张表，不改代码
_SECTIONS = (
    ("self_checks", "已自检（1 秒，未跑实验）——直接执行不会失败：", textfmt.CHECK),
    ("problems",    "预检不通过（先修这些）：",                      textfmt.CROSS),
    ("produces",    "跑完之后你会得到：",                             textfmt.BULLET),
    ("on_failure",  "如果失败：",                                     textfmt.BULLET),
)


def _status_line(plan: dict) -> str:
    tail = "  remaining=%s" % plan["remaining"] if "remaining" in plan else ""
    return ("当前状态：记录 %d 条（可 claim %d）/ 待做 %d / 拟做 %d / 已判死 %d%s"
            % (plan.get("n_records", 0), plan.get("n_claimable", 0),
               plan.get("n_pending", 0), plan.get("n_proposed", 0),
               plan.get("n_refuted", 0), tail))


def render(plan: dict) -> str:
    d = plan.get("detail") or {}
    act = plan.get("action")
    parts = ["下一步：%s（%s）" % (_ACTION_LABEL.get(act, act), act),
             "", "为什么：%s" % plan.get("why", ""),
             "", "照做（复制粘贴即可）：", "    %s" % plan.get("cmd", "")]
    for key, title, mark in _SECTIONS:
        if d.get(key):
            parts += textfmt.section(title, textfmt.bullets(d[key], mark))
    if d.get("expect_seconds"):
        parts += textfmt.section("预计耗时：%s" % d["expect_seconds"])
    if d.get("also"):
        parts += textfmt.section("之后还有：%s" % ", ".join(map(str, d["also"])))
    if d.get("spec"):
        parts += textfmt.section("这条候选的说明：",
                                 textfmt.kv((k, str(v)[:160])
                                            for k, v in d["spec"].items() if v))
    parts += ["", _status_line(plan)]
    return textfmt.doc(parts)
```

**加一个新段落 = 在 `_SECTIONS` 加一行。** 空行、符号、缩进自动正确。

## 五、其余四处的预期收益

| 函数 | 现状 | 重构后 | 主要手段 |
|---|---|---|---|
| `write_report` | 89 行 / 56 append | ~55 行 | `table()` 吃掉 3 张表；`section()` 吃掉 6 段 |
| `iterate.render_prompt` | 57 / 39 | ~35 | `bullets()` + `section()` |
| `propose.render` | 42 / 19 | ~30 | `banner()` + `kv()` |
| `prompt.render_student_prompt` | 45 / 2 | 不动 | 它本来就结构清晰 |

**合计 283 行 → ~190 行，且格式规则只有一个真源。**

## 六、安全协议（必须先做，否则不许重构）

这些输出是**对外文本契约**（测试、文档、agent 都在读）。所以：

**第 1 步：黄金测试（先写，此时还没改代码）**

```python
# tests/test_render_golden.py
GOLDEN = Path(__file__).parent / "golden"

def test_nextstep_render_golden():
    for f in GOLDEN.glob("plan_*.json"):
        plan = json.loads(f.read_text())
        assert ns.render(plan) == (GOLDEN / f.with_suffix(".txt").name).read_text()
```

用当前实现生成 `golden/*.txt`（覆盖 run/implement/explore/fix/stop 五种 action + 有/无 detail 的组合）。

**第 2 步：重构**
**第 3 步：黄金测试必须 100% 通过（逐字相同），否则立刻回滚**

## 七、这条不该现在做的理由

**手上还有 3 轮实验在跑**（21:10 重启，第 1 轮记录还没落）。
- 重构 `write_report` 会碰 `loop.py`，而正在跑的 run 正在调用它
- 黄金测试需要先冻结当前输出，但当前输出可能马上要随 BUG A/B 的重跑而变

**建议顺序**：
1. 等 3 轮跑完、结果验收（这是主线）
2. 再做 `textfmt` + 黄金测试重构（半天，独立于主线）

## 八、如果只做一件事

**做 `textfmt.section()` + `bullets()` 这两个函数，先只改 `nextstep.render`。**

理由：
- 它是对 agent 输出的**第一屏**，可读性收益最直接
- 50 → 32 行，风险最小
- 黄金测试只需要冻结一种输出

其余四处等这套模式被证明好用再推。
