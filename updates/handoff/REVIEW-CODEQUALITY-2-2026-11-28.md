# 代码质量审查 · 之二 · `selftest()` 该不该面向对象

> 业主问：这段能不能像 Java 那样用面向对象写得好看点。
> 结论先说：**能不用类，但这个方向不是解。真正的问题是另外四件事。**

---

## 一、先量化：这不是一个函数，是**十个文件各有一套**

```
$ grep -rln "def selftest" --include='*.py' Plugins scripts
Plugins/ops/engine/floor.py
Plugins/ops/skills/op-profiling/tools/pmc_collect.py
Plugins/ops/skills/op-roofline/templates.py
scripts/geomean.py
scripts/opbench.py
scripts/atlas_query.py
scripts/rank_impls.py
scripts/design_space.py
scripts/op_bootstrap.py
scripts/ingest_check.py
```

`floor.py` 621 行，其中 `selftest()` 占 **约 80 行（13%）**。
调用点：`engine/cli.py:185 → floor_io.selftest()`。

---

## 二、丑在哪（四条，按严重度）

### ① `assert` 在 `python -O` 下会被剥掉 —— **这是真 bug，不是风格问题**

我用字节码验证过：

```
默认模式：  COMPARE_OP → POP_JUMP_IF_TRUE → LOAD_ASSERTION_ERROR → RAISE_VARARGS
-O 模式：   （断言整段消失）→ LOAD_GLOBAL print("PASS")
```

即 **`python3 -O floor.py --selftest` 会打印 `selftest PASS`，一个断言都不跑。**
部署时用 `-O` 是常见做法 —— 那时这个自检就是**绿灯摆设**。

### ② 测试住在生产文件里

`floor.py` 是产品代码，却带 80 行回归测试。后果：
- 改行情价被抬高（改一个判据要读 621 行）
- 想单跑这个测试必须 `--selftest`，CI 里没有天然入口
- `pytest` 收集不到（`test_floor_block.py` 测的是另一件事）

### ③ 每个场景手写三步，重复 10 遍

```python
r = analyze({...}, fake)          # 1. 造输入
assert r["gate"] == "at_floor", r # 2. 断言
print("[ok] 贴地板 …")            # 3. 打印
```

10 个场景 = 30 行样板 + 60 行真实意图。

### ④ 魔数内联

`268.4` / `223.7` / `223.7 * 1.15` / `279.0` / `58.2` —— 这些数字的**含义**（268MB 对应 223.7µs 是怎么算的）只靠注释。
`223.7 * 1.15` 这种"写死的相对值"没法表达"我要的是 15% 空间"这个意图。

---

## 三、为什么"面向对象"不是解

业主说的"像 Java 那样"，我理解是想要**清晰的结构**。但 Java 那边真正起作用的不是"类"，是三件事：

| Java 里 | 起作用的是 | Python 对应物 |
|---|---|---|
| `class FooTest` | **测试与生产分离** | `tests/test_foo.py` |
| `@Test` + JUnit 运行器 | **框架负责运行与报告** | `pytest` |
| `@ParameterizedTest @CsvSource` | **数据驱动** | `@pytest.mark.parametrize` |
| `assertEquals(expected, actual)` | 断言库 | `assert` / `pytest` |

**类本身在这件事上没有任何贡献** —— 这里的场景全是无状态的（造输入 → 算 → 断言），
没有多态、没有生命周期、没有需要封装的字段。写成类只会多出一层 `self`，
代码量反而增加 2–3 倍。

**结论：要的是"数据驱动 + 测试与生产分离"，不是"类"。**

---

## 四、两个可选写法

### 方案 A（推荐）：搬进 `tests/`，用 parametrize

```python
# tests/test_floor_caliber.py
"""floor 口径的回归测试。

守的不是「代码能跑」，而是三个已被真实事故验证过的口径：
  · 两个地板取慢的
  · launch 是加性项（小 kernel 的地板 = 搬运 + 43µs）
  · 归约要打折（0.79）
"""
import pytest
from engine.floor import analyze

FAKE = {"layers": {
    "2_rw_1to1":       {"achievable": 1200, "confidence": "high"},
    "2_read_only":     {"achievable": 2000, "confidence": "high"},
    "2_rw_2to1":       {"achievable": 1730, "confidence": "high"},
    "3_cross_row":     {"achievable": None, "reason": "未标定"},
    "3_scatter_gather": {"achievable": None, "reason": "未标定"},
}}

# 268.4 MB / 1200 GB/s = 223.7 µs —— 地板本身
FLOOR_US = 268.4 / 1200 * 1000

CASES = [
    # (名字, 输入, 期望字段)
    ("贴地板",      dict(bytes_mb=268.4, pattern="rw_1to1", t_actual_us=FLOOR_US),
     dict(gate="at_floor")),
    ("15%空间",     dict(bytes_mb=268.4, pattern="rw_1to1", t_actual_us=FLOOR_US * 1.15),
     dict(gate="structural")),
    ("40%空间",     dict(bytes_mb=268.4, pattern="rw_1to1", t_actual_us=FLOOR_US * 1.40),
     dict(gate="headroom", advice_contains="先怀疑")),
    ("快过地板",    dict(bytes_mb=268.4, pattern="rw_1to1", t_actual_us=200.0),
     dict(gate="below_floor")),
    ("未标定模式",  dict(bytes_mb=268.4, pattern="scatter_gather", t_actual_us=100.0),
     dict(has_error=True)),
    # GEMM 4096³：算力腿 492µs，只算带宽会得到 58µs（8× 假空间）
    ("算力绑定",    dict(bytes_mb=100.7, pattern="rw_2to1",
                        flops=2 * 4096 ** 3, peak_tf=279.0, t_actual_us=505.0),
     dict(binding="compute", gate="at_floor")),
    ("带宽绑定",    dict(bytes_mb=268.4, pattern="rw_1to1",
                        flops=1.0e9, peak_tf=279.0, t_actual_us=242.2),
     dict(binding="memory")),
]

@pytest.mark.parametrize("name,case,expect", CASES, ids=[c[0] for c in CASES])
def test_caliber(name, case, expect):
    r = analyze(case, FAKE)
    for key, want in expect.items():
        if key == "has_error":
            assert "error" in r, r
        elif key == "advice_contains":
            assert want in r.get("advice", ""), r
        else:
            assert r[key] == want, (name, key, r)
```

**收益**：
- 80 行 → 约 45 行，且**意图全在数据表里**（一行一个场景，一眼看完）
- 失败时 pytest 直接报 `贴地板 / 15%空间 / …`
- `-O` 不再影响断言
- `floor.py` 少 80 行
- 魔数变成**算式**（`268.4 / 1200 * 1000`），不再是神秘常数

### 方案 B（如果坚持留在 `floor.py`）：数据类 + 循环

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Case:
    name: str
    given: dict
    expect: dict
    check: Any = None          # 可选：自定义断言（如 launch 的 remaining 区间）

def selftest() -> int:
    for c in _CASES:
        r = analyze(c.given, _fake())
        for k, want in c.expect.items():
            if callable(want):
                assert want(r[k]), (c.name, k, r)
            else:
                assert r[k] == want, (c.name, k, r)
        print("[ok] %s" % c.name)
    print("\nselftest PASS")
    return 0
```

这是"Java 风格"的合理形态：**用数据类描述用例，用统一循环执行**，
而不是为每个场景建一个类。

---

## 五、还有一件必须一起修的事

**`assert` → 显式异常**（或直接搬 pytest）：

```python
def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)     # -O 下不会被剥
```

或者搬进 pytest（方案 A）后这个问题自动消失。

**在搬之前，至少给 `selftest()` 加一道守卫**：

```python
def selftest() -> int:
    if not __debug__:
        raise SystemExit("✗ `python -O` 会剥掉 assert，selftest 将失效；请用默认模式运行")
```

---

## 六、优先级（和渲染层分开排）

| 项 | 价值 | 风险 | 建议 |
|---|---|---|---|
| **修 `-O` 静默失效** | 高（假绿灯） | 极低（3 行） | **现在就做** |
| `selftest` 搬 `tests/` + parametrize | 中高 | 低（纯搬） | 等 3 轮跑完 |
| 另外 9 个文件的 selftest 统一 | 中 | 中 | 最后做，不着急 |

---

## 七、一句话回答业主

> **不用改成 Java 那种类结构** —— 这里是"一堆无状态场景"，类只会让代码更长。
> **要的是 JUnit 的那种"参数化用例表"**：Python 里就是 `pytest.mark.parametrize`，
> 一行一个场景，魔数变成算式，断言交给框架。
> 顺手能修掉一个真 bug：`python -O` 下这个自检现在是**静默通过**的。
