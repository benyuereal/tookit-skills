# 代码优化计划 · 2026-11-28 21:20 · 不删代码，只做收敛与加固

> 业主定调：**不删代码，删除要谨慎**。本计划一行不删。
> 目标：**同一件事只有一个真源、最长的函数能读懂、没测试的地方补测试。**

---

## 一、先纠正我自己的误判（已核实）

| 我之前的说法 | 实际 | 结论 |
|---|---|---|
| "iterate.py 有重复的 load_candidates" | 它只是**薄包装**：`return _engine("card").load_candidates(card)` | ✅ 不是重复，不用动 |
| "三个编排器有大量同名重复函数" | 实测交集只有 `main` | ✅ 没有大面积复制粘贴 |
| "启动慢" | `loop.py --help` **0.07s** | ✅ 性能不是问题 |
| "advance.py 是死代码" | ❌ 已撤回，它有 2 轮实证 + D1 守卫 | ✅ 保留 |

**所以优化的重点不在"砍"，在"读不懂"和"没人验"。**

---

## 二、优化项 A：拆 `cmd_run`（446 行）—— 唯一的大可读性问题

实测 `loop.py` 79 个函数，最长的：

```
cmd_run          446 行   ← 比第 2 名（cmd_resume 115）大 4 倍
cmd_resume       115
decide           108
bottleneck_problems 106
main             106
```

**446 行意味着改一处要滚 4 屏。** 纯重构（提取函数，行为不变）：

```
cmd_run
├─ _resolve_candidates()      # 选候选 / 校验登记
├─ _preflight_gate()          # 六道门（若还没拆出来）
├─ _run_measurement()         # 调 bench、收数
├─ _decide_and_record()       # 判定 + 写账本
└─ _post_run_hooks()          # 晋升 / atlas 回写
```

**验收**：`cmd_run` ≤ 120 行；全仓 `pytest Plugins/ops -q` 仍 13 failed；`loop.py run` 在 `op-fused-add-rmsnorm` 上行为与重构前一致
（同一 tag 跑两次，`cand_ms` 差 <2%、`decision` 相同）。

---

## 三、优化项 B：补测试盲区（1,732 行产品代码 0 测试）

| 模块 | 行数 | 测试 | 风险 |
|---|---|---|---|
| `floor.py` | 621 | **0** | 地板算错 → 所有"剩余空间"判断错 |
| `ledger.py` | 339 | **0** | 账本读写错 → 历史结论错 |
| `judge.py` | 288 | **0** | 判据生成错 → 全盘错 |
| `propose.py` | 255 | **0** | 出题错 → 空转 |
| `signals.py` | 229 | **0** | 信号算错 → 方向错 |

对比：`loop.py` 2,889 行有 **11 个测试文件** —— 测试分布和风险是反的。

**优先补 3 个（按风险）**：
1. `judge.py`：容差 sha256 自校验、fail-closed（生成不出 → 不写文件）、两条证据路径
2. `ledger.py`：追加/读取/去重、坏行不炸、tag 过滤
3. `floor.py`：三层分母不混用、缺数据时的降级

**验收**：新增 ≥15 条用例，全绿；覆盖 `judge.py` 的 fail-closed 分支。

---

## 四、优化项 C：把 `advance.py` 里已验证的组件**提取共用**

这是本次优化里**唯一直接提升 driver 可靠性**的一项：

| 提取 | 现状 | 收益 |
|---|---|---|
| **`engine/guard.py`** ← `advance.py::_guard_no_bench` | driver 只靠 prompt 拦"自己跑 bench" | 变成**代码级拒绝**，直接补上 D1 强制（P4 的一半） |
| **`engine/progress.py`** ← `advance.py` 的 progress 状态机（round/stage/history） | driver 只有记录文件，无 stage，`--resume` 没法做 | driver 可续跑；两套编排共享同一状态模型 |

**做法**：把函数**复制**到新模块，`advance.py` 改成 `from guard import ...`（原实现保留，
先跑测试确认行为一致，再让 advance 切换）。**不删原文件里的任何一行，直到共用模块被测试锁住。**

**验收**：
- `guard`：命令含 `ab.py`/`bench/` → 抛异常；`ls bench/`（只读）→ 放行
- `driver.py` 调 `guard` 后，fake 场景里"agent 试图跑 ab.py"必须被拒
- `progress`：写/读/推进 stage 各一条用例

---

## 五、优化项 D：`iterate.py` 改为 `driver.py` 的薄 CLI（**保留文件，不删**）

现状：`iterate.py` 703 行 / 28 个函数，**自己实现了一整套轮次控制流**，但缺少 driver 已有的
noop 守卫、judge 接线、重复测量噪声底（R1–R4 的修复只进了 driver）。

后果：**同一个 bug 要修两遍，或者只修一处、另一处继续错。**

**做法（不删文件）**：
1. 先给 `iterate.py` 的 docstring 加一行「**已由 `driver.py` 取代，新代码请用 driver**」
2. 下一步把它变成薄包装：`from driver import iterate, RealBackend`，CLI 参数保留
3. 现有 703 行实现**保留在文件里**，用 `if False:` 或注释标 `# legacy` —— 等薄包装跑通两个真实算子后再删

**验收**：`iterate.py iterate --max-rounds 3` 与 `driver.py --real --demo 3` 产出**同构**的记录
（字段一致、verdict 口径一致）。

---

## 六、优化项 E：一份 `docs/ENGINE.md` 写清三者分工

现在三个入口并存但**没有一处文档说清谁该在什么时候用**：

```
loop.py      一次实验（六道门 + 记账）        ← 核心，所有人都调它
advance.py   参数枚举式多轮（自己 forge/扫参）  ← 同一实现扫参数
driver.py    agent 驱动式多轮（LLM 写实现）     ← 要新实现/新思路
iterate.py   旧的工作流反转 CLI               ← 保留入口，内部收敛到 driver
```

**验收**：文档里每个入口都有「什么时候用 / 什么时候不用 / 唯一验收命令」。

---

## 七、执行顺序（每步都能独立验收）

```
1. A 拆 cmd_run            ← 纯重构，先做（后面补测试更容易）
2. B 补 judge/ledger/floor 测试
3. C 提取 guard + progress（advance → 共用，driver 接入）
4. D iterate.py 薄包装
5. E 文档
```

**每一步的通用验收**：`pytest Plugins/ops -q` 仍 = 13 failed（不许新增红）。

---

## 八、明确不做

- ❌ 不删任何文件（业主定调）
- ❌ 不动 `driver.py` / `nextstep.py` / `card.py` / `judge.py` 的**接口**（只补测试）
- ❌ 不动 `loop.py` 的**判定逻辑**（只拆函数，不改行为）
- ❌ 不为"整齐"重命名公共函数
