# URGENT-NOTE · 2026-11-28 21:10 · 3 轮实验跑到第 1 轮时发现的 2 个 bug

> 现状：老卡 `op-fused-add-rmsnorm/fused-new` 第 1 轮 agent 已结束（21:06），
> 引擎正在测量；新卡仍在跑。以下两条**现在就能修**，不必等整轮跑完。

---

## ✅ 先说好消息：D1 这次是教科书级

老卡 agent 的收尾原文（`claude -p` 的 `result` 字段）：

> **按规矩我没有测、也没有写任何「快了多少」，本轮验证记进了 `ledger.md`（轮 7 复验段）。
> 判定请走 `loop.py run`。**

它做了：重编 `new.so` 验证逐字节一致、`opcheck correctness` 三个变体全 PASS
（`max_ulp=0.50` / `cosine=0.9999986`）、`test_adapter.py` 7 passed、尾块 12 个 M 全验。
**一行性能数字都没编。** 这是我们想要的 agent 行为。

---

## 🔴 BUG A：commit message 是 66 KB 的 JSON（污染 git 历史 + 记录）

```
$ git -C .artifacts/driver/op-fused_p32/workspace log --format='%h|%s' -1
97e24df|{"duration_api_ms":442177,"stop_reason":"end_turn","session_id":"9b463e30-...","total_cost_usd":1.3704,...
```

**根因**：`RealBackend.run_agent` 最后一行

```python
return r.stdout.strip()        # ← claude -p --output-format json 的 stdout 是整段 JSON
```

`iterate()` 把返回值直接当 `commit message`，也直接写进 `IterationRecord.change`。

**修**：`run_agent` 解析 JSON 取 `.result`（并保留 `duration_api_ms` / `num_turns` / `cost_usd` 到诊断字段）：

```python
import json as _json
try:
    d = _json.loads(r.stdout)
    self.last_meta = {k: d.get(k) for k in ("duration_api_ms","num_turns","total_cost_usd")}
    return str(d.get("result") or "").strip()[:4000]
except Exception:
    return r.stdout.strip()[:4000]
```

**验收**：`git log` 的 commit message 是自然语言；`IterationRecord.change` < 4 KB。

---

## 🔴 BUG B（更严重）：`AGENT_PATHS` 太宽，`ledger.md` 也算"agent 改动" —— **又一类假 KEEP**

```python
AGENT_PATHS = ("Plugins/ops/skills/",)      # ← 连 SKILL.md / ledger.md / variants.yaml 都算
```

这轮 agent **只改了 `ledger.md`**（实现文件 `fa-rms-new` 是 19:37 上一轮就有的）：

```
$ git show --stat HEAD
 Plugins/ops/skills/op-fused-add-rmsnorm/ledger.md | 29 +++++++++++++++++++++++
 1 file changed
```

但 `agent_changes()` 认它 → **noop 守卫不会触发** → 引擎会去测 `fa-rms-new`，
而它相对 base 的提升与"这轮改了什么"无关 → **很可能又是一个归因错误的 KEEP**。

这与 20:00 那次"noop 却 KEEP"是同一类问题，只是绕过的路径从「空改动」换成了「文档改动」。

**修**：把 AGENT_PATHS 收窄到**代码文件**：

```python
AGENT_PATHS = (
    "Plugins/ops/skills/",      # 起点
)
# 但必须排除文档/镜像类，只认实现：
AGENT_CODE = re.compile(r"(_common\.py|bench/.*\.py|kernels/.*\.(hip|cuh|cu|so|sh)|tests/.*\.py)$")
def agent_changes(files):
    return [f for f in files
            if any(f.startswith(p) for p in AGENT_PATHS)
            and AGENT_CODE.search(f)
            and not any(f.startswith(q) for q in ENGINE_PATHS)]
```

**验收**：
- `changed_files=[".../ledger.md"]` → `agent_changes == []` → `INVALID_NOOP`
- `changed_files=[".../_common.py"]` → 非空
- 补一条 fake 用例锁住。

---

## 建议

1. **先修 A+B 再继续跑**：现在这轮哪怕跑出 KEEP，归因也不可信（agent 只改了文档）。
2. 两条都是十几行改动；改完重跑 3 轮。
3. 记录现在还没落（引擎在测量阶段），修完 A 之后重跑才干净。
