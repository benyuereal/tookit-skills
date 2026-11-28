# REVIEW-P3.2（复审）· 2026-11-28 · lead 结论：**这不是闭环，是误判**

验收对象：commit `4262331 fix(driver): eval 前生成 judge，闭环第一次出现 KEEP`
你说「第一次闭环了」。我复跑后**不同意这个结论**。

## 一、K1/K2/K3 我复跑的结果

```
K1  KEEP 条数 = 1                     ← 你说是的就是这条
K2  该条 claim=PASS, vs_base=0.5664   ← 表面看没问题
K3  judge/tolerance.json 确实被自动生成了   ← 接线修对了
```

**但这条 KEEP 不是 agent 带来的。** 看记录本身：

```json
{"iter":1, "round_id":"r1",
 "change":"noop",                    ← agent 什么都没做
 "changed_files":[],                 ← 一个文件都没改
 "correctness":true, "verdict":"KEEP",
 "base_ms":0.2531, "cand_ms":0.1434, "vs_base":0.5664,
 "claim":"PASS"}
```

同一批运行的 round 2：

```json
{"iter":2, "change":"noop",
 "changed_files":["memory/ledger.jsonl","results/bench_std.json"],   ← 这两个是**引擎自己写的**
 "verdict":"NEUTRAL", "base_ms":0.1435, "cand_ms":0.1432, "claim":"PASS"}
```

## 二、真实发生的事（时序还原）

```
r1: base=0.2531（旧的/受干扰的基线）  vs  cand=0.1434（卡里早就实现好的 reg-resident）
    agent 没改任何东西 → 但 0.5664 远小于噪声底 → 判 KEEP
    ↓
    P1 的基线晋升被触发 → base.json = {"base":"fa-rms-mr2","ms":0.1434}
    ↓
r2: base=0.1435（刚晋升的它自己） vs cand=0.1432（还是它自己） → NEUTRAL
```

**也就是说：引擎把一个"本来就已经优化好的实现"跟一个慢基线比了一次，
判成 KEEP，然后把它升成新基线，下一轮再跟它自己比。**

agent 全程 `noop`。**这不是自主迭代，这是把历史成果记到了这一轮头上。**

## 三、两个必须修的问题

### 问题 1：没有 noop 守卫（致命）

`changed_files == []` 却判 KEEP。**"没有改动"绝不该产生"提升"判定** ——
否则循环可以永远"进步"（每次换一个慢基线比一次），KEEP 就失去含义。

**修**：verdict 前面加一道判据：

```python
AGENT_PATHS = ("Plugins/ops/skills/",)          # 只认 agent 该改的地方

def _agent_changes(changed_files):
    return [f for f in changed_files
            if any(f.startswith(p) for p in AGENT_PATHS)]

# 在 _verdict 里：
if not _agent_changes(changed_files):
    return "INVALID_NOOP"        # 不 KEEP、不 REVERT、**不晋升**
```

`INVALID_NOOP` 语义：
- **不参与** KEEP/REVERT 统计；
- **不触发基线晋升**；
- 连续 `MAX_NOOP=1` 次就停（说明 agent 没干活，或 prompt 没让它动）。

### 问题 2：`changed_files` 把引擎输出算成 agent 改动

round 2 的 `changed_files = ['memory/ledger.jsonl', 'results/bench_std.json']` ——
这两个是 `loop.py run` 自己写的。它会让 noop 看起来"改了东西"。

**修**：`git diff --name-only` 之后先过滤掉引擎产物：

```
memory/
results/
.artifacts/
**/runs/loop/
```

## 四、怎么才算真闭环（验收标准，明确写死）

**K-v2**：

```bash
cd /workspace/tookit-skills
# 1) 存在一条 KEEP，且该轮 agent 真的改了实现文件
python3 - <<'PY'
import glob,json
ok=False
for f in glob.glob('.artifacts/driver/**/*iter*.jsonl', recursive=True):
    for l in open(f):
        d=json.loads(l)
        if d.get('verdict')=='KEEP':
            ch=[c for c in d.get('changed_files') or [] if c.startswith('Plugins/ops/skills/')]
            print(f.split('/')[-1], 'KEEP changed=', ch)
            ok = ok or bool(ch)
print('K-v2a 通过:', ok)
PY
# 2) 不存在「noop 却 KEEP」的记录
grep -h '"change": "noop"' .artifacts/driver/**/*iter*.jsonl 2>/dev/null | grep -c KEEP   # 期望 0
# 3) 晋升只发生在有 agent 改动的 KEEP 上：base.json 的 ts 与那条 KEEP 的时间对应
```

**要跑出这条，必须让 agent 真的实现一个新变体**（例如 `fa-rms-mr4` 或新候选），
而不是 noop。**用 noop 脚本永远证明不了闭环。**

## 五、做对了的（保留）

- ✅ `RealBackend.eval` 前生成 judge —— 接线修对了，K3 过；
- ✅ `FAIL:evidence` / `FAIL:caliber` 分类 —— 有用了；
- ✅ 基线晋升链路在跑（`base.json` 内容正确、`evidence_kind=vendor_library`）；
- ✅ 每轮 git commit（虽然内容是 noop）。

## 六、结论

> **「第一次闭环」这个说法要撤回。** 机制链路是通的（这确实值钱），
> 但**这条 KEEP 是误判**：agent 没改任何东西。
>
> 修完 noop 守卫 + `changed_files` 过滤，再用一个**真会改代码的 agent**
> 跑出「有改动的 KEEP」，才能说闭环。

改动量不大（两份小函数 + 一次重跑），但**这一条不修，后面的 9 分指标全部不可信**。
