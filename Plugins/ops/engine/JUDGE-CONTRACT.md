# JUDGE-CONTRACT（草案，from loop.py 侧）

> 目的：把"被审计的对象"（judge + 任务定义 + 冻结容差）与"审计层"（loop.py / record /
> claim）之间的接口定死。本文只写**接口与验收**，不写 teacher prompt 怎么设计。
>
> 作者：loop.py 侧。judge 生成的实现（`engine/judge.py`）归 lead 侧；本文件是他们
> 的需求输入。有冲突以实现为准，但请把差异回写到这份文件。

## 1. 目录与不可变性

```
tasks/<op>/                       # 进 git，只有问题，没有过程
  description.md                  # 必填：语义、签名、dtype/layout、边界约定
  shapes.json                     # 必填：一或多档 shape
  task.toml                       # 可选：language/arch/target/deny
  reference/                      # 可选：已有实现（有就读源码/黑盒；没有就现写）

.artifacts/<arm>/<op>_<id>/       # 不进 git，一次 run 一个目录
  task.snapshot/                  # 开始时的 tasks/<op> 只读快照（含 sha256）
  judge.py                        # teacher 生成：build_inputs + reference
  tolerance.json                  # 校准后冻结的容差（见 §3）
  runs/loop/<date>.jsonl          # 引擎账本（record 写入）
  runs/loop/base.json             # 基线晋升（见 §4）
  results/bench_std.json          # --bench-std 的交付物
```

**硬规则**：`run` 不接受任何覆盖任务定义的参数（description / shapes / 容差 / deny）。
要改问题就改 `tasks/` 或新建 task。"这次特殊，覆盖一下"一律拒绝。

## 2. judge 必须提供的 API

```python
# judge.py（每 run 一份，由 teacher 生成）
def build_inputs(params: dict, seed: int = 0, device=None) -> tuple:
    """返回 (…输入张量, meta)。meta 必须是 dict 且放在最后一位。
    必须满足 description 里的输入合法性（索引越界、单调性、mask 等）。"""

def reference(*inputs, params: dict | None = None) -> tuple:
    """返回 (…输出张量)。fp32 语义。不允许调用被优化的那个实现。"""

def get_runner(name: str, **kwargs):
    """变体入口。name 至少要有 'ref-heavy'（= reference 本身）。
    库/手写变体在各自的 _common 里登记，judge 只保证 ref-heavy 存在。"""
```

- `build_inputs` 必须能**在 CPU 上跑**（loop 的 `next` 用 CPU smoke 做 1 秒自检）。
- `reference` 不允许 import 被优化对象所在库（否则"参考答案"和"被测对象"同一份，
  正确性门形同虚设）。
- 生成后必须能被 `python3 -c "import judge; judge.build_inputs(...); judge.reference(...)"`
  跑通，否则对象不可用。

## 3. 校准与冻结容差

```json
// tolerance.json
{
  "source": "reference-vs-independent",   // 或 "self-consistent-only"
  "metrics": {"max_ulp": 2.0, "cosine": 0.99999},
  "measured": {"max_ulp": 0.5, "cosine": 0.9999986},
  "shapes": ["s0"],
  "judge_sha256": "...",
  "frozen_at": "2026-11-28T12:00:00",
  "independent_check": {
    "kind": "second-impl|library|invariants",
    "detail": "..."
  }
}
```

**放行线**（loop 侧按这个判）：
- 有独立核验：`max_ulp ≤ 2.0`（bf16）或 `cosine ≥ 0.99999`，且 `frozen_at` 之后
  judge.py 的 sha256 未变 → `claim` 可放行"数值正确"。
- 没有独立核验：`source = self-consistent-only` → `claim` **不放行**数值正确，
  记录标黄。宁可标黄，不可标绿。
- 容差文件缺失、sha256 与 judge.py 不符、或 run 期间被修改 → `run` 拒绝执行。

## 4. 引擎消费方式（loop.py 已实现的部分）

| 行为 | 现状 |
|---|---|
| `run` 读 `tolerance.json` 作为判定容差 | 待接：现在读 `contract.yaml` / `--tol` |
| `record` 落账时带 judge sha256 + 容差 | 待接 |
| 候选 `purpose: calibration` → `CALIBRATION_ONLY` | ✅ 已实现（0e1ff3e） |
| 无 competitor 时允许参考基线，记录 `base_role=correctness_only` | ✅ 已实现（0e1ff3e） |
| 被接受的实现晋升为下一轮 base（`runs/loop/base.json`） | ✅ 已实现并验证：第二次 run 自动 `--base fa-rms-mr2`，Δ≈-0.1% 判 INCONCLUSIVE |
| 合计判定不再假报 `ACCEPTED_BITEXACT` | ✅ 已实现（0e1ff3e） |

## 5. 对象的验收测试（一条竖切片）

```
① tasks/<op>/ 只有 description.md + shapes.json
② judge 生成 → CPU smoke 通过
③ GPU 校准 → tolerance.json 冻结（有独立核验则标 reference-vs-independent）
④ loop.py run <baseline> → runs/loop/*.jsonl 一条记录
⑤ loop.py claim --tag <tag> → 五项全 PASS
⑥ 再 run 一条真实候选 → 判定 + 晋升
⑦ 故意改 judge.py 一个字节 → run 必须拒绝（sha256 不符）
```

⑧ 缺一项就不算"对象做完了"。

## 6. 留给 lead 的问题

1. `independent_check` 在没有第二实现时，可以用"不变量套件"（如 topk 索引逐位一致、
   softmax 和=1、tie-break 稳定）代替吗？我倾向可以，但要写进 `detail` 并在报告里注明。
2. `.artifacts/<arm>/...` 的 `<arm>` 由谁注入（opbench 的 arm 名 vs 引擎自己的 run id）？
   这会影响三臂隔离的目录权限。
3. judge 生成失败时的行为：`run` 直接拒绝，还是生成一个 `self-consistent-only` 的
   降级对象让人先看数字？我倾向**拒绝**（fail-closed），但那样第一轮可能什么都跑不了。
