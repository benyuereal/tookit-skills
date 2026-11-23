# 技术债登记（每条都要能被机器发现，否则等于没记）

> 规则：**登记在这里的债，必须有一条命令能查出来，或一个测试能守住。**
> 只写"以后要改"不算登记——那是许愿。

---

## D1 · `ab.py` 有 7 份，且**没有两份相同**

**发现**：2026-11-24。业主指出 `scripts/ab.py` 不存在（文档里写错了），实际在
各 skill 的 `bench/` 下。查下来不是"5 份重复"，是 **7 份各自演化**：

```bash
ls Plugins/ops/skills/*/bench/ab.py | wc -l     # 7
md5sum Plugins/ops/skills/*/bench/ab.py | awk '{print $1}' | sort -u | wc -l   # 7（全不同）
```

涉及：`op-fused-add-rmsnorm`、`op-gemm-fp8`、`op-moe-align-permute`、
`op-moe-topk-gating`、`op-moe-w8a8`、`op-rmsnorm-fused`、`op-sparse-mla`。

**风险**：ABBA 交替测 + 3×MAD 判定是本仓所有性能结论的地基。
7 份各自演化意味着**不同卡的"显著"判据可能不一样**，而没人察觉得到。
这正是「校验本身没被校验」的又一次。

**裁决（业主 2026-11-24）**：不许再新建第 6（第 8）份。

**还债路径**（未做，需单独立项）：
1. 先写测试固定「所有 ab.py 的 3×MAD 判据与 ABBA 顺序逻辑必须一致」
   —— 这会立刻暴露 7 份之间差在哪；
2. 再抽公共实现，各卡改为薄封装；
3. **不许**在没有第 1 步的情况下直接合并（会丢掉各卡的特殊口径）。

**当前防回退**：新增 `ab.py` 前必须先跑
```bash
ls Plugins/ops/skills/*/bench/ab.py | wc -l   # 超过 7 就要说明为什么
```

---

## D2 · `validate()` 545 行（目标 ≤200）

2026-11-24：加 E39/E40 后从 540 涨到 545。Q5 目标 200。
拆分不在当前冲刺范围（风险高、收益低），但数字必须留痕：

```bash
python3 -I - <<'PY'
import ast, pathlib
t = ast.parse(pathlib.Path("Plugins/ops/op-atlas/validate.py").read_text())
best = max(((n.end_lineno - n.lineno + 1, n.name) for n in ast.walk(t)
            if isinstance(n, ast.FunctionDef)), default=(0, ""))
print(best)
PY
```

---

## D3 · 17 个 E25 警告中的 6 张无任何可补证据

`fused-add-rmsnorm`、`gemm-int8`、`layernorm`、`moe-combine-reduce`、`qk-norm`、`moe-w8a8`
——它们的数字只存在于卡片 `ratio` 里，没有 `results/*.json`，也没有可复跑的 bench 入口。

**待业主裁决**：降 `draft` 会让 measured 从 27 掉到 21（分母按裁决②锁 27）。

---

## D4 · `SQ_LDS_BANK_CONFLICT` 在本机采不到真实值

2026-11-24 实测：故意构造 bank conflict 的 kernel 上，
`SQ_INSTS_LDS=25600`、`SQ_WAIT_INST_LDS=9202` 都非零，**唯独它仍是 0**。

**影响**：任何基于它的归因都不成立。已从 S4 五件套移除，改用
`SQ_WAIT_INST_LDS/SQ_INSTS_LDS` 与 `SQ_INSTS_LDS/MFMA` 两个代理。
详见 `Plugins/ops/skills/op-profiling/references/counters.md §1.1`。
