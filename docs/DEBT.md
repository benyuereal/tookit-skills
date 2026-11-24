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

---

## D5 · `mlp-gelu-geglu` 的语义自相矛盾

**发现**：2026-11-24。`Plugins/ops/op-atlas/cards/mlp-gelu-geglu.yaml` 把
**MLP 的 GeGLU**（门控激活）和 **GroupNorm+SiLU 复合算子**拼成了一件事。

原文（卡上同时存在，逐条可 grep）：

```yaml
# 1) 卡名与 math 说的是 MLP 的 GeGLU
math: y = gelu(x*W_up) * (x*W_gate) * W_down

# 2) 但主变体指向的是 GroupNorm+SiLU
variants[0].import_path: lightop.GNSilu
variants[0].signature:   GNSilu()(input) 元素级激活     # ← 「元素级激活」与签名自相矛盾

# 3) ratio 存的是两个算子的跨形状比值
ratio.value: 0.33
ratio.vs:    GNSilu 相对 torch F.silu 的带宽比（153/468）
ratio.scope.in_len: 64x32x64x64 (GNSilu) / 4096x4096 (F.silu)

# 4) conclusion / hard_preconditions 自己承认了「不是同一个算子」
conclusion: GNSilu 是 GroupNorm+SiLU 复合算子，不能当纯激活；纯 silu 达 468 GB/s
hard_preconditions[0]: lightop.GNSilu 是 GroupNorm+SiLU，需要 num_groups 与 num_channels，不是纯激活
```

**矛盾在四处**（不是一处）：

1. `math` 是 `gelu(...)*...*W_down`（MLP），`conclusion` 谈的是 `GNSilu`/`GroupNorm` —— 两个算子。
2. `signature` 写「元素级激活」，但实测 `lightop.GNSilu(num_groups, num_channels)` 是
   `torch.nn.Module` 且必传两个参数（`inspect.signature` 确认）—— 它**不可能是**纯元素级激活。
   照着 `modify_path.how` 的 `lightop.GNSilu()(x)` 调用会直接 `TypeError`。
3. `ratio.scope` 的形状是 4 维 NCHW `64x32x64x64`（图像 norm 的形状），与 `math` 的
   MLP 矩阵 `(M,H)` 不在一个口径 —— 这正是「跨形状拼接比值」。
4. `0.33 = 153/468` 的分子分母**分别来自不同形状**：153 是 GNSilu 在 `64x32x64x64`
   （bf16）上的带宽，468 是 torch `F.silu` 在 `4096x4096` 上的带宽。两者不可相除。

**2026-11-24 本机复测（gfx938 / BW1100 / torch 2.11.0 / lightop 0.6.0）**：

| 算子 | 形状 | dtype | 实测带宽 |
|---|---|---|---|
| `lightop.GNSilu` | 64x32x64x64 | bf16 | **~150 GB/s**（卡里写 153，吻合） |
| `lightop.GNSilu` | 128x32x64x64 | bf16 | ~150 GB/s |
| `lightop.GNSilu` | 64x64x64x64 | bf16 | ~151 GB/s |
| `torch F.silu` | 4096x4096 | fp32 | ~516 GB/s |
| `torch F.silu` | 4096x4096 | bf16 | ~272 GB/s |
| `torch F.silu` | 32768x4096 | bf16 | ~380 GB/s |

结论：`468` 既不是 GNSilu 的数、也不是任何一个标称形状上稳定复现的 F.silu 数；
它落在 fp32 大形状 F.silu（516/721）与 bf16（272–380）之间的空档。
**卡里没有 `results/*.json`，153 与 468 均无可复跑来源**（`gaps.md` 已判 `missing_evidence`，
连续 3 批缺证据）。`op-optimize/cases/` 下唯一的 SwiGLU case
（`001-swiglu-fused-vs-split`）用的是 `lightop.FuseBiasSwiGLU`，与本卡无关。

**本机实际可用的相关实现**（实测清单，供修卡时替换）：

| 库 | 名字 | 语义 | 与 math 的关系 |
|---|---|---|---|
| `aiter` | `gelu_and_mul(out, x)` | `gelu(x[:, :H]) * x[:, H:]`（实测证实） | **直接对应 math** |
| `aiter` | `gelu_tanh_and_mul` | tanh 近似 GeGLU | 对应 math（近似式） |
| `lightop` | `FuseBiasGelu()(x, bias)` | `gelu(x+bias)`，元素级 | 对应 math 的单步 |
| `lightop` | `FuseBiasSwiGLU()(x, bias)` | SwiGLU（silu 门控） | 同族不同激活 |
| `lightop` | `GNSilu(gs, ch)` | GroupNorm+SiLU，**需 NCHW 4 维** | **与 math 无关** |
| `torch` | `F.gelu` / `F.silu` | 逐元素 | 参考 |

**没有任何库提供名字叫 `geglu` 的入口**；`math` 的公式需由 `gelu_and_mul` 或
`gelu → mul` 两步拼出。这就是为什么把 `GNSilu` 写进 `import_path` 是错的。

**三个处理选项**（需业主裁决，本文件只登记不决定）：

- **A. 拆成两张卡**：`geglu`（family=activation，math 保留，import_path 改 `aiter.gelu_and_mul`）
  + `gn-silu`（family=norm，承接 `GNSilu` 的 153 与 4 维形状）。
  优点：语义最干净，符合 `mlp-swiglu` 的既有结构；缺点：`gn-silu` 目前同样无 results，
  会新增一张 `missing_evidence` 卡，measured 分母要重算。
- **B. 保留 `mlp-gelu-geglu`，把 GNSilu 内容移到 `layernorm` 或新 norm 卡**。
  优点：不新增卡；缺点：`layernorm` 与 GroupNorm 语义也不同，仍是借位。
- **C. 整张卡降 draft**：理由是语义不清 + 无可复跑数据（E25 已连续 3 批判 missing）。
  `gaps.md` 的生成物已经写「降级 draft（连续 ≥2 批）」，本 D5 与该处置一致。
  优点：最省事、风险最低；缺点：丢掉了「MLP GeGLU 无本机实测」这条真实覆盖缺口的信息。

**建议**：先按 C 降 draft 止住误用（`GNSilu()(x)` 会直接报错），再按 A 拆分。
不建议 B —— 把一个 norm 复合算子塞进另一张 norm 卡，等于把矛盾挪个位置。

**谁裁决**：算子卡业主（`op-atlas` 卡片的维护者）。涉及 `measured` 计数分母
（当前锁 27），降级会影响 `gaps.md` 与 INDEX，须与 D3 的裁决一并处理。

**当前防回退**：`Plugins/ops/tests/test_mlp_gelu_semantics.py`

```bash
python3 -I -m pytest Plugins/ops/tests/test_mlp_gelu_semantics.py -v
# 期望：3 skipped（D5 已知矛盾，按设计登记不 fail）+ 7 passed
```

该测试对**四条判据**做机器检查（math vs conclusion 的算子名、scope 形状维度、
主变体 import_path 族别、跨形状拼接比值），并用合成反例 + 干净对照卡
（`mlp-swiglu`）证明判据「该报错时真会报错」—— 避免 skip 变成假绿。
卡修好后 skip 自动转 pass，无需改测试。
