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

---

## D6 · 前置推理闸门若强制，会 100% 拒跑

**发现**：2026-11-24。`scripts/check_means.py`（借鉴 v7.1「借 1」）能检出手工候选
缺 `means` / `predicted_effect` / `expected_bottleneck`。

但实测全仓：**11 个 pending 候选，0 个写了这三个字段**。

```bash
python3 -I scripts/check_means.py Plugins/ops/skills/op-rmsnorm-fused   # rc=1
python3 -I scripts/check_means.py Plugins/ops/skills/op-moe-w8a8        # rc=1
python3 -I scripts/check_means.py Plugins/ops/skills/op-sparse-mla      # rc=1
```

**风险**：如果直接把闸门接进 `loop.py run` 的注册门（rc=2），
所有实验立刻停摆。这违反业主明确说过的「约束不能卡住能力上限」。

**裁决（2026-11-24，主线程自定）**：分三档接线，不许一刀切：
1. **warn**（当前）：只在 `advance` 报告里列出，不拦运行
2. **per-card block**：某张卡的候选**全部**补齐后才对该卡转为强制
3. **global block**：全仓候选都补齐后才全局强制

**当前防回退**：`Plugins/ops/tests/test_check_means.py`（15 条，其中反向 7 条）

---

## D7 · `ablate.py::_case` 返回值类型混合

`_case()` 正常时返回 `dict`，异常时**把 `SpecError` 当返回值**返回。
调用方必须每次 `isinstance` 检查，容易漏。

**影响**：主线程写独立测试时就踩了 —— 用 `pytest.raises` 断言失败，
因为异常根本没抛出来。已在 `test_ablate.py` 里按实际契约断言并注明。

**还债路径**：让 `_case` 只返回 dict，异常照常抛；selftest 改用 `pytest.raises`
或 try/except。**这个改动会同时动 selftest 的 8 处断言**，不在本轮范围。

**当前防回退**：`Plugins/ops/tests/test_ablate.py::test_missing_tag_is_spec_error_not_verdict_failure`
按实际契约断言（`isinstance(r, SpecError)`），并在 docstring 里写明这是缺陷不是设计。

---

## D8 · `mhc-fused.yaml` 卡与源文件语义不符

**发现**：2026-11-24，读 mHC 语料时。卡上写：

```yaml
math: conv1d + hadamard + norm 的融合链
variants[0].file_hint: kernels/mhc_fused.hip
```

但 `mhc_fused.hip` 的头注释是：

```c
// Post + next-layer-projection fusion: the step-3 experiments.
```

即「post + 下一层 proj 的 Step-3 融合」，与 conv1d/hadamard/norm 无关。
真对应 conv1d+hadamard+norm 的应是 `hc-split-sinkhorn.yaml`。

**这是与 D5 同类的语义错配**（D5 是 MLP GeGLU vs GroupNorm+SiLU）。

**影响**：照卡的 `file_hint` 去读源码会对不上；`best_for` 写的
「验证 conv1d+hadamard+norm 三者融合是否划算」也不成立。

**处置（2026-11-24 已完成）**：
- `mhc-fused.yaml` 的 `title` / `math` / `best_for` 已改成 Step-3 融合的真实语义
- `cons` 补上了两条实测结论：融合净负 1.7x（流碎裂）、同 TU 编译改寄存器分配
  （118→83 VGPRs，掉 30% 带宽）
- `source_ref.note` 留痕说明这个 TU 是 post+proj，**不是** conv1d+hadamard+norm

**当前防回退**：`scripts/check_card_semantics.py`

```bash
python3 -I scripts/check_card_semantics.py --selftest   # 4 项，反向 1 + 边界 2
python3 -I scripts/check_card_semantics.py              # rc=0（修好后）
```

判据：从卡的 `math` 抽算子族关键词，去 `source_ref.path` 的标识符里找对应。
**一个都找不到就报**。已实测：修之前它精确抓出 mhc-fused（且不是误报其余 38 张卡）。
这是「一定会报出来」的判据，不是「一定会判对」——误报可由人一眼排除，漏报不可接受。

---

## D9 · `ablate.py` 分不清「冠军的消融」与「基准的另一条实验线」

**发现**：2026-11-24，首次在真实数据上跑 `ablate.py`（mqa_opt 的 LDS swizzle 实验）。

数据（`ab.py --rows 48 --ctx 32768`，event_isolated，3 轮 20 迭代）：

| variant | ms | vs group3_ref |
|---|---|---|
| `group3_ref` | 0.2372 | 1.000 |
| `group3_swz` | 0.2364 | 1.004 |
| `group3_pk` | 0.2354 | 1.008 |
| `group3_bp` | **0.1873** | **1.266** |
| `bp_noswz` | 0.1893 | 1.253 |
| `bp_pk` | 0.1862 | 1.274 |
| `abl_noepi` | 0.1694 | 1.400 |
| `abl_noshfl` | 0.1768 | 1.342 |
| `abl_neither` | **0.1163** | **2.040** |

把它们当「冠军 `group3_bp` 的消融项」喂给 `ablate.py`，得到：

```
epilogue  delta=-0.01789  share=-0.359  负贡献
shuffle   delta=-0.01054  share=-0.211  负贡献
both      delta=-0.07106  share=-1.424  负贡献
status unexplained  sum(share)=-1.96
```

**问题**：`abl_*` 是从 `group3_ref` 派生的**另一条实验线**（去掉 epilogue/shuffle
两个功能），不是 `group3_bp` 的消融。工具按「冠军减去该项」计算，自然全为负。
`share` 为负说明归因公式的前提不成立，但**工具没有拦住**，只报了 `unexplained`。

**影响**：如果人只看 `share` 列，会得出「epilogue 有害」的错误结论 ——
实际上 `abl_noepi` 是**完全不同的 kernel**（少了功能，也少了正确性保证）。

**正确做法（未实现）**：
1. spec 里每个消融项必须声明 `derived_from: champion | baseline`
2. 只有 `derived_from: champion` 的项才做归因；`baseline` 派生的单独列，不参与 share
3. `share` 出现负值时应当**直接报错**，而不是当作「负贡献」写进表 ——
   负 share 几乎总意味着 spec 写错了，而不是「这项有害」

**根因补充**：源码 `mkern_swz.hip:584` 明确写着

```cpp
// ablations (not numerically valid; timing-only diagnostics)
m.def("abl_noepi", &run<3, true, 1>);
```

即这三个是**测量探针**，不产生正确数值。所以它们**根本不该进归因** ——
归因的对象是「能算出正确结果、但慢一些」的版本，不是「故意算错以测某段耗时」的探针。

**处置（2026-11-24 已完成）**：`ablate.py` 现在强制 spec 声明两个字段：

```yaml
ablations:
  - id: epilogue
    tag: abl_noepi
    derived_from: baseline        # champion | baseline，必填
    numerically_valid: false      # true | false，必填
```

三类项分流：
  · `derived_from: champion` + `numerically_valid: true` → 参与归因
  · `derived_from: baseline` → 进「旁证」区，**不参与**（它是另一条实验线）
  · `numerically_valid: false` → 进「旁证」区（计时探针，测的是耗时不是收益）
  · 真消融项若 delta 为负 → **标为不可 claim 并报「负归因几乎总是 spec 写错」**，
    不再当「负贡献」写进表

真实 mqa 数据复测（spec3）：三支探针全部进旁证，归因只剩 `no_bp`（share 0.039），
status 如实报 `unexplained`（96% 收益没归因到）。比之前全负 share 可读得多。

**当前防回退**：`Plugins/ops/tests/test_ablate.py` 新增 6 条 D9 用例：
  · 缺 derived_from / numerically_valid → 拒
  · baseline 派生项进侧栏、不进 items
  · 探针进侧栏
  · **负 delta 不许产出 share**（核心）
  · 旁证再多也不改变 sum_share

**这条债的价值**：它是「工具第一次遇到真实复杂场景就暴露设计缺口」的实例 ——
比在合成数据上跑 100 条测试更能说明问题。同时也说明：
**spec 的正确性没有校验器**，人和工具都可能把「探针」当「消融」。

---

## D10 · `moe-align-permute` 卡与 lightop 的算子不是同一个

**发现**：2026-11-25，做第 3 张卡时。

卡片 `math`：`dst = permute(src, slot_map)` —— 描述的是**搬运数据**。
`_common.py` 的 reference 也确实返回 `(permuted[total_slots, H], slot_map[M, K])`。

但卡上挂的 `import_path: lightop.moe_align_block_size` 实测返回：

```python
out = lightop.moe_align_block_size(idx, block_size=128, num_experts=128)
# (sorted_ids:int32[81792], expert_ids:int32[639], num_tokens_post_pad:int32[1])
```

**只算索引，一个字节数据都不搬**。

所以这两者不可直接比较：
  · 我们的 reference：读 x[M,H] + 写 permuted[total_slots,H] —— 访存主导
  · lightop：只读 topk_idx[M,K] + 写索引 —— 纯计算/小数据

拿它们比加速比，等于比两件不同的活（本仓最贵的一类错，已出现 5 次）。

**这是 D5/D8 之后第三次「卡片语义与源实现不符」。**

**处理选项**（需裁决）：
- **A. 改卡的 math**：把它定义成「算索引」(`sorted_ids`/`expert_ids`/`diagnostics`)，
  reference 相应改为只算索引。优点：与 lightop 对齐、可比。
  缺点：丢掉「permute 数据」这个更接近生产的部分。
- **B. 保留 math，换 baseline**：找一个真做 permute 的实现当对手
  （如 `lightop.groupgemm_permute` 或 aiter 的 permute 族），
  `moe_align_block_size` 作为「索引部分」的参照，不参与加速比。
- **C. 拆成两张卡**：`moe-align-index`（对齐算索引）+ `moe-permute-data`（搬数据）。

**本卡当前状态**：draft，ratio=null，无可复跑证据。**在裁决前不动它。**

**当前防回退**：无机器判据。若要防，可在 `check_card_semantics.py` 里加一条：
「卡的 variants 里 import_path 的返回结构必须与 reference 的输出数量/类型一致」——
但这需要真的调用库函数，成本较高。

---

## D11 · 单位换算算错过三次

**发现**：2026-11-25。三次都发生在人读数字时，不在代码里：

| 次 | 错法 | 错值 | 真值 |
|---|---|---|---|
| 1 | 把 121µs 的带宽安到 140µs 上 | 1664 GB/s | 1438 |
| 2 | `MB/ms × 1000` 当成 GB/s | 419576 | 420 |
| 3 | 同上，在 permute 卡上再犯一次 | 同上 | 同上 |

**根因**：`MB / ms` 本身就等于 `GB/s`（1 MB/ms = 1000 MB/s = 1 GB/s），
再乘 1000 就大了三个数量级。而 `floor.py` 里的公式是 `bytes_mb / t_us * 1000`
（注意是 **µs** 不是 ms），两个公式长得像、单位不同 —— 混用必然出错。

**处理**：
- `floor.py` 的注释里补一句单位说明
- 任何手算带宽前，先写一行量纲检查（如 `assert 100 < gbs < 3000`）
- 本文登记为「人读数字比代码更容易错」的实例：
  代码里的公式三年没出错，手算三次全错

**当前防回退**：无机器判据（人读数字无法机检）。
但可加一条：`floor.py` 输出 achievable 时做**物理上限校验**（> 2500 GB/s 就报警）。
