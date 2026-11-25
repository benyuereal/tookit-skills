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

---

## D12 · 共享卡被占满，gen2 迭代做不出可信判定

**发现**：2026-11-25 19:30，做 `fused-add-rmsnorm` 的 gen2 迭代时。

### 症状

以 gen1 赢家（`fa-rms-reg`）为基线跑 A/B：

```
# fa-rms-mr2: 0.1648ms (0.9572x) 在噪声内（|Δ|=0.00736 <= 3.0*MAD=0.0312）
# fa-rms-mr4: 0.1666ms (0.9675x) 在噪声内（|Δ|=0.00560001 <= 3.0*MAD=0.0312）
```

**MAD = 0.0312（约 19%）** —— 正常情况下是 0.0005（0.3%）。噪声大了 60 倍。

### 根因

```
HCU[0] use: 100%    vram 132GB / 147GB
HCU[1] use:  78%    vram 137GB / 147GB
HCU[2] use: 100%    vram 137GB / 147GB
HCU[3] use: 100%    vram 137GB / 147GB
HCU[7] use:   ?     vram 139GB / 147GB   ← 我们用的卡
```

CLAUDE.md 的硬约束写明「每卡 used 需 < 30 GB」，实际 139 GB。
**卡被别的进程占满了**（不是本容器的）。

### 为什么不能"凑合判"

`judge()` 已经正确地把三个变体都判成 `INCONCLUSIVE`，这是**判据在起作用**。
如果放宽 MAD 阈值强行判"mr2 比 reg 快 4.3%"，就是在噪声上做决策 ——
本仓已经因此犯过错（「快 24%」同批复测只有 1.7%）。

### 处理

1. **gen2 暂停**，等卡空出来再跑
2. gen2 的**代码已就绪**（`kernels/fa-rms-reg-mr.hip`，2 行/4 行两版，都 private=0，
   正确性 1 ULP 通过），只是测不出可信数字
3. 已有的 gen1 封存物**不受影响**（它是卡空闲时测的，MAD=0.0005）

### 教训

**`self_evolve.py --seal` 应该在封存前检查卡的状态。**
现在它只读历史记录，不看「测这些记录时卡是不是干净的」。
可加一条：封存时检查 `base_mad/base_ms < 0.01`，超过就警告「这条记录可能不干净」。

**当前防回退**：无机器判据。这条登记为已知缺陷。

---

## D13 · FlashMLA-das 的 fp8 kernel 只支持 d=576，GLM-5.3 需要 512

**发现**：2026-11-25，读 `/workspace/FlashMLA-das` 源码时。

### 症状

`csrc/api/dense_decode_kvfp8_hip.h:75` 的 API 契约：

```cpp
TORCH_CHECK(head_size_k == 576 || head_size_k == 512,
            "Only head_size_k == 576 or 512 is supported");
```

**但 kernel 层只实例化了 576**：

```bash
$ grep -o '<[^>]*>' csrc/extension/*.cu | sort -u
<bfloat16_t, 576> <bfloat16_t,bfloat16_t,576>
<float_e4m3_t,bfloat16_t,576> <half_t, 576>
```

`flash_fwd_mla_kernel_fp8_hip.h:2925` 更硬：

```cpp
template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla_fp8(...) {
    static_assert(Headdim == 576);          // ← 传 512 直接编不过
    using Kernel_traits = ...<576, 16, 64, 4, ...>;   // ← 内部还硬编码 576
```

### 影响

**GLM-5.3 flash 的 d_qk == d_v == 512（无 rope 尾），用不了 fp8 路径。**

- DeepSeek DSA / V4.1 是 576（512 latent + 64 rope tail）—— 它们能用
- GLM-5.3 是 512 —— **API 说支持，底层没有**

### 处理

已改（2026-11-25）：

1. `csrc/extension/flash_fwd_mla_kernel_fp8.h` —— 把函数内 5 处 `<576, ...>` 改成
   `<Headdim, ...>`，`static_assert` 放开到 `576 || 512`
2. `csrc/extension/flash_fwd_mla_fp8_gfx938.cu` —— 加 512 的模板实例化
3. 同步 `_hip.h`（hipify 产物，构建时会重新生成，但本地编译需要）

**注意**：改的是 `.h` / `.cu` 源文件，**不是 `_hip.h`**。
第一版改错了对象（改了 hipify 出来的 `_hip.h`），后来发现 `setup.py` 只编 `.cu`。

### 待验证

- [ ] 编译通过（进行中）
- [ ] d=512 路径在 GLM-5.3 形状上正确性通过
- [ ] 性能对比：fp8 512 vs bf16 512

**这是「API 契约与实现不一致」的实例**，和 D10（卡片 math 与源实现不符）同类。

---

## D14 · FlashMLA fp8 的 d=512 编译通过但运行 VMFault（D13 的后续）

**状态**：🔴 **未解决**。已定位到 3 个层次，修了 2 个，第 3 个没找到。

### 已修的（两个真实 bug）

**① kernel 名字硬编码** —— `flash_fwd_mla_kernel_fp8_hip.h:2892`

```cpp
std::string kernel_name = params.is_causal ?
    "_ZN5flash..._qkvfp8ILi576ELi16E...Lb1E..." :   // ← 写死 ILi576E
    "...";
status = hipModuleGetFunction(&flash_mla_func, mod, kernel_name.c_str());
status = hipModuleLaunchKernel(flash_mla_func, ...);   // ← 返回值没检查
```

512 的 kernel 编译出来了，但按 576 的名字查不到句柄。返回值没检查，
拿 `nullptr` 去 launch → 崩在 AQL 层，只看到垃圾 `grid.z = 3000719448`。

修：名字按 `Kernel_traits::kHeadDim` 动态拼 + 加空句柄检查。

**② 模板参数名** —— 那个作用域里没有 `Headdim`，只有 `Kernel_traits`。
错误用 `Kernel_traits::kHeadDim`。

### 未解决的（第 3 层）

修完上面两个后，**512 仍然 VMFault**，但错误形态变了：

```
>>>>>>>> KERNEL VMFault !!!! <<<<<<
```

不再是垃圾 grid（说明句柄对了），是 kernel 内部越界。

### 已排除

| 假设 | 排除方式 |
|---|---|
| 模板没实例化 | `nm -D` 有 44 个 `ILi512E` 符号 |
| `kHeadDimV <= kHeadDim` 不满足 | 512 <= 512 ✓ |
| stride 按 576 算 | `params.q_row_stride = q.stride(-3)` 是动态的 |
| smem 布局写死 | `SmemLayoutQ/O` 由 `kHeadDim`/`kHeadDimV` 模板化 |
| asm 特化路径的问题 | `FLASH_MLA_ENABLE_ASM=0` 也崩 |
| 512 与 576 的平台差异 | 576 同参数跑通（`out (1,32,64,512)`） |

### 进展：主 kernel 已跑通，combine 仍崩（2026-11-25 20:15）

修完上面两个后重测，**形态再变一次**：

```
✓ d=512 跑通! out: (1, 32, 64, 512)      ← 主计算 kernel 成功返回
>>>>>>>> KERNEL VMFault !!!! <<<<<<       ← 之后 combine 阶段崩
```

用 `FLASH_MLA_DEBUG_SMEM=1` 打开诊断后主 kernel 就过了（说明 smem 没超），
所以 **smem 假设排除**，问题精确定位到 **combine 阶段**。

### 对照组（同机同代码）

| d_qk | 主 kernel | combine | 完整流程 |
|---|---|---|---|
| 576 | ✓ | ✓ | **✓ out (1,32,64,512)** |
| 512 | ✓ | ✗ VMFault | ✗ |

**576 用完全相同的改动跑通** —— 所以不是我改动引入的，
是 512 路径本身有未适配处。

### 下一步的精确方向

`flash_fwd_splitkv_mla_combine_kernel` 在 `flash_fwd_mla_kernel.h:4300`
（注意：fp8 路径复用的是 bf16 的 combine，不是独立的）。

它用 `row_offset_lseaccum = split_offset * hs + hs_idx`，
其中 `split_offset` 来自 `num_splits` 前缀和、`hs = params.h * params.seqlen_q`。

**最可疑的假设**：576 时 `d_v(512) < d_qk(576)` 严格成立，
512 时两者相等 —— 任何 `if (d_v < d_qk)` 的分支都不会进。
要在 `compute_attn_*` 里搜这类分支。

### 教训

**「编译通过 + API 文档说支持」不等于「能跑」。**
这个 fp8 kernel 的 512 支持是半截图：模板参数化了、实例化加了、
API check 写了，但查表那一步漏了，而且错误检查也没有。
第一个错误（崩溃）完全没有指向性 —— 只看到垃圾 grid 值。

**判断一段「已支持某形状」的代码能不能信，必须真跑一次。**

---

## D14 结案 · 根因不是 d=512，是 H≥64 走了未初始化指针的分支

**结论先行**：把 `d_qk` 改成 512 完全是被冤枉的。
真正的原因在 **API 层一个 `if/else` 只在一个分支里初始化了指针**。

### 根因（一行代码）

`csrc/api/dense_decode_hip.h:166`（修复前）：

```cpp
if ((num_heads_q < 64 && num_heads_k == 1) || num_heads_k > 1 || arch.is_gfx928()) {
    params.tile_scheduler_metadata_ptr = (...)->data_ptr();   // ← 只有这里赋值
    ...
} else {
    // 从来不赋值
}
...
run_mha_fwd_splitkv_mla_fp8(...);   // 第 87 行却直接用
```

fp8 主 kernel 第一件事就是解引用它
（`flash_fwd_mla_kernel_fp8_hip.h:2615`）：

```cpp
int *p = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
int4 meta = *(reinterpret_cast<int4 *>(p));   // 未初始化 → VMFault
```

**H ≥ 64 走 else 分支 → 指针是调用方残留值 → VMFault。**

### 怎么定位到的（这段比结论值钱）

前 8 个假设全错，包括我自己写的「d_v < d_qk 分支」。
真正把它逼出来的是**按变量扫参数**，不是按猜测改代码：

| SQ | H | SKV | 结果 |
|---|---|---|---|
| 16 | **32** | 32768 | ✓ |
| 16 | **32** | 65536 | ✓ |
| 32 | **64** | 32768 | ✗ |

`SKV` 翻倍没事、`H` 从 32 到 64 就崩 —— 分界点精确落在
代码里的 `num_heads_q < 64` 上。**这个 64 是源码里的字面量，
不是硬件限制，所以它一定是代码分支，不是形状问题。**

### 过程中两个把自己带偏的坑

**① 设备端 printf 在崩溃时丢失。**
我在 combine kernel 里加 printf 定位，`.so` 里确实有字符串（`strings` 6 处），
但运行时一个字都没打出来。原因是 kernel 崩了，设备端缓冲区直接丢弃。
**结论：崩溃场景下 printf 不可靠，不能用「没输出」推断「没执行到」。**

**② `_hip.h` 是 hipify 产物，和 `.h` 要手工同步。**
我改了 `.h` 没同步 `_hip.h`，编译用的却是 `_hip.h`，白等一轮编译。
而且两边字段名/id 可用性不同（`.hip` 里没有 `getenv`，`params` 里叫 `d` 不叫 `d_qk`）。

### 教训

1. **「改 A 之后 B 崩了」不等于「A 导致 B 崩」。**
   我花了两轮在 d=512 上找原因，实际和 512 无关。
   应该更早做「只改 H」这个单一变量对照。

2. **同一份 `params`，两个分支只初始化一半，是这类 bug 的温床。**
   共同字段应该在 if 之前赋值。已按此修复。

3. **报错信息越少，越要先做「变量扫描」而不是「代码猜测」。**
   `KERNEL VMFault` 没有任何信息量，此时逐个参数二分比读代码快。

---

## D15 · 编译缓存与手工补丁的时序陷阱（本次实际踩到）

**现象**：改了 `dense_decode_hip.h` 的修复，`--check` 确认补丁在文件里，
编译也报成功，但跑起来仍然 VMFault。反复两次。

**根因**：`setup.py build_ext` 按时间戳决定是否重编。观察到的时序：

```
21:16:23  补丁写入 dense_decode_hip.h
21:16      api_hip.o 已编译完成（早于补丁，内容不含修复）
           → 链接成功、运行崩溃、看起来"修了没用"
```

注意 `ls` 只显示到分钟，肉眼看不出先后。

**判据（必须机器化）**：

```bash
# 改完源文件后，先确认 .o 比源文件新
[ build/.../api_hip.o -nt csrc/api/dense_decode_hip.h ] || rm -f build/.../api_hip.o
```

**已做**：`scripts/fmla_patch.py --check` 确认补丁在文件里。
**待做**：把「`.o` 必须比源文件新」也加进 `--check`，否则还是靠人看。

## D15 附带：文件"神秘消失"两次

同一份改动两次从工作区消失，`cp` 备份救回。**未查明原因** ——
那次仓库有一次 `clone: from github.com:HYGON-AI/FlashMLA-das.git`
（reflog 可见），怀疑是自动化流程重新拉取覆盖，但没有确证。

**应对**：`fmla_patch.py` 让补丁可重现，备份放 `/workspace/.fmla_patch_backup/`。
**不要**在没确认的情况下假设改动能留住。

---

## D16 · 验证器本身失效：`--check` 报"补丁在"，其实没打

**这是今天最严重的一个 bug，因为它是"守门人"自己坏了。**

### 现象

```
$ python3 -I scripts/fmla_patch.py --check
✓ 未初始化指针（H>=64 崩溃的根因）      ← 说补丁在
$ grep -c "两个分支都必须赋值" csrc/api/dense_decode_hip.h
0                                        ← 其实不在
```

### 根因（一行判据写得太聪明）

第一版 `_already_applied()` 的判据是「取 new 里最长的一行代码，
它在文件里就算打过补丁」：

```python
code = [ln.strip() for ln in new.splitlines()
        if ln.strip() and not ln.strip().startswith("//")]
return max(code, key=len) in s
```

问题在于**这个补丁的 new 里，最长的那行本来就存在于原文件**——
补丁只是把

```cpp
params.tile_scheduler_metadata_ptr = (...)->data_ptr();
```

从 `if` 里面复制到 `if` 之前。所以：

```
new in s         : False   ← 补丁确实没打
old in s         : True    ← 锚点还在，铁证
_already_applied : True    ← 但启发式说"打了"
```

### 为什么没被 selftest 抓到

当时的 5 项 selftest 测的是「未打补丁时 check 返回 False」，
用的是 `PATCHES[1]`（`static_assert` 那个补丁），它的 new 只有一行、
且那行确实不存在于原文件。**用另一个补丁测，测不出这个补丁的 bug。**

### 修复

1. **判据顺序改对**：`old in s` 优先 —— 锚点还在就一定没打，
   不允许被任何启发式覆盖。
2. 加第 6 项 selftest，专门复现「old 还在但某行已存在」这个场景。

### 教训

**「验证器通过」本身需要被验证，而且要用会失败的输入验。**
如果 selftest 只用一种补丁形态，它就是在测自己的实现细节，不是在测判据。

这和 `D12`（gpu_guard 把可用卡全判死）是同一类错误：
**判据写反/写松，比没有判据更危险 —— 因为它会让人以为已经检查过了。**
