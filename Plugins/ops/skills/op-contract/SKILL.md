---
name: op-contract
description: "算子契约（contract.yaml）规范：把「这个算子怎么算对、算多快算够」变成机器可读的验收对象——形状/dtype/布局约束/容差分级/不变量/门槛/性能地板。被每张 op-* 算子卡引用，被 scripts/opverify.py 执行。触发场景：新增算子卡、给已有算子补契约、两个实现/变体之间的容差怎么定、判断某次优化算不算通过、把口头结论沉淀成可复验条目。不适用：单次怎么量（口径、噪声判定）走 op-bench；改动能不能合入的判定走 op-gate。"
version: 1.0.1
author: DCU Toolkit Contributors
license: MIT
kind: knowledge
---

# 算子契约（contract.yaml）规范

## 一、为什么要有契约

没有契约时，「这个优化通过了」= **模型自己觉得**。三种失败我们都遇到过：微基准 bit-identical 但生产掉分（输入布局 ≠ 生产 kernel 输出布局）；「对比 torch 参考误差 1e-7」但同一 token 输出依赖 batch 同伴（ASM 内核串读）；「快 9ms」来自把 600MHz 与 1150MHz 两张卡的绝对值相减。契约的作用是把这些问题**在写代码之前**变成必填项。原则一句话：

> **模型负责判断，脚本负责事实。** `SKILL.md` 给人看，`contract.yaml` 给 `opverify.py` 执行。

## 二、最小可用契约

```yaml
op: <算子名，与目录名一致去 op- 前缀>
version: 1
math: "<一行数学定义，含哨兵语义>"
shapes:
  <输入名>: [符号维度...]        # 用符号，具体数值在 shapes.yaml 档位段里给
  out: [...]
dtypes:
  in: bfloat16 | float16 | float8_e4m3 | int32
  acc: float32                   # 必须显式写：fp16/bf16 与 fp32 累加结论会相反
  out: bfloat16
layout_constraints:              # 只写会导致结论反转的布局事实
  - "<一句话>：<为什么>"
tolerance:
  vs_fp32_ref:                   # 与 fp32 参考实现的容差
    max_abs: 2.0e-3
    cosine: 0.9999
  variant_vs_variant: bit_exact  # 变体之间默认逐位一致（不 bit-exact 必须写理由与容差）
invariants:                      # 必须全过，见 §四
  - determinism
  - padding_invariance
  - batch_independence
gates:                           # 可执行的门，opverify 逐条跑
  - {name: correctness, template: correctness, required: true}
  - {name: invariants,  template: invariants,  required: true}
  - {name: perf,        script: bench/bench.py, require: "median_ms <= baseline_ms * 1.02", repeats: 3}
floor:                           # 性能地板：什么时候可以宣布「别再优化了」
  metric: hbm_bandwidth
  peak: 1200                     # 单位见 unit，物理常数，不是实测值
  unit: GB/s
  basis: "roofline：该算子每 token 必须读 X 字节，峰值带宽 Y"
  at_floor_threshold: 0.85       # 达成率 ≥ 85% 判 at_floor
```

## 三、字段判据

| 字段 | 判据 |
|---|---|
| `math` | 写明哨兵值语义（`-1` 无效索引、`-inf` 死区、`0` 填充）。这些是正确性 bug 的高发区 |
| `shapes` | 用**符号**，不放具体数值；具体数值进算子卡的 `shapes.yaml`，一档一个顶层键 |
| `dtypes` | `acc` 显式写。fp16/bf16 累加与 fp32 累加结论会相反 |
| `layout_constraints` | 只写**会导致结论反转**的布局事实。例：「kv 在 packed `[slots,132]` 下向量化被限制在 ≤4B，与分离平面的带宽差 2×，两者性能**不可跨布局比较**」 |
| `tolerance` | 分两级：`vs_fp32_ref`（宽松，检数值正确）+ `variant_vs_variant`（严格，检实现等价）。**默认 `bit_exact`**，不 bit-exact 必须写明理由与容差 |
| `invariants` | 见 §四。至少 `determinism`。**注意**：如果哨兵位置的支持度因变体而异（有的实现只支持后缀填充），那它就不是「每次 run 都要判」的不变量，而是**变体能力**——写进 `variants.yaml` 并单独出一个检查脚本 |
| `gates` | 每条要能**真的被执行**：优先写 `template:`（共享实现 `scripts/opcheck.py` 的子命令，见 §八），只在流程确实算子特有（如手写 `.so` 的构建核验）时才写 `script:` 指向卡内路径。跑不了的写 `required: false` 并注明 `skip_reason` |
| `floor` | `peak` 必须是**物理常数**（rocminfo/规格书的峰值带宽、峰值算力），不是实测值。`basis` 必须能复算出「这个算子至少要多快」 |
| `manifest` | 声明**要冻结哪些执行身份**（见 §六）。只写 role 与 env 键名，**不写路径**——路径由采集时命令行显式给 |

 > 缺项按失败处理。`opverify.py` 对任何 `required: true` 但缺数据/缺脚本的条目一律判 FAIL——宁可拦下，不糊过去。

## 四、不变量清单（通用 + 算子特定）

不变量和「对比参考」是两件事：参考对比抓数值偏差，**不变量抓实现结构错误**。

### 通用（所有算子都该有）

| 不变量 | 检什么 | 我们踩过的坑 |
|---|---|---|
| `determinism` | 同输入重复 N 次（默认 5），输出 bit-exact | M≥6 decode 非确定性（state-copy 用 batch_idx 索引错 block） |
| `padding_invariance` | padding/哨兵位置变化不影响有效位结果（**仅当算子定义确实位置无关时**） | topk 的 `-1` 填充位污染有效位；反过来也有坑：tilelang sparse-MLA 用「块首≥0」计数截断循环，只支持后缀填充，前置 `-1` 静默算错 → 这类写成变体能力而不是不变量 |
| `index_permutation_invariance` | 乱序输入索引不改变结果（gather 类算子） | 稀疏 attention 的 indices 顺序 |
| `batch_independence` | 同一样本在 batch 大小变化时输出不变 | ASM MoE 内核串读 batch 同伴数据（自洽性测试坐实） |
| `layout_invariance` | 输入满足声明布局的等价形式时结果不变 | packed vs 分离平面 |

### 算子特定（写进具体卡的 `invariants`）

- 注意力类：`split_invariance`（SplitKV 不同切分必须 bit-exact）
- GEMM 类：`shape_tiling_invariance`（不同 tile 配置结果一致）
- 归一化类：`eps_boundary`（接近 0 的方差处理）

## 五、地板（floor）怎么写才算数

地板是**唯一允许说「别再优化了」的依据**，所以必须可复算：

```yaml
floor:
  metric: hbm_bandwidth
  peak: 1200
  unit: GB/s
  basis: |
    每 step 必须读 indexer KV = rows × topk × 576 × 2B。
    batch=8/320k 时 = 32 × 2048 × 576 × 2 = 75.5 MB。
    峰值 1200 GB/s → 理论下限 63 µs/次。
  at_floor_threshold: 0.85
```

三条要求：
1. `peak` 是**硬件常数**，写来源（规格书 / `rocminfo` / 实测 STREAM）；实测 STREAM 要注明是实测。
2. `basis` 能被人用计算器验算一遍。
3. 结论只写「达成率 x%」与状态，不写绝对 µs（绝对数进 `kb/op-opt/`）。

**反例**（我们真犯过）：把「激活 expert 数线性增长的带宽」当成「45% 就还有 2× 空间」，gfx938 实测已 88.8% 峰值（其他 gfx 未验证）。地板必须贴带宽模型算，不能按直觉。

## 六、执行身份（`manifest` 段）

**要回答的问题**：这个 219.9µs，是**这份源码编出来的这个产物**跑出来的吗？

结论里所有数字都依附于「哪份代码、在什么环境」。没有这条链，下面三种错配事后完全看不出来：

| 错配 | 症状 |
|---|---|
| 源码改了没重建 | 测的是旧产物，改动其实没生效 |
| 产物被另一次构建覆盖 | 测的是别人的产物 |
| 进程加载的不是这个产物 | 测的是缓存里另一个（tilelang 缓存下有 1700+ 个 `executable.so`） |

契约里只声明**要哪些 role、哪些 env 键**，路径不进来（机器路径不进仓库，也猜不准）：

```yaml
manifest:
  required_roles: [source, loaded]          # tilelang 这类：编出来的就是加载的那份
  env_keys: [gfx, dtk, torch, python, sclk_mhz, device]
```

| role | 含义 | 典型填法 | 是否必填 |
|---|---|---|---|
| `source` | 算子源码 | `sparse_mla_fwd.py` | 必填 |
| `loaded` | 进程**实际加载**的那个产物 | 运行时从 kernel 的 `adapter.libpath` / `module.__file__` 抓 | 必填 |
| `build` | 编译产物 | 缓存里的 `executable.so` / 编译出的 `.co` | **仅当产物 ≠ 加载产物时**（如 `.co` 编译后拷进安装树），用 mtime 链查「编完没拷」 |

三条判据：

1. **`build` 与 `loaded` 至少要有一个**，且 `source` 必须在。只钉源码等于没钉产物（源码没变、产物被覆盖照样测错）；只钉产物则不知道是哪份源码编的。
2. **`loaded` 必须在运行时抓**，不能事后猜。事后按名字 grep 缓存目录抓不到（生成的代码里没有算子名）——这条我们实测过（2026-10-28：`~/.tilelang/cache` 下 1700+ 个 `executable.so`，生成代码里无算子名）。
3. **`env_keys` 的每个值都不能是占位符**。`unknown` / `unavailable(...)` / `<模板>` 一律算「没记录」，判定失败。env 键名必须用采集脚本真产出的名字（例：`sclk_mhz`，不是自造的 `clock_ghz`）——否则通道永远是占位符，闸永远 FAIL。

采集与复核（产物路径直接从 bench 报告里取；没有 bench 集成时用 `--artifact role=path` 显式给，glob 必须唯一匹配；`validate` exit 0 = sha256 与 mtime 链完好）：

```bash
python3 scripts/opmanifest.py record   --out <run>/manifest.json --from-bench <run>/bench.json --variant tilelang --env-json <run>/bench.json
python3 scripts/opmanifest.py validate --manifest <run>/manifest.json --contract <card>/contract.yaml
```

`opverify.py` 在**契约声明了 `manifest.required_roles` 时**自动做这道闸（闸名 `identity`），默认读 `<run>/manifest.json`：

- 没有 manifest → `REJECTED`「执行身份未冻结」
- sha256 不符 / 源码比产物新 → `REJECTED`，理由指到具体文件名
- 显式 `--allow-missing manifest` → 闸记 `skipped` 并留痕，**不会悄悄 pass**

契约没写 `manifest` 段 → 不出现这道闸，老契约的输出结构不变（兼容既有台账）。

> **不 fail-open**：`opmanifest.py` 自身异常一律转成「不一致」。判定脚本自己坏掉时写 `sys.exit(0)` 会让门静默退化（坏了也放行、没人察觉）——我们不抄。

## 七、常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 契约里写具体 shape | 换档位就失效 | 符号维度 + `shapes.yaml` 分档 |
| gate 脚本路径不存在 | 门永远「无法执行」→ 被跳过 | 要么写脚本，要么 `required: false` + `skip_reason` |
| 契约与 SKILL.md 各说一套 | 模型按文档、脚本按契约，冲突 | 改一处要改另一处 |

## 八、与工具的关系

```bash
# 跑门：流程由共享模板提供，判据读本卡 YAML（template: correctness / invariants / padding）
python3 scripts/opcheck.py correctness --contract <card>/contract.yaml --shape <档位> --variants a,b
python3 scripts/opcheck.py invariants  --contract <card>/contract.yaml --variant a [--tilelang-root <tl>]
python3 scripts/opcheck.py padding     --contract <card>/contract.yaml --shape <档位>
# 校验单次 run（exit 0 = 通过） / schema 自检
python3 scripts/opverify.py --contract <card>/contract.yaml --run runs/op-optimize/<id> --json
python3 scripts/opverify.py --contract <card>/contract.yaml --check-contract
```

### 共享门模板与 adapter（`scripts/opcheck.py`）

**不要再给每张卡抄一套测试脚本**：流程、报告字段、退出码只有一份。卡目录下放一个 adapter
模块（默认 `_common.py`），只写本算子的数学：

| adapter 符号 | 必需 | 作用 |
|---|---|---|
| `build_inputs(params, seed=, device=)` | 是 | 造输入，返回 `(inp..., meta)`；`meta` 里的口径字段（池大小、`effective_topk`、warning）随报告输出 |
| `reference(*inp, params, chunk=)` | 是 | 高精度 oracle（fp32 或更高） |
| `runner_kwargs(params)` | 是 | 传给 `get_runner` 的形状参数（宽度换了 `sm_scale` 就换，别在门里另算一份） |
| `get_runner(name, tilelang_root=None, **kwargs)` | 是 | 变体注册表，返回 `fn(*inp)` |
| `run_isolated(fn, *inp)` | 是 | 可信调用：输入副本 + 输出毒化，必须可只给 `inp` 调用 |
| `load_shape` / `compare` | 否 | 缺省由模板直接读 `shapes.yaml` / 用 torch 现算 |
| `split_runner` / `permute_valid` / `head_layout` | 否 | 对应 `split_invariance` / `index_permutation_invariance` / 布局能力门；缺了该门报 SKIP，不默认通过 |

card 侧只需在 `contract.yaml:gates` 写 `template: <子命令>`。`opcheck.py` 缺必需符号时退出码 2。

- `scripts/opverify.py` 只读 `contract.yaml`，**不读 SKILL.md**——文档写错不影响判定，契约写错立刻暴露。
- 编排器 `dcu-orchestrator` 的 `op-validate` 节点以同一份契约作为 acceptance 依据。
- 容差与不变量属于**可复用事实**，升级需走 `kb/` 晋升提案（见 `data/kb/README.md`）。
