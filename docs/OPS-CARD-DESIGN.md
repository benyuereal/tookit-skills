# 算子卡域（ops）设计文档

> dcu-toolkit 1.6.0 新增的第 8 个插件域
> 阅读顺序：① 工具箱架构 → ② 算子卡架构 → ③ 怎么运转 → ④ 判定/沉淀 → ⑤ 实战与对比

---

## 1. 要解决什么问题

旧 7 个域回答「怎么做」。做算子优化时卡住人的是另一类问题：

> 这个算子是什么？有几种实现？哪个档位用哪个？判据是什么？上次结论是什么？

结论散在聊天记录、个人笔记、`/tmp` 里，导致三个反复发生的事故：

1. **重走老路**：别人已证伪的方向，又花一周重测；
2. **口径打架**：600MHz 与 1150MHz 相减，得出过 1.5× 假收益；
3. **模型自己拍板**：AI 说「通过了」，没人知道判据。

**答案：一算子一目录——契约 + 判定 + 台账。**

> 原则：**模型负责判断，脚本负责事实。** `SKILL.md` 给人看，`contract.yaml` 给 `opverify.py` 执行。

---

## 2. dcu-toolkit 架构

```
Plugins/
├── dcu-core          路由 + 编排（workflow 契约）
├── dcu-env           基础环境
├── dcu-perf          性能剖析方法
├── dcu-inference     推理框架适配
├── dcu-operators     算子开发（HIP/Triton、CUDA→HIP）
├── dcu-troubleshooting
├── dcu-engineering
└── ops               * 算子卡域（以算子为对象）
```

前 7 个按**能力**组织，ops 按**对象**组织。三者正交：

| 域 | 回答 |
|---|---|
| `dcu-operators` | 怎么写 kernel |
| `dcu-perf` | 怎么剖析、怎么闭环 |
| `ops` | 怎么判、怎么记 |

---

## 3. 算子卡架构：固定七件套

```
Plugins/ops/skills/op-sparse-mla/
├── SKILL.md            怎么用（给人/agent 读）
├── contract.yaml   *   机器可读契约（opverify 唯一事实来源）
├── variants.yaml   *   实现变体矩阵
├── candidates.yaml     可证伪候选清单
├── shapes/             形状档位
├── bench/              微基准 + PMC 数据
├── tests/              正确性 / 不变量测试
├── ledger.md       *   结论台账
└── references/
```

### 3.1 contract.yaml

```yaml
tolerance:
  vs_fp32_ref: { max_abs: 2.0e-3, cosine: 0.9999 }
  variant_vs_variant: { max_abs: 3.0e-3 }   # 实测 1.95e-3，故上限 3e-3
invariants:
  - determinism                  # bit_exact ×5
  - index_permutation_invariance
  - split_invariance
  - padding_layout
```

要点：容差必须带**实测依据**；布局约束要把会改变结论的事实写死（`-1` 是哨兵、kv 同张量兼作 K/V、packed 布局向量宽度限制）。

### 3.2 variants.yaml

```yaml
variants:
  - id: prod-vllm        # 生产 AOT ASM，无源码
    status: active
  - id: tilelang         # 可改
    status: usable
  - id: tilelang-prefetch
    status: usable
```

每条记录 `status / constraints / padding_support / mutates_inputs`。状态机：`draft → measured → tuned → at_floor → deprecated`，登记在 `data/ops/registry.json`。

### 3.3 ledger.md

只写**档位 / 变体 / 相对值 / 状态**；每行必须能回到一次可复跑的 run；作废结论不删，划掉写理由。

---

## 4. 怎么运转（竖向流程）

一次请求从进来到沉淀，经过 12 站。**每站只干一件事，交出固定格式的文件；下一站不认聊天记录，只认文件。**

```
① 人 · 提问
   「sparse-mla 这次改动算不算通过？」
        ↓ 自然语言
② router · 路由 / 转发（两条真实机制）
   dcu-skill-router：SKILL.md 里的七域路由表，覆盖约 140 个技能
   dcu-dispatcher：intent_classifier.py 程序化意图分类，管编码类四意图
   算子优化 → ops 域（op-sparse-mla / op-gate / op-optimize 工作流）
   模型适配 → dcu-inference 域（model-deploy / infer-optimize 工作流）
   → skill 集合 + 工作流选择
        ↓
③ orchestrator · 规划
   读 workflow：节点 / 依赖 / 并行度 / 审批级别 → 建 run 目录
   workflow_state.py init → workflow-state.json
   → 节点 DAG + runs/op-optimize/
        ↓ 节点计划
④ op-profile · 采集（只读）
   op-bench 口径计时 + PMC
   → profile/result.json（含环境指纹）
        ↓ 结果 + 指纹
⑤ verify_evidence · 门禁
   核对 path + fingerprint，缺一项就停
   workflow_state.py verify → 重算产物 sha256，变了就降级 stale
   → pass / stop
        ↓ pass
⑥ op-roofline · 推天花板
   判 memory/compute bound，算地板达成率
   → roofline/result.json
        ↓ cost_bound
⑦ op-constrain · 硬约束
   映射 allowed_actions，越界方案直接退回
   → candidates.yaml
        ↓ 候选清单
⑧ 人 · 审批 ①
   改代码/编译属高风险，proceed: ask 写死在 YAML 里
   → 人工决策
        ↓ 批准
⑨ op-optimize · 实施
   单点改动 + opverify 自检 + 可回退
   → optimize/result.json
        ↓ 改动 + 自检
⑩ op-validate · 判定
   四闸四档，verdict 只认脚本输出
   → validate/result.json
        ↓ ACCEPTED
⑪ 人 · 审批 ②
   写持久知识属高风险
   → 人工决策
        ↓ 批准
⑫ op-distill · 沉淀 → 复用
   五段台账；workflow_state.py finish 过三道收尾门禁
   → kb/op-opt/*.md → 下次秒回
```

> 每个节点完成时都会 `mark --status done --artifact <path>`，把产物指纹写进 `workflow-state.json`。

### 4.1 数据接力

```
runs/op-optimize/
├── profile/result.json      ← 指纹 + 热点
├── roofline/result.json     ← 天花板
├── constrain/candidates.yaml
├── optimize/result.json     ← diff + 自检
├── validate/result.json     ← verdict
└── distill/…                → 知识库
```

交接契约：`path + fingerprint（gfx/dtk/image/model）+ status`。**上一站没产出或指纹对不上，下一站不启动。**

### 4.2 三个角色

| 角色 | 负责 | 不负责 |
|---|---|---|
| **Agent** | 理解意图、出计划、写代码、总结 | 不决定数字 |
| **Script** | 计时、判定、校验指纹 | 不做意图理解 |
| **人** | 两处审批：改代码前 / 写知识前 | 不算每一个数 |

风险分级（`proceed: auto / ask`）写在 workflow YAML 里，不靠 agent 自觉。

### 4.3 状态轨：workflow_state.py

`Plugins/dcu-core/skills/dcu-orchestrator/scripts/workflow_state.py`（纯标准库，带文件锁 + 原子写），管 `<run-dir>/workflow-state.json`：

| 子命令 | 作用 |
|---|---|
| `init` | 建状态文件，登记节点清单 |
| `mark` | 节点完成时写 status + 产物 `{sha256, size, mtime}` |
| `verify` | 重算哈希；内容变了 → 降级 `stale`，退出码非零 |
| `inspect` | 查看进度（人类可读 / `--json`） |
| `finish` | 收尾门禁 |

**收尾门禁**（任一不过就拒绝收尾）：

- `MISSING_ROOFLINE_EVIDENCE` — 缺 roofline 证据；
- `UNRESOLVED_CEILING` — `measured_us > ceiling_us × 1.5`；
- `UNFINISHED_NODES` — 还有节点没跑完。

两个价值：**断点续跑**（跑到一半断了不用从头来）、**产物不留白**（结果被改过会被判 `stale` 并强制重跑）。

---

## 5. 判定：四闸四档

```
gate0 执行身份 ──fail──► REJECTED（数字配不上代码）
   │  源码/产物/实际加载的 .so sha256 一致；指纹非占位
gate1 正确性 ──fail──► REJECTED
gate2 不变量 ──fail──► REJECTED（参考过了也不行）
   │
gate3 A-B 性能 ──噪声内──► INCONCLUSIVE
   │显著
ACCEPTED
```

四档：`ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC-ONLY`。

- gate0 最便宜也最常被跳过——测出的 219.9µs 可能来自缓存里的另一个 `.so`（真实踩过）；
- INCONCLUSIVE 不许当通过；
- REJECTED 必须归因「方向错」还是「实现错」，**不许把实现 bug 记成方向证伪**。

工具：`scripts/opverify.py`，退出码可直接进 CI。

---

## 6. 口径与沉淀

### 6.1 微基准口径三问 + 六条铁律（op-bench）

三问：**测什么**（kernel 单发 / 算子调用 / 端到端）、**含不含分配**、**冷还是热**。答不出就别测。

六条铁律对应六次真实误判：
1. 输入布局必须与生产 kernel 输出一致（用生产路径 dump 的真实张量）；
2. 跨卡只比 ratio，不比绝对 µs（同卡同进程交替）；
3. 折算有效量（稀疏 padding、MoE 未激活 expert）；
4. MAD 显著性（噪声内的差值不是收益）；
5. 地板判定（离理论天花板多近）；
6. 冷热两轮。

工具：`scripts/opbench.py`，禁止手搓 `time.time()`。

### 6.2 五段台账

```
## 结论       改了什么、收益多少、正确性如何
## 前提       结论在什么条件下成立
## 已证伪     排除了哪些方向 ← 最有价值
## 复现命令   能重现数字的确切命令
## 口径       计时/频率档/分母/指标定义
```

纪律：生成方只填能确定性推出的部分，**推不出的段留空占位，不编造**；`kbq.py index` 机器检查缺段；每个数字必须能回溯到 `result.json`。

---

## 7. 质量保障：CI 不需要 DCU

```yaml
- python3 -I scripts/run_evals.py
- python3 -I -m unittest discover -s Plugins/ops/tests -p 'test_*.py'
```

卡的结构性质量（契约字段、台账五段、variants 声明一致性）在 CI 就能验，上机的只有真实性能与正确性。

---

## 8. 实战：HIP 版 flash_mla 的真实站位

场景：GLM5.3 架构，**H=16**，pool=full，同进程交替执行。

### 8.1 GLM 完整档位站位

`hip / flash` = HIP 耗时 / flash_mla 耗时，**<1 表示 HIP 更快，>1 表示更慢**。

| S | ctx | hip / flash | 判定 |
|---|---|---|---|
| 1 | 128k | 0.28 | **领先 3.6×** |
| 16 | 128k | 0.36 | **领先 2.8×** |
| 32（decode-glm53 主档） | 128k | 0.67 | **领先 1.49×** |
| 64 | 128k | 1.33 | 落后 1.33× |
| 128 | 320k | 1.32 | **落后**（目标区间） |
| 256 | 320k | 1.34 | **落后**（目标区间） |
| 512 | 320k | 1.38 | **落后**（目标区间） |

**结论：decode 档（S ≤ 32）领先 1.49~3.6×；S=64 与 320k 目标区间落后 1.32~1.38×。不能一句话概括为「领先」或「落后」。**

### 8.2 320k 详细对照

| S | hip-glm | flash_mla | tilelang | hip / flash | tile / flash |
|---|---|---|---|---|---|
| 128 | 0.9520 ms | **0.7238 ms** | 0.8402 ms | **1.315** | 1.161 |
| 256 | 1.8806 ms | **1.4077 ms** | 1.6203 ms | **1.336** | 1.151 |
| 512 | 3.6813 ms | **2.6627 ms** | 3.1478 ms | **1.383** | 1.182 |

（`hip / flash`、`tile / flash` 为耗时倍率，**>1 表示更慢**。）

**320k 目标区间：落后。**

- HIP-GLM 比 flash_mla 慢 **1.32~1.38×**；
- 比 tilelang 也慢约 **13%~17%**；
- 该区间三个档位均落后（decode 档的领先见 §8.1）。

**这组数据说明了什么：**

1. **skill 保证方向，不保证赢**——skill 路径让实现可验证、可复跑，但性能上限要靠额外工程；
2. 硬件 `buffer_load`、`aicc+mllvm` 编译都带来过实际提升，但只缩小了差距，没能反超；
3. 下一步方向：**gather 访存与预取、tile 策略、split 配置**；
4. **诚实记录**：该实现进 `variants.yaml` 时状态是 `draft`，而不是「已完成」——这组数据原样进台账，作为下一次优化的起点。

### 8.3 历史记录：H=16 早期一轮（保留）

> 保留作演进记录。该轮比值口径为 **>1 表示更快**，与上表「>1 表示更慢」不同；两轮采集口径不同，**不可混比**。

| 320k · H=16 | S=128 | S=256 | S=512 |
|---|---|---|---|
| 仅 skill（标量读写） | 0.424 ms（0.87×，**落后**） | 0.840 ms（1.27×，**领先**） | 1.655 ms（1.28×，**领先**） |
| + 硬件 buffer_load | 0.365 ms（≈1.00×，持平） | 0.727 ms | 1.443 ms |
| + aicc + mllvm | **0.327 ms（1.13×，领先）** | **0.651 ms（1.64×，领先）** | **1.291 ms（1.65×，领先）** |

**关键：仅走 skill 时，S=256 / S=512 已经领先（1.27× / 1.28×），只有 S=128 落后（0.87×）。** 加上编译优化后，三个档位全部领先。这一轮记录的是「标量 → buffer_load → 编译」三步带来的相对改善，也说明 skill 路径本身能覆盖中大 S 档位，下沿 S=128 是短板。

复现命令：

```bash
python3 bench/bench_long.py \
    --s 128,256,512 --ctx 327680 --h 16 --topk 2048 \
    --variants hip,flash_mla,tilelang \
    --pool=full --repeats=5 --warmup=40ms
```

---

## 9. 怎么用

| 场景 | 入口 |
|---|---|
| 现在哪个实现最好 | `op-sparse-mla/ledger.md` |
| 改完算不算通过 | `python3 -I scripts/opverify.py …` |
| 跑完整流程 | `op-optimize` 工作流（两处 ASK） |
| 新增算子卡 | `Plugins/ops/skills/op-<name>/` + registry 登记 draft |
| 写新实现 | 进 `variants.yaml`（draft）→ tests / opbench / opverify |

---

## 10. 当前状态

- 方法论三篇 + 首版三张卡：`op-sparse-mla`（measured）、`op-moe-w8a8`、`op-gemm-fp8`（draft）；
- HIP 实现的 flash_mla 算子已完成初步优化：**decode 档（S ≤ 32）领先 flash_mla 1.49~3.6×**，但**320k 目标区间（S ≥ 128）仍落后 1.32~1.38×**、落后 tilelang 约 13%~17%（见 §8），尚未正式登记进 variants；
- 代码在 `feat/ops-operator-cards` 分支，**尚未 commit**（ops 域核心文件仍是 untracked）——合并前先跑 `run_evals.py` + 单测；
- `prod-vllm`（生产 AOT ASM）尚未纳入基准，是下一步优先事项。

---

## 11. 与 cannbot 的对比

> 参考 cannbot（`gitcode.com/cann/cannbot`）可见目录结构。**cannbot 记「怎么生产」，ops 记「怎么验收」。**

### 11.1 目录结构

**cannbot**（Ascend NPU，生成流水线）：

```
archive_tasks/
└── flash_attention/
    ├── design/
    ├── kernel/
    │   ├── op_host/  op_kernel/  utils/
    │   ├── CMakeLists.txt  build.sh
    │   └── ops.h  register.cpp
    ├── model.py
    ├── model_new_ascendc.py
    └── model_new_tilelang.py
```

**ops**（Hygon DCU，验收与沉淀）：`contract.yaml` / `variants.yaml` / `candidates.yaml` / `shapes/` / `bench/` / `tests/` / `ledger.md`。

### 11.2 对比

| 维度 | cannbot | ops |
|---|---|---|
| 核心动作 | 生成（tilelang → AscendC） | 判定 + 沉淀 |
| 变体 | 同目录多文件 | `variants.yaml` 结构化矩阵 |
| 正确性 | 模板 + 人工 | tests + 四闸，机器可判 |
| 性能口径 | 未见统一口径 | 三问 + 六铁律 + MAD |
| 沉淀 | 代码与 design 文档 | 五段台账 + 已证伪 |
| 构建 | CMake + build.sh | 不绑定构建系统 |
| CI | 未见算子级质量门 | 静态检查 + 单测，无需 DCU |
| 最强 | 生成流水线、工程闭环 | 验收纪律、结论可复用 |

### 11.3 互相借鉴

- **ops 可借鉴**：补 `design/`（设计意图落纸）、补 `kernel/` 与构建脚本；**构建配置本身就是变体的一部分**。
- **cannbot 可借鉴**：生成的 kernel 也需要 `contract`；性能需要统一口径；每次改动应留可复跑台账。

> 一句话：cannbot 回答「怎么生产出来」，ops 回答「做出来的能不能信」。理想形态是生成流水线 + 验收契约。

> ※ 本对比基于 cannbot 仓库可见目录结构（2026-10），未做全量代码审计。
