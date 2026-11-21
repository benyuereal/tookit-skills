# Agent 驱动 kernel 优化 flow（KDA + BBuf，DCU 适配待验证）

> 来源：Nvidia Kernel Design Agents（mit-han-lab/kernel-design-agents）+ BBuf/AI-Infra-Auto-Driven-SKILLS。
> 2026-10-16 自 `operators/dcu-kernel-dev/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 十二、Agent 驱动 kernel 优化 flow（参考上游 Nvidia KDA + BBuf，DCU 适配待验证）

> **来源与标注**：源自 Nvidia Kernel Design Agents（KDA，`mit-han-lab/kernel-design-agents`，含 `prompts/basic-flow.md` + `docs/agent-flow.md`）+ BBuf 实战（`BBuf/AI-Infra-Auto-Driven-SKILLS` 的 `prompts/sglang-sota-*` + `model-pr-optimization-history`）。BBuf 用这套 flow（Fable + 简单 prompt + 2 个 skill）**2 周把 35 个 Diffusion 模型在 Eager 下性能全部持平/超越 Torch Compile**——即 Torch Compile 的 fuse 优化可被 Agent 自动逆向复刻成手写 eager kernel。
> **⚠️ 这是 CUDA 平台方法论，DCU/ROCm 上是否同样有效未经我们实测，仅作"参考、可尝试"用，禁止当作 DCU 已验证纪律。**

### 12.1 Task Contract 模板（每次优化任务起手必填）
直接复用 KDA 的 `basic-flow.md` 契约，DCU 化改写：
```
- Task name: <算子/模型名>
- Objective: <用户侧目标，如"把 X kernel 在 DCU 上持平/超越 Y">
- Correctness: <精度容差/不变量，如 max diff<0.03>
- Performance target: <可量化指标，如 端到端延迟/吞吐>
- Allowed approaches: <语言/库约束，如 TileLang/HIP/Triton + DTK 版本>
- Validation cmd: <证明正确性的命令，如 precision_test.py>
- Evaluation cmd: <测量目标的命令，打印 METRIC 行>
- Promotion: <候选被接受必须满足的条件>
```
**铁律**：draft 没写出来不准动手（与 §十、§十一 同构）。

### 12.2 最小循环（与 §十一 profile 循环对齐）
1. 定义 Task Contract → 2. agent 读本地代码/测试 → 3. 写 `docs/draft.md` →
4. 转可执行 plan → 5. 实现第一个候选 → 6. 验证正确性 → 7. 测 METRIC →
8. **记录证据并决定 keep/revise/reject** → 9. 循环直到满足 Promotion 或 blocker 显式。
- **晋升规则（KDA 原文）**：候选只有**满足契约 + 有证据证其改善/保住目标指标**才晋升；被拒的候选**记录原因**，不要静默丢弃（避免重复试错）。
- DCU 上每轮必须真机（有 `/dev/kfd` + hyhal 挂载）跑 `METRIC` 行，禁止凭估算。

### 12.3 证据留痕（KDA 推荐，DCU 落地）
任务工作区放：`docs/draft.md`、`docs/plan.md`、`benchmark.csv`（或表格日志）、`candidates.jsonl`（候选名+父链接+状态）、`profile/`（hipprof/rocprof 产物）。
> 与 `dcu-op-optimization` 的「输出物约定」模板一致：数字不进 skill，留痕在各项目。

### 12.4 与本项目既有流程的关系
- 本 § 是 §十（结构化 playbook）的**自动化/Agent 版**：把"约束→算法→正确性→性能"交给 Agent 按契约循环，而非人工逐步。
- 与 §十一（profile 驱动循环）**同一铁律**：只看 `METRIC` 行、瓶颈只能源于本次 profiler 报告、正确性门禁不过再快也废纸、git 为真相。
- **DCU 待验证项**：KDA 的 prompt/skill 在 CUDA 上跑通；DCU 上能否同样用 Agent 自动逆向复刻 fuse kernel，需你在 <device-A>/<device-B> 真机试——若发现 DCU 栈差异（如 TileLang/ROCm inductor 行为），按 §十二（原 §十二）的硬约束处理。
