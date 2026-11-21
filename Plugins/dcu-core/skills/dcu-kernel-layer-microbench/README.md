# dcu-kernel-layer-microbench — DCU kernel 层微基准

DCU 上 LLM 推理 **kernel 层微基准**方法论。当 profiler 定位不到 decode 时间去向时，用微基准量化 kernel
开销与优化收益（BW100 / DTK / SGLang DCU build 实测）。

## 核心内容

- 微基准 rooflining（替代缺失的 GPU kernel profile）
- 各 ops 微基准配方（memory-bound / launch-bound 判定）
- **DCU 专属坑**（每类坑都实测过，会真实消耗时间）
- SGLang DCU build 下的 spec-decode 事实
- overlap 注意点

## 衔接

- 与 `dcu-perf-profiling`（全栈 profile）互补：profile 看不出 kernel 时间去向时，转本 skill 做 kernel 层量化。
- 量化出候选后，用 `dcu-op-optimization` 做优化闭环。

详见 `SKILL.md`。
