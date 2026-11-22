---
name: cross-engine-acceptance-gate
description: 移植/换引擎后的精度验收门禁。bf16 跨引擎逐 bit 一致不可作硬门禁，用判别实验定性后分级放行。
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# cross-engine-acceptance-gate — 移植精度验收门禁（bf16 分级判据）

## When to Use

- 模型移植/换引擎/换镜像后，用户要求验证"输出与官方基线/原引擎一致"
- 门禁出现少量 token 分叉，需判定是 bf16 数值噪声还是移植缺陷
- 门禁不过且疑似真缺陷（数值崩坏/重复循环/A-B 不一致）时，转 `dcu-vllm-precision` 做逐算子定位——两 skill 正交：本 skill 管整体放行判定，那管算子级根因

## 核心认知

**bf16 贪心解码下，跨引擎逐 bit 一致本不可达。** attention kernel、累加顺序的任何差异都会在概率接近的 token 上让 argmax 翻面，之后整段合法漂移。硬卡"全匹配"会错杀正确实现。

## 门禁设计（实测：VeriLoop-E2 27B hybrid，DCU vLLM 0.28 vs HF transformers，23 条）

1. **建基线**：同权重 HF transformers，23 条覆盖 数学/代码/长文(8000+字)/中文 的 prompt，greedy（temperature=0），存逐字输出。serving 镜像缺 accelerate 时卷装注入（见 dcu-vllm-deploy §4.6）。
2. **跑门禁**：同批 prompt 发 vLLM（temperature=0、固定 max_tokens、**raw completions 端点绕开 chat 模板差异**），逐条 token 级 diff。
3. **判别实验定性分叉**（分叉≠失败，先定性再放行）：
   - **自一致性**：vLLM 同 prompt 连发两次应逐字节相同 → 排除非确定性。**前提：必须单并发串行**——并发下 batch 组成变化本身就会让 argmax 翻面，会误判"非确定性"。
   - **分块边界 A/B**：`--max-num-batched-tokens` 两档（如 8192 vs 32768）各跑一遍全同 → 排除 chunked scan/state 传递 bug（hybrid 线性注意力模型重点）。
   - **prefix-caching A/B**：开/关前缀缓存全同 → 排除 mamba state 池恢复 bug。
   - **fp32 控制组**：HF fp32 再跑一遍基线。若 fp32 vs bf16 自身就有分歧（实测 3/23），实锤"逐 bit 一致不可达"，分歧属 bf16 数值刀锋。
4. **分级放行判据**：无崩坏症状（乱码/重复循环/发散）+ 分叉条逐条语义审查通过（等价措辞/格式位置差）+ A/B 实验排除移植嫌疑 = 放行。真移植缺陷的形态：数值崩坏、重复循环、A/B 不一致、分叉点在输出极前部且语义偏离。

## 口径纪律

- "N/23 一致"这类具体数字与镜像版本强绑定，只作该版本观测值注记，**不作通用基线**。可复用的是方法论与结论（判定移植成功与否）。
- graph/eager 两遍门禁差异（cudagraph 融合 kernel 数值差推动更多敏感 token 翻面）是预期行为，记录即可，不作为缺陷依据。

## 实测记录（VeriLoop-E2，2026-10，仅作案例非通用基线）

- 内部 0.28.0-r3 镜像：eager 19/23、graph 15/23，分叉全为语义等价微差（措辞一字/加粗位置），四组判别实验全过 → 判定移植成功。
- 公开 0.28.1rc1 镜像：eager 17/23，新增分叉全在长文复读区 echo 边界（顿号取舍），语义零差 → 同判定。两镜像差异归因 kernel 版本数值差。
- 启动三坑（retention 整除性 / VLM 壳 / libcudart）见 `dcu-vllm-deploy` §4.5。

## 验收

1. 给出 bf16 分级判据的判定结果（通过 / 不通过 / 需人工）。
2. 对照侧与候选侧的输入完全一致（同一 tokenizer、同一 seed、同一 prompt 集）。
3. 差异最大的若干条能定位到具体层/算子。
4. 结论带样本数与统计口径（多少条、什么分布）。

## 不适用场景

同引擎内的小版本升级（差异不足以跨引擎对比）；
只想看单条样本的数值（那走 `opcheck` 的正确性门）；
对照侧本身不可信（先验证参照实现）。
