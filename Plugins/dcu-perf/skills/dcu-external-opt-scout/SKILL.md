---
name: dcu-external-opt-scout
description: "Vet GitHub GPU/DCU optimization projects for DCU testing."
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# DCU 外部优化项目侦察与筛选

## When to Use

用户要求在 GitHub 上找 GPU/DCU 的 kernel 或推理优化类项目，并判断哪些值得在 DCU 上实测、价值点是什么。典型触发："去 github 查一下有没有同架构/类似架构的优化项目，筛选出最有价值的一批"、"看看哪些可以在 DCU 上测一下"。不适用于：单纯查某个已知项目的用法（直接读其 README）、部署/适配某个具体软件包（用 `dcu-software-adaptation-guide`）、自己写 kernel（用 `dcu-kernel-dev`）。

## 概述

给定一台 DCU（先确认其 gfx 架构），在 GitHub 上找同/近架构的优化类项目，**读源码验证技术细节**后，判断哪些值得在 DCU 上实测、价值点是什么。核心纪律：**不信任 README/博客的自报数字，必须 clone 下来 grep 源码验证关键声明，并用 Amdahl 定律判断端到端收益上限。**

## Step 0 — 先确认目标 DCU 的 gfx 架构

一切筛选以架构为锚。连设备查（见 `dcu-ssh`）：
```bash
ssh <alias> "source /opt/dtk/env.sh 2>/dev/null; rocminfo 2>/dev/null | grep -iE 'gfx|gcn' | head"
```
`rocminfo` 的 `Name: gfxXXX` 是权威来源（`hy-smi` 概览不直接给架构）。记下 gfx 号 + 卡型 + DTK 版本 + 空闲卡号（`hy-smi` 看 VRAM%/HCU%，只挑空闲卡）。

## Step 1 — 按架构名搜索，不是按 "dcu"

用 GitHub Search API（未认证 10 req/min，`sleep 8` 间隔）。**多组查询并行覆盖**：
- 目标架构名本身：`gfx936`（DCU 专属，命中少但最对口）
- 最近 AMD 架构：见 `references/arch-mapping.md`（gfx936 最近是 gfx942/CDNA3，方法论可参考 gfx90a/CDNA2）
- 通用优化关键词：`triton AMD GPU`、`composable kernel`、`flash attention ROCm`、`hip kernel optimization`、`aiter`、`rocm quantization kernel`
- DCU 专项：`vllm dcu`、`sglang dcu`、`dcu inference optimization`

```bash
curl -s -H "Accept: application/vnd.github+json" \
  "https://api.github.com/search/repositories?q=$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "<query>")&sort=stars&order=desc&per_page=8" \
| python3 -c "import json,sys;d=json.load(sys.stdin);[print(f\"{r['stargazers_count']:>5}* {r['full_name']:<42} {r['pushed_at'][:10]} {(r['description'] or '')[:80]}\") for r in d.get('items',[])]"
```

**分档认知**：DCU 专属架构（gfx936）项目 star 极少（个位数）但环境最对口；AMD 近架构（gfx942/CDNA3）项目 star 多但需移植。两档都要看，不能只按 star 排序——star 高 ≠ 对口。

## Step 2 — 拉 README 初筛，再 clone 读源码验证（关键纪律）

README 初筛后，**对候选项目 `git clone --depth 1`，grep 源码验证博客/README 里的关键技术声明是否真实存在**。这是本 skill 的核心，跳过它等于把自报数字当事实：

```bash
git clone --depth 1 https://github.com/<owner>/<repo> <dir>
# 验证博客声称的指令/技巧在 .cu/.hip/.py 里真实存在：
grep -n "v_mmac_i32_16x16x32_i8\|v_mfma\|shfl_xor\|__shfl_xor" <dir>/csrc/**/*.cu
grep -n "laneid % 16\|lane_col\|fragment" <dir>/csrc/**/*.cu
```

验证要点：
- 声称用了某条矩阵指令（MMAC/MFMA）→ grep 内联汇编 `asm volatile` 确认指令字符串真在
- 声称做了 fragment/lane 映射 → grep 映射公式（如 `laneid % 16`、输出列交织 `+ {0,4,8,12}`）
- 声称 wave64 归约 → grep `__shfl_xor`（DCU wavefront=64，不是 32；来自其他架构的 DPP 归约不能直接套）
- 声称的量化/解码技巧 → 读对应 kernel 函数体确认逻辑

**博客/README 与源码不符 → 该项目降级或剔除。** 源码验证通过才谈价值。

## Step 3 — Amdahl 定律判断端到端收益上限

算子级加速 ≠ 端到端加速。对每个候选优化，先问：**目标算子占 GPU kernel 总时间百分之几？**
- 若项目自报"算子 2x"但目标算子只占 kernel 时间 ~10%，端到端上限 ≈ 1/(1-0.1+0.1/2) ≈ +5%，别期待翻倍
- 让项目作者/博客里的 profile 占比数据说话（如"attention 只占 7.5%+3.9%"→ INT8 attention 端到端只 +0.4~0.8%）
- **真正的大头往往是非量化优化**（分块调优、GEMV 分流、消除重复计算），量化路径常被高估

## Step 4 — 环境匹配检查（决定"测不测"的成本）

连目标设备 `docker ps` 看**是否已有匹配框架/DTK 版本的容器**：
```bash
ssh <alias> "docker ps --format '{{.Names}}\t{{.Image}}' | head"
# 对候选容器查版本
docker exec <c> bash -c 'python -c "import vllm,torch;print(vllm.__version__,torch.__version__)"'
```
- 有现成匹配容器（如项目基于 vLLM 0.18.1+DTK 26.04，设备上正好有该版本容器）→ 免搭环境，测的成本低，倾向测
- 无匹配容器 → 需重建环境，成本高，除非价值极高否则降级

## Step 5 — 决策框架 + 价值点提炼

对每个候选，按四维打分后给结论（测/不测/留作资产/必读不测）：
1. **技术深度**（源码验证通过？关键声明属实？）
2. **环境匹配**（Step 4，测的成本）
3. **E2E 收益上限**（Step 3，Amdahl）
4. **模型形状兼容性**（很多优化 gate 死在特定形状，如 head_dim=256/GQA 6:1，换模型不激活——必须确认用户实际跑的模型是否命中）

**价值点提炼**：区分"性能价值"（能带来可测 E2E 改善）和"知识价值"（唯一一份可参考的实现/踩坑经验，即使不测也值得读）。一个项目可能性能价值低但知识价值高（如踩坑记录、跨架构移植方法论）。

**交付格式**：分档（同架构直接可测 / 近架构可迁移 / 参考不测），每档列项目 + 价值点 + 测的前提（用哪张卡、哪个容器、跑哪个模型）。最后给明确建议：测哪个、目标定准（不是复现自报的 X 倍，而是验证用户自己模型上的可测改善）、需要用户拍板什么（如"你们现在主要跑哪个模型"）。

## 关键坑

- **只按 star 排序会漏掉最对口的**：DCU 专属架构项目 star 个位数但环境完全匹配，AMD 高 star 项目可能需大量移植。必须按架构名搜，star 只作辅助。
- **信 README 自报数字 = 把未验证声明当事实**：必须 clone + grep 源码。博客写得再详细，指令/映射不在源码里就是吹的。
- **算子加速当端到端加速**：不查目标算子占比就报"2x"，Amdahl 一算可能只有 +5%。量化路径尤其常被高估，非量化优化（分块/GEMV/去重）常是真正大头。
- **忽略模型形状 gate**：很多 kernel 优化只在特定 head_dim/GQA/无 sliding-window 时激活，换模型直接不生效。交付前必须问清用户实际跑的模型。
- **wavefront 宽度**：DCU 是 64 lane，来自 32-lane 架构（或 NVIDIA）的归约/映射代码不能直接套，grep 确认 `__shfl_xor` 覆盖 64 lane。
- **别在共享机上乱占卡**：`hy-smi` 先确认哪些卡空闲，只挑空闲卡；0-N 卡被他人服务占着就耐心等或先问用户（见 `dcu-ssh`/`dcu-hy-smi`）。

## 参考

- `references/arch-mapping.md` — DCU gfx 架构 ↔ AMD 架构映射 + 搜索查询模式
