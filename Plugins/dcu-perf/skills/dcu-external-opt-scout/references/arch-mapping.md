# DCU gfx 架构 ↔ AMD 架构映射 + 搜索查询模式

## 架构映射（决定"同架构"还是"近架构"）

| DCU 卡型 | gfx | 最近 AMD 架构 | 移植难度 |
|---|---|---|---|
| BW100 / BW150 | gfx936 | gfx942 (MI300X, CDNA3) | 中——同 CDNA3 家族，MFMA/wave64 相近，但 FP8 编码(FNUZ vs OCP)与部分指令不同 |
| BW1000 | gfx936 | gfx942 (CDNA3) | 同上 |
| K100_AI | gfx928 | gfx90a (MI210/MI250, CDNA2) | 高——跨 CDNA 代，FP8 ALU 有无、INT8 MFMA K 维不同 |
| BW1100 | gfx938 | gfx942 (CDNA3) | 中 |

**关键**：DCU 的 gfx 号是海光专属，AMD 阵营没有"同架构"，只有"最近架构"。所以搜索要同时覆盖：
1. DCU 专属 gfx 号（gfx936/gfx928/gfx938）——命中少但环境最对口
2. 最近 AMD 架构（gfx942/CDNA3、gfx90a/CDNA2）——方法论可迁移，需移植

## 搜索查询模式（GitHub Search API）

按优先级分组，每组 `sleep 8`（未认证 10 req/min）：

**A. 目标 DCU 架构名**（最对口）
- `gfx936`、`gfx928`、`gfx938`（按实际卡型选）
- `dcu hygon kernel`、`vllm dcu`、`sglang dcu`、`dcu inference optimization`

**B. 最近 AMD 架构**（方法论迁移）
- `gfx942 kernel`、`CDNA3 kernel`、`MI300X optimization`
- `gfx90a`、`CDNA2`（若目标是 K100_AI/gfx928）

**C. 通用优化关键词**（跨架构，按 star 筛）
- `triton AMD GPU`、`composable kernel`、`flash attention ROCm`
- `hip kernel optimization`、`aiter`、`rocm quantization kernel`、`rocm mxfp4`

**D. 赛事/社区专项**（中文 DCU 优化常以赛事形式开源）
- `先导杯`、`dcu 优化`、`海光 推理优化`

## 分档判定

- **同架构直接可测**：项目明确标注目标 DCU gfx 号 + 同 DTK 版本，环境完全匹配
- **近架构可迁移**：AMD 近架构（gfx942/90a）项目，方法论/指令可参考，需移植验证
- **参考不测**：高 star 通用项目（如 AITER 本体、composable_kernel），当对照/知识库，不直接跑

## 跨架构移植的硬件阻塞分类（来自 aiter-cdna2 方法论）

判断"近架构项目能否移植到 DCU"时，按硬件能力分类阻塞，而非打包问题：
- **FP8/BF8 ALU**：CDNA2(gfx90a) 无 FP8 ALU，`v_cvt_pk_fp8_f32` 不汇编；CDNA3(gfx942) 有但 FNUZ 编码，与 gfx950 的 OCP 不位兼容
- **INT8 MFMA K 维**：gfx90a 是 `16x16x16i8`，gfx942 是 `16x16x32i8`，两种拼写互不兼容
- **packed-bf16 atomic**：`global_atomic_pk_add_bf16` 仅 CDNA3+，CDNA2 无
- **SMFMAC(稀疏)/XF32**：CDNA3+ 才有
- **wavefront 宽度**：DCU 是 64 lane，32-lane 架构的归约/映射代码不能直接套

移植前先确认目标 DCU 的硬件能力（FP8 有无、INT8 MFMA K 维、wavefront 宽度），再判断哪些 kernel 可转、哪些被硬件卡死。
