---
name: dcu-kernelwiki
description: "海光DCU kernel优化知识库，agent可查询的可复用方法与事实"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# 海光 DCU Kernel 优化知识库（dcu-kernelwiki）

> 定位：把分散在 内部文档 / 上游仓库 / 实测里的 **DCU 算子优化事实与坑** 结构化组织，
> 让 agent 能像查 ROCmKernelWiki 一样"查一下 DCU 上某算子怎么写才对"。
> 设计参考 `jhinpan/ROCmKernelWiki`（AMD CDNA 版，带证据飞轮与真机验证），此处 DCU 化。

## 一、设计原则（与 ROCmKernelWiki 一致，DCU 化）

1. **架构作用域隔离**：DCU 有多种架构（K100/Z100→gfx906、K100-AI→gfx928 等）。
   每条知识必须标注适用的 `gfx` / 卡型，默认查询结果只返回当前作用域内的事实。
   > ⚠️ 来源：你本机/容器里**不要假设某代卡**。编译前用 `dcu-kernel-dev/../dcu-kernel-dev/scripts/dcu_detect_arch.sh` 现查。
2. **溯源 + 证据**：每条"已验证事实"必须挂来源（内部文档 文件 ID / 上游 commit / 实测日期 + 设备 + DTK 版本）。
   不收"听说""可能"。模糊结论标 `unverified`。
3. **真机验证纪律**：凡涉及数值/性能/指令布局的结论，必须在挂载 `/opt/hyhal:ro` 且有 `/dev/kfd` 的
   DCU 物理机上编译运行验证，容器里 `rocminfo` 不可靠（见 memory 纪律，gfx 用物理机查）。
4. **性能数字不入本库**：基线 ns / GB/s / 加速比等逐条数据进 `dcu-perf-cases` 或本地留痕，
   本库只留**方法 + 架构事实 + 结论性判断**（与 dcu-kernel-dev 纪律一致）。

## 二、硬件作用域（持续补全）

| 卡型 | gfx | 架构 | Matrix Core | 关键约束 | 验证状态 |
|---|---|---|---|---|---|
| K100 / Z100 | `gfx906` | 类 GCN/CDNA 早期 | 视具体型号 | 早期 DTK；见 XDZS2025 Conv2d 路径 | 见 dcu-kernel-dev §13.4（XDZS2025） |
| K100-AI | `gfx928` | RDNA 衍生（含 MC） | ✅ `__builtin_amdgcn_mmac_f32_16x16x8f32` 等 | wavefront=64；每 CU 64KB LDS / 256KB 寄存器；每 block 512 线程(8 wave) | 已实测(dtk25.04) |
| **BW150** | `gfx936` | DCU 第二代 | 视具体型号（非 928 MC intrinsic） | 实测 <device-A> 设备：8×BW150；`hipcc --offload-arch=gfx936` 通用 HIP kernel 编译运行通过 | ✅ **真机验证 2026-08-06（设备 <device-A>, dtk26.04）** |
| **BW100** | `gfx936` | DCU（与 BW150 同 gfx936 代） | 视具体型号 | 实测 <device-B> 设备：8×BW100；gfx=936（见 dcu-kernel-dev §13.5 验证优先级） | ✅ **真机验证 2026-08-06（设备 <device-B>, dtk26.04）** |
| **BW1000** | `gfx936` | DCU 第二代（同 BW100/BW150 代） | 视具体型号（TileLang MLS 白名单不含 gfx936，同 BW100/BW150） | 2026-09-10 单卡容器实测：gfx936 / 80 CU / wavefront 64 / 64GB HBM / Max Clock 1500MHz / `sramecc+:xnack-` / Fast f16 TRUE；DTK 26.04（HIP 6.3.26113）+ torch 2.10.0+das.opt1 + triton 3.5.0+das185 + tilelang 0.1.9+das185；torch fp16 matmul 4096³ ~318 TFLOPS、TileLang mmac 路径 GEMM ~74 TFLOPS（粗测单次含启动）、triton vadd/tl.dot matmul PASS；`hy-smi` 正常但 `rocminfo` 打印 `rocm_smi cmake 不存在` 警告（hyhal 6.3.30 驱动目录下无该 cmake，无实际影响） | ✅ **真机验证 2026-09-10（BW1000 容器, dtk26.04）** |
| **BW1100** | `gfx938` | DCU 第三代（DCU-3G，OAM+HSW 全互联） | 视具体型号（FP8/BF8 MFMA intrinsic 可识别，见 dcu-kernel-dev §15.2b） | 2026-09-16 单卡容器实测（<device-alias>, pytorch:2.7.1-dtk26.04 镜像）：gfx938 / **64 CU**（注意比 BW1000 的 80 CU 少）/ 每 CU 40 wave（max_threads_per_mp=2560）/ **144GB HBM** / `sramecc+:xnack-` / 寄存器 128KB/CU；DTK 26.04-rc4（HIP 6.3.26045）+ torch 2.7.1，torch arch_list 含 gfx938；torch fp16 matmul 4096³ ~306 TFLOPS（bf16 ~303）；⚠️ `torch._scaled_mm`（hipBLASLt 0.10）FP8 GEMM 报 `could not find valid hipblaslt solution`（e4m3fnuz/e5m2fnuz 均无 solution），FP8 推理不能走该路径 | ✅ **真机验证 2026-09-16（<device-alias> BW1100 容器, dtk26.04）** |
> 实际 gfx 以物理机 `rocminfo` 为准，不假设。新增卡型时在此表追加一行并标验证来源。
> <device-A>=BW150、<device-B>=BW100 均为 **gfx936**（2026-08-06 实测），故 XDZS2025 的 gfx928 专属 MC intrinsic **不适用这几台**，XDZS2025 仅作离线参考（见 dcu-kernel-dev §13.5）。BW1000 亦为 gfx936（2026-09-10 实测），MLS 白名单限制同上。

### 硬件资源速查（P0 真机实测 2026-10-20，DTK-26.04 系，hipGetDeviceProperties + rocminfo 双源）

| 卡(机) | gfx | CU | 线程/CU | L1 | L2 | LDS/CU | boost 时钟 | HBM | PwrCap |
|---|---|---|---|---|---|---|---|---|---|
| K100-AI (<device-C>) | gfx928 | 120 | 2560(40 wave) | 16KB | 8MB | 64KB | 1400MHz | 64GB(875MHz×4096bit) | 400W |
| BW150 (<device-A>) | gfx936 | 80 | 2560(40 wave) | 32KB | 8MB | 64KB | 1400MHz | 64GB(1800MHz×4096bit) | 600W |
| BW100 (<device-B>) | gfx936 | 64 | 2560(40 wave) | 32KB | 8MB | 64KB | 1300MHz | 64GB(1800MHz×4096bit) | 400W |
| BW1100 (<device-D>) | gfx938 | 64 | 2560(40 wave) | 未测 | 8MB | 64KB | 1350MHz | 144GB | 800W |

> 三代 928/936/938 的 L2 均为 8MB（四卡实测）；L1 代际不同（928=16KB、936=32KB、938 待补）。
> BW1100 镜像内 rocminfo 缺 libhsa-runtime64.so.1 跑不了，L1/SIMD-per-CU 待补（HIP 结构体字段已全）。
> K100-AI "每 block 512 线程" 是 XDZS2025 实践值，硬件 max = 1024 线程/block（maxThreadsPerBlock 实测）。
> 方法与三宿主编译/运行配方见 `wiki/hwprobe_p0_20261020.md`；性能数字（HBM/launch/LDS 带宽）→ `dcu-perf-cases`（`perf/dcu-perf-cases/references/hwprobe-baseline-20261020.md`）；型号规格总表 → 包根 `../_dcu-toolkit-shared/references/dcu-models.md`。

## 三、已验证事实索引（wiki/ 目录）

本知识库正文按主题拆到 `wiki/` 下，每条文件含 frontmatter（scope / source / verified_on）。
当前已建骨架：

- `wiki/matrixcore_layout.md` — DCU K100-AI Matrix Core 指令集与 wavefront 数据布局（源自 XDZS2025 反向工程）
- `wiki/lds_bank_conflict.md` — LDS bank conflict 与 padding 规避（BM×(BK+2) padding 法）
- `wiki/rmsnorm_precision_cliff.md` — RMSNorm bf16 舍入悬崖（block 增大→diff>0.03），判据 <0.03 放行
- `wiki/tilelang_dcu_known_issues.md` — TileLang 在 DCU 的 12 项已知问题（target=hip / warp_size=64 / alloc_shared 竞争 等）
- `wiki/optimization_loop_protocol.md` — profile 驱动优化循环协议（METRIC 行、profiler 证据、MAD 噪声判定）
- `wiki/gemm-bank-conflict-case.md` — GEMM 缩放因子 SFA 的 bank conflict 分析与修复（Colfax RTX Pro 6000 SM120 案例，方法论可迁移 DCU）
- `wiki/megamoe_fused_moe_kernel.md` — MegaMoE 单 kernel 通算融合 MoE 层（SM100 案例）：symmetric memory 物理机制、性能账算法、7 个可迁移设计（pull 模型/round-robin+min-peeling/分槽位 combine/self-resetting barrier/尾部并行等）+ SM100→DCU 概念映射表
- `wiki/flashattn_pipeline_overlap_case.md` — Flash Attention mainloop 流水线重构与双缓冲（SM103 案例）：结构性 overlap 优先、优化上下文比优化本身重要、unroll×模板组合爆炸消融归因、barrier 粒度细化、收益递减换架构 + SM103→DCU 概念映射表
- `wiki/bandwidth_bound_l2_thrashing.md` — bandwidth-bound kernel 的 L2 thrashing 悬崖与 NVIDIA cache-hint 优化（TMA prefetch/streaming store/evict-first）的 DCU 适用性判据：BW150/BW100 L2=8MB 无悬崖、naive 即 HBM 最优（2026-09-28 真机验证）
- `wiki/blackwell_gemm_dcu_transfer.md` — Blackwell GEMM 8步优化在 DCU 的可迁移性实测（<device-B> gfx936）：128B XOR swizzle 机制真实但净收益取决于访问模式（向量化净正/scalar 净负，DCU 无 TMA）、多 stage 双缓冲只对 latency-bound kernel 有效、可迁移的是方法论不是手法

> 以上文件由 `dcu-kernel-dev` 与既有 dcu skills 已沉淀的实战结论回填，
> 并持续按"证据飞轮"补充：发现新事实 → 真机验证 → 写 frontmatter(source+verified_on) → 提交。

## 四、证据飞轮（self-evolution）

1. 任何 agent / 人在 DCU 上做了 kernel 实验，产出"新事实"时：
   - 在 `wiki/` 建/改对应文件，frontmatter 填 `source`（内部文档 ID 或 commit）、`verified_on`（日期+设备+gfx+DTK）。
   - 不确定者标 `status: unverified`。
2. 性能数据不在此留，转 `dcu-perf-cases`。
3. 提交到 dcu skill 仓（master），保持知识库随源文档/实测同步。

## 五、与 dcu-kernel-dev 的分工

- **dcu-kernel-dev**：算子开发**方法学**（两条路径：手写 HIP / TileLang）+ 编译调试工具链 + 精度陷阱。
- **dcu-kernelwiki**：算子优化**结构化知识 + 真机验证证据**，agent 查询"DCU 上某件事是不是这样"。
- 二者通过 `related_skills` 互链；不重复，wiki 只引用 dev 的方法，不重述。

## 六、何时选它 / 何时别选

- **选它**：动手前要一条**架构事实**——某代卡的 CU/LDS/L2 数值、某 intrinsic 是否存在、bank conflict 的规避手法、某个「看起来该快」的写法在 DCU 上是不是根本不成立。这类问题查表比实验便宜两个数量级。
- **选它**：判断一条经验能不能迁移（SM100/SM103 上成立的手法到 DCU 要不要打折扣）——库里有带映射结论的案例。
- **选它**：要给一条结论挂 `source` + `verified_on` 时，先看这里有没有现成的，避免重复验证。
- **别选（结构性吃亏）**：要**性能数字**。本库按纪律不收基线吞吐/带宽/加速比，数字在 `dcu-perf-cases` 与用户知识库；拿本库的定性结论当分母是一次分母漂移。
- **别选**：要一个能跑的 kernel 或编译器行为——它是检索层，不是实现层也不是编译器；要写代码去 `dcu-kernel-dev`，要编译 backend 去 `dcu-triton-source`。
- **别选**：当前作用域（gfx/卡型/DTK）没覆盖时硬套邻代结论——928 的 MC intrinsic 不适用于 gfx936，这类跨代套用在库里被显式标为仅离线参考。
- **反例**：查到的条目若标 `unverified`，它在决策里的权重应低于一次真机测量；不要把未验证条目当已证实事实引用。

## 七、使用须知

- 本库是"事实检索"层，不是编译器也不是 benchmark。
- 任何结论先核对：设备 gfx + 驱动/DTK 版本 + 报错原文（与 dcu-kernel-dev 同纪律）。
- 无卡环境无法验证；涉及编译/运行的结论务必在真机复核。
