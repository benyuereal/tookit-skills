# 性能优化案例 §41（多模态视觉 patch_embed conv3d→matmul）

> 2026-10-20 <device-B>（8x BW100，gfx936，Kylin V10，DTK 26.04）实测。模型：MCG-NJU/VideoJev-4B（视频时序定位 + 选择式视频问答，候选打分头替代 LM 输出头，非生成式；Qwen3.5-4B backbone = 24 层线性注意力 GDN + 8 层 full attention + 24 层视觉 ViT）。底座镜像 torch 2.10.0+dtk2604 / transformers 5.17.0 / triton 3.6+dtk2604 / flash-attn 2.8.3（DCU 构建）/ causal-conv1d 1.5.4+das / flash-linear-attention 0.4.2。纯 HF trust_remote_code 推理路径（非 serving 框架）。

## 现象 / 目标

VideoJev-4B 官方 quickstart（10s 视频 + 2 题一次 forward，~2.8k tokens）在 BW100 单卡上跑通并验证正确性后，做端到端延迟优化。初始"稳态"数字 7.5s 与文章（H200，1.123s/题）差距过大，需要定位真实瓶颈并给出可信数字。

## 根因（kernel/测量级）

1. **测量假象（最大坑，先于一切优化）**：DCU 空闲卡默认 sclk level 1 (600MHz)，动态调频爬升主导短时工作负载：同一工作负载**未锁频 7.5s、锁频（level 6=1300MHz，回读确认）0.80s，10 倍差**。锁频前所有数字作废（纪律已补进 dcu-llm-benchmark §1.7）。
2. **单一最大 GPU 热点 = 视觉 patch_embed MIOpen conv3d 182ms**（torch.profiler，GPU 968ms 口径）：Qwen3.5 视觉 patch_embed 为 Conv3d(3→1024, k=(2,16,16), s=(2,16,16), 无 padding)，输入 8800 个独立 patch。MIOpen 对 gfx936 该形状**只有 gfx928 隐式 GEMM 可选**（conv3d_ncdhw_bf16_fwd_implicitgemm_wg64x128_tt4x8_su32_group_common_gfx928），`MIOPEN_FIND_MODE=EXHAUSTIVE`（32687 候选）搜出的 algo 不变，0 收益。
3. 其余（非本次目标）：elementwise/copy ~240ms（935 次 copy_ 多为 GDN 路径 bf16/fp32 互转，mamba_ssm_dtype=float32 为精度设计）；GEMM ~190ms 正常；FA2 varlen + 视觉 FA 26ms；GDN fla triton + causal_conv1d 55ms（DCU 适配 kernel 工作正常，hub-kernel 装饰器本地解析 fla.ops.gated_delta_rule，无运行时联网）。

## 优化动作

**patch_embed conv3d → matmul 等价替换（运行时 monkey-patch，env 门控，可回滚）**：
- 依据：全 stride 无 padding 的 Conv3d 每个 patch (3,2,16,16) 独立无跨 patch 卷积，数学上逐 patch 等价于 Linear。
- 实现：`visual.patch_embed.proj.forward` 替换为 `F.linear(x.reshape(N,-1), W.reshape(O,-1), b)`——权重 reshape 视图零拷贝、不动权重文件；env `VIDEOJEV_OPT=1` 默认开，`=0` 回参考路径；conv 形态不符（k≠s 或 padding≠0）时拒绝安装（fail-closed）。
- 备选路线先排除：MIOpen 环境变量调参（EXHAUSTIVE 实测 0 收益）；GDN 路径 elementwise 融合（需自研 HIP kernel 且动 fp32 SSM 精度设计，收益/风险比不够）。

## 量化结果（卡 4 锁频 1300MHz；交错 3 轮 × 各 5 次中位数，A/B 过噪声门禁；两卡交叉验证 0.798/0.798 一致）

| 指标 | 优化前 (conv3d) | 优化后 (matmul) | 变化 |
|---|---|---|---|
| 视觉编码阶段 | 0.297s | 0.114s | −0.183s（与 profile conv 182ms 吻合）|
| 稳态 e2e（含 CPU 预处理 0.30s） | 0.804s | **0.613s** | **−24%** |
| 冷启动首跑（新进程） | 7.48s | 6.73s | −10%（一次性 MIOpen/hipBLASLt 初始化 ~6s 主导） |

**正确性**：官方 quickstart 双任务——定位区间 [3.0, 10.0]s、答案 A. Pasta. (p=0.999916)——patch 前后逐位不变；候选概率最大偏差 0.007（bf16 舍入量级，小于 H200/DCU 跨硬件 0.05 偏差）；同模式重复跑 bit-identical。

**与文章口径对比（H200）**：文章报 4B LVBench 1.123s/题 = 长视频（分钟级）全集按题数分摊、**不含预处理/数据传输**。本案例为 10s 短视频 demo 规模：GPU 模型计算（剔除 CPU 预处理 0.30s）= 0.494s/2 题 ≈ 0.25s/题（优化前）、0.16s/题（优化后）。两者工作量相差一个量级（视频 token 数），**数字不可直接对比，只可比口径**；严格对标需拉 LVBench 集在 DCU 重跑（后续项）。

## 可复用经验

1. **多模态视觉 patch_embed 通式**：Qwen-VL 系列视觉 patch_embed 均为全 stride 无 padding Conv3d → 在 DCU 上可一律等价替换 F.linear（先查 k==s 且 padding==0）。MIOpen 对 gfx936 的 conv3d 小批量大批数形状算法覆盖弱（只有 gfx928 隐式 GEMM），EXHAUSTIVE 也搜不出更优——**视觉侧 conv 热点优先代数等价变换，别指望 MIOpen 选优**。量级参考：10s 级视频编码省 ~180ms（占 e2e ~24%）。
2. **短时工作负载先锁频再测**（10x 假象，见 dcu-llm-benchmark §1.7）；冷启动与稳态口径分开报（新进程首次 predict 含一次性库初始化）。
3. **交错 A/B（A/B/A/B/A/B）防卡状态漂移**：本案差 0.19s >> run-to-run 方差 0.01-0.02s，门禁大幅通过；共享机测前 hy-smi 选空卡，测中被占立即换卡重测（卡 2 被占 56GB 即换卡 4）。
4. **HF 推理路径的 kernel 覆盖检查法**：torch.profiler key_averages 按 flash_attn / triton(GDN) / gemm / elementwise / conv 五类归并，先看 conv 与 elementwise 占比再动手。

## 验证方法

- 正确性：官方 quickstart 双任务对参考输出（区间 + 答案 + 概率峰 >0.8）自检脚本 PASS；patch 开/关输出确定性复现。
- 性能：同卡锁频、交错 3 轮 × 各 5 次中位数；冷启动 3 次独立进程口径单列。
- 交付物：<device-B> 镜像 videojev-dcu:4b-20261020-opt1 + /data/videojev_assets/{infer.py, ab_patchembed.py, bench.py, cold_bench.py, profile_ops.py, README.md}。