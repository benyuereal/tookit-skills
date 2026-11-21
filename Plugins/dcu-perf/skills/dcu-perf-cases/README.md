# dcu-perf-cases（DCU 真实性能优化案例库）

用户向 README。

## 是什么
**案例沉淀**（不是方法论）。每个案例按「现象 → 根因 → 优化动作 → 量化结果」四段式记录，方便横向对照同类问题。分析方法论见 `dcu-perf-analysis` + `dcu-op-optimization`（强制量化对比、留痕、禁"感觉快了"）。案例源自《DCU性能分析及常用优化技术手册》外链实战文档（内部文档 file_id=<internal-file-id>，作者<internal-author>）。

## 已收录案例
1. **LSTM（BW10 vs A10）**：小算子爆炸 / Persistent RNN 缺失 → 改写 Persistent RNN，耗时 8–10s → 3.08s。
2. **UNet（BW10 vs A10）**：FP32 精度壁垒 + Launch Overhead → AMP + channels_last + Graph，97.7s → 33.9s（16.4 → 47.2 samples/s）。
3. **GNN / GPT-2（BW10 vs A10）**：访存墙（Softmax / Elementwise 未融合）→ FlashAttention + 图融合。
4. **三类根因横向对照**（速查表：小算子爆炸 / 精度壁垒 / 访存墙）。
5. **Qwen3-32B SGLang（BW1000×2）**：decode 纯权重带宽瓶颈 → 提升带宽利用率（更大 batch / fp8 / Aiter GEMM）。
6. **BW100 Qwen3.5-2B 多模态 vLLM**：prefill 串行排队根因（非算力）→ 先拆 prefill/decode 占比。
7. **SGLang mamba cache size 陷阱**：多模态并发被静默限流 → 显式设 `max-mamba-cache-size`；附完整启动参数模板（Aiter/Marlin/fp8/mamba/EAGLE 投机）。
8. **GLM-5.1 K100 AI INT8 部署**：TileLang + Sparse MLA 组合优化。
9. **Qwen3-8B LoRA SFT 吞吐（K100-AI×8）**：Llama-Factory + Liger-Kernel 一行收益。
10. **K100 Qwen3.6-35B-A3B W8A8 单卡 Decode**：专用 INT8 算子 + 融合 + MTP3（纯模型 ~53.5 → 叠 MTP3 稳定 ~85）。
11. **Qwen3.6-27B W8A8 K100AI 单卡 Decode（R067 MTP4）**：shape 感知 runtime patch + 两级 draft selector，~3.2 → 42.3 tok/s。
12. **TFRS 推荐排序多热 Embedding 融合**：TF Profiler 定位事件数瓶颈。
13. **BW10 gemma-4-31B AWQ 两阶段优化**：aiter w4a16 GEMM + flash fp8 KV head512 + MTP，TP4 TPOT 2.59x（90.72→35.02ms）。
14. **曙光 SHCA IB BW1000 双机 TP16 RDMA 调优**：国产 IB 网卡走 Socket→RDMA 全链路（官方 RCCL 插件 + 通道锁死 + topo 按节点），TTFT 2803→1121ms、吞吐 22→29.6；启动脚本见 `scripts/shca_ib_tp16_start.sh`。

**附录**：并发压测通用方法论（去标识化，源自客户现场规范——固定档位多轮、掐头去尾取均值、按阈值筛最大并发、整机折算）。

## 扩展规范
新增案例在文末「## N. 案例名」追加，保持四段式；跨平台对标须注明硬件 / 精度 / 框架版本，量化结果必须给优化前后两组数。

## 相关 skill
dcu-perf-analysis（总纲）/ dcu-op-optimization / dcu-perf-profiling / dcu-hipprof / dcu-framework-profilers

## 溯源
案例原文 内部文档 file_id=<internal-file-id>（内部文档链接，已略）
