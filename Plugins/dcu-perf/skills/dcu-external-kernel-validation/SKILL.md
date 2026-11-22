---
name: dcu-external-kernel-validation
description: "在 DCU 上端到端验证外部 GPU kernel 优化（external kernel validation）。何时用：拿到一个号称更快的第三方 kernel，要判断它在 DCU 上是否真的成立。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# DCU 外部内核优化全链路验证方法论

## When to Use

用户发来一篇 GPU kernel 优化文章/博客/PR（如 Blackwell/ROCm 上的 attention 优化），要求"在 DCU 上验证有没有价值 / 实际测试一下 / 生产环境能提升多少"。本 skill 是从**文章思想 → kernel 实现 → 生产框架接入 → E2E 实测**的完整工作流，含每步的坑。

前置：`dcu-external-opt-scout` 管 GitHub 项目侦察（star/架构分档），本 skill 管**单篇文章/kernel 的实测验证闭环**，两者互补。

不适用：纯部署/适配任务（用 `dcu-software-adaptation-guide`）；自己从零写 kernel 的规范（用 `dcu-kernel-dev`）。

## 已验证案例（本方法论的产出，2026-10-09，<device-B> 8xBW100 gfx936）

文章：Colfax CUTLASS PR #2973 Blackwell GQA decode（swap-AB 布局 + 双 warpgroup + PDL + split-K，对 FA4 中位数 1.46x）。

| 验证层 | 结果 |
|---|---|
| kernel 级：swap-AB vs 朴素标准布局（同 Triton 写法） | **3.0x**（MMA M 维从 8 拉到 64/128，tensor core 利用率上来；不只是"消 padding"） |
| kernel 级：vs FA2 2.8.3 paged（`flash_attn_with_kvcache`，page=256） | 交叉点 L≈8k；L=16k/32k 时 **+12-17%**；短序列负收益 |
| E2E vLLM 0.18.1（triton 后端）eager | +5~6.4% TPOT |
| E2E vLLM 0.18.1（triton 后端）**CUDA graph（生产默认）** | **+14.8%**（32k 上下文 TPOT 39.3→33.5ms，输出逐 token 一致） |
| E2E SGLang 0.5.18（grouped decode 后端） | **负收益 ~5%**——它的 `decode_attention_fwd_grouped`（stage1/stage2 + GQA 打包）已内置文章思想 |

最终判断：优化价值锚点是 **DCU 版 vLLM**（triton 后端还是老 unified kernel）；SGLang 侧不用做。文章在 Blackwell 上只有 1.46x 的原因也是 FA4 已迭代到类似水平。

## 已验证案例二（2026-10-20，dcu-bw150 8xBW150 gfx936，DTK 26.04）

文章：SGLang Low-Latency Mode（`sgl-nvidia-mega` / `sgl-amd-mono`，LMSYS 2026-10-18）。
只验证 AMD 路径（sgl-amd-mono）的两个**架构无关**思想；NVIDIA 路径核心是 PDL，DCU 无对应指令，直接排除（见 Step 0）。

| 验证项 | 结果 |
|---|---|
| persistent monokernel（单次启动融合 norm→QKV gemm→rope 链） | ✅ 可行且正确（diff ~2e-5，fp32 内） |
| naive `__syncthreads` 融合 vs 拆开（3 launch） | ❌ **融合反而慢 4–8%** |
| tag 数据就绪同步（原子+threadfence+自旋） | ✅ 可行且正确（diff = 0） |
| tag 异步重叠（producer/consumer 重叠） vs split | ⚠️ 收益不显著（gpu 0.97–0.98×，wall 1.02×，噪声内） |
| 跨卡 all-reduce 数据通路 | ✅ `hipMemcpyPeer` 可用且正确 |
| kernel 内直接访问对端显存（IPC） | ❌ `hipIpcOpenMemHandle` 返回 invalid argument |
| cooperative launch / grid sync | ✅ DCU 支持（单设备=1，多设备=1） |

**可复用预判（下次遇到"融合 / persistent 化 kernel / 低延迟 mode"类文章先看这条）**：
这类把"整个 decoder 层合并成 1–2 次 persistent 启动 + 内核内同步"的上游 kernel 文章，在 DCU 上
**"能实现、正确"，但"不能指望拿到文章宣称的性能收益"**。原因：
1. DCU 的 kernel 间 launch 边界空闲本来就小（2–4 µs），省 1 次 launch 收益有限；
2. 无论 barrier（`__syncthreads`）还是 tag 自旋，kernel 内同步都有额外开销，在该场景下抵消甚至反超省下的 launch；
3. `hipIpcOpenMemHandle` 失败 → 无法像 sgl-amd-mono 那样在单 kernel 内直接原子写对端显存做 all-reduce，
   跨卡只能走 `hipMemcpyPeer` / RCCL。
**真正评估这类融合/tag 的价值，需放到贴近真实 decoder 层（复杂依赖 + 长计算链 + 大量 token 流水线）的端到端路径测**，
而非最小 kernel 级复现。相关平台硬约束见 `dcu-kernel-dev` §15.1。

## Step 0 — 拆文章：架构无关思想 vs 硬件专属机制

把文章的每个优化点分类：
- **架构无关思想**（可迁移，值钱的）：布局/算法层（如 swap-AB：算 S^T=KQ^T/O^T=V^TP^T 把 KV 位置放大矩阵维）、调度模式（双累加器、split-K 拉并行度）、"少一次 launch/少一次 memset"类
- **硬件专属机制**（不搬，gfx936 没有）：TMEM、TMA、PDL（Programmatic Dependent Launch）、thread block cluster + distributed SMEM、`setmaxregister` 按角色分寄存器、WGMMA

只搬第一类；但**文章的收益数字通常是思想+机制的混合**，别指望纯思想移植拿到同等倍数——DCU 上实测往往更小（因为基线强弱不同）。

## Step 1 — 环境摸底

1. `hy-smi` 选**空闲卡**（VRAM%=0 且 HCU%=0）；共享机被占就换卡，**绝不碰别人的服务**（占卡进程用 `/proc/<pid>/cgroup` 反查容器归属再决定）
2. 宿主机 `source /opt/dtk-*/env.sh && rocminfo | grep gfx` 确认架构（BW100=gfx936）
3. 起容器（DCU docker 无 `--gpus`）：
   ```bash
   docker run -d --name test --privileged --security-opt seccomp=unconfined \
     --device /dev/kfd --device /dev/dri --device /dev/mkfd \
     -v /opt/hyhal:/opt/hyhal -v /data:/data \
     -e HIP_VISIBLE_DEVICES=<card> -e 'AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938' \
     <image> sleep infinity
   ```
   **必须挂 /opt/hyhal**（驱动库，不挂 torch 报 `librocm_smi64.so` 缺失）；AMDGPU_TARGETS 里的分号要用引号包住，否则 shell 截断
4. 容器内 `source /opt/dtk/env.sh` 再跑 python；确认 `torch.cuda.get_device_properties(0).gcnArchName` = gfx936

## Step 2 — kernel 级 A/B（先数值，再性能）

1. **先对 torch fp32 参考验数值**：bf16 正常误差线 ~0.0001-0.001，超过说明 kernel 有 bug。用错误模式定位（per-head / per-dim 的 diff 分布）：全 head 均匀偏大 = 数值精度问题；个别 head/dim 集中 = 索引/布局 bug
2. **对拍生产 kernel 必须同布局同形状**：paged 对 paged（同 page size、同 KV 张量结构），连续 KV 只能测布局无关的优化
3. **弱基线陷阱**：拿"无 split-K 的连续版"当基线会把 12-15% 的收益虚报成 3-6x。查清生产路径实际跑哪个 kernel 变体再选基线
4. 计时用 CUDA event + 足够 warmup（Triton 首次 JIT 要编译，warmup 50+）；同一进程内交替采样避免时钟漂移
5. kernel 测试**必须包含 B>1**（见 Pitfall #1）

## Step 3 — 接入生产框架（E2E）

1. **先 probe 生产后端，再打 patch**。DCU 上 vLLM 0.18.1 的 decode 走 `TritonAttentionImpl → unified_attention`（`triton_attn.py`），**不是 flash_attn**——打错模块会"E2E 无变化且输出正确"，白跑一轮。probe 方法：找到模型的 attention 层，打印 `type(layer.impl).__name__` 或看 backend 目录。SGLang DCU 版 decode 走 `decode_attention_fwd`（`sglang.kernels.ops.attention.decode_attention`，triton 后端的 `forward_decode` 调它）
2. patch 点选 **kernel 入口函数**（如 `triton_attn.unified_attention`），不要选 Python 调度层；条件门控只放行你支持的形状（ndim/shape/scale/特殊参数），其余全部 fallback 原路径
3. **call counter 严格验证 patch 真被调用**（36 层 × N 步应等于 36×N，无 fallback 计数）。patch 装了 ≠ 被调用；被调用 ≠ 走对分支——gate 条件里每一项都要用 probe 打印的真实参数核过一遍（本案例 SGLang 侧 gate 连错三次：K/V buffer 是 4D HND 不是 3D、k_scale 是 float 1.0 不是 tensor、v_buffer.shape != k_buffer.shape 是正常的 HND 差异）
4. **eager 和 CUDA graph 两种模式都测**（graph 是生产默认，`enforce_eager=True` 只是调试模式）。graph 模式收益可能更大或更小（固定开销占比不同）
5. 正确性：greedy 下输出逐 token 与基线对比
6. E2E 收益 = attention 占比 × kernel 提升（Amdahl）。decode 步里 attention 占 ~25-30%（其余 GEMM/norm/elementwise），kernel 1.4x → E2E ~15%。**数字对不上先查这个**，再怀疑 patch 没生效

## Step 4 — 跨框架核查（避免重复造轮子）

同一思想在另一个框架可能**已内置**。接新框架前先读它的 decode kernel 源码：
- SGLang 0.5.18 DCU 版 `decode_attention_fwd_grouped`：stage1/stage2 两阶段 + GQA 打包 query head + 预分配 fp32 logits/lse buffer——文章的"打包维度"思想已有且更细，我的 swap-AB 跑不过它（负收益），直接放弃该框架
- 判断标准：读 kernel 的 tile 维度/累加器形状/是否有打包或 split，思想重叠度高就先做 kernel 级对拍，E2E 大概率没空间

## Step 5 — 结论表达

分三层报数：kernel 级（对谁、什么布局、多少倍）→ E2E（什么框架、什么模式、上下文长度、TPOT 变化）→ 边界（哪些场景负收益/未测）。**弱基线的数字必须作废并明确说明**（本案例"3-6x"作废，真实是长上下文 +12-17% kernel / +15% E2E）。

## Pitfalls（全部实测踩过）

1. **batch>1 越界 bug 在 batch=1 下永远暴露不出**：本案例 kernel 里 PM/PL 缓冲（无 D 维）的 batch 步长误用了 PO 的步长（含 D 维），B=1 时 pid_b 恒 0 不触发；CUDA graph 捕获遍历 batch 1~512 时 `pid_b>0` 溢出 → VMFault 崩溃。**kernel 测试必须含 B>1；graph 捕获是最强暴露器**
2. **graph-safe 三要素**：热路径无 `.item()`/GPU→CPU 同步；workspace 在 graph 捕获**前**预分配（lazy 分配可能落在捕获里炸）；无捕获期间的任何新分配
3. **VMFault 崩溃后显存不释放**：进程死了卡上 VRAM 仍被占（host queue 残留），查 `hy-smi --showpids` 找 PID，cgroup 反查容器归属，确认是自己的孤儿进程再 kill；是别人的就换卡重跑
4. **FA2 paged 要求 page size 是 256 倍数**（vLLM DCU 用 page=16 走不了 FA2 路径——这也是平台选 Triton 后端的因素之一）；`flash_attn_with_kvcache` 的 `out=` 参数在 DCU 构建里有形状校验 bug，用普通调用
5. **`flash_attn_func` 默认 BSHD 不是 BHSD**：传 `(1, Hq, 1, d)` 形状会报 "Number of heads in key/value must divide number of heads in query"（seqlen 维被当成 head 数），要 `bhsd=True` 或改形状。这不是 DCU 构建的问题，是上游 API 默认值
6. **SGLang 0.5.18 DCU 镜像默认 AITER 后端是坏的**：`paged_attention_ragged is not defined`（NameError，CUDA graph 捕获时崩），必须显式 `attention_backend="triton"`。镜像自身问题，值得反馈海光
7. **spawn 子进程**：benchmark 脚本必须有 `if __name__ == "__main__":` 保护（子进程会重新 import 主模块）；vLLM patch 用 `VLLM_ENABLE_V1_MULTIPROCESSING=0`（engine 同进程）；SGLang scheduler 是 spawn 子进程，patch 要用 **sitecustomize.py + PYTHONPATH** 注入（环境变量门控），普通 import-time patch 只进主进程
8. **SGLang 的 decode 实际 batch 是 CUDA graph padding 值**（如单请求也跑 B=8），kernel 必须对 padding 请求（L=0）有显式零输出分支，否则污染输出
9. **SGLang HND KV 布局**：K 是 `(pages, H, ps, D)`、V 是 `(pages, H, D, ps)`（shape 不同是**正常的**），ps=1 时退化为 per-token stride(0) + per-head stride(1)，和 3D 布局偏移等价
10. **probe 比猜测快**：gate 条件不过时，打印前 N 次调用的真实参数（shape/ndim/各 scale 值/OK 标志），一次定位所有不匹配项。本案例 SGLang gate 连修三次都是 probe 一枪一个

## 参考

- `references/sglang_swapab_template.md`：SGLang CSR 布局 kernel 的完整适配点（HND 4D、graph padding batch、gate 调试法）+ 负收益结论
- `references/sitecustomize_patch_inject.py`：spawn 子进程 patch 注入模板（SGLang 类多进程框架通用）
- 案例脚本（开发期在 `<agent-home>/cache/scratch/swapab/`，**scratch 会被 24h 清理**，重要版本以本 skill references 为准）：vLLM paged kernel + patch、SGLang CSR kernel + patch、E2E 模板、FA2 公平对比、背靠背 A/B 模板
- `dcu-external-opt-scout`：GitHub 项目侦察（本 skill 的前置）
- `dcu-op-optimization`：性能数据留痕模板
