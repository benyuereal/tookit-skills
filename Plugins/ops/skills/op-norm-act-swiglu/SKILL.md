---
name: op-norm-act-swiglu
description: "add + RMSNorm + SwiGLU 三合一融合。输入 x/residual [M,H] 与 w[H]，输出 [M,I]（I=H/2）。用途：把两步法（fused_add_rms_norm + fuse_silu_and_mul）的中间张量 h[M,H] 消掉，流量 369.1→167.8 MB。触发场景：MLP 前段的残差+归一化+门控激活连续三步、想减少 kernel 数与 HBM 往返。不适用：H≠4096 或已融合的上游。"
---

# op-norm-act-swiglu

```
xf      = x + residual                        读 x, residual
h       = w * xf / sqrt(mean(xf^2,-1)+eps)    读 w；归约沿 H
gate,up = h[:, :I], h[:, I:]                  I = H/2
out     = silu(gate) * up                     写 out，I 宽
```

M=8192, H=4096, I=2048，全 bf16，eps=1e-6。

## 为什么值得融合

基线是两步法，中间张量 `h[M,4096]` 要落一次 HBM 再读回来：

| | 流量 | 明细 |
|---|---|---|
| 两步法 | 369.1 MB | ① 读x+读r+写x+写r (4H) ② 读h+写out (1.5H) |
| 三合一 | **167.8 MB** | 读x+读r+读w+写out (2.5H) |

**降到 45.5%。** 融合成立的根据：`gate` 与 `up` 来自同一个 `h`，
归一化因子对两者相同 —— 一个 kernel 内先归约、再同时算两半，不需要把 `h` 写出去。

## 实测（event_isolated，base 每轮重测）

| 实现 | 耗时 | 相对基线 |
|---|---|---|
| `two-step`（lightop 两步法） | 0.3050 ms | 1.00x |
| `fused-hip-mr2`（本卡） | **0.1496 ms** | **2.03x** |

精度：**1.00 ULP**，cosine 1.0000001。
地板：82.6%（1:1 档 1680 GB/s × 归约折扣 0.81）。
holdout（M=13312）：0.21552 ms，73% 地板。

## 用法

```bash
# 编译（BLOCK_ROWS=2 是实测最优；4 行因 vgpr=84 退化）
bash Plugins/ops/skills/op-norm-act-swiglu/kernels/build.sh

# A/B
python3 -I bench/ab.py --shape s0 --base two-step --variants fused-hip-mr2 \
        --rounds 5 --measure-mode event_isolated
```

## 配置约束

`H == THREADS*8*VEC` 且 `THREADS*8 == I` 且 `VEC == 2` ⇒ H=4096 固定。
最后那条是**设计选择**：它让同一线程的 j=0 持有 gate、j=1 持有 up，配对天然成立。
换形状必须重新生成 kernel，不提供运行期兜底。

## 下一步

`next_probe: barrier-width` —— 剩余 17% 来自行内归约的两道 barrier。
用 `ds_bpermute` 替掉 LDS 表可省一次 `__syncthreads`，预期 -5%~-10%。
