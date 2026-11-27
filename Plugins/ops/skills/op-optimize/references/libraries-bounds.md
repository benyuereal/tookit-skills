# 竞品库可用边界

> 记录「这个库在什么范围内可信、超出范围会怎样」。
>
> 与 `libraries.md` 的分工：**那个讲「有哪些库、怎么调」，这个讲「哪里不能用」。**
>
> 为什么需要单独一页：atlas 的核心命题是「去哪个库找、用哪个、上限多少」，
> 但**边界**这件事一直只写在 commit message 里 —— 换个人、换一天就找不到了。
>
> 规则：**每条必须有实测数据表。** 没有数据的边界不准写进来。

---

## lightop

### lightop.gemm

**可用范围：M ≤ 16**

| M | K | N | 相对误差 | 判定 |
|---|---|---|---|---|
| 8 | 512 | 512 | 0.002249 | 合格 |
| 16 | 512 | 512 | 0.002160 | 合格 |
| 24 | 512 | 512 | **1.064266** | **完全错** |
| 32 | 512 | 512 | **4.475079** | **完全错** |

**入参约定**：吃 `(A[M,K], B[N,K])`，**不是** `(A, B[K,N])`。
传错时（直接给 `B[K,N]`）误差 1.36。

```python
# 正确
out = lightop.gemm(A, B.t().contiguous())

# 错误（形状对得上，但数值全错 —— 最难发现的那种）
out = lightop.gemm(A, B)
```

**复现**：

```bash
HIP_VISIBLE_DEVICES=N python3 -I -c "
import torch, lightop
torch.manual_seed(0)
for M in (8, 16, 24, 32):
    A = torch.randn(M, 512, dtype=torch.bfloat16, device='cuda:0') * 0.1
    B = torch.randn(512, 512, dtype=torch.bfloat16, device='cuda:0') * 0.1
    out = lightop.gemm(A, B.t().contiguous()); torch.cuda.synchronize()
    rel = ((out.float() - A.float() @ B.float()).abs().max()
           / (A.float() @ B.float()).abs().max()).item()
    print(M, '%.6f' % rel)
"
```

**根因**：未查明。猜测是 tile 分块假设了 M ≤ 16 的 warp 布局，但**未验证**。

**护栏**：`Plugins/ops/skills/op-gemm-bf16/_common.py` 里 M ≥ 24 直接抛异常，
不返回错误结果 —— 悄悄算错比报错危险得多。

**一次误判值得记**：第一次测到 K=4096 错、K=256 对，就归因成「K > 256 有问题」
并写进了卡片。后来控制 M 不变再扫 K，发现 K=256 且 N=4096 时也是错的 ——
**真正的变量是 M**。这是 `wrong-variable` 故障面的一个实例。

---

## FlashMLA

### FlashMLA fp8 decode

**可用范围：H ≤ 32**

| H | D(qk) | 结果 |
|---|---|---|
| 32 | 512 | ✓ |
| 63 | 512 | ✓ |
| **64** | **512** | ✗ 崩 |
| **64** | **576** | ✗ 崩 |
| 65 / 96 | 512 | ✗ 崩 |

**关键**：`D=576` 是上游原生支持的形状，H=64 时**也崩** ——
所以问题不在 d=512 的参数化上，而在 H ≥ 64 这条路径。

**根因**：**未查明。** 已排除的假设：
- 不是 combine 阶段（`FLASH_MLA_SKIP_COMBINE=1` 仍崩）
- 不是序列长度（SKV=1024/4096/16384 全崩）
- 不是 `Traits_Block_M_64`（该模板未被使用）
- 不是设备绑定方式（用 `HIP_VISIBLE_DEVICES` 重映射仍崩）

未查方向见 `docs/DEBT.md` 的 D22。

**注意**：GLM-5.3 线上配置就是 `num_attention_heads=64`，
所以这条路径是**生产主路径**，不是测试形状。

---

## 记录模板

新增一条时按这个写：

```markdown
### <库>.<入口>

**可用范围：<条件>**

| 参数 | 结果 | 判定 |
|---|---|---|
| ... | ... | ... |

**复现**：<能跑的命令>

**根因**：<查明 / 未查明 + 已排除哪些>

**护栏**：<哪个文件里加了什么保护>
```

---

## lightop（GDN 分支 `qwen3.8-27b-perf-gdn`）

> 源码位置：`csrc/gdn/`，Python 接口 `lightop.gdn`
> 状态：**已有一套手写 HIP 实现**，不是空的。
> 详见 `hw_kb/hip/gfx938/gdn_recurrent_structure.md`。

### lightop.gdn 的精度陷阱（**给错不报错，结果全错**）

`A_log` 和 `dt_bias` 在 kernel 里按 **`const float*`** 读。

```python
# ❌ 不报错，但每个头的 decay 都错 —— kernel 读越界
lightop.gdn.gdn_verify(q, k, v, a, b, A_log_bf16, dt_bias_bf16, ...)

# ✅ 走 Python 层的 wrapper（它内部转 fp32）
lightop.gdn.gdn_verify(...)   # wrapper 有 f32() 转换
```

**为什么危险**：给错精度不会抛异常，现象是「结果算错」，
和本仓 `wrong-variable` / `uninit-struct` 是同一族 —— **最难发现的那种**。

**判据**：任何传给 HIP kernel 的 float32 指针参数，
传之前确认 dtype 是 `float32`。§

### lightop.gdn 的能力降级接口（值得抄的设计）

```python
lightop.gdn.gdn_conv1d_unsupported_reason(...) -> Optional[str]
# None = HIP 能跑；字符串 = 为什么必须留在 Triton
```

**不是布尔，是一句原因。** 调用方据此决定 fallback，不是二元判断 + 猜。

对照本仓：`candidates.yaml` 的 `status: blocked` + `blocker:` 是同一思路，
但那条是**静态的**（写在文件里），这条是**运行期问出来的**。
对于「某些形状能跑、某些不能」的场景，后者更准确。


---

## GEMM 库实测对比（2026-11-27，prefill-8k：M=8192 K=4096 N=4096，bf16）

| 库 | 耗时 | 相对 torch-mm | 结论 |
|---|---|---|---|
| `torch.mm`（hipBLASLt 默认启发式） | **0.9374 ms** | 1.00x | **最快** |
| `aiter.gemm_a16w16_opus` | 10.9784 ms | **11.7x 慢** | 不适用大 M |
| `lightop.gemm` | — | — | **结果错误，见下** |

### lightop.gemm：所有 M≥24 都算错

之前的记录说「M≥16 可用、M≥24 错」。2026-11-27 复测更严重：

| M | 误差 | 判定 |
|---|---|---|
| 24 | 4.614309 | **错** |
| 64 | 4.872625 | **错** |
| 128 | 3.823497 | **错** |
| 1024 | 1.000000 | **错** |
| 8192 | 4.218176 | **错** |

**两种调用方式（`gemm(A,B.t())` / `gemm(A,B)`）都错。**
结论更新：**`lightop.gemm` 在本机完全不可用**，不是「小 M 可用」。
根因未查明（可能缺少某个上下文初始化，或需要特定的量化前置）。

### aiter.gemm_a16w16_opus：入参要转置，且大 M 极慢

**入参约定**：吃 `(A[M,K], B[N,K])`，不是 `(A, B[K,N])`。
```
gemm(A, B)     误差 1.36   错
gemm(A, B.T)   误差 0.0037 对
```

**性能**：prefill-8k 上 10.98ms vs torch.mm 0.94ms —— **慢 11.7 倍**。
不排除是形状不适配（它可能在等某个特定 tile 配置），但**在实测形状上就是慢**。

**判据**：凡是「库 A 比库 B 慢一个数量级」的结果，
先查入参格式对不对（我们踩了两次：lightop 和 aiter 都需要 B.T），
再判断是不是形状不适配。**不要直接写进结论。**

