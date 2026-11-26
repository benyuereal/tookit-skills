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
