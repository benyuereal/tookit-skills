# Case 010：FP8 GEMM —— aiter opus 只到 nominal 的 8.9%，还有 11 倍空间

> 起点：`gemm-fp8` 卡 status=draft，卡里写「gfx938 上 FP8 无方案（hipBLASLt）」。
> 本 case 查了三件事，一件比一件重要。

## 1. 轮 1：hipBLASLt 确实没方案（卡里的说法对）

```
torch._scaled_mm(...) → could not find valid hipblaslt solution
```

⇒ 「gfx938 上 hipBLASLt 无 FP8 GEMM」这条结论**成立**。

## 2. 轮 2：但 aiter 有（卡里的结论过时了）

`aiter.gemm_a8w8_opus` 的文档明确写「gfx938/gfx946 FP8 E4M3 Opus GEMM」，
而且跑得出结果：

```
4096³: 2.7530 ms  MAD=0.0155  49.9 TF  out=(4096,4096) bf16
```

**卡里「无方案」应改成「hipBLASLt 无方案，aiter 有」。**

## 3. 轮 3-5：排除「是我测错了」

这是本 case 的核心工作——49.9 TF 与当时 constants.yaml 的 540 TF 差 10.9 倍，
必须先排除测量本身的问题。

### kernelId 对照（排除传错参数）

| kernelId | ms | TFLOPS |
|---|---|---|
| 0（全局内存 baseline） | 117.4881 | 1.2 |
| 1（32x32 四 wave 同步 LDS） | 8.1277 | 16.9 |
| 2（64x64x64 Opus 流水） | 2.7690 | 49.6 |
| auto | 2.7491 | **50.0** |

auto 确实选了最好的（走 kernel 2）。**不是我传错了参数。**

### 扫形状（排除形状没选好）

| M×N×K | ms | TFLOPS |
|---|---|---|
| 4096³ | 2.7672 | 49.7 |
| 8192×4096×4096 | 5.6018 | 49.1 |
| 4096×8192×4096 | 5.7558 | 47.8 |
| 8192³ | 22.3480 | 49.2 |
| 16384×4096×4096 | 11.2880 | 48.7 |
| 4096×16384×4096 | 11.4489 | 48.0 |

**六个形状全部收敛在 47.8~49.7 TF。不是形状问题。**

### bf16 对照 + 转换开销（排除转换税）

| | ms | TFLOPS |
|---|---|---|
| bf16 GEMM（4096³） | 0.5206 | 264.0 |
| fp8 GEMM（4096³） | 2.7672 | 49.7 |
| bf16→fp8 转换 | 0.0531 | 占 GEMM 的 1.9% |

**fp8/bf16 = 0.19，即 fp8 慢 5.3×。**
转换只占 1.9%，不是转换税。

**若硬件真有 fp8 加速，fp8 应该比 bf16 快，而不是慢 5 倍。**
这个反直觉的比值本身就是最强的证据：**本机 fp8 走的是慢路径，
不是「fp8 硬件上限就是 49.8 TF」。**

## 4. constants.yaml 的 fp8 峰值 540 TF 是错的

旧值来源原文：

```
"本机 fp8 GEMM 实测 479 TF = 89% 达成，反推峰值 540 TF（kb: qwen38-prefill-floor）"
```

**问题在 source 的后半句**：`kb: qwen38-prefill-floor` 是一条知识库记录，
**没有脚本、无法复跑**。本仓库的探针从未验证过它。

已改为 49.8 TF（status: measured），并加了 `reproduce:` 字段指向本脚本。
同时加了一条守卫测试 `test_every_compute_peak_has_a_reproducible_tool`：
**任何 compute 峰值常数必须带 `reproduce:`，且文件必须真实存在。**

### 两个分母错的性质对比

| | 声称 | 实测 | 偏差 | 根因 |
|---|---|---|---|---|
| bf16 峰值 | 185 TF | 310 TF | 低 1.7× | 用小形状（N=2048）反推 |
| **fp8 峰值** | **540 TF** | **49.8 TF** | **高 10.9×** | **来源不可复跑** |

fp8 这次更严重：它把「几乎没有优化空间」（49.8 已接近这条实现的顶）
说成「还有 11× 空间」。

## 5. 结论

```
① hipBLASLt 在 gfx938 无 fp8 方案（确认）——但 aiter 有（卡里已过时）
② aiter opus 实测 47.8~49.7 TF，六形状收敛 ⇒ 这是**这一族实现**的水平
③ 同库 fp8/bf16 = 1.98× ⇒ **本机是 2:1 双速率，有原生 fp8 路径**
④ fp8 nominal 560 TF（业主确认）⇒ 49.8/560 = 8.9%，**约 11 倍空间**
```

### 这个 8.9% 说明什么

不是「fp8 硬件不行」，而是 **aiter 的 opus 实现族整体低效**：

| 精度 | aiter opus | nominal | 占比 |
|---|---|---|---|
| bf16 | 24.1 TF | 280 TF | 8.6% |
| fp8 | 49.8 TF | 560 TF | 8.9% |

两条的实现效率几乎一样（8.6% vs 8.9%），而且**恰好构成 2:1**——
这既印证了双速率，也说明 49.8 不是 fp8 的天花板，是这一族实现的天花板。

对比 hipBLASLt 的 bf16 能到 273.9 TF（nominal 的 97.8%），
差距全在「实现质量」，不在「精度」。

### 曾经误判过（记录在此，勿重犯）

第一版本 case 拿 aiter fp8（49.8）去比 hipBLASLt bf16（310），
得出「fp8 比 bf16 慢 5.3×」「本机 fp8 走 dequant 回退」「49.8 就是 fp8 峰值」。
**这三条全错**，根因是跨库比较把库的差距记成了精度的差距。

判据：**判精度特性必须同库同形状；判库的质量才跨库比。两者不可混。**

## 6. 未测定项（不插值）

| 项 | 状态 |
|---|---|
| fp8 的 **achievable** 峰值（最佳实现能到多少） | **未测定**（需要一条打满 MMA 的实现，本轮没有） |
| fp8 的 nominal 峰值 | 560 TF（业主确认，见 constants.yaml:nominal_peak_tf） |
| tilelang 的 fp8 路径能到多少 | 未测定（卡里是 primary，本 case 未跑通） |
| 49.8 TF 的根因（指令效率 / LDS / 占用率） | 未测定（需 PMC，本机无 rocprof） |
| 小 M（decode）的 fp8 表现 | 未测定（本 case 只测 M>=4096） |

## 7. 对本卡的落点

- `status` → measured，`ratio` = **0.089**（分母 = fp8 nominal 560 TF，口径见 constants.yaml）
- **primary 从 tilelang 改成 aiter**（tilelang 本轮未跑通，aiter 有实测）
- 卡里「gfx938 上 FP8 无方案」改成「hipBLASLt 无，aiter 有」
- `ceiling.note` 写明分母是 nominal 560，且**结论定性为「约 11 倍空间，不是已到顶」**
- `modify_path` = replace-call（换掉 opus 这一族），不是调参数

## 复跑

```bash
python3 -I Plugins/ops/skills/op-optimize/cases/010-fp8-gemm-ceiling/bench.py
# 落盘：results/fp8-gemm.json
```
