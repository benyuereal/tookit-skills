# Case 008：paged decode —— flash_attn 的 decode 路径在本机是坏档

> 起点：`paged-attention-decode` 卡 status=draft，`impls` 里 flash-attn 是 primary。
> 本 case 要回答：它在 decode 形状到底能到多少？值不值得当 primary？

## 1. 硬前提：block_size 必须被 64 整除

| block_size | 结果 |
|---|---|
| 16 | 拒绝 |
| 32 | 拒绝 |
| 64 | 通过 |

报错原文：

```
Paged KV cache block size must be divisible by 64 at gfx936 platform
```

**注意报错写的是 gfx936，实测机器是 gfx938。** 错误文案与实际不符——
说明这条路径的硬件分派逻辑本身没把 gfx938 列全，报错兜底到了别名的文案。
这不影响功能（拒绝是对的），但排查时会被误导。

## 2. 轮 2：paged vs 连续 KV（B=8, KV=8192）

| | median | GB/s | of peak |
|---|---|---|---|
| 连续 KV | 1.0235 ms | 33 | 0.02 |
| paged | 1.6338 ms | 21 | 0.01 |

paged 慢 1.60×。**但这不是重点**——重点是**连续 KV 本身就只有 33 GB/s**。
paged 的开销（1.6×）是小问题，整个 decode 路径的低效才是大问题。

## 3. 轮 3：扫每序列 query 数（关键实验）

| SQ | median | MAD | 算力 TFLOPS |
|---|---|---|---|
| 1 | 3.1643 ms | 0.0024 | 1.4 |
| 2 | 3.1619 ms | 0.0021 | 2.7 |
| 4 | 3.1635 ms | 0.0027 | 5.4 |
| 8 | 3.1592 ms | 0.0013 | 10.9 |
| 16 | 3.1642 ms | 0.0013 | 21.7 |
| 32 | 3.1747 ms | 0.0013 | 43.3 |

**耗时 3.1592~3.1747 ms，波动 0.5%。与 SQ 完全无关。**

这条比任何单一数字都有信息量，它说明：
- 不是 compute-bound（否则 SQ 翻倍耗时也该翻倍）
- 耗时被**与 query 数无关的固定代价**钉死
- 反过来说：**多算 query 是免费的**（SQ=1→32 耗时不变，算力从 1.4 涨到 43.3 TF）

这个固定代价是什么？结合 case 006 的 MLA（同机 0.35 of peak）与
已知的 gather 结论（`sparse-mla` 卡记的「gather 与计算不重叠」），
最可能是 **KV 读取的调度方式**——kv 块按页取，取块之间串行等待。

## 4. 轮 4：扫 batch（decode 的真实扩展维度）

| B | median | GB/s | of peak | 每序列摊销 |
|---|---|---|---|---|
| 1 | 0.4696 ms | 9 | 0.01 | 469.6 µs |
| 8 | 1.0179 ms | 33 | 0.02 | 127.2 µs |
| 32 | 3.1606 ms | 42 | 0.03 | 98.8 µs |
| 64 | 6.0782 ms | 44 | 0.03 | 95.0 µs |

B 从 32 到 64，达成率只从 0.03 到 0.03（44 GB/s），每序列摊销停在 ~95 µs。
**已经饱和在 0.03 of peak。**

## 5. 结论：flash-attn 不适合当本机 decode 的 primary

```
① block_size 必须被 64 整除（报错文案写 gfx936，实际 gfx938）
② paged 比连续 KV 慢 1.60×，但两者都慢（33 / 21 GB/s）
③ decode 达成率只到 0.01~0.03 of peak，且与 query 数完全无关
④ 同机 MLA 吸收式 decode 能到 0.35（case 006）——**差一个数量级**
```

### 对照表（同一台机器，decode 形态）

| 实现 | 场景 | 达成率 | 来源 |
|---|---|---|---|
| MLA 吸收式（flash_mla） | KV 压进 latent | **0.35** | case 006 |
| flash_attn paged decode | 标准 MHA/GQA | 0.01~0.03 | 本 case |

**这个对照直接决定选型**：本机做 decode，MLA 类模型必须走 flash_mla 的吸收式路径；
标准 MHA 的 decode 用 flash_attn 会掉到 0.03，需要另找实现
（`boltops` 在 `impls` 里，但本 case 未测）。

## 6. 未测定项（不插值）

| 项 | 状态 |
|---|---|
| boltops 的 decode 实现 | **未测定**（卡里有此 impl，本 case 未跑通） |
| 0.03 的根因是否就是「块之间串行等待」 | 未测定（需 PMC，本机无 rocprof） |
| 连续 KV 的 33 GB/s 与 paged 21 GB/s 差在哪 | 未测定（1.6× 像是页表跳转开销，但无计数器证据） |
| GQA（H_kv < H_q）能否改善 | 未测定 |
| 与 vLLM/sglang 里实际用的 decode kernel 对比 | 未测定 |

## 7. 对本卡的落点

- `status` → measured，`ratio` = 0.03（B=32, KV=8192, D=128）
- **flash-attn 从 primary 降级**，理由是本 case 的 0.03 与 case 006 的 0.35 对照
- `modify_path` 改成「换实现」而不是「调参数」——参数调不动 0.03

## 复跑

```bash
python3 -I Plugins/ops/skills/op-optimize/cases/008-paged-decode-ceiling/bench.py
# 落盘：results/paged-decode.json
```
