# op-gemm-fp8 台账

> 绝对数字进 `~/.config/dcu-toolkit/kb/op-opt/`；本文件只留相对倍数、机制与判定。

## 一、当前状态

**draft**。契约、形状档位、变体矩阵、adapter、A/B 适配器均已就绪；
缺的是**本机安装 TileLang(hcu) / aiter 的 FP8 GEMM 后端**——门会 FAIL 并说明原因。

## 二、已确立的事实（来自 contract.yaml，有实测支撑）

| 事实 | 机制 | 影响 |
|---|---|---|
| gfx938 上 hipBLASLt 无 FP8 方案 | 实测确认 | `torch._scaled_mm` 这条路在本平台不可用 |
| fp8 在 ROCm 是 **fnuz** 语义（gfx950 除外） | 平台 ISA 约定 | 量化代码不能照搬 NVIDIA e4m3 |
| 裸 `cvt_pk_fp8_f32` 需 fp8-insts | gfx938 不提供该目标特性 | 必须用 `cast_to_f8` / `cast_from_f8` |
| tilelang 必须显式 `warp_size=64`、`target=hip` | 默认 32 会越界返回自身值 | 漏了会得到错误结果而非报错 |
| `k_pack=2` 是 CDNA 系取向 | 硬件 gather 向量化宽度 | bf16 用不了 MLS 路径 |
| M 分档是两个瓶颈类型 | M≤32 带宽型 / M≥128 算力型 | 不可用同一地板判据 |

## 三、待验证（open_questions）

1. `example_tilelang_gemm_amd_fp8_preshuffle.py` 在 gfx938 上的编译/精度/达成率
2. `k_pack=2` 与 `BLOCK_N` 的最优组合

## 四、复现方式

```bash
# 正确性门（tiny 档，adapter 自检；真实档留给真机 bench）
python3 scripts/opcheck.py correctness \
    --contract Plugins/ops/skills/op-gemm-fp8/contract.yaml \
    --shape tiny --variants tilelang

# A/B（需先装后端）
python3 Plugins/ops/skills/op-gemm-fp8/bench/ab.py \
    --variants tilelang --shape decode-m32 --rounds 5
```

## 五、纪律

- 参考实现用 `torch.mm`（fp32），**不用 `_scaled_mm`**——后者是被测对象。
- 分母按档位选：decode 档用带宽 1724 GB/s；prefill 档用 fp8 算力峰值。
  两档混用会把结论判反。
- `tiny` 档只用于验证门是否通，**绝不用于性能结论**。
