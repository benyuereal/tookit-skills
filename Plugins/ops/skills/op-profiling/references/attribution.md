# 归因表（模板）

> **只允许报一个主导瓶颈。** 堆指标等于没归因。
> 这张表回答的问题是「为什么慢」，不是「有多慢」——后者归 `op-bench`。

## 表格（每张深挖卡必填）

```yaml
attribution:
  dominant: l2-miss            # 五类之一，只许一个
  share: 0.62                  # 该瓶颈能解释的时间占比（0~1）
  evidence:                    # 必须是 PMC 数值，不许定性描述
    l2_hit_rate: 0.41          # TCC_HIT / (TCC_HIT + TCC_MISS)
    sectors_per_request: 3.8   # TA_FLAT_READ_WAVEFRONTS 与 DRAM 字节折算得出
    hbm_util: 0.52             # DRAM 字节 / (峰值带宽 × 墙钟)
  ruled_out: [hbm, lds, vgpr]  # 排除了什么 + 凭什么排除
  measured:
    kernel: "xxx_kernel"       # pmc.py 选中的主 kernel
    csv: /tmp/pmcout/pmc.csv   # 原始 CSV（可复跑）
    cu: 64
    sclk_mhz: 1350
    ts: '2026-11-23T13:40:00'
```

## 五类主导瓶颈与判据

判据全部来自 `scripts/pmc.py` 输出的**原值**（它不解释，解释在这张表）。

| dominant | 判据（同时满足） | 怎么算 |
|---|---|---|
| `hbm` | DRAM 有效带宽 ≥ 0.85 × 峰值；且 L2 命中不低 | `TCC_MISS 求和 × 128B / 墙钟` ÷ 峰值带宽 |
| `l2-miss` | 命中率低（<0.6）；且 HBM 未饱和（<0.7）；且 sector 效率差 | `TCC_HIT / (TCC_HIT + TCC_MISS)`；sector 效率 = 有用字节 / 请求字节 |
| `lds-conflict` | `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS` 高于同形状已知高效实现 2× 以上 | 必须**同形状三方对照**（本实现 / 对手 / 已知高效 matmul） |
| `vgpr-spill` | 编译产物 `scr > 0` 或 `arch_vgpr` 接近上限 | `pmc.py` 的 `geometry.scr`；或 `llvm-readelf --notes` |
| `issue-stall` | MFMA 占用低；且 `SQ_WAIT_INST_LDS` 或 `TCP_TCP_TA_DATA_STALL_CYCLES` 占比高 | 见下方「两条比值」 |

## 两条比值（跨形状可比，优先于绝对值）

```
(lane 指令占比) = (SQ_INSTS_VALU − MFMA) / MFMA
(LDS 指令占比)  = SQ_INSTS_LDS / MFMA
```

已知参照：调优 matmul 0.23 / 0.25，flash_mla 0.73 / 0.53。
比值高 ⇒ 每条 MMA 配的搬运指令太多 ⇒ 是**调度**问题，不是带宽问题。

## 三条纪律

1. **`evidence` 里出现「看起来 / 应该是 / 可能」⇒ 这张表判无效。**
   与项目已有的「机制化证伪」同源：机制能迁移，印象不能。
2. **稳态行取 GRBM 最小**，不取首行（PMC 下首次 launch 的 GRBM 虚高 ~40%）。
   这条由 `scripts/pmc.py` 强制，归因表只记录它选了哪一行。
3. **`ruled_out` 必须写「凭什么」**，不能只列名词。
   例：`hbm: DRAM 字节 42 MB / 0.9 ms = 46 GB/s，只有峰值 3% ⇒ 不可能受 HBM 限`

## 与 `share` 的关系

`share` 是**估计值**，允许粗（0.5 / 0.6 / 0.7 这种粒度）。它不用来算加速比，
只用来回答「先治哪个」。如果两个瓶颈的 share 接近（差 <0.15），
说明归因没收敛——回去补一组 PMC，而不是二选一硬报。
