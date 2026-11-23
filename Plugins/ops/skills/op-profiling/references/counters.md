# PMC 计数器可用性表（gfx938 实测）

> 由 `scripts/pmc_probe.py` 对照真跑出来的 CSV 逐列核对。
> **本文件的每一行都有实测证据**；没有证据的一律标「未验证」。
> 判据纪律：**只有 `present` 的字段能写进瓶颈判据**。
> `all_zero` 与 `missing` 都不许当成「没问题」——那是把「没采到」当成「没发生」。

实测环境：gfx938 / BW1100 / DTK 25.10，采集命令见文末。

---

## 一、本机实测结论（2026-11-24，torch elementwise + 分配器 kernel）

样本：`HIP_VISIBLE_DEVICES=6 hipprof --pmc --pmc-type 3 -o /tmp/pmcout/x python3 k.py`
（`k.py` = 对 16 MB 张量做 5 次 `torch.add`）

| 字段 | 状态 | 列数 | 实测合计 | 能写进判据？ |
|---|---|---|---|---|
| `GRBM_COUNT` | **present** | 1 | 383397 | ✅ 周期基准 |
| `SQ_INSTS_VALU` | **present** | 1 | 5350400 | ✅ 含 MFMA，未拆分 |
| `SQ_ACTIVE_INST_VALU` | **present** | 1 | 5490176 | ✅ |
| `SQ_INSTS_VMEM_RD` | **present** | 1 | 81920 | ✅ |
| `SQ_INSTS_VMEM_WR` | **present** | 1 | 147456 | ✅ |
| `TA_TA_BUSY` | **present** | 16 | 10567736 | ✅ 跨 SE 实例求和 |
| `TCP_TCP_TA_DATA_STALL_CYCLES` | **present** | 16 | 2879178 | ✅ 访存停顿 |
| `TA_FLAT_READ_WAVEFRONTS` | **present** | 16 | 81920 | ✅ |
| `TCC_HIT` | **present** | 32 | 7906 | ✅ hit 不是字节 |
| `TCC_MISS` | **present** | 32 | 2228371 | ✅ miss × 128 B = DRAM 读字节 |
| `SQ_INSTS_LDS` | **present**（LDS kernel 实测） | 1 | 25600 | ✅ 需带 LDS 的 kernel 才非零 |
| `SQ_WAIT_INST_LDS` | **present**（LDS kernel 实测） | 1 | 9202 | ✅ 需带 LDS 的 kernel 才非零 |
| `SQ_LDS_BANK_CONFLICT` | **all_zero**（**故意构造冲突后仍为 0**） | 1 | 0 | ❌ **本机采不到真实值**——见下方 §1.1 |
| `SQ_WAVES` | **missing** | 0 | — | ❌ **本机 CSV 不提供这一列** |
| `SQ_INSTS_BARRIER` | **missing** | 0 | — | ❌ 采不到 |
| `SQ_INSTS_SALU` | **missing** | 0 | — | ❌ 采不到 |
| `SQ_INSTS_VMEM` | **missing** | 0 | — | ❌ 采不到（只有 RD/WR 分开的两列） |

### 1.1 `SQ_LDS_BANK_CONFLICT` 被判「采不到真实值」的证据

第二次实验专门构造了会冲突的 LDS 访问：

```c
__shared__ float s[256];
int t = threadIdx.x;
s[t] = in[t];
__syncthreads();
out[t] = s[(t * 7 + 13) & 255];   // 步长 7，256 个 lane 必然撞 bank
```

同一次采集里 `SQ_INSTS_LDS = 25600`、`SQ_WAIT_INST_LDS = 9202` 都正常非零，
**唯独 `SQ_LDS_BANK_CONFLICT` 仍是 0**。

⇒ 结论不是「没有冲突」，而是**这个计数器在本机不反映冲突**。
任何基于它的归因（"conflict 低所以没问题"或"conflict 高所以该 swizzle"）**都不成立**。
替代：手算 bank 映射（`(addr/4) % 32`），或用耗时消融（改 swizzle 后测同一 kernel）。

### 由此产生的三处文档更正

1. **`SQ_WAVES` 采不到。** `surfaces.yaml` 的 `occupancy` 条目写着
   `check: "pmc: GRBM_COUNT / SQ_WAVES"`，`cards.md` 写着
   「Occupancy = `SQ_WAVES ÷ GRBM_COUNT`」——**这两处在本机无法执行**。
   替代口径：用 CSV 自带的 `grd`（网格）与 `wgr`（每工作组线程数）
   算出 CTA 总数，再除以 CU 数与实测驻留档位。
2. **`SQ_INSTS_BARRIER` 采不到。** `barrier-width` 攻击面的 check 写的是
   `SQ_INSTS_BARRIER / SQ_INSTS_VALU`——本机无此列。改用
   `SQ_WAIT_INST_LDS / SQ_INSTS_VALU`（已在 LDS kernel 上实测非零 = 9202）。
3. **`SQ_LDS_BANK_CONFLICT` 采不到真实值。** 见 §1.1：故意制造冲突后仍为 0。
   `layout-pack` 里凡是以它为判据的写法都要改成手算 bank 映射或耗时消融。

---

## 二、CSV 白送的放置条件（G5 不需要 `.so`）

`hipprof --pmc --pmc-type 3` 的 CSV 里就有这几列，**是本次测量实际跑的值**，
比从编译产物读更贴近事实（产物可能被重新编译过）：

| 列 | 含义 | 实测样本 |
|---|---|---|
| `grd` | 网格大小（总 work-item 数） | 163840 |
| `wgr` | 每工作组线程数 | 256 |
| `lds` | 动态 LDS 字节 | 0 |
| `arch_vgpr` | 架构 VGPR 数 | 48 |
| `accum_vgpr` | 累加器 VGPR 数 | 0 |
| `sgpr` | SGPR 数 | 48 |
| `wave_size` | wavefront 宽度 | 64 |

⇒ **`scripts/results_schema.py` 的 placement 字段可以直接从这份 CSV 填**，
不必依赖 `inspect_so.py` 去找 `.so`。两者都可用，优先级：
**本次测量的 CSV > 编译产物**（后者可能不是当时跑的那个二进制）。

---

## 三、复现命令

```bash
# 注意：本版 hipprof **不认 `--` 分隔符**（加了会报 Invalid arg: --）。
# op-profiling/SKILL.md:36 的示例带 `--`，那是错的，照抄会直接失败。
HIP_VISIBLE_DEVICES=<idle> hipprof --pmc --pmc-type 3 -o /tmp/pmcout/x python3 your_bench.py

python3 -I scripts/pmc_probe.py /tmp/pmcout/x.csv        # 三态表
python3 -I scripts/pmc_probe.py /tmp/pmcout/x.csv --json # 机器读
```

---

## 四、未验证（下次补）

- `--pmc-type 3` 之外的档位（0/1/2）是否提供额外列（也许 bank conflict 在别的档位里）。
- `--kernel-name` 过滤后，实例列（`[0..N]`）的编号是否仍然跨 SE。
- bank conflict 是否需要用 `--pmc-read` / `--pmc-write` 两种 focus 模式才出现。

---

## 五、一句话

> 这份表的意义不是「有哪些计数器」，而是**「哪些没有」**。
> 决策树里写了采不到的指标，agent 会拿 0 当「没问题」——
> 那比不做归因更糟，因为它会给出一个看起来有依据的错结论。
