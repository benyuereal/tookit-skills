---
name: dcu-perf-profiling
description: "DCU性能剖析方法论：rocprof/rocTracer命令模板与瓶颈判定"
version: 1.1.0
author: DCU Engineer
license: MIT
---

# DCU 性能剖析方法论（dcu-perf-profiling）

本 skill 定义**如何量化 DCU 上算子/模型性能**的标准动作与指标约定。是 `dcu-op-optimization` 的度量支撑，也可独立用于"这个服务为什么慢"的排查。

## 前置（必读纪律）
- **黄金法则：Profile → Diagnose → Plan，按这个顺序，绝不猜。** 没有报告之前不发明假设；没把观察到的模式匹配到已知诊断（见 §D 细粒度诊断）之前不动手写修复；不写一墙建议——按证据和预期收益排序。**不外包理解**：自己跑 profile、自己读报告、引用具体计数器值，禁止写「profile 显示是 memory-bound」这种空话。
- **gfx 标识、卡型号一律用物理机 `rocminfo` 查**（带 `ROCM_PATH`/`LD_LIBRARY_PATH`）。容器内 rocminfo 不可靠，曾踩坑。
- 若要在**容器内**跑 profiling：容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见（DEVCOUNT=0）。
- 指定部分卡：`HIP_VISIBLE_DEVICES=0,1,2,3`（编号从 0 起），不靠 `-g N` 自行挑卡。
- 进容器跑测试前先查 `NCCL_NCHANNELS_PER_PEER` 是否已设（镜像常预置=2），有则 unset 再测。

## 工具栈
| 工具 | 用途 |
|---|---|
| `rocprof` | 指令级采样、kernel 耗时、HW 计数器（occupancy、L2 hit、MEM 带宽） |
| `rocTracer` / `roctracer` | API+HIP 时间线、kernel 依赖、launch 开销 |
| `hy-smi` | 设备级监控（温度/功耗/利用率/显存），见 `dcu-hy-smi` |
| `rocminfo` | 设备 agent 信息、gfx、CU 数（物理机查） |
| `MIOpen` (MIOPEN_*) | 卷积/深度学习算子库分析：当热点是卷积类时，查 MIopen 是否选到最优算法 |

> ⚠ **版本/镜像工具差异（2026-08-28 <device-A> 双镜像实测）**：
> - **旧 26.04 镜像**（<device-B> 设备 vllm015 基线）：`/opt/dtk/bin` 只有 `hipprof`/`hipprof_ctrl`，**没有** `rocprof`/`rocprofv2`；`rocprof`/`rocprofv2` 在宿主物理机 DTK-25.04 的 `/opt/dtk-25.04/rocprofiler/bin/`。
> - **新 26.04 镜像**（<device-A> 设备 20260824 sglang 基线，DTK-26.04-rc4）：**带** `rocprof`/`rocprofv2`/`rocsys`，位于 `/opt/dtk/rocprofiler/bin/`——但该目录**不在默认 PATH**（env.sh 未加），`source /opt/dtk/env.sh` 后 `which rocprof` 仍失败，必须用绝对路径 `/opt/dtk/rocprofiler/bin/rocprof`（详见 dcu-dtk-layout §3.5）。
> → 通用纪律：容器内 `which rocprof` 失败**不等于**"没有 profiler"——先 `ls /opt/dtk/rocprofiler/bin/` 确认；hipprof（在 PATH 里）任何 26.04 镜像都有，本 skill §D 的 hipprof 计数器名正好对应。

> **hipprof 补充（DCU 原生 profiler，与 rocprof 互补）**：本仓库 `dcu-hipprof` 是 hipprof 总纲，此处给常用命令速查（整合自个人 skill 仓库 `<personal_skill_repo>` 的 `auto-*-opt`，2026-08-13）：
> ```bash
> # 指令级 PMC（始终先跑这个定位瓶颈）
> hipprof --pmc --pmc-type 3 <app>
> hipprof --pmc-read  --pmc-type 3 <app>   # 内存读剖析（memory-bound 时）
> hipprof --pmc-write --pmc-type 3 <app>  # 内存写剖析
> # ISA 验证（改完底层后必查编译产物）
> hipcc -o kernel.co kernel.hip --offload-arch=gfx936 -c
> dccobjdump --inputs=kernel.co --show-sass --show-instruction-encoding --separate-functions
> # 寄存器压力 / codeobj 分析
> hipcc --resource-usage kernel.hip --offload-arch=gfx936
> hipprof --codeobj-analyze <elf-or-so-file>
> # SQTT（难诊断的 stall）
> hipprof --sqtt --sqtt-type 1 --kernel-name <name> --output-type 0 -d <dir>/ <app>
> ```
> 关键 PMC 信号：SQ_INSTS_MMOP（MMAC 活动）、SQ_WAVES（occupancy，<20 提示 launch 轴）、SQ_LDS_BANK_CONFLICT（bank conflict）、SQTT waitcnt stalls（>30% 提示双缓冲缺失）。峰值参考（gfx936）：FP16 Tensor ≈442.4 TFLOPS、HBM ≈1.8 TB/s，以 `rocminfo` + 实测为准。

### MIopen 热点算子分析（卷积/深度学习算子库）
当框架 Profiler（见 `dcu-framework-profilers`）定位到热点来自 MIOpen（如卷积），用环境变量抓取 MIOpen 的算法选择日志，确认是否选到最优实现：
```bash
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_LOG_LEVEL=2
# 或用 miopenDriver 工具直接测不同算法/配置下的性能，确保卷积实现最优
```
> MIOpen 日志分析 SOP 与「算子 size 抽取」见下方「算子 size 抽取（给产品部优化用）」。卷积类热点优先确认 MIOpen 算法选择，再决定是否改 kernel。

## 标准剖析流程

### A. 宏观：定位慢在哪一层
1. 用框架/服务的 benchmark（如 vLLM 的 `benchmark_serving.py`）拿到端到端指标：TTFT、TPOT、吞吐(req/s)。
2. `hy-smi` 看 DCU 利用率是否打满——若利用率低但吞吐低，多为**调度/通信/IO**瓶颈，不是算子问题。

### B. 微观：定位哪个 kernel 慢
```bash
# 基本 kernel 计时（耗时排行）
rocprof --timestamp on --stats python your_inference.py

# 带 HW 计数器（occupancy / L2 / MEM）
rocprof -i ./profile_metric.txt python your_inference.py
# profile_metric.txt 例：
#   pmc: GPUBusy, SQ_WAVES, L2CacheHit, MEM_BUSY, VALUUtilization
```
- 产出：kernel 耗时排序表（按 `Time(ms)` 降序）。
- 取耗时占比 >10% 的 kernel 进入 `dcu-op-optimization` 第 2 步。

### C. 瓶颈判定（核心）
| 指标现象 | 判定 | 优化方向（见 dcu-op-optimization 第3步） |
|---|---|---|
| MEM_BUSY 高、ArithmeticUtil 低、算术强度低 | Memory-bound | 合并访存 / 向量化 / LDS tile |
| VALUUtilization 高、GPUBusy 高、算术强度高 | Compute-bound | 提升 occupancy / MFMA / 指令并行 |
| GPUBusy 低、kernel 小且多 | Latency-bound | kernel fusion / 减少 launch |
| kernel 间大段 gap（tracer 可见） | 同步/通信瓶颈 | 计算通信重叠 / custom all-reduce |

### D. 细粒度瓶颈诊断（六维分析 + 14 模式，读 hipprof/rocprof 输出）

> 定位：§C 给出粗判定（memory/compute/latency/通信四类）。当需要**精确定位到具体模式、决定先修哪个**时，走本节。
> 方法论参考：`mit-han-lab/kernel-design-agents` → 子模块 `ncu-report-skill`（NVIDIA B200 / Nsight Compute 版），此处 **DCU 化**到 hipprof / rocprof。
> **纪律**：计数器名只用本 skill 已验证的（见 §指标约定 映射表）；未验证的一律标「待真机验证」，**禁止编造** hipprof/rocprof 计数器名。结论必须引用具体计数器值。

#### D.0 一次运行一个目录（强制）
```
profile/<run_name>/
  harness/     # 独立 harness（若需隔离编译，见 dcu-kernel-dev）
  reports/     # hipprof/rocprof 原始输出
  analysis/    # 解析后的指标、stall 热点、时间线
  REPORT.md    # 最终诊断报告（排序优化计划，见 dcu-op-optimization 输出物模板）
```
**一次运行一个子目录，绝不复用旧目录**，避免跨运行污染。`<run_name>` 用「算子名_日期_改动标签」。

#### D.1 六维分析（走完全部六维，别停在第一个发现）
某次可能只有 1–2 维主导，但**六维都要走一遍**。

- **Dim 1 — Launch 几何 & 占用率（wave 数学）**：grid 够不够填满 DCU？占用率被寄存器/LDS/block 大小哪个卡住？
  - 已验证计数器：`SQ_WAVES`（<20 提示 launch 轴问题）。
  - wave 数学（DCU 用 CU 数替换 SM 数，CU 数从物理机 rocminfo 查）：
    ```python
    blocks_per_cu = min(occ_limit_blocks, occ_limit_registers, occ_limit_lds, occ_limit_waves)
    wave_size = blocks_per_cu * num_cus
    num_waves = (total_blocks + wave_size - 1) // wave_size
    last_wave_blocks = total_blocks - (num_waves - 1) * wave_size
    last_wave_utilization_pct = last_wave_blocks / wave_size * 100
    ```
  - 读法：waves/CU<1 → grid 太小填不满；∈[1,2) → 有尾波（尾效应量级≈`last_wave_blocks/wave_size × block_exec_time/total_time`）；>4 → 已摊平；理论占用率 100% 但实测<<100% → 瓶颈是 stall 转 Dim 3；理论<100% 且寄存器最紧 → 降寄存器/`__launch_bounds__`；LDS 最紧 → 降 tile。
- **Dim 2 — 尾效应 / 负载不均**：各 block 是否差不多时间结束，还是少数 outlier 拖长？
  - 信号：输入 `max_seq_len/avg_seq_len>3`；每 CU 活跃周期最快/最慢差 5–100×；时间线长尾；`SQ_WAVES>1.05` 有 partial wave。
  - 来源：①变长 per-CTA 工作（prefix-sum/累计长度驱动）②kernel 内分支提前退出 ③work-stealing 没做好负载均衡。
  - 修复：变长工作做 chunk（序列类 time-chunking）或 oversubscribe+work-stealing。
- **Dim 3 — Stall 分解 + 逐行热点**：wave 没在 issue 时在等什么？哪些源码行产生最多 stall？
  - 已验证计数器：`SQTT waitcnt stalls`（>30% 提示双缓冲缺失）。SQTT 是 DCU 指令级 stall 采样（`hipprof --sqtt`）。
  - 读法：长延迟依赖 stall 高 → 等 global load 返回（合并访存/复用/加 ILP，转 Dim 6）；短延迟依赖 stall 高 → LDS/计算链（加 ILP/缩短依赖链）；barrier/同步 stall 高 → 减同步/修发散；计算管饱和 → 真 compute-bound；not_selected/selected → 健康信号忽略。
  - 逐行热点：用 SQTT 输出定位到具体 `buffer_load`/`s_barrier`/计算指令行。
- **Dim 4 — Tensor Core（MFMA）利用率**：用没用 tensor core？用得好不好？
  - 已验证计数器：`SQ_INSTS_MMOP`（MMAC 活动）。
  - 读法：=0 → matmul 类 kernel 几乎总是漏优化；>0 但<<50% → 没喂饱（数据没及时到 Dim 6 或 tile 不对）；>50% → 做得不错重点放别处。
  - 修复：0% 且矩阵乘形状 → 围绕 MFMA 重设计（`v_mmac`/`tl.dot`，见 `dcu-kernel-dev`），通常 2–10×。
  - DCU 注意：不同 gfx 的 MFMA 指令集不同（gfx928 有 `__builtin_amdgcn_mmac_f32_16x16x8f32` 等；gfx936 的 BW150/BW100 **不是** 928 那套 MC intrinsic）。先 `dcu-kernel-dev/scripts/dcu_detect_arch.sh` 现查 gfx，别假设。
- **Dim 5 — 时间线形态**：CU 利用率随 kernel 生命周期怎么变？
  - 工具：hipprof Perfetto trace（`--hip-trace` 导出，`https://ui.perfetto.dev/` 读）。
  - 形态读法见 D.3。
- **Dim 6 — 访存模式 & 缓存效率**：global load 合并了吗？缓存命中吗？HBM 真忙吗？
  - 已验证计数器：`MEM_BUSY`（HBM 忙）、`L2CacheHit`（L2 命中）。
  - 读法：`MEM_BUSY`≈80–100% → 真 HBM 带宽 bound（减字节/用 LDS 摊读）；`MEM_BUSY`<<10% 但慢 → **不是**带宽 bound，是 latency-bound（Dim 3）或 compute-bound（看 `VALUUtilization`）；`L2CacheHit`>90% → 局部性好；<50% → L2 被打穿读落 HBM；合并质量差 → Pattern C/D；寄存器溢出（local memory，HBM-backed）→ 降寄存器压力。

#### D.2 诊断 Playbook：信号 → 成因 → 修复（14 模式，DCU 化）
> 用法：对每个**观察**读 信号/成因/一线修复/深层修复/例外。多数 kernel 同时命中 2–4 个模式，**按量级排序**（stall 百分比+预期收益），先修最大的。计数器名：已验证的直接写，未验证标「待真机验证」，**不编造**。

- **A 小 grid / CU 空闲**：信号 `SQ_WAVES<0.5`、grid block 数<CU 数。成因：CTA 比 CU 少，部分 CU 全程空闲。一线：增大 grid（split-K / 按 head·channel 拆 / grid-stride loop 仅当 work unit 便宜）。深层：persistent kernel（每 CU 一 block + atomic counter 取活）/ 与相邻 kernel 融合。例外：LLM decode（batch=1）天然小，split-K over KV 是标准缓解；多级 reduction 末级天然小，融进生产 kernel。
- **B 尾效应（变长输入）**：信号 输入 max/avg>3、每 CU 活跃周期差 5–100×、时间线长尾、`SQ_WAVES>1.05` 有 partial wave。成因：每 CTA 迭代变长内循环，序列长度差异大时少数长序列 CTA 拖尾。一线（便宜）：packed batching/按长度排序（应用层）；长序列拆到更多 CTA（加 `split_factor` grid 维 + 小 post-reduction）。深层：chunkwise kernel（每序列切固定 chunk 并行 + 小 recurrence 缝合，如 flash-linear-attention `chunk_delta_rule_fwd`）；classify-and-dispatch（短走简单路径、长走 chunk 路径）。例外：短 kernel（<10µs）partial-wave 绝对成本小；已预排序/预打包。
- **C 非合并 global load**：信号 sectors/request 远高于理想值（理想 4，>8 严重；DCU 对应计数器名待真机验证）、肇事 load 行主 stall 是长延迟依赖。成因：warp 内 lane 访问非连续地址，硬件取多余 sector 只有少数 lane 用。一线：重排 thread↔data 映射——`x[lane*K+i]`（stride K）翻成 `x[lane+i*32]`（合并）；AoS→SoA。深层：LDS 做转置（合并 load 进 LDS 再任意访问）；向量化（标量 load → `float2/float4` 宽 load）。例外：随机索引 gather/scatter（稀疏 matmul、embedding）天然非合并，能排序就排序；图/树遍历。
- **D 稀疏写（低 store 效率）**：信号 store 每 sector 有效字节远低于上限（DCU 对应计数器名待真机验证）、代码有 `if (lane_id < K) { out[...] = ... }` 模式。成因：只有部分 lane 写，L1 store buffer 刷半空 sector。一线：打包写——warp 先用 shuffle/LDS reduction 集体产出 K 个值，再让恰好 K 个连续 lane 做 K 次连续写；K≥32 全 lane 写，K<8 把多次迭代结果打包成一次向量化写。深层：先写 LDS，block 末尾做一次合并 global store。例外：histogram/scatter 天然稀疏，走 Pattern G。
- **E Latency-bound（长延迟依赖主导）**：信号 长延迟依赖 stall 占采样>40%、`MEM_BUSY<10%`（→非 HBM 带宽 bound）、热点行是 global load。成因：wave issue 一个 load 后 stall 等它返回，常叠加低占用率或 ILP 不足。一线：增加 in-flight 访存——unroll load 循环（4–8 个 load 发出后再用值）；加独立 wave 提占用率（Pattern J）；异步 bulk load（DCU 对应机制待真机验证，NVIDIA 是 cp.async/TMA）。深层：软件流水线（算 tile N 时预 load tile N+1 进 LDS）；复用数据移 LDS 让后续 load 命中 L1。例外：pointer chasing/图遍历，数据依赖链根本性。
- **F Compute-bound 但没走 tensor core**：信号 `VALUUtilization` 高（>50%）、`SQ_INSTS_MMOP=0`、workload 是 matmul 类。成因：用标量 FMA 走 ALU 管而非 tensor core。一线：用 MFMA（`v_mmac`/`tl.dot`，见 `dcu-kernel-dev`）；手搓太多就用 rocBLAS/MIOpen（已按目标 arch 调优）。深层：重排数据布局满足 MMA tile 形状约束；LDS staging。例外：非矩阵 workload（reduction/sort/elementwise）tensor core 帮不上；小矩阵（M,N,K<32）tile 太粗。
- **G Atomics 竞争**：信号 长延迟 stall 集中在 ATOM/RED 指令、L2 吞吐高但计算吞吐低。成因：大量线程原子更新少数位置→串行化。一线：分层 reduction——warp 内 `__shfl_down_sync`（无原子）；block 内 LDS reduction（无原子）；block 间末尾单次原子。深层：LDS histogram 再一次合并刷 global；bucketing（写 `out[tid % N_buckets]` + merge kernel）。例外：RCCL 类通信，原子根本性。
- **H LDS bank conflict**：信号 `SQ_LDS_BANK_CONFLICT` 高、短延迟 stall 集中在 LDS load 行、访问模式有对齐 bank 边界的规则 stride。成因：LDS 32 bank/4B，同 bank 访问串行化。一线：padding——`__shared__ float tile[32][33]` 而非 `[32][32]` 打破规则 bank 对齐（BM×(BK+2) padding 法，见 `dcu-kernelwiki/wiki/lds_bank_conflict.md`）。深层：swizzle（XOR 扰索引让访问散到不同 bank）；重排布局让 lane 访问不同 bank。例外：broadcast 读（全 lane 读同址）无冲突；LDS 访问量低不值得折腾。
- **I 同步开销**：信号 barrier stall 占采样>20%、源码热点行是 `s_barrier`/`__syncthreads`。成因：`__syncthreads` 等最慢 wave，叠加 per-wave 不均会放大。一线：只需 warp 级同步时换 warp 级原语（`__shfl_sync`/`__ballot_sync`/`__syncwarp`）；减少同步总次数，合并多个同步阶段。深层：warp 专用化（producer/consumer wave + named barrier，DCU 对应 `__builtin_amdgcn_sched_barrier`，见 `dcu-op-optimization` 案例八）。
- **J 实测占用率 << 理论占用率**：信号 理论占用率>50% 但实测<<50%。成因：理论是「能驻留」的最大 wave 数，实测是「真在跑」的，差距来自 stall（留空槽）/不均（部分 CU 空）/短 kernel（warmup 主导）。读法：差距大且命中 Pattern B（尾效应）→修不均会闭合差距；无不均→看 stall 原因（Pattern E/H/I）。一线：找造成差距的 stall 原因，处理对应 pattern。
- **K 寄存器溢出**：信号 local memory load/store 计数>0（DCU 对应计数器名待真机验证）、每线程寄存器数高（DCU 具体阈值待真机验证）。成因：编译器装不下所有活变量，溢出到 local memory（HBM-backed）。一线：`__launch_bounds__(maxThreadsPerBlock, minBlocksPerCU)` 告诉编译器守寄存器预算。深层：减少活值（重算代替缓存、拆 kernel）；per-thread 数组移 LDS 显式索引。例外：大融合 kernel（FlashAttention）接受少量溢出换上游更大收益。
- **L 无意用了 FP64**：信号 本应 FP32 的 kernel 里 FP64 管活跃>0。成因：C/C++ 浮点字面量（`1.0`/`0.5`/`3.14`）默认 `double`，`float x = a + 1.0*b` 会把 `a+1.0*b` 提升成 double。一线：所有字面量加 `f` 后缀（`1.0f`/`0.5f`/`3.14f`）；超越函数用 `__expf`/`__logf`/`__sinf` 变体。
- **M Pipeline 气泡（无计算/访存重叠）**：信号 时间线 `CU 吞吐` 与 `HBM 吞吐` 锯齿交替、长延迟 stall 高但 HBM 吞吐也高。成因：load 一个 tile、算它、load 下一个——单缓冲。一线：双缓冲——两个 LDS tile，算 tile A 时 load tile B，阶段间 `__syncthreads`。深层：多阶段 pipeline（3–4 阶段）；异步 load（DCU 对应机制待真机验证）。
- **N Wavefront 发散**：信号 每指令执行的 thread 数远低于 wavefront 宽度（DCU wavefront=64，理想 64）、发散分支聚集在特定源码行。成因：warp 内 lane 在分支走不同路径，硬件串行化。一线：重排让同 wavefront 内 lane 走同分支（能排序/分区就排）；`if(c) a else b` 转无分支 `mask*a + (1-mask)*b`（两侧都便宜时）。例外：warp 内树 reduction（末几步半/四分之一活跃）用 `__shfl_down_sync` 干净处理；边界处理（tensor 边缘少数 wave）不值得较真。

> **DCU 特有提醒**：wavefront=64（非 NVIDIA 的 32），所有「per-warp / 32 lane」的直觉要按 64 重算；LDS 32 bank/4B；MFMA 指令集随 gfx 变（先查 gfx）。

#### D.3 时间线形态读法（hipprof Perfetto）
用 `hipprof --hip-trace` 导出 Perfetto trace，`https://ui.perfetto.dev/` 读。看 **CU 吞吐 + HBM 吞吐 + 长延迟 stall** 多条曲线并排，区分形态：

| 形态 | 含义 | 指向 |
|---|---|---|
| 平高 → 干净落 | 理想，均衡、CU 填满 | 无 |
| 平高 → 长渐降尾 | 尾效应 | Dim 2 / Pattern B |
| 平低 | grid 太小 或 严重 stall-bound | Dim 1 / Dim 3 |
| 周期锯齿（compute↔memory） | 无计算-访存重叠，缺 pipeline/双缓冲 | Pattern M |
| 慢爬升 → 平中 → 干净落 | 有 warmup 后稳态 | 通常正常 |

> 短 kernel（<20µs）采样少，时间线解读要谨慎。

#### D.4 输入分布检查（尾效应前置，必做）
若 per-CTA 工作由某数组驱动（如 `per_element_lengths`/prefix-sum）：
```python
work_per_cta = [...]  # 从驱动内循环次数的量推导
avg = sum(work_per_cta) / len(work_per_cta)
print(f"max/avg = {max(work_per_cta)/avg:.2f}x, max/min = {max(work_per_cta)/min(work_per_cta):.2f}x")
```
**`max/avg > 5×` → 显著尾效应风险**，优先走 Pattern B。变长输入 workload 的 #1 瓶颈经常就是负载不均，**永远先查输入分布**。

#### D.5 排序优化计划（写进 REPORT.md）
按 `(预期收益) × (工作量比)` 排序，stall 百分比+预期收益是最佳估计器。**最多 3–5 条**（>5 条会稀释信号，且 >5 优先级通常各 <5% 收益）。被拒的候选要记原因，不要静默丢弃。
```
Priority 1: <pattern> — <具体修复>
  Evidence: <计数器值>
  预期收益: X%
  Effort: low / medium / high
  Why now: <为什么这是最高杠杆的修复>
Priority 2: ...
```
> 完整留痕模板（环境/baseline/改动/正确性/性能）见 `dcu-op-optimization` 输出物约定；多轮迭代的预注册实验计划（门禁/停止条件）见 `humanize-gen-plan`。

## 常见优化方向（通用 checklist）

> 源自《DCU性能分析及常用优化技术手册》Ch5（内部文档 file_id=<internal-file-id>）。下列方向跨框架通用，落实前**必须先量化对比**（见本 skill 指标约定 + `dcu-op-optimization` 闭环），禁止凭经验直接改。

- **计算-通信重叠**：Timeline 上 RCCL/NCCL 是否与 Kernel 真正并行。理想情况下反向传播时通信应完全隐藏在计算之后；若出现大片连续 AllReduce 且 GPU 空闲 → 调 `bucket_size` 或启用梯度累加。详细通信分析见 `dcu-hipprof` / `dcu-rccl-test`。
- **Pipeline Bubbles（管道气泡）**：Megatron-LM 等 PP 下，各 micro-batch 执行间隙的气泡占比（Idle/总时间）不应超过 `(pp_size - 1) / num_microbatches` 的近似上界；实际更大可能是 load balancing 不均或通信延迟。
- **调度器延迟（vLLM）**：vLLM trace 中关注 scheduler 耗时，以及 Prefill/Decode 切换时的显存整理（block copy）耗时；调度延迟长会抬高 TTFT，调 `max_num_seqs` / `block_size`。vLLM 框架级 Profiler 见 `dcu-framework-profilers`。
- **CPU 侧瓶颈**：用 `ncu --section "OSRuntime"` 或 nsys 的 osrt trace 查 CPU 线程是否在 `cudaLaunchKernel`/`cudaStreamSynchronize` 处长时间阻塞；常见原因为 Python GIL、DataLoader 线程数不合理、过多同步。若 `hipExtModuleLaunchKernel` 次数爆炸（典型如 LSTM 案例），用 **HIP Graph / CUDA Graph** 回放绕过逐算子下发 → 详见 `dcu-perf-cases`。

## 算子 size 抽取（给产品部优化用）

> 源自外链 SOP 文档（内部文档 file_id=<internal-file-id>）。当 hipprof/rocprof 定位到热点算子、但自身无优化想法时，抽取该算子的 **shape/size 日志**交给产品部，由其在指定 size 上做 rocblas/miopen 优化。

### MIopen 算子 size 抽取
程序执行前加环境变量，运行后重定向文件做后处理：
```shell
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_LOG_LEVEL=6

cat xxx.txt | grep MIOpenDriver | sort -n | uniq -c > miopen.log
```

### rocBLAS 算子 size 抽取
```shell
export ROCBLAS_LAYER=3
cat xxx.log | grep "./rocblas-bench -f gemm" | sort -n | uniq -c > rocblas_size.log
```

### 完整示例（含 MLP 抽取）
```shell
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCBLAS_LAYER=3
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_LOG_LEVEL=6

<command> 2>&1 | tee -a ${logfile}

python demo.py 2>&1 | tee log.log
cat log.log | grep "./rocblas-bench -f gemm" | sort -n | uniq -c > rocblas_size.log
```

### 用抽取到的 size 在 DTK 下复测耗时
拿到 `rocblas_size.log` 中的 `./rocblas-bench -f gemm_ex ...` 命令，可在 DTK 自带 benchmark 工具直接复测该 size 耗时（验证产品部优化前后差异）：
```shell
export HIP_VISIBLE_DEVICES=0
cd /opt/dtk/lib/rocblas/benchmark_tool
./rocblas-bench -f gemm_ex --transposeA T --transposeB N -m 11264 -n 1 -k 4096 \
  --alpha 1 --a_type bf16_r --lda 4096 --b_type bf16_r --ldb 4096 --beta 0 \
  --c_type bf16_r --ldc 11264 --d_type bf16_r --ldd 11264 \
  --compute_type f32_r --algo 0 --solution_index 0 --flags 0
# 输出含 rocblas-Gflops 与耗时 us，按 size 对比优化前后
```
> 复测用 `HIP_VISIBLE_DEVICES` 指定单卡，size 必须与线上抽取到的完全一致才有可比性。

## 指标约定（写报告/留痕时统一用语）
- **耗时**：单调用 `ms`，括号内注明 shape 与精度（BF16/FP16/FP8）。
- **带宽利用率** = achieved bandwidth / 理论峰值（HBM 带宽查卡规格）。
- **算力利用率** = achieved FLOPS / peak FLOPS。
- **加速比** = 旧 ms / 新 ms（同条件）。
- 任何对比必须注明：**卡型号(gfx) / DTK 版本 / 精度 / shape** 四要素，缺一对比无效。

### 计数器名映射（NCU 概念 → DCU，只用已验证的）
> §D 细粒度诊断引用计数器时，**只用下表已验证的 DCU 计数器名**；未验证的映射标「待真机验证」，**禁止编造** hipprof/rocprof 计数器名。新增映射前先在真机 `hipprof --pmc` / `rocprof` 跑一遍确认名字存在。

| NCU 概念 | DCU 计数器（已验证） | 备注 |
|---|---|---|
| occupancy / waves | `SQ_WAVES` | <20 提示 launch 轴问题 |
| LDS bank conflict | `SQ_LDS_BANK_CONFLICT` | 32 bank/4B |
| tensor core 活动 | `SQ_INSTS_MMOP` | =0 提示没用 MMAC |
| stall（指令级） | `SQTT waitcnt stalls` | >30% 提示双缓冲缺失 |
| HBM 忙 | `MEM_BUSY` | |
| 计算利用率 | `VALUUtilization` | |
| GPU 忙 | `GPUBusy` | |
| L2 命中 | `L2CacheHit` | |
| 算术强度 | `ArithmeticUtil` | |

**待真机验证（§D 用到但尚未在 DCU 确认计数器名的）**：逐行 stall 热点的 SQTT 具体输出字段名、寄存器溢出（local memory load/store）的 PMC 计数器名、每 CU 活跃周期分布（尾效应 per-CU 方差）、合并质量（sectors/request 等价物）、异步 bulk load 机制（NVIDIA cp.async/TMA 的 DCU 对应）。

## 输出物
- 一份 `rocprof --stats` 原始输出（保留，别只存结论）
- 一张 kernel 耗时排序表（前 N 个）
- 瓶颈判定结论（上表 C 的某一类）

## 与 SGLang 端到端 profiling 衔接（借鉴自 generate-profile + benchmark-and-profile）

SGLang 的端到端 profiling 与 DCU 的框架级 profiling 互补。做 DCU 推理服务（vLLM/SGLang）整体耗时排查时可借鉴：
- **服务启动 + 健康检查**：`sglang serve --model-path <model> --port <port>` 后轮询 `/health` 等就绪，再下发请求。
- **e2e profiling**：`sglang.test.send_one --profile` 触发 profiler，产出 Chrome trace（`*.trace.json.gz`）与 server_args.json；`--profile-by-stage` 可拆 prefill/decode。
- **基准 + 对比**：`benchmark-and-profile.md` 里先跑 baseline 再跑优化后，用 `compare_perf.py` 输出 Markdown 对比表；注意**仅以 denoise latency 或 TTFT/TPOT 作主指标**，正确性门禁不过不能算提升。
- **阶数纪律**：`ncu` / `nsys` 在 DCU 没有对应（用 `rocprof`/`hipprof`）；但「先基准→改→复测→对比」的流程思想通用。

## 单请求 SGLang profiling + Perfetto 读 trace（借鉴 SGLang 官方博客，DCU 需重测）

> **来源与标注**：源自 SGLang 官方博客实战文（NVIDIA **L4** + **Qwen3.5-0.8B**，单请求 prefill/decode trace 阅读）。**这是 CUDA 平台方法，DCU/ROCm 上内核名、带宽数字、CUDA Graph 行为均需在海光 DCU 上重测**，仅作"参考、可尝试"用，禁止把 L4 的 187 tok/s、grid=248320/8 等当 DCU 事实硬编码。

### 核心结论（先记住，再学方法）
- **Prefill**：由重复的 `GDN + FFN` 块主导，末尾是**巨贵的词表投影 GEMV**（vocab projection，`[1,1024]@[1024,248320]`，grid=248320/8，仅跑一次却是最慢内核之一）。
- **batch=1 Decode**：被一系列**瘦长 GEMV** 主导，**内存带宽受限（bandwidth-bound）而非计算受限**——这是 decode 瓶颈的底层物理解释，也印证前文「decode 需 CUDA/HIP Graph」的根因（不只是 launch overhead，更是 GEMV 算术强度低、权重读多算少）。
- **batch 增大改变故事**：瘦长 GEMV 变肥接近 GEMM，GPU 才有机会复用权重 → **单看 batch=1 会误判，必须多 batch 对比**（与"单点结论不可靠"纪律一致）。

### 方法（DCU 上几乎 1:1 可复用，端点/流程与 GPU 厂商无关）
1. **数学 roofline 预判（先做）**：`带宽上限(GB/s) / 每 token 权重(GB) = 理论 tok/s`。文章 L4：300/1.6≈187 tok/s → 假设 decode 带宽受限。DCU 上换 **BW150/BW100/BW1000 对应 HBM 带宽**重算。
2. **单请求 trace 抓取**：SGLang 自带 profiling 端点 ——
   ```bash
   curl -X POST http://127.0.0.1:30000/start_profile -H "Content-Type: application/json"
   # 随后下发请求（如 bench_serving --num-prompts 1，或真实单请求）
   # 结束：/stop_profile，产物在 Perfetto trace
   ```
   - 设 warmup 若干步再剖析后续步，避免首步初始化噪声。
   - **把 prefill 与 decode 拆成两个 trace**（SGLang 支持按 stage 分离），便于阅读。
3. **读 trace（Perfetto UI `https://ui.perfetto.dev/`）双时间线**：上半 CPU 侧、下半 GPU 侧。CUDA 启动异步 → 先读 CPU 端启动、再看 GPU 端执行。decode 被 **CUDA Graph 捕获**（CPU 时间线不逐层显示），要点进 **graph replay** 看真实 GPU 内核。
4. **读图三步法（可迁移到任意模型/卡）**：
   - ① **架构预判**：先按模型结构（如 6 组 GDN+FFN + 注意力 成组重复）预期 trace 峰值形态；
   - ② **对照 trace 验证**：成组重复峰值 ↔ 模型重复块；内核名对照张量形状（如 vocab projection grid=248320/8 反推词表大小）才能读懂；
   - ③ **瓶颈归属**：GPU 时间线里重复蓝色 GEMV 条柱 = 瘦长投影 = 带宽受限；落到"该路径是 bandwidth-bound 还是 compute-bound"。

### DCU 适配注意
- 海光 fork 的 SGLang（如 <device-A> 上 0.5.12）**同样支持 `/start_profile` + Perfetto**，单请求剖析流程可直接复用；区别在 GPU 时间线里是 **HIP kernel 名**（非 CUDA kernel），且带宽数字换成 DCU 规格。
- **必须先做 roofline 预判再抓 trace**：DCU 各机型 HBM 带宽不同（BW150/BW100/BW1000），decode 是否带宽受限以 DCU 实测为准，不抄 L4 结论。
- 与 `dcu-sglang-source`（海光 SGLang 适配索引）交叉引用：DCU 上跑本流程前先确认 kernel 命名与 `--attention-backend` 配置。
- 已跑出 trace.json 想做统一 kernel/空泡/融合分析（torch.profiler 系，跨 SGLang/vLLM/TRT-LLM）→ 通用互补 skill `llm-torch-profiler-analysis`（已整合进本仓库，自动出分析文档）。

## CUDA/Kernel 崩溃调试（借鉴自 debug-cuda-crash）

处崩溃（illegal access / device-side assert / NaN）时，用「输入保留」三步法定位：
1. **启用日志装饰**：`@debug_kernel_api` 装饰器在崩溃前记录调用边界（函数名、输入张量 shape/dtype/device、统计 min/max/nan/inf）；级别可调（level 1 函数名、level 3 输入元数据、level 10 完整 snapshot 含 inputs.pt）。
2. **复现崩溃**：把 crash 程序用临时脚本复现，日志保留最后一次 API 边界（崩溃前输入），`SGLANG_KERNEL_API_DUMP_DIR=/tmp/...` 留 dump。
3. **二分排查**：从 dump 里读 inputs.pt/metadata.json，比对合法/非法形状、NaN/Inf 分布，锁定损坏输入或 kernel 逻辑。

DCU 上对应用：`hipprof` 的 trace/PMC 段捕捉非法访问与断言；`hy-smi` 监控 NaN/温度；数值边界检查见 `dcu-train-precision`。

## 与兄弟 skill 关系
- `dcu-op-optimization`：拿到本 skill 的量化结果后走"改→验证"闭环。
- `dcu-vllm-source`：定位慢 kernel 属于 vLLM 哪段源码/后端。
- `dcu-dtk`：rocprof/rocTracer 随 DTK 提供，版本选型和路径见它。
- `dcu-dtk-layout`：DTK 内部目录/工具速查（hipcc·rocminfo·rocprof 等工具路径与用法、lib 库分组），排查「工具找不到 / 库缺」时先查它。

## 使用须知
- 经验仅供参考。先核对**设备型号 + 驱动/DTK 版本 + 报错原文**再下手。
- profiling 数字**严禁写入本 skill**（会变/过期），只存方法。
- 高危操作谨慎；优先测试机验证，确认带外恢复通道。

## § PMC 计数器速查（DTK-26.04 实测，2026-10-20 P0 采集）


## 计数器目录（静态，版本绑定）
- 位置：/opt/dtk*/share/profiler/counters/{basic_counters.xml, derived_counters.xml}（DTK 安装目录内，随版本变）
- DTK-26.04：basic 1647 + derived；XML 已拉档（<device-C> 实拉）
- 自定义计数器名有效性 = 该目录 + 工具 metric 解析（名字有效 ≠ 能取到值）

## 运行时验证（gfx936, <device-A> 卡 4, 2026-10-20）
1. **hipprof --pmc --pmc-type 3 <app>** 可用：输出 pmc_results_<pid>.csv；默认集 ~100 计数器有值：
   - 每 kernel 行含 arch_vgpr / sgpr（每 kernel 寄存器实际用量，如 1GB copy kernel arch_vgpr=12/sgpr=32）
   - 有值：SQ_INSTS_VALU/VMEM_RD/VMEM_WR/LDS/FLAT_LDS_ONLY、SQ_WAIT_INST_LDS、SQ_LDS_BANK_CONFLICT、GRBM_COUNT、TCC_HIT[32]/TCC_MISS[32]（L2 每 slice）、TA_FLAT_READ_WAVEFRONTS[16]、TA_TA_BUSY[16]、TCC_EA_WRREQ_STALL[32]、TCP_TCP_TA_DATA_STALL_CYCLES[16]、Dispatch/Begin/End/CompleteNs
   - ⚠ TCC_HIT 每片≈196608 整值（疑似计数饱和），L2 命中解读需谨慎
2. **rocprof -i metric.txt 自定义计数器：DTK-26.04 不可用**（★ 避坑）
   - 4/4 失败（<device-A> 宿主卡4/卡2 + <device-B> 特权容器）：Error: Context Create failed → abort
   - 20 个候选名（TA_BUFFER_COALESCABLE_WAVEFRONTS、TCC_SECTOR_HIT、SQ_INSTS_VMEM_RD_REPLAY、SPI_RA_VGPR_SIMD_FULL_PS 等）解析通过（目录内有效名），值取不到
   - 替代：hipprof --pmc 默认集已覆盖 L2/TA/LDS/GRBM 主链路；sectors/store 效率类（NVIDIA sectors-per-request 对应物 = TA_BUFFER_COALESCABLE_WAVEFRONTS + TCC_SECTOR_HIT）当前**名字坐实、数值未测**，DTK 26.10+ 复测

## 按用途速查（DTK-26.04 目录内名字；★=运行时已验证有值）
| 用途 | 计数器 |
|---|---|
| 占用率/RA 瓶颈 | SPI_RA_VGPR_SIMD_FULL_PS / SPI_RA_SGPR_SIMD_FULL_PS / SPI_RA_LDS_CU_FULL_PS / SPI_RA_WAVE_SIMD_FULL_PS / SPI_RA_WVLIM_STALL_PS / SPI_RA_RES_STALL_PS |
| LDS | SQ_LDS_BANK_CONFLICT★ / SQ_LDS_ADDR_CONFLICT / SQ_LDS_UNALIGNED_STALL / SQ_INSTS_LDS★ / GDS_DS_BANK_CONFL |
| 合并/sector 效率 | TA_BUFFER_COALESCABLE_WAVEFRONTS / TA_BUFFER_READ_WAVEFRONTS / TCC_SECTOR_HIT / SQ_INSTS_VMEM_RD_REPLAY |
| L2 | TCC_HIT[32]★ / TCC_MISS[32]★ / TCC_EA_WRREQ_STALL[32]★ / TCP_TCP_TA_DATA_STALL_CYCLES[16]★ / TA_TA_BUSY[16]★ |
| Stall 分解 | SQ_WAIT_CNT_VM / SQ_WAIT_CNT_LGKM / SQ_WAIT_INST_VMEM / SQ_WAIT_INST_LDS / SQ_WAIT_IFETCH / SQ_WAIT_BARRIER |
| TLB | SQ_UTCL1_TRANSLATION_MISS / SQ_UTCL1_REQUEST |
| MMOP(MFMA) 按 dtype | SQ_INSTS_MMOP / SQ_INSTS_MMOP_F32 / F16 / BF16 / I8 / **FP8**（gfx938 FP8 验证可用） |
| VALU 按 op/dtype | SQ_INSTS_VALU_ADD/MUL/FMA/TRANS_F16/F32/F64 / INT32 / INT64 / CVT |
| 基础 | SQ_WAVES / SQ_CYCLES / SQ_BUSY_CYCLES / GRBM_CP_BUSY / SQ_INSTS_VMEM_RD★/WR★ / SQ_INSTS_BRANCH |
| 指令缓存 | SQC_ICACHE_HITS/MISSES / SQC_DCACHE_* |
