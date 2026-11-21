# 跨架构微优化速查（NVIDIA Blackwell NVFP4 GEMM → DCU 可迁移手法）

> 来源：cudaforfun《在NVFP4上超越cuBLAS》（https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4）。
> 2026-10-16 自 `operators/dcu-kernel-dev/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 十六、跨架构微优化速查（来源：NVIDIA Blackwell NVFP4 GEMM，手法 DCU 可迁移）

> 来源：cudaforfun《在NVFP4上超越cuBLAS: 从零手写+Claude极限优化Blackwell GEMM》（微信转载，原始 https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4）。
> **标注**：这些是 NV 平台文章里的**手法本身**（与具体架构无关），已剥离 TMEM/tcgen05/TMA/NVFP4 等 NV 专属细节。所有**百分比数字都是 NVIDIA GB300 实测，非 DCU 基线**，DCU 收益以真机 `METRIC` 行为准。完整案例见 `dcu-op-optimization` §案例八。

### 16.1 手法 → DCU 对应物 速查表

| 手法（跨架构可迁移） | NV 平台写法/信号 | DCU 对应物 | 何时用 |
|---|---|---|---|
| **特殊寄存器做 warp-uniform 判定** | `mov.u32 %laneid` 代替 `threadIdx.x%32==0`；benchmark 单点 **+77.5%** | DCU 特殊寄存器 `s_laneid` / `__builtin_amdgcn_lane_id`。凡 SMEM/TMEM 描述符、barrier 地址、循环状态、lane0 选择等 warp-uniform 判定，直接读特殊寄存器，避免从 per-thread `threadIdx` 推 uniform 判定，让编译器省去等价 R2UR 指令 | 任何"warp 内选一个 lane 干活"的场景 |
| **宽向量化存储** | `st.global.v8.b32`（8×b32 一次写）benchmark **+7.5%** | `global_store_dwordx4`（或更宽），对齐 16B/128B | epilogue 写回、大块搬出 |
| **缓存策略提示（写一次的输出）** | `L1::no_allocate.L2::evict_first` 保护可复用输入的 L2 驻留，benchmark **+1.2%** | ROCm/DCU 有 GL2 策略（`buffer_store` 的 `glc`/`dlc` 位、L2 驱逐提示）；写一次不再读的输出用 no-allocate + evict-first | 输出只写一次、不再被 kernel 读 |
| **双缓冲重叠隐藏 drain 延迟** | 两个输出 tile 重叠 36 列，MMA 等 consumer 排空重叠边缘而非整块，benchmark **+2.2%** | 无 TMEM 等价物，但「输出 tile 双缓冲 + 重叠边界」是 `num_stages=2` 双缓冲的进阶思路（见 §15.2） | compute 等 epilogue drain 的 GEMM/conv |
| **精确线程数提示去收敛保护** | `__block_size__(224)` (reqntid) 去掉 BSSY/BSYNC，VGPR 62→52 | `__launch_bounds__`（见 §14.1）+ 给编译器精确线程数，避免收敛保护开销 | kernel 被插收敛保护导致 SASS 膨胀 |
| **尾部精确化，不在 padding 上跑 MMA** | K=8192 用 K=768 主循环 + 显式 tail，算 K=8256 而非 padding 出的 K=8448；K64 折叠尾部凑精确 8192 | 固定 K 主循环 + 显式 tail group，不跑被零填充的 MFMA；尾部混不同 K 的 MFMA 凑精确 | 非整除 K 形状，避免 tensor core 算 0 |
| **量化 stage 数对性能的影响** | SMEM 占 99.1%；减一个 A/B stage 损 12%、减一个 scale stage 损 1.2% | LDS 占用率同样需量化（见 §15.5 ISA 自检清单） | 调 double-buffer stage 数 |

### 16.2 跨 die L2 分区调度（advanced/experimental）

- **现象（NV GB300）**：双 die 各管约半 L2；物理地址由 hash 定 home side，SM 由所在 die 定 near side，二者可 mismatch，跨 die 链往返慢 ~2.3x（148ns vs 344ns）。分割比例不固定（76/76、72/80、74/78…），**每卡不同**。
- **手法**：把可复用维度（如 A 的 M 行）按 requester side 分组，使复用留在 near side，route table 每 tile 一个条目仅几 KB，每 SM 查询算哪个 tile（NV 实测 +~1%）。
- **DCU 适配**：AMD CDNA 多 die（gfx94x）有**完全相同的「每 die 半 GL2 + Infinity Fabric 跨 die 延迟」特征**（见 `dcu-topology`）；kernel 内按 die/SA 分组 tile 是可行方向，但 DCU 无现成地址 hash 公式，需自己 microbenchmark 定 home side。**DCU 上收益与实现待真机验证**。

### 16.3 基准测试纪律（强化 §十一 / `dcu-op-optimization` 第 5 步）

- 轮换输入：每次启动用不同 A/B 副本，kernel 看不到自己之前的 L2 状态（避免 self-warm 虚高）。
- 多轮交替 + 轮换先后（如与 rocBLAS 交替 5 轮、每轮连续 25 启动取均值）。
- 冷却：每定时 arm 至少 8s 冷却，轮询直到空闲时钟恢复初始；DCU 对应 `rocm-smi --setclock` / `hy-smi`。
- **关键警示**：同内核用不同测法可得出 1.039x~1.078x 不同 apparent 优势——**基准协议本身影响 apparent 加速比**，DCU 上必须同协议同条件对比。
- **同进程连续测多配置会 VMFault（2026-08-20 实测）**：同一 Python 进程内连续调用同一 vendor kernel 的不同配置（如 paged-attention 的 page=1→16→32→64…）会触发 `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`（状态泄漏，单个配置单独跑全部正常）。**每个配置必须独立进程测**（脚本收 argv 配置、外层 shell 循环起进程）。
- **standalone 复现必须复刻生产张量 layout**：复现 paged kernel 时 K/V cache 的 dim 顺序、block_table 语义（page 索引 vs token 级 cache location）必须与生产调用点逐字一致，否则"kernel 是坏的"这类结论不可信——判据：生产服务走同一路径工作正常，则错在复现而非 kernel。

