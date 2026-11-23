# ops 工具总览与选型

先读这张表再动手：为同一件事写第二套工具，会把老坑（首次 launch 虚高、噪声 kernel、CU 归一化）重踩一遍——本域真发生过。

| 我要干什么 | 用哪个 |
|---|---|
| 这个改动快了多少 | `scripts/opbench.py`（HIP event，中位数 + MAD，`\|Δ\| > 3×MAD`） |
| 我们和 flash_mla 谁快、交叉点在哪 | `op-sparse-mla/bench/bench_long.py`（`pool=full`，比值走 `significant()`） |
| 这个档位的索引池够不够大 | `op-sparse-mla/_common.build_inputs`（看 `pool_extent`/`pool_mb`，池 <64MB 告警） |
| 卡在算力、内存还是指令发射 | 跑 `scripts/pmc.py` 解析 PMC CSV（只出机械原值；三级证据按 `op-profiling/SKILL.md §四` 由人判） |
| 多个变体并排比 | 用 `scripts/pmc.py` 同一口径各算一遍再并排（**不要为它再维护一个解析器**） |
| 这个算子该往哪优化 | `op-roofline/derive.py`（shape + 设计 → 预测耗时 + 绑定约束；唯一天花板权威） |
| 编出来的 kernel 实际用了多少 VGPR / SGPR / LDS | `op-hip-compile/inspect_so.py <.so>`（走 `dccobjdump --extract-elf` + `llvm-readelf --notes`）。**不要猜 occupancy，读二进制** |
| 想反汇编看厂商实际用什么指令 | **本机做不到**：无 `llvm-objdump`；`dccobjdump --architecture` 只支持 gfx906/926/928，不含 gfx938。能拿到原始指令字节（`llvm-readelf -x .text`），但没有字节→助记符这一步 |
| 这个设计会不会寄存器溢出 | `op-roofline/tools/hipbudget.py`（活集合 vs 每线程 256）。**只作相对比较**：绝对 VGPR 判定实测不准（算 283 / 实际 256），不要拿它一票否决设计 |
| 只想隔离一个变量（occupancy / launch 配置 / 分配） | 改一个**不改变语义**的量做探针：动态 LDS 申请量（kernel 仍只读前 N 字节，输出必须 bit-identical）、launch 配置、分配策略；先证 bit-identical 再比耗时 |
| 是计算密集还是带宽密集 | `op-sparse-mla/bench/bench.py --mode workset`（行为级，不替代 PMC） |
| 这个改动精度够不够 | `scripts/opcheck.py correctness`（对 fp32 稠密参考，容差读 `contract.yaml`） |
| 改了这条指令能省多少 | `op-sparse-mla/bench/bench.py --mode workset\|config`（实测消融） |
| 下一步做什么 | `op-optimize/loop.py`（`status`/`queue`；`refuted` 强制拒跑）。**本卡 `run` 不可用**（变体适配器已删），验证走 `bench/bench.py` + `scripts/opcheck.py` |
| 这个候选到底做完没有 | `op-optimize/loop.py verify --tag <id>`：跑候选的 `done_when`，**rc=0 才写 adopted**；`adopt` 是它的旧名（等价，不跑命令就采纳的路已关掉） |
| 这个数到底合不合理 | `op-gate`（四道门 + 四档决策） |
| 这些门什么时候自己跑 | `Plugins/ops/hooks/ops_gate.py`（PreToolUse，只挂写源码/跑 bench/`git commit`） |
| 卡在哪一类瓶颈、该试哪类手法 | `op-roofline/derive.py` 出 `bound` → 查 `bound-to-class.yaml` 得允许的 class → `surfaces.yaml` 挑该 class 的手法。**`loop.py run` 没有 bottleneck 记录会拒跑** |
| 我们这个仓库优化得有多深 | `scripts/depth.py`（五维 DS 分：基线强度/达成率/跨形状/轮次/归因）。**只读，不阻断** |
| 上次那条结论/那个坑记在哪 | `dcu-orchestrator/scripts/kbq.py`（`index`/`search`，精确子串，不做语义打分） |

## 完成判定（`done_when`）的执行口径

- 命令在**仓库根**执行（`scripts/opcheck.py` 这类共享脚本按仓库根相对写）；卡目录用 `$OPS_CARD`、
  仓库根用 `$OPS_REPO_ROOT`。
- 超时上限 `VERIFY_TIMEOUT_S = 3600` 秒，是常量不是开关——「跑不完」按失败算（记 rc=124），
  不能读成通过。
- 失败**不改** YAML：保持原状态不是「降级」许可，而是「它本来就没完成」。
- 退出码是唯一判据：组合命令用 `&&` 串，任何一段失败都必须让整体非 0；日志里出现 "pass" 不算数。

## 已弃用（不要再用 / 不要重开）

- 为同一件事新建第二套工具（与现有工具重合就改那个）——**同一个数两处算必然漂**：今天的 87%/55%（标称 vs 实测峰值分母）与 38.9/77.9µs（pmc vs derive 两个天花板）都是这么来的。
- PMC 的五个坑（PMC 墙钟当 bench、取首行当稳态、`max(TA_*)/GRBM` 算占用率、`SQ_INSTS_VALU` 当纯向量、报 L1 命中率）——权威口径见 `op-profiling/SKILL.md §五`，本表不复述。
- 用型号标称 GB/s 当 DRAM 带宽（会算出物理不可能值）。
- 按「某段占比 7.5%」推算「删掉省 7.5%」（本项目实测反了 1.1%）。
- 缺输入时静默用默认值凑（必须 `null` + 缺口原因）。
- 口径只写在文档里、不写进代码（断言/默认值/输出字段名才算数）。
- `OPS_GATE=0` 之类环境变量后门（要停就改 `.claude/settings.json`）。
- 门禁内部异常 fail-open（必须 `exit 2`）。
- 靠 run 目录名认领 manifest / result.json（认文件里的 `op` 字段）。
