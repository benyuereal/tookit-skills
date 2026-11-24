# 工具层可用性（`scripts/tool_probe.py` 实测生成，勿手改）

> 每一条都是**真跑**的结果，不是文档抄来的。
> 判据里只能写 `available` 的工具；`degraded` 要标明弱在哪；
> `unverified` 不许当成可用。

| 工具 | 阶段 | 状态 | 实测细节 | 缺失时的降级路径 |
|---|---|---|---|---|
| `hipprof` | S3/S4 | **available** | 路径 /opt/dtk/bin/hipprof；--pmc=True；--hip-trace=True | 无替代：只能靠耗时消融（改一处看差多少） |
| `hipprof_pmc_csv` | S3/S4 | **available** | CSV 206 列，关键列齐全；实例列如 TA_TA_BUSY[0..] | 退化为「只有墙钟」，归因只能靠消融 |
| `hy_smi` | S3/S4/S6 | **available** | hy-smi 可读（13 行） | 无法确认 sclk → 结论不可复现（本仓栽过） |
| `dccobjdump` | S4 | **degraded** | dccobjdump 可用；--architecture 支持列表里**不**含 gfx938（已知只到 gfx928） | 用 hipcc --save-temps 拿中间产物 |
| `llvm_readelf` | S4 | **available** | AOMP-16.0-45 (http://github.com/ROCm-Developer-Tools/aomp): | 无替代：资源画像做不了 → S4 只能标 partial |
| `llvm_objdump` | S4 | **missing** | llvm-objdump 不存在（助记符级反汇编缺这一入口） | **放弃**：只用资源字段 + 微基准推断，不许声称改了一条指令 |
| `hipcc` | S4/S6 | **available** | 编译 gfx938 .so 成功（16120 B） | 只能用现成库，不能自研实现 |
| `inspect_so` | S4 | **available** | 能读 VGPR/spill/LDS 并算 CTA/CU | S4 只能手工做，无法自动化 |
| `rocgdb` | S6 | **degraded** | 只有通用 gdb；无 ROCm 扩展（GPU 侧无 coredump 支持） | 靠 kernel 内的 printf / 边界检查 |
| `bw_calibrate` | S1 | **available** | 四档齐全，sclk=1350 | 用 bw-ladder 的静态值，无法重标定 |
| `compute_peak` | S1 | **available** | bf16=279 TF，reproduce=Plugins/ops/skills/op-roofline/tools/calibrate_bf16_peak.py | 用 constants.yaml 的既定值，无法重标定 |
| `flock` | 全阶段 | **available** | flock 可用；锁文件 /tmp/opbench-gpu.lock（已存在） | 无保障：多个 bench 可能同时上卡，结论全部作废 |
| `kfd` | 全阶段 | **available** | /dev/kfd 存在，renderD* 8 个 | 无法跑任何实测 |

统计：available 10 / degraded 2 / missing 1

生成命令：`python3 -I scripts/tool_probe.py --write`
