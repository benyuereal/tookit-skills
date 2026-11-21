---
name: dcu-skill-on-device-verification
description: "在真机/容器实跑DCU技能自带脚本，对照SKILL.md基线判断生效与准确。用户让扫描或去设备验证技能时加载。"
version: 1.0.0
author: <maintainer>
license: MIT
kind: tool
---

# DCU 技能库真机验证（dcu-skill-on-device-verification）

## 定位

用户会周期性要求核对 DCU 技能库「哪些内容能在设备上验证、技能是否生效、内容是否准确」。
本 skill 是**跨技能的验证方法论**：把技能按「需要什么资源才能验证」分层，挑便宜的优先实跑，
跑技能自身随附的脚本，并对照 SKILL.md 里写死的基线数字判断准确性。
它不是某一次验证任务的记录（那放 references/），而是可重复的验证流程。

## When to Use

- 用户问「dcu skills 里哪些内容能在设备上验证一下 / 看技能是否生效、内容是否准确」
- 需要系统性核对一批技能的准确性，而非单个技能的排障
- 评估技能库的「地基」是否可信（环境/编译层技能是地基）

Don't use for: 单个技能的具体操作（进对应技能）；给技能新增经验（那是各技能自己的事）。

## 可验证性分层（核心判断框架）

先按此把技能分类，决定验证顺序与资源投入：

| 层 | 特征 | 验证方式 | 资源成本 |
|---|---|---|---|
| **A 只读环境类** | 查询命令（hy-smi/lspci/numactl/ls DTK），只读无副作用 | 真机直接跑命令对照 | 极低，先做 |
| **B 编译/运行类** | HIP/Triton/CUDA→HIP 编译，需 DTK 工具链 | 容器里编译+跑 demo | 中，抽 2-3 个代表 |
| **C 重型推理/多卡类** | vLLM/SGLang 起服务、rccl 多卡、profiling | 需要模型+整机 | 高，验证阶段一般跳过 |
| **D 故障排查类** | 需特定故障场景才触发（VMFault/精度/分布式卡死） | 无法主动构造 | 不可主动验证 |
| **E 文档/方法论类** | router、源码索引、工程流程 | 看流程合理性，非真机 | 不依赖设备 |

经验法则：先做 Tier A（最便宜、最能证明技能生效准确），再挑 Tier B 有代表性的 1-2 个验证编译链路。
Tier C/D/E 不建议在「验技能」阶段碰。用 Todo 列表管理逐项验证进度。

## 验证循环（每技能）

1. **加载技能正文**（skill_view）→ 找出该技能声称的「可执行产物 + 基线数值」：
   - 随附脚本（scripts/）：如 dcu-env 的 check_iommu_acs.sh、dcu-topology 的 collect_topology.sh、
     dcu-cpu-affinity 的 gen_affinity.sh、dcu-kernel-dev 的 dcu_detect_arch.sh / dcu_build_kernel.sh
   - SKILL.md 里写死的实测基线：如「<device-A>=BW150/gfx936」「max_diff=0.000977」「hop 组内1/跨组2」
2. **在真机/容器实跑**，看输出是否与技能描述一致。
3. **对照基线数字**：实跑结果与 SKILL.md 记录精确吻合（数量级、型号、gfx、状态）才判「准确」。
   - 例：dcu-kernel-dev 记录 <device-A> 首批 GEMM status=PASS max_diff=0.000977，实跑正好 0.000977 → 强佐证。
4. **有出入 → 先定性再更新（用户硬要求，勿直接改）**：把每处差异归到三类之一，**定性后才回填**：
   - **通用差异**（上游/平台整体演进，跨版本都成立）：如 vLLM `--device` 参数在 0.15.1 和 0.18.1 都没有（platform plugin 机制替代）→ 技能该条目整体过时，标「已废弃」。
   - **版本特定差异**（海光 fork 跨版本演进，技能记录的是旧版、新版变了）：如 SGLang 0.5.12 `is_dcu()`/`dcu_mla` → 0.5.15 `is_hcu()`/`hcu_mla`。技能记录旧版**本身没错**，补一条「版本演进对照」即可，别删旧版内容。
   - **技能错**（技能写错了，任何版本都不成立）：直接修正。
   - **环境/镜像差异**（非技能问题，是这台机器/这个镜像的状态）：只记录，不改技能。
5. **定性手段——双版本对照（关键技巧）**：判断「通用 vs 版本特定」时，**同时起技能来源版本 + 一个更新版本的镜像**，对同一断言各跑一遍：
   - 两版本都无该特性 → 通用演进（如 `--device` 0.15.1/0.18.1 都无）。
   - 旧版有、新版变 → 版本特定（如 `is_dcu` 0.5.12 有 / 0.5.15 变 `is_hcu`）。
   - 技能来源版本镜像通常还在 `docker images` 里（如 `dsv4-flash-k100ai-sglang0.5.12-20260728`），直接起它做旧版基线，别凭记忆。
6. 回填：通用差异→改技能主体；版本特定→补对照表（保留旧版）；技能错→修正；环境差异→只记录。改完 commit+push（只 add 自己改的文件）。

## 真机/容器执行环境（照抄可用）

SSH 辅助：凭证从 `~/.config/dcu-toolkit/credentials.json` 读入 `SSHPASS`（sshpass -e），
写个临时 python 封装（见 references/）复用连接，**不落盘密码、不回显**。设备 IP 从 `~/.ssh/known_hosts` 全文找
（按编号定位，如 <device-A> → `<internal_device_ip>`），不要搜短串。

起验证容器选**轻量标准镜像**（如 pytorch:<ver>-dtk26.04），挂 `--device=/dev/kfd --device=/dev/mkfd --device=/dev/dri -v /opt/hyhal:/opt/hyhal:ro --shm-size`。

## Pitfalls（本次实测确认）

1. **`docker run --rm` + 内联 `bash -c` 会吞掉输出**：一次性容器把 banner（"8 DCU were detected"）打出后，
   后续命令 stdout 被截断、看不到编译/运行结果。**改用常驻容器**（`docker run -itd --name <c>`）+
   `docker exec <c> bash -ilc '...'` 分离启动与执行，才能拿到完整输出。
2. **`docker exec` 必须 `-ilc`**：非交互非登录 shell 不加载 `.bashrc`/DTK 环境，hipcc/rocblas 找不到或性能劣化。
3. **ssh 写文件进「容器挂载目录」要写宿主机路径**：容器内 `/work` 若来自 `-v /tmp/v:/work`，
   ssh 要写 **宿主机的 `/tmp/v/...`**，不是容器的 `/work`（ssh 到宿主机,`/work` 可能不存在 → rc=1 静默失败）。
   本次就因写成 `/work/vecadd.cu` 导致文件没进去，hipify 报 could not open。
4. **别信凭证里的机型标签**：<device-A> 凭证记 `BW100X8`，但 `lspci -nn` 显示 1d94:6320（BW150/BW1000 系）。
   型号/gfx 以 `lspci -nn`/`hy-smi`/`rocminfo` 实测为准，凭证标签只作非敏感索引。
5. **hy-smi 在 `/opt/hyhal/bin/hy-smi`，不需 source DTK**（驱动自带）；DTK 工具链才需 `source /opt/dtk/env.sh`。
6. **`--showtoponuma` 是唯一可信卡↔NUMA 源**（lspci 在海光全报 node0）；仅该选项支持 `--json`。
7. **验证完清理**：`docker rm -f` 自己起的验证容器 + `rm -rf` 宿主机临时文件（只删自己的，不动他人容器）。

## 验证基线汇总（参考）

已确认准确的骨架技能与对照点（详见 references/bw150-gfx936-verification-2026-08.md）。
**<device-A> 设备累计验证 17 个技能全部生效**，覆盖环境/编译/通信/部署/性能/包源/引擎全链路：

- **环境/拓扑（Tier A）**：dcu-env（check_iommu_acs.sh 判 all_disabled=true exit=0）、dcu-hy-smi（showtoponuma/json、topotype 表格、hop 组内1跨组2、showpids）、dcu-dtk-layout（DTK-26.04-rc4、AMDGPU_TARGETS、软链 llvm→dcc/rocm-smi→hy-smi-wrapper）、dcu-topology（collect_topology.sh 识别单NUMA）、dcu-cpu-affinity（gen_affinity.sh dry-run 安全）
- **编译/算子（Tier B）**：dcu-kernel-dev（detect_arch→gfx936，build GEMM→max_diff=0.000977 status=PASS 与记录吻合）、dcu-cuda-hip-port（hipify-perl 路径对，cudaMalloc→hipMalloc 残留0，hipcc --offload-arch=gfx936 编译运行 PASS）
- **通信**：dcu-rccl-test（物理机 rocminfo 确认 gfx936，commit 5e838ad + make 编译 rccl-tests，8卡 all_reduce/alltoall 跑通 #wrong=0，设备识别 BW150）
- **部署**：dcu-vllm-deploy（TP=8 起 Qwen3.6-27B 推理"1+1=2"正确）、dcu-sglang-source（0.5.12 镜像 is_dcu/dcu_mla_backend 存在，与记录吻合）
- **性能**：dcu-perf-profiling（hipprof --pmc 采集 GEMM，计数器 SQ_INSTS_MMOP/SQ_LDS_BANK_CONFLICT/wave_size=64 与映射表吻合）、dcu-hipprof（--hip-trace --stats 出 hipkernel.csv/hiptrace.csv/.db，--leak-check 正常）
- **包源**：dcu-das-hub-api（X-Das-Source 头必需、列版本 API、内网→公网 URL 改写下载 sha256 全匹配）
- **benchmark**：dcu-llm-benchmark（vllm bench serve --help=all 选项与 §2.2 全吻合；sglang bench_serving CLI 与 §3.2 全吻合）
- **引擎**：dcu-migraphx-perf（migraphx-driver 5.2.0 子命令与 §10 速查表一致）、dcu-ort-perf（ORT 1.19.2 + MIGraphXExecutionProvider 存在）

### BW1100 验证成果（2026-09-15，<device-alias>/<<internal-ip>>）

在 BW1100（OAM+HSW 形态，gfx938，新代际 DCU-3G）上验证了一批**代际敏感**的 skill（详见 references/bw1100-gfx938-verification-2026-09.md）：
- **dcu-kernel-dev**：`dcu_detect_arch.sh` → **gfx938**，`dcu_build_kernel.sh` 用 `--offload-arch=gfx938` 编译 `hip_gemm_tiled.cpp` 成功，运行 **status=PASS（max_diff=0.000977，与 <device-A> gfx936 完全一致）** → 已回填 §13.5 补充 BW1100=gfx938 实测。
- **dcu-topology**：`collect_topology.sh -j` 正确识别 8 卡横跨 6 个 NUMA 节点（0,1,3,4,5,7，cross_numa=true，dcu_source=hy-smi）→ 已补充「OAM+HSW 全互联」拓扑形态（全 1-hop、全 HSW 链路）。
- **dcu-hy-smi**：`--showtoponuma` 显示 8 卡散落 6 节点；`--showtopotype` 全 HSW、`--showtopohops` 全 1（OAM+HSW 全互联）→ 已补充 OAM+HSW 形态说明。
- **dcu-env**：`check_iommu_acs.sh` **all_disabled=true**（OAM 不受 IOMMU/ACS 影响）→ 已补充 BW1100/OAM 实机确认（1d94:6430、gfx938、8HCU+7HSW）。
- **dcu-dtk-layout**：DTK-25.04.4 结构**与 26.04 基线显著不同**（缺 aillvm/dushmem 目录、AMDGPU_TARGETS 空、rocm_agent_enumerator 未含 gfx938、hipprof 在 /opt/dtk/bin 直放）→ 已补充第三基线（**宿主** DTK-25.04 结构，**注意：宿主 DTK-25.04 不满足 BW 系列要求，BW 系列须用容器 DTK-26.04**）。

### 验证中发现的 3 处差异（已定性 + 回填 + commit e26a0fd）

| 技能 | 差异 | 定性 | 处理 |
|---|---|---|---|
| dcu-vllm-deploy | `--device rocm` 参数在 0.15.1 和 0.18.1 都没有 | **通用差异**（vLLM platform plugin 机制替代，跨版本都成立） | 标「已废弃」 |
| dcu-sglang-source | 0.5.12 `is_dcu()`/`dcu_mla` → 0.5.15 `is_hcu()`/`hcu_mla` | **版本特定**（海光 fork 术语演进，技能记录 0.5.12 本身没错） | 补版本演进对照表，保留旧版 |
| dcu-perf-profiling | DTK-26.04 容器只有 hipprof，rocprof 在宿主 25.04 | **版本/镜像特定** | 工具栈表补版本说明 |

> 定性方法见「验证循环」第 4-5 步：双版本对照（起技能来源版本 + 更新版本镜像各跑一遍）区分通用 vs 版本特定。

## 与 dcu-skill-router 的关系

本 skill 是「验证技能库质量」的横切流程；dcu-skill-router 是「用技能做事」的入口。两者互补。

**发现差异后的回填纪律**（本 skill 库是 curator 管理的 dcu 仓库，用户已授权直接改）：
- 按「验证循环」第 4-6 步定性后，**直接 patch 对应技能**（通用差异改主体 / 版本特定补对照表保留旧版 / 技能错修正 / 环境差异只记录）。
- 改完 `git add` **只 add 自己改的文件**（别碰他人未跟踪/已改文件），commit + push origin master。
- 用户若要求「先定性再更新」，严格先定性、汇报定性结论、再回填；不要跳过定性直接改。
- 若某技能是**用户自有**（非本 dcu 仓库、curator 未 adopt），则不直接改，报告并建议 `agent curator adopt <name>`。

## References

- `references/bw150-gfx936-verification-2026-08.md` — <device-A> 设备(BW150/gfx936)验证完整命令脚手架 + 设备快照 + 各技能对照结果
- `references/bw1100-gfx938-verification-2026-09.md` — BW1100 设备(<device-alias>/gfx938/OAM+HSW)验证完整命令脚手架 + 设备快照 + 各技能对照结果
