---
name: dcu-hicache
description: "SGLang HiCache 三层缓存（L1 GPU/L2 host/L3 磁盘）在 DCU 上的部署、补丁与验证；含 DeepSeek-V4(DSV4)/DSPARK 多轮变长场景"
version: 1.1.0
author: DCU Engineer
license: MIT
kind: tool
---

# SGLang HiCache 三层缓存 DCU 部署（dcu-hicache）

适用场景：在 DCU 上用 SGLang **内置** HiCache（hierarchical cache）做三层 KV/Mamba 缓存卸载（L1 GPU 显存 / L2 host 内存 / L3 磁盘 file 后端），缓解 HBM 显存墙、降低长前缀 TTFT、提升并发吞吐。

> **与 `dcu-lmcache` 的区别**：LMCache 是独立 pip 库（vLLM/SGLang 都接，需编译安装）；HiCache 是 SGLang 内置（`--enable-hierarchical-cache` 开关，无需额外装库）。本 skill 只讲 SGLang 原生 HiCache 路线，两者不要混。

## When to Use

- 在 DCU 上用 SGLang 部署 HiCache 三层缓存（`--enable-hierarchical-cache`）
- SGLang HiCache 启动崩（seccomp / NUMA OOM / 补丁 anchor 错）
- HiCache 命中明细"错位"、L3 磁盘不写入、fact_ok 全 False
- 需要验证 L1/L2/L3 三层缓存是否真实工作
- 选 `--hicache-ratio` / `--numa-node` 配置
- **DeepSeek-V4（DSV4）/ DSPARK 投机解码**的多轮对话、长前缀、多并发场景的 HiCache 部署与压测（`--hicache-size` 不支持 DSV4、L1/L2/L3 容量公式、主动压池制造驱逐）——详见 `references/hicache-dsv4-deepseek-multiturn.md`
- **HiCache 宿主池初始化在 DCU 上 HSA stream 卡死**（`alloc_mmap`+register 坏 DMA/HSA 状态）——见 §6 坑 5（与 §3 坑 1 seccomp 是不同 bug）

Don't use for：
- vLLM 的 KV 卸载（用 `dcu-lmcache`）
- SGLang 源码结构（用 `dcu-sglang-source`）
- 容器启动 / seccomp 通用问题（用 `dcu-docker`）

> **一键适配脚本**：`templates/hicache_adapt_dcu.sh`（5 步幂等：预检→建容器→打补丁+ast 验证→起服务→验服务，`--rebuild` 强制重建容器防环境污染 / `--step N` 单步 / `--dry-run` 干跑，<device-A> 4 卡 + 2 卡 HCU 实测 health=200 跑通，含 2026-09-03 干净环境 rebuild 抓到的 4 个 bug 修复）。改配置区变量后 `bash hicache_adapt_dcu.sh` 即可。

## 0. 三层缓存架构

```
L1 GPU 显存 (KV + Mamba 状态)  →  L2 host 内存 (hicache-ratio × GPU 池)  →  L3 磁盘 file 后端
   热数据                          次热数据                                  冷数据（持久化，跨重启召回）
```

- Qwen3.8-27B 是 **Mamba 混合模型**（日志 `hybrid Mamba pool stack: pools=KV + MAMBA`），驱逐由 Mamba 状态主导（host 池大），不是纯 KV 模型。
- L3 磁盘缓存**跨重启持久化**：`/public/hicache_l3` 保留 `.bin` 文件，重启后 fresh 请求可直接命中 storage 层（实测有效）。

## 1. 环境（实测镜像）

- 镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620`
- SGLang 0.5.12（DCU dev 编译，`sglang.__version__` 显示 0.0.0.dev0）/ PyTorch 2.9.0 / DTK 26.04 / Ubuntu 22.04 / Python 3.10
- 容器必带 `--security-opt seccomp=unconfined`（见 §3 坑 1）+ 必挂 `/opt/hyhal:ro` + DCU 设备透传（见 `dcu-docker`）

## 2. 代码补丁（DCU 版 SGLang 0.5.12 必打）

DCU 版 SGLang 0.5.12 的 HiCache 有两个 bug，不打补丁启动即崩。补丁打在容器内 `dist-packages`，**镜像重建后需重打**。

### Bug B — memory_pool_host.py 分页布局 D2H/H2D 回退
- 文件：`/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/memory_pool_host.py`
- 需在 `MHATokenToKVPoolHost` 类内补 `_dcu_paged_device` 等方法（分页布局的 D2H/H2D 回退路径）。
- ⚠ **anchor 坑（最严重，会静默打错）**：若用 `def load_to_device_per_layer(` 做 anchor，该方法名在文件里出现 **9 次**，第一次在**基类 HostKVCache**（line 222，line 221 是 `@abc.abstractmethod`）。`s.replace(anchor, methods+anchor, 1)` 会把方法插到基类装饰器和 def 之间 → 装饰器"抢"走 `_dcu_paged_device` 变成抽象方法 → 启动崩 `TypeError: Can't instantiate abstract class MHATokenToKVPoolHost with abstract method _dcu_paged_device`。
- **正确做法**：用 MHA 类内**唯一**的 anchor（`def v_buffer` property，line 394，全文件唯一）插入，避开基类 @abstractmethod。
- 验证：`ABSTRACT_METHODS: []`（无残留抽象方法）+ 方法落在 MHA 类内（line 406/412/426）。
- **崩溃 op 的真实归属（判断升级路径时别查错包）**：Bug B 崩溃的 `transfer_kv_direct` 是从 **`sgl_kernel.kvcacheio`**（sgl-kernel 包，0.3.21）import 的，**不在 flash-attn 里**——flash-attn 只含 `libflash_attention.so`、无 kvcacheio，其依赖仅 `torch==2.9.0`+`einops`，不依赖 sgl-kernel。所以"升级 flash-attn 修 Bug B"方向不对：若某版本修了该 op，修复来自 **sgl-kernel** 而非 flash-attn。判断升级路径查 `sgl_kernel.kvcacheio` 的版本/符号，别查 flash-attn。
- **升级 flash-attn 修不了 Bug B（2026-09-02 <device-A> 实测定论）**：flash-attn 升到 `2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be`（sgl-kernel 保持 0.3.21 不变，`--no-deps` 不连带升级）后，去掉 Bug B 补丁重启 → 驱逐写回**照样崩**（2 vs 64）；重新打补丁 → 正常。**结论：Bug B 补丁不可省略，与 flash-attn 版本无关。**
- **Bug B 是确定性 kernel bug，与硬件/设备无关（2026-09-03 跨设备实测）**：在另一台 2 卡 HCU 设备（sgl-kernel 0.3.21 / flash-attn 2607280958 / sglang 0.5.12 / KV_LAYOUT 默认 true / page-size 64，与 <device-A> 完全一致）上对照实验：不打补丁 + 触发驱逐 → 崩（崩溃路径与 <device-A> 一字不差）；打补丁 → 13 请求全成功。**"某设备没有 Bug B"的真相是它没真触发过驱逐**（KV 池没灌满，"30K 能跑"陷阱）。判断某环境是否有 Bug B：不看"跑没崩"，看"是否真触发过驱逐 + sgl-kernel 版本"。触发 4 条件缺一不可：① sgl-kernel 0.3.21 ② KV 分页布局（SGLANG_KV_LAYOUT_DCU_FA 默认 true）③ direct 后端 ④ 真触发驱逐（GPU KV 池灌满 → 写回 host）。

### Bug C — hi_mamba_radix_cache.py load_back protect/release
- 文件：`/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/hi_mamba_radix_cache.py`
- load_back 路径补 protect（1 处）+ release（3 处）。
- Bug C 的 anchor 有 assert（插错会报错，不会静默打错），比 Bug B 安全。

> 补丁 anchor 行号、错误签名与验证命令见 `references/hicache_deploy.md`。

## 3. 启动配置与坑

### 标准启动参数（4 卡 TP=4 实测稳定）
```
--numa-node 0 1 2 3          # rank 分散到 4 个 node（ratio 1 下任何 node 都够，见坑 2）
--enable-hierarchical-cache --hicache-ratio 1   # 稳定值（见坑 2）
--hicache-io-backend direct
--hicache-storage-backend file --hicache-storage-prefetch-policy timeout
--mem-fraction-static 0.83
--enable-cache-report        # 必须，否则 cached_tokens_details 为空
--attention-backend fa3 --page-size 64
--kv-cache-dtype fp8_e5m2
# 不要加 --max-total-tokens（会限死 KV 池，显存只 50%，非正常业务场景）
```
**两个必配环境变量（2026-09-03 <device-A> 真机实测抓到，缺了启动即失败）**：
- **L3 磁盘目录用环境变量，不是命令行参数**：`--hicache-storage-path` 在 sglang 0.5.12 **不存在**，写了直接 `unrecognized arguments: --hicache-storage-path` 启动失败。正确：`export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/public/hicache_l3` + 启动前 `mkdir -p /public/hicache_l3`，命令行只保留 `--hicache-storage-backend file --hicache-storage-prefetch-policy timeout`。
- **`--context-length` 超模型派生值需放行**：`--context-length 1000000` > 模型派生 262144 时启动报 `User-specified context_length (1000000) is greater than the derived context_length (262144)`。需 `export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`（配合 rope override 的 `original_max_position_embeddings` 才能真扩上下文）。

### 坑 1 — seccomp 拦截 mbind（TP≥2 必崩）
- **现象**：scheduler 初始化崩溃（exit 1），traceback 被吞；单卡小内存能跑，TP≥2 大内存分配就死。
- **根因**：DCU 运行时 ROCT Thunk 做 NUMA 内存绑定需要 `mbind` 系统调用，默认 seccomp 策略拦截。hylog 日志（`/var/log/hylog/`）明确写 `mbind is blocked by seccomp`。
- **修复**：容器 `--security-opt seccomp=unconfined`（必需，不是推荐）。
- **诊断线索**：单卡 matmul 能跑但 TP≥2 死 → 大概率 seccomp（单卡小内存不走 mbind 路径）。

### 坑 2 — NUMA 策略 OOM（CONSTRAINT_MEMORY_POLICY）
- **现象**：`RuntimeError: Rank N scheduler died (exit code: -9)`（SIGKILL），dmesg 报 `oom-kill:constraint=CONSTRAINT_MEMORY_POLICY, nodemask=X, task=sglang::schedul`。
- **根因**：`--numa-node 0 1 2 3` 把每个 rank 的 host 池**钉死在单个 NUMA node** 上。单 rank host 池 = KV + Mamba（ratio 2 时 ≈81.64GB）。若该 node 空闲内存 < 单 rank host 池（被 hymgr/apport/coredns 等占用），触发 NUMA 策略 OOM（**不是普通内存不足**，总内存可能还很空）。
- **为什么之前能跑、重启就崩**：配置没变，是节点内存格局变了（其他进程占了 node 内存）。
- **修复**（二选一）：
  1. 降 `--hicache-ratio`（2→1，host 池减半到 ~40.84GB/rank，任何 node 都放得下）——**稳定优先**
  2. 把 rank 分散到空闲内存多的 node（`--numa-node 0 1 2 4` 避开被占满的 node 3）
- **node 选择取决于单 rank host 池大小（= ratio 决定）**：ratio 2 时单 rank 池 ~81.64GB，必须挑空闲 >85GB 的 node（如 node 4）；ratio 1 时单 rank 池 ~40.84GB，**任何 node（含 node 3 的 75.9GB）都够放，应改回 `--numa-node 0 1 2 3`**——别把 ratio 2 时代的"避开 node 3"遗留到 ratio 1（用户纠正过：ratio 1 下 0 1 2 4 是多余的）。
- **诊断**：`dmesg -T | grep -iE "oom|killed process"` 看 `constraint=CONSTRAINT_MEMORY_POLICY` + `nodemask=X`；`grep MemFree /sys/devices/system/node/nodeX/meminfo` 看各 node 空闲。

### 坑 3 — Mamba 混合模型三层命中"错位"
- **现象**：L1/L2/L3 命中明细"错位"（如 L2 期望 host 实际 storage，L3 期望 storage 实际 host）。
- **根因**：Qwen3.8-27B 是 Mamba 混合模型，驱逐由 Mamba 状态（host 15.24GB/rank，大）主导而非 KV（host 3.28GB/rank，小）。Phase 2 Mamba 状态写 L3 → storage 命中；Phase 3 从 L3 回载到 host → host 命中。是回载机制正常行为，**不是 bug**。
- **核心信号**：事实召回（fact_ok）三层全 True 才是真信号，不要纠结命中层"错位"。

### 坑 4 — 通过 SSH 起服务：`docker exec -d nohup` 会占住 SSH 会话
- **现象**：在宿主机上通过 `ssh` 跑"起服务"脚本（脚本里 `docker exec -d ... nohup bash serve.sh > log 2>&1`），SSH 会话不返回、最终被 SIGTERM（exit 143），且留下"旧服务已 pkill、新服务没起来"的半状态。
- **根因**：`docker exec -d` 启动的进程通过 docker 管道占住 SSH 会话的 stdout，sshpass 不返回 → SSH 被杀。脚本设计是"在 DCU 宿主机本地跑"（无 SSH 中断问题），通过 SSH 跑就踩到。
- **修复**：起服务命令加 `< /dev/null` 断开管道：`docker exec -d <容器> bash -c 'cd /root && nohup bash serve.sh > /root/serve.log 2>&1 < /dev/null'`。或把"起服务"和"等 health"拆成两个独立 SSH 调用（起服务立即返回，health 单独轮询）。
- **诊断**：serve.log 没被 truncate（新 nohup 会 truncate 它）→ 说明启动命令没执行到；`pgrep -f sglang.launch_server` 空 + health=000 → 服务没起来。

## 4. 验证方法

1. **基础**：health=200 + 推理正常 + `cached_tokens_details` 非空（需 `--enable-cache-report`）。
2. **L1/L2/L3 命中明细**：发同前缀请求，看 `cached_tokens_device/host/storage` 哪层 >0。
3. **事实召回**：长前缀里埋事实，驱逐后重问，fact_ok=True 证明缓存真实工作（max_tokens 给够，推理模型 thinking 会吃掉小 max_tokens 的输出空间）。
4. **L3 磁盘**：`ls /public/hicache_l3 | wc -l` + `du -sh`，确认 `.bin` 文件真实写入；重启后 fresh 请求命中 storage 证明跨重启持久化。
5. **并发压测**：多 worker 各持独立长前缀（工作集 > GPU KV 池），三阶段（预热/稳态/压力），看缓存加速比 + 成功率。
6. **稳定性**：serve.log 无 OOM/traceback/crash；hy-smi 显存占用正常。

> 验证脚本结构与命令见 `references/hicache_deploy.md`。

## 5. 与兄弟 skill 关系

- `dcu-lmcache`：LMCache 路线（独立 pip 库，vLLM/SGLang 都接）；本 skill 是 SGLang 原生 HiCache 路线。
- `dcu-sglang-source`：SGLang 源码结构（HiCache 相关源码落点）。
- `dcu-docker`：容器启动（seccomp=unconfined 必需，见本 skill 坑 1）。
- `dcu-cpu-affinity`：NUMA 绑核（本 skill 坑 2 的 numa-node 交互）。
- `dcu-vllm-deploy`：vLLM 部署（对比参考）。

## 6. DeepSeek-V4（DSV4）+ DSPARK 多轮变长场景（与 §0~5 的 Qwen3.8 场景不混用）

> 本场景：DeepSeek-V4-Flash-0731（**纯 KV/MLA 模型**，非 Mamba 混合）在 8× DCU BW1100、SGLang **0.5.15（安装版，dev fork 无 DSPARK）**、DP=8、**DSPARK 投机解码** 下的 HiCache 三层部署与多轮 Agent 长前缀压测。完整启动脚本、容量公式、性能对比、patch 见 `references/hicache-dsv4-deepseek-multiturn.md`。

### 坑 5 — HiCache 宿主池初始化把 DCU HSA stream 卡死（DSV4 场景最严重 bug，与坑 1 seccomp 不同）
- **现象**：`--enable-hierarchical-cache` 启动时，worker 在 `memory_pool_host.py` 的指针张量构造处无限期卡死。native 栈 `hipMemcpyWithStream → hsa_signal_wait_scacquire(∞)`，GPU 0% 利用率，8 rank 全冻结；两次启动均复现。
- **根因**：宿主池分配器 `HostTensorAllocator` 用 `alloc_mmap`（`mmap(MAP_SHARED|MAP_POPULATE)`）分配几百 GB host 内存再 `cudaHostRegister`。该路径在 DCU 上**损坏 GPU DMA/HSA 状态**。隔离验证：`torch.empty`+`cudaHostRegister` 单进程 405GB 正常；`alloc_mmap`+register 单进程 500s 无进展。
- **修复**：`HostTensorAllocator.allocate` 改为 `torch.empty(dims, dtype, device="cpu")`（patch `host_allocator_torch_empty_dcu_fix.patch`）。修复后启动 ~5 分钟正常。
- **诊断线索**：启动只在 HiCache 宿主池初始化处卡死、GPU 0%、native 栈 `hsa_signal_wait_scacquire` → 大概率此 bug。

### 坑 6 — 驱逐 double-free 崩溃（`_evict_backuped` 对 None 节点 free）
- **现象**：高并发（users=5/10）下 `_evict_backuped` 对已驱逐节点 `node.value is None` 再 `free(None)` → `AttributeError`。
- **修复**：`_evict_backuped` 开头加 `if node.value is None: 刷新叶子簿记, return 0`，幂等（patch `hiradix_evict_backuped_None_value_fix.patch`）。

### 坑 7 — `--hicache-size` 不支持 DSV4
- **现象**：启动报 `DeepSeek V4 HiCache currently does not support --hicache-size`。
- **处理**：DSV4 只能用 `--hicache-ratio` 表达 L2 host 池容量（`--hicache-size <GB>` 不行）。

### DSV4 关键配置要点
- **DCU 必用 `--hicache-io-backend direct`**（kernel 后端 JIT kernel 在 DCU 写野指针）；`--hicache-mem-layout layer_first`（扁平 host）。
- **L1/L2/L3 容量公式**（DSV4，`KV_BYTES_PER_TOKEN ≈ 10.7 KB/token`，由 `mem_fraction × 可用显存 / 默认池tokens` 推出）：
  - `L1_tokens = min(mem_fraction × 可用显存 / KV_BYTES, max_total_tokens)`
  - `L2_tokens = L1_tokens × hicache_ratio`；经验 `ratio ≈ 工作集W/L1 − 1`，有 L3 兜底时 ratio 可小（0.3~1）
  - `L3_GB ≈ 总工作集KV − L1_GB − L2_GB`
- **压池制造驱逐是让 HiCache "有意义"的关键前置**：默认 6.4M/卡（realistic 负载仅填 1~2%、不驱逐 → HiCache 无收益）。用 `--max-total-tokens 500000` 压到 500K/卡，users=64（每卡 8 会话 × 67K ≈ 536K > 500K）才触发驱逐。这是多轮变长场景验证 HiCache 的标准手法。
- **DSPARK 下 `--mem-fraction-static` 有 0.406 硬下限**（低于启动报错），压池只能靠 `--max-total-tokens`，不动 mem-fraction。
- **L3+minL2 主动预取是命中率高的关键**：`--hicache-storage-prefetch-policy timeout` 提前把磁盘前缀塞进 L1，请求到达即命中。L2（无磁盘）溢出即丢 → 命中率上不去；L3（有磁盘）溢出保留 → 命中率高。

### 多轮变长场景结论（本实验，users=64 满载驱逐）
- **L3+minL2 全面优于 baseline/L2**：users=64 总吞吐 +14.4%（6,421→7,347）、p50 ↓21%（174→137.5s）、pending 峰值 ↓39%（581K→355K）、命中率 ~6%→~40%。
- **users=16（中等压力）是 L3 最大收益点**：总吞吐 +57%（9,780→15,393）、p50 ↓56%。
- 命中率定义：`prefill_cache / (prefill_cache + prefill_compute)`（总缓存命中率，含各层）。
- **成功率 ~90% 的失败全是 context 超限 400**：`--context-length 81920` 是保守设定，模型实际支持 1M（`max_position_embeddings=1048576`，YaRN×16）。把 `--context-length` 提到 131072 即达 ~100%，且不影响池大小（池由 `--max-total-tokens` 单独控制）。

## 使用须知

- DCU 版 SGLang 0.5.12 必打 Bug B/C 补丁（镜像重建后重打）。
- 容器必带 `--security-opt seccomp=unconfined`（TP≥2 必需）。
- `--hicache-ratio` 按节点内存格局选：node 空闲多可 ratio 2（L2 容量大），node 被占则 ratio 1（稳定）。
- Mamba 混合模型看 fact_ok 不看命中层"错位"。
- 不存性能基线/逐行日志；数字用溯源纪律处理。
