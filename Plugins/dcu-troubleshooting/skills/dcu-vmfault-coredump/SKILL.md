---
name: dcu-vmfault-coredump
description: Locate the faulting kernel and the offending memory value from a Hygon DCU / ROCm HSA VMFault crash (HSA_STATUS_ERROR_EXCEPTION 0x1016, TRAPSTS=0x80000000), using the crash log's HSA QUEUE ANALYSIS (kernel name + args hex dump) and the HSA_ENABLE_COREDUMP=1 gpucore ELF memory snapshot. Answers "which kernel, which tensor, which index value" when hy-coredump faultinfo alone shows nothing useful.
version: 1.1.0
author: DCU Engineer
license: MIT
kind: tool
---

# DCU HSA VMFault Coredump 定位

> 来源：整合自个人 skill 仓库 `<personal_skill_repo>`（`dcu-vmfault-coredump`），落库为 `dcu-toolkit-skills` 仓库独立 skill（2026-08-13）。脚本已 `py_compile` 通过，仅依赖 `c++filt`(binutils) + python3 stdlib，无第三方依赖。补强 `dcu-hipprof` 的 CoreDump 实战域（hipprof 仅列子域，本 skill 提供可运行的解码脚本）。

在 Hygon DCU（ROCm/HSA 生态）上，崩溃日志只给 `HSA_STATUS_ERROR_EXCEPTION 0x1016` + TRAPSTS=0x80000000（VMFault = 页错误）。
hy-coredump 的 faultinfo 给 faulting VA（越界读时是"src 缓冲区前 4KB"或"无符号回绕后的大地址"，单独看无意义）、faulting PC 和 TRAPSTS。
faulting PC **符号化成功时 faultinfo 会直接报 kernel 名 + kernel 内指令偏移（汇编位置）**——这是"哪个 kernel"的最快证据；
但符号化可遇不可求（code object 被 strip / 重定位不全 / 内联符号不唯一 / 编译器与驱动版本不匹配时只留裸 PC），且即便点名，
faultinfo 也**没有 kernel 参数和 fault-time 数据值**——"哪个张量、索引值多少"仍要靠 args 解码 + gpucore。

**核心心法：faultinfo（符号化成功时）只点名"哪个 kernel + 哪条指令"，真正的诊断靠"内核名 + 参数 ABI + 内存快照 + 配置对号"。**

本 skill 用四个信息源把崩溃闭环定位到"哪条 kernel、操作哪个张量、越界值是多少"：

| 层 | 信息源 | 产出 |
|----|--------|------|
| 1 | 崩溃日志的 **HSA QUEUE ANALYSIS**（VMFault 回调打印） | faulting kernel 符号名 |
| 2 | 同一回调的 **DUMP KERNEL ARGS** hex dump + 内核 ABI（或 `--raw` 盲解码） | 指针、形状、标量 |
| 3 | **gpucore ELF 内存快照**（`HSA_ENABLE_COREDUMP=1`）按 VA 读内存 | fault-time 真实索引值/数据 |
| 4 | 模型配置 / 框架代码 / 队列顺序 | 缓冲区身份、op 归属、注入点 |

**通用性**：第 1/3/4 层与内核类型无关（任何 kernel 的 faulting 名字都从队列分析来、gpucore 按 VA 读任意地址）。
第 2 层是唯一需要适配的：PyTorch 内核可用 ABI 精确解码；Triton/CK/自定义 HIP 内核没有 TensorInfo 布局，
用 `--raw` 盲解码 + `--gpucore` 段交叉引用即可定位指针，见下。

配套脚本见 `scripts/`（全部读入 crash log / gpucore 路径即可跑，见 Usage）。参考 ABI、已知模式、队列分析选块、gpucore 现场数据恢复原理见 `references/`。

---

## 先判定：是不是 VMFault（闸门）

**本流程专为 VMFault 设计**：TRAPSTS 第 31 位（`0x80000000`，页错误/越界访存），或异常文本含
"memory access fault"/"VMFault"。只有 VMFault 才有"从 gpucore 内存快照读真实越界值"的意义；
非法指令、特权指令、XNACK、算术异常病根在指令/驱动/硬件而非数据，**别硬套**。

```bash
python scripts/classify_trap.py CRASH.log            # 扫描日志判 trap 类型
python scripts/classify_trap.py --trapsts 0x80000000
python scripts/classify_trap.py CRASH.log --json     # 机器可读
```

| TRAPSTS / 异常文本 | 判定 | 走不走本流程 |
|---------------------|------|-------------|
| `0x80000000`（bit31）/ "memory access fault"/"VMFault"/"page fault" | VMFault（页错误/越界访存） | ✅ 走 |
| bit0 / "illegal instruction" | 非法指令 | ❌ 查 kernel ISA/gfx arch 编译选项、代码段损坏 |
| bit1 / "privileged instruction" | 特权指令 | ❌ 查驱动/ROCm 运行时版本不匹配 |
| "XNACK"/"XNWAIT" | XNACK / wave 异常 | ❌ 硬件/驱动或二次异常 |
| 无 TRAPSTS、host 侧 | segfault / malloc 失败 | ❌ 看 host 栈，不走 |
| bit2 / "memory violation" | 内存相关但非页错误（常为二次异常） | ⚠️ 谨慎，先看 faultinfo VA |

> 非 VMFault 时 gpucore 取证意义有限，返回码 `2` 即"不走本流程"。

---

## 触发条件

- 日志或 `hy-smi`/`dmesg` 出现 HSA VMFault：`HSA_STATUS_ERROR_EXCEPTION 0x1016`、TRAPSTS=0x80000000。
- 需要定位"哪条 kernel 崩、越界值是什么"（尤其 OOB 读/写、数组索引越界）。
- 现场必须有 crash log（含 HSA QUEUE ANALYSIS）和/或 gpucore。
- **先跑 `classify_trap.py` 确认是 VMFault**，再进入工作流。

**前置/关键环境**：复现/现场进程需带 `HSA_ENABLE_COREDUMP=1`，才有 gpucore 内存快照。没有它，只能到第 2 层（指针+形状），拿不到 fault-time 数据。

---

## 输入材料规范（遇到 VMFault 带什么材料、能定位到哪层）

把本 skill 当"取证工具"：**带得越全，定位越深**。分档：

| 带什么 | 能定位到哪层 |
|--------|-------------|
| 崩溃日志（含 `Callback: Queue ... aborting` / `TRAPSTS=0x80000000` / HSA QUEUE ANALYSIS） | 第 1-2 层：faulting kernel + 指针/形状 |
| + gpucore ELF（`HSA_ENABLE_COREDUMP=1` 产物） | 第 1-3 层：+ fault-time 越界值、dst/src 逐字节验证 |
| + 环境/配置/复现材料 | 第 4-6 层：+ op 归属、根因注入点、最小复现 |

按此模板给齐材料（缺的注明"无"）：

```
DCU VMFault 现场材料
- 崩溃日志：<路径>                        # 必须，含队列分析
- gpucore：<路径>                         # HSA_ENABLE_COREDUMP=1 才有；无则说明
- server 启动：<run_server.sh 关键行/启动命令，含 env vars>
- client 触发：<什么请求/负载触发，是否 spec-decode/MTP/PD>
- 模型配置：<模型名、vocab_size、hidden_size、层数、dtype>
- 框架版本：<vllm / vllm_hcu 版本，是否 V1，VLLM_HCU_* 开关>
- 已确认闸门：TRAPSTS=0x80000000 / "memory access fault"（VMFault）
```

各材料用途：

| 材料 | 喂给哪层 | 解决什么 |
|------|---------|---------|
| 崩溃日志 | 第 1-2 层 | faulting kernel、args 指针/形状 |
| gpucore | 第 3 层 | fault-time 真实索引值/数据 |
| 模型配置 | 第 3 层 | 缓冲区身份（字节数×形状 vs vocab/hidden） |
| server/client 脚本 | 第 4 层 + 复现 | 归属哪条 forward、哪种负载触发、复现入口 |
| 复现脚本 | 第 5 层 | 最小触发条件，锤死根因 |
| 框架版本 | 第 6 层 | 注入点回溯（scheduler 行号随版本漂移，需按当前版本确认） |

> 唯一硬性前置：**崩溃日志必须有**（否则无从选块）。gpucore 决定能否答出"越界值"。env 信息决定能否复现与归属。

---

## 工作流

### 第 1 层：拿到 faulting kernel 名字

> **捷径**：先看 faultinfo 是否已符号化——若它直接报了 kernel 名 + 汇编位置，这一问当场答完，跳到第 2 层解码 args。符号化失败才走下面的队列分析。

**从崩溃日志选块（兜底，永远可靠）**：一次 VMFault 会让运行时把**所有 HOSTQUEUE（stream）**都 dump 队列分析，每个 queue 再 dump 一滑动窗口的 packet
（真实案例：1239 个 `>>>>>>>> MATCH KERNEL DETAILS <<<<<<<<<` 块，绝大部分是无关上下文）。faulting 的那个用语义判断，不看位置：

1. `Callback: Queue 0x... aborting ... 0x1016` 行点名了 faulting queue 的地址（锚点）。
2. faulting queue 是**唯一** `write index != read index` 的 queue（其余都打 `all command executed` = 已排空 = 无辜）。
   真实案例：5 个 queue，只有 stream 0 的 `write=35452708, read=35451391`（1317 个 pending），且 base_address 对上了锚点。
3. 该 queue 里 faulting packet = **read index** 那条（trap 发生时正在执行的 packet），dump 从它开始。

```bash
python scripts/find_faulting_kernel.py CRASH.log --demangle
# -> 打印 faulting queue / packet index / kernel 符号名 / args dump 行号（与 grep -n 一致）
```

在该 packet 的 `>>>>>>>> MATCH KERNEL DETAILS <<<<<<<<<` 块里，`kernel name:` 即 faulting 内核：

```
kernel name: _ZN2at6native12_GLOBAL__N_121indexSelectSmallIndexIN3c104HalfEljLi2ELi2ELin2EEEvNS_4cuda6detail10TensorInfoIT_T1_EENS7_IKS8_S9_EENS7_IKT0_S9_EEiiS9_l
```

去符号名（macOS/容器都有 `c++filt`）：

```bash
c++filt '_ZN2at6native...'
# -> at::native::{anonymous}::indexSelectSmallIndex<c10::Half, long, unsigned int, 2, 2, -2>(...)
```

模板参数逐位拆解（以 indexSelectSmallIndex 为例）：
| 参数 | 值 | 含义 |
|------|-----|------|
| T | `c10::Half` | 张量元素类型 |
| IndicesType | `long` | 索引张量 dtype = int64 |
| IndexType | `unsigned int` | TensorInfo 寻址类型 |
| DstDim/SrcDim | 2/2 | 2D 张量 |
| IdxDim | -2 | 索引维模板参数 |

这步把"访存越界"收敛到具体 op 类型（如 `index_select`/高级索引 + 2D fp16 + int64 索引）。**但还是不知道是哪个张量、索引值多少。**

### 第 2 层：解 kernel args（内核 ABI 或盲解码 + hex dump）

同一回调里 AQL packet 带 kernel 参数缓冲区：

```
>>>>>>>> DUMP KERNEL ARGS: size: 928 <<<<<<<<<
00 2a 60 ff 25 7f 00 00 01 00 00 00 10 00 00 00 ...
```

两条路径：

**(a) ABI 已知**（框架/算子 launch 处源码有布局）——PyTorch 类 kernel 常见为若干 `TensorInfo` + 标量：

```
struct TensorInfo {          // 本 fork 用 int32 sizes/strides
    T*      data;            // @0    8 字节
    int32   sizes[25];       // @8    100 字节
    int32   strides[25];     // @108  100 字节
    uint32  dims;            // @208  4 字节
    // pad -> 216 字节
};
```

```bash
python scripts/parse_kernel_args.py CRASH.log                          # 默认 indexSelectSmallIndex ABI
python scripts/parse_kernel_args.py CRASH.log ABI.json                 # 自定义 ABI（标量/u64 指针/tensorinfo）
python scripts/parse_kernel_args.py CRASH.log ABI.json --gpucore gpucore.21292  # 指针自动对上 gpucore 段
```

**(b) ABI 未知**（Triton / CK / 自定义 HIP kernel）——`--raw` 盲解码：不需要布局，按 u64 指针 + 标量扫，
再 `--gpucore` 把每个指针对上 PT_LOAD 段，识别"哪个 buffer 是哪个"：

```bash
python scripts/parse_kernel_args.py CRASH.log --raw --gpucore gpucore.21292
```

输出示例（ABI 路径 + `--gpucore`，真实案例）：
```
dst: data=0x7ef3247aaa00 dims=2 sizes=[4, 2048]  -> in seg 0x7ef3247aaa00 memsz=0x4000
src: data=0x7ef1cb000000 dims=2 sizes=[248320, 2048]  -> in seg 0x7ef1cb000000 memsz=0x3ca00000
indices: data=0x7ef3cd1e3400 dims=1 sizes=[4]  -> in seg 0x7ef3cd1e3400 memsz=0x20
srcSelectDimSize=248320  innerSize=2048
```
关键结论：src 形状、indices 长度、越界判定上界（srcSelectDimSize）、每个指针落在哪个 gpucore 段。

### 第 3 层：读 gpucore ELF 内存（指针处的真实值）

把第 2 层的指针当作 **GPU 虚拟地址**，在 gpucore ELF 的 PT_LOAD 段里按偏移读内容（VA→文件偏移换算、mmap、bss、逐字节可信性验证见 `references/gpucore-snapshot.md`）：

```bash
python scripts/decode_gpucore.py gpucore.21292 \
    0x7ef3cd1e3400:64  0x7ef1cb000000:96  0x7ef3247aaa00:64
```

读三个目标：indices 指针（期望 int64[N]）、src 头、dst 头。真实案例读到 index = **[40, 0, 29348, -1]**。

**两个决定性判定：**

**(a) src 是谁？** 字节数 = prod(shape) × dtype size，与模型配置对号。案例：`248320×2048×2 = 0x3CA00000`，且 gpucore 里恰好有 memsz=0x3ca00000 的 PT_LOAD 段、VA 恰为 src data ptr → 对照 `vocab_size=248320, hidden=2048` → src = `embed_tokens.weight`。

**(b) 索引值是 fault-time 的真实值吗？** 用 `compare_buffers.py` 逐字节比对 dst 各行与 src 词嵌入：

```bash
python scripts/compare_buffers.py gpucore.21292 \
    --src 0x7ef1cb000000:248320:2048 \
    --dst 0x7ef3247aaa00:4:2048 \
    --index 0x7ef3cd1e3400 --dtype float16
```
案例：dst row0/1/2 与 src row 40/0/29348 逐字节 64/64 匹配，row3 NaN（未写）→ kernel 按顺序写完前 3 行后在第 4 个 index(-1) 上 fault。字节级匹配排除"崩溃后内存被复用"。

### 第 4 层：归属 op（队列顺序 + 模型结构）

崩溃日志队列分析里 faulting kernel **之后**紧排的内核暴露调用上下文。案例：faulting embed 之后是 `_causal_conv1d_update_kernel`（GDN decode conv）、`fused_sigmoid_gating_delta_rule_update_kernel`（GDN decode attn）——这两个 kernel 只在 TARGET 模型的 GDN 层存在 → 归属到 TARGET `embed_tokens` 的 decode 入口。模型代码：哪些层有这些 op、哪个 forward 入口先做 embedding，对号入座。

### 第 5 层（可选）：最小复现

拿到越界值后，在同 fork 框架/同型号 DCU 上写最小脚本复现，锁定触发条件。案例：

```python
w = torch.randn(320, 16, dtype=torch.float16, device="cuda")
torch.nn.functional.embedding(
    torch.tensor([1, 2, 3, -1], dtype=torch.int64, device="cuda"), w)  # -1 -> 崩溃
```
control `[0,1,2,3]` 正常。→ 单个 -1 即触发同 kernel 同 0x1016。

### 第 6 层：回溯框架侧注入点

从越界值回溯框架（vLLM / vllm_hcu）注入点闭环根因。案例：-1 来自 scheduler 的无效 spec token 补齐 `spec_token_ids.extend([-1] * num_invalid_tokens)`（scheduler.py:2352-2356）→ `update_req_spec_token_ids` 写入 input batch → embed → 越界。

---

## 不同内核族的处理差异

崩溃的 kernel 不一定来自 PyTorch。第 2 层的解码方式按内核族选，其余层不变：

| 内核族 | 例子 | ABI 从哪来 | args 解码 |
|--------|------|-----------|-----------|
| PyTorch aten | `indexSelectSmallIndex`、elementwise | launch 处源码（`Indexing.cu` 等），TensorInfo 布局 | ABI 模式（默认/自定义 JSON） |
| torch.compile Triton | `triton_poi_fused_*`、`triton_red_fused_*` | 生成的 `.ttir`/HIP 源码，扁平 struct（指针 + int） | 有 ABI 用 ABI，没有用 `--raw` |
| Triton（手写 kernel） | vLLM 自定义 attention 等 | Triton 编译产物的参数布局 | `--raw` 盲解码 |
| CK 模板 GEMM | `Cijk_Ailk_Bljk*`、`GemmD_{...}` | CK 模板实例化的 arg struct | `--raw`（struct 里前几个是 A/B/C 指针，后面全是 stride int） |
| 自定义 HIP kernel | vLLM `hcu_cache_kernel.cu`、flash-attn HIP | 函数签名 → 扁平 struct（指针 + 标量自然对齐） | `--raw` 或自定义 JSON |

`--raw` + `--gpucore` 对以上所有族通用：盲扫出指针 → 对上 PT_LOAD 段 → 用段大小和已知张量形状对号。
归属逻辑（字节数×形状 vs 配置）与内核族无关。

---

## Usage

```bash
# 0. 闸门：先确认是 VMFault（否则不走本流程）
python scripts/classify_trap.py CRASH.log

# 1. 在 1000+ 个 FIND MATCH KERNEL DETAILS 块里选 faulting 的（write!=read 的 queue + read-index packet）
python scripts/find_faulting_kernel.py CRASH.log --demangle

# 2. 解 kernel args（crash log -> 指针 + 形状）
python scripts/parse_kernel_args.py CRASH.log                        # 默认 indexSelectSmallIndex ABI
python scripts/parse_kernel_args.py CRASH.log ABI.json               # 自定义内核 ABI（标量/u64 指针/tensorinfo）
python scripts/parse_kernel_args.py CRASH.log --raw --gpucore gpucore.21292  # ABI 未知:盲解码 + 段交叉引用
python scripts/parse_kernel_args.py CRASH.log --raw                  # 没有 gpucore 时的盲解码

# 2. 读 gpucore 内存（VA -> 真实值）
python scripts/decode_gpucore.py gpucore.21292 VA[:nbytes] [VA[:nbytes] ...]
python scripts/decode_gpucore.py gpucore.21292 --ptrs PTR [PTR ...] # 指针归属:每个指针对应哪个段

# 3. dst vs src 逐字节验证（确认 fault-time 数据；index_select/embed 类模式专用）
python scripts/compare_buffers.py gpucore.21292 \
    --src SRVA:nrow:ncol --dst DSTVA:nrow:ncol --index IDXVA [--dtype float16]

# 4. 去符号（也可直接用 c++filt）
python scripts/demangle.py 'KERNEL_NAME'
```

## 通用步骤清单

0. **闸门**：`classify_trap.py` 确认是 VMFault（TRAPSTS=0x80000000 / memory access fault）。非 VMFault → 换诊断路径。
1. 崩溃日志 → `find_faulting_kernel.py` 在众多队列分析块里选 faulting 的（`write!=read` 的 queue + read-index packet）→ 记下 **kernel 符号名** + `DUMP KERNEL ARGS` 起始行。
2. `c++filt` 解符号 → 确定内核族（PyTorch aten / Triton / CK / 自定义 HIP）+ op 类型 + dtype。
3. 查该 kernel 的 launch 处源码 → 拿**参数 ABI**（TensorInfo 布局 / 扁平 struct）；查不到就用 `--raw` 盲解码。
4. 解 args dump → 指针 + 形状（有 `--gpucore` 则指针直接对上段）。
5. 读 gpucore ELF 在指针处的内存 → 真实索引值、src 数据、dst 内容。
6. 用**字节数×形状 vs 模型配置**判定缓冲区身份；index_select/embed 类用 dst vs src 逐字节匹配确认 fault-time 数据。
7. 用 faulting 之后的队列内核 + 模型层结构归属 op。
8. 最小脚本在相同 DCU 上复现 → 锁定触发条件。
9. 回溯框架侧注入点 → 根因闭环。

## 工具依赖

- `c++filt`（binutils，macOS/容器通用）
- python3 + struct（脚本无第三方依赖）

## 关键环境

- 复现/现场进程保留 `HSA_ENABLE_COREDUMP=1`（run_server.sh 已带），否则没有 gpucore 内存快照。
- gpucore 很大（真实案例 56GB），用 PT_LOAD 段按偏移读，不加载全量。

## 驱动自带的 core/调试工具（/opt/hyhal/bin，<device-A> 实测）

> 来源：2026-08-28 在 <device-A> 设备对 `/opt/hyhal/bin` 扫描 + `--help` 实测。除本 skill 的 python 解码脚本外，
> 驱动还自带两个取证工具，崩溃现场可先用它们做初步解析：

| 工具 | 用途 | 关键参数 |
|------|------|------|
| `hycoredump`（v1.1.0） | **core 文件解析**（对应 faultinfo/gpucore 的驱动侧工具） | `-i <core>` 输入（默认 `core_0`）、`-o <dir>` 输出目录、`-m <dir>` mem 目录、`-r <dir>` rocm 目录（默认 `/opt/rocm`，**DTK 环境应指到 `/opt/dtk`**）、`-l <error/info/debug>`、`--dis_all` 反汇编全部 code object |
| `hycapture`（v1.0.0） | **HCU 调试器抓取**（对运行中进程抓 HCU 侧状态） | `-p <pid>` 指定进程、`-a` 抓全部进程信息、`-nd` driverless 模式、`--log-level` |

```bash
# core 解析示例（DTK 环境）
/opt/hyhal/bin/hycoredump -i core_0 -o /tmp/coreout -r /opt/dtk -l info
# 运行中进程 HCU 状态抓取（先确认目标 pid，会 attach 进程）
/opt/hyhal/bin/hycapture -p <pid>
```

> 关联：`hiplogdump`（`/opt/hyhal/bin/hiplogdump sort|thread|unsort|rcdf <log>`）解码 HIP_LOG 压缩日志，
> 崩溃日志里的 HSA QUEUE ANALYSIS 若来自 HIP 日志文件，可先用它解压排序再喂给本 skill 的 `find_faulting_kernel.py`。
