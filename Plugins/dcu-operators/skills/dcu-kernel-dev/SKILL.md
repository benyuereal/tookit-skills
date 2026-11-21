---
name: dcu-kernel-dev
description: "DCU算子开发规范：HIP kernel开发与TileLang DSL写算子两路径"
version: 1.1.0
author: DCU Engineer
license: MIT
---

# DCU 算子开发规范（dcu-kernel-dev）

> 定位：海光 DCU 上写算子的**方法学 + 踩坑**沉淀。两条路径：
> - **路径 A · HIP kernel**：CUDA→HIP 迁移、手写/融合算子，落点见 `dcu-vllm-op-replace` 第 1 步。
> - **路径 B · TileLang DSL**：用类 Triton 的 Python DSL 写算子，编译进 vLLM/SGLang，**绕开 csrc 重编译**。本文档把 TileLang 在 DCU 上的安装、写法、已知问题、调试系统沉淀于此（源文档三篇：tilelang 主实践、tilelang 用法总结、GLM-5.1 K100 TileLang 部署）。
>
> 内容纪律：性能基线数字/逐行日志**不入库**（见 `dcu-perf-cases` 附录方法论），本页只留可复用的「方法 + DCU 专属坑」。所有数值结论以源文档 内部文档 为准。

---

## 一、何时用 TileLang（路径 B）vs 手写 HIP（路径 A）

- **TileLang**：DSL 代码量约为手写 HIP 的 1/5，支持 **HCU 后端**，kernel 迁移周期约 1 周（手写以周计且风险高）。适合「链路未打通、上游接口可能变」的迭代期，以及中小算子融合。
  - 实战依据：GLM-5.1 Sparse MLA 三个缺失算子（MQA Logits 非分页/分页、Sparse MLA Forward）全用 TileLang 打通，性能达手写汇编（lightop）的 80%–108%。
- **手写 HIP**（路径 A）：追求极限性能、或 TileLang/编译器尚不支持的算子。落点 `csrc/opt/`（海光新增优化）或 `csrc/rocm/`，头带 `#ifdef USE_ROCM`，重编译见 `dcu-vllm-op-replace`。

---

## 二、TileLang 安装与环境

### 2.1 基础依赖
```bash
# 基础环境（清华源，通用）
pip install apache-tvm-ffi pytest z3-solver -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- **libz3.so 缺失**：`OSError: libz3.so: cannot open shared object file`
  → `export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/z3/lib:/usr/local/lib/python3.11/site-packages/z3/lib:$LD_LIBRARY_PATH`

### 2.2 tilelang 安装（两条源，版本不同）
| 来源 | 安装串 | 备注 |
|---|---|---|
| 海光内网 git（含 lower-pass 修复） | `git clone http://<internal_ip_ci>:10068/<internal-group>/tilelang.git` → `git switch release/0.1.7.post3` → `USE_HCU=ON USE_CUDA=OFF pip install -e . -v --config-settings=install.strip=false --config-settings=cmake.build-type=Debug` | 内网，需权限；K100 AI(RDNA) 用 `USE_HCU=ON` |
| PyPI（sourcefind 海光定制轮） | `pip install tilelang==0.1.9+das.opt1.dtk2604.torch290.2605071954.g88ce2a -i https://pypi.sourcefind.cn/release/dtk2604/ --trusted-host pypi.sourcefind.cn --no-deps` | 记验证过的路径是 `/release/dtk<ver>/`（非 `/nightly/`） |
| 官方 git | `https://developer.sourcefind.cn/codes/tsoc/tilelang.git` / `github.com/tile-ai/tilelang.git -b v0.1.10` | 上游，版本更新 |

- 重装须加 `--no-build-isolation`；配套 tvm：`pip install apache-tvm-ffi==0.1.9`。
- **python 版本坑**：sourcefind 轮标 py38，但在 py310/py311 实测可用（无强制版本需求）。

### 2.3 DTK / 镜像
- DTK 选 `dtk2604`（文档实测）。
- 可用镜像：`<internal_registry>/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`（及 py3.11）。
- 文档作者 vllm0.18.1 镜像：`<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`（启动须 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见）。

---

## 三、TileLang 架构与写法核心（DCU 必读）

### 3.1 后端选择（最关键）
```python
backend_target = "hip" if torch.version.hip else "cuda"
_TILELANG_KERNEL_CACHE[cache_key] = tl.compile(gemma_rms_norm_kernel, target=backend_target)
```
→ **DCU 上必须显式 `target="hip"`**，否则编译出 CUDA 代码在 DCU 跑错。

### 3.2 动态维度用符号，别硬编
```python
M = T.dynamic('M', 'int32')   # batch 等动态维度
```
用 Python 变量硬编 → 编译器无法识别传入参数 M。

### 3.3 内存层级：alloc_local vs alloc_shared（精度命门）
- `T.alloc_local` = **寄存器**（线程私有，安全，映射到寄存器）。
- `T.alloc_shared` = **共享内存**（块共享，映射到 LDS）。
- ⚠ **精度敏感的中间量必须用 `alloc_local`**。误用 `alloc_shared` 时，多线程竞争写同地址、且每次执行写入顺序不同 → 结果随机、每跑一次都不一样。实战踩坑：把 `local_sum` 误声明为 shared 导致精度时好时坏。

### 3.4 DCU warp_size = 64（warp shuffle 精度坑）
- `T.shfl_down()` 等 warp shuffle 默认 `warp_size=32`；**DCU 上须显式指定 `warp_size=64`**。
- 不指定 → 编译器按 32 处理，0 号线程以 offset 32 读 32 线程值判为越界、越界直接返回自身值 → 实际 `warp_sum[0] = 自身+自身` → 规约结果错。

### 3.5 k_pack 与架构（RDNA/CDNA）
- `k_pack=1 for RDNA (gfx936)`；`k_pack=2 would be for CDNA`。按 topk 分档：topk=1024→k_pack=2，topk=2048→k_pack=4（平衡 LDS 带宽与稀疏适应性）。

### 3.6 调试：导出 / 替换 lower 产物
```bash
TILELANG_KERNEL_DUMP=1 TILELANG_CACHE_DIR=./cache python xxx.py   # 导 lower 后 CU 源码 + 汇编
TILELANG_OVERRIDE_DEVICE_SOURCE=cache/xxx.cu python xxx.py        # 跳过 lower，直接用本地 CU（二分定位编译器 bug）
```
- 验证产生的 HIP 代码：`export TVM_LOG_DEBUG=1`，`print(_TILELANG_KERNEL_CACHE[cache_key].get_kernel_source())`（向量化实现中以 float4/int4 load/store 打满 128-bit 位宽）。

### 3.7 PyTorch dynamo 墙
tilelang 算子体内不允许存在 I/O 等 → `torch.compile` 会图 break（dynamo 墙）。写 kernel 时保持纯计算。

---

## 四、TileLang 在 DCU 的专属已知问题（实测汇总）

| # | 问题 | 现象 / 根因 | 对策 |
|---|---|---|---|
| 1 | 后端选错 | 未显式 `target="hip"` → CUDA 代码跑在 DCU | `backend_target = "hip" if torch.version.hip else "cuda"` |
| 2 | warp_size | DCU=64，shfl 默认 32 → 越界返回自身值 | 显式 `warp_size=64` |
| 3 | alloc_shared 竞争写 | 误当寄存器用 → 结果随机 | 精度敏感量用 `alloc_local` |
| 4 | 动态参数 M | 编译器不识别 Python 变量 | `T.dynamic('M','int32')` |
| 5 | libz3.so | `OSError: libz3.so` | 见 §2.1 `LD_LIBRARY_PATH` |
| 6 | hipprof libLLVM | `libLLVM-17git.so: cannot open shared object file` | `export LD_LIBRARY_PATH=/opt/dtk-26.04-DCC2602-0317/dcc/lib:$LD_LIBRARY_PATH` |
| 7 | amd tensor layout | 部分 layout 在 DCU 报错；同类型不同 dtype 共用内存属性 cast → 框架布局推断失败 | 换 shared memory 做 cast；资源越界时调低块尺寸 |
| 8 | im2col 不支持 | `tilelang.language` 无 `im2col` | 规避 / 等支持 |
| 9 | 不支持 CUDA arch 引 | `from tilelang.carver.arch import CUDA` 在 DCU 报错 | 改用 `from tilelang.carver.arch import CDNA` |
| 10 | 不支持 93 arch | Analyzer 的 ARCH_CONFIGS 无 93 | 等框架支持 |
| 11 | Analyzer 不输出 flops | `Analyzer.analysis` 功能问题 | 手动算 / 报告 bug |
| 12 | invalid argument | 一般是**资源越界** | 调低块尺寸（如 block_M=32,block_N=32,num_stages=1,threads=128）；断言 `total_shared <= max_shared` 提前发现 |

> 通用修复「资源越界」：`assert total_shared <= max_shared, "Shared memory too high"`，继续调低 block/threads。

---

## 五、编译安装与 lower-pass 调试实战（GLM-5.1 K100 案例）

### 5.1 安装 release/0.1.7.post3（HCU 后端）
```bash
pip uninstall tilelang -y
git clone http://<internal_ip_ci>:10068/<internal-group>/tilelang.git
git switch release/0.1.7.post3
USE_HCU=ON USE_CUDA=OFF pip install -e . -v \
  --config-settings=install.strip=false --config-settings=cmake.build-type=Debug \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
# 重装须 --no-build-isolation；配套 tvm:
pip uninstall apache-tvm-ffi -y
pip install apache-tvm-ffi==0.1.9 -i https://pypi.tuna.tsinghua.edu.cn/simple
python sparse_mla_fwd.py --check   # 先功能测试
```

### 5.2 lower-pass bug 二分定位（必学）
**问题**：Sparse MLA Forward kernel 初期，相同输入多次运行结果不一致，与 PyTorch 参考偏差超合理范围。
**定位（四步）**：
1. `TILELANG_KERNEL_DUMP=1 TILELANG_CACHE_DIR=./cache python sparse_mla_fwd.py` 导 CU 源码 + 汇编；
2. 逐函数比对 CU 源码，排除逻辑错误；
3. `TILELANG_OVERRIDE_DEVICE_SOURCE=cache/xxx.cu python sparse_mla_fwd.py` 用本地 CU 替换 lower 输出，逐文件二分；
4. 锁定 **lower pass 在分支内部错误插入 `__syncthreads()`** → 死锁/数据竞争。
**修复**：回退 tilelang 至 `release/0.1.7.post3` 重新编译。
**启示**：选 DSL/编译器方案优先考察其调试工具链完整性；`TILELANG_OVERRIDE_DEVICE_SOURCE` 的「跳过 lower、直接用本地 CU」机制使编译器 bug 排查可控、可二分，否则周期从数天膨胀到数周。

---

## 六、参考实现（scripts/）

| 文件 | 内容 | 溯源 |
|---|---|---|
| `scripts/tl_gemma_rmsnorm_naive.py` | tilelang naive GemmaRMSNorm（含寄存器/共享内存用法、backend=hip） | 内部文档 旧主文档 §2.2（file_id=<internal-file-id>） |
| `scripts/tl_paged_mqa_logits.py` | tilelang paged_mqa_logits (BF16) 完整实现：kernel + PyTorch 参考 + lightop 接口 + 正确性校验 + benchmark | 内部文档 doc3 附录 C.1（file_id=<internal-file-id>） |
| `scripts/tl_rmsnorm_precision_test.py` | RMSNorm 精度验证 demo（独立 f32 reference 对比，判据 <0.03 安全） | 内部文档 主文档 §2.4（file_id=<internal-file-id>） |
| `scripts/tl_rmsnorm_benchmark.sh` | 端到端 profiling 流水线（hipprof --hip-trace 包 vllm bench，产出 .hipkernel.csv/.hiptrace.csv） | 内部文档 主文档 §3.1（file_id=<internal-file-id>） |
| `references/launch.json` | debugpy 调试配置（HIP_VISIBLE_DEVICES 指定卡） | 内部文档 主文档 §1.3（file_id=<internal-file-id>） |

> 运行 `tl_paged_mqa_logits.py`：`python tl_paged_mqa_logits.py --validate`（正确性）/ `--benchmark`（性能）。代码内 `device="cuda"` 在 hip 运行时下可用（torch.cuda.* 被 hip 映射），`tl.compile` 须 `backend_target="hip"`。
> `tl_rmsnorm_precision_test.py` 依赖 vllm 已注册 `torch.ops.vllm.qwen3_5_rms_norm` 并设 `USE_TILELANG=1`；若只验证 tilelang kernel 本身，可改为直接调用 `tl_gemma_rmsnorm_naive.get_cached_tilelang_kernel`。
> `tl_rmsnorm_benchmark.sh` 须挂载 `/opt/hyhal` 的 DCU 容器内运行；路径/卡号用环境变量覆盖（见脚本头注释）。

---

## 七、交叉引用

- 算子替换进 vLLM（csrc 路径 / TileLang 路径）→ `dcu-vllm-op-replace`。
- GLM-5.1 K100 AI 用 TileLang 打通 Sparse MLA 的端到端案例、量化结论、部署 env → `dcu-perf-cases` §8。
- 算子优化闭环/留痕 → `dcu-op-optimization`；度量 → `dcu-perf-profiling`。
- DTK/镜像 → `dcu-dtk`、`dcu-docker`。

## 八、RMSNorm TileLang 优化方法论（三级 + 精度陷阱）

> 来源：内部文档 主文档 §4–§5（file_id=<internal-file-id>）。
> 本节只留**优化方向方法论**，不存性能基线数字（ns 仅作示例量级，以本地实测为准）。

### 8.1 优化方向总览
RMSNorm 分两阶段：①规约（整行 x² 求和得标量 RMS）②逐元素归一化×(1+w)。三个优化方向：

| 方向 | 目标 | 对应手段 |
|---|---|---|
| 减少访存延迟 | 打满显存带宽 | 向量化访存 VEC=8（每线程连续 8×f16=128bit，对齐 GLOBAL_LOAD_DWORDX4 128-bit 位宽） |
| 增大并行度 | 提升 SM 占用率 | Block Size 手动寻优（256→1024，权衡单线程工作量 vs 并行） |
| 消除同步瓶颈 | 减少规约 sync 气泡 | Warp Shuffle 替代 `T.serial(8)`+`sync_threads` 树状规约（warp_size=64，见 §3.4） |

### 8.2 向量化访存（打满带宽）
- 瓶颈：naive 每线程一次 load 1 个 f16（16bit），128-bit 指令仅用 12.5%。
- 做法：每线程访问连续 8 个 f16 → 8×16bit=128bit，打满 load/store 位宽。
- 验证：`export TVM_LOG_DEBUG=1`，`print(_TILELANG_KERNEL_CACHE[cache_key].get_kernel_source())`，向量化实现应见 `float4`/`int4` 做 load/store。

### 8.3 Block Size 手动寻优
单线程工作量与并行度之间的平衡，需实测（DCU 纪律：不入库基线，本地跑 `tl_rmsnorm_benchmark.sh` 取本机量级）。规律（示例量级，非基线）：block 越大平均耗时越低，但见 §8.5 精度代价。

### 8.4 Warp Shuffle 消除规约同步
用 `T.shfl_down()` 做 warp 内规约，免去多级 `sync_threads` 气泡。**必须 `warp_size=64`**（DCU ≠ NVIDIA 32，见 §3.4），否则越界返回自身值 → 规约结果错。

### 8.5 ★精度陷阱（核心收获，必读）
- **现象**：block_size 增大 → 规约的「加法顺序」改变 → IEEE754 舍入顺序变化 → bf16 写回撞**舍入悬崖（Rounding Cliff）**。
- **机理**：`(A+B)+C != A+(B+C)`；block=1024 时 `local_sum` 在规约树最后阶段才相加远距元素，顺序与 torch.mean() 不同；强转 bf16 时一方向上舍入、另一方下舍入，差达 1 ULP。
- **量级**：block=256 → max diff 可 0；block=1024 → max diff=0.03125（bf16 在 [4,8) 区间 1 ULP）。
- **判据**：max diff < 0.03（≈1 ULP）视为**安全范围内**，可放行；否则回退较小 block 或对齐 torch 的规约顺序。
- **工具**：`tl_rmsnorm_precision_test.py` 用独立 f32 reference 对比，输出 max/mean diff 与 NaN 检查。

### 8.5b ★bit-identical 门禁：复刻 torch 两步运算的中间舍入（Triton 融合 kernel 必读）

- **现象**：把 torch 的 `a * torch.sigmoid(g)` 两步融合成一个 Triton kernel，直接 `a * tl.sigmoid(g)` 后转 bf16，结果 ~75% 元素与 torch 逐位一致、maxdiff 达 1.5e-2（1 ULP 量级）——**不是 bit-identical**。
- **根因**：torch 两步执行时 `sigmoid` 的结果**先舍入到 bf16**，再做 bf16×bf16（fp32 opmath）乘并舍入回 bf16。融合 kernel 若在 fp32 里一路算到底，省掉了中间那次 bf16 舍入，结果不同。
- **对策**：kernel 里显式复刻中间舍入——`s = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)` 再 `out = (a * s).to(tl.bfloat16)`，即可 100% bit-identical（2026-08-21 <device-A> 实测，全形状 maxdiff 0.0）。
- **纪律**：融合替换现有 kernel 时，门禁必须用**真实布局**验证（含非连续 stride 视图，如 qkv split 出来的 k），不能只测 contiguous；bit-identical 通过才允许设生产默认（无质量风险）。

### 8.6 优先级结论（方法论，非基线）
GemmaRMSNorm 在推理耗时中占比极小（量级 ~0.01%），对端到端吞吐几乎无影响。**DCU 上应优先把精力投入 fa / gemm / elementwise 等热点算子**（占推理耗时大头），RMSNorm 这类算子做到「正确 + 不拖后腿」即可。

### 8.7 端到端 profiling 流水线
1. 起服务 + 压测，用 `hipprof --hip-trace --output-type 0 bash <脚本>` 包裹（见 `tl_rmsnorm_benchmark.sh`）。
2. 产物：`result_*.json`（Chrome Trace/Perfetto 可视化）、`*.hipkernel.csv`（各 kernel 调用次数/总/均耗时/占比）、`*.hiptrace.csv`（HIP API 统计）。
3. 读数：原生 RMSNorm 在 PyTorch 拆成多个 element-wise/reduce kernel，不单一名出现；Triton 融合版名字含 `mean/mul/pow/rsqrt` 即 RMSNorm。

---

## 九、tilelang 各 example 在 DCU 的通过性矩阵（已下沉）

各 example 的编译通过性矩阵已移至 [references/tilelang-pass-matrix.md](references/tilelang-pass-matrix.md)。

---

## 十、结构化 kernel 开发 playbook（源自 tensormux/kernel-skills）

> 来源：`github.com/tensormux/kernel-skills`（⭐53, MIT）。其 README 证明：同一模型同一 prompt，
> 有无一个结构化 skill 文件，softmax kernel 的正确性/带宽差异巨大（naive 在 N>256 静默错、对抗形状 8/8 全 fail；
> skill 引导版 0/16 失败，带宽与 torch.softmax 差 1.2%）。DCU 上写算子同样适用：**先约束、后算法、再正确性**。

写任何 DCU kernel 前，agent 必须强制走完下面清单（禁止拿到 prompt 就写代码）：

1. **收集约束（动手前）**：目标 shape/dtype/精度容差、目标 gfx（**用 §13.2 的探测脚本现查，不假设**）、
   DTK 版本、是否融合、参考实现来源。
2. **选算法与内存策略**：根据问题选 tiling / 向量化 / Matrix Core / warp shuffle，明确 memory-bound vs compute-bound。
3. **显式推理正确性风险**：边界 shape（257/质数/odd）、对齐、NaN/Inf、舍入顺序（见 §八 精度悬崖）。
4. **拒绝伪性能声明**：任何 speedup 数字必须来自真机 `METRIC` 行（见 §十一），禁止估算/编造。
5. **知道何时不该写自定义 kernel**：框架已有高效实现（如 hipBLAS/rocBLAS 覆盖的 GEMM）时不要重复造。

## 十一、profile 驱动优化循环协议（DCU 版，源自 auto-kernel-research）

> 来源：`github.com/Dominic789654/auto-kernel-research`（MIT，纯协议）。与 `dcu-op-optimization`
> （profile→定位→改→验证）同构，此处给可执行循环。完整版见 `dcu-kernelwiki/wiki/optimization_loop_protocol.md`。

铁律：
- **metric 只认 `METRIC name=value` 行**（测量命令打印，只解析这些行；无行=crash 不是 0）。
- **瓶颈声明只能源于本次会话落到磁盘的 profiler 报告**（hipprof/rocprof），禁止读代码猜。
- **正确性门禁不过，再快也是废纸**。
- **git 为真相**：每实验一 commit，败者 `git reset --hard HEAD~1`。
- **置信度用 MAD**：GPU jitter 大，metric 跑 5+ 次，`|delta| > 2×MAD` 才算真改善；锁时钟（`rocm-smi --setclock`）更佳。

循环每轮：HEARTBEAT → THINK（假设必须引 profile 某行）→ EDIT（≤5 文件、小 diff）→ COMMIT →
CHECK → MEASURE → DECIDE（keep/loss 回退）→ 重大胜出后 RE-PROFILE（瓶颈会移动）→ LOG。
反停滞：`stale_count≥2` 停调旋钮去重 profile；`≥4` 换结构性方向。

## 十二、Agent 驱动 kernel 优化 flow（已下沉）

KDA + BBuf 的 Task Contract / 最小循环 / 证据留痕 / 与既有流程关系全文移至 [references/agent-optimization-flow.md](references/agent-optimization-flow.md)；设备/编译探测脚本仍在本文件 §13.2。


## 十三、DCU 编译硬约束与设备架构探测（不硬编码 gfx）

### 13.1 为什么不能写死 gfx

DCU 有多代架构（K100/Z100→`gfx906`、K100-AI→`gfx928` 等），且算子编译/调优通常在**远端 DCU 物理机**
（本机无卡）。原 `shibingzhang/XDZS2025` 把 `--offload-arch=gfx928/gfx906` 写死在 Makefile/compile.md，
是不可移植的反例。正确做法：**编译前在目标机现查设备架构**。

### 13.2 探测 + 编译脚本（本 skill 提供）

- `scripts/dcu_detect_arch.sh` — 运行时探测 gfx（依次试 `rocminfo` / `hipconfig` / `hy-smi`，
  兼容无卡场景返回 product 名提示人工映射）。输出如 `gfx928`。
- `scripts/dcu_build_kernel.sh <src.cpp> [额外参数]` — 自动探测 arch 后用该 gfx 作 `--offload-arch` 编译；
  允许 `DCU_ARCH=gfx928` 手动覆盖。

```bash
# 在挂载 -v /opt/hyhal:/opt/hyhal:ro 且有 /dev/kfd 的 DCU 容器内：
bash scripts/dcu_detect_arch.sh            # 例：gfx928
bash scripts/dcu_build_kernel.sh <你的kernel>.cpp -o out
DCU_ARCH=gfx906 bash scripts/dcu_build_kernel.sh <你的kernel>.cpp
```

### 13.3 容器/物理机纪律（与 memory 一致）

- 进 DCU 容器必须 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见（DEVCOUNT=0）。
- gfx 用**物理机** `rocminfo` 查；容器内 `rocminfo` 不可靠（PATH/LD 不全）。
- 所有数值/正确性结论必须在真机（有 `/dev/kfd` + hyhal 挂载）验证，本机/容器里只做代码审查。

### 13.4 XDZS2025 抽取参考（DCU 原生算子）

XDZS2025 仓库（`shibingzhang/XDZS2025`，开源包未随附代码）含：
- `SGEMM/MCv1.cpp, MCv2.cpp` — Matrix Core GEMM（tiling + 寄存器复用 + LDS padding，K100-AI/dtk25.04）
- `SpMM/spmm.cpp` — 稀疏矩阵乘（依赖 hipsparse）
- `Conv2d/` — 卷积前向（gfx906 路径）
- `README_XDZS2025.md` — 原作者的 Matrix Core 指令/布局/资源实测结论

> 该仓是**唯一公开面向 DCU K100-AI 的算子实现**，含 `__builtin_amdgcn_mmac_f32_16x16x8f32` 等
> Matrix Core 指令的 wavefront 数据布局反编译结论，是 DCU 手写算子的第一手参考。已并入本 skill 供离线查阅。
> 注意：无 license、已停更，商用/分发前需评估；生产落地建议对照 `ROCm/rocm-libraries` golden reference 复核。

### 13.5 验证优先级（设备，不写死型号）

DCU 机型多，**本 skill 一律不在脚本/文档里写死卡型号或 gfx**——编译目标由 `dcu_detect_arch.sh` 在目标机运行时现查。

- **首批真机验证（2026-08-06 实测）**：
  - **<device-A> = BW150（8 卡），gfx936** —— `dcu_build_kernel.sh` 探测到 gfx936 并编译通用 GEMM demo（`hip_gemm_tiled.cpp`），运行 `status=PASS`（max_diff=0.000977, errors=0）。证据已回填 `dcu-kernelwiki` 硬件作用域表。
  - **<device-B> = BW100（8 卡），gfx936** —— `rocminfo` 确认 gfx936×8。
  - ⚠️ 注意：<device-A>/<device-B> **不是 K100-AI**，而是 BW150/BW100，且**两者都是 gfx936**（与最初"<device-A>/<device-B> 首批、BW1000/BW1100 后续"的预估不同，以真机为准）。
- **BW1100 实测（2026-09-15，<device-alias>/<<internal-ip>>，OAM+HSW 形态）**：
  - **<device-D> = BW1100（8 卡，HYGON DCU-3G），gfx938** —— `dcu_detect_arch.sh` 探测到 **gfx938**，`dcu_build_kernel.sh` 用 `hipcc --offload-arch=gfx938` 编译 `hip_gemm_tiled.cpp` 成功，运行 `status=PASS`（**max_diff=0.000977, errors=0**，与 <device-A> gfx936 完全一致）。
  - 说明：gfx938 是 **DCU-3G 新一代架构**（与 BW150/BW100 的 gfx936 不同代），但通用 tiled GEMM 的浮点结果与 gfx936 一致（无 Matrix Core 依赖的 float GEMM 结果不受代际影响）。
  - ⚠️ 注：<device-alias> **宿主机** DTK-25.04.4 上 `/opt/dtk/bin/rocm_agent_enumerator` 只列到 `gfx936`（**未含 gfx938**），但 `rocminfo` 实测为 gfx938、`hipcc --offload-arch=gfx938` 也能正常编译——说明该版本 enumerator 的架构列表滞后，不影响编译，勿以 enumerator 为准。**且 BW 系列须用 DTK-26.04+，宿主 DTK-25.04 不满足要求（见 §15.2b）**。
- **后续补充（待实测设备）**：BW1000 等 —— 同一流程，探测到的 gfx 自动成为编译目标，脚本零改动。
- **XDZS2025 抽取代码的架构约束**：其 `SGEMM/MCv1.cpp`（及 MCv2，源码在 GitHub 仓库，未随包分发）用了 `gfx928`（K100-AI）专属的 Matrix Core intrinsic `__builtin_amdgcn_mmac_f32_16x16x8f32`。在 gfx936（<device-A>/<device-B>）等非 928 设备上，若该 intrinsic 不可用，`hipcc --offload-arch=<探测gfx>` 会**编译失败并报 intrinsic 不支持**——这是预期内的架构差异，不是脚本 bug。遇到时：
  1. 用 `dcu_detect_arch.sh` 确认目标 gfx，查该架构的 Matrix Core 指令集；
  2. 若目标 gfx 无对应 MC intrinsic，回退到该架构支持的 GEMM 实现（rocBLAS/hipBLAS 或通用 tiling 版，如 `scripts/hip_gemm_tiled.cpp`），**不要把 928 专属 intrinsic 硬套**。
- 每在一款新设备验证通过，建议在 `dcu-kernelwiki` 的「硬件作用域」表补一行（gfx + 验证来源），形成证据飞轮。

## 十四、HIP kernel 调优速查表与示例代码（基础补全）

> 来源：整合自个人 skill 仓库 `<personal_skill_repo>` 的 `hipc_kernel_opt_skills`（`DCU-kernel-optimize`，2026-08-13）。本仓库 `dcu-kernel-dev` 已覆盖更深的方法学，此处补全「可直接抄的通用示例代码 + 现象→根因→方案 速查表」，作为新手入门与快速排查的索引。

### 14.1 常见性能瓶颈速查表

| 现象 | 可能原因 | 解决方案 |
|---|---|---|
| 占用率低 | 寄存器过多 / LDS 过大 | 减少变量作用域，使用 `__launch_bounds__` |
| 内存带宽低 | 非合并访问 | 调整线程块布局，使用向量类型（float4/double2） |
| 计算单元闲置 | 分支发散 / 同步开销大 | 重构分支，减少 `__syncthreads()` |
| L2 命中率低 | 数据重用差 | 使用 LDS 缓存，调整 tile 大小 |
| 峰值计算未达 | 未使用向量/矩阵指令 | 手动向量化，启用 fast-math，GEMM 走 v_mmac |
| SMEM 访问串行化（bank conflict） | SMEM 布局导致多地址落同 bank（如同行 r 与 r+8 同 bank） | 量化 bank_index 分布（byte_offset 公式），改 stride/padding 打破冲突（见 `dcu-kernelwiki/wiki/lds_bank_conflict.md` + `dcu-op-optimization` 案例七） |

### 14.2 向量化加载示例（打满 128-bit 位宽）

```cpp
// 使用 float4 一次加载 16 字节（对齐到 128 字节：__align__(16)）
__global__ void vector_load(float4* in, float4* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float4 data = in[tid];
    data.x += 1.0f;
    out[tid] = data;
}
```

### 14.3 LDS 树形规约示例

```cpp
__global__ void reduce_lds(float* in, float* out) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = in[idx];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {  // 树形规约
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
```

### 14.4 矩阵乘法 Tiling 示例（LDS 分块）

```cpp
__global__ void matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    float sum = 0.0f;
    for (int k = 0; k < N / 16; k++) {
        As[ty][tx] = A[(by * 16 + ty) * N + (k * 16 + tx)];
        Bs[ty][tx] = B[(k * 16 + ty) * N + (bx * 16 + tx)];
        __syncthreads();
        for (int i = 0; i < 16; i++) sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    C[(by * 16 + ty) * N + (bx * 16 + tx)] = sum;
}
// 块大小 (16→32) + 向量加载可进一步提升
```

> 注：以上为通用 HIP 基础范式。DCU 上的**架构硬约束**（wavefront=64、LDS=64KB、gfx 现查不写死）与**正确/精度陷阱**见本 skill §十~§十三，禁止把示例当 DCU 已验证实现直接上线。

---

## 十五、DCU 专属优化轴与工具速查（去重自 auto-*-opt）

> 来源：整合自个人 skill 仓库 `<personal_skill_repo>` 的 `auto-kernel-opt-skill` / `auto-vllm-opt-skill`（Claude Code agentic 工作流，其 `agents/commands/templates` 因 agent 运行时不支持未搬入；仅抽取 DCU 专属且本仓库未覆盖的干货，2026-08-13）。

### 15.1 DCU ≠ NVIDIA 关键事实（写 kernel 前必记）

- Wavefront = **64** 线程（非 warp 32）。
- 矩阵指令 = `v_mmac`（`__builtin_amdgcn_mmac_*`，非 CUDA `mma.sync`）。
- LDS = **64 KB**（非 NVIDIA 48KB 可配）。
- 峰值参考（gfx936）：FP16 Tensor ≈ **442.4 TFLOPS**；HBM 带宽 ≈ **1.8 TB/s**。（数值随机型/gfx 变化，以 `rocminfo` + 实测为准，禁止硬编码进 skill）
- **DCU≠NVIDIA 硬约束（每 block 线程上限 256 / 宏括号 / 跨卡 IPC 失败 / "融合不一定更快"）**：详见 `references/sglang-mono-repro-20261020.md`（2026-10-20 dcu-bw150 sgl-amd-mono 复现实测）。

### 15.2 优化轴优先级（按 kernel 类型）

1. **结构先行（所有类型）**：tile/block 尺寸（block 维必须是 64 的倍数）→ LDS double buffer（`num_stages=2`）→ 向量化加载（float4，对齐 16 字节）→ warp/thread 映射（GROUP_SIZE_M、64/wavefront）。
2. **GEMM/Attention（矩阵密集）**：MMAC 利用率（确认 `tl.dot`/`v_mmac` 被用）→ LDS 矩阵读取布局（`ds_read_m32x16_b16` 而非 `ds_read_b32` 喂 MMAC）→ Split-K/并行规约 → epilogue 融合 → gfx938 上 FP8/BF8 MMAC 提 2×。
3. **Elementwise/Reduction**：persistent kernel（grid 大时）→ 寄存器 tiling（每线程 2-8 元素）→ SGPR 化指针/边界（释放 VGPR）→ `tl.arange` 索引必须编译期常量 → 跨 lane 操作用 `tl.inline_asm_elementwise` 代替 LDS 规约。

### 15.2b ★ BW 系列 (gfx938/BW1100) 的 FP8/BF8 支持实测（2026-09-15，决定性结论）

> ⚠️ **先决条件：BW 系列必须用 DTK-26.04 及以上版本**。**宿主机默认 /opt/dtk 可能指向 DTK-25.04（如 <device-alias> 宿主 = DTK-25.04.4），不满足 BW 系列要求**。**用容器镜像自带的 DTK（容器内 /opt/dtk = DTK-26.04-rc4）**，不要挂载宿主机 /opt/dtk 覆盖它。之前用宿主 DTK-25.04 测 FP8 得出"fp8-insts 不可用"是**用错版本**，在 DTK-26.04 下结论完全不同（见下）。

**一、在正确 DTK-26.04 下，FP8/BF8 硬件支持已实测跑通（决定性）**：
- **实测**：在 DTK-26.04-rc4 容器（BW1100/gfx938）内，用 HIP 封装函数 `internal::cast_to_f8<float,false>` + `internal::cast_from_f8<float,false>` 写 FP8/BF8 roundtrip kernel（`--offload-arch=gfx938`），**编译通过并成功在硬件上运行**：
  - `METRIC fp8_max_quant_error=0.500000`（FP8 E4M3）
  - `METRIC bf8_max_quant_error=1.000000`（BF8 E5M2）
  - `METRIC fp8_status=PASS`
  - → **gfx938 硬件实际执行了 FP8/BF8 转换**，量化误差在合理范围。
- 编译器 `BuiltinsAMDGPU.def`（`<dtk>/dcc/include/clang/Basic/`）定义了 gfx938 的完整 **FP8/BF8 Matrix Core intrinsic**（`__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8`、`_16x16x32_bf8_bf8`、`_bf8_fp8`、`__builtin_amdgcn_smfmac_f32_16x16x64_*` 等）→ 硬件有 FP8/BF8 MMAC（矩阵乘）能力。
- 佐证：`amd_hip_fp8.h` 宏判断将 gfx938 归入「OCP+FNUZ 两种 FP8 类型都启用」分支；hipblaslt v0.10 有 `hipblaslt_f8_fnuz`/`hipblaslt_bf8_fnuz` 类型；MIOpenDriver 支持 `conv[fp8|bfp8]`。

**二、⚠️ 唯一易踩坑：裸转换 intrinsic 需 fp8-insts，用 HIP 封装或库规避**：
- **裸的 `__builtin_amdgcn_cvt_pk_fp8_f32` / `cvt_f32_fp8`（FP8 标量转换指令）直接调用报 `needs target feature fp8-insts`**，且 `+fp8-insts` feature 在 gfx938 上**不被识别**（`'+fp8-insts' is not a recognized feature for this target`，DTK-25.04 与 26.04 一致）。
- **规避**：用 **HIP 封装函数**（`internal::cast_to_f8<float,is_fnuz>` / `internal::cast_from_f8<float,is_fnuz>`，`<hip/amd_detail/amd_hip_fp8_cast.h>`）或 **hipblaslt / MIOpen 库**，这些封装已处理 feature（实测 DTK-26.04 下 HIP 封装函数可编译可运行）。**不要手写裸 cvt intrinsic**。

**三、FP8 GEMM 的正确路径**（gfx938 上）：
- 优先用 **hipblaslt**（`#define ROCM_USE_FLOAT8 1` 后启用 fp8 类型）或 MIOpen 的 fp8 算子，让库处理 FP8 kernel。
- 若手写转换，用 HIP 封装的 `cast_to_f8`/`cast_from_f8`（见上），**不要**用裸 cvt intrinsic；矩阵乘用 `v_mfma_f32_32x32x16_fp8_fp8` 类 intrinsic。

**四、⚠️ 端到端 FP8 GEMM 实测（2026-09-16，<device-alias> BW1100，pytorch:2.7.1-dtk26.04 容器）**：
- **`torch._scaled_mm`（走 hipBLASLt 0.10）在 BW1100 上无 solution**：`could not find valid hipblaslt solution`（e4m3fnuz 与 e5m2fnuz 均失败，`HIPBLAS_STATUS_NOT_SUPPORTED`）。即**当前镜像栈里 FP8 推理不能走 torch._scaled_mm / hipBLASLt 路径**（与 gfx936/BW1000 行为一致，是库 solution 库未覆盖 gfx938 的 FP8 GEMM，非硬件不支持）。
- **FP8 标量转换指令级 OK**：`__hip_fp8_e4m3_fnuz` 封装类型在 gfx938 编译+运行通过（`gcnArchName=gfx938:sramecc+:xnack-`，cast roundtrip 正常）——硬件确实执行 FP8 转换，只是 GEMM 库层没接上。
- 容器内**无 aiter**（`import aiter` 失败）。要跑 FP8 GEMM 需换带 aiter 或更新 hipBLASLt 的镜像，或手写 MFMA kernel。

> 版本对比：宿主 <device-alias> 默认 `/opt/dtk` = **DTK-25.04.4**（不满足 BW 系列要求）；容器 `harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.7.1-kylin10sp3-dtk26.04-py3.10` 内 `/opt/dtk` = **DTK-26.04-rc4**（正确）。用 BW 系列必须进 DTK-26.04 容器。

### 15.3 瓶颈 → 方法 映射（先量化再改）

| 观察（来自 hipprof PMC / dccobjdump） | 根因 | 方法 |
|---|---|---|
| SQ_INSTS_MMOP=0 的 GEMM | MMAC 未被使用 | 改用 v_mmac / tl.dot |
| ISA 全是 `global_load_dword`（无 x4） | 未向量化 | 向量化全局访问（float4） |
| 高 LDS bank conflict + MMAC 错布局 | LDS 读取布局错 | 改用 `ds_read_m32x16_b16` |
| VGPR 高、occupancy 低 | 寄存器压力 | 降低 VGPR / `__launch_bounds__` |
| 串行化 load/compute | 无 double buffer | `num_stages=2` 双缓冲 |
| 跨步 HBM 访问、带宽低 | 非合并 | 合并访问 + 对齐 |
| 小 grid、CU 空闲 | 利用率不足 | persistent scheduler |
| 重复 HBM 读可复用数据 | 缺 LDS 暂存 | LDS tiling |

### 15.4 DCU 工具速查（profiling / ISA 验证）

```bash
# GPU 状态
hy-smi
rocm-smi

# Profiling（始终先跑这个）
hipprof --pmc --pmc-type 3 <app>                       # 指令级 PMC
hipprof --pmc-read  --pmc-type 3 <app>                 # 内存读剖析
hipprof --pmc-write --pmc-type 3 <app>                 # 内存写剖析

# ISA 验证（改完底层后必查编译产物）
hipcc -o kernel.co kernel.hip --offload-arch=gfx936 -c
dccobjdump --inputs=kernel.co --show-sass --show-instruction-encoding --separate-functions

# 寄存器压力
hipcc --resource-usage kernel.hip --offload-arch=gfx936
hipprof --codeobj-analyze <elf-or-so-file>

# SQTT（难诊断的 stall）
hipprof --sqtt --sqtt-type 1 --kernel-name <name> --output-type 0 -d <dir>/ <app>
```

### 15.5 ISA 自检清单（每次改完扫一眼 dccobjdump）

| 看什么 | 好信号 | 坏信号 |
|---|---|---|
| 向量化加载 | `global_load_dwordx4` | 只有 `global_load_dword` |
| GEMM 的 MMAC | `v_mmac_f32_16x16x16f16` | 只有 `v_fma_f32` / `v_mul_f32` |
| 打包 FP16 | `v_pk_fma_f16` / `v_pk_add_f16` | 只有 `v_fma_f16` |
| 正确 LDS 喂 | `ds_read_m32x16_b16` | `ds_read_b32` 喂 MMAC |
| 双缓冲 | load 与 compute 交织、`s_waitcnt` 在 tile 边界 | 每次小 compute 前 `s_waitcnt vmcnt(0)` |
| 异步 LDS | `buffer_load_dwordx4 ... lds` | load→VGPR→store 到 LDS |
| 低寄存器压力 | VGPR ≤ 64（num_warps=4 时 10 CTAs/CU） | VGPR ≥ 200（1 CTA/CU） |

> 进阶瓶颈→指令映射与优化目录见 `dcu-perf-profiling`（hipprof/rocprof 量化）与 `dcu-op-optimization`（改→验证闭环）。本 §15 只给 DCU 专属「速查锚点」。

---

## 十六、跨架构微优化速查（已下沉）

NVFP4 GEMM 手法→DCU 对应物速查表、跨 die L2 分区调度、基准测试纪律移至 [references/cross-arch-microopt.md](references/cross-arch-microopt.md)。

---

## 使用须知
- **DCC 编译器官方文档**（编译选项/-mllvm/GCVM IR/LingShu 后端 SGPR/VGPR/dccprof）：索引见 `dcu-triton-source/references/dcc-compiler-docs.md`（共用参考，2026-10-14 建档）。
- 经验/方法仅供参考，不保证修复。先核对设备型号(gfx)+驱动/DTK 版本+报错原文。
- 不存性能基线/逐行日志；数字以 `dcu-perf-cases` 模板或源文档留痕。
- 高危操作谨慎，优先测试机验证。
- 由 GitHub/内部文档 检索提炼并入的外来章节：§十（tensormux/kernel-skills）、§十一（auto-kernel-research）、§13.4（XDZS2025）在本文件；
  §九（tilelang 内部文档 矩阵）、§十二（KDA+BBuf agent flow）、§十六（cudaforfun NVFP4）已下沉 `references/` 同名文件。
  均只沉淀方法/约束，不搬运性能数字与逐行日志。
