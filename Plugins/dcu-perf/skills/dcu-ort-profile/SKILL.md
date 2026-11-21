---
name: dcu-ort-profile
description: "ORT在DCU上的Profile机制解析+MOE显存雪崩HIP流修复"
version: 1.0.0
author: DCU Engineer（整理自<internal-author> 内部文档 文档）
license: MIT
kind: tool
---

# ORT 1.19.2 Profile 机制解析与 DCU/ROCm MOE 显存优化

本 skill 整合 4 篇内部文档，覆盖 ONNX Runtime（ORT）在 DCU/ROCm 上的性能剖析与 MOE 模型显存优化实战。

## 来源文档（建议溯源）

| 文档 | 内部文档 链接 | file_id |
|---|---|---|
| ORT 1.19.2 Profile 机制解析和实践（主文档：Profile 机制 + 自定义 HIP 流优化） | https://www.<internal-doc-link>/l/<内部文档_token> | <internal-file-id> |
| ORT 在 DCU 上推理 MOE 模型显存异常问题的分析和解决（根因 + 环境变量缓解 + 句柄分析） | https://www.<internal-doc-link>/l/<内部文档_token> | <internal-file-id> |
| ort-moe 优化多线程性能对比测试（多线程多 Session 并发验证） | https://www.<internal-doc-link>/l/<内部文档_token> | <internal-file-id> |
| ORT 上的算子耗时统计工具（Profile JSON → 算子耗时排序 CSV） | https://www.<internal-doc-link>/l/<内部文档_token> | <internal-file-id> |

> 关键实测结论（来自上述文档）：自定义单 HIP 流优化后，MOE 推理整体耗时从 3.45s 降至 0.99s（↓71.4%），If 算子 ↓98.7%；DCU 显存占用从「持续增长爆显存」降至稳定 2.5G（单 session，64G 卡上仅 4%），并发数提升约 4 倍。
> **性能数字剥离纪律**：上述结论是**文档实测的具体数字**，属「基线/性能数据」，**不入本方法论 skill**；如需引用数字，转 `dcu-perf-cases` 项目留痕。本节只陈述「自定义单 HIP 流修复显存雪崩」的**方法路径**。

---

## 1. ORT 1.19.2 原生 Profile 机制

### 1.1 核心定义与设计目标
ORT 原生 Profile 是内置于引擎的轻量级性能追踪体系，无需第三方工具即可在推理时采集全流程性能数据。核心目标：统计算子执行耗时、追踪内存分配/释放开销、记录执行器调度延迟、区分主机（CPU）与设备（GPU/DCU）端耗时、生成标准化可解析报告。

1.19.2 相对旧版的三项优化：
- 降低采集开销（整体推理耗时增加 < 3%）
- 优化 JSON 日志格式可读性与兼容性
- 兼容精细化采集配置；支持 Python/C++ 多语言 API，Windows/Linux/macOS 全平台

### 1.2 核心配置项（均为官方标准有效参数）
| 配置项 | 作用 | 正确用法 |
|---|---|---|
| `enable_profiling` | 布尔，开关 Profile | `SessionOptions().enable_profiling = True`；支持会话内动态开关，无需重启会话 |
| `end_profiling()` | 结束采集并导出日志 | 无入参，自动在当前目录生成带时间戳的 `onnxruntime_profile_*.json`，返回日志完整路径 |

### 1.3 架构与流程（会话级开启-埋点采集-异步落盘-统一解析）
1. **会话配置开启**：`sess_options.enable_profiling = True`，设置基础采集模式
2. **运行时埋点采集**：模型图优化 / 算子调度 / 内存分配 / 设备交互 / 推理执行 / 结果回传全链路插入埋点，记录每个节点的开始/结束时间、耗时、线程 ID、执行设备
3. **数据异步缓存**：先存入内存缓存，避免频繁 IO 阻塞推理（1.19.2 优化了缓存策略）
4. **会话结束输出**：调用 `end_profiling()` 导出标准 JSON 日志，可用 Chrome Tracing / Perfetto UI 可视化

### 1.4 Profile 日志格式（Chrome Tracing 兼容 JSON）
| 字段 | 含义 |
|---|---|
| `cat` | 事件分类：Node（模型节点）/ Kernel / Memory / Session |
| `name` | 事件名称（算子名 / 内核名 / 操作名） |
| `ts` / `te` | 开始 / 结束时间戳（微秒） |
| `dur` | 事件总耗时（微秒） |
| `pid` / `tid` | 进程 ID / 线程 ID |
| `args` | 扩展参数：执行设备、输入尺寸、内存占用等 |

解析工具：Chrome `chrome://tracing/` 导入 JSON；或本 skill 的 `scripts/analyze_ort_profile.py`（按算子降序统计耗时 + 落盘 CSV）。

---

## 2. ORT-ROCM-EP Profile 专项（1.19.2 核心突破）

### 2.1 背景与价值
1.19.2 之前，ROCm/DCU 下 Profile 仅能追踪主机端 CPU 调度耗时，无法采集 GPU/DCU 端内核执行、数据拷贝、设备同步耗时，且需手动对接第三方 ROCm 工具、数据无法与 ORT 原生日志融合。

1.19.2 官方特性：**原生集成 ROCm 官方 roctracer**，无需 rocprofiler，实现 GPU/DCU 端内核级 + 数据传输级耗时采集，与 ORT 原生 Profile 无缝融合，生成一份含 CPU+GPU/DCU 的全链路统一 Chrome Tracing 日志，并优化流调度逻辑缓解显存异常增长。

### 2.2 ROCm 专属日志分类
| 分类 | 含义 | 瓶颈信号 |
|---|---|---|
| `ROCm_Kernel` | DCU 端算子内核执行事件，`dur` 为内核纯计算耗时 | 占比过高 → 优化算子精度（如 FP16）或替换高效内核 |
| `ROCm_Memcpy` | GPU/DCU 数据拷贝，区分 H2D / D2H / D2D | 耗时过高 → 启用固定内存、减少频繁拷贝 |
| `ROCm_Sync` | GPU/DCU 同步等待事件 | 耗时过高 → 调整流调度策略（改用自定义单流）避免阻塞与显存异常 |

### 2.3 环境与编译要求（强制）
- **ROCm 版本**：必须为 ROCm 5.6 / 5.7（ORT 1.19.2 官方兼容列表限定）；低于 5.6 无完整 roctracer 支持 → GPU/DCU 端日志无内核耗时
- **核心依赖库**：必须安装 `roctracer`（Debian 系 `roctracer-dev` / RPM 系 `roctracer-devel`）。`rocprofiler-dev` 非必需，1.19.2 ROCm Profile 仅依赖 roctracer
- **ORT 版本**：必须使用 1.19.2 官方预编译包（含对应 ROCm 版本）或源码编译开启对应参数；旧版未实现 ROCm Profile 与原生 Profile 融合，且流调度逻辑存在缺陷
- **环境变量**：`LD_LIBRARY_PATH` 必须含 `$ROCM_HOME/lib`，否则运行时无法加载 roctracer

源码编译开启 ROCm Profile 的专属参数（仅依赖 roctracer）：
```bash
./build.sh \
    --config Release \
    --use_rocm \
    --rocm_home=/opt/rocm \
    --enable_rocm_profiling \   # 1.19.2 新增，开启 ROCm 专属 Profile
    --build_shared_lib \
    --parallel
```

---

## 3. DCU + ORT 推理 MOE 模型显存异常根因

### 3.1 问题现象
1.7G 的 MOE 轻量模型，首次推理无法运行，DCU 显存连续、无节制增长，短时间内占满全部可用显存，推理进程崩溃。

### 3.2 根因链（rocBLAS handle 硬绑定）
1. MOE 有多个独立 expert（Gate→Expert0..N 无数据依赖），ORT 识别为并行子图
2. ORT 从内部流池为每个并行分支分配独立 HIP stream
3. ROCm EP 对每个 stream 创建 `RocmContext`，并为每个 context 创建 **1 个 rocBLAS handle**
4. **硬绑定铁律**：`1 stream = 1 rocBLAS handle`，rocBLAS 不允许多流共用同一 handle（源码：`PerThreadContext` 中 `rocblas_create_handle` + `rocblas_set_stream`）
5. 结果：ORT 流越多 → rocBLAS handle 越多 → 显存雪崩

> 实测：某 MOE 场景创建了 **388 个 rocBLAS handle**（与 `rocblas_destroy_handle` 数量一致），按 rocBLAS handle 显存占用推算至少占 40G+ 显存，而模型本身仅 1.7G。

### 3.3 智算早期提供的临时环境变量缓解（不再推荐，自定义流方案更彻底）
| 环境变量 | 作用 | 来源 |
|---|---|---|
| `export PADDING_MALLOC=0` | 降低显存开销（进展1：降至约 70%+ 50G，可单 session 运行） | 匹配进展1 |
| `export HIP_REUSE_MODULE=1` | 缩短首次预热（进展2：49s → 3s） | 匹配进展2 |

> 注：这两项是「治标」绕行，且对具体项目方不可接受。下方 3.4 的自定义 HIP 流方案实施后可不再需要它们。

### 3.4 根因定位手段（辅助排查）
- **ORT 日志**：`sess_options.log_severity_level = 0`（VERBOSE），展示计算图处理过程
- **MIOpen 日志**：`export MIOPEN_ENABLE_LOGGING=1` / `export MIOPEN_LOG_LEVEL=5`，观察每算子绑定哪个 HIP 流（`SetStream stream: 0x...`），确认多流；`MIOPEN_COMPILE_PARALLEL_LEVEL=4` 间接影响内存
- **rocBLAS 句柄日志**：`export ROCBLAS_LAYER=3`，统计 `rocblas_create_handle` 个数（应与 handle 泄漏量正相关）

---

## 4. 自定义 HIP 流修复方案（核心，1.19.2 推荐）

### 4.1 思路
在 ROCm 底层机制不调整前，于 ORT 框架层规避：通过 ctypes 手动创建「自定义单 HIP 流」，以 `user_compute_stream`（**十进制字符串**）传给 `ROCMExecutionProvider`，强制全程单流串行，消除多流→多 handle→显存泄漏根因；同时关闭 ORT 多线程并行，规范流生命周期（推理结束主动 `hipStreamDestroy`）。

### 4.2 关键约束（ORT 1.19.2 专属坑）
- `user_compute_stream` 必须是**十进制字符串**（`str(stream_ptr.value)`），ORT 1.19.2 **不支持十六进制**
- 默认单流场景下，1.19.2 会残留 **1 个多余的初始化 handle**（ORT 1.20+ 已修复：设置 `user_compute_stream` 时跳过默认 context 创建，仅 1 个 handle）。该残留 handle 最终也切到同一自定义流，**无害、无额外显存开销**
- `libhiprtc.so` 路径默认 `/opt/dtk/lib/libhiprtc.so`，按实际 DTK 安装路径调整
- `Devices incompatible` 是 HIP 底层设备兼容性警告，确认 device_id 正确即可，不影响推理与 Profile 采集

### 4.3 完整可用脚本
见 `scripts/ort_custom_stream_infer.py`（含 `create_hip_stream` / `destroy_hip_stream` 与单流推理 + Profile 导出）。核心片段：
```python
import onnxruntime as ort, ctypes

def create_hip_stream(hip_lib_path="/opt/dtk/lib/libhiprtc.so"):
    hip_lib = ctypes.CDLL(hip_lib_path)
    hip_lib.hipStreamCreate.restype = ctypes.c_int
    hip_lib.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    stream_ptr = ctypes.c_void_p()
    hip_lib.hipStreamCreate(ctypes.byref(stream_ptr))
    return stream_ptr, hip_lib

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

stream_ptr, hip_lib = create_hip_stream()
rocm_provider_options = {"device_id": "0", "user_compute_stream": str(stream_ptr.value)}
session = ort.InferenceSession("encoder.onnx", sess_options=sess_options,
    providers=[("ROCMExecutionProvider", rocm_provider_options), "CPUExecutionProvider"])
# ... run ...
session.end_profiling()           # 导出 JSON
destroy_hip_stream(stream_ptr, hip_lib)  # 释放，防泄漏
```

### 4.4 优化前后核心对比（定性结论，encoder.onnx + MOE）

自定义单 HIP 流优化后：单次推理耗时**降约 71%**（默认流 → 自定义流），算子数不变（算子逻辑一致），显存从**持续增长至爆显存 → 稳定约 2.5G 无泄漏**，MOE 显存雪崩彻底解决。

> **数字剥离纪律**：具体实测数字属性能基线，不入本方法论 skill；如需引用转 `dcu-perf-cases` 或溯源 内部文档（file_id `<internal-file-id>`）。本节只证明「自定义单 HIP 流优化」方法有效。

---

## 5. 算子耗时统计工具（Profile JSON → 排序 CSV）

`scripts/analyze_ort_profile.py` 解析 `onnxruntime_profile_*.json`（首尾含 `[]`），按 `算子名|原始Provider` 为唯一键分别统计调用次数 / 累计 dur / 单次平均 dur，终端降序打印 + 落盘 `ort_op_complete_analysis.csv`。

特性：
- 缺失 provider 默认填充 `CPUExecutionProvider`（格式化后 CPU）
- 同算子不同 EP（ROCM / CPU）分开统计
- 仅处理 `cat == "Node"` 的日志行

用法：
```bash
python3 scripts/analyze_ort_profile.py onnxruntime_profile_2026-03-03_*.json
# 不传参默认读 ort_profile.log
```

---

## 6. 多线程多 Session 并发验证

`scripts/ort_multithread_bench.py` 对比默认流 vs 自定义流在高并发下的单次推理延时与显存占用。`create_ort_session` 自动为每个 ROCm Session 绑定自定义 HIP 流，1 Session 对应 1 独立锁 + 1 流，测完统一销毁。

用法：
```bash
python3 scripts/ort_multithread_bench.py --session_num 8 --thread_num 64 --loop_times 10
# 可选 --provider / --enable_lock / --hip_lib_path
```

实测并发结论（定性）：自定义 HIP 流方案下，**并发线数提升约 4 倍**（K100-AI 16→64 线、BW150 32→128 线，均为 8 session），预热时间减少 50%+，显存占用降至约 36%（64G 卡）；而优化前默认流**无法跑 2+ session（爆显存）**。单次推理耗时优化前后基本一致（偏差 10% 内视为性能一致）。

> 具体并发/耗时/显存数字属性能基线，不入本方法论 skill；如需引用转 `dcu-perf-cases` 或溯源 内部文档（file_id `<internal-file-id>`）。

---

## 7. Pitfalls（必看）

1. **`user_compute_stream` 必须十进制**：ORT 1.19.2 不支持十六进制流指针，务必 `str(stream_ptr.value)`
2. **lib 路径**：`/opt/dtk/lib/libhiprtc.so` 按实际 DTK 安装路径调整，否则 `CDLL` 抛异常
3. **残留 handle 无害**：1.19.2 单流下会有 2 个 rocBLAS handle（1 个自定义流 + 1 个初始化残留），均绑同一流，不影响性能/显存，1.20+ 才修成 1 个
4. **必须显式销毁流**：`destroy_hip_stream` 放在 `finally` 或统一清理，避免显存残留累积
5. **环境变量绕行非根治**：`PADDING_MALLOC=0` / `HIP_REUSE_MODULE=1` 只是智算早期缓解，自定义流方案实施后无需它们
6. **roctracer 依赖**：ROCm-EP Profile 强依赖 `roctracer` + `LD_LIBRARY_PATH` 含 `$ROCM_HOME/lib`，否则日志无 GPU/DCU 内核耗时
7. **Profile JSON 首尾 `[]`**：解析前需剥掉外层方括号再逐行 `json.loads`（脚本已处理）
8. **`Devices incompatible` 警告**：HIP 设备兼容性提示，device_id 正确即可忽略，不影响推理与采集

## 8. 验证步骤（落地自检）

- [ ] `python3 -m py_compile scripts/*.py` 全部通过（已验证）
- [ ] `bash -n` 不适用（纯 Python）；脚本无全角/不可见空白（已校验 `\u3000/\u00a0/\u202f` 均为 none）
- [ ] ORT 版本为 1.19.2 + ROCm 5.6/5.7；`LD_LIBRARY_PATH` 含 `$ROCM_HOME/lib`
- [ ] 运行 `ort_custom_stream_infer.py`：能加载模型、导出 `onnxruntime_profile_*.json`、流销毁无报错
- [ ] 运行 `analyze_ort_profile.py <json>`：终端打印降序算子表 + 生成 CSV
- [ ] 对比优化前后：MOE 显存稳定 ≤ 2.5G，总耗时 ↓ 70%+，If 算子 ↓ 98%+

## 9. 脚本索引（scripts/）

| 脚本 | 用途 |
|---|---|
| `analyze_ort_profile.py` | Profile JSON → 算子耗时降序表 + CSV |
| `ort_custom_stream_infer.py` | 自定义 HIP 单流推理 + Profile 导出（显存修复核心） |
| `ort_multithread_bench.py` | 多线程多 Session 并发对比测试（含自定义流绑定） |

> 注：以上脚本在「用户 DCU 环境 + ORT 1.19.2 + DTK/ROCm」下运行；本机无 onnxruntime/torch 依赖，仅做语法与空白校验。
