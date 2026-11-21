---
name: dcu-inference-optimization
description: DCU通用模型推理优化方法论
version: 1.2.4
author: DCU Engineer
license: MIT
kind: tool
---

# DCU 通用模型推理优化（dcu-inference-optimization）

> 定位：**方法论 + 可复用流程**，面向计算机视觉、语音、时序预测等中小规模模型，也适用于大模型推理的 GEMM 寻优。核心思路（来自 内部文档 主文档 file_id=<internal-file-id>，作者 <internal-author>）：
>
> ```
> **第 0 步：先 profile 定位真瓶颈**（看时间真正花在哪，再决定动哪个算子，见 §十四·1）
> → Baseline 测试
> → torch.compile 图编译优化
> → NHWC / channels_last 数据布局优化
> → TF32 / FP16 等计算精度优化
> → MIOpen 卷积类算子寻优
> → rocBLAS GEMM 类算子选优
> → 性能测试与回归验证
> ```
>
> 分模型侧重：
> - CNN 类：NHWC、TF32、MIOpen 寻优；
> - Transformer/MLP 类：torch.compile、rocBLAS GEMM 选优、精度优化；
> - RNN/LSTM 类：GEMM 调用形态、小 kernel launch、persistent kernel 或自定义算子。
> - ⚠ 不同 DTK 版本、不同硬件，模型可能已用最优 miopen/rocblas，需**实测**确认提升有效性。
> - 大模型推理类也可用 rocBLAS GEMM 寻优。

---

## 一、torch.compile 优化

### 1.1 backend
| backend | 说明 | 适用 |
|---|---|---|
| `inductor`（默认） | TorchInductor 生成 Triton/后端 kernel，支持算子融合、图优化、减少 Python 调度与 kernel launch | 模型结构稳定、输入 shape 固定、可融合算子多 |
| `eager` | 不真编译，仅调试图捕获 | 排查是不是 Inductor 导致的问题 |
| `aot_eager` | AOTAutograd 拆前后向，仍 eager 执行 | 验证模型能否被图化 |

### 1.2 mode 与 fullgraph（推荐顺序，别一上来 max-autotune）
```
baseline
→ torch.compile default + fullgraph=False
→ torch.compile reduce-overhead + fullgraph=False   # 小模型/小batch/延迟敏感
→ torch.compile max-autotune + fullgraph=False       # 固定shape、性能冲刺
→ 尝试 fullgraph=True
```
- `reduce-overhead`：减少 Python 调度/框架开销，适合小 batch、小模型、频繁调用。
- `max-autotune`：更激进自动调优，但首次编译更久、ROCm/DCU 下更易触发 Triton/Inductor 编译错误或 fallback。
- `fullgraph=True`：要求整图捕获，遇不支持控制流/动态 shape 直接报错 → 退回 `False`。
- 初始模板：`torch.compile(model, backend="inductor", mode="default", fullgraph=False)`（放在 `model.eval()` 后）。

### 1.3 注意事项
- compile 并非对所有模型都加速；ROCm/DCU 下可能：首次编译久、算子 fallback、channels_last+max-autotune 触发编译错误、多线程并发不稳定、瓶颈在单个大算子时收益有限、动态控制流只能部分图化。
- 每次优化都做对照测试：baseline / compile default / reduce-overhead / max-autotune / +NHWC / +TF32 / +MIOpen Find / +rocBLAS 选优。

---

## 二、NHWC / channels_last 优化

- 原理：NHWC 让同一空间位置多 channel 在内存更连续，匹配现代 GPU/DCU 卷积 kernel 访存模式，利于 MFMA/Tensor Core/TF32。
- 用法（**模型和输入都要转**）：
```python
model = model.to("cuda:0").to(memory_format=torch.channels_last)
input_tensor = input_tensor.to("cuda:0").to(memory_format=torch.channels_last)
```
- 验证生效：
```python
print(input_tensor.is_contiguous(memory_format=torch.channels_last))  # True
```
- 叠加环境变量（后端选择建议，与上面代码配合）：
```bash
PYTORCH_MIOPEN_SUGGEST_NHWC=1
```
- 注意：非卷积模型收益小；部分算子不支持 channels_last 会产生额外 layout 转换；与 compile+max-autotune 组合可能触发兼容问题；**有些模型单独 NHWC 基本无效果，但叠加 MIOpen 寻优后效果显著提升**。

---

## 三、TF32 优化

- 原理：输入/权重/输出仍是 FP32，底层卷积或矩阵计算走 TF32 路径提速。
- DCU/ROCm/MIOpen 控制（FP32 卷积允许 TF32 中间计算）：
```bash
MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1
```
- 组合（NHWC+TF32+MIOpen Find）：
```bash
MIOPEN_FIND_MODE=1 \
PYTORCH_MIOPEN_SUGGEST_NHWC=1 \
MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1 \
python model_test.py
```
- 精度验证（TF32 有轻微误差，先对齐）：
```python
max_abs_error = (out_tf32 - out_fp32).abs().max()
mean_abs_error = (out_tf32 - out_fp32).abs().mean()
```
- 注意：不改变 Tensor dtype（仍显示 float32）；已是 FP16/INT8 时 TF32 不是主要优化点。

---

## 四、MIOpen 寻优（Conv/Norm/Pool 类）

- `MIOPEN_FIND_MODE=1`：完整算法搜索+benchmark，建 FindDb 缓存（首次慢，适合模型首部署/固定 shape）。
- `MIOPEN_FIND_MODE=3`：优先查 FindDb 缓存，命中即复用（后续正式测试用，启动快）。
- 推荐流程：先用 1 建缓存，再用 3 跑正式测试；结合 TF32 同理。
- **需重新寻优的变化**：batch size、输入分辨率、模型结构、NCHW/NHWC layout、FP32/TF32/FP16 策略、MIOpen/PyTorch/驱动版本、更换硬件型号。
- 与 torch.compile 关系：compile 后部分算子被 Inductor/Triton 接管，不再走 MIOpen 路径 → 测试时分别测「关 compile+ MIOpen」与「开 compile + MIOpen」比较。

---

## 五、BLAS GEMM 日志自动选优（rocBLAS Layer3/4 + hipBLASLt）

- 原理：同 GEMM 不同 size 性能受 M/N/K、transA/B、数据类型、compute type、lda/ldb、硬件、rocBLAS 内部 kernel 选择影响。默认选择不一定最优。方法：用 `ROCBLAS_LAYER=4` 采集**真实业务** GEMM 调用 → 自动选优工具针对这些真实 size 选更优 kernel → 写回 rocBLAS 配置目录，后续自动生效。
- Layer4 日志字段：M/N/K、transA/transB、a/b/c/d_type、compute_type、lda/ldb/ldc/ldd、call_count（出现次数，优先关注高频）。
- 操作流程：
```
运行业务模型 ROCBLAS_LAYER=4
→ 复制 GEMM 日志，存到 rocblas_layer4_size/xxx.log
→ 改 configs/tuning_args.txt (source=layer4, hardware_type=BW, gpu_ids=空闲卡, should_clean=0, update_target_config=1)
→ python Auto_tuning.py
→ 回业务模型重测
```
- 工具路径（示例，以实际 DTK 为准）：
```bash
cd /opt/dtk/lib/rocblas/auto_select_test/auto_select_tools   # 或 /opt/dtk-26.04-*/lib/rocblas/auto_select_test/auto_select_tools
ROCBLAS_LAYER=4 python resnet.py        # 采集
# 粘贴日志到 rocblas_layer4_size/resnet.log
vim configs/tuning_args.txt            # source=layer4; hardware_type=BW; update_target_config=1
python Auto_tuning.py                   # 生成并更新 /opt/dtk/lib/rocblas/library_gpuN
python resnet.py                        # 重测，自动读新配置
```
- tuning_args 关键项：`source=layer4`、`hardware_type=BW`（可选 ZF/ZF2/KM/KME/ECO/BW/NMZ/YY）、`gpu_ids=空闲DCU`、`should_clean=0`（更新前先备份）、`update_target_config=1`、`strategy=common`。
- 适用：GEMM/MatMul/Linear 占比高、shape/batch 固定、重复运行、默认 kernel 非最优、且不改模型代码。
- 注意：① 日志须来自真实业务输入（batch/resolution/seq 一致）；② gpu_ids 选空闲卡（benchmark 会被占用干扰）；③ **先备份原始 rocBLAS 配置目录**再 update；④ xxx.csv 空通常表示无该类型 GEMM，非错误；⑤ 端到端收益看瓶颈是否在 GEMM；⑥ shape 变化后重采重选；⑦ 与 MIOpen/compile 作用层级不同，分别测。

### 三者层级对照
| 优化方式 | 主要对象 | 说明 |
|---|---|---|
| MIOpen Find | Conv/Norm/Pool | 偏卷积类 |
| rocBLAS Layer4 选优 | GEMM/MatMul/Linear | 偏矩阵乘 |
| torch.compile | PyTorch 图级 | 融合/减调度，可能改底层调用路径 |

### 5.1 BW1000 实测：rocBLAS shape 调优的端到端收益与适用场景（2026-09-12）

> 环境：BW1000 2卡（gfx936/80CU/64GB/1000W）TP=2，Qwen3.8-27B（Qwen3.5 hybrid：GDN 线性注意力+每4层 full attention，hidden=5120，64层，dense），SGLang，无 EAGLE，`--flush-cache`+random-ids 排除 APC。数字为 `[案例历史数据，随硬件/DTK/版本变化]`，只作方法论示例。

**完整流程（可复用）**：
1. 带 `ROCBLAS_LAYER=3` 起服务跑真实业务 → 日志里 `rocblas-bench -f gemm_ex ... -m M -n N -k K` 就是实际 GEMM shape
2. 提取去重 shape 写 CSV（表头 `gemmtype,TransA,TransB,M,N,K,B`，bf16 用 `bf16`），放 `auto_select_tools/csv_size/csv/`
3. 改 `configs/tuning_args.txt`：`source=csv`、`gpu_ids=<空闲卡>`、`cu_count=0`（BW1000 80CU 无 80cu 配置，走默认无后缀那套）、`update_target_config=0`（不污染全局）
4. `python3 Auto_tuning.py`（工具自带 A/B：旧配置 vs 新配置各 2 轮）
5. 服务带 `ROCBLAS_TENSILE_LIBPATH=<ast>/optimization_configs/new/config/library_gpu5` 重启即生效

**实测结论**：
- 算子级：163 个 decode GEMM shape（M∈{5120,7168,8192,17408,124160}，N=1~60，K∈{3072,5120,8704,10240}）平均 **1.285x**，中位 1.118x，0 个下降
- 端到端单流 decode（batch=1）：**+25~31%，且随上下文长度（1k→256k）基本稳定**（256k 略降，attention 占比上升）
- 高并发 batch=64：**无收益**（attention+TP 通信占大头，GEMM 占比被稀释）
- **适用场景：低并发、长输出的 decode**（单用户对话、agent 推理、长文生成）；高并发吞吐场景不受益
- 配置是"超集"：调过的 shape 用最优 kernel，没调过的回退默认启发式（不亏不赚），放心部署
- 配置作用域：绑定 (硬件型号, DTK 版本, TP 切分, 模型)——TP 切分会改 GEMM 的 M/K 维度，**shape 必须在目标机上采集**，BW1000 调的不能给 BW150 用

**踩坑（实测）**：
1. **bench_serving 的 "Output token throughput" 含 prefill 墙钟时间**，长上下文冷启动 TTFT 会污染它（64k 不预热 TTFT 12.5s vs `--warmup-requests 3` 后 0.7s）→ decode 对比必须用 **TPOT 或单流 decode 吞吐（1000/TPOT）**，别用总吞吐
2. **EAGLE 投机解码有 run 间噪声**（accept length 波动，吞吐 ±15%）→ 受控 A/B 要关 EAGLE（去掉 `--speculative-*` 和 `SGLANG_ENABLE_SPEC_V2`）
3. **APC 缓存**：`--flush-cache` + random-ids（无共享前缀）排除
4. **`pkill -f 'sglang.launch_server'` 会自匹配杀掉自己的 ssh 会话**（命令行里含该字符串）→ 用 `pgrep -f 'sglang[.]launch_server'` 字符类技巧取 PID 再 kill
5. 256k 这种超长 prefill 基准很慢，用 `background=true` 后台跑写日志，别前台阻塞
6. 工具目录拷到临时目录再改（`cp -r auto_select_tools /tmp/ast`），别动 `/opt/dtk` 原工具；`update_target_config=0` 保证全局配置不被污染

### 5.2 rocBLAS Auto_tuning 官方教程补全（2026-10-16，飞书《rocblas自动化tuning工具及实战方案》）

> 与 §5.1 自采路线（layer4 混合日志 / 手工 CSV）互补的第三种采集口径 + 官方 tuning_args 全参数 + DSv4-FP8 实测数据点。

- **layer3 独立日志抓取**（bench 记录单独落文件，不与启动日志混写）：
  ```bash
  # 服务启动脚本副本的环境变量处加：
  export ROCBLAS_LAYER=3
  export ROCBLAS_LOG_BENCH_PATH=bench.log
  # 启动命令末尾追加 >&rocmblas.txt，启动日志与 rocBLAS 日志分两路 tail -f 监控
  ```
  抓完把 `bench.log` 拷到 `auto_select_tools/rocblas_layer3_size/`，`tuning_args.txt` 用 `source=layer3`。
- **抓取前置坑**：① 先关 MLA Host Staging Buffer（`SGLANG_DISAGG_HOST_STAGING_BUFFER` 及相关变量），否则与抓取过程冲突报错；② 开 cuda graph 时压测可能不向 bench.log 写日志，**属正常现象**，等压测执行完毕即可。
- **tuning_args.txt 全参数表**（官方完整口径，§5.1 只列关键项）：

  | 参数 | 说明 |
  |---|---|
  | `mode` | user / developer（普通用户仅支持 user） |
  | `source` | layer3 / layer4 / csv |
  | `hardware_type` | ZF / ZF2 / KM / KME / ECO / BW / NMZ / **YY**（按实际平台改） |
  | `gpu_ids` | 卡号，多卡空格分隔 |
  | `cu_count` | 按实际硬件 CU 数填（`rocminfo \| grep Com` 查看），匹配不同 cu 配置文件；0=走默认无后缀那套（见 §5.1） |
  | `strategy` | common / quick（user 模式仅 common） |
  | `chunk_size` | 每卡每轮选优 size 数（默认 10，size 少可调） |
  | `tune_warmup_times` / `tune_loop_times` | 选优预热/循环次数（默认 2/4） |
  | `bench_warmup_times` / `bench_loop_times` | 测试预热/循环次数（默认 2/10） |
  | `test_times` | 测试轮次（默认 2） |
  | `enable_bestkernel_csv` | 读 bestkernel CSV + MT 校验过滤 kernel（0 禁用默认 / 1 启用） |
  | `update_target_config` | 是否把生成配置写回目标包配置目录（0 不更新；删除/写回前先备份） |
  | `enable_mk_n_threshold` | 同一 MK 组合的 N 值数量 ≥30 时自动展开（0 禁用默认 / 1 启用） |

- **产物与生效**：寻优日志尾部出现生成完成即产出 `optimization_configs/new/config/library_gpu<N>`（N=卡号），服务启动脚本加 `export ROCBLAS_TENSILE_LIBPATH=<该路径>` 重启生效——与 §5.1 第 5 步同口径。
- **DSv4-0731-FP8 实测** `[案例历史数据，随硬件/DTK/版本变化]`：1P 1D、长上下文**单并发**，TPOT 14.67→7.99ms（30K/1K）、15.1→8.43（50K/1K）、15.54→8.41（70K/1K），吞吐 67.57→123.99 / 65.25→116.47 / 61.34→117.77 tok/s（**约 +78~84%**）；IFB 64K/4K、128K/4K 场景原文未填数（待补）。与 §5.1 Qwen3.8-27B BW1000（单流 decode +25~31%）互证：**低并发长输出 decode 收益最大**，DSv4-FP8 长上下文场景收益更显著。

- **抓 size / 调优相关环境变量必须在服务启动前注入**（2026-04-21 同事项目实录，2026-10-15 衍生整理版）：rocBLAS 侧抓 GEMM size 需先 `export ROCBLAS_LAYER=3` + `export TENSILE_DB=0x8000` 再起服务打流，原文步骤即「容器重启一下」再 export——已运行的服务进程无法补注入，操作等价 = 重启容器后重跑压测。
- **tuning 不必占用目标机**（同上来源）：GEMM size 日志可拷别人已抓好的，在自己环境 tuning；生成的 `library_gpu*` 配置拷回目标机 `export ROCBLAS_TENSILE_LIBPATH=<路径>` 生效。但原文同时警示跨环境配置「不一定能用，需要重新 tuning」——配置绑定（卡型、DTK 版本、模型），借用成品必须按 §5.1 口径端到端 A/B 验证。

- **rocBLAS 日志→CSV 整理脚本**（原始文档附件，2026-10-19 归档入库）：`python3 references/extract_rocblas_gemm_to_csv.py -i rocmblas.txt --tp 1`——识别 `rocblas-bench` 命令行（`--transposeA/B` + `-m/-n/-k`）与 `rocblas_gemm_ex`/`rocblas_gemm_strided_batched_ex` API 两种日志口径，输出 detail CSV（每次调用一行）与 summary CSV（同 M/K 组合的 N 聚合为区间、tp 列只首行显示、utf-8-sig 防 Excel 乱码），产物可作为 §5.2 Auto_tuning 的 `source=csv` 输入或人工核对。
- **hipBLASLt 调优日志→按 n 全覆盖映射表脚本**（衍生文档附件，2026-10-20 经用户补发入库）：`python3 references/hipblas_config_expand.py hipblas-cofig.txt -o hipblas-cofig-expanded-28col.txt`——把 Auto_tuning/bench 产出的 hipBLASLt 调优日志扩成按 n 全覆盖的 (m,k,n)→solution_index 映射表：默认删 8 个布局列（lda/stride_a/ldb/stride_b/ldc/stride_c/ldd/stride_d，实测均为 m,n,k 的机械推导量），再按 (m,k) 分组把 n 在 `--min-n`~`--max-n`（默认 256~2048，step 1）补齐；新行只改 n 与 solution_index，solution_index 按实测锚点推导（`--rule nearest/floor/ceil`，`--tie lower` 默认），其余列逐字节沿用来源行（故新行 Gflops/us 仍是来源 n 实测值）；先跑 `--dry-run --summary` 看统计与分段表、`--list-columns` 查列名；分组内非 n 差异列须并入 `--key` 或 `--allow-varying`，否则报错退出。

### 5.3 hipBLASLt GEMM size 统计与 tuning（飞书原始文档 2026-04-21《GLM5 hipblaslt rocblas size统计与tuning》，GLM-5-W8A8 ep8 项目实录；2026-10-15 为衍生整理版、补 §2.4 点对点平滑控制；原文注明步骤整理自 DTK《自动化选优脚本使用说明》《hipblaslt-bench使用说明》等参考文献，未本机复测）

> §5/§5.1/§5.2 是 **rocBLAS/Tensile** 链路（`ROCBLAS_LAYER` + `Auto_tuning.py` + `ROCBLAS_TENSILE_LIBPATH`）。若模型稠密层 GEMM 走 **hipBLASLt**（W8A8/int8 路径常见），用本节独立链路；同一模型两条链可并存，调优配置要一起注入（见④）。

- **① 抓真实 GEMM size**：服务启动前 `export HIPBLASLT_LOG_MASK=32`，启动命令末尾 `>&hipblas.txt` 重定向，再打真实流量或冒烟压测（例：`vllm bench serve --dataset-name random --num-prompts 1 --random-input-len 32 --random-output-len 1 --temperature 0.0 --ignore-eos`）。**抓取期间 API server 进程数必须设 1（vLLM 加 `--api-server-count 1`），否则日志无打印**（原文实测口径；抓完改回原值）。日志里每个 GEMM 是一条可复放的 `hipblaslt-bench` 命令：`--api_method c -m M -n N -k K --lda/--ldb/--ldc/--ldd --stride_a/b/c/d --alpha/--beta --transA/--transB --batch_count 1 --scaleA/--scaleB --a_type/--b_type/--c_type/--d_type --compute_type --activation_type …`；w8a8 dense 层实例形态 `a_type/b_type=i8_r, c_type=i32_r, d_type=bf16_r, compute_type=i32_r`。
- **② 去重生成 sample.sh**：日志中同 shape 大量重复，按参数指纹去重成每 shape 一条：`python3 references/extract_hipblaslt_dedup_from_log.py -i hipblas.txt -o sample.sh`（脚本已随本 skill 入库；先折叠空白再截取 `hipblaslt-bench` 到 `--activation_type` 的完整段，缺关键字段的行丢弃并打印候选/无效/去重后计数）。另入库 `references/extract_hipblaslt_gemm_to_csv.py`（同源脚本），按 TP 切分输出 detail/tuning 统计 CSV：`python3 references/extract_hipblaslt_gemm_to_csv.py -i hipblas.txt --tp 1`。
- **③ 跑 tuning 生成 config**：先 `chmod +x <DTK>/lib/hipblaslt/benchmark_tool/hipblaslt-bench` 与同目录 `hipblaslt_gentest.py`、把该目录加入 PATH（tar 拷包丢执行位，坑同 `dtk-blas-ab-benchmark`），再 `HIPBLASLT_TUNING_FILE=hipblaslt.config ./sample.sh`。产出格式：首行 `Git Version: <hash>`，之后**每条数据行前重复一遍表头**（格式如此，非脏数据；尾列 `…hipblaslt-Gflops,us,solution_index,gcnArchName,CUs`）。**行数核验**：`grep -o "xnack" hipblaslt.config | wc -l`（表头含 `xnack-`，计数=数据行数），少于 ② 去重条数即有 shape 未出结果。
- **④ 注入生效 + 验证**：`export HIPBLASLT_TUNING_OVERRIDE_FILE=<绝对路径>/hipblaslt.config` 后重启服务生效；加 `export HIPBLASLT_LOG_LEVEL=4`，日志出现 `problem_override_from_file Find solution with index: <n>` 即确认命中。模型同时走 rocBLAS 调优时，本变量与 `ROCBLAS_TENSILE_LIBPATH` 一并注入，只注入一半等于 A/B 错位。
- **★⑤ 复用性硬约束（本案例最有价值的一条）**：tuning config **仅支持点对点优化（原文注明已和 DTK 研发确认）**——只有与 tuning 时 shape 完全一致的调用才命中，且 **N 随输入长度变化**。`[案例历史数据]` 原文实录：客户数据集 300 条输入集中在 7K token（6800~7800 波动），仅取前 10 条做 tuning，后续数据推理**性能回退到初期最差水平**（现象记录，机制未解释）。缓解两手：**采样要铺开**——输入按长度 sort 后首尾交替取样重排，再取子集做 tuning，别按自然顺序取前 N；**N 扩区间**——config 可裁剪只留 `m,n,k,solution_index`（`stride_*`/`ld*` 可不保留），把 N 在波动区间内手动展开以覆盖大多数 size（原文用附件脚本 `hipblas_config_expand.py input.config --min-n 256 --max-n 4096 -o output.config`，附件在原文档内但 drive API 侧无下载权限，脚本本体暂未归档；用时从原文档附件区手动下载或向原作者索取）。
- **⑥ 端到端参考线（Decode TPOT，原始文档 §5 实测图表，2026-04-21 三组采样 14:18/15:17/15:52）**：该部署（GLM-5-W8A8 ep8+tp1+MTP，`vllm bench serve` random 数据集，context 约 1K–46K × bs=1~32）decode TPOT 绝对水平：**bs=1/2 ≈ 97–123 ms；bs=4 ≈ 122–163 ms；bs=8 ≈ 160–207 ms；bs=16 ≈ 267–545 ms；bs=32 ≈ 474–720 ms（一组仅测到约 16K 上下文）**。形态：bs≤8 时 TPOT 对 context 长度基本不敏感（组内起伏 <±15%），随 batch size 近线性上升；**bs=8→16 出现明显劣化拐点**，bs≥16 波动大（相邻采样点最高跳升约 1.5×）。仅作同部署配置的相对锚点使用（项目实录、未本机复测），任何调优收益结论仍按 §5.1 端到端口径自测。
- **口径边界**：与 §5.1 的 rocBLAS 超集「不亏不赚」**语义不同**——hipBLASLt override 逐点匹配，tuning 集采样窄了就只在采样点生效、其余 shape 走默认解，两条链路结论勿互抄。与 `dcu-llm-benchmark` 的「W8A8 tune-cache miss 亚秒级、别按多秒首 token 归因」也是两回事：那边是 TTFT 归因口径，本节是 decode GEMM 选解覆盖面。配置绑定（卡型、DTK 版本、TP、模型），端到端收益按 §5.1 口径自测（只认吞吐/TTFT/TPOT）。

---

## 六、通用优化流程（固定测试流程，便于复用）

1. **Baseline**：原始 eager，记录单次延迟、batch 吞吐、DCU 利用率、显存、输出正确性、环境版本（PyTorch/ROCm/DTK/MIOpen/rocBLAS）。
2. **torch.compile**：default → reduce-overhead → max-autotune；fullgraph False → True。
3. **NHWC**：代码转 channels_last + `PYTORCH_MIOPEN_SUGGEST_NHWC=1`。
4. **TF32**：`MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1`。
5. **MIOpen 寻优**：FIND_MODE=1 建缓存 → 3 正式测。
6. **rocBLAS Layer4 选优**：采集→Auto_tuning→重测。
7. **组合测试**（不一定线性叠加，逐项记录对比）：
```
baseline → NHWC → TF32 → MIOpen Find → torch.compile → rocBLAS 选优
→ NHWC+TF32+MIOpen → compile+NHWC → compile+rocBLAS → 最优组合回归
```

---

## 七、性能测试规范（必含）

- **必须 warmup**（首跑含初始化/编译/缓存，不计入）：
```python
with torch.no_grad():
    for _ in range(10):
        _ = model(input_tensor)
torch.cuda.synchronize()
```
- **正式计时前后都要同步**（DCU 异步执行）：
```python
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(iterations):
        _ = model(input_tensor)
torch.cuda.synchronize()
end = time.time()
```
- 吞吐/延迟：`latency = total_time/iterations`；`throughput = iterations*batch_size/total_time`；batch>1 区分 batch latency / sample latency / throughput。
- 推荐测试模板见下方代码块。

---

## 八、分模型类型优化建议

- **CNN**（VGG/ResNet/U-Net/YOLO/MobileNet/EfficientNet）：NHWC → TF32 → MIOpen Find → compile → batch 调优 → rocBLAS 选优（补充 Linear/1x1 Conv/隐式 GEMM）。
- **Transformer/MLP**（Linear/MatMul/Attention/LayerNorm/Softmax/GELU）：rocBLAS 选优 → compile → FP16/BF16/TF32 → batch/seq 调优 → attention kernel 优化。
- **RNN/LSTM**：确认 GEMM 调用形态 → rocBLAS 选优 → 合并 gate 计算 → batched GEMM → 减少 timestep 小 kernel launch → persistent kernel/自定义算子 → compile 辅助。固定 hidden/batch 可进一步考虑 persistent kernel 或自定义 MMAC/MFMA kernel。

---

## 九、常见问题排查

1. **torch.compile 报错**：关 compile 确认 eager 正常 → backend="eager" 测图捕获 → mode 从 default 起 → fullgraph=False → 去掉 NHWC 再测 → 最后组合。
2. **NHWC 无收益**：非卷积密集 / 模型或输入没同时转 / 中间算子不支持致频繁 layout 转换 / batch 太小 / 后端没选 NHWC kernel。
3. **TF32 精度不一致**：对比 FP32 baseline 误差；检查业务指标；敏感模型关 TF32，性能敏感保留。
4. **MIOpen 首跑慢**：FIND_MODE=1 建缓存，后续用 3。
5. **rocBLAS 选优后端到端无提升**：瓶颈不在 GEMM / 日志 shape 与正式不一致 / gpu_ids 卡被占 / 默认已近最优 / compile 后不走原 rocblas 路径 / GEMM 调用少。排查：对比前后端到端、查 Layer4 call_count 看高频、对比 merged_results_*.csv 新旧、确认加载更新后配置、shape 变则重采。

---

## 十、实测记录（定性结论 + 溯源）

> 原始基线总表来自 内部文档《移动无线所小模型优化记录表（bw150/bw10）》(file_id=<internal-file-id>，作者 <internal-author>)。按 skill 内容纪律，**具体基线数字不入本方法论 skill**（如需引用转 `dcu-perf-cases` 或溯源上述 内部文档）；此处只保留跨表可迁移的规律性结论。

**规律性结论（跨表可迁移）**：CNN（vgg/resnet）靠 compile+NHWC+TF32+MIOpen 提升最显著（100%+）；LSTM 须 persistent/重写底层才显著；Transformer 延迟类靠 migraphx/ort 后端或 gemm 融合；部分大模型（视觉/手机助手）开箱已近最优，单点优化无提升——印证「组合优化未必线性叠加、以端到端为准」。

---

## 十一、推荐测试模板（scripts 同目录）

```python
import torch, time
model.eval()
# warmup
with torch.no_grad():
    for _ in range(10):
        _ = model(input_tensor)
torch.cuda.synchronize()
iterations = 100
start = time.time()
with torch.no_grad():
    for _ in range(iterations):
        _ = model(input_tensor)
torch.cuda.synchronize()
end = time.time()
total_time = end - start
latency = total_time / iterations
throughput = iterations * batch_size / total_time
sample_latency = latency / batch_size
print(f"Batch latency: {latency*1000:.3f} ms")
print(f"Sample latency: {sample_latency*1000:.3f} ms")
print(f"Throughput: {throughput:.2f} samples/s")
```

---

## 十二、大模型长上下文 / 多产品线系统性调优方法论（2026-08-24 Qwen3.8-27B 三产品线研究提炼）

> 适用：大模型（尤其 hybrid attention / 长上下文 / 多 TP 产品线）从「跑通」到「系统性特化」的阶段。本节只沉淀**方法论**，具体数字/案例见 `dcu-perf-cases` 案例 15。

### 12.1 优化方法论总纲（correctness floor → matched A/B → specialization → fail-closed）
1. **先建 correctness floor**：任何性能优化前，先有可复现的质量基线（arithmetic/critical case/long retrieval/repeated SHA），否则「变快」可能是「变错」（如 kernel 漏写）。
2. **matched A/B 找根因**：对比两组除被测项外逐字一致（同卡、同镜像、同启动参数、同负载），性能判定必须 matched，不能跨 run 拼点。
3. **按多维 specialization**：phase × shape × K-family × layer × TP-rank geometry × KV length，而非全局一刀切。
4. **所有激进路径 fail-closed**：未审计 shape/geometry 一律 fallback 到 correct parent，不为代码统一硬开。
5. **offline benchmark 发现 crossover 后固化 dispatch**：不凭直觉写阈值，用数据找 crossover 点再固化。

### 12.2 Unified Adaptive 路由（多 shape 收敛为可维护执行策略）
- **RouteKey 要带全 feature vector**：至少含 phase / q_len / kv_len / cached_prefix_len / new_tokens / batch_size / TP geometry / rowparallel_Klocal / transformer_layer / speculative_state / draft_block / cuda_graph_shape。**不要只看 context_length**——一个「128K 请求」可能是 cold（q≈128K, kv≈128K），也可能是 Agent continuation（cached_prefix=120K, q=8K, kv≈128K），两者最佳 kernel 很可能完全不同。
- **offline autotune 流程**：candidate generation → isolated correctness → full-model exact semantic → quality gate → 3× matched performance → variance/contamination filter → winner + runner-up crossover analysis → merge adjacent regions → frozen dispatch table → runtime fail-closed selection（unknown shape → fallback）。
- **候选 shape 不要只扫正式点**：short/mid 要密扫（512/1K/2K/3K/4K/6K/8K/10K/12K/16K/24K/32K），long 回归（32K…257.9K），**必须加 prefix-hit**（(prefix,new)=(32K/64K/120K/240K, 1/4/8K)），否则 offline router 只学会 cold benchmark 而非 Agent workload。
- **CUDA Graph 管理限制成少量 graph family**（short/medium/long/fallback），不要为每个长度 capture；prefill graph 单独 correctness promotion（见 `dcu-sglang-source` DFlash2 段）。

### 12.3 reject 的是 (算法, parent_stack, shape) 组合，不是算法名
- 一个算法在某 parent stack + 某 shape 被 reject，**不代表永久拉黑**——parent stack 演进（新 long-KV/tail/DFlash2/verifier 体系成熟）后，旧 reject 的算法回移可能重新赢过旧路径。
- 工程教训：reject 决策要记录 (算法, parent_stack, shape) 三元组，parent 变化时重新评估，不要把算法名写进永久黑名单。

### 12.4 hybrid attention 长 prefill 的 O(N)+O(N²) 建模
- 含线性 attention（GDN/Mamba 类）+ full attention 的 hybrid 模型，长 prefill TTFT 呈 **O(N)（线性层）+ O(N²)（full attention 逐 chunk long-KV 累积）** 曲率。
- 先对 dense TTFT 做二次拟合（`TTFT ≈ a + b·K + c·K²`，K=context/1024），看 R² 是否高、是否与模型结构（线性层数 vs full 层数）一致，再判断「cliff」是真实 kernel 阈值还是二次曲率的视觉放大。
- 避免把「64K cliff」这类视觉拐点误判为 kernel 阈值去硬改 dispatch——先建模确认是平滑曲率还是真实断点。

### 12.5 TP + 动态 per-token 量化的 shared-scale 语义（正确性坑，vLLM 同样适用）
- W8A8/FP8 的 activation scale 若是 **dynamic per-token**，TP K-sharding 后各 rank 的 local scale ≠ global scale（数学不等价，是量化定义被 TP 改写而非噪声）。
- 修复：跨 rank 取 global max（TP-MAX）恢复单卡语义，并**逐维收窄共享范围**（K family / phase / layer）而非全局付费；路由解析失败要 fail-closed 报错而非猜。
- 详见 `dcu-perf-cases` 案例 15 根因 1。

## 十三、框架级优化与选型入口（大模型 / 通用模型 / 科学计算 / 智驾）

> 沉淀"某框架/某场景下可用的优化手段与框架选型"，不存性能数字。具体收益以本环境实测为准。

### 13.1 SGLang 大模型推理融合优化入口（BW1000 类）
- 可逐项开启的融合/通信优化（按 profile 占比决定投入，别全开）：
  - **SiLU-Mul-Quant 融合**、**RMSNorm-Quant 融合**：MoE/FFN 路径的 elementwise+quant 融合，降 kernel launch 与中间张量往返。
  - **custom all-reduce**：TP 通信换自定义 all-reduce，降通信开销。
- ⚠ **冲突**：RMSNorm-Quant 融合与 HIP Graph custom all-reduce **不可同时开**（logits 异常/乱码），二选一或都关（见 `dcu-ontrack-experience` B1）。
- 先 profile 看 elementwise/通信占比再决定开哪些（占比低则融合收益≈0，见 `dcu-sglang-source` §三·补2 融合 ROI 判据）。

### 13.2 通用模型 / 科学计算软件移植优化（Triton 算子 + 流程）
- **Triton 算子编译**：客户自研软件（物探/科学计算类）里性能瓶颈算子用 Triton 重写/编译，常能拿到可观单卡提升（某物探软件实战）。
- **模型处理流程优化**：算子级之外，审视数据搬运/预处理/后处理流程（减少 host-device 往返、合并小操作），与算子优化叠加。
- 适用：非 LLM 的自研数值软件 DCU 移植，先 profile 定位瓶颈算子再决定 Triton 重写还是流程优化。

### 13.3 智驾模型适配优化组合拳
- 智驾模型（BEV Fusion / ALOCC / BEVDet 等）在 DCU 上从"跑通"到"达标"的通用手段组合：
  - **算子优化**（瓶颈算子 Triton/HIP 重写或调优）
  - **依赖库升级**（DTK/框架/算子库版本对齐，消除兼容回退）
  - **共享内存提效**（kernel 内 shared memory 使用优化，降 HBM 往返）
- 多模型共性优化点可沉淀为通用手段，跨模型复用。

### 13.4 通用模型推理框架选型
- 不同框架对常见模型的性能/精度表现不同，选型按模型类型 + 实测：
  - **TritonServer + ONNXRuntime**：CV/通用模型（YOLO 类）推理吞吐优化入口。
  - **TorchServe + PyTorch**：BERT 类 NLP 模型推理优化入口。
  - 选型前先做框架性能对比（PyTorch / ORT / MIGraphX），确定不同框架对目标模型的提升比，再定部署框架（为类似项目提供参考）。

### 13.5 GEMM 性能的两个隐性变量（hyqual vs 业务实测差距排查）
- 客户用 cuda sample 的 gemm 测性能与 hyqual/业务实测差距大时，先查两个隐性变量：
  - **GEMM 形状（M/N/K）**：sample 用的形状与业务真实形状不同，性能不可比。
  - **数据类型（小数尾数位长度）**：同一精度类别下不同 dtype（如 fp16 vs bf16 vs 不同尾数位）对 GEMM 性能影响显著。
- 排查顺序：先对齐形状，再对齐 dtype，最后才怀疑硬件/驱动。

## 十四、优化方法论补充（外部借鉴，未本机实测）

> 本节提炼自外部案例《调优 VAE 解码器：单 chunk 从 133.8ms 压到 60.2ms》（reactor.inc / 公众号转载，
> 优化的是 NVIDIA Blackwell 上 Wan 2.1 VAE 解码器）。**仅提炼去平台化的通用方法论，不含 B200/cuDNN 专属数值**，
> 具体数字/实现已在原平台验证，**本机（DCU）尚未实测**；下述手法在 DCU 上是否成立、收益多少需按 §六 端到端口径重新实测确认。
> 编号与 §十四 内部一致，其中「①先 profile 定位」已并入开头核心思路的「第 0 步」。

### 14.1 先 profile 定位真瓶颈，再决定动哪个算子（对应核心思路第 0 步）
- 原案例：不 profile 就优化，人容易先顺手量化线性层，结果是「算子优化 8%、端到端只快 1%」；
  真正的瓶颈在解码器（占单 chunk 最大份额且在关键路径上），这才是该动刀的地方。
- 手法：按 GPU kernel 时间归类找占比最大的阶段，而不是凭直觉选算子。
- 呼应：与 §13.1「先 profile 看占比再决定开哪些融合」一致，但本节强调这是**任何优化开始前的第 0 步**，
  不局限于 SGLang 融合，也适用于算子/权重/精度层面的取舍。

### 14.2 统一内存布局，消除框架与加速库间的「静默转换」（完善 §二）
- 原案例：PyTorch 默认 channels-first（NCDHW），加速库卷积内核跑 channels-last（NDHWC），两边不一致时
  加速库会**静默转换**，且一次转换还打散 elementwise 融合。统一成 channels-last 后杀掉 312 次转换、单 chunk 大降。
- **隐性成本要点（§二 未强调的角度）**：layout 不一致引起的转换**不报错**、只在 kernel 里悄悄吃时间，
  容易漏看 → 这正是 §二 NHWC 之外需要警惕的点。DCU 上对应 hipdnn/MIGraphX 的 layout 行为与 cuDNN 不同，需实测确认。
- 保留 §二 已有动作：模型和输入都要转 `channels_last`、`PYTORCH_MIOPEN_SUGGEST_NHWC=1`、验证 `is_contiguous`。

### 14.3 一次处理多单元，摊薄内核启动（新增）
- 原案例：缓存热了之后，一次调用解多个隐帧与分多次调用是**同一批计算**，只是 kernel 更少更大（更少启动开销、更大 kernel 更利于利用率）。
- 代价：显存峰值上升（原案例约 +8GB）。取舍见 §14.4。
- 手法：对「每次调用处理一个单元」的路径，评估能否合并成一次更大调用，用 kernel 启动次数换计算规模。

### 14.4 资源权衡视角：显存换算力，瓶颈在算力不在显存（新增）
- 原案例：接受多占约 8GB 显存换取启动开销下降，因为瓶颈在算力、显存通常有富余。
- 手法：优化取舍要基于「瓶颈在哪」的资源视角判断，**不是无脑最小化资源**。若瓶颈在算力/延迟，适度多用显存换性能是划算的；
  若显存本身吃紧（DCU 显存富余度通常不如此案例的 B200），则需重新权衡。是否成立要实测。

### 14.5 区分「精度损失」与「被否决的真实原因」（完善 §三）
- 原案例：被否决的 2D-tap 版本**不是因更不准**（fp32 各方案都在 0.1dB 内），而是因**慢**被否决。
- 手法：评价一个备选方案时，把「精度差异」和「被否决的真实原因（慢 / 复杂 / 不可用）」分开判别，
  不要因表面精度差就把「慢」误判成「不准」。精度验证动作沿用 §三（max_abs_error / mean_abs_error）与 §12.1（correctness floor）。

### 14.6 优化成果的标尺：对比强者基线，而非自己上次版本（新增）
- 原案例：优化成果不是只跟自己原版比，而是对比业界同类优化栈（vLLM-Omni / SGLang 自带的 fast path）。
- 手法：评估优化价值时，用「市场/同类引擎的强者基线」做标尺，避免「自我进步」的假象。
  与 §12.1 的 matched A/B（对照组逐字一致）不同——A/B 保证严谨性，本条决定**跟谁比**才是有效的标尺。

## 交叉引用

- 案例沉淀（LSTM/UNet/GNN 四段式）→ `dcu-perf-cases` §1-§3（B 文档源 tbYA3th1orMhJ4SCfe4X1xf4MdUvKV1P8）。
- 算子级优化闭环/留痕 → `dcu-op-optimization`；度量 → `dcu-perf-profiling`。
- persistent kernel / 自定义算子（LSTM）→ `dcu-kernel-dev`。
- 环境/版本对齐 → `dcu-env`。

## 使用须知
- 经验/方法仅供参考，不保证修复。先核对设备型号(gfx)+驱动/DTK 版本+报错原文。
- 性能数字以源文档/实测为准；本页总表为作者公开汇总，仅供方法佐证。
- 高危操作谨慎，优先测试机验证。
