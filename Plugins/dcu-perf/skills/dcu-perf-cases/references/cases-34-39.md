# 性能优化案例 §34-§39（kernel/系统级）

> 2026-10-16 自 `perf/dcu-perf-cases/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），案例四段式原文照录、未改一字；编号与原文一致，索引见主 SKILL.md 案例索引表。

## 34. 高并发 KERNEL VMFault（paged attention 越界）

> 场景：8 卡 TP=8 部署 Qwen3.5-35B-A3B，高并发压测时 GPU 崩溃
> 模型：Qwen3.5-35B-A3B（MoE），max-model-len 260000，block-size 256
> 硬件：海光 DCU ×8（BW1000 拓扑绑核）
> 软件：vLLM 0.15 海光定制镜像 + DTK 26.04

## 现象

evalscope 高并发压测（parallel 200-400）时服务崩溃：

```
Invalid address access: 0x7ef578e20000, Error code: 1.
KERNEL VMFault !!!! <<<<<<
PID: 234970, SIGNAL: 0 !!!! <<<<<<
```

复现特征：
- **batch 超过 200 很容易复现**（命中 full cuda graph 的大 batch）；
- PIECEWISE 模式下累积运行也会出现；
- FULL_DECODE_ONLY 模式也会报错；
- 降 gpu-memory-utilization 到 0.8 不能解决，仍然崩溃。

## 定位

**第一步：从 HSA queue dump 定位到 kernel。** 崩溃时 DTK 打印 HOSTQUEUE 分析，dump 出出错的 AQL packet 和 kernel 名：

```
kernel name:
_Z22paged_attention_kernelIttLb0ELi256ELi64ELi256ELi4EEvPT_S1_PKS0_PKT0_S6_iiPKiS8_iPKfiiSA_SA_iiS3_ib
```

→ 问题出在 **paged attention kernel**（KV cache 按 block 寻址的访存越界）。

**第二步：用 rocm-debug-agent 精确定位（DCU 上 VMFault 排查首选工具）。**

```bash
# 编译带调试信息（-O1，-O3 会导致行号偏移）
dcc -g -O1 your_kernel.cpp -o your_program
# 运行
ROCM_DEBUG_AGENT_OPTIONS="--all --save-code-objects=/tmp/co --output=/tmp/dump.txt" \
HSA_TOOLS_LIB=/opt/dtk-26.04/lib/librocm-debug-agent.so.2 ./your_program
```

agent 通过 HSA_TOOLS_LIB 注入进程，异常时转储所有 wavefront 的 PC、寄存器、LDS、反汇编 + 源码行号。定位四步：

1. 看 `Faulting page` → 与合法 buffer 地址范围比对，判断越界方向/幅度；
2. 找 `reason: MEMORY_VIOLATION` 的 wavefront → 看 PC 和反汇编定位到指令；
3. 看该 wavefront 的 v0/v1（实际访问地址）和 exec mask（是否边界 wavefront——越界几乎总发生在边界 wavefront）；
4. 有源码行号直接修；没有就 `llvm-objdump -d` 反汇编 code object，结合 kernel 名回查源码。

错误代码速查：1=读访问违规，2=非法指令，3=写访问违规，4=对齐错误。

**第三步：变量控制缩小范围。** 逐一试了以下组合：

| 实验 | 结果 |
|------|------|
| 限制 cuda graph 捕获大小（最大 180） | 仍崩溃 |
| 仅 PIECEWISE 模式 | 崩溃 |
| 去掉 piecewise（纯 full） | 崩溃 |
| gpu util 0.8 | 崩溃 |
| 限制并发数量 | 可缓解 |

结合 vLLM 框架层面 block table / KV cache 的排查（参考上游 PR #37728 及海光 vllm 对应 commit），确认是 **full cuda graph 在大 batch 下 paged attention 的 block table 索引越界**。

## 方案

1. **cuda graph mode 改为 PIECEWISE**（`--compilation-config '{"cudagraph_mode": "PIECEWISE"}'`）：piecewise 切图后 paged attention 走独立分段，不再触发 full graph 下的越界路径；
2. **限制 `--max-num-seqs`**：把运行 batch 压在安全范围内（如 180/256），配合 `cudagraph_capture_sizes` 显式列出捕获档位。

```bash
vllm serve <model_dir>/Qwen3.5-35B-A3B --served-model-name qwen3.5 \
  -tp 8 --dtype bfloat16 --gpu-memory-utilization 0.9 \
  --max-model-len 260000 --max-num-seqs 256 \
  --compilation-config '{ "cudagraph_mode": "PIECEWISE",
    "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128,144,168,180,196,256] }' \
  --trust-remote-code --block-size 256 --reasoning-parser qwen3
```

## 效果

- PIECEWISE + max-num-seqs 限制后，parallel 200-400 压测不再出现 KERNEL VMFault，服务稳定。
- 长稳运行（累积运行）验证通过。

## 可复用经验

1. **KERNEL VMFault 第一手信息在 HSA queue dump 里**：DTK 崩溃时打印的 `DUMP AQL PACKET` + `kernel name` 能直接告诉你哪个 kernel 出事，比任何日志都快。
2. **rocm-debug-agent 是 DCU 上 VMFault 排查的 99% 首选**：一行命令拿到源码行号 + 寄存器 + 汇编；vLLM 这类长驻服务推荐 `--output` 到文件 + `--save-code-objects`，避免 stderr 被日志框架吞掉；服务 hang 住不 crash 时 `kill -s SIGQUIT <pid>` 可手动触发 wavefront 转储。
3. **Triton JIT kernel 没有 C++ 源码行号**，但反汇编 + 寄存器值仍能定位到"哪个 kernel、哪条访存指令、什么地址"，再结合 kernel 名回查 Triton 源码。
4. **cuda graph 模式是 VMFault 的高频变量**：FULL / PIECEWISE / FULL_DECODE_ONLY 走不同的切图路径，同一 bug 可能只在某一种模式下触发；排查时把 cudagraph_mode 作为必测变量。
5. **越界几乎总发生在边界 wavefront**：exec mask 只有部分 lane 活跃时，优先怀疑 grid 尺寸与 buffer 大小不匹配（block table 行数 < 实际请求数）。
6. 常见 VMFault 根因四类：数组越界（KV cache / block_table 索引）、空指针、主机/设备指针混用、访问已释放内存。

## 关联文档

- [vllm_compilation_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_compilation_guide.md) — CUDA Graph、FULL/PIECEWISE 切图
- [vllm_kvcache_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kvcache_guide.md) — PagedAttention、block table
- [vllm_attention_backend_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_attention_backend_guide.md) — paged attention kernel

## 35. 编译 FlashAttention 解决 FA VMFault

> 场景：推理/训练过程中频繁遇到 KERNEL VMFault，定位发现多数是 FlashAttention（FA）的 bug 引起
> 硬件：BW1000（gfx938/gfx936/gfx928）
> 软件：DTK 26.04 容器 + AICC 编译器 + FA aicc-master-dev 分支

## 现象

推理/训练（vLLM 压测、llama_factory 微调、ms-swift 全参训练等）中反复出现 `KERNEL VMFault`，提交 DPS 后响应慢。历史案例（<internal-ticket> / <internal-ticket> / <internal-ticket>）的最终解决方式**都是更新 FA 或 triton 等包**。

## 定位

- 在 CSD 中搜索 vmfault，确认多数 VMFault 与 FA 版本相关；
- OpenDas 开源的 FA 版本落后于内网版本，遇到问题时**优先从内网拉取分支，按自己的环境编译**。

## 方案

**环境准备（BW1000 容器）：**

1. 启动包含 DTK 的镜像：`<internal_registry>/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk26.04-py3.11`
2. 拉取源码（注意都用 aicc 版本，master 已年久失修）：
   - FA：`flash-attention-cutlass` 的 `aicc-master-dev` 分支
   - cutlass：`cutlass_3.2.1` 的 `feature/16x64-mmac-aicc` 分支
3. 安装 AICC 编译器：下载后执行 `bash dtk_llvm.run`

**编译：**

```bash
# cutlass 解压后放到 flashattention 的 csrc 中
export FLASH_ATTN_OPT=1
python setup.py sdist bdist_wheel
# 产物在 dist/ 下，如 flash_attn-2.8.3+das.opt1.dtk2604-cp311-cp311-linux_x86_64.whl
```

编译耗时接近一小时，建议放 tmux 中执行。

**替换 FA：**

```bash
pip3 uninstall flash_attn
pip3 install flash_attn-2.8.3+das.opt1.dtk2604-cp311-cp311-linux_x86_64.whl
```

**编译问题速查：**

| 报错 | 原因 | 解决 |
|------|------|------|
| `aicc: not found` | 没装 AICC | 执行环境准备第 3 步 |
| 提示没有 `FLASH_ATTN_OPT` | 环境变量未设 | `export FLASH_ATTN_OPT=1` |
| `cannot find ROCm device library for gfx92a` | 当前环境没有 K100/K100_AI 相关库 | 修改 setup.py 去掉 gfx92a（324 行 `extra_options_raw`、989 行 `--offload-arch`、1142 行 `extra_options_raw` 三处） |

## 效果

- 用自编译的 aicc 版 FA 替换后，FA 引起的 VMFault 消失，推理/训练恢复正常。
- 形成标准处置流程：**VMFault → 查 CSD 确认是否 FA 相关 → 内网拉 aicc 分支自编译 → 替换 FA 包**。

## 可复用经验

1. **VMFault 先查包版本再查代码**：DCU 上 FA/triton 版本问题引发的 VMFault 占比很高，升级/重编译 FA 是成本最低的修复路径。
2. **aicc 分支 ≠ master 分支**：FA 和 cutlass 都要用 aicc 版本，master 已年久失修，直接编译 master 会踩坑。
3. **编译目标架构要与环境匹配**：环境里没有 K100 库时，setup.py 里的 `gfx92a` 目标必须去掉，否则 clang++ 报 device library 缺失。
4. **编译产物带版本标记**（`+das.opt1.dtk2604`），替换前确认与容器 DTK 版本一致。

## 关联文档

- [vllm_attention_backend_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_attention_backend_guide.md) — FlashAttention 后端
- [vllm_compilation_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_compilation_guide.md) — 算子编译与融合

## 36. 视觉链路 SDPA 累加器精度（TRITON_ATTN）

> 场景：交付某金融机构的 DCU 上做火焰图检测验收（Qwen3.5-35B-A3B 识别图中是否有火焰，仅输出"有"/"无"）
> 模型：Qwen3.5-35B-A3B（多模态，vision encoder + MoE LLM）
> 硬件：海光 K100 DCU ×2（测试覆盖 2-8 卡，64GB/卡）
> 软件：vLLM 海光 DCU 分支（0.15 → 0.18 两个版本先后出问题）

## 现象

同一测试集（80 张火焰图，20 并发），两个版本先后出现精度问题：

| 版本 | 现象 | 数据 |
|------|------|------|
| vLLM 0.15 | 火焰图识别乱码、多输出字 | 原 prompt 有效率仅 52.5%（42/80 含"有"/"无"） |
| vLLM 0.18 | 视觉处理严重异常，正确率下降 80% | 更新 prompt 后有效率 100%，但正确率从 97.5% 掉到 25-30%（几乎全部识别为"无"） |

关键对照实验（`huggingface_inference.py` 对比测试）：

| 方案 | 视觉精度 | 注意力后端 | 输出 | 状态 |
|------|---------|-----------|------|------|
| HuggingFace bf16 | bfloat16 | PyTorch SDPA | 识别错误 | ❌ |
| HuggingFace fp32 | float32 | PyTorch SDPA | 正确识别 | ✅ |
| vLLM 默认 | bfloat16 | SDPA | 识别错误 | ❌ |
| 方案三（推荐） | bfloat16 | TRITON_ATTN | 正确识别 | ✅ |

**文本推理始终正常，只有视觉链路异常** → 问题在视觉编码器，不在 LLM。

## 定位

**阶段一（vLLM 0.15）：prompt 问题。** 原 prompt"检测图片中是否存在火焰…存在则输出'有'"导致模型输出"是"或其他内容；改为"检测图片中是否有火焰。规则：仅输出一个中文汉字，有则输出'有'，没有则输出'无'，禁止输出任何其他字符"后，有效率 52.5% → 100%，正确率 97.5%。该问题在 0.18 不再出现。

**阶段二（vLLM 0.18）：视觉注意力累加器精度不足。** 调整 shm 配置、禁用 tp embedding 多模态数据等均无效，排除 embedding 数据分片通信问题。对照实验锁定根因：

| 原因 | 说明 | 影响权重 |
|------|------|---------|
| 注意力累加器精度不足 | SDPA 在 bfloat16 下累加器使用 bfloat16，softmax 数值精度不足 | ~80% |
| 视觉模块敏感 | ViT 类编码器对数值精度敏感，低精度导致特征提取偏差 | ~15% |
| 累积误差 | 视觉编码器层数多，低精度误差逐层放大 | ~5% |

为什么文本正常、视觉异常：文本链路只经过 LLM（bf16 足够）；图像要经过 vision encoder 的注意力计算，SDPA bf16 累加器精度不足导致特征偏差。TRITON_ATTN 的 Triton kernel 内部累加器（`m_i`/`l_i`/`acc`）固定使用 float32，解决了这个问题。

## 方案

三个方案演进，推荐方案三：

| 方案 | 配置 | 显存开销 | 精度 | 状态 |
|------|------|---------|------|------|
| 一：自定义 SDPA 算子 | 视觉 bf16 + 自定义 SDPA | profiler 需多卡 | 正常 | ⚠️ 不推荐 |
| 二：float32 + TRITON_ATTN | `--mm-encoder-dtype float32` + TRITON_ATTN | 额外 ~0.5 GB | 最高 | ✅ 可用 |
| 三：bfloat16 + TRITON_ATTN（推荐） | 仅指定 TRITON_ATTN 后端 | 无额外开销 | 高 | ✅ 推荐 |

```bash
# 方案三（推荐）：无需改代码，现有镜像即可
vllm serve <model_dir>/Qwen3.5-35B-A3B \
  --port 8000 --dtype bfloat16 \
  --mm-encoder-attn-backend TRITON_ATTN \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 2 --max-model-len 32000 \
  --gpu-memory-utilization 0.9 --trust-remote-code
```

方案二额外加 `--mm-encoder-dtype float32`（视觉模块权重转 float32，需改代码提供新镜像；实测视觉权重 +0.48GB，但转换触发 gc 碎片整理释放 ~0.7GB，单卡总显存反而 44.43 → 44.18 GB）。

## 效果

| 测试项 | HuggingFace float32（基准） | 方案二/三 | 结果 |
|--------|---------------------------|----------|------|
| 文本理解 | 正常 | 正常 | 一致 |
| 图像描述 | 正常 | 正常 | 一致 |
| 火焰识别准确率 | 100%（40/40） | 100%（40/40） | 一致 |

性能（2 卡、64 并发、39 请求）：成功率 100%；TTFT 平均 1.333s（方案三）vs 1.465s（方案二），+9.9% 在可接受范围；全局吞吐 16.62 vs 16.41 images/sec，基本持平。

## 可复用经验

1. **多模态精度问题先做"文本 vs 视觉"对照**：文本正常 + 视觉异常 → 直接锁定 vision encoder，不用查 LLM 链路。
2. **精度问题用对照实验定位，不要猜**：HF bf16/fp32 × vLLM 默认/各后端，2×2 矩阵一跑就知道是 dtype 问题还是后端问题。
3. **注意力后端的累加器精度是隐藏变量**：SDPA 累加器跟随输入 dtype（bf16），TRITON_ATTN 内部固定 float32 累加——ViT 这类对精度敏感的编码器在 bf16 下可能系统性偏差，且表现为"识别全错"而非"偶发错"。
4. **prompt 约束要写死输出格式**："仅输出一个中文汉字，有则输出'有'…禁止输出任何其他字符"比"存在则输出'有'"有效率高得多（52.5% → 100%）。
5. **换镜像/升级版本后必须回归精度**：0.15 → 0.18 升级引入了新的视觉精度问题，版本升级 ≠ 质量提升。
6. **显存账要算细**：视觉模块转 fp32 只 +0.48GB，且 gc 碎片整理可能让总显存反降——精度方案的显存代价往往比想象中小。

## 关联文档

- [vllm_attention_backend_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_attention_backend_guide.md) — SDPA / TRITON_ATTN 后端、累加器精度
- [vllm_model_runner_v2_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_model_runner_v2_guide.md) — 多模态处理链路
- [vllm_production_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_production_guide.md) — 精度验收方法

## 37. DS-V4-Flash PD 分离 vs IFB 选型

> 场景：DeepSeek-V4-Flash（FP8 / INT8 两版）在 BW1100 上的长上下文推理部署，P/D 分离 vs 单实例 IFB 的选型
> 模型：DeepSeek-V4-Flash-FP8-Channel（w8a8）/ INT8-Channel（slimquant_marlin）
> 硬件：BW1100 ×2 节点（每节点 8 卡），IB 10 口（mlx5_1-4, 7-10）
> 软件：SGLang 0.5.12 海光定制版 + DTK 2604 + DeepEP + EAGLE/MTP

## 现象 / 目标

长上下文（4K-262K 输入）推理需要同时满足 TTFT 和 TPOT 两类 SLO。单实例（IFB，prefill+decode 同节点）与 P/D 分离（1P1D 两节点）两种部署形态，需要实测数据支撑选型。

## 方案

**三种部署形态：**

| 形态 | 说明 |
|------|------|
| 统一 IFB | 单节点 8 卡同时跑 prefill + decode（TP8/DP8/EP8，DP Attention 开，EAGLE 2/1/3，KV FP8，radix cache 开，cuda graph bs≤64） |
| P 节点（P-like） | `--disaggregation-mode prefill`，TP8 + **NSA prefill CP8**（round-robin-split），DeepEP normal，关 MTP/radix cache/cuda graph，KV BF16，static 专家布局 + 64 冗余专家 |
| D 节点（D-like） | `--disaggregation-mode decode`，TP8/DP8 + DP Attention + DP LM head，DeepEP auto，EAGLE/MTP 2/1/3，cuda graph bs≤64/128，KV BF16 |

router 做 PD 路由：

```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://<P_IP>:30000 --decode http://<D_IP>:30000 \
  --host 0.0.0.0 --port 10015
```

关键环境变量（P/D 通用）：`SGLANG_USE_LIGHTOP=1`（fused topk/moe_align/moe_sum）、`SGLANG_USE_FP8_W8A8_MOE=1`（FP8 版；INT8 版必须 =0）、`SGLANG_USE_DEEPGEMM_MOE=1`、`SGLANG_ROCM_USE_AITER_MOE=1`、`NCCL_MIN/MAX_NCHANNELS=16`、`ROCSHMEM_HEAP_SIZE≈3GB`、`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`。PD 分离还需替换 topo 及 rocblas/hipblaslt 库（`LD_LIBRARY_PATH` 指向自装目录 + `HIPBLASLT_TUNING_OVERRIDE_FILE`）。

## 效果

**1. IFB vs 1P1D（按节点归一化，out=1024）：**

| SLO | in | 胜者 | 收益 |
|-----|-----|------|------|
| TPOT≤25 | 512 / 1024 | IFB | +268.8% / +333.9% |
| TPOT≤25 | 4096 | PD | +60.8% |
| TPOT≤25 | 16384 / 32768 / 65536 | PD | +317.6% / +468.1% / **+1362.4%** |
| TTFT≤1000 & TPOT≤30 | 4096 / 16384 / 32768 | **仅 PD 有可行点** | ∞ |
| TTFT≤5000 & TPOT≤50 | 512 / 1024 / 4096 | IFB | +124.5% / +107.9% / +9.7% |

**2. P-like（CP8）长上下文 prefill 提升（单并发，IFB 模拟 PD 对比）：**

| 输入 | 吞吐比 | TTFT 比 |
|------|--------|---------|
| 512 | 0.56×（更慢） | 1.83× |
| 4K | 2.10× | 0.48× |
| 16K | 5.27× | 0.19× |
| 32K | 6.77× | 0.15× |
| 64K | 7.20× | 0.14× |
| 131K | 8.20× | 0.12× |
| 262K | 7.15× | 0.14× |

明确的长度分界：**512-1K 时 CP8 通信/调度开销占主导更慢；4K 起占优；16K-262K prefill 吞吐提升 5-8 倍、TTFT 降到原来的 12%-19%**。131K 以上在"平均 TTFT≤40s"阈值下，合并 PD 无合格点，P-like 仍可维持并发（新增可用区间）。

**3. D-like（DP8 + MTP）decode 侧：** 短输入与合并 PD 基本持平（±6%），131K/176K/196K 长输入分别 +9.4% / +20.9% / +28.0%。

**4. 推荐选型规则：**

| 业务特征 | 推荐 | 原因 |
|---------|------|------|
| 512-1K 短输入，TTFT 允许 2-10s | IFB | 单节点生成吞吐显著更高，少用 1 台节点 |
| 4K 输入，TTFT 必须 1-2s | P/D | IFB 无可行点或吞吐很低 |
| 4K 输入，TTFT 可放宽 5-10s | IFB 倾向 | 单位节点吞吐略高 |
| 16K 输入 | P/D 倾向 | 多数 SLO 下更优（宽松 TTFT 下优势仅 ~8%） |
| 32K-64K 且有明确 TTFT SLO | P/D | 多次出现只有 P/D 能满足 SLO |

## 可复用经验

1. **PD 分离不是无条件更优**：短输入 + 宽松 TTFT 时 IFB 单节点效率更高（省一台机器）；PD 的价值在长上下文 TTFT 可行性和 TPOT 约束下的吞吐上限。选型必须按"输入长度分布 × SLO"实测，不能拍脑袋。
2. **CP（context parallel）是长 prefill 的杠杆**：DP8 → DP1+CP8 把单条长序列的 prefill 分散到 8 卡，输入越长收益越大（16K 起 5-8 倍）；但短输入下 CP 通信开销反超收益，存在明确的长度分界点。
3. **P/D 可以分别扩容和调优**：1P1D 只是起点，生产上可按输入长度分布、到达率、P/D 队列压力选 1P2D / 2P1D 等配比再端到端验证。
4. **P 节点和 D 节点的最优配置完全不同**：P 侧关 MTP/radix cache/cuda graph、开 CP8、DeepEP normal、static 专家布局 + 冗余专家；D 侧开 DP Attention + MTP、DeepEP auto、cuda graph。混用配置会互相拖累。
5. **量化版本决定算子开关**：INT8（slimquant_marlin）版必须 `SGLANG_USE_FP8_W8A8_MOE=0` 并关 aiter moe，与 FP8 版的环境变量集不同，部署脚本要按量化版本分开维护。
6. **PD 分离有隐性依赖**：topo 文件、rocblas/hipblaslt 自装库、ep_config、专家分布文件（`--init-expert-location`）缺一不可，迁移环境时最容易漏。

## 关联文档

- [vllm_kv_transfer_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_kv_transfer_guide.md) — P-D 分离、KV 传输、KVConnector
- [vllm_parallel_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_parallel_guide.md) — TP/DP/CP/EP 并行策略、DeepEP
- [vllm_spec_decode_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_spec_decode_guide.md) — EAGLE/MTP 投机解码
- [vllm_quantization_guide](dcu-vllm-source:references/upstream-vllm-026/vllm_quantization_guide.md) — FP8 / INT8 量化部署

## 38. BW150/BW100 HBM 实测峰值与 QKNorm+RoPE naive kernel 带宽曲线（L2 thrashing 悬崖验证）

> 场景：验证 NVIDIA 侧 bandwidth-bound kernel 优化文章（QKNorm+RoPE：TMA prefetch + L2 分层 + occupancy，H100 NVL 实测）能否迁移 DCU
> 硬件：<device-A> = BW150 ×8（gfx936, 80 CU, L2 8MB, HBM 理论 1843 GB/s）；<device-B> = BW100 ×8（gfx936, 64 CU, L2 8MB, HBM 理论 1843 GB/s）
> 软件：DTK 26.04 容器（<device-A>: vllm-0151 / <device-B>: qwen38-official），hipcc --offload-arch=gfx936 -O3，HIP_VISIBLE_DEVICES=0
> 日期：2026-09-28

## 现象 / 目标

H100 上 naive QKNorm+RoPE 存在"数据量越大带宽越差"的 L2 thrashing 悬崖（~3400→~2500 GB/s，-26%），文章靠 cache 分层救回。需确认 BW 系列（L2 仅 8MB）是否存在同样悬崖、naive 实现是否有优化 headroom。

## 方案

1. HBM 峰值基准：纯 copy（read+write）kernel，256/512/1024 MB × block 128/256/512/1024，best-of-10 hipEvent 计时。
2. QKNorm+RoPE naive kernel：直接 global load/store、无 cache hint，float4 向量化，256 线程/block；Q/K read-once（N×16 heads×128 dim bf16 等价 float4），cos_sin 每 token 复用，weight 全局复用；N = 512…61440 tokens，best-of-10。
3. 正确性门禁：全量回读对比参考值，max_diff 判据。

## 效果

**1. HBM 实测峰值（BW150，copy read+write）：**

| size | block=128 | block=256 | block=512 | block=1024 |
|------|-----------|-----------|-----------|------------|
| 256MB | 1132.8 | 1351.9 | **1359.6** | 1357.9 |
| 512MB | 1138.8 | 1347.6 | **1361.3** | 1357.4 |
| 1024MB | 1141.8 | 1343.7 | 1360.6 | **1361.1** |

单位 GB/s。峰值 ~1361（理论 1843 的 74%），跨 size/block 稳定；block=128 明显低（~1140）。

**2. QKNorm+RoPE naive kernel 带宽曲线（无悬崖）：**

| tokens | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 30720 | 61440 |
|--------|-----|------|------|------|------|-------|-------|-------|
| BW150 GB/s | 740.7 | 983.1 | 1144.3 | 1246.6 | 1316.7 | 1350.7 | 1361.9 | **1366.8** |
| BW100 GB/s | 643.7 | 886.4 | 1092.4 | 1225.4 | 1322.7 | 1351.7 | 1357.3 | **1369.1** |

正确性：两台全部 max_diff=0.0000。

**3. 结论**：BW150/BW100 上 naive 实现带宽单调上升后平台化在 ~1360 GB/s（= HBM copy 峰值），8k→60k 不衰减，**无 L2 thrashing 悬崖**；naive 已打满 HBM，NVIDIA 侧 cache-hint 优化（TMA prefetch/streaming store/evict-first）在此无 headroom。根因：L2 仅 8MB（H100 为 50MB），数据从一开始就不 fit L2，无"fit→overflow"相变。

## 可复用经验

1. **评估 NVIDIA kernel 优化文章迁移 DCU，先查 L2 大小**（`hipDeviceProp_t.l2CacheSize`）：依赖"数据量跨 L2 容量"相变的优化（prefetch 预热/evict-first/streaming store）只在 L2 大到工作集能从小于 L2 长到大于 L2 的卡上有收益；BW 系列 8MB L2 上无效。
2. **"打满带宽"的结论必须先测纯 copy 峰值**（多 size × 多 block 扫描）再对比 kernel 实测，否则无法判断 headroom 是否存在。
3. **block=128 在 BW150 上带宽显著低**（~1140 vs ~1360 GB/s），bandwidth-bound kernel 建议 block≥256。
4. TMA 指令 DCU 无对应硬件；PTX cache hint 无直接等价物（AMD 侧为 buffer 指令 + descriptor cache-control 位，本 DTK 无 nontemporal_store builtin）。

## 关联文档

- [bandwidth_bound_l2_thrashing](dcu-kernelwiki:wiki/bandwidth_bound_l2_thrashing.md) — 方法+结论（本案例为性能留痕）
- [dcu-kernel-dev](dcu-kernel-dev) §15.2/§15.3 — bandwidth-bound kernel 优化轴与瓶颈映射

## 39. Qwen3.8-27B GDN prefill kernel 优化（vLLM fla/ops，QK 共享去冗余）

> 场景：Qwen3.5/3.8 系列 GDN hybrid 模型（vLLM `fla/ops` 路径）在 DCU 上跑 vLLM，prefill TTFT 的 kernel 级优化
> 模型：Qwen3.8-27B（model_type=qwen3_5，64 层 = 48 层 linear_attention(GDN) + 16 层 full_attention，75% 层走 GDN 路径）
> 硬件：<device-B> 节点 4×DCU TP4
> 软件：vLLM 0.18.1 + hcu 插件，DTK 26.04
> 日期：2026-10-08

## 现象 / 目标

GDN 层形状（优化 gate 精确匹配）：`H=48`（value heads）、`Hg=16`（query heads）、`K=128`、`V=128`、`BT=64`（chunk size）、bf16。prefill TTFT 随输入长度单调放大，需 kernel 级优化。

## 根因（kernel 级）

**核心洞察**：`H/Hg = 3`，每个 query head 被 3 个 value head 共享。原版 `chunk_fwd_kernel_o` 的 grid 是 `(V/BV, NT, B*H)`——每个 value head 各算一遍 `b_A = Q@K`，但 3 个 value head 共享同一份 Q/K，**QK 矩阵被冗余算 3 次、K 被冗余加载 3 次**。

## 优化动作（3 个，按收益排序）

1. **chunk_o.py 新增 share_qk3 kernel（prefill 大头）**：保留原 `chunk_fwd_kernel_o` 不动，新增 `chunk_fwd_kernel_o_share_qk3`（grid 改 `(NT, B*Hg)`，一个 program 算一次 `b_A`，`static_range` 循环 3 个 value head 复用，后两个 head 不再加载 K），在 `chunk_fwd_o()` 加 env gate + 形状 gate。完整代码见 [`references/gdn-prefill-share_qk3.py`](./references/gdn-prefill-share_qk3.py)。
   - **gate 纪律**：env `VLLM_XIANDAO_GDN_CHUNK_O_SHARE_QK3` 默认开，设 `0` 回退原 kernel 做 A/B；形状 gate 精确锁 Qwen3.8-27B GDN 形状（`H==48 and Hg==16 and K==128 and V==128 and BT==64 and bf16`），其它模型/形状自动 fallback 不误伤；`static_range(1,3)` 硬编码 3 个 value head，依赖 `H/Hg==3`，换模型（H/Hg≠3）要改循环上界。
2. **solve_tril.py 避免整矩阵零化**：`Ai = torch.zeros_like(...)` → `torch.empty_like`（不预清零），但**必须**在 kernel 里显式零化严格上三角的 6 个 16×16 块（`Ai_12/Ai_13/Ai_14/Ai_23/Ai_24/Ai_34`，坐标 (0,16)(0,32)(0,48)(16,32)(16,48)(32,48)），用 `tl.store`（block_ptr）或 `desc_o.store`（TMA，更快）。⚠ **正确性关键**：`empty_like` 后不补 0 这 6 块，上三角是垃圾值，三角求解结果错，数学/代码类任务掉分。改完必跑精度验证。
3. **chunk_delta_h.py 删 aiter 分支（可选）**：hcu 插件 patch 进来的版本有 aiter 自定义 kernel 分支（`VLLM_HCU_USE_CUSTOM_AITER_FLA` 命中走 `launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64`）。删掉该分支 + 顶部两个 import，统一走 Triton kernel。收益不如 #1，属统一/稳定性质；环境 aiter 路径没问题时可只做 #1+#2。

## 量化结果（2026-10-08，4×DCU TP4，Qwen3.8-27B，vLLM 0.18.1+hcu 插件，同卡同插件状态 A/B）

| 输入 | TTFT 原版→优化 | TPOT 原版→优化 |
|------|--------------|--------------|
| 8K  | 2.56s → 1.71s（快 33%） | 46.8 → 43.0ms（快 8%） |
| 16K | 7.07s → 3.58s（快 49%） | 48.2 → 44.1ms（快 8%） |
| 32K | 21.82s → 8.20s（快 62%） | 50.6 → 46.7ms（快 8%） |

**特征**：prefill 收益随输入长度**单调放大**（kernel 级优化特征，与 `dcu-inference-optimization` §12.4 的 O(N) 线性层建模一致）；decode 稳定 ~8%（decode 是 memory-bound 状态递推，GDN decode kernel 已 88-90% roof 无 headroom，见 `dcu-op-optimization` 微基准 roofline 节）。

## 验证方法（必须做，别只信性能）

1. **性能 A/B**：同卡、同插件状态、唯一变量是 kernel。`VLLM_XIANDAO_GDN_CHUNK_O_SHARE_QK3=1` vs `=0` 各起服务，固定 token 长度 prompt 测 8K/16K/32K × 128 输出的 TTFT/TPOT。
2. **精度**：GSM8K 或数学 benchmark，优化 vs 原版 ±0.5pp 内；temperature=0 下逐 token 对比应一致。重点盯优化 #2 的 6 块零化——没补 0 数学/代码任务会掉分。

## 可复用经验

1. **GDN 类 linear-attention 的 prefill chunk kernel 有 QK 冗余**：当 `H/Hg > 1`（多个 value head 共享 query head）时，原版按 value head 展开的 grid 会冗余算 QK 矩阵 + 冗余加载 K。按 query head 展开、循环 value head 复用 QK 是通用去冗余手法（本例 H/Hg=3）。
2. **kernel 级优化的收益随序列长度单调放大**（区别于 launch-overhead 类优化），与 O(N)/O(N²) 建模一致——这是判断"优化是否真在 kernel 层"的特征信号。
3. **`empty_like` 替代 `zeros_like` 省整矩阵清零，但必须补零 kernel 不写的区域**（本例 6 个上三角块），否则垃圾值污染三角求解。
4. **形状 gate 精确锁目标模型 + env 开关做 A/B** 是 DCU 上注入专用 kernel 的标准范式（与 `dcu-vllm-runtime-patch` 一致）：不误伤其它模型、可回退、可对照。

## 踩坑

1. **`VLLM_PLUGINS=`（空串）会静默禁用 hcu 插件** ★最隐蔽：未设置（None）→ 加载全部；空串 → 解析成 `['']` → `"hcu" in ['']` False → 插件被跳过，**无任何报错**，只少了 "Platform plugin hcu is activated" 一行。后果：attention backend/算子分发全变，decode 掉速 7-8%（曾误判为 fork 代码问题）。A/B 时两组 `VLLM_PLUGINS` 必须完全一致。详见 `dcu-vllm-hcu-source`。
2. **源码编译 fork 会覆盖 hcu 插件 patch 过的文件**：插件靠字符串 patch 往 vLLM 源码注入类（如 `CpCommonAttentionMetadata`），源码编译会把 patch 过的文件覆盖回原版 → 插件激活时 `ImportError`。修复：编译后手动 `from vllm_hcu.patches import apply_patches; apply_patches()`（幂等）或跑 `vllm-hcu-apply-patches`。详见 `dcu-vllm-hcu-source`。
3. **共享机选卡**：跑前 `hy-smi` 看显存；杀自己残留 `pgrep -f "vllm serve" | xargs -r kill -9`，注意 EngineCore/Worker_TP 子进程要单独杀（父进程 kill 后子进程可能残留占显存）；绝不杀别人的服务。
4. **csrc MMAC i8 stub**：fork 的 `csrc/rocm/attention.cu` 把 `v_mmac_i32_16x16x32_i8` stub 掉（dcc 25.10 不支持）——int8 矩阵乘，bf16 dense 用不到，不影响本方案；换 DTK 版本时注意是否要恢复。

---

## §39 使用须知

- 本方案针对 **vLLM 0.18.1 + hcu 插件** 的 `fla/ops` 路径；SGLang 的 GDN kernel 在 `sglang/kernels/ops/attention/fla/`，改法思路相同（QK 共享）但代码不同，勿直接搬。
- 形状 gate 锁死 Qwen3.8-27B；其它 GDN 模型（H/Hg 不同）需按 gate 纪律调整 gate 和 `static_range` 上界。
- 性能数字是 2026-10-08 单点实测，随硬件/DTK/版本变化，只作量级参考。
- 完整方案文档（含 stock/fork 源码对照）：内部文档 <internal-doc-link>
