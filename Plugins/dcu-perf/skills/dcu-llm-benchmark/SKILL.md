---
name: dcu-llm-benchmark
description: DCU大模型推理压测 + W8A8 INT8 首字慢归因、tune-cache 补桶、冷编译 warmup
version: 1.2.0
author: DCU Engineer
license: MIT
---

# DCU 大模型推理性能测试（dcu-llm-benchmark）

> 定位：海光 DCU 上**端到端推理吞吐 / 延迟压测**的方法学与纪律，不是算子级 profile（那是 dcu-hipprof / dcu-perf-profiling）。
> 典型意图：「跑一下 vLLM / SGLang 的 bench，看吞吐和延迟」「vllm bench serve 怎么用」「bench 数字怎么才可信」。
> 内容纪律：**不存会被视为固定基线的性能数字**（token 数 / ns / 吞吐绝对数属项目侧留痕）；本文前几条纪律里的 15-20%、5~7ms、~7% 是**经验量级/坑的定性警示**，不是本环境实测基线，数值以项目实测为准。本页只留「正确跑出可信数据」的方法 + DCU 专属坑 + 命令模板。

---

## 一、前置纪律（跑任何 bench 之前必做，否则数字失真）

按你的既定项目测试纪律，顺序不可省：

1. **先定镜像**：bench 必须基于你指定的 Docker 镜像起的专用容器跑，**编译/运行都在容器内**。先确认镜像（如 `<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-gemma4-0413`），后续全基于此。
2. **dcu-env 摸底**：跑前确认硬件拓扑、驱动、DTK 版本、gfx（物理机 `rocminfo`，容器内不可靠）。详见 `dcu-env`。
3. **容器挂载**：必须 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见（DEVCOUNT=0）。
4. **docker exec 必须 `-ilc`**：`docker exec -d bash -ilc '...'`。用 `-c`（非交互非登录）不会加载 `.bashrc`/`/etc/profile`，导致 rocblas/hipblaslt 未加载，性能掉 15–20%。**所有性能命令都走 `-ilc`**。详见 `dcu-docker` 测评章节。
5. **挑卡用 `HIP_VISIBLE_DEVICES=0,1,2,3`**（编号从 0 起），不靠 `-g N` 自行挑卡。
6. **进容器先查 `NCCL_NCHANNELS_PER_PEER`**：镜像常预置 `=2`，有则 `unset` 再测。
7. **锁频检查（DCU 动态调频会吃掉 bench 数字）**：DCU 默认按负载动态调频，decode 低负载时频率掉下来，TPOT 虚高 5~7ms、吞吐掉 ~7%。bench 前把用到的卡锁到 sclk level 6（760MHz）：
   ```bash
   for i in 0 1 2 3; do   # 换成实际 HIP_VISIBLE_DEVICES 的卡号
     rocm-smi -d $i --setperflevel manual
     rocm-smi -d $i --setsclk 6
   done
   rocm-smi --showclocks 2>&1 | grep sclk   # 确认：所有卡 level 6 (760Mhz)
   ```
   **某些容器下 rocm-smi 锁频会静默失败（权限不足）**，必须用 `--showclocks` 回读确认；异常形态：level 0 (300Mhz) 或 Perf 列是 `auto`。未确认锁频成功的 bench 数字不可用于版本间对比。
   **<device-B>/BW100 实测（2026-10-20，短时推理场景）**：BW100 sclk level 6 = **1300MHz**（level 频率值以 `--showclocks` 回读为准，不是上面示例的 760MHz）；空闲卡默认 level 1 (600MHz)。短时工作负载（秒级推理，e2e ~0.8s 量级）频率爬升主导：同一工作负载实测**未锁频 7.5s、锁频 0.80s，10 倍差**——比 decode 低负载场景 5~7ms/TPOT 的误差大一个量级；短时工作负载务必先锁频再测，否则数字无意义。

> 汇报测试结果时：输出指令完整原始 stdout，**不裁剪、不算均值、不自行分析/下结论**；异常原样保留，分析单独标「观察」与结果分离。

---

## 二、vLLM 端到端压测：`vllm bench serve`（以 0.18.1 为例）

`vllm bench serve` 是 vLLM 自带在线服务吞吐压测器（Benchmark the online serving throughput）。

### 2.1 最小可用命令（用户实证模板，GLM-5.2 INT4 量化模型）

```bash
vllm bench serve \
  --model /data/models/GLM-5.2-INT4-w4a8 \
  --random-input-len 4096 \
  --random-output-len 1024 \
  --port 8000 \
  --dataset-name random \
  --num-prompts 64 \
  --max-concurrency 64 \
  --ignore-eos \
  --served-model-name glm-5.2
```

- `--model`：本机模型路径（或 HuggingFace ID）。
- `--random-input-len / --random-output-len`：`--dataset-name random` 专用，直接指定输入输出长度。
- `--dataset-name random`：用随机合成数据，无需外部数据集，最常用作快速吞吐摸底。
- `--num-prompts 64`：总请求数。
- `--max-concurrency 64`：最大并发请求数（模拟上层组件限流；与 `--request-rate` 配合时实际速率可能更低）。
- `--ignore-eos`：忽略 EOS 强制跑满 output-len（吞吐压测常用，避免短回复提前终止）。
- `--served-model-name glm-5.2`：服务侧模型名（与 `vllm serve --served-model-name` 对齐，否则 bench 按 `/v1/models` 首个模型兜底）。注意正确参数是 `--served-model-name`（非 `--served-model`）。

### 2.2 关键选项速查（**vLLM 0.18.1 专属**，基于容器内 `vllm bench serve --help` 全量整理）

> ⚠ **版本专属**：下表所有默认值/选项名/取值集均以 **vLLM 0.18.1** 的 `vllm bench serve --help` 为准。不同 vLLM 大版本（如 0.15.x / 0.21.0 / 0.23.0）选项可能有增删/改名，**跑之前务必用容器内对应版本的 `vllm bench serve --help` 复核**。
> 完整 0.18.1 原文见用户侧 `<local_bench_help_0181>`（已核对；本 skill 内所有 `/root/*.txt` 均为用户侧本地文件，非仓库内）。下表节选压测最常用项，其余（如 multimodal/blazedit/asr/sonnet 专用、采样参数）查阅该文件或 `--help`。
> **版本差异**：已基于 `<local_bench_help_0151>`（0.15.1）、`<local_bench_help_0210>`（0.21.0）、`<local_bench_help_0230>`（0.23.0）逐选项 diff，见 §2.2.1 / §2.5 四版本对照。

#### 2.2.1 vLLM 版本差异对照（0.15.1 vs 0.18.1 vs 0.21.0 vs 0.23.0，实测 diff）

> 总体结论：四版本 `bench serve` **主体高度一致**（选项名/指标/默认绝大多数相同）。差异仅在少数枚举值与少量新增选项，详见下表。

| 差异点 | 0.15.1 | 0.18.1 | 0.21.0 | 0.23.0 | 影响 / 应对 |
|---|---|---|---|---|---|
| `--backend` 取值集 | 无 `vllm-pooling` | 含 `vllm-pooling` | 含 `vllm-pooling` | 含 `vllm-pooling` | 0.15.1 不能 bench pooling 模型 |
| `--dataset-name` 取值集 | 无 `custom_mm` | 含 `custom_mm` | 含 `custom_mm`+`speed_bench` | 含 `custom_mm`+`speed_bench`+`timed_trace`+`custom_audio`+`custom_image` | 0.15.1 无自定义多模态；0.21.0+ 多 SPEED-Bench；0.23.0 再加 timed_trace/自定义音图数据集 |
| 6 个增强选项（`--asr-*`/`--insecure`/`--plot-*`） | 无 | 有 | 有 | 有 | 0.15.1 无；0.18.1+/0.21.0/0.23.0 有 |
| `--lora-assignment` | 无 | 无 | 有 | 有 | 0.21.0+ 支持 LoRA 分配策略 |
| `--speed-bench-*`(3) | 无 | 无 | 有 | 有 | 0.21.0+ 支持 SPEED-Bench 数据集 |
| `--chat-template-kwargs` | 无 | 无 | 无 | 有 | 0.23.0 支持 client-side apply_chat_template 的 kwargs（如 reasoning） |
| `--timed-trace-*`(7) | 无 | 无 | 无 | 有 | 0.23.0 新增 timed_trace 数据集全套参数 |
| `--random-range-ratio` 语义 | 单 float | 单 float | 单 float 或 JSON dict 分设 ISL/OSL | 同 0.21.0 | 0.21.0+ 可独立控输入/输出采样 |
| `--served-model`（无 -name） | **不存在** | **不存在** | **不存在** | **不存在** | ⚠ 四版本参数名均为 `--served-model-name`；勿写 `--served-model` |

> 数据来源：`<local_bench_help_0151>`(0.15.1)、`<local_bench_help_0181>`(0.18.1)、`<local_bench_help_0210>`(0.21.0)、`<local_bench_help_0230>`(0.23.0)。

| 选项 | 含义 | 默认 | 备注 |
|---|---|---|---|
| `--backend {vllm,openai,openai-chat,...}` | 压测目标后端类型 | `openai` | 打 vLLM 服务通常用 `openai` |
| `--base-url` | 服务/API base url（不用 host:port 时） | None | 默认 `http://{host}:{port}` |
| `--burstiness` | 请求到达突发度（Poisson=1；<1 更突发，>1 更均匀） | `1.0` | 仅当 `--request-rate` 非 inf 生效 |
| `--dataset-name {sharegpt,burstgpt,sonnet,random,random-mm,random-rerank,hf,custom,custom_mm,prefix_repetition,spec_bench}` | 数据集 | `random` | 真实分布用 `sharegpt`；快速摸底用 `random` |
| `--dataset-path` | sharegpt/sonnet 数据集路径 或 HF dataset ID | None | `random` 不需要 |
| `--disable-shuffle` | 关闭样本打乱 | False | 确定性顺序 |
| `--disable-tqdm` | 关闭进度条 | False | |
| `--enable-multimodal-chat` | 多模态 chat 变换 | False | |
| `--endpoint` | API endpoint | `/v1/completions` | chat 场景改 `/v1/chat/completions` |
| `--extra-body` | 每请求附加 JSON body，如 `'{"chat_template_kwargs":{"enable_thinking":false}}'` | None | 控制思考/采样等 |
| `--goodput GOODPUT` | SLO `KEY:VALUE`(ms)，如 `ttft:300 tpot:50`，指标名 `ttft/tpot/e2el` | None | 参考 DistServe |
| `--header [KEY=VALUE ...]` | 每请求附加 header | None | 覆盖后端常量 |
| `--host / --port` | 服务地址 | localhost / 8000 | |
| `--ignore-eos` | 请求忽略 EOS（强制跑满 output） | False | deepspeed_mii/tgi 不支持 |
| `--input-len / --output-len` | 通用输入/输出长度（映射到 random-input-len 等） | None | 未指定用数据集默认 |
| `--insecure` | 禁用 SSL 校验（自签证书时） | False | |
| `--label` | 结果 JSON 前缀标签（缺省用 backend） | None | 归档区分多次跑 |
| `--logprobs` | 每 token 计算并返回的 logprobs 数 | None | 不指定则单 dummy logprob |
| `--lora-modules` | 服务端已加载的 LoRA 子集，每请求随机选一个 | None | |
| `--max-concurrency` | 最大并发请求数 | None | 见 2.1 说明 |
| `--metadata [KEY=VALUE ...]` | 写入结果 JSON 的元数据，如 `version=0.3.3 tp=1` | None | 留痕用 |
| `--metric-percentiles` | 上报分位数，如 `25,50,75` | `99` | 配合 `--percentile-metrics` |
| `--model` | 模型名（缺省取 `/v1/models` 第一个） | None | |
| `--no-oversample` | 样本不足时不超采样 | False | |
| `--no-stream` | 不以流式加载数据集 | False | |
| `--num-prompts` | 处理请求总数 | `1000` | |
| `--num-warmups` | 预热请求数 | `0` | 建议设几个排除冷启动 |
| `--percentile-metrics` | 对哪些指标报分位数：`ttft/tpot/itl/e2el` | None（生成模型默认 `ttft,tpot,itl`） | |
| `--plot-dataset-stats` | 生成数据集统计图（prompt/output/combined token 分布） | False | |
| `--plot-timeline` | 生成 HTML 请求执行时间线图 | False | |
| `--profile` | 启用 vLLM Profiling（服务端需 `--profiler-config`） | False | |
| `--ramp-up-strategy {linear,exponential}` + `--ramp-up-start-rps`/`--ramp-up-end-rps` | 请求速率斜坡上升策略 | None | 渐进加压场景 |
| `--ready-check-timeout-sec` | 等端点 ready 超时（默认跳过 ready check） | `0` | |
| `--request-id-prefix` | 请求 ID 前缀 | `bench-...` | |
| `--request-rate` | 请求发起速率（inf=全速压，否则 Poisson/gamma 合成到达） | `inf` | 与 `--max-concurrency` 配合 |
| `--result-dir` | 结果 JSON 保存目录 | 当前目录 | |
| `--result-filename` | 结果 JSON 文件名（缺省 `{label}-{qps}qps-{model}-{dt}.json`） | None | |
| `--save-detailed` | 结果含每请求明细（response/error/ttfts/tpots） | False | |
| `--save-result` | 保存结果到 JSON | False | 留痕必开 |
| `--seed` | 随机种子 | None | |
| `--served-model-name` | API 侧模型名（缺省同 `--model`） | None | 即 2.1 的 `--served-model` 同义 |
| `--skip-chat-template` | 跳过 chat 模板 | False | |
| `--skip-tokenizer-init` | 跳过 tokenizer/detokenizer 初始化 | False | |
| `--timeline-itl-thresholds` | 时间线图 ITL 着色阈值(ms)，两值分三档 | `25 50` | |
| `--tokenizer` | 指定 tokenizer 名/路径 | None | |
| `--tokenizer-mode {auto,hf,slow,mistral,deepseek_v32,qwen_vl}` | tokenizer 模式 | `auto` | |
| `--trust-remote-code` | 信任 HF 远程代码 | False | |
| `--use-beam-search` | 启用 beam search | False | |
| sampling：`--temperature/--top-p/--top-k/--min-p/--frequency-penalty/--presence-penalty/--repetition-penalty` | 采样参数（仅 openai 系后端生效） | None | |

> random 数据集专用：`--random-input-len`（默认 1024）、`--random-output-len`（默认 128）、`--random-prefix-len`、`--random-range-ratio`（长度采样范围比）。
> 输出结果 JSON 用 `--save-result --result-dir <dir>` 落盘，结合 `--label`/`--metadata` 做多次对比留痕（**项目侧留痕，不入库本 skill**）。

### 2.3 关键指标与读法

`vllm bench serve` 输出的核心指标（按 `--percentile-metrics` 选择，默认报 `99` 分位）：

| 指标 | 含义 | 关注点 |
|---|---|---|
| **Total throughput (requ/s)** | 总请求吞吐 | 端到端服务能力，最常被汇报 |
| **Output token throughput (tok/s)** | 输出 token 吞吐 | 推理生成速率 |
| **TTFT** (Time To First Token) | 首 token 延迟 | 用户体验、prefill 性能 |
| **TPOT** (Time Per Output Token) | 每输出 token 延迟 | decode 性能、与并发强相关 |
| **ITL** (Inter-Token Latency) | token 间延迟 | 流式平滑度 |
| **E2EL** (End-to-End Latency) | 端到端延迟 | 单请求总耗时 |
| **Goodput** | 满足 SLO（goodput 阈值）的请求占比 | 服务质量 |

- `--request-rate inf` + `--max-concurrency N`：全速压、限并发 → 看饱和吞吐。
- 固定 `--request-rate R`（如 Poisson）：看给定负载下的延迟分布（TTFT/TPOT 分位）。
- `--num-warmups` 设几个，排除冷启动对首请求的影响。

### 2.4 真实分布的压测（sharegpt）

```bash
vllm bench serve \
  --model /data/models/Qwen3-8B \
  --dataset-name sharegpt \
  --dataset-path /data1/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --request-rate 10 \
  --max-concurrency 32 \
  --served-model-name qwen3-8b
```

### 2.5 vLLM 版本差异（0.15.1 vs 0.18.1 vs 0.21.0 vs 0.23.0，已补齐）

> ⚠ 不同 vLLM 大版本 `bench serve` 选项可能增删/改名/改默认。本 skill 主体以 **0.18.1** 为准（见 §2.2 + `<local_bench_help_0181>`）。
> **0.15.1**（`<local_bench_help_0151>`）、**0.21.0**（`<local_bench_help_0210>`）、**0.23.0**（`<local_bench_help_0230>`）的完整 `--help` 均已提供，经逐选项 diff，**四版本主体高度一致**，仅少数枚举/新增选项差异（详见 §2.2.1）：
> - 0.15.1：比 0.18.1 少 `vllm-pooling` backend、`custom_mm` 数据集、以及 6 个增强选项（`--asr-*`/`--insecure`/`--plot-*`）；
> - 0.21.0：在 0.18.1 基础上新增 `--lora-assignment`、`--speed-bench-*`(3)、`--dataset-name` 新增 `speed_bench`，且 `--random-range-ratio` 支持 JSON dict 分设 ISL/OSL；
> - 0.23.0：在 0.21.0 基础上新增 `--chat-template-kwargs`、`--timed-trace-*`(7) 数据集全套、`--dataset-name` 新增 `timed_trace`/`custom_audio`/`custom_image`；
> - 四版本**均无** `--served-model`，正确参数名为 `--served-model-name`（本 skill 模板已用正确名）。
>
> **跨版本命令模板**：§2.1 / §2.4 模板四版本通用（均用 `--served-model-name`）。0.15.1 不支持那 6 个增强选项（勿加 `--plot-timeline` 等）；0.21.0+ 可额外用 `--lora-assignment`/`--speed-bench-*`；0.23.0 还可加 `--chat-template-kwargs`/`--timed-trace-*`。跑任意版本前，**以容器内 `vllm bench serve --help` 实际输出为准**。

| 差异点 | 0.15.1 | 0.18.1 | 0.21.0 | 0.23.0 | 影响 / 应对 |
|---|---|---|---|---|---|
| `--backend` 取值集 | 无 `vllm-pooling` | 含 | 含 | 含 | 0.15.1 不能 bench pooling 模型 |
| `--dataset-name` 取值集 | 无 `custom_mm` | 含 `custom_mm` | +`speed_bench` | +`timed_trace`/`custom_audio`/`custom_image` | 0.15.1 无自定义多模态；0.21.0+ 多 SPEED-Bench；0.23.0 再加 timed_trace/音图 |
| 6 个增强选项 | 无 | 有 | 有 | 有 | 0.15.1 无；0.18.1+/0.21.0/0.23.0 有 |
| `--lora-assignment` | 无 | 无 | 有 | 有 | 0.21.0+ 支持 LoRA 分配策略 |
| `--speed-bench-*`(3) | 无 | 无 | 有 | 有 | 0.21.0+ 支持 SPEED-Bench |
| `--chat-template-kwargs` | 无 | 无 | 无 | 有 | 0.23.0 支持 client-side chat template kwargs（如 reasoning） |
| `--timed-trace-*`(7) | 无 | 无 | 无 | 有 | 0.23.0 新增 timed_trace 数据集全套参数 |
| `--random-range-ratio` | 单 float | 单 float | JSON dict 分设 ISL/OSL | 同 0.21.0 | 0.21.0+ 可独立控输入/输出采样 |
| `--served-model-name` | 有（同名） | 有 | 有 | 有 | 四版本均用此名，勿写 `--served-model` |

---

## 三、SGLang 等效压测路径（**0.5.10 / 0.5.12 专属**，基于 `<local_bench_help_0510>` 与 `<local_bench_help_0512>`）

SGLang 自带压测：`python -m sglang.bench_serving`（参数与 vLLM `bench serve` 高度对齐，但选项名/枚举有差异，且**没有 `--ignore-eos`，对应是 `--disable-ignore-eos`**）。

> ⚠ **版本专属**：下表以 **0.5.12** 为准（基于 `<local_bench_help_0512>`）。0.5.10（`<local_bench_help_0510>`）差异见 §3.2.1，两版本主体一致，仅少数数据集/选项新增。

### 3.1 最小可用命令（用户实证模板，Kimi-K2.6 + random-ids）

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --model /data1/models/Kimi-K2.6 \
  --port 8000 --host 0.0.0.0 \
  --dataset-name random-ids \
  --tokenize-prompt \
  --num-prompts 1 --max-concurrency 1 \
  --random-input-len 512 --random-output-len 512 \
  --random-range-ratio 1
```

- `--backend sglang`：打 SGLang 服务（也可用 `sglang-oai` / `sglang-native` 等，依服务暴露的接口）。
- `--dataset-name random-ids`：随机 id 数据集（见 §3.3 坑：DS 类模型必须用 `--tokenize-prompt` 配合）。
- `--tokenize-prompt`：**关键**（见 §3.3）。用整数 id 而非字符串作输入，精确控制 prompt 长度，避免 tokenizer 长度膨胀。
- `--random-range-ratio 1`：输入/输出长度采样范围为 `[len*(1-ratio), len*(1+ratio)]`，=1 即全范围采样（长度波动大）；要固定长度设 0。
- `--host 0.0.0.0` / `--port 8000`：SGLang 默认 host 0.0.0.0、默认端口按引擎不同（常 30000），显式指定更稳。

### 3.2 关键选项速查（**0.5.12 专属**，基于 `<local_bench_help_0512>`）

| 选项 | 含义 | 默认 | 备注 |
|---|---|---|---|
| `--backend {sglang,sglang-native,sglang-oai,sglang-oai-chat,sglang-embedding,vllm,vllm-chat,lmdeploy,lmdeploy-chat,trt,gserver,truss}` | 压测目标后端 | 必填 | 打 SGLang 服务用 `sglang` 系 |
| `--base-url` | 服务 base url（不用 host:port 时） | None | |
| `--host` / `--port` | 服务地址 | host 0.0.0.0；port 按引擎默认 | |
| `--ready-check-timeout-sec` | 等端点 ready 超时（0=跳过） | `60` | |
| `--dataset-name {autobench,sharegpt,custom,openai,random,random-ids,generated-shared-prefix,mmmu,image,mooncake,longbench_v2}` | 数据集 | — | `random-ids` 为随机 id 数据集（DS 类必配 `--tokenize-prompt`）；0.5.10 无 `autobench`/`longbench_v2` |
| `--dataset-path` | 数据集路径 | None | |
| `--model` | 模型名/路径（缺省请求 `/v1/models`） | None | |
| `--served-model-name` | 服务侧模型名（缺省同 `--model`） | None | 注意是 `--served-model-name`（同 vLLM，无裸 `--served-model`） |
| `--tokenizer` | 指定 tokenizer | 模型默认 | |
| `--num-prompts` | 请求总数 | `1000` | |
| `--sharegpt-output-len` / `--sharegpt-context-len` | sharegpt 数据集输出/上下文长度 | None | 仅 sharegpt |
| `--random-input-len` / `--random-output-len` | random/image 数据集的输入/输出长度 | None | |
| `--random-range-ratio` | 长度采样范围比 `[len*(1±r), len*(1+r)]` | `0.0` | `=1` 全范围；`=0` 固定长度 |
| `--request-rate` | 请求速率（inf=全速） | `inf` | Poisson 合成 |
| `--max-concurrency` | 最大并发 | None | 限流/与 request-rate 配合 |
| `--output-file` | 结果 JSONL 文件名 | None | |
| `--output-details` / `--print-requests` | 输出明细 / 实时打印请求 | False | 排查用 |
| `--disable-tqdm` / `--disable-stream` | 关进度条 / 关流式 | False | |
| `--disable-ignore-eos` | **禁用**忽略 EOS（即默认是「忽略 EOS」） | 默认忽略 EOS | ⚠ 与 vLLM 相反：vLLM 是 `--ignore-eos`（默认不忽略），SGLang 是 `--disable-ignore-eos`（默认忽略） |
| `--return-logprob` / `--top-logprobs-num` / `--token-ids-logprob` / `--logprob-start-len` | 返回 logprob 相关 | False / None | |
| `--return-routed-experts` | 返回路由专家（MoE） | False | |
| `--extra-request-body` | 附加 JSON 到请求体（采样参数等） | None | |
| `--apply-chat-template` | 应用 chat 模板 | False | |
| `--profile` / `--profile-activities` / `--profile-output-dir` 等 | Torch Profiler（需服务端 `SGLANG_TORCH_PROFILER_DIR`） | — | 0.5.12 `--profile-activities` 多 `MEM`（torch.cuda.memory 快照） |
| `--plot-throughput` | 画吞吐/并发随时间图（需 termplotlib+gnuplot） | False | |
| `--lora-name` / `--lora-request-distribution` / `--lora-zipf-alpha` | LoRA 压测 | None | |
| `--warmup-requests` | 预热请求数 | `0` | 排除冷启动 |
| `--tokenize-prompt` | 用整数 id 而非字符串输入，精确控制 prompt 长度 | False | **⚠ 见 §3.3，DS 类必加** |
| `--flush-cache` | bench 前清缓存 | False | |
| `--pd-separated` / `--profile-prefill-url` / `--profile-decode-url` | PD 分离部署压测 | — | |
| `--fake-prefill` | 纯 decode 性能压测（配 fake prefill 后端） | False | **0.5.12 新增；0.5.10 无** |
| `--tag` | 结果 tag | None | 留痕 |
| `--header` | 自定义 HTTP header | None | |
| `generated-shared-prefix` 专用：`--gsp-*` | 共享前缀数据集参数（组数/长度等） | — | |
| `mooncake` 专用：`--mooncake-*` | mooncake trace 回放参数 | — | `--request-rate` 在 mooncake 模式被忽略 |

> 完整 0.5.12 原文见 `<local_bench_help_0512>`；0.5.10 见 `<local_bench_help_0510>`。**版本专属**：不同 SGLang 版本选项可能增删，跑前以 `python -m sglang.bench_serving --help` 为准。
> 指标对齐 vLLM：`requ/s`、`TTFT`、`TPOT`、`ITL`；SGLang 默认端口常 30000（显式 `--port` 更稳）。

#### 3.2.1 SGLang 版本差异对照（0.5.10 vs 0.5.12，实测 diff）

> 总体结论：两版本 `bench_serving` **主体完全一致**（选项名/指标/默认绝大多数相同）。差异仅在少数枚举值与 1 个新增选项。

| 差异点 | 0.5.10（`<local_bench_help_0510>`） | 0.5.12（`<local_bench_help_0512>`） | 影响 / 应对 |
|---|---|---|---|
| `--dataset-name` 取值集 | 9 项（无 `autobench`/`longbench_v2`） | 11 项（多 `autobench`、`longbench_v2`） | 0.5.10 无 autobench/longbench_v2 数据集 |
| `--profile-activities` 枚举 | `{CPU,GPU,CUDA_PROFILER,XPU}` | 多 `MEM` | 0.5.12 可抓 torch.cuda.memory 快照（memory_viz） |
| `--fake-prefill` | **无** | 有（纯 decode 压测，配 fake prefill 后端） | 0.5.10 无纯 decode 压测快捷选项 |
| `--tokenize-prompt` / `--random-range-ratio` / `--served-model-name` 等 | 均有 | 均有（同名同义） | 两版本核心参数一致；§3.3 实战坑两版本通用 |

> 数据来源：`<local_bench_help_0510>`(0.5.10)、`<local_bench_help_0512>`(0.5.12)。§3.1/§3.2 命令模板两版本通用（0.5.10 勿加 `--fake-prefill` 即可）。

### 3.3 ⚠ 实战坑：DeepSeek 类模型必须 `--tokenize-prompt`（tokenizer 长度膨胀）

> 来源：用户 DCU 实测经验（Kimi-K2.6 vs DeepSeek 对比）。**0.5.10 / 0.5.12 均适用**。

- **现象**：`--dataset-name random-ids` 会在模型中 **decode 后再 encode**，存在「长度变化过程」。
- **根因**：不同 tokenizer 对字符的处理差异巨大：
  - **DeepSeek 类 tokenizer**：对部分字符会拆分成更多 token，**长度膨胀 2–3 倍**（如本该 512 token 的输入被撑到 1000+）。这会严重污染 bench 的输入长度假设，导致吞吐/延迟不可比、甚至 OOM。
  - **Kimi 类 tokenizer**：处理方式不同，长度膨胀远小于 DS（可能只差几个~十几个 token 量级），影响很小。
- **对策**：**DeepSeek 类模型跑 bench 必须加 `--tokenize-prompt`**（用整数 id 直接作输入，绕过字符串→token 的再编码膨胀，精确控制长度）。Kimi 等影响小的模型可不加，但加上也无害、且结果更可控。
- **教训**：bench 前先确认模型家族（DS 系 vs 其他），DS 系默认带 `--tokenize-prompt`；比较不同模型吞吐时，必须保证输入长度口径一致（都 `--tokenize-prompt` 或都不加），否则数字不可比。

### 3.4 ⚠ 实战坑：超长上下文（>256k）`random-ids` 被截断，`--tokenize-prompt` 也救不了（2026-08-25 10921 实测）

- **现象**：`--dataset-name random-ids --random-input-len 700000` 和 `999000` 两轮，客户端 `#Input tokens` 与 soak 日志 "Total input tokens" **完全相同**（619,794，每请求仅 ~128k），KV 水位峰值只有 0.25（预期 0.9+）——请求被截断，"700k/1M 压测通过"的结论实际只验证了 ~128k。
- **根因线索**：服务端日志有警告 `Target model's context_length (1000000) is greater than the derived context_length (262144)`——yarn/rope override 把 context-length 扩到 1M 后，工具链某处仍按模型 config 推导长度（262144）截断；加 `--tokenize-prompt`（绕过 §3.3 的 decode/encode 往返）后长度**仍不变**，截断点不在 tokenizer 往返。与 §3.3 是同一族坑（长度口径失真）但机制不同：§3.3 是膨胀，本条是截断。
- **对策**：超长上下文（>256k）压测**放弃 bench_serving 数据集路径，直接调 `/generate` 发 `input_ids`**——POST `{"input_ids": [随机 id × N], "sampling_params": {"max_new_tokens": M, "temperature": 0.0}}`，服务端不截断（`meta_info.prompt_tokens` 精确等于 N）；并发用 ThreadPoolExecutor 同时发（串行只顶到单请求水位，压不出高水位场景）。
- **铁律**：长上下文压测后**必须核对服务端实际收到的长度**（`prompt_tokens` / soak "Total input tokens" / 服务端 prefill 日志 `#pending-token` 初值）与设定值一致，不能只看 exit=0 / 0 崩溃——截断会让"极限测试"静默降级成中长度测试。

### 3.5 ⚠ 实战坑：无外网容器 `bench_serving` 直接不可用（2026-08-20 <device-A> 实测）

- **现象**：离线 DCU 容器里 `sglang.bench_serving --dataset-name random`（或 random-ids）报 `hf_hub_download` RuntimeError——它要联网下载 tokenizer；`--tokenizer` 指本地模型目录也不够（内部仍走 HF 下载路径）。
- **对策**：自写压测客户端——本地 `AutoTokenizer(local_files_only=True)` 随机采样 token id 拼 prompt（随机 token 天然无公共前缀，顺带排除 prefix cache 干扰），直接打 `/v1/completions`（`stream=True`，payload 用 `"prompt": [token_ids]` + `ignore_eos` + `max_tokens`），自测 TTFT（首个非空 chunk）/TPOT（`(total-ttft)/(n_out-1)`）/吞吐。每批前调 `/flush_cache` 清前缀缓存。
- **A/B 纪律**：对比两组启动参数除被测项外逐字一致（speculative/radix/mamba 策略全对齐）；注意 `--disable-radix-cache` 与 mamba `extra_buffer` 硬冲突（须改 `no_buffer`）。

### 3.6 ⚠ A/B 纪律：投机解码（EAGLE/MTP）与前缀缓存必须显式排除（2026-08-21 <device-A> 实测）

- **投机解码是启动参数，运行时切不了**：要测"纯 decode"必须去掉全部 `--speculative-*` 参数**重启服务**（其余参数逐字一致）。生产配置常默认开 EAGLE，**所有绝对吞吐数字都含投机解码加速**——汇报前必须先说清数字是否含 EAGLE，否则口径不可比。
- **Accept length 是 EAGLE 生效的直接证据**：bench 输出 `Accept length ≈ 3.8`（上限 = num_draft_tokens）即 EAGLE 每步接受 ~3.8 个 draft token；关 EAGLE 后该指标不存在。看到 Accept length 就知道数字被投机解码主导。
- **前缀缓存用 random-ids 天然排除**：`--dataset-name random-ids --tokenize-prompt` 随机 token id，任意两条 prompt 从第 0 个 token 起即不同，radix cache 命中率≈0（服务侧 `disable_radix_cache=False` 也等效排除）。
- **EAGLE 增益量级 ~2x，且随输入长度单调递减、超长上下文负优化**（以下量级为 2026-08-21 <device-A>/BW150/sglang 0.5.12 fork/Qwen3.8-27B 实测，换硬件/版本/模型须重测，勿直接套用）：128k 输入下开 EAGLE 比不开还慢（draft 模型也要处理长 KV，验证开销超过接受收益；长 prefill 排队期间投机无收益）。**建议：短/中上下文（≤16k）开 EAGLE，超长上下文（128k）关投机解码**——该建议基于上述实测，其他环境先小样本 A/B 确认拐点。
- **相对增益 vs 绝对数字**：A/B 两组都开（或都关）EAGLE 时，被测项的相对增益仍有效（EAGLE 是常量）；但"带 EAGLE"与"纯 decode"的绝对数字不可互比，须分开汇报。

### 3.7 ⚠ 对比两个投机解码方案（DFlash2 vs 原生 MTP 等）的三条铁律（2026-08-21 <device-B> 实测）

对比"方案 A vs 方案 B 哪个命中率高/吞吐高"时，以下三条不遵守会得出**方向性错误**的结论：

1. **随机文本是 OOD 陷阱（最易踩，夸大差距 2 倍+）**：用随机文本/随机 token 做长上下文 prompt 时，**独立小 draft**（如 DFlash2 1.92B）的 accept 会崩到 0.1-0.2（分布外），而**主模型自带的 MTP 层**（与 target 同分布、共享权重）accept 仍 0.6-0.85 → 严重夸大方案差距。必须用**真实业务负载**（代码补全/自然语言，in-distribution）复测。实测：DFlash2 长上下文随机文本 37-44 tok/s → 真实代码 61.0 tok/s，与 MTP 的差距从 1.4-1.6x 缩到 1.23x。
2. **overlap v1/v2 是头号混淆变量**：不同方案可能跑在不同 overlap 状态（如本 build 对 DFLASH 强制 spec v1 无 overlap，EAGLE/MTP 默认 v2 overlap 开）。**串行 bs=1 下 overlap 只值 ~+10 tok/s，但高并发下收益随并发放大**（实测 MTP 并发 4/8/16/32 比 DFlash2 快 2.7-3.3x，串行只 1.2-1.4x）。对比时若两方案 overlap 状态不同，**必须标注**，否则差距归因错误（会把 overlap 收益算到算法头上）。
3. **accept rate 口径要归一**：不同方案 per-token 命中率不可直接比（DFlash2 一次 draft 3-7 个 vs MTP 每轮 3 个，预测越远越难命中）。用 **accept len（每轮提交 token 数）+ 吞吐**做公平口径，并主动说明 per-token 口径差异。取数用日志 `grep -o "accept len: [0-9.]*, accept rate: [0-9.]*"`（`/get_internal_state` 的 avg_spec_accept_length 常为 None）。

> 结论模板（<device-B> Qwen3.8-27B mamba 混合 + DCU build）：DFlash2 优化到天花板后仍全面落后原生 MTP（串行真实负载 1.2-1.4x、高并发 2.7-3.3x），未复现论文"DFlash2 命中率更高"（block 对齐后命中率持平 ~72%）。剩余差距是算法本质（MTP draft 是主模型自带 1 层、同分布、每轮 3 次便宜前向）+ build 限制（DFlash2 被锁 spec v1），非可消除的接口 handicap。

### 3.8 ⚠ 投机解码（DFlash/MTP）调参的 telemetry 口径 + 长上下文质量门契约（2026-08-24 Qwen3.8-27B 三产品线研究提炼）

**DFlash/投机解码调参的 objective 必须是「单位 accepted visible token 的总时间」，不是单纯 maximize acceptance**：
```
cost_per_visible = (draft_ms + selector_ms + verify_ms + runtime_ms) / accepted_tokens
```
- 上游已有实例：acceptance length 约 2.0 时 DFlash 仍使某 MoE 模型从 ~146 tok/s 降到 ~93 tok/s——「acceptance 不错」不等于「系统更快」（量级参考，换模型/硬件须重测）。
- 因此 spec 调参前必须给多档/多产品线统一加 **DFlash stage telemetry**（accept / draft_ms / selector_ms / verify_ms / accepted_tokens_per_round），否则后续所有 spec tuning 都无法严谨归因。
- selector 数值路径本身也是性能变量（社区实测 selector ranking 改 FP32 后 acceptance 3.01→3.24、C1 28.52→34.08 tok/s，量级参考）——不能只 sweep draft block 而忽略 selector 实现。
- DFlash 可见 decode 可越过 true-M1 weight-bandwidth roof（target verifier 一次确认多个 visible token，摊薄 weight streaming），所以「visible decode 超过 M1 roof」不是物理矛盾，别拿 M1 roof 当利用率 KPI。

**长上下文（>128K）质量门契约（cold benchmark 之外必跑）**：
- **arithmetic20**（20 题算术，记录 known_misses 集合，如 [8,17]）+ **critical case15**（单题精确值，如 303）——量化质量回归的最小集。
- **repeated SHA**：同一长上下文请求跑 3 次独立 cold run，输出 SHA 必须完全一致（抓 SHA 漂移/acceptance 崩溃/偶发 SIGSEGV）。
- **long exact/needle**（128K 精确检索）+ **long retrieval P95**（257.9K 检索 P95）。
- **contamination=0 / restart=0 / OOM=false**：无污染、无重启、无 OOM 是发布硬门禁。
- 数据层级不允许混用：Final/Accepted（当前唯一主口径）/ Corrected research authority（解释因果与历史演进）/ Historical invalid（只解释错误根因，不得当冠军性能）。

### 3.9 PD 分离部署：配比选型与单卡口径折算方法论（脱敏方法论）

> 本节为**纯方法论**，来自某 PD 分离项目现场测试的提炼。**不涉及具体模型名/客户/友商/具体并发数**。适用于 prefill/decode 分离（PD 分离）部署时：①怎么选 P/D 副本配比；②跨厂家横向对比时折算口径。

#### 3.9.1 PD 分离配比选型（1P1D vs 2P1D 及副本内配比）

PD 分离把 prefill（P）与 decode（D）拆到不同 GPU 副本，P/D 副本数及副本内部的 CP（context parallel）/EP（expert parallel）/DP（data parallel）配比，**必须针对目标场景的「输入上下文长度 + 期望输出吞吐 + 是否要求低 TTFT」设计**，不是拍脑袋。

- **prefill 负担重的场景**（超长上下文输入，如 128k 级输入、输出相对短、**不要求低 TTFT**）：用 **1P1D**（1 个 prefill 副本 + 1 个 decode 副本）即可满足，是此类 SLA 下的最简/最优配比；P/D 副本内部按模型 MoE 结构配 CP/EP（如 P:CP+EP、D:DP+EP）。
- **吞吐/单卡并发要求更高、prefill 相对轻**的场景（输入更短、要求更高吞吐）：可增加 prefill 副本数用 **2P1D**（2 P + 1 D），性能通常更优（分摊 prefill 计算，decode 端承接更多并发）。

选型要点：
- **配比是"为特定 SLA 设计的"**：后台/规划侧针对目标 SLA（输入长度+吞吐+TTFT 要求）算出的配比才是最优；现场临时拼的配比往往**不满足最优**，仅为满足该场景基本要求。上线前应让规划侧按 SLA 重算，而非沿用临场配比。
- **验证手段**：用真实/客户负载压测确认满足 SLA；并观察总并发是否还有余量（"还能摸高"），有余量说明配比偏保守、可上探。
- **同场景换配比要重测**：不同输入输出长度、不同模型结构，最优配比不同；不要跨场景套用同一配比。

#### 3.9.2 跨厂家/跨方案横向对比：单卡口径折算

- **背景**：不同厂家（或同一厂家不同方案）的 PD 分离**方案本身不同**——用的卡数、P/D 配比、切分方式都不同。**直接比"总并发数"或"总吞吐"不可比**（卡数不同、配比不同，总量被规模放大）。
- **方法**：**折算成单卡口径**——`单卡并发 = 总并发 ÷ 总卡数`，或 `单卡输出吞吐 = 总输出吞吐 ÷ 总卡数`。用单卡口径对比，才能公平反映硬件/方案的真实单卡能力，而非总规模。
- **纪律**：横向对比报告中，除总量外**必须同时给出单卡口径**；对比前先确认双方卡数/配比口径一致，否则数字会误导（规模效应被误判为硬件差异）。

#### 3.9.3 混布（多实例）vs PD 分离：第一层选型判据

> 多机多卡场景，先定拓扑（混布 vs PD 分离），再定 P/D 配比（3.9.1）。纯方法论；SGLang 具体部署参数见 `dcu-sglang-source` §四·补5。

判定顺序：

1. **IB 组网是硬前提**：PD 分离的 KV cache 传输只能走 200G IB（RDMA），不走 TCP/Socket——P/D 节点间无 IB 互联则 PD 分离直接排除，只能混布；组网规划须先预留 IB 链路，否则 KV 传输成为瓶颈。
2. **输入/输出比**：长输入、输出相对短的场景 PD 分离收益大（prefill 与 decode 资源画像完全不同，分离后互不阻塞）；输入输出均衡或输出偏长的场景，混布吞吐上限更高（无 KV 传输一跳、无 P/D 配比调优成本）。
3. **SLA/稳定性**：PD 分离把 prefill 尖峰与 decode 侧隔离，decode 侧 TTFT/TPOT 更稳、P/D 可独立扩缩；代价是网络一跳 + 组件更多（Router），故障面更大（已知问题：KV cache 达临界值时 P 节点崩溃未定位、D 节点误把对外服务网卡指定为管理网卡宕机）。混布组件少、故障面小，但 prefill 长序列会阻塞 decode（chunked prefill、前缀缓存等缓解）。
4. **框架开关约束**：双机混布场景 APC（前缀缓存）与 MTP（投机解码）不可同时开启——方案若要求两者同开，本身就推向 PD 分离（PD 模板中投机解码放 D 侧、前缀缓存常禁用，两者天然不冲突）。

结论模板：**无 IB → 混布；有 IB + 长输入短输出 + SLA 严 → PD 分离；有 IB + 均衡场景求吞吐上限 → 混布优先、PD 分离作 A/B 候选**——结论必须用真实负载压测验证，不靠判据空推。

---

### 3.10 ⚠ TTFT 长度曲线测量 + 首请求 JIT 归因（复现客户「首字慢」类症状）

复现客户「首字慢 / 波动 / 首请求慢重复快 / 重启后首批慢」这类 TTFT 症状、或画 prefill TTFT 随输入长度曲线时，两条测量纪律 + 一条归因判断：

1. **每档长度用独立探针，勿用同一段 base 文本 repeat 到多档**（最易踩，直接画出假曲线）：用同一段文本 repeat 出不同长度时，**长档的前缀恰好等于已发过的短档**，radix/prefix 缓存命中共享前缀 → 长档**假性偏快**、曲线非单调（更长的档反而比更短的快，纯假象）。对策：每档 `input_ids` 用独立随机序列，或每档头部塞一个该长度独有的 token（保证各档从第 0 token 起互不相同）。这是 §3.6「两条 prompt 不共享前缀」在**长度曲线**场景的加强版——专防「长档前缀含短档」。
2. **首请求（一次性 JIT）与稳态分开测、分开报**：引擎 warmup 通常只覆盖一小撮 shape（不少 INT8 patch 链把 warmup 尺寸硬编码成固定小列表，未必覆盖你测的大长度）。首个落在 warmup 覆盖之外的大 shape 会触发**一次性 Triton JIT（秒级，可与稳态差数倍）**。要复现客户「首请求慢 / 重复快 / 重启后首批慢」，就是分别量「该长度首请求」与「重复/稳态」：前者含 JIT，后者才是真实 prefill 速率——别把 JIT 的一次性开销误当稳态 prefill 慢。
3. **W8A8 sparse tune-cache miss 是亚秒级，不是多秒 TTFT 的根因**：w8a8 模型的 tune cache 若只有少数 M 桶，短 prefill miss 落到的是**启发式 stock kernel，很快（亚秒）**。多秒级首字慢**不要**归给 W8A8 cache miss——归到 ①一次性 JIT（见上条）或 ②**baseline prefill 速度**（INT8 的 native 加速多为 decode 侧 gemv，**prefill 不受益**，靠 TP 摊算力解决）。**别为多秒症状去「补桶」**（往 tune cache 加 M 桶）——那补的是一个亚秒级小头。

### 3.11 W8A8 INT8 tune cache 生成：真要补桶时的 DCU 方法（配合 §3.10·3「别为多秒症状补桶」）

客户**明确要求**消除 W8A8 sparse tune cache 的 miss（要把舍入表能产出的 M 桶都补进 cache，让短/中 prefill 命中调优配置而非 stock 启发式）时，在 DCU（K100-AI/gfx928）上按此法生成——**不要**靠 lmslim 自带的 autotune 工具：

1. **lmslim 的 W8A8 INT8 autotune 内核在 DCU 上 VMFault，别用它**：`lmslim/tools/w8a8_int8_tools.py` 的 autotune 入口（`_multiprocess_tuning` / `matmul_kernel_tuning_low/mid/high`）在 K100-AI 上跑第一个 shape 就 `KERNEL VMFault / HSA memory aperture violation`（autotune 内核越界），0 个桶落盘。改用**服务同款推理内核** `matmul_int8_infer`（底层是服务实际用的 `matmul_kernel`，能正常跑）手动 benchmark 一组候选配置、挑最快、写进 cache。benchmark 前先 smoke 测几个形状，确认 `matmul_int8_infer` 在目标卡不崩再全量跑。
2. **cache 文件是嵌套格式，别用工具 API 读改写（会写坏）**：`W8A8_{N}_{K}_{device_name}.json` 是**嵌套** `{ "N_K": { "M": {config} } }`。但工具 `get_triton_cache()` 返回**拍平**的 `{ "M_N_K": {config} }`——经它读再写回会写成扁平+嵌套混杂、文件损坏。正确做法：直接 `json.load` 嵌套文件、按 `nested[f"{N}_{K}"][str(M)]` 访问/写入、`json.dump` 写回嵌套。
3. **只补舍入表能产出的 M 桶**（运行时只有这些会命中，其余补了白补）：m≤16 逐值；17-64 取 4 倍数；65-160 取 8 倍数；161-199→160；200-479→256；480-959→512；960-2047→1024；2048-4095→2048；4096-5999→4096；≥6000→8192（封顶）。短/中 prefill 可达桶即 {2-16, 20,24,…,64, 72,80,…,160, 256, 512, 1024, 2048, 4096}，8192 是 prefill 封顶。诊断提示：客户报「某长度区间首字异常/只有某几个长度冷」时，把长度**映射到本表 m_ 桶**、看分界在 m_ 桶边界（如 4096/6000），**不是**对齐倍数（div16/32/64 解释不了——落同一 m_ 桶的长度行为一致，跨桶才变）；再用 §3.10·1 的独立探针长度曲线确认是真断档还是线性中段，别急着「补走廊」。
4. **补桶只消「命中调优配置 vs stock」的稳态差，不消首请求 JIT**：补桶让短/中 prefill 命中调优配置（稳态 ≥ stock 启发式），但**调优配置首用仍要 Triton JIT**——每次重启重付的 JIT 要靠**预编译**消（warmup 扩到覆盖这些 M 桶，或让 TRITON_CACHE_DIR 持久化/烤进镜像），**不是补桶消的**。别把补桶当「重启重付 JIT」的解。
5. **在空闲卡上 benchmark**（§5·11：别污染在跑的服务）；每个 (M,N,K) 用 `matmul_int8_infer` 计时取中位，挑最快配置写回。
6. **消首请求 JIT 的 warmup 配方（A'，补桶的配套）**：引擎自带 warmup 只覆盖一小撮 shape（有些 INT8 patch 链把 warmup 尺寸硬编码成 1024/2048/2230 → 只预编译 m_={1024,2048}），**其余 m_ 桶首请求仍付一次性 Triton JIT**（含 6000-16383 落 m_=8192 的大桶）。消法：server ready 后、放真实流量前，对**上表每个 m_ 桶（含 8192）**各发一个 prefill 请求，`input_ids` 用该桶长度 + **独立随机 token**（勿共享前缀，否则 radix 命中不真跑 prefill），把每个 m_ 桶的调优配置预编译掉；端口可配（env/argv）；做成 post-ready 钩子（等端口监听→跑 warmup→再标 healthy）。脚本 `scripts/w8a8_warmup_all_mbuckets.py`；跑完同长度首请求应 = 稳态。
7. **交付成一键升级包时，别让用户填深层内部路径**（用户反馈：`TRITON_JSON_DIR` 普通人不知道）：把「填充 cache + warmup」打成 tar.gz（cache/*.json + warmup 脚本 + apply 脚本 + README）。apply 脚本**自动探测 cache 目录**——接收**部署根目录**（用户知道解压到哪了），在其下找稳定标记子路径（本部署 `data/qwen38-27b-k100ai-int8-opt/cache/tp4`；launch 里 `TRITON_JSON_DIR=$PKG_DIR/data/…/cache/tp4`，`PKG_DIR`=解压根，稳定标记是中间那段、不随解压位置变），或任意含 `W8A8_*_<device>.json` 的目录；不传参时在 . / /data / /opt / $HOME 找。apply 前**自动备份**原 cache（`cache/tp4.bak_<ts>`）、幂等可重跑、导入后**校验桶数**。别要求用户知道 `TRITON_JSON_DIR`——他只知道解压根。

---

## 四、度量叠加：hipprof 包住 bench 抓 kernel 级耗时

bench 给出端到端吞吐/延迟，但**哪个 kernel 吃时间**要靠 hipprof。把 bench 整段用 `hipprof --hip-trace` 包裹：

```bash
hipprof --hip-trace --output-type 0 bash -c '
  vllm bench serve --model /data/models/GLM-5.2-INT4-w4a8 \
    --random-input-len 4096 --random-output-len 1024 \
    --dataset-name random --num-prompts 64 --max-concurrency 64 \
    --ignore-eos --served-model-name glm-5.2
'
```

产物（详见 `dcu-hipprof`）：
- `result_*.json`：Chrome Trace / Perfetto 可视化（HIP Runtime API 时间线）。
- `result_*.hipkernel.csv`：各 GPU kernel 调用次数 / 总耗时 / 平均 / 占比。
- `result_*.hiptrace.csv`：HIP API 调用统计。

> 也可以直接复用 `dcu-kernel-dev/scripts/tl_rmsnorm_benchmark.sh` 的思路（起服务 + 压测 + hipprof 包裹）做组合压测。

---

## 五、DCU 专属坑（bench 数字失真来源）

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| 1 | docker exec 用 `-c` 非登录 shell | rocblas/hipblaslt 未加载，性能掉 15–20% | 一律 `docker exec -d bash -ilc '...'`（见 `dcu-docker`） |
| 2 | 未挂 `/opt/hyhal` | DCU 不可见（DEVCOUNT=0） | 启动必带 `-v /opt/hyhal:/opt/hyhal:ro` |
| 3 | 挑卡用 `-g N` | 与框架选卡逻辑冲突 | 用 `HIP_VISIBLE_DEVICES=0,1,2,3` |
| 4 | `NCCL_NCHANNELS_PER_PEER` 预置=2 未清 | 通信路径非最优 | 进容器先 `unset NCCL_NCHANNELS_PER_PEER` |
| 5 | 冷启动计入延迟 | TTFT 首请求虚高 | `--num-warmups` 设几个 |
| 6 | 混用不同 input/output 长度 | 吞吐不可比 | 同次对比固定 `--random-input-len/--random-output-len` 或同数据集 |
| 7 | 直接信 `data.data.content` 解析 内部文档 | KeyError（旧文档写法） | 用 `data.content`（见 `内部文档-doc-extraction`） |
| 8 | `bench_serving --seed N`（0.5.12 海光 fork） | 客户端挂死（26 分钟无请求发出，服务日志停更） | A/B 一律不加 `--seed`（2026-08-21 <device-A> 实测） |
| 9 | 容器内 `ss -tlnp` 判端口 | 显示无监听但服务正常 | 健康检查用 `curl -s -m 5 http://127.0.0.1:<port>/get_server_info` |
| 10 | 杀挂死的 bench 客户端后不重启服务 | scheduler 卡死（长时间不处理请求，短 generate 超时） | 必须重启整个服务再测 |
| 11 | 生产服务在跑时跑 kernel 微基准（GEMM/norm 等） | GPU 争抢把基线抬高 20-30%（**幅度仅参考**），tuned/优化收益被夸大，A/B 结论方向性错误 | 微基准必须停服务在干净 GPU 上跑；或至少确认 `rocm-smi` 无其他进程占卡。端到端 bench 同理——对比两组时确保 GPU 独占 |
| 12 | TTFT 长度曲线复用同一段 base 文本 repeat 出多档长度 | 长档前缀命中 radix/prefix 缓存，假性偏快、曲线非单调（更长档比更短档还快） | 每档 input_ids 独立随机、或头部塞该长度独有 token（见 §3.10） |
| 13 | 思考模型（enable_thinking）数字与非思考基线直接比 | 思考/reasoning token 占输出预算、撑大 completion_tokens，"1024 输出"被思考吃掉一截，TPOT/吞吐与非思考口径不可直接比 | 把 thinking 当显式测量轴：thinking on/off 分开跑分开报；跨基线对比先对齐 thinking 状态并固定 reasoning_effort 档位（高档位增加等待与 token 消耗）。vLLM：serve 侧 `--default-chat-template-kwargs '{"enable_thinking":false,...}'` 定服务默认；客户端 bench 侧 `--chat-template-kwargs`（0.23.0+）或 `--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'` 逐请求覆盖 |
| 14 | 跨不同 DCU 卡型/机型（测试机卡 vs 现场生产卡）直接比**绝对** TTFT/吞吐 | 硬件不同，同配置同代码绝对 TTFT 系统性差约 1.5×（量级警示，随机型/DTK 版本变，非固定基线），直接比会误判「优化无效/退步」 | 跨卡型/跨机型对比只比**形状**（长度曲线是否线性、有无断档、各档相对排序），不比绝对数；要对绝对值先在同机型上 A/B。同一机型内不同启动参数才可直接比绝对 TTFT |

---

## 六、交叉引用

- 跑前环境摸底 → `dcu-env`；容器执行纪律 → `dcu-docker`；部署启动参数 → `dcu-vllm-deploy` / `dcu-inference-infra`。
- 算子/kernel 级度量 → `dcu-hipprof`（父，路由 5 子 skill）/ `dcu-perf-profiling`（rocprof 系）。
- TileLang 算子 bench 组合示例（hipprof 包 vllm bench）→ `dcu-kernel-dev/scripts/tl_rmsnorm_benchmark.sh`。
- 性能优化闭环/留痕 → `dcu-op-optimization`；度量口径 → `dcu-perf-profiling`。
- 跨框架自动搜索最优部署命令（同模型比 SGLang/vLLM/TRT-LLM 吞吐）→ 通用互补 skill `llm-serving-auto-benchmark`（已整合进本仓库，注意它依赖 dcu-inference-cookbook，zip 未打包需自备）。

## 使用须知
- 方法仅供参考，不保证修复；先核对设备型号(gfx)+驱动/DTK 版本+报错原文。
- **不存会被视为固定基线的性能数字**：token 数、吞吐、延迟绝对数以项目侧留痕为准，本页只沉淀方法；文中的经验量级数字仅作坑的定性警示，非本环境实测基线。
- 高危操作谨慎，优先测试机验证。
