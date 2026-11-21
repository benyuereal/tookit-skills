# dcu-llm-benchmark — DCU 大模型推理性能测试

> 海光 DCU 上端到端推理**吞吐 / 延迟压测**的方法学与纪律。覆盖 vLLM `vllm bench serve`、SGLang `bench_serving` 的完整参数与版本差异、hipprof 抓 kernel 级耗时、以及 DCU 专属 bench 数字失真坑。
> 不存性能基线/逐行日志——只沉淀"怎么正确跑出可信 bench 数据"的方法（数字以你项目侧留痕为准）。

## 这个 skill 解决什么
- 想跑 vLLM / SGLang 的在线服务压测，看吞吐(requ/s)、首 token 延迟(TTFT)、每 token 延迟(TPOT)、token 间延迟(ITL)、端到端延迟(E2EL)、goodput。
- 担心 bench 数字不准（容器执行方式、挑卡、预置环境变量都会让数据失真）。
- 想知道 `vllm bench serve` / `sglang bench_serving` 一堆选项怎么用、默认值/版本差异是啥。

## 版本覆盖（主 SKILL.md 已实测 diff 对齐）
- **vLLM**：0.15.1 / 0.18.1 / 0.21.0 / 0.23.0 四版本差异对照（选项名/枚举/新增项）。
- **SGLang**：0.5.10 / 0.5.12 两版本差异对照。
- 主体以 **vLLM 0.18.1** / **SGLang 0.5.12** 为准，所有差异见主 SKILL.md §2.2.1 / §2.5 / §3.2.1。

## 快速开始（vLLM 0.18.1，GLM-5.2 INT4 示例）

```bash
vllm bench serve \
  --model /data/models/GLM-5.2-INT4-w4a8 \
  --random-input-len 4096 --random-output-len 1024 \
  --port 8000 --dataset-name random \
  --num-prompts 64 --max-concurrency 64 \
  --ignore-eos --served-model-name glm-5.2
```

> ⚠ 参数是 `--served-model-name`（三版本/四版本均无 `--served-model` 这个写法）。

## SGLang 等效（0.5.12，Kimi-K2.6 + random-ids 实战模板）

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

> ⚠ SGLang **没有 `--ignore-eos`**，对应是 `--disable-ignore-eos`（默认即忽略 EOS，与 vLLM 相反）。
> ⚠ **DeepSeek 类模型必须加 `--tokenize-prompt`**：`random-ids` 会 decode 后再 encode，DS tokenizer 长度膨胀 2–3 倍污染输入长度假设；Kimi 类影响小。跨模型比较须统一长度口径。详见主 SKILL.md §3.3。

## 跑 bench 前的纪律（否则数字失真）
1. 先定镜像，基于该镜像起的专用容器跑；编译/运行都在容器内。
2. `dcu-env` 摸清硬件拓扑/驱动/DTK/gfx。
3. 容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`（否则 DCU 不可见）。
4. **docker exec 必须 `bash -ilc`**（非 `-c`），否则 rocblas/hipblaslt 不加载，性能掉 15–20%。
5. 挑卡用 `HIP_VISIBLE_DEVICES=0,1,2,3`，不靠 `-g N`。
6. 进容器先 `unset NCCL_NCHANNELS_PER_PEER`（镜像常预置=2）。

## 关键指标读法
- 饱和吞吐：`--request-rate inf` + `--max-concurrency N`。
- 给定负载延迟分布：`--request-rate R`（Poisson）+ `--percentile-metrics ttft,tpot,itl`。
- 排除冷启动：`--num-warmups` 设几个。

## 度量叠加（看哪个 kernel 慢）
用 `hipprof --hip-trace` 包住 bench，产出 `.hipkernel.csv`（各 kernel 耗时占比）、`.hiptrace.csv`、Chrome Trace JSON。

## 完整参数
- vLLM：主 SKILL.md §2.2 选项表（以 0.18.1 为准）+ §2.2.1/§2.5 四版本差异。
- SGLang：主 SKILL.md §3.2 选项表（以 0.5.12 为准）+ §3.2.1 两版本差异。
- 原始 `--help` 输出见用户侧：`<local_bench_help_0181>`(0.18.1)、`<local_bench_help_0151>`(0.15.1)、`<local_bench_help_0210>`(0.21.0)、`<local_bench_help_0230>`(0.23.0)、`<local_bench_help_0512>`(SGLang 0.5.12)、`<local_bench_help_0510>`(SGLang 0.5.10)。

## 交叉引用
- 环境摸底 → `dcu-env`；容器执行纪律 → `dcu-docker`；部署启动 → `dcu-vllm-deploy` / `dcu-inference-infra`
- kernel 级度量 → `dcu-hipprof`（父，路由 5 子 skill）/ `dcu-perf-profiling`
- 算子 bench 组合示例 → `dcu-kernel-dev/scripts/tl_rmsnorm_benchmark.sh`
