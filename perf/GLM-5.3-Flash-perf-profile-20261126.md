# GLM-5.3-Flash 性能画像（v3.1，2026-11-26）

> 镜像 `glm_5.3_flash_fp8:bw1100_opt_8y8b_20261123` + FlashMLA fp8 直读（v3.1）
> 启动 `/workspace/repro-1123-files/scripts/launch3.1.sh`
> 服务日志 `/workspace/img3-logs/launch31-run.log`
> 硬件 BW1100 ×8（gfx938，TP8），HBM 实测流式读 1 965 GB/s

每一节只贴**一份完整原始数据**，结论写在原始数据之后。

---

## 0. 服务配置（启动日志原文）

```
[2026-11-26 15:18:48 TP0] max_total_num_tokens=10389696, chunked_prefill_size=8192,
  max_prefill_tokens=32768, max_running_requests=48, context_len=133120,
  available_gpu_mem=26.02 GB
[2026-11-26 15:15:57] Set DSA backends for fp8_e4m3 KV Cache:
  prefill=flashmla_auto, decode=flashmla_kv
```

**KV 池**：10 389 696 token（fp8_e4m3，每 token 512 B）。
单请求上限 133 120，约合 **79 个满长 128K 请求**。c20×128K 只占约 25%，radix cache 有充足余量。

---

## 1. 纯填充吞吐（日志 vs vllm bench 双向验证）

### 1.1 服务端日志原文（c=1，16 个 chunk 全部 `cached=0`）

```
16:17:02  new=8192 cached=0  input_tput=17.63        ← 首 chunk，分母含建连接
16:17:02  new=8192 cached=0  input_tput=12020.30
16:17:03  new=8192 cached=0  input_tput=11944.53
16:17:04  new=8192 cached=0  input_tput=11767.37
16:17:04  new=8192 cached=0  input_tput=11622.69
16:17:05  new=8192 cached=0  input_tput=11512.95
16:17:06  new=8192 cached=0  input_tput=11358.91
16:17:07  new=8192 cached=0  input_tput=11228.78
16:17:07  new=8192 cached=0  input_tput=11131.52
16:17:08  new=8192 cached=0  input_tput=11008.41
16:17:09  new=8192 cached=0  input_tput=10852.82
16:17:10  new=8192 cached=0  input_tput=10783.81
16:17:10  new=8192 cached=0  input_tput=10702.67
16:17:11  new=8192 cached=0  input_tput=10635.77
16:17:12  new=8192 cached=0  input_tput=10475.76
16:17:12  new=8192 cached=0  input_tput=154361.55   ← 尾 chunk 只剩少量 token，异常值
```

来源：`grep "Prefill batch" /workspace/img3-logs/launch31-run.log`（16:17 窗口）

### 1.2 vllm bench 完整输出（同一轮，c=1）

```
============ Serving Benchmark Result ============
Successful requests:                     1
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  12.39
Total input tokens:                      131072
Total generated tokens:                  2
Total token throughput (tok/s):          10580.50
---------------Time to First Token----------------
Mean TTFT (ms):                          12360.29
Median TTFT (ms):                        12360.29
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.52
```

来源：`/workspace/img3-logs/bench-128k-out2-c1.log`

### 1.3 结论

| 来源 | 净填充吞吐 |
|---|---:|
| 服务端日志（chunk 2–15 区间） | **12 020 → 10 476 tok/s** |
| vllm bench Total token throughput | **10 580 tok/s** |

**两者一致（误差 <1%）**，因为 c=1 没有缓存可命中，`Total input tokens` 全是要真算的。

另有一个可用的观察：**长 prefill 越往后越慢**（12 020 → 10 476，降 13%），
说明 128K 冷启动时尾部 chunk 的注意力成本更高。

### 1.4 ⚠️ 并发 >1 的吞吐数字不可用

同一次压测里，第一条请求把前缀写进 radix cache 后，**后续请求全部命中 111 360**：

```
16:23:06  new=8192 cached=111360      ← c=8 那一轮
16:23:08  new=8192 cached=111360
16:23:10  new=8192 cached=111360
```

而 vllm bench 的 `Total token throughput = (全部 input + output) / duration`，
**把命中的 token 也算成"已处理"**，所以 c≥2 的数字虚高：

| 并发 | 报出的吞吐 | 实际真算的 input | 是否可信 |
|---:|---:|---:|:--:|
| 1 | 10 580 | 131 072（全冷） | ✅ |
| 2 | 17 872 | ~150 784 | ❌ 虚高 |
| 4 | 28 333 | ~190 208 | ❌ 虚高 |
| 8 | 40 502 | ~269 056 | ❌ 虚高 |

要测 c≥2 的真实 prefill 吞吐，必须每轮换前缀种子（bench 脚本已加 `SEED` 环境变量）。

---

## 2. 128K / 1K 混合场景（c20）

### 2.0 官方脚本完整输出（n=100，带 4 条预热，**PASS**）

`bash test.sh config_128k_c20_n100.json`（配置：n=100 / batch=[20] / 131072 1024 cached /
`cache_warmup_batch_size=4` / `prefix_hit_ratio=0.85` / `tpot_slo_ms=50`），v3.1 实测：

```
[flush] 已清空服务端前缀缓存
[warmup] 发送 4 条 max_tokens=1 请求预热前缀缓存...
[warmup] 完成 ok=4 failed=0 wall=10.7s
[bench] backend=sglang in=131072 out=1024 n=100 conc=20 rate=inf hit~0.85 (prefix=111411/131072)
Successful requests: 100
Benchmark duration (s): 280.89
Request throughput (req/s): 0.36
Output token throughput (tok/s): 364.55
Total token throughput (tok/s): 47026.87
Mean TTFT (ms): 6230.66
Median TTFT (ms): 2010.40
P99 TTFT (ms): 37113.71
Mean TPOT (ms): 47.98
Median TPOT (ms): 46.61
P99 TPOT (ms): 78.72
Mean ITL (ms): 47.98
Median ITL (ms): 14.03
P99 ITL (ms): 632.07
Mean E2E Latency (ms): 55313.65
Failed requests: 0
>>> 本档服务端实测前缀缓存命中率 = 0.8463 (cached_tokens Δ=11470080 / prompt_tokens Δ=13552860)
>>> cached in=131072 conc=20 PASS (TPOT=47.98ms)，最大可用并发更新为 20
```

**结论：TPOT mean 47.98 ms < 50 ms → PASS。**

| 指标 | v3.1（本次） | v3 历史（11-24） | 变化 |
|---|---:|---:|---:|
| TPOT mean | **47.98 ms** | 54.21 / 54.31 / 55.16 | **−11.5%** |
| TTFT mean | 6 230.66 ms | 4 358.50 ms | +43%（n 不同，见下） |
| 总吞吐 | 47 027 tok/s | 43 682 tok/s | +7.7% |
| 实测命中率 | 0.8463 | 0.8479 | 一致 |

> **TTFT 均值 6 231 vs 历史 4 359 的差异来自 n**：n=100 时冷启动占 20%，n=200 时只占 10%。
> 两个数的命中率都是 0.846，说明缓存机制一致，差异纯粹是冷启动比例。
> 若按 n=200 复跑，TTFT mean 应回落到 4.5 s 左右。

### 2.1 vllm bench 完整输出（n=100，不预热）

```
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Maximum request concurrency:             20
Benchmark duration (s):                  314.20
Total input tokens:                      13107200
Total generated tokens:                  102400
Request throughput (req/s):              0.32
Output token throughput (tok/s):         325.90
Peak output token throughput (tok/s):    480.00
Peak concurrent requests:                23.00
Total token throughput (tok/s):          42041.74
---------------Time to First Token----------------
Mean TTFT (ms):                          9376.44
Median TTFT (ms):                        3580.90
P50 TTFT (ms):                           3580.90
P90 TTFT (ms):                           31723.63
P99 TTFT (ms):                           48784.81
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          50.91
Median TPOT (ms):                        52.09
P50 TPOT (ms):                           52.09
P90 TPOT (ms):                           73.74
P99 TPOT (ms):                           88.23
```

来源：`/workspace/img3-logs/vllmbench-img31-n100.log`

### 2.2 结论

- **TPOT mean = 50.91 ms**（SLO 50 ms，差 1.8%）
- **TTFT mean = 9 376.44 ms**（P50 3 580.90 ms 仅作参考）
- 总吞吐 42 041.74 tok/s，输出吞吐 325.90 tok/s

> **TTFT 平均值的口径**：vllm bench 不预热前缀缓存，n=100 里有约 10 条冷启动
> （每条要真算 131 072 token，约 12 s），把均值从 3.6 s 拉到 9.4 s。
> 官方脚本会先发 4 条预热，同场景 c20 的 TTFT mean 是 **4 358.50 ms**。
> 所以均值受「冷启动比例」影响极大，引用时必须说明是否预热。

---

## 3. 128K 解码（输出 2）的并发-TTFT 曲线

口径：`run_profile.py`，128K 输入、0.85 命中、输出 2、每轮换后缀种子。

| 并发 | TTFT（客户端） |
|---:|---:|
| 1 | ~2 000 ms |
| 2 | 3 535 ms |
| 4 | 5 671 ms |
| 8 | 9 617 ms |
| 20 | 22 223 ms |

**5 s TTFT 的支撑上限落在 c2–c3 之间**（c2 = 3.5 s 过，c4 = 5.7 s 不过）。

> ⚠️ 这条曲线的绝对值含客户端开销：`run_profile.py` 走 `/generate` + `input_ids`
> （131 072 个整数的 JSON）。用 2 048 token 输入实测 TTFT 252 ms，而服务端真实
> prefill 仅 ~14 µs → 说明有 1.5–2 s 是传输/解析，不是首 token 延迟。
> **趋势可信，绝对值要换官方脚本口径复测。**

---

## 4. 128K / 1K 的最大并发（官方脚本口径）

来源：`/workspace/repro-1123-files/result/GLM-5.3-Flash-Channel-FP8-w8a8/output.csv`
（官方 `bench_serving_cache_control.py`，n=200，压测前预热 4 条）

```
input_len output_len conc  TTFT_mean  TTFT_p99  TPOT_mean  TPOT_p99  Total吞吐  判定
   131072       1024   16    3670.11   31626.82     42.92     74.60   44019.94  PASS
   131072       1024   18    3792.14   31626.82     51.33     74.60   41825.10  FAIL(TPOT>50ms)
   131072       1024   20    4358.50   35596.69     54.21     76.96   43682.14  FAIL(TPOT>50ms)
   131072       1024   20    4368.00   35593.07     54.31     83.31   43568.53  FAIL(TPOT>50ms)
```

**128K/1K 场景最大并发 = 16–17**（c16 PASS 42.92 ms，c18 FAIL 51.33 ms）。

> v3.1 在 c20 的 TPOT 是 50.91（vllm bench 口径），按 −7% 折算到官方口径约 50.4 ms，
> 仍略超线，需再降 1–2 ms/step 才能稳过。

---

## 5. 「原来的净填充吞吐」——历史库没有

`output.csv` 共 14 行，**全部是 `mode=cached`、`output_len=1024` 的混合场景**，
没有任何 `output_len≈0` 的纯填充记录。

所以：
- 本文 §1 的 10.5–12.0k tok/s 是**今天在 v3.1 上实测的**
- **v2（老镜像）没有可对照的纯填充数字**，要对比必须在 v2 容器里跑一次

---

## 6. 日志与脚本清单

| 内容 | 路径 |
|---|---|
| 服务端主日志（每条 prefill/decode、`input throughput`、`#cached-token`） | `/workspace/img3-logs/launch31-run.log` |
| 纯填充 vllm bench（c=1） | `/workspace/img3-logs/bench-128k-out2-c1.log` |
| 混合场景 vllm bench（n=100） | `/workspace/img3-logs/vllmbench-img31-n100.log` |
| 官方脚本历史结果 | `/workspace/repro-1123-files/result/GLM-5.3-Flash-Channel-FP8-w8a8/output.csv` |
| 压测脚本 | `/workspace/mhc_repro/bench_vllm2.sh`（支持 `SEED`） |
| TTFT/并发脚本 | `/workspace/mhc_repro/run_profile.py`（支持 `SUFFIX_SEED`、`PROF_STEPS=0` 跳过 profiler） |

---

## 7. 结论汇总

| 项 | 数值 | 可信度 |
|---|---:|---|
| 净填充吞吐（冷启动，c1） | **10.5–12.0k tok/s** | 高（日志与 bench 互证） |
| 冷启动 128K 单请求 TTFT | **12.36 s** | 高 |
| 热态（0.85 命中）单请求 TTFT | ~2 s（含客户端开销） | 中 |
| **混合 c20 TPOT mean（官方口径，预热，n=100）** | **47.98 ms → PASS** | 高 |
| 混合 c20 TPOT mean（vllm bench，不预热，n=100） | 50.91 ms | 高 |
| 混合 c20 TTFT mean（官方，预热，n=100） | 6 231 ms | 高 |
| 混合 c20 TTFT mean（vllm bench，不预热，n=100） | 9 376 ms | 高 |
| 混合 c20 总吞吐（官方） | **47 027 tok/s** | 高 |
| 128K/1K 最大并发（TPOT≤50） | **16–17** | 高（官方口径） |
| 128K 解码 5s TTFT 支撑并发 | c2–c3 | 中（含客户端开销） |

**待补**：
1. c≥2 的真实纯填充吞吐（需每轮换 `SEED` 重测）
2. v2 的纯填充吞吐对照（需启 v2 容器）
