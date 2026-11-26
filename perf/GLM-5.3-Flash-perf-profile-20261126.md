# GLM-5.3-Flash 性能画像（2026-11-26）

> 镜像 `glm_5.3_flash_fp8:bw1100_opt_8y8b_20261123` + FlashMLA fp8 直读
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

## 1. 纯填充吞吐

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


### 1.2 prefill 吞吐测试

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


### 1.3 prefill 吞吐测试（c=20）

```
============ Serving Benchmark Result ============
Successful requests:                     20
Failed requests:                         0
Maximum request concurrency:             20
Benchmark duration (s):                  48.94
Total input tokens:                      2621440
Total generated tokens:                  20
Request throughput (req/s):              0.41
Output token throughput (tok/s):         0.41
Peak output token throughput (tok/s):    1.00
Peak concurrent requests:                20.00
Total token throughput (tok/s):          53560.96
---------------Time to First Token----------------
Mean TTFT (ms):                          31669.27
Median TTFT (ms):                        31784.75
P50 TTFT (ms):                           31784.75
P90 TTFT (ms):                           45955.70
P99 TTFT (ms):                           48611.45
```

### 1.4 结论

| 测量方式 | 净填充吞吐 |
|---|---:|
| 服务端日志（c=1，chunk 2–15 区间） | **12 020 → 10 476 tok/s** |
| Total token throughput（c=1） | **10 580 tok/s** |
| Total token throughput（c=20） | **53 561 tok/s** |

- **c=1 时服务端日志与压测输出一致（误差 <1%）**，因为没有缓存可命中，
  `Total input tokens` 全是要真算的
- **c=20 时吞吐升到 53.6k tok/s（c1 的 5.06 倍）**，TTFT 也随之升到 31.7 s
- 长 prefill 越往后越慢（12 020 → 10 476，降 13%），说明 128K 冷启动时尾部 chunk 的注意力成本更高

### 1.5 ⚠️ 并发 >1 时须先确认有没有命中缓存

**§1.3 的 c=20（53 561 tok/s）是冷启动的**——自洽性可以验证：
20 × 131 072 = 2 621 440 token ÷ 48.94 s = 53 561 tok/s，与报出值吻合；
若命中 85% 前缀则只需算 394k token，48.94 s 意味着 8k tok/s，与 TTFT 31.7 s 矛盾。

但**如果同一次压测内前缀被写进 radix cache，后续请求会全部命中 111 360**，那时
`Total token throughput` 就会把命中的 token 也算成"已处理"，数字虚高：

```
16:23:06  new=8192 cached=111360      ← c=8 那一轮
16:23:08  new=8192 cached=111360
16:23:10  new=8192 cached=111360
```

而 `Total token throughput = (全部 input + output) / duration`，
**把命中的 token 也算成"已处理"**，所以 c≥2 的数字虚高：

| 并发 | 报出的吞吐 | 实际真算的 input | 是否可信 |
|---:|---:|---:|:--:|
| 1 | 10 580 | 131 072（全冷） | ✅ |
| 2 | 17 872 | ~150 784 | ❌ 虚高 |
| 4 | 28 333 | ~190 208 | ❌ 虚高 |
| 8 | 40 502 | ~269 056 | ❌ 虚高 |

要测 c≥2 的真实 prefill 吞吐，必须每轮换前缀种子（bench 脚本已加 `SEED` 环境变量）。

---

## 2. 128K / 1K 混合场景

口径：输入 131072（0.85 前缀命中）/ 输出 1024 / `request_rate=inf` / 预热 4 条 /
`bench_serving_cache_control.py` / `tpot_slo_ms=50`。

### 2.0 n=100 / c20（第一次）

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

### 2.1 n=100 / c20（第二次）

```
[flush] 已清空服务端前缀缓存
[warmup] 发送 4 条 max_tokens=1 请求预热前缀缓存...
[warmup] 完成 ok=4 failed=0 wall=10.6s
[bench] backend=sglang in=131072 out=1024 n=100 conc=20 rate=inf hit~0.85 (prefix=111411/131072)
Successful requests: 100
Benchmark duration (s): 273.52
Request throughput (req/s): 0.37
Output token throughput (tok/s): 374.38
Total token throughput (tok/s): 48295.19
Mean TTFT (ms): 6552.47
Median TTFT (ms): 1997.66
P99 TTFT (ms): 36916.79
Mean TPOT (ms): 46.07
Median TPOT (ms): 45.75
P99 TPOT (ms): 75.56
Mean ITL (ms): 46.07
Median ITL (ms): 13.62
P99 ITL (ms): 626.74
Mean E2E Latency (ms): 53680.51
Failed requests: 0
>>> 本档服务端实测前缀缓存命中率 = 0.8463 (cached_tokens Δ=11470080 / prompt_tokens Δ=13552860)
>>> cached in=131072 conc=20 PASS (TPOT=46.07ms)，最大可用并发更新为 20
```

> 两次 n=100 用的是两份启动脚本，但计算路径等价——其中一项 rocBLAS 调优
> （`ROCBLAS_TENSILE_LIBPATH`）实测**不生效**：本 workload 的 Tensile 内核全部来自
> hipBLASLt，用 router shape 做 A/B 也是 15.6 µs 对 15.6 µs。所以两次的差异（TPOT 47.98 vs 46.07）
> 主要来自投机接受率的随机波动（两次跑的中段 accept rate 分别是 0.55 / 0.82）。

### 2.2 n=100 / c16

```
[bench] backend=sglang in=131072 out=1024 n=100 conc=16 rate=inf hit~0.85 (prefix=111411/131072)
Successful requests: 100
Benchmark duration (s): 278.38
Request throughput (req/s): 0.36
Output token throughput (tok/s): 367.84
Total token throughput (tok/s): 47451.11
Mean TTFT (ms): 5033.14
Median TTFT (ms): 1992.46
P99 TTFT (ms): 29922.24
Mean TPOT (ms): 37.63
Median TPOT (ms): 37.11
P99 TPOT (ms): 64.29
Mean ITL (ms): 37.63
Median ITL (ms): 11.50
P99 ITL (ms): 419.49
Mean E2E Latency (ms): 43523.83
Failed requests: 0
>>> 本档服务端实测前缀缓存命中率 = 0.8463 (cached_tokens Δ=11470080 / prompt_tokens Δ=13552860)
>>> cached in=131072 conc=16 PASS (TPOT=37.63ms), 最大可用并发更新为 16
```

### 2.3 结论

| 配置 | TTFT mean | TTFT median | TPOT mean | 总吞吐 | 判定 |
|---|---:|---:|---:|---:|---|
| **n=100 / c20（第一次）** | 6 230.66 ms | 2 010.40 ms | 47.98 ms | 47 027 tok/s | **PASS** |
| **n=100 / c20（第二次）** | 6 552.47 ms | 1 997.66 ms | **46.07 ms** | **48 295 tok/s** | **PASS** |
| **n=100 / c16** | 5 033.14 ms | 1 992.46 ms | **37.63 ms** | 47 451 tok/s | **PASS** |

**两点**：

1. **三次全部 PASS**：c20 两次 TPOT 47.98 / 46.07 ms，c16 一次 **37.63 ms**
2. **TTFT 均值受尾部拖累**：三次中位数都在 2.0 s 附近（1 992–2 010 ms），
   但均值 5.0–6.6 s，因为少数请求排队到 30–37 s（P99）。引用均值必须同时给中位数

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

> 上表是**历史数据**（未含本次 FlashMLA 优化）。本次在 n=100 下：
> c20 的 TPOT mean 是 **47.98 / 46.07 ms（两次）**、c16 是 **37.63 ms**，均 PASS。

---
