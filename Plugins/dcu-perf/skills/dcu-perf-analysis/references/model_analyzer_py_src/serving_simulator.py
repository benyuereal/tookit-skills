

import math
import os
import random
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .model_analyzer import model_analyzer, run_cli
except ImportError:  # 允许 `python serving_simulator.py` 直接运行
    from model_analyzer import model_analyzer, run_cli


@dataclass
class ServingConfig:
    """单池 (mixed: prefill 与 decode 共享同一组实例) 服务仿真输入。"""
    # ---- 核心新参数 ----
    req_rate: float                       # QPS, 到达率
    input_len: int = 4096                 # 每请求输入 (prompt) 长度
    output_len: int = 256                 # 每请求输出 (decode) token 数

    # ---- 模型 / 硬件 / 并行 (透传给 model_analyzer) ----
    model: str = "deepseek_v3"
    gpu: str = "NMZ"
    tp: int = 8
    dp: int = 1
    pp: int = 1
    ep: int = 8
    cp: int = 1
    nodes: int = 1
    dp_attention: bool = False
    # 拓扑带宽 (跟随 model_analyzer 默认; 显式设置覆盖)
    gpus_per_node: int = 8
    nvlink_domain_size: int = 8
    intra_node_bw_gbs: float = 140.0
    inter_node_bw_gbs: float = 50.0
    kv_cache_dtype: str = "fp8"
    attention_compute_dtype: str = "fp8"
    fuse_moe: bool = True
    # 任意额外 model_analyzer kwargs (mtp_param / sequence_parallel / num_images ...)
    analyzer_kwargs: dict = field(default_factory=dict)

    # ---- 仿真控制 (sglang 调度语义, §2) ----
    num_requests: int = 500               # 仿真请求数 (越多 p99 越稳)
    max_running_requests: int = 256       # 准入闸门: running 满则排队
    chunked_prefill_size: int = 8192      # 单次 prefill token 预算 (限制 prefill batch 宽度)
    arrival: str = "poisson"              # 'poisson' | 'uniform'
    warmup: int = -1                      # 丢弃前 N 个完成请求; -1 = 自动 (10%)
    ctx_bucket: int = 512                 # decode ctx_len 离散化粒度 (提高缓存命中, §7.1)
    seed: int = 0                         # 到达过程随机种子 (poisson 复现)

    # 失控保护: 仿真总迭代上限 (饱和时队列发散, 防止死循环)
    max_iters: int = 0                    # 0 = 自动 (num_requests * (output_len+4) * 3)


@dataclass
class ServingMetrics:
    """仿真输出指标 (设计文档 §4 / §5)。"""
    req_rate: float = 0.0
    ttft_ms_mean: float = 0.0
    ttft_ms_p99: float = 0.0
    tpot_ms_mean: float = 0.0
    tpot_ms_p99: float = 0.0
    itl_ms_mean: float = 0.0
    itl_ms_p99: float = 0.0
    achieved_qps: float = 0.0             # 实际完成速率
    decode_bs_mean: float = 0.0
    decode_bs_max: int = 0
    saturated: bool = False               # achieved_qps < req_rate → 队列发散
    completed: int = 0                    # 计入统计的请求数 (已剔除 warmup)
    sim_time_ms: float = 0.0              # 仿真总时长 (clock)

    def as_row(self) -> dict:
        return {
            "req_rate": round(self.req_rate, 4),
            "ttft_mean_ms": round(self.ttft_ms_mean, 2),
            "ttft_p99_ms": round(self.ttft_ms_p99, 2),
            "tpot_mean_ms": round(self.tpot_ms_mean, 3),
            "tpot_p99_ms": round(self.tpot_ms_p99, 3),
            "itl_mean_ms": round(self.itl_ms_mean, 3),
            "itl_p99_ms": round(self.itl_ms_p99, 3),
            "achieved_qps": round(self.achieved_qps, 4),
            "decode_bs_mean": round(self.decode_bs_mean, 1),
            "decode_bs_max": self.decode_bs_max,
            "saturated": self.saturated,
            "completed": self.completed,
        }



@contextmanager
def _silenced():
    """同时屏蔽 fd 1/2 (C++ printf) 与 sys.stdout/stderr (Python print)。"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_fd1, saved_fd2 = os.dup(1), os.dup(2)
    old_out, old_err = sys.stdout, sys.stderr
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stdout.close()
        sys.stderr.flush(); sys.stderr.close()
        sys.stdout, sys.stderr = old_out, old_err
        os.dup2(saved_fd1, 1)
        os.dup2(saved_fd2, 2)
        os.close(saved_fd1); os.close(saved_fd2); os.close(devnull)


class _KernelCache:
    """
    把 model_analyzer 当 building block 反复调用, 按 (kind, bs, len_bucket) 缓存。

      T_prefill(bs, in_len)  -> prefill_latency_ms (整批 prefill 耗时)
      T_decode_step(bs, ctx) -> decode_latency_ms (单步 decode 延迟)

    内核结果只依赖 (bs, 序列长度) 与固定的模型/并行/硬件配置, 故可安全缓存。
    """

    def __init__(self, cfg: ServingConfig):
        self.cfg = cfg
        self._prefill: Dict[Tuple[int, int], float] = {}
        self._decode: Dict[Tuple[int, int], float] = {}
        self.calls = 0  # 实际穿透到 C++ 的次数 (诊断缓存命中率)

    def _base_kwargs(self, bs: int, seq_len: int) -> dict:
        cfg = self.cfg
        kw = dict(
            prefill_tp=cfg.tp, prefill_dp=cfg.dp, prefill_pp=cfg.pp,
            prefill_ep=cfg.ep, prefill_cp=cfg.cp,
            decode_tp=cfg.tp, decode_dp=cfg.dp, decode_pp=cfg.pp,
            decode_ep=cfg.ep, decode_cp=1,
            prefill_nodes=cfg.nodes, decode_nodes=cfg.nodes,
            batch_size=bs, seq_len=seq_len, output_length=1,
            chunk_prefill_size=max(cfg.chunked_prefill_size, seq_len),
            gpu=cfg.gpu, model=cfg.model,
            dp_attention=cfg.dp_attention,
            gpus_per_node=cfg.gpus_per_node,
            nvlink_domain_size=cfg.nvlink_domain_size,
            intra_node_bw_gbs=cfg.intra_node_bw_gbs,
            inter_node_bw_gbs=cfg.inter_node_bw_gbs,
            kv_cache_dtype=cfg.kv_cache_dtype,
            attention_compute_dtype=cfg.attention_compute_dtype,
            fuse_moe=cfg.fuse_moe,
            generate_csv=False, verbose=False, skip_memory_check=True,
        )
        kw.update(cfg.analyzer_kwargs)
        return kw

    def _run(self, bs: int, seq_len: int) -> dict:
        self.calls += 1
        with _silenced():
            return model_analyzer(**self._base_kwargs(bs, seq_len))

    def t_prefill(self, bs: int, in_len: int) -> float:
        """整批 prefill (bs 个请求, 每个 in_len token) 的耗时 (ms)。"""
        key = (bs, in_len)
        v = self._prefill.get(key)
        if v is None:
            v = float(self._run(bs, in_len)["prefill_latency_ms"])
            self._prefill[key] = v
        return v

    def t_decode_step(self, bs: int, ctx_len: int) -> float:
        """running batch=bs, 平均上下文=ctx_len 时单步 decode 延迟 (ms)。"""
        bucket = max(self.cfg.ctx_bucket,
                     int(math.ceil(ctx_len / self.cfg.ctx_bucket)) * self.cfg.ctx_bucket)
        key = (bs, bucket)
        v = self._decode.get(key)
        if v is None:
            v = float(self._run(bs, bucket)["decode_latency_ms"])
            self._decode[key] = v
        return v


# ---------------------------------------------------------------------------
# 请求对象与到达过程
# ---------------------------------------------------------------------------

class _Req:
    __slots__ = ("idx", "arrival", "input_len", "output_len",
                 "ctx", "gen", "ttft", "itls", "first_token_clock", "finish_clock")

    def __init__(self, idx: int, arrival: float, input_len: int, output_len: int):
        self.idx = idx
        self.arrival = arrival          # 到达时刻 (ms)
        self.input_len = input_len
        self.output_len = output_len
        self.ctx = input_len            # 当前上下文长度 (decode 每步 +1)
        self.gen = 0                    # 已生成 token 数 (prefill 出首 token 后 =1)
        self.ttft: float = 0.0
        self.itls: List[float] = []     # 各 decode 步延迟 (inter-token latency)
        self.first_token_clock: float = 0.0
        self.finish_clock: float = 0.0  # 末 token 生成时刻 (吞吐统计用)


def _gen_arrivals(cfg: ServingConfig) -> List[float]:
    """生成到达时刻序列 (ms), 单调递增。"""
    rng = random.Random(cfg.seed)
    rate_per_ms = cfg.req_rate / 1000.0   # QPS → 每 ms 到达数
    times: List[float] = []
    t = 0.0
    for _ in range(cfg.num_requests):
        if cfg.arrival == "uniform":
            t += 1.0 / rate_per_ms
        elif cfg.arrival == "poisson":
            # 指数间隔: -ln(U)/rate
            t += -math.log(1.0 - rng.random()) / rate_per_ms
        else:
            raise ValueError(f"arrival={cfg.arrival!r} 非法 (poisson|uniform)")
        times.append(t)
    return times



def simulate_serving(cfg: ServingConfig, kernel: Optional[_KernelCache] = None) -> ServingMetrics:
    """
    单池 iteration 级 DES。返回 TTFT / TPOT / ITL 的 mean/p99 + 饱和判定。

    照搬 sglang `get_next_batch_to_run`:每个 iteration 一次仲裁,
    Prefill 优先 (能组 prefill 就跑 prefill, 否则跑 decode), 连续批 decode 全体 +1 token,
    max_running_requests 作准入闸门 (满则排队, 等待计入 TTFT)。
    """
    if cfg.req_rate <= 0:
        raise ValueError("req_rate 必须 > 0")
    kernel = kernel or _KernelCache(cfg)

    arrivals = _gen_arrivals(cfg)
    pending = [_Req(i, arrivals[i], cfg.input_len, cfg.output_len)
               for i in range(cfg.num_requests)]
    next_arrival = 0           # pending 中下一个未到达请求的下标
    waiting: deque = deque()   # 已到达、未开始 prefill (FCFS)
    running: List[_Req] = []   # 正在 decode

    clock = 0.0
    finished: List[_Req] = []
    decode_bs_samples: List[int] = []
    max_iters = cfg.max_iters or (cfg.num_requests * (cfg.output_len + 4) * 3 + 10000)
    iters = 0
    saturated = False

    def admit_arrived():
        nonlocal next_arrival
        while next_arrival < len(pending) and pending[next_arrival].arrival <= clock:
            waiting.append(pending[next_arrival])
            next_arrival += 1

    while len(finished) < cfg.num_requests:
        iters += 1
        if iters > max_iters:
            saturated = True
            break

        admit_arrived()

        slots = cfg.max_running_requests - len(running)
        can_prefill = bool(waiting) and slots > 0

        if can_prefill:
            batch: List[_Req] = []
            tokens = 0
            while waiting and len(batch) < slots:
                req = waiting[0]
                if batch and tokens + req.input_len > cfg.chunked_prefill_size:
                    break
                batch.append(req)
                tokens += req.input_len
                waiting.popleft()

            # 假设同一仿真内 input_len 一致 → 缓存键收敛
            dt = kernel.t_prefill(len(batch), batch[0].input_len)
            clock += dt
            admit_arrived()  # prefill 期间也可能有新到达
            for req in batch:
                # prefill 完成即出首 token → TTFT = 完成时刻 − 到达时刻 (含排队)
                req.ttft = clock - req.arrival
                req.first_token_clock = clock
                req.gen = 1
                if req.gen >= req.output_len:
                    req.finish_clock = clock      # output_len==1: 无 decode
                    finished.append(req)
                else:
                    running.append(req)

        elif running:
            # ---- decode iter (连续批: running 全体 +1 token) ----
            avg_ctx = sum(r.ctx for r in running) / len(running)
            dt = kernel.t_decode_step(len(running), int(round(avg_ctx)))
            decode_bs_samples.append(len(running))
            clock += dt
            still: List[_Req] = []
            for req in running:
                req.gen += 1
                req.ctx += 1
                req.itls.append(dt)
                if req.gen >= req.output_len:
                    req.finish_clock = clock
                    finished.append(req)
                else:
                    still.append(req)
            running = still

        else:
            # ---- 空转: 快进到下一个到达时刻 (避免空步) ----
            if next_arrival < len(pending):
                clock = max(clock, pending[next_arrival].arrival)
                admit_arrived()
            else:
                break  # 无在跑、无等待、无未来到达 → 结束

    return _aggregate(cfg, finished, decode_bs_samples, clock, saturated)


def _percentile(xs: List[float], q: float) -> float:
    """线性插值百分位 (q ∈ [0,100]); 空列表返回 0。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate(cfg: ServingConfig, finished: List[_Req],
               decode_bs_samples: List[int], clock: float,
               saturated: bool) -> ServingMetrics:
    """统计 mean/p99 + 饱和判定 (§5)。丢弃前 warmup 个完成请求避免冷启动污染 (§7.6)。"""
    # 按完成顺序丢 warmup (系统未达稳态)
    warmup = cfg.warmup if cfg.warmup >= 0 else max(1, cfg.num_requests // 10)
    warmup = min(warmup, max(0, len(finished) - 1))
    stat_reqs = finished[warmup:] if finished else []

    ttfts = [r.ttft for r in stat_reqs]
    tpots = [_mean(r.itls) for r in stat_reqs if r.itls]   # TPOT = 单请求 decode 段均值
    itls = [dt for r in stat_reqs for dt in r.itls]        # ITL = 全体 decode 步样本

    m = ServingMetrics(req_rate=cfg.req_rate)
    m.ttft_ms_mean = _mean(ttfts)
    m.ttft_ms_p99 = _percentile(ttfts, 99)
    m.tpot_ms_mean = _mean(tpots)
    m.tpot_ms_p99 = _percentile(tpots, 99)
    m.itl_ms_mean = _mean(itls)
    m.itl_ms_p99 = _percentile(itls, 99)
    m.decode_bs_mean = _mean([float(x) for x in decode_bs_samples])
    m.decode_bs_max = max(decode_bs_samples) if decode_bs_samples else 0
    m.completed = len(stat_reqs)
    m.sim_time_ms = clock

    # achieved_qps: 持续吞吐 = 全部完成数 / (末请求完成时刻 − 首请求到达时刻)。
    # 稳态下 ≈ req_rate; 饱和时分母被 backlog drain 拉长 → 明显 < req_rate。
    if finished:
        t0 = min(r.arrival for r in finished)
        span = max(1e-9, (clock - t0) / 1000.0)
        m.achieved_qps = len(finished) / span

    ttft_diverging = False
    if len(stat_reqs) >= 6:
        by_arrival = sorted(stat_reqs, key=lambda r: r.arrival)
        k = max(1, len(by_arrival) // 3)
        head = _mean([r.ttft for r in by_arrival[:k]])
        tail = _mean([r.ttft for r in by_arrival[-k:]])
        # 后段比前段高 50% 以上, 且绝对增幅超一个 prefill 量级 (排除抖动)
        ttft_diverging = (tail > head * 1.5) and (tail - head > max(head, 1.0))
    m.saturated = bool(saturated or ttft_diverging)
    return m


def sweep_req_rate(base: ServingConfig, rates: List[float],
                   stop_on_saturate: bool = True,
                   verbose: bool = True) -> List[ServingMetrics]:
    """
    从低到高扫 req_rate, 观察 mean/p99 TTFT/TPOT 在哪个 rate 开始爆 (队列发散)。
    该拐点即"系统 QPS 能力上限" 。

    内核缓存在所有 rate 间共享 (同一模型/并行配置, 内核结果与 rate 无关) → 大幅提速。
    """
    from dataclasses import replace
    kernel = _KernelCache(base)
    results: List[ServingMetrics] = []
    if verbose:
        print(f"\n{'='*92}")
        print(f"  req_rate sweep: model={base.model} gpu={base.gpu} "
              f"tp{base.tp}dp{base.dp}pp{base.pp}ep{base.ep}cp{base.cp} "
              f"in={base.input_len} out={base.output_len} "
              f"max_run={base.max_running_requests} arrival={base.arrival}")
        print(f"{'='*92}")
        print(f"  {'QPS':>7} | {'TTFT mean':>9} {'TTFT p99':>9} | "
              f"{'TPOT mean':>9} {'TPOT p99':>9} | {'ITL p99':>8} | "
              f"{'achQPS':>7} {'dec_bs':>7} | {'sat':>4}")
        print(f"  {'-'*88}")
    for rate in rates:
        cfg = replace(base, req_rate=rate)
        m = simulate_serving(cfg, kernel=kernel)
        results.append(m)
        if verbose:
            print(f"  {rate:>7.2f} | {m.ttft_ms_mean:>9.1f} {m.ttft_ms_p99:>9.1f} | "
                  f"{m.tpot_ms_mean:>9.2f} {m.tpot_ms_p99:>9.2f} | {m.itl_ms_p99:>8.2f} | "
                  f"{m.achieved_qps:>7.2f} {m.decode_bs_mean:>7.1f} | "
                  f"{'YES' if m.saturated else 'no':>4}")
        if stop_on_saturate and m.saturated:
            if verbose:
                print(f"  → 在 QPS={rate:.2f} 检测到饱和 (队列发散), 停止扫描。")
            break
    if verbose:
        print(f"  内核穿透 C++ 调用次数 (缓存后): {kernel.calls}")
        print(f"{'='*92}\n")
    return results


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def serving_sim(
    req_rate: float = 0.0,
    rates: str = "",
    model: str = "deepseek_v3",
    gpu: str = "NMZ",
    tp: int = 8,
    dp: int = 1,
    pp: int = 1,
    ep: int = 8,
    cp: int = 1,
    nodes: int = 1,
    dp_attention: bool = False,
    input_len: int = 4096,
    output_len: int = 256,
    num_requests: int = 500,
    max_running_requests: int = 256,
    chunked_prefill_size: int = 8192,
    arrival: str = "poisson",
    warmup: int = -1,
    ctx_bucket: int = 512,
    seed: int = 0,
    kv_cache_dtype: str = "fp8",
    attention_compute_dtype: str = "fp8",
    fuse_moe: bool = True,
    gpus_per_node: int = 8,
    nvlink_domain_size: int = 8,
    intra_node_bw_gbs: float = 140.0,
    inter_node_bw_gbs: float = 50.0,
):
    """
    服务仿真 CLI。

    两种用法:
      - 单点: 传 --req_rate <QPS>, 输出该到达率下的 mean/p99 指标。
      - 扫描: 传 --rates "1,2,4,8,16" (逗号分隔), 从低到高扫到饱和拐点。
    """
    base = ServingConfig(
        req_rate=req_rate or 1.0,
        input_len=input_len, output_len=output_len,
        model=model, gpu=gpu, tp=tp, dp=dp, pp=pp, ep=ep, cp=cp, nodes=nodes,
        dp_attention=dp_attention,
        gpus_per_node=gpus_per_node, nvlink_domain_size=nvlink_domain_size,
        intra_node_bw_gbs=intra_node_bw_gbs, inter_node_bw_gbs=inter_node_bw_gbs,
        kv_cache_dtype=kv_cache_dtype, attention_compute_dtype=attention_compute_dtype,
        fuse_moe=fuse_moe,
        num_requests=num_requests, max_running_requests=max_running_requests,
        chunked_prefill_size=chunked_prefill_size, arrival=arrival,
        warmup=warmup, ctx_bucket=ctx_bucket, seed=seed,
    )

    if rates.strip():
        rate_list = [float(x) for x in rates.replace(" ", "").split(",") if x]
        return sweep_req_rate(base, rate_list)

    if req_rate <= 0:
        raise SystemExit("请指定 --req_rate <QPS> (单点) 或 --rates \"1,2,4,...\" (扫描)")

    m = simulate_serving(base)
    print(f"\n{'='*60}")
    print(f"  Serving Sim @ QPS={req_rate}  model={model}  in={input_len} out={output_len}")
    print(f"{'='*60}")
    print(f"  TTFT  mean/p99 : {m.ttft_ms_mean:10.2f} / {m.ttft_ms_p99:10.2f} ms")
    print(f"  TPOT  mean/p99 : {m.tpot_ms_mean:10.3f} / {m.tpot_ms_p99:10.3f} ms")
    print(f"  ITL   mean/p99 : {m.itl_ms_mean:10.3f} / {m.itl_ms_p99:10.3f} ms")
    print(f"  decode_bs mean/max : {m.decode_bs_mean:.1f} / {m.decode_bs_max}")
    print(f"  achieved_qps : {m.achieved_qps:.3f}  (target {req_rate})  "
          f"saturated={m.saturated}")
    print(f"{'='*60}\n")
    return m


if __name__ == "__main__":
    run_cli(
        serving_sim,
        prog="python -m model_analyzer.serving_simulator",
        description="请求率 (req_rate/QPS) 服务仿真: 迭代级 DES, 输出 TTFT/TPOT/ITL mean&p99 + 扫 SLO 拐点",
    )
