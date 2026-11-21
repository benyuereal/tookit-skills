#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 调度器最小化 POC —— 动态批处理 / Chunked Prefill / 抢占

只依赖 Python 标准库，CPU 直接跑：
    python mini_scheduler.py

对应文档：docs/vllm_scheduler_guide.md

这个 POC 把 vLLM V1 调度器的核心决策浓缩成 ~200 行，目的是让你"看见"
动态批处理是怎么工作的：

  1. 动态批处理 (Dynamic Batching)
     每个 step 把"正在 decode 的请求"和"新到达要 prefill 的请求"混进同一个
     batch 一起算，而不是等一个请求算完再算下一个。

  2. Chunked Prefill
     长 prompt 的 prefill 不一定要一步算完，可以按 token 预算切成多块，
     和 decode 混批，避免一个长 prefill 把整步占满、拖慢所有 decode。

  3. Token Budget (max_num_batched_tokens)
     每个 step 最多处理 N 个 token（prefill + decode 合计），控制单步延迟。

  4. KV Cache 容量 + 抢占 (Preemption)
     KV cache 块不够时，把"最晚进入 running 的请求"踢回 waiting 队列，
     它的 KV 释放掉，之后重新 prefill（recompute 抢占，最简策略）。

运行后会打印每个 step 的 batch 明细，最后汇总每个请求的 TTFT / 总耗时。
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Request:
    req_id: str
    prompt_len: int          # prompt token 数
    max_tokens: int
    # 运行态
    computed: int = 0        # 已计算的 token 数（prompt 已算 + 已生成）
    generated: int = 0       # 已生成的 token 数
    state: str = "waiting"   # waiting / running / finished
    ttft: Optional[int] = None   # 首 token 产出的 step
    finish_step: Optional[int] = None

    @property
    def is_prefill(self) -> bool:
        """还有 prompt 没算完 → 这一步对它来说是 prefill"""
        return self.computed < self.prompt_len

    @property
    def remaining_prompt(self) -> int:
        return max(0, self.prompt_len - self.computed)

    @property
    def remaining_total(self) -> int:
        """还需要算多少 token 才算完（prompt 剩余 + 还要生成的）"""
        return self.remaining_prompt + max(0, self.max_tokens - self.generated)


@dataclass
class ScheduleResult:
    """一个 step 的调度结果：哪些请求进 batch、各算多少 token"""
    running: List[Request] = field(default_factory=list)   # 已在 running 的请求
    prefill: List[Request] = field(default_factory=list)   # 本步新准入的 prefill 请求
    # 每个请求这一步实际分配的 token 数
    alloc: dict = field(default_factory=dict)
    # 每个请求这一步的类型（调度时定，避免 execute 后 is_prefill 变化导致标错）
    kinds: dict = field(default_factory=dict)
    # 本步被抢占的请求 id
    preempted: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------

class MiniScheduler:
    def __init__(self, max_num_batched_tokens: int = 64,
                 max_num_seqs: int = 8,
                 kv_cache_blocks: int = 20,
                 block_size: int = 4):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.block_size = block_size
        # KV cache 总块数（模拟显存容量）
        self.kv_total_blocks = kv_cache_blocks
        self.kv_used_blocks = 0

        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.all_reqs: List[Request] = []   # 跟踪所有请求（含已完成），用于汇总

    # ---- KV 块需求：一个请求占 ceil(computed / block_size) 块 ----
    def _blocks_needed(self, req: Request) -> int:
        return (req.computed + self.block_size - 1) // self.block_size

    def _kv_free_blocks(self) -> int:
        return self.kv_total_blocks - self.kv_used_blocks

    def _kv_used_by(self, reqs: List[Request]) -> int:
        return sum(self._blocks_needed(r) for r in reqs)

    # ---- 核心：调度一个 step ----
    def schedule(self) -> ScheduleResult:
        result = ScheduleResult()
        budget = self.max_num_batched_tokens
        proj_kv = 0   # 本步算完后，batch 里所有请求预计占用的 KV 块数

        def blocks_after(req: Request, n: int) -> int:
            """这一步给它算 n 个 token 后，它占多少 KV 块"""
            return (req.computed + n + self.block_size - 1) // self.block_size

        # 1) 先保证 running 里的请求（decode 每请求 1 token，prefill 续算）
        #    按"进入 running 的先后"保序，越晚的越容易被抢占
        for req in self.running:
            if budget <= 0:
                break
            if req.is_prefill:
                n = min(req.remaining_prompt, budget)   # chunked prefill 续算
                result.kinds[req.req_id] = "P"
            else:
                n = 1                                    # decode 一步 1 token
                result.kinds[req.req_id] = "D"
            # KV 预测：算完这一步后总占用是否超容量
            if proj_kv + blocks_after(req, n) > self.kv_total_blocks:
                # 放不下 → 抢占：踢回 waiting，computed 清零（recompute 策略）
                self._preempt(req, result)
                continue
            proj_kv += blocks_after(req, n)
            result.running.append(req)
            result.alloc[req.req_id] = n
            budget -= n

        # 2) 从 waiting 里挑新请求进 batch（prefill），受预算 + 序列数 + KV 限制
        #    贪心：按到达顺序，能塞多少塞多少，长 prompt 可 chunk
        for req in self.waiting:
            if budget <= 0:
                break
            if len(result.running) + len(result.prefill) >= self.max_num_seqs:
                break
            n = min(req.remaining_prompt, budget)
            if proj_kv + blocks_after(req, n) > self.kv_total_blocks:
                continue   # KV 不够，这个请求本步进不来（等下一步）
            proj_kv += blocks_after(req, n)
            result.prefill.append(req)
            result.alloc[req.req_id] = n
            result.kinds[req.req_id] = "P"
            budget -= n

        return result

    def _preempt(self, req: Request, result: ScheduleResult):
        """抢占：把 running 请求踢回 waiting，KV 释放，computed 清零重算"""
        if req in self.running:
            self.running.remove(req)
        req.state = "waiting"
        req.computed = 0
        req.generated = 0
        req.ttft = None
        if req not in self.waiting:
            self.waiting.append(req)
        result.preempted.append(req.req_id)

    # ---- 执行一个 step 的结果（模拟 GPU 算完）----
    def execute(self, result: ScheduleResult, step: int):
        # 1) 新 prefill 的请求：waiting → running（先移，后面算 KV 才准）
        for req in result.prefill:
            if req.state == "waiting":
                self.waiting.remove(req)
                self.running.append(req)
                req.state = "running"

        # 2) 推进每个被调度请求的进度
        for req in result.running + result.prefill:
            n = result.alloc[req.req_id]
            req.computed += n
            # prefill 算完（computed 达到 prompt_len）后，超出部分算"生成"
            if req.computed >= req.prompt_len:
                req.generated = req.computed - req.prompt_len
                if req.ttft is None and req.generated >= 1:
                    req.ttft = step
                if req.generated >= req.max_tokens:
                    req.state = "finished"
                    req.finish_step = step

        # 3) 移除完成的请求（释放 KV）
        for req in [r for r in self.running if r.state == "finished"]:
            self.running.remove(req)

        # 4) 重算 KV 占用（只算还在 running 的）
        self.kv_used_blocks = self._kv_used_by(self.running)

    # ---- 主循环 ----
    def run(self, arrivals: List[tuple], max_steps: int = 200):
        """arrivals: [(step, req_id, prompt_len, max_tokens), ...] 按 step 到达"""
        arrival_map = {}
        for step, rid, plen, mtok in arrivals:
            arrival_map.setdefault(step, []).append((rid, plen, mtok))

        max_arrival = max((s for s, *_ in arrivals), default=0)
        step = 0
        while step < max_steps:
            # 全部请求都到齐且都算完 → 结束
            if step > max_arrival and not self.waiting and not self.running:
                break
            # 新请求到达
            for rid, plen, mtok in arrival_map.get(step, []):
                req = Request(rid, plen, mtok)
                self.waiting.append(req)
                self.all_reqs.append(req)

            result = self.schedule()
            if not result.running and not result.prefill:
                # 没有可调度的（KV 满且无预算），空转一步
                step += 1
                continue
            self.execute(result, step)
            self._print_step(step, result)
            step += 1

        self._print_summary()
        return step


    # ---- 打印 ----
    def _print_step(self, step: int, result: ScheduleResult):
        parts = []
        for req in result.running + result.prefill:
            kind = result.kinds[req.req_id]   # 调度时定的 P/D
            parts.append(f"{req.req_id}[{kind}x{result.alloc[req.req_id]}]")
        total = sum(result.alloc.values())
        pre = f"  ⚡抢占 {','.join(result.preempted)}" if result.preempted else ""
        print(f"step {step:3d} | batch={len(parts):2d} | tokens={total:3d}/{self.max_num_batched_tokens} "
              f"| kv={self.kv_used_blocks}/{self.kv_total_blocks} | {' '.join(parts)}{pre}")

    def _print_summary(self):
        print("\n===== 汇总 =====")
        print(f"{'req':6s} {'prompt':>6s} {'gen':>4s} {'TTFT':>5s} {'finish':>7s}  {'state':>10s}")
        for req in self.all_reqs:
            ttft = str(req.ttft) if req.ttft is not None else "-"
            fin = str(req.finish_step) if req.finish_step is not None else "-"
            print(f"{req.req_id:6s} {req.prompt_len:6d} {req.generated:4d} {ttft:>5s} {fin:>7s}  {req.state:>10s}")


# ---------------------------------------------------------------------------
# 演示场景
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("场景：4 个请求在不同 step 到达，演示动态批处理 + chunked prefill")
    print("  - A: 长 prompt(40)，step 0 到达")
    print("  - B: 短 prompt(8)， step 0 到达")
    print("  - C: 中 prompt(20)，step 2 到达")
    print("  - D: 短 prompt(6)， step 4 到达")
    print("  预算 max_num_batched_tokens=32, max_num_seqs=4, KV=16 块(block=4)")
    print("=" * 70 + "\n")

    sched = MiniScheduler(
        max_num_batched_tokens=32,
        max_num_seqs=4,
        kv_cache_blocks=16,
        block_size=4,
    )
    arrivals = [
        (0, "A", 40, 10),   # 长 prompt
        (0, "B", 8, 6),     # 短 prompt
        (2, "C", 20, 8),    # 中 prompt
        (4, "D", 6, 5),     # 短 prompt
    ]
    sched.run(arrivals)


if __name__ == "__main__":
    main()
