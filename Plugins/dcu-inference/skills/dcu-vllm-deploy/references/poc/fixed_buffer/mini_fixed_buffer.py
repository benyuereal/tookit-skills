#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM V1 InputBatch 固定 buffer 机制最小化 POC —— "固定停车位"

只依赖 Python 标准库，CPU 直接跑：
    python3 mini_fixed_buffer.py

对应文档：docs/vllm_input_batch_guide.md

这个 POC 让你"看见" vLLM V1 的 InputBatch 为什么用"固定 buffer"：

  真实 vLLM 按 max_num_seqs 预分配一批 CPU buffer（input_ids / positions /
  slot_mapping / block_table / req_to_token ...），每个请求占一个"停车位"
  （req_index）。推理循环里：

    - batch 组成不变（没有请求加入/离开）→ 复用 buffer，只更新变化的 slot
      （比如 decode 时每个请求只多 1 个 token，只写新 token 那几格）
    - batch 变化（请求加入/离开）→ 才重建 buffer（condense / 重排）

  对比"朴素实现"每步都重建整个 batch 张量，固定 buffer 把每步的 CPU 张量
  构造开销从 O(batch) 降到 O(变化量)。

  组件名对齐真实 vLLM：InputBatch、block_table、slot_mapping、positions、
  req_to_token。
"""

MAX_NUM_SEQS = 4     # 预分配的"停车位"数（= 真实 vLLM 的 max_num_seqs）
NUM_STEPS = 100      # 模拟 100 步推理


# ---------------------------------------------------------------------------
# 一个请求在某一步的"状态"（模拟调度器给出的 batch 组成）
# ---------------------------------------------------------------------------

class ReqState:
    def __init__(self, req_id: str, num_tokens: int):
        self.req_id = req_id
        self.num_tokens = num_tokens   # 该请求当前已算的 token 数

    def decode(self):
        self.num_tokens += 1


# ---------------------------------------------------------------------------
# 朴素实现：每步重建整个 batch 张量
# ---------------------------------------------------------------------------

class NaiveBatch:
    """每步都重新构造 input_ids / positions / slot_mapping / block_table"""

    def __init__(self, max_num_seqs: int):
        self.max_num_seqs = max_num_seqs
        self.rebuilds = 0
        self.slots_written = 0

    def step(self, reqs: list):
        # 每步都重建 4 个张量（input_ids/positions/slot_mapping/block_table）
        self.rebuilds += 1
        n = len(reqs)
        # 重建 = 把每个请求的所有 token 都重新写一遍
        self.slots_written += sum(r.num_tokens for r in reqs)
        return f"重建 4 个张量（{n} 请求 x 全部 token）"


# ---------------------------------------------------------------------------
# 固定 buffer 实现：预分配 max 大小，不变时只写变化 slot
# ---------------------------------------------------------------------------

class FixedInputBatch:
    """
    对齐真实 vLLM InputBatch：按 max_num_seqs 预分配 buffer，
    每个请求占一个 req_index（停车位）。
    """

    def __init__(self, max_num_seqs: int):
        self.max_num_seqs = max_num_seqs
        # 预分配的 buffer（长度 = max_num_seqs，每个请求一格）
        self.input_ids = [0] * max_num_seqs        # 每请求本步要算的 token
        self.positions = [0] * max_num_seqs        # 每请求本步 token 的位置
        self.slot_mapping = [0] * max_num_seqs     # 每请求 KV 写入的物理槽位
        self.block_table = [[0] * 8 for _ in range(max_num_seqs)]  # 每请求 block table
        self.req_to_token = [[0] * 256 for _ in range(max_num_seqs)]  # 逻辑位置->物理槽
        self.req_id_to_index = {}                  # req_id -> 停车位
        self.rebuilds = 0
        self.slots_written = 0
        self.reuses = 0

    def _rebuild(self, reqs: list):
        """batch 变化时重建：重新分配停车位 + 填 block_table/req_to_token"""
        self.rebuilds += 1
        self.req_id_to_index = {}
        for i, r in enumerate(reqs):
            self.req_id_to_index[r.req_id] = i
            # 重建时把该请求的 block_table / req_to_token 整行填一遍
            self.slots_written += 8 + r.num_tokens
        return f"batch 变化，重建 buffer（{len(reqs)} 请求）"

    def step(self, reqs: list):
        # 判断 batch 组成是否变化（req_id 集合）
        cur_ids = {r.req_id for r in reqs}
        old_ids = set(self.req_id_to_index.keys())
        if cur_ids != old_ids:
            # 有请求加入/离开 → 重建
            return self._rebuild(reqs)
        # 组成不变 → 复用 buffer，只更新变化的 slot（decode 每请求 1 个新 token）
        self.reuses += 1
        for r in reqs:
            i = self.req_id_to_index[r.req_id]
            # 只写本步新增的那 1 个 token 对应的几格
            self.input_ids[i] = r.num_tokens
            self.positions[i] = r.num_tokens - 1
            self.slot_mapping[i] = r.num_tokens - 1
            self.req_to_token[i][r.num_tokens - 1] = r.num_tokens - 1
            self.slots_written += 4   # 4 个 buffer 各写 1 格
        return f"复用 buffer，只更新 {len(reqs)} slot（每请求 1 新 token）"


# ---------------------------------------------------------------------------
# 模拟推理循环：每步 batch 有 加入/离开/不变 三种情况
# ---------------------------------------------------------------------------

def simulate():
    print("=" * 70)
    print(f"场景：模拟 {NUM_STEPS} 步推理，max_num_seqs={MAX_NUM_SEQS}")
    print("  每步 batch 请求有 加入/离开/不变 三种情况：")
    print("    - 不变：所有请求 decode（各 +1 token）→ 固定 buffer 只写变化 slot")
    print("    - 加入：新请求 prefill 进来 → 固定 buffer 重建")
    print("    - 离开：请求生成完退出 → 固定 buffer 重建")
    print("  对比朴素实现（每步重建整个 batch 张量）的 CPU 开销")
    print("=" * 70 + "\n")

    naive = NaiveBatch(MAX_NUM_SEQS)
    fixed = FixedInputBatch(MAX_NUM_SEQS)

    # 用一个简单的请求池模拟加入/离开
    active = []          # 当前 batch 里的请求
    next_id = 0
    stats = {"加入": 0, "离开": 0, "不变": 0}

    for step in range(NUM_STEPS):
        # ---- 模拟调度：决定本步 batch 组成 ----
        # 规则：每 10 步加入 1 个新请求（若没满），每 15 步离开 1 个（若 >1）
        # 离开用 %15==7 错开，避免和加入（%10==0）撞在同一步
        joined = step % 10 == 0 and len(active) < MAX_NUM_SEQS
        left = step % 15 == 7 and len(active) > 1
        if joined:
            r = ReqState(f"req{next_id}", 8)   # 新请求 prefill 8 token
            next_id += 1
            active.append(r)
            stats["加入"] += 1
        if left:
            active.pop()   # 最后一个请求完成离开
            stats["离开"] += 1

        if not joined and not left:
            # 不变：所有请求 decode
            for r in active:
                r.decode()
            stats["不变"] += 1

        if not active:
            continue

        # ---- 两种实现各跑一步 ----
        naive_msg = naive.step(active)
        fixed_msg = fixed.step(active)

        # 只打印前 12 步 + 每次 batch 变化，避免刷屏
        if step < 12 or joined or left:
            tag = "加入+离开" if (joined and left) else \
                  "加入" if joined else "离开" if left else "不变"
            print(f"step {step:3d} [{tag}] batch={len(active)}")
            print(f"    朴素: {naive_msg}")
            print(f"    固定: {fixed_msg}")

    # ---- 汇总对比 ----
    print("\n" + "=" * 70)
    print(f"===== {NUM_STEPS} 步汇总对比 =====")
    print("=" * 70)
    print(f"  batch 组成变化次数: 加入 {stats['加入']} + 离开 {stats['离开']} "
          f"= {stats['加入'] + stats['离开']} 次")
    print(f"  batch 不变步数:     {stats['不变']} 次")
    print()
    print(f"  {'实现':10s} {'重建次数':>8s} {'写入 slot 总数':>14s}")
    print(f"  {'朴素':10s} {naive.rebuilds:8d} {naive.slots_written:14d}")
    print(f"  {'固定buffer':10s} {fixed.rebuilds:8d} {fixed.slots_written:14d}")
    print()
    ratio = naive.slots_written / max(1, fixed.slots_written)
    print(f"  → 固定 buffer 只重建 {fixed.rebuilds} 次（= batch 变化次数），")
    print(f"    朴素实现每步都重建 {naive.rebuilds} 次。")
    print(f"    写入 slot 总数：朴素 {naive.slots_written} vs 固定 {fixed.slots_written}，")
    print(f"    固定 buffer 少写约 {ratio:.1f}x 的 CPU 数据。")
    print("  真实 vLLM 里这就是 InputBatch 用固定 buffer 的原因：")
    print("  decode 占绝大多数步，每步只更新 1 个新 token 的几格，")
    print("  避免每步重建整批张量带来的 CPU 开销和 H2D 拷贝。")


if __name__ == "__main__":
    simulate()
