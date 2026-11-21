#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM PagedAttention 最小化 POC —— KV cache 分页

只依赖 Python 标准库，CPU 直接跑：
    python3 mini_paged_attention.py

对应文档：docs/vllm_paged_attention_guide.md

这个 POC 把 PagedAttention 的核心机制浓缩成 ~200 行，让你"看见"：

  1. 物理 block 池 (Block Pool)
     显存被切成固定大小的物理块（这里 16 个 block，每 block 4 个 token 的
     K/V 槽位），所有请求共享这个池子。

  2. Block Table（逻辑块 → 物理块）
     每个请求维护一张 block table：第 i 个逻辑块存在哪个物理块里。
     物理块可以任意散布，请求的 KV 在显存里是"交错"的。

  3. 分页 attention
     给定 1 个新 token 的 query + 该请求的 block table，从分页 KV 里
     gather 出 K/V 再算 attention —— 每个请求只 gather 自己 block table
     指向的物理块，各取各的，互不干扰。

  4. 对比不分页
     不分页（连续预留 max_len）时，每个请求要预留 max_len 的连续空间，
     实际只用了一小部分 → 显存浪费。最后打印两种方案的占用对比。
"""

import math

NUM_BLOCKS = 16   # 物理 block 池大小
BLOCK_SIZE = 4    # 每 block 存 4 个 token 的 K/V
DIM = 2           # 向量维度（故意取 2，方便肉眼看数值）


# ---------------------------------------------------------------------------
# 物理 block 池
# ---------------------------------------------------------------------------

class BlockPool:
    """固定物理 block 池：每 block 有 BLOCK_SIZE 个 K 槽位和 V 槽位"""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # k_blocks[pb][slot] = 该物理块第 slot 个 token 的 K 向量
        self.k_blocks = [[None] * block_size for _ in range(num_blocks)]
        self.v_blocks = [[None] * block_size for _ in range(num_blocks)]
        self.free = list(range(num_blocks - 1, -1, -1))   # free list（栈）

    def alloc(self) -> int:
        """从 free list 分配一个物理块"""
        return self.free.pop()

    def free_block(self, pb: int):
        """释放物理块（清掉内容，防止读到脏数据）"""
        for s in range(self.block_size):
            self.k_blocks[pb][s] = None
            self.v_blocks[pb][s] = None
        self.free.append(pb)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free)


# ---------------------------------------------------------------------------
# 请求：block table + 分页读写
# ---------------------------------------------------------------------------

class Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.block_table: list = []   # 逻辑块 i -> 物理块 block_table[i]
        self.num_tokens = 0           # 已写入 KV 的 token 数

    def ensure_blocks(self, pool: BlockPool, n_tokens: int):
        """保证 block table 能覆盖 n_tokens 个 token（向上取整到 block）"""
        need = (n_tokens + pool.block_size - 1) // pool.block_size
        while len(self.block_table) < need:
            pb = pool.alloc()
            self.block_table.append(pb)
            print(f"  [alloc] {self.req_id} 逻辑块 {len(self.block_table)-1} "
                  f"-> 物理块 {pb}  (free list 剩 {len(pool.free)} 块)")

    def write_kv(self, pool: BlockPool, token_idx: int, k, v):
        """把第 token_idx 个 token 的 K/V 写进分页 KV cache"""
        lb = token_idx // pool.block_size          # 逻辑块号
        slot = token_idx % pool.block_size         # 块内槽位
        pb = self.block_table[lb]                  # 查 block table 得物理块
        pool.k_blocks[pb][slot] = k
        pool.v_blocks[pb][slot] = v
        self.num_tokens = max(self.num_tokens, token_idx + 1)
        print(f"  [write] {self.req_id} token {token_idx} "
              f"-> 物理块 {pb} slot {slot}")

    def free(self, pool: BlockPool):
        for pb in self.block_table:
            pool.free_block(pb)
        self.block_table = []
        self.num_tokens = 0


# ---------------------------------------------------------------------------
# 分页 attention：从 block table gather K/V
# ---------------------------------------------------------------------------

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def paged_attention(req: Request, pool: BlockPool, q):
    """
    给定 1 个新 token 的 query q，按 req 的 block table 从分页 KV gather
    出全部 K/V，算 attention。每个请求只碰自己 block table 里的物理块。
    """
    # 1) gather：token i 的 K/V 在 物理块 block_table[i//4] 的 slot i%4
    ks, vs, phys_blocks = [], [], []
    for i in range(req.num_tokens):
        pb = req.block_table[i // pool.block_size]
        ks.append(pool.k_blocks[pb][i % pool.block_size])
        vs.append(pool.v_blocks[pb][i % pool.block_size])
        if pb not in phys_blocks:
            phys_blocks.append(pb)
    print(f"  [attention] {req.req_id}: 从物理块 {phys_blocks} gather "
          f"{req.num_tokens} 个 token 的 K/V")

    # 2) scores = q · k，softmax，加权求和 V
    scores = [dot(q, k) for k in ks]
    weights = softmax(scores)
    out = [sum(w * v[j] for w, v in zip(weights, vs)) for j in range(DIM)]
    for i, (s, w) in enumerate(zip(scores, weights)):
        print(f"      token {i}: score={s:6.2f}  weight={w:6.3f}")
    print(f"      output = [{out[0]:.3f}, {out[1]:.3f}]")
    return out


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("场景：2 个请求共享 16 块物理 block 池（每块 4 token 的 K/V 槽位）")
    print("  - A: prompt 6 token   - B: prompt 5 token")
    print("  交错写入 KV（A0,B0,A1,B1,...），再各自 decode 1 步")
    print("=" * 70 + "\n")

    pool = BlockPool(NUM_BLOCKS, BLOCK_SIZE)
    a = Request("A")
    b = Request("B")

    # ---- 1) 交错 prefill：两个请求交替写 KV，物理块互相穿插 ----
    print("---- 交错 prefill（A/B 交替写 KV）----")
    a.ensure_blocks(pool, 6)
    b.ensure_blocks(pool, 5)
    print(f"  A block_table = {a.block_table}")
    print(f"  B block_table = {b.block_table}\n")

    for i in range(6):
        # token i 的 K/V 用确定性小向量，方便核对
        a.write_kv(pool, i, [i + 1, (i + 1) * 0.5], [i + 1, 0.0])
        if i < 5:
            b.write_kv(pool, i, [10 + i, (10 + i) * 0.5], [10 + i, 0.0])

    print(f"\n  此时池子占用: {pool.used_blocks}/{pool.num_blocks} 块")
    print(f"  A 的 KV 在物理块 {a.block_table}，B 的 KV 在物理块 {b.block_table}")
    print("  → 两个请求的 KV 在池子里交错存放，但 block table 各指各的\n")

    # ---- 2) decode：各取各的，互不干扰 ----
    print("---- decode：A/B 各算 1 个新 token 的 attention ----")
    q_a = [1.0, 1.0]
    q_b = [1.0, 1.0]
    paged_attention(a, pool, q_a)
    print()
    paged_attention(b, pool, q_b)
    print("\n  → A 只 gather 了自己的物理块，B 只 gather 了自己的，")
    print("    虽然 KV 在池子里交错，attention 结果互不污染\n")

    # ---- 3) 对比：分页 vs 不分页（连续预留 max_len）----
    print("---- 显存占用对比（max_len = 16 token/请求）----")
    max_len = 16
    # 不分页：每请求预留 max_len 个连续 token 槽位
    nopaged_slots = 2 * max_len
    nopaged_blocks = nopaged_slots // BLOCK_SIZE
    # 分页：只按实际 token 数向上取整
    paged_blocks = pool.used_blocks
    print(f"  不分页（连续预留 max_len）: 2 请求 x {max_len} token "
          f"= {nopaged_slots} 槽位 = {nopaged_blocks} 块")
    print(f"  分页（按实际 token 数）:   A 6 + B 5 = 11 token "
          f"= {paged_blocks} 块")
    waste = nopaged_blocks - paged_blocks
    print(f"  浪费: {waste} 块（{waste / nopaged_blocks * 100:.0f}%）"
          f"  ← 请求越短、max_len 越大，浪费越严重")
    print("  真实 vLLM 里这就是 PagedAttention 的核心收益：KV cache 利用率")
    print("  从 ~20-40% 提到 98%+，从而能塞下更大的 batch。")


if __name__ == "__main__":
    main()
