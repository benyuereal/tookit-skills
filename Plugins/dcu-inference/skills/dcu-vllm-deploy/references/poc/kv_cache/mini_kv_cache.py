#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM KV cache 管理器最小化 POC —— BlockPool + 前缀缓存 (APC)

只依赖 Python 标准库，CPU 直接跑：
    python3 mini_kv_cache.py

对应文档：docs/vllm_kv_cache_guide.md

这个 POC 把 vLLM V1 KVCacheManager 的核心机制浓缩成 ~200 行，让你"看见"：

  1. BlockPool（固定 block 数 + free list）
     显存切成固定数量的物理块，free list 管理空闲块，
     分配/释放都是 O(1)。

  2. 按 block_size 向上取整分配
     请求有 N 个 token 就占 ceil(N / block_size) 块；
     完成/抢占时整块释放回 free list。

  3. 前缀缓存 APC (Automatic Prefix Caching)
     每个"写满"的 block 按内容哈希（这里简化为前缀 token 序列）登记到
     block_hash -> 物理块 的映射，并带引用计数。
     新请求的相同前缀直接命中复用，跳过这部分 prefill → TTFT 降低。

  4. 引用计数
     前缀 block 被多个请求引用时，前几个完成不释放，
     最后一个引用释放时才真正回到 free list。
"""

BLOCK_SIZE = 4
NUM_BLOCKS = 12   # 物理 block 池大小


# ---------------------------------------------------------------------------
# BlockPool：固定 block 数 + free list
# ---------------------------------------------------------------------------

class BlockPool:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free = list(range(num_blocks - 1, -1, -1))   # free list（栈）
        # 每个物理块的内容（token 列表），用于演示"块里存了什么"
        self.contents = [None] * num_blocks

    def alloc(self) -> int:
        return self.free.pop()

    def free_block(self, pb: int):
        self.contents[pb] = None
        self.free.append(pb)

    @property
    def used(self) -> int:
        return self.num_blocks - len(self.free)


# ---------------------------------------------------------------------------
# 前缀缓存：block 内容哈希 -> 物理块（带引用计数）
# ---------------------------------------------------------------------------

class PrefixCache:
    """
    简化版 APC：key = 该 block 的前缀 token 元组（真实 vLLM 用
    (parent_hash, token_ids) 的链式哈希，保证前缀语义）。
    value = (物理块号, 引用计数)
    """

    def __init__(self):
        self.cache: dict = {}   # hash -> (pb, refcount)

    def lookup(self, h) -> int:
        """命中返回物理块号，否则 -1"""
        entry = self.cache.get(h)
        return entry[0] if entry else -1

    def ref(self, h, pb):
        """引用计数 +1（新请求复用 / 首次登记）"""
        if h in self.cache:
            self.cache[h] = (self.cache[h][0], self.cache[h][1] + 1)
        else:
            self.cache[h] = (pb, 1)

    def release(self, h) -> bool:
        """引用计数 -1；归零时从缓存移除（块由调用方释放）。返回是否移除"""
        entry = self.cache.get(h)
        if entry is None:
            return False
        pb, rc = entry
        if rc <= 1:
            del self.cache[h]
            return True
        self.cache[h] = (pb, rc - 1)
        return False


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------

class Request:
    def __init__(self, req_id: str, tokens: list):
        self.req_id = req_id
        self.tokens = tokens            # prompt token 序列
        self.block_table: list = []     # 逻辑块 -> 物理块
        self.block_hashes: list = []    # 每个块的哈希（用于释放时查 APC）
        self.ttft = None                # 首 token 时间（= 实际 prefill token 数）

    @property
    def num_blocks(self) -> int:
        return (len(self.tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE

    def is_full(self, lb: int) -> bool:
        """第 lb 个逻辑块是否写满（只有满块才进 APC）"""
        return (lb + 1) * BLOCK_SIZE <= len(self.tokens)


# ---------------------------------------------------------------------------
# KV cache 管理器
# ---------------------------------------------------------------------------

class MiniKVCacheManager:
    def __init__(self, pool: BlockPool, prefix_cache: PrefixCache):
        self.pool = pool
        self.apc = prefix_cache

    def _block_hash(self, req: Request, lb: int):
        """第 lb 个逻辑块的"内容哈希"：简化为前 (lb+1)*4 个 token 的元组"""
        end = (lb + 1) * BLOCK_SIZE
        return tuple(req.tokens[:end])

    # ---- 分配：先查前缀缓存，命中部分直接复用 ----
    def allocate(self, req: Request) -> int:
        hit_tokens = 0
        for lb in range(req.num_blocks):
            if not req.is_full(lb):
                break   # 不满的块不进 APC，也不可能命中
            h = self._block_hash(req, lb)
            pb = self.apc.lookup(h)
            if pb != -1:
                # 命中前缀缓存：复用物理块，引用计数 +1
                self.apc.ref(h, pb)
                req.block_table.append(pb)
                req.block_hashes.append(h)
                hit_tokens = (lb + 1) * BLOCK_SIZE
            else:
                break
        # 未命中的部分：从 free list 新分配
        for lb in range(req.num_blocks):
            if lb < len(req.block_table):
                continue
            pb = self.pool.alloc()
            req.block_table.append(pb)
            req.block_hashes.append(self._block_hash(req, lb))
            # 把 token 写进块（模拟 prefill 算完 KV 写 cache）
            start = lb * BLOCK_SIZE
            self.pool.contents[pb] = list(req.tokens[start:start + BLOCK_SIZE])

        if hit_tokens > 0:
            print(f"  [APC] {req.req_id} 命中前缀缓存，跳过 {hit_tokens} token prefill "
                  f"(复用物理块 {req.block_table[:hit_tokens // BLOCK_SIZE]})")
        else:
            print(f"  [APC] {req.req_id} 未命中，全部 {len(req.tokens)} token 需 prefill")
        print(f"  [alloc] {req.req_id} block_table={req.block_table} "
              f"池子占用 {self.pool.used}/{self.pool.num_blocks}")
        return hit_tokens

    # ---- 模拟 prefill：只算未命中的 token ----
    def run_prefill(self, req: Request, hit_tokens: int):
        work = len(req.tokens) - hit_tokens
        req.ttft = work   # 时间单位 = 实际 prefill token 数
        # 写满的块登记进 APC（真实 vLLM 只缓存写满的块，部分块不缓存）
        for lb in range(req.num_blocks):
            if not req.is_full(lb):
                break
            h = req.block_hashes[lb]
            if self.apc.lookup(h) == -1:
                self.apc.ref(h, req.block_table[lb])
        print(f"  [prefill] {req.req_id} 实际算 {work} token，TTFT={req.ttft} "
              f"(池子占用 {self.pool.used}/{self.pool.num_blocks})")

    # ---- 释放：引用计数归零的块才真正释放 ----
    def free(self, req: Request):
        for pb, h in zip(req.block_table, req.block_hashes):
            if h not in self.apc.cache:
                # 不在 APC 里（不满的块从没被缓存）→ 直接释放
                self.pool.free_block(pb)
                print(f"  [free] {req.req_id} 释放物理块 {pb} "
                      f"(非缓存块，直接回 free list)")
                continue
            removed = self.apc.release(h)
            if removed:
                self.pool.free_block(pb)
                print(f"  [free] {req.req_id} 释放物理块 {pb} "
                      f"(引用计数归零，回到 free list)")
            else:
                print(f"  [free] {req.req_id} 物理块 {pb} 仍被其他请求引用，暂不释放")
        req.block_table = []
        req.block_hashes = []


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def main():
    # 3 个请求共享相同 system prompt 前缀（8 token = 2 个满块）
    system_prompt = [101, 102, 103, 104, 105, 106, 107, 108]
    print("=" * 70)
    print("场景：3 个请求共享相同 system prompt 前缀（8 token = 2 个满块）")
    print(f"  池子 {NUM_BLOCKS} 块，block_size={BLOCK_SIZE}")
    print("  A 先到（未命中，全量 prefill）→ B、C 后到（命中前缀缓存）")
    print("=" * 70 + "\n")

    pool = BlockPool(NUM_BLOCKS)
    apc = PrefixCache()
    mgr = MiniKVCacheManager(pool, apc)

    reqs = {
        "A": Request("A", system_prompt + [201, 202, 203]),
        "B": Request("B", system_prompt + [301, 302, 303]),
        "C": Request("C", system_prompt + [401, 402, 403]),
    }

    print("---- 请求 A 到达（第一个，无前缀缓存）----")
    hit = mgr.allocate(reqs["A"])
    mgr.run_prefill(reqs["A"], hit)
    print()

    print("---- 请求 B 到达（相同 system prompt）----")
    hit = mgr.allocate(reqs["B"])
    mgr.run_prefill(reqs["B"], hit)
    print()

    print("---- 请求 C 到达（相同 system prompt）----")
    hit = mgr.allocate(reqs["C"])
    mgr.run_prefill(reqs["C"], hit)
    print()

    # ---- TTFT 对比 ----
    print("---- TTFT 对比（时间单位 = 实际 prefill token 数）----")
    full = len(reqs["A"].tokens)   # 无 APC 时每个请求都要全量 prefill
    for rid in "ABC":
        r = reqs[rid]
        saved = full - r.ttft
        print(f"  {rid}: TTFT={r.ttft:3d}  (无 APC 需 {full}，"
              f"省 {saved} token 的 prefill)")
    print("  → B/C 的 TTFT 显著低于 A：前缀 KV 直接复用，不用重算\n")

    # ---- 引用计数演示 ----
    print("---- 引用计数：前缀块被 3 个请求引用 ----")
    prefix_pb = reqs["A"].block_table[:2]
    rcs = [apc.cache[tuple(system_prompt[: (i + 1) * BLOCK_SIZE])][1]
           for i in range(2)]
    print(f"  前缀物理块 {prefix_pb} 当前引用计数: {rcs}")
    print()
    print("  A 完成 → 释放：")
    mgr.free(reqs["A"])
    print()
    print("  B 完成 → 释放：")
    mgr.free(reqs["B"])
    print()
    print("  C 完成 → 释放（最后一个引用）：")
    mgr.free(reqs["C"])
    print()
    print(f"  最终池子占用: {pool.used}/{pool.num_blocks}，"
          f"free list 剩 {len(pool.free)} 块")
    print("  → 前 2 个完成时前缀块不释放（还有引用），第 3 个完成才真正释放")


if __name__ == "__main__":
    main()
