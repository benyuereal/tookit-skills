#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 采样管线最小化 POC —— temperature / top-k / top-p / penalties

只依赖 Python 标准库，CPU 直接跑：
    python mini_sampling.py

对应真实源码（vllm/v1/sample/）：
  - sampler.py::Sampler.forward / sample
      管线顺序：penalties → temperature → top_k/top_p → 按分布采样
      （temperature=0 走 greedy argmax 快路径，不采样）
  - ops/topk_topp_sampler.py::apply_top_k_top_p
      top-p 实现：logits 升序排序 → softmax → cumsum →
      mask 掉 probs_sum <= 1-p 的 token（最后一个强制保留）
  - ops/topk_topp_sampler.py::random_sample
      不用 multinomial，用 argmax(probs / q)，q ~ Exp(1)（Gumbel-max 技巧）
  - ops/penalties.py::apply_all_penalties
      logits -= frequency_penalty * 出现次数
      logits -= presence_penalty  * (是否出现过)

这个 POC 把整条管线浓缩成 ~200 行，目的是让你"看见"每一步的中间状态：
  1. 完整管线 trace：logits → temperature → top-k → top-p → 采样
  2. batch 3 个请求（greedy / top-k=5 / top-p=0.9），mask 前后分布对比
  3. top-p 的 cumsum 曲线，标出截断点
  4. temperature=0.1 / 1.0 / 2.0 对分布形状的影响（尖锐 → 平坦）
  5. frequency / presence penalty 对 logits 的影响（重复 token 被压低）
  6. 固定 seed 重复采样 20 次，实测频率 vs 理论概率
"""

import math
import random

# 小词表，方便打印分布
VOCAB = ["the", "cat", "sat", "on", "mat", "and", "dog", "saw", "a", "fish"]
NEG_INF = float("-inf")
EPS = 1e-5   # 对齐 sampler.py 的 _SAMPLING_EPS：temperature < EPS 视为 greedy

# 固定的"模型输出" logits（模拟一次 forward 的结果）
LOGITS = [2.1, 3.4, 1.2, 0.5, 2.8, 0.1, 1.9, 0.3, 1.5, 0.8]


# ---------------------------------------------------------------------------
# 基础算子（对齐 vllm/v1/sample/ 的实现）
# ---------------------------------------------------------------------------

def softmax(logits):
    """数值稳定 softmax：先减 max，避免 exp 溢出"""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def apply_temperature(logits, temp):
    """sampler.py::apply_temperature：logits / temperature"""
    if temp < EPS:
        return list(logits)   # greedy 快路径，不缩放
    return [x / temp for x in logits]


def apply_top_k(logits, k):
    """topk_topp_sampler.py::apply_top_k_only：非 top-k 的 token 置 -inf"""
    order = sorted(range(len(logits)), key=lambda i: -logits[i])
    keep = set(order[:k])
    return [x if i in keep else NEG_INF for i, x in enumerate(logits)]


def apply_top_p(logits, p):
    """topk_topp_sampler.py::apply_top_k_top_p 的 top-p 部分。

    升序排序 → softmax → cumsum → mask 掉 probs_sum <= 1-p 的 token，
    最后一个（最大）强制保留（top_p_mask[:, -1] = False）。
    返回 (masked_logits, 排序信息) 供打印 cumsum 曲线。
    """
    order = sorted(range(len(logits)), key=lambda i: logits[i])  # 升序
    s_logits = [logits[i] for i in order]
    s_probs = softmax(s_logits)
    cum, s = [], 0.0
    for pr in s_probs:
        s += pr
        cum.append(s)
    mask = [c <= 1 - p for c in cum]
    mask[-1] = False   # at least one
    out = [NEG_INF] * len(logits)
    for i, m in zip(order, mask):
        out[i] = NEG_INF if m else logits[i]
    return out, order, s_probs, cum, mask


def apply_penalties(logits, output_token_ids, freq_pen, pres_pen):
    """penalties.py::apply_all_penalties（OpenAI 定义）：
    logits -= freq_pen * 出现次数;  logits -= pres_pen * (是否出现过)"""
    counts = [0] * len(VOCAB)
    for t in output_token_ids:
        counts[t] += 1
    out = []
    for i, x in enumerate(logits):
        v = x
        if freq_pen:
            v -= freq_pen * counts[i]
        if pres_pen and counts[i] > 0:
            v -= pres_pen
        out.append(v)
    return out


def random_sample(probs, rng):
    """topk_topp_sampler.py::random_sample：argmax(probs / q)，q ~ Exp(1)。
    等价于按 probs 采样，但不用 multinomial（避免 CPU-GPU 同步）。"""
    best, best_i = -1.0, 0
    for i, p in enumerate(probs):
        if p <= 0:
            continue
        q = -math.log(rng.random())   # Exp(1) 随机数
        v = p / q
        if v > best:
            best, best_i = v, i
    return best_i


# ---------------------------------------------------------------------------
# 采样器（对齐 sampler.py::Sampler.sample 的管线顺序）
# ---------------------------------------------------------------------------

class MiniSampler:
    def sample(self, logits, temperature=1.0, top_k=None, top_p=None,
               rng=None, trace=False):
        # 1) greedy 快路径：temperature < EPS 直接 argmax，不采样
        if temperature < EPS:
            if trace:
                print("    [greedy] temperature=0 → argmax 快路径，不采样")
            return max(range(len(logits)), key=lambda i: logits[i])
        # 2) temperature 缩放
        l = apply_temperature(logits, temperature)
        if trace:
            self._show("temperature", l)
        # 3) top-k 截断
        if top_k is not None:
            l = apply_top_k(l, top_k)
            if trace:
                self._show(f"top-k={top_k}", l)
        # 4) top-p 截断
        if top_p is not None:
            l = apply_top_p(l, top_p)[0]
            if trace:
                self._show(f"top-p={top_p}", l)
        # 5) softmax + 按分布采样
        probs = softmax(l)
        if trace:
            self._show("最终分布", probs, is_prob=True)
        return random_sample(probs, rng)

    @staticmethod
    def _show(name, vals, is_prob=False):
        cells = []
        for i, v in enumerate(vals):
            if v == NEG_INF:
                cells.append(f"{VOCAB[i]}:  -inf")
            elif is_prob:
                cells.append(f"{VOCAB[i]}:{v:6.3f}")
            else:
                cells.append(f"{VOCAB[i]}:{v:6.2f}")
        print(f"    {name:12s} | " + " ".join(cells))


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def main():
    s = MiniSampler()

    print("=" * 78)
    print("演示 1：完整管线 trace（temperature=0.8, top-k=6, top-p=0.9）")
    print("=" * 78)
    print("  原始 logits | " + " ".join(f"{VOCAB[i]}:{x:5.2f}" for i, x in enumerate(LOGITS)))
    s.sample(LOGITS, temperature=0.8, top_k=6, top_p=0.9,
             rng=random.Random(0), trace=True)

    print("\n" + "=" * 78)
    print("演示 2：batch 3 个请求，参数不同，mask 前后分布对比")
    print("  req0: greedy (temperature=0)   req1: top-k=5   req2: top-p=0.9")
    print("=" * 78)
    base = softmax(LOGITS)
    s._show("mask 前", base, is_prob=True)
    for name, kw in [("req0 greedy", dict(temperature=0.0)),
                     ("req1 top-k=5", dict(temperature=1.0, top_k=5)),
                     ("req2 top-p=0.9", dict(temperature=1.0, top_p=0.9))]:
        l = apply_temperature(LOGITS, kw.get("temperature", 1.0))
        if "top_k" in kw:
            l = apply_top_k(l, kw["top_k"])
        if "top_p" in kw:
            l = apply_top_p(l, kw["top_p"])[0]
        s._show(f"{name} mask 后", softmax(l), is_prob=True)

    print("\n" + "=" * 78)
    print("演示 3：top-p=0.9 的 cumsum 曲线（升序排列，标出截断点）")
    print("=" * 78)
    _, order, s_probs, cum, mask = apply_top_p(LOGITS, 0.9)
    for rank, i in enumerate(order):
        mark = "  ← 截断点(最后一个 probs_sum<=0.1)" if (mask[rank] and rank < len(order) - 1 and not mask[rank + 1]) else ""
        kept = "保留" if not mask[rank] else "mask"
        print(f"    rank{rank:2d} {VOCAB[i]:5s} p={s_probs[rank]:7.4f}  cumsum={cum[rank]:7.4f}  [{kept}]{mark}")

    print("\n" + "=" * 78)
    print("演示 4：temperature 的效果（同一份 logits，分布形状变化）")
    print("=" * 78)
    for t in [0.1, 1.0, 2.0]:
        s._show(f"t={t}", softmax(apply_temperature(LOGITS, t)), is_prob=True)
    print("    t 越小分布越尖锐（接近 one-hot），t 越大越平坦（接近均匀）")

    print("\n" + "=" * 78)
    print("演示 5：frequency / presence penalty（已生成 [cat, cat, dog]）")
    print("  freq_pen=1.0, pres_pen=0.5 → 重复 token 的 logits 被压低")
    print("=" * 78)
    s._show("penalty 前", LOGITS)
    penalized = apply_penalties(LOGITS, [1, 1, 6], freq_pen=1.0, pres_pen=0.5)
    s._show("penalty 后", penalized)
    for i, n in [(1, 2), (6, 1), (3, 0)]:
        delta = -1.0 * n - (0.5 if n > 0 else 0.0)
        why = f"出现 {n} 次: -1.0*{n}" + (f"-0.5" if n > 0 else "") if n else "未出现: 不变"
        print(f"    {VOCAB[i]:5s}: {LOGITS[i]:5.2f} → {penalized[i]:5.2f}  ({why})")

    print("\n" + "=" * 78)
    print("演示 6：固定 seed=42 重复采样 20 次，实测频率 vs 理论概率")
    print("  参数：temperature=1.0，无 top-k/top-p（理论分布 = softmax(logits)）")
    print("=" * 78)
    theory = softmax(LOGITS)
    rng = random.Random(42)
    hits = [0] * len(VOCAB)
    for _ in range(20):
        hits[random_sample(theory, rng)] += 1
    print(f"    {'token':6s} {'理论':>8s} {'实测':>8s} {'次数':>6s}")
    for i in range(len(VOCAB)):
        print(f"    {VOCAB[i]:6s} {theory[i]:8.4f} {hits[i] / 20:8.4f} {hits[i]:5d}/20")
    print("    20 次样本少、噪声大；把循环改成 2000 次，实测会明显贴近理论值")


if __name__ == "__main__":
    main()
