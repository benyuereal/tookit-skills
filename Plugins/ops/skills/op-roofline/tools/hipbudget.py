#!/usr/bin/env python3
"""从**生成代码的声明段**算寄存器预算 —— 作用域感知，不把不同作用域的声明平铺求和。

为什么不能平铺求和（2026-10-28 修正）
--------------------------------------
bm32/thr256 的生产生成代码里，函数体作用域（depth-1）声明约 216 VGPR；循环内
还有若干**兄弟块**（depth-3，形如 ``{ bfloat16_t B_local[8]; ... }``）合计 68 VGPR。
旧版把它们相加得 283 ≥ 256，报「保证溢出」；实测编译产物是 **256 VGPR / spill 0**。
两者不同时活跃：兄弟块各自在右花括号处结束，寄存器被复用。

模型（静态、保守，不做语句级 liveness）
---------------------------------------
  * 同一作用域内的声明相加；
  * 兄弟块（不同分支 / 不同嵌套块）**取最大值**，不相加；
  * 一个块内的活集合 = 它自己的声明 + 所有**祖先作用域**的声明
    （外层变量在进入内层时通常仍活着）；
  * 对所有作用域取「祖先链」之和的最大值作为活集合峰值。
  * 分支按花括号分块：`if/else { }`、独立的 `{ }` 块天然是兄弟块，取 max。
    无花括号的单语句分支不建模（生成代码总是带花括号），遇到时保守相加。
  * 这是静态下界近似：临时量不进声明段，**绝对数值以编译产物
    vgpr_count / vgpr_spill_count 为准**。

校准（gfx938 / tilelang-hygon 生成代码）
----------------------------------------
| 设计        | 旧版平铺 | 本版活集合峰值 | 编译器实报            | 判定      |
|-------------|----------|----------------|-----------------------|-----------|
| bm32/thr256 | 283      | ≤ 256          | 256 / spill 0         | 未超限 ✓  |
| bm64/thr256 | 451      | > 256          | 256 / 有 spill(≥30 B) | 判为溢出 ✓ |

能说的 / 不能说的
-----------------
  能说：活集合峰值 ≥ 单线程上限 ⇒ 溢出高度可预期；两个设计的峰值之差 ≈ 改动的寄存器代价。
  不能说：「没超 ⇒ 一定不溢出」（临时量不进声明段），也不能拿它做绝对数值判定。

用法
----
    python3 hipbudget.py <file.hip>                     # 列全部声明 + 作用域聚合
    python3 hipbudget.py <file.hip> --live acc_o,Q,KV   # 只算名字含这些前缀的声明
    python3 hipbudget.py <file.hip> --live Q,acc_o --threads 256
"""
import argparse
import re
import sys
from pathlib import Path

# 类型 → 每个元素的字节数
BYTES = {
    "float": 4, "float32_t": 4, "int": 4, "int32_t": 4, "unsigned": 4,
    "unsigned int": 4, "uint": 4, "uint32_t": 4,
    "float16_t": 2, "half": 2, "_Float16": 2, "__half": 2,
    "bfloat16_t": 2, "bfloat16": 2, "__bf16": 2, "short": 2,
    "unsigned short": 2, "uint16_t": 2, "int16_t": 2,
    "double": 8, "float64_t": 8, "long": 8, "int64_t": 8,
}
DECL = re.compile(r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
VEC = re.compile(r"\b(float|int|unsigned)([248])\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")

VF_LIMIT = 196608           # 每 CU 的 VGPR 文件（32-bit 寄存器数）；768 模式 = 4 SIMD × 64 lane × 768。
                            # 未开 `-mllvm=-support-768-vgprs=true` 时是 65536（= 4 × 64 × 256）。
                            # 依据：hbird-gemm 172 VGPR × 512 线程实测 2 CTA/CU，只有 196608 能复现。
VGPR_PER_THREAD_CAP = 256   # 单线程硬上限

# 这些限定符声明的数组不在 per-thread VGPR 里
NON_VGPR_QUAL = ("__shared__", "__constant__", "__device__", "constant__", "device__")


def strip_code(src):
    """去掉注释与字符串/字符字面量，用等长空格替换（保持正则偏移不变）。"""
    out = list(src)
    i, n, mode = 0, len(src), None
    while i < n:
        c = src[i]
        d = src[i + 1] if i + 1 < n else ""
        if mode == "line":
            if c == "\n":
                mode = None
            else:
                out[i] = " "
        elif mode == "block":
            if c == "*" and d == "/":
                out[i] = out[i + 1] = " "
                i += 1
                mode = None
            elif c != "\n":
                out[i] = " "
        elif mode in ("str", "chr"):
            if c == "\\":
                out[i] = " "
                if i + 1 < n and src[i + 1] != "\n":
                    out[i + 1] = " "
                i += 1
            elif (mode == "str" and c == '"') or (mode == "chr" and c == "'"):
                mode = None
            elif c != "\n":
                out[i] = " "
        else:
            if c == "/" and d == "/":
                mode = "line"
                out[i] = out[i + 1] = " "
                i += 1
            elif c == "/" and d == "*":
                mode = "block"
                out[i] = out[i + 1] = " "
                i += 1
            elif c == '"':
                mode = "str"
                out[i] = " "
            elif c == "'":
                mode = "chr"
                out[i] = " "
        i += 1
    return "".join(out)


def _is_non_vgpr(src, pos):
    """声明前（上一条 ; { } 之后）若有 shared/constant 限定符，则不是 VGPR。"""
    start = max(src.rfind(";", 0, pos), src.rfind("{", 0, pos), src.rfind("}", 0, pos))
    head = src[start + 1:pos]
    return any(q in head for q in NON_VGPR_QUAL)


def scan(path):
    """解析文件，返回声明（带作用域）与作用域树。

    返回 dict：
      decls   : [{name, n, bytes, ty, scope, depth}]
      parent  : 每个作用域 id → 父 id（根为 -1）
      depth   : 每个作用域 id → 嵌套深度
      degraded: 花括号不配对时为 True（此时调用方不得做绝对判定）
    """
    raw = Path(path).read_text(errors="ignore")
    src = strip_code(raw)

    # 1) 收集候选声明（同名只取第一次出现，避免 VEC 重复计）
    cands, seen = [], set()
    for m in DECL.finditer(src):
        ty, name, n = m.group(1), m.group(2), m.group(3)
        if ty in BYTES and name not in seen:
            cands.append((m.start(), name, int(n), BYTES[ty], ty))
            seen.add(name)
    base = {"float": 4, "int": 4, "unsigned": 4}
    for m in VEC.finditer(src):
        ty, vec, name, n = m.groups()
        if name not in seen:
            cands.append((m.start(), name, int(n) * int(vec), base[ty], ty + vec))
            seen.add(name)
    cands.sort()

    # 2) 花括号 → 作用域树；声明归属于它所在的最内层块
    unbalanced = src.count("{") != src.count("}")
    events = [(m.start(), m.group()) for m in re.finditer(r"[{}]", src)]
    parent = [-1]
    depth = [0]
    stack_cur = 0
    ei = 0
    decls = []
    for pos, name, n, b, ty in cands:
        while ei < len(events) and events[ei][0] < pos:
            _, ch = events[ei]
            ei += 1
            if ch == "{":
                sid = len(parent)
                parent.append(stack_cur)
                depth.append(depth[stack_cur] + 1)
                stack_cur = sid
            elif stack_cur != 0:
                stack_cur = parent[stack_cur]
        if _is_non_vgpr(src, pos):
            continue
        decls.append({"name": name, "n": n, "bytes": b, "ty": ty,
                      "scope": stack_cur, "depth": depth[stack_cur]})

    return {"decls": decls, "parent": parent, "depth": depth, "degraded": unbalanced}


def _chain_sum(scope, parent, self_sum):
    total, s = 0.0, scope
    while s != -1:
        total += self_sum[s]
        s = parent[s]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hip")
    ap.add_argument("--live", help="逗号分隔的名字前缀，只统计这些（活集合）")
    ap.add_argument("--threads", type=int, default=256)
    a = ap.parse_args()

    info = scan(a.hip)
    decls = info["decls"]
    if not decls:
        raise SystemExit("没解析到定长数组声明：%s" % a.hip)

    prefixes = [p.strip() for p in a.live.split(",") if p.strip()] if a.live else None
    rows = [d for d in decls
            if prefixes is None or any(d["name"].startswith(p) for p in prefixes)]
    if not rows:
        raise SystemExit("--live 一个都没匹配上；可用名字：%s"
                         % ", ".join(sorted({d["name"] for d in decls})))

    for d in rows:
        d["vgpr"] = d["n"] * d["bytes"] / 4.0

    w = max(len(d["name"]) for d in rows)
    for d in sorted(rows, key=lambda x: x["name"]):
        print("  %-*s %4d × %-10s = %6.1f VGPR  [d%d]"
              % (w, d["name"], d["n"], d["ty"], d["vgpr"], d["depth"]))

    # 3) 作用域聚合：同作用域相加；祖先链求和后取最大（兄弟块自然取 max）
    parent, depth = info["parent"], info["depth"]
    self_sum = [0.0] * len(parent)
    for d in rows:
        self_sum[d["scope"]] += d["vgpr"]

    cap_thread = VGPR_PER_THREAD_CAP
    cap_cta = VF_LIMIT / a.threads
    cap = min(cap_thread, cap_cta)

    if info["degraded"]:
        flat = sum(d["vgpr"] for d in rows)
        print("\ncaveat 花括号不配对，作用域解析失败 —— 退回平铺求和（仅可作相对比较）")
        print("平铺声明量合计 %6.1f VGPR/线程" % flat)
        print("单线程硬上限   %6.0f       每 CU 按 %d 线程折算上限 %.0f"
              % (cap_thread, a.threads, cap_cta))
        print("⇒ **不做绝对判定**：绝对数值需以编译产物 vgpr_count / vgpr_spill_count 为准。")
        return 0

    chain = [_chain_sum(s, parent, self_sum) for s in range(len(parent))]
    peak_sid = max(range(len(chain)), key=lambda s: chain[s])
    peak = chain[peak_sid]

    print("\n按作用域聚合（同作用域相加；兄弟块取 max，不跨作用域平铺求和）：")
    for s in range(len(parent)):
        if self_sum[s] == 0:
            continue
        print("  #%-3d d%-2d 自身 %6.1f  → 祖先链 %6.1f VGPR"
              % (s, depth[s], self_sum[s], chain[s]))
    print("\n活集合峰值（祖先链最大值） %6.1f VGPR/线程  @ 作用域 #%d (d%d)"
          % (peak, peak_sid, depth[peak_sid]))
    print("单线程硬上限   %6.0f       每 CU 按 %d 线程折算上限 %.0f（取小 = %.0f）"
          % (cap_thread, a.threads, cap_cta, cap))

    if peak >= cap:
        print("⇒ **判为溢出**（保守判据：活集合峰值 ≥ 单线程上限）。"
              "绝对数值需以编译产物 vgpr_count / vgpr_spill_count 为准。")
        return 1
    print("⇒ 未超限 ⇒ **不保证**不溢出（临时量不进声明段，实测可多出 ~148 VGPR）。"
          "下一步是编译后读 vgpr_count / vgpr_spill_count；绝对数值以编译产物为准。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
