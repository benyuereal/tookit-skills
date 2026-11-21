#!/usr/bin/env python3
"""把「下一块 gather 提前到本次 GEMM 之前」的预取改写做成可复跑脚本。

为什么需要它：`sparse_mla` 是访存受限（算术强度 ≈30 FLOP/B，交叉点 ≈143），
但 gather 已经打到流式峰值的 94~97%（`bench/bench.py --mode workset` 的 mem-only
变体），整 kernel 却只有 ~55%。根因不是带宽也不是向量化，是**每 CU 只驻 1 个 CTA**
（vgpr_count=217 × 256 线程 = 55552，两个 CTA 超 64K 寄存器堆），线程在等 gather
时没有任何其他工作可填。于是把下一块的 gather 提前发出去，用寄存器做预取缓冲。

改写内容（对每个 `T.Pipelined(valid_NI[0], ...)` 主循环各做一遍）：
  1. 循环体开头：把上一轮预取到的寄存器内容 `T.copy(..._local -> ..._shared)`；
  2. 原位：gather 的索引从 `i_i * BI` 换成 `min(i_i + 1, valid_NI[0] - 1) * BI`
     （最后一块重复取一次，无害且不会越界）；spilt0~3 与 tail 全部纳入；
  3. 循环前加 prologue：block 0 先取到寄存器。

**只预取 spilt0~2**。把 spilt3 / tail 也搬进 shared 加预取（另一个版本，pf2）会慢 1.7×：
spilt3 现在直接从寄存器喂 MMA，搬进 shared 后多出 12KB 的 local->shared 写 + shared 读，
LDS 流水立刻成为新的瓶颈（VGPR 217->231，未 spill，纯 LDS 代价）。这条否掉了，别再试。

实测（2026-10-27，gfx938，单卡独占，ctx=320k，ns=1）：整 kernel 快 5~6%，**输出逐位一致**。

用法：
    # 只导出改写后的源码（人工阅读 / 打补丁）
    python3 bench/prefetch.py --vendor /path/to/tilelang-hygon --emit prefetch_fwd.py

    # 直接跟原版 A/B（同进程交替，比值才有意义）
    python3 bench/prefetch.py --vendor /path/to/tilelang-hygon --shape decode-s128 --json prefetch.json
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

CARD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD_DIR))

LOOP_RE = re.compile(r"^(?P<ind>[ \t]*)for i_i in T\.Pipelined\(valid_NI\[0\], num_stages=num_stages\):\n", re.M)


def _body_span(src, m):
    """返回 (body_start, body_end, indent)：循环体在源码里 [start, end) 的跨度与语句缩进。"""
    head_end = m.end()
    ind = len(m.group("ind"))
    body_ind = None
    pos = head_end
    end = len(src)
    for line in src[head_end:].splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            cur = len(line) - len(line.lstrip())
            if body_ind is None:
                body_ind = cur
            elif cur <= ind:
                end = pos
                break
        pos += len(line)
    return head_end, end, (body_ind if body_ind is not None else ind + 4)


def _dedent(text, n):
    return "".join(l[n:] if l[:n].strip() == "" else l for l in text.splitlines(keepends=True))


def _reindent(text, n):
    pad = " " * n
    return "".join(pad + l if l.strip() else l for l in text.splitlines(keepends=True))


# 两种块索引写法：no-split 内核用 `i_i * BI`，split 内核先算 `idx_in_split = i_i * split_stride + split_idx * BI`。
# 认不出的写法一律跳过（宁可不动，也不能把 copy 挪到循环开头而 gather 还用旧索引——那会静默算错）。
FORMS = (
    dict(cur="i_i * BI + bi_i", first="bi_i",
         nxt="T.min(i_i + 1, valid_NI[0] - 1) * BI + bi_i"),
    dict(cur="idx_in_split + bi_i", first="split_idx * BI + bi_i",
         nxt="T.min(i_i + 1, valid_NI[0] - 1) * split_stride + split_idx * BI + bi_i"),
)


def _transform_body(body):
    """body 已 dedent 到语句 0 缩进。返回 (改写后的 body, prologue 的块 0 索引) 或 None。"""
    copy0 = ("T.copy(KV_spilt0_local, KV_spilt0_shared)\n"
             "T.copy(KV_spilt1_local, KV_spilt1_shared)\n"
             "T.copy(KV_spilt2_local, KV_spilt2_shared)\n")
    if copy0 not in body or "KV_spilt3_local[bi_i, d_i]" not in body:
        return None
    key = "KV_spilt0_local[bi_i, d_i]"
    seg_beg = body.rindex("for bi_i, d_i in T.Parallel(BI, D_spilt):", 0, body.index(key))
    nxt_beg = body.rindex("for bi_i, d_i in T.Parallel(BI, D_spilt):",
                          0, body.index("KV_spilt3_local[bi_i, d_i]"))
    seg = body[seg_beg:nxt_beg]
    form = next((f for f in FORMS if f["cur"] in seg), None)
    if form is None:
        return None
    # 1) 只改写 spilt0~2 的 gather 索引；mask 初始化、spilt3/tail 仍用当前块
    #    （spilt3/tail 是寄存器直喂 MMA，搬 shared 会慢 1.7×）
    #    先改这一段再搬 copy —— copy0 在前面插入会挪动所有偏移
    body = body[:seg_beg] + seg.replace(form["cur"], form["nxt"]) + body[nxt_beg:]
    # 2) 原来在全部 gather 之后的三个 copy 挪到循环体开头（读上一轮预取到的寄存器）
    body = copy0 + body.replace(copy0, "", 1)
    return body, form["first"]


def _prologue(indent, first):
    """循环前把 block 0 取到寄存器（缩进 = 主循环头缩进）。"""
    pad = " " * indent
    out = pad + "# --- 预取 prologue: block 0 先取到寄存器 ---\n"
    out += pad + "for bi_i, d_i in T.Parallel(BI, D_spilt):\n"
    for j, off in enumerate(("d_i", "D_spilt + d_i", "2 * D_spilt + d_i")):
        out += pad + "    KV_spilt%d_local[bi_i, d_i] = T.if_then_else(\n" % j
        out += pad + "        Indices[b_i, s_i, g_i, %s] >= 0, KV[b_i, Indices[b_i, s_i, g_i, %s], g_i, %s], 0\n" % (first, first, off)
        out += pad + "    )\n"
    return out


def transform(src):
    """对源码里所有主循环做预取改写；返回 (新源码, 改写个数)。"""
    n = 0
    spans = []
    for m in LOOP_RE.finditer(src):
        body_start, body_end, body_ind = _body_span(src, m)
        spans.append((m.start(), m.group("ind"), body_start, body_end, body_ind))
    for loop_start, ind, s, e, bi in reversed(spans):
        body = _dedent(src[s:e], bi)
        got = _transform_body(body)
        if got is None:
            continue
        new_body, first = got
        src = src[:loop_start] + _prologue(len(ind), first) + src[loop_start:s] \
            + _reindent(new_body, bi) + src[e:]
        n += 1
    return src, n


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    p = argparse.ArgumentParser(description="sparse-mla gather 预取改写 + A/B")
    p.add_argument("--vendor", required=True, help="tilelang-hygon 仓库根目录")
    p.add_argument("--shape", default=None, help="不给就只做源码改写，不跑 GPU")
    p.add_argument("--emit", default=None, help="把改写后的模块写到这个路径")
    p.add_argument("--json", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    vendor = Path(a.vendor).resolve()
    src_path = vendor / "perf/sparse_mla/sparse_mla_fwd.py"
    sys.path.insert(0, str(vendor))
    src = src_path.read_text(encoding="utf-8")
    new_src, n = transform(src)
    print("# 改写主循环 %d 处" % n)
    if not n:
        print("# verdict=fail（没有匹配到主循环，厂商文件结构可能已变）")
        return 1
    if a.emit:
        Path(a.emit).write_text(new_src, encoding="utf-8")
        print("# wrote %s" % a.emit)
    if not a.shape:
        return 0

    # 只在需要跑数值时导入 torch / 卡片工具
    import torch  # noqa: F401
    from _common import DV, SM_SCALE, build_inputs, compare, load_shape, _load_tilelang_module  # noqa: E402
    from opbench import bench_fn  # noqa: E402

    base = _load_tilelang_module(str(vendor))
    tmp = tempfile.mkdtemp(prefix="sparse_mla_prefetch_")
    patched_path = os.path.join(tmp, "sparse_mla_prefetch_fwd.py")
    Path(patched_path).write_text(new_src, encoding="utf-8")
    patched = _load(patched_path, "sparse_mla_prefetch_fwd")

    shape = load_shape(a.shape)
    params = dict(shape.get("params") or {})
    q, kv, idx, meta = build_inputs(params, seed=0, device=a.device)
    out = {}
    ref = None
    for tag, mod in (("tilelang", base), ("tilelang-prefetch", patched)):
        o = mod.tilelang_sparse_fwd(q, kv, idx, SM_SCALE, d_v=DV)
        torch.cuda.synchronize()
        o = o.reshape(q.shape[0], q.shape[1], DV) if hasattr(o, "reshape") else o
        if ref is None:
            ref = o.clone()
            ok = {"max_abs": 0.0, "cosine": 1.0}
        else:
            ok = compare(o, ref)
        ms = bench_fn(lambda: mod.tilelang_sparse_fwd(q, kv, idx, SM_SCALE, d_v=DV),
                      warmup_ms=25.0, rep_ms=150.0, repeats=3)["median_ms"]
        out[tag] = {"median_ms": ms, "vs_tilelang": ok}
        print("%-20s %.4f ms  vs tilelang max_abs=%.3e cos=%.7f" % (tag, ms, ok["max_abs"], ok["cosine"]))
    r = out["tilelang-prefetch"]["median_ms"] / out["tilelang"]["median_ms"]
    print("# 预取版/原版 = %.4f  (%.1f%%)" % (r, (r - 1) * 100))
    if a.json:
        Path(a.json).write_text(json.dumps({"op": "sparse-mla", "shape": shape.get("shape"),
                                            "params": meta, "result": out, "ratio": r},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
        print("# wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
