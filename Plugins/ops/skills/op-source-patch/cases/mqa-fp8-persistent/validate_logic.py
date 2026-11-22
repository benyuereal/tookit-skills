#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖 GPU 的逻辑回归：patch 前后 persistent 的 dtype 白名单是否按预期变化。

为什么要有这一步：patch 改的是「哪些 dtype 允许走 persistent」，这是一个
**纯逻辑判断**，不需要显卡就能验证。把不需要 GPU 的部分先验掉，
上机时只需验性能，排查面小很多。

用法:
    python3 validate_logic.py --before <原始 mqa.py> --after <patch 后 mqa.py>
"""
import argparse
import sys


def _slice_function(lines, name):
    """按缩进切出一个函数的函数体行。"""
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("def %s(" % name):
            start = i
            break
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for ln in lines[start + 1:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        body.append(ln)
    return body


def _slice_branch(body, marker):
    """按缩进切出 if 分支（marker 是分支行里的特征串）。"""
    start = None
    for i, ln in enumerate(body):
        if marker in ln:
            start = i
            break
    if start is None:
        return None
    indent = len(body[start]) - len(body[start].lstrip())
    seg = [body[start]]
    for ln in body[start + 1:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        seg.append(ln)
    return "\n".join(seg)


def persistent_dtypes(path):
    """从源码里抽出 _persistent_supported 对 gfx938 允许的 dtype。

    只做源码级解析，**不 import torch** —— 无卡环境也能跑。
    按缩进切片而不是正则：函数体里有多个 `return False`，
    非贪婪正则会停在第一个上（第一版就栽在这里）。
    """
    lines = open(path, encoding="utf-8").read().split("\n")
    body = _slice_function(lines, "_persistent_supported")
    if body is None:
        return None
    seg = _slice_branch(body, 'arch == "gfx938"')
    if seg is None:
        return None
    return {
        "int8": "q.dtype == torch.int8" in seg,
        "fp8": "float8_e4m3fn" in seg,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="mqa.py persistent dtype 门禁逻辑回归")
    p.add_argument("--before", required=True, help="patch 前的 mqa.py")
    p.add_argument("--after", required=True, help="patch 后的 mqa.py")
    a = p.parse_args(argv)

    try:
        before = persistent_dtypes(a.before)
        after = persistent_dtypes(a.after)
    except OSError as e:
        print("读不到文件: %s" % e, file=sys.stderr)
        return 2

    print("patch 前 gfx938 persistent 允许:", before)
    print("patch 后 gfx938 persistent 允许:", after)

    if before is None or after is None:
        print("FAIL: 找不到 _persistent_supported 的 gfx938 分支")
        return 1

    ok = True
    # int8 上 persistent 实测有效 → 必须保留
    if not after["int8"]:
        print("FAIL: patch 后 int8 仍应允许 persistent（实测 int8 上有效）")
        ok = False
    # FP8 实测变慢 + 小 page 分组 RuntimeError → 必须移除
    if after["fp8"]:
        print("FAIL: patch 后 fp8 不应再允许 persistent")
        ok = False
    if ok:
        print("PASS: fp8 已从 persistent 白名单移除；int8 保留")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
