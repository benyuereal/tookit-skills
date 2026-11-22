#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖 GPU 的逻辑回归：patch 前后的 dtype 判据是否按预期变化。

为什么要有这一步：patch 改的是「哪些 dtype 允许走 persistent」。
这个判断是纯逻辑，不需要显卡就能验证；把不需要 GPU 的部分先验掉，
上机时只需要验性能，排查面小很多。
"""
import argparse
import importlib.util
import sys


def load_mqa(path):
    spec = importlib.util.spec_from_file_location("_mqa_" + str(abs(hash(path)) % 99999), path)
    mod = importlib.util.module_from_spec(spec)
    # 不执行整个模块（它会 import torch），只读源码做静态判断
    return open(path, encoding="utf-8").read()


def persistent_dtypes(src):
    """从源码里抽出 _persistent_supported 对 gfx938 允许的 dtype 集合。

    只做源码级判断，不 import torch——这样在无卡环境也能跑。
    """
    import re
    m = re.search(r'(?s)def _persistent_supported.*?return False', src)
    if not m:
        return None
    body = m.group(0)
    # 只看 gfx938 分支
    g = re.search(r'(?s)if arch == "gfx938".*?(?=return False|if arch ==)', body)
    if not g:
        return None
    seg = g.group(0)
    return {
        "int8": "q.dtype == torch.int8" in seg,
        "fp8": "float8_e4m3fn" in seg,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="mqa.py persistent dtype 门禁逻辑回归")
    p.add_argument("--before", required=True, help="patch 前的 mqa.py")
    p.add_argument("--after", required=True, help="patch 后的 mqa.py")
    a = p.parse_args(argv)

    before = persistent_dtypes(load_mqa(a.before))
    after = persistent_dtypes(load_mqa(a.after))

    print("patch 前 gfx938 persistent 允许:", before)
    print("patch 后 gfx938 persistent 允许:", after)

    ok = True
    if before is None or after is None:
        print("FAIL: 找不到 _persistent_supported 的 gfx938 分支")
        return 1
    if not after["int8"]:
        print("FAIL: patch 后 int8 仍应允许 persistent（实测 int8 上 persistent 有效）")
        ok = False
    if after["fp8"]:
        print("FAIL: patch 后 fp8 不应再允许 persistent")
        ok = False
    if before["fp8"] and not after["fp8"]:
        print("PASS: fp8 已从 persistent 白名单移除；int8 保留")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
