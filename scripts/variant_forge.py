#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""variant_forge —— 从现有实现生成**参数化变体**。

为什么只做 Triton
-----------------
Triton 的 `tl.constexpr` 与 `run(...)` 的关键字参数能干净地覆盖 BLOCK / num_warps。

**HIP 做不到**——tile 大小会影响 LDS 声明、循环边界、寄存器分配，
改一个 `BLOCK_M` 要重写 kernel 结构，不是宏能解决的。
所以 HIP 侧的候选标 `forge: manual`，由 agent 写代码而不是自动生成。
这不是偷懒，是**承认自动化的边界**。硬做会产出「看起来改了但没改」的变体——
那种东西比没有更糟，因为它会被当成实验证据。

反作弊：锚点找不到必须报错
--------------------------
`surfaces.yaml` 的教训是「产出看起来跑了但没改的变体」。所以本工具：
  - 每个 patch 项必须在源码里找到**恰好一个**锚点，找到 0 个或 >1 个都报错
  - 生成后必须做**回读验证**：新值真的出现在源码里，且与原值不同
  - `--dry-run` 打印 diff 但不写文件

用法
----
    python3 -I scripts/variant_forge.py --list
    python3 -I scripts/variant_forge.py --parent triton-reduce --patch num_warps=8 --out /tmp/v.py
    python3 -I scripts/variant_forge.py --parent triton-reduce --patch BLOCK=2048 --dry-run
    python3 -I scripts/variant_forge.py --selftest
"""
import argparse
import difflib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAFFOLD = REPO / "scripts/kernel_scaffold.py"

# 每个参数：在源码里的锚点形态 + 合法值
# 锚点用「恰好出现一次」的字符串，避免误替换。
PATCHABLE = {
    "num_warps": {
        "anchor": "num_warps=4",
        "render": lambda v: "num_warps=%d" % int(v),
        "range": (1, 32),
        "why": "Triton 的 warp 数。影响占用率与归约的 lane 分布。",
    },
    "BLOCK": {
        # 模板里 BLOCK 通过位置参数传，与 H 相同
        "anchor": "eps, H, num_warps=num_warps",
        "render": lambda v: "eps, %d, num_warps=num_warps" % int(v),
        "range": (64, 65536),
        "why": "每行一次载入的元素数。H 不是 2 的幂时需要 mask。",
    },
    "vec": {
        # 预留：模板若加了向量宽度参数，这里扩展
        "anchor": None,
        "render": None,
        "range": None,
        "why": "（当前模板未参数化向量宽度——Triton 会自动向量化）",
    },
}

PARENTS = {"triton-reduce": "rmsnorm_k / run 的 Triton 实现"}


def _scaffold():
    spec = importlib.util.spec_from_file_location("_ks", str(SCAFFOLD))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_parent(name):
    """取父模板源码。目前只有一个 triton-reduce。"""
    if name not in PARENTS:
        return None, "未知 parent %r（可选 %s）" % (name, sorted(PARENTS))
    m = _scaffold()
    src, err = m.generate("reduce-lastdim", "triton")
    if err:
        return None, err
    return src, None


def parse_patch(spec):
    """'num_warps=8,BLOCK=2048' → {'num_warps': '8', 'BLOCK': '2048'}"""
    out = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            return None, "patch 项 %r 不是 key=value 形式" % item
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out, None


def apply_patch(src, patch):
    """应用参数覆盖。返回 (new_src, changes, errors)。

    **每条 patch 必须在源码里找到恰好一个锚点。**
    找到 0 个 → 模板变了或参数名错；找到 >1 个 → 会误改别处。
    两种都报错，不许静默跳过。
    """
    changes, errors = [], []
    new = src
    for k, v in patch.items():
        spec = PATCHABLE.get(k)
        if not spec:
            errors.append("不支持的参数 %r（支持：%s）" % (k, sorted(PATCHABLE)))
            continue
        if spec["anchor"] is None:
            errors.append("参数 %r 在当前模板里不可参数化：%s" % (k, spec["why"]))
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            errors.append("参数 %r 的值 %r 不是整数" % (k, v))
            continue
        lo, hi = spec["range"]
        if not (lo <= iv <= hi):
            errors.append("参数 %r=%d 越界（允许 %d~%d）" % (k, iv, lo, hi))
            continue
        n = new.count(spec["anchor"])
        if n != 1:
            errors.append("参数 %r 的锚点在源码里出现 %d 次（必须恰好 1 次）：%r"
                          % (k, n, spec["anchor"]))
            continue
        repl = spec["render"](iv)
        new = new.replace(spec["anchor"], repl)
        changes.append({"param": k, "old": spec["anchor"], "new": repl,
                        "why": spec["why"]})
    return new, changes, errors


def verify(new_src, changes, old_src):
    """回读验证：新值真的出现了，且与原不同。

    这一步防的是「锚点替换了但语义没变」——
    比如把 `num_warps=4` 换成 `num_warps=4`（值没变但调用方以为改了）。
    """
    errs = []
    if new_src == old_src:
        errs.append("生成结果与 parent 完全相同 —— patch 没生效")
    for c in changes:
        if c["new"] not in new_src:
            errs.append("回读失败：%r 不在生成结果里" % c["new"])
        if c["old"] == c["new"]:
            errs.append("参数 %r 的新旧值相同，等于没改" % c["param"])
    return errs


def diff_text(old, new, name="parent", name2="variant"):
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=name, tofile=name2))


def selftest():
    src, err = load_parent("triton-reduce")
    assert src and not err, err
    print("[ok] 父模板可取（%d 行）" % len(src.splitlines()))

    # 1) num_warps 覆盖
    new, ch, errs = apply_patch(src, {"num_warps": "8"})
    assert not errs, errs
    assert "num_warps=8" in new, "新值没进去"
    assert "num_warps=4" not in new
    print("[ok] num_warps=8 覆盖生效")

    # 2) BLOCK 覆盖
    new2, ch2, errs2 = apply_patch(src, {"BLOCK": "2048"})
    assert not errs2, errs2
    assert "eps, 2048," in new2
    print("[ok] BLOCK=2048 覆盖生效")

    # 3) 两个 patch 生成的文件必须不同
    a, _, _ = apply_patch(src, {"num_warps": "8"})
    b, _, _ = apply_patch(src, {"num_warps": "4", "BLOCK": "4096"})
    assert a != b, "两个不同 patch 生成了相同结果"
    print("[ok] 不同 patch → 不同结果")

    # 4) 反向：不认识的参数
    _, _, e = apply_patch(src, {"num_frobs": "3"})
    assert e and "不支持" in e[0], e
    print("[ok] 反向：未知参数报错")

    # 5) 反向：越界
    _, _, e = apply_patch(src, {"num_warps": "0"})
    assert e and "越界" in e[0], e
    print("[ok] 反向：num_warps=0 越界报错")

    # 6) 反向：锚点找不到（改坏的模板）
    broken = src.replace("num_warps=4", "num_warps=XX")
    _, _, e = apply_patch(broken, {"num_warps": "8"})
    assert e and "0 次" in e[0], e
    print("[ok] 反向：锚点找不到报错（不静默跳过）")

    # 7) 反向：新旧值相同
    _, ch, _ = apply_patch(src, {"num_warps": "4"})
    v = verify(src, ch, src)
    assert any("相同" in x or "没生效" in x for x in v), v
    print("[ok] 反向：新旧值相同被 verify 抓到")

    # 8) 生成结果语法正确
    import ast
    new3, _, _ = apply_patch(src, {"num_warps": "8", "BLOCK": "2048"})
    ast.parse(new3)
    print("[ok] 生成结果通过 ast.parse")

    print("\nselftest PASS（8 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="从 Triton 模板生成参数化变体")
    ap.add_argument("--parent", default="triton-reduce")
    ap.add_argument("--patch")
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.list:
        print("可用的 parent：")
        for k, v in PARENTS.items():
            print("  %-16s %s" % (k, v))
        print("\n可 patch 的参数：")
        for k, v in PATCHABLE.items():
            rng = ("%d~%d" % v["range"]) if v["range"] else "（当前不可参数化）"
            print("  %-12s %-16s %s" % (k, rng, v["why"]))
        return 0
    if not a.patch:
        ap.error("给 --patch 或 --list 或 --selftest")

    src, err = load_parent(a.parent)
    if err:
        print("ERROR: %s" % err, file=sys.stderr)
        return 2
    patch, err = parse_patch(a.patch)
    if err:
        print("ERROR: %s" % err, file=sys.stderr)
        return 2
    new, changes, errs = apply_patch(src, patch)
    if errs:
        print("ERROR: patch 失败", file=sys.stderr)
        for e in errs:
            print("  - %s" % e, file=sys.stderr)
        return 2
    verrs = verify(new, changes, src)
    if verrs:
        print("ERROR: 回读验证失败", file=sys.stderr)
        for e in verrs:
            print("  - %s" % e, file=sys.stderr)
        return 2

    d = diff_text(src, new)
    print(d if d else "（无变化）")
    print()
    print("改动 %d 处：%s" % (len(changes),
                            ", ".join(c["param"] for c in changes)))
    if a.dry_run:
        print("[dry-run] 未写文件")
        return 0
    if not a.out:
        ap.error("要给 --out（或 --dry-run）")
    out = Path(a.out)
    out.write_text(new, encoding="utf-8")
    print("已写 %s（%d 行）" % (out, len(new.splitlines())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
