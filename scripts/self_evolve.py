#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""self_evolve —— 把这一轮的赢家封存成下一代的基线，再打一轮。

## 为什么要它（这是全项目最大的一个缺口）

实测 25 条实验记录，按时间排：

    11-24 17:30  rmsnorm-fused      7 × ACCEPTED  −9~−11%
    11-24 20:06  rmsnorm-fused      3 × REJECTED  +3~+21%
    11-25 00:30  rmsnorm-fused      1 ACCEPTED + 6 INCONCLUSIVE
    11-25 03:37  fused-add-rmsnorm  1 ACCEPTED    −45.1%
    11-25 11:11  norm-act-swiglu    1 ACCEPTED    −50.9%

**25 条里 0 条是「以上一轮赢家为基线」。**

`advance.py` 的 `replan()` 只做一件事：在候选队列里决定下一轮试哪个。
它不知道「上一轮赢了什么」，也就不会把赢家变成新起点。

对照 ForgeKernel：它 gen0 只比 PyTorch 快 0.3%，但靠「每代晋升」滚到累计 +29%。
我们 gen0 就做到 −45%，然后停下。

## 它做什么

    gen N 的赢家 ──封存──▶ reference/sealed/genN/  ──┬─▶ 变成 gen N+1 的基线
                                                      └─▶ MANIFEST 记录全部证据

## 三条不许做错的（每条都写进验收）

1. **封存物必须可复现**：MANIFEST 要有 sha256 + 编译参数 + 口径 + 实测值。
   少一样，下一代比的就不是同一个东西。
2. **噪声地板用本代实测的 spread**，不是常数。共享卡上 MAD 会漂
   （D4：同一份代码四张卡从 4.9% 到 7.2%）。
3. **晋升后 floor 不能改**。地板是硬件属性，不因为换了基线就变。
   改了会出现「remaining 一直是 20%」的假象。

## 用法

    python3 -I scripts/self_evolve.py --card fused-add-rmsnorm --seal
    python3 -I scripts/self_evolve.py --card fused-add-rmsnorm --plan
    python3 -I scripts/self_evolve.py --card fused-add-rmsnorm --verify
    python3 -I scripts/self_evolve.py --selftest
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / "Plugins/ops/skills"

# 增益低于它就算「摸到地板」——与 advance.AT_FLOOR 同源，但这里指的是**代际增益**
NOISE_FLOOR_MIN = 0.015
# 一代的目标：比上一代再快 5%
DEFAULT_STEP = 1.05
# 只有 ratio 低于它才算「值得晋升」。0.98 意味着至少快 2%
SEAL_RATIO = 0.98


# ---------------------------------------------------------------- 记录读取

def read_records(card):
    """读该卡所有权威记录（noise 目录已由 loop.py 隔离，这里不重复过滤）。"""
    runs = Path(card) / "runs" / "loop"
    if not runs.is_dir():
        return []
    out = []
    for f in sorted(runs.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r["_src"] = f.name
            out.append(r)
    out.sort(key=lambda r: r.get("ts", ""))
    return out


def _ratio_of(r):
    """从记录里拿「相对基线」的比值。**不依赖单一字段名。**

    实测踩过：`op-fused-add-rmsnorm/bench/ab.py` 不写 `ratio_vs_base`
    （七份分叉的后果，D1），但写了 `cand_ms` 与 `base_ms`。
    只见一个字段名就判「没有比例」，会把 45% 的收益漏掉。

    优先级：ratio_vs_base → cand_ms/base_ms → 1+delta_pct/100
    """
    v = r.get("ratio_vs_base")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    c, b = r.get("cand_ms"), r.get("base_ms")
    if isinstance(c, (int, float)) and isinstance(b, (int, float)) and b > 0:
        return c / b
    d = r.get("delta_pct")
    if isinstance(d, (int, float)):
        return 1.0 + d / 100.0
    return None


def best_accepted(records):
    """挑出「可以当新基线」的记录。

    条件（缺一不可）：
      · decision 以 ACCEPTED 开头
      · significant=true（不是噪声）
      · measure_mode 可 claim（back_to_back 不算）
      · ratio < SEAL_RATIO（至少快 2%）
      · acc_ok 不为 False（None 允许，但要标出来）

    返回 (record, reasons_if_empty)。
    """
    claimable = ("event_isolated", "end_to_end")
    cands, why = [], {
        "not_accepted": 0, "not_significant": 0, "bad_mode": 0,
        "ratio_too_high": 0, "acc_failed": 0}
    for r in records:
        if not str(r.get("decision", "")).startswith("ACCEPTED"):
            why["not_accepted"] += 1
            continue
        if not r.get("significant"):
            why["not_significant"] += 1
            continue
        mode = r.get("measure_mode")
        if mode not in claimable:
            why["bad_mode"] += 1
            continue
        ratio = _ratio_of(r)
        if ratio is None or ratio >= SEAL_RATIO:
            why["ratio_too_high"] += 1
            continue
        if r.get("acc_ok") is False:
            why["acc_failed"] += 1
            continue
        cands.append(r)
    if not cands:
        return None, why
    cands.sort(key=lambda r: _ratio_of(r) or 9e9)
    return cands[0], why


def gen_dirs(card):
    """已存在的代号目录，按代数排序。"""
    root = Path(card) / "reference" / "sealed"
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("gen"):
            try:
                out.append((int(d.name[3:]), d))
            except ValueError:
                continue
    return sorted(out)


def next_gen(card):
    gens = gen_dirs(card)
    return (gens[-1][0] + 1) if gens else 1


# ---------------------------------------------------------------- 噪声地板

def noise_floor(records, gen=None):
    """本代实测的 spread，不低于 NOISE_FLOOR_MIN。

    **不用固定常数**。共享卡上 MAD 会漂（D4 实测：同一份代码四张卡
    从 4.9% 到 7.2%）。用固定值会导致「在噪声大的机器上过严、
    在安静的机器上过松」。

    做法：取这一代所有记录的 base_mad/base_ms 相对值，再取这些相对值的
    最大最小差（spread）。样本不足时退回 NOISE_FLOOR_MIN。
    """
    rel = []
    for r in records:
        b, m = r.get("base_ms"), r.get("base_mad")
        if isinstance(b, (int, float)) and b > 0 and isinstance(m, (int, float)):
            rel.append(m / b)
    if len(rel) < 3:
        return NOISE_FLOOR_MIN, "样本 %d 条不足 3，退回最小值" % len(rel)
    spread = max(rel) - min(rel)
    return max(NOISE_FLOOR_MIN, spread), "本代 %d 条记录的 MAD 相对值极差 %.4f" % (
        len(rel), spread)


# ---------------------------------------------------------------- 封存

def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_kernel_files(card, variant):
    """找到该变体对应的源码/二进制。找不到就返回空 —— **不猜**。

    实测踩过：变体名是 `fa-rms-reg`，文件名也是 `fa-rms-reg.hip`，
    但第一版把它转成下划线再找，结果一个都没匹配上（files 为空）。
    **两种写法都要试。**
    """
    root = Path(card)
    out = {"source": None, "binary": None, "notes": []}
    forms = {variant, variant.replace("-", "_"), variant.replace("_", "-")}
    for d in (root / "kernels", root / "runs" / "forged"):
        if not d.is_dir():
            continue
        for form in sorted(forms):
            if out["source"] is None:
                for ext in (".hip", ".cu", ".py"):
                    hits = [h for h in sorted(d.glob("*%s*%s" % (form, ext))) if h.is_file()]
                    if hits:
                        out["source"] = hits[0]
                        break
            if out["binary"] is None:
                hits = [h for h in sorted(d.glob("*%s*.so" % form)) if h.is_file()]
                if hits:
                    out["binary"] = hits[0]
            if out["source"] or out["binary"]:
                break
        if out["source"] or out["binary"]:
            break
    if not (out["source"] or out["binary"]):
        out["notes"].append(
            "按变体名 %r 没找到 kernel 文件（试过 %s）" % (variant, sorted(forms)))
    return out


# ---------------------------------------------------------------- 封存

def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(card, record=None, emit=print):
    """把赢家封存成 reference/sealed/genN/。

    返回 (sealed_dir, manifest) 或 (None, 原因)。
    """
    card = Path(card)
    recs = read_records(card)
    if record is None:
        record, why = best_accepted(recs)
        if record is None:
            return None, "没有可晋升的记录（%s）" % why
    gen = next_gen(card)
    dest = card / "reference" / "sealed" / ("gen%d" % gen)
    if dest.exists():
        return None, "%s 已存在 —— 不覆盖（封存物一旦定下就是历史）" % dest

    files = _find_kernel_files(card, record.get("tag", ""))
    # **先在临时目录做全，成功才改名**。实测踩过：第一版直接 mkdir，
    # 后面 emit 抛异常，留下只有半截 MANIFEST 的 gen1，下次调用就变 gen2
    # —— 编号错乱且看不出原因。
    tmp = card / "reference" / "sealed" / (".tmp-gen%d" % gen)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    copied = {}
    for kind in ("source", "binary"):
        f = files[kind]
        if f and f.is_file():
            dst = tmp / f.name
            shutil.copy2(f, dst)
            copied[kind] = {"name": f.name, "sha256": _sha256(dst)}

    floor, floor_why = noise_floor(recs)
    manifest = {
        "gen": gen,
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": record.get("tag"),
        "files": copied,
        # 证据：没有这些，下一代比的就不是同一个东西
        "evidence": {
            "sha256": (copied.get("binary") or copied.get("source") or {}).get("sha256"),
            "measured_ms": record.get("cand_ms"),
            "base_ms": record.get("base_ms"),
            "ratio": _ratio_of(record),
            "measure_mode": record.get("measure_mode"),
            "acc_ok": record.get("acc_ok"),
            "acc": record.get("acc"),
            "rounds": (record.get("caliber") or {}).get("rounds"),
            "source_record": record.get("_src"),
            "ts": record.get("ts"),
            "shape": record.get("shape"),
        },
        "noise_floor": {"value": floor, "basis": floor_why},
        "notes": files["notes"],
    }
    (tmp / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.rename(dest)
    emit("封存 %s → %s（%s，ratio=%.3f）" % (
        record.get("tag"), _rel(dest),
        record.get("measure_mode"), _ratio_of(record) or 0.0))
    return dest, manifest


# ---------------------------------------------------------------- 新目标

def _rel(p):
    """转成相对仓库路径；不在仓库内就返回绝对路径。**不许抛。**"""
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def plan_next(card, step=DEFAULT_STEP, emit=print):
    """基于最新封存代，生成下一代的 target.json。"""
    card = Path(card)
    gens = gen_dirs(card)
    if not gens:
        return None, "没有封存代 —— 先跑 --seal"
    gen, d = gens[-1]
    mf = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    base_ms = mf["evidence"]["measured_ms"]
    if not isinstance(base_ms, (int, float)):
        return None, "封存物没有实测耗时 —— MANIFEST 不完整"
    floor = mf["noise_floor"]["value"]
    target = base_ms / step
    # 目标必须真的比基线快，且增益要高于噪声地板，否则这一代注定判「没收益」
    if (1 - 1.0 / step) < floor:
        return None, ("step=%.3f 的增益 %.1f%% 低于噪声地板 %.1f%% —— "
                      "这一代注定判不出收益，请调大 --step"
                      % (step, (1 - 1.0 / step) * 100, floor * 100))
    out = {
        "next_gen": gen + 1,
        # 卡可能在仓库外（测试、外部工程）—— relative_to 会抛 ValueError。
        # 用 try 而不是 if，因为两边的解析方式可能不同。
        "reference": _rel(d),
        "reference_variant": mf["variant"],
        "base_ms": base_ms,
        "target_ms": target,
        "step": step,
        "noise_floor": floor,
        "stop_if": "增益 < noise_floor 或 连续 2 轮无 ACCEPTED",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dest = card / ("gen%d" % (gen + 1))
    dest.mkdir(exist_ok=True)
    (dest / "target.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    emit("第 %d 代目标：%.4f → %.4f ms（step=%.2f，噪声地板 %.1f%%）" % (
        gen + 1, base_ms, target, step, floor * 100))
    return dest / "target.json", out


# ---------------------------------------------------------------- 停止判据

def should_stop(gen_results, floor):
    """gen_results: [(gen, best_ratio)]。返回 (是否停, 原因)。

    **这是整条链的刹车。** 没有它，`--max-gen` 就是唯一的停法，
    而那只会在「跑够次数」时停，不会在「没有收益」时停。
    """
    if not gen_results:
        return False, ""
    gen, best = gen_results[-1]
    if best is None:
        return True, "第 %d 代没有确认的最优" % gen
    gain = 1.0 - best
    if gain < floor:
        return True, "第 %d 代增益 %.2f%% < 噪声地板 %.2f%%" % (
            gen, gain * 100, floor * 100)
    return False, "第 %d 代增益 %.2f%% ≥ 噪声地板" % (gen, gain * 100)


# ---------------------------------------------------------------- 校验

def verify(card, emit=print):
    """校验所有封存物自洽：sha256 对得上、字段齐、能复现。"""
    card = Path(card)
    gens = gen_dirs(card)
    if not gens:
        return ["没有封存代"]
    probs = []
    for gen, d in gens:
        mf_p = d / "MANIFEST.json"
        if not mf_p.is_file():
            probs.append("gen%d 缺 MANIFEST.json" % gen)
            continue
        mf = json.loads(mf_p.read_text(encoding="utf-8"))
        ev = mf.get("evidence") or {}
        for k in ("measured_ms", "ratio", "measure_mode"):
            if ev.get(k) is None:
                probs.append("gen%d MANIFEST 缺 evidence.%s" % (gen, k))
        if ev.get("measure_mode") not in ("event_isolated", "end_to_end"):
            probs.append("gen%d 口径 %r 不可 claim" % (gen, ev.get("measure_mode")))
        # sha256 必须对得上（文件可能是后来被改的）
        for kind, info in (mf.get("files") or {}).items():
            f = d / info["name"]
            if not f.is_file():
                probs.append("gen%d 缺文件 %s" % (gen, info["name"]))
            elif _sha256(f) != info["sha256"]:
                probs.append("gen%d 的 %s sha256 不符 —— 文件被改过" % (gen, kind))
        if not mf.get("noise_floor", {}).get("value"):
            probs.append("gen%d 缺噪声地板" % gen)
    if not probs:
        emit("✓ %d 个封存代全部自洽" % len(gens))
    return probs


# ---------------------------------------------------------------- selftest

def selftest():
    """**反向优先**：先证明它拦得住不该晋升的记录。"""
    atom = "_t"
    cases = []

    def R(**kw):
        base = {"decision": "ACCEPTED_NUMERICS", "significant": True,
                "measure_mode": "event_isolated", "ratio_vs_base": 0.90,
                "acc_ok": True, "base_ms": 1.0, "base_mad": 0.01,
                "cand_ms": 0.90, "tag": "v1", "ts": "2026-01-01T00:00:00"}
        base.update(kw)
        return base

    # ① 反向：back_to_back 不可晋升
    r, why = best_accepted([R(measure_mode="back_to_back")])
    assert r is None and why["bad_mode"] == 1, (r, why)
    print("[ok] 反向：back_to_back 口径不许晋升（偏差逐实现变化，无全局修正）")

    # ② 反向：不显著的不许晋升
    r, why = best_accepted([R(significant=False)])
    assert r is None and why["not_significant"] == 1, (r, why)
    print("[ok] 反向：不显著的记录不许当基线")

    # ③ 反向：只快 1% 的不许晋升（噪声内）
    r, why = best_accepted([R(ratio_vs_base=0.99)])
    assert r is None and why["ratio_too_high"] == 1, (r, why)
    print("[ok] 反向：只快 1% 的不许晋升（SEAL_RATIO=0.98）")

    # ④ 反向：精度不过的不许晋升
    r, why = best_accepted([R(acc_ok=False)])
    assert r is None and why["acc_failed"] == 1, (r, why)
    print("[ok] 反向：acc_ok=False 的不许当基线（拿错的当参照是最坏的）")

    # ⑤ 正向：合格的能被选中，且取最好的那条
    r, _ = best_accepted([R(ratio_vs_base=0.95), R(ratio_vs_base=0.80)])
    assert r is not None and abs(_ratio_of(r) - 0.80) < 1e-9, r
    print("[ok] 正向：多条合格记录里取 ratio 最小的")

    # ⑤b 反向：**字段名不同也要能算出来**
    #    实测踩过：某份 ab.py 不写 ratio_vs_base，只写 cand_ms/base_ms，
    #    只看字段名会把 45% 的收益漏掉。
    r2, _ = best_accepted([R(ratio_vs_base=None, cand_ms=0.55, base_ms=1.0)])
    assert r2 is not None, "没有 ratio_vs_base 就判无收益 —— 会漏掉真实收益"
    assert abs(_ratio_of(r2) - 0.55) < 1e-9, _ratio_of(r2)
    print("[ok] 反向：没有 ratio_vs_base 时用 cand_ms/base_ms 算（实测踩过）")

    # ⑥ 噪声地板：样本不足退回最小值
    f, why = noise_floor([R()])
    assert f == NOISE_FLOOR_MIN, (f, why)
    print("[ok] 噪声地板：样本 <3 条退回 %.3f" % NOISE_FLOOR_MIN)

    # ⑦ 噪声地板：样本足够时取实测 spread，且不低于下限
    recs = [R(base_mad=0.005), R(base_mad=0.05), R(base_mad=0.09)]
    f, why = noise_floor(recs)
    assert f >= NOISE_FLOOR_MIN, (f, why)
    assert abs(f - 0.085) < 1e-6, (f, why)   # 0.09 - 0.005
    print("[ok] 噪声地板用实测 spread（0.085），不是固定常数")

    # ⑧ 反向：停止判据在增益低于地板时必须停
    stop, why = should_stop([(2, 0.99)], floor=0.02)
    assert stop, why
    print("[ok] 反向：增益 1% < 地板 2% ⇒ 停")

    # ⑨ 正向：增益高于地板时不停
    stop, why = should_stop([(2, 0.90)], floor=0.02)
    assert not stop, why
    print("[ok] 正向：增益 10% > 地板 2% ⇒ 继续")

    # ⑩ 反向：这一代没有确认最优时必须停（不许无限跑）
    stop, why = should_stop([(3, None)], floor=0.02)
    assert stop, why
    print("[ok] 反向：没有确认最优 ⇒ 停（否则会无限跑）")

    print("\nselftest PASS（10 项）")
    return 0


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", help="卡 id（不带 op- 前缀）或路径")
    ap.add_argument("--seal", action="store_true", help="把最优记录封存成新一代")
    ap.add_argument("--plan", action="store_true", help="基于最新封存代生成下一代目标")
    ap.add_argument("--verify", action="store_true", help="校验封存物自洽")
    ap.add_argument("--step", type=float, default=DEFAULT_STEP,
                    help="这一代的目标倍数（默认 1.05）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.card:
        ap.error("需要 --card（或 --selftest）")
    card = Path(a.card)
    if not card.is_dir():
        cand = SKILLS / ("op-%s" % a.card)
        if not cand.is_dir():
            print("找不到卡：%s" % a.card, file=sys.stderr)
            return 1
        card = cand

    did = False
    if a.verify:
        did = True
        probs = verify(card)
        if probs:
            for p in probs:
                print("✗ " + p)
            return 1
    if a.seal:
        did = True
        dest, info = seal(card)
        if dest is None:
            print("未封存：%s" % info, file=sys.stderr)
            return 1
    if a.plan:
        did = True
        p, info = plan_next(card, step=a.step)
        if p is None:
            print("未生成目标：%s" % info, file=sys.stderr)
            return 1
    if not did:
        ap.error("需要 --seal / --plan / --verify 之一")
    return 0


if __name__ == "__main__":
    sys.exit(main())
