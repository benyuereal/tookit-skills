#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_curriculum —— 校验 surfaces.yaml 与 curriculum.yaml 的一致性。

## 它守什么

`surfaces.yaml` 是平铺的 17 条攻击面，`curriculum.yaml` 把它们分成 M0..M3
四层（先定架构、后调参数）。两者的关系是**一一对应**：
每条 surface 有且只有一个层。

这个一致性**不会自动保持**：加一条新 surface 忘了分层、或者把一条
surface 挪层时改了一处漏了另一处，都不会让任何测试变红 ——
只会在 agent 真正按课程跑的时候发现「这条没层，不知道什么时候做」。

## 四条判据（每条都有反向测试）

1. surfaces.yaml 的每条 surface 在 curriculum 里**有且只有一个**层
2. curriculum 里的每条 surface 在 surfaces.yaml 里**真实存在**
3. M0 层必须声明 `gate`（验收判据）—— 没有 gate 的架构层等于没有冻结
4. 不许有空层

## 用法

    python3 -I scripts/check_curriculum.py            # 检查
    python3 -I scripts/check_curriculum.py --selftest # 自检
    python3 -I scripts/check_curriculum.py --json     # 机器可读
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SURFACES = REPO / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"
CURRICULUM = REPO / "Plugins/ops/skills/op-optimize/references/curriculum.yaml"

LAYER_ORDER = ["M0", "M1", "M2", "M3"]


def _load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def surface_ids(surfaces):
    """surfaces.yaml 是一个 list，每条有 id。"""
    out = []
    for s in surfaces or []:
        if isinstance(s, dict) and s.get("id"):
            out.append(s["id"])
    return out


def curriculum_map(cur):
    """返回 {surface_id: [层名...]}，顺便检查层名合法。"""
    m, bad_layers = {}, []
    layers = (cur or {}).get("layers") or {}
    for layer, spec in layers.items():
        if layer not in LAYER_ORDER:
            bad_layers.append(layer)
        for sid in (spec or {}).get("surfaces") or []:
            m.setdefault(sid, []).append(layer)
    return m, bad_layers


def check(surfaces, cur):
    """返回问题列表（空 = 一致）。**不抛异常** —— 编排器不该因为数据问题崩。"""
    probs = []
    sids = surface_ids(surfaces)
    cmap, bad_layers = curriculum_map(cur)

    if not sids:
        probs.append("surfaces.yaml 里一条 surface 都没有读出来（格式变了？）")
    if bad_layers:
        probs.append("curriculum 里有未知层 %s（合法：%s）"
                     % (sorted(bad_layers), LAYER_ORDER))

    # ① 每条 surface 有且只有一个层
    for sid in sids:
        got = cmap.get(sid, [])
        if not got:
            probs.append("surface %r 没有分层 —— agent 不知道什么时候该做" % sid)
        elif len(got) > 1:
            probs.append("surface %r 出现在多个层 %s —— 有且只有一个" % (sid, got))

    # ② curriculum 里的每条必须真实存在
    for sid in cmap:
        if sid not in sids:
            probs.append("curriculum 里的 %r 在 surfaces.yaml 里不存在" % sid)

    # ③ M0 必须有 gate
    layers = (cur or {}).get("layers") or {}
    m0 = layers.get("M0") or {}
    if not (m0.get("gate") or "").strip():
        probs.append("M0 没写 gate —— 没有验收判据的架构层等于没有冻结")

    # ④ 不许有空层
    for layer in LAYER_ORDER:
        spec = layers.get(layer)
        if spec is None:
            probs.append("缺层 %s" % layer)
        elif not (spec.get("surfaces") or []):
            probs.append("层 %s 是空的" % layer)

    return probs


def summary(surfaces, cur):
    cmap, _ = curriculum_map(cur)
    layers = (cur or {}).get("layers") or {}
    out = {"total_surfaces": len(surface_ids(surfaces)),
           "layers": {k: len((v or {}).get("surfaces") or [])
                      for k, v in layers.items()}}
    return out


def selftest():
    """**反向优先**：先证明它拦得住不一致。"""
    surfaces = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    # ① 一致时无问题
    cur = {"layers": {
        "M0": {"gate": "通过正确性 + 达到预测地板", "surfaces": ["a"]},
        "M1": {"surfaces": ["b"]},
        "M2": {"surfaces": ["c"]},
        "M3": {"surfaces": []},   # 空层，下面单独测
    }}
    probs = check(surfaces, cur)
    assert any("M3 是空的" in p for p in probs), probs
    cur["layers"]["M3"]["surfaces"] = []   # 故意留空
    print("[ok] 反向：空层被报出来")

    # ② 修复后干净（M0..M3 每层都要非空，而只有 3 条 surface —— 改成 4 条）
    surfaces4 = surfaces + [{"id": "d"}]
    cur["layers"]["M3"] = {"surfaces": ["d"]}
    probs = check(surfaces4, cur)
    assert probs == [], probs
    print("[ok] 正向：一致时无问题")

    # ③ 反向：某条 surface 没分层
    cur2 = {"layers": {
        "M0": {"gate": "g", "surfaces": ["a"]},
        "M1": {"surfaces": []},
        "M2": {"surfaces": ["b"]},
        "M3": {"surfaces": []},
    }}
    probs = check(surfaces, cur2)
    assert any("'c' 没有分层" in p for p in probs), probs
    print("[ok] 反向：surface 没分层被报出来")

    # ④ 反向：一条 surface 出现在多个层
    cur3 = {"layers": {
        "M0": {"gate": "g", "surfaces": ["a", "b"]},
        "M1": {"surfaces": ["b"]},
        "M2": {"surfaces": ["c"]},
        "M3": {"surfaces": []},
    }}
    probs = check(surfaces, cur3)
    assert any("'b' 出现在多个层" in p for p in probs), probs
    print("[ok] 反向：一条 surface 分到多个层被报出来")

    # ⑤ 反向：curriculum 里有不存在的 surface
    cur4 = {"layers": {
        "M0": {"gate": "g", "surfaces": ["a", "zombie"]},
        "M1": {"surfaces": ["b"]},
        "M2": {"surfaces": ["c"]},
        "M3": {"surfaces": []},
    }}
    probs = check(surfaces, cur4)
    assert any("'zombie' 在 surfaces.yaml 里不存在" in p for p in probs), probs
    print("[ok] 反向：curriculum 里编造的 surface 被报出来")

    # ⑥ 反向：M0 没 gate
    cur5 = {"layers": {
        "M0": {"surfaces": ["a"]},
        "M1": {"surfaces": ["b"]},
        "M2": {"surfaces": ["c"]},
        "M3": {"surfaces": []},
    }}
    probs = check(surfaces, cur5)
    assert any("M0 没写 gate" in p for p in probs), probs
    print("[ok] 反向：M0 缺 gate 被报出来（没有验收判据的架构层等于没冻结）")

    # ⑦ 正向：summary 数得对
    s = summary(surfaces, cur)
    assert s["total_surfaces"] == 3, s
    print("[ok] 正向：summary 统计正确")

    print("\nselftest PASS（7 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    for p in (SURFACES, CURRICULUM):
        if not p.is_file():
            print("找不到 %s" % p, file=sys.stderr)
            return 1
    surfaces = _load_yaml(SURFACES)
    cur = _load_yaml(CURRICULUM)
    probs = check(surfaces, cur)
    if a.json:
        print(json.dumps({"problems": probs, "summary": summary(surfaces, cur)},
                         ensure_ascii=False, indent=1))
        return 1 if probs else 0
    if probs:
        for p in probs:
            print("✗ " + p)
        return 1
    s = summary(surfaces, cur)
    print("✓ %d 条 surface 分在 %s" % (
        s["total_surfaces"],
        " / ".join("%s=%d" % (k, v) for k, v in s["layers"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
