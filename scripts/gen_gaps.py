#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_gaps —— 生成 op-atlas/gaps.md：已知未覆盖 + 证据缺口。

为什么需要它
------------
46 张卡容易被当成「全覆盖」。实际：
  - 五个家族零实测（linear-attn / kv / sampling / quant / memory）
  - 24 张 measured 里 14 张挂不上可复跑的证据文件
「用覆盖数掩盖深度不足」是本项目一路最警惕的那类自我美化。

gaps.md 是**生成物**，不手写。两份内容：

1. **证据缺口**（业主 2026-11-23 裁决要求）：
   `missing_evidence` 不许只打一行 stderr 就过去，必须机读列出卡片名。
   连续两批仍 missing 的卡要降级 draft —— 由 E25 读本文件的历史判断。

2. **未覆盖算子 + 零实测家族**：为什么没覆盖、补的前提。

用法
----
    python3 -I scripts/gen_gaps.py            # 写 gaps.md
    python3 -I scripts/gen_gaps.py --check    # 只校验是否同步（CI 用）
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "Plugins/ops/op-atlas"
CARDS = ATLAS / "cards"
GAPS = ATLAS / "gaps.md"
STATE = ATLAS / ".gaps-state.json"     # 记录「连续几批 missing」，供 E25 降级用

# 已知未覆盖：人工维护，每条必须写「为什么」与「补的前提」。
# 不手写生成的理由：这些判断不是从卡里推出来的，是取舍。
UNCOVERED = [
    ("all_reduce / all_gather", "属 RCCL，本仓库改不动其 kernel",
     "只能做重叠编排（归 engine 层，不是算子卡）"),
    ("dequant_fp8_w4", "无公开入口，本机也没有可调用的 w4a16 反量化算子",
     "需先有 w4a16 量化卡与可跑通的权重格式"),
    ("sampling_top_p", "映射有、实测无；top-p 的判据（累积概率截断点）与 top-k 不同",
     "需先定 sampling 族的判据，再建 case"),
    ("moe_grouped_fp8_ll", "deepgemm 的 ll 路径要求 6D packed layout，本机造不出",
     "需先解决 pack 格式（case 002 记为未测定）"),
    ("flash_mla sparse prefill 形态", "本机 flash_mla 只跑通 decode 剂量",
     "需确认真实 trace 里 prefill 是否真走这条路径"),
]

# 零实测家族的原因（人工写，因为它需要判断而不是统计）
ZERO_MEASURED_REASON = {
    "linear-attn": "9 条映射已有，但 0 张实测；本机未找到可跑通的 chunked-scan / gated-linear-attn 入口",
    "kv": "3 张卡全 draft；paged 写入的测量被 launch 开销淹没（见 case 009 的同类问题）",
    "sampling": "3 张卡全 draft；判据未定（top-k 与 top-p 的截断语义不同）",
    "quant": "4 条映射 0 实测；per-token/per-block 判据未落地",
    "memory": "embedding-lookup 1 张 draft；属访存类，判据与 rope 同源但未跑",
}


def _load_cards():
    import yaml
    out = []
    for p in sorted(CARDS.glob("*.yaml")):
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        out.append(d)
    return out


def evidence_status(card):
    """返回 (状态, 证据列表)。状态 ∈ ok / missing / hash_mismatch。"""
    import hashlib
    lv = card.get("last_verdict") or {}
    if lv.get("status") not in ("measured", "tuned"):
        return None, []
    ev = lv.get("evidence") or []
    if not ev:
        return "missing", []
    bad = []
    for e in ev:
        path = REPO / str(e.get("path") or "")
        want = str(e.get("sha256") or "")
        if not path.is_file():
            bad.append("%s 不存在" % e.get("path"))
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if want and got != want:
            bad.append("%s 哈希不符（记录 %s… 实际 %s…）"
                       % (Path(e["path"]).name, want[:8], got[:8]))
    return ("hash_mismatch" if bad else "ok"), bad


def compute_state(cards):
    """当前一批的 missing 名单 + 与上一批比的连续计数。

    **计数按「批」而非「运行次数」递增**：同一批内重复生成不累加。
    判据是 missing 名单是否变化——名单没变就是同一批（幂等），变了才算新一批。
    第一版每次生成都 +1，于是同一批跑三次就变成「连续 3 批」，
    会误触发降级。
    """
    prev = {}
    if STATE.is_file():
        try:
            prev = json.loads(STATE.read_text(encoding="utf-8"))
        except ValueError:
            prev = {}
    prev_streak = prev.get("streak") or {}
    prev_missing = set(prev.get("missing") or [])
    streak, missing, mismatched = {}, [], []
    for c in cards:
        st, bad = evidence_status(c)
        if st == "missing":
            missing.append(c.get("id"))
        elif st == "hash_mismatch":
            mismatched.append((c.get("id"), bad))
    this_missing = set(missing)
    same_batch = (this_missing == prev_missing)
    for cid in missing:
        prev_n = int(prev_streak.get(cid, 0))
        # 名单没变 = 同一批（幂等，不递增）；名单变了 = 新一批（递增）
        streak[cid] = prev_n if same_batch else prev_n + 1
    return missing, mismatched, streak


def render(cards, missing, mismatched, streak):
    fams = {}
    for c in cards:
        fams.setdefault(c.get("family") or "?", []).append(c)
    measured = [c for c in cards if (c.get("last_verdict") or {}).get("status")
                in ("measured", "tuned")]

    L = []
    L.append("# GAPS —— 已知未覆盖与证据缺口（生成物，勿手改）")
    L.append("")
    L.append("> 由 `scripts/gen_gaps.py` 生成。改这张表请改脚本或卡片，不要直接编辑本文件。")
    L.append("> 生成时间：%s" % __import__("time").strftime("%Y-%m-%d %H:%M"))
    L.append(">")
    L.append("> **它防的是「用覆盖数掩盖深度不足」。** 46 张卡容易被当成全覆盖，")
    L.append("> 实际有 %d 张挂不上可复跑证据、%d 个家族零实测。"
             % (len(missing), sum(1 for f in fams
                                  if not any((c.get("last_verdict") or {}).get("status")
                                             in ("measured", "tuned") for c in fams[f]))))
    L.append("")

    # ---- 一、证据缺口（业主裁决要求：机读列出卡片名）
    L.append("## 一、证据缺口（`missing_evidence`）")
    L.append("")
    L.append("`measured` 的本义是「有数**且可追**」。下列 %d 张卡有结论但挂不上可复跑的"
             " results 文件（E25 判 missing）。" % len(missing))
    L.append("")
    L.append("| 卡片 | 连续缺证据批数 | 处置 |")
    L.append("|---|---|---|")
    for cid in sorted(missing):
        n = streak.get(cid, 1)
        action = "**降级 draft**（连续 ≥2 批）" if n >= 2 else "本批 warning，下批仍缺则降级"
        L.append("| `%s` | %d | %s |" % (cid, n, action))
    L.append("")
    if mismatched:
        L.append("### 哈希不符（E25 判 error，必须修）")
        L.append("")
        for cid, bad in mismatched:
            L.append("- `%s`：%s" % (cid, "；".join(bad)))
        L.append("")

    # ---- 二、零实测家族
    L.append("## 二、零实测家族")
    L.append("")
    L.append("| 家族 | 卡数 | 为什么零实测 |")
    L.append("|---|---|---|")
    for fam in sorted(fams):
        cs = fams[fam]
        if any((c.get("last_verdict") or {}).get("status") in ("measured", "tuned") for c in cs):
            continue
        L.append("| `%s` | %d | %s |"
                 % (fam, len(cs), ZERO_MEASURED_REASON.get(fam, "原因未记录——需要补")))
    L.append("")
    covered = [f for f in sorted(fams)
               if any((c.get("last_verdict") or {}).get("status") in ("measured", "tuned")
                      for c in fams[f])]
    L.append("有实测的家族 %d / %d：%s" % (len(covered), len(fams),
                                          "、".join("`%s`" % f for f in covered)))
    L.append("")

    # ---- 二b、分母未标定（业主 2026-11-23 要求）
    L.append("## 二b、分母未标定")
    L.append("")
    L.append("下列常数被当作达成率的分母，但**全仓没有可复跑的测量记录**。")
    L.append("在补上标定前，凡用它们算出的 utilization 必须与实测峰值并列双报。")
    L.append("")
    L.append("| 常数 | 值 | 现状 | 实测参考 | 补的前提 |")
    L.append("|---|---|---|---|---|")
    L.append("| `memory.stream_read_peak` | 1724 GB/s | **只有断言，无测量记录** | "
             "`scripts/bw_calibrate.py` 实测：copy 1051 / mul 1287 / add 1634 / sum 865 GB/s"
             "（2026-11-23，sclk 1350），**无一到 1724** | 跑 `bw_calibrate.py` 并据实改值，或标 unverified |")
    L.append("")
    L.append("> 这是「分母错了 10 倍」那类问题的第三次：1100 标称 → 185 反推 → 540 无来源。")
    L.append("> 前三次都是「分母的来源不是本仓库能跑的测量」。现在补了 `bw_calibrate.py`，")
    L.append("> 但 1724 这个数本身还没被替换——它需要先跑一次并确认时钟档位。")
    L.append("")

    # ---- 三、已知未覆盖算子
    L.append("## 三、已知未覆盖算子")
    L.append("")
    L.append("| 未覆盖 | 为什么没覆盖 | 补的前提 |")
    L.append("|---|---|---|")
    for name, why, pre in UNCOVERED:
        L.append("| `%s` | %s | %s |" % (name, why, pre))
    L.append("")
    L.append("## 四、当前实测覆盖")
    L.append("")
    L.append("有数卡 %d / 全部 %d 张。其中证据齐全 %d 张、缺证据 %d 张。"
             % (len(measured), len(cards), len(measured) - len(missing), len(missing)))
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成 gaps.md（含证据缺口）")
    ap.add_argument("--check", action="store_true", help="只校验是否同步（CI 用）")
    a = ap.parse_args(argv)
    cards = _load_cards()
    missing, mismatched, streak = compute_state(cards)
    want = render(cards, missing, mismatched, streak)

    if a.check:
        cur = GAPS.read_text(encoding="utf-8") if GAPS.is_file() else ""
        # 只比正文（时间戳会变）
        strip = lambda t: "\n".join(l for l in t.splitlines()
                                    if not l.startswith("> 生成时间"))
        if strip(cur) != strip(want):
            print("gaps.md 与卡片不同步。跑 python3 -I scripts/gen_gaps.py 重新生成",
                  file=sys.stderr)
            return 1
        print("gaps.md 与卡片一致（缺证据 %d 张）" % len(missing))
        return 0

    GAPS.write_text(want, encoding="utf-8")
    STATE.write_text(json.dumps({"streak": streak, "missing": missing},
                                indent=2, ensure_ascii=False), encoding="utf-8")
    print("已生成 %s（缺证据 %d 张，哈希不符 %d 张）"
          % (GAPS.relative_to(REPO), len(missing), len(mismatched)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
