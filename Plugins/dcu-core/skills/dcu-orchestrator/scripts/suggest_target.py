#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""suggest_target.py — 把候选事实匹配到「拟并入的目标 skill 文件」建议路径。

用途：promotion-review-convention 的审批块里「拟并入：<目标 skill 文件路径>」常要 agent 手填，
本脚本按域把候选事实关键词匹配到**已存在的目标文件**，无匹配则按各域命名规范建议新建文件名，
让 agent 直接拿到"建议拟并入 <路径>"，不必从零想落点。

仅依赖 Python 3 标准库；确定性匹配（关键词子串命中计分，取最高；无命中→建议新建）。

用法：
  python3 suggest_target.py --domain op --fact "LDS bank conflict 用 padding 消" [--fact ...]
      [--repo <dcu-toolkit-skills 仓库根>] [--out <md>]
      --repo 存在时校验目标文件是否真实存在；不存在→标「建议新建」（按命名规范给建议名）。
      输出：建议拟并入的目标文件路径（每事实一条；匹配现有 → <路径>，未匹配 → 建议新建 <建议名>）。
"""
import argparse
import os
import sys


# 每个目标：rel_path（相对仓库根）+ keywords（小写子串命中）。
# 目标清单取自各域真实目录（2026-10 核对）。
TARGETS = {
    "op": [
        ("operators/dcu-kernelwiki/wiki/lds_bank_conflict.md", ["lds", "bank conflict", "bank_conflict", "padding"]),
        ("operators/dcu-kernelwiki/wiki/matrixcore_layout.md", ["matrix core", "matrixcore", "mmac", "wavefront", "k100"]),
        ("operators/dcu-kernelwiki/wiki/rmsnorm_precision_cliff.md", ["rmsnorm", "precision cliff", "舍入", "精度悬崖"]),
        ("operators/dcu-kernelwiki/wiki/tilelang_dcu_known_issues.md", ["tilelang"]),
        ("operators/dcu-kernelwiki/wiki/optimization_loop_protocol.md", ["profile", "优化循环", "metric", "mad 噪声"]),
        ("operators/dcu-kernelwiki/wiki/gemm-bank-conflict-case.md", ["gemm", "sfu", "sfa"]),
        ("operators/dcu-kernelwiki/wiki/megamoe_fused_moe_kernel.md", ["moe", "megamoe", "fused moe"]),
        ("operators/dcu-kernelwiki/wiki/flashattn_pipeline_overlap_case.md", ["flash attention", "pipeline", "overlap", "双缓冲"]),
        ("operators/dcu-kernelwiki/wiki/bandwidth_bound_l2_thrashing.md", ["bandwidth", "l2", "thrashing", "cache"]),
        ("operators/dcu-kernelwiki/wiki/blackwell_gemm_dcu_transfer.md", ["blackwell", "swizzle", "xor", "xorswizzle"]),
    ],
    "fault": [
        ("troubleshooting/dcu-ontrack-experience/references/xid_sxid_errors.md", ["xid", "sxid", "vmfault", "hsa"]),
        ("troubleshooting/dcu-ontrack-experience/references/diagnosis_rules.md", ["driver_load", "drv-", "诊断规则", "规则引擎", "pcie_link"]),
        ("troubleshooting/dcu-ontrack-experience/SKILL.md", ["hy-smi", "devcount", "rccl", "vllm", "sglang", "编译", "libxxx"]),
    ],
    "deploy": [
        # deploy 落点是 data/product-matrix/ 方案卡（按命名规范新建），没有通用匹配文件，走新建建议。
    ],
    "research": [
        # research 落点建议 references/<模型>_<来源>.md 或 dcu-kernelwiki/wiki/<topic>.md，走新建建议。
    ],
}

# 从事实文本里探测的「公开 token」（设备/模型/代际），用于生成干净的建议文件名（不含占位符）。
PUBLIC_TOKEN_PATTERNS = [
    r"BW\d+(?:000)?", r"gfx\d{3}", r"qwen[-\w.]*", r"glm[-\w.]*", r"llama[-\w.]*", r"deepseek[-\w.]*", r"k\d{2}[-]?ai",
]


def detect_public(fact):
    """从事实文本抓公开 token（卡型/模型/gfx 代际），去重保序。"""
    import re
    seen, out = set(), []
    for pat in PUBLIC_TOKEN_PATTERNS:
        for m in re.findall(pat, str(fact), re.I):
            tok = m.lower()
            if tok not in seen:
                seen.add(tok)
                out.append(m)
    return out


def slug(text):
    import re
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(text)).strip("._-")
    return s or "untitled"


# 各域「建议新建」命名规范（data/product-matrix / 一般 references）
def _is_public_token(tok):
    """meta 兜底值也必须命中公开 token 模式：裸设备号（<device-A>/<device-B>）、MISSING、内部名一律丢弃。"""
    return bool(tok) and any(re_match(p, tok) for p in PUBLIC_TOKEN_PATTERNS)


def new_name(domain, fact, meta=None):
    """生成「建议新建」文件名。只含探测到的公开 token，不写占位符；探测不到则省略该段。"""
    toks = detect_public(fact)
    dev = meta.get("device") if meta else None
    mdl = meta.get("model") if meta else None
    device = next((t for t in toks if re_match(r"bw|gfx|k\d", t)), None)
    if device is None and _is_public_token(dev):
        device = dev
    model = next((t for t in toks if re_match(r"qwen|glm|llama|deepseek", t)), None)
    if model is None and _is_public_token(mdl):
        model = mdl
    point = slug(fact)[:24] or "方案"
    if domain == "deploy":
        # {卡型}_{场景}_{模型}.md；卡型/模型都探测不到才用「要点」兜底作关键信息，避免占位符/冗余
        scene = "LLM_推理"
        parts = [p for p in (device, scene, model) if p]
        if len(parts) <= 1:
            parts = [scene, point]
        return "data/product-matrix/%s.md" % "_".join(parts)
    if domain == "research":
        base = model or device or "调研"
        return "references/%s_research_notes.md" % slug(base)
    # op / fault 兜底：建议新建 wiki 主题文件
    if domain == "op":
        return "operators/dcu-kernelwiki/wiki/%s.md" % slug(fact)[:40]
    return "troubleshooting/dcu-ontrack-experience/references/%s.md" % slug(fact)[:40]


def re_match(pat, s):
    import re
    return bool(re.search(pat, str(s), re.I))


def match(domain, fact, meta=None):
    """返回 (目标路径, is_new)。is_new=True 表示无匹配、建议新建。"""
    fact_low = str(fact).lower()
    best = None
    best_score = 0
    for path, kws in TARGETS.get(domain, []):
        score = sum(1 for k in kws if k.lower() in fact_low)
        if score > best_score:
            best_score = score
            best = path
    if best and best_score > 0:
        return best, False
    return new_name(domain, fact, meta), True


def main():
    ap = argparse.ArgumentParser(description="候选事实 → 拟并入目标文件路径建议")
    ap.add_argument("--domain", required=True, choices=["op", "fault", "deploy", "research"])
    ap.add_argument("--fact", action="append", default=[], help="候选事实文本（可多次）")
    ap.add_argument("--repo", default="", help="dcu-toolkit-skills 仓库根（校验目标文件是否存在）")
    ap.add_argument("--meta", default="{}", help='JSON 可选 meta，如 {"device":"BW1000","model":"qwen3-32b"}')
    ap.add_argument("--out", default="", help="输出到文件（缺省仅打印）")
    args = ap.parse_args()

    try:
        meta = json_loads(args.meta)
    except Exception:
        meta = {}

    facts = args.fact
    if not facts:
        print("[WARN] 未给 --fact，无法匹配；建议先跑 extract_candidate_facts.py 获取候选事实。")
        return 2

    lines = ["## 建议拟并入目标（suggest_target.py 自动匹配）\n"]
    for fact in facts:
        path, is_new = match(args.domain, fact, meta)
        exists = False
        if args.repo:
            exists = os.path.isfile(os.path.join(args.repo, path))
        if is_new:
            note = "建议新建（未命中现有目标）" + ("；已存在同名？" if exists else "")
            lines.append("- 建议新建：`%s`  %s" % (path, note))
        else:
            note = "（匹配现有文件）" if (not args.repo or exists) else "（仓库里不存在？请核对）"
            lines.append("- 拟并入：`%s`  %s" % (path, note))
    lines.append("\n> 请人工核对后填入 promotion-review-convention 审批块的「拟并入」。")

    text = "\n".join(lines)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        print("[OK] 落点建议已写入：%s" % args.out)
    else:
        print(text)
    return 0


def json_loads(s):
    import json
    return json.loads(s) if s.strip() else {}


if __name__ == "__main__":
    sys.exit(main())
