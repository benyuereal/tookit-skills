#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_candidate_facts.py — 从一次 run 产物(及可选 notes)抽取「可复用候选事实」草稿。

用途：把 dcu_distill.py 产出的「候选事实（待填）」占位，替换成**预填草稿**——按已知字段
从 run 产物/notes 里抽出可复用的事实，自动填 scope（语义化指纹）+ status（凭指纹完整性）
+ 依据（证据路径），并把「环境专属 / 纯数字 / 未验证」归入明确不并入。agent 复核/修改后
即得到标准候选事实行，可填入晋升提案并按 promotion-review-convention 呈现审批块。

仅依赖 Python 3 标准库；确定性变换，不推断、不补全、不美化（缺字段标 MISSING）。

用法：
  python3 extract_candidate_facts.py --run runs/op-optimize/validate/result.json \
      [--notes runs/.../notes.md] [-domain op|fault|deploy|research] [--out <md>]
      产出：候选事实 markdown 块（含 scope/status/依据 + 明确不并入清单），打印或写 --out。
"""
import argparse
import json
import os
import sys


# 语义化：机型 → 公开占位（对外包禁裸号，见 skill-package-hygiene；内部版另含机号映射）
DEVICE_SEMANTIC = {
    "BW150": "<gfx936-代>", "BW100": "<gfx936-代>", "BW1000": "<gfx936-代>", "BW1100": "<gfx938-代>",
}
GFX_SEMANTIC = {"gfx906": "gfx906-代", "gfx928": "gfx928-代", "gfx936": "gfx936-代", "gfx938": "gfx938-代"}


def sem_device(device):
    """设备号/机型 → 公开占位/语义化；未知则原样（调用方负责）。"""
    if not device or device == "MISSING":
        return device
    return DEVICE_SEMANTIC.get(str(device), "<device-model>")


def sem_gfx(gfx):
    if not gfx or gfx == "MISSING":
        return gfx
    return GFX_SEMANTIC.get(str(gfx), "<gfx代>")


def load_json(path):
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def pick(data, *keys, default="MISSING"):
    pool = []
    if isinstance(data, dict):
        pool.append(data)
        for k in ("fingerprint", "metrics", "result", "env", "performance", "evidence"):
            v = data.get(k)
            if isinstance(v, dict):
                pool.append(v)
    for key in keys:
        for src in pool:
            if key in src and src[key] not in (None, ""):
                return src[key]
    return default


def extract_notes(lines):
    """从 notes/free-text 里粗抽「可能是可复用事实」的行（以手法/坑/约束/注意 开头）。"""
    hits = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if any(k in low for k in ("手法", "坑", "注意", "约束", "必须", "不要", "避免", "建议", "要点", "关键")):
            hits.append(s)
    return hits


# 各域的候选事实字段映射：label -> 候选键名列表（.json 里可能出现的键，依次 pick）。
# 让 fault/deploy/research 域也能预填各自典型字段，而不是只剩「待人工补充」。
# 纯数值校验类字段（accuracy/correctness/max_diff 等）不进候选，归「明确不并入（仅台账备注）」。
DOMAIN_FIELDS = {
    "op": [
        ("手法", ["change", "diff_summary", "change_path"]),
        ("结论/坑", ["note", "notes"]),
    ],
    "fault": [
        ("根因", ["root_cause", "conclusion"]),
        ("修复建议", ["fix_action", "fix", "workaround"]),
        ("置信度", ["confidence"]),
        ("报错签名", ["key_error", "error", "fault_type"]),
    ],
    "deploy": [
        ("部署要点", ["change", "note", "config", "startup_args"]),
        ("一致性结论", ["consistency", "mismatch_ratio", "cases_total"]),
        ("精度锚点", ["accuracy", "max_diff"]),
    ],
    "research": [
        ("来源锚点", ["source_url", "source_version", "repo"]),
        ("目标模型", ["target_model", "model"]),
        ("适配现状", ["adapt_status", "dcu_status", "image", "wheel"]),
        ("复现要点", ["launch_cmd", "command", "config"]),
    ],
}
# 纯数值校验/精度值字段：可回查但与「可复用事实」无关，不应建议落点/新建文件，只留台账。
NUMERIC_CHECK_KEYS = {"accuracy", "correctness", "max_diff", "gain", "gain_percent", "speedup",
                      "mismatch_ratio", "cases_total"}
# 这些字段值可能内嵌敏感值（IP/路径/容器），合入前必须过脱敏闸复核并占位。
SENSITIVE_KEYS = {"serve_endpoint", "source_url", "image", "repo", "launch_cmd", "command"}


def main():
    ap = argparse.ArgumentParser(description="从 run 产物抽取可复用候选事实草稿")
    ap.add_argument("--run", required=True, help="run 产物 result.json（唯一事实来源）")
    ap.add_argument("--notes", default="", help="可选 notes.md（自由文本，辅助抽候选事实行）")
    ap.add_argument("--domain", default="op", choices=["op", "fault", "deploy", "research"])
    ap.add_argument("--out", default="", help="输出到文件（缺省仅打印）")
    args = ap.parse_args()

    if not os.path.isfile(args.run):
        print("[FAIL] 找不到 run 产物：%s" % args.run)
        return 2
    data = load_json(args.run)

    # --- scope：语义化指纹 ---
    device = sem_device(pick(data, "device", "device_model", "card"))
    gfx = sem_gfx(pick(data, "gfx", "arch"))
    dtk = pick(data, "dtk", "dtk_version")
    framework = pick(data, "framework", "engine")
    model = pick(data, "model")
    scope_parts = [p for p in (gfx, device, dtk, framework, model) if p and p != "MISSING"]
    scope = "/".join(scope_parts) if scope_parts else "<gfx代际/卡型>"

    # --- status：指纹完整性（缺 device/dtk/gfx/framework/gain 任一即 unverified）---
    fp_keys = ("device", "dtk", "gfx", "framework", "gain")
    fp_missing = [k for k in fp_keys if pick(data, k) == "MISSING"]
    status = "unverified" if fp_missing else "verified"

    # --- 依据：证据路径 ---
    source_run = pick(data, "source_run", "run", default=args.run)
    logs = pick(data, "logs", "profile", "trace", default="MISSING")
    evidence = source_run + ((", logs: " + logs) if logs != "MISSING" else "")

    # --- 候选事实（按域字段映射抽；每项都可回查；敏感字段标「需脱敏占位」）---
    # 纯数值校验类字段不进候选（否则会把 max_diff/gain 等误当成要新建文件的可复用事实），归「台账备注」。
    candidates = []
    numeric_notes = []
    seen = set()
    for label, keys in DOMAIN_FIELDS.get(args.domain, []):
        if any(k in NUMERIC_CHECK_KEYS for k in keys):
            val = pick(data, *keys, default="MISSING")
            if val != "MISSING":
                numeric_notes.append("%s=%s" % (label, val))
            continue
        val = pick(data, *keys, default="MISSING")
        if val == "MISSING" or val in seen:
            continue
        seen.add(val)
        suffix = "（含敏感值，合入前需脱敏占位）" if any(k in SENSITIVE_KEYS for k in keys) else ""
        candidates.append(("[ ] %s：%s%s" % (label, val, suffix), scope, status, evidence))
    # 公共跨域字段补充（兜底抽 note/notes 作为可复用事实；accuracy/correctness 等数值类归台账备注）
    note_val = pick(data, "note", "notes", default="MISSING")
    if note_val != "MISSING" and note_val not in seen:
        candidates.append(("[ ] 补充结论/坑：%s" % note_val, scope, status, evidence))
    for k in ("accuracy", "correctness", "max_diff"):
        v = pick(data, k, default="MISSING")
        if v != "MISSING":
            numeric_notes.append("正确性/精度锚点=%s" % v)
    # notes 自由文本抽取
    if args.notes and os.path.isfile(args.notes):
        for line in extract_notes(open(args.notes, encoding="utf-8-sig").read().splitlines()):
            candidates.append(("[ ] %s" % line, scope, status, evidence))
    if not candidates:
        candidates.append(("[ ] （待人工从任务上下文补充可复用事实）", scope, status, evidence))

    # --- 明确不并入 ---
    exclude = []
    env_specific = pick(data, "env_specific", default="MISSING")
    if env_specific != "MISSING":
        exclude.append("环境专属：%s" % env_specific)
    gain = pick(data, "gain", "gain_percent", "speedup", default="MISSING")
    if gain != "MISSING":
        exclude.append("纯性能数字（标[历史数据]，只留台账）：gain=%s" % gain)
    if fp_missing:
        exclude.append("指纹不完整（%s）→ 候选标 unverified，未验证结论不进 skill 正文" % ",".join(fp_missing))

    # --- 输出候选事实 markdown 块 ---
    out = []
    out.append("## 候选事实（自动抽取草稿，人工复核/修改后合入）\n")
    out.append("> 来源：`%s`；scope=%s；status=%s" % (args.run, scope, status))
    out.append("> 脱敏：设备号/机型已语义化（%s→占位）；合入前仍需按 skill-package-hygiene 复核 IP/用户名/内网仓库/工单号。\n" % device)
    for cand, sc, st, ev in candidates:
        out.append("- %s | scope=%s | status=%s | 依据=%s" % (cand, sc, st, ev))
    if exclude:
        out.append("\n### 明确不并入")
        for e in exclude:
            out.append("- %s" % e)
    if numeric_notes:
        out.append("\n### 台账备注（纯数值校验值，只留台账，不进 skill，不建议落点）")
        for n in numeric_notes:
            out.append("- %s" % n)

    # --- 拟并入落点建议（suggest_target 自动匹配，免手填）---
    try:
        import suggest_target as st_mod
    except Exception:
        st_mod = None
    if st_mod is not None and candidates:
        meta = {"device": pick(data, "device", "device_model", "card"),
                "model": pick(data, "model")}
        out.append("\n### 拟并入落点建议（建议人工核对后填入「拟并入」）")
        for cand, sc, ststatus, ev in candidates:
            fact = cand.split("：", 1)[-1] if "：" in cand else cand
            # 占位/待补充候选不是可复用事实，不参与落点匹配，也不建议新建文件
            if "待人工" in cand or cand.strip().startswith("[ ] （"):
                out.append("- （占位提示：候选为空，请先人工补充可复用事实，暂不建议落点/新建）")
                continue
            try:
                path, is_new = st_mod.match(args.domain, fact, meta)
                if is_new and ("MISSING" in path or "untitled" in path or "待人工" in path):
                    out.append("- （未识别到可复用事实，暂不建议新建文件——请补充具体事实后再定落点）")
                    continue
                tag = "建议新建" if is_new else "匹配现有"
                out.append("- 拟并入 `%s`（%s）" % (path, tag))
            except Exception:
                out.append("- （落点建议失败，请手填拟并入 skill 文件）")

    out.append("\n> 请选择：①合入 ②只留台账 ③修改后再合入 ④放弃本次沉淀")

    text = "\n".join(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("[OK] 候选事实草稿已写入：%s" % args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
