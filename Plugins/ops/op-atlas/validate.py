#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-atlas 防漂移校验。退出码非 0 = CI 失败。

用法:
    python3 -I validate.py --atlas Plugins/ops/op-atlas
    python3 -I validate.py --atlas Plugins/ops/op-atlas --json

设计前提：**图谱是索引层**。所以校验的重点不是「卡片写得对不对」，
而是「索引与真相源是否还一致」——卡片最容易的腐化方式是悄悄变成副本，
或指向一个已经不存在的 §号 / L 号。

八类检查（E1~E8）：
  E1  cards ↔ INDEX 双向一致（每张卡在 INDEX 出现；INDEX 每个链接真实存在）
  E2  impls.library_profile_ref 的 §号在 libraries.md 里真实存在
  E3  impls.law_ref 的 L 号在 pipeline-laws.md 里真实存在
  E4  status 与 data/ops/registry.json 对齐（registry 无此算子时 op_card 必须为 null）
  E5  ceiling.ref 可解析（文件存在 / constants 键存在 / missing 时 ref 为 null）
  E6  字段白名单 —— 卡片结构上**没有字段**可以承载技术描述
  E7  结论红线 —— last_verdict 不许写裸绝对耗时、不许无分母百分比
  E8  家族文件齐备（每个 family 都有 families/<family>.md）
"""
import argparse
import json
import re
import sys
from pathlib import Path

FAMILIES = {"gemm", "attention", "moe", "elementwise", "comm", "sampling",
            "norm", "rope", "activation", "quant", "kv", "linear-attn", "memory"}
STATUSES = {"draft", "measured", "tuned", "at_floor", "deprecated"}
TIERS = {"primary", "structural", "reference", "fallback", "avoid", "unavailable"}
CEIL_KINDS = {"roofline_model", "constants", "measured", "analytic", "missing"}
VERDICT_STATUSES = {"none", "predicted", "measured", "refuted", "at_floor"}

# E24：ratio.value 的量纲。四种语义完全不同，混在一列会让聚合与排序全错。
#   speedup      相对基线的加速倍数（>1 = 更快）——只有这类能算几何平均
#   utilization  占峰值的比例（0~1，偶尔 >1 是分母口径错）
#   throughput   绝对吞吐（GB/s / TFLOPS），不是倍数
#   ratio_other  非收益比值（如「M=128 与 M=1024 的耗时比」），不许当加速比
RATIO_KINDS = {"speedup", "utilization", "throughput", "ratio_other"}
RATIO_OTHER_LIMIT = 5     # E24b：ratio_other 超过这个数就提醒复核分类
LAYERS = {"kernel", "engine"}

# E6：顶层字段白名单。只有这些，多的一个字也进不来。
ALLOWED_TOP = {"id", "title", "family", "math", "status", "op_card", "impls", "at_floor",
               "ceiling", "shapes", "last_verdict", "surfaces", "updated", "layer",
               # L3：这张卡最可行的一条改法（不是「可以调参」这种废话）
               "modify_path",
               # 风险分级：不满足会崩 vs 不满足只是不赚
               "hard_preconditions", "soft_preconditions"}
REQUIRED_TOP = {"id", "title", "family", "math", "status", "op_card", "impls",
                "ceiling", "last_verdict", "updated"}
ALLOWED_IMPL = {"repo", "library_profile_ref", "file_hint", "tier", "law_ref",
                "best_for", "not_for", "pros", "cons", "source",
                # L1 映射三字段：从「知道在哪个库」到「知道调哪个函数」
                "import_path", "signature", "smoke_test",
                # 源码树/自研的锚点：库名不够，要指到文件与符号
                "source_ref", "l1_exempt_reason"}
REQUIRED_IMPL = {"repo", "library_profile_ref", "file_hint", "tier"}
ALLOWED_CEIL = {"kind", "ref", "bound", "model_name", "note"}
REQUIRED_CEIL = {"kind", "ref"}
ALLOWED_VERDICT = {"status", "kb_path", "ratio", "geomean", "evidence",
                   "next_probe", "conclusion", "verified_on",
                   # E27：访存口径。跨实现比较前必须先对齐 (inputs, outputs)
                   "traffic_passes", "inputs", "outputs", "comparison"}
REQUIRED_VERDICT = {"status", "kb_path", "conclusion", "verified_on"}

# E7：裸绝对耗时（0.887 ms / 1.29s / 83 us）——必须带分母或干脆不写
BARE_TIME = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:ms|µs|us|ns)\b", re.I)
# E7：无分母百分比（97% / 87 %）——必须写清"相对谁"
BARE_PCT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*%")

# E18：卡片正文里引用的机器峰值，必须与 constants.yaml 的实测值一致。
# 由来：bf16 峰值曾在 5 处各写各的（185/217/310），带宽峰值写 1728（常数是 1724）。
# 一个常数散落多处、只有一处对，就是「分母用错、结论反转」的温床。
PEAK_CLAIMS = {
    "bf16": re.compile(r"bf16[^。；\n]{0,20}?峰值[^0-9\n]{0,10}(\d{2,4})", re.I),
    "bw": re.compile(r"流式峰值\s*(\d{3,4})"),
}

errors, warns = [], []
L1_PENDING = [0]   # L1 回填进度计数（全部填完后改为 err）
L1_EXEMPT = [0]    # 不适用 L1 的条目数（in-house / 源码树）
_NON_INSTALLABLE = set()


def err(code, where, msg):
    errors.append("[%s] %s: %s" % (code, where, msg))


def warn(code, where, msg):
    warns.append("[%s] %s: %s" % (code, where, msg))


def _load_yaml(path):
    try:
        import yaml
    except ImportError:
        return None, "PyYAML 未安装，无法解析 %s（本校验器需要它；CI 已装）" % path
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except Exception as e:                                    # noqa: BLE001
        return None, "%s 解析失败: %s" % (path, e)


def _truth_sources(repo):
    lib_p = repo / "Plugins/ops/skills/op-stack-design/references/libraries.md"
    laws_p = repo / "Plugins/ops/skills/op-roofline/references/pipeline.md"
    lib = lib_p.read_text(encoding="utf-8") if lib_p.is_file() else ""
    laws = laws_p.read_text(encoding="utf-8") if laws_p.is_file() else ""
    lib_sections = {int(m) for m in re.findall(r"(?m)^###\s+(\d+)\.\s", lib)}
    law_ids = {m for m in re.findall(r"(?m)^#{1,4}\s+([A-Za-z]\d+)\b", laws)}
    return lib_sections, law_ids


def _check_index_matches_cards(cards_dir, index_txt):
    """E19：INDEX 的「算子一览」表逐行核对卡片。

    只核状态（最易漂）与家族、上限；结论文本是截断的，不做全文比对。
    """
    row_re = re.compile(
        r"^\|\s*\[`([a-z0-9-]+)`\]\(cards/[^)]+\)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|",
        re.M)
    seen = set()
    for m in row_re.finditer(index_txt):
        cid = m.group(1)
        idx_status = m.group(3).strip()
        idx_family = m.group(2).strip()
        idx_bound = m.group(5).strip()
        seen.add(cid)
        p = cards_dir / ("%s.yaml" % cid)
        if not p.is_file():
            continue                                   # E1b 已报
        c, problem = _load_yaml(p)
        if problem or not isinstance(c, dict):
            continue
        card_status = str(c.get("status") or "").strip()
        card_family = str(c.get("family") or "").strip()
        card_bound = str((c.get("ceiling") or {}).get("bound") or "").strip()
        if idx_status != card_status:
            err("E19", "INDEX.md",
                "%s 状态不一致：INDEX=%r 卡片=%r（INDEX 是生成物，跑 sync_index.py）"
                % (cid, idx_status, card_status))
        if idx_family and card_family and idx_family != card_family:
            err("E19", "INDEX.md", "%s 家族不一致：INDEX=%r 卡片=%r"
                % (cid, idx_family, card_family))
        if idx_bound and card_bound and idx_bound != card_bound:
            err("E19", "INDEX.md", "%s 上限口径不一致：INDEX=%r 卡片=%r"
                % (cid, idx_bound, card_bound))


# stop_if 必须含机制：不涨时凭什么判死，总不能只写「不涨就停」
STOP_MECHANISM_WORDS = ("⇒", "因为", "由于", "机制", "说明", "证明", "→")


def _check_next_probe(cards_dir, repo):
    """E26 实现。"""
    surf_ids = set()
    sp = repo / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"
    if sp.is_file():
        try:
            import yaml
            data = yaml.safe_load(sp.read_text(encoding="utf-8")) or []
            surf_ids = {str(x.get("id")) for x in data if isinstance(x, dict)}
        except Exception:                                        # noqa: BLE001
            surf_ids = set()

    for p in sorted(cards_dir.glob("*.yaml")):
        d, problem = _load_yaml(p)
        if problem or not isinstance(d, dict):
            continue
        lv = d.get("last_verdict") or {}
        if lv.get("status") not in ("measured", "tuned"):
            continue
        np_ = lv.get("next_probe")
        if not isinstance(np_, dict):
            err("E26", p.name,
                "measured/tuned 卡缺 next_probe —— 没有「下一步改什么」的卡只是基线快照")
            continue
        extra = set(np_) - {"surface", "hypothesis", "expected", "stop_if",
                            "effort", "none_reason", "tuned_from", "falsifier"}
        if extra:
            err("E26", p.name, "next_probe 未知字段 %s" % sorted(extra))

        surf = str(np_.get("surface") or "")
        if not surf:
            err("E26", p.name, "next_probe.surface 缺失（必须是 surfaces.yaml 的 id 或 none）")
        elif surf == "none":
            if not str(np_.get("none_reason") or "").strip():
                err("E26", p.name,
                    "surface=none 必须写 none_reason（如「已到地板」或「改法属 source-patch」）")
        elif surf_ids and surf not in surf_ids:
            err("E26", p.name,
                "next_probe.surface=%r 不在 surfaces.yaml 里（可选：%s）"
                % (surf, ", ".join(sorted(surf_ids))))

        hyp = str(np_.get("hypothesis") or "")
        if not hyp:
            err("E26", p.name, "next_probe.hypothesis 缺失")
        elif not re.search(r"\d", hyp):
            err("E26", p.name,
                "next_probe.hypothesis 里没有数字 —— 预期不可证伪，"
                "写成「再看看」也能过机检，那就失去意义了")

        stop = str(np_.get("stop_if") or "")
        if not stop:
            err("E26", p.name, "next_probe.stop_if 缺失（不涨时凭什么判死）")
        elif not any(w in stop for w in STOP_MECHANISM_WORDS):
            err("E26", p.name,
                "next_probe.stop_if 里没有机制说明 —— 只写「不涨就停」不算判据，"
                "要写清「不涨说明什么」（可用 ⇒/因为/由于/机制/说明/证明/→）")

        if lv.get("status") == "tuned" and not np_.get("tuned_from"):
            err("E26", p.name, "status=tuned 但 next_probe 里没有 tuned_from（基线值 + 改法）")


AT_FLOOR_KINDS = {"hardware_floor", "impl_ceiling"}


def _check_depth_rules(cards_dir, repo):
    """E28：深度优化要求的机检。

    1. top-level at_floor 必须带 kind ∈ {hardware_floor, impl_ceiling} 与 falsifier
    2. next_probe.surface == "none" 必须带 none_reason 与 falsifier
    3. at_floor 卡不许再写 next_probe 里有实际攻击面（矛盾）
    """
    for p in sorted(cards_dir.glob("*.yaml")):
        d, problem = _load_yaml(p)
        if problem or not isinstance(d, dict):
            continue
        lv = d.get("last_verdict") or {}
        af = d.get("at_floor")

        # 1) at_floor 块
        if d.get("status") == "at_floor" or af:
            if not isinstance(af, dict):
                err("E28", p.name,
                    "status=at_floor 但没有 at_floor 块 —— "
                    "必须写 kind（hardware_floor/impl_ceiling）与 falsifier")
            else:
                kind = str(af.get("kind") or "")
                if kind not in AT_FLOOR_KINDS:
                    err("E28", p.name,
                        "at_floor.kind=%r 非法（只允许 %s）"
                        % (kind, sorted(AT_FLOOR_KINDS)))
                if not str(af.get("falsifier") or "").strip():
                    err("E28", p.name,
                        "at_floor 缺 falsifier —— 必须写「换哪种实现/哪种方法能推翻这个结论」。"
                        "写不出来就说明这是工具限制而不是物理极限，不该标 at_floor")

        # 2) surface=none
        np_ = lv.get("next_probe") or {}
        if isinstance(np_, dict) and np_.get("surface") == "none":
            if not str(np_.get("falsifier") or "").strip():
                # 允许用 none_reason 里的「推翻」/「除非」表达
                nr = str(np_.get("none_reason") or "")
                if not any(k in nr for k in ("推翻", "除非", "若", "如果")):
                    err("E28", p.name,
                        "next_probe.surface=none 缺 falsifier（或用 none_reason 写清"
                        "「什么情况下会重新推进」）—— 否则等于无声放弃")


def _load_baseline_policy(repo):
    p = repo / "Plugins/ops/op-atlas/baseline-policy.yaml"
    if not p.is_file():
        return None
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:                                            # noqa: BLE001
        return None


def _check_baseline_policy(cards_dir, repo):
    """E27 硬校验：comparison 的 baseline 必须是允许作性能基准的类型。

    由来：2026-11-23 用「triton 融合 vs torch 三段式」算出 3.65x，
    这个数在工程上等于零——torch 三段式是未优化参考实现。
    24 张卡里 19 张「有数但没优化」，根子就在这里。
    """
    pol = _load_baseline_policy(repo)
    if not pol:
        return
    roles = pol.get("roles") or {}
    forbidden = pol.get("forbidden_impl_patterns") or []

    for c in sorted(cards_dir.glob("*.yaml")):
        d, problem = _load_yaml(c)
        if problem or not isinstance(d, dict):
            continue
        lv = d.get("last_verdict") or {}
        comp = lv.get("comparison")
        if not comp or not isinstance(comp, dict):
            continue
        base = str(comp.get("baseline") or "")
        role = str(comp.get("baseline_role") or "")
        if not base:
            err("E27", c.name, "comparison 缺 baseline（必须是某个变体名）")
            continue
        if not role:
            err("E27", c.name,
                "comparison.baseline=%r 未声明 baseline_role —— "
                "必须标明它凭什么能当性能基准" % base)
            continue
        info = roles.get(role) or {}
        allowed = info.get("allowed_as_performance_baseline")
        if allowed is False:
            err("E27", c.name,
                "comparison.baseline_role=%r（%s）不能当性能基准：%s"
                % (role, base, info.get("note", "")))
        import fnmatch
        for pat in forbidden:
            if fnmatch.fnmatch(base, pat):
                err("E27", c.name,
                    "comparison.baseline=%r 命中禁止模式 %r —— "
                    "未优化的参考实现打赢了没有工程意义" % (base, pat))
                break


def _check_evidence(cards_dir, repo):
    """E25 实现。见上方注释里的分档规则。"""
    import hashlib
    streak = {}
    state_p = repo / "Plugins/ops/op-atlas/.gaps-state.json"
    if state_p.is_file():
        try:
            streak = json.loads(state_p.read_text(encoding="utf-8")).get("streak", {})
        except ValueError:
            streak = {}
    missing = []
    for p in sorted(cards_dir.glob("*.yaml")):
        d, problem = _load_yaml(p)
        if problem or not isinstance(d, dict):
            continue
        lv = d.get("last_verdict") or {}
        if lv.get("status") not in ("measured", "tuned"):
            continue
        cid = d.get("id") or p.stem
        ev = lv.get("evidence")
        if not ev:
            n = int(streak.get(cid, 0))
            missing.append(cid)
            warns.append("[E25] %s: 有结论但没挂 evidence（missing_evidence，"
                         "不进聚合）%s" % (p.stem,
                                          "；连续 %d 批 → 应降级 draft" % n if n >= 2 else ""))
            continue
        if not isinstance(ev, list):
            err("E25", p.name, "evidence 必须是列表")
            continue
        for i, e in enumerate(ev):
            if not isinstance(e, dict):
                err("E25", p.name, "evidence[%d] 必须是 mapping" % i)
                continue
            extra = set(e) - {"path", "sha256", "verified_on"}
            if extra:
                err("E25", p.name, "evidence[%d] 未知字段 %s" % (i, sorted(extra)))
            path = repo / str(e.get("path") or "")
            want = str(e.get("sha256") or "")
            if not path.is_file():
                err("E25", p.name, "evidence 指向不存在的文件 %s（path 相对仓库根）"
                    % e.get("path"))
                continue
            if not want:
                err("E25", p.name, "evidence 缺 sha256（没有指纹就无法发现数字漂移）")
                continue
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != want:
                err("E25", p.name,
                    "evidence 哈希不符：%s 记录 %s… 实际 %s…（results 被改过而卡片没更新）"
                    % (path.name, want[:10], got[:10]))
            if not e.get("verified_on"):
                err("E25", p.name, "evidence 缺 verified_on")
    if missing:
        print("[E25] %d 张卡缺 evidence（warning，不进聚合）：%s"
              % (len(missing), ", ".join(sorted(missing))), file=sys.stderr)


def _peak_constants(repo):
    """从 constants.yaml 取两个达成率分母：bf16 峰值 TFLOPS、流式峰值带宽 GB/s。

    只认实测值。取不到返回 None，让 E18 明确报「读不到」而不是静默跳过——
    静默跳过正是 185 那个错了几个月没被发现的原因。
    """
    p = repo / "Plugins/ops/skills/op-roofline/constants.yaml"
    if not p.is_file():
        return None, None
    txt = p.read_text(encoding="utf-8")

    def _val(key, section):
        # 在指定 section 下找 key 的 value
        m = re.search(r"(?ms)^%s:\s*\n(.*?)(?=^\S)" % section, txt)
        if not m:
            return None
        blk = m.group(1)
        m2 = re.search(r"(?m)^\s{2}%s:\s*\n\s{4}value:\s*([\d.]+)" % re.escape(key), blk)
        return float(m2.group(1)) if m2 else None

    return _val("bf16_gemm_peak_tf", "compute"), _val("stream_read_peak", "memory")


def _library_registry(repo):
    """读 data/ops/libraries-registry.yaml → (canonical 集合, removed 映射)。"""
    p = repo / "data/ops/libraries-registry.yaml"
    if not p.is_file():
        return None, {}
    data, problem = _load_yaml(p)
    if problem or not isinstance(data, dict):
        return None, {}
    libs = {l.get("canonical_id") for l in (data.get("libraries") or [])
            if isinstance(l, dict)}
    removed = {r.get("canonical_id"): r.get("replaced_by")
               for r in (data.get("removed") or []) if isinstance(r, dict)}
    return libs, removed


def _non_installable_libs(repo):
    """registry 里 installed: false 的库（源码树/未装），L1 不适用。"""
    p = repo / "data/ops/libraries-registry.yaml"
    if not p.is_file():
        return set()
    data, problem = _load_yaml(p)
    if problem or not isinstance(data, dict):
        return set()
    return {l.get("canonical_id") for l in (data.get("libraries") or [])
            if isinstance(l, dict) and l.get("installed") is False}


def _registry(repo):
    p = repo / "data/ops/registry.json"
    if not p.is_file():
        return {}, {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}, {}
    return ({o.get("name"): o.get("status") for o in d.get("ops", [])},
            {o.get("name"): o.get("path") for o in d.get("ops", [])})


def _check_surface_laws(repo, atlas):
    """E9：surfaces.yaml 里每条攻击面的 law 必须能解析到实体锚点。

    为什么需要这条：law 曾经写成 `L1`..`L10`，而 pipeline-laws.md 根本不存在
    ——定律本体悬空，`law` 变成装饰性编号，「判死要能追溯到机制」成了空话。
    现在的约定是 `<file>#<anchor>`，anchor 必须在那个文件里有对应标题。
    """
    sp = repo / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"
    if not sp.is_file():
        warn("E9", str(sp), "surfaces.yaml 不存在，跳过定律校验")
        return
    data, problem = _load_yaml(sp)
    if problem:
        err("E9", sp.name, problem)
        return
    if not isinstance(data, list):
        err("E9", sp.name, "顶层应为列表")
        return

    refs_dir = repo / "Plugins/ops/skills/op-optimize/references"
    for s in data:
        if not isinstance(s, dict):
            continue
        sid = s.get("id", "?")
        law = s.get("law")
        if not law:
            err("E9", sp.name, "surface %s 缺 law（判死时无法追溯机制）" % sid)
            continue
        if "#" not in str(law):
            err("E9", sp.name,
                "surface %s 的 law=%r 没有锚点（应写成 <file>#<anchor>）" % (sid, law))
            continue
        fname, _, anchor = str(law).partition("#")
        # 允许两种写法：相对于 op-optimize/references/ 的文件名，
        # 或相对于仓库根的路径（定律本体常在 op-roofline/references/）
        target = refs_dir / fname
        if not target.is_file():
            target = repo / fname
        if not target.is_file():
            err("E9", sp.name, "surface %s 的 law 指向不存在的文件 %s" % (sid, fname))
            continue
        body = target.read_text(encoding="utf-8", errors="ignore")
        # 锚点允许写成 `### Q5` / `## Q5` / `# Q5`，标题后必须跟着锚点名
        if not re.search(r"(?m)^#{1,4}\s+%s\b" % re.escape(anchor), body):
            err("E9", sp.name,
                "surface %s 的锚点 %s 在 %s 中找不到对应标题" % (sid, anchor, fname))


def validate(atlas, repo):
    cards_dir = atlas / "cards"
    cards = sorted(cards_dir.glob("*.yaml"))
    if not cards:
        err("E1", str(atlas), "cards/ 下没有卡片")
        return

    lib_sections, law_ids = _truth_sources(repo)
    reg_status, reg_path = _registry(repo)
    lib_canon, lib_removed = _library_registry(repo)
    peak_bf16, peak_bw = _peak_constants(repo)
    _NON_INSTALLABLE.clear()
    _NON_INSTALLABLE.update(_non_installable_libs(repo))

    index_p = atlas / "INDEX.md"
    index_txt = index_p.read_text(encoding="utf-8") if index_p.is_file() else ""
    if not index_txt:
        err("E1", str(index_p), "INDEX.md 不存在或为空")

    seen_ids = set()
    for card_path in cards:
        c, problem = _load_yaml(card_path)
        where = card_path.name
        if problem:
            err("E0", where, problem)
            continue
        if not isinstance(c, dict):
            err("E0", where, "顶层不是 mapping")
            continue

        cid = c.get("id")
        if cid != card_path.stem:
            err("E0", where, "id=%r 与文件名 %r 不符" % (cid, card_path.stem))
        if cid in seen_ids:
            err("E1", where, "id %r 重复" % cid)
        seen_ids.add(cid)

        # ---- E6 字段白名单（事前防副本，最重要的一条）
        extra = set(c) - ALLOWED_TOP
        if extra:
            err("E6", where, "未知顶层字段 %s；卡片不接受自由文本" % sorted(extra))
        missing = REQUIRED_TOP - set(c)
        if missing:
            err("E6", where, "缺必填字段 %s" % sorted(missing))

        if c.get("family") not in FAMILIES:
            err("E8", where, "family=%r 不在 %s" % (c.get("family"), sorted(FAMILIES)))
        if c.get("status") not in STATUSES:
            err("E6", where, "status=%r 非法" % c.get("status"))
        hps = c.get("hard_preconditions")
        if hps is not None:
            if not isinstance(hps, list):
                err("E17", where, "hard_preconditions 必须是列表")
            else:
                for i, hp in enumerate(hps):
                    if not isinstance(hp, dict):
                        err("E17", where, "hard_preconditions[%d] 必须是 mapping" % i)
                        continue
                    extra = set(hp) - {"statement", "check"}
                    if extra:
                        err("E17", where,
                            "hard_preconditions[%d] 未知字段 %s" % (i, sorted(extra)))
                    for f in ("statement", "check"):
                        if not str(hp.get(f) or "").strip():
                            # hard 类不满足会崩，必须有可执行 check 才拦得住
                            err("E17", where,
                                "hard_preconditions[%d] 缺 %s（hard 类每条都要有可执行 check）"
                                % (i, f))

        mp = c.get("modify_path")
        if mp is not None:
            if not isinstance(mp, dict):
                err("E14", where, "modify_path 必须是 mapping")
            else:
                extra = set(mp) - {"kind", "what", "how", "requires"}
                if extra:
                    err("E14", where, "modify_path 未知字段 %s" % sorted(extra))
                if mp.get("kind") not in {"env-flag", "replace-call", "source-patch"}:
                    err("E14", where,
                        "modify_path.kind=%r 非法（env-flag / replace-call / source-patch）"
                        % mp.get("kind"))
                for f in ("what", "how"):
                    if not str(mp.get(f) or "").strip():
                        err("E14", where, "modify_path 缺 %s" % f)
                if mp.get("how") and len(str(mp["how"])) < 15:
                    err("E14", where, "modify_path.how 太短（%r 是废话，要写具体动作）"
                        % mp.get("how"))

        if c.get("layer") not in LAYERS:
            err("E11", where, "layer=%r 非法或缺失（kernel / engine）" % c.get("layer"))
        if c.get("layer") == "engine":
            lv = c.get("last_verdict") or {}
            if isinstance(lv, dict) and lv.get("status") not in (None, "none"):
                # engine 卡**可以**有自己的结论，但必须标明是 engine 级，
                # 且 ratio 必须带 scope——否则「受并行策略/显存水位影响的数字」
                # 会看起来像算子级结论，污染算子层判断。
                if (lv.get("ratio") or {}).get("scope") is None:
                    err("E11", where,
                        "layer=engine 的 last_verdict.ratio 必须带 scope"
                        "（引擎层数字受并行策略与显存水位影响，无 scope 会被误当算子级结论）")

        # ---- E1 cards ↔ INDEX 双向
        if index_txt and ("cards/%s.yaml" % card_path.stem) not in index_txt:
            err("E1", where, "%s 未在 INDEX.md 登记" % cid)

        # ---- E2 / E3 指针可解析
        impls = c.get("impls") or []
        if not isinstance(impls, list) or not impls:
            err("E6", where, "impls 必须是非空数组")
            impls = []
        for i, impl in enumerate(impls):
            _no_import_pre = impl.get("repo") in ("in-house",) or \
                impl.get("repo") in _NON_INSTALLABLE
            if not isinstance(impl, dict):
                err("E6", where, "impls[%d] 不是 mapping" % i)
                continue
            iextra = set(impl) - ALLOWED_IMPL
            if iextra:
                err("E6", where, "impls[%d] 未知字段 %s" % (i, sorted(iextra)))
            imiss = REQUIRED_IMPL - set(impl)
            if imiss:
                err("E6", where, "impls[%d] 缺必填 %s" % (i, sorted(imiss)))
            if impl.get("tier") not in TIERS:
                err("E6", where, "impls[%d].tier=%r 非法" % (i, impl.get("tier")))

            ref = impl.get("library_profile_ref")
            if ref is None:
                if impl.get("repo") not in ("in-house", None):
                    err("E2", where, "impls[%d] 非 in-house 缺 library_profile_ref" % i)
            else:
                m = re.fullmatch(r"libraries\.md §(\d{1,2})", str(ref))
                if not m:
                    err("E2", where, "impls[%d] ref 格式非法 %r" % (i, ref))
                elif int(m.group(1)) not in lib_sections:
                    err("E2", where, "impls[%d] %s 在 libraries.md 不存在（现有 §%s）"
                        % (i, ref, sorted(lib_sections)))

            law = impl.get("law_ref")
            if law is not None:
                txt = str(law)
                if "#" in txt:
                    f, _, anchor = txt.partition("#")
                    target = repo / f
                    if not target.is_file():
                        err("E3", where, "impls[%d] law_ref 文件不存在 %r" % (i, f))
                    elif anchor not in {
                            m for m in re.findall(r"(?m)^#{1,4}\s+([A-Za-z]\d+)\b",
                                                  target.read_text(encoding="utf-8",
                                                                  errors="ignore"))}:
                        err("E3", where, "impls[%d] 锚点 %s 在 %s 中不存在" % (i, anchor, f))
                elif txt not in law_ids:
                    err("E3", where, "impls[%d] law_ref=%r 无法解析（用 <file>#<anchor> 或 Q5 这类锚点；现有 %s）"
                        % (i, txt, sorted(law_ids)))
            if impl.get("tier") == "avoid" and not law:
                warn("W1", where, "impls[%d] avoid 条目无 law_ref，机制会丢失" % i)

            # E12：repo 必须是 libraries-registry.yaml 里的 canonical id。
            # 这条拦的是「同一库多个异名」（DeepGEMM / deepgemm-das 并存），
            # 那种状态下机器无法对账，人也会以为它们是两个库。
            repo_name = impl.get("repo")
            if lib_canon is None:
                warn("E12", where, "libraries-registry.yaml 不可读，跳过库名校验")
            elif repo_name in lib_removed:
                err("E12", where,
                    "impls[%d] repo=%r 已从环境移除（替换为 %r）"
                    % (i, repo_name, lib_removed[repo_name]))
            elif repo_name not in lib_canon:
                err("E12", where,
                    "impls[%d] repo=%r 不在 libraries-registry.yaml 里"
                    "（canonical id 见该文件；异名会被这条拦下）"
                    % (i, repo_name))

            # 场景化字段：tier=primary 必须说清「什么场景用我、什么场景别用我」。
            # 只写 tier 只回答了「谁快」，没回答「在什么条件下快」。
            if impl.get("tier") == "primary":
                for f in ("best_for", "not_for"):
                    v = impl.get(f)
                    if not v:
                        err("E10", where, "impls[%d] tier=primary 缺 %s" % (i, f))
                # L1：primary 必须能落到具体函数并验证它存在。
                # 只写「lightop 有 rope 融合族」只是知道在哪个库，不是映射。
                # 源码树/自研必须有 source_ref——只写库名等于「不知道自己长什么样」。
                # 允许 path 为 null（写明「尚无实现」），但不允许整块缺失。
                sr = impl.get("source_ref")
                if _no_import_pre:
                    if not isinstance(sr, dict):
                        err("E15", where,
                            "impls[%d] repo=%r 必须带 source_ref（path/symbol/line/build_cmd）"
                            % (i, impl.get("repo")))
                    else:
                        miss_k = {"path","symbol","line","build_cmd","note"} - set(sr)
                        if miss_k:
                            err("E15", where, "impls[%d] source_ref 缺 %s"
                                % (i, sorted(miss_k)))
                # in-house 与源码树类库（installed: false）没有可导入的函数，
                # L1 对它不适用——它们的「映射」是源码路径而非 库.函数。
                _no_import = _no_import_pre
                for f in ("import_path", "signature", "smoke_test"):
                    if not impl.get(f):
                        if _no_import:
                            # 源码树/in-house 必须写明为什么 L1 不适用
                            if not impl.get("l1_exempt_reason"):
                                err("E13", where,
                                    "impls[%d] 是源码树/自研，缺 l1_exempt_reason"
                                    "（为什么没有 库.函数）" % i)
                            else:
                                L1_EXEMPT[0] += 1
                        else:
                            err("E13", where,
                                "impls[%d] tier=primary 缺 %s（L1：要能落到 库.函数）"
                                % (i, f))
            for f in ("best_for", "not_for", "pros", "cons"):
                v = impl.get(f)
                if v is not None and not isinstance(v, list):
                    err("E10", where, "impls[%d] %s 必须是列表" % (i, f))

        # ---- E4 status 与 registry 对齐
        reg = reg_status.get(cid)
        if reg is None:
            if c.get("op_card") is not None:
                err("E4", where, "registry 无此算子，但 op_card=%r 非空" % c.get("op_card"))
        elif reg != c.get("status"):
            err("E4", where, "status=%r != registry %r" % (c.get("status"), reg))
        oc = c.get("op_card")
        if oc and not (repo / oc).is_dir():
            err("E4", where, "op_card=%r 指向的目录不存在" % oc)

        # ---- E5 ceiling 可解析
        ce = c.get("ceiling") or {}
        if not isinstance(ce, dict):
            err("E6", where, "ceiling 必须是 mapping")
            ce = {}
        cextra = set(ce) - ALLOWED_CEIL
        if cextra:
            err("E6", where, "ceiling 未知字段 %s" % sorted(cextra))
        if CEIL_KINDS - {ce.get("kind")} and ce.get("kind") not in CEIL_KINDS:
            err("E6", where, "ceiling.kind=%r 非法" % ce.get("kind"))
        if ce.get("kind") == "missing":
            if ce.get("ref") is not None:
                err("E5", where, "ceiling.kind=missing 时 ref 必须为 null")
        elif ce.get("ref"):
            ref = str(ce["ref"])
            if "#" in ref:
                f, _, key = ref.partition("#")
                if not (repo / f).is_file():
                    err("E5", where, "ceiling.ref 文件不存在: %s" % f)
                elif key and (repo / f).is_file():
                    body = (repo / f).read_text(encoding="utf-8", errors="ignore")
                    if key.split(".")[-1] not in body:
                        err("E5", where, "ceiling.ref 键 %r 在 %s 中找不到" % (key, f))
            elif not (repo / ref).exists():
                err("E5", where, "ceiling.ref 路径不存在: %s" % ref)

        # ---- E7 结论红线
        lv = c.get("last_verdict") or {}
        if not isinstance(lv, dict):
            err("E6", where, "last_verdict 必须是 mapping")
            lv = {}
        lextra = set(lv) - ALLOWED_VERDICT
        if lextra:
            err("E6", where, "last_verdict 未知字段 %s" % sorted(lextra))
        lmiss = REQUIRED_VERDICT - set(lv)
        if lmiss:
            err("E6", where, "last_verdict 缺必填 %s" % sorted(lmiss))
        if lv.get("status") not in VERDICT_STATUSES:
            err("E6", where, "last_verdict.status=%r 非法" % lv.get("status"))

        # ---- geomean：主指标（跨 shape 几何平均加速比）
        # ratio（达成率）降为诊断量：它会饱和，且依赖「峰值」这个易错的分母。
        gm = lv.get("geomean")
        if gm is not None:
            if not isinstance(gm, dict):
                err("E6", where, "last_verdict.geomean 必须是 mapping 或 null")
            else:
                gext = set(gm) - {"value", "shapes", "coverage", "vs", "gate"}
                if gext:
                    err("E6", where, "geomean 未知字段 %s" % sorted(gext))
                if "value" not in gm:
                    err("E6", where, "geomean 必须带 value")
                shapes = gm.get("shapes")
                # 单形状的几何平均就是那一个比值，防不了「挑好看的形状」，不许当主指标
                if not shapes or (isinstance(shapes, list) and len(shapes) < 2):
                    err("E22", where,
                        "geomean 覆盖少于 2 个 shape —— 单形状的几何平均没有意义"
                        "（它就是那一个比值），不能当主结论")
                if gm.get("gate") is False:
                    err("E22", where, "geomean.gate=False 却写进了卡（未过门的指标不许当结论）")

        conc = str(lv.get("conclusion") or "")
        if BARE_TIME.search(conc):
            err("E7", where, "结论含裸绝对耗时 %r —— 绝对数字进 kb，卡里只留相对倍数"
                % BARE_TIME.search(conc).group(0))
        if BARE_PCT.search(conc):
            err("E7", where, "结论含无分母百分比 %r —— 百分比必须写明相对谁"
                % BARE_PCT.search(conc).group(0))
        if len(conc) > 60:
            err("E7", where, "结论 %d 字，超过 60 字上限（会变成技术描述）" % len(conc))

        ratio = lv.get("ratio")
        if ratio is not None:
            if not isinstance(ratio, dict):
                err("E6", where, "last_verdict.ratio 必须是 mapping 或 null")
            else:
                # scope：跨口径比较是最隐蔽的错误来源。
                # 内部材料里就有一句自我纠正——「那个 1.67–1.86 倍是 4 卡
                # rocBLAS 换核的结果，不是 8 卡同口径对比」。
                rextra = set(ratio) - {"kind", "unit", "value", "vs", "scope",
                                       "traffic_passes", "inputs", "outputs"}
                if rextra:
                    err("E6", where, "ratio 未知字段 %s" % sorted(rextra))
                # 有 value 就必须有 scope：没有口径的倍数不可比
                if ratio.get("value") is not None and not ratio.get("scope"):
                    err("E16", where,
                        "ratio 有 value 但缺 scope（cards/concurrency/in_len/out_len/warmup）"
                        "——跨口径比较是最隐蔽的错误来源")
                if "value" not in ratio or "vs" not in ratio:
                    err("E6", where, "ratio 必须带 value 与 vs（分母不许省略）")

                # ---- E24 量纲：value 必须声明它是什么
                # 由来：24 条实测混着 加速比 7 / 达成率 5 / 绝对吞吐 10 / 其它 2，
                # 「2.39」（倍）与「887」（GB/s）放在同一字段，任何跨卡聚合都会错，
                # 也排不出「哪张卡优化得最好」。
                #   速度类：speedup（相对基线的倍数）| utilization（占峰值）| throughput（绝对吞吐）
                #   非收益比值（如「M=128 与 M=1024 的耗时比」）不许算速度，用 ratio_other
                kind = ratio.get("kind")
                val = ratio.get("value")
                if kind is None:
                    err("E24", where,
                        "ratio 缺 kind（speedup/utilization/throughput/ratio_other）"
                        "——量纲不明的 value 无法聚合、也无法排序")
                elif kind not in RATIO_KINDS:
                    err("E24", where, "ratio.kind=%r 非法（只允许 %s）"
                        % (kind, sorted(RATIO_KINDS)))
                else:
                    if not ratio.get("unit"):
                        err("E24", where,
                            "ratio 缺 unit（x / ratio / GB/s / TFLOPS）——跨卡聚合要按 unit 分组")
                    # utilization 是「占峰值」，不可能超过 150%（超了是分母错）
                    if kind == "utilization" and val is not None:
                        try:
                            if float(val) > 1.5:
                                err("E24", where,
                                    "kind=utilization 但 value=%s > 1.5 —— "
                                    "达成率不会超过 150%%，这多半是分母口径错（见 E23）"
                                    % val)
                        except (TypeError, ValueError):
                            err("E24", where, "ratio.value 不是数字：%r" % (val,))
                    # 防「887 GB/s 被读成 887 倍」：非 speedup 不该出现 >1.5 的裸倍数
                    if kind in ("throughput", "ratio_other") and val is not None:
                        try:
                            v = float(val)
                            if v > 1.5 and not ratio.get("unit"):
                                err("E24", where,
                                    "kind=%s 且 value=%.3g 但没有 unit —— "
                                    "容易被读成「%.0f 倍」" % (kind, v, v))
                        except (TypeError, ValueError):
                            pass

                # E18：vs 里若引用了机器峰值，必须等于 constants.yaml 的实测值。
                # 历史：bf16 峰值在 5 处各写各的（185/217/310），带宽写 1728（常数 1724）。
                # 「一个常数散落多处、只有一处对」正是分母用错的温床。
                vs = str(ratio.get("vs") or "")
                m = PEAK_CLAIMS["bf16"].search(vs)
                if m and peak_bf16 is not None:
                    got = float(m.group(1))
                    if abs(got - peak_bf16) / peak_bf16 > 0.02:
                        err("E18", where,
                            "vs 里写的 bf16 峰值 %g 与 constants.yaml 实测 %g 不符"
                            "（偏差 %.0f%%）——达成率分母必须引用实测值"
                            % (got, peak_bf16, abs(got - peak_bf16) / peak_bf16 * 100))
                m = PEAK_CLAIMS["bw"].search(vs)
                if m and peak_bw is not None:
                    got = float(m.group(1))
                    if abs(got - peak_bw) / peak_bw > 0.02:
                        err("E18", where,
                            "vs 里写的流式峰值带宽 %g 与 constants.yaml 实测 %g 不符"
                            "——达成率分母必须引用实测值" % (got, peak_bw))

    # ---- E1b INDEX 指向的卡必须存在
    for m in re.findall(r"cards/([a-z0-9-]+)\.yaml", index_txt):
        if not (cards_dir / ("%s.yaml" % m)).is_file():
            err("E1", "INDEX.md", "指向不存在的卡 %s" % m)

    # ---- E26：measured/tuned 卡必须写「下一步改什么、改到什么时候停」
    # 由来：22 张 measured 里 0 张写了下一个攻击面，于是 24 张卡退化成 24 张基线快照。
    # 但仅要求「字段存在」会变成形式主义——写「再看看」也能过机检。
    # 所以三条硬要求：
    #   surface     必须命中 surfaces.yaml 的 id（或 none + 理由）
    #   hypothesis  必须含**一个数字**（可证伪的预期）
    #   stop_if     必须含**机制词**（不涨时凭什么判死，不能只写「不涨就停」）
    _check_next_probe(cards_dir, repo)

    # ---- E25 证据指纹：measured/tuned 卡的数字必须能追到可复跑的文件
    # 这是「分母漂移」的同族问题：数字写在卡片与 results 两处，只有一处对。
    # 已经栽过两次（540 无来源、49.8 当峰值）。这里用 sha256 把两处钉在一起。
    #
    # 分档（业主 2026-11-23 裁决）：
    #   hash_mismatch  → error（真漂移，必须修，不许降级）
    #   missing        → warning + 不进聚合；连续 ≥2 批则应由 GAPS 判降级
    # 不立刻对 missing 报 error 的理由：14 张卡本来就没有可挂的证据文件，
    # 强制 error 只会逼人回去造一个假 results 骗门禁 —— 与「不为凑数造假」冲突。
    _check_evidence(cards_dir, repo)

    # ---- E21 顶层 status 与 last_verdict.status 不许互相矛盾
    # 两者语义不同（顶层是「这个算子推到哪一步」，last_verdict 是「上次结论的性质」），
    # 但下面两种组合是明确矛盾：
    #   顶层 draft/tuned 之外，last_verdict 却写 measured  —— 有实测却不算数
    #   顶层 measured，last_verdict 却是 none             —— 标了 measured 却没有结论
    # moe-w8a8 是唯一允许的例外形态：last_verdict 只覆盖某个分量（Marlin），
    # 整体未判，所以顶层仍是 draft。这种情况必须用 scope 写明「仅某分量」。
    for c in cards:
        data, problem = _load_yaml(c)
        if problem or not isinstance(data, dict):
            continue
        top = str(data.get("status") or "")
        lv = data.get("last_verdict") or {}
        lvs = str(lv.get("status") or "")
        if top in ("draft",) and lvs in ("measured", "tuned"):
            scope = str((lv.get("ratio") or {}).get("scope") or "")
            note = str(((lv.get("ratio") or {}).get("scope") or {}).get("note") or "") \
                if isinstance((lv.get("ratio") or {}).get("scope"), dict) else ""
            if "仅" not in note and "部分" not in note:
                err("E21", c.name,
                    "顶层 status=draft 但 last_verdict.status=%r；"
                    "若只测了某个分量，必须在 ratio.scope.note 写明「仅 XX 分量」" % lvs)
        if top in ("measured", "tuned") and lvs == "none":
            err("E21", c.name,
                "顶层 status=%r 但 last_verdict.status=none（没有结论的卡不能标 measured）" % top)

    # ---- E23 引用「峰值」做分母必须标明口径（nominal/achievable/impl_measured）
    # 由来：2026-11-23 一天之内两处分母错（185 反推、540 无来源、310 单轮采样、
    # peak/steady 混用）。口径不写清，读者无法判断「达成率」该不该信，
    # 也分不清「离硬件上限多远」与「这条实现多快」。
    for c in cards:
        data, problem = _load_yaml(c)
        if problem or not isinstance(data, dict):
            continue
        lv = data.get("last_verdict") or {}
        if lv.get("status") not in ("measured", "tuned"):
            continue
        ratio = lv.get("ratio") or {}
        vs = str(ratio.get("vs") or "")
        if not re.search(r"峰值|peak", vs, re.I):
            continue
        note = ""
        sc = ratio.get("scope")
        if isinstance(sc, dict):
            note = str(sc.get("note") or "")
        blob = vs + " " + note
        if not re.search(r"nominal|achievable|impl_measured|实测|标称", blob):
            err("E23", c.name,
                "ratio.vs 引用了「峰值」但没标口径（nominal/achievable/impl_measured）"
                "——三种口径的达成率不可互比，读者也无从判断该不该信")

    # ---- E24b ratio_other 是逃生口，不是常驻地
    # 若它涨上去，说明 kind 分类没收敛（有东西本该判成 speedup/utilization/throughput
    # 却被塞进「其它」）。超过 5 条就提醒复核分类，而不是继续往兜底里丢。
    n_other = 0
    for c in cards:
        data, problem = _load_yaml(c)
        if problem or not isinstance(data, dict):
            continue
        lv = data.get("last_verdict") or {}
        if lv.get("status") not in ("measured", "tuned"):
            continue
        if (lv.get("ratio") or {}).get("kind") == "ratio_other":
            n_other += 1
    if n_other > RATIO_OTHER_LIMIT:
        warns.append("[E24] ratio_other 已 %d 条（上限 %d）：它是逃生口不是常驻地，"
                     "涨上去说明 kind 分类没收敛，请复核这几条是否本该判成 "
                     "speedup/utilization/throughput" % (n_other, RATIO_OTHER_LIMIT))

    # ---- E28 「到顶」必须带 falsifier；surface=none 必须说明为什么不再推进
    # 由来：2026-11-23 我在 rmsnorm 上试了 cross-lane-fold 后判死并标 at_floor，
    # 后来发现 tilelang 能做到 0.82（我 0.78）——那是 triton 的 codegen 上限，
    # 不是方法极限。**没有 falsifier 的「到顶」等于把工具限制记成物理规律。**
    _check_depth_rules(cards_dir, repo)

    # ---- E27 基准合法性与语义对齐（读 baseline-policy.yaml）
    _check_baseline_policy(cards_dir, repo)

    # ---- E27b 跨实现比较必须对齐访存口径
    # 由来：2026-11-23 我们拿「只写 1 份输出（201 MB）」的 kernel 去比
    # lightop 的「写 2 份（268 MB）」，得出「快 1.18x」——**少干 1/3 的活然后说自己快**。
    # 等语义下实际慢 6.7%。这条防线要求：只要写了 comparison，就必须写明对齐口径。
    for c in cards:
        data, problem = _load_yaml(c)
        if problem or not isinstance(data, dict):
            continue
        lv = data.get("last_verdict") or {}
        comp = lv.get("comparison")
        if not comp:
            continue
        ins = lv.get("inputs", (lv.get("ratio") or {}).get("inputs"))
        outs = lv.get("outputs", (lv.get("ratio") or {}).get("outputs"))
        align = (comp or {}).get("alignment") if isinstance(comp, dict) else None
        if ins is None or outs is None:
            err("E27", c.name,
                "有 comparison 但没写 inputs/outputs —— "
                "跨实现比较前必须先对齐访存口径（不对齐就是在比两件不同的活）")
        elif not align:
            err("E27", c.name,
                "有 comparison 但 comparison.alignment 缺失 —— "
                "必须写明双方各自的 (inputs, outputs) 是否一致")

    # ---- E19 INDEX 表里的状态/家族/上限必须与卡片一致
    # 由来：13 张卡已 promoted 到 measured，INDEX 里还写着 draft。
    # 「一个事实在多处登记、只有一处对」正是分母用错那类错误的温床。
    # INDEX 现在是生成物（sync_index.py），这里做反向核对。
    _check_index_matches_cards(cards_dir, index_txt)

    # ---- E8 家族文件齐备
    for c in cards:
        data, problem = _load_yaml(c)
        if problem or not isinstance(data, dict):
            continue
        fam = data.get("family")
        if fam in FAMILIES and not (atlas / "families" / ("%s.md" % fam)).is_file():
            err("E8", c.name, "缺家族文件 families/%s.md" % fam)

    if L1_EXEMPT[0]:
        print("[L1] %d 个 impl 因源码树/自研豁免（均带 l1_exempt_reason）"
              % L1_EXEMPT[0], file=sys.stderr)

    # ---- E9 surfaces.yaml 的 law 必须解析到实体（防悬空引用）
    _check_surface_laws(repo, atlas)


def main():
    ap = argparse.ArgumentParser(description="op-atlas 防漂移校验")
    ap.add_argument("--atlas", default="Plugins/ops/op-atlas")
    ap.add_argument("--repo", default=None, help="仓库根（默认从 atlas 路径推）")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    atlas = Path(a.atlas).resolve()
    repo = Path(a.repo).resolve() if a.repo else atlas.parents[2]
    validate(atlas, repo)

    if a.json:
        print(json.dumps({"errors": errors, "warns": warns}, ensure_ascii=False, indent=2))
    else:
        for e in errors:
            print(e)
        for w in warns:
            print(w)
        print("\n结果：%d 错误 / %d 警告" % (len(errors), len(warns)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
