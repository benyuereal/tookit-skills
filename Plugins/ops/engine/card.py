#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""card —— 读算子卡的**唯一入口**（格式归一化只在这里发生）。

## 为什么收口（这是修 bug，不是洁癖）

`candidates.yaml` 在仓库里同时存在两种顶层格式：

  · 老卡是 dict：`defaults / directions / candidates / refuted / stalled`
  · 新工坊是**裸 list**：文件内容就是一串候选。实测 2026-11-27，9 张卡里 4 张
    （op-gemm-bf16 / op-mqa-logits / op-fusion-rmsnorm-rope /
     op-fusion-add-rmsnorm-quant）

这个兼容原本只修在 `engine/loop.py::load_candidates()` 一处。后果不是「风格不统一」，
而是**别的读卡点直接崩**：`scripts/check_expect.py` 按字母序扫到第 5 张卡
（`op-fusion-add-rmsnorm-quant`）就 `AttributeError: 'list' object has no attribute 'get'`，
整轮扫描中止 —— 它后面所有卡（含任何新卡）的 expect 质量根本没被检查过。
一个兼容补丁只修在**一个调用点**，等于只修了九分之一的读卡路径。

所以这里把「读卡」收成唯一入口：谁来读卡都调这里，归一化只发生一次。
新读卡点如果绕过本模块，`tests/` 之外没人拦得住 —— 但至少现在只有一处要改。

## 边界（为什么判定不在这里）

本模块只做 I/O 与 schema 归一化，**不做任何判定**：
「这条候选能不能跑」是 `loop.py` 六道门的事（方向预算、bottleneck 归因、
done_when 外部裁决……），读卡层一旦参与判定，就会出现「读卡层和门禁各有一份
口径」的老问题 —— 这正是本轮重构要消灭的东西。

也不做缓存：卡片是活文档，人/agent 在循环中途改它是正常操作，
缓存会让「改了没生效」变成最难查的一类 bug。
"""
from __future__ import annotations

from pathlib import Path

# 卡片里的文件名一律在这里写死一次。散落在各脚本里的字面量拼错不会报错，
# 只会静默读不到（`Path("shape.yaml")` 不存在 ⇒ 空字典 ⇒ 上层「缺数据」）。
CANDIDATES = "candidates.yaml"
CONTRACT = "contract.yaml"
SHAPES = "shapes.yaml"
VARIANTS = "variants.yaml"
ROOFLINE = "roofline.json"                    # op-roofline/derive.py 的正式产物

# 多轮编排自动生成的候选登记处。**advance 不写卡片**（写范围只到 runs/），
# 生成的变体要进六道门只能通过这份 overlay —— loop.py 读卡时合并它。
OVERLAY = Path("runs") / "advance" / "overlay.yaml"

# candidates.yaml 的五个段。缺省值必须是「空容器」而不是 None：
# 下游全是 `for c in (d.get("candidates") or [])` 和 `d["directions"]`，
# None 会在某几处变成 TypeError，且只在没有该段的卡上出现。
_SECTIONS = (("defaults", dict), ("directions", list), ("candidates", list),
             ("refuted", list), ("stalled", list))


def _as_path(card):
    """接受 str / Path，统一成 Path。调用点传参习惯不统一（有 Path 有 str）。"""
    return card if isinstance(card, Path) else Path(str(card))


def read_yaml(path, default=None):
    """容错读 YAML：文件不存在或解析失败都返回 default。

    故意不抛：卡片缺文件是**常态**（只有参与优化循环的卡才有
    candidates.yaml；纯知识 skill 本来就没有），把「没有」当异常处理，
    调用方就得为每个可选文件写一遍 try —— 实测那样写出来的第三份实现
    会把 OSError 吞成空字典、把 ImportError 也吞成空字典，最后分不清
    「没装 PyYAML」和「文件里就是空」。
    """
    p = _as_path(path)
    try:
        import yaml
    except ImportError:                                  # 无 PyYAML 环境
        return {} if default is None else default
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default
    if d is None:
        return {} if default is None else default
    return d


def read_json(path, default=None):
    """容错读 JSON（roofline.json / runs 记录）。与 read_yaml 同一套契约。"""
    import json
    p = _as_path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def normalize_candidates(d, path="<candidates>"):
    """把 candidates.yaml 的两种顶层格式归一成同一套 dict。

    这是**唯一**的兼容点。要做的事只有三件：
      ① 裸 list → `{"candidates": [...]}`（新工坊写法）
      ② 五个段补齐缺省（老卡只写 candidates 段是合法的）
      ③ 既不是 dict 也不是 list ⇒ 明确报错（不许静默当空卡：
         静默会让「文件写坏了」看起来像「这张卡没有候选」，白跑一整轮）

    `<path>` 只用于报错信息 —— 定位不到文件的报错等于没报错，
    实测过一次「必须是 dict 或 list」但不知道是哪张卡。
    """
    if isinstance(d, list):
        d = {"candidates": d}
    if not isinstance(d, dict):
        raise SystemExit(
            "candidates.yaml 必须是 dict 或 list，实际是 %s：%s"
            % (type(d).__name__, path))
    for key, ctor in _SECTIONS:
        d.setdefault(key, ctor())
    return d


def load_candidates(card):
    """读卡片的候选队列（归一化后），**并合并 runs/advance/overlay.yaml**。

    overlay 是「多轮编排自动生成的候选」的登记处。它解决一个真实冲突：
      · ENGINE.md 规定 advance **不许写卡片**（写范围只到 runs/）
      · 但 loop.py 的登记门要求「候选必须在队列里」，否则拒绝跑
    没有 overlay 时，自动生成的变体无法进入六道门 —— 要么绕过门（不可接受），
    要么让 advance 去改 candidates.yaml（违反边界）。

    overlay 只是**追加**，永远不能：
      · 覆盖已存在的候选 id（重复即冲突，直接报错，不静默取其一）
      · 改 refuted 清单（已判死的不能被 overlay 复活）
      · 改 defaults / directions 的状态与计数
    所以它不能让任何一条已经生效的约束失效。

    每个 overlay 候选会打上 `_origin: advance-overlay`，记录里看得出来源。

    没有 candidates.yaml 时返回**空壳**而不是报错：空白卡是合法状态
    （bootstrap 刚建出来、或纯知识 skill），`status`/`queue` 要能照常跑。
    """
    card = _as_path(card)
    p = card / CANDIDATES
    if not p.exists():
        d = normalize_candidates({}, p)
    else:
        d = normalize_candidates(read_yaml(p, {}), p)
    _merge_overlay(d, card / OVERLAY)
    return d


def _merge_overlay(d, ov):
    """把 overlay 里的候选追加进 d["candidates"]，冲突即报错。"""
    if not ov.is_file():
        return d
    o = read_yaml(ov, {})
    if not isinstance(o, dict):
        return d
    have = {c.get("id") for c in d["candidates"] if isinstance(c, dict)}
    dead = {r.get("id") for r in d["refuted"] if isinstance(r, dict)}
    for c in (o.get("candidates") or []):
        if not isinstance(c, dict) or not c.get("id"):
            continue
        cid = c["id"]
        if cid in have:
            raise SystemExit(
                "overlay 冲突：候选 %r 与卡片已有候选同 id。"
                "overlay 只能新增，不能覆盖（否则登记的候选会被悄悄换掉）。" % cid)
        if cid in dead:
            raise SystemExit(
                "overlay 冲突：候选 %r 已在 refuted 里。"
                "已判死的候选不能被 overlay 复活。" % cid)
        have.add(cid)
        d["candidates"].append(dict(c, _origin="advance-overlay"))
    return d


def load_contract(card):
    """读 contract.yaml（精度容差、op 名、点击参数的唯一真源）。缺文件返回 {}。"""
    d = read_yaml(_as_path(card) / CONTRACT, {})
    return d if isinstance(d, dict) else {}


# contract.yaml 里精度容差的两条合法写法。第一种是新 schema（分口径写），
# 第二种是老卡（只有一个 max_abs）。顺序即优先级，**不许合并成 max()**：
# 新写法和老写法同时出现时，语义明确的那条（vs_fp32_ref）才是主口径。
_TOL_PATHS = (("tolerance", "vs_fp32_ref", "max_abs"), ("tolerance", "max_abs"))


def contract_tol(contract, default=2e-3):
    """从已读出的 contract dict 里取 max_abs 容差；没有就返回 default。

    单独一个纯函数（不读文件）是刻意的：调用方已经读好 contract 时不必再读一遍，
    而「容差在哪几个键下」这件事只该有**一处**知道 —— 原先 loop.py 与
    bench 适配器各写了一份键路径，改 schema 时必漏一处。
    """
    for path in _TOL_PATHS:
        node = contract
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            return float(node)
    return default


def load_contract_tol(card, default=2e-3):
    """`load_contract` + `contract_tol` 的合并入口（卡片不存在也返回 default）。"""
    try:
        return contract_tol(load_contract(card), default)
    except Exception:                                    # noqa: BLE001
        # contract.yaml 写坏了不该让整轮跑不起来：容差有默认值，而
        # 「读不到」与「没写」在这里是同一件事（都用默认容差）。
        return default


def load_shapes(card):
    """读 shapes.yaml（档位 → params）。缺文件返回 {}。"""
    d = read_yaml(_as_path(card) / SHAPES, {})
    return d if isinstance(d, dict) else {}


def load_variants(card):
    """读 variants.yaml 的 `variants:` 列表。缺文件返回 []。

    这里只把「列表」取出来；元素是不是 dict、有没有 role 由消费方判 ——
    读卡层对元素做 schema 校验的话，一张老卡里一个畸形元素会让整张卡读不出来。
    """
    d = read_yaml(_as_path(card) / VARIANTS, {})
    vs = d.get("variants") if isinstance(d, dict) else d
    return [v for v in (vs or []) if isinstance(v, dict)]


def load_directions(card):
    """读 directions 段。裸 list 卡没有这一段 ⇒ []（不是错误）。

    先走 load_candidates 是刻意的：overlay 不能加方向，但归一化与
    「文件写坏了要报错」这两件事必须与其它读卡路径完全一致，
    否则 `load_directions()` 会把一张写坏的卡读成「没有方向」。
    """
    return [d for d in (load_candidates(card).get("directions") or [])
            if isinstance(d, dict)]


def load_roofline(card):
    """读 roofline.json（derive.py 的正式产物）。缺文件 / 坏 JSON 返回 {}。"""
    d = read_json(_as_path(card) / ROOFLINE, {})
    return d if isinstance(d, dict) else {}


def load_roofline_measurements(card):
    """读 runs/loop/*.jsonl 里 loop.py 落的实测记录（按文件名序）。

    之所以放在 card.py 而不放回 loop.py：用它的地方（explore 的达成率、
    advance 的 judge）都只读不写，而 loop.py 的写记录路径要读模块级
    开关（MEMORY_WRITE）—— 读与写在这里是两件事，混在一个模块里
    会让「只想读一下记录」的脚本被迫连带加载记账逻辑。
    """
    import json
    d = _as_path(card) / "runs" / "loop"
    out = []
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def atlas_id(card):
    """卡片目录名 → atlas 卡 id（`op-gated-add` → `gated-add`）。

    atlas 卡文件名去掉 `op-` 前缀是这个仓库的约定（see op_bootstrap）。
    两处各写一遍 `name.startswith("op-")` 的话，第三处一定会漏掉前缀判断。
    """
    n = _as_path(card).name
    return n[3:] if n.startswith("op-") else n


def atlas_card_path(card, repo_root):
    """atlas 卡路径。`repo_root` 由调用方传（engine 不知道自己被装到哪）。"""
    return _as_path(repo_root) / "Plugins/ops/op-atlas/cards" / ("%s.yaml" % atlas_id(card))


def load_atlas_card(card, repo_root):
    """读 atlas 卡（floor_inputs / variants 的登记处）。读不到返回 {}。"""
    p = atlas_card_path(card, repo_root)
    d = read_yaml(p, {})
    return d if isinstance(d, dict) else {}
