#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""advance —— 多轮编排的**唯一入口**（v5.1 §6.2）。

它只做一件事：把「计划 → 前提门 → 跑实验 → 判 → 记 → 重规划 → 停」串起来，
**不重实现任何判定**。

铁律（违反即为安全性倒退）
--------------------------
`loop.py run` 已有六道门：
  ① candidates 登记  ② bottleneck 归因  ③ done_when 外部裁决
  ④ 方向预算  ⑤ refuted 拒跑  ⑥ allowed_actions 天花板约束
本模块自己跑实验，这六道门全绕过。所以：

  * run 只通过子进程调 `python3 -I <engine>/loop.py -C <card> run <tag>`
  * judge 只读 loop.py 写进 `runs/loop/*.jsonl` 的 `decision`，不自己算显著性
  * 任何要执行的命令里出现 `ab.py` / `bench/` 一律拒绝（`_guard_no_bench`）

唯一验收
--------
`advance.py --card X --max-rounds 1` 跑完一轮，中间没人敲命令，
`runs/loop/*.jsonl` 多一条记录，且 advance 自己的 `progress.stage` 前进。

    每轮：plan → gate(precondition) → run(调 loop.py) → judge → record
          → replan → stop

结构：本文件只剩「编排」这一件事
--------------------------------
纯逻辑已经搬进 `Plugins/ops/engine/`：
  · `ledger.py`   —— Cand/Top、parse_gain、replan、ledger_cites、overlay、progress
  · `card.py`     —— 读卡（含 overlay 合并、candidates.yaml 两种顶层格式）
  · `floor.py`    —— 地板（at_floor 前置门用的 remaining 也来自它）
本文件保留的是**只剩这里能做的事**：一轮一轮地推进、通过子进程调 loop.py、
以及 `SURFACES_PATH` 这个模块级常量 —— 测试用 monkeypatch 把它指向临时
surfaces.yaml（`test_advance.py::surfaces_file`），把它或读它的函数搬走，
补丁会静默失效：测试仍然"通过"，但测的已经不是真实路径了。
这类耦合比"文件行数"重要得多 —— 宁可这个文件长，也不许测试变成假绿。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# scripts/advance.py → 向上找到 .git 即为仓库根（与 loop.py 同一策略）
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[1])
ENGINE_DIR = REPO_ROOT / "Plugins/ops/engine"
ENGINE_LOOP = ENGINE_DIR / "loop.py"
SURFACES_PATH = REPO_ROOT / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"

def _engine(name):
    """按路径加载 engine/<name>.py（见 engine/loop.py::_engine 的说明）。

    必须在常量之前定义：下面的 ADVANCE_DIR / Cand 等都是对 engine 模块的转发，
    它们在**模块导入时**就要能取到值。这类顺序问题在脚本里最容易埋雷 ——
    `python3 -m py_compile` 过得去，只有真的 import 时才炸。
    """
    key = "_ops_engine_" + name
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(ENGINE_DIR / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# 读卡的唯一入口。**这里曾经 importlib 加载 loop.py 只为了借它的
# load_candidates** —— 为了「别自己再写一份 overlay 合并」而把整个 2200 行
# 引擎拖进编排层。现在那份实现在 engine/card.py，直接调它即可。
card_io = _engine("card")
# 地板入口：三处各算一遍地板的重复已收敛到 engine/floor.py。
floor_io = _engine("floor")
# 记账入口：纯数据结构、replan、overlay、progress 都在 engine/ledger.py。
# 本文件只留「编排」——它读模块级 SURFACES_PATH（测试 monkeypatch 的那个名字），
# 搬走会让补丁静默失效，见 run_advance 的说明。
_ledger = _engine("ledger")

# 进度文件 / stage 语义 / overlay 命名定义在 engine/ledger.py（与 explore 共用）。
# 这里保留同名转发：既有调用点与测试直接读 advance.ADVANCE_DIR 之类。
ADVANCE_DIR = _ledger.ADVANCE_DIR
STAGES = _ledger.STAGES
TERMINAL = _ledger.TERMINAL

# floor.py 的 at_floor 阈值（remaining < 0.10 ⇒ 别再折腾）
AT_FLOOR = 0.10
# 连续 refuted 多少次就停（「连续」只看本轮 decision，不看噪声）
MAX_CONSECUTIVE_REFUTED = 2


# ---------------------------------------------------------------- 数据结构

# Cand / RejectedList / Top 定义在 engine/ledger.py。这里 re-export 是**契约**：
# 测试直接构造 `adv.Cand(...)` / `adv.Top(...)`，文档里也这么写。
Cand = _ledger.Cand
RejectedList = _ledger.RejectedList
Top = _ledger.Top


# ---------------------------------------------------------------- 知识读取

def load_surfaces():
    d = card_io.read_yaml(SURFACES_PATH, {}) if Path(SURFACES_PATH).is_file() else {}
    return [s for s in d if isinstance(s, dict) and s.get("id")] if isinstance(d, list) else []


def parse_gain(text):
    """surfaces.yaml 的 expected_gain 文本 → 0..1 代理值（实现在 engine/ledger.py）。

    口径写死在那个函数的 docstring 里；这里保留名字是因为 plan/selftest 与测试
    都直接点名它。
    """
    return _ledger.parse_gain(text)


def read_roofline(card):
    """读 `<卡>/roofline.json`（derive 的离线产物）。读不到返回 {}。

    这个函数保留的原因是**返回值**：`lang_select.route_scenario()` 要的是整份
    roofline（含场景签名），不只是地板。谁要地板谁去 `floor_io.resolve()`。
    """
    return card_io.load_roofline(card)


def card_id(card, roof=None):
    roof = roof if roof is not None else read_roofline(card)
    return str(roof.get("card") or Path(card).name)


# read_records / judge / confirm_record 都在 engine/ledger.py：它们只读 loop.py
# 的产出，不产生判定，编排层不该自己再写一遍「怎么读 runs/loop」。
read_records = _ledger.read_records


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(Path(p).resolve())


def pick_results_json(card):
    """给 `<results.json>` 占位符挑一个真实存在的 results 文件；没有返回 None。"""
    d = Path(card) / "results"
    preferred = d / "impl-ranking.json"
    if preferred.is_file():
        return _rel(preferred)
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            return _rel(p)
    return None


# ---------------------------------------------------------------- plan（简化版）

def _load_candidates_via_loop(card):
    """读候选队列（含 overlay 合并），返回 (candidates, 来源)。

    函数名保留 `_via_loop` 是为了不动调用点与日志里的来源串，但**实现已经
    不经过 loop.py**：合并 runs/advance/overlay.yaml 的逻辑收在
    engine/card.py::load_candidates 里，与 loop.py 用的是同一个函数。
    原先为了借这一步而 importlib 整个 loop.py，等于让编排层依赖引擎的
    全部导入副作用（它还会去找 measure_mode、注册常量、读 references/），
    而真正需要的只有「读卡 + 合并」这十行。

    读卡失败时退回 {}（plan 会得到「没有 pending 候选」）：
    plan 是只读的规划步骤，它不该因为卡片写坏了而崩掉整轮编排 ——
    真坏了由 loop.py 的门去拒，那里才是该报错的地方。
    """
    try:
        return card_io.load_candidates(Path(card)), "card.load_candidates"
    except Exception:                           # noqa: BLE001
        return {}, "card-unreadable"


def _sweep_key(cid):
    """同一批 sweep 的变体共享前缀（实现在 engine/ledger.py::sweep_key）。"""
    return _ledger.sweep_key(cid)


def build_plan(card, ledger=None):
    """读候选队列（含 overlay）+ surfaces.yaml，给出 top3。

    **只在这里读 SURFACES_PATH**：测试用 monkeypatch 把它指到临时文件，
    所以"从哪读 surfaces"这个知识必须留在本模块（这也是 build_plan 没搬进
    engine 的唯一原因，见 run_advance 的说明）。纯排序逻辑在 ledger.build_plan。
    """
    cands, _src = _load_candidates_via_loop(card)
    return _ledger.build_plan(cands, load_surfaces(), ledger=ledger)


# ---------------------------------------------------------------- gate（前提门）

# 前提检查支持的全部占位符。**每加一个都必须给出落地来源**，
# 否则它会被 shell 当成重定向（`pmc_probe.py <csv>` → `/bin/sh: Syntax error`），
# 而报错看起来像「命令写错了」，真因是占位符没人填。
PLACEHOLDERS = ("<card>", "<results.json>", "<csv>", "<X>")


def _find_pmc_csv(card):
    """在卡的 runs/ 与 /tmp 里找最近一份 PMC CSV。找不到返回 None（不猜）。"""
    import glob
    pats = [str(Path(card) / "runs" / "**" / "*.csv"),
            "/tmp/pmc*/**/*.csv", "/tmp/pmc*/*.csv"]
    cands = []
    for pat in pats:
        cands += glob.glob(pat, recursive=True)
    cands = [c for c in cands if Path(c).is_file()]
    if not cands:
        return None
    return max(cands, key=lambda c: Path(c).stat().st_mtime)


def _find_co(card):
    """找一份已编好的设备目标文件（.co / .hsaco / .so）给 ISA 类前提检查用。"""
    import glob
    # 本机可以反汇编的只有 .hip 源（→ --save-temps 的 *.out）与 *.out 本身。
    # .so/.co/.hsaco 实测都被 dccobjdump 拒（"not recognized as a valid object
    # file"）：.so 是宿主 ELF，dccobjdump 只认 elf64-amdgpu。所以优先 .hip。
    pats = [str(Path(card) / "runs" / "**" / "*.hip"),
            str(Path(card) / "runs" / "**" / "*gfx*.out"),
            str(Path(card) / "runs" / "**" / "*.hsaco"),
            str(Path(card) / "runs" / "**" / "*.co")]
    cands = []
    for pat in pats:
        cands += glob.glob(pat, recursive=True)
    cands = [c for c in cands if Path(c).is_file()]
    return max(cands, key=lambda c: Path(c).stat().st_mtime) if cands else None


def _subst_check(cmd, card, cid, results_rel):
    s = str(cmd).replace("<card>", cid)
    if "<results.json>" in s:
        if not results_rel:
            return None                      # 占位符无法落地 = unrunnable
        s = s.replace("<results.json>", results_rel)
    if "<csv>" in s:
        csv = _find_pmc_csv(card)
        if not csv:
            return None                      # 没有 PMC 采集 → unrunnable，不假装通过
        s = s.replace("<csv>", csv)
    if "<X>" in s:
        co = _find_co(card)
        if not co:
            return None                      # 没有已编译产物 → unrunnable
        # 模板里写成 `<X>.co` / `<X>.co.ISA`，**带后缀**。硬拼 stem+".co" 会拼出
        # 不存在的路径（实测：找到的是 auto-gen1.so，拼出来是 auto-gen1.co，
        # 报「目标不存在」——看起来像磁盘上没文件，其实是后缀拼错了）。
        # 所以先按出现的完整后缀整体替换，再把剩下的裸 <X> 换掉。
        real = str(co)
        for suf in (".co.ISA", ".hsaco.ISA", ".co", ".hsaco", ".so", ".hip"):
            s = s.replace("<X>" + suf, real if suf.endswith(".ISA")
                          else real)
        s = s.replace("<X>", real)
    return s


def gate(cands, card, runner):
    """跑候选的 precondition_check。返回 (通过的候选列表, 逐条日志)。

    **不跑实验**：precondition_check 是 roofline/PMC/ISA 的读数命令，不是 bench。
    任一命令含 ab.py / bench/ 直接拒绝（见 _guard_no_bench）。
    """
    cid = card_id(card)
    results_rel = pick_results_json(card)
    passed, log = [], []
    for c in cands:
        if not c.precondition_check:
            passed.append(c)
            log.append({"id": c.id, "check": None, "ok": True, "why": "无 precondition_check"})
            continue
        cmd = _subst_check(c.precondition_check, card, cid, results_rel)
        if cmd is None:
            log.append({"id": c.id, "check": c.precondition_check, "ok": False,
                        "why": "unrunnable：占位符无法落地"})
            continue
        rc, out = runner.precondition(cmd, cwd=REPO_ROOT)
        ok = (rc == 0)
        log.append({"id": c.id, "check": cmd, "ok": ok, "rc": rc,
                    "why": "" if ok else "precondition 未通过 (rc=%d)" % rc})
        if ok:
            passed.append(c)
    return passed, log


# ---------------------------------------------------------------- run（只调 loop.py）

def _guard_no_bench(cmd):
    """硬防护：advance 绝不直接跑卡片的 bench。

    唯一的实验入口是 loop.py；出现 `ab.py` 或 `bench/` 就说明有人想绕过六道门。
    """
    s = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    low = s.lower()
    if "ab.py" in low or "/bench/" in low or "bench/ab" in low:
        raise SystemExit("advance 拒绝直接跑 bench（%s）：实验只能经 loop.py run" % s)


def loop_run_cmd(card, tag):
    """tag 可以是单个 id，也可以是 id 列表（一次 bench 多变体，共用基线测量）。

    **批量是刻意的**：loop.py 的多 tag 路径只测一次 base，各变体同口径可比。
    逐个调用会让每个变体各自重测 base，base 漂移就被算进了变体差异里。
    """
    tags = [str(tag)] if isinstance(tag, str) else [str(t) for t in tag]
    return [sys.executable, "-I", str(ENGINE_LOOP), "-C", str(card), "run", *tags]


class Runner:
    """默认执行器。测试可替换这些方法，而无需真 GPU。"""

    def __init__(self, dry_run=False, timeout=600):
        self.dry_run = dry_run
        self.timeout = timeout
        self.commands = []          # 审计：记录所有实际执行的命令

    def precondition(self, cmd, cwd):
        _guard_no_bench(cmd)
        self.commands.append(cmd)
        try:
            cp = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                                text=True, timeout=self.timeout)
            return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
        except (OSError, subprocess.SubprocessError) as e:      # noqa: BLE001
            return 1, str(e)

    def loop_run(self, card, tag):
        cmd = loop_run_cmd(card, tag)
        _guard_no_bench(cmd)
        self.commands.append(" ".join(cmd))
        cp = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                            timeout=self.timeout)
        return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


# ---------------------------------------------------------------- judge / record

# 只读 loop.py 的 decision，不重判 —— 实现在 engine/ledger.py（read_records 的同处）。
judge = _ledger.judge
confirm_record = _ledger.confirm_record


# ---------------------------------------------------------------- forge（写新实现）

# 「诊断 → 造变体 → 编译 → 测量」里的第二环。第一环是诊断（roofline / PMC / ISA，
# 由 op-roofline / op-profiling 负责），第三环是本文件的 run（六道门），
# 本段只负责**把诊断结论变成一个可测的东西**。
#
# 三条硬约束（都不是风格问题）：
#   ① advance 不 import forge_step 的实现细节，只走它导出的 forge_variant()
#      —— 造变体属于「算子知识」，不能长进编排层（ENGINE.md §一）。
#   ② 生成物只能落 runs/。要进六道门，必须写 runs/advance/overlay.yaml
#      （loop.py 只读合并它），**绝不改 candidates.yaml**。
#   ③ 体检不过就不登记 —— 一个算错的变体进了队列，会把这轮方向判死。
FORGE_OVERLAY = _ledger.FORGE_OVERLAY


def _load_forge():
    """按路径加载 scripts/forge_step.py（不 import、不靠 sys.path）。"""
    import importlib.util
    p = REPO_ROOT / "scripts" / "forge_step.py"
    if not p.is_file():
        return None, "找不到 %s" % p
    spec = importlib.util.spec_from_file_location("_forge_step", str(p))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        return None, "加载 forge_step 失败：%s: %s" % (type(e).__name__, e)
    return mod, None


# overlay 的读写实现在 engine/ledger.py：`explore --apply` 与 `advance --forge`
# 写的是同一份文件、同一个格式，两处各写一份 sync 检查迟早会不一致。
write_overlay = _ledger.write_overlay
_overlay_ids = _ledger.overlay_ids


def _route_language(card, emit=print, selector=None):
    """按 roofline 的场景签名决定「先试哪个语言」。**不猜，读不到就说读不到。**

    为什么必须有这一步：`lang_select.py` 里写着「rmsnorm: HIP 197µs vs
    triton 242µs，triton 慢 23%」，判据和数据都齐 —— 但没有任何调用方。
    结果是路线选择这个影响最大的决定，反而交给了 agent 的习惯。

    返回 {"rule":..., "langs":[...], "action":...} 或 None（无 roofline 时不猜）。
    """
    import importlib.util
    p = REPO_ROOT / "scripts" / "lang_select.py"
    if not p.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_lang_select_for_adv", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        selector = mod
    except Exception as e:                       # noqa: BLE001
        emit("  路线：加载 lang_select 失败（%s）—— 跳过，不猜" % e)
        return None
    roof = read_roofline(card)
    if not roof:
        emit("  路线：无 roofline.json —— 不猜语言，先补 roofline")
        return None
    r = selector.route_scenario(roof, roof.get("bound"))
    if not r:
        return None
    sig = r.get("signature") or {}
    emit("  场景签名：%s" % sig)
    emit("  路线：%s" % r.get("action"))
    if r.get("langs"):
        emit("  语言优先：%s" % " > ".join(r["langs"]))
    return r


def _shape_of(card):
    """读**真正会被 bench 的那个形状**的 params。读不到返回 {}（不猜）。

    **不能取 shapes.yaml 的第一个**。实测踩过：第一版就是这么写的，取到 `tiny`
    （H=512），而 bench 实际跑的是 `prefill-8k`（H=4096）。生成的变体按 H=512
    编译，跑到 bench 里直接 ValueError —— 但报错在 ab.py 里，advance 只看到
    rc=1，看起来像「六道门拒绝了」，完全指不到真因。

    正确的源有两个，按优先级：
      1. candidates.yaml 的 defaults.bench_args 里的 `--shape <name>`
      2. shapes.yaml 里 note 不含「不用于 bench」的第一个
    都找不到就返回 {}，由调用方报错 —— **不猜**。
    """
    d = card_io.load_shapes(card)
    if not d:
        return {}

    wanted = None
    try:
        cy = card_io.load_candidates(card)
        args = ((cy.get("defaults") or {}).get("bench_args") or [])
        if "--shape" in args:
            i = args.index("--shape")
            if i + 1 < len(args):
                wanted = str(args[i + 1])
    except Exception:                           # noqa: BLE001
        wanted = None

    if wanted and isinstance(d.get(wanted), dict):
        return dict(d[wanted].get("params") or {})
    for v in d.values():
        if not isinstance(v, dict) or not isinstance(v.get("params"), dict):
            continue
        if "不用于 bench" in str(v.get("note") or ""):
            continue
        return dict(v["params"])
    return {}


def forge_next(card, shape, direction=None, emit=print, sweep=3,
               template="fused-add-reduce", forge_fn=None):
    """当队列里没有更好的变体时，造一个新的。

    返回 {"forged": bool, "id":..., "reason":..., "evidence":...}。
    **不抛异常**（造不出来是正常结果，不该让整轮编排崩），但必须带原因。
    """
    if forge_fn is None:
        mod, err = _load_forge()
        if err:
            return {"forged": False, "reason": err}
        forge_fn = mod
    # 方向必须来自 surfaces.yaml，**不许发明**。第一次实现硬编码
    # direction="fused-kernel"，loop.py 立刻拒绝：「不在 surfaces.yaml 里，
    # 无法判断它属于哪类攻击面」—— 门拦得对，是生成端错了。
    # 默认取卡上第一个 open 方向；卡上没有就报错（不猜）。
    if direction is None:
        try:
            # directions 段的读法（裸 list 卡没有这一段 ⇒ []）在 card 里。
            opens = [x.get("id") for x in card_io.load_directions(card)
                     if str(x.get("status")) == "open"]
        except Exception:                       # noqa: BLE001
            opens = []
        if not opens:
            return {"forged": False,
                    "reason": "卡上没有 open 方向，无法为生成变体挂方向（不猜）"}
        direction = opens[0]
    valid = {str(s["id"]) for s in load_surfaces()}
    if valid and direction not in valid:
        return {"forged": False,
                "reason": "方向 %r 不在 surfaces.yaml（可选 %s）"
                          % (direction, sorted(valid)[:8])}
    H = int(shape.get("H") or 0)
    if not H:
        return {"forged": False, "reason": "形状里没有 H，无法定 THREADS/VEC"}
    try:
        # **一次枚举 K 组，不是选一组**。选一组等于让默认值替我决定胜负，
        # 而实测两组都合法、都编译、都不报错，只差 24% 速度 —— 这种差别
        # 只能测出来。排序只决定先试谁。
        combos = forge_fn.enumerate_params(H, max_variants=sweep)
    except Exception as e:                      # noqa: BLE001
        return {"forged": False, "reason": str(e)}

    have = _overlay_ids(card)
    entries, made, skipped = [], [], []
    for params in combos:
        vid = "auto-%s-h%d-t%d-v%d" % (template.replace("-", "")[:8], H,
                                       params["THREADS"], params["VEC"])
        if vid in have:
            skipped.append({"id": vid, "reason": "已生成过"})
            continue
        try:
            r = forge_fn.forge_variant(vid, template, params, H, None, card=card)
        except Exception as e:                  # noqa: BLE001
            # 单个组合体检不过（如 VEC=8 溢出寄存器）不是整轮失败：
            # 记下原因继续试下一个。**但绝不把它登记进队列** ——
            # 一个算错/溢出的变体进队列会把这轮方向判死。
            skipped.append({"id": vid, "params": params, "reason": str(e)[:200]})
            emit("    %s 体检不过，跳过：%s" % (vid, str(e)[:90]))
            continue
        emit("    %s 通过体检 %s" % (vid, r.get("profile")))
        made.append({"id": vid, "params": params, "evidence": r})
        entries.append({"id": vid, "direction": direction, "status": "pending",
                        "priority": len(entries) + 1,
                        "action_class": "improve-locality",
                        # expect 必须可机检（check_expect.py 拦散文）。生成变体的
                        # 预期写 0：它不是「预期能快多少」，是「还没测」。写成 -5%
                        # 会让 EV 排序把没实测的猜想排到真结论前面。
                        # **同一批 sweep 的候选 expect 全为 0、priority 按先验序**，
                        # 所以谁先跑由先验决定，谁赢由 judge() 决定。
                        "expect": "0%", "expect_basis": "生成变体，尚未实测",
                        "expect_confidence": "low",
                        "variant": vid, "forge_evidence": r})
    if not entries:
        # 汇总必须带**逐条原因**。只报「全部不合格」会让「体检拦住了」
        # 和「已经生成过」看起来一样 —— 前者是正确拦截，后者是重复劳动，
        # 处置完全不同。第一版就是只报条数，测试和人都分不出来。
        fails = [x for x in skipped if "已生成过" not in str(x.get("reason"))]
        reason = "全部组合都不合格或已生成过（%d 条跳过）" % len(skipped)
        if fails:
            reason += "；体检不过 %d 条：%s" % (
                len(fails), fails[0].get("reason", "")[:120])
        return {"forged": False, "sweep": sweep, "made": made, "skipped": skipped,
                "reason": reason}
    try:
        f = write_overlay(card, entries)
    except SystemExit as e:
        return {"forged": False, "reason": str(e)}
    emit("  forge：生成 %d/%d 组（%s）→ 由 judge 决定谁赢，不由默认值决定"
         % (len(entries), sweep, ", ".join(m["id"] for m in made)))
    return {"forged": True, "sweep": sweep, "made": made, "skipped": skipped,
            "overlay": str(f), "id": made[0]["id"]}


# ---------------------------------------------------------------- replan（v5.1 §6.3）

# §6.3 的防塌缩判据与「新 top1 必须引用上一轮 verdict」都在 engine/ledger.py。
ledger_cites = _ledger.ledger_cites
replan = _ledger.replan


# ---------------------------------------------------------------- progress 状态

# progress.json 的读写与「stage 只许前进」的判据在 engine/ledger.py。
progress_path = _ledger.progress_path
load_progress = _ledger.load_progress
save_progress = _ledger.save_progress
_advance_stage = _ledger.advance_stage


# ---------------------------------------------------------------- 编排

def run_advance(card, max_rounds, runner=None, dry_run=False, emit=print,
                write_state=None, forge_mode=False, forge_module=None,
                forge_sweep=3):
    """跑多轮。返回报告 dict。dry-run 跳过 run，且不落任何状态。"""
    card = Path(card)
    if write_state is None:
        write_state = not dry_run
    runner = runner or Runner(dry_run=dry_run)
    report = {"card": card_id(card), "dry_run": bool(dry_run), "rounds": 0,
              "status": "done", "reason": "", "gate": [], "decisions": [],
              "replan": None, "forge": None, "route": None, "progress_stage": "idle",
              "commands": runner.commands}

    if max_rounds <= 0:
        emit("max-rounds=0：立即退出，不跑任何东西。")
        report.update(status="noop", reason="max-rounds=0", progress_stage="idle")
        return report

    # at_floor 前置门。**只信本卡自己的 derive 产物**（roofline.json），
    # 不吃 atlas 卡的 floor_inputs 兜底 —— 这不是漏了统一，是刻意的：
    # atlas 的 t_actual 是**仓库级**的历史测量，不是本次编排的起点；拿它
    # 判「这张卡已经到顶」会让「先量一次基线」永远跑不起来。实测代价：
    # op-gemm-bf16 的 roofline.json 里还没有 remaining，而 atlas 卡里算出来
    # 是 0.072（< 0.10）—— 吃兜底就会把它从「没有待做候选」改成「at_floor 停机」，
    # 也就是这张卡再也进不了循环。
    fl = floor_io.resolve(card, sources=("roofline",))
    remaining = fl.remaining
    if isinstance(remaining, (int, float)) and remaining < AT_FLOOR:
        emit("at_floor：remaining=%.4f < %.2f，停在计划前。" % (remaining, AT_FLOOR))
        report.update(status="stopped", reason="at_floor", progress_stage="idle")
        return report

    if dry_run:
        # dry-run 必须是**纯函数**：不读也不写磁盘状态，否则上一次真跑留下的
        # 终态（blocked/stopped）会让本次 stage 永远无法前进，看到的也是旧进度。
        state = {"card": report["card"], "round": 0, "stage": "idle",
                 "since": time.strftime("%Y-%m-%dT%H:%M:%S"), "history": []}
    else:
        state = load_progress(card)
        # 终态只对**单次调用**终态：重新调用意味着「人已经处理过阻塞」，
        # 从 idle 重开；否则 blocked 一次之后这张卡永远推不动。
        if state.get("stage") in TERMINAL:
            state["stage"] = "idle"
        state["card"] = report["card"]
        state["since"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _advance_stage(state, "planned")

    ledger = read_records(card)
    prev_top = build_plan(card, ledger=ledger)
    consecutive_refuted = 0

    def _emit_plan(top):
        for i, c in enumerate(top, 1):
            emit("  top%d  %-22s direction=%-18s ev=%.3f  gain=%r"
                 % (i, c.id, c.direction, c.ev, (c.surface or {}).get("expected_gain")))

    # **列候选前先定实现路线**。这一步此前完全缺失：lang_select.py 早就能
    # 从 roofline 的数字判出「该先试 HIP 还是 Triton」，但没有任何代码调用它，
    # 于是 agent 按习惯从 Triton 起手（opbench 三臂对照实测：三臂都没写过 HIP）。
    # 放在 plan 之前而不是之后：路线决定「用什么写」，候选决定「改哪里」，
    # 前者是后者的前提。
    route = _route_language(card, emit=emit)
    report["route"] = route

    emit("plan（简化版：按 expected_gain 排序，不做完整 EV）")
    _emit_plan(prev_top)
    if not prev_top and forge_mode and not dry_run:
        # 队列空了不等于没路走：这正是「写新实现」该出场的时刻。
        # 造出来之后重排一次，让它照常过六道门 —— 不绕过任何一道。
        fr = forge_next(card, _shape_of(card), emit=emit,
                        sweep=forge_sweep, forge_fn=forge_module)
        report["forge"] = fr
        if fr.get("forged"):
            emit("  forge 成功：%d 组进队列 → 重排（生成的候选同样要过六道门）"
             % len(fr.get("made") or []))
            prev_top = build_plan(card, ledger=read_records(card))
            _emit_plan(prev_top)
        else:
            emit("  forge 未产出：%s" % fr.get("reason"))
    if not prev_top:
        emit("  没有 pending 候选 → blocked")
        state["stage"] = "blocked"
        if write_state:
            save_progress(card, state)
        report.update(status="blocked", reason="没有 pending 候选",
                      progress_stage=state["stage"])
        return report

    while report["rounds"] < max_rounds:
        report["rounds"] += 1
        ledger = read_records(card)
        plan = build_plan(card, ledger=ledger)
        _advance_stage(state, "planned")

        # ---- gate
        passed, glog = gate(list(plan), card, runner)
        report["gate"] = glog
        _advance_stage(state, "gated")
        emit("\nround %d gate：" % report["rounds"])
        for g in glog:
            emit("  %-22s %s %s" % (g["id"], "✓" if g["ok"] else "✗", g.get("why", "")))
        if not passed:
            emit("  全部 precondition 未通过 → blocked（不进 run）")
            state["stage"] = "blocked"
            if write_state:
                save_progress(card, state)
            report.update(status="blocked", reason="precondition 全部失败",
                          progress_stage=state["stage"])
            return report

        # ---- run（只调 loop.py；被它自己的门拒了就顺延下一个）
        before = len(read_records(card))
        chosen = None
        run_rc = None
        if dry_run:
            chosen = passed[0]
            emit("\nround %d run：[dry-run] 跳过，不调 loop.py（将跑 %s）"
                 % (report["rounds"], chosen.id))
        else:
            # **一次把整批 top-K 交给 loop.py，不是成功一个就 break**。
            # 原来每个候选单独 invoke，第一个成功就停 —— 于是 forge 一次枚举
            # 出来的 K 组只测了第 1 组，sweep 白做。实测（2026-11-24）：三组
            # 0.1790/0.1792/0.1808 ms 并排才看得出「VEC>=2 之后都收敛」，
            # 只跑第 1 组会得出「64线程/VEC8 最优」的错误结论（差 <1%，在噪声内）。
            # loop.py 本身支持多 tag（一次 bench 多变体、共用同一次基线测量），
            # 这正是它比「逐个跑」更准的地方：base 只测一次，各变体同口径。
            batch = [c.id for c in passed]
            rc, out = runner.loop_run(card, batch)
            run_rc = rc
            if rc == 0:
                # 一批都跑过了；chosen 记第一个（replan 用 top1 语义）
                chosen = passed[0]
            else:
                # rc!=0 是 loop.py 的门拒绝（登记/归因/预算/refuted/天花板）——
                # advance 不重判，只顺延逐条重试（批跑失败可能是因为某一条被拦，
                # 单独跑能定位到具体是哪条）。
                emit("  批量 run 失败（rc=%d），逐条重试以定位被拦的候选" % rc)
                for cand in passed:
                    one_rc, _ = runner.loop_run(card, [cand.id])
                    if one_rc == 0:
                        chosen = cand
                        run_rc = 0
                        break
                    emit("  loop.py run %s → rc=%d，顺延" % (cand.id, one_rc))
            if chosen is None:
                state["stage"] = "blocked"
                if write_state:
                    save_progress(card, state)
                report.update(
                    status="blocked",
                    reason="loop.py 拒绝了全部候选（rc=%r）：六道门生效，未被绕过"
                           % run_rc,
                    progress_stage=state["stage"])
                return report
            _advance_stage(state, "ran")

        # ---- judge（只读 decision）
        if dry_run:
            j = judge(card, before)
            decision = j["decision"] or "DRY_RUN"
            emit("round %d judge：[dry-run] %s（未新增记录时不编造判定）"
                 % (report["rounds"], decision))
        else:
            j = judge(card, before)
            decision = j["decision"]
            emit("\nround %d judge：decision=%s（读自 loop.py 产出）"
                 % (report["rounds"], decision))
        _advance_stage(state, "judged")

        # ---- record（确认，不代写）
        if dry_run:
            emit("round %d record：[dry-run] 不写记录" % report["rounds"])
        else:
            ok, new = confirm_record(card, before)
            emit("round %d record：%s（新增 %d 条）"
                 % (report["rounds"], "已落盘" if ok else "未落盘", len(new)))
            _advance_stage(state, "recorded")
        report["decisions"].append(decision)

        # ---- 连续 refuted 计数
        if decision in ("REJECTED_NO_GAIN", "REJECTED_ACCURACY"):
            consecutive_refuted += 1
        else:
            consecutive_refuted = 0

        # ---- replan
        # **用实际执行的候选（chosen）比，不用 plan.top1**。
        # 两者可能不同：gate 会按 precondition 顺延、loop.py 的门也会拒后顺延。
        # 用 plan.top1 会导致「明明换了候选却判 CONTINUE_SAME」（实测踩过：
        # 第 1 轮跑 triton、第 2 轮跑 tilelang，replan 却说"top1 未变"）。
        new_top = build_plan(card, ledger=read_records(card))
        prev = Top([chosen] if chosen else [], rejected=plan.rejected, ledger=ledger)
        action = replan(prev, new_top, ledger=read_records(card))
        report["replan"] = action
        _advance_stage(state, "replanned")
        emit("round %d replan：%s —— %s%s"
             % (report["rounds"], action["action"], action["reason"],
                "" if action["action"] != "CONTINUE_SWITCH"
                else "（citation=%s）" % (action.get("cited") or "无")))
        state.setdefault("history", []).append(
            {"round": report["rounds"], "decision": decision,
             "replan": action["action"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        state["round"] = report["rounds"]

        # ---- stop
        if consecutive_refuted >= MAX_CONSECUTIVE_REFUTED:
            emit("stop：连续 %d 轮 refuted。" % consecutive_refuted)
            report.update(status="stopped", reason="连续 refuted")
            break
        if report["rounds"] >= max_rounds:
            emit("stop：达到预算（max-rounds=%d）。" % max_rounds)
            report.update(status="done", reason="budget")
            break
        if action["action"] == "STOP":
            emit("stop：replan 判停 —— %s" % action["reason"])
            report.update(status="stopped", reason=action["reason"])
            break
        if action["action"] == "STALLED":
            # 抖动：不换向，保持 prev_top 再跑一轮
            emit("stalled：没拉开差距，本轮不换向，沿用当前方向。")
        elif action["action"] == "CONTINUE_SWITCH":
            prev_top = new_top
        else:                                  # CONTINUE_SAME
            prev_top = new_top

    report["progress_stage"] = state.get("stage")
    if write_state:
        save_progress(card, state)
    return report


# ---------------------------------------------------------------- selftest

def selftest():
    """不依赖 GPU / 卡片的纯逻辑自检。"""
    def C(i, ev):
        return Cand(id=i, direction=i, ev=ev)

    T = lambda *xs: Top(list(xs))
    cases = []
    cases.append(("replan 无方向 → STOP",
                  replan(T(C("a", 0.8)), T())["action"] == "STOP"))
    cases.append(("EV<0.3 → STOP",
                  replan(T(C("a", 0.8)), T(C("b", 0.2)))["action"] == "STOP"))
    cases.append(("EV 掉一半 → STOP",
                  replan(T(C("a", 0.8)), T(C("b", 0.35)))["action"] == "STOP"))
    cases.append(("top1 未变 → CONTINUE_SAME",
                  replan(T(C("a", 0.5)), T(C("a", 0.5)))["action"] == "CONTINUE_SAME"))
    cases.append(("未领先 1.5× → STALLED",
                  replan(T(C("a", 0.5)), T(C("b", 0.5), C("c", 0.4)))["action"] == "STALLED"))
    sw = replan(T(C("a", 0.5)), T(C("b", 0.8), C("c", 0.2)))
    cases.append(("正常换向 → CONTINUE_SWITCH", sw["action"] == "CONTINUE_SWITCH"))
    cases.append(("parse_gain 区间中点", abs(parse_gain("20~60%（x）") - 0.40) < 1e-9))
    cases.append(("parse_gain 上界", abs(parse_gain("≤15%（x）") - 0.15) < 1e-9))
    banned = False
    try:
        _guard_no_bench("python3 bench/ab.py --json x")
    except SystemExit:
        banned = True
    cases.append(("ab.py 被拦", banned))
    n = 0
    for name, ok in cases:
        print("[%s] %s" % ("ok" if ok else "FAIL", name))
        n += 1 if ok else 0
    print("\nselftest %s（%d/%d）" % ("PASS" if n == len(cases) else "FAIL", n, len(cases)))
    return 0 if n == len(cases) else 1


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="多轮编排唯一入口（只编排，不重实现判定）")
    ap.add_argument("--card", help="算子卡目录（含 candidates.yaml / roofline.json）")
    ap.add_argument("--max-rounds", type=int, default=3, dest="max_rounds")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="跳过 run（无需 GPU），只验证编排逻辑")
    ap.add_argument("--json", action="store_true", help="机器可读报告")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--forge", action="store_true", dest="forge_mode",
                    help="队列空了就造新变体（生成→编译→体检→写 overlay）")
    ap.add_argument("--sweep", type=int, default=3, dest="forge_sweep",
                    help="forge 一次枚举几组参数（默认 3；赢家由 judge 判，不由默认值）")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.card:
        ap.error("--card 必填（或用 --selftest）")

    # --json 时把人读日志关掉，避免污染机器可读输出
    emit = (lambda *args, **kw: None) if a.json else print
    report = run_advance(Path(a.card).resolve(), a.max_rounds,
                         dry_run=a.dry_run, emit=emit, forge_mode=a.forge_mode,
                         forge_sweep=a.forge_sweep)
    if a.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n== advance 结束 ==")
        print("status   %s（%s）" % (report["status"], report.get("reason") or "-"))
        print("rounds   %d" % report["rounds"])
        print("stage    %s" % report.get("progress_stage"))
        print("decisions %s" % (report.get("decisions") or "-"))
        fg = report.get("forge")
        if fg:
            print("forge    %s（%s）" % ("是" if fg.get("forged") else "否",
                                        fg.get("reason") or "-"))
    return 0 if report["status"] in ("done", "noop", "stopped") else 2


if __name__ == "__main__":
    sys.exit(main())
