#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""variant_forge —— 从现有 Triton 模板生成**参数化变体**（v6 的 C2）。

为什么只做 Triton
-----------------
Triton 的 `tl.constexpr` 能干净地覆盖 BLOCK / BM / BN / BK / num_warps：
它们是 launch 时传进去的编译期常量，改值只改 launch 行或函数默认值，
kernel 结构一字不动。

HIP 做不到。tile 大小会影响 LDS 声明、循环边界、寄存器分配——
改一个 `BLOCK_M` 要重写 kernel 结构，不是宏能解决的。
所以 HIP 侧的候选在 candidates.yaml 里标 `forge: manual`，
由 agent 写代码；本工具**明确拒绝**自动生成（不产出假变体）。

本工具补的缺口
--------------
`design_space` 能枚举参数、`loop` 能跑候选，但两者之间缺一步：
**把候选变成能跑的源码**。没有这一步，agent 只能手改模板——
改没改到、改对没改对，全靠人眼。

验收为什么不是 isa_probe
------------------------
Triton 不经 hipcc，产出是 `.py`，`isa_probe` 吃不了。
所以验收改成机器判据：生成后**重新解析 AST，顺着「launch 实参 /
run() 默认值」把锚点上的值读回来**，必须等于请求值。
不是「字符串里搜一下」——那会放过「改了注释、没改调用」的假阳性。

用法
----
    # 直接对内置模板打补丁
    python3 -I scripts/variant_forge.py --parent reduce-lastdim \\
        --patch BLOCK=2048,num_warps=8 --out /tmp/v.py

    # 按算子卡的候选生成；candidates.yaml 里该候选需声明 forge 块
    python3 -I scripts/variant_forge.py --card op-rmsnorm-fused --cand bm64 --dry-run

    # 自检（不依赖 GPU / torch）
    python3 -I scripts/variant_forge.py --selftest

候选的 forge 声明格式（写在 candidates.yaml 的候选条目下）
----------------------------------------------------------
    forge:
      parent: reduce-lastdim
      patch: {BLOCK: 2048, num_warps: 8}

    # HIP 侧，不自动生成：
    forge: manual

失败必须明确
------------
未知参数名 / 值越界 / 找不到锚点，全部报错并给可用集合。
**绝不静默跳过**——静默跳过会产出「看起来改了但没改」的变体，
那种变体跑出来的数会污染整轮优化结论。
"""
import argparse
import ast
import difflib
import importlib.util
import io
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAFFOLD = REPO / "scripts/kernel_scaffold.py"


class ForgeError(Exception):
    """所有可预期的失败：未知参数 / 越界 / 锚点缺失 / 生成后校验失败。"""


# ---------------------------------------------------------------- 模板与参数

@dataclass(frozen=True)
class Template:
    name: str
    source: str
    params: dict          # 参数名 -> 校验规则（由 discover_params 实测得出）


# 允许 forge 的参数名 → 校验规则。**不是「模板里出现的任何名字都可改」**：
# 只有 kernel 签名里的这些 constexpr / launch 关键字才进可选集合；
# 未登记的名字（如 H / eps，改它们不是参数化而是改语义）一律报错。
# 具体某个模板认哪些，由 discover_params 实测 + 本表求交得出。
REGISTRY = {
    "BLOCK": {
        "kind": "constexpr", "min": 16, "max": 1 << 20,
        "power_of_two": True,
        "help": "tl.arange(0, BLOCK) 要求 2 的幂；本模板还要求 BLOCK == 运行时 H",
    },
    "BM": {
        "kind": "constexpr", "min": 16, "max": 1 << 16,
        "power_of_two": True, "help": "tile 行数，需 2 的幂",
    },
    "BN": {
        "kind": "constexpr", "min": 16, "max": 1 << 16,
        "power_of_two": True, "help": "tile 列数，需 2 的幂",
    },
    "BK": {
        "kind": "constexpr", "min": 8, "max": 1 << 12,
        "power_of_two": True, "help": "tile K 维，需 2 的幂",
    },
    "num_warps": {
        "kind": "launch_kw", "min": 1, "max": 32,
        "power_of_two": True, "help": "Triton launch 的 warp 数；必须 2 的幂",
    },
    "num_stages": {
        "kind": "launch_kw", "min": 1, "max": 8,
        "power_of_two": False, "help": "Triton launch 的流水级数",
    },
}

# 别名（示例命令里的 warps=4 / bm64 等）
ALIASES = {
    "warps": "num_warps",
    "numwarps": "num_warps",
    "stages": "num_stages",
    "block": "BLOCK",
    "bm": "BM", "bn": "BN", "bk": "BK",
}


def discover_params(source):
    """扫描模板，返回**实际可改**的参数集合。

    只有同时满足「是 REGISTRY 里登记的名字」+「在模板里有锚点」才收：
      constexpr：出现在 @triton.jit kernel 签名里
      launch_kw：出现在 launch 调用关键字里
    这样 reduce 模板只认 BLOCK / num_warps，不会让 matmul 的 BM 混进来。
    """
    try:
        tree = ast.parse(source)
        kernel = _kernel_def(tree)
        run = _run_def(tree)
        call = _launch_call(run, kernel.name)
    except ForgeError:
        return {}   # 没有锚点的模板不接受任何 patch（后续会给具体错）
    sig = {a.arg for a in kernel.args.args} | {a.arg for a in kernel.args.kwonlyargs}
    kws = {k.arg for k in call.keywords if k.arg}
    out = {}
    for name, spec in REGISTRY.items():
        kind = spec["kind"]
        if kind == "constexpr" and name in sig:
            out[name] = dict(spec)
        elif kind == "launch_kw" and name in kws:
            out[name] = dict(spec)
    return out


def load_scaffold():
    """按文件路径加载 kernel_scaffold，不靠 sys.path / 模块名。"""
    if not SCAFFOLD.is_file():
        raise ForgeError("找不到模板源 %s" % SCAFFOLD)
    spec = importlib.util.spec_from_file_location("_kernel_scaffold", str(SCAFFOLD))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_template(name):
    """模板名 → Template。也接受 `.py` 路径（用于测试 / 外部模板）。"""
    if not name:
        raise ForgeError("缺 --parent（模板名，如 reduce-lastdim）")
    p = Path(name)
    if p.is_file():
        src = p.read_text(encoding="utf-8")
        return Template(p.stem, src, discover_params(src))
    ks = load_scaffold()
    entry = ks.TEMPLATES.get(str(name))
    if entry is None:
        raise ForgeError("未知模板 %r（可选 %s）" % (name, sorted(ks.TEMPLATES)))
    src = entry.get("triton")
    if not src:
        raise ForgeError("模板 %s 没有 Triton 版本；HIP 侧请标 forge: manual"
                         % (name))
    params = discover_params(src)
    if not params:
        raise ForgeError("模板 %s 没有可 forge 的锚点（kernel 签名 / launch "
                         "关键字里没有登记的参数）" % name)
    return Template(str(name), src, params)


def _supported(template):
    names = sorted(template.params)
    if not names:
        return "<无>（模板里没有登记过的可 forge 参数）"
    extras = [k for k, v in ALIASES.items() if v in template.params]
    s = ", ".join(names)
    return s + ("（别名 %s）" % ", ".join(sorted(extras)) if extras else "")


def _validate_value(name, value, spec):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForgeError("%s 必须是整数，收到 %r" % (name, value))
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and value < lo:
        raise ForgeError("%s=%d 越界：必须 >= %d（%s）"
                         % (name, value, lo, spec.get("help", "")))
    if hi is not None and value > hi:
        raise ForgeError("%s=%d 越界：必须 <= %d（%s）"
                         % (name, value, hi, spec.get("help", "")))
    if spec.get("power_of_two") and (value <= 0 or value & (value - 1)):
        raise ForgeError("%s=%d 不是 2 的幂（%s）"
                         % (name, value, spec.get("help", "")))
    return value


def normalize_patches(patches, template):
    """别名归一 + 未知参数报错 + 值校验。返回有序 dict。"""
    if not isinstance(patches, dict) or not patches:
        raise ForgeError("patch 为空：至少给一个参数，如 BLOCK=2048")
    out = {}
    for raw_name, raw_val in patches.items():
        if not isinstance(raw_name, str):
            raise ForgeError("参数名必须是字符串，收到 %r" % (raw_name,))
        name = ALIASES.get(raw_name.strip(), raw_name.strip())
        if name not in template.params:
            raise ForgeError(
                "模板 %s 不认识参数 %r；支持 %s"
                % (template.name, raw_name, _supported(template)))
        if name in out:
            raise ForgeError("参数 %s 重复出现（别名归一后冲突）" % name)
        out[name] = _validate_value(name, raw_val, template.params[name])
    return out


def parse_patch_arg(text):
    """'BLOCK=2048,num_warps=8' → {'BLOCK': 2048, 'num_warps': 8}。"""
    if not text:
        raise ForgeError("--patch 为空")
    out = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ForgeError("--patch 项 %r 不是 KEY=VALUE" % part)
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise ForgeError("--patch 项 %r 缺参数名或值" % part)
        try:
            iv = int(v, 10)
        except ValueError:
            raise ForgeError("--patch %s 的值 %r 不是整数" % (k, v))
        if k in out:
            raise ForgeError("--patch 里参数 %s 重复" % k)
        out[k] = iv
    return out


def _as_int(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    return None


# ---------------------------------------------------------------- AST 锚点

def _kernel_def(tree):
    """找 @triton.jit 修饰的顶层 kernel。"""
    found = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if (isinstance(d, ast.Attribute) and d.attr == "jit") or \
                    (isinstance(d, ast.Name) and d.id == "jit"):
                found.append(node)
                break
    if not found:
        raise ForgeError("模板里找不到 @triton.jit kernel —— 没有可锚定的 kernel")
    if len(found) > 1:
        raise ForgeError("模板里有 %d 个 @triton.jit kernel，无法确定改哪个"
                         % len(found))
    return found[0]


def _run_def(tree):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    raise ForgeError("模板里找不到 def run(...) —— 无法定位 launch 入口")


def _launch_call(run_node, kernel_name):
    """run() 里形如 kernel[(...)](...) 的 launch 调用，必须恰好一个。"""
    calls = []
    for node in ast.walk(run_node):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == kernel_name):
            calls.append(node)
    if not calls:
        raise ForgeError("run() 里找不到 %s[(...)](...) 的 launch 调用 —— "
                         "锚点缺失，拒绝生成" % kernel_name)
    if len(calls) > 1:
        raise ForgeError("run() 里有 %d 处 %s 的 launch 调用，无法确定改哪个"
                         % (len(calls), kernel_name))
    return calls[0]


def _param_index(func, name):
    names = [a.arg for a in func.args.args]
    return names.index(name) if name in names else None


def _run_default_node(run_node, name):
    """run() 签名里 name 的默认值节点；没有返回 None。"""
    args = run_node.args
    positional = args.args
    defaults = args.defaults
    if defaults:
        offset = len(positional) - len(defaults)
        for a, d in zip(positional[offset:], defaults):
            if a.arg == name:
                return d
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        if a.arg == name:
            return d
    return None


def _lines(src):
    return src.splitlines(keepends=True)


def _line_start_offsets(src):
    offs, pos = [], 0
    for line in _lines(src):
        offs.append(pos)
        pos += len(line)
    return offs


def _char_col(line_text, byte_col):
    """AST 的 col_offset 是 UTF-8 字节偏移；转成字符偏移。"""
    prefix = line_text.encode("utf-8")[:byte_col]
    try:
        return len(prefix.decode("utf-8"))
    except UnicodeDecodeError:
        return byte_col


def _span(src, node):
    lines = _lines(src)
    offs = _line_start_offsets(src)
    sl, el = lines[node.lineno - 1], lines[node.end_lineno - 1]
    s = offs[node.lineno - 1] + _char_col(sl, node.col_offset)
    e = offs[node.end_lineno - 1] + _char_col(el, node.end_col_offset)
    return s, e


def _call_arg(call, param_name, index):
    """launch 调用里某个 kernel 参数对应的实参节点。

    优先按关键字（BLOCK=...），否则按 kernel 签名里的位置。
    返回 (node, kind)；找不到返回 (None, None)。
    """
    for kw in call.keywords:
        if kw.arg == param_name:
            return kw.value, "kwarg"
    if index is not None and index < len(call.args):
        return call.args[index], "positional"
    return None, None


@dataclass
class Edit:
    param: str
    line: int
    old: str
    new: str
    anchor: str
    start: int = field(repr=False, default=0)
    end: int = field(repr=False, default=0)


def _plan_edits(src, template, patches):
    """为每个参数找锚点，返回 Edit 列表。任一锚点缺失 → ForgeError。"""
    tree = ast.parse(src)
    kernel = _kernel_def(tree)
    run = _run_def(tree)
    call = _launch_call(run, kernel.name)
    edits = []

    for name, value in patches.items():
        spec = template.params[name]

        if spec["kind"] == "constexpr":
            # 锚点 = launch 调用里该 kernel 参数对应的实参。
            # 位置参数按 kernel 签名定位（本模板 BLOCK 是最后一个位置参数）。
            idx = _param_index(kernel, name)
            node, kind = _call_arg(call, name, idx)
            if node is None:
                raise ForgeError(
                    "锚点缺失：kernel 签名里有 %s，但 launch 调用没有对应的"
                    "位置/关键字实参 —— 拒绝生成（否则会产出没改的假变体）"
                    % name)
            start, end = _span(src, node)
            old = src[start:end]
            new = old if old.strip() == str(value) else str(value)
            edits.append(Edit(name, node.lineno, old, new,
                              "%s launch %s实参" % (
                                  kernel.name,
                                  "关键字" if kind == "kwarg" else "位置"),
                              start, end))

        elif spec["kind"] == "launch_kw":
            kw = next((k for k in call.keywords if k.arg == name), None)
            if kw is None:
                raise ForgeError(
                    "锚点缺失：run() 的 launch 调用没有 %s=... —— "
                    "改 run() 默认值不会生效，拒绝生成" % name)
            vnode = kw.value
            if isinstance(vnode, ast.Name):
                default = _run_default_node(run, vnode.id)
                if default is None:
                    raise ForgeError(
                        "锚点无效：launch 传的是 %s=%s，但 run() 里没有"
                        "这个参数 —— 改默认值不会生效" % (name, vnode.id))
                target = default
                anchor = "def run() 的 %s 默认值（经 launch 关键字生效）" % vnode.id
            elif _as_int(vnode) is not None:
                target = vnode
                anchor = "launch 的 %s 字面量" % name
            else:
                raise ForgeError(
                    "锚点无效：launch 的 %s 不是简单常量或 run() 参数，"
                    "无法保证改动能生效" % name)
            start, end = _span(src, target)
            old = src[start:end]
            new = old if old.strip() == str(value) else str(value)
            edits.append(Edit(name, target.lineno, old, new, anchor, start, end))

        else:
            raise ForgeError("内部错误：未知参数类型 %r" % spec["kind"])

    edits.sort(key=lambda e: e.start)
    for a, b in zip(edits, edits[1:]):
        if a.end > b.start:
            raise ForgeError("内部错误：参数 %s 与 %s 的替换区间重叠"
                             % (a.param, b.param))
    if not edits:
        raise ForgeError("没有任何锚点被改动 —— 拒绝生成空变体")
    return edits


def _apply_edits(src, edits):
    out = src
    for e in sorted(edits, key=lambda x: x.start, reverse=True):
        out = out[:e.start] + e.new + out[e.end:]
    return out


# ---------------------------------------------------------------- 生成后校验

def _effective_value(src, name):
    """单个参数在锚点上的实际值；解析不出返回 None。

    constexpr：launch 里该 kernel 参数的位置/关键字实参；
    launch_kw：launch 关键字的值，或它引用的 run() 默认值。
    """
    tree = ast.parse(src)
    kernel = _kernel_def(tree)
    run = _run_def(tree)
    call = _launch_call(run, kernel.name)
    kind = REGISTRY.get(name, {}).get("kind")
    if kind == "constexpr":
        node, _ = _call_arg(call, name, _param_index(kernel, name))
        return _as_int(node) if node is not None else None
    if kind == "launch_kw":
        kw = next((k for k in call.keywords if k.arg == name), None)
        if kw is None:
            return None
        if _as_int(kw.value) is not None:
            return _as_int(kw.value)
        if isinstance(kw.value, ast.Name):
            return _as_int(_run_default_node(run, kw.value.id))
        return None
    return None


def _effective_params(src, names=None):
    """把锚点上的**实际值**读回来，非字面量返回 None。

    这是本工具的核心验收：不是搜字符串，而是解析 AST 后顺着
    「launch 实参 / run 默认值」把值解出来。
    """
    names = list(names) if names else ["BLOCK", "num_warps"]
    return {n: _effective_value(src, n) for n in names}


def _effective_text(src, name):
    try:
        v = _effective_value(src, name)
    except ForgeError:
        return "<无法解析>"
    return "<非字面量>" if v is None else str(v)


def verify_patch(src, patches, template):
    """生成后机器校验：值必须真的接在锚点上，且出现在源码文本里。

    失败抛 ForgeError。**这是防「看起来改了但没改」的唯一一道闸。**
    """
    norm = normalize_patches(patches, template)
    try:
        compile(src, "<variant>", "exec")
    except SyntaxError as e:
        raise ForgeError("生成物语法错误：%s" % e)
    eff = _effective_params(src, norm.keys())
    problems = []
    for name, value in norm.items():
        got = eff.get(name)
        if got != value:
            problems.append("%s 请求 %d，锚点上实际是 %s"
                            % (name, value, _effective_text(src, name)))
        if str(value) not in src:
            problems.append("%s=%d 没有出现在源码文本里" % (name, value))
    if problems:
        raise ForgeError("生成后校验失败（拒绝产出假变体）：" + "；".join(problems))
    return eff


# ---------------------------------------------------------------- 主流程

@dataclass
class ForgeResult:
    parent: str
    patches: dict
    source: str
    diff: str
    edits: list
    changed: bool
    effective: dict
    warnings: list


def forge(source, patches, template=None, parent_name="parent"):
    """核心入口：模板源码 + patch → 变体源码（带 diff 与校验）。"""
    if template is None:
        template = Template(parent_name, source, discover_params(source))
    norm = normalize_patches(patches, template)
    edits = _plan_edits(source, template, norm)
    new_src = _apply_edits(source, edits)
    ast.parse(new_src)                                  # 语法闸
    effective = verify_patch(new_src, norm, template)   # 接线闸

    parent_eff = _effective_params(source, norm.keys())
    warnings = []
    noop = [n for n, v in norm.items() if parent_eff.get(n) == v]
    if noop:
        warnings.append(
            "%s 的目标值与 parent 相同，生成物在该锚点上无差异"
            % ", ".join(noop))
    if "BLOCK" in norm:
        warnings.append(
            "本模板无 mask：BLOCK 只改 launch 实参，必须等于运行时 H，"
            "run() 里的 H 不受影响")

    diff = "".join(difflib.unified_diff(
        _lines(source), _lines(new_src),
        fromfile="%s (parent)" % parent_name,
        tofile="%s + %s" % (parent_name, ", ".join(
            "%s=%d" % (k, v) for k, v in norm.items())),
        n=2))
    return ForgeResult(
        parent=parent_name, patches=norm, source=new_src, diff=diff,
        edits=edits, changed=(new_src != source), effective=effective,
        warnings=warnings)


def render_report(res, dry_run=False, out=None):
    lines = []
    changed = [e for e in res.edits if e.old != e.new]
    if changed:
        lines.append("改动 %d 行：" % len(changed))
        for e in changed:
            lines.append("  L%-4d %-10s %s → %s   [%s]"
                         % (e.line, e.param,
                            e.old.replace("\n", "\\n"),
                            e.new.replace("\n", "\\n"), e.anchor))
    else:
        lines.append("改动 0 行（目标值已等于 parent 的锚点值）")
    lines.append("")
    lines.append(res.diff.rstrip("\n") if res.diff
                 else "（无 diff：生成物与 parent 内容相同）")
    for w in res.warnings:
        lines.append("⚠ " + w)
    lines.append("")
    lines.append("接线校验：" + ", ".join(
        "%s=%s" % (k, v) for k, v in res.effective.items()))
    if dry_run:
        lines.append("[dry-run] 未写文件（目标 %s）"
                     % (out if out else "<未指定 --out>"))
    return "\n".join(lines)


# ---------------------------------------------------------------- 算子卡

def _card_dir(ref):
    p = Path(ref)
    if p.is_dir():
        return p
    for cand in (REPO / "Plugins/ops/skills" / str(ref),
                 REPO / "Plugins/ops/skills" / ("op-%s" % ref)):
        if cand.is_dir():
            return cand
    raise ForgeError("找不到算子卡 %r（试过 %s 与 op- 前缀）" % (ref, ref))


def resolve_candidate(card_ref, cand_id):
    """从卡的 candidates.yaml 取 (candidate, forge_spec)。

    `forge: manual` → 明确拒绝（HIP 侧由 agent 写，不产出假变体）。
    """
    import yaml  # 延迟导入：--parent / --selftest 路径不依赖 yaml
    card = _card_dir(card_ref)
    p = card / "candidates.yaml"
    if not p.is_file():
        raise ForgeError("%s 没有 candidates.yaml" % card)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cands = [c for c in (data.get("candidates") or []) if isinstance(c, dict)]
    for c in cands:
        if c.get("id") != cand_id:
            continue
        spec = c.get("forge")
        if spec is None:
            raise ForgeError(
                "候选 %s 没有声明 forge。Triton 自动生成请加：\n"
                "  forge: {parent: reduce-lastdim, patch: {BLOCK: 2048}}\n"
                "HIP 侧手写请标：forge: manual" % cand_id)
        if isinstance(spec, str):
            if spec.strip().lower() == "manual":
                raise ForgeError(
                    "候选 %s 标了 forge: manual —— HIP 侧 tile 改动会动 "
                    "LDS 声明/循环边界/寄存器分配，不能靠宏替换自动生成；"
                    "请由 agent 手写代码，本工具拒绝产出假变体。" % cand_id)
            raise ForgeError("候选 %s 的 forge 字符串 %r 不认识"
                             "（只支持 manual 或 {parent, patch}）"
                             % (cand_id, spec))
        if not isinstance(spec, dict):
            raise ForgeError("候选 %s 的 forge 块必须是 dict，收到 %r"
                             % (cand_id, type(spec).__name__))
        return c, spec
    ids = [str(c.get("id")) for c in cands]
    raise ForgeError(
        "候选 %r 不在 %s/candidates.yaml；现有候选：%s。\n"
        "若只想 forge 一个参数变体，可不走卡片："
        "--parent reduce-lastdim --patch num_warps=8"
        % (cand_id, card.name, ", ".join(ids) or "<空>"))


def forge_candidate(card_ref, cand_id, dry_run=False, out=None):
    cand, spec = resolve_candidate(card_ref, cand_id)
    parent = spec.get("parent") or "reduce-lastdim"
    patch = spec.get("patch")
    if isinstance(patch, str):
        raw = parse_patch_arg(patch)
    elif isinstance(patch, dict):
        raw = dict(patch)
    else:
        raise ForgeError("候选 %s 的 forge.patch 必须是 dict 或字符串"
                         % cand_id)
    tpl = load_template(parent)
    res = forge(tpl.source, raw, template=tpl, parent_name=tpl.name)
    if out and not dry_run:
        _write(out, res.source)
    return res


def _write(out, source):
    p = Path(out)
    if p.parent and not p.parent.is_dir():
        raise ForgeError("输出目录不存在：%s" % p.parent)
    p.write_text(source, encoding="utf-8")
    print("已写入 %s（%d 行）" % (p, len(source.splitlines())))


# ---------------------------------------------------------------- selftest

def selftest():
    tpl = load_template("reduce-lastdim")
    parent = tpl.source

    a = forge(parent, {"num_warps": 8}, tpl, "reduce-lastdim")
    assert "num_warps=8" in a.source, "num_warps=8 未出现在源码里"
    assert a.effective["num_warps"] == 8, a.effective
    print("[ok] num_warps=8 接进 run() 默认值，锚点读回 8")

    b = forge(parent, {"num_warps": 4}, tpl, "reduce-lastdim")
    assert "num_warps=4" in b.source
    assert a.source != b.source, "两个 patch 生成物相同 —— 参数没真的接进去"
    print("[ok] num_warps=4 与 num_warps=8 生成物不同（反向）")

    c = forge(parent, {"BLOCK": 2048}, tpl, "reduce-lastdim")
    assert c.effective["BLOCK"] == 2048, c.effective
    assert "2048" in c.source
    print("[ok] BLOCK=2048 接进 launch 最后一个位置参数，锚点读回 2048")

    d = forge(parent, {"BLOCK": 2048, "num_warps": 8}, tpl, "reduce-lastdim")
    assert d.effective["BLOCK"] == 2048 and d.effective["num_warps"] == 8
    assert len([e for e in d.edits if e.old != e.new]) == 2
    print("[ok] 两个 patch 同时生效，改动两行")

    for r in (a, b, c, d):
        compile(r.source, "<variant>", "exec")
    print("[ok] 生成物全部通过 compile() 语法闸")

    try:
        forge(parent, {"BM": 128}, tpl, "reduce-lastdim")
    except ForgeError as e:
        assert "BLOCK" in str(e) and "num_warps" in str(e), e
        print("[ok] 反向：未知参数报错并列出支持集合")
    else:
        raise AssertionError("未知参数 BM 竟然没报错")

    try:
        forge(parent, {"num_warps": 0}, tpl, "reduce-lastdim")
    except ForgeError as e:
        assert "越界" in str(e), e
        print("[ok] 反向：num_warps=0 报错")
    else:
        raise AssertionError("num_warps=0 竟然没报错")

    try:
        forge(parent, {"BLOCK": 1000}, tpl, "reduce-lastdim")
    except ForgeError as e:
        assert "2 的幂" in str(e), e
        print("[ok] 反向：BLOCK=1000（非 2 的幂）报错")
    else:
        raise AssertionError("BLOCK=1000 竟然没报错")

    broken = parent.replace("def run(x, w, eps=1e-6, num_warps=4):",
                            "def run(x, w, eps=1e-6):")
    assert broken != parent
    try:
        forge(broken, {"num_warps": 8}, tpl, "broken")
    except ForgeError as e:
        assert "锚点" in str(e), e
        print("[ok] 反向：坏模板锚点缺失报错（不静默跳过）")
    else:
        raise AssertionError("坏模板竟然没报错")

    try:
        verify_patch(parent, {"num_warps": 8}, tpl)
    except ForgeError:
        print("[ok] 反向：校验器本身被测 —— 未改动的 parent 被 verify_patch 拒绝")
    else:
        raise AssertionError("verify_patch 竟然接受未改动的 parent")

    out = Path(tempfile.mkdtemp()) / "v.py"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--parent", "reduce-lastdim", "--patch", "num_warps=8",
                   "--out", str(out), "--dry-run"])
    text = buf.getvalue()
    assert rc == 0 and not out.exists(), "dry-run 写了文件"
    assert "num_warps=8" in text and "+" in text, text
    print("[ok] 反向：--dry-run 不写文件，且 diff 含被改的行")

    print("\nselftest PASS（11 项，含 6 项反向）")
    return 0


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="从 Triton 模板生成参数化变体（HIP 侧请 forge: manual）")
    ap.add_argument("--parent", help="模板名（如 reduce-lastdim）或 .py 路径")
    ap.add_argument("--patch", help="KEY=VAL[,KEY=VAL]，如 BLOCK=2048,num_warps=8")
    ap.add_argument("--out", help="输出路径（--dry-run 时不写）")
    ap.add_argument("--card", help="算子卡 id 或目录（配合 --cand）")
    ap.add_argument("--cand", help="candidates.yaml 里的候选 id")
    ap.add_argument("--dry-run", action="store_true", help="只打印 diff，不写文件")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    try:
        if a.cand and not a.card:
            raise ForgeError("--cand 需要 --card")
        if a.card and not a.cand:
            raise ForgeError("--card 需要 --cand")
        if a.cand:
            if a.parent or a.patch:
                raise ForgeError("--cand 与 --parent/--patch 不能同时给"
                                 "（patch 由卡片候选的 forge 块提供）")
            res = forge_candidate(a.card, a.cand, a.dry_run, a.out)
            print("候选 %s → 模板 %s" % (a.cand, res.parent))
        else:
            if not a.parent:
                ap.error("给 --parent + --patch，或 --card + --cand，或 --selftest")
            if not a.patch:
                ap.error("--parent 需要 --patch（如 BLOCK=2048,num_warps=8）")
            tpl = load_template(a.parent)
            res = forge(tpl.source, parse_patch_arg(a.patch),
                        template=tpl, parent_name=tpl.name)
            if a.out and not a.dry_run:
                _write(a.out, res.source)
        print(render_report(res, dry_run=a.dry_run, out=a.out))
        return 0
    except ForgeError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
