#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ops 域文档的真值门：SKILL 里写下的路径必须真的存在。

为什么要有这一层：Agent 是**按文档执行**的，一条「看着对、跑起来 404」的命令
会烧掉一整轮；而这种漂移是静默的（文件合并/改名后没人扫文档）。本测试把
「文档里出现的仓内路径」变成机器可验证的事实，CI 里跑（`.gitlab-ci.yml` 的
`unittest discover -s Plugins/ops/tests`）。

检查四类：
  ① 路径 token 可解析——仓外源码走 `EXTERNAL_PREFIXES` 白名单（必须写字面前缀，
     不能只按文件名放过），卡内示意路径走 `ILLUSTRATIVE`（逐条写理由）；
  ② `contract.yaml` 的 gates：`template:` 必须是 `scripts/opcheck.py` 的子命令；
     `script:` 必须存在，**除非**该门显式 `required: false` 且带 `skip_reason`
     （「声明了但还不能跑」是合法的，「声明能跑但文件没了」不是）；
  ③ `data/ops/registry.json` ↔ 卡目录 ↔ `ops/SKILL.md` 索引 三方一致；
  ④ 指向用户知识库的路径必须写全 `~/.config/dcu-toolkit/kb/…`——只写 `kb/op-opt/x.md`
     时 Agent 会按包内相对路径去找，必然找不到。

刻意不做：不校验语义、不连网、不碰 GPU。纯静态，无卡环境也能跑。
"""
import argparse
import glob
import json
import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SKILLS = REPO / "Plugins" / "ops" / "skills"
OPS_ENTRY = REPO / "Plugins" / "ops" / "SKILL.md"
REGISTRY = REPO / "data" / "ops" / "registry.json"

sys.path.insert(0, str(REPO / "scripts"))
from opverify import parse_yaml  # noqa: E402

CODE_SPAN = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
PATHISH = re.compile(r"^[\w./<>~$-]+\.(?:py|ya?ml|md|hip|sh|json)$")
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")
TOKEN_SPLIT = re.compile(r"[\s,;()\[\]\"'=|]+")

# 仓外源码/示例：这些文件名在本仓必然找不到，按**前缀**放过（不按文件名，
# 否则任何同名文件都会被误放过）。
EXTERNAL_PREFIXES = (
    "perf/",            # tilelang-hygon 的 perf/ 树
    "examples/",        # tilelang 示例
    "tilelang/",        # tilelang Python 包内部
    "src/",             # tilelang C++ 源码
)
EXTERNAL_BASENAMES = {
    "sparse_mla_fwd.py",              # tilelang-hygon 生产 kernel（外部仓库）
    "result.json", "manifest.json",   # 运行产物，跑完才有
}
# 卡内示意路径 / 占位：逐条写理由，别用通配符。
ILLUSTRATIVE = {
    "dev/x.hip": "op-hip-compile 的命令示例，不是本仓文件",
    "path/to/generated.hip": "同上",
}
# 卡 schema 的固定文件名：文档里当「概念名」用（「见 ledger.md」「填 contract.yaml」），
# 不指某个具体路径。它们是否该存在由 test_card_has_required_files 管。
SCHEMA_BASENAMES = {
    "SKILL.md", "contract.yaml", "variants.yaml", "shapes.yaml", "_common.py",
    "candidates.yaml", "ledger.md", "bench.py", "cards.md",
}
# 这几类登记在卡内，但别处也常被当范例引用（`bench/bench.py`）；当前卡没有、
# 兄弟卡有时跳过，避免把「举例」误判成死链。
CARD_RELATIVE_DIRS = ("bench/", "dev/", "references/", "models/", "runs/", "tests/")


def _index_names():
    """仓内所有可用文件名 → 路径列表（用于「同名文件是否存在于别处」判定）。"""
    out = {}
    for base in (SKILLS, REPO / "scripts", REPO / "data"):
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d != "__pycache__"]
            for f in fn:
                out.setdefault(f, []).append(os.path.join(dp, f))
    return out


NAMES = _index_names()


def _skill_dirs():
    return sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").is_file())


def _path_tokens(text):
    """正文里的路径 token（inline code + 代码块），去掉 `:行号` 后缀。"""
    chunks = CODE_SPAN.findall(text)
    for blk in FENCE.findall(text):
        chunks.extend(blk.splitlines())
    for chunk in chunks:
        for tok in TOKEN_SPLIT.split(chunk):
            tok = tok.strip().rstrip(".,:;")
            if not tok or "/" not in tok:
                continue
            tok = LINE_SUFFIX.sub("", tok)
            if PATHISH.match(tok):
                yield tok


def _resolves(tok, skill_dir):
    for base in (skill_dir, SKILLS, REPO, REPO / "scripts"):
        if (base / tok).exists():
            return True
    return False


def _exists_in_sibling_tree(tok):
    """卡内相对路径：同名文件是否在任一卡里存在（举例用法）。"""
    base = os.path.basename(tok)
    return any(os.path.basename(p) == base for p in NAMES.get(base, []))


def classify(tok, skill_dir):
    """返回 None（放过）或失败原因。"""
    if tok.startswith("/") or tok.startswith("$") or "<" in tok or "$" in tok:
        return None
    if tok.startswith("~"):
        if "/kb/" in tok and not tok.startswith("~/.config/dcu-toolkit/kb/"):
            return "用户知识库路径必须写全 ~/.config/dcu-toolkit/kb/…"
        return None
    if tok in ILLUSTRATIVE:
        return None
    if tok.startswith(EXTERNAL_PREFIXES):
        return None
    if re.match(r"^(?:\.\./)*kb/(?:op-opt|issue|solution)/", tok):
        return "用户知识库路径必须写全 ~/.config/dcu-toolkit/kb/…"
    base = os.path.basename(tok)
    if "/" not in tok:
        if base in SCHEMA_BASENAMES or base in EXTERNAL_BASENAMES:
            return None
        return None if NAMES.get(base) else "仓内找不到同名文件"
    if _resolves(tok, skill_dir):
        return None
    if tok.startswith(CARD_RELATIVE_DIRS) and _exists_in_sibling_tree(tok):
        return None
    if base in EXTERNAL_BASENAMES or tok.startswith(EXTERNAL_PREFIXES):
        return None
    sibling = SKILLS / tok.split("/")[0]
    if sibling.is_dir():
        return "指向兄弟 skill 的路径不存在"
    return "路径不存在（也没有同名文件可回落）"


def _opcheck_subcommands():
    sys.path.insert(0, str(REPO / "scripts"))
    import opcheck
    parser = opcheck.build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


class TestSkillPaths(unittest.TestCase):
    def test_every_path_token_resolves(self):
        bad = []
        targets = [(p / "SKILL.md", p) for p in _skill_dirs()] + [(OPS_ENTRY, REPO / "Plugins" / "ops")]
        for skill_md, skill_dir in targets:
            text = skill_md.read_text(encoding="utf-8")
            for tok in sorted(set(_path_tokens(text))):
                why = classify(tok, skill_dir)
                if why:
                    bad.append("%s -> `%s`: %s" % (skill_md.relative_to(REPO), tok, why))
        self.assertEqual(bad, [], "文档里写了不存在的路径：\n" + "\n".join(bad))

    def test_card_skill_md_stays_readable(self):
        """卡面是 agent 每次进卡都要读的上下文，不是仓库文档。

        上限 240 = 现状最长（`op-contract` 235 行，方法学卡）×约 1.02，只做「别继续长」的护栏：
        超了先删重复表述；真删不动就拆进 `references/`，别让必读段无界膨胀。
        """
        LIMIT = 240
        bad = ["%s: %d 行" % (p.parent.name, n)
               for p, n in ((p, len(p.read_text(encoding="utf-8").splitlines()))
                            for p in sorted(SKILLS.glob("*/SKILL.md")))
               if n > LIMIT]
        self.assertEqual(bad, [], "SKILL.md 超过 %d 行（拆进 references/）：\n%s"
                         % (LIMIT, "\n".join(bad)))


class TestContractGates(unittest.TestCase):
    def test_templates_are_known_subcommands(self):
        known = _opcheck_subcommands()
        self.assertTrue(known, "没能从 scripts/opcheck.py 读出子命令")
        bad = []
        for card in _registered_cards():
            contract = parse_yaml((card / "contract.yaml").read_text(encoding="utf-8"))
            for gate in contract.get("gates") or []:
                tpl = gate.get("template")
                if tpl and tpl not in known:
                    bad.append("%s: gate %r 的 template=%r 不是 opcheck 子命令" %
                               (card.name, gate.get("name"), tpl))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_scripts_exist_or_gate_is_declared_missing(self):
        bad = []
        for card in _registered_cards():
            contract = parse_yaml((card / "contract.yaml").read_text(encoding="utf-8"))
            for gate in contract.get("gates") or []:
                script = gate.get("script")
                if not script or (card / script).exists():
                    continue
                if gate.get("required") is False and str(gate.get("skip_reason", "")).strip():
                    continue
                bad.append("%s: gate %r 指向 %s，文件不存在且未声明 required:false + skip_reason"
                           % (card.name, gate.get("name"), script))
        self.assertEqual(bad, [], "\n".join(bad))


class TestRegistry(unittest.TestCase):
    def test_registry_paths_and_names_agree(self):
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
        bad = []
        for op in reg.get("ops") or []:
            card = REPO / op["path"]
            if not card.is_dir():
                bad.append("registry 的 path 不存在: %s" % op["path"])
                continue
            if card.name != "op-" + op["name"]:
                bad.append("%s: 目录名应为 op-%s" % (card.name, op["name"]))
            contract_path = card / "contract.yaml"
            if not contract_path.is_file():
                bad.append("%s: 缺 contract.yaml" % card.name)
                continue
            got = parse_yaml(contract_path.read_text(encoding="utf-8")).get("op")
            if got != op["name"]:
                bad.append("%s: contract.op=%r 与 registry name=%r 不一致" %
                           (card.name, got, op["name"]))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_every_card_is_registered(self):
        registered = {op["name"] for op in json.loads(REGISTRY.read_text(encoding="utf-8"))["ops"]}
        missing = [c.name for c in _card_dirs() if c.name[3:] not in registered]
        self.assertEqual(missing, [], "卡未在 registry.json 登记: %s" % missing)

    def test_entry_index_lists_every_skill(self):
        text = OPS_ENTRY.read_text(encoding="utf-8")
        indexed = set(re.findall(r"(?m)^\|\s*`(op-[\w-]+)`\s*\|", text))
        actual = {p.name for p in _skill_dirs()}
        self.assertEqual(indexed - actual, set(), "ops/SKILL.md 索引里有不存在的 skill")
        self.assertEqual(actual - indexed, set(), "ops/SKILL.md 索引漏登记 skill")

    def test_entry_index_status_matches_registry(self):
        """索引里的状态列必须等于 registry 的 status——两处各写一遍必然漂。"""
        text = OPS_ENTRY.read_text(encoding="utf-8")
        shown = {name: cell.strip().strip("`")
                 for name, cell in re.findall(r"(?m)^\|\s*`(op-[\w-]+)`\s*\|\s*([^|]*?)\s*\|", text)}
        bad = []
        for op in json.loads(REGISTRY.read_text(encoding="utf-8"))["ops"]:
            got = shown.get("op-" + op["name"])
            if got is None:
                continue            # 漏登记由 test_entry_index_lists_every_skill 报
            if got != op["status"]:
                bad.append("%s: ops/SKILL.md 状态=%r，registry.json=%r"
                           % (op["name"], got, op["status"]))
        self.assertEqual(bad, [], "\n".join(bad))


def _card_dirs():
    return [p for p in _skill_dirs() if (p / "contract.yaml").is_file()]


def _registered_cards():
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [REPO / op["path"] for op in reg["ops"]]


if __name__ == "__main__":
    unittest.main()
