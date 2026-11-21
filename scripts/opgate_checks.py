#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ops 门禁的检查函数（从 hooks/ops_gate.py 拆出来）。

hook 自己只负责「什么时候问、问到了怎么拦」；这里放「问什么」——A/B/C/D 契约与
身份、E 编译产物、F verified 提案。判定本身仍然只调用 opverify / opmanifest /
loop.py 里已有的函数，本模块不重新实现一遍判定逻辑。

`Checks` 以 root 为构造参数，测试可以指向临时仓库树。
"""

import collections
import json
import os
import pathlib
import re
import subprocess
import sys

RESULT_REQUIRED = ("node", "op", "fingerprint")
PROPOSAL_KEYS = ("target", "evidence", "approved_by")
VERIFIED_STATUS_RE = re.compile(r"(?i)^\s*(?:[-*+]\s+)?status\s*:\s*[\"']?verified\b")
VERIFIED_WORD_RE = re.compile(r"已验证|\b[Vv]erified\b")

# 超过这个天数的 manifest 只提示不拦截：结论本身已经过期，该重新量而不是继续引用。
STALE_DAYS = 7


def _manifest_age_days(man):
    """manifest 记录距今多少天；读不出时间就当无限旧（不阻塞无关动作）。"""
    import datetime as _dt
    raw = (man.get("recorded_at") or "") if isinstance(man, dict) else ""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            then = _dt.datetime.strptime(raw[:len(raw)], fmt)
        except (ValueError, TypeError):
            continue
        now = _dt.datetime.now(then.tzinfo) if then.tzinfo else _dt.datetime.now()
        return max(0, (now - then).days)
    return 10 ** 6


def card_op(card):
    """卡片目录名 → 契约里的算子名。`op-sparse-mla` → `sparse-mla`。"""
    name = pathlib.Path(card).name
    return name[3:] if name.startswith("op-") else name


class Checks:
    """一棵仓库树上的全部检查。root 是构造参数，只为测试能指向临时目录。"""

    def __init__(self, root):
        self.root = pathlib.Path(root).resolve()
        self.skills = self.root / "Plugins" / "ops" / "skills"
        self.runs = self.root / "runs" / "op-optimize"
        self.scripts = self.root / "scripts"

    # ------------------------------------------------------------ 基础设施
    def _mod(self, name):
        s = str(self.scripts)
        if s not in sys.path:
            sys.path.insert(0, s)
        return __import__(name)

    def cards(self):
        return sorted(p.parent for p in self.skills.glob("*/contract.yaml"))

    def manifests(self):
        return sorted(self.runs.glob("*/*/manifest.json"))

    def _read_manifest(self, p):
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))

    def _rel(self, p):
        try:
            return str(pathlib.Path(p).relative_to(self.root))
        except ValueError:
            return str(p)

    def newest_manifest(self, card):
        """按 manifest 自己的 `op` 字段认领，只认最新的一份（老 manifest 必然过期）。"""
        op, best = card_op(card), None
        for p in self.manifests():
            try:
                man = self._read_manifest(p)
            except Exception:
                continue
            if isinstance(man, dict) and str(man.get("op") or "") == op:
                if best is None or p.stat().st_mtime > best.stat().st_mtime:
                    best = p
        return best

    @staticmethod
    def _under(p, root):
        try:
            pathlib.Path(os.path.realpath(p)).relative_to(os.path.realpath(root))
            return True
        except ValueError:
            return False

    def _target(self, payload):
        ti = payload.get("tool_input") or {}
        raw = ti.get("file_path") or ti.get("notebook_path") or ""
        if not raw:
            return ""
        p = pathlib.Path(raw)
        if not p.is_absolute():
            p = pathlib.Path(payload.get("cwd") or self.root) / p
        return os.path.realpath(p)

    # ---------------------------------------------------------------- 检查
    def check_contracts(self, card_list):
        ov, probs = self._mod("opverify"), []
        for card in card_list:
            c = ov.parse_yaml((card / "contract.yaml").read_text(encoding="utf-8"))
            for p in ov.check_contract(c, base_dir=str(card)):
                probs.append(("A", "[%s] 契约：%s" % (card.name, p)))
        return probs

    def check_identity(self, card_list):
        """身份闸：只核验**本次动作相关**的卡，且陈旧 manifest 不阻塞无关动作。

        为什么要这条：manifest 记的是某次 run 的产物指纹，它随机器/缓存目录变动而失效
        （换一台机器，`~/.tilelang/cache/...` 必然不存在）。让一条 2026-10 的记录永久挡住
        所有 `git commit`，就把「身份闸」变成了「仓库锁」——这正是要避免的重约束。

        规则：只核验 card_list 里的卡；每张卡只看最新一份 manifest；**记录早于
        STALE_DAYS 天**的只报为提示（进 note，不拦），因为它的结论已经过期，
        该做的是重新量而不是继续引用。`OPS_GATE_STRICT=1` 可恢复全量拦截。
        """
        om, probs, seen, notes = self._mod("opmanifest"), [], [], []
        strict = os.environ.get("OPS_GATE_STRICT") == "1"
        for m in self.manifests():          # 坏 manifest 不能静默消失（fail-open）
            try:
                man = self._read_manifest(m)
            except Exception as e:
                probs.append(("B", "%s 读不出来（%s: %s）—— 身份无法核验"
                              % (self._rel(m), type(e).__name__, e)))
                continue
            if not isinstance(man, dict):
                probs.append(("B", "%s 不是 JSON 对象 —— 身份无法核验" % self._rel(m)))
        for card in card_list:
            m = self.newest_manifest(card)
            if m is None:
                continue
            man = self._read_manifest(m)
            stale = _manifest_age_days(man) > STALE_DAYS
            roles, keys = om.contract_requirements(str(card / "contract.yaml"))
            for p in om.validate_manifest(man, roles, keys):
                if stale and not strict:
                    notes.append("[%s] 陈旧 manifest（%s 天前）不再拦截：%s"
                                 % (card.name, _manifest_age_days(man), p))
                    continue
                probs.append(("B", "[%s] 身份（%s）：%s" % (card.name, self._rel(m), p)))
            seen.append(self._rel(m))
        return probs, seen, notes

    def check_budget(self, card_list):
        probs = []
        loop = self.skills / "op-optimize" / "loop.py"
        for card in card_list:
            if not (card / "candidates.yaml").exists():
                continue
            cp = subprocess.run([sys.executable, "-I", str(loop), "-C", str(card), "queue"],
                                capture_output=True, text=True, timeout=120)
            if cp.returncode != 0:
                probs.append(("C", "[%s] 方向预算：loop.py queue 退出码 %d\n%s"
                              % (card.name, cp.returncode, (cp.stdout + cp.stderr).strip())))
        return probs

    def check_result_shape(self):
        """节点交付件 runs/op-optimize/<run>/<node>/result.json 至少能回溯 node/op/fingerprint。"""
        probs = []
        for rj in sorted(self.runs.glob("*/*/result.json")):
            rel = self._rel(rj)
            try:
                d = json.loads(rj.read_text(encoding="utf-8"))
            except Exception as e:
                probs.append(("D", "%s 读不出来（%s: %s）" % (rel, type(e).__name__, e)))
                continue
            if not isinstance(d, dict):
                probs.append(("D", "%s 不是 JSON 对象" % rel))
                continue
            miss = [k for k in RESULT_REQUIRED if k not in d]
            if miss:
                probs.append(("D", "%s 缺字段 %s（交付件必须能回溯）" % (rel, ", ".join(miss))))
        return probs

    def declared_paths(self, role=None):
        """所有 manifest 里声明过的产物路径（按 role 过滤）。"""
        out = set()
        for m in self.manifests():
            for a in self._read_manifest(m).get("artifacts") or []:
                if isinstance(a, dict) and a.get("path") and role in (None, a.get("role")):
                    out.add(os.path.realpath(a["path"]))
        return out

    # ------------------------------------------------- F verified 资产的提案门
    @staticmethod
    def is_verified_doc(p):
        p = pathlib.Path(p)
        return p.suffix.lower() == ".md" and (
            p.name == "SKILL.md" or any(part == "references" for part in p.parts))

    @staticmethod
    def verified_marks(text):
        return [s for s in (line.strip() for line in (text or "").splitlines())
                if s and (VERIFIED_STATUS_RE.search(s) or VERIFIED_WORD_RE.search(s))]

    def verified_delta(self, payload, current):
        """这次写新增/改写了哪些 verified 标记；返回 (added, None) 或 (None, 原因)。"""
        ti = payload.get("tool_input") or {}
        tool = payload.get("tool_name") or ""

        def delta(old_text, new_text):
            return list((collections.Counter(self.verified_marks(new_text))
                         - collections.Counter(self.verified_marks(old_text))).elements())

        if tool == "Write":
            new = ti.get("content")
            return (delta(current, new), None) if isinstance(new, str) else (None, "缺 content")
        if tool == "Edit":
            if not isinstance(ti.get("new_string"), str) or not isinstance(ti.get("old_string"), str):
                return None, "缺 old_string/new_string"
            return delta(ti["old_string"], ti["new_string"]), None
        if tool == "MultiEdit":
            edits = ti.get("edits")
            if not isinstance(edits, list) or not edits:
                return None, "缺 edits"
            added = []
            for e in edits:
                if not isinstance(e, dict) or not isinstance(e.get("new_string"), str) \
                        or not isinstance(e.get("old_string"), str):
                    return None, "edits 里有条目缺 old_string/new_string"
                added += delta(e["old_string"], e["new_string"])
            return added, None
        return None, "工具 %s 的写入内容读不出来" % (tool or "?")

    def check_proposal(self, proposal, target):
        """同目录提案是否成立。返回问题清单，空 = 提案齐了。"""
        if not proposal.exists():
            return ["缺提案文件"]
        try:
            data = json.loads(proposal.read_text(encoding="utf-8"))
        except Exception as e:
            return ["提案读不出来（%s: %s）" % (type(e).__name__, e)]
        if not isinstance(data, dict):
            return ["提案不是 JSON 对象"]
        probs = ["缺 %s" % k for k in PROPOSAL_KEYS if k not in data]
        if data.get("target") != target.name:
            probs.append("target=%r 与目标文件名 %r 不符" % (data.get("target"), target.name))
        ev = data.get("evidence")
        if not isinstance(ev, list) or not ev or not all(
                isinstance(x, str) and x.strip() for x in ev):
            probs.append("evidence 必须是非空字符串数组（可回溯的路径或命令）")
        if not str(data.get("approved_by") or "").strip():
            probs.append("approved_by 为空（谁批的）")
        return probs

    def check_verified_write(self, payload):
        """新增/改写 verified 标记 ⇒ 必须带提案；普通编辑一律放行。"""
        target = pathlib.Path(self._target(payload))
        if not self.is_verified_doc(target) or not self._under(target, self.root):
            return []
        try:
            current = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        except Exception:
            current = ""
        added, why = self.verified_delta(payload, current)
        if added is None:
            return [("F", "%s 的写入内容读不出来（%s）—— 无法判定是否新增 verified，失败关闭"
                     % (self._rel(target), why))]
        if not added:
            return []
        proposal = target.with_name(target.name + ".proposal.json")
        probs = self.check_proposal(proposal, target)
        if not probs:
            return []
        sample = "；".join(added[:3]) + ("…" if len(added) > 3 else "")
        return [("F", "写 verified 需要提案：%s\n"
                      "    目标 %s 这次新增/改写了 %d 个 verified 标记（%s）\n"
                      "    %s\n"
                      "    提案格式：{\"target\": \"%s\", \"evidence\": [\"<可回溯路径或命令>\"], "
                      "\"approved_by\": \"<人名或标记>\"}"
                 % (self._rel(proposal), self._rel(target), len(added), sample,
                    "\n    ".join(probs), target.name))]
