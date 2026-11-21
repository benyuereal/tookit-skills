#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ops 门禁 —— 本项目唯一的 PreToolUse hook（T10/P0-4）。

只挂四个动作，动作发生之前跑同一组检查：

    动作            触发条件
    write_source    Write/Edit/MultiEdit/NotebookEdit 的目标是 kernel 源码
    write_verified  写入目标是 `SKILL.md` 或 `references/*.md`，且这次写**新增/改写**了
                    `status: verified` 或「已验证 / verified」标记（普通文档编辑不拦）
    run_bench       Bash 跑的是卡片里的 bench 脚本 / `loop.py … run` / `scripts/opbench.py`
    git_commit      Bash 跑的是 `git commit`

    检查
    A 契约         每张卡的 contract.yaml 必须过 opverify.check_contract
    B 执行身份     每张卡**最新**一份 manifest.json 必须过 opmanifest.validate_manifest
    C 方向预算     有 candidates.yaml 的卡，`loop.py queue` 必须 rc=0（只在 commit 时查）
    D 交付件结构   `runs/op-optimize/<run>/<node>/result.json` 必须是对象且含 node/op/fingerprint
                   （只看节点交付件，不看 `<run>/result.json` 那种一次 run 一个文件的旧布局）
    E 不许手改产物 写入目标不得是任何 manifest 里 role=build/loaded 的产物
    F verified 提案 新增/改写 verified 标记时，同目录必须有 `<目标文件名>.proposal.json`
                   （含 target / evidence[] / approved_by）——verified 是资产，不是一句形容词

判定只调用 opverify / opmanifest / loop.py 里已有的函数，本文件不重新实现一遍判定逻辑
（一个概念只留一个权威点）。本文件只负责「什么时候问」和「问到了怎么拦」。

**门禁自己坏了必须失败关闭**：任何内部异常 ⇒ exit 2 并说清坏在哪，不是 exit 0。
反过来写 `sys.exit(0)` 就是「门禁一坏就全放行」——静默退化的门比没有门更危险，因为没人会发现。

协议（Claude Code PreToolUse）：stdin 收 {"tool_name","tool_input","cwd"}；
exit 0 = 放行，exit 2 = 拦下，stderr 回给模型。

没有环境变量开关。真要停掉门禁，改 .claude/settings.json —— 那是显式的、看得见的动作；
留一个 `OPS_GATE=0` 之类的后门，等于把「门禁坏了」和「门禁不存在」变成同一个状态。
"""

import collections
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[3]
SKILLS = ROOT / "Plugins" / "ops" / "skills"
RUNS = ROOT / "runs" / "op-optimize"
SCRIPTS = ROOT / "scripts"
LOG = ROOT / "runs" / "ops-gate.jsonl"

WRITE_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")

# D 的必填键：一份交付件至少要能回溯到「哪个算子、哪个节点、哪次运行的指纹」。
# 这里只挡「字段被删空」这一类结构性问题（commit 时点，便宜）。字段级契约在
# `mark done` 那层由 workflow_state.py 按节点任务契约（tasks/<node>.yaml 的
# evidence.fingerprint）逐键校验 —— 两层不同口径、不同落点，各管一段，别在这里重复实现。
RESULT_REQUIRED = ("node", "op", "fingerprint")


def card_op(card):
    """卡片目录名 → 契约里的算子名。`op-sparse-mla` → `sparse-mla`。"""
    name = pathlib.Path(card).name
    return name[3:] if name.startswith("op-") else name

SRC_SUFFIX = (".hip", ".cu", ".cuh")

# F：什么算「一个 verified 标记」。两类别名，都按**行**判（本项目文档里一段就是一行）：
#   1) YAML 状态行 `status: verified`（卡片/知识卡的机器可读断言）
#   2) 出现过「已验证」或独立单词 `verified` 的行（叙述性断言）
# 刻意避开：`verified_on`（下划线是 word 字符，\b 不成立）、`unverified`（前有 n）。
VERIFIED_STATUS_RE = re.compile(r"(?i)^\s*(?:[-*+]\s+)?status\s*:\s*[\"']?verified\b")
VERIFIED_WORD_RE = re.compile(r"已验证|\b[Vv]erified\b")

# 提案文件必须有的键（ProposeSkills 的 review card 同款：evidence 让你回到现场）
PROPOSAL_KEYS = ("target", "evidence", "approved_by")

# 带参数的 git 前置选项：跳过它时要连同参数一起跳过
GIT_OPTS_WITH_ARG = ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path")

HELP = {
    "A": "契约不合格 → 修 contract.yaml，复现：python3 -I scripts/opverify.py --contract <卡>/contract.yaml --check-contract",
    "B": "执行身份漂了 → 重建产物后重新 `opmanifest.py record`；改完源码不重建就是这样被拦的",
    "C": "方向预算违规 → python3 -I Plugins/ops/skills/op-optimize/loop.py -C <卡> queue 看缺什么",
    "D": "交付件缺 node/op/fingerprint → 补齐 result.json 再提交（字段级契约在 mark done 那层由任务契约逐键校验）",
    "E": "不许手改编译产物 → 改源码后重建，别动 build/loaded 文件",
    "F": "写 verified 需要提案 → 在同目录补 <目标文件名>.proposal.json"
         "（target / evidence[] / approved_by），evidence 要能回到现场；"
         "普通文档编辑不需要提案",
}


class Gate:
    """把 root 做成构造参数只为测试能指向临时目录；运行时 main() 只传 ROOT。"""

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

    def newest_manifest(self, card):
        """这张卡最新的一份 manifest。

        按 manifest 自己的 `op` 字段认领，不靠 run 目录名猜 —— 目录名是人为起的
        （`2026-10-28-p04` 里根本没有算子名），拿它当键会把门禁挂到别人那份 manifest 上。

        只认最新的：老 manifest 随源码演进必然过期，拿它当门禁会把正常推进判成身份漂移。
        """
        op = card_op(card)
        best = None
        for p in self.manifests():
            try:
                man = self._read_manifest(p)
            except Exception:
                continue
            if not isinstance(man, dict):
                continue
            if str(man.get("op") or "") != op:
                continue
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
        return best

    def _read_manifest(self, p):
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))

    def _rel(self, p):
        try:
            return str(pathlib.Path(p).relative_to(self.root))
        except ValueError:
            return str(p)

    # ---------------------------------------------------------------- 检查
    def check_contracts(self, card_list):
        ov = self._mod("opverify")
        probs = []
        for card in card_list:
            cpath = card / "contract.yaml"
            c = ov.parse_yaml(cpath.read_text(encoding="utf-8"))
            for p in ov.check_contract(c, base_dir=str(card)):
                probs.append(("A", "[%s] 契约：%s" % (card.name, p)))
        return probs

    def check_identity(self, card_list):
        om = self._mod("opmanifest")
        probs, seen = [], []
        # 先单独查「manifest 读不出来」：newest_manifest 为了按 `op` 认领会跳过坏文件，
        # 若只在它里面处理，一个坏 manifest 就等于让身份闸静默消失（fail-open）。
        for m in self.manifests():
            try:
                man = self._read_manifest(m)
            except Exception as e:
                probs.append(("B", "%s 读不出来（%s: %s）—— 身份无法核验" %
                              (self._rel(m), type(e).__name__, e)))
                continue
            if not isinstance(man, dict):
                probs.append(("B", "%s 不是 JSON 对象 —— 身份无法核验" % self._rel(m)))
        for card in card_list:
            m = self.newest_manifest(card)
            if m is None:
                continue
            roles, keys = om.contract_requirements(str(card / "contract.yaml"))
            for p in om.validate_manifest(self._read_manifest(m), roles, keys):
                probs.append(("B", "[%s] 身份（%s）：%s" % (card.name, self._rel(m), p)))
            seen.append(self._rel(m))
        return probs, seen

    def check_budget(self, card_list):
        probs = []
        loop = self.skills / "op-optimize" / "loop.py"
        for card in card_list:
            if not (card / "candidates.yaml").exists():
                continue
            cp = subprocess.run([sys.executable, "-I", str(loop), "-C", str(card), "queue"],
                                capture_output=True, text=True, timeout=120)
            if cp.returncode != 0:
                detail = (cp.stdout + cp.stderr).strip()
                probs.append(("C", "[%s] 方向预算：loop.py queue 退出码 %d\n%s"
                              % (card.name, cp.returncode, detail)))
        return probs

    def check_result_shape(self):
        """**所有节点**交付件都查，不只是冻过 manifest 的那几个：
        删字段的人不必先删 manifest，只按 manifest 目录查会留下一个能绕过去的洞。

        范围只到 `runs/op-optimize/<run>/<node>/result.json` 这一层 —— 那是「节点交付件」。
        `<run>/result.json`（老布局，一次 run 一个文件）不是节点产物，不在这条规则里。"""
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

    # ------------------------------------------------- F verified 资产的提案门
    @staticmethod
    def is_verified_doc(p):
        """只有 `SKILL.md` 与 `references/*.md` 是「verified 资产」的落点。"""
        p = pathlib.Path(p)
        if p.suffix.lower() != ".md":
            return False
        if p.name == "SKILL.md":
            return True
        return any(part == "references" for part in p.parts)

    @staticmethod
    def verified_marks(text):
        """一行一个标记：`status: verified` 或含「已验证 / verified」的行。"""
        out = []
        for line in (text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            if VERIFIED_STATUS_RE.search(s) or VERIFIED_WORD_RE.search(s):
                out.append(s)
        return out

    def verified_delta(self, payload, current):
        """这次写**新增/改写**了哪些 verified 标记（相对当前文件内容）。

        返回 (added, None)；`added` 为空 = 普通编辑，放行。
        读不出「新内容」（缺 content / old_string）⇒ 返回 (None, 原因)：
        没法判定就失败关闭，不许默认放行。

        粒度是**标记行**而不是整个文件：
          - 改标题 / 改错别字 / 改不含标记的段落 ⇒ 不产生新标记行 ⇒ 放行；
          - 在已 verified 的文件里改别的段落、原样保留标记行 ⇒ 放行；
          - 新增一行 `status: verified`、或把标记行改写（哪怕只是加个注释）⇒ 拦；
          - **删掉**标记行不拦（删断言不会凭空造出一条 verified 结论）。
        """
        ti = payload.get("tool_input") or {}
        tool = payload.get("tool_name") or ""
        old = collections.Counter(self.verified_marks(current))

        if tool == "Write":
            new = ti.get("content")
            if not isinstance(new, str):
                return None, "缺 content"
            return list((collections.Counter(self.verified_marks(new)) - old).elements()), None

        if tool == "Edit":
            if not isinstance(ti.get("new_string"), str) or not isinstance(ti.get("old_string"), str):
                return None, "缺 old_string/new_string"
            frag = collections.Counter(self.verified_marks(ti["old_string"]))
            return list((collections.Counter(self.verified_marks(ti["new_string"])) - frag).elements()), None

        if tool == "MultiEdit":
            edits = ti.get("edits")
            if not isinstance(edits, list) or not edits:
                return None, "缺 edits"
            added = []
            for e in edits:
                if not isinstance(e, dict) or not isinstance(e.get("new_string"), str) \
                        or not isinstance(e.get("old_string"), str):
                    return None, "edits 里有条目缺 old_string/new_string"
                frag = collections.Counter(self.verified_marks(e["old_string"]))
                added += list((collections.Counter(self.verified_marks(e["new_string"])) - frag).elements())
            return added, None

        return None, "工具 %s 的写入内容读不出来" % (tool or "?")

    def check_proposal(self, proposal, target):
        """同目录提案是否成立。返回问题清单，空 = 提案齐了。"""
        if not proposal.exists():
            return ["缺提案文件"]
        try:
            data = json.loads(proposal.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return ["提案读不出来（%s: %s）" % (type(e).__name__, e)]
        if not isinstance(data, dict):
            return ["提案不是 JSON 对象"]
        probs = []
        for k in PROPOSAL_KEYS:
            if k not in data:
                probs.append("缺 %s" % k)
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
        """新增/改写 verified 标记 ⇒ 必须带提案。普通编辑一律放行。"""
        target = pathlib.Path(self._target(payload))
        if not self.is_verified_doc(target) or not self._under(target, self.root):
            return []
        try:
            current = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
        except Exception:  # 读不到旧内容时按「空文件」比：只会更严，不会更松
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

    def declared_paths(self, role=None):
        """所有 manifest 里声明过的产物路径（按 role 过滤）。用于回答「这是不是 kernel 源码」。"""
        out = set()
        for m in self.manifests():
            for a in self._read_manifest(m).get("artifacts") or []:
                if not isinstance(a, dict) or not a.get("path"):
                    continue
                if role is None or a.get("role") == role:
                    out.add(os.path.realpath(a["path"]))
        return out

    # ---------------------------------------------------------------- 动作识别
    def classify(self, payload):
        """返回 (动作名, 目标卡列表, 说明)。动作名 None = 不管。"""
        tool = payload.get("tool_name") or ""
        ti = payload.get("tool_input") or {}
        cwd = payload.get("cwd") or str(self.root)

        if tool in WRITE_TOOLS:
            raw = ti.get("file_path") or ti.get("notebook_path") or ""
            if not raw:
                return None, [], ""
            p = pathlib.Path(raw)
            if not p.is_absolute():
                p = pathlib.Path(cwd) / p
            p = pathlib.Path(os.path.realpath(p))
            # F：verified 资产（SKILL.md / references/*.md）。不在这里判「有没有提案」——
            # 那要在 decide() 里做；classify 只回答「这算不算这次要管的事」。
            if self._under(p, self.root) and self.is_verified_doc(p):
                return "write_verified", [], "%s（SKILL.md / references）" % self._rel(p)
            srcs = self.declared_paths("source")
            hit = str(p) in srcs
            why = "manifest 里声明的源码"
            if not hit:
                if str(p) in (self.declared_paths("build") | self.declared_paths("loaded")):
                    return "write_artifact", self.cards(), "编译产物 %s" % p.name
                # 没被任何 manifest 声明过：只有落在本仓库里的才归我们管，
                # 否则装到 workspace 级会把别人项目的 .hip 也一起拦（跨项目耦合）
                if not self._under(p, self.root):
                    return None, [], ""
                if p.suffix in SRC_SUFFIX:
                    hit, why = True, "%s 源码" % p.suffix
                elif p.suffix == ".py" and "perf" in p.parts:
                    hit, why = True, "perf/ 下的 .py"
            if not hit:
                return None, [], ""
            cards, owner = [], False
            for card in self.cards():
                m = self.newest_manifest(card)
                if m and str(p) in {os.path.realpath(a["path"])
                                    for a in self._read_manifest(m).get("artifacts") or []
                                    if isinstance(a, dict) and a.get("path")}:
                    cards.append(card)
                    owner = True
            return "write_source", cards or self.cards(), \
                ("已声明源码（%s）" % why) + ("" if owner else "，但没有 manifest 声明它")

        if tool == "Bash":
            cmd = ti.get("command") or ""
            toks = self._tokens(cmd)
            if self._is_git_commit(toks):
                return "git_commit", self.cards(), "git commit"
            cards = self._bench_cards(toks, cwd)
            if cards is not None:
                return "run_bench", cards or self.cards(), "bench"
        return None, [], ""

    @staticmethod
    def _under(p, root):
        try:
            pathlib.Path(os.path.realpath(p)).relative_to(os.path.realpath(root))
            return True
        except ValueError:
            return False

    def _git_toplevel(self, cwd):
        """cwd 所在的 git 仓库根。不是仓库 / 没有 git ⇒ None。"""
        try:
            cp = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
                                capture_output=True, text=True, timeout=20)
        except Exception:
            return None
        if cp.returncode != 0 or not cp.stdout.strip():
            return None
        return pathlib.Path(os.path.realpath(cp.stdout.strip()))

    @staticmethod
    def _tokens(cmd):
        try:
            return shlex.split(cmd, posix=True)
        except ValueError:
            return cmd.split()

    @staticmethod
    def _is_git_commit(toks):
        """只认 `git [-C dir] commit` 这一形态。
        走 token 而不是正则：`git log --grep commit` 里的 commit 不是子命令。"""
        for i, tok in enumerate(toks):
            if tok != "git":
                continue
            j = i + 1
            while j < len(toks) and toks[j].startswith("-"):
                j += 2 if toks[j] in GIT_OPTS_WITH_ARG else 1
            if j < len(toks) and toks[j] == "commit":
                return True
        return False

    def _bench_cards(self, toks, cwd):
        """命令里有没有跑 bench 脚本。返回命中的卡（可能为空 = 命中但认不出卡），
        不命中返回 None。"""
        hit, cards = False, []
        for i, tok in enumerate(toks):
            if not tok.endswith(".py"):
                continue
            p = pathlib.Path(tok)
            if not p.is_absolute():
                p = pathlib.Path(cwd) / p
            p = pathlib.Path(os.path.realpath(p))
            if p.name == "opbench.py" and p.parent == self.scripts:
                hit = True
                continue
            if p == self.skills / "op-optimize" / "loop.py":
                if "run" in toks[i + 1:]:
                    hit = True
                continue
            for parent in p.parents:
                if parent.name == "bench":
                    hit = True
                    owner = parent.parent
                    if (owner / "contract.yaml").exists() and owner not in cards:
                        cards.append(owner)
                    break
        return cards if hit else None

    # ---------------------------------------------------------------- 判定
    def decide(self, payload):
        """返回 (exit_code, 消息, 动作名)。exit 0 = 放行，2 = 拦下。"""
        action, card_list, note = self.classify(payload)
        if action is None:
            return 0, "", None

        probs = []
        seen = []
        if action == "write_verified":
            probs += self.check_verified_write(payload)
        elif action == "write_artifact":
            target = self._target(payload)
            probs.append(("E", "%s 是 manifest 里 role=build/loaded 的产物" % self._rel(target)))
        elif action == "write_source":
            target = self._target(payload)
            probs += self.check_contracts(card_list)
            # 写「已声明的源码」本身就是要制造漂移，这时再拿 B 拦自己是自相矛盾；
            # 只有写一条**别的** kernel 源码、而身份链已经断了，才拦。
            if not target or os.path.realpath(target) not in self.declared_paths("source"):
                ip, seen = self.check_identity(card_list)
                probs += ip
        elif action == "run_bench":
            probs += self.check_contracts(card_list)
            ip, seen = self.check_identity(card_list)
            probs += ip
        else:  # git_commit
            probs += self.check_contracts(card_list)
            ip, seen = self.check_identity(card_list)
            probs += ip
            probs += self.check_budget(card_list)
            probs += self.check_result_shape()

        if probs:
            head = "ops-gate 拦下了 %s（%s）：%d 项不过" % (action, note, len(probs))
            lines = [head]
            for _, msg in probs:
                lines.append("  - " + msg)
            used = []
            for code, _ in probs:
                if code not in used:
                    used.append(code)
            lines.append("怎么解：")
            for code in used:
                lines.append("  - " + HELP[code])
            return 2, "\n".join(lines), action

        if action == "write_verified":
            return 0, ("ops-gate ✓ %s（%s）：verified 提案门通过"
                       "（无新增标记，或标记有提案背书）" % (action, note)), action
        where = ("，检查了 %s" % ", ".join(seen)) if seen else ""
        return 0, "ops-gate ✓ %s（%s%s）契约/身份通过" % (action, note, where), action

    def _target(self, payload):
        ti = payload.get("tool_input") or {}
        raw = ti.get("file_path") or ti.get("notebook_path") or ""
        if not raw:
            return ""
        p = pathlib.Path(raw)
        if not p.is_absolute():
            p = pathlib.Path(payload.get("cwd") or self.root) / p
        return os.path.realpath(p)


def _log(entry):
    """留痕，best-effort：日志写不进去不影响判定（它不参与任何门禁）。"""
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        rc, msg, action = Gate(ROOT).decide(payload)
    except Exception as e:
        # 失败关闭：门禁自己坏了，不许静默放行
        print("ops-gate 自己坏了（%s: %s）—— 失败关闭，拦下这次动作。\n"
              "先修门禁，别绕过。" % (type(e).__name__, e), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    if msg:
        print(msg, file=sys.stderr)
    if action:
        _log({"ts": _now(), "action": action, "rc": rc,
              "tool": payload.get("tool_name"),
              "head": (msg or "").splitlines()[0] if msg else ""})
    return rc


def _now():
    import datetime
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    sys.exit(main())
