#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ops 门禁 —— 本项目唯一的 PreToolUse hook。

只关心下面这几类动作，其余一律立刻放行：`_worth_checking` 是毫秒级字符串预筛，
不读文件、不 import 重型模块、不 glob 全仓。

    write_source    写 kernel 源码（.hip/.cu/.cuh、perf/*.py、manifest 声明的 source）
    write_artifact  写 manifest 里 role=build/loaded 的产物
    write_verified  写 SKILL.md / references/*.md 且这次新增或改写了 verified 标记
    run_bench       Bash 跑卡片 bench 脚本 / loop.py … run / scripts/opbench.py
    git_commit      Bash 跑 git commit

检查函数在 `scripts/opgate_checks.py`；本文件只负责「什么时候问、问到了怎么拦」。
C/D 只在 commit 时查，不挂在 Write/Edit 路径上。

**门禁自己坏了失败关闭**：任何内部异常 ⇒ exit 2 并说清坏在哪。

协议：stdin 收 {"tool_name","tool_input","cwd"}；exit 0 = 放行，exit 2 = 拦下。
没有环境变量开关——要停就改 .claude/settings.json，那是显式、看得见的动作。
"""

import json
import os
import pathlib
import shlex
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[3]
LOG = ROOT / "runs" / "ops-gate.jsonl"
SCRIPTS = ROOT / "scripts"

WRITE_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
SRC_SUFFIX = (".hip", ".cu", ".cuh")
GIT_OPTS_WITH_ARG = ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path")

HELP = {
    "A": "契约不合格 → 修 contract.yaml，复现：python3 -I scripts/opverify.py --contract <卡>/contract.yaml --check-contract",
    "B": "执行身份漂了 → 重建产物后重新 `opmanifest.py record`；改完源码不重建就是这样被拦的",
    "C": "方向预算违规 → python3 -I Plugins/ops/skills/op-optimize/loop.py -C <卡> queue 看缺什么",
    "D": "交付件缺 node/op/fingerprint → 补齐 result.json 再提交",
    "E": "不许手改编译产物 → 改源码后重建，别动 build/loaded 文件",
    "F": "写 verified 需要提案 → 同目录补 <目标文件名>.proposal.json"
         "（target / evidence[] / approved_by），evidence 要能回到现场",
}


def _worth_checking(payload):
    """快速路径：这次动作值不值得进判定。只做字符串/后缀判断，毫秒级。"""
    tool, ti = payload.get("tool_name") or "", payload.get("tool_input") or {}
    if tool == "Bash":
        cmd = ti.get("command") or ""
        return "commit" in cmd or ".py" in cmd      # 其余 Bash 永远不查
    if tool in WRITE_TOOLS:
        raw = (ti.get("file_path") or ti.get("notebook_path") or "").lower()
        if not raw:
            return False
        if raw.endswith(SRC_SUFFIX):
            return True
        if raw.endswith(".md"):
            return raw.endswith("skill.md") or "/references/" in raw
        return (raw.endswith(".py") and "/perf/" in raw) \
            or raw.endswith((".so", ".o", ".a")) or "/build/" in raw
    return False


class Gate:
    """检查函数在 scripts/opgate_checks.py（用到时才导入）；这里只加动作识别与判定编排。"""

    def __init__(self, root):
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        import opgate_checks
        self._impl = opgate_checks.Checks(root)

    def __getattr__(self, name):
        """检查函数与 root/skills/runs/scripts 全部转发到那棵 Checks 实例。"""
        if name == "_impl":                     # __init__ 期间尚未赋值，别递归
            raise AttributeError(name)
        return getattr(self._impl, name)

    def classify(self, payload):
        """返回 (动作名, 目标卡列表, 说明)。动作名 None = 不管。"""
        tool, ti = payload.get("tool_name") or "", payload.get("tool_input") or {}
        cwd = payload.get("cwd") or str(self.root)

        if tool in WRITE_TOOLS:
            target = self._target(payload)
            if not target:
                return None, [], ""
            p = pathlib.Path(target)
            if self._under(p, self.root) and self.is_verified_doc(p):
                return "write_verified", [], "%s（SKILL.md / references）" % self._rel(p)
            hit, why = str(p) in self.declared_paths("source"), "manifest 里声明的源码"
            if not hit:
                if str(p) in (self.declared_paths("build") | self.declared_paths("loaded")):
                    return "write_artifact", self.cards(), "编译产物 %s" % p.name
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
            toks = self._tokens(ti.get("command") or "")
            if self._is_git_commit(toks):
                return "git_commit", self.cards(), "git commit"
            cards = self._bench_cards(toks, cwd)
            if cards is not None:
                return "run_bench", cards or self.cards(), "bench"
        return None, [], ""

    @staticmethod
    def _tokens(cmd):
        try:
            return shlex.split(cmd, posix=True)
        except ValueError:
            return cmd.split()

    @staticmethod
    def _is_git_commit(toks):
        """只认 `git [-C dir] commit`；走 token，`git log --grep commit` 不算。"""
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
        """命令里有没有跑 bench 脚本。返回命中的卡（可能为空），不命中返回 None。"""
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
            elif p in (self.skills / "op-optimize" / "loop.py",   # shim（保留期）
                       self.skills.parent / "engine" / "loop.py"):  # 迁后的真身
                # 两个路径都要认：实现已迁到 engine/，但 shim 还在，
                # 用户从任一入口调 run 都必须被 hook 拦到。
                hit = hit or "run" in toks[i + 1:]
            else:
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

        probs, seen, notes = [], [], []
        if action == "write_verified":
            probs += self.check_verified_write(payload)
        elif action == "write_artifact":
            probs.append(("E", "%s 是 manifest 里 role=build/loaded 的产物"
                          % self._rel(self._target(payload))))
        elif action == "write_source":
            target = self._target(payload)
            # 写「已声明的源码」本身就是在制造漂移，拿身份闸拦自己自相矛盾。
            if not target or os.path.realpath(target) not in self.declared_paths("source"):
                ip, seen, notes = self.check_identity(card_list)
                probs += ip
        elif action == "run_bench":
            probs += self.check_contracts(card_list)
            ip, seen, notes = self.check_identity(card_list)
            probs += ip
        else:  # git_commit
            probs += self.check_contracts(card_list)
            ip, seen, notes = self.check_identity(card_list)
            probs += ip
            probs += self.check_budget(card_list)
            probs += self.check_result_shape()

        if probs:
            lines = ["ops-gate 拦下了 %s（%s）：%d 项不过" % (action, note, len(probs))]
            lines += ["  - " + msg for _, msg in probs]
            lines.append("怎么解：")
            for code in list(dict.fromkeys(c for c, _ in probs)):
                lines.append("  - " + HELP[code])
            return 2, "\n".join(lines), action
        if action == "write_verified":
            return 0, ("ops-gate ✓ %s（%s）：verified 提案门通过"
                       "（无新增标记，或标记有提案背书）" % (action, note)), action
        where = ("，检查了 %s" % ", ".join(seen)) if seen else ""
        hint = ("".join("\n  提示：" + n for n in notes)) if notes else ""
        return 0, "ops-gate ✓ %s（%s%s）契约/身份通过%s" % (action, note, where, hint), action


def _log(entry):
    """留痕，best-effort：日志写不进去不影响判定。"""
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
        if not isinstance(payload, dict) or not _worth_checking(payload):
            return 0
        rc, msg, action = Gate(ROOT).decide(payload)
    except Exception as e:
        print("ops-gate 自己坏了（%s: %s）—— 失败关闭，拦下这次动作。\n"
              "先修门禁，别绕过。" % (type(e).__name__, e), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    if msg:
        print(msg, file=sys.stderr)
    if action:
        import datetime
        _log({"ts": datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
              "action": action, "rc": rc, "tool": payload.get("tool_name"),
              "head": (msg or "").splitlines()[0] if msg else ""})
    return rc


if __name__ == "__main__":
    sys.exit(main())
