#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""skill 评测执行器：把 `<skill>/evals/evals.json` 里的用例真的跑一遍。

和 `run_evals.py` 的分工（一个管静态、一个管动态，别混）：
  - `run_evals.py`  —— 静态门：结构、id 唯一性、true/false 约定。零成本，进每次 CI。
  - 本脚本         —— 动态门：起真模型，看**这条 prompt 会不会把该 skill 拉起来**，
                       并（可选）判答得对不对。烧 token，需要 Claude Code 登录态，
                       默认不进 CI，按需手动跑。

判法（默认只判「路由」，最客观、最便宜）：
  - true  用例：本次会话第一个 `Skill` 调用必须是被评的这个 skill（除此以外都是没命中）；
  - false 用例：本次会话**不得**调用被评的 skill（形似问题应路由到别处，别抢活）。
  `--judge` 再加上一层内容判定：把回答与 `expect` 一起交给模型打分（1~5，≥4 算过）。
  `forbid` 是**零成本**的那层：路由对之后，在回答文本里找禁用说法（写「错话」本身，
  不是关键词；命中位置前有否定词则跳过）——拉起对了但把 `sshpass -p` 教出去，照样判错。

抖动：单次判分是概率事件。`--repeat N` 同一用例跑 N 次，报「k/N」——全过 PASS、
部分过 FLAKY、全败 FAIL；失败再归类成 **未拉起**（模型没加载任何 skill）/ **拉错**
（邻居抢走）/ **误拉**（false 用例被误触发），三类的修法不同（见 docs/EVALS.md §四）。

隔离：每个用例在独立临时目录里跑；`claude -p` 挂**仓内全部**插件（路由本来就发生在
全库之间，只挂被测域测不出「被邻居抢走」），并给本次会话一个空的 `CLAUDE_CONFIG_DIR`，
使 `~/.claude/skills` 里的已安装旧副本不可见——否则改工作区的 SKILL.md 对结果没有影响。
禁用 Bash/Write/Edit 等，避免评测过程真的去改东西。插件挂载的 skill 带命名空间
（`ops:op-bench`），判分前统一去掉前缀。

用法：
    python3 scripts/skill_eval.py --list                     # 列出有哪些 skill 带 evals
    python3 scripts/skill_eval.py --skill dcu-ssh            # 跑一个（4 例）
    python3 scripts/skill_eval.py --domain dcu-core          # 跑一个域
    python3 scripts/skill_eval.py --skill op-bench --judge   # 连内容一起判
    python3 scripts/skill_eval.py --domain ops --json /tmp/eval.json
退出码：0 全过；1 有失败；2 参数/环境问题（找不到 claude、没登录）。
完整工作流（用例怎么写、失败怎么修、CI 策略）见 docs/EVALS.md。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 允许的 / 禁止的工具：评测只需要「读」，不需要「做」。
ALLOW_TOOLS = "Skill Read Glob Grep"
DENY_TOOLS = ("Bash Write Edit NotebookEdit WebFetch WebSearch Task "
              "TodoWrite KillShell WebFetch")

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "runs"}


# ---------------------------------------------------------------- discovery

def iter_skill_dirs(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # 只认 <...>/skills/<skill>/SKILL.md（包根与插件根的 SKILL.md 是入口文档）
        if "SKILL.md" in filenames and os.path.basename(os.path.dirname(dirpath)) == "skills":
            yield dirpath


def load_case_file(skill_dir):
    path = os.path.join(skill_dir, "evals", "evals.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def plugin_dir_of(skill_dir):
    """skill 目录向上找带 .claude-plugin/plugin.json 的插件根。"""
    cur = os.path.abspath(skill_dir)
    while True:
        if os.path.isfile(os.path.join(cur, ".claude-plugin", "plugin.json")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def all_plugin_dirs(root):
    """仓内所有插件根。评测一次挂全量，才是真实的跨域抢活现场。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if os.path.basename(dirpath) == ".claude-plugin" and "plugin.json" in filenames:
            out.append(os.path.dirname(dirpath))
    return sorted(out)


def collect(root, skill_filter, domain_filter):
    out = []
    for d in sorted(iter_skill_dirs(root)):
        name = os.path.basename(d)
        if skill_filter and name not in skill_filter:
            continue
        plugin = plugin_dir_of(d)
        if not plugin:
            continue
        if domain_filter and os.path.basename(plugin) not in domain_filter:
            continue
        data = load_case_file(d)
        if not data:
            continue
        out.append((name, d, plugin, data))
    return out


# ------------------------------------------------------------------- runner

def parse_skill_calls(stdout):
    """从 stream-json 里按顺序取出 Skill 调用名，以及 usage。"""
    picked, tokens = [], None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            content = (ev.get("message") or {}).get("content") or []
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" \
                        and block.get("name") == "Skill":
                    picked.append((block.get("input") or {}).get("skill"))
        elif ev.get("type") == "result":
            u = ev.get("usage") or {}
            total = (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
            tokens = total or None
    return picked, tokens


def run_claude(prompt, plugins, timeout, model, isolate=True):
    """一次独立会话。返回 (stdout, stderr, rc, 秒数)。

    isolate=True 时给本次会话一个空 `CLAUDE_CONFIG_DIR`：用户已安装在
    `~/.claude/skills` 的旧副本不会被看到——否则测的是上次安装的那份描述，
    改工作区里的 SKILL.md 对结果毫无影响（这个坑真的踩过）。
    鉴权走 `ANTHROPIC_AUTH_TOKEN` 等环境变量，不依赖配置文件。
    """
    cwd = tempfile.mkdtemp(prefix="skill-eval-")
    cfg = tempfile.mkdtemp(prefix="skill-eval-cfg-") if isolate else None
    cmd = ["claude", "-p", prompt,
           "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--allowedTools", ALLOW_TOOLS,
           "--disallowedTools", DENY_TOOLS]
    for p in plugins or []:
        cmd.extend(["--plugin-dir", p])
    if model:
        cmd.extend(["--model", model])
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    if cfg:
        env["CLAUDE_CONFIG_DIR"] = cfg
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", "timeout after %ss" % timeout, -9
    except FileNotFoundError:
        raise RuntimeError("找不到 `claude` 命令：评测需要本机 Claude Code 登录态")
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
        if cfg:
            shutil.rmtree(cfg, ignore_errors=True)
    return stdout, stderr, rc, time.time() - t0


def judge_answer(prompt, answer, expect, model, timeout):
    """可选：让模型给回答打分。返回 (score, reason)。"""
    rubric = (
        "你在做一次严格的评分。下面是用户问题、应命中的要点、以及助手回答。\n"
        "按 1~5 打分：5=要点齐全且无编造；4=关键要点齐全，少量细节缺失；"
        "3=方向对但缺关键要点；2=方向偏；1=答非所问或编造。\n"
        "只输出一行 JSON：{\"score\": <1-5>, \"missing\": \"<缺的要点，没有就空>\"}\n\n"
        "【用户问题】\n%s\n\n【应命中的要点】\n%s\n\n【助手回答】\n%s\n"
        % (prompt, expect, answer[:6000])
    )
    stdout, _, _, _ = run_claude(rubric, [], timeout, model)
    text = last_assistant_text(stdout)
    if not text.strip():
        return None, "judge 未返回"
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return int(obj.get("score")), str(obj.get("missing") or "")
    except (ValueError, KeyError, TypeError):
        return None, "judge 输出无法解析：%s" % text.strip()[:120]


def normalize_skill(name):
    """插件挂载的 skill 会带命名空间（`ops:op-bench`），判分只认 skill 名。"""
    return name.split(":", 1)[1] if name and ":" in name else name


def score_routing(skill, case, picked):
    """判一条用例的路由是否达标，返回 (ok, detail)。纯函数，不碰模型（便于单测）。

    true  用例：第一个 Skill 调用必须是被评 skill —— 第一个就错了，说明描述没把
                别的 skill 挡在外面；
    false 用例：本次会话不得调用被评 skill —— 形似问题应路由到别处，别抢活。
    picked 可以是原始名（`ops:op-bench`），判分前统一去掉插件前缀。
    """
    names = [normalize_skill(p) for p in picked]
    if case.get("expect"):
        first = names[0] if names else None
        return (first == skill,
                "期望拉起 %s，实际 %s" % (skill, first or "未拉起任何 skill"))
    ok = skill not in names
    return ok, "未拉起" if ok else "不应拉起 %s，实际 %s" % (skill, names)


def failure_reason(skill, case, picked):
    """把一次失败归类，返回 `未拉起` / `拉错` / `误拉`；通过返回空串。

    三种失败的表态完全不同：
      - 未拉起：本次会话一个 skill 都没加载 → 模型觉得通用知识够用，是**内容**问题；
      - 拉错：  加载了别的 skill 且第一个不是目标 → 邻居抢活，是**描述**问题；
      - 误拉：  false 用例里被评 skill 出现在调用列表 → 描述过宽。
    """
    names = [normalize_skill(p) for p in picked]
    if case.get("expect"):
        if names and names[0] == skill:
            return ""
        return "拉错" if names else "未拉起"
    return "误拉" if skill in names else ""


def forbidden_hit(case, answer):
    """返回答案里命中的第一条禁用说法；没有返回空串。

    `forbid` 写的是**错误说法本身**，不是关键词；允许 `re:` 前缀（正则），
    用来覆盖同一错话的不同措辞。
    命中片段本身或它前面 24 个字符里若有否定词（不/别/禁止/避免/勿/禁用…）则跳过，
    因为「不要用 sshpass -p」是正确说法。这是启发式：**宁可漏报，不可误报**——
    一个会冤枉正确答案的门，很快就没人看了。
    """
    text = answer or ""
    negation = ("不", "别", "禁止", "避免", "勿", "禁用", "严禁", "切莫", "不能", "杜绝", "忌")
    for phrase in case.get("forbid") or []:
        if not phrase:
            continue
        spans = []
        if phrase.startswith("re:"):
            try:
                spans = [(m.start(), m.end()) for m in re.finditer(phrase[3:], text)]
            except re.error:
                continue
        else:
            i = text.find(phrase)
            while i >= 0:
                spans.append((i, i + len(phrase)))
                i = text.find(phrase, i + 1)
        for start, end in spans:
            window = text[max(0, start - 24):end]
            if not any(neg in window for neg in negation):
                return phrase
    return ""


def aggregate_attempts(attempts):
    """把同一用例的 N 次尝试合成一行：全过 PASS、部分过 FLAKY、全败 FAIL。

    `ok` 只有全过才为真——单次通过不能洗白抖动，抖动本身就是信号。
    `reason` 取失败尝试里出现最多的那类，用来看「到底卡在哪」。
    """
    n = len(attempts)
    k = sum(1 for a in attempts if a["ok"])
    first = attempts[0]
    fails = [a for a in attempts if not a["ok"]]
    verdict = "PASS" if k == n else ("FLAKY" if k else "FAIL")
    reason = Counter(a.get("reason") or "" for a in fails).most_common(1)[0][0] if fails else ""
    detail = first["detail"] if verdict == "PASS" else fails[0]["detail"]
    return dict(id=first["id"], skill=first["skill"], kind=first["kind"],
                ok=k == n, verdict=verdict, passed=k, attempts=n, reason=reason,
                picked=first["picked"], detail=detail,
                sec=round(sum(a["sec"] for a in attempts), 1),
                tokens=sum(a["tokens"] or 0 for a in attempts) or None,
                prompt=first.get("prompt"), runs=attempts)


def run_case(skill, case, plugins, args):
    true_case = bool(case.get("expect"))
    cid = case.get("id")
    prompt = case.get("prompt") or ""
    stdout, stderr, rc, sec = run_claude(prompt, plugins, args.timeout, args.model,
                                         isolate=not args.no_isolate)
    picked, tokens = parse_skill_calls(stdout)
    ok, detail = score_routing(skill, case, picked)
    reason = "" if ok else failure_reason(skill, case, picked)
    answer = last_assistant_text(stdout)
    row = dict(id=cid, skill=skill, kind="true" if true_case else "false",
               ok=ok, picked=picked, sec=round(sec, 1), tokens=tokens,
               rc=rc, detail=detail, prompt=prompt, reason=reason,
               answer=answer[:800])
    if rc != 0 and not stdout:
        row["ok"] = False
        row["reason"] = "执行失败"
        row["detail"] = (stderr or "").strip()[:300] or "claude -p 失败"
    # forbid 只看路由已经对的情况：路由错了，先修路由。
    bad = forbidden_hit(case, answer) if (ok and true_case) else ""
    if bad:
        row["ok"] = False
        row["reason"] = "说错"
        row["detail"] = "答案命中禁用说法：%s" % bad
    if row["ok"] and true_case and args.judge:
        score, missing = judge_answer(prompt, answer, case.get("expect") or "",
                                      args.model, args.timeout)
        row["judge"] = dict(score=score, missing=missing)
        row["ok"] = bool(score is not None and score >= 4)
        if not row["ok"]:
            row["detail"] = "内容未达标（%s）：%s" % (score, missing)
    return row


def last_assistant_text(stdout):
    text = ""
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            for b in (ev.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "text":
                    text = b.get("text") or text
        elif ev.get("type") == "result" and ev.get("result"):
            text = ev["result"]
    return text


# --------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="跑 skill 的动态评测（真模型；会烧 token，默认不进 CI）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=REPO, help="包根（默认仓库根）")
    ap.add_argument("--skill", action="append", default=[],
                    help="只跑指定 skill（可重复）")
    ap.add_argument("--case", action="append", default=[],
                    help="只跑指定用例 id（可重复；改完描述验证单条失败时用）")
    ap.add_argument("--domain", action="append", default=[],
                    help="只跑指定插件域，如 ops / dcu-core（可重复）")
    ap.add_argument("--list", action="store_true", help="只列出带 evals 的 skill")
    ap.add_argument("--limit", type=int, default=0, help="最多跑多少个用例（0=不限）")
    ap.add_argument("--jobs", type=int, default=1, help="并发数（默认 1，避免限流）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="同一用例重复次数（默认 1）。>1 时报 k/N，用来看抖动"
                         "「稳定不拉」还是「时好时坏」")
    ap.add_argument("--timeout", type=int, default=300, help="单例超时秒数")
    ap.add_argument("--model", default="", help="传给 claude -p 的 --model")
    ap.add_argument("--judge", action="store_true", help="true 用例再跑一次内容评分（翻倍成本）")
    ap.add_argument("--json", default="", help="把明细写到 JSON")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要跑的用例")
    ap.add_argument("--no-isolate", action="store_true",
                    help="不隔离 CLAUDE_CONFIG_DIR（会看到 ~/.claude/skills 里的已安装副本，"
                         "测的就不是工作区这份了；仅用于复现用户现场）")
    args = ap.parse_args(argv)

    targets = collect(args.root, set(args.skill), set(args.domain))
    if not targets:
        print("没有找到带 evals/evals.json 的 skill（用 --list 看现状）", file=sys.stderr)
        return 2
    if args.list:
        for name, _d, plugin, data in targets:
            print("%-28s %-14s %2d 例  %s"
                  % (name, os.path.basename(plugin), len(data["cases"]),
                     (data.get("description") or "")[:60]))
        return 0

    # 一次挂全量插件：路由本来就发生在全库之间，只挂被测域测不出「被邻居抢走」。
    plugins = all_plugin_dirs(args.root)
    missing_plugin = sorted({os.path.basename(p) for n, _d, p, _x in targets
                             if p not in plugins})
    if missing_plugin:
        print("这些插件没有 .claude-plugin/plugin.json，挂不上：%s" % missing_plugin,
              file=sys.stderr)
        return 2

    jobs = []
    for name, _d, _p, data in targets:
        for case in data["cases"]:
            jobs.append((name, case))
    if args.case:
        want = set(args.case)
        jobs = [j for j in jobs if j[1].get("id") in want]
        if not jobs:
            print("没有匹配 --case 的用例：%s" % ", ".join(sorted(want)), file=sys.stderr)
            return 2
    if args.limit:
        jobs = jobs[:args.limit]

    if args.dry_run:
        for name, case in jobs:
            print("%-28s %-12s %s" % (name, case.get("id"), (case.get("prompt") or "")[:70]))
        print("\n共 %d 例 × %d 次（dry-run，未调用模型）"
              % (len(jobs), max(1, args.repeat)))
        return 0

    if shutil.which("claude") is None:
        print("找不到 `claude` 命令：动态评测需要本机 Claude Code", file=sys.stderr)
        return 2

    repeat = max(1, args.repeat)
    print("跑 %d 例 × %d 次，并发 %d，挂载 %d 个插件%s。每次尝试起一次独立会话（token 消耗见末尾）。\n"
          % (len(jobs), repeat, args.jobs, len(plugins),
             "" if args.no_isolate else "（隔离 ~/.claude/skills）"), flush=True)
    t0 = time.time()
    flat = [(n, c) for n, c in jobs for _ in range(repeat)]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        attempts = [f.result() for f in
                    [pool.submit(run_case, n, c, plugins, args) for n, c in flat]]
    # futures 按提交顺序取结果，同一用例的 N 次尝试是连续片段
    rows = [aggregate_attempts(attempts[i * repeat:(i + 1) * repeat])
            for i in range(len(jobs))]
    for i, row in enumerate(rows, 1):
        print("[%2d/%2d] %-5s %d/%d  %-22s %5.0fs  %s"
              % (i, len(rows), row["verdict"], row["passed"], row["attempts"],
                 row["id"], row["sec"], row["detail"]), flush=True)

    by_skill = {}
    for r in rows:
        by_skill.setdefault(r["skill"], []).append(r)
    print("\n%-28s %s" % ("skill", "通过/例"))
    for name in sorted(by_skill):
        rs = by_skill[name]
        print("%-28s %d/%d" % (name, sum(r["ok"] for r in rs), len(rs)))
    n_ok = sum(1 for r in rows if r["ok"])
    flaky = [r["id"] for r in rows if r["verdict"] == "FLAKY"]
    reasons = Counter(r["reason"] for r in rows if not r["ok"] and r["reason"])
    if reasons:
        print("失败归类：%s" % "，".join("%s %d" % (k, v) for k, v in reasons.most_common()))
    print("\n总计 %d/%d 例通过%s，耗时 %.0fs，token 合计 %s"
          % (n_ok, len(rows),
             ("（其中 %d 例抖动：%s）" % (len(flaky), "、".join(flaky))) if flaky else "",
             time.time() - t0, sum(r["tokens"] or 0 for r in rows) or "未知"))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           total=len(rows), passed=n_ok, repeat=repeat, rows=rows),
                      f, ensure_ascii=False, indent=1)
        print("明细 → %s" % args.json)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
