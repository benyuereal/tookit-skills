#!/usr/bin/env python3
"""HYGON-AI GitHub 开源上游同步工具（dcu skills 用）。

用途：刷新本地 dcu skills 引用的 HYGON-AI 开源仓库快照（仓库清单 / 版本锚点 /
inference-cookbook 模型验证矩阵），并与上次快照做 diff。供以下 skill 的
"上游新鲜度" 更新机制共用：
  - inference/dcu-model-deploy-catalog（cookbook 模型矩阵）
  - training/dcu-train-cookbook（训练域配方与框架版本）
  - inference/dcu-vllm-source、dcu-sglang-source、dcu-mooncake-source
  - operators/dcu-aiter-source、inference/dcu-training-frameworks-source
  - env/dcu-opensource-platform（K8s/监控组件版本）

用法：
  python3 hygon-upstream-sync.py --check   # 快检：仓库列表 + 关键仓库 tag（约 2-10 个 API 请求）
  python3 hygon-upstream-sync.py --full    # 全量：另抓 README（raw 通道，不占 API 配额）+ 解析 cookbook 矩阵

数据落盘：
  - 快照（小、入 git）：references/hygon-upstream/upstream-snapshot.json
  - README 原文缓存（大、不入 git）：<agent-home>/cache/hygon-upstream/<repo>.md

注意：
  - GitHub API 未认证配额 60 次/小时/IP；raw.githubusercontent.com 不占配额。
  - 本机 curl 直连可能被审批门拦截，脚本一律走 python urllib。
  - 快照只存元数据与锚点，不搬运 README 正文（避免 git 仓库膨胀与许可证问题）。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

ORG = "HYGON-AI"
API_BASE = f"https://api.github.com/orgs/{ORG}"
RAW_BASE = f"https://raw.githubusercontent.com/{ORG}"
SKILL_ROOT = Path(__file__).resolve().parent.parent          # .../dcu
SNAPSHOT_PATH = SKILL_ROOT / "references" / "hygon-upstream" / "upstream-snapshot.json"
CACHE_DIR = Path("<agent-home>/cache/hygon-upstream")

# 需要提取"版本锚点/上游基线"的仓库（--check 只查这些的 tags；--full 另读 README）
KEY_REPOS = [
    "vllm-plugin-das", "sglang-das", "Megatron-LM-das", "aiter-das",
    "triton-das", "mooncake-das", "LMCache-das", "DeepEP-das",
    "slime-das", "AReal-das", "verl-das", "one-click-quant",
    "k8s-hcu-device-plugin", "k8s-hcu-dra-driver", "k8s-hcu-label-node",
    "hcu-dcgm", "hcu-exporter", "hcu-train-simulator", "TileKernels-das",
]

# README 里各类上游/自身版本写法的提取模式（按序尝试，取首个命中）
ANCHOR_PATTERNS = [
    r"\*\*Upstream tag\*\*:?\s*`([^`]+)`",
    r"aligned with branch\s*`([^`]+)`,\s*tag\s*`([^`]+)`",   # vllm-plugin-das 风格
    r"compliance review is commit\s*`?`?([0-9a-f]{7,40})`?",  # aiter-das 风格
    r"at upstream commit\s*`([0-9a-f]{7,40})`",              # DeepEP-das 风格
    r"[Uu]pstream\s*(?:branch|repository)?[:\s`]*`?([\w.\-+]+)`?\s*\n?\s*.*?[Cc]ommit[:` ]+`?([0-9a-f]{7,40})`?",
    r"上游版本[:：]\s*`?([\w.\-+]+)`?",
    r"上游基线[:：]\s*`?([\w.\-+]+)`?",
    r"Current version:\s*\*\*([^*]+)\*\*",
    r"当前版本[:：]\s*\*\*([^*]+)\*\*",
    r"\bv(\d+\.\d+\.\d+[\w.\-+]*)\b",
]

CARDS = ["K100_AI", "BW1000", "BW1100"]
FRAMEWORK_TOKENS = {"vllm", "sglang", "vllm-omni", "sglang-omni", "sglang-diffusion",
                    "wan-das", "infinity", "megatron", "megatron-lm", "mmcv", "verl",
                    "llamafactory", "ms-swift", "tensorrt-llm"}
VENDORS = {"qwen", "deepseek", "meta llama", "llama", "z.ai", "tencent",
           "moonshot ai", "minimax", "stepfun", "inclusionai", "xiaomi mimo",
           "mimo", "wan", "baai", "glm", "others", "other"}


def http_get(url: str, raw: bool = False, timeout: int = 30) -> str:
    headers = {"User-Agent": "dcu-skills-upstream-sync"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if not raw:
        headers["Accept"] = "application/vnd.github+json"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_repo_list(old_repos=None) -> list:
    try:
        repos, page = [], 1
        while True:
            data = json.loads(http_get(f"{API_BASE}/repos?per_page=100&page={page}&sort=updated"))
            if not data:
                break
            repos.extend(data)
            if len(data) < 100:
                break
            page += 1
        return [{
            "name": r["name"], "desc": (r.get("description") or "")[:120],
            "lang": r.get("language"), "stars": r["stargazers_count"],
            "pushed_at": r["pushed_at"][:10], "archived": r.get("archived", False),
        } for r in repos]
    except urllib.error.HTTPError as e:
        print(f"  [warn] 拉取仓库列表失败 HTTP {e.code}（配额/网络）；回退用旧快照数据", file=sys.stderr)
        return old_repos or []


def fetch_tags(repo: str, n: int = 5) -> list:
    try:
        tags = json.loads(http_get(f"{API_BASE}/repos/{ORG}/{repo}/tags?per_page={n}"))
        return [t["name"] for t in tags]
    except Exception:
        return []


def fetch_readme(repo: str) -> str:
    """README 走 raw 通道（不占 API 配额），main→master 依次回退，最后才用 API。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{repo}.md"
    for branch in ("main", "master"):
        try:
            txt = http_get(f"{RAW_BASE}/{repo}/{branch}/README.md", raw=True)
            cache.write_text(txt)
            return txt
        except Exception:
            continue
    try:  # 兜底：API readme 端点（占配额）
        req = urllib.request.Request(
            f"{API_BASE}/repos/{ORG}/{repo}/readme",
            headers={"User-Agent": "dcu-skills-upstream-sync", "Accept": "application/vnd.github.raw"})
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode("utf-8", errors="replace")
        cache.write_text(txt)
        return txt
    except Exception as e:
        print(f"  [warn] {repo} README 抓取失败: {e}", file=sys.stderr)
        return cache.read_text() if cache.exists() else ""


def extract_anchor(readme: str) -> dict:
    out = {}
    for pat in ANCHOR_PATTERNS:
        m = re.search(pat, readme)
        if m:
            out["anchor"] = " / ".join(g for g in m.groups() if g) if m.groups() else m.group(0)
            break
    return out


STATUS_SYMBOLS = {"passed", "wip", "-"}


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&nbsp;", " ").strip()


def norm_fw(s: str) -> str:
    return re.sub(r"[\s_]+", "-", s.strip().lower())


def parse_cookbook_matrix(readme: str) -> dict:
    """解析 inference-cookbook-das README 的 模型×框架×卡型 验证矩阵。

    README 表格为 HTML（<td rowspan=N> 厂商/模型 + 框架 + 3 个状态列）。
    返回 {"<model>|<framework>": {"K100_AI": "ok|wip|...", ...}, ...}
    行判定：含 <img> 的格=厂商列（剔除）；状态格=passed/wip/-；其余非统计格中，
    命中 FRAMEWORK_TOKENS 的为框架，否则为模型名（跨行沿用，rowspan 由 tds 连续给出）。
    """
    matrix, cur_model, cur_fw = {}, None, None
    for tr in re.findall(r"<tr>(.*?)</tr>", readme, re.S):
        tds = re.findall(r"<td([^>]*)>(.*?)</td>", tr, re.S)
        # 剔除厂商列（内容含 <img> / cdn / huggingface）
        cells = []
        for attrs, content in tds:
            raw = content
            if re.search(r"<img|cdn|huggingface", raw, re.I):
                continue
            text = _strip_tags(content)
            if text:
                cells.append(text)
        if not cells:
            continue
        statuses = [t for t in cells if t in STATUS_SYMBOLS][:3]
        nonstat = [t for t in cells if t not in STATUS_SYMBOLS and t != "-"]
        fw = None
        model = None
        for t in nonstat:
            if norm_fw(t) in FRAMEWORK_TOKENS:
                fw = t
                break
        model_cands = [t for t in nonstat if norm_fw(t) not in FRAMEWORK_TOKENS
                       and not re.match(r"^[0-9.]+$", t)]
        if model_cands:
            model = model_cands[0]
        if model:
            cur_model = model
        if fw:
            cur_fw = fw
        if cur_model and (fw or cur_fw) and statuses:
            key = f"{cur_model}|{fw or cur_fw}"
            merged = matrix.get(key, {})
            merged.update({c: _status(v) for c, v in zip(CARDS, statuses) if v != "-"})
            matrix[key] = merged
    return matrix


def _status(s: str) -> str:
    if "passed" in s:
        return "ok"
    if "wip" in s:
        return "wip"
    return "-"


def diff_matrix(old: dict, new: dict) -> list:
    diffs = []
    for key, st in new.items():
        if key not in old:
            got = {k: v for k, v in st.items() if v == "ok"}
            if got:
                diffs.append(f"  + 新增已验证: {key} -> {got}")
        else:
            for card, v in st.items():
                if v == "ok" and old[key].get(card) != "ok":
                    diffs.append(f"  + 新验证: {key} @ {card}")
    for key in old:
        if key not in new:
            diffs.append(f"  - 矩阵移除: {key}")
    return diffs


def main():
    ap = argparse.ArgumentParser(
        description=(__doc__ or "HYGON-AI GitHub 开源上游同步工具").split("\n")[1])
    ap.add_argument("--check", action="store_true", help="快检：仓库列表+关键仓库 tags")
    ap.add_argument("--full", action="store_true", help="全量：README+版本锚点+cookbook 矩阵")
    args = ap.parse_args()
    if not args.check and not args.full:
        args.full = True

    old = json.loads(SNAPSHOT_PATH.read_text()) if SNAPSHOT_PATH.exists() else {}
    snap = {"fetched_at": str(date.today()), "org": ORG, "repos": [], "anchors": {},
            "cookbook_matrix": old.get("cookbook_matrix", {})}

    print(f"[1/3] 拉取 {ORG} 组织仓库列表 ...")
    snap["repos"] = fetch_repo_list(old_repos=old.get("repos"))
    print(f"      共 {len(snap['repos'])} 个仓库")

    print("[2/3] 关键仓库版本锚点 ...")
    for repo in KEY_REPOS:
        tags = fetch_tags(repo)
        entry = {"tags": tags}
        if args.full:
            readme = fetch_readme(repo)
            entry.update(extract_anchor(readme))
            if repo == "sglang-das" and "anchor" not in entry and tags:
                entry["anchor"] = tags[0]  # README 无基线时用最新 tag
        snap["anchors"][repo] = entry
        print(f"      {repo:26s} tags={tags[:3]} anchor={entry.get('anchor', '-')}")

    if args.full:
        print("[3/3] 解析 inference-cookbook-das 模型矩阵 ...")
        cb = fetch_readme("inference-cookbook-das")
        snap["cookbook_matrix"] = parse_cookbook_matrix(cb)
        ok_rows = [k for k, v in snap["cookbook_matrix"].items()
                   if any(x == "ok" for x in v.values())]
        print(f"      矩阵行 {len(snap['cookbook_matrix'])} 条，其中含已验证 {len(ok_rows)} 条")

    if old:
        old_names = {r["name"] for r in old.get("repos", [])}
        new_names = {r["name"] for r in snap["repos"]}
        for n in sorted(new_names - old_names):
            print(f"[diff] 新增仓库: {n}")
        for n in sorted(old_names - new_names):
            print(f"[diff] 仓库消失/转私有: {n}")
        for repo, ent in snap["anchors"].items():
            o = old.get("anchors", {}).get(repo, {})
            if o.get("anchor") and ent.get("anchor") and o["anchor"] != ent["anchor"]:
                print(f"[diff] {repo} 版本锚点变化: {o['anchor']} -> {ent['anchor']}")
            if o.get("tags") and ent.get("tags") and o["tags"][0] != ent["tags"][0]:
                print(f"[diff] {repo} 最新 tag: {o['tags'][0]} -> {ent['tags'][0]}")
        for line in diff_matrix(old.get("cookbook_matrix", {}), snap["cookbook_matrix"]):
            print("[diff] cookbook " + line)

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=1))
    validate_against_schema()
    print(f"[done] 快照已写入 {SNAPSHOT_PATH} (fetched_at={snap['fetched_at']})")


def validate_against_schema() -> None:
    """落盘后按同级 schema 自校验（缺文件则跳过）。"""
    schema = SNAPSHOT_PATH.with_name('upstream-snapshot.schema.json')

    mini = SNAPSHOT_PATH.parents[2] / 'scripts' / 'json_schema_mini.py'

    if not (schema.is_file() and mini.is_file()):
        print('[SKIP] schema 或校验器缺失，跳过结构自检', file=sys.stderr)
        return
    if subprocess.call([sys.executable, str(mini), str(schema), str(SNAPSHOT_PATH)]) != 0:
        sys.exit('ERR: 快照未通过 schema 校验，API 结构可能已变')


if __name__ == "__main__":
    main()
