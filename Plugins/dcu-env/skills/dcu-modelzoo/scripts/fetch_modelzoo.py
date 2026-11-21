#!/usr/bin/env python3
"""抓取光合开发者社区 ModelZoo（GitLab group modelzoo）全量模型适配仓库清单 + 元数据。

数据源（2026-09-02 逆向验证，匿名可访问，GitLab API v4）：
  - GET /codes/api/v4/groups/modelzoo/projects?per_page=100&page=N  （X-Total 翻页，622 个 @2026-09-02）
  - GET /codes/api/v4/projects/modelzoo%2F<path>/repository/files/model.properties/raw?ref=<default_branch>
    （机器可读元数据：modelCode/modelName/processType 推理|训练/appCategory/frameType vllm,sglang/accelerateType BW1000,K100AI）
  - GET /codes/api/v4/projects/modelzoo%2F<path>/repository/files/README.md/raw?ref=<branch>
    （标准模板：环境依赖 DTK/框架版本表 + 推荐镜像 + 支持的 DCU 型号/最低卡数 + 推理启动命令）

用法:
  python3 fetch_modelzoo.py [输出.json] [--with-readme]
    默认只抓 projects 列表 + model.properties（快，~2min）
    --with-readme 额外抓每个仓库 README.md 并提取 DTK/镜像/卡型号行（慢，~8min，8 并发）

输出 JSON: {fetched_at, total, projects:[{path,name,description,default_branch,web_url,
last_activity,star_count, props:{...}, readme_hints:{dtks, images, cards}[]}]}
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

BASE = "https://developer.sourcefind.cn/codes/api/v4"
UA = {"User-Agent": "Mozilla/5.0"}


def http_get(url: str, retries: int = 3, binary: bool = False):
    last: Optional[Exception] = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            r = urllib.request.urlopen(req, timeout=30)
            data = r.read()
            return (data if binary else data.decode("utf-8", "replace")), r.headers
        except Exception as e:  # noqa: BLE001 - 网络重试
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} retries: {url}") from last


def get_json(url: str):
    data, _ = http_get(url)
    return json.loads(data)


def list_all_projects():
    """翻页拉全量 group 项目（X-Total 头控制终止）。"""
    out, page = [], 1
    while True:
        d, headers = http_get(
            f"{BASE}/groups/modelzoo/projects?per_page=100&page={page}"
            f"&order_by=last_activity_at&sort=desc"
        )
        items = json.loads(d)
        out += items
        total = int(headers.get("X-Total") or 0)
        print(f"  page {page}: +{len(items)} (total {len(out)}/{total})", file=sys.stderr)
        if not items or len(out) >= total:
            return out
        page += 1
        time.sleep(0.3)


def parse_properties(text: str) -> dict:
    props = {}
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            props[k.strip()] = v.strip()
    return props


def fetch_one(p: dict, with_readme: bool) -> dict:
    full_path = p["path_with_namespace"]  # 2026-10-14: 存全串（modelzoo/<子组>/<repo>），防 group 改名与短名撞车；快照 diff/合并一律以 full path 为键
    path = full_path.split("/", 1)[1]
    ref = p.get("default_branch") or "master"
    enc = urllib.parse.quote(path, safe="")
    item: dict = {
        "path": path,
        "path_with_namespace": full_path,
        "name": p.get("name"),
        "description": (p.get("description") or "").strip(),
        "default_branch": ref,
        "web_url": p.get("web_url"),
        "last_activity": (p.get("last_activity_at") or "")[:10],
        "star_count": p.get("star_count", 0),
    }
    # model.properties
    try:
        t, _ = http_get(f"{BASE}/projects/modelzoo%2F{enc}/repository/files/model.properties/raw?ref={ref}")
        item["props"] = parse_properties(t)
    except RuntimeError as e:
        if "404" in str(e) or "403" in str(e):
            item["props"] = {}
        else:
            item["props"] = {}
            item["props_error"] = str(e)[:80]
    # README hints（可选，慢）
    hints: dict = {"dtks": [], "images": [], "cards": [], "readme_error": False}
    if with_readme:
        try:
            t, _ = http_get(f"{BASE}/projects/modelzoo%2F{enc}/repository/files/README.md/raw?ref={ref}")
            hints["dtks"] = sorted(set(re.findall(r"DTK\s*\|?\s*([0-9]{2}\.[0-9]{2}(?:\.[0-9]+)?(?:-rc[0-9]+)?)", t)))
            hints["images"] = sorted(set(re.findall(r"(?:image\.sourcefind\.cn:5000|harbor\.sourcefind\.cn:5443)/[A-Za-z0-9_./:-]+", t)))
            hints["cards"] = sorted(set(re.findall(r"(K100AI|BW1000|BW1100|BW1101|K100|Z100|Z800)", t)))
        except RuntimeError:
            hints["readme_error"] = True
        item["readme_hints"] = hints
    return item


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "modelzoo_index.json"
    with_readme = "--with-readme" in sys.argv

    print("listing modelzoo projects...", file=sys.stderr)
    projects = list_all_projects()
    print(f"projects: {len(projects)}; fetching properties{' + readme' if with_readme else ''} (8 workers)...", file=sys.stderr)

    items = []
    with ThreadPoolExecutor(8) as ex:
        for i, item in enumerate(ex.map(lambda p: fetch_one(p, with_readme), projects), 1):
            items.append(item)
            if i % 100 == 0:
                print(f"  {i}/{len(projects)}", file=sys.stderr)

    json.dump(
        {
            "fetched_at": time.strftime("%Y-%m-%d"),
            "source": BASE + "/groups/modelzoo/projects",
            "total": len(items),
            "projects": items,
        },
        open(out_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    n_prop = sum(1 for i in items if i.get("props"))
    print(f"saved {out_path}: {len(items)} projects, {n_prop} with model.properties", file=sys.stderr)
    validate_against_schema(out_path)


def validate_against_schema(out_path: str) -> None:
    """落盘后按 data/modelzoo_index.schema.json 自校验（包根零依赖 mini 校验器；缺文件则跳过）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    schema = os.path.join(here, "..", "data", "modelzoo_index.schema.json")
    mini = os.path.join(here, "..", "..", "..", "scripts", "json_schema_mini.py")
    if not (os.path.isfile(schema) and os.path.isfile(mini)):
        print("SKIP: schema 或 mini 校验器缺失，跳过结构校验", file=sys.stderr)
        return
    if subprocess.call([sys.executable, mini, schema, out_path]) != 0:
        sys.exit("ERR: 快照未通过 schema 校验，请检查 GitLab API 结构是否变化")


if __name__ == "__main__":
    main()
