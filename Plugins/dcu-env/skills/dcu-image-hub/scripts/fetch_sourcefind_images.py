#!/usr/bin/env python3
"""抓取 developer.sourcefind.cn 光合开发者社区 DCU 镜像仓库全量清单 + 版本 tag。

数据源（2026-09-02 逆向验证，匿名可访问）：
  - GET https://developer.sourcefind.cn/api-repo/image/ListImage?Page=1&PageSize=200
  - GET https://developer.sourcefind.cn/api-repo/image/ListTag?ImageUUID=<uuid>&Page=1&PageSize=500

用法:
  python3 fetch_sourcefind_images.py [输出.json]

输出: 精简 JSON（镜像元信息 + 全部 tag: name/size/pull_url/push_time/release_type）
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Optional

BASE = "https://developer.sourcefind.cn/api-repo/image"
UA = {"User-Agent": "Mozilla/5.0"}


def get(url: str, retries: int = 3) -> dict:
    last: Optional[Exception] = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            data: dict = json.load(urllib.request.urlopen(req, timeout=30))
            return data
        except Exception as e:  # noqa: BLE001 - 网络重试
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} retries: {url}") from last


def main():
    out_path = sys.argv[1] if len(sys.argv) == 2 else "sourcefind_dcu_images.json"
    # 项目过滤：默认全抓（dcu/hcu/cpu 全要 —— hcu 项目下有独立维护的镜像线，
    # 如 vllm0.28.1rc1 只发在 hcu/admin/base/custom，2026-10-12 实测漏 hcu 导致漏版本）
    project_filter = sys.argv[2] if len(sys.argv) > 2 else None

    d = get(f"{BASE}/ListImage?Page=1&PageSize=200")
    assert d.get("code") == "success", d.get("code")
    images = d["data"]["list"]
    total = d["data"]["total"]
    print(f"images: {len(images)}/{total}")

    result = []
    for img in images:
        if project_filter and img["ProjectName"] != project_filter:
            continue
        uuid = img["UUID"]
        t = get(f"{BASE}/ListTag?ImageUUID={uuid}&Page=1&PageSize=500")
        tags = t.get("data", {}).get("list") or []
        tag_list = [
            {
                "name": x["Name"],
                "size": x.get("Size"),
                "pull_url": x.get("PullUrl"),
                "push_time": (x.get("PushTime") or "")[:10],
                "release_type": x.get("ReleaseType"),
            }
            for x in tags
        ]
        # push 时间倒序 = 最新在前
        tag_list.sort(key=lambda x: x["push_time"], reverse=True)
        result.append(
            {
                "id": img["ID"],
                "uuid": uuid,
                "name": img["Name"],
                "repo": img["RepoName"],
                "project": img["ProjectName"],
                "accelerator": img["Accelerator"],
                "abstract": img.get("Abstract") or "",
                "pull_count": img.get("PullCount"),
                "updated": (img.get("UpdatedAt") or "")[:10],
                "detail_url": f"https://developer.sourcefind.cn/servicelist/detail?uuid={uuid}",
                "tags": tag_list,
            }
        )
        print(f"  [{img['ProjectName']}] {img['RepoName']}: {len(tag_list)} tags")
        time.sleep(0.2)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.strftime("%Y-%m-%d"), "source": BASE, "images": result},
                  f, ensure_ascii=False, indent=1)
    n_tags = sum(len(r["tags"]) for r in result)
    print(f"saved {out_path}: {len(result)} images, {n_tags} tags")
    validate_against_schema(out_path)


def validate_against_schema(out_path: str) -> None:
    """落盘后按 data/sourcefind_dcu_images.schema.json 自校验（包根零依赖 mini 校验器；缺文件则跳过）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    schema = os.path.join(here, "..", "data", "sourcefind_dcu_images.schema.json")
    mini = os.path.join(here, "..", "..", "..", "scripts", "json_schema_mini.py")
    if not (os.path.isfile(schema) and os.path.isfile(mini)):
        print("SKIP: schema 或 mini 校验器缺失，跳过结构校验")
        return
    if subprocess.call([sys.executable, mini, schema, out_path]) != 0:
        sys.exit("ERR: 快照未通过 schema 校验，请检查 API 结构是否变化")


if __name__ == "__main__":
    main()
