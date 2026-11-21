#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opmanifest — 执行身份冻结（T10 / P0-1，原 T2，缺陷 D9）。

问题
----
结论说「这个 kernel 是 219.9µs」。但没人能证明 219.9µs 是**这份源码编出来的产物**跑出来的。
三种错配都会让数字变成别的代码的，而且事后看不出来：

  1. 源码改了没重建        -> 测的是旧产物
  2. 产物被另一次构建覆盖  -> 测的是别人的产物
  3. 进程加载的不是这个产物 -> 测的是缓存里另一个

做法
----
把「源码 source / 编译产物 build / 进程实际加载的产物 loaded」三者都钉上 sha256 + mtime：
采集时 `record` 落一份 manifest.json，判定时 `validate` 复核。

契约只声明**要哪些 role、哪些 env 键**；路径由命令行显式给，不扫全盘、不把机器路径写进仓库。

刻意不做
--------
- **不猜路径**：`--artifact role=path` 显式给；glob 必须唯一匹配，否则报错而不是随便挑一个
  （`~/.tilelang/cache` 下有 1700+ 个 `executable.so`，猜必错）
- **不 fail-open**：本工具自身异常 -> 转成问题并按失败处理。
  反过来 `sys.exit(0)  # don't block if the check itself is broken` 会让判定门静默退化
  （坏了也放行、没人察觉，能悄悄退化一个月），我们不抄。
- **不验数字对不对**：只验「这份数字配这份代码」。数字本身归 opverify / op-roofline。

退出码
------
record   : 0=写入成功，1=参数或 IO 问题
validate : 0=一致，1=不一致或无法判定（问题逐条打印）
"""
import argparse
import datetime
import glob as globmod
import hashlib
import json
import os
import socket
import sys

SCHEMA = "opmanifest/v1"
ROLES = ("source", "build", "loaded")

# 这些值等于「没记录」。列出来是为了让校验能明确说「这是占位符」而不是「字段缺失」。
PLACEHOLDERS = {"", "unknown", "unavailable", "n/a", "na", "none", "null", "nil",
                "-", "tbd", "todo", "unset"}

# 源码比产物新多少秒以内不算违规（文件系统 mtime 抖动 + 同秒重建）
MTIME_GRACE_S = 1.0


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def is_placeholder(v):
    """None / unknown / unavailable(<...>) / <模板> 都算「没记录」。"""
    if v is None:
        return True
    if not isinstance(v, str):
        return False
    s = v.strip().lower()
    if s in PLACEHOLDERS:
        return True
    if s.startswith("unavailable("):
        return True
    if s.startswith("<") and s.endswith(">"):
        return True
    return False


def parse_artifact_spec(spec):
    """把 `role=path` 或 `role=glob` 展开成一个 artifact 记录。glob 必须唯一匹配。"""
    if "=" not in spec:
        raise ValueError("--artifact 需要 role=path 形式，收到 %r" % (spec,))
    role, path = spec.split("=", 1)
    role, path = role.strip(), path.strip()
    if role not in ROLES:
        raise ValueError("未知 role %r；只认 %s" % (role, ", ".join(ROLES)))
    if not path:
        raise ValueError("role=%s 的路径为空" % role)
    path = os.path.expanduser(os.path.expandvars(path))
    if any(ch in path for ch in "*?["):
        hits = sorted(globmod.glob(path, recursive=True))
        if len(hits) != 1:
            raise ValueError(
                "role=%s 的 glob 匹配到 %d 个文件，必须正好 1 个（不许随便挑）：%s"
                % (role, len(hits), path))
        path = hits[0]
    if not os.path.isfile(path):
        raise ValueError("role=%s 的文件不存在：%s" % (role, path))
    st = os.stat(path)
    return {"role": role, "path": os.path.abspath(path),
            "sha256": sha256_file(path), "bytes": st.st_size, "mtime": st.st_mtime}


def load_env_from_json(path):
    """从 bench.json 之类的报告里取 env 对象。取不到就报错，不静默放过。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    env = d.get("env") if isinstance(d, dict) else None
    if not isinstance(env, dict):
        raise ValueError("%s 里没有 env 对象（顶层需要 \"env\": {...}）" % path)
    return env


def artifacts_from_bench(path, variant):
    """从 bench.json 的 `results[<variant>].artifacts` 取产物路径，展开成 `role=path` 列表。

    这些路径是 bench 进程在**首次调用之后当场抓的**（tilelang 从 JITImpl._kernel_cache
    的 adapter.libpath，其它变体从 /proc/self/maps 增量），不是事后猜的。
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    results = d.get("results") if isinstance(d, dict) else None
    if not isinstance(results, dict):
        raise ValueError("%s 里没有 results 对象" % path)
    if variant not in results:
        raise ValueError("%s 里没有 results[%s]（有：%s）"
                         % (path, variant, ", ".join(sorted(results)) or "无"))
    art = results[variant].get("artifacts")
    if not isinstance(art, dict):
        raise ValueError("results[%s] 里没有 artifacts —— 该 bench 脚本没做执行身份采集" % variant)
    specs = []
    for role in ROLES:
        for p in (art.get(role) or []):
            specs.append("%s=%s" % (role, p))
    if not specs:
        raise ValueError("results[%s].artifacts 是空的：%s"
                         % (variant, art.get("loaded_note") or "bench 没抓到产物路径"))
    return specs


def validate_manifest(man, required_roles=None, env_keys=None):
    """返回问题清单（空 = 一致）。任何内部异常都转成问题，**不 fail-open**。"""
    probs = []
    try:
        if not isinstance(man, dict):
            return ["manifest 不是 JSON 对象"]
        if man.get("schema") != SCHEMA:
            probs.append("manifest.schema=%r，期望 %s" % (man.get("schema"), SCHEMA))

        arts = man.get("artifacts")
        if not isinstance(arts, list) or not arts:
            return probs + ["manifest.artifacts 为空：没有冻结任何产物"]

        # --- 1. 每个声明的产物必须存在，且 sha256 未变 -------------------------
        live = []
        for i, it in enumerate(arts):
            if not isinstance(it, dict):
                probs.append("artifacts[%d] 不是对象" % i)
                continue
            role, path, sha = it.get("role"), it.get("path"), it.get("sha256")
            if not role or not path or not sha:
                probs.append("artifacts[%d] 缺 role/path/sha256 之一" % i)
                continue
            if not os.path.isfile(path):
                probs.append("%s 不存在了（记录于 %s）：%s"
                             % (role, man.get("recorded_at"), path))
                continue
            now = sha256_file(path)
            if now != sha:
                probs.append("%s 的 sha256 与记录不符：记录 %s… 现在 %s…（%s）"
                             % (role, sha[:12], now[:12], path))
            live.append({"role": role, "path": path, "mtime": os.stat(path).st_mtime})

        # --- 2. 契约要求的 role 必须齐 ---------------------------------------
        declared = {it.get("role") for it in arts if isinstance(it, dict)}
        for r in (required_roles or []):
            if r not in declared:
                probs.append("缺少 role=%s 的产物（契约要求 %s）"
                             % (r, ", ".join(required_roles)))

        # --- 3. 源码不得比产物新（改完没重建）---------------------------------
        srcs = [x for x in live if x["role"] == "source"]
        outs = [x for x in live if x["role"] in ("build", "loaded")]
        if srcs and outs:
            newest_src = max(srcs, key=lambda x: x["mtime"])
            oldest_out = min(outs, key=lambda x: x["mtime"])
            gap = newest_src["mtime"] - oldest_out["mtime"]
            if gap > MTIME_GRACE_S:
                probs.append(
                    "源码比产物新 %.1fs：%s 晚于 %s —— 疑似改完没重建"
                    % (gap, os.path.basename(newest_src["path"]),
                       os.path.basename(oldest_out["path"])))

        # --- 4. 环境键必须齐且不是占位符 --------------------------------------
        env = man.get("env") or {}
        if not isinstance(env, dict):
            probs.append("manifest.env 不是对象")
            env = {}
        for k in (env_keys or []):
            if k not in env:
                probs.append("env.%s 缺失（结论不可回溯到运行环境）" % k)
            elif is_placeholder(env[k]):
                probs.append("env.%s=%r 是占位符，不算已记录" % (k, env[k]))
    except Exception as e:  # 门禁自己坏了也必须失败关闭
        probs.append("校验过程异常（按不一致处理，不 fail-open）：%s: %s"
                     % (type(e).__name__, e))
    return probs


def contract_requirements(contract_path):
    """从 contract.yaml 的 manifest 段取 (required_roles, env_keys)。

    复用 opverify 的 YAML 安全子集解析器，避免第二份 YAML 实现（一个概念一个权威点）。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import opverify  # noqa: E402
    with open(contract_path, "r", encoding="utf-8") as f:
        c = opverify.parse_yaml(f.read())
    mf = (c.get("manifest") or {}) if isinstance(c, dict) else {}
    roles = mf.get("required_roles") or []
    keys = mf.get("env_keys") or []
    return (list(roles) if isinstance(roles, list) else [],
            list(keys) if isinstance(keys, list) else [])


# ------------------------------------------------------------------ 命令

def cmd_record(a):
    specs = list(a.artifact or [])
    if a.from_bench:
        if not a.variant:
            raise SystemExit("--from-bench 必须同时给 --variant")
        try:
            specs += artifacts_from_bench(a.from_bench, a.variant)
        except Exception as e:
            raise SystemExit("读 --from-bench 失败：%s" % e)
    arts = []
    seen = set()
    for spec in specs:
        art = parse_artifact_spec(spec)
        if (art["role"], art["path"]) in seen:
            continue
        seen.add((art["role"], art["path"]))
        arts.append(art)
    if not arts:
        raise SystemExit("至少给一个 --artifact role=path，或 --from-bench")

    env = {}
    for p in (a.env_json or []):
        try:
            env.update(load_env_from_json(p))
        except Exception as e:
            raise SystemExit("读 --env-json 失败：%s" % e)
    for kv in (a.env or []):
        if "=" not in kv:
            raise SystemExit("--env 需要 k=v 形式，收到 %r" % (kv,))
        k, v = kv.split("=", 1)
        env[k.strip()] = v.strip()

    man = {
        "schema": SCHEMA,
        "op": a.op,
        "shape": a.shape,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "artifacts": arts,
        "roles": sorted({x["role"] for x in arts}),
        "env": env,
        "note": a.note or "",
    }
    out = os.path.abspath(os.path.expanduser(a.out))
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("已记录 %s（%d 个产物，role=%s）"
          % (out, len(arts), ",".join(man["roles"])))
    return 0


def cmd_validate(a):
    roles, keys = (None, None)
    if a.contract:
        roles, keys = contract_requirements(a.contract)
    if a.required_role:
        roles = list(a.required_role)
    if a.env_keys is not None:
        keys = [x.strip() for x in a.env_keys.split(",") if x.strip()]

    try:
        with open(a.manifest, "r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception as e:
        print("无法读取 manifest %s：%s: %s" % (a.manifest, type(e).__name__, e))
        print("判定：执行身份未冻结（失败关闭）")
        return 1

    probs = validate_manifest(man, roles, keys)
    if a.json:
        print(json.dumps({"ok": not probs, "problems": probs,
                          "required_roles": roles or [], "env_keys": keys or [],
                          "manifest": a.manifest}, ensure_ascii=False, indent=2))
    else:
        if probs:
            print("执行身份不一致（%d 项）：" % len(probs))
            for p in probs:
                print("  - " + p)
        else:
            print("执行身份一致：%d 个产物的 sha256 与 mtime 链未断裂"
                  % len(man.get("artifacts") or []))
    return 0 if not probs else 1


def main(argv=None):
    p = argparse.ArgumentParser(
        description="opmanifest — 冻结/复核「源码→产物→加载产物」执行身份链",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="采集时刻记录 manifest.json")
    r.add_argument("--out", required=True, help="输出 manifest.json 路径")
    r.add_argument("--artifact", action="append", metavar="ROLE=PATH",
                   help="可重复；ROLE ∈ source|build|loaded；PATH 可含 glob（须唯一匹配）")
    r.add_argument("--from-bench", metavar="PATH",
                   help="从 bench.json 的 results[<variant>].artifacts 取 source/build/loaded")
    r.add_argument("--variant", default=None, help="配合 --from-bench 指定变体名")
    r.add_argument("--env-json", action="append", metavar="PATH",
                   help="从该 json 的顶层 env 对象取环境指纹（如 bench.json）")
    r.add_argument("--env", action="append", metavar="K=V", help="补充/覆盖环境键")
    r.add_argument("--op", default=None)
    r.add_argument("--shape", default=None)
    r.add_argument("--note", default=None)
    r.set_defaults(func=cmd_record)

    v = sub.add_parser("validate", help="复核 manifest 是否仍与磁盘一致")
    v.add_argument("--manifest", required=True)
    v.add_argument("--contract", help="从该契约的 manifest 段取 required_roles / env_keys")
    v.add_argument("--required-role", action="append", metavar="ROLE")
    v.add_argument("--env-keys", default=None, help="逗号分隔，覆盖契约里的 env_keys")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_validate)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except ValueError as e:
        print("参数错误：%s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
