# -*- coding: utf-8 -*-
"""`engine/_loader.py` 与 `card.load_candidates` 缓存的用例。

锁三件事：
  1. 加载成功走 `sys.modules` 缓存（同一 key 返回同一对象，不重复 exec）；
  2. 加载**失败**的模块不留缓存 —— 第一次抛错后，修好文件第二次能重试成功
     （审计指出的脏模块 bug：失败后缓存里留着执行了一半的模块）；
  3. `sibling()` / `script()` 的路径与 key 正确；`load_candidates` 按
     `(mtime, size)` 失效 —— 文件没变返回同一对象，变了必须重读。
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ENGINE = REPO / "Plugins/ops/engine"

# `_loader` 是 engine/ 下的私有模块；`python3 -I` 下脚本目录不在 sys.path，
# 测试进程同理，所以按 engine 目录加载一次。
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import _loader  # noqa: E402


# ---------------------------------------------------------------- load_module

def test_load_module_hits_cache(tmp_path):
    key = "_test_loader_cached"
    p = tmp_path / "m.py"
    p.write_text("X = 1\n", encoding="utf-8")
    try:
        a = _loader.load_module(key, p)
        b = _loader.load_module(key, p)
        assert a is b, "同一 key 第二次调用必须命中 sys.modules 缓存"
        assert sys.modules[key] is a
        assert a.X == 1
    finally:
        sys.modules.pop(key, None)


def test_failed_load_is_not_cached_and_can_retry(tmp_path):
    """第一次 exec 抛错 → 缓存里不能留半成品；第二次（修好后）必须成功。"""
    key = "_test_loader_retry"
    p = tmp_path / "m.py"
    p.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="boom"):
            _loader.load_module(key, p)
        assert key not in sys.modules, "失败的模块不许留在 sys.modules（脏模块 bug）"

        # 修好文件后重试 —— 若脏模块还在缓存，这里会再次拿到坏对象/不执行新代码。
        p.write_text("VALUE = 42\n", encoding="utf-8")
        m = _loader.load_module(key, p)
        assert m.VALUE == 42
        assert sys.modules[key] is m
    finally:
        sys.modules.pop(key, None)


def test_failed_load_syntax_error_also_clears_cache(tmp_path):
    key = "_test_loader_syntax"
    p = tmp_path / "m.py"
    p.write_text("def broken(:\n", encoding="utf-8")
    try:
        with pytest.raises(SyntaxError):
            _loader.load_module(key, p)
        assert key not in sys.modules
    finally:
        sys.modules.pop(key, None)


# ---------------------------------------------------------------- path helpers

def test_sibling_path_and_key():
    mod = _loader.sibling("card")
    assert "_ops_engine_card" in sys.modules
    assert sys.modules["_ops_engine_card"] is mod
    assert Path(mod.__file__).resolve() == (ENGINE / "card.py").resolve()


def test_script_path_and_key(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "demo_script.py").write_text("VALUE = 7\n", encoding="utf-8")
    # 不碰真实仓库 scripts/：只验证 `REPO_ROOT/scripts/<name>.py` 的拼法。
    monkeypatch.setattr(_loader, "REPO_ROOT", tmp_path)
    try:
        mod = _loader.script("demo_script")
        assert mod.VALUE == 7
        assert sys.modules["_op_script_demo_script"] is mod
        assert Path(mod.__file__).resolve() == (scripts / "demo_script.py").resolve()
    finally:
        sys.modules.pop("_op_script_demo_script", None)


# ---------------------------------------------------------------- card 缓存

def test_load_candidates_cached_until_file_changes(tmp_path):
    card_mod = _loader.sibling("card")
    card = tmp_path / "op-cache"
    card.mkdir()
    p = card / "candidates.yaml"
    p.write_text("candidates:\n  - {id: a, status: pending}\n", encoding="utf-8")

    d1 = card_mod.load_candidates(card)
    d2 = card_mod.load_candidates(card)
    assert d1 is d2, "同一个 YAML 没变时必须返回同一对象"
    assert [c["id"] for c in d1["candidates"]] == ["a"]

    # 改内容并显式推进 mtime（有些文件系统的 mtime 粒度会让快速重写看起来没变）。
    p.write_text("candidates:\n  - {id: b, status: pending}\n", encoding="utf-8")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))

    d3 = card_mod.load_candidates(card)
    assert d3 is not d1, "mtime 变了必须重新读"
    assert [c["id"] for c in d3["candidates"]] == ["b"]
