#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fmla_patch —— 给 FlashMLA-das 打 d=512 支持补丁。

## 为什么要有这个脚本（不是为了省事）

2026-11-25 踩了两次：

  ① 直接改 `csrc/api/dense_decode_hip.h`，改完被仓库重新 clone 覆盖，
     白等一轮编译（两次，共约 20 分钟）
  ② 改了 `csrc/extension/flash_fwd_mla_kernel_fp8.h` 但没同步 `_hip.h`
     （hipify 产物），编译用的是 `_hip.h`，等于没改

**凡是「手工编辑 → 编译 → 发现没生效」的循环，都应该变成幂等脚本。**
这样至少能一行命令确认「补丁在不在」。

## 用法

    python3 -I scripts/fmla_patch.py --check     # 只检查，不修改（退出码 0=全在）
    python3 -I scripts/fmla_patch.py --apply     # 打补丁（幂等）
    python3 -I scripts/fmla_patch.py --selftest
"""
import argparse
import pathlib
import sys

REPO = pathlib.Path("/workspace/FlashMLA-das")

# (文件, 锚点, 替换成, 说明)
PATCHES = [
    (
        "csrc/api/dense_decode_hip.h",
        "    if ((num_heads_q < 64 && num_heads_k == 1) || num_heads_k > 1 || arch.is_gfx928()) {\n"
        "        params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();\n"
        "        params.num_sm_parts = num_sm_parts;\n"
        "        params.num_splits_ptr = num_splits->data_ptr<int>();",

        "    // **这两个字段两个分支都必须赋值**（D14）。\n"
        "    // fp8 主 kernel 第一件事就解引用 tile_scheduler_metadata_ptr\n"
        "    // （flash_fwd_mla_kernel_fp8_hip.h:2615），而原来只在 if 分支赋值。\n"
        "    // H >= 64 走 else → 指针是残留值 → VMFault，报错无任何指向性。\n"
        "    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();\n"
        "    params.num_sm_parts = num_sm_parts;\n"
        "    params.num_splits_ptr = num_splits->data_ptr<int>();\n"
        "    if ((num_heads_q < 64 && num_heads_k == 1) || num_heads_k > 1 || arch.is_gfx928()) {",
        "未初始化指针（H>=64 崩溃的根因）",
    ),
    (
        "csrc/extension/flash_fwd_mla_kernel_fp8.h",
        "    static_assert(Headdim == 576);",
        "    static_assert(Headdim == 576 || Headdim == 512);",
        "放开 d=512（GLM-5.3）",
    ),
]


def _p(path):
    return REPO / path


def check(emit=print):
    """检查补丁是否都在。**锚点变了也要报 False** —— 不能静默通过。"""
    ok = True
    for f, old, new, desc in PATCHES:
        p = _p(f)
        if not p.is_file():
            emit("✗ 缺文件 %s" % f)
            ok = False
            continue
        s = p.read_text(encoding="utf-8")
        if new in s:
            emit("✓ %s" % desc)
        elif old in s:
            emit("✗ %s —— 未打（锚点还在）" % desc)
            ok = False
        else:
            emit("? %s —— 锚点已变，需人工确认" % desc)
            ok = False
    return ok


def apply(emit=print):
    n = 0
    for f, old, new, desc in PATCHES:
        p = _p(f)
        if not p.is_file():
            emit("跳过（文件不存在）：%s" % f)
            continue
        s = p.read_text(encoding="utf-8")
        if old in s:
            p.write_text(s.replace(old, new, 1), encoding="utf-8")
            emit("已打补丁：%s" % desc)
            n += 1
        elif new in s:
            emit("已存在，跳过：%s" % desc)
        else:
            emit("⚠ 锚点不匹配，跳过：%s（%s）" % (desc, f))
    return n


def selftest():
    """不碰真仓库 —— 用临时目录验 check/apply 的幂等性。"""
    import tempfile
    global REPO
    saved = REPO
    try:
        with tempfile.TemporaryDirectory() as td:
            REPO = pathlib.Path(td)
            f, old, new, desc = PATCHES[1]
            fp = _p(f)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text("void f() {\n" + old + "\n}\n", encoding="utf-8")

            assert not check(emit=lambda *_: None), "未打补丁时 check 应为 False"
            print("[ok] 反向：未打补丁时 check 返回 False")

            assert apply(emit=lambda *_: None) == 1, "第一次 apply 应改 1 处"
            print("[ok] 正向：apply 生效")

            assert apply(emit=lambda *_: None) == 0, "幂等：第二次 apply 不应再改"
            print("[ok] 幂等：重复 apply 不改动")

            assert new in fp.read_text(encoding="utf-8"), "改动应落盘"
            print("[ok] 改动确实写入文件")

            fp.write_text("void f() { /* 完全不认识 */ }\n", encoding="utf-8")
            assert not check(emit=lambda *_: None), "锚点不匹配时 check 应为 False"
            print("[ok] 反向：锚点消失时 check 返回 False（不会静默通过）")

        print("\nselftest PASS（5 项，不碰真实仓库）")
        return 0
    finally:
        REPO = saved


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.check:
        return 0 if check() else 1
    apply()
    return 0


if __name__ == "__main__":
    sys.exit(main())
