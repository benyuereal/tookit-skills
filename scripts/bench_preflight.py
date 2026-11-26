#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_preflight —— 跑任何 bench 之前的前置检查。**默认开，不是可选项。**

## 为什么要有它

2026-11-25 的 5 小时，根因是 `cuda:N`(N>=4) 让自定义 kernel 崩。
工具里没有任何地方在「跑 bench 之前」检查这个。
`env_probe.py` 写出来了，但**如果没人记得跑它，等于没写**。

判决：把它接进 bench 入口，默认执行。想跳过必须显式 `--skip-preflight`，
而那个开关只在「已经手工验过环境」时才该用。

## 三层检查，按代价从低到高

1. **设备绑定**（~10 秒）：cuda:N vs HIP_VISIBLE_DEVICES + cuda:0 对照
   —— 最贵的一坑，最便宜的一查
2. **构建产物**（~1 秒）：传给 bench 的 .so 比源码新吗？符号在吗？
3. **卡干净**（~2 秒）：委托 gpu_guard（使用率 / 可分配）

## 用法（在 ab.py 的 main 里）

    from bench_preflight import preflight
    preflight(device=a.device, sources=[("kernels/x.so", "kernels/x.hip")])

返回 True 表示可以继续。**任何一项失败都返回 False，不抛异常** ——
让调用方决定是退出还是显式跳过。
"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _norm_device(device):
    """把 'cuda:4' / '4' / None 统一成物理卡号或 None。"""
    if device is None:
        return None
    s = str(device)
    if s.startswith("cuda:"):
        s = s[5:]
    try:
        return int(s)
    except ValueError:
        return None


def preflight(device=None, sources=None, emit=None, skip_bind=False):
    """跑前置检查。返回 True=可以继续，False=不要跑。

    device  : 'cuda:4' 或 4，用于定位检查哪张卡
    sources : [(so_path, src_path, [symbols...]), ...] 要验的构建产物
    emit    : 输出函数，默认 print
    skip_bind: 跳过设备绑定检查（**只在已经手工验过时用**）
    """
    if emit is None:
        def emit(m):
            print(m, file=sys.stderr)

    ok_all = True
    card = _norm_device(device)

    # ① 设备绑定 —— 最贵的一坑
    if not skip_bind:
        try:
            import env_probe
            # **必须区分两种调用方式，它们要查的卡不一样：**
            #
            # (a) 调用方传了 --device N
            #     -> 要查物理卡 N（因为 ab.py 内部会转成重映射）
            # (b) 调用方没传 --device，但环境里设了 HIP_VISIBLE_DEVICES=M
            #     -> 进程内的 cuda:0 就是物理卡 M，要查 M
            #     **这是 loop.py 的调用方式**（它设 env 不传 --device）
            # (c) 都没有 -> 进程内 cuda:0 就是物理卡 0
            #
            # 2026-11-26 实测踩过：第一版在 (b) 情况下去查卡 0，
            # 而实际用的是卡 7 —— 查错了对象，报告还是「✓」。
            import os
            if card is None:
                hv = os.environ.get("HIP_VISIBLE_DEVICES")
                if hv and hv.strip():
                    first = hv.split(",")[0].strip()
                    try:
                        card = int(first)
                        emit("[preflight] 设备绑定：从 HIP_VISIBLE_DEVICES 取到卡 %d"
                             % card)
                    except ValueError:
                        card = 0
                else:
                    card = 0

            ok_bind, why = env_probe.probe_bind_modes(card)
            if ok_bind:
                emit("[preflight] 设备绑定（卡 %d）：✓ %s" % (card, why))
            else:
                emit("[preflight] 设备绑定（卡 %d）：✗ %s" % (card, why))
                emit("[preflight]   -> 用 HIP_VISIBLE_DEVICES=%d + cuda:0" % card)
                ok_all = False
        except Exception as ex:                        # noqa: BLE001
            emit("[preflight] 设备绑定：? 检查本身出错（%s）" % str(ex)[:60])

    # ② 构建产物
    if sources:
        try:
            import env_probe
            for item in sources:
                so, src = item[0], item[1]
                syms = item[2] if len(item) > 2 else None
                ok_b, detail = env_probe.check_build(src, so, symbols=syms)
                for d in detail:
                    emit("[preflight] 构建：%s" % d)
                if not ok_b:
                    ok_all = False
        except Exception as ex:                        # noqa: BLE001
            emit("[preflight] 构建：? 检查本身出错（%s）" % str(ex)[:60])

    # ③ 卡干净（可选：只在有用率数据时）
    if card is not None:
        try:
            import gpu_guard
            c, ok_g, why = gpu_guard.check(card, settle=0)
            if ok_g:
                emit("[preflight] 卡状态：✓ %s" % why)
            else:
                emit("[preflight] 卡状态：✗ %s" % why)
                ok_all = False
        except Exception as ex:                        # noqa: BLE001
            emit("[preflight] 卡状态：? %s" % str(ex)[:60])

    if not ok_all:
        emit("[preflight] **前置检查未全过。测出来的数字不可信。**")
    return ok_all


def selftest():
    """不依赖 GPU —— 验设备号解析与失败传播。"""
    assert _norm_device("cuda:4") == 4
    assert _norm_device(4) == 4
    assert _norm_device("4") == 4
    assert _norm_device(None) is None
    assert _norm_device("cuda") is None
    print("[ok] 设备号解析：'cuda:4' / 4 / '4' -> 4；None / 'cuda' -> None")

    # 构建产物检查失败必须让 preflight 返回 False
    import tempfile
    msgs = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "k.hip"
        src.write_text("// x", encoding="utf-8")
        # 故意不给 .so -> 必须失败
        ok = preflight(device=None, emit=msgs.append,
                       sources=[(str(d / "k.so"), str(src))])
        assert not ok, "缺产物时 preflight 应返回 False"
        assert any("构建" in m for m in msgs), msgs
    print("[ok] 反向：构建产物缺失 -> preflight 返回 False")

    msgs2 = []
    ok2 = preflight(device=None, emit=msgs2.append, skip_bind=True)
    assert ok2, ("无 sources、无 device 时应通过，实际 msgs=%r" % (msgs2,))
    print("[ok] 正向：无待检查项 -> 返回 True")

    print("\nselftest PASS（3 项）")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
