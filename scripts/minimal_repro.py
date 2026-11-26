#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""minimal_repro —— kernel 崩了/算错了，用递减实验定位到根因。

## 为什么需要它（D24 的 90 分钟）

2026-11-25 FlashMLA 崩了，我花了 90 分钟才想到「做一个 5 行最小 kernel 试试」。
那次递减实验的走向：

```
① 完整 kernel                    -> 崩
② 35 行 __shfl_down 归约         -> 崩
③ 空 kernel，64 线程             -> 崩
④ 空 kernel，32/128/256 线程     -> 全崩
⑤ 换成别的算子预编译的 .so       -> 崩   ← 到这里会误判成「平台坏了」
⑥ 换设备绑定方式                 -> **好了**
```

**关键在第 ⑦ 层：递减只减了「被测对象」，没减「怎么调用它」。**
第 ⑤ 层之后我已经证明了「不是我的代码」，但没证明「不是我的用法」。
又花了 40 分钟。

## 这个工具做什么

它不「自动找出根因」——那不可能。它做两件事：

1. **强制你按固定顺序做递减实验**，每一层只改一个变量。
   顺序是按「最可能翻转」排的：先减计算，再减尺寸，再换实现，
   **最后换调用方式**（那是最容易被忘掉的一层）。

2. **记录每一层的结果**，并**在最后两层特意提示**：
   「你已经证明了不是被测对象的问题 —— 现在该怀疑调用方式了。」
   这句话是 D24 的教训变成的机器判据。

## 用法

    # 层的定义写在 JSON 里，工具负责跑、记录、并在该提醒时提醒
    python3 -I scripts/minimal_repro.py --plan plan.json
    python3 -I scripts/minimal_repro.py --init > plan.json   # 生成模板
    python3 -I scripts/minimal_repro.py --selftest

plan.json 结构：
    {
      "what": "FlashMLA H=64 崩",
      "layers": [
        {"id": "L1", "desc": "完整 kernel", "cmd": "python3 probe.py", "expect": "crash"},
        ...
      ]
    }

每层跑完记 ok / crash / timeout。**工具会在「所有层都崩」时
明确提示去查调用方式** —— 这是它存在的唯一理由。
"""
import argparse
import json
import subprocess
import sys

# 递减实验的标准层序。**调用方式放在最后是刻意的。**
CANONICAL_LAYERS = [
    ("L1", "完整实现", "被测对象：完整 kernel"),
    ("L2", "去掉主体计算", "只保留一次最简访存"),
    ("L3", "空 kernel", "去掉全部计算，只写一个值"),
    ("L4", "空 kernel + 变尺寸", "去掉线程数/块大小依赖"),
    ("L5", "换实现", "用**别的算子**已编译好的产物 —— 排除「我的代码」"),
    ("L6", "换调用方式", "**最容易漏掉的一层**：设备绑定 / 环境变量 / API 用法"),
]


def run_layer(layer, timeout=180):
    """跑一层，返回 'ok' / 'crash' / 'timeout' / 'error'。**不抛异常。**"""
    try:
        p = subprocess.run(layer["cmd"], shell=True, capture_output=True,
                           text=True, timeout=timeout)
        if p.returncode == 0:
            return "ok"
        # 段错误 (139) / 被信号杀 (负返回值) 都算 crash
        if p.returncode in (139, -11) or p.returncode < 0:
            return "crash"
        return "crash" if "core" in (p.stderr or "").lower() else "error"
    except subprocess.TimeoutExpired:
        return "timeout"
    except OSError as ex:
        return "error:%s" % str(ex)[:40]


def analyze(results):
    """从层结果推断该往哪查。返回 (结论, 是否该提醒查调用方式)。

    判据（这是 D24 唯一的可复用结论）：
      · 找到第一个 ok —— 它前一层就是翻转点，那就是根因层
      · **全部崩** —— 说明问题不在被测对象。**必须去查调用方式。**
        这正是 D24 第 ⑤ 层之后的状态，当时我没想到。
    """
    statuses = [r["status"] for r in results]
    if "ok" not in statuses:
        return ("所有层都崩 —— 问题不在被测对象。", True)

    first_ok = statuses.index("ok")
    if first_ok == 0:
        return ("第 1 层就通过 —— 递减实验的条件可能和真实场景不一致。", False)
    flip = results[first_ok - 1]
    return ("翻转点：第 %s 层（%s）崩、下一层通过。"
            "**根因是被减掉的那个成分。**" % (flip["id"], flip["desc"]), False)


def plan_doc(what):
    """生成 plan 模板。层用 CANONICAL_LAYERS，命令留空待填。"""
    return {
        "what": what,
        "layers": [
            {"id": lid, "desc": desc, "why": why, "cmd": "", "expect": "crash"}
            for lid, desc, why in CANONICAL_LAYERS
        ],
        "note": "x".replace("x", ""),
    }


def selftest():
    """不依赖 GPU —— 用假命令验判据。"""
    # 反向：全崩时必须提醒查调用方式
    res = [{"id": "L%d" % i, "desc": "d", "status": "crash"} for i in range(1, 6)]
    msg, remind = analyze(res)
    assert remind is True, ("全崩时必须提醒查调用方式，实际 %r" % (remind,))
    assert "不在被测对象" in msg, msg
    print("[ok] 反向：所有层都崩 -> 提醒去查调用方式（D24 的教训）")

    # 正向：找翻转点
    res2 = [{"id": "L1", "desc": "完整实现", "status": "crash"},
            {"id": "L2", "desc": "去掉主体计算", "status": "ok"}]
    msg2, remind2 = analyze(res2)
    assert remind2 is False and "L1" in msg2, msg2
    print("[ok] 正向：找到翻转点 -> 指出根因层")

    # 边界：第 1 层就过
    msg3, _ = analyze([{"id": "L1", "desc": "d", "status": "ok"}])
    assert "条件可能和真实场景不一致" in msg3, msg3
    print("[ok] 边界：第 1 层就通过 -> 提示条件不一致（不是「没问题」）")

    # 真跑一个命令验 run_layer
    st = run_layer({"cmd": "exit 0"})
    assert st == "ok", st
    st2 = run_layer({"cmd": "sh -c 'kill -SEGV $$'"})
    assert st2 in ("crash", "error"), st2
    print("[ok] run_layer：成功判 ok，段错误判 crash")

    print("\nselftest PASS（4 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", help="实验计划 JSON")
    ap.add_argument("--init", action="store_true", help="生成 plan 模板")
    ap.add_argument("--what", default="未命名问题")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.init:
        print(json.dumps(plan_doc(a.what), ensure_ascii=False, indent=2))
        return 0
    if not a.plan:
        ap.error("需要 --plan 或 --init")

    plan = json.loads(open(a.plan, encoding="utf-8").read())
    layers = plan.get("layers") or []
    if not layers:
        print("plan 里没有 layers", file=sys.stderr)
        return 2

    print("递减实验：%s\n" % plan.get("what", ""))
    results = []
    for lyr in layers:
        if not lyr.get("cmd"):
            print("  %-4s %-22s  (未填 cmd，跳过)" % (lyr["id"], lyr["desc"]))
            continue
        st = run_layer(lyr)
        results.append({"id": lyr["id"], "desc": lyr["desc"], "status": st})
        print("  %-4s %-22s  %s" % (lyr["id"], lyr["desc"], st))

    if not results:
        print("\n没有可跑的层 —— 先填 cmd")
        return 2

    print()
    msg, remind = analyze(results)
    print("结论：%s" % msg)
    if remind:
        print()
        print("=" * 62)
        print("**你已经证明了问题不在被测对象。现在该怀疑调用方式。**")
        print()
        print("按这个顺序查（每条 10 秒）：")
        print("  1. 设备绑定：cuda:N vs HIP_VISIBLE_DEVICES + cuda:0")
        print("     -> python3 -I scripts/env_probe.py --card N")
        print("  2. 构建产物：.so 比源码新吗？符号在吗？")
        print("  3. arch 匹配：编译目标 vs 实际设备")
        print()
        print("2026-11-25 就是这么漏掉的：递减只减了被测对象，")
        print("没减「怎么调用它」，于是多花 40 分钟。")
        print("=" * 62)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
