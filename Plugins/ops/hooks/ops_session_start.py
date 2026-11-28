#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionStart hook —— 把「算子优化入口」直接注入会话上下文。

## 为什么需要 hook，而不是只写在 SKILL.md 里

Claude Code 只自动读取 **cwd 及其祖先目录** 的 CLAUDE.md / AGENTS.md。
插件装在 `<cwd>/skill/Plugins/ops/` 时，它的 `SKILL.md`、仓库根 `CLAUDE.md`
都不在自动读取路径上——除非模型主动去 `cat`。实测（2026-11-27）C 臂就是
只读了算子卡、直接手搓测量，`engine/loop.py` 一次没跑：**入口没暴露**。

SessionStart 是官方机制：插件加载时执行本脚本，把 additionalContext
塞进模型上下文。这样无论模型从哪个文件开始读，"必须走 op-optimize →
engine/loop.py" 这条规则都在它的上下文里。

## 输出协议

stdout 必须是 JSON：
    {"hookSpecificOutput": {"hookEventName": "SessionStart",
                            "additionalContext": "..."}}
调试：设 OPS_HOOK_DEBUG=<路径> 会把触发记录写进去（默认不写，不污染）。
"""
import json
import os
import sys


def _msg() -> str:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "<plugin>")
    return (
        "# 算子优化：这是一个循环，不是一份清单\n"
        "\n"
        "做算子优化时，**每一步都由引擎决定下一步**。不要自己先建任务清单、\n"
        "不要自己设计测量流程——你的计划会和引擎的状态机打架，而实测结果是\n"
        "引擎赢不了（三轮实测里它都把自己的基准排在引擎前面）。\n"
        "\n"
        "## 唯一需要记住的循环\n"
        "\n"
        f"    python3 -I {root}/engine/loop.py -C <卡目录> next\n"
        "\n"
        "它输出三行：下一步是什么、为什么、照做的命令。照做，然后再问一次 next。\n"
        "直到它返回 `下一步：停止（stop）` 或队列耗尽。\n"
        "\n"
        "## next 可能给你的五种动作\n"
        "\n"
        "| action | 你要做什么 |\n"
        "|---|---|\n"
        "| `run` | 复制它给的命令执行，等引擎判定并记账。**不要自己再测一遍**。 |\n"
        "| `implement` | 这是唯一需要你动脑的一步：把候选实现成变体，登记进\n"
        "|           | `_common.VARIANTS` 并给 `get_runner` 加分支。其余不用管。 |\n"
        "| `baseline` | 跑它给的命令，先把基线量出来。 |\n"
        "| `explore` | 跑它给的命令出题（生成「拟做」），然后回到 next。 |\n"
        "| `stop` | 收工。写 `report.md`，引用的每个数字都要能指到\n"
        "|        | `<卡>/runs/loop/*.jsonl` 里的某条记录。 |\n"
        "\n"
        "## 硬要求（少一项判未完成）\n"
        "\n"
        "1. `<卡>/runs/loop/*.jsonl` 必须有记录 —— 引擎实测的账本，手搓测量不算。\n"
        "2. `results/bench_std.json` 必须由 `loop.py run --bench-std` 生成：\n"
        f"\n       python3 -I {root}/engine/loop.py -C <卡> run <tag> --bench-std results/bench_std.json\n"
        "\n"
        "3. `report.md` 里每个耗时数字都要指到上面两个来源之一。\n"
        "\n"
        "## 不知道用哪张卡时\n"
        "\n"
        "`Plugins/ops/op-atlas/INDEX.md` 是按场景/族列的一览；\n"
        "没有语义相同的卡就用 `python3 -I {root}/../scripts/op_bootstrap.py` 建一张，\n"
        "再对它 `next`。**「没有现成卡」不是退回手搓测量的理由。**\n"
    )


def main() -> int:
    try:
        # 事件 JSON 从 stdin 给；读不到也不影响输出（纯注入型 hook）
        try:
            sys.stdin.read()
        except Exception:
            pass
        dbg = os.environ.get("OPS_HOOK_DEBUG")
        if dbg:
            try:
                with open(dbg, "a", encoding="utf-8") as f:
                    f.write("fired cwd=%s\n" % os.getcwd())
            except OSError:
                pass
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": _msg(),
            }
        }, ensure_ascii=False))
        return 0
    except Exception as e:  # 失败关闭：hook 坏了要看得见，但不能让会话起不来
        print(f"ops_session_start: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
