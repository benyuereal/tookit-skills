# -*- coding: utf-8 -*-
"""_loader —— 「按路径加载模块」的**唯一实现**。

## 为什么要有它

`importlib.util.spec_from_file_location` 在本仓被写了十遍，每处只差错误处理，
而其中几处（`loop.py::_load_measure_mode` 等）**失败时不清理 `sys.modules`**：
第一次 exec 抛异常后，缓存里留着一个执行了一半的模块，后续调用拿到的是它 ——
"第一次明明报错了，第二次却不报错、行为还不对"是这类 bug 最难查的形态。

所以这里只做一件事：**先挂 `sys.modules` 再 exec，失败就删掉再抛**。
挂载必须在 exec 之前：模块之间会互相加载（floor 要用 card，cli 要用全部），
不先挂就会把同一个文件执行两遍，模块级缓存跟着分裂成两份。

## 为什么不能裸 import

`Plugins/ops` 不是可安装的包，脚本用 `python3 -I` 跑（脚本目录不进 sys.path），
测试用 `spec_from_file_location` 按路径加载，shim 用 `runpy` 转发。
三种方式下 `__package__` 都不对，相对导入必然失败 —— 与 loop.py::_engine
是同一个原因，这里只是把它收成一份。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: engine/ 目录。`sibling()` 用它拼兄弟模块路径。
ENGINE_DIR = Path(__file__).resolve().parent

#: 仓库根。engine/ 在 Plugins/ops/ 下（与 loop.py / signals.py 同口径）。
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[3])


def load_module(key: str, path):
    """按路径加载模块，**带 `sys.modules` 缓存**；失败时从缓存删除再抛。

    `key` 是模块在 `sys.modules` 里的名字。同一个 key 第二次调用直接返回
    第一次的模块对象（不重复 exec）。
    """
    if key in sys.modules:
        return sys.modules[key]
    p = Path(path)
    spec = importlib.util.spec_from_file_location(key, str(p))
    if spec is None:
        raise ModuleNotFoundError(str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # 关键：失败的模块不许留在缓存里，否则第二次调用拿到半成品。
        del sys.modules[key]
        raise
    return mod


def sibling(name: str):
    """加载 engine/ 下的兄弟模块（key 与 loop/cli/iterate 的 `_engine` 一致）。"""
    return load_module("_ops_engine_" + name, ENGINE_DIR / (name + ".py"))


def script(name: str):
    """加载仓库 `scripts/` 下的脚本（key `_op_script_<name>`）。"""
    return load_module("_op_script_" + name, REPO_ROOT / "scripts" / (name + ".py"))
