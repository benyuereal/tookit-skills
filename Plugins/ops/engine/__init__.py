# -*- coding: utf-8 -*-
"""engine —— 算子优化引擎（判定、地板、读写卡、编排的**唯一实现**）。

## 分层（下层不认识上层）

    card.py      读卡：candidates/contract/shapes/variants/roofline 的 I/O 与归一化
    floor.py     地板：derive 模型 → roofline.json → atlas floor_inputs 的唯一取数顺序
    probe.py     采信号：跑 derive、读 runs/loop、扫已实现变体、数方向预算
    propose.py   信号 → 候选（拟做，不是待做）；人读渲染
    ledger.py    记账：Cand/Top、replan、overlay、progress
    loop.py      六道门 + 12 个子命令（`cmd_run` 的唯一实现，含方向预算常量）
    cli.py       explore / floor 的薄 CLI（参数、打印、退出码）

## 为什么这个文件几乎是空的

因为 `Plugins/ops` 不是可安装的包：脚本用 `python3 -I` 跑（脚本目录不进
sys.path），测试用 `spec_from_file_location` 按路径加载，shim 用 `runpy`
转发。三种方式下 `__package__` 都不对，相对导入必然失败。所以 engine 下的
模块互相之间按**路径**加载（见 loop.py::_engine 的注释），本 `__init__.py`
只在有人 `import engine` 时才生效 —— 而仓库里没有这种人。

保留它是因为：① 让 engine/ 是个包这件事对读者/工具可见；
② 上面这张分层表比任何一处注释都更容易先被看到。
"""
