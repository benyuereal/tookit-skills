# memory —— 跨卡记忆索引

> **只有 `engine/loop.py` 写。** 人手写、脚本自动生成、agent 补写一律禁止。
> 理由：本仓已有两套记账（`ledger.md` 散文 vs `runs/loop/*.jsonl` 机器记录），
> DS 的「≥3 轮」数的是散文。再开一个人写入口就是第三套。

## 两个文件的关系

| 文件 | 位置 | 权威性 | 用途 |
|---|---|---|---|
| `runs/loop/<date>.jsonl` | 各 skill 下 | **权威** | 单卡实验流水 |
| `memory/ledger.jsonl` | 这里 | 索引 | 跨卡按 (bound, signal) 检索 |

`loop.py` 写完流水后**顺手 append 一条**到这里。字段基本相同，
只多一个 `card` 字段用于跨卡检索。

## 检索

```bash
python3 -I scripts/recall.py --bound compute --signal wait_lds_high
```

返回的每条必须带 `citation`（哪张卡、哪一轮、verdict）。没有 citation 的结果等于没有。
