# 算子实现任务{{card_suffix}}

你只负责**写实现**。测量、判定、记账全部由引擎完成；不要在会话里自己跑 bench 或计时。

## 时间预算与停止条件
- 本轮到点即停（默认 20 分钟），**只做本候选的实现**。
- 不要读仓库其他文件、不要翻账本/历史记录、不要自己对比变体——
  测量、判定、记账都是引擎的事。时间到就把已完成的部分写进卡里收工。

## 落卡硬约束（违反即判 INVALID_NOOP）
- **所有实现必须写进 `<卡>/_common.py`**（`VARIANTS` / `HIP_VARIANTS` + `get_runner` 分支）；需要新 kernel 就加 `<卡>/kernels/*.hip` 并登记。
- **每完成一段就落卡**，不要等「最后一次性写回」。引擎只认工作区里的改动。
- `/tmp` 只允许临时验证；临时文件请放 `<工作区>/.scratch/`（引擎看得见）。

## 任务

{{task_lines}}

{{candidate_block}}
{{feedback_block}}
{{refuted_block}}
{{round_block}}
