# 状态看板

> 自动生成，勿手改。重新生成：`python3 -I scripts/state.py --write`
> 阶段定义来自 `Plugins/ops/op-atlas/validate.py` 的 `STAGES` / `STAGE_NEEDS` / `BLOCKED_BY`。

阶段分布（自动生成，勿手改）
{{stage_distribution}}
{{blocked_summary}}

## 分阶段清单
{{stage_sections}}

### blocked（{{blocked_count}} 张）
{{blocked_body}}

### 未接入（{{unadmitted_count}} 张，未写 progress.stage）
{{unadmitted_body}}

### 非法（{{illegal_count}} 张）
{{illegal_body}}{{extra_sections}}
