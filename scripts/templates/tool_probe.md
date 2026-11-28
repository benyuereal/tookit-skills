# 工具层可用性（`scripts/tool_probe.py` 实测生成，勿手改）

> 每一条都是**真跑**的结果，不是文档抄来的。
> 判据里只能写 `available` 的工具；`degraded` 要标明弱在哪；
> `unverified` 不许当成可用。

| 工具 | 阶段 | 状态 | 实测细节 | 缺失时的降级路径 |
|---|---|---|---|---|
{{rows}}
统计：{{stats}}

生成命令：`python3 -I scripts/tool_probe.py --write`
