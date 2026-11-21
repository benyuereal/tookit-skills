# dcu-train-cookbook — DCU/HCU 训练域

DCU/HCU 训练域 skill：训练/微调/RL 强化学习配方速查 + 训练框架版本锚点。覆盖：
- RL 强化学习（verl-das / slime-das / AReaL-das，GRPO 矩阵）
- 训练/微调配方（train-cookbook-das）
- 训练框架（Megatron-LM-das = hcu-megatron）
- 训练性能估算（hcu-train-simulator）
- 算子/其他训练组件

## 核心内容

- **更新机制**：配方与版本锚点由 `scripts/hygon-upstream-sync.py` 从上游同步，保证最新
- 各域配方速查（RL / 微调 / 框架 / 估算）
- 版本锚点（框架 × 版本 × 适配状态）

## 衔接

- 训练/RL 类任务先查本 skill 拿配方与版本；算子层面用 `dcu-op-optimization`。

详见 `SKILL.md`。
