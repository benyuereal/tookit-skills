# 产品方案矩阵 (Product Matrix)
> 本目录是规范与模板（只读）；沉淀记录写入用户知识库 `~/.config/dcu-toolkit/kb/`（见 `data/kb/README.md`），不写回本目录（skill 树/共享层随包升级，写入会被厂商版本覆盖）。

此目录存储DCU产品方案卡片，每个方案一个markdown文件。

## 谁写入（沉淀约定）

部署验证类任务收尾时，由 agent 按 `dcu-orchestrator` 的 `model-deploy` 工作流沉淀：
`deploy-verify` 通过后，把本次验证过的「模型×框架×卡型」方案按下方模板写入用户知识库 `~/.config/dcu-toolkit/kb/`，
内容遵循 `docs/STANDARDS.md` 脱敏红线（零内网地址、零真实凭证、设备号占位化）。

## 文件命名规范

```
{DCU型号}_{场景}_{关键信息}.md
```

示例：
- `Z100_LLM_GLM5.2_8卡_推理方案.md`
- `K100-AI_CV_目标检测_4卡方案.md`
- `Z100_多卡互联_带宽优化方案.md`

## 方案卡片模板

见 `../kb/templates/product-card-template.md`

## 方案卡片内容

每张卡片包含：
1. 硬件配置（卡型号、数量、CPU、内存、网络）
2. 软件栈版本（DTK、驱动、框架版本）
3. 关键调优参数（绑核策略、batch size、精度模式等）
4. 性能数据（throughput、latency、对比基准）
5. 适用场景与不适用场景
6. 已知问题与规避方案
7. 部署步骤（可直接给客户）
