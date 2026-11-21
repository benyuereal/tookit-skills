# 问题排查知识库 (Issue Knowledge Base)
> 本目录是规范与模板（只读）；沉淀记录写入用户知识库 `~/.config/dcu-toolkit/kb/`（见 `data/kb/README.md`），不写回本目录（skill 树/共享层随包升级，写入会被厂商版本覆盖）。

此目录存储DCU相关问题的排查记录，每个问题一个markdown文件。

## 谁写入（沉淀约定）

故障排查类任务收尾时，由 agent 按 `dcu-orchestrator` 的 `fault-triage` 工作流沉淀：
`fault-fix` 通过后，把本次故障按下方模板写入用户知识库 `~/.config/dcu-toolkit/kb/`（按下方模板）（文件名见命名规范），
内容遵循 `docs/STANDARDS.md` 脱敏红线（零内网地址、零真实凭证、设备号占位化）。

## 文件命名规范

```
{问题类型}_{简要描述}.md
```

示例：
- `模型推理_OOM显存不足.md`
- `性能_绑核后性能下降.md`
- `环境_DTK版本不兼容.md`
- `通信_多卡互联报错.md`

## 问题记录模板

见 `../kb/templates/issue-record-template.md`

## 问题记录内容

每条记录包含：
1. 问题描述（现象、报错信息）
2. 环境信息（DCU型号、DTK版本、模型、框架）
3. 排查过程（步骤、命令、中间结果）
4. 根因分析
5. 解决方案
6. 复现条件（可选）
7. 关联问题（可选）
