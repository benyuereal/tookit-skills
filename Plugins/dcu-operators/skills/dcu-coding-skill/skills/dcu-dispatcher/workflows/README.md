# DCU Dispatcher Workflows

workflow YAML 文件目录。每个 `.yaml` 文件定义一个多 skill 工作流。

## 可用工作流

| workflow_id | 描述 |
|-------------|------|
| `migrate-and-test` | CUDA 代码迁移 → 自动生成测试用例 |

## 新增工作流

直接在本目录添加 `{workflow_id}.yaml` 文件，无需修改任何 Python 代码。

YAML schema：
```yaml
workflow_id: <全局唯一字符串>
description: <描述>
on_error: stop | continue | skip   # v1.0 仅支持这 3 个值

steps:
  - id: <step_id>
    skill: <skill_name>
    action: <capability_action>
    needs: [<step_id>, ...]         # 依赖的前序步骤（可选）
    inputs:
      key: "{{ context.query }}"    # 支持模板替换
    on_error: stop | continue | skip  # 覆盖全局策略（可选）
```
