---
name: dcu-ci-testcase
version: 1.0.1
license: MIT
description: CI 测试用例管理，支持测试用例创建、验证和组织，覆盖 unit/integration 场景
author: DCU Team





---

# DCU CI 测试用例技能

## 快速开始

### 1. 生成单元测试
```
用户：给这段 HIP 代码生成 pytest 单元测试
技能：dcu-ci-testcase
结果：可执行的 pytest 测试代码
```

### 2. 生成集成测试
```
用户：为这个 kernel 调用流程生成 integration test
技能：dcu-ci-testcase
结果：包含准备/执行/校验的集成测试模板
```

### 3. 构建测试套件
```
用户：把这些测试组织成一个 suite 并输出运行命令
技能：dcu-ci-testcase
结果：测试套件定义 + 运行建议
```

## 核心概念

将 HIP/Triton 代码片段结构化转换为可验证、可执行、可组合的 CI 测试资产。

## 核心函数

| 函数 | 描述 |
|------|------|
| `create_testcase()` | 根据代码与测试类型生成测试代码 |
| `validate_testcase()` | 验证测试代码结构与关键断言 |
| `run_testcase()` | 运行测试（当前返回执行计划） |
| `create_test_suite()` | 将多个测试组合为 suite |
| `generate_test_data()` | 生成示例输入数据配置 |

## 二分回归定位（借鉴自 sglang-bisect-ci-regression）

问题复现后用二分法定位回归源头：先取中间版本做基线对照，通过逐步缩小提交范围定位首个破坏提交。
操作要点：①每次仅测试一个中间版本；②对照已知正确基线；③失败→向前二分，成功→向后二分；④收敛到首个破坏提交后做根因确认。
适用：CI 回归、kernel 变更、环境漂移等「版本边界」类故障。

## 测试编写规范（借鉴自 write-sglang-test）

三层校验：基线对照（现有已知正确行为）→ 边界用例（新行为验证）→ 失败回归（复现→定位→修复→确认）。
- 测试用例须用 `CustomTestCase`（自定义基类），保证 `tearDownClass` 在 `setUpClass` 失败时仍执行，避免资源泄漏。
- 单元测试优先放 `unit/`，集成/端到端测试放 `registered/` 对应分类；Mock 场景优先用 `unittest.mock.patch`。
- 模型无关逻辑用通用小模型；模型相关测试按模型逐类注册到对应 CI suite。

## 基本用法

本目录名为连字符 `dcu-ci-testcase`（Python 包名不能含连字符），需按下述方式加载为 `dcu_ci_testcase` 后使用：

**方式 A（独立加载，推荐）**——用 `importlib` 从本目录加载为 `dcu_ci_testcase`：

```python
import importlib.util, sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[0]  # 指向本 skill 目录
INIT = SKILL_DIR / "__init__.py"
spec = importlib.util.spec_from_file_location(
    "dcu_ci_testcase", INIT, submodule_search_locations=[str(SKILL_DIR)])
MODULE = importlib.util.module_from_spec(spec)
sys.modules["dcu_ci_testcase"] = MODULE
spec.loader.exec_module(MODULE)

from dcu_ci_testcase import TestCaseManager

manager = TestCaseManager()
testcase = manager.create_testcase(
    hip_code="void saxpy(...) { /* ... */ }",
    test_type="unit_test",
)
is_valid = manager.validate_testcase(testcase)
print(is_valid)
```

**方式 B（已由框架/测试加载过该包名时）**——若 `dcu_ci_testcase` 已被外部框架或测试装置（参照 `tests/test_testcase_manager.py` 的加载方式）注册到 `sys.modules`，则可直接简写：

```python
from dcu_ci_testcase import TestCaseManager
```

## 配置

```yaml
default_framework: pytest
enable_static_validation: true
strict_assertion_check: true
```

## 参考文档

- [测试指南](references/test-guidelines.md)
- [覆盖率规范](references/coverage-guidelines.md)
- [示例集合](references/test-examples.md)
