"""dcu-ci-testcase package exports."""

from common.intent import IntentRegistry, IntentSpec

from .generator import TestCaseGenerator
from .suite_builder import TestSuiteBuilder
from .testcase_manager import TestCaseManager
from .validator import TestCaseValidator

IntentRegistry.register(IntentSpec(
    type="ci-test",
    target_skill="dcu-ci-testcase",
    description="生成或管理 CI 测试用例",
    examples=["生成 pytest 单元测试", "创建 gtest 测试用例"],
    priority=100,
))

__version__ = "1.0.0"
__author__ = "DCU Team"

__all__ = [
    "TestCaseManager",
    "TestCaseGenerator",
    "TestCaseValidator",
    "TestSuiteBuilder",
]
