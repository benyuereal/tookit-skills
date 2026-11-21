"""Facade manager for testcase generation, validation, and orchestration."""

from __future__ import annotations

from typing import Any, Dict, List

from .generator import TestCaseGenerator
from .suite_builder import TestSuiteBuilder
from .validator import TestCaseValidator


class TestCaseManager:
    """High-level API for dcu-ci-testcase skill."""

    def __init__(self) -> None:
        """Initialize manager dependencies."""
        self.generator = TestCaseGenerator()
        self.validator = TestCaseValidator()
        self.suite_builder = TestSuiteBuilder()

    def create_testcase(self, hip_code: str, test_type: str) -> str:
        """Create testcase source code."""
        try:
            return self.generator.create_testcase(hip_code=hip_code, test_type=test_type)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def validate_testcase(self, testcase: str) -> bool:
        """Validate testcase content."""
        return self.validator.validate_testcase(testcase)

    def run_testcase(self, testcase: str) -> Dict[str, Any]:
        """Provide execution plan for running a testcase."""
        if not self.validate_testcase(testcase):
            return {
                "success": False,
                "message": "Testcase validation failed.",
                "command": "pytest -q",
            }
        return {
            "success": True,
            "message": "Testcase is valid and ready to run.",
            "command": "pytest -q",
        }

    def create_test_suite(self, tests: List[str]) -> str:
        """Create a suite descriptor from test ids/files."""
        return self.suite_builder.create_test_suite(tests)

    def generate_test_data(self, hip_function: str) -> Dict[str, Any]:
        """Generate example data payload for tests."""
        return self.generator.generate_test_data(hip_function)
