"""Validation logic for generated testcases."""

from __future__ import annotations


class TestCaseValidator:
    """Validate testcase content with lightweight static checks."""

    def validate_testcase(self, testcase: str) -> bool:
        """Validate required structure of generated tests.

        Args:
            testcase: Generated test source.

        Returns:
            True when basic assertions and function definition exist.
        """
        if not isinstance(testcase, str) or not testcase.strip():
            return False
        has_test_function = "def test_" in testcase
        has_assert = "assert " in testcase
        has_import = "import pytest" in testcase
        return has_test_function and has_assert and has_import
