"""Build testcase suite content."""

from __future__ import annotations

from typing import List


class TestSuiteBuilder:
    """Combine multiple tests into a suite manifest."""

    def create_test_suite(self, tests: List[str]) -> str:
        """Create a serialized suite descriptor.

        Args:
            tests: List of generated testcase file names or ids.

        Returns:
            Suite text in YAML-like format.
        """
        lines = ["suite:", "  name: dcu-ci-suite", "  tests:"]
        for item in tests:
            lines.append(f"    - {item}")
        lines.append("  run_command: pytest -q")
        return "\n".join(lines) + "\n"
