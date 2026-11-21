"""Unit tests for dcu-ci-testcase manager APIs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
INIT_FILE = SKILL_DIR / "__init__.py"
SPEC = importlib.util.spec_from_file_location(
    "dcu_ci_testcase",
    INIT_FILE,
    submodule_search_locations=[str(SKILL_DIR)],
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Failed to load dcu-ci-testcase package for tests.")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["dcu_ci_testcase"] = MODULE
SPEC.loader.exec_module(MODULE)

TestCaseManager = MODULE.TestCaseManager


def test_create_and_validate_unit_testcase() -> None:
    """Manager can create and validate a unit testcase."""
    manager = TestCaseManager()
    testcase = manager.create_testcase("hipMalloc(&ptr, n);", "unit_test")
    assert "def test_" in testcase
    assert manager.validate_testcase(testcase) is True


def test_create_suite() -> None:
    """Manager can create suite descriptor."""
    manager = TestCaseManager()
    suite = manager.create_test_suite(["tests/test_a.py", "tests/test_b.py"])
    assert "dcu-ci-suite" in suite
    assert "pytest -q" in suite


def test_run_testcase_plan_with_invalid_input() -> None:
    """Invalid testcase produces failed run plan."""
    manager = TestCaseManager()
    result = manager.run_testcase("not a testcase")
    assert result["success"] is False


def test_generate_test_data() -> None:
    """Manager returns deterministic generated test data."""
    manager = TestCaseManager()
    payload = manager.generate_test_data("saxpy")
    assert payload["function_name"] == "saxpy"
    assert payload["seed"] == 42
