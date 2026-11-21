import importlib.util
from pathlib import Path


def _load_skill_class():
    root = Path(__file__).resolve().parents[3]
    target = root / "skills" / "dcu-hip-reference" / "hip_reference.py"
    spec = importlib.util.spec_from_file_location("dcu_hip_reference_runtime", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DCUHipReferenceSkill


def test_spi_contract():
    skill = _load_skill_class()()
    caps = skill.capabilities()
    assert len(caps) >= 2
    assert any(c.action == "lookup_api" for c in caps)


def test_health_check_degraded_when_no_sqlite():
    skill = _load_skill_class()()
    hs = skill.health_check()
    assert hs.healthy is True
