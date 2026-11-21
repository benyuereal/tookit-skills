"""dcu-dispatcher 单元测试"""
import importlib.util
from pathlib import Path

import pytest
from common.intent import IntentRegistry, IntentSpec

ROOT = Path(__file__).resolve().parents[3]


def _load_module(rel_path: str, module_name: str):
    target = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clear_registry():
    """每个测试前清空 IntentRegistry，避免状态污染"""
    IntentRegistry.clear()
    yield
    IntentRegistry.clear()


@pytest.fixture
def marketplace():
    return {
        "skills": [
            {
                "name": "dcu-hip-reference",
                "category": "core",
                "keywords": ["hip", "hip-api", "lookup", "查询"],
                "intents": [{"type": "api-lookup", "description": "HIP API 查询", "examples": []}],
            },
            {
                "name": "dcu-code-migrator",
                "category": "core",
                "keywords": ["cuda", "hipify", "migration", "迁移"],
                "intents": [{"type": "code-migrate", "description": "CUDA 迁移", "examples": []}],
            },
        ]
    }


@pytest.fixture
def installed_skills():
    return {"dcu-hip-reference", "dcu-code-migrator"}


def _register(marketplace, installed):
    for skill in marketplace["skills"]:
        if skill["name"] in installed:
            for intent_def in skill.get("intents", []):
                IntentRegistry.register(IntentSpec(
                    type=intent_def["type"],
                    target_skill=skill["name"],
                    description=intent_def.get("description", ""),
                ))


class TestIntentClassifier:
    def test_keyword_classify_migrate(self, marketplace, installed_skills):
        IntentClassifier = _load_module(
            "skills/dcu-dispatcher/intent_classifier.py", "intent_classifier_mod"
        ).IntentClassifier
        _register(marketplace, installed_skills)
        clf = IntentClassifier(marketplace, installed_skills=installed_skills)
        intent = clf.classify("把这段 cuda 代码迁移到 HIP")
        assert intent.target_skill == "dcu-code-migrator"
        assert intent.strategy == "keyword"

    def test_keyword_classify_lookup(self, marketplace, installed_skills):
        IntentClassifier = _load_module(
            "skills/dcu-dispatcher/intent_classifier.py", "intent_classifier_mod"
        ).IntentClassifier
        _register(marketplace, installed_skills)
        clf = IntentClassifier(marketplace, installed_skills=installed_skills)
        intent = clf.classify("hipMalloc 怎么用，查询一下")
        assert intent.target_skill == "dcu-hip-reference"

    def test_fallback_to_hip_reference(self, marketplace, installed_skills):
        IntentClassifier = _load_module(
            "skills/dcu-dispatcher/intent_classifier.py", "intent_classifier_mod"
        ).IntentClassifier
        _register(marketplace, installed_skills)
        clf = IntentClassifier(marketplace, installed_skills=installed_skills)
        intent = clf.classify("完全不相关的问题 xyzzy")
        assert intent.target_skill == "dcu-hip-reference"
        assert intent.strategy == "fallback"

    def test_fallback_self_when_no_hip_reference(self, marketplace):
        IntentClassifier = _load_module(
            "skills/dcu-dispatcher/intent_classifier.py", "intent_classifier_mod"
        ).IntentClassifier
        installed = {"dcu-code-migrator"}   # 不含 hip-reference
        _register(marketplace, installed)
        # host_skill 非 None 时不触发 dispatcher 不变式
        clf = IntentClassifier(
            marketplace, installed_skills=installed, host_skill="dcu-code-migrator"
        )
        intent = clf.classify("xyzzy unknown")
        assert intent.target_skill == "__self__"

    def test_dispatcher_host_requires_hip_reference(self, marketplace):
        IntentClassifier = _load_module(
            "skills/dcu-dispatcher/intent_classifier.py", "intent_classifier_mod"
        ).IntentClassifier
        with pytest.raises(RuntimeError, match="dcu-hip-reference"):
            IntentClassifier(
                marketplace,
                installed_skills={"dcu-code-migrator"},  # 故意不含 hip-reference
                host_skill=None,   # dispatcher 宿主
            )


class TestRouter:
    def test_on_error_validation(self, tmp_path):
        DCURouter = _load_module("skills/dcu-dispatcher/router.py", "router_mod").DCURouter
        IntentClassifier = _load_module(
            "skills/dcu-dispatcher/intent_classifier.py", "intent_classifier_mod"
        ).IntentClassifier
        from common.intent import IntentRegistry, IntentSpec
        IntentRegistry.register(IntentSpec("api-lookup", "dcu-hip-reference", ""))
        mp = {"skills": [{"name": "dcu-hip-reference", "category": "core", "keywords": ["hip"]}]}
        clf = IntentClassifier(mp, installed_skills={"dcu-hip-reference"})
        router = DCURouter(classifier=clf)

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("workflow_id: bad\non_error: rollback\nsteps: []\n")
        with pytest.raises(ValueError, match="on_error"):
            router.load_workflow.__func__(router, "bad") if False else None
            import yaml
            wf = yaml.safe_load(bad_yaml.read_text())
            router._validate_workflow(wf)


class TestDispatcher:
    def test_detect_installed_skills_from_entry_paths(self, tmp_path):
        dispatcher_mod = _load_module("skills/dcu-dispatcher/dispatcher.py", "dispatcher_mod")
        fake_root = tmp_path
        skill_dir = fake_root / "skills" / "dcu-code-migrator"
        (fake_root / "skills" / "dcu-dispatcher").mkdir(parents=True)
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: dcu-code-migrator\n---\n", encoding="utf-8")

        dispatcher_mod.__file__ = str(fake_root / "skills" / "dcu-dispatcher" / "dispatcher.py")
        installed = dispatcher_mod.DCUDispatcher._detect_installed_skills({
            "skills": [
                {"name": "dcu-code-migrator", "entry": "skills/dcu-code-migrator/SKILL.md"},
                {"name": "dcu-hip-reference", "entry": "skills/dcu-hip-reference/SKILL.md"},
            ]
        })

        assert installed == {"dcu-code-migrator"}
