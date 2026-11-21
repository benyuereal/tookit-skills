"""
DCUDispatcher —— 整包模式的唯一入口（§3.1）

负责：意图分类 → 简单路由 / workflow 编排
不负责：知识检索、能力执行（这些属于 core skill）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from .intent_classifier import IntentClassifier
from .router import DCURouter
from .domain_router import DomainRouter
from common.intent import IntentRegistry, IntentSpec
from common.spi.workflow import RoutingContext, WorkflowResult


class DCUDispatcher:
    """
    v4.1 纯路由入口。

    使用方式：
        dispatcher = DCUDispatcher.from_marketplace()
        ctx = dispatcher.dispatch("把这段 CUDA 迁移到 HIP")
        # ctx.target_skill == "dcu-code-migrator"
    """

    def __init__(
        self,
        marketplace: Dict[str, Any],
        installed_skills: Optional[Set[str]] = None,
        llm_client: Any = None,
        skill_caller: Optional[Callable[[str, str, Dict[str, Any]], Any]] = None,
    ):
        """
        Args:
            marketplace:      marketplace.json 的解析结果
            installed_skills: 运行时已安装的 skill 名集合
            llm_client:       LLM 客户端（提供 classify_intent 方法；None 则纯关键词）
            skill_caller:     fn(skill, action, inputs) → output，由宿主注入
        """
        if installed_skills is None:
            installed_skills = self._detect_installed_skills(marketplace)
        self._installed_skills = installed_skills

        # 注册所有 skill 的 intents
        self._register_intents(marketplace, installed_skills)

        self._classifier = IntentClassifier(
            marketplace=marketplace,
            llm_client=llm_client,
            host_skill=None,                    # dispatcher 宿主
            installed_skills=installed_skills,
        )
        self._router = DCURouter(
            classifier=self._classifier,
            skill_caller=skill_caller,
        )
        self._domain_router = DomainRouter(
            marketplace=marketplace,
            installed_skills=installed_skills,
        )
        self._marketplace = marketplace

    # ---- 主接口 ----

    def dispatch(self, query: str) -> RoutingContext:
        """意图分类 + 简单路由，返回 RoutingContext"""
        decision = self._domain_router.pre_route(query)
        forced_target_skill = "dcu-hip-reference" if decision.force_dcu_first else None
        extra_metadata = {
            "domain_guard": {
                "force_dcu_first": decision.force_dcu_first,
                "bypass_skill": decision.bypass_skill,
                "matched_keywords": decision.matched_keywords,
                "reason": decision.reason,
            }
        }
        if decision.bypass_skill:
            forced_target_skill = "__self__"
            extra_metadata["domain_guard"]["forced_target"] = forced_target_skill
        return self._router.route_request(
            query,
            forced_target_skill=forced_target_skill,
            extra_metadata=extra_metadata,
        )

    def dispatch_workflow(self, query: str, workflow_id: str) -> WorkflowResult:
        """意图分类 + 工作流执行"""
        decision = self._domain_router.pre_route(query)
        forced_target_skill = "dcu-hip-reference" if decision.force_dcu_first else None
        extra_metadata = {
            "domain_guard": {
                "force_dcu_first": decision.force_dcu_first,
                "bypass_skill": decision.bypass_skill,
                "matched_keywords": decision.matched_keywords,
                "reason": decision.reason,
            }
        }
        if decision.bypass_skill:
            forced_target_skill = "__self__"
            extra_metadata["domain_guard"]["forced_target"] = forced_target_skill
        ctx = self._router.route_request(
            query,
            workflow_id=workflow_id,
            forced_target_skill=forced_target_skill,
            extra_metadata=extra_metadata,
        )
        wf  = self._router.load_workflow(workflow_id)
        return self._router.execute_workflow(wf, ctx)

    # ---- 工厂方法 ----

    @classmethod
    def from_marketplace(
        cls,
        marketplace_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> "DCUDispatcher":
        """从 marketplace.json 文件创建 dispatcher"""
        if marketplace_path is None:
            marketplace_path = Path(__file__).parent.parent.parent / "marketplace.json"
        with open(marketplace_path, encoding="utf-8") as f:
            marketplace = json.load(f)
        return cls(marketplace=marketplace, **kwargs)

    # ---- 内部辅助 ----

    @staticmethod
    def _detect_installed_skills(marketplace: Dict[str, Any]) -> Set[str]:
        """
        探测运行时已安装的 skill（通过 importlib 检测 Python 包是否可导入）。
        未能导入的 skill 视为未安装。
        """
        import importlib
        installed: Set[str] = set()
        repo_root = Path(__file__).resolve().parents[2]
        for skill in marketplace.get("skills", []):
            name = skill.get("name", "")
            entry = skill.get("entry", "")
            if entry and (repo_root / entry).exists():
                installed.add(name)
                continue
            python_name = name.replace("-", "_")
            try:
                importlib.import_module(f"skills.{python_name}")
                installed.add(name)
            except ImportError:
                pass
        return installed

    @staticmethod
    def _register_intents(
        marketplace: Dict[str, Any],
        installed_skills: Set[str],
    ) -> None:
        """从 marketplace.json 的 intents 字段批量注册到 IntentRegistry"""
        for skill in marketplace.get("skills", []):
            name = skill.get("name", "")
            if name not in installed_skills:
                continue
            for intent_def in skill.get("intents", []):
                IntentRegistry.register(IntentSpec(
                    type=intent_def["type"],
                    target_skill=name,
                    description=intent_def.get("description", ""),
                    examples=intent_def.get("examples", []),
                ))
