"""
DCURouter —— 简单路由 + 工作流编排（§8.1-8.2）

on_error 策略严格限定为 {stop, continue, skip}（v1.0 冻结，不支持 rollback/compensate）。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import yaml

from common.intent import IntentRegistry
from common.spi.routing import resolve_target_skill
from common.spi.workflow import (
    RoutingContext,
    StepResult,
    WorkflowResult,
)

from .intent_classifier import Intent, IntentClassifier

# P5: 特殊 action 处理函数（懒加载）
_retrieval_handler = None
_error_extractor = None


def _get_retrieval_handler():
    """懒加载检索处理器"""
    global _retrieval_handler
    if _retrieval_handler is None:
        from common.kb.aggregator import ShardAggregator
        from common.kb.query_parser import parse_query
        from common.spi.retrieval_hint import get_hint_for_intent

        def retrieval_handler(skill_name: str, action: str, inputs: dict, context: Optional[RoutingContext]):
            """内置 retrieve action 处理"""
            # 解析 query
            raw_query = inputs.get("query", context.query if context else "")
            extract_errors = inputs.get("extract_errors", False)
            parsed = parse_query(raw_query, extract_errors=extract_errors)

            # 生成 hint（RoutingContext 使用 intent_type 字段，无 .intent 对象）
            intent_type = inputs.get("intent_type")
            if not intent_type and context is not None:
                intent_type = context.intent_type
            if not intent_type:
                intent_type = "unknown"
            hint = get_hint_for_intent(intent_type)

            # 如果用户指定了 custom_hint，可以覆盖
            if "domains" in inputs:
                hint.primary_domains = inputs["domains"]
            if "channels" in inputs:
                hint.channels = inputs["channels"]

            # 执行检索：parent.parent 已是 skills/，不要再拼一层 skills/
            skills_root = Path(__file__).resolve().parent.parent
            agg = ShardAggregator(skill_dir=skills_root / skill_name)
            results = agg.search_with_hint(
                parsed.nl_query or raw_query,
                top_k=inputs.get("top_k", 5),
                hint=hint,
                parsed_query=parsed,
                deduplicate=inputs.get("deduplicate", True),
            )
            return results
        _retrieval_handler = retrieval_handler
    return _retrieval_handler


def _get_error_extractor():
    """懒加载错误提取器"""
    global _error_extractor
    if _error_extractor is None:
        from common.kb.query_parser import parse_query

        def error_extractor(skill_name: str, action: str, inputs: dict, context: Optional[RoutingContext]):
            """内置 extract_error action 处理"""
            raw_query = inputs.get("query", (context.query if context else ""))
            parsed = parse_query(raw_query, extract_errors=True)
            return {
                "error_snippet": parsed.error_snippet,
                "symbols": parsed.symbols,
                "code_blocks": parsed.code_blocks,
            }
        _error_extractor = error_extractor
    return _error_extractor

OnErrorPolicy = Literal["stop", "continue", "skip"]


class DCURouter:
    """
    纯路由层。不持有知识分片，不直接执行能力。
    通过 skill_caller 回调调用具体 core skill 的 action。
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        skill_caller: Optional[Callable[[str, str, Dict[str, Any]], Any]] = None,
        workflows_dir: Optional[Path] = None,
    ):
        """
        Args:
            classifier:   已初始化的 IntentClassifier
            skill_caller: fn(skill_name, action, inputs) → output
                          由运行时（AI 工具 / Python SDK）注入具体调用逻辑
            workflows_dir: workflow YAML 文件目录，默认 skills/dcu-dispatcher/workflows/
        """
        self._classifier   = classifier
        self._skill_caller = skill_caller
        self._workflows_dir = workflows_dir or (
            Path(__file__).parent / "workflows"
        )

    # ---- 简单路由 ----

    def route_request(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        forced_target_skill: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingContext:
        """分类意图并返回 RoutingContext"""
        intent: Intent = self._classifier.classify(query)
        if forced_target_skill and intent.type == "unknown":
            intent.target_skill = forced_target_skill
            # 当 domain guard 强制 DCU-first 时，意图默认标记为 api-lookup
            intent.type = "api-lookup"
        if forced_target_skill == "__self__":
            # 全绕过模式：不解析到任何子 skill，直接交回宿主处理
            metadata = {
                "confidence": intent.confidence,
                "strategy":   intent.strategy,
                "reasoning":  intent.reasoning,
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            return RoutingContext(
                query=query,
                intent_type="unknown",
                target_skill="__self__",
                metadata=metadata,
            )
        resolved_skill = resolve_target_skill(
            intent.target_skill,
            self._classifier._host_skill,
        )
        metadata = {
            "confidence": intent.confidence,
            "strategy":   intent.strategy,
            "reasoning":  intent.reasoning,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return RoutingContext(
            query=query,
            intent_type=intent.type,
            target_skill=resolved_skill,
            metadata=metadata,
        )

    # ---- 工作流编排 ----

    def load_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """从 YAML 文件加载 workflow 定义"""
        wf_path = self._workflows_dir / f"{workflow_id}.yaml"
        if not wf_path.exists():
            raise FileNotFoundError(f"Workflow '{workflow_id}' not found at {wf_path}")
        with open(wf_path, encoding="utf-8") as f:
            wf = yaml.safe_load(f)
        self._validate_workflow(wf)
        return wf

    def execute_workflow(
        self,
        workflow: Dict[str, Any],
        context: RoutingContext,
    ) -> WorkflowResult:
        """
        按 DAG 拓扑序执行 steps（§8.2）。

        on_error 策略（v1.0 冻结为 3 值，不支持 rollback/compensate）：
        - stop:     中止后续步骤（默认）
        - continue: 跳过失败步骤，继续执行
        - skip:     跳过该 step 及其下游，整体仍判 succeeded
        """
        workflow_id = workflow.get("workflow_id", "unknown")
        global_on_error: OnErrorPolicy = workflow.get("on_error", "stop")
        steps_def: List[Dict[str, Any]] = workflow.get("steps", [])

        step_results: Dict[str, StepResult] = {}
        skipped_due_to: set = set()     # 因上游 skip 而被跳过的 step_id

        for step in steps_def:
            step_id    = step["id"]
            skill_name = step["skill"]
            action     = step["action"]
            needs      = step.get("needs", [])
            step_on_error: OnErrorPolicy = step.get("on_error", global_on_error)

            # 若上游 skip 则该步骤也 skip
            if any(n in skipped_due_to for n in needs):
                skipped_due_to.add(step_id)
                step_results[step_id] = StepResult(
                    step_id=step_id, skill=skill_name, status="skipped"
                )
                continue

            # 准备 inputs（支持 {{ steps.X.outputs.Y }} 和 {{ context.query.Z }} 模板）
            raw_inputs = step.get("inputs", {})
            inputs = self._resolve_inputs(raw_inputs, context, step_results)

            # 执行
            t0 = time.monotonic()
            result = self._call_step(step_id, skill_name, action, inputs, step_on_error, context)
            result.duration_ms = int((time.monotonic() - t0) * 1000)
            step_results[step_id] = result

            # 处理失败
            if result.status == "failed":
                if step_on_error == "stop":
                    remaining = steps_def[steps_def.index(step) + 1:]
                    for r in remaining:
                        step_results[r["id"]] = StepResult(
                            step_id=r["id"], skill=r["skill"], status="skipped"
                        )
                    return WorkflowResult(
                        workflow_id=workflow_id,
                        status="failed",
                        steps=list(step_results.values()),
                    )
                elif step_on_error == "skip":
                    skipped_due_to.add(step_id)

        final_status = "succeeded"
        if any(r.status == "failed" for r in step_results.values()):
            final_status = "failed"

        last_success = next(
            (r.output for r in reversed(list(step_results.values())) if r.status == "succeeded"),
            None,
        )
        return WorkflowResult(
            workflow_id=workflow_id,
            status=final_status,
            steps=list(step_results.values()),
            final_output=last_success,
        )

    # ---- 内部辅助 ----

    def _call_step(
        self,
        step_id: str,
        skill_name: str,
        action: str,
        inputs: Dict[str, Any],
        on_error: OnErrorPolicy,
        context: Optional[RoutingContext] = None,
    ) -> StepResult:
        # P5: 处理内置 action
        if action == "retrieve":
            try:
                handler = _get_retrieval_handler()
                output = handler(skill_name, action, inputs, context)
                return StepResult(step_id=step_id, skill=skill_name, status="succeeded", output=output)
            except Exception as e:
                return StepResult(step_id=step_id, skill=skill_name, status="failed", error=str(e))

        if action == "extract_error":
            try:
                handler = _get_error_extractor()
                output = handler(skill_name, action, inputs, context)
                return StepResult(step_id=step_id, skill=skill_name, status="succeeded", output=output)
            except Exception as e:
                return StepResult(step_id=step_id, skill=skill_name, status="failed", error=str(e))

        # 原有逻辑：调用 skill_caller
        if self._skill_caller is None:
            return StepResult(
                step_id=step_id, skill=skill_name, status="skipped",
                error="skill_caller 未注入，跳过执行"
            )
        try:
            output = self._skill_caller(skill_name, action, inputs)
            return StepResult(step_id=step_id, skill=skill_name, status="succeeded", output=output)
        except Exception as e:
            return StepResult(step_id=step_id, skill=skill_name, status="failed", error=str(e))

    def _resolve_inputs(
        self,
        raw_inputs: Dict[str, Any],
        context: RoutingContext,
        step_results: Dict[str, StepResult],
    ) -> Dict[str, Any]:
        """简单模板替换：{{ steps.X.outputs.Y }} / {{ context.query }}"""
        import re

        def _resolve_value(v: Any) -> Any:
            if not isinstance(v, str):
                return v
            pattern = r"\{\{\s*(.*?)\s*\}\}"
            def _replace(m: "re.Match[str]") -> str:
                expr = m.group(1)
                if expr.startswith("steps."):
                    parts = expr.split(".")
                    sid = parts[1]
                    sr = step_results.get(sid)
                    if sr and sr.output is not None:
                        if len(parts) > 3 and parts[2] == "outputs":
                            key = parts[3]
                            if isinstance(sr.output, dict):
                                return str(sr.output.get(key, ""))
                        return str(sr.output)
                    return ""
                elif expr == "context.query":
                    return context.query
                elif expr.startswith("context.query."):
                    key = expr.split(".", 2)[2]
                    return str(context.skill_inputs.get(key, ""))
                return m.group(0)
            return re.sub(pattern, _replace, v)

        return {k: _resolve_value(v) for k, v in raw_inputs.items()}

    def _validate_workflow(self, wf: Dict[str, Any]) -> None:
        """基础 schema 校验"""
        valid_on_error = {"stop", "continue", "skip"}
        if wf.get("on_error", "stop") not in valid_on_error:
            raise ValueError(
                f"workflow '{wf.get('workflow_id')}' on_error 值非法: "
                f"'{wf['on_error']}'，必须为 {valid_on_error}"
            )
        for step in wf.get("steps", []):
            if step.get("on_error", "stop") not in valid_on_error:
                raise ValueError(
                    f"step '{step.get('id')}' on_error 值非法，必须为 {valid_on_error}"
                )
