"""
workflow 数据契约（§8.2）

RoutingContext / StepResult / WorkflowResult 放在 common/spi/ 而非 dispatcher，
使 core skill、测试工具、第三方 runner 均可复用，dispatcher 保持"纯路由"定位。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..intent.registry import IntentSpec


StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


@dataclass
class RoutingContext:
    """单次请求的路由上下文，贯穿 dispatcher 的整个处理周期"""
    query: str
    intent_type: str                          # 意图类型字符串（动态）
    target_skill: str                         # 解析后的目标 skill 名
    skill_inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """workflow 单步执行结果"""
    step_id: str
    skill: str
    status: StepStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class WorkflowResult:
    """workflow 整体执行结果"""
    workflow_id: str
    status: StepStatus
    steps: List[StepResult] = field(default_factory=list)
    final_output: Optional[Any] = None

    def succeeded_steps(self) -> List[StepResult]:
        return [s for s in self.steps if s.status == "succeeded"]

    def failed_steps(self) -> List[StepResult]:
        return [s for s in self.steps if s.status == "failed"]
