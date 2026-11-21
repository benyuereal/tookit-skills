"""
DCU Coding Skill Common Module

v3 兼容导出（保留至 PR9 删除废弃文件）+ v4.1 新子包快捷导出。

v4.1 推荐使用子包路径：
    from common.kb import get_aggregator
    from common.spi import SkillProvider, Capability, HealthStatus
    from common.intent import IntentRegistry, IntentSpec
    from common.utils import Logger, PathHandler, ConfigLoader
"""

# v3 兼容导出（PR9 前保留）
from .dcu_kb_core import KnowledgeBase
from .document_parser import DocumentParser
# utils/ 子包已建立，旧 utils.py 被遮蔽；直接从子包导入保持兼容
from .utils import Logger, PathHandler, ConfigLoader

# v4.1 新子包快捷导出
from .kb import get_aggregator, ShardAggregator
from .spi import SkillProvider, Capability, HealthStatus
from .intent import IntentRegistry, IntentSpec

__all__ = [
    # v3 兼容
    "KnowledgeBase",
    "DocumentParser",
    "Logger",
    "PathHandler",
    "ConfigLoader",
    # v4.1 新增
    "get_aggregator",
    "ShardAggregator",
    "SkillProvider",
    "Capability",
    "HealthStatus",
    "IntentRegistry",
    "IntentSpec",
]

COMPONENTS_INFO = {
    "knowledge-base": {
        "name": "DCU Knowledge Base",
        "version": "1.0.0",
        "type": "core",
        "description": "DCU编程知识库"
    },
    "code-migrator": {
        "name": "CUDA to HIP Migrator",
        "version": "1.0.0",
        "type": "core",
        "description": "CUDA代码自动迁移到HIP"
    },
    "ci-testcase": {
        "name": "CI Test Case Manager",
        "version": "1.0.0",
        "type": "core",
        "description": "测试用例管理"
    },
    "triton-kernel": {
        "name": "Triton Kernel",
        "version": "1.0.0",
        "type": "core",
        "description": "Triton kernel 生成、优化、模板"
    },
    "code-generator": {
        "name": "Code Generator",
        "version": "1.0.0",
        "type": "auxiliary",
        "description": "HIP/Triton 代码生成"
    },
    "perf-optimizer": {
        "name": "Performance Optimizer",
        "version": "1.0.0",
        "type": "auxiliary",
        "description": "性能优化建议"
    },
    "debug-assistant": {
        "name": "Debug Assistant",
        "version": "1.0.0",
        "type": "auxiliary",
        "description": "调试助手"
    }
}
