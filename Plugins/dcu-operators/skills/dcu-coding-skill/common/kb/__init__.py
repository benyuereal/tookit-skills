"""
common.kb — 知识分片聚合检索

v4.1 核心：可发现、可合并的分片（shard）机制。
每个 core skill 自带 kb_shard/ 目录，聚合器在运行时自动发现并合并检索。

快速使用：
    from common.kb import get_aggregator
    from pathlib import Path

    kb = get_aggregator(skill_dir=Path(__file__).parent)
    results = kb.search("hipMalloc 参数", domains=["hip-api"], top_k=5)
    api_hits = kb.search_api("hipMalloc", domains=["hip-api"], top_k=5)
"""
import json
from pathlib import Path

import yaml

from .aggregator import ShardAggregator, SearchResults
from .shard import Shard
from .rrf import weighted_rrf


def get_aggregator(skill_dir=None) -> ShardAggregator:
    """
    获取聚合器实例。

    Args:
        skill_dir: 调用方 skill 所在目录（Path 或 None）。
                   传入时，聚合器发现该目录下的 kb_shard/、kb_shard_ext/，
                   再扫描同级 skill 分片与安装根 docs/self_docs/。

    Returns:
        配置好 skill_dir 的 ShardAggregator 实例（分片仅发现：
        本 skill / 同级 skills 的 kb_shard 与 kb_shard_ext，以及安装根 docs/self_docs）。
    """
    runtime_cfg = {}
    if skill_dir:
        sdir = Path(skill_dir)
        agent_root = sdir.parent.parent
        cfg_path = agent_root / "dcu-coding-skill.config.yaml"
        if not cfg_path.exists():
            cfg_path = agent_root / "config.yaml"
        if cfg_path.exists():
            runtime_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        manifest_path = agent_root / "dcu-coding-skill.package-manifest.json"
        if not manifest_path.exists():
            manifest_path = agent_root / "package-manifest.json"
        if manifest_path.exists():
            pm = json.loads(manifest_path.read_text(encoding="utf-8"))
            runtime_cfg.setdefault("package_capabilities", {})
            runtime_cfg["package_capabilities"]["vectorization"] = bool(
                pm.get("capabilities", {}).get("vectorization", False)
            )
        else:
            internal_caps = agent_root / "packaging" / "package_capabilities.json"
            if internal_caps.exists():
                caps = json.loads(internal_caps.read_text(encoding="utf-8"))
                runtime_cfg.setdefault("package_capabilities", {})
                runtime_cfg["package_capabilities"]["vectorization"] = bool(
                    caps.get("vectorization", False)
                )
    return ShardAggregator(skill_dir=skill_dir, config=runtime_cfg)


__all__ = ["ShardAggregator", "SearchResults", "Shard", "weighted_rrf", "get_aggregator"]
