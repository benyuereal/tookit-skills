"""DCUCodeMigratorSkill - SkillProvider implementation for the fast-based migrator."""
from __future__ import annotations

from pathlib import Path
from typing import List

from common.spi import Capability, HealthStatus, SkillProvider


class DCUCodeMigratorSkill(SkillProvider):
    name = "dcu-code-migrator"
    version = "1.0.0"

    def capabilities(self) -> List[Capability]:
        return [
            Capability(action="migrate_cuda_to_hip", cost="medium", description="Generate CUDA to HIP migration output"),
            Capability(action="get_migration_report", cost="medium", description="Summarize migration status and risks"),
            Capability(action="check_remaining_cuda", cost="cheap", description="Detect remaining CUDA symbols after migration"),
        ]

    def health_check(self) -> HealthStatus:
        native = self.skill_dir() / "kb_shard" / "shard.sqlite"
        ext = self.skill_dir() / "kb_shard_ext" / "shard.sqlite"
        if not native.exists() and not ext.exists():
            return HealthStatus(
                healthy=True,
                degraded=True,
                reason="native/ext shard 都不存在，降级为全局聚合检索",
                missing_shards=["dcu-code-migrator-native", "dcu-code-migrator-ext"],
            )
        return HealthStatus(healthy=True)

    def skill_dir(self) -> Path:
        return Path(__file__).parent
