# 工具注册表 —— 六类排查定位能力
#
# **硬规则：没有 probe 证据不许写「实测可用」（status: available）。**
# 本文件由 scripts/gen_tool_registry.py 从 tool_probe.py 的实跑结果生成，
# 每条 status 都对应一次真实探测，证据在 probe 字段里。
#
# 复跑：python3 -I scripts/tool_probe.py --write-registry
# 校验：python3 -I scripts/gen_tool_registry.py --check

version: 1
registered_on: 2026-11-24
machine: gfx938 / BW1100 / DTK 26.04
generated_by: scripts/tool_probe.py

tools:{{tools_body}}

# ===== playbooks：按顺序的工具链 =====
# 规则：chain 里的每个 id 必须在上面的 tools 里存在（gen --check 会验）。
playbooks:{{playbooks_body}}
