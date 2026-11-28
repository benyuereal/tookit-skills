#!/bin/bash
# 库存 agent wrapper：把全局「实现者契约」作为 system prompt 注入。
# 环境（orchestrator 会传）：OPS_WS / OPS_ENV_FILE / OPS_SYSTEM_PROMPT_FILE / OPS_GPU
set -a
[ -r "${OPS_ENV_FILE:-/workspace/tsoc.env}" ] && . "${OPS_ENV_FILE:-/workspace/tsoc.env}"
[ -r /opt/dtk/env.sh ] && . /opt/dtk/env.sh >/dev/null 2>&1
set +a
export ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL%/}
export CLAUDE_CODE_ENABLE_TELEMETRY=0 DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 \
       DISABLE_AUTOUPDATER=1 DISABLE_NON_ESSENTIAL_MODEL_CALLS=1 \
       CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
cd "${OPS_WS:?OPS_WS 未设置}" || exit 1
SYS=()
if [ -n "${OPS_SYSTEM_PROMPT_FILE:-}" ] && [ -r "${OPS_SYSTEM_PROMPT_FILE}" ]; then
  SYS=(--append-system-prompt "$(cat "${OPS_SYSTEM_PROMPT_FILE}")")
fi
exec /opt/opbench-node/bin/claude -p --output-format json \
  --allowedTools Bash Read Write Edit Glob Grep TodoWrite \
  --dangerously-skip-permissions "${SYS[@]}"
