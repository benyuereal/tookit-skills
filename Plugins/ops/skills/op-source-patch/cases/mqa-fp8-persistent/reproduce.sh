#!/bin/bash
# 复现 mqa-fp8-persistent 案例：定位→取源→编译→改(patch)→回归→留痕。
#
# 本案例的源码来自本地（**不重拉**）：
#   /workspace/lightop/lzh_glm_new  —— LightOp 源码树（已解包）
#   若只有 tgz：/workspace/lightop/lightop_lzh_glm_new.tgz
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../../../.." && pwd)
PY=${PYTHON:-python3}
SRC=${LIGHTOP_SRC:-/workspace/lightop/lzh_glm_new}

usage() {
  cat <<'USAGE'
用法: ./reproduce.sh <目标>
  logic    逻辑回归（不需 GPU）：patch 前后 persistent 的 dtype 白名单是否按预期变化
  apply    在临时副本上应用 patch 并验证干净套用（不改原源码树）
  all      logic + apply

环境变量:
  LIGHTOP_SRC  LightOp 源码树（默认 /workspace/lightop/lzh_glm_new）
  PYTHON       解释器（默认 python3）
USAGE
}

need_src() {
  [ -d "$SRC" ] || { echo "找不到源码树 $SRC；先解包 lightop 的 tgz" >&2; exit 2; }
}

case "${1:-logic}" in
  logic)
    need_src
    TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
    cp "$SRC/lightop/attention/mqa.py" "$TMP/after.py"
    ( cd "$TMP" && mkdir -p lightop/attention && cp after.py lightop/attention/mqa.py \
      && patch -p1 --silent < "$HERE/patches/0001-disable-persistent-for-fp8.patch" ) \
      || { echo "patch 套用失败" >&2; exit 1; }
    "$PY" "$HERE/validate_logic.py" \
      --before "$SRC/lightop/attention/mqa.py" \
      --after "$TMP/lightop/attention/mqa.py"
    ;;
  apply)
    need_src
    TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
    cp -r "$SRC/lightop" "$TMP/"
    ( cd "$TMP" && patch -p1 --dry-run < "$HERE/patches/0001-disable-persistent-for-fp8.patch" ) \
      && echo "PASS: patch 干净套用（无 fuzz）"
    ;;
  all) "$0" logic; "$0" apply ;;
  *) usage; exit 2 ;;
esac
