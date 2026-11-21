#!/usr/bin/env bash
# install_sync_selftest.sh —— 校验 install.sh 的清单式增量同步（sync_tree）合并语义。
# 用法：bash scripts/install_sync_selftest.sh [仓库根]
# 覆盖：用户新增文件不删 / 上游移除且未改过→删 / 上游移除但本地改过→保留 /
#       厂商更新→覆盖并备份 .local.bak / 源路径带尾斜杠仍能拷贝（回归） / 重跑幂等。
set -u
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
TMP="$(mktemp -d)"
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then echo "  PASS $1"; pass=$((pass+1)); else echo "  FAIL $1: got[$2] want[$3]"; fail=$((fail+1)); fi; }

# 抽取 install.sh 里的同步函数（从 MANIFEST_DIR= 到 install_one_skill 结束）
awk '/^MANIFEST_DIR=/{f=1} f{print} /^install_one_skill\(\)/{p=1} p&&/^}/{exit}' "$ROOT/install.sh" > "$TMP/blk.sh"
[ -s "$TMP/blk.sh" ] || { echo "FATAL: 未能从 install.sh 抽取同步函数"; exit 2; }
warn() { echo "    [warn] $*"; }
err() { echo "    [ERR] $*"; }
CONF_DIR="$TMP/conf"; mkdir -p "$CONF_DIR"
. "$TMP/blk.sh"

SRC="$TMP/src"; DST="$TMP/dst"
mkdir -p "$SRC/sub"
printf 'a1\n' > "$SRC/a.md"; printf 'c1\n' > "$SRC/sub/c.md"; printf 'k1\n' > "$SRC/keep.md"; printf 'g1\n' > "$SRC/gone.md"

# 第一次同步：源路径**带尾斜杠**（回归用例）
sync_tree "$SRC/" "$DST" >/dev/null 2>&1
chk "尾斜杠源路径仍能拷全(=4)" "$(find "$DST" -type f | wc -l)" "4"
mkdir -p "$DST"
sync_tree "$SRC/" "$DST" >/dev/null 2>&1
chk "尾斜杠源路径仍能拷全(=4)" "$(find "$DST" -type f | wc -l)" "4"

# 用户沉淀 + 本地改动
printf 'USER\n' > "$DST/user_added.md"
printf 'LOCAL\n' >> "$DST/keep.md"
printf 'LOCAL2\n' >> "$DST/a.md"

# 上游：a.md 改内容、移除 keep.md 与 gone.md、新增 new.md
printf 'a2\n' > "$SRC/a.md"; rm -f "$SRC/keep.md"; rm -f "$SRC/gone.md"; printf 'n1\n' > "$SRC/new.md"
sync_tree "$SRC/" "$DST" >/dev/null 2>&1

chk "用户新增文件保留" "$(cat "$DST/user_added.md" 2>/dev/null)" "USER"
chk "上游更新生效" "$(cat "$DST/a.md")" "a2"
chk "上游新文件同步" "$(cat "$DST/new.md")" "n1"
chk "上游移除且未改过→删" "$([ -e "$DST/gone.md" ] && echo present || echo absent)" "absent"
chk "上游移除但本地改过→保留" "$(cat "$DST/keep.md")" "$(printf 'k1\nLOCAL')"
chk "本地改动被覆盖前已备份" "$(cat "$DST/a.md.local.bak")" "$(printf 'a1\nLOCAL2')"

# 幂等
M1="$(find "$DST" -type f -printf '%T@ %p\n' 2>/dev/null | sort)"
M1="$(find "$DST" -type f -printf '%T@ %p\n' | sort)"
sync_tree "$SRC/" "$DST" >/dev/null 2>&1
M2="$(find "$DST" -type f -printf '%T@ %p\n' | sort)"
chk "重跑幂等（不改动已有文件）" "$([ "$M1" = "$M2" ] && echo same || echo changed)" "same"

echo ""
echo "RESULT: pass=$pass fail=$fail"
rm -rf "$TMP"
[ "$fail" = 0 ] || exit 1