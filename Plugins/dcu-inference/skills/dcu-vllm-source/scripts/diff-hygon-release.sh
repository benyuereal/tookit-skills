#!/usr/bin/env bash
# diff-hygon-release.sh — 对比上游 vLLM 与海光发布版，定位真实差异
#
# 用法: bash diff-hygon-release.sh <上游目录> <海光发布版目录>
#
# 方法: diff -rq 列出差异文件 + 逐文件行级计数（新增/删除行）。
# 背景: 海光发布版往往 Python 层零差异、仅 CMakeLists.txt 裁 gfx；
#       本脚本用于"新版本到手先跑，不要臆测适配点"
#       （见 dcu-inference-infra / dcu-aiter-source 的引用）。
# 运行前提: 两个目录都已在本地（上游 tarball 解压 + 海光发布版 wheel 解包）。
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "用法: bash $0 <上游目录> <海光发布版目录>" >&2
  exit 2
fi

UPSTREAM="$1"
HYGON="$2"

if [ ! -d "$UPSTREAM" ]; then
  echo "ERR: 上游目录不存在: $UPSTREAM" >&2
  exit 1
fi
if [ ! -d "$HYGON" ]; then
  echo "ERR: 海光发布版目录不存在: $HYGON" >&2
  exit 1
fi

echo "=== diff-hygon-release ==="
echo "上游:       $UPSTREAM"
echo "海光发布版: $HYGON"
echo

# 1) 文件级差异（diff -rq）
echo "--- 文件级差异（diff -rq）---"
DIFF_OUT=$(diff -rq "$UPSTREAM" "$HYGON" || true)
if [ -z "$DIFF_OUT" ]; then
  echo "OK: 两目录完全一致，无差异文件。"
  exit 0
fi
echo "$DIFF_OUT"
echo

# 2) 逐文件行级计数（仅对"内容不同"的文件）
echo "--- 行级计数（海光相对上游: +新增 / -删除）---"
TOTAL_ADD=0
TOTAL_DEL=0
while IFS= read -r line; do
  case "$line" in
    "Files "*differ)
      f_up=$(printf '%s\n' "$line" | sed -n 's/^Files \(.*\) and \(.*\) differ$/\1/p')
      f_hy=$(printf '%s\n' "$line" | sed -n 's/^Files \(.*\) and \(.*\) differ$/\2/p')
      if [ -f "$f_up" ] && [ -f "$f_hy" ]; then
        add=$(diff "$f_up" "$f_hy" | grep -c '^>' || true)
        del=$(diff "$f_up" "$f_hy" | grep -c '^<' || true)
        TOTAL_ADD=$((TOTAL_ADD + add))
        TOTAL_DEL=$((TOTAL_DEL + del))
        rel="${f_up#"$UPSTREAM"/}"
        printf '  %-58s +%s -%s\n' "$rel" "$add" "$del"
      fi
      ;;
    "Only in "*)
      echo "  [仅一方存在] $line"
      ;;
  esac
done <<< "$DIFF_OUT"

echo
echo "=== 汇总 ==="
echo "海光相对上游: 新增 $TOTAL_ADD 行, 删除 $TOTAL_DEL 行"
echo "OK: diff 完成。"
