#!/usr/bin/env bash
set -euo pipefail

repo=${1:-.}
if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
  echo "not a git repository: $repo" >&2
  exit 2
fi

git -C "$repo" worktree list --porcelain | awk '
  /^worktree / { if (path != "") print path "\t" branch; path=substr($0, 10); branch="(detached)" }
  /^branch / { branch=substr($0, 8) }
  END { if (path != "") print path "\t" branch }
' | while IFS=$'\t' read -r worktree branch; do
  printf '\n## %s\nbranch: %s\n' "$worktree" "$branch"
  git -C "$worktree" log -5 --date=short --format='commit: %ad %h %s'
  find "$worktree/docs/reports" "$worktree/.qoder/plans" \
    -type f \( -name 'PLAN*.md' -o -name 'REPORT*.md' -o -name '*SUMMARY*.md' \) \
    2>/dev/null | sort | tail -20 | sed 's/^/artifact: /'
done
