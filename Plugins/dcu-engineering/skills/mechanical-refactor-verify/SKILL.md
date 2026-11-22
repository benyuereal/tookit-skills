---
name: mechanical-refactor-verify
description: "用可复现的变换 + 严格的语义变更边界来验证机械重构（mechanical refactor verify）：拆文件、移函数、重命名、抽模块。何时用：提交或审阅机械重构 commit/PR。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# Mechanical Refactor — Reproducible Verification

## Core Principle

The deliverable of a mechanical move (file split, function move, module extraction) is NOT the diff — it is **the script that produces the diff**.
A script is auditable; a diff is not.

## Workflow

Regardless of who did the move (human or agent) and when (before or after committing), the workflow is the same:

### Step 1: Write the transform script to /tmp/

Write the script to `/tmp/transform_<short_description>.py`. **Never write it inside the repo.**

The scaffold (worktree creation, diff check, ruff format, result reporting) lives in `mechanical_refactor_verify_utils.py` next to this skill.

**MANDATORY**: The transform script MUST use `verify_mechanical_refactor()` from the utils module. Do NOT reimplement the verification scaffold — no hand-written worktree management, no hand-written diff checking. The script only defines `transform()` and calls `verify_mechanical_refactor`.

Script template (follow this structure exactly):

```python
#!/usr/bin/env python3
"""Reproducible transform for: <describe the mechanical move>

Run from the repo root:  python3 /tmp/transform_<short_description>.py
"""
import sys
from pathlib import Path

sys.path.append(".claude/skills/mechanical-refactor-verify")
from mechanical_refactor_verify_utils import verify_mechanical_refactor, exec_command, git_add_and_commit, dedent

BASE_COMMIT = "<base_sha>"
TARGET_COMMIT = "<pr_mechanical_move_final_sha>"


def transform(dir_root: Path) -> None:
    """Perform the mechanical transformation and commit each step.

    Args:
        dir_root: Path to the worktree (checked out at BASE_COMMIT).
    """
    # --- Step 1: Split source file ---
    source = dir_root / "path/to/source.py"
    content = source.read_text()
    lines = content.splitlines(keepends=True)

    splits = [
        ("path/to/pkg/target_a.py", 1, 50),
        ("path/to/pkg/target_b.py", 51, 120),
    ]
    for target_path, start, end in splits:
        target = dir_root / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines[start - 1 : end]))

    source.unlink()
    (dir_root / "path/to/pkg/__init__.py").touch()

    git_add_and_commit("mechanical: split source.py", cwd=str(dir_root))

    # --- Step 2: Fix imports ---
    # <edit files>
    # git_add_and_commit("fix imports", cwd=str(dir_root))

    # Note: pre-commit run --all-files is run automatically after transform() returns


if __name__ == "__main__":
    verify_mechanical_refactor(
        base_commit=BASE_COMMIT,
        target_commit=TARGET_COMMIT,
        transform=transform,
    )
```

### Step 2: Run the script from the repo root

```bash
cd <repo_root>
python3 /tmp/transform_<short_description>.py
# Expected: "PASS: transform reproduces the commit exactly."
```

If FAIL, fix the script and re-run until PASS.

### Step 3: Upload gist, delete local file, update PR description

One gist per PR. Do all three:

```bash
# 1. Create gist (or update existing)
gh gist create --public -d "Mechanical refactor transform: <description>" /tmp/transform_<short_description>.py
# Or update: gh gist edit <gist_id> -a /tmp/transform_<short_description>.py

# 2. Delete local file
rm /tmp/transform_<short_description>.py

# 3. Update PR description (paste the block below)
```

PR description must include:

````markdown
## Mechanical Move

Transform script: <gist_url>

### One-click verification

```bash
python3 <(curl -sL <gist_raw_url>)
```
````

### Step 4: PR scope

A mechanical refactor PR must contain **only** mechanical changes (moves, splits, renames, import fixes, formatting). All of these must be reproducible by the transform script.

Semantic changes (new logic, API restructuring, behavior changes) belong in a **separate PR**.

For performance work, keep kernel, dispatch, allocation, graph, synchronization, and configuration changes out of the mechanical commit. Reproduce the move first, then benchmark the semantic follow-up against the untouched baseline; otherwise a performance delta cannot be attributed.

## Verifying hand-written edits: use oracles, not recall

Bulk edits (renames, path/identifier rewrites, doc-口径 sync) fail in a way diff review misses: the
token you typed is *plausibly* right but not what exists — an existing skill name that shares a stem
with the pattern, a tool name whose digits are easy to transpose. The two spellings are visually
indistinguishable, so reading the file back cannot catch it. Verify against the environment:

- **Derive canonical values from the filesystem, then normalise to them.** Discover the real
dir/file/module name (`ls -d <glob>`, `ls <parent> | grep -i <prefix>`) and have the rewrite script
  load those values (JSON/env), instead of carrying a typed literal in the script body. Idempotent
  re-runs then cannot drift.
- **Token-existence check:** for every command a script or shell recipe calls, resolve it with
  `command -v <token>`; for module attributes, diff the tokens your file uses against the module's
  `dir()` set. This finds attribute typos statically, which a syntax check cannot.
- **When two spellings look identical to you, measure instead of reading:** compare the module's own
  `__name__` length, print the characters as codes (`printf %s "$n" | od -tu1`), or ask the FS for the
  real name (`ls /usr/bin/<prefix>*<sufix>`). Take the value the oracle returns; never re-decide by eye.
- **Syntax gates are necessary, not sufficient:** a shell `-n` parse and a Python byte-compile catch
  structure only — library argument actions keyed by string, module-attribute typos and missing paths
  only fail at runtime. Run the new script once on synthetic input, and exercise both branches of any
  tool-availability fallback before declaring it verified.
- **A search-replace family pattern over-reaches; bound it and re-grep.** A rule for the token
  `op-optimize` rewrote the longer *skill name* that shares the stem and turned a run path into
  `<path>.yaml/<subpath>`. Add boundaries / negative look-ahead (`(?<![-A-Za-z])tok(?!ization)`), then
  grep the touched files for the intended token; a zero-hit or a mangled path proves the over-reach.
- **Re-read the changed hunks once the automations finish** (the tool reports its diff) — a
  block-anchor edit can land one line off or drop an indent level, which only shows in the diff.

## Verifying an existing PR (`/mechanical-refactor-verify verify`)

1. Find the gist URL and one-click command in the PR description
2. Run the one-click command from the repo root
3. Report: PASS or show the diff
