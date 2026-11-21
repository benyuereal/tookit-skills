---
name: large-class-init-style
description: Structure SGLang Scheduler, TokenizerManager, and ModelRunner constructors as explicit, overridable lifecycle steps. Use when modifying or reviewing initialization logic in those three classes, especially resource allocation, backend selection, graph setup, or distributed construction.
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# `__init__` Style for Scheduler / TokenizerManager / ModelRunner

Apply when modifying the `__init__` of:

- `Scheduler` — `python/sglang/srt/managers/scheduler.py`
- `TokenizerManager` — `python/sglang/srt/managers/tokenizer_manager.py`
- `ModelRunner` — `python/sglang/srt/model_executor/model_runner.py`

## Why

- Downstream forks override one piece (tokenizer, KV cache, IPC, …).
- Inline logic forces them to copy the whole `__init__`, which rots against upstream.
- Splitting into `init_*` helpers lets them override exactly what they need.
- Reference shape: `TokenizerManager.__init__` in `python/sglang/srt/managers/tokenizer_manager.py`.

## Rules

- **`__init__` is an orchestrator.** Sequence of `self.init_*(...)` calls + minimal glue. No non-trivial construction inlined.
- **One helper per overridable unit.** Each `init_*` = one concern a subclass might swap. Don't lump.
- **Naming:** `init_<thing>` (snake_case, names the component). Conditional construction → `maybe_init_<thing>`, gate inside the helper.
- **No silent state coupling.** A helper only reads `self.*` set by earlier helpers. Ordering lives in `__init__`. Shared intermediates → pass as args, not via `self.*`.
- **New logic = new helper.** Default to adding `init_<thing>`, not another inline block. One-line `self.foo = server_args.foo` is fine; structured logic is not.
- **Preserve override points.** Prefer additive changes to existing `init_*` signatures. Breaking changes → call out in PR.
- **No hidden device work.** Do not conceal GPU allocation, synchronization, compilation, graph capture, communicator creation, or network I/O in a generic helper. Name the lifecycle stage and make ordering explicit.
- **Rollback-friendly gates.** Hardware-specific or experimental initialization must have a clear default, activation log, and fallback path.
- **Verify every lifecycle.** Test eager and graph modes, single and distributed ranks, startup failure cleanup, and repeated construction where supported.

## Scope

Only the three classes listed above. Not other manager-style classes, not small dataclass/utility constructors.
