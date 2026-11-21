---
name: scripted-runtime-notes
description: Design and review SGLang scripted-runtime tests and harness APIs while preserving real scheduler lifecycle semantics. Use when adding probes, control primitives, hooks, or tests around request, batch, KV-cache, pause/resume, abort, or scheduler behavior.
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# Scripted Runtime — Notes

Notes for anything related to the SGLang scripted runtime.

The harness must drive the same state transitions as production. A passing assertion is insufficient if the test bypasses request ingestion, scheduler ordering, cache ownership, or worker synchronization. When testing an optimized path, add an observable activation consequence and verify output IDs or hashes at exact token counts.

## When to Add an API

Tests read `r.req.*` and `t._scheduler.*` directly — there is no encapsulation boundary. A thin wrapper buys zero isolation; it only grows the surface.

Add an API only if it does real work:

1. **Control primitive** — drives the engine through a real path (`start_req`, `pause_generation`, `abort`, `evict_radix`, `exhaust_kv`). Reuse the real path; never hand-mutate state.
2. **Hook-backed** — value cannot be read from a snapshot; accumulate via `scheduler_hook.on_run_batch` or the recv proxy (`chunks_done`). Read-only; never monkey-patch; never add `*_count` to `srt/`.
3. **Multi-structure derivation, widely reused** — scans `chunked_req` + `waiting_queue` + `running_batch` + `last_batch` (`is_idle`, `status`, `batch_composition`).

Else: don't. Read `r.req.X` / `t._scheduler.X` in the test; inline single-use accessors.

Never:

- Weaken an assertion to fit a missing probe.
- Probe implementation details ("field non-None", "which branch ran") — assert the consequence.

## Other Tips

- **Engine-self-driven behavior: drive the real loop, don't call the private.** Never synchronously call a scheduler private (e.g. `scheduler._abort_on_waiting_timeout()`) from the harness/test — it runs at the wrong loop phase, bypasses the ordered `recv_requests` → `process_input_requests` injection, and can fire in states the real loop never reaches (e.g. while paused). For sweeps the engine runs itself (timeout/idle), enable the config/env and advance the loop with `yield`.
