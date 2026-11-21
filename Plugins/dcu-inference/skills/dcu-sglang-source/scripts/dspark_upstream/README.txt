DSpark (Confidence-Driven Speculative Decoding) upstream source snapshot
Source: sgl-project/sglang main branch (post PR #30261, integrated 2026-07-06)
Captured: 2026-08-06
Layout mirrors sglang package:
  sglang/srt/speculative/dspark_components/   (12 files, no kernels/ subdir - refactored upstream)
  sglang/srt/speculative/ragged_verify.py
  sglang/srt/speculative/spec_registry.py / spec_info.py / base_spec_worker.py (modified by PR)
  sglang/srt/models/dspark.py / deepseek_v4_dspark.py
  sglang/srt/arg_groups/speculative_hook.py
  sglang/kernels/ops/speculative/dspark/      (4 Triton kernels: accept/draft_model/schedule/verify_window)
NOTE: kernels were originally under dspark_components/kernels/ in PR #30261 but refactored
      into sglang/kernels/ops/speculative/dspark/ on main. The 7 PR-listed kernels/ files do NOT
      exist on main anymore (merged). This snapshot uses the current main layout.
