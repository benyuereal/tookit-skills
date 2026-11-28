# {{op_name}} 契约（bootstrap 生成，需人工核对）

op: {{op_name}}
version: 1
math: "{{math}}"

shapes:
{{shapes_body}}
dtypes:
{{dtypes_body}}
tolerance:
  vs_fp32_ref:
{{tolerance_body}}
  variant_vs_variant:
    mode: tolerance

invariants:
  - name: determinism
    mode: bit_exact
    repeats: 5

gates:
  - name: correctness
    template: correctness
    required: true
  - name: invariants
    template: invariants
    required: true
  - name: perf
    script: bench/ab.py
    required: false
    skip_reason: "bootstrap 生成时还没有实测，bench 待接"

floor:
  metric: hbm_bandwidth
  # peak 必须来自 bw-ladder 的 achievable，不是型号标称值（1100 是型号名）
  peak: 1730
  unit: GB/s
  peak_source: Plugins/ops/op-atlas/bw-ladder.yaml
  basis: "bootstrap 按访问模式算出的字节数，见 floor_inputs"
  at_floor_threshold: 0.10
  status: unmeasured
