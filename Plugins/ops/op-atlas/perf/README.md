# perf/ —— 性能数据的唯一落点

> **禁止手写性能数字进卡片。** 卡片的 ratio / last_verdict 必须引用本目录的 JSON。

## 每条记录的格式（缺一不予采纳）

```json
{
  "variant": "lightop-fused",
  "median_us": 207.4,
  "unit": "us",
  "caliber": {
    "what": "op_call",
    "alloc_in_timing": false,
    "state": "hot",
    "input_strategy": "preclone_pool_rotating",
    "pool_size": 64,
    "timer": "hip_event",
    "rounds": 7,
    "statistic": "median",
    "mad": 1.2,
    "n": 7,
    "clock_mhz": 1350,
    "gpu_exclusive": true
  },
  "traffic_passes": {"reads": 2, "writes": 2},
  "shape": {"M": 8192, "H": 4096, "dtype": "bfloat16", "name": "prefill-8k"},
  "evidence": {"file": "results/raw.json", "sha256": "...", "verified_on": "2026-11-23"},
  "reproduce": "python3 -I bench/bench.py --shape prefill-8k"
}
```

## 三条纪律

1. **无 caliber → 门禁拒绝入库**（`verified_on` 超 30 天 → 报警）
2. **无 reproduce → 标 unverified**
3. **不同 caliber 的数不许放同一张排序表** —— `atlas_query.py select` 会按 caliber 分组，跨组直接拒绝合并

## 文件名

`<family>-<card>.json`，如 `gemm-bf16.json` / `fusion-rmsnorm-rope.json`。
