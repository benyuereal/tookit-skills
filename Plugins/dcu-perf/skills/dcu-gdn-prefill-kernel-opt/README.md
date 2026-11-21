# dcu-gdn-prefill-kernel-opt — GDN 线性注意力 Prefill Kernel 优化

优化 DCU 上 GDN（线性注意力）**prefill** kernel、降低 prefill TTFT。以 Qwen3.8-27B + vLLM 0.18.1 + hcu 插件为
实测基线（2026-10-08，4×DCU TP4），给出从源码改动到收益验证的完整闭环。

## 核心内容

- §0 为什么优化 GDN prefill kernel
- **优化 #1（prefill 大头）**：`chunk_o.py` 新增 share_qk3 kernel + `chunk_fwd_o()` gate
- **优化 #2**：`solve_tril.py` 避免整矩阵零化
- 优化 #3（可选）：`chunk_delta_h.py` 删 aiter 分支
- 实测收益（4×DCU TP4 / Qwen3.8-27B）
- 验证方法（必须做，别只信性能）+ 踩坑

## 目录

```
dcu-gdn-prefill-kernel-opt/
└── SKILL.md
```

## 衔接

- 优化后务必按 §5 验证（数值 + 端到端），再用 `dcu-llm-benchmark` 出可信吞吐/延迟数据。

详见 `SKILL.md`。
