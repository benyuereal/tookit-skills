# dcu-tilelang-source — TileLang DCU 适配源码索引

TileLang（TVM 系 DSL 编译器）在海光 DCU 上的适配源码结构：`contrib/hcu.py` 海光 HIP 后处理、`TILELANG_OVERRIDE_DEVICE_SOURCE` 替换生成 kernel 的机制、libtilelang 链接 DTK、MLS/GEMM_MLS 架构白名单（gfx936 限制）。

## 目录

```
dcu-tilelang-source/
└── SKILL.md   # agent 向：源码地图 + override 机制 + MLS 白名单 + 验证
```

## 快速用法 / 要点

- 写 TileLang kernel 在 DCU 上跑 → 先看 hcu 后端与编译 flags（gfx936 是合法目标）。
- **MLS 白名单坑**：`T.matrix_load`/`T.ds_read_format` 走的 MLS 路径仅 gfx938/gfx92a/gfx946，gfx936（BW100/BW1000/K100-AI 档）上用了必崩——规避见 SKILL.md §2b。
- kernel cache 会跳过 lowering，override 调试时设 `TILELANG_DISABLE_CACHE=1`。

> 详细内容见 `SKILL.md`。