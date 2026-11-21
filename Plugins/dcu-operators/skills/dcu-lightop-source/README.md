# dcu-lightop-source — LightOp 自研融合算子库源码

LightOp（海光自研融合算子库）源码结构、构建、DTK/gfx 适配要点、算子族索引。与 dcu-aiter-source 并列：AITER=AMD ROCm 上游的 DCU 适配版；LightOp=海光自研，二者均做 MoE/GEMM 类融合算子。

## 目录

```
dcu-lightop-source/
└── SKILL.md
```

## 快速用法 / 要点

- 找 LightOp 算子在源码落点 → 算子族索引
- DTK/gfx 适配要点见对应段
- 构建见 SKILL.md

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
