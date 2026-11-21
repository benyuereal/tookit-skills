# dcu-lightop-moe — LightOp MoE 融合算子深度适配

LightOp 的 MoE（混合专家）融合算子深度适配索引。覆盖自研 MoE GEMM 多量化路径（w8a8/w4a8/w16a16 + Marlin）、asm 后端、Kimi-K2/sqrtsoftplus 专用 fused gate、permute/unpermute。

## 目录

```
dcu-lightop-moe/
└── SKILL.md
```

## 快速用法 / 要点

- 多量化路径选型（w8a8/w4a8/w16a16/Marlin）→ SKILL.md 量化路径段
- permute/unpermute 与 AITER MoE 的差异见对照
- 与 dcu-lightop-source 配套

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
