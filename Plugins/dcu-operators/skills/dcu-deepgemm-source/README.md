# dcu-deepgemm-source — DeepGEMM 自研 GEMM 库源码

DeepGEMM（海光自研/适配的通用 GEMM 库，基于 DeepSeek DeepGEMM 移植+改造）源码结构、构建、海光适配要点、GEMM 路径索引。专注 grouped GEMM（普通+MoE 分组），支持 FP8/INT8/BF16。

## 目录

```
dcu-deepgemm-source/
└── SKILL.md
```

## 快速用法 / 要点

- 普通 GEMM / MoE 分组 GEMM 路径索引 → SKILL.md GEMM 路径段
- 与 AITER·LightOp GEMM 选型对比见对应 skill
- 量化/精度支持（FP8/INT8/BF16）见适配要点

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
