# dcu-aiter-moe — AITER MoE 融合算子深度适配

AITER 的 MoE（混合专家）融合算子深度适配索引。覆盖四后端调度框架（moe_c/asm/triton/ck）、get_aiter_moe_config 后端选择、aiter_moe 完整调用树、shuffle 必做踩坑、分 gfx 调优表、csrc/asm kernel 定位。

## 目录

```
dcu-aiter-moe/
└── SKILL.md
```

## 快速用法 / 要点

- 查某 MoE 算子该走哪个后端（moe_c/asm/triton/ck）→ SKILL.md 四后端调度
- shuffle 是必做前置，漏做结果错；调优按 gfx 查分 gfx 调优表
- 与 dcu-aiter-source 配套（source 看结构，本 skill 看 MoE 算子细节）

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
