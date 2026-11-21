# dcu-triton-source — Triton DCU/HCU 源码适配索引

Triton 在海光 DCU/HCU 上的源码结构与适配点：独立 hcu backend（DTK 特化：libgalaxyhip / aillvm / gfx936-938 768-vgprs）+ 调优 CLI。写/调 Triton kernel 前先了解 hcu backend 与上游 triton 的差异。

## 目录

```
dcu-triton-source/
└── SKILL.md   # agent 向：源码结构 + hcu backend 特化点 + 调优 CLI
```

## 快速用法 / 要点

- hcu backend 是独立 backend（非上游 nvidia/AMD 后端改），编译路径与约束不同。
- 768-vgprs、gfx936/gfx938 差异等硬件相关特化见 SKILL.md 对应段。
- 实际写 kernel 的方法论转 `dcu-triton-kernels`（coding-skill 子包），本 skill 管源码结构。

> 详细内容见 `SKILL.md`。