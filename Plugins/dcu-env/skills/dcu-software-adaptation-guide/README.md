# dcu-software-adaptation-guide — DCU 软件适配通用参考

拿到任意新软件包要适配到 DCU 时的**总方法论**：5 大适配形态决策树 + 7 步 SOP + DTK 与 ROCm 核心差异 + 适配点定位技巧 + 验证清单，由 19 个包的实测经验归纳。与 `dcu-adaptation-overview`（查表现成结论）互补——本 skill 教「怎么做」，总览查「结论是什么」。

## 目录

```
dcu-software-adaptation-guide/
└── SKILL.md   # agent 向：形态决策 + SOP + DTK 差异 + 定位技巧 + 验证清单 + 踩坑
```

## 快速用法 / 要点

- 先走 §0 快速决策：纯 Python / wheel 带 HIP 源码 / patch 层 / 需源码编译 / 不可适配，五形态选对路径。
- §3 DTK vs ROCm 差异（libgalaxyhip / HIP runtime 命名 / gfx 架构）是所有适配的公共背景，必读。
- §4 定位技巧：wheel 解包 + ldd + grep 上游 diff，快速找到海光改了哪里。
- 19 包踩坑集中在 §7，遇到报错先对照。

> 详细内容见 `SKILL.md`。