# dcu-adaptation-overview — DCU 适配总览

DCU 适配的**总入口速查**：版本兼容矩阵（DTK × torch × 框架 × 算子包）、软件能力矩阵（哪些能跑/验证状态/限制）、真机验证结果、升级 SOP。拿到「这环境装什么版本 / 这包能不能用 / 怎么升级」类问题先看这里，再下钻到对应专项 skill。

## 目录

```
dcu-adaptation-overview/
└── SKILL.md   # agent 向：兼容矩阵 + 能力矩阵 + 真机验证 + 升级 SOP
```

## 快速用法 / 要点

- 先查 §0 快速查表定位问题类型，再按需进 §1-§4。
- 版本兼容矩阵以 DTK 26.04 体系为基线；能力矩阵标注了验证状态与限制。
- 升级前必读 §4 升级 SOP（顺序错会白装）。
- 逐包细节下钻：`dcu-env-packages`（包清单）、`dcu-dtk`/`dcu-dtk-layout`（DTK 安装与目录）、各 `*-source`（源码级适配点）。

> 本 skill 详细内容（矩阵全表、SOP 步骤、踩坑）见 `SKILL.md`。