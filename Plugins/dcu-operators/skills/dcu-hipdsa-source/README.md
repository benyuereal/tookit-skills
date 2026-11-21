# dcu-hipdsa-source — hip-dsa / hipdnn DCU 适配源码索引

hip-dsa / hip-dsa-kernels / hipdnn 在海光 DCU 上的适配源码索引：cudnn.DSA（DeepSeek 稀疏注意力）的 HIP 替代实现（14 个 .hip kernel 源码在 wheel 内）+ hipdnn PReLU 适配点。

## 目录

```
dcu-hipdsa-source/
└── SKILL.md   # agent 向：源码结构 + kernel 清单 + 适配落点
```

## 快速用法 / 要点

- 查「DeepSeek 稀疏注意力在 DCU 上用什么替代 cudnn」→ cudnn.DSA 的 HIP 实现与 kernel 源码位置。
- 与 `dcu-flashattn-source`（flash-attn 库）、`dcu-aiter-source`（AITER）按库区分，别交叉找。

> 详细内容见 `SKILL.md`。