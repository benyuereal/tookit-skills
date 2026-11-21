# dcu-flashattn-source — flash-attn 海光 DCU 适配源码索引

flash-attn（独立 attention kernel 库）在海光 DCU 上的适配源码索引：三层适配结构（kernel hipify → gfx936 / Python 接口扩展 / 运行时架构分支）、FA3 不可移植结论、版本谱系（DCU 源 release wheel vs 上游 sdist diff）。

## 目录

```
dcu-flashattn-source/
└── SKILL.md   # agent 向：三层适配结构 + 版本谱系 + 验证方法
```

## 快速用法 / 要点

- 核心结论先行：FA3 不可移植，DCU 上最新可用是 FA2 系。
- 查「flash-attn 适配改了哪些文件 / 版本串怎么读 / 哪个 build 最新」→ §2 版本谱系。
- 与 `dcu-aiter-source`（AITER 算子库）、`dcu-vllm-source`（vLLM 框架层 attention）区分：本 skill 专讲 flash-attn 这个独立库。

> 详细内容见 `SKILL.md`。