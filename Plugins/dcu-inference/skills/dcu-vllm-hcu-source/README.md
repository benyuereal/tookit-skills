# dcu-vllm-hcu-source — vllm-hcu 插件式 patch 层源码索引

vLLM 在海光 DCU/HCU 上的插件式适配包 vllm-hcu 的源码结构：105 个 patch、HCUPlatform 插件机制、模块交换（module swap）如何在不改 vLLM 本体的前提下注入 DCU 适配。适用于 0.21+ 时代独立包形态。

## 目录

```
dcu-vllm-hcu-source/
└── SKILL.md   # agent 向：patch 清单 + HCUPlatform + 模块交换机制
```

## 快速用法 / 要点

- 查「vLLM 的 DCU 适配改在哪」→ 105 个 patch 按模块索引。
- 与 `dcu-vllm-source`（vLLM 本体各版本源码结构）、`dcu-vllm-runtime-patch`（运行时 monkey-patch 范式）、`dcu-vllm-custom-op`（自定义算子）分工明确，别混。

> 详细内容见 `SKILL.md`。