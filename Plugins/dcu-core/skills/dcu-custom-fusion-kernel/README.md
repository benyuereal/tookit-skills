# dcu-custom-fusion-kernel — DCU 自定义融合 kernel

当框架（vLLM/SGLang）对某模型配置**没有现成融合 kernel**时，写 DCU HIP 融合 kernel 填补缺口的 skill。
以 QK-Norm + RoPE 融合为范式，覆盖触发判断、kernel 设计、接线进框架（env 开关 + 可回滚）、两级验证。

## 核心内容

- **先确认"现成 kernel 真的不支持"**（别重复造轮子）
- QK-Norm + RoPE 融合 kernel 设计（raw-HIP）
- **DCU warp=64 的 shuffle 规约坑**（核心坑点）
- 接线进 vLLM / SGLang：env 开关 + 可回滚
- 两级验证（数值 + 端到端）
- SGLang 在 DCU 跑 Qwen3-Next 的配置坑

## 目录

```
dcu-custom-fusion-kernel/
├── SKILL.md
└── references/
    └── qwen38_qknorm_rope_kernel.md
```

## 衔接

- 先查 `dcu-kernel-layer-microbench` / `dcu-kernelwiki` 确认无现成 kernel，再进入本 skill 试写。
- 框架侧接线后，用 `dcu-op-optimization` 跑优化闭环验证。

详见 `SKILL.md`。
