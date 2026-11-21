# dcu-vllm-custom-op — 海光 DCU 版 vLLM 自定义算子开发流程

> agent 向规范见 [`SKILL.md`](./SKILL.md)。本文件面向使用者。

## 它解决什么

在 **海光 DCU 版 vLLM** 中**从零开发一个自定义算子**的标准流程，并标注 ROCm 下的关键改造点。

与 [`dcu-vllm-op-replace`](../dcu-vllm-op-replace/) 互补：

| skill | 解决 |
|---|---|
| `dcu-vllm-op-replace` | 已有算子，**替换/重定向**进 vLLM 原生调用 |
| `dcu-vllm-custom-op`（本 skill） | **从零造一个算子** + DCU 适配（dispatch `is_rocm`、HIP 替换 CUDA、重编译纪律） |

## 包含

- **11 步通用开发流程**（PyTorch 参考 → Triton → CUDA/HIP 优化 → dispatch → binding → fallback → 单测 → 性能/精度评估）
- **3 个 DCU 适配要点**：① `is_cuda()` 在 ROCm 为 False，dispatch 须加 `is_rocm()`（★头号坑）② kernel 后端 CUDA→HIP ③ 重编译必挂 `/opt/hyhal:ro` + `VLLM_TARGET_DEVICE=rocm`
- **真实代码骨架**：基于 vLLM 官方 `merge_attn_states.py` dispatch 层 + `_custom_ops.py` 注册层
- **可复制模板**：[`scripts/vllm_custom_op_dispatch_template.py`](./scripts/vllm_custom_op_dispatch_template.py)（含海光适配分支 + 11 步核对清单）
- **一手溯源**：vLLM PR #16173（merge_attn_states, 3x~4.4x 加速）、官方 CustomOp 文档、同源知乎问答；并说明原文（知乎 zhuanlan p/1892966682634473987）在本环境无法读取

## 快速使用

```bash
# 1. 复制模板
cp scripts/vllm_custom_op_dispatch_template.py your_op_dispatch.py
# 2. 把 YourOp / 张量参数 / 约束校验改成你的算子
# 3. 按 SKILL.md §一 11 步 + §二 DCU 适配点 + §四 验证闭环 走完
```

## 适用版本

机制对 vLLM 上游与海光版通用（注册机制未改）；DCU 适配点在 `current_platform.is_rocm()` 与 csrc HIP 分支。跨版本请先读对应 `dcu-vllm-source-*`。
