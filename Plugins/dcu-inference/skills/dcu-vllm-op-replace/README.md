# dcu-vllm-op-replace — vLLM 自定义算子替换流程

在 vLLM（海光 DCU 版）中把自定义算子替换/重定向原生算子的标准流程：HIP kernel 编写→torch_bindings 注册→_custom_ops.py 登记→platforms/rocm.py 的 Dispatch/compilation_config 重定向→重编译→验证。

## 目录

```
dcu-vllm-op-replace/
└── SKILL.md
```

## 快速用法 / 要点

- 全流程：kernel 编写 → torch_bindings → _custom_ops → rocm.py 重定向 → 重编译 → 验证
- 第 1 步写 kernel 的细节见 dcu-kernel-dev（骨架）
- 验证正确性+性能闭环见 dcu-op-optimization

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
