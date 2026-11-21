# dcu-fastsafetensors-source — fastsafetensors DCU 适配源码索引

fastsafetensors（safetensors 并行加载库）在海光 DCU 上的源码结构与适配点：`gpu_compat.h` 运行时 dlopen CUDA/HIP 双路径、GDS/ThreeFS copier 的 DCU 处理。基于 wheel 解包 + 上游 diff 实测。

## 目录

```
dcu-fastsafetensors-source/
└── SKILL.md   # agent 向：源码地图 + 适配点 + 验证入口
```

## 快速用法 / 要点

- 查「fastsafetensors 在 DCU 上改了哪里 / 为什么能同时兼容 CUDA/HIP」→ gpu_compat.h 双路径设计。
- GDS/ThreeFS 文件拷贝路径在 DCU 上的落点见 SKILL.md 对应段。
- 与 `dcu-vllm-source`/`dcu-sglang-source`（框架层适配）互补，本 skill 只讲这个库本身。

> 详细内容见 `SKILL.md`。