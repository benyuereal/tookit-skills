# dcu-mhc-kernels-source — MHC/线性注意力 kernel 包 DCU 适配索引

MHC（Meta Hidden Computations / 线性注意力）kernel 包在海光 DCU 上的适配源码索引：causal-conv1d、fast-hadamard-transform、mhc-fused-kernel 三个包的 HIP 适配（Megatron MHC 融合 kernel 路径）。

## 目录

```
dcu-mhc-kernels-source/
└── SKILL.md   # agent 向：三包源码结构 + 适配点
```

## 快速用法 / 要点

- 查「causal-conv1d / fast-hadamard / mhc-fused 在 DCU 上的适配落点」→ 对应小节。
- 线性注意力模型（MHC 系）部署遇到 kernel 缺失/编译问题时先看这里。

> 详细内容见 `SKILL.md`。