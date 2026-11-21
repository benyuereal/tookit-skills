# dcu-train-precision — 海光 DCU 训练精度排查

海光 DCU 平台深度学习训练精度排查最佳实践：5 类根因分类、通用排查四步法、逐层 forward hook 调试工具、常见处理（AMP/MMCV/Flash-Attention/Paddle softmax）、快速参考表、NaN/Inf 诊断脚本。

## 目录

```
dcu-train-precision/
├── SKILL.md                      # agent 向：完整方法 + 参考表 + 脚本引用
└── scripts/
    ├── layer_hook_debug.py       # 逐层 forward hook：打印每层输入/权重/输出前10值
    └── diagnose_nan.py           # NaN/Inf 快速诊断（遍历前5 batch）
```

## 快速用法 / 要点

- 核心：NV vs DCU **同随机种子**各跑一遍，对比 debug.log 逐层定位**首个异常层**
- 四步法：环境复现 → 逐层采集 → 差异定位 → 根因修复
- 四类常见问题：AMP loss NaN（升级 Transformers / sdpa 加 contiguous / FP32 兜底）、MMCV（FP16→FP32）、Flash-Attention 兼容（升级 / eager 禁用）、Paddle softmax 溢出（手写 stable_softmax）
- 外置脚本已 py_compile 验证；不存性能基线/逐行日志

> 详细内容（参考表、patch 片段、诊断流程）见 `SKILL.md`。
