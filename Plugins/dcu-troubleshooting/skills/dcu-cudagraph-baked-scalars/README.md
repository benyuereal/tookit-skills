# dcu-cudagraph-baked-scalars — CUDA/HIP graph 烧死 Python 标量排查

排查「CUDA/HIP graph 捕获时把每 batch 变化的 Python 标量烧死在 capture」导致的静默读错：
症状 = accept 从第 1 轮起为 0、输出错但不崩不报错。常见于 sglang/vLLM 投机解码 draft、attention backend
等 graph 化路径。

## 核心内容
- 症状特征：不崩不报错但结果静默错（如投机解码 accept 恒 0）
- 排查思路：先证伪 tensor 假设 → 再查 Python 标量是否被烧死
- 修复：把会变化的标量改为常量上界/正确传入

## 目录
```
dcu-cudagraph-baked-scalars/
└── SKILL.md   排查流程 + 修复手法
```

## 快速用法
vLLM/SGLang graph 化路径结果错但无异常时读 `SKILL.md` 按症状匹配。
