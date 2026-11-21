# dcu-ort-source — ONNX Runtime DCU 适配源码索引

ONNX Runtime 在海光 DCU 上的源码结构与适配点：ROCm EP 链接 DTK `libgalaxyhip`、MIGraphX EP 的接入方式、二进制 wheel 的适配落点。运行时调优（非源码）见 `dcu-ort-perf`。

## 目录

```
dcu-ort-source/
└── SKILL.md   # agent 向：源码地图 + EP 链接关系 + wheel 适配点
```

## 快速用法 / 要点

- 查「ORT 在 DCU 上用哪个 EP、链接哪个库」→ ROCm EP ↔ libgalaxyhip ↔ MIGraphX 的链路图。
- 源码级问题（为什么某 op 走了别的 EP）看 SKILL.md 落点分析；性能问题转 `dcu-ort-perf`。

> 详细内容见 `SKILL.md`。