# dcu-mooncake-source — Mooncake Transfer Engine DCU 适配源码索引

Mooncake Transfer Engine（KV cache 跨节点传输）在海光 DCU 上的源码结构与适配点：libtransfer_engine 链接 DTK、HIP copy kernel、EP（expert parallel）buffer 适配。基于 wheel 解包 + ldd 实测。

## 目录

```
dcu-mooncake-source/
└── SKILL.md   # agent 向：源码地图 + DTK 链接 + HIP copy kernel 落点
```

## 快速用法 / 要点

- 查「Mooncake 在 DCU 上跨节点 KV 传输怎么跑通」→ libtransfer_engine 的 DTK 链接与 HIP kernel。
- 与 PD 分离部署配合使用：传输层问题看这里，部署配比看 `dcu-llm-benchmark`，SGLang 侧看 `dcu-sglang-source`。

> 详细内容见 `SKILL.md`。