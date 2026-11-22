# 案例：MQA 的 FP8 不该走 persistent

## 这一例演示什么

`op-source-patch` 六步的完整闭环：定位 → 取源 → 编译 → 改(patch) →
回归 → 留痕。**重点在「回归」那一步做了什么**。

## 问题

`lightop/attention/mqa.py` 的 `_persistent_supported` 对 gfx938 返回：

```python
return ((q.dtype == torch.int8 and fused_kv_cache.dtype == torch.int8)
        or (q.dtype == torch.float8_e4m3fn and fused_kv_cache.dtype == torch.uint8))
```

即 **FP8 也被允许走 persistent**。但实测表明：

1. FP8 + persistent 比 default 后端**慢**——persistent 的跨 iteration 驻留
   假设建立在 BF16/int8 的寄存器与 LDS 布局上，FP8 的 scale 广播打破了它
2. 在强制的小 page 分组形状上，persistent 路径直接 **RuntimeError**

判据见 `op-roofline/references/pipeline.md` 的 Q5 与 Q6。

## 改法

`patches/0001-disable-persistent-for-fp8.patch` —— 把 gfx938 的白名单
收窄为「仅 int8」。

## 回归做了什么（这一例的关键）

**没有直接上机测性能**，而是先做不需要 GPU 的逻辑回归：

```bash
./reproduce.sh logic
# patch 前 gfx938 persistent 允许: {'int8': True, 'fp8': True}
# patch 后 gfx938 persistent 允许: {'int8': True, 'fp8': False}
# PASS
```

这一步的价值：patch 改的是「哪些 dtype 允许」这个**纯逻辑判断**，
不需要显卡就能验证。把不需要 GPU 的部分先验掉，上机时只需验性能，
排查面小很多。

另加 `./reproduce.sh apply` 验证 patch 能干净套用（无 fuzz）。

## 结果

见 `results/result.json`。**明确写了「not_measured」**：
端到端性能未测，需要一张空闲卡。本案例只验证逻辑正确性与 patch 可套用。

## 库与版本

- `lightop` 0.6.0+dtk2604.torch2110
- 源码：`/workspace/lightop/lzh_glm_new`（本地已解包，**未重拉**——重拉一次是 200MB+）

## 复现

```bash
./reproduce.sh all      # logic + apply
```
