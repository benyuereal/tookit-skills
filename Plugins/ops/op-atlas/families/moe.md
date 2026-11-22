# moe

token 按专家重排后分组计算。**能否快取决于 (量化格式 × gfx) 的查表命中**。

## 本族特有的判据

1. **token 组织方式决定了权重布局，布局决定了能不能复用**。
   三条路径（masked / 物理连续 / low-latency）的权重格式互不通用：
   HIPC / W6 packed / marlin 三种 layout 混用不是慢，是**静默算错**。
2. **「慢」先查表，别先改 kernel**。AITER 的 `get_aiter_moe_config` 按
   (量化类型 × dtype) 选后端，选不中就整体降级。查表命中率是第一个要看的数。
3. **8 档精度的覆盖面不同**：WFP4A8 / WFP4A16 是冷门档，覆盖面明显窄于主流档，
   没命中就走 fallback（正确但慢）。
4. **融合不一定赢**：共享专家与 routed 专家的占比决定了融合的收益边界。

## 本族攻击面（跨族通用的见 surfaces.yaml）

- `table-miss`：该 (量化, gfx) 组合的调优表有没有这一档（**未命中往往是最大收益**）
- `fusion-boundary`：融合是否把可并行的 shared/routed 串成依赖链
- `layout-pack`：每步是否多付一次权重 shuffle（漏了是静默错误）

## 已知的结构性陷阱

| 陷阱 | 机制 | 出处 |
|---|---|---|
| 在 AITER 上层反复调参 | 后端都没选中时，参数根本不生效 | libraries.md §4 |
| 漏掉权重 shuffle | `aiter_moe_shfl_weight` 未调用是**静默错误**不是报错 | libraries.md §4 |
| 用稠密 GEMM 循环模拟 | 实测是 graph 化稠密路径的 2.9x | libraries.md §一 |

## 库选择

| 场景 | 首选 |
|---|---|
| 通用 MoE，想要统一 API + 回退 | `libraries.md §4`（aiter-das） |
| 某量化组合没解 | `libraries.md §5`（moe_c，直接改表） |
| grouped GEMM 精细控制 | `libraries.md §6`（DeepGEMM） |
| 量化融合链 | `libraries.md §3`（LightOp） |
