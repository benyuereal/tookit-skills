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

## 实现横评

> **优劣与场景写在家族层，不重复写进每张卡**——同一个实现会被多张卡引用，
> 写进卡片就会有 N 份会漂移的真相。卡片只留 `best_for` / `not_for` 的简短版。

| 实现 | 层级 | 是什么 | 适合什么 | 不适合什么 | 证据 |
|---|---|---|---|---|---|
| `lightop-4stage` | kernel | align→grouped→act→reduce 四段融合，包内无持久化调度 | decode 小 batch（16-row pooled cache）；shared 与 routed 同容器 | FP8 + persistent 后端（会 RuntimeError）；需要跨 gfx 单一二进制 | `libraries.md §3`；本仓 FP8 消融记录 |
| `lightop-asm` | kernel | mode≥1000 的汇编路径 | 特定档位 | **混合专家 batch**（串读其他 token/专家的数据，违反 `batch_independence`） | `libraries.md §3`；`op-moe-w8a8/ledger.md` |
| `aiter-das` | kernel | AITER 移植，`get_aiter_moe_config` 按 (量化×dtype) 选 moe_c/asm/triton/ck | 标准接入兜底；需要统一 Python API | 需要单一路径可预测性能（查表未命中会静默降级） | `libraries.md §4` |
| `moe_c` | kernel | 按「精度模式 × gfx」组织的调优表矩阵 | AITER 选中 MOE_C 且慢时直接改表 | AITER 根本没选它时（改表无效，先强制 `spec_sol_type` 验证） | `libraries.md §5` |
| `MegaMoE + CP8/DP8` | **engine** | 8 卡并行策略 + 图捕获 + DSPARK，不是 kernel | 整模型部署的并行策略选型 | **不属算子层**：结论受并行策略、显存水位、预热共同影响，**不许进任何算子卡的 last_verdict** | 方法见 `docs/ENGINE.md`；数字留 kb |

**读这张表的方式**：先按量化格式与 page 布局缩小到 1-2 行，再看「不适合」列排除。
四者不在同一层——前四个是 kernel，最后一个是 engine，**同表比会得出错误结论**。
