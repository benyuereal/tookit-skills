# FlashMLA 深度优化 · 执行计划

> 建立于 2026-11-25。依据业主定下的顺序。
> **迭代顺序：flash MLA → DeepGEMM → flash attention**

## 仓库

| 仓库 | 路径 | 作用 |
|---|---|---|
| `HYGON-AI/FlashMLA-das` | `/workspace/FlashMLA-das` | 当前主战场 |
| `HYGON-AI/deepgemm-das` | `/workspace/deepgemm-das` | 第二阶段 |
| `HYGON-AI/flash-attention-das` | 未克隆 | 第三阶段 |

## 已确认的基线（2026-11-25，卡空闲时测）

| 形状 | flash_mla bf16 | MAD |
|---|---|---|
| decode-s128 | 0.3429 ms | 0.75% |
| decode-m1 | 0.3600 ms | 2.49% |
| decode-m32 | 0.3592 ms | 0.40% |

## 当前阻塞：D14 · fp8 d=512 VMFault

### 什么是「d=512」

- DeepSeek DSA / V4.1：`d_qk = 576`（512 latent + 64 rope tail）
- **GLM-5.3 flash：`d_qk = d_v = 512`（无 rope 尾）**

上游 FlashMLA-das 只实例化了 576。**GLM-5.3 用不了 fp8 路径。**

### 已修的

1. `static_assert(Headdim == 576)` → 放开到 `576 || 512`
2. 函数内 5 处 `<576, ...>` → `<Headdim, ...>`
3. 加 512 的模板实例化（`.cu` 与 `.h`）
4. **kernel 名字硬编码 `ILi576E`** → 按 `Kernel_traits::kHeadDim` 动态拼
5. 查表失败没检查 → 加空句柄检查

### 未解决

修完 1-5 后 512 仍 VMFault，但**错误形态变了**：
从「垃圾 grid.z」变成「KERNEL VMFault」，说明句柄对了，是 kernel 内部越界。

### 排查方向（按可能性排）

1. **smem 超 64KB**（当前诊断就查这个）
   - 证据：`group_segment_size: 65536` 已用满
   - 若超了：减 `kBlockN` 64 → 32
2. 某处 `kBlockN` / `kBlockM` 的 smem 布局与 512 不兼容
3. `d_v == d_qk == 512` 时，某个「k 和 v 分开处理」的假设塌了
   - 576 时 `kHeadDimV(512) < kHeadDim(576)` 严格成立
   - 512 时两者相等，任何 `if (d_v < d_qk)` 分支都不会进

## 纪律（针对这次任务）

1. **每次测量前跑 `gpu_guard.py`** —— 显存占用不是判据，使用率 + 分配探针才是
2. **576 始终作对照组** —— 任何「512 有问题」的结论必须配 576 正常
3. **不猜** —— 改动前先看源码，不试参数
4. **每修一个 bug 记进 DEBT.md** —— 这次的坑（硬编码 kernel name）
   如果早有人记过，能省两小时
