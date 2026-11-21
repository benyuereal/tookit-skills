---
name: dcu-triton-kernels
description: "Use when writing/optimizing Triton kernels on DCU (Hygon BW1000), or migrating PyTorch operators to Triton — covers correctness-first workflow, five optimization patterns, ten task solutions, and DCU-specific pitfalls."
version: "1.0.0"
author: <internal-user>
license: MIT
---

# DCU Triton Kernel 优化（dcu-triton-kernels）

> 适用：HIP 后端（BW1000/C500）、NPU 后端（Ascend 910B）。Triton 原生 DSL，不涉及 csrc 重编译。
> 区别于 `dcu-kernel-dev` 路径 B（TileLang DSL）：本文档聚焦 Triton 原生写法。

---

## 一、什么时候值得写 Triton

| 机会类型 | 场景 | 收益来源 |
|---|---|---|
| **融合** | 相邻多个逐元素操作 | 减少 kernel launch 和中间张量读写 |
| **专用化** | 固定形状 / 消除动态分支 | 循环展开 + 消除分支预测失败 |
| **数学变换** | 单调函数 / 公共因子抵消 | 跳过中间计算步骤 |

**先判断瓶颈类别：**
- 小张量 + 多步骤 → **Launch-bound**，融合优先
- 逐元素 + 大张量 → **Memory-bound**，一次 load 多次复用
- 大量乘加 / GEMM → **Compute-bound**，合理 tile + 低精度累加

无法说明"减少什么"的修改不值得做。

---

## 二、正确性优先的闭环

```
确认接口和形状
  → 跑通 reference（torch 版本能跑通）
  → 写中间张量表（每个步骤的输出形状和代价）
  → 测 reference 基线
  → 提出可验证的瓶颈假设
  → 写最小 Triton 版本
  → 先校验正确性（官方评测器，atol=1e-2, rtol=1e-2）
  → 再测性能
  → 扫参数（BLOCK / num_warps）
  → 回归全部样例
```

---

## 三、如何跑官方评测

### 必备结构（reference.py / solution.py）

```python
# reference.py — PyTorch 基线
class Model(nn.Module):
    def __init__(self, ...): super().__init__(...)
    def forward(self, ...): ...

def get_init_inputs(): return [...]
def get_inputs(): return [...]

# solution.py — Triton 优化（类名必须是 ModelNew）
class ModelNew(nn.Module):
    def __init__(self, ...): super().__init__(...)
    def forward(self, ...): ...

def get_init_inputs(): return [...]
def get_inputs(): return [...]
```

### 官方评测命令

```bash
# BW1000 / C500
DEVICE_ID=0 python3 evaluator/auto_bench.py \
  --v0_file reference.py \
  --v1_file solution.py \
  --warmup 200 --repeat 500

# Ascend 910B
CANN_DEVICE_ID=0 python3 evaluator/auto_bench.py \
  --v0_file reference.py \
  --v1_file solution.py \
  --warmup 200 --repeat 500
```

**输出格式：**
```
PASS accuracy; v0=0.275516 ms, v1=0.088945 ms, speedup=3.098x
```

评测器位置：`references/evaluator/auto_bench.py`（DCU 适配版，支持 gcu/npu/mlu 自动检测）。

### 快速本地诊断（不等官方评测器）

```python
import torch
import triton
import triton.testing

def benchmark(fn, warmup=100, repeat=300):
    # 预热
    for _ in range(warmup): fn()
    torch.cuda.synchronize()  # gcu / npu 对应同步接口
    # 计时
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)
```

---

## 四、如何监控性能

### hy-smi（设备状态）

```bash
# 显存使用和利用率
hy-smi --showmeminfo vram --showuse

# 实时监控（每秒刷新）
watch -n 1 hy-smi

# 特定设备
hy-smi -d 0
```

### rocprof（kernel 级别 Timeline）

```bash
# 单次运行抓取 kernel 级别 timeline
rocprof --hip-trace on \
  python3 evaluator/auto_bench.py \
    --v0_file reference.py \
    --v1_file solution.py \
    --warmup 10 --repeat 50

# 结果输出在 ./output/
# 查看：chrome://tracing 加载 output/*.json
```

### 诊断顺序

```
1. hy-smi 确认设备空闲、无其他进程占卡
2. 官方评测器确认正确性 PASS
3. 对比 PyTorch vs Triton 中位数时间，计算加速比
4. rocprof 确认时间花在哪个 kernel（是 launch 多还是某个 kernel 慢）
5. 瓶颈假设 → 针对性优化 → 回归正确性
```

---

## 五、五大核心优化模式

### 5.1 融合：缩短数据"片外旅程"

融合一层中间张量节省：
```
N × 4B 写 + N × 4B 读 = 8N 字节
```
**判断标准：** 中间张量是否被多次读写？是 → 融合。

### 5.2 数学等价变换

| 变换 | 场景 | 示例 |
|---|---|---|
| 单调性跳过 | Top-k / argmax | `argmax(softmax(x)) == argmax(x)` |
| 公共因子抵消 | 二次归一化 | `softmax(x)/Σsoftmax == exp(x)/Σexp` |
| 运算后移 | ReLU+log1p 后置 | 移到 max pooling 之后 |

### 5.3 在线 Softmax

列分块，保持片上状态：
```
m_new = max(m_old, m_blk)
p_blk = exp(score_blk - m_new)
l_new = α*l_old + Σp_blk
acc_new = α*acc_old + p_blk @ V_blk
out = acc / l
```

### 5.4 program 粒度选择

| 粒度 | 适用场景 | 任务 |
|---|---|---|
| 每行一个 program | 一行元素归约或复用 | Task01, 07, 09 |
| 每矩阵一个 program | 小矩阵多轮迭代 | Task08 |
| 每 query tile 一个 program | K/V 流式扫描共享 | Task03, 06 |
| 每 sample 一个 program | 独立随机变换 | Task09 |

**经验：** 粒度太小重复加载增加 launch；粒度太大寄存器压力降低 occupancy。

### 5.5 合并访存

让连续线程访问连续地址：
```python
hidden = block * BLOCK + tl.arange(0, BLOCK)
ptrs = base + row * hidden_size + hidden  # 最后一维连续
```

跨距访问使一次 wave 的地址分散，显存事务利用率下降。

---

## 六、十题速查

| 任务 | 优化模式 | 加速比 | 关键代码 |
|---|---|---|---|
| **Task01** GroupedTopk | 单调性跳过 Softmax + 公共因子抵消 | 3.10× | `tl.argmax` 逐轮选 Top-k |
| **Task02** FusedMoE | 三阶段 GEMM 分块融合 | 17.92× | `tl.dot` 分块 + 专家选择 |
| **Task03** FlexAttention | 在线 Softmax | 1.66× | `tl.exp(score - m_new)` 稳定指数 |
| **Task04** SPLADESparsePooler | GELU+LayerNorm 融合 + 数学变换 | 2.93× | `exp(erf(x/sqrt(2)))` GELU 融合 |
| **Task05** RotaryEmbedding | 广播三角函数链融合 | 2.60× | `x1*cos - x2*sin` 单 kernel |
| **Task06** MMEncoderAttention | 流式 K/V + 在线 Softmax | 1.66× | query tile 共享 K/V 流 |
| **Task07** mhc_post | 数据复用（一次 load 4次复用） | 11.51× | `(batch,token)` 合并为 row |
| **Task08** hc_split_sinkhorn | 小矩阵片上驻留 20 轮 | 13.57× | `tl.constexpr(20)` 静态循环 |
| **Task09** CRA | 语义一致性随机变换融合 | 5.55× | seed 确定随机数，同样本一致 |
| **Task10** head_compute_mix_bwd | 反向多路归约 + atomic 汇总 | 1.85× | `tl.atomic_add` 跨 program 归约 |

**十题推荐学习顺序：**
```
Task05 → Task08 → Task07 → Task01 → Task10
       → Task02/04 → Task03/06 → Task09
```

题解详情：`references/tasks/0X_*.md`

---

## 七、DCU 陷阱清单

| 陷阱 | 后果 | 修复 |
|---|---|---|
| `alloc_shared` 用于精度敏感状态 | 结果随机、每次不同 | 用 `alloc_local` |
| warp shuffle 未指定 `warp_size=64` | DCU 归约结果错误 | 显式传 `warp_size=64` |
| `num_warps` 过大 | 小任务反而更慢 | 先试 `1`，再扫到最优 |
| 融合后块张量过大 | VGPR spill，occupancy 下降 | 拆成多 kernel |
| 配置专用化未检查 | 静默错误 | `__init__` 中 `assert` 不支持配置 |
| atomic 操作用于非交换操作 | 结果错误 | 确保操作满足交换律 |
| eps 顺序错误 | 数值结果漂移 | 与参考实现的括号顺序保持一致 |
| 输出 dtype 不匹配 | 官方评测器比较失败 | store 时显式 `.to(output_dtype)` |

---

## 八、调优记录模板

每次调优后记录（下一题遇到相似形状时这些失败数据比最终答案更有价值）：

```markdown
### 假设
reference 慢在 20 轮小 kernel launch，不在算术量。

### 改动
一个 program 保存 4×4 矩阵完成全部迭代。

### 正确性
max_abs_diff = ...，PASS。

### 性能（BLOCK/num_warps = x）
中位数 ... ms；加速比 ...x。

### 结论
kernel 数下降；增大 num_warps 无收益 → 块内工作量太小。
```

---

## 九、跨平台迁移

| 平台 | 通用性 | 特殊处理 |
|---|---|---|
| BW1000 | 直接复用 | — |
| C500 | 大部分可复用 | 矩阵 tile 仍可调优；设 NCHW 布局 |
| Ascend 910B | 部分可迁移 | query tile 调为 32×32；watchdog 需拆分大 kernel |

详见：`references/platforms/migration_guide.md`