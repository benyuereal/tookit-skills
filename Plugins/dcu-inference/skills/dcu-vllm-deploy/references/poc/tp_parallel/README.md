# Tensor Parallel 最小化 POC —— Column/Row Parallel Linear + all-reduce

一个 ~200 行的纯 Python 脚本，把 vLLM 的 Tensor Parallel（TP）核心浓缩出来，
让你**跑起来、看见**权重是怎么切的、每个 rank 算出什么、all-reduce 在哪一步发生。

只依赖 Python 标准库，CPU 直接跑，不需要 GPU/DCU：

```bash
python3 mini_tp_parallel.py
```

对应真实 vLLM 源码：

| POC 里的东西 | 真实 vLLM 源码位置 |
|-------------|-------------------|
| `ColumnParallelLinear`（按输出维切 W，不通信） | `vllm/model_executor/layers/linear.py` → `class ColumnParallelLinear`（`forward` 里 `gather_output=False` 时不通信） |
| `RowParallelLinear`（按输入维切 W，输出是部分和） | `vllm/model_executor/layers/linear.py` → `class RowParallelLinear`（`forward` 里 `reduce_results=True` 时调 all-reduce） |
| `all_reduce`（部分和求和） | `vllm/distributed/communication_op.py` → `tensor_model_parallel_all_reduce`（`get_tp_group().all_reduce`） |
| gate/up 交错切分 | `vllm/model_executor/layers/linear.py` → `class MergedColumnParallelLinear`（每个 rank 从 gate 和 up 各拿 `inter/TP` 列） |

## 它演示了什么

| 机制 | 在 trace 里怎么看 |
|------|------------------|
| **Column 并行切输出维** | `qkv (8x24) 按输出维切 -> W0(8x12) W1(8x12)`，每卡只存 1/TP 的输出列 |
| **Row 并行切输入维** | `o (24x8) 按输入维切 -> W0(12x8) W1(12x8)`，每卡只存 1/TP 的输入行 |
| **局部输出** | 每个 rank 打印自己的 `qkv_shard` / `gu_shard` / `h_shard` |
| **部分和 → all-reduce** | `o_partial`（两个 rank 各一个）→ `[all-reduce] 部分和求和` → 完整结果 |
| **哪一步要通信** | 每行末尾标 `(不通信)` 或 `[all-reduce #N]` |
| **正确性** | 最后 `assert` 单卡完整计算 == 多卡 TP 计算（逐元素，容差 1e-5） |
| **通信次数** | 一个 transformer 层 TP=2 需要 2 次 all-reduce（O 投影 + down 投影） |

## 怎么读 trace

```
===== 权重切分（Layer 0）=====
  qkv      (8x24) 按输出维切 -> W0(8x12)  W1(8x12)
  o        (24x8) 按输入维切 -> W0(12x8)  W1(12x8)
  gate_up  (8x32) 按输出维切 -> W0(8x16)  W1(8x16)
  down     (16x8) 按输入维切 -> W0(8x8)   W1(8x8)
```

- **Column 并行**（qkv、gate_up）：W 是 `(in, out)`，沿**输出维**（列）切。
  rank r 拿 `W[:, r*out/TP : (r+1)*out/TP]`。输入 X 在所有 rank 上**复制**（不切），
  所以每卡算出 `X @ W_r`，是完整输出的一个**列块**（局部结果）。
- **Row 并行**（o、down）：W 是 `(in, out)`，沿**输入维**（行）切。
  rank r 拿 `W[r*in/TP : (r+1)*in/TP, :]`。输入必须是**切好的 shard**，
  每卡算出 `X_r @ W_r`，是完整输出的一个**部分和**（partial sum）。

```
  [QKV Column] W(8x24) 按输出维切 -> W0(8x12)  W1(8x12)   (不通信)
    rank0 qkv_shard (1x12): [...]
    rank1 qkv_shard (1x12): [...]
  [Attention ] 各 rank 本地算自己的 head   (不通信)
  [O Row     ] W(24x8) 按输入维切 -> W0(12x8)  W1(12x8)
    rank0 o_partial (1x8): [...]
    rank1 o_partial (1x8): [...]
  [all-reduce #1] 部分和求和 -> (1x8): [...]
```

- `qkv_shard`：Column 并行的局部结果，两个 rank 的 shard **横向拼起来**才是完整 qkv。
- `o_partial`：Row 并行的部分和，两个 rank 的部分和**逐元素相加**才是完整 o。
- `[all-reduce #1]`：这就是 `tensor_model_parallel_all_reduce`，把各 rank 的部分和求和，
  结果在每个 rank 上都有一份（all-reduce 的 "all"）。

## 为什么 MLP 是 Column + Row 配对

这是 TP 最核心的设计，POC 里 gate/up 用 Column、down 用 Row：

```
        Column 并行（切输出维）              Row 并行（切输入维）
X ──> [gate/up] ──> h_shard (每卡 1/TP) ──> [down] ──> d_partial ──> all-reduce ──> d
        不通信          每卡算局部结果          不通信        部分和        求和
```

1. **Column 切输出维**：`h = silu(X@G) * (X@U)`。G、U 按输出维切，每卡算出
   `h` 的 1/TP 列（`h_shard`）。此时**不需要通信**——每卡的输入 X 是完整的（复制的），
   输出只是列块，各算各的。
2. **Row 切输入维**：`d = h @ D`。D 按输入维切，每卡拿 `h_shard`（正好是 h 的 1/TP 列）
   乘 `D_r`，得到 `d` 的一个**部分和** `d_partial`。
3. **all-reduce 求和**：`d = Σ_r d_partial`。因为 `h @ D = h0 @ D0 + h1 @ D1`
   （D 按行堆叠），部分和相加正好等于完整结果。

**关键**：Column 的输出（列块）正好是 Row 的输入（切好的 shard），
所以中间**不需要任何通信**，只在 Row 的末尾 all-reduce 一次。
这就是为什么 attention 是 `QKV(Column) → O(Row)`、MLP 是 `gate/up(Column) → down(Row)`
——**Column 和 Row 总是配对出现**，一对里只通信一次。

## 通信统计

```
一个 transformer 层 (TP=2) 需要 2 次 all-reduce：attention 的 O 投影 + MLP 的 down 投影
2 层共 4 次 all-reduce
```

- 每层固定 **2 次** all-reduce（O 投影 + down 投影），与 TP 度无关。
- **trade-off**：TP 越大，每层 all-reduce 次数不变（仍是 2 次），
  但每次 all-reduce 参与的 rank 更多、数据量更大、延迟更高；
  好处是每卡权重/激活显存按 TP 线性下降。
  所以 TP 一般限制在单机内（如 8 卡），跨机扩展用 PP（流水线）/ EP（专家并行）。

## 和真实 vLLM 的差异（简化点）

| 简化 | 真实 vLLM |
|------|----------|
| all-reduce 用 Python 列表逐元素求和 | NCCL/RCCL 集合通信（`get_tp_group().all_reduce`），GPU 上走 NVLink/PCIe |
| 没有 TP group 概念，直接 `range(TP)` | 有 `parallel_state` 的 TP group，rank 由 `get_tensor_model_parallel_rank()` 决定 |
| attention 直接透传 qkv shard | 真实 attention 在这里做 per-head 的 QK^T、softmax、V 加权（POC 里省略） |
| QKV 用普通 Column 切分 | 真实用 `QKVParallelLinear`，按 head 切，GQA/MQA 时 KV head 会复制 |
| gate/up 手动交错拼权重 | 真实用 `MergedColumnParallelLinear`，`output_sizes` 自动处理 gate/up 切分 |
| 权重随机生成 | 真实从 checkpoint 加载，`weight_loader` 按 `tp_rank` narrow 出本卡分片 |
| 单 token（batch=1） | 真实 batch 维是 (num_tokens, hidden)，TP 只切 hidden 维 |

想扩展的话，可以加：GQA 的 KV head 复制、`gather_output=True` 的 all-gather、
bias 的处理（Row 并行里 bias 只在 rank 0 加一次，避免重复）、以及 batch 维。
