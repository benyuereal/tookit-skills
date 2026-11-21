# InputBatch 固定 buffer 机制最小化 POC —— "固定停车位"

一个 ~200 行的纯 Python 实现，让你**跑起来、看见** vLLM V1 `InputBatch`
为什么用"固定 buffer"（用户俗称"固定停车位"）：按 `max_num_seqs` 预分配
buffer，batch 组成不变时复用只更新变化部分，batch 变化才重建。

只依赖 Python 标准库，CPU 直接跑，不需要 GPU/DCU：

```bash
python3 mini_fixed_buffer.py
```

## 它演示了什么

| 机制 | 在 trace 里怎么看 |
|------|------------------|
| **预分配 buffer** | `FixedInputBatch` 按 `max_num_seqs=4` 预分配 `input_ids`/`positions`/`slot_mapping`/`block_table`/`req_to_token`，每个请求占一个"停车位"（`req_index`） |
| **batch 不变 → 复用** | `复用 buffer，只更新 2 slot（每请求 1 新 token）`：decode 步只写每个请求新增 1 个 token 的几格 |
| **batch 变化 → 重建** | `batch 变化，重建 buffer（3 请求）`：请求加入/离开时重新分配停车位、重填 block_table/req_to_token |
| **量化对比** | 100 步里：朴素重建 100 次、写 11664 slot；固定 buffer 只重建 15 次、写 3006 slot（少写 ~3.9x） |

## 怎么读 trace

```
step   1 [不变] batch=1
    朴素: 重建 4 个张量（1 请求 x 全部 token）
    固定: 复用 buffer，只更新 1 slot（每请求 1 新 token）
step  10 [加入] batch=2
    朴素: 重建 4 个张量（2 请求 x 全部 token）
    固定: batch 变化，重建 buffer（2 请求）
```

- `[不变]`：所有请求 decode，batch 组成没变 → 固定 buffer 复用，只写变化 slot
- `[加入]`/`[离开]`：batch 组成变了 → 固定 buffer 重建（对应真实 vLLM 的 `condense`/重排）
- 朴素实现每步都重建整个 batch 张量（把每个请求的所有 token 重写一遍）

## 汇总

```
batch 组成变化次数: 加入 9 + 离开 6 = 15 次
batch 不变步数:     85 次

实现             重建次数     写入 slot 总数
朴素              100          11664
固定buffer         15           3006
```

decode 占绝大多数步（85/100），固定 buffer 在这些步只更新 1 个新 token 的几格，
避免每步重建整批张量带来的 CPU 开销和 H2D 拷贝。这就是 InputBatch 用固定
buffer 的原因。

## 组件名对齐真实 vLLM

| POC 组件 | 真实 vLLM |
|----------|----------|
| `FixedInputBatch` | `vllm/v1/worker/gpu_input_batch.py` 的 `InputBatch` |
| `input_ids` / `positions` | `InputBatch.token_ids_cpu` / `gpu_model_runner.positions` buffer |
| `slot_mapping` | `gpu_model_runner` 的 `slot_mapping`（KV 写入的物理槽位） |
| `block_table` | `InputBatch.block_table`（`MultiGroupBlockTable`） |
| `req_to_token` | `gpu_model_runner.req_to_token`（逻辑位置 → 物理槽） |
| `req_id_to_index`（停车位） | `InputBatch.req_id_to_index` |
| batch 变化重建 | `InputBatch.condense()` / `add_request` / `remove_request` |

## 和真实 vLLM 的差异（简化点）

| 简化 | 真实 vLLM |
|------|----------|
| buffer 用 Python 列表模拟 | 真实是 `torch.Tensor`（CPU pinned + GPU 双份），`pin_memory` 加速 H2D |
| 重建 = 重填 block_table/req_to_token | 真实 `condense()` 把活跃请求前移、压缩空洞，更精细 |
| 加入/离开用简单规则模拟 | 真实由调度器 `ScheduleResult` 决定，含 prefill/decode 混合 |
| 只统计"写入 slot 数" | 真实开销还包括 H2D 拷贝、kernel launch、采样元数据构造 |
| 单 batch | 真实还有 ubatch（`gpu_ubatch_wrapper`）、spec decode 的额外 buffer |

对应真实源码：`vllm/v1/worker/gpu_input_batch.py`（`InputBatch.__init__` 预分配
buffer、`add_request`/`remove_request`/`condense`）、`vllm/v1/worker/gpu_model_runner.py`
（`positions`/`slot_mapping`/`req_to_token` buffer 的 `_make_buffer`）。

想扩展的话，可以加：condense 压缩空洞、prefill+decode 混合步、H2D 拷贝开销建模。
