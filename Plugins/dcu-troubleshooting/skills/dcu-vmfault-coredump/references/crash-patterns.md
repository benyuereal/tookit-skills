# 已知 VMFault 崩溃模式

实战积累的 OOB 崩溃模式。用于快速对号：症状 → 判定 → 归属。

> **方法通用性**：下面这些是"PyTorch index_select/embed"这个族的案例。崩溃在 Triton/CK/自定义 HIP kernel 时，
> 症状→判定→归属的逻辑不变（越界值、OOB 地址、dst vs src 逐字节验证、队列归属），只是 args 解码改用
> `--raw` 盲解码 + `--gpucore` 段交叉引用（见 `kernel-abi.md`）。
>
> **所需材料**：带什么材料、能定位到哪一层，见 `SKILL.md`「输入材料规范」——崩溃日志必带（否则无从选块），
> gpucore 决定能否读到 fault-time 越界值，env/配置/复现决定能否归属到 op 与根因。

## 模式 1：负索引直达 embed（`-1` token id）

**症状**：faulting kernel = `indexSelectSmallIndex<c10::Half, long, unsigned int, 2, 2, -2>`，
src = `(vocab_size, hidden)` fp16，indices = int64[N]，fault-time index 含 **-1**。

**机制**：`embed_tokens`（= `weight.index_select(0, ids)`）对 -1 越界。PyTorch fork 的
`indexSelectSmallIndex` 内核**只有上界断言**：

```cpp
IndexType srcIndex = indices.data[get(dstIndex, indices)];
CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);   // 只有上界
// srcOffset = srcIndex * src.strides[srcSelectDim];  // -1 -> 负 offset -> 读 src 前 4KB 之前的页
```

-1 以带符号小值通过上界检查，随后算出负 offset，读缓冲区之前的内存 → VMFault。

**归属（真实案例）**：faulting embed 之后紧排 `_causal_conv1d_update_kernel`（GDN decode conv）、
`fused_sigmoid_gating_delta_rule_update_kernel`（GDN decode attn）→ TARGET 模型 decode forward 入口，
非 MTP drafter（drafter 全 full_attention 无 GDN 层）。

**根因（vLLM MTP spec-decode）**：scheduler `update_draft_token_ids_in_output`
（scheduler.py:2352-2356）把 draft 不足的槽位补 `[-1]`：

```python
spec_token_ids.extend([-1] * num_invalid_tokens)
```

`num_draft_tokens[req_idx] = len(draft_token_ids)`（gpu_model_runner.py:1855）把 -1 补齐位计为真实 draft；
`_prepare_input_ids` 无过滤散点进 input batch；`_calc_spec_decode_metadata` 原样带出；
`num_invalid_spec_tokens` 只在统计扣除，不阻止 embed。→ 修复 = 写入点 sanitize（-1→0），
见 vllm-hcu commit `4d47ad2`（gated on `VLLM_HCU_USE_PD_SPLIT`）。

**最小复现**（3 行）：
```python
w = torch.randn(320, 16, dtype=torch.float16, device="cuda")
torch.nn.functional.embedding(
    torch.tensor([1, 2, 3, -1], dtype=torch.int64, device="cuda"), w)  # 崩溃
# control [0,1,2,3] 正常
```

## 模式 2：无符号回绕的大地址（OOB 读/写通用症状）

**症状**：faulting VA 是一个"莫名其妙的大地址"（如 `0xFFFFFFFF...`），TRAPSTS=0x80000000。

**机制**：索引值作为无符号 64 位回绕（如 -1 → 0xFFFFFFFFFFFFFFFF），乘以 stride 后远超缓冲区，
读/写到高地址页。与模式 1 同源（-1 哨兵/脏索引）。

**判定**：gpucore 读 index 张量，若出现 `0xFFFFFFFFFFFFFFFF`（= -1）即此模式。

## 一般排查原则

- **faultinfo 不可用 ≠ 无法定位**：faultinfo 只给 faulting VA + trap 类型；真正的上下文在
  队列分析（kernel 名 + args）+ gpucore 内存（fault-time 值）+ 配置（缓冲区身份）。
- **先定 op，再定张量，再定值**：kernel 名 → 参数 ABI 解指针/形状 → gpucore 读真实值。
- **dst vs src 逐字节匹配**是"fault-time 数据"的最强证据，排除崩溃后内存复用。
- **字节数×形状 = 缓冲区身份**：`prod(shape)×dtype_size` 与模型配置（vocab/hidden 等）对号，
  且 gpucore PT_LOAD 段 memsz 精确匹配时几乎可锤死。
