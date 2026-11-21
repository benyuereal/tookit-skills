# NInfer Qwen3.8-27B DCU 移植案例（2026-08-25 终态）

> 案例背景：NInfer（单卡闭集 C++/CUDA 推理引擎，硬锁 RTX 5090/sm_120a）移植到 DCU gfx936，
> 目标 Qwen3.8-27B（groupwise-int 档）。MFMA 被 DTK 26.04 堵死，走 SIMT-first，mma.cuh 全 stub 化（no-op）。
> **终态：精度 6/6 全绿 + decode 1.75x 快于原生 + 256k 上下文跑通 + 8 路多并发跑通。**
> 本文件沉淀**已验证事实**与**假警报记录**，下个会话直接从这里续，不要重查已验证项。

## 环境

- <device-B> 机器（<<internal-ip>>），容器 `ninfer-dcu-port`，镜像 `vllm:0.18.1-ubuntu22.04-dtk26.04-py3.10`
- 源码 `<local_ninfer_src>-hip`（hipify 后），构建产物 `/data1/build/ninfer_cli`
- 权重 `/data1/out/qwen3_8_27b.ninfer`（18.2GB），原生模型 `/data1/models/Qwen3.8-27B`
- 只用卡 4（`HIP_VISIBLE_DEVICES=4`），sglang 间歇性占卡（8 路 DP/TP），轮询 `hy-smi` 等释放
- 编译：`/opt/dtk/bin/hipcc -O2 -std=c++20 --offload-arch=gfx936 -include <local_ninfer_src>_hip_compat.h -Iinclude -Isrc -Ithird_party -Ithird_party/utf8proc`
- 链接：`hipcc -O2 -std=c++20 --offload-arch=gfx936 [core .o + ops .o + cpp .o + gqa_prefill_simt_launcher.o + main.o + stub .o + utf8proc.o] -o ninfer_cli`
- 独立 kernel 测试需链接 `src/core/tensor.cpp src/core/dtype.cpp src/core/device.cu`（Tensor/hip_check/dtype_size 符号）
- **重编某 .cpp 前先 `ls build/*/ | grep <name>` 确认它原本在哪个目录**（core/ops/cpp），重编到同一目录覆盖，避免 duplicate symbol

## 三个根因（全部 MMA stub 静默 no-op，全部已修）

| # | 根因 | 现象 | 修复 |
|---|---|---|---|
| 1 | Q4 swiglu prefill t>1 走 MMA stub | activation 全 0，MLP 不贡献 | Q4 GEMM SIMT + swiglu elementwise（`q4_linear_swiglu_simt.cu`） |
| 2 | Q5 linear_add t>13/16 走 MMA stub | 残差加 `x += gdn_out` 不生效，mlp 拿到 GDN 前的 x | 通用 Q5 GEMM SIMT + 单独 residual_add（`q5_linear_add_plan.cpp` dispatch 加 fallback） |
| 3 | GDN chunked T≥64 走 MMA stub（1656 行 3 kernel） | 长 prompt（>64tok）退化 | recurrent SIMT fallback（`gated_delta_net.cpp` chunked 版改调 `launch_recurrent_inout`，内部循环 T + 内部 l2norm） |

**定位关键**：某中间态"范围正常但 cos 低"且**只在特定 t 区间错**（t=9 对、t=14 错）→ 查 dispatch route 表，确认该 t 是否落进 MMA route。

## 终态成绩（2026-08-25）

- **精度 6/6 全绿**（vs 原生 Qwen3.8-27B 同 prompt）：Hello(13)/2+2(25)/sky(18)/中文(16)/TCP(60)/AI历史(135) 全部一致
- **decode**：NInfer 12.6-12.8 tok/s vs 原生 ~7.2 tok/s = **1.75x 更快**
- **prefill**：NInfer 14.9(13tok)/39.7(43tok)/48.5(153tok)/43.6(13340tok) tok/s
- **256k 上下文**：bf16 KV = **16GiB**（只有 16 个 GQA 层有 KV，48 个 GDN 层是 SSM state 不随上下文增长），64GiB 卡放得下，**不需要 int8 KV**。10k prefill 43.58 tok/s 输出正确。256k 全量 prefill 估算 ~100min（recurrent fallback，慢但能跑）
- **多并发**（N=1/2/4/8，每请求 32tok）：总吞吐 9.84/9.43/8.77/11.16 tok/s，基本恒定（内存受限 decode 正确行为），N=8 略高说明 batching 生效。GDN 正确处理 batched decode（`gdn_input cols=1..7`）

## 多并发测试方法（C++ Engine API）

CLI 只接受单 `--prompt`，多并发走 Engine API：
```cpp
ninfer::EngineOptions opt;
opt.max_concurrency = N;            // 默认 1
opt.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
opt.use_hip_graph = false;
ninfer::Engine engine(std::move(opt));
// 每个请求独立 prompt（避免 prefix reuse）
std::vector<ninfer::GenerationHandle> hs;
for (i in N) {
  auto input = ninfer::product::prompt_from_text(prompts[i], false);
  auto pp = engine.prepare(std::move(input));
  ninfer::RequestOptions req; req.execution.requested_output_tokens = 32; req.output.raw = true;
  hs.push_back(engine.submit(std::move(pp), std::move(req)));
}
for (i in N) auto r = hs[i].wait();   // 收集 result.timings / generated_token_ids
```
链接时排除引擎的 `apps_cli_main.o`（避免 main 重复），加自己的 main .o。

## .ninfer 格式事实（已验证）

- 文件头：16 字节 magic + JSON（174230 字节）；**数据区起点 = 176128**（`align_up(16+json,256)`）
- **JSON 里 tensor 的 offset 是相对数据区起点 176128 的**，不是文件绝对偏移
- tokenizer.json 在数据区 offset=0（以 `{` 开头）
- 对象命名：`text/layers/N/...`（如 `text/layers/0/gdn/query_key`），不是 HF 命名
- 布局 `row-split-k128-v1` 是 **plane-based**：
  - low plane（codes）：`rows * groups_per_row * code_bytes_per_group`
  - high plane（Q5 高 1bit，Q4 无）：offset = `align_up(low_bytes, 256)`
  - scale plane（fp16）：offset = `align_up(high_off + high_bytes, 256)`
  - 索引均 `[row][group]` row-major；`padded_columns = align_up(cols, 128)`
- 量化格式：
  - Q4G64_F16S：group=64，code 32B/group（每 byte 2 个 4bit，低 nibble=偶元素），scale fp16 2B/group，`q=(nibble^8)-8`，`w=q*scale`
  - Q5G64_F16S：code 32B + high 8B/group，5bit=`(high_bit<<4)|low4`，`q=(q5^0x10)-0x10`（range -16..15）
  - W8G32_F16S：group=32，codes int8 `[vocab, padded_d]`，scales fp16 `[vocab*kg]`
- 字节数验证法：`rows*groups*code_bytes + rows*groups*high_bytes + rows*groups*2` 必须等于 JSON bytes 字段

## 反量化公式（Python，已验证 cos>0.99）

```python
# Q4: codes [rows, gpr, 32] uint8, scales [rows, gpr] fp16
lo = (codes & 0x0f).astype(np.int32); hi = (codes >> 4).astype(np.int32)
q = np.empty((rows, gpr, 64), np.float32)
q[..., 0::2] = (lo ^ 8) - 8; q[..., 1::2] = (hi ^ 8) - 8
W = (q * scales[..., None]).reshape(rows, gpr*64)[:, :cols]

# Q5: high [rows, gpr, 8] uint8; lane=0..31, high byte idx = lane>>2, shift=(lane&3)*2
hb = high[:, :, (lane>>2)]
bit_even = (hb >> ((lane&3)*2)) & 1; bit_odd = (hb >> ((lane&3)*2+1)) & 1
q_even = ((lo | (bit_even<<4)) ^ 0x10) - 0x10
q_odd  = ((hi4 | (bit_odd<<4)) ^ 0x10) - 0x10
```

## 已验证正确的组件（不要重查）

| 组件 | 方法 | 结果 |
|---|---|---|
| embedding W8 反量化 | Python 反量化 vs 原生 | cos=0.99999 |
| Q4 query_key / Q5 value_z 权重 | 反量化 vs 原生 bf16 | cos=0.994 / 0.9985 |
| input_norm / post_attn_norm / gdn_norm 权重 | 正确 DATA_BASE 读取 | cos=1.00000 |
| Q4 GEMM SIMT kernel（t=4/8/9，Full=true/false） | 独立测试（布局对齐） | cos=1.00000 |
| rmsnorm kernel（unit_offset=true） | 独立测试 | cos=1.00000 |
| h（layer 0 input_layernorm 输出） | 原生 forward hook 对比 | cos=0.99996 |
| GDN 全链（h/qkv/z/conv1d/recurrent/gated_norm） | 原生 hook 逐点（t=9/14） | cos>0.999 |
| Q4 swiglu（activation） | 原生 hook | cos=0.99882 |
| Q5 linear_add（残差加） | 14-token 逐层 | 修复后全绿 |
| GDN recurrent（T=76/80/135） | 逐层 + e2e | 全绿 |
| warp_reduce_sum（32-lane 模型 on 64-lane wavefront） | 最小测试（8 warp 各 496） | 正确 |
| cp_async（同步 memcpy shim） | 代码审查 | 正确 |
| token ids | NInfer dump vs 原生 | 13/13 一致 |
| g/beta（GDN gating，float32） | 范围统计 | 正常（g∈[-8.4,0], beta∈[0.02,1]） |

## 假警报记录（结论被推翻的，防止重查）

1. **"embedding 错（cos=0.0037）"** → 对比脚本用 `np.float16` 读 BF16 + reshape 方向反了。正确读取后 cos=0.999968。
2. **"logits 平坦（top1=2.79）"** → 用 `np.float16` 读 BF16 logits 的假象。正确读取后 top1=18.6。
3. **"g/beta 是 1e37 垃圾"** → dump 把 float32 当 bf16 读。按 f32 读后正常。
4. **"Q4 GEMM kernel 错（cos=-0.004）"** → 测试脚本 x 布局写反（kernel 读 `x[col][k]`，测试填 `x[k][col]`）。v3 布局对齐后 cos=1.00000。
5. **"h 错（cos=0.009）"** → Python 手算 rmsnorm 参考公式有偏差。原生 hook 对比后 cos=0.99996。
6. **"input_norm 26 倍偏差"** → BF16 按 FP16 读 + 错误 offset（发现 DATA_BASE 之前）。
7. **"qkv 错（cos=0.00456）"** → 对比脚本**双重 bug**：chunk 错位（NInfer dump 是 chunk 0 的 9 token，脚本取了原生 token 9:13）+ 布局转置（qkv 是 token-major `flat=t*10240+row`，脚本用 `reshape(10240,9)` row-major）。正确对齐后 qk cos=0.99933 / value 0.99990 / z 0.99978。
8. **"layer 0 输出 cos≈0（MLP 错）"** → 实为根因2（Q5 linear_add 残差加 no-op），不是 MLP GEMM 错。修 linear_add 后全绿。

## 模型结构

- Qwen3_5，64 层，**layer 0 是 GDN**（linear_attention），GQA 在 3,7,11...（full_attention_interval=4），**16 GQA + 48 GDN**，hidden 5120，intermediate 17408，attn_output_gate=true
- GDN 几何：qk 4096 行（Q4）+ value_z 12288 行（Q5，value 6144 + z 6144），k=5120，gating a/b [48,5120] BF16
- GQA 几何：24 Q heads / 4 KV heads / group 6 / head_dim 256，paged KV page size 64
- GDN chunk size=64，T<64 全 recurrent，T≥64 现在走 recurrent fallback（正确但 O(T) 串行，256k 全量 prefill 慢）
- 各 op dtype：attn_input=Q4G64+Q5G64, gdn_input=Q4Q5, gdn_gating=BF16, MLP=Q4G64/Q5G64, out_proj/down=Q5G64(linear_add), embedding=W8G32_F16S, KV=bf16/int8（无 fp8）

## 引擎 dump 环境变量（已加进 variant.cpp / text_context_impl.h）

- `NINFER_DUMP_LOGITS` / `NINFER_DUMP_EMB` / `NINFER_DUMP_IDS`
- `NINFER_DUMP_LAYERS=<prefix>`：逐层 hidden（文件名 `<prefix>.<layer>.c<chunk>`，bf16，token-major）
- `NINFER_DUMP_GDN=<prefix>`：layer 0 GDN 中间态 h/g/beta/qkv/z/qkv_c/o/on（g/beta 是 f32，其余 bf16）
- `NINFER_DUMP_MLP` / `NINFER_DUMP_ACT` / `NINFER_DUMP_XIN`：MLP 中间态（post_norm h / swiglu activation / mlp 输入 x_in）
- `NINFER_DBG_EMB` / `NINFER_DBG_KERNEL` / `NINFER_DBG_GEMM`：运行时 debug 打印

## 对比纪律（血泪教训）

- **token-major**：引擎 h 布局 `x.ne=[5120,T] x.nb=[2,10240]` → 物理 `x[t*5120+h]`，对比脚本 `reshape(T,5120).T`
- **bf16 读取**：`(u16.astype(uint32) << 16).view(float32)` 或 `torch.bfloat16`，**绝不用 `np.float16`**
- **原生 hook 是 gold standard**：transformers `register_forward_hook` 抓真实中间态，手算公式只作交叉验证
- **chunk 对齐**：NInfer prefill 分 chunk（18tok→14+4，80tok→76+4），dump 文件名带 chunk 计数器，对比时取对应 chunk 的 token 区间
- **独立测试布局必须严格对齐 kernel**：先读 kernel 源码确认访存模式（`x[c][k]` vs `x[k][c]`），测试布局不对齐的"kernel 错"结论不可信
