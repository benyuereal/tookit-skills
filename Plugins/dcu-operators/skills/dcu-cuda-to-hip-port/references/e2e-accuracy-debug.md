# e2e 精度对拍与退化根因定位（NInfer→DCU 2026-08 实战）

整引擎链接通过、e2e 能跑之后，输出退化（垃圾/重复/不连贯）但不崩溃时的完整定位流程。
本文件是 SKILL.md §5.6 的展开：3 个根因的完整细节、6 次假警报的教训、对比脚本模板。

## 1. 核心原则

1. **原生模型 forward hook 是 gold standard**。用 transformers 跑原生模型，
   `module.register_forward_hook` 抓每层中间态 dump 成 .bin。
   **绝不用手算 Python 参考**——手算公式（silu 系数、norm 的 eps、gate/up 顺序）
   总有偏差，会出假警报。本会话 6 次假警报全部来自手算参考或布局错位。
2. **独立测试布局必须严格对齐 kernel**。测试脚本的 reshape 方向、chunk 对齐、
   dtype 读取方式，任何一项错都会把"对的 kernel"判成"错的"。
3. **逐层对比找第一个错层**，再在该层内定位 op，再查该 op 的 dispatch t 阈值。
   不要跳步猜。

## 2. 三个根因（全是 dispatch t 阈值 → MMA stub 静默垃圾）

### 根因1：Q4 swiglu（MLP gate_up）
- dispatch：`t=1→gemv_pair(SIMT) / t>1→small_t_exact / mma_split_half_pair_*(MMA)`
- t>1 全落 MMA stub → activation 全 0 → MLP 不贡献 → 输出退化
- 修复：t>1 时复用已验证的 Q4 GEMM SIMT kernel（`launch_q4_simt_r8_c8`）算
  gate_up [34816,t]，再跑 swiglu elementwise kernel（`silu(gate)*up`）。
  gate_up 权重布局：前 17408 行 gate，后 17408 行 up（Qwen3 标准 [gate; up] 拼接）。
- 验证：activation cos=0.99882 ✅

### 根因2：Q5 linear_add（out_proj / down_proj，GEMM+残差加融合）
- dispatch route 表（kK6144Routes = out_proj）：
  - t=1: GemvResidual
  - t=2-13: Split2ExactResidual (SIMT)
  - t=14-32: MmaResidualR64C16 (MMA) ← t=14 落这里
  - t=33-48: MmaResidualR64C24 (MMA)
  - t=49-128: MmaResidualR64C64 (MMA)
  - t=129+: MmaResidualR64C128 (MMA)
- t=9 走 SIMT（对），t=14 落 MMA stub（no-op → x 不变 → 残差加没生效）
- 现象：x_in（mlp_tail 输入）range 异常小（[-0.43,0.50]，等于 GDN 前的 layer_in
  而非 GDN 后的 x+gdn_out）
- 修复：dispatch 入口加 fallback——t 超出 SIMT exact 范围时，用通用 Q5 GEMM SIMT
  （`q5_dispatch`，已验证 t=14 正确）+ 单独 residual_add kernel。
- 验证：18/25/60 token 全对 ✅

### 根因3：GDN chunked（linear-attention prefill）
- dispatch：`T_full = (T/64)*64`；T_full>0 → l2norm + launch_chunked(MMA) + recurrent tail；
  T_full=0（T<64）→ 全 recurrent（SIMT）
- T=76 → T_full=64 → chunked MMA stub → 前 64 token 错
- 修复：chunked 版 `gated_delta_net` 整个函数体改成
  `launch_recurrent_inout(q,k,v,g,beta,scale,normalize_qk, ssm_state_in, ssm_state_out, out, stream)`
  ——recurrent kernel 内部循环整个 T（支持任意 T，in-kernel 做 l2norm）。
- 性能：引擎 prefill 本身分 chunk（`--prefill-chunk`），recurrent fallback 只处理
  小 chunk + 传 state，10k prefill 43.58 tok/s（和短 prompt 相当），不是灾难。
  256k 全量 prefill 估算 ~100min（慢但能跑），后续移植 chunked 成 SIMT 提性能。
- 验证：T=80 和原生完全一致 ✅

## 3. 六次假警报（对比脚本 bug，不是 kernel bug）

| # | 假警报 | 真因 |
|---|---|---|
| 1 | embedding "索引 bug" | 对比脚本 reshape 布局反了 |
| 2 | Q4 GEMM "kernel 错" | 测试脚本 x 布局写反 |
| 3 | h "错" | Python 手算参考公式有偏差 |
| 4 | g/beta "1e37 垃圾" | dump 把 float32 当 bf16 读 |
| 5 | qkv "cos=0.004" | chunk 错位（引擎 18tok 切 14+4，脚本取了 chunk 1）+ 布局转置 |
| 6 | "on 对但 mlp.h 错" 误判为 MLP 错 | 实际是 out_proj（Q5 linear_add）MMA stub，残差没加 |

**教训**：
- 对比脚本的 reshape 方向、chunk 对齐、dtype 读取，每一项都要先单独验证
  （用已知正确的数据对一遍），再用来判 kernel 对错。
- 手算参考只能做 sanity check（量级/范围），不能做 cos 对比。
- "某中间态对、下一态错"时，先确认中间态的**语义**（是 out_proj 输入还是输出？
  是 GDN 前还是 GDN 后的残差？），再定位 op。

## 4. 对比脚本模板

### 4.1 bf16 读取（numpy，不用 torch）
```python
import numpy as np
def read_bf16(path):
    raw = open(path, "rb").read()
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)
```

### 4.2 token-major 布局
引擎 dump 的 hidden 是 token-major：物理 `x[t*hidden + h]`。
对比时 `reshape(T, hidden)`（**不是** `reshape(hidden, T)`）。
若 kernel 输出是 [hidden, T]（row-major），要 `.T` 转置。

### 4.3 chunk 对齐
引擎 prefill 分 chunk（如 18tok → 14+4）。对比脚本必须取**同一 chunk** 的 token：
- 先 dump 文件大小判断 chunk 大小（`size / hidden / 2` = token 数）
- 原生 dump 是完整 T 个 token，取 `[:chunk_size]` 对齐引擎 chunk 0

### 4.4 逐层对比
```python
for i in range(64):
    ni = read_bf16(f"ninfer_layer.{i}.c0").reshape(T0, 5120)
    nat = read_bf16(f"native_layer.{i}.bin").reshape(T, 5120)[:T0]
    cos = float(np.dot(ni.reshape(-1), nat.reshape(-1)) /
                (np.linalg.norm(ni) * np.linalg.norm(nat) + 1e-12))
    if cos < 0.99:
        print(f"layer {i}: cos={cos:.5f} WRONG")
        break
```

### 4.5 原生 forward hook dump
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True).cuda().eval()
dumps = {}
def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple): out = out[0]
        dumps[name] = out.detach().cpu().numpy()
    return hook
# 挂到每层的 input_layernorm / post_attention_layernorm / mlp 等
for i, layer in enumerate(model.model.layers):
    layer.input_layernorm.register_forward_hook(make_hook(f"layer{i}_h"))
    layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer{i}_mlp_h"))
# forward 后 dumps 里就是各中间态
```

## 5. 混合注意力 KV cache 显存

混合模型（GQA full-attention + GDN/linear-attention）的 KV cache 显存**只算
full-attention 层**——linear-attention 层用 SSM state（固定大小，不随 context 增长）。

```
KV bytes = full_attn_layers × kv_heads × head_dim × 2(K+V) × dtype_bytes × context
```

Qwen3.8-27B 实例（64 层 = 16 GQA + 48 GDN，kv_heads=4，head_dim=256）：
- 256k bf16：`16 × 4 × 256 × 2 × 2B × 262144 = 16GiB`（64GiB 卡放得下）
- 误算全部 64 层：`64 × 4 × 256 × 2 × 2B × 262144 = 64GiB`（放不下）→ 错

**务必用引擎实际 `--kv-capacity` summary 验证**（`free after weights` /
`KV cache payload` / `free after startup`），别信手算。本发现省掉了写 SIMT
i8 prefill attention 的大工程（原以为必须 int8 KV，实际 bf16 就放得下）。

## 6. 性能基准纪律

- 测 agent 自身在跑的模型时 agent 须静默 + 严格串行（自身请求争抢后端拉低吞吐 40-70%）。
- 性能数据以用户后台日志为准（用户说日志某值就直接采信，不客户端复测）。
- 最终报告只留有效结果，不标早期失败轮次。
- 原生 transformers 的 prefill 数字不可靠（单次 forward 被固定开销主导，
  13/43/153tok 都 ~600ms），decode 数字才可靠。NInfer 的 prefill 随长度正常提升
  才符合物理规律。
