# Qwen3.8-27B 融合 QK-Norm+RoPE kernel — 完整源码与集成配方

> 2026-09-28 在 <device-A>（BW150/gfx936/DTK-26.04）实测验证。
> 配套脚本（patch_vllm.py / patch_sglang.py / qwen38_kernel.cu）在本地 `/root/dcu-qwen38-fused-qknorm-rope/`。

## 1. 背景：为什么不能直接用现成方案

**hipBLASLt epilogue 融合路径 DCU 走不通**：DTK-26.04 的 hipBLASLt 0.10 epilogue 只支持 pointwise（RELU/BIAS/GELU/SIGMOID/TANH/LEAKYRELU/DGELU/BGRAD），**没有 RMSNorm/RoPE**。`hipblaslt-ext-op.h` 的 LayerNorm 是独立函数（仅 FP32、m≤4096），不是 epilogue。

**LightOp 现成融合 kernel 都不支持 Qwen3.8 配置**（head_size=256 + partial rotary_dim=64）：

| kernel | 问题 |
|---|---|
| `rms_rotary_embedding_fuse` | 接受 head_size=256 但假设**全维旋转**（rotary_dim==head_size），按 512 宽读 cos_sin_cache，Qwen3.8 的 cache 只有 64 宽 → 越界读 → cos_sim=0.0 |
| `fuse_rms_mrope_cuda` | 有 rotary_dim 参数，但 ① 断言 `head_size ∈ {64,128}` ② RoPE 逻辑对全部 head_size 元素套旋转（cos 索引用 `%half_rd` 回绕），partial rotary 的尾部 pass 维会被错误旋转 |

**Qwen3.8-27B 关键配置**（config.json text_config）：
- 64 层 = 16 全注意力（25%，`full_attention_interval=4`）+ 48 线性注意力（GatedDeltaNet，不受益）
- head_size=256，rotary_dim=64（`partial_rotary_factor=0.25`），neox style，theta=1e7
- `attn_output_gate=True`（QKV 投影带 gate，q 要先 split 出 gate）
- QK-Norm 是 GemmaRMSNorm（`x * rsqrt(mean(x²)+eps) * (1.0+weight)`，注意是 **1+weight**，weight 初始化为 0）
- text-only 时 positions 是 1D（M-RoPE 的 interleaved 逻辑只在多模态 2D positions 时生效，text-only 就是标准 neox partial rotary）

## 2. Kernel 设计

- 1 block = 1 token，4 warp（256 线程），每 warp 处理 1 个 head（grid-stride over heads）
- Phase 1：warp 内对全 head_size=256 维做 GemmaRMSNorm（warp reduce + broadcast）
- Phase 2：前 64 维 neox RoPE（前半/后半配对），后 192 维 pass-through
- cos/sin 从 cos_sin_cache 加载到 smem（每 token 一次）
- **s_head 必须 per-warp**（`s_head[4][HEAD]`），否则跨 warp 竞争

## 3. 完整 kernel 源码（已验证）

```cuda
// Qwen3.8-27B 全注意力层: 融合 QK-Norm(GemmaRMSNorm) + RoPE(partial rotary)
// 配置: head_size=256, rotary_dim=64 (neox, 前64维旋转, 后192维pass), text-only
// DCU: warp=64, bf16
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>

#define HEAD 256
#define RD 64
#define HALF 32
#define LANES 64
#define PER_LANE (HEAD / LANES)  // 4

template <typename scalar_t>
__global__ void qwen38_fused_qknorm_rope_kernel(
    scalar_t* __restrict__ q,            // [num_tokens, n_qh*HEAD]
    scalar_t* __restrict__ k,            // [num_tokens, n_kh*HEAD]
    const scalar_t* __restrict__ cos_sin, // [max_pos, RD]
    const int64_t* __restrict__ positions, // [num_tokens]
    const scalar_t* __restrict__ gamma_q,  // [HEAD]
    const scalar_t* __restrict__ gamma_k,  // [HEAD]
    int num_tokens, int n_qh, int n_kh, float eps)
{
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    int warp = threadIdx.x / LANES;
    int lane = threadIdx.x % LANES;
    int nwarps = blockDim.x / LANES;

    __shared__ scalar_t s_cos[HALF];
    __shared__ scalar_t s_sin[HALF];
    __shared__ scalar_t s_head[4][HEAD];  // per-warp, 避免跨 warp 竞争

    int pos = positions[token];
    if (lane < HALF) {
        s_cos[lane] = cos_sin[(int64_t)pos * RD + lane];
        s_sin[lane] = cos_sin[(int64_t)pos * RD + HALF + lane];
    }
    __syncthreads();

    for (int h = warp; h < n_qh + n_kh; h += nwarps) {
        bool is_q = (h < n_qh);
        int hidx = is_q ? h : (h - n_qh);
        scalar_t* base = is_q ? (q + (int64_t)token * n_qh * HEAD + hidx * HEAD)
                              : (k + (int64_t)token * n_kh * HEAD + hidx * HEAD);
        const scalar_t* gamma = is_q ? gamma_q : gamma_k;

        // Phase 1: load + GemmaRMSNorm over full HEAD
        float x[PER_LANE];
        float sumsq = 0.f;
        #pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            x[i] = (float)base[lane * PER_LANE + i];
            sumsq += x[i] * x[i];
        }
        // warp reduce over full 64-lane wavefront (no mask, DCU wavefront=64)
        #pragma unroll
        for (int off = LANES / 2; off > 0; off >>= 1)
            sumsq += __shfl_down(sumsq, off);
        // broadcast lane 0's total to all lanes (shfl_down 后仅 lane0 持有总和)
        sumsq = __shfl(sumsq, 0);
        float rstd = rsqrtf(sumsq / HEAD + eps);
        #pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane * PER_LANE + i;
            s_head[warp][d] = (scalar_t)(x[i] * rstd * (1.0f + (float)gamma[d]));
        }
        __syncwarp();

        // Phase 2: RoPE on first RD dims (neox), pass-through rest
        #pragma unroll
        for (int i = 0; i < PER_LANE; i++) {
            int d = lane * PER_LANE + i;
            if (d < RD) {
                float v = (float)s_head[warp][d];
                float c = (float)s_cos[d % HALF];
                float s = (float)s_sin[d % HALF];
                if (d < HALF) {
                    float partner = (float)s_head[warp][d + HALF];
                    base[d] = (scalar_t)(v * c - partner * s);
                } else {
                    float partner = (float)s_head[warp][d - HALF];
                    base[d] = (scalar_t)(v * c + partner * s);
                }
            } else {
                base[d] = s_head[warp][d];
            }
        }
        __syncwarp();
    }
}

void qwen38_fused_qknorm_rope(
    torch::Tensor q, torch::Tensor k,
    torch::Tensor cos_sin, torch::Tensor positions,
    torch::Tensor gamma_q, torch::Tensor gamma_k,
    double eps)
{
    int num_tokens = q.size(0);
    int n_qh = q.size(1) / HEAD;
    int n_kh = k.size(1) / HEAD;
    int nwarps = 4;
    dim3 grid(num_tokens);
    dim3 block(nwarps * LANES);
    auto stream = c10::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "qwen38_fused_qknorm_rope", [&] {
        qwen38_fused_qknorm_rope_kernel<scalar_t><<<grid, block, 0, stream>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            cos_sin.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
            gamma_q.data_ptr<scalar_t>(), gamma_k.data_ptr<scalar_t>(),
            num_tokens, n_qh, n_kh, (float)eps);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qwen38_fused_qknorm_rope", &qwen38_fused_qknorm_rope,
          "Fused QK-Norm + RoPE for Qwen3.8");
}
```

## 4. 编译

```python
from torch.utils.cpp_extension import load
ext = load(
    name="qwen38_fuse",
    sources=["/path/to/qwen38_fuse_kernel.cu"],
    extra_cuda_cflags=["-O3"],          # 不要加 --use_fast_math
    build_directory="/opt/qwen38_fuse/build",  # 目录必须先 mkdir
)
# 产物: /opt/qwen38_fuse/build/qwen38_fuse.so
```
- **torch 版本必须匹配**：.so 是 torch extension，torch 2.9 编的不能在 2.11 容器用（反之亦然）。每个容器单独编。
- 加载：`importlib.util.spec_from_file_location("qwen38_fuse", "/opt/qwen38_fuse/qwen38_fuse.so")`

## 5. vLLM 集成（已验证 +9.9%）

**patch 点**：`vllm/model_executor/models/qwen3_next.py` 的 `Qwen3NextAttention.forward`，把
```python
q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)
q, k = self.rotary_emb(positions, q, k)
```
替换为 env 门控的融合路径（完整 patch 脚本见 `/root/dcu-qwen38-fused-qknorm-rope/patch_vllm.py`）：
```python
if (os.environ.get("VLLM_QWEN38_FUSED_QKNORM_ROPE", "0") == "1"
        and positions.ndim == 1
        and self.head_dim == 256
        and getattr(self.rotary_emb, "rotary_dim", 256) == 64):
    cos_sin_cache = self.rotary_emb.cos_sin_cache
    if (cos_sin_cache.device != q.device or cos_sin_cache.dtype != q.dtype):
        cos_sin_cache = cos_sin_cache.to(q.device, dtype=q.dtype, non_blocking=True)
        self.rotary_emb.cos_sin_cache = cos_sin_cache
    q = q.contiguous()
    k = k.contiguous()   # gate 分支里 k 来自 qkv.split, 非 contiguous, 必须加
    _qwen38_fused_qknorm_rope(q, k, cos_sin_cache, positions,
                              self.q_norm.weight, self.k_norm.weight,
                              self.q_norm.variance_epsilon)
else:
    # 原分离路径
```
**安全守卫三件套**（缺一不可）：env 开关（默认关）+ `positions.ndim==1`（text-only）+ `head_dim==256 and rotary_dim==64`（防误用于其他 Qwen3-Next 变体）。
**启用**：`VLLM_QWEN38_FUSED_QKNORM_ROPE=1 vllm serve ...`

## 6. SGLang 集成（kernel 级已验证，e2e 差最后一步）

**patch 点**：`sglang/srt/models/qwen3_next.py` 的 `self_attention`（0.5.12）/ `forward_prepare_native`（0.5.18），把
```python
q, k = self._apply_qk_norm(q, k)
q, k = self.rotary_emb(positions, q, k)
```
替换为同样的 env 门控融合路径（env 名 `SGLANG_QWEN38_FUSED_QKNORM_ROPE`，patch 脚本见 `/root/dcu-qwen38-fused-qknorm-rope/patch_sglang.py`）。

**SGLang 测试环境专属坑**（生产镜像 sglang 0.5.12-20260620，torch 2.9.0 + vllm 0.15.1）：
1. **vllm `_C` op 未注册**（镜像预存问题，`torch.ops._C.rms_norm` 缺失）：SGLang 的 `GemmaRMSNorm.forward_hip` 依赖它，且用于**所有层归一化**（不只 qk_norm），所以 baseline 和 fused 都会撞。绕法：patch `sglang/srt/layers/layernorm.py` 把 `_has_vllm_rms_norm` 强制置 False（走 forward_native 纯 torch）。注意 `SGLANG_USE_AITER=1` 不行（触发 aiter 其他不兼容 import）。
2. **attention backend**：默认 aiter 会报 `paged_attention_ragged` 未定义（aiter 版本不匹配），必须 `attention_backend="fa3"`（对齐生产脚本）。
3. **page_size**：gfx936 要求 64 的倍数；且 Mamba 调度 `no_buffer` 模式会强制 page_size=1 → 必须 `mamba_scheduler_strategy="extra_buffer"`（生产脚本同款）。
4. **生产配置参考**（<device-A> 上 `/data0/<user>/qwen/run_qwen3.8-27b.sh`，路径因人而异）：`--attention-backend fa3 --page-size 64 --mamba-scheduler-strategy extra_buffer --json-model-override-args '{"rope_parameters": {...yarn...}}'`。
5. **生产镜像是 `sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620`**（注意 tag 是 `dtk26.04` 不是 `dtk2604`），不是名字带 qwen38 的 14e5255 镜像（那个是别人实验镜像，vllm 0.21.0）。查生产用哪个镜像：看跑过该模型的容器（`docker ps -a | grep qwen3.8`）的 Image 字段，别信镜像名。

## 7. 验证方法（两级）

**kernel 级**（快，分钟级）：
- 参考路径用**真实框架类**（vLLM: `MRotaryEmbedding`+`GemmaRMSNorm`；SGLang: `get_rope`+纯 torch GemmaRMSNorm 参考公式）
- 判据：cos_sim > 0.999 且 maxdiff ≤ 0.0312（bf16 1 ULP）
- 多 shape：tokens = 1/64/512/2048/8192
- vLLM 的 CustomOp 实例化需要 `set_current_vllm_config(VllmConfig())` 上下文
- SGLang 的 `get_rope` 签名：`get_rope(head_size, rotary_dim, max_position, base, is_neox_style, dtype, partial_rotary_factor)`（rotary_dim 传 head_size，partial 因子单独传，内部相乘）

**端到端**（慢，27B TP=2 加载 ~5 分钟）：
- 正确性：greedy（temperature=0）5 个 prompt × 128 token，对比**token 序列逐位一致**（最强证据）
- 性能：64 prompt × 256 token 批量生成，测吞吐 tok/s
- **第一次跑含 aiter JIT 编译开销（~150s vs 正常 ~30s），性能对比必须重跑一次公平的**
- vLLM 脚本必须有 `if __name__ == "__main__":` 守卫（spawn 多进程）
- SGLang 0.5.12 的 `Engine.generate` 用 dict 形式 sampling_params（`{"temperature":0.0,"max_new_tokens":128,"sampling_seed":42}`），不是 SamplingParams 对象；token 在 `meta_info` 里（0.5.12 的 key 名与 0.5.18 不同，需现查）

## 8. 实测结果（<device-A> / BW150 / gfx936 / DTK-26.04）

**kernel 级**（Qwen3.8 全注意力层配置，BF16）：

| tokens | 分离 3 kernel | 融合 1 kernel | 加速 |
|---|---|---|---|
| 64 | 0.63ms | 0.017ms | 36.6x |
| 512 | 0.63ms | 0.037ms | 17.0x |
| 2048 | 1.02ms | 0.122ms | 8.4x |
| 8192 | 3.48ms | 0.405ms | 8.6x |
| 16384 | 6.69ms | 0.780ms | 8.6x |

**vLLM 端到端**（Qwen3.8-27B，TP=2，enforce_eager，64 prompt × 256 token）：

| 指标 | 分离 | 融合 | 提升 |
|---|---|---|---|
| 吞吐 | 524.3 tok/s | 576.2 tok/s | **+9.9%** |
| 总耗时 | 31.25s | 28.43s | -9.0% |
| 正确性 | — | 5 prompt token 逐位一致 | PASS |

**为什么 kernel 快 8-36x 但端到端只 +9.9%**：Qwen3.8 是混合架构，只有 25% 层（16/64）是全注意力受益，其余 75% 线性注意力不受益；且 QK-Norm+RoPE 只占 forward 一小部分（大头是 GEMM/attention/线性注意力）。

## 9. 交叉引用

- DCU kernel 开发规范（warp=64、gfx 现查、精度陷阱）→ `dcu-kernel-dev`
- LightOp 融合方法论（六大模式族、八大手法）→ `dcu-lightop-fusion`
- 算子优化闭环（profile→定位→改→验证）→ `dcu-op-optimization`
- vLLM 算子替换流程 → `dcu-vllm-op-replace`
- H100 Fused QKV GEMM+QKNorm+RoPE 类文章对 DCU 的评估结论：方法论有用（NormRope 占 GEMM 60% 的判断被实机证实），实现路径不可搬（hipBLASLt epilogue 无 NormRope）
