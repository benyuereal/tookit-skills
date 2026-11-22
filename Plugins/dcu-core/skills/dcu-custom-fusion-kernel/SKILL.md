---
name: dcu-custom-fusion-kernel
description: "为模型不支持的结构手写 DCU HIP 融合 kernel（HIP fusion kernel）。何时用：现有算子库缺这个组合、或需要把多个算子合成一个 kernel。"
version: 1.0.1
author: DCU Engineer
license: MIT
kind: tool
---

# DCU 自定义融合 kernel：填补框架融合缺口（dcu-custom-fusion-kernel）

> 定位：当框架自带的融合 kernel（LightOp / sgl_kernel / vllm `_C`）**不支持目标模型的配置**（如 head_size=256 + partial rotary、特殊 norm 变体）时，**手写一个 DCU HIP kernel 填补缺口，并用 env 开关接进 vLLM/SGLang**。本 skill 沉淀方法 + DCU 专属坑 + 一个已验证的完整案例（Qwen3.8-27B / Qwen3-Next 的 QK-Norm+RoPE 融合，vLLM 端到端实测 +9.9%）。
> 与 `dcu-kernel-dev`（写 HIP kernel 的通用方法学）互补：本 skill 聚焦「**填框架融合缺口 + 接线进 serving 栈**」这条具体工作流。

## 触发条件
- 目标模型有 norm+rope（或类似 memory-bound 算子链），但框架现成融合 kernel 报「不支持该配置」（如 `Expected head_size == 64 || head_size == 128`）
- 或现成 kernel 只支持全维旋转，而模型是 partial rotary（rotary_dim < head_size）
- 想省掉分离 kernel 的 launch 开销 + 中间 HBM 往返

## 一、先确认「现成 kernel 真的不支持」（别重复造）
0. **hipBLASLt epilogue 融合路径 DCU 走不通**（先排除）：DTK-26.04 的 hipBLASLt 0.10 epilogue 只支持 pointwise（RELU/BIAS/GELU/SIGMOID/TANH/LEAKYRELU/DGELU/BGRAD），**没有 RMSNorm/RoPE**；`hipblaslt-ext-op.h` 的 LayerNorm 是独立函数（仅 FP32、m≤4096），不是 epilogue。所以「融进 GEMM epilogue」这条路在 DCU 上不可行，只能走独立融合 kernel。
1. 查框架融合 kernel 的 C++ 源码断言（LightOp 源码在 `/root/dcu-src/lightop-master/csrc/`；预编译 .so 无源码时看本地缓存）。
2. 典型缺口（实测）：
   - `fuse_rms_mrope_cuda` 断言 `head_size ∈ {64,128}`，且 RoPE 逻辑对**全部 head_size 元素**套旋转（cos 索引用 `%half_rd` 回绕）→ 只支持 `rotary_dim == head_size`（全维旋转）。
   - `rms_rotary_embedding_fuse` 支持 head_size=256，但假设**全维旋转**，按 `head_size*2` 宽读 cos_sin_cache；模型是 partial rotary（cache 只有 `rotary_dim*2` 宽）→ 越界读 → 全错（cos_sim=0.0）。
3. 若模型是 partial rotary（如 Qwen3.8：head_size=256, rotary_dim=64）→ 现成 kernel 都不适用 → 需手写。

## 二、kernel 设计（QK-Norm + RoPE 融合范式）
- **GemmaRMSNorm 作用全 head_size**（不是 rotary_dim）：`x * rsqrt(mean(x²)+eps) * (1.0 + weight)`（注意 `1+weight`，Gemma 风格，weight 初始化为 0）。
- **RoPE 只作用前 rotary_dim 维**（neox：前半/后半配对），**后 (head_size - rotary_dim) 维 pass-through**（直接写回 norm 结果，不旋转）。
- 一个 block 处理一个 token，多 warp 各处理不同 head；cos/sin 从 cos_sin_cache 读进 smem。
- cos_sin_cache 布局：`cat((cos, sin), dim=-1)` = `[max_pos, rotary_dim*2]`（vLLM 与 SGLang 一致，kernel 接口可直接复用）。
- 完整已验证 kernel 源码见 `references/qwen38_qknorm_rope_kernel.md`。

## 三、★DCU warp=64 的 raw-HIP shuffle 规约坑（核心，必读）
写 raw HIP kernel 做 warp 内 reduce 时，DCU wavefront=64，有两个 NVIDIA 习惯会**直接算错**：
1. **mask 截断**：`__shfl_down_sync(mask, val, off)` 的 mask 是 **32-bit unsigned**。传 64-bit `0xffffffffffffffffULL` 会被截断成 `0xffffffff`，**只覆盖前 32 lane**，后 32 lane 不参与规约 → 结果错。
2. **缺 broadcast**：`shfl_down` 规约后**只有 lane 0 持有总和**，其他 lane 是部分和。若所有 lane 用各自的（错误的部分）和算 rstd → norm 全错。

**正确写法**：
```cpp
// 无 mask 版本覆盖整个 64-lane wavefront
for (int off = 32; off > 0; off >>= 1)
    sumsq += __shfl_down(sumsq, off);
// 规约后广播 lane 0 的总和给所有 lane
sumsq = __shfl(sumsq, 0);
float rstd = rsqrtf(sumsq / HEAD + eps);
```
- 同理 `__syncwarp()` 用**无参版本**（覆盖全 wavefront），别传 64-bit mask。
- 症状：norm-only 测试 maxdiff 达 ~1.8（远超 bf16 1 ULP 的 ~0.03），cos_sim ~0.996（不是 0.9999）。
- 对照：`dcu-kernel-dev` §3.4 讲的是 **TileLang** 的 `warp_size=64`；本条是 **raw HIP** 的 mask/broadcast 坑，两者都要记。

## 四、接线进 vLLM / SGLang（env 开关 + 可回滚）
1. **编译 .so**：`torch.utils.cpp_extension.load(sources=[kernel.cu], extra_cuda_cflags=["-O3"])`。**DTK 不支持 `--use_fast_math`，别加**（报 `unsupported option`）。**torch 版本必须和容器一致**（.so 是 torch extension，跨版本加载失败）→ 每个容器单独编译。
2. **lazy loader**：模型文件顶部加 `_get_xxx_fuse()`，用 `importlib.util.spec_from_file_location` 加载 .so，避免 import 期副作用。
3. **env 门控 + 安全守卫**：在 forward 的 norm+rope 处加
   ```python
   if (os.environ.get("VLLM_XXX_FUSED", "0") == "1"
           and positions.ndim == 1          # text-only
           and self.head_dim == 256         # 硬编码配置守卫
           and getattr(self.rotary_emb, "rotary_dim", 256) == 64):
       ... 调融合 kernel ...
   else:
       ... 原分离路径 ...
   ```
   守卫确保只在该模型配置下生效，不误伤其他变体。
4. **备份可回滚**：patch 前 `cp model.py model.py.bak_xxx`。
5. **contiguous 坑**：`qkv.split` 出来的张量非 contiguous，融合 kernel 前对 q/k 加 `.contiguous()`（q 经 reshape 后通常已 contiguous，k 不一定）。
6. **vLLM CustomOp 实例化需要 config 上下文**：单测里直接 `MRotaryEmbedding(...)`/`GemmaRMSNorm(...)` 会报 `Current vLLM config is not set`，用 `with set_current_vllm_config(VllmConfig()):` 包裹。

## 五、验证（两级，都要过）
1. **kernel 级**：新 kernel vs 框架**真实**分离路径（用框架自己的 norm+rope 类做参考，别手写参考——手写易有偏差）。判据 cos_sim > 0.999 且 maxdiff < 0.03（bf16 1 ULP）。多组 shape（1/64/512/2048/8192 token）。
2. **端到端**：起服务，greedy 解码（temperature=0）对比融合 vs 分离的 **token 序列逐位一致**（最强证据）。再跑性能基准（多 prompt 批量，测吞吐）。
- 注意：第一次跑可能含 aiter JIT 编译开销，**性能对比要重跑一次公平 baseline**（本会话第一次 baseline 157s 含 JIT，重跑后 30.75s）。

## 六、SGLang 在 DCU 跑 Qwen3-Next 的配置坑（接线时可能遇到）
> 标注：以下为 <device-A> 上 sglang 0.5.18 dtk2604 镜像实测遇到的配置项。**SGLang 端到端本会话未完全跑通**（kernel 级已验证正确，e2e 被镜像配置问题逐个卡住），下列是「让 SGLang 能跑 Qwen3-Next」需要核对的配置，**非已验证完整流程**，别当 best practice 直接套。
- 生产部署脚本（`run_qwen3.8-27b.sh`）关键项：`SGLANG_USE_LIGHTOP=1`、`SGLANG_USE_AITER_LINEAR_ATTN=1`、`--attention-backend fa3`、`--page-size 64`（gfx936 要求 page size 是 64 倍数）、`--json-model-override-args` 带 rope 参数（mrope_interleaved / mrope_section / partial_rotary_factor / rope_type）。
- 干净测试容器常见报错及绕法：
  - vllm `_C` op 未注册（`rms_norm` 缺失，`torch.ops._C` 只有 1 个 op）→ SGLang 的 `GemmaRMSNorm.forward_hip` 依赖它；可 patch `layernorm.py` 把 `_has_vllm_rms_norm` 置 False 走 `forward_native`（纯 torch），或开 aiter。
  - aiter attention backend `paged_attention_ragged` 未定义 → 改用 `--attention-backend fa3`。
  - fa3 报 `Paged KV cache block size must be divisible by 64` → `page_size=64`。
  - Mamba 调度 `no_buffer` 模式会强制 page_size=1 → 必须 `mamba_scheduler_strategy="extra_buffer"`（生产脚本同款）。
  - 生产镜像是 `sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620`（tag 是 `dtk26.04` 不是 `dtk2604`），不是名字带 qwen38 的 14e5255 镜像（那是别人实验镜像，vllm 0.21.0）。查生产用哪个镜像：看跑过该模型的容器（`docker ps -a | grep qwen3.8`）的 Image 字段，别信镜像名。
- SGLang `Engine.generate` 用 **dict 形式** sampling_params（`{"temperature":0,"max_new_tokens":N,"sampling_seed":42}`），不是 vLLM 的 SamplingParams 对象；参数名是 `sampling_seed` 不是 `seed`。

## 七、收益预期（方法论，非基线）
- 单算子（QK-Norm+RoPE）融合可快数倍到数十倍（省 2 次 launch + 中间 HBM 往返）。
- 但**端到端收益取决于受益层占比**：Qwen3-Next 混合架构只有 ~25% 层是全注意力（受益），其余线性注意力不受益；且 QK-Norm+RoPE 只占 forward 一小部分。实测 Qwen3.8-27B 端到端吞吐 ~+10%。
- **别用单算子加速比外推端到端**。

## 参考
- `references/qwen38_qknorm_rope_kernel.md` — 完整 kernel 源码 + vLLM/SGLang patch 配方 + 构建/测试命令 + 实测数据。

## 使用须知
- 本 skill 的 kernel 是 Qwen3.8 配置（head_size=256 / rotary_dim=64）的已验证实现；换模型配置需改 kernel 常量并重新验证。
- 数字以真机 METRIC 行为准；本 skill 只留方法 + 坑 + 一个案例。
- 与 `dcu-kernel-dev`（通用 HIP 方法学）、`dcu-lightop-fusion`（LightOp 融合方法论）、`dcu-vllm-op-replace`（算子替换进 vLLM）有交叉，可考虑整合。

## 验收

1. 融合 kernel 编译通过（`hipcc` 无 warning 视为通过的下限）。
2. 与「未融合的两个独立算子」逐位对比 `max_abs <= contract 容差`；不逐位必须给理由。
3. 端到端 step 时间**显著下降**（`|Δ| > 3×MAD`），不是单 kernel 快。
4. warp=64 规约正确：全 lane 参与、尾部 lane 不读垃圾值。

## 不适用场景

现有算子库已覆盖这个组合（先查 `libraries.md` 的匹配表，别重复造）；
只想调某个已有 kernel 的参数（走 `op-optimize`）；
compute-bound 主体做融合（净负，见 `libraries.md §3`）。
