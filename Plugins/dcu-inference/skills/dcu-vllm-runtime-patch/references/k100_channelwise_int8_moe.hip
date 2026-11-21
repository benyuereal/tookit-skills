// ============================================================================
// 实战 native INT8 MoE kernel（K100 AI / gfx928 / vLLM 0.18.1 / DTK 26.04 / Qwen3.6-35B-A3B W8A8）
// ----------------------------------------------------------------------------
// 来源：本地复现工程 /root/K100_QWEN36_REPRODUCTION_2026-08-07.gz
//        native_ext/k100_channelwise_int8_moe.hip
// 安全说明：纯 HIP/C++ kernel 源码，无密码 / 密钥 / 凭据。
// 功能：K100 专用 channelwise INT8 MoE（gate_up / down 两个 fused GEMM），
//       固定 template 参数吃满 Qwen3.6 decode 形状（gate_up N=512,K=2048；down N=2048,K=256）。
// 必读前提（离开 K100 AI / gfx928 / vLLM 0.18.1 / DTK 26.04 须重新验证）：
//   - 须配合 build_k100_channelwise_int8_moe.py 编译（其内 AITER 模板须 patch
//     group_size_n==1 为真 per-output-channel，否则数值错）。
//   - 该 .hip 不在本仓库编译，仅作源码归档与审计；编译产物 .so 由 build 脚本生成。
// 脱敏：本文件无内嵌绝对路径，无需脱敏。
// ============================================================================

#include <torch/extension.h>
#include <algorithm>
#include <hip/hip_runtime.h>

// aicc exposes this HCU-private builtin; hipcc does not. Standard HIP
// atomicAdd has the same FP32 global accumulation semantics required here.
#define __builtin_hcu_global_atomic_fadd_f32(ptr, value) atomicAdd((ptr), (value))
#include "moe_w8a8_block_wise_hip.h"

namespace {

int token_blocks(const torch::Tensor& sorted_token_ids, int size_m, int topk) {
    int64_t em = sorted_token_ids.size(0);
    if (size_m <= 16) {
        em = std::min<int64_t>(em, static_cast<int64_t>(size_m) * 16 * topk);
    }
    return static_cast<int>((em + 15) / 16);
}

void validate_common(
    const torch::Tensor& input,
    const torch::Tensor& a_scales,
    const torch::Tensor& output,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scales,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& expert_ids,
    const torch::Tensor& num_tokens_post_pad) {
    TORCH_CHECK(input.is_cuda(), "input must be on HCU/CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kInt8, "input must be int8");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bfloat16");
    TORCH_CHECK(a_scales.scalar_type() == torch::kFloat32, "activation scales must be float32");
    TORCH_CHECK(weight_scales.scalar_type() == torch::kFloat32, "weight scales must be float32");
    TORCH_CHECK(sorted_token_ids.scalar_type() == torch::kInt32, "sorted ids must be int32");
    TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert ids must be int32");
    TORCH_CHECK(num_tokens_post_pad.scalar_type() == torch::kInt32, "post-pad count must be int32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(a_scales.is_contiguous(), "activation scales must be contiguous");
    TORCH_CHECK(weight_scales.is_contiguous(), "weight scales must be contiguous");
}

}  // namespace

// Qwen3.6-35B TP2 gate/up: [M,2048] x [E,512,2048], top-k=8.
torch::Tensor k100_moe_gemm_gate_up(
    torch::Tensor input,
    torch::Tensor a_scales,
    torch::Tensor output,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad) {
    validate_common(input, a_scales, output, weight, weight_scales,
                    sorted_token_ids, expert_ids, num_tokens_post_pad);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    auto output_fp32 = torch::zeros(output.sizes(), output.options().dtype(torch::kFloat32));
    const int size_m = input.size(0);
    const int size_n = weight.size(1);
    const int size_k = input.size(1);
    TORCH_CHECK(size_n == 512 && size_k == 2048,
                "gate/up expects N=512,K=2048, got ", size_n, ",", size_k);
    run_moe_w8a8_gemm_block_wise<
        half, 8, 8, 16, 64, 128, false, false,
        1, 128, 1, 2, 1, true, 256>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<int8_t>()),
        a_scales.data_ptr<float>(),
        output_fp32.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(weight.data_ptr<int8_t>()),
        nullptr,
        weight_scales.data_ptr<float>(),
        nullptr,
        nullptr,
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tokens_post_pad.data_ptr<int32_t>(),
        token_blocks(sorted_token_ids, size_m, 8),
        size_m, size_n, size_k);
    output.copy_(output_fp32.to(torch::kBFloat16));
    return output;
}

// Qwen3.6-35B TP2 down: [M*8,256] x [E,2048,256], top-k=1.
torch::Tensor k100_moe_gemm_down(
    torch::Tensor input,
    torch::Tensor a_scales,
    torch::Tensor output,
    torch::Tensor weight,
    torch::Tensor weight_scales,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad) {
    validate_common(input, a_scales, output, weight, weight_scales,
                    sorted_token_ids, expert_ids, num_tokens_post_pad);
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
                "topk weights must be float32");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    auto output_fp32 = torch::zeros(output.sizes(), output.options().dtype(torch::kFloat32));
    const int size_m = input.size(0);
    const int size_n = weight.size(1);
    const int size_k = input.size(1);
    TORCH_CHECK(size_n == 2048 && size_k == 256,
                "down expects N=2048,K=256, got ", size_n, ",", size_k);
    run_moe_w8a8_gemm_block_wise<
        half, 8, 1, 16, 64, 128, false, true,
        1, 128, 1, 1, 1, true, 256>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<int8_t>()),
        a_scales.data_ptr<float>(),
        output_fp32.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(weight.data_ptr<int8_t>()),
        nullptr,
        weight_scales.data_ptr<float>(),
        nullptr,
        topk_weights.data_ptr<float>(),
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tokens_post_pad.data_ptr<int32_t>(),
        token_blocks(sorted_token_ids, size_m, 1),
        size_m, size_n, size_k);
    output.copy_(output_fp32.to(torch::kBFloat16));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_gate_up", &k100_moe_gemm_gate_up, "K100 channelwise INT8 MoE gate/up");
    m.def("gemm_down", &k100_moe_gemm_down, "K100 channelwise INT8 MoE down");
}
