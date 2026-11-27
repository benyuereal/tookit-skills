// ============================================================================
// 手写 HIP 版稀疏 MLA 前向（GLM5.3 flash 吸收式 MLA 档，H=64，split 结构）
//
// 形状（decode-glm53 / prefill-glm53）：
//   q   [1, S, H=64, DQK=512]  bf16   （无 rope 尾，tail_dim=0，d_qk==d_v==512）
//   kv  [1, SKV, G=1, 512]     bf16   （同一张量兼 K 与 V）
//   idx [1, S, G=1, topk=2048] int32   （-1 = 无效；继承 tilelang「-1 只垫队尾」语义）
//   out [1, S, H=64, 512]      bf16
//   sm_scale 传自然值 1/sqrt(512)，host 乘 log2(e) 后喂 kernel（kernel 用 exp2）。
//
// 结构：镜像 tilelang 对本档自动选的 config（本机 cu_count=70，H=64，
// block_I=32 / threads=256 / num_stages=0 / batch_head=1，num_split 随 S 变）：
//   S <= 16   : num_split=4 —— split 阶段 grid=(S*REPLICATE_H, 1, 4)，block=256；
//               REPLICATE_H=2（H=64/max_block_m=32），bx 编码 (s, hblk)，每 CTA 算
//               1 个 query token 的 32 个 head；bz=split_idx 交错取 KV 块（块号
//               i*128+z*32）；combine 阶段 grid=(S, H, 1)，block=128。
//   17 <= S <= 35 : num_split=2（decode-glm53 档 S=32 即此路）—— 同上，块号 i*64+z*32。
//   S >= 36   : no-split —— 单 kernel grid=(S*REPLICATE_H, 1, 1)，直接写 out
//               （含 prefill S=4096；与 tilelang 选路一致，split_invariance 已验）。
//   partial 中间量 [1, S, num_split, H, 512] fp16 + glse [1, S, num_split, H] fp16
//   （glse = log2(sumexp) + m_i*sm_scale，combine 用 exp2 归并）。
//
// 正确性依据：kernel 体逐字节来自 tilelang 对本机同形状生成并验证过的内核
// （_get_sparse_mla_fwd_kernel(64,512,0,2048,1,1/sqrt(512),32,256,0,{1,2,4},0,1) 的
// get_kernel_source() 导出），仅做三处替换：
//   (a) kernel 名 main_*_kernel(_1) -> sparse_mla_glm_*_kernel
//   (b) 烘焙的 sm_scale 常量 6.375872e-02f（=1/sqrt(512)*log2e）-> 形参 sm_scale
//   (c) C API 补 sm_scale 形参 + host dispatcher 按 S 分派
//
// 设备侧：include 真实 tilelang device 头（common/gemm/reduce/barrier）；copy.h
// 仍绕开（本机 DTK 缺 async builtins），只 verbatim vendor 用到的
// hcu_buffer_load / hcu_buffer_store / hcu_buffer_load_cache_swizzle 三个顶层
// 包装（Q 的加载走 cache_swizzle<1024> 描述符变体；底层原语在 common.h 引入的
// amd_buffer_addressing.hpp 里，与 DeepSeek 版同一套结论）。
// KV gather 用硬件 buffer_load（-1 哨兵 pred=false -> OOB 0-fill，免分支）。
//
// 编译（与 DeepSeek 版同一批 aicc + -mllvm flag，缺了慢 ~1.12x/CTA）：
//   /opt/dtk/bin/aicc -x hip --offload-arch=gfx938:sramecc+ -std=c++20 -O3 \
//     -I $TILELANG_HYGON_ROOT/src \
//     -mllvm=-enable-latency-hack=true -mllvm=-mmac-latency=5 \
//     -mllvm=-ds-load-store-latency=6 -mllvm=-disable-machine-sink=True \
//     -mllvm=-check-valu-data-forward-hazards=0 -mllvm=-disable-cluster-lds-memops=true \
//     -mllvm=-amdgpu-disable-backoff-barrier=false -mllvm=-support-768-vgprs=true \
//     -mllvm=-hcu-update-wait-by-reverse-search=true -mllvm=-hcu-pre-emit-load-store-opt=false \
//     -mllvm=-enable-hcu-approx-func-fp-math=true \
//     sparse_mla_hip_glm.hip -shared -fPIC -o libsparse_mla_hip_glm.so
//
// flag 说明（2026-10-29 同进程 A/B 实测，S=128/320k，pool=full）：
//   - -enable-hcu-approx-func-fp-math：tilelang pass_config 开了 TL_ENABLE_FAST_MATH，
//     其 aicc 后端就追加这个 flag —— 手写版之前漏了它，慢 1.16x/CTA（0.952 vs 0.810 ms，
//     与 tilelang 0.839 持平）。补上后输出与原 tilelang 路径逐位一致（exp2f 精度在 2e-3 门内）。
//   - WDRA（-vgpr-greedy-alloc-mode=local-wave）：同体 kernel 上 +1.2%（0.964 vs 0.952），
//     逐位一致但无净收益，**不要加**。
//
// 调用：hip_sparse_mla_fwd(Q, KV, Indices, Out, S, SKV, sm_scale, stream)
// ============================================================================

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

#include <tl_templates/hcu/common.h>
#include <tl_templates/hcu/gemm.h>
#include <tl_templates/hcu/reduce.h>
#include <tl_templates/hcu/barrier.h>

// ---- 顶层 hcu_buffer_load / hcu_buffer_store / hcu_buffer_load_cache_swizzle
//   （verbatim 抄自 copy.h 的三个本 kernel 用到的顶层包装）----
namespace tl {
template <typename T, index_t N, bool oob_conditional_check = true>
TL_DEVICE thread_buffer<T, N>
hcu_buffer_load(const T* p_src_wave, index_t src_thread_element_offset,
                bool src_thread_element_valid, index_t /*src_element_space_size*/) {
  const int32x4_t res = make_wave_buffer_resource(p_src_wave);
  index_t byte_off = [&]() {
    if constexpr (oob_conditional_check)
      return src_thread_element_valid ? src_thread_element_offset * sizeof(T) : 0xffffffff;
    else
      return src_thread_element_offset * sizeof(T);
  }();
  return hcu_buffer_load_impl<T, N>(res, byte_off, 0);
}

template <uint32_t CacheSwizzleStrideBytes, typename T, index_t N, bool oob_conditional_check = true>
TL_DEVICE thread_buffer<T, N> hcu_buffer_load_cache_swizzle(
    const T* p_src_wave, index_t src_thread_element_offset,
    bool src_thread_element_valid, index_t src_element_space_size) {
  const int32x4_t res = make_wave_buffer_resource_cache_swizzle(
      p_src_wave, src_element_space_size * sizeof(T), CacheSwizzleStrideBytes);
  index_t byte_off = [&]() {
    if constexpr (oob_conditional_check)
      return src_thread_element_valid ? src_thread_element_offset * sizeof(T) : 0xffffffff;
    else
      return src_thread_element_offset * sizeof(T);
  }();
  return hcu_buffer_load_impl<T, N>(res, byte_off, 0);
}

template <typename T, index_t N, bool oob_conditional_check = true>
TL_DEVICE void hcu_buffer_store(const thread_buffer<T, N>& src_thread_data, T* p_dst_wave,
                                index_t dst_thread_element_offset, bool dst_thread_element_valid,
                                index_t /*dst_element_space_size*/) {
  const int32x4_t res = make_wave_buffer_resource(p_dst_wave);
  index_t byte_off = [&]() {
    if constexpr (oob_conditional_check)
      return dst_thread_element_valid ? dst_thread_element_offset * sizeof(T) : 0xffffffff;
    else
      return dst_thread_element_offset * sizeof(T);
  }();
  hcu_buffer_store_impl<T, N>(src_thread_data, res, byte_off, 0);
}
} // namespace tl

extern "C" __global__ void __launch_bounds__(256) sparse_mla_glm_split4_kernel(int* __restrict__ Indices, bfloat16_t* __restrict__ KV, half_t* __restrict__ Output_partial, bfloat16_t* __restrict__ Q, half_t* __restrict__ glse, int batch, int batch_head, int seq_len, int seq_len_kv, float sm_scale) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* KV_spilt0_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_oshared0 = ((void*)((char*)buf_dyn_shmem + 0));
  void* KV_spilt1_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* acc_oshared1 = ((void*)((char*)buf_dyn_shmem + 8192));
  void* KV_spilt2_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 24576));
  int valid_NI[1];
  float sumexp[1];
  float acc_o[64];
  bfloat16_t Q_spilt0_shared[32];
  bfloat16_t Q_spilt1_shared[32];
  bfloat16_t Q_spilt2_shared[32];
  bfloat16_t Q_spilt3_shared[32];
  float acc_o0[16];
  float acc_o1[16];
  float acc_o2[16];
  float acc_o3[16];
  float sumexp_1[2];
  float m_i[2];
  half_t Output_partial_local_cast[8];
  int indices_mask[4];
  signed char mask[4];
  float acc_s[8];
  bfloat16_t KV_spilt0_local[16];
  bfloat16_t KV_spilt1_local[16];
  bfloat16_t KV_spilt2_local[16];
  bfloat16_t KV_spilt3_local[16];
  float m_i_prev[2];
  float m_i_clear[2];
  float alpha[2];
  float sumexp_i[2];
  bfloat16_t gemm2_kv_split0_local[16];
  bfloat16_t S_shared_local_cast_1[4];
  half_t acc_oshared0_local_cast_2[4];
  half_t acc_oshared1_local_cast_3[4];
  half_t acc_oshared0_local_cast_4[4];
  half_t acc_oshared1_local_cast_5[4];
  valid_NI[0] = 0;
  for (int i_i = 0; i_i < 16; ++i_i) {
    int first_indices = Indices[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32))];
    if (0 <= first_indices) {
      valid_NI[0] = (valid_NI[0] + 1);
    }
  }
  if (valid_NI[0] == 0) {
    sumexp[0] = -HUGE_VALF;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0.000000e+00f;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    bool pred = ((((int)threadIdx.x) >> 5) == 0);
    tl::hcu_buffer_store<half_t, 1, true>(tl::bit_cast<tl::thread_buffer<half_t, 1>>(((half_t)sumexp[0])), reinterpret_cast<half_t*>(glse), ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)256) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)256)) + (((int64_t)((int)blockIdx.z)) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) & (int64_t)31)), pred, ((seq_len * batch_head) * 256));
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __1;
        float4 v_ = *(float4*)(acc_o + ((i_1 * 8) + (vec * 4)));
        *((half_t*)(&(((half2*)(&(__1.x)))->x))) = (half_t)(v_.x);
        *((half_t*)(&(((half2*)(&(__1.x)))->y))) = (half_t)(v_.y);
        *((half_t*)(&(((half2*)(&(__1.y)))->x))) = (half_t)(v_.z);
        *((half_t*)(&(((half2*)(&(__1.y)))->y))) = (half_t)(v_.w);
        *(uint2*)(Output_partial_local_cast + (vec * 4)) = __1;
      }
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(Output_partial_local_cast + 0)), reinterpret_cast<half_t*>(Output_partial), (((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)131072) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)131072)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_1) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), true, ((seq_len * batch_head) * 131072));
    }
  } else {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(uint4*)(Q_spilt0_shared + (i_2 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), ((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      *(uint4*)(Q_spilt1_shared + (i_3 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_3) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_3) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)128), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      *(uint4*)(Q_spilt2_shared + (i_4 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_4) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)256), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(uint4*)(Q_spilt3_shared + (i_5 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_5) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_5) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)384), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float broadcast_var_1 = 0.000000e+00f;
      *(float4*)(acc_o0 + (i_6 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      float broadcast_var_2 = 0.000000e+00f;
      *(float4*)(acc_o1 + (i_7 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      float broadcast_var_3 = 0.000000e+00f;
      *(float4*)(acc_o2 + (i_8 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      float broadcast_var_4 = 0.000000e+00f;
      *(float4*)(acc_o3 + (i_9 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    float broadcast_var_5 = 1.000000e+00f;
    *(float2*)(sumexp_1 + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    float broadcast_var_6 = -1.073742e+09f;
    *(float2*)(m_i + 0) = make_float2(broadcast_var_6, broadcast_var_6);
    for (int i_i_1 = 0; i_i_1 < valid_NI[0]; ++i_i_1) {
      *(int4*)(indices_mask + 0) = *(int4*)&(tl::hcu_buffer_load<float, 4, false>(reinterpret_cast<const float*>(Indices), ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)4) * (int64_t)4)), true, ((seq_len * batch) * 2048)).get());
      int broadcast_var_7 = 0;
      int __2;
      ushort4 __3;
        int4 v__1 = make_int4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
        int4 v__2 = *(int4*)(indices_mask + 0);
        __3.x = (v__1.x<=v__2.x);
        __3.y = (v__1.y<=v__2.y);
        __3.z = (v__1.z<=v__2.z);
        __3.w = (v__1.w<=v__2.w);
      __2=((signed char)(__3.x) << 0);
      __2=__2 & ~(0x000000ff << 8) |((signed char)(__3.y) << 8);
      __2=__2 & ~(0x000000ff << 16) |((signed char)(__3.z) << 16);
      __2=__2 & ~(0x000000ff << 24) |((signed char)(__3.w) << 24);
      *(int*)(mask + 0) = __2;
      #pragma unroll
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        float condval;
        if (((bool)mask[(i_10 & 3)])) {
          condval = 0.000000e+00f;
        } else {
          condval = -HUGE_VALF;
        }
        acc_s[i_10] = condval;
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        bfloat16_t broadcast_var_8 = bfloat16_t(0.000000e+00f);
        bool pred_1 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
        *(uint4*)(KV_spilt0_local + (i_11 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), (((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), pred_1, ((seq_len_kv * batch) * 512)).get());
        bfloat16_t broadcast_var_9 = bfloat16_t(0.000000e+00f);
        bool pred_2 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
        *(uint4*)(KV_spilt1_local + (i_11 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)128), pred_2, ((seq_len_kv * batch) * 512)).get());
        bfloat16_t broadcast_var_10 = bfloat16_t(0.000000e+00f);
        bool pred_3 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
        *(uint4*)(KV_spilt2_local + (i_11 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)256), pred_3, ((seq_len_kv * batch) * 512)).get());
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        bfloat16_t broadcast_var_11 = bfloat16_t(0.000000e+00f);
        bool pred_4 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))]);
        *(uint4*)(KV_spilt3_local + (i_12 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + (((int64_t)i_12) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)384), pred_4, ((seq_len_kv * batch) * 512)).get());
      }
      __syncthreads();
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((i_13 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt0_local + (i_13 * 8));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        *(uint4*)(((bfloat16_t*)KV_spilt1_shared) + (((i_14 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt1_local + (i_14 * 8));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        *(uint4*)(((bfloat16_t*)KV_spilt2_shared) + (((i_15 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt2_local + (i_15 * 8));
      }
      {
        bfloat16_t B_local[8];
        __syncthreads();
        for (int ki = 0; ki < 2; ++ki) {
          *(uint4*)(B_local + 0) = *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
          for (int kp = 0; kp < 2; ++kp) {
            for (int i_16 = 0; i_16 < 2; ++i_16) {
              *(((float32x4*)acc_s) + i_16) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt0_shared) + (((ki * 4) + (i_16 * 2)) + kp)), *(((bfloat16x4_vec*)B_local) + kp), *(((float32x4*)acc_s) + i_16), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t B_local_1[8];
        for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
          *(uint4*)(B_local_1 + 0) = *(uint4*)(((bfloat16_t*)KV_spilt1_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki_1 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
          for (int kp_1 = 0; kp_1 < 2; ++kp_1) {
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              *(((float32x4*)acc_s) + i_17) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt1_shared) + (((ki_1 * 4) + (i_17 * 2)) + kp_1)), *(((bfloat16x4_vec*)B_local_1) + kp_1), *(((float32x4*)acc_s) + i_17), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t B_local_2[8];
        for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
          *(uint4*)(B_local_2 + 0) = *(uint4*)(((bfloat16_t*)KV_spilt2_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki_2 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
          for (int kp_2 = 0; kp_2 < 2; ++kp_2) {
            for (int i_18 = 0; i_18 < 2; ++i_18) {
              *(((float32x4*)acc_s) + i_18) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt2_shared) + (((ki_2 * 4) + (i_18 * 2)) + kp_2)), *(((bfloat16x4_vec*)B_local_2) + kp_2), *(((float32x4*)acc_s) + i_18), 1, 0);
            }
          }
        }
      }
      {
        for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
          for (int kp_3 = 0; kp_3 < 2; ++kp_3) {
            for (int i_19 = 0; i_19 < 2; ++i_19) {
              *(((float32x4*)acc_s) + i_19) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt3_shared) + (((ki_3 * 4) + (i_19 * 2)) + kp_3)), *(((bfloat16x4_vec*)KV_spilt3_local) + ((ki_3 * 2) + kp_3)), *(((float32x4*)acc_s) + i_19), 1, 0);
            }
          }
        }
      }
      *(float2*)(m_i_prev + 0) = *(float2*)(m_i + 0);
      __syncthreads();
      tl::AllReduce<tl::SumOp, 256, 128, 0, tl::SyncThreadsBarrier, 4, 256>::run_batch((&(acc_s[0])), (&(((float*)workspace_2)[0])));
      __syncthreads();
      tl::AllReduce<tl::SumOp, 256, 128, 0, tl::SyncThreadsBarrier, 4, 256>::run_batch((&(acc_s[4])), (&(((float*)workspace_2)[0])));
      __syncthreads();
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        m_i_clear[i_20] = -HUGE_VALF;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          m_i_clear[i_20] = max(m_i_clear[i_20], acc_s[((i_20 * 4) + rv)]);
        }
        m_i_clear[i_20] = tl::AllReduce<tl::MaxOp, 128, 16, 0>::run(m_i_clear[i_20], (&(((float*)workspace_1)[0])));
        m_i[i_20] = max(m_i[i_20], m_i_clear[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        alpha[i_21] = exp2f(((m_i_prev[i_21] - m_i[i_21]) * sm_scale));
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        acc_s[i_22] = exp2f(((acc_s[i_22] * sm_scale) - (m_i[(i_22 >> 2)] * sm_scale)));
      }
      __syncthreads();
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        sumexp_i[i_23] = 0.000000e+00f;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
          sumexp_i[i_23] = (sumexp_i[i_23] + acc_s[((i_23 * 4) + rv_1)]);
        }
        sumexp_i[i_23] = tl::AllReduce<tl::SumOp, 128, 16, 0>::run(sumexp_i[i_23], (&(((float*)workspace)[0])));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        sumexp_1[i_24] = ((sumexp_1[i_24] * alpha[i_24]) + sumexp_i[i_24]);
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 16; ++i_25) {
        acc_o0[i_25] = (acc_o0[i_25] * alpha[(i_25 >> 3)]);
        acc_o1[i_25] = (acc_o1[i_25] * alpha[(i_25 >> 3)]);
        acc_o2[i_25] = (acc_o2[i_25] * alpha[(i_25 >> 3)]);
        acc_o3[i_25] = (acc_o3[i_25] * alpha[(i_25 >> 3)]);
      }
      __syncthreads();
      #pragma unroll
      for (int i_26 = 0; i_26 < 16; ++i_26) {
        gemm2_kv_split0_local[i_26] = ((bfloat16_t*)KV_spilt0_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + ((i_26 & 7) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_26 >> 3) * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + (i_26 & 7))) * 8)) + (((int)threadIdx.x) & 7))];
      }
      if ((((int)threadIdx.x) >> 7) == 0) {
        #pragma unroll
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          uint2 __4;
          float4 v__3 = *(float4*)(acc_s + (i_27 * 4));
          ((bfloat16_t*)(&(__4.x)))[0] = (bfloat16_t)(v__3.x);
          ((bfloat16_t*)(&(__4.x)))[1] = (bfloat16_t)(v__3.y);
          ((bfloat16_t*)(&(__4.y)))[0] = (bfloat16_t)(v__3.z);
          ((bfloat16_t*)(&(__4.y)))[1] = (bfloat16_t)(v__3.w);
          *(uint2*)(S_shared_local_cast_1 + 0) = __4;
          *(uint2*)(((bfloat16_t*)S_shared) + ((((i_27 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 127) >> 5) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(S_shared_local_cast_1 + 0);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int i_28 = 0; i_28 < 2; ++i_28) {
        *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (i_28 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8))) = *(uint4*)(KV_spilt3_local + (i_28 * 8));
      }
      {
        bfloat16_t A_local[16];
        for (int i_29 = 0; i_29 < 2; ++i_29) {
          *(uint4*)(A_local + (i_29 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_29 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        for (int kp_4 = 0; kp_4 < 2; ++kp_4) {
          for (int i_30 = 0; i_30 < 2; ++i_30) {
            for (int j = 0; j < 2; ++j) {
              *(((float32x4*)acc_o0) + ((i_30 * 2) + j)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local) + ((i_30 * 2) + kp_4)), *(((bfloat16x4_vec*)gemm2_kv_split0_local) + ((j * 2) + kp_4)), *(((float32x4*)acc_o0) + ((i_30 * 2) + j)), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t A_local_1[16];
        bfloat16_t B_local_3[16];
        for (int i_31 = 0; i_31 < 2; ++i_31) {
          *(uint4*)(A_local_1 + (i_31 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_31 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          for (int local_id = 0; local_id < 8; ++local_id) {
            B_local_3[((j_1 * 8) + local_id)] = ((bfloat16_t*)KV_spilt1_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_1 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id)) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int kp_5 = 0; kp_5 < 2; ++kp_5) {
          for (int i_32 = 0; i_32 < 2; ++i_32) {
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              *(((float32x4*)acc_o1) + ((i_32 * 2) + j_2)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_1) + ((i_32 * 2) + kp_5)), *(((bfloat16x4_vec*)B_local_3) + ((j_2 * 2) + kp_5)), *(((float32x4*)acc_o1) + ((i_32 * 2) + j_2)), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t A_local_2[16];
        bfloat16_t B_local_4[16];
        for (int i_33 = 0; i_33 < 2; ++i_33) {
          *(uint4*)(A_local_2 + (i_33 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_33 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          for (int local_id_1 = 0; local_id_1 < 8; ++local_id_1) {
            B_local_4[((j_3 * 8) + local_id_1)] = ((bfloat16_t*)KV_spilt2_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id_1 * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_3 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id_1)) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int kp_6 = 0; kp_6 < 2; ++kp_6) {
          for (int i_34 = 0; i_34 < 2; ++i_34) {
            for (int j_4 = 0; j_4 < 2; ++j_4) {
              *(((float32x4*)acc_o2) + ((i_34 * 2) + j_4)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_2) + ((i_34 * 2) + kp_6)), *(((bfloat16x4_vec*)B_local_4) + ((j_4 * 2) + kp_6)), *(((float32x4*)acc_o2) + ((i_34 * 2) + j_4)), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t A_local_3[16];
        bfloat16_t B_local_5[16];
        for (int i_35 = 0; i_35 < 2; ++i_35) {
          *(uint4*)(A_local_3 + (i_35 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_35 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        __syncthreads();
        for (int j_5 = 0; j_5 < 2; ++j_5) {
          for (int local_id_2 = 0; local_id_2 < 8; ++local_id_2) {
            B_local_5[((j_5 * 8) + local_id_2)] = ((bfloat16_t*)KV_spilt0_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id_2 * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_5 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id_2)) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int kp_7 = 0; kp_7 < 2; ++kp_7) {
          for (int i_36 = 0; i_36 < 2; ++i_36) {
            for (int j_6 = 0; j_6 < 2; ++j_6) {
              *(((float32x4*)acc_o3) + ((i_36 * 2) + j_6)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_3) + ((i_36 * 2) + kp_7)), *(((bfloat16x4_vec*)B_local_5) + ((j_6 * 2) + kp_7)), *(((float32x4*)acc_o3) + ((i_36 * 2) + j_6)), 1, 0);
            }
          }
        }
      }
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 16; ++i_37) {
      acc_o0[i_37] = (acc_o0[i_37] / sumexp_1[(i_37 >> 3)]);
      acc_o1[i_37] = (acc_o1[i_37] / sumexp_1[(i_37 >> 3)]);
      acc_o2[i_37] = (acc_o2[i_37] / sumexp_1[(i_37 >> 3)]);
      acc_o3[i_37] = (acc_o3[i_37] / sumexp_1[(i_37 >> 3)]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 2; ++i_38) {
      sumexp_1[i_38] = (__log2f(sumexp_1[i_38]) + (m_i[i_38] * sm_scale));
    }
    __syncthreads();
    #pragma unroll
    for (int i_39 = 0; i_39 < 4; ++i_39) {
      uint2 __5;
      float4 v__4 = *(float4*)(acc_o0 + (i_39 * 4));
      *((half_t*)(&(((half2*)(&(__5.x)))->x))) = (half_t)(v__4.x);
      *((half_t*)(&(((half2*)(&(__5.x)))->y))) = (half_t)(v__4.y);
      *((half_t*)(&(((half2*)(&(__5.y)))->x))) = (half_t)(v__4.z);
      *((half_t*)(&(((half2*)(&(__5.y)))->y))) = (half_t)(v__4.w);
      *(uint2*)(acc_oshared0_local_cast_2 + 0) = __5;
      *(uint2*)(((half_t*)acc_oshared0) + (((((i_39 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_39 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared0_local_cast_2 + 0);
    }
    __syncthreads();
    #pragma unroll
    for (int i_40 = 0; i_40 < 4; ++i_40) {
      uint2 __6;
      float4 v__5 = *(float4*)(acc_o1 + (i_40 * 4));
      *((half_t*)(&(((half2*)(&(__6.x)))->x))) = (half_t)(v__5.x);
      *((half_t*)(&(((half2*)(&(__6.x)))->y))) = (half_t)(v__5.y);
      *((half_t*)(&(((half2*)(&(__6.y)))->x))) = (half_t)(v__5.z);
      *((half_t*)(&(((half2*)(&(__6.y)))->y))) = (half_t)(v__5.w);
      *(uint2*)(acc_oshared1_local_cast_3 + 0) = __6;
      *(uint2*)(((half_t*)acc_oshared1) + (((((i_40 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_40 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared1_local_cast_3 + 0);
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 2; ++i_41) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared0) + (((i_41 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), ((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)131072) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)131072)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_41) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), true, ((seq_len * batch_head) * 131072));
    }
    __syncthreads();
    #pragma unroll
    for (int i_42 = 0; i_42 < 2; ++i_42) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared1) + (((i_42 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)131072) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)131072)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_42) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)128), true, ((seq_len * batch_head) * 131072));
    }
    __syncthreads();
    #pragma unroll
    for (int i_43 = 0; i_43 < 4; ++i_43) {
      uint2 __7;
      float4 v__6 = *(float4*)(acc_o2 + (i_43 * 4));
      *((half_t*)(&(((half2*)(&(__7.x)))->x))) = (half_t)(v__6.x);
      *((half_t*)(&(((half2*)(&(__7.x)))->y))) = (half_t)(v__6.y);
      *((half_t*)(&(((half2*)(&(__7.y)))->x))) = (half_t)(v__6.z);
      *((half_t*)(&(((half2*)(&(__7.y)))->y))) = (half_t)(v__6.w);
      *(uint2*)(acc_oshared0_local_cast_4 + 0) = __7;
      *(uint2*)(((half_t*)acc_oshared0) + (((((i_43 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_43 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared0_local_cast_4 + 0);
    }
    __syncthreads();
    #pragma unroll
    for (int i_44 = 0; i_44 < 4; ++i_44) {
      uint2 __8;
      float4 v__7 = *(float4*)(acc_o3 + (i_44 * 4));
      *((half_t*)(&(((half2*)(&(__8.x)))->x))) = (half_t)(v__7.x);
      *((half_t*)(&(((half2*)(&(__8.x)))->y))) = (half_t)(v__7.y);
      *((half_t*)(&(((half2*)(&(__8.y)))->x))) = (half_t)(v__7.z);
      *((half_t*)(&(((half2*)(&(__8.y)))->y))) = (half_t)(v__7.w);
      *(uint2*)(acc_oshared1_local_cast_5 + 0) = __8;
      *(uint2*)(((half_t*)acc_oshared1) + (((((i_44 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_44 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared1_local_cast_5 + 0);
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared0) + (((i_45 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)131072) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)131072)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_45) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)256), true, ((seq_len * batch_head) * 131072));
    }
    __syncthreads();
    #pragma unroll
    for (int i_46 = 0; i_46 < 2; ++i_46) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared1) + (((i_46 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)131072) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)131072)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_46) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)384), true, ((seq_len * batch_head) * 131072));
    }
    if (((((((int)threadIdx.x) & 63) >> 4) * 4) + (((int)threadIdx.x) >> 6)) == 0) {
      #pragma unroll
      for (int i_47 = 0; i_47 < 2; ++i_47) {
        tl::hcu_buffer_store<half_t, 1, false>(tl::bit_cast<tl::thread_buffer<half_t, 1>>(((half_t)sumexp_1[i_47])), reinterpret_cast<half_t*>(glse), (((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)256) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)256)) + (((int64_t)((int)blockIdx.z)) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)32)) + (((int64_t)i_47) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15)), true, ((seq_len * batch_head) * 256));
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) sparse_mla_glm_combine4_kernel(bfloat16_t* __restrict__ Output, half_t* __restrict__ Output_partial, half_t* __restrict__ glse, int batch, int batch_head, int seq_len) {
  float lse_logsum_local[1];
  float o_accum_local[4];
  float lse_max_local[1];
  float lse_local_split[4];
  half_t po_local[4];
  float scale_local[1];
  bfloat16_t Output_local_cast_6[4];
  lse_logsum_local[0] = 0.000000e+00f;
  float broadcast_var = 0.000000e+00f;
  *(float4*)(o_accum_local + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  lse_max_local[0] = -HUGE_VALF;
  for (int k = 0; k < 4; ++k) {
    lse_local_split[k] = ((float)glse[((((((int64_t)((int)blockIdx.x)) * (int64_t)256) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)256)) + (((int64_t)k) * (int64_t)64)) + ((int64_t)((int)blockIdx.y)))]);
    lse_max_local[0] = max(lse_max_local[0], lse_local_split[k]);
  }
  for (int k_1 = 0; k_1 < 4; ++k_1) {
    lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[k_1] - lse_max_local[0])));
  }
  lse_logsum_local[0] = (__log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_2 = 0; k_2 < 4; ++k_2) {
    *(uint2*)(po_local + 0) = *(uint2*)&(tl::hcu_buffer_load<half_t, 4, false>(reinterpret_cast<const half_t*>(Output_partial), (((((((int64_t)((int)blockIdx.x)) * (int64_t)131072) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)131072)) + (((int64_t)k_2) * (int64_t)32768)) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), true, ((seq_len * batch_head) * 131072)).get());
    scale_local[0] = exp2f((lse_local_split[k_2] - lse_logsum_local[0]));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      o_accum_local[i] = (o_accum_local[i] + (((float)po_local[i]) * scale_local[0]));
    }
  }
  uint2 __1;
  float4 v_ = *(float4*)(o_accum_local + 0);
  ((bfloat16_t*)(&(__1.x)))[0] = (bfloat16_t)(v_.x);
  ((bfloat16_t*)(&(__1.x)))[1] = (bfloat16_t)(v_.y);
  ((bfloat16_t*)(&(__1.y)))[0] = (bfloat16_t)(v_.z);
  ((bfloat16_t*)(&(__1.y)))[1] = (bfloat16_t)(v_.w);
  *(uint2*)(Output_local_cast_6 + 0) = __1;
  tl::hcu_buffer_store<unsigned short, 4, false>(tl::bit_cast<tl::thread_buffer<unsigned short, 4>>(*(uint2*)(Output_local_cast_6 + 0)), reinterpret_cast<unsigned short*>(Output), ((((((int64_t)((int)blockIdx.x)) * (int64_t)32768) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)32768)) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), true, ((seq_len * batch) * 32768));
}


extern "C" __global__ void __launch_bounds__(256) sparse_mla_glm_split_kernel(int* __restrict__ Indices, bfloat16_t* __restrict__ KV, half_t* __restrict__ Output_partial, bfloat16_t* __restrict__ Q, half_t* __restrict__ glse, int batch, int batch_head, int seq_len, int seq_len_kv, float sm_scale) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* KV_spilt0_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_oshared0 = ((void*)((char*)buf_dyn_shmem + 0));
  void* KV_spilt1_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* acc_oshared1 = ((void*)((char*)buf_dyn_shmem + 8192));
  void* KV_spilt2_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 24576));
  int valid_NI[1];
  float sumexp[1];
  float acc_o[64];
  bfloat16_t Q_spilt0_shared[32];
  bfloat16_t Q_spilt1_shared[32];
  bfloat16_t Q_spilt2_shared[32];
  bfloat16_t Q_spilt3_shared[32];
  float acc_o0[16];
  float acc_o1[16];
  float acc_o2[16];
  float acc_o3[16];
  float sumexp_1[2];
  float m_i[2];
  half_t Output_partial_local_cast[8];
  int indices_mask[4];
  signed char mask[4];
  float acc_s[8];
  bfloat16_t KV_spilt0_local[16];
  bfloat16_t KV_spilt1_local[16];
  bfloat16_t KV_spilt2_local[16];
  bfloat16_t KV_spilt3_local[16];
  float m_i_prev[2];
  float m_i_clear[2];
  float alpha[2];
  float sumexp_i[2];
  bfloat16_t gemm2_kv_split0_local[16];
  bfloat16_t S_shared_local_cast_1[4];
  half_t acc_oshared0_local_cast_2[4];
  half_t acc_oshared1_local_cast_3[4];
  half_t acc_oshared0_local_cast_4[4];
  half_t acc_oshared1_local_cast_5[4];
  valid_NI[0] = 0;
  for (int i_i = 0; i_i < 32; ++i_i) {
    int first_indices = Indices[(((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32))];
    if (0 <= first_indices) {
      valid_NI[0] = (valid_NI[0] + 1);
    }
  }
  if (valid_NI[0] == 0) {
    sumexp[0] = -HUGE_VALF;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0.000000e+00f;
      *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    bool pred = ((((int)threadIdx.x) >> 5) == 0);
    tl::hcu_buffer_store<half_t, 1, true>(tl::bit_cast<tl::thread_buffer<half_t, 1>>(((half_t)sumexp[0])), reinterpret_cast<half_t*>(glse), ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)128) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) & (int64_t)31)), pred, ((seq_len * batch_head) * 128));
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __1;
        float4 v_ = *(float4*)(acc_o + ((i_1 * 8) + (vec * 4)));
        *((half_t*)(&(((half2*)(&(__1.x)))->x))) = (half_t)(v_.x);
        *((half_t*)(&(((half2*)(&(__1.x)))->y))) = (half_t)(v_.y);
        *((half_t*)(&(((half2*)(&(__1.y)))->x))) = (half_t)(v_.z);
        *((half_t*)(&(((half2*)(&(__1.y)))->y))) = (half_t)(v_.w);
        *(uint2*)(Output_partial_local_cast + (vec * 4)) = __1;
      }
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(Output_partial_local_cast + 0)), reinterpret_cast<half_t*>(Output_partial), (((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)65536)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_1) * (int64_t)2048)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), true, ((seq_len * batch_head) * 65536));
    }
  } else {
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(uint4*)(Q_spilt0_shared + (i_2 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), ((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      *(uint4*)(Q_spilt1_shared + (i_3 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_3) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_3) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)128), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      *(uint4*)(Q_spilt2_shared + (i_4 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_4) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)256), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(uint4*)(Q_spilt3_shared + (i_5 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_5) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_5) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)384), true, ((seq_len * batch) * 32768)).get());
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float broadcast_var_1 = 0.000000e+00f;
      *(float4*)(acc_o0 + (i_6 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      float broadcast_var_2 = 0.000000e+00f;
      *(float4*)(acc_o1 + (i_7 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      float broadcast_var_3 = 0.000000e+00f;
      *(float4*)(acc_o2 + (i_8 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      float broadcast_var_4 = 0.000000e+00f;
      *(float4*)(acc_o3 + (i_9 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    float broadcast_var_5 = 1.000000e+00f;
    *(float2*)(sumexp_1 + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    float broadcast_var_6 = -1.073742e+09f;
    *(float2*)(m_i + 0) = make_float2(broadcast_var_6, broadcast_var_6);
    for (int i_i_1 = 0; i_i_1 < valid_NI[0]; ++i_i_1) {
      *(int4*)(indices_mask + 0) = *(int4*)&(tl::hcu_buffer_load<float, 4, false>(reinterpret_cast<const float*>(Indices), ((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)4) * (int64_t)4)), true, ((seq_len * batch) * 2048)).get());
      int broadcast_var_7 = 0;
      int __2;
      ushort4 __3;
        int4 v__1 = make_int4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
        int4 v__2 = *(int4*)(indices_mask + 0);
        __3.x = (v__1.x<=v__2.x);
        __3.y = (v__1.y<=v__2.y);
        __3.z = (v__1.z<=v__2.z);
        __3.w = (v__1.w<=v__2.w);
      __2=((signed char)(__3.x) << 0);
      __2=__2 & ~(0x000000ff << 8) |((signed char)(__3.y) << 8);
      __2=__2 & ~(0x000000ff << 16) |((signed char)(__3.z) << 16);
      __2=__2 & ~(0x000000ff << 24) |((signed char)(__3.w) << 24);
      *(int*)(mask + 0) = __2;
      #pragma unroll
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        float condval;
        if (((bool)mask[(i_10 & 3)])) {
          condval = 0.000000e+00f;
        } else {
          condval = -HUGE_VALF;
        }
        acc_s[i_10] = condval;
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        bfloat16_t broadcast_var_8 = bfloat16_t(0.000000e+00f);
        bool pred_1 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
        *(uint4*)(KV_spilt0_local + (i_11 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), (((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), pred_1, ((seq_len_kv * batch) * 512)).get());
        bfloat16_t broadcast_var_9 = bfloat16_t(0.000000e+00f);
        bool pred_2 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
        *(uint4*)(KV_spilt1_local + (i_11 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)128), pred_2, ((seq_len_kv * batch) * 512)).get());
        bfloat16_t broadcast_var_10 = bfloat16_t(0.000000e+00f);
        bool pred_3 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
        *(uint4*)(KV_spilt2_local + (i_11 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((int64_t)i_11) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)256), pred_3, ((seq_len_kv * batch) * 512)).get());
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        bfloat16_t broadcast_var_11 = bfloat16_t(0.000000e+00f);
        bool pred_4 = (0 <= Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))]);
        *(uint4*)(KV_spilt3_local + (i_12 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((((int64_t)Indices[(((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)64)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + (((int64_t)i_12) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)384), pred_4, ((seq_len_kv * batch) * 512)).get());
      }
      __syncthreads();
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((i_13 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt0_local + (i_13 * 8));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        *(uint4*)(((bfloat16_t*)KV_spilt1_shared) + (((i_14 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt1_local + (i_14 * 8));
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        *(uint4*)(((bfloat16_t*)KV_spilt2_shared) + (((i_15 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt2_local + (i_15 * 8));
      }
      {
        bfloat16_t B_local[8];
        __syncthreads();
        for (int ki = 0; ki < 2; ++ki) {
          *(uint4*)(B_local + 0) = *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
          for (int kp = 0; kp < 2; ++kp) {
            for (int i_16 = 0; i_16 < 2; ++i_16) {
              *(((float32x4*)acc_s) + i_16) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt0_shared) + (((ki * 4) + (i_16 * 2)) + kp)), *(((bfloat16x4_vec*)B_local) + kp), *(((float32x4*)acc_s) + i_16), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t B_local_1[8];
        for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
          *(uint4*)(B_local_1 + 0) = *(uint4*)(((bfloat16_t*)KV_spilt1_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki_1 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
          for (int kp_1 = 0; kp_1 < 2; ++kp_1) {
            for (int i_17 = 0; i_17 < 2; ++i_17) {
              *(((float32x4*)acc_s) + i_17) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt1_shared) + (((ki_1 * 4) + (i_17 * 2)) + kp_1)), *(((bfloat16x4_vec*)B_local_1) + kp_1), *(((float32x4*)acc_s) + i_17), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t B_local_2[8];
        for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
          *(uint4*)(B_local_2 + 0) = *(uint4*)(((bfloat16_t*)KV_spilt2_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki_2 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
          for (int kp_2 = 0; kp_2 < 2; ++kp_2) {
            for (int i_18 = 0; i_18 < 2; ++i_18) {
              *(((float32x4*)acc_s) + i_18) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt2_shared) + (((ki_2 * 4) + (i_18 * 2)) + kp_2)), *(((bfloat16x4_vec*)B_local_2) + kp_2), *(((float32x4*)acc_s) + i_18), 1, 0);
            }
          }
        }
      }
      {
        for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
          for (int kp_3 = 0; kp_3 < 2; ++kp_3) {
            for (int i_19 = 0; i_19 < 2; ++i_19) {
              *(((float32x4*)acc_s) + i_19) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt3_shared) + (((ki_3 * 4) + (i_19 * 2)) + kp_3)), *(((bfloat16x4_vec*)KV_spilt3_local) + ((ki_3 * 2) + kp_3)), *(((float32x4*)acc_s) + i_19), 1, 0);
            }
          }
        }
      }
      *(float2*)(m_i_prev + 0) = *(float2*)(m_i + 0);
      __syncthreads();
      tl::AllReduce<tl::SumOp, 256, 128, 0, tl::SyncThreadsBarrier, 4, 256>::run_batch((&(acc_s[0])), (&(((float*)workspace_2)[0])));
      __syncthreads();
      tl::AllReduce<tl::SumOp, 256, 128, 0, tl::SyncThreadsBarrier, 4, 256>::run_batch((&(acc_s[4])), (&(((float*)workspace_2)[0])));
      __syncthreads();
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        m_i_clear[i_20] = -HUGE_VALF;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          m_i_clear[i_20] = max(m_i_clear[i_20], acc_s[((i_20 * 4) + rv)]);
        }
        m_i_clear[i_20] = tl::AllReduce<tl::MaxOp, 128, 16, 0>::run(m_i_clear[i_20], (&(((float*)workspace_1)[0])));
        m_i[i_20] = max(m_i[i_20], m_i_clear[i_20]);
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        alpha[i_21] = exp2f(((m_i_prev[i_21] - m_i[i_21]) * sm_scale));
      }
      #pragma unroll
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        acc_s[i_22] = exp2f(((acc_s[i_22] * sm_scale) - (m_i[(i_22 >> 2)] * sm_scale)));
      }
      __syncthreads();
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        sumexp_i[i_23] = 0.000000e+00f;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
          sumexp_i[i_23] = (sumexp_i[i_23] + acc_s[((i_23 * 4) + rv_1)]);
        }
        sumexp_i[i_23] = tl::AllReduce<tl::SumOp, 128, 16, 0>::run(sumexp_i[i_23], (&(((float*)workspace)[0])));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        sumexp_1[i_24] = ((sumexp_1[i_24] * alpha[i_24]) + sumexp_i[i_24]);
      }
      #pragma unroll
      for (int i_25 = 0; i_25 < 16; ++i_25) {
        acc_o0[i_25] = (acc_o0[i_25] * alpha[(i_25 >> 3)]);
        acc_o1[i_25] = (acc_o1[i_25] * alpha[(i_25 >> 3)]);
        acc_o2[i_25] = (acc_o2[i_25] * alpha[(i_25 >> 3)]);
        acc_o3[i_25] = (acc_o3[i_25] * alpha[(i_25 >> 3)]);
      }
      __syncthreads();
      #pragma unroll
      for (int i_26 = 0; i_26 < 16; ++i_26) {
        gemm2_kv_split0_local[i_26] = ((bfloat16_t*)KV_spilt0_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + ((i_26 & 7) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_26 >> 3) * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + (i_26 & 7))) * 8)) + (((int)threadIdx.x) & 7))];
      }
      if ((((int)threadIdx.x) >> 7) == 0) {
        #pragma unroll
        for (int i_27 = 0; i_27 < 2; ++i_27) {
          uint2 __4;
          float4 v__3 = *(float4*)(acc_s + (i_27 * 4));
          ((bfloat16_t*)(&(__4.x)))[0] = (bfloat16_t)(v__3.x);
          ((bfloat16_t*)(&(__4.x)))[1] = (bfloat16_t)(v__3.y);
          ((bfloat16_t*)(&(__4.y)))[0] = (bfloat16_t)(v__3.z);
          ((bfloat16_t*)(&(__4.y)))[1] = (bfloat16_t)(v__3.w);
          *(uint2*)(S_shared_local_cast_1 + 0) = __4;
          *(uint2*)(((bfloat16_t*)S_shared) + ((((i_27 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 127) >> 5) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(S_shared_local_cast_1 + 0);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int i_28 = 0; i_28 < 2; ++i_28) {
        *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (i_28 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8))) = *(uint4*)(KV_spilt3_local + (i_28 * 8));
      }
      {
        bfloat16_t A_local[16];
        for (int i_29 = 0; i_29 < 2; ++i_29) {
          *(uint4*)(A_local + (i_29 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_29 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        for (int kp_4 = 0; kp_4 < 2; ++kp_4) {
          for (int i_30 = 0; i_30 < 2; ++i_30) {
            for (int j = 0; j < 2; ++j) {
              *(((float32x4*)acc_o0) + ((i_30 * 2) + j)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local) + ((i_30 * 2) + kp_4)), *(((bfloat16x4_vec*)gemm2_kv_split0_local) + ((j * 2) + kp_4)), *(((float32x4*)acc_o0) + ((i_30 * 2) + j)), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t A_local_1[16];
        bfloat16_t B_local_3[16];
        for (int i_31 = 0; i_31 < 2; ++i_31) {
          *(uint4*)(A_local_1 + (i_31 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_31 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          for (int local_id = 0; local_id < 8; ++local_id) {
            B_local_3[((j_1 * 8) + local_id)] = ((bfloat16_t*)KV_spilt1_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_1 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id)) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int kp_5 = 0; kp_5 < 2; ++kp_5) {
          for (int i_32 = 0; i_32 < 2; ++i_32) {
            for (int j_2 = 0; j_2 < 2; ++j_2) {
              *(((float32x4*)acc_o1) + ((i_32 * 2) + j_2)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_1) + ((i_32 * 2) + kp_5)), *(((bfloat16x4_vec*)B_local_3) + ((j_2 * 2) + kp_5)), *(((float32x4*)acc_o1) + ((i_32 * 2) + j_2)), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t A_local_2[16];
        bfloat16_t B_local_4[16];
        for (int i_33 = 0; i_33 < 2; ++i_33) {
          *(uint4*)(A_local_2 + (i_33 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_33 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          for (int local_id_1 = 0; local_id_1 < 8; ++local_id_1) {
            B_local_4[((j_3 * 8) + local_id_1)] = ((bfloat16_t*)KV_spilt2_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id_1 * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_3 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id_1)) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int kp_6 = 0; kp_6 < 2; ++kp_6) {
          for (int i_34 = 0; i_34 < 2; ++i_34) {
            for (int j_4 = 0; j_4 < 2; ++j_4) {
              *(((float32x4*)acc_o2) + ((i_34 * 2) + j_4)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_2) + ((i_34 * 2) + kp_6)), *(((bfloat16x4_vec*)B_local_4) + ((j_4 * 2) + kp_6)), *(((float32x4*)acc_o2) + ((i_34 * 2) + j_4)), 1, 0);
            }
          }
        }
      }
      {
        bfloat16_t A_local_3[16];
        bfloat16_t B_local_5[16];
        for (int i_35 = 0; i_35 < 2; ++i_35) {
          *(uint4*)(A_local_3 + (i_35 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_35 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
        }
        __syncthreads();
        for (int j_5 = 0; j_5 < 2; ++j_5) {
          for (int local_id_2 = 0; local_id_2 < 8; ++local_id_2) {
            B_local_5[((j_5 * 8) + local_id_2)] = ((bfloat16_t*)KV_spilt0_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id_2 * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_5 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id_2)) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int kp_7 = 0; kp_7 < 2; ++kp_7) {
          for (int i_36 = 0; i_36 < 2; ++i_36) {
            for (int j_6 = 0; j_6 < 2; ++j_6) {
              *(((float32x4*)acc_o3) + ((i_36 * 2) + j_6)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_3) + ((i_36 * 2) + kp_7)), *(((bfloat16x4_vec*)B_local_5) + ((j_6 * 2) + kp_7)), *(((float32x4*)acc_o3) + ((i_36 * 2) + j_6)), 1, 0);
            }
          }
        }
      }
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 16; ++i_37) {
      acc_o0[i_37] = (acc_o0[i_37] / sumexp_1[(i_37 >> 3)]);
      acc_o1[i_37] = (acc_o1[i_37] / sumexp_1[(i_37 >> 3)]);
      acc_o2[i_37] = (acc_o2[i_37] / sumexp_1[(i_37 >> 3)]);
      acc_o3[i_37] = (acc_o3[i_37] / sumexp_1[(i_37 >> 3)]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 2; ++i_38) {
      sumexp_1[i_38] = (__log2f(sumexp_1[i_38]) + (m_i[i_38] * sm_scale));
    }
    __syncthreads();
    #pragma unroll
    for (int i_39 = 0; i_39 < 4; ++i_39) {
      uint2 __5;
      float4 v__4 = *(float4*)(acc_o0 + (i_39 * 4));
      *((half_t*)(&(((half2*)(&(__5.x)))->x))) = (half_t)(v__4.x);
      *((half_t*)(&(((half2*)(&(__5.x)))->y))) = (half_t)(v__4.y);
      *((half_t*)(&(((half2*)(&(__5.y)))->x))) = (half_t)(v__4.z);
      *((half_t*)(&(((half2*)(&(__5.y)))->y))) = (half_t)(v__4.w);
      *(uint2*)(acc_oshared0_local_cast_2 + 0) = __5;
      *(uint2*)(((half_t*)acc_oshared0) + (((((i_39 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_39 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared0_local_cast_2 + 0);
    }
    __syncthreads();
    #pragma unroll
    for (int i_40 = 0; i_40 < 4; ++i_40) {
      uint2 __6;
      float4 v__5 = *(float4*)(acc_o1 + (i_40 * 4));
      *((half_t*)(&(((half2*)(&(__6.x)))->x))) = (half_t)(v__5.x);
      *((half_t*)(&(((half2*)(&(__6.x)))->y))) = (half_t)(v__5.y);
      *((half_t*)(&(((half2*)(&(__6.y)))->x))) = (half_t)(v__5.z);
      *((half_t*)(&(((half2*)(&(__6.y)))->y))) = (half_t)(v__5.w);
      *(uint2*)(acc_oshared1_local_cast_3 + 0) = __6;
      *(uint2*)(((half_t*)acc_oshared1) + (((((i_40 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_40 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared1_local_cast_3 + 0);
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 2; ++i_41) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared0) + (((i_41 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), ((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)65536)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_41) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), true, ((seq_len * batch_head) * 65536));
    }
    __syncthreads();
    #pragma unroll
    for (int i_42 = 0; i_42 < 2; ++i_42) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared1) + (((i_42 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)65536)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_42) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)128), true, ((seq_len * batch_head) * 65536));
    }
    __syncthreads();
    #pragma unroll
    for (int i_43 = 0; i_43 < 4; ++i_43) {
      uint2 __7;
      float4 v__6 = *(float4*)(acc_o2 + (i_43 * 4));
      *((half_t*)(&(((half2*)(&(__7.x)))->x))) = (half_t)(v__6.x);
      *((half_t*)(&(((half2*)(&(__7.x)))->y))) = (half_t)(v__6.y);
      *((half_t*)(&(((half2*)(&(__7.y)))->x))) = (half_t)(v__6.z);
      *((half_t*)(&(((half2*)(&(__7.y)))->y))) = (half_t)(v__6.w);
      *(uint2*)(acc_oshared0_local_cast_4 + 0) = __7;
      *(uint2*)(((half_t*)acc_oshared0) + (((((i_43 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_43 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared0_local_cast_4 + 0);
    }
    __syncthreads();
    #pragma unroll
    for (int i_44 = 0; i_44 < 4; ++i_44) {
      uint2 __8;
      float4 v__7 = *(float4*)(acc_o3 + (i_44 * 4));
      *((half_t*)(&(((half2*)(&(__8.x)))->x))) = (half_t)(v__7.x);
      *((half_t*)(&(((half2*)(&(__8.x)))->y))) = (half_t)(v__7.y);
      *((half_t*)(&(((half2*)(&(__8.y)))->x))) = (half_t)(v__7.z);
      *((half_t*)(&(((half2*)(&(__8.y)))->y))) = (half_t)(v__7.w);
      *(uint2*)(acc_oshared1_local_cast_5 + 0) = __8;
      *(uint2*)(((half_t*)acc_oshared1) + (((((i_44 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_44 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared1_local_cast_5 + 0);
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared0) + (((i_45 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)65536)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_45) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)256), true, ((seq_len * batch_head) * 65536));
    }
    __syncthreads();
    #pragma unroll
    for (int i_46 = 0; i_46 < 2; ++i_46) {
      tl::hcu_buffer_store<half_t, 8, false>(tl::bit_cast<tl::thread_buffer<half_t, 8>>(*(uint4*)(((half_t*)acc_oshared1) + (((i_46 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<half_t*>(Output_partial), (((((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)65536)) + (((int64_t)((int)blockIdx.z)) * (int64_t)32768)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)16384)) + (((int64_t)i_46) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)384), true, ((seq_len * batch_head) * 65536));
    }
    if (((((((int)threadIdx.x) & 63) >> 4) * 4) + (((int)threadIdx.x) >> 6)) == 0) {
      #pragma unroll
      for (int i_47 = 0; i_47 < 2; ++i_47) {
        tl::hcu_buffer_store<half_t, 1, false>(tl::bit_cast<tl::thread_buffer<half_t, 1>>(((half_t)sumexp_1[i_47])), reinterpret_cast<half_t*>(glse), (((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)128) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)128)) + (((int64_t)((int)blockIdx.z)) * (int64_t)64)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)1) * (int64_t)32)) + (((int64_t)i_47) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15)), true, ((seq_len * batch_head) * 128));
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) sparse_mla_glm_combine_kernel(bfloat16_t* __restrict__ Output, half_t* __restrict__ Output_partial, half_t* __restrict__ glse, int batch, int batch_head, int seq_len) {
  float lse_logsum_local[1];
  float o_accum_local[4];
  float lse_max_local[1];
  float lse_local_split[2];
  half_t po_local[4];
  float scale_local[1];
  bfloat16_t Output_local_cast_6[4];
  lse_logsum_local[0] = 0.000000e+00f;
  float broadcast_var = 0.000000e+00f;
  *(float4*)(o_accum_local + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  lse_max_local[0] = -HUGE_VALF;
  for (int k = 0; k < 2; ++k) {
    lse_local_split[k] = ((float)glse[((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)128)) + (((int64_t)k) * (int64_t)64)) + ((int64_t)((int)blockIdx.y)))]);
    lse_max_local[0] = max(lse_max_local[0], lse_local_split[k]);
  }
  for (int k_1 = 0; k_1 < 2; ++k_1) {
    lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[k_1] - lse_max_local[0])));
  }
  lse_logsum_local[0] = (__log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_2 = 0; k_2 < 2; ++k_2) {
    *(uint2*)(po_local + 0) = *(uint2*)&(tl::hcu_buffer_load<half_t, 4, false>(reinterpret_cast<const half_t*>(Output_partial), (((((((int64_t)((int)blockIdx.x)) * (int64_t)65536) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)65536)) + (((int64_t)k_2) * (int64_t)32768)) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), true, ((seq_len * batch_head) * 65536)).get());
    scale_local[0] = exp2f((lse_local_split[k_2] - lse_logsum_local[0]));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      o_accum_local[i] = (o_accum_local[i] + (((float)po_local[i]) * scale_local[0]));
    }
  }
  uint2 __1;
  float4 v_ = *(float4*)(o_accum_local + 0);
  ((bfloat16_t*)(&(__1.x)))[0] = (bfloat16_t)(v_.x);
  ((bfloat16_t*)(&(__1.x)))[1] = (bfloat16_t)(v_.y);
  ((bfloat16_t*)(&(__1.y)))[0] = (bfloat16_t)(v_.z);
  ((bfloat16_t*)(&(__1.y)))[1] = (bfloat16_t)(v_.w);
  *(uint2*)(Output_local_cast_6 + 0) = __1;
  tl::hcu_buffer_store<unsigned short, 4, false>(tl::bit_cast<tl::thread_buffer<unsigned short, 4>>(*(uint2*)(Output_local_cast_6 + 0)), reinterpret_cast<unsigned short*>(Output), ((((((int64_t)((int)blockIdx.x)) * (int64_t)32768) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_len)) * (int64_t)32768)) + (((int64_t)((int)blockIdx.y)) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), true, ((seq_len * batch) * 32768));
}


extern "C" __global__ void __launch_bounds__(256) sparse_mla_glm_nosplit_kernel(int* __restrict__ Indices, bfloat16_t* __restrict__ KV, bfloat16_t* __restrict__ Output, bfloat16_t* __restrict__ Q, int batch, int seq_len, int seq_len_kv, float sm_scale) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* KV_spilt0_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_oshared0 = ((void*)((char*)buf_dyn_shmem + 0));
  void* KV_spilt1_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* acc_oshared1 = ((void*)((char*)buf_dyn_shmem + 8192));
  void* KV_spilt2_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 24576));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 24576));
  bfloat16_t Q_spilt0_shared[32];
  bfloat16_t Q_spilt1_shared[32];
  bfloat16_t Q_spilt2_shared[32];
  bfloat16_t Q_spilt3_shared[32];
  float acc_o0[16];
  float acc_o1[16];
  float acc_o2[16];
  float acc_o3[16];
  float sumexp[2];
  float m_i[2];
  int valid_NI[1];
  int indices_mask[4];
  signed char mask[4];
  float acc_s[8];
  bfloat16_t KV_spilt0_local[16];
  bfloat16_t KV_spilt1_local[16];
  bfloat16_t KV_spilt2_local[16];
  bfloat16_t KV_spilt3_local[16];
  float m_i_prev[2];
  float m_i_clear[2];
  float alpha[2];
  float sumexp_i[2];
  bfloat16_t gemm2_kv_split0_local[16];
  bfloat16_t S_shared_local_cast[4];
  bfloat16_t acc_oshared0_local_cast_1[4];
  bfloat16_t acc_oshared1_local_cast_2[4];
  bfloat16_t acc_oshared0_local_cast_3[4];
  bfloat16_t acc_oshared1_local_cast_4[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(Q_spilt0_shared + (i * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), ((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)), true, ((seq_len * batch) * 32768)).get());
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(Q_spilt1_shared + (i_1 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_1) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_1) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)128), true, ((seq_len * batch) * 32768)).get());
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    *(uint4*)(Q_spilt2_shared + (i_2 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)256), true, ((seq_len * batch) * 32768)).get());
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    *(uint4*)(Q_spilt3_shared + (i_3 * 8)) = *(uint4*)&(tl::hcu_buffer_load_cache_swizzle<1024, unsigned short, 8, false>(reinterpret_cast<const unsigned short*>(Q), (((((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + ((((int64_t)i_3) & (int64_t)1) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_3) >> (int64_t)1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)384), true, ((seq_len * batch) * 32768)).get());
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(acc_o0 + (i_4 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    float broadcast_var_1 = 0.000000e+00f;
    *(float4*)(acc_o1 + (i_5 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    float broadcast_var_2 = 0.000000e+00f;
    *(float4*)(acc_o2 + (i_6 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    float broadcast_var_3 = 0.000000e+00f;
    *(float4*)(acc_o3 + (i_7 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  float broadcast_var_4 = 1.000000e+00f;
  *(float2*)(sumexp + 0) = make_float2(broadcast_var_4, broadcast_var_4);
  float broadcast_var_5 = -1.073742e+09f;
  *(float2*)(m_i + 0) = make_float2(broadcast_var_5, broadcast_var_5);
  valid_NI[0] = 0;
  for (int i_i = 0; i_i < 64; ++i_i) {
    int first_indices = Indices[((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i) * (int64_t)32))];
    if (0 <= first_indices) {
      valid_NI[0] = (valid_NI[0] + 1);
    }
  }
  for (int i_i_1 = 0; i_i_1 < valid_NI[0]; ++i_i_1) {
    *(int4*)(indices_mask + 0) = *(int4*)&(tl::hcu_buffer_load<float, 4, false>(reinterpret_cast<const float*>(Indices), (((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)4) * (int64_t)4)), true, ((seq_len * batch) * 2048)).get());
    int broadcast_var_6 = 0;
    int __1;
    ushort4 __2;
      int4 v_ = make_int4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
      int4 v__1 = *(int4*)(indices_mask + 0);
      __2.x = (v_.x<=v__1.x);
      __2.y = (v_.y<=v__1.y);
      __2.z = (v_.z<=v__1.z);
      __2.w = (v_.w<=v__1.w);
    __1=((signed char)(__2.x) << 0);
    __1=__1 & ~(0x000000ff << 8) |((signed char)(__2.y) << 8);
    __1=__1 & ~(0x000000ff << 16) |((signed char)(__2.z) << 16);
    __1=__1 & ~(0x000000ff << 24) |((signed char)(__2.w) << 24);
    *(int*)(mask + 0) = __1;
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      float condval;
      if (((bool)mask[(i_8 & 3)])) {
        condval = 0.000000e+00f;
      } else {
        condval = -HUGE_VALF;
      }
      acc_s[i_8] = condval;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      bfloat16_t broadcast_var_7 = bfloat16_t(0.000000e+00f);
      bool pred = (0 <= Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((int64_t)i_9) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
      *(uint4*)(KV_spilt0_local + (i_9 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), (((((int64_t)Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((int64_t)i_9) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), pred, ((seq_len_kv * batch) * 512)).get());
      bfloat16_t broadcast_var_8 = bfloat16_t(0.000000e+00f);
      bool pred_1 = (0 <= Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((int64_t)i_9) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
      *(uint4*)(KV_spilt1_local + (i_9 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((int64_t)Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((int64_t)i_9) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)128), pred_1, ((seq_len_kv * batch) * 512)).get());
      bfloat16_t broadcast_var_9 = bfloat16_t(0.000000e+00f);
      bool pred_2 = (0 <= Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((int64_t)i_9) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]);
      *(uint4*)(KV_spilt2_local + (i_9 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((int64_t)Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((int64_t)i_9) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)256), pred_2, ((seq_len_kv * batch) * 512)).get());
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      bfloat16_t broadcast_var_10 = bfloat16_t(0.000000e+00f);
      bool pred_3 = (0 <= Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))]);
      *(uint4*)(KV_spilt3_local + (i_10 * 8)) = *(uint4*)&(tl::hcu_buffer_load<unsigned short, 8, true>(reinterpret_cast<const unsigned short*>(KV), ((((((((int64_t)Indices[((((((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)2048)) + (((int64_t)i_i_1) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)6) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))]) * (int64_t)512) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + (((int64_t)i_10) * (int64_t)32)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)8)) + (int64_t)384), pred_3, ((seq_len_kv * batch) * 512)).get());
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((i_11 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt0_local + (i_11 * 8));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      *(uint4*)(((bfloat16_t*)KV_spilt1_shared) + (((i_12 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt1_local + (i_12 * 8));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      *(uint4*)(((bfloat16_t*)KV_spilt2_shared) + (((i_13 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8))) = *(uint4*)(KV_spilt2_local + (i_13 * 8));
    }
    {
      bfloat16_t B_local[8];
      __syncthreads();
      for (int ki = 0; ki < 2; ++ki) {
        *(uint4*)(B_local + 0) = *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
        for (int kp = 0; kp < 2; ++kp) {
          for (int i_14 = 0; i_14 < 2; ++i_14) {
            *(((float32x4*)acc_s) + i_14) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt0_shared) + (((ki * 4) + (i_14 * 2)) + kp)), *(((bfloat16x4_vec*)B_local) + kp), *(((float32x4*)acc_s) + i_14), 1, 0);
          }
        }
      }
    }
    {
      bfloat16_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        *(uint4*)(B_local_1 + 0) = *(uint4*)(((bfloat16_t*)KV_spilt1_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki_1 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
        for (int kp_1 = 0; kp_1 < 2; ++kp_1) {
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            *(((float32x4*)acc_s) + i_15) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt1_shared) + (((ki_1 * 4) + (i_15 * 2)) + kp_1)), *(((bfloat16x4_vec*)B_local_1) + kp_1), *(((float32x4*)acc_s) + i_15), 1, 0);
          }
        }
      }
    }
    {
      bfloat16_t B_local_2[8];
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        *(uint4*)(B_local_2 + 0) = *(uint4*)(((bfloat16_t*)KV_spilt2_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (ki_2 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8)));
        for (int kp_2 = 0; kp_2 < 2; ++kp_2) {
          for (int i_16 = 0; i_16 < 2; ++i_16) {
            *(((float32x4*)acc_s) + i_16) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt2_shared) + (((ki_2 * 4) + (i_16 * 2)) + kp_2)), *(((bfloat16x4_vec*)B_local_2) + kp_2), *(((float32x4*)acc_s) + i_16), 1, 0);
          }
        }
      }
    }
    {
      for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
        for (int kp_3 = 0; kp_3 < 2; ++kp_3) {
          for (int i_17 = 0; i_17 < 2; ++i_17) {
            *(((float32x4*)acc_s) + i_17) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)Q_spilt3_shared) + (((ki_3 * 4) + (i_17 * 2)) + kp_3)), *(((bfloat16x4_vec*)KV_spilt3_local) + ((ki_3 * 2) + kp_3)), *(((float32x4*)acc_s) + i_17), 1, 0);
          }
        }
      }
    }
    *(float2*)(m_i_prev + 0) = *(float2*)(m_i + 0);
    __syncthreads();
    tl::AllReduce<tl::SumOp, 256, 128, 0, tl::SyncThreadsBarrier, 4, 256>::run_batch((&(acc_s[0])), (&(((float*)workspace_2)[0])));
    __syncthreads();
    tl::AllReduce<tl::SumOp, 256, 128, 0, tl::SyncThreadsBarrier, 4, 256>::run_batch((&(acc_s[4])), (&(((float*)workspace_2)[0])));
    __syncthreads();
    if ((i_i_1 % 32) == 0) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        m_i_clear[i_18] = -HUGE_VALF;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          m_i_clear[i_18] = max(m_i_clear[i_18], acc_s[((i_18 * 4) + rv)]);
        }
        m_i_clear[i_18] = tl::AllReduce<tl::MaxOp, 128, 16, 0>::run(m_i_clear[i_18], (&(((float*)workspace_1)[0])));
        m_i[i_18] = max(m_i[i_18], m_i_clear[i_18]);
      }
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      alpha[i_19] = exp2f(((m_i_prev[i_19] - m_i[i_19]) * sm_scale));
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      acc_s[i_20] = exp2f(((acc_s[i_20] * sm_scale) - (m_i[(i_20 >> 2)] * sm_scale)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      sumexp_i[i_21] = 0.000000e+00f;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        sumexp_i[i_21] = (sumexp_i[i_21] + acc_s[((i_21 * 4) + rv_1)]);
      }
      sumexp_i[i_21] = tl::AllReduce<tl::SumOp, 128, 16, 0>::run(sumexp_i[i_21], (&(((float*)workspace)[0])));
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      sumexp[i_22] = ((sumexp[i_22] * alpha[i_22]) + sumexp_i[i_22]);
    }
    if ((i_i_1 % 32) == 0) {
      #pragma unroll
      for (int i_23 = 0; i_23 < 16; ++i_23) {
        acc_o0[i_23] = (acc_o0[i_23] * alpha[(i_23 >> 3)]);
        acc_o1[i_23] = (acc_o1[i_23] * alpha[(i_23 >> 3)]);
        acc_o2[i_23] = (acc_o2[i_23] * alpha[(i_23 >> 3)]);
        acc_o3[i_23] = (acc_o3[i_23] * alpha[(i_23 >> 3)]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_24 = 0; i_24 < 16; ++i_24) {
      gemm2_kv_split0_local[i_24] = ((bfloat16_t*)KV_spilt0_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + ((i_24 & 7) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_24 >> 3) * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + (i_24 & 7))) * 8)) + (((int)threadIdx.x) & 7))];
    }
    if ((((int)threadIdx.x) >> 7) == 0) {
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        uint2 __3;
        float4 v__2 = *(float4*)(acc_s + (i_25 * 4));
        ((bfloat16_t*)(&(__3.x)))[0] = (bfloat16_t)(v__2.x);
        ((bfloat16_t*)(&(__3.x)))[1] = (bfloat16_t)(v__2.y);
        ((bfloat16_t*)(&(__3.y)))[0] = (bfloat16_t)(v__2.z);
        ((bfloat16_t*)(&(__3.y)))[1] = (bfloat16_t)(v__2.w);
        *(uint2*)(S_shared_local_cast + 0) = __3;
        *(uint2*)(((bfloat16_t*)S_shared) + ((((i_25 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 127) >> 5) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(S_shared_local_cast + 0);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      *(uint4*)(((bfloat16_t*)KV_spilt0_shared) + (((((((int)threadIdx.x) & 127) >> 6) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 7) * 8) + (i_26 * 4)) + ((((int)threadIdx.x) & 63) >> 4)) ^ (((int)threadIdx.x) & 15)) * 8))) = *(uint4*)(KV_spilt3_local + (i_26 * 8));
    }
    {
      bfloat16_t A_local[16];
      for (int i_27 = 0; i_27 < 2; ++i_27) {
        *(uint4*)(A_local + (i_27 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_27 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
      }
      for (int kp_4 = 0; kp_4 < 2; ++kp_4) {
        for (int i_28 = 0; i_28 < 2; ++i_28) {
          for (int j = 0; j < 2; ++j) {
            *(((float32x4*)acc_o0) + ((i_28 * 2) + j)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local) + ((i_28 * 2) + kp_4)), *(((bfloat16x4_vec*)gemm2_kv_split0_local) + ((j * 2) + kp_4)), *(((float32x4*)acc_o0) + ((i_28 * 2) + j)), 1, 0);
          }
        }
      }
    }
    {
      bfloat16_t A_local_1[16];
      bfloat16_t B_local_3[16];
      for (int i_29 = 0; i_29 < 2; ++i_29) {
        *(uint4*)(A_local_1 + (i_29 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_29 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
      }
      for (int j_1 = 0; j_1 < 2; ++j_1) {
        for (int local_id = 0; local_id < 8; ++local_id) {
          B_local_3[((j_1 * 8) + local_id)] = ((bfloat16_t*)KV_spilt1_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_1 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id)) * 8)) + (((int)threadIdx.x) & 7))];
        }
      }
      for (int kp_5 = 0; kp_5 < 2; ++kp_5) {
        for (int i_30 = 0; i_30 < 2; ++i_30) {
          for (int j_2 = 0; j_2 < 2; ++j_2) {
            *(((float32x4*)acc_o1) + ((i_30 * 2) + j_2)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_1) + ((i_30 * 2) + kp_5)), *(((bfloat16x4_vec*)B_local_3) + ((j_2 * 2) + kp_5)), *(((float32x4*)acc_o1) + ((i_30 * 2) + j_2)), 1, 0);
          }
        }
      }
    }
    {
      bfloat16_t A_local_2[16];
      bfloat16_t B_local_4[16];
      for (int i_31 = 0; i_31 < 2; ++i_31) {
        *(uint4*)(A_local_2 + (i_31 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_31 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
      }
      for (int j_3 = 0; j_3 < 2; ++j_3) {
        for (int local_id_1 = 0; local_id_1 < 8; ++local_id_1) {
          B_local_4[((j_3 * 8) + local_id_1)] = ((bfloat16_t*)KV_spilt2_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id_1 * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_3 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id_1)) * 8)) + (((int)threadIdx.x) & 7))];
        }
      }
      for (int kp_6 = 0; kp_6 < 2; ++kp_6) {
        for (int i_32 = 0; i_32 < 2; ++i_32) {
          for (int j_4 = 0; j_4 < 2; ++j_4) {
            *(((float32x4*)acc_o2) + ((i_32 * 2) + j_4)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_2) + ((i_32 * 2) + kp_6)), *(((bfloat16x4_vec*)B_local_4) + ((j_4 * 2) + kp_6)), *(((float32x4*)acc_o2) + ((i_32 * 2) + j_4)), 1, 0);
          }
        }
      }
    }
    {
      bfloat16_t A_local_3[16];
      bfloat16_t B_local_5[16];
      for (int i_33 = 0; i_33 < 2; ++i_33) {
        *(uint4*)(A_local_3 + (i_33 * 8)) = *(uint4*)(((bfloat16_t*)S_shared) + (((i_33 * 512) + ((((int)threadIdx.x) & 15) * 32)) + ((((((int)threadIdx.x) & 63) >> 4) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)));
      }
      __syncthreads();
      for (int j_5 = 0; j_5 < 2; ++j_5) {
        for (int local_id_2 = 0; local_id_2 < 8; ++local_id_2) {
          B_local_5[((j_5 * 8) + local_id_2)] = ((bfloat16_t*)KV_spilt0_shared)[((((((((int)threadIdx.x) & 63) >> 4) * 1024) + (local_id_2 * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + (j_5 * 2)) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((((int)threadIdx.x) & 31) >> 4) * 8) + local_id_2)) * 8)) + (((int)threadIdx.x) & 7))];
        }
      }
      for (int kp_7 = 0; kp_7 < 2; ++kp_7) {
        for (int i_34 = 0; i_34 < 2; ++i_34) {
          for (int j_6 = 0; j_6 < 2; ++j_6) {
            *(((float32x4*)acc_o3) + ((i_34 * 2) + j_6)) = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(((bfloat16x4_vec*)A_local_3) + ((i_34 * 2) + kp_7)), *(((bfloat16x4_vec*)B_local_5) + ((j_6 * 2) + kp_7)), *(((float32x4*)acc_o3) + ((i_34 * 2) + j_6)), 1, 0);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_35 = 0; i_35 < 16; ++i_35) {
    acc_o0[i_35] = (acc_o0[i_35] / sumexp[(i_35 >> 3)]);
    acc_o1[i_35] = (acc_o1[i_35] / sumexp[(i_35 >> 3)]);
    acc_o2[i_35] = (acc_o2[i_35] / sumexp[(i_35 >> 3)]);
    acc_o3[i_35] = (acc_o3[i_35] / sumexp[(i_35 >> 3)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_36 = 0; i_36 < 4; ++i_36) {
    uint2 __4;
    float4 v__3 = *(float4*)(acc_o0 + (i_36 * 4));
    ((bfloat16_t*)(&(__4.x)))[0] = (bfloat16_t)(v__3.x);
    ((bfloat16_t*)(&(__4.x)))[1] = (bfloat16_t)(v__3.y);
    ((bfloat16_t*)(&(__4.y)))[0] = (bfloat16_t)(v__3.z);
    ((bfloat16_t*)(&(__4.y)))[1] = (bfloat16_t)(v__3.w);
    *(uint2*)(acc_oshared0_local_cast_1 + 0) = __4;
    *(uint2*)(((bfloat16_t*)acc_oshared0) + (((((i_36 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_36 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared0_local_cast_1 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_37 = 0; i_37 < 4; ++i_37) {
    uint2 __5;
    float4 v__4 = *(float4*)(acc_o1 + (i_37 * 4));
    ((bfloat16_t*)(&(__5.x)))[0] = (bfloat16_t)(v__4.x);
    ((bfloat16_t*)(&(__5.x)))[1] = (bfloat16_t)(v__4.y);
    ((bfloat16_t*)(&(__5.y)))[0] = (bfloat16_t)(v__4.z);
    ((bfloat16_t*)(&(__5.y)))[1] = (bfloat16_t)(v__4.w);
    *(uint2*)(acc_oshared1_local_cast_2 + 0) = __5;
    *(uint2*)(((bfloat16_t*)acc_oshared1) + (((((i_37 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_37 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared1_local_cast_2 + 0);
  }
  #pragma unroll
  for (int i_38 = 0; i_38 < 2; ++i_38) {
    tl::hcu_buffer_store<unsigned short, 8, false>(tl::bit_cast<tl::thread_buffer<unsigned short, 8>>(*(uint4*)(((bfloat16_t*)acc_oshared0) + (((i_38 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<unsigned short*>(Output), ((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((int64_t)i_38) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), true, ((seq_len * batch) * 32768));
  }
  __syncthreads();
  #pragma unroll
  for (int i_39 = 0; i_39 < 2; ++i_39) {
    tl::hcu_buffer_store<unsigned short, 8, false>(tl::bit_cast<tl::thread_buffer<unsigned short, 8>>(*(uint4*)(((bfloat16_t*)acc_oshared1) + (((i_39 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<unsigned short*>(Output), (((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((int64_t)i_39) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)128), true, ((seq_len * batch) * 32768));
  }
  __syncthreads();
  #pragma unroll
  for (int i_40 = 0; i_40 < 4; ++i_40) {
    uint2 __6;
    float4 v__5 = *(float4*)(acc_o2 + (i_40 * 4));
    ((bfloat16_t*)(&(__6.x)))[0] = (bfloat16_t)(v__5.x);
    ((bfloat16_t*)(&(__6.x)))[1] = (bfloat16_t)(v__5.y);
    ((bfloat16_t*)(&(__6.y)))[0] = (bfloat16_t)(v__5.z);
    ((bfloat16_t*)(&(__6.y)))[1] = (bfloat16_t)(v__5.w);
    *(uint2*)(acc_oshared0_local_cast_3 + 0) = __6;
    *(uint2*)(((bfloat16_t*)acc_oshared0) + (((((i_40 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_40 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared0_local_cast_3 + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_41 = 0; i_41 < 4; ++i_41) {
    uint2 __7;
    float4 v__6 = *(float4*)(acc_o3 + (i_41 * 4));
    ((bfloat16_t*)(&(__7.x)))[0] = (bfloat16_t)(v__6.x);
    ((bfloat16_t*)(&(__7.x)))[1] = (bfloat16_t)(v__6.y);
    ((bfloat16_t*)(&(__7.y)))[0] = (bfloat16_t)(v__6.z);
    ((bfloat16_t*)(&(__7.y)))[1] = (bfloat16_t)(v__6.w);
    *(uint2*)(acc_oshared1_local_cast_4 + 0) = __7;
    *(uint2*)(((bfloat16_t*)acc_oshared1) + (((((i_41 >> 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + ((((((((int)threadIdx.x) >> 6) * 4) + ((i_41 & 1) * 2)) + ((((int)threadIdx.x) & 63) >> 5)) ^ (((int)threadIdx.x) & 15)) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(acc_oshared1_local_cast_4 + 0);
  }
  #pragma unroll
  for (int i_42 = 0; i_42 < 2; ++i_42) {
    tl::hcu_buffer_store<unsigned short, 8, false>(tl::bit_cast<tl::thread_buffer<unsigned short, 8>>(*(uint4*)(((bfloat16_t*)acc_oshared0) + (((i_42 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<unsigned short*>(Output), (((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((int64_t)i_42) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)256), true, ((seq_len * batch) * 32768));
  }
  __syncthreads();
  #pragma unroll
  for (int i_43 = 0; i_43 < 2; ++i_43) {
    tl::hcu_buffer_store<unsigned short, 8, false>(tl::bit_cast<tl::thread_buffer<unsigned short, 8>>(*(uint4*)(((bfloat16_t*)acc_oshared1) + (((i_43 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((int)threadIdx.x) & 15) ^ (((int)threadIdx.x) >> 4)) * 8)))), reinterpret_cast<unsigned short*>(Output), (((((((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)32768) + (((int64_t)((int)blockIdx.x)) * (int64_t)16384)) + (((int64_t)i_43) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)384), true, ((seq_len * batch) * 32768));
  }
}



// ============================================================================
// host dispatcher + C API
//   sm_scale 传自然值 1/sqrt(512)，host 乘 log2(e) 后喂 kernel（exp2 口径）。
//   动态 shared = 28672 字节（24576 KV/S + 4096 workspace，与 tilelang 生成内核一致）。
//   num_split 分派镜像 tilelang get_best_config(1, S, 64) 的实测选择表
//   （cu_count=70，2026-10-29 本机核对：S<=16 -> 4（S=16/17..26 档取 4/8，本实现
//     统一 4 走可整除值）；17<=S<=35 -> 2；S>=36 -> 1/no-split）。
//   中间缓冲：glse 按 num_split 最大 4 预留 S*4*64 fp16；partial 同，
//   persistent 按 S 缓存（tilelang 侧 torch.empty 走 caching allocator 近零成本，
//   本 .so 若每次 hipMalloc/hipFree 走驱动会吃 0.4~0.8ms —— 实测 2.9x 假差距的来源）。
// ============================================================================
#define SMLA_GLM_SHARED_BYTES 28672
#define LOG2E_F 1.4426950408889634f
#define SMLA_GLM_H 64
#define SMLA_GLM_REPLICATE_H 2
#define SMLA_GLM_DV 512
#define SMLA_GLM_MAX_SPLIT 4

namespace {
struct GlmWorkspace {
  void* glse = nullptr;
  void* partial = nullptr;
  int capacity_S = 0;
  void ensure(int S, hipStream_t st) {
    if (S <= capacity_S) return;
    if (glse) hipFree(glse);
    if (partial) hipFree(partial);
    hipMalloc(&glse, (size_t)S * SMLA_GLM_MAX_SPLIT * SMLA_GLM_H * sizeof(half_t));
    hipMalloc(&partial, (size_t)S * SMLA_GLM_MAX_SPLIT * SMLA_GLM_H * SMLA_GLM_DV * sizeof(half_t));
    capacity_S = S;
  }
};
GlmWorkspace& glm_ws() {
  static GlmWorkspace ws;
  return ws;
}
} // namespace

extern "C" void hip_sparse_mla_fwd(const void* Q, const void* KV, const void* Indices,
                                   void* Out, int S, int SKV, float sm_scale,
                                   void* stream) {
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
  const float scale_log2e = sm_scale * LOG2E_F;
  const int batch = 1, batch_head = 1, H = SMLA_GLM_H;
  int* idx = const_cast<int*>(reinterpret_cast<const int*>(Indices));
  bfloat16_t* kv = const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(KV));
  bfloat16_t* q = const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(Q));
  bfloat16_t* out = reinterpret_cast<bfloat16_t*>(Out);

  int num_split;
  if (S <= 16)      num_split = 4;
  else if (S <= 35) num_split = 2;
  else              num_split = 1;

  if (num_split == 1) {
    // no-split：单 kernel 直接写 out
    dim3 grid(S * SMLA_GLM_REPLICATE_H, 1, 1), block(256);
    sparse_mla_glm_nosplit_kernel<<<grid, block, SMLA_GLM_SHARED_BYTES, st>>>(
        idx, kv, out, q, batch, S, SKV, scale_log2e);
    return;
  }

  GlmWorkspace& ws = glm_ws();
  ws.ensure(S, st);
  half_t* glse = reinterpret_cast<half_t*>(ws.glse);
  half_t* partial = reinterpret_cast<half_t*>(ws.partial);

  if (num_split == 2) {
    dim3 grid(S * SMLA_GLM_REPLICATE_H, batch_head, 2), block(256);
    sparse_mla_glm_split_kernel<<<grid, block, SMLA_GLM_SHARED_BYTES, st>>>(
        idx, kv, partial, q, glse, batch, batch_head, S, SKV, scale_log2e);
    dim3 grid_c(S, H, 1), block_c(128);
    sparse_mla_glm_combine_kernel<<<grid_c, block_c, 0, st>>>(out, partial, glse,
                                                              batch, batch_head, S);
  } else {  // num_split == 4
    dim3 grid(S * SMLA_GLM_REPLICATE_H, batch_head, 4), block(256);
    sparse_mla_glm_split4_kernel<<<grid, block, SMLA_GLM_SHARED_BYTES, st>>>(
        idx, kv, partial, q, glse, batch, batch_head, S, SKV, scale_log2e);
    dim3 grid_c(S, H, 1), block_c(128);
    sparse_mla_glm_combine4_kernel<<<grid_c, block_c, 0, st>>>(out, partial, glse,
                                                               batch, batch_head, S);
  }
}
