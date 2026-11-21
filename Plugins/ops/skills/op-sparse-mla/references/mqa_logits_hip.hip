// MQA logits（DSA indexer 打分）—— 手写 HIP 版。里程碑 M1：先正确，再谈快。
//
// 公式：logits[m,n] = Σ_h relu((q[m,h,:]*w[m,h]) · k[n,:])，窗口 [ks[m], ke[m])
//   w 已折进 q（relu 正齐次：relu(x)·w = relu(x·w)，w>0）⇒ epilogue 里没有乘法。
//
// 为什么必须 MMA：每输出 32 head × 128 维 = 4096 MAC；标量 FMA 版实测 860 µs 级，
// 而 MMA(fp8 16x16x32) 版地板 16 µs。lightop 的 mqa kernel 本身就是手写 MMA
// （csrc/mqa_logits/mqa_logits_asm.cu），复用 Tensile 的 GEMM 调度 + 轻量 epilogue。
//
// 本文件的分工（对齐 lightop 的结构，但用我们 sparse_mla_hip_glm.hip 里已验证的 idiom）：
//   1. K 块 [BN, D] 走 tl::hcu_buffer_load（向量化 + 硬件 OOB 钳位）
//   2. Q（M 条 query 的全部 head）常驻寄存器/LDS，一次加载
//   3. MMA 算 raw[BN, M*H]（fp8×fp8 → fp32）
//   4. epilogue：relu → 按 h 归约 → × SQ × KSV[n] → 写 OUT
//
// 状态：**骨架，未跑通**。M1 待办见文件末尾 TODO。
// 编译（aicc，与 sparse MLA 同一套 11 条 -mllvm，见 dev/COMPILE.md）：
//   /opt/dtk/bin/aicc -x hip --offload-arch=gfx938:sramecc+ -std=c++20 -O3 \
//     -I $TILELANG_HYGON_ROOT/src -mllvm=-support-768-vgprs=true \
//     ... mqa_logits_hip.hip -shared -fPIC -o libmqa_logits.so

#include <hip/hip_runtime.h>
#include <tl_templates/hcu/hcu_fp8.h>
#include <tl_templates/hcu/copy.h>
#include <tl_templates/hcu/reduce.h>
#include <tl_templates/hcu/barrier.h>

#define DEV_M 8        // query 数（生产 M≈8）
#define DEV_H 32       // indexer head 数
#define DEV_D 128      // indexer 头维
#define DEV_BN 128     // 每个 CTA 处理多少 KV 位置
#define DEV_THREADS 256

extern "C" __global__ void __launch_bounds__(DEV_THREADS)
mqa_logits_kernel(const unsigned char* __restrict__ Q,   // [M, H, D] fp8
                  const unsigned char* __restrict__ K,   // [N, D]    fp8
                  const int*           __restrict__ KS,  // [M]
                  const int*           __restrict__ KE,  // [M]
                  const float*         __restrict__ KSV, // [N] per-row scale
                  float*               __restrict__ OUT, // [M, N]
                  int N, float SQ)
{
  extern __shared__ __align__(1024) unsigned char buf_dyn_shmem[];
  unsigned char* q_sh = buf_dyn_shmem;                        // [M*H, D] fp8 = 32 KB
  unsigned char* k_sh = buf_dyn_shmem + DEV_M * DEV_H * DEV_D; // [BN, D]  fp8 = 16 KB

  // TODO(M1-1)  Q 常驻：M 次 2D 拷贝进 q_sh。
  //             注意 32 KB + 16 KB = 48 KB ⇒ 1 CTA/CU；这与我们实测的
  //             「1 vs 2 CTA/CU 差 67~69%」冲突，所以这一步很可能要改成
  //             「Q 常驻寄存器」或「每轮只驻一个 head 块」——见 TODO(M1-4)。
  // TODO(M1-2)  K 块走 tl::hcu_buffer_load<unsigned short, 8, true>（硬件 OOB 钳位）。
  // TODO(M1-3)  MMA：__builtin_hcu_mmac_f32_16x16x32_fp8_fp8
  //             A = q_sh 的 (M*H, D) 片段，B = k_sh 的 (BN, D) 片段，accum fp32
  //             输出 raw[BN, M*H] 在寄存器（BN=128, M*H=256 ⇒ 128 regs/线程，需分块）
  // TODO(M1-4)  epilogue：relu → 按 h 归约 → ×SQ×KSV[n] → 写 OUT
  //             归约若跨线程，用 tl::AllReduce<SumOp,...>::run_batch
  // TODO(M1-5)  验证：对 fp32 参考比 max_abs / cosine（参考用反量化后的 q/k）
}

// M1 完成判据（按顺序，每步都要有数字）：
//   1. 编译通过（上面那条 aicc 命令）
//   2. S=8 / N=8192 单形状 cos ≥ 0.999
//   3. N=131072 同进程 ABBA 对 lightop 的比值
//   4. 若比值 > 1.3×，回 op-roofline/references/pipeline.md 的四问重新设计，
//      不要直接调参数（见 Plugins/ops/SKILL.md §四 5b）
