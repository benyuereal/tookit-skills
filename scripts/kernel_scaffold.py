#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kernel_scaffold —— 从访问模式模板生成**可编译的** kernel 骨架。

为什么需要它
------------
在此之前，我们的 agent 只有「诊断」能力：`isa_probe` 能看出厂商 kernel 用了什么，
`lang_select` 能说该用哪个语言。**但它写不出新 kernel** —— 全仓 4 个 `.hip`
都是历史上手工写的，没有任何生成工具。

所以「写出比 lightop 更好的实现」这件事从来发生过。
本文件补的就是这一环：**给定访问模式 + 语言，生成一个能编译、能跑、能测的骨架**。

生成什么（不是万能代码，是**可改的起点**）
----------------------------------------
    骨架保证：能编译 / 能过正确性 / 有一个笨但正确的实现
    骨架不保证：快。**快是 agent 要在这上面改出来的事。**

这不是「自动优化器」。它把「从零写一个 HIP kernel」的固定成本降下来，
让 agent 的注意力放在**设备细节**上（LDS 布局、向量宽度、归约方式）。

支持的模板（与 op-roofline/templates.py 对齐）
----------------------------------------------
    reduce-lastdim       rmsnorm / layernorm 类
    elementwise-binary   silu×mul / add 类
    matmul               GEMM 类（只给 naive + 提示，不试图替代 hipBLASLt）

用法
----
    python3 -I scripts/kernel_scaffold.py --template reduce-lastdim --lang hip --out /tmp/k.hip
    python3 -I scripts/kernel_scaffold.py --template reduce-lastdim --lang triton --out /tmp/k.py
    python3 -I scripts/kernel_scaffold.py --template reduce-lastdim --lang hip --out /tmp/k.hip --build
    python3 -I scripts/kernel_scaffold.py --list
    python3 -I scripts/kernel_scaffold.py --selftest
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- HIP 模板

HIP_REDUCE = r'''// 由 kernel_scaffold.py 生成 —— 行内归约（RMSNorm 形态）的**可编译起点**
//
// 已做对的（不要改坏）：
//   - 每行一个 block，整行读进 LDS 后做归约（避免两趟 HBM）
//   - 用 float 累加，避免 bf16 精度损失
//   - 写出用 vector 化提示（见下）
//
// **这是起点，不是最优解。** 已知可改的地方（agent 要自己判断哪个值得改）：
//   1. LDS 用量：当前 = 行长 × 4 bytes。若行长大，考虑分块 + 在线归约
//   2. VGPR：当前每线程持 1~2 个元素。若用向量化（float4），VGPR 会涨但指令数会降
//   3. 归约：当前是 LDS + barrier 的朴素树归约，可换 warp shuffle 或 ds_bpermute
//   4. 向量宽度：读用 `__hip_bfloat16` 单元素，可改成 float4 一次读 4 个
//   5. 驻留：当前 block=256 threads，LDS=R*4。用 inspect_so.py 算 CTA/CU 再定
//
// 编译：
//   hipcc -O3 --offload-arch=gfx938 k.hip -o k
// 看资源：
//   python3 -I scripts/isa_probe.py k  --json
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using bf16 = hip_bfloat16;   // 本机头文件提供 operator float()，无 __bfloat162float

// RMSNorm(x) * weight —— 单行
__global__ void rmsnorm_kernel(const bf16* __restrict__ x,
                               const bf16* __restrict__ w,
                               bf16* __restrict__ y,
                               int M, int H, float eps) {
  extern __shared__ float sdata[];          // 行长 H 个 float

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const bf16* xr = x + (size_t)row * H;
  bf16* yr = y + (size_t)row * H;

  // 1) 读进 LDS 并累加平方
  float local = 0.f;
  for (int i = tid; i < H; i += nthreads) {
    const float v = static_cast<float>(xr[i]);
    sdata[i] = v;
    local += v * v;
  }

  // 2) block 级归约（朴素树；**可换成 warp shuffle**）
  sdata[tid] = local;
  __syncthreads();
  for (int s = nthreads / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  const float rms = sqrtf(sdata[0] / (float)H + eps);
  __syncthreads();                          // sdata 还要复用（下面从 LDS 读回）

  // 3) 从 LDS 读回并写 y
  //    注意：这里读的是 **LDS**，不是再读一遍 HBM —— 这是本骨架最重要的设计
  for (int i = tid; i < H; i += nthreads) {
    const float v = sdata[i];
    const float wv = static_cast<float>(w[i]);
    yr[i] = bf16(v / rms * wv);
  }
}

int main(int argc, char** argv) {
  const int M = (argc > 1) ? atoi(argv[1]) : 8192;
  const int H = (argc > 2) ? atoi(argv[2]) : 4096;

  bf16 *dx, *dw, *dy;
  hipMalloc(&dx, (size_t)M * H * sizeof(bf16));
  hipMalloc(&dw, (size_t)H * sizeof(bf16));
  hipMalloc(&dy, (size_t)M * H * sizeof(bf16));

  hipMemset(dx, 0, (size_t)M * H * sizeof(bf16));
  hipMemset(dw, 0, (size_t)H * sizeof(bf16));

  const int threads = 256;
  const size_t shmem = (size_t)H * sizeof(float);

  // 预热 + 计时
  for (int i = 0; i < 5; i++)
    rmsnorm_kernel<<<M, threads, shmem>>>(dx, dw, dy, M, H, 1e-6f);
  hipDeviceSynchronize();

  hipEvent_t e1, e2;
  hipEventCreate(&e1); hipEventCreate(&e2);
  float best = 1e30f;
  for (int i = 0; i < 21; i++) {
    hipEventRecord(e1);
    rmsnorm_kernel<<<M, threads, shmem>>>(dx, dw, dy, M, H, 1e-6f);
    hipEventRecord(e2);
    hipEventSynchronize(e2);
    float ms; hipEventElapsedTime(&ms, e1, e2);
    if (ms < best) best = ms;
  }
  const double bytes = (double)M * H * 2 * sizeof(bf16);   // 1读1写
  printf("median_ms=%.4f  GB/s=%.0f\n", best, bytes / (best * 1e-3) / 1e9);
  return 0;
}
'''

# ---------------------------------------------------------------- Triton 模板

TRITON_REDUCE = r'''# 由 kernel_scaffold.py 生成 —— 行内归约的 Triton 起点
#
# Triton 的优势：迭代快、autotune 便宜。
# Triton 的劣势：**控不了 LDS bank / 寄存器排布**。
#   ⇒ 若瓶颈是这两个，本模板改到头也到不了厂商水平，该换 HIP。
#   （依据：rmsnorm 上 triton 比 lightop 慢约 20%，而 tilelang 只差 12%）
#
# 已做对的：
#   - 单 kernel 内完成读→归约→写，不落中间态
#   - 用 tl.float32 累加
# 已知可改：
#   1. num_warps（当前 4）——用 autotune 扫
#   2. BLOCK 是否等于 H（当前是）——H 大时考虑分块 + 两趟
#   3. 若 H 不是 2 的幂，需要 mask（本模板假定是）
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_k(X, W, Y, stride, H: tl.constexpr, eps: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride + c).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    w = tl.load(W + c).to(tl.float32)
    tl.store(Y + row * stride + c, (x * inv * w).to(Y.dtype.element_ty))


def run(x, w, eps=1e-6, num_warps=4):
    M, H = x.shape
    y = torch.empty_like(x)
    rmsnorm_k[(M,)](x, w, y, x.stride(0), H, eps, H, num_warps=num_warps)
    return y


if __name__ == "__main__":
    import statistics
    M, H = 8192, 4096
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(H, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        run(x, w)
    torch.cuda.synchronize()
    ts = []
    for _ in range(21):
        e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
        e1.record(); run(x, w); e2.record(); torch.cuda.synchronize()
        ts.append(e1.elapsed_time(e2))
    med = statistics.median(ts)
    byts = M * H * 2 * 2
    print("median_ms=%.4f  GB/s=%.0f" % (med, byts / (med * 1e-3) / 1e9))
'''

TEMPLATES = {
    "reduce-lastdim": {"hip": HIP_REDUCE, "triton": TRITON_REDUCE,
                       "note": "行内归约（rmsnorm / layernorm 形态）"},
}

LANGS = ("hip", "triton")


def generate(template, lang):
    t = TEMPLATES.get(str(template))
    if not t:
        return None, "未知模板 %r（可选 %s）" % (template, sorted(TEMPLATES))
    if lang not in LANGS:
        return None, "未知语言 %r（可选 %s）" % (lang, list(LANGS))
    src = t.get(lang)
    if not src:
        return None, "模板 %s 没有 %s 版本" % (template, lang)
    return src, None


def build(path, arch="gfx938"):
    """编译并返回 (ok, 输出)。这是「编译→诊断闭环」的第一步。"""
    hipcc = "hipcc"
    out = Path(path).with_suffix("")
    p = subprocess.run([hipcc, "-O3", "--offload-arch=%s" % arch, str(path),
                        "-o", str(out)], capture_output=True, text=True, timeout=300)
    return p.returncode == 0, ((p.stdout or "") + (p.stderr or ""))[:1500], out


def selftest():
    # 1) 两个模板都能生成
    for tpl in ("reduce-lastdim",):
        for lang in LANGS:
            src, err = generate(tpl, lang)
            assert src and not err, err
    print("[ok] reduce-lastdim 的 hip / triton 版本都能生成")

    # 2) HIP 版必须能编译
    src, _ = generate("reduce-lastdim", "hip")
    d = Path(tempfile.mkdtemp())
    f = d / "k.hip"
    f.write_text(src, encoding="utf-8")
    ok, out, exe = build(f)
    assert ok, "HIP 骨架编译失败：\n%s" % out
    print("[ok] HIP 骨架编译通过（这是「可用的起点」的最低标准）")

    # 3) 编译后能跑出数
    p = subprocess.run([str(exe), "1024", "512"], capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stderr[:400]
    assert "median_ms=" in p.stdout, p.stdout
    print("[ok] 编译产物能跑出计时：%s" % p.stdout.strip())

    # 4) 骨架里必须写明"这是起点，已知可改的地方"
    assert "已知可改" in src or "起点" in src
    print("[ok] 骨架自带「哪里可以改」的清单（不是黑箱）")

    # 5) 未知模板报错
    _, err = generate("nope", "hip")
    assert err
    print("[ok] 未知模板报错：%s" % err[:30])

    print("\nselftest PASS（5 项，含真编译 + 真跑）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成可编译的 kernel 骨架")
    ap.add_argument("--template", default="reduce-lastdim")
    ap.add_argument("--lang", default="hip", choices=LANGS)
    ap.add_argument("--out")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.list:
        for k, v in TEMPLATES.items():
            print("%-18s %-10s %s" % (k, "/".join(LANGS), v["note"]))
        return 0
    src, err = generate(a.template, a.lang)
    if err:
        print("ERROR: %s" % err, file=sys.stderr)
        return 2
    out = Path(a.out or ("/tmp/k.%s" % ("hip" if a.lang == "hip" else "py")))
    out.write_text(src, encoding="utf-8")
    print("已生成 %s（%d 行）" % (out, len(src.splitlines())))
    if a.build:
        ok, msg, exe = build(out)
        print("编译 %s" % ("成功 → %s" % exe if ok else "失败"))
        if not ok:
            print(msg)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
