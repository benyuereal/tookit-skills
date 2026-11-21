# 手写 HIP 交付物的编译与排查（DCU 本机配方，非通用方法）

> 本文件是**某一台机器、某一套编译器**的可复现配方，不属于算子卡的可迁移方法。
> 换语言（Triton / CUDA / 别家 HIP）时整份作废，只需要保留下面两条**原则**：
>
> 1. **镜像对标实现的实际编译选项，而不是抄一份静态 flag 列表。** 对标实现（这里是 tilelang JIT）
>    的选项由代码在运行时拼出来，可能带条件项；漏一条就可能慢 10%+，且**数值完全一致**，
>    所以只有逐 flag A/B 才找得到。
> 2. **怀疑 codegen 时读二进制，不要读源码猜。** `llvm-readelf --notes <.so>` 看内嵌设备 ELF 的
>    `.text` 大小 / VGPR / SGPR，用 VGPR 反推每 CU 能驻几个 CTA。

## 为什么必须用 aicc

`aicc`（DTK 自带，`/opt/dtk/bin/aicc`）是本机 tilelang JIT 实际调用的编译器；
**hipcc（后端 dcc/lld）拒收这批 `-mllvm` 项**（报 `Unknown command line argument`）。
flag 本质是后端 LLVM 的指令调度 / 延迟建模 / 寄存器预算（mmac 与 LDS 延迟建模、
768 VGPR 预算、waitcnt 时机），**不改变数值**（与无 flag 路径输出位级一致）。

## canonical 构建命令（GLM 档，11 条 flag）

```bash
/opt/dtk/bin/aicc -x hip --offload-arch=gfx938:sramecc+ -std=c++20 -O3 \
  -I $TILELANG_HYGON_ROOT/src \
  -mllvm=-enable-latency-hack=true -mllvm=-mmac-latency=5 \
  -mllvm=-ds-load-store-latency=6 -mllvm=-disable-machine-sink=True \
  -mllvm=-check-valu-data-forward-hazards=0 -mllvm=-disable-cluster-lds-memops=true \
  -mllvm=-amdgpu-disable-backoff-barrier=false -mllvm=-support-768-vgprs=true \
  -mllvm=-hcu-update-wait-by-reverse-search=true -mllvm=-hcu-pre-emit-load-store-opt=false \
  -mllvm=-enable-hcu-approx-func-fp-math=true \
  dev/sparse_mla_hip_glm.hip -shared -fPIC -o dev/libsparse_mla_hip_glm.so
```

前 10 条是 `tilelang/contrib/hcu.py: get_hcu_compile_flags()` 的无条件项；
第 11 条是 **pass_configs 条件项**：`TL_ENABLE_FAST_MATH` 打开时追加
`-enable-hcu-approx-func-fp-math`（`hcu.py:271`），而 `sparse_mla_fwd` 的 `@tilelang.jit`
正是开了 FAST_MATH。漏掉它 S=128/320k 从 0.810 变 0.952 ms（1.16×，两版输出位级一致）。

WDRA 对应项（`-mllvm=-vgpr-greedy-alloc-mode=local-wave`）实测 **+1.2% 更慢，不要加**。

## 「同体不同速」三步排查

1. `get_kernel_source()` 导出对标 kernel 源码，与手写体做 token diff；
2. `llvm-readelf --notes <.so>` 读内嵌 GPU ELF 的 `.text` 大小 / VGPR / SGPR 对照；
3. 同进程 A/B 逐 flag 变体（见 `candidates.yaml: kernel_body_bytecopy_needs_flag_mirror`）。

## 两条配套事实

- **内层 KV gather 用 `tl::hcu_buffer_load`**（单条向量化硬件 `buffer_load_dwordx4` + buffer
  描述符 + 硬件 OOB clamp，`-1` 哨兵自动 0-fill 免分支）；标量 global load 循环恒定慢
  **~1.35×/CTA** 且 S=128 落后 flash_mla 13%。两条路径**位级一致**，纯速度差异。
  可迁移的原则：**gather 用硬件向量化 + 硬件越界钳位，别在 kernel 里手写分支做哨兵**。
- 设备头只 include 真实 tilelang device 头 `tl_templates/hcu/{common,gemm,reduce,barrier}.h`；
  `copy.h` 混入本机 DTK 没有的 async builtins（`__builtin_amdgcn_asyncmark` 等）会编译失败，
  所以 kernel 里 verbatim vendor 了 `copy.h` 的两个顶层 `hcu_buffer_load/hcu_buffer_store`
  包装（真正的搬运原语 `make_wave_buffer_resource` + `*_impl` 在
  `core/arch/amd_buffer_addressing.hpp`，`common.h` 已引入）。
