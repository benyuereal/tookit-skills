# tilelang 各 example 在 DCU 的通过性矩阵

> 来源：内部文档 子文档《tilelang 使用方法与问题总结》§3。
> 2026-10-16 自 `operators/dcu-kernel-dev/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 九、tilelang 各 example 在 DCU 的通过性矩阵

> 来源：内部文档 子文档《tilelang 使用方法与问题总结》§3（file_id=<internal-file-id>）。
> 汇总 0.1.9/0.1.10 在 DCU 的实测通过性，供选型避坑（结论性，非性能基线）。

| 测试目录 | 结论 | 当前问题记录 |
|---|---|---|
| `tilelang/testing/python/amd` | `test_tilelang_test_amd.py` 通过、精度正常；其他用例能跑但精度有问题 | MatrixMul=OK；mfma core 精度校验失败；mfma BW1000 fp8 跑不通 |
| `tilelang/testing/python/dcu` | 0.1.9 无法跑通；前期 0.1.6 可跑通 | — |
| `tilelang/testing/python/analysis` | OK | — |
| `tilelang/examples/deepseek_v32` | sparse_mla_fwd 编译成功，调用参数问题 | 一般资源越界，调低块尺寸继续；待验证 |
| `tilelang-0.1.10/examples/amd` | autotune 阶段 acc_s / acc_s_cast 布局推断冲突 | 两者 dtype 不同 → 换 acc_s_cast 为共享内存做 cast 可解；待验证 |
| `tilelang-0.1.10/examples/gemm/example_gemm.py` | 测试 OK | — |
| `tilelang-0.1.10/examples/blocksparse_gemm` | autotune OK，测试 OK | — |
| `tilelang-0.1.10/examples/attention_sink/example_mha_sink_fwd_bhsd.py` | OK，tilelang vs torch 加速比约 **3.85×** | — |

**通用修复「资源越界 / invalid argument」**：调低块尺寸（`block_M=32, block_N=32, num_stages=1, threads=128`），并 `assert total_shared <= max_shared` 提前发现（见 §四 已知问题 12）。
**amd 布局推断失败共性**：部分 amd tensor layout 在 DCU 报错；不同 dtype 共用内存属性 cast 易布局推断失败 → 改用 shared memory 做 cast。

