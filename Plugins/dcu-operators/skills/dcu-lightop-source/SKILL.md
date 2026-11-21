---
name: dcu-lightop-source
description: LightOp海光融合算子库源码结构与适配
version: 1.0.0
author: DCU Engineer
license: MIT
---

# LightOp 源码适配索引（dcu-lightop-source）

> 基于 `<local_src_cache>/lightop-master/`（自研融合算子库源码）。README 实测：norm 类对 torch native 有 2-5× 提升。海光**自研**，非 AMD ROCm 移植。

## §0 定位：LightOp 是什么、与 AITER 的关系

LightOp = 海光**自研**的融合算子库（C++/HIP kernel + PyTorch 绑定），把 torch 原生算子融合优化（norm/dropout/gemm/softmax 组合），norm 类 2-5× 提升。

**与 AITER 的选型关系（关键，避免以后混淆"改哪个库"）**：
| 维度 | LightOp | AITER（dcu-aiter-source） |
|---|---|---|
| 来源 | 海光**自研** | AMD ROCm/aiter 上游的 DCU 适配版（das） |
| 上游对照 | **无公开上游**，适配要点是 DTK/gfx 矩阵 | 有 ROCm/aiter 上游，可 diff（见 dcu-aiter-source §3） |
| 算子侧重 | norm/dropout/activation 融合 + quant_gemm/moe_gemm + DeepSeek-V4 专用 kernel | MoE 四后端调度框架 + Attention/Quant/GEMM |
| 复用关系 | csrc 含 `aiter_hip_common.h`（复用 AITER 的 HIP 公共头） | — |
| 选型建议 | 融合 norm/激活类、DeepSeek-V4 类专用 kernel 优先看 LightOp | MoE 多后端调度、标准 Attention/GEMM 优先看 AITER |

**链路**（你"写适配优化算子"）：LightOp 取/改 kernel → `dcu-kernel-dev`（HIP 规范）→ `dcu-vllm-op-replace`（接入 vLLM 6 步）。与 AITER 走同一条下游链路。

## §1 源码地图（实测目录）

```
lightop-master/
├── python/lightop/      # Python 包（算子接口 + 绑定）
├── csrc/                # HIP/C++ kernel（大量 .cu） + export.cpp（pybind 总注册）
│   ├── aiter_hip_common.h   # 复用 AITER 的 HIP 公共头（两库耦合点）
│   ├── asm_cu/ awq_w4a16_gemm/ blockwise_scaling/ bmm_scale/ allreduce_fuse/ concat/  # 子模块
│   ├── moe_*/ gemm_*/ quant_*  # 核心 GEMM/MOE/量化 kernel
│   ├── fuse_rms_*/ fuse_layernorm_*/ fuse_*dropout*  # 融合 norm/dropout/activation
│   └── fused_deepseek_v4_*_kernel.cu  # DeepSeek-V4 专用（qnorm_rope_kv_insert 等）
├── configs/             # 分 gfx 调优表（MOE_BLOCKINT4_*_gfx936/gfx938.json、BATCHED_GEMM_W8A8_*_GFX938.json）
├── third_party/         # librarygemmbiasf / librarygemmdbiasb（GEMM bias 库）
├── docs/migration/      # 迁移文档
├── test/                # 算子测试（moe_*/ groupgemm/ deepgemm/ mqa_logits/ run_*_tests.py + all_test.py + benchmark_test_kernel.sh）
├── hsa/ lib/ tools/ compat/   # 运行时/工具
└── setup.py setup_torch29.py get_version.py
```

## §2 构建与安装（海光实测）

- **依赖 DTK**：README 支持 dtk22.04 / 22.10 / 23.04 / 23.10 / 24.04；当前体系 dtk26.04 经实测同样可用（需 pytorch 2.1+/2.3+）。pytorch 1.10/1.13/2.0/2.1/2.3。
- **安装（develop/install）**：
  ```bash
  python3 setup.py install          # 或 bdist_wheel 打 whl
  # 验证
  python -c "import lightop; print(lightop.__version__)"
  ```
- **gfx 架构指定（23.10 前去 928）**：
  ```bash
  PYTORCH_ROCM_ARCH='gfx906;gfx926' python3 setup.py install   # 旧 DTK 无 928
  # 新 DTK（936/938）按卡指定，如 gfx936;gfx938
  ```
- **测试**：`cd test && bash benchmark_test_kernel.sh`（生成性能/精度表）；或 `python3 all_test.py`、`python3 test_ln_add_dropout.py` 单测。
- **torch2.9 专用**：`setup_torch29.py`（适配新 torch 的构建入口）。

## §3 海光适配要点（无上游 diff，以 DTK/gfx 矩阵为准）

> **与 AITER 的方法差异**：AITER 有 ROCm/aiter 上游可 `diff -rq` 对照（dcu-aiter-source §3）；LightOp 是海光自研、**无公开上游**，故不跑上游 diff，适配要点改为 **DTK 版本支持矩阵 + gfx 列表 + 与 AITER 的代码复用点**。

- **DTK 支持矩阵**（README 实证 + 26.04 实测）：dtk22.04 ~ dtk24.04 全系列；dtk26.04 实测可用。pytorch 1.10/1.13/2.0/2.1/2.3。
- **gfx 支持**：gfx906 / gfx926 / gfx928 / gfx936 / gfx938。注意 **23.10 之前 DTK 不支持 gfx928，需 `PYTORCH_ROCM_ARCH` 显式去掉 928**（见 §2）。
- **与 AITER 耦合**：`csrc/aiter_hip_common.h` 直接复用 AITER 的 HIP 公共头——改 LightOp 的 HIP 代码时需同步关注 AITER 该头改动（两库可能共享底层定义）。
- **调优表分 gfx**：`configs/` 下 json 按 `GFX936`/`GFX938` 命名（MOE_BLOCKINT4、BATCHED_GEMM_W8A8），换 gfx 必须用对应调优表。
- **[待核验] 精确 DTK 对应关系**：README 给的是 dtk22.04~24.04 区间；具体某算子在某 DTK/gfx 的可用性以实际编译/测试为准（见 dcu-dtk）。

## §4 算子族索引（实现位置速查）

| 算子族 | 代表算子 | csrc kernel | 调优表/后端 | 说明 |
|---|---|---|---|---|
| 核心 GEMM/量化 | quant_gemm(w8a8)、gemm_awq_w4a16 | `csrc/*gemm*`, `awq_w4a16_*`, `blockwise_scaling/` | `configs/BATCHED_GEMM_W8A8_*_GFX938.json` | README 核心算子 |
| MoE GEMM | moe_gemm_w8a8 / w4a8 / w4a16 | `csrc/moe_*`, `deepgemm_moe_permute.cu`, `fuse_groupgemm_permute_quant.cu` | `configs/MOE_BLOCKINT4_*_gfx936/gfx938.json` + Marlin | README 核心算子（w4a16 Marlin） |
| 融合 Norm/Dropout | layernorm/rmsnorm/groupnorm + dropout + bias 组合 | `fuse_rms_*`, `fuse_layernorm_*`, `fuse_*dropout*`, `fuse_bias_*` | — | README "torch 算子"表（2-5× 提升主力） |
| 融合 Activation | bias_swiglu / bias_gelu / silu 类 | `bias_swiglu.cu`, `fuse_silu_mul_*_quant.cu` | — | SwiGLU/GeLU 融合 |
| Attention/MQA | mqa_logits | `csrc/mqa_logits*`（同 AITER 命名） | — | test/mqa_logits |
| DeepSeek-V4 专用 | qnorm_rope_kv_insert / kv_rope_quant | `fused_deepseek_v4_*_kernel.cu` | — | DS-V4 特化 kernel（LightOp 独有，AITER 无） |
| AllReduce 融合 | allreduce_fuse | `csrc/allreduce_fuse/` | — | 通信融合 |

## §5 调优表与实现落点

- **分 gfx 调优**：`configs/MOE_BLOCKINT4_*_{gfx936,gfx938}.json`（Marlin W4A16）、`configs/BATCHED_GEMM_W8A8_*_GFX938.json`（W8A8 批 GEMM）。换 gfx 必须换对应 json。
- **GEMM bias 库**：`third_party/librarygemmbiasf`、`librarygemmdbiasb`（第三方 GEMM 带 bias 实现）。
- **Python 接口**：`python/lightop/`；pybind 总注册 `csrc/export.cpp`。

## §5.1 W8A8 GEMM 调优配置机制（gemmopt.py，2026-08 实测）

> 来源：Qwen3.8-27B-INT8-w8a8 在 gfx936/80cu 4 卡 TP 上的适配实测。LightOp 的 W8A8 smooth-quant GEMM 走 `gemmopt.py::w8a8_smooth_quant_config(m,n,k)` 选 kernel 配置，**这是 decode 小 M 场景性能的关键开关**——配置选错会静默回退到慢数倍的 triton（**下文所有具体倍数/百分比/噪声幅度均为该次实测，仅参考**，随 shape/gfx/DTK/模型变，须本机复测）。

- **配置选择是精确匹配**：`w8a8_smooth_quant_config` 用 key `M{m}N{n}K{k}` 查 tuned 字典（`config_smooth_gemm.py` 的 `w8a8_smooth_quant_config_dict_{80,64}`，按 `LIGHTOP_GPU_CUS`/`LIGHTOP_GPU_TARGET` 选）。**匹配不到就返回 status=False**，上层（lmslim `triton_scaled_mm`）回退到慢 3-5x 的 triton `matmul_int8`。
- **坑：tuned 字典是别的模型 shape**：字典里上千条 key 全是其他模型（N=2112/5120/4096 配 K=7168/4096/12800 等），**本模型 TP 切分后的 per-rank shape 一条都不在** → 生产里 decode 全部 fallback 到 triton。换模型/换 TP 数后 per-rank shape 变，必须重新核对字典命中（`python -c "from lightop.config_smooth_gemm import gemm_w8a8_smooth_config as c; d=c.get_w8a8_smooth_quant_config_dict(); print('M4N4352K5120' in d)"`）。
- **修复模式（fallback 默认配置）**：patch `w8a8_smooth_quant_config` 的 else 分支——未命中且 `k%256==0 and n%32==0`（kernel 约束）时返回安全默认配置 `(4,32,256,1,False)` 而非回退 triton。env `LIGHTOP_FALLBACK_DEFAULT=0` 可关。纯 kernel 比 triton 快数倍（**具体倍数仅参考**，随 shape/gfx/DTK 变，须本机微基准实测）。
- **per-shape 调优（进一步）**：默认配置不是 per-shape 最优。全网格调优 = 各 per-rank shape × M=1..16 × (warps 1/2/4/8 × nperblock 16/32/64/128 × kperwarp 64/128/256/512 × splitk 1/2/4 × use_n_warps T/F)，直接调 `op.gemm_w8_a8_smooth_colmajor(a, b, scale_a, scale_b, bias, out_dtype, warps, nperblock, kperwarp, splitk, use_n_warps)` 测纯 kernel 时间（min-of-N 轮取最小）。**weight 布局**：lightop 要 non-contiguous `(K,N)` 转置视图（`b.is_contiguous()==False`），生产 `process_weights_after_loading` 的 `weight.t()` 满足；triton 要 contiguous `(K,N)`。
- **集成 tuned 配置（外部 JSON）**：把 per-shape 最优写 `tuned_*.json`（key→`[warps,nperblock,kperwarp,splitk,use_n_warps]`），patch else 分支先查外部 JSON 再回退默认，env `LIGHTOP_TUNED_JSON` 改路径/=0 关闭。
- **⚠ 坑：gemmopt.py 缺 `import json`**——patch 里用 `json.load` 但原文件没 import json，NameError 被 `except: pass` 静默吞掉，tuned 查找**从未生效**（表面 status=True 走默认配置，看不出问题）。调试方法：临时把 `except: pass` 改成 `except Exception: traceback.print_exc()` 暴露真实异常。**给第三方库打 patch 用 json/os 等标准库时，先确认原文件已 import**。
- **⚠ 坑：微基准必须在干净 GPU 上跑**——生产服务在跑时跑 GEMM 微基准，GPU 争抢把 default 基线抬高 20-30%，tuned 收益被夸大。停服务再测才可信。
- **端到端传导**：GEMM 微基准提速**不线性**传导到端到端。把 GEMM 砍半（fallback patch）端到端 +6%；再 per-shape 调优省 13% GEMM 端到端却 wash（±2% 噪声）——GEMM 占比下降后边际收益递减，且 20-prompt 基准噪声 ±2-3%。

## §6 和 vLLM / AITER 的衔接

- **vLLM 接入**：经 `dcu-vllm-op-replace` 6 步（HIP kernel → torch_bindings → `_custom_ops.py` 登记 → `rocm.py` 重定向 → 重编译 → 验证）。LightOp 的 MoE/GEMM kernel 可替换 vLLM 原生路径（参照 vLLM 0.15.1 有 `moe_fused_gate.cu`、AITER 有 `dcu-aiter-moe` 专题）。
- **与 AITER 协作**：优先按 §0 选型表决定改 LightOp 还是 AITER；注意 `aiter_hip_common.h` 共享头，改动需双向确认。
- **开发规范 / 验证闭环**：`dcu-kernel-dev`（HIP 语法/编译调试）、`dcu-op-optimization` + `dcu-perf-profiling`（profile→定位→改→验证）。
- **DTK 前置**：`dcu-dtk`（DTK 版本选型、容器内必挂 `-v /opt/hyhal:/opt/hyhal:ro`）。

## 使用须知
- 经验仅供参考，不保证修复。**先核对**：设备型号（gfx936/938 等）、DTK 版本、pytorch 版本、报错原文。LightOp 无公开上游，适配点以 DTK 矩阵 + 实际编译为准。
- 改 kernel 后必须重编（`setup.py install`）；`PYTORCH_ROCM_ARCH` 与 gfx 强相关——**换 gfx 必须重编**（旧 DTK 去 928）。
- `aiter_hip_common.h` 是 LightOp↔AITER 耦合点，改前确认是否影响另一库。
- 高危操作（重装驱动/刷 vbios）需谨慎，确认带外恢复通道、优先测试机验证（见 dcu-ontrack-experience 使用须知）。
