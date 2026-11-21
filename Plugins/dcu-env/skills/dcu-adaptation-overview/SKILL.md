---
name: dcu-adaptation-overview
description: DCU 适配总览——版本兼容矩阵（DTK×torch×框架×算子包）+ 软件能力矩阵（哪些能跑/验证状态/限制）+ 真机验证结果 + 升级 SOP（19 包实测 + <device-A>/<device-B> 真机验证）
version: 1.2.0
author: DCU Engineer
license: MIT
---

# DCU 适配总览（dcu-adaptation-overview）

> 定位：**DCU 适配的"查表"总览**——版本兼容矩阵（DTK×torch×框架×算子包怎么配）+ 软件能力矩阵（哪些软件在 DCU 上真能跑、验证状态、限制）+ 升级 SOP。让 agent/用户快速回答"这个环境该装什么版本""这个包在 DCU 上能不能用"。
> 与 `dcu-env-packages` 的关系：那个讲**源里有哪些包 + 怎么装**（包清单/索引路径/排障）；本 skill 讲**版本怎么配 + 能不能用**（兼容矩阵/能力矩阵/验证状态）。
> 与 `dcu-software-adaptation-guide` 的关系：那个讲**怎么适配一个新软件**（方法论 SOP）；本 skill 讲**已适配的软件现状**（查表）。
> 与 10 个 source skill 的关系：那些讲**具体包源码结构**；本 skill 是**跨包总览**，source skill 是细节。
> 数据来源：2026-09-01 19 包 wheel 解包分析 + <device-A>/<device-B> 真机验证。**版本会更新**，装前用 `dcu-das-hub-api` 复核最新。

> **DTK 版本现状（2026-10-09 实测）**：DTK-26.10 工具包已发布（2026-09-28，ROCm 7.2.1 + CUDA 12.9），但 `release/dtk2610/` 索引**暂无适配轮**（0 个 das wheel），框架仍按 **dtk2604** 构建（dtk2604 已有 torch 轮到 2.11.0）。DTK-26.04.1（2026-08-27）为新增补丁版，配套驱动 rock-6.4.0-V1.7.0。26.10 发布摘要 + GEMM 实测（BW1100：+40~60%/+150% 未复现，26.10 算法选择不稳定）见 `dcu-dtk` §5.5。

## 0. 快速查表

| 问题 | 看哪节 |
|---|---|
| 这个环境（DTK+torch）该装什么版本的 vLLM/SGLang/算子包？ | §1 版本兼容矩阵 |
| 这个包在 DCU 上能不能用？验证过没？有什么限制？ | §2 软件能力矩阵 |
| 真机验证结果（哪些 import OK / 算子 smoke 通过） | §3 真机验证结果 |
| 换 torch 大版本 / 升 DTK 怎么操作？ | §4 升级 SOP |
| 怎么适配一个**新**软件？ | `dcu-software-adaptation-guide` |
| 源里有哪些包 / 怎么装 / 装不上排障？ | `dcu-env-packages` |

## 1. 版本兼容矩阵（DTK 26.04 体系）

> 核心规则：**torch 是基座，其他包绑 torch 标记**。换 torch 大版本要整体重装一整套（triton/flash_attn/vllm/算子包全跟着换），不能只升 torch。
> 标记格式：`torch251`=2.5.1 / `torch271`=2.7.1 / `torch280`=2.8.0 / `torch290`=2.9.0 / `torch2100`=2.10.0。

### 1.1 torch 基座（dtk2604）
| torch 版本 | 可用性 | 备注 |
|---|---|---|
| 2.5.1 | ✅ | 老组合，部分包已停更 |
| 2.7.1 | ✅ | <device-A> 基础镜像（py3.11） |
| 2.8.0 | ✅ | triton 有 torch280 版 |
| 2.9.0 | ✅ | <device-B> 基础镜像（py3.10），**主力组合** |
| 2.10.0 | ✅ | 最新，vllm-hcu/deep-ep/tilelang 新版绑它 |

### 1.2 框架包 torch 绑定矩阵（dtk2604，2026-09-01 枚举）
| 包 | 基础版本 | 可绑 torch 标记 | 验证状态（§3） |
|---|---|---|---|
| **vllm** | 0.9.2 / 0.11.0 / 0.15.1 / 0.18.1 | torch2100/251/271/290 | 框架本体，e2e 见 `dcu-vllm-deploy` |
| **vllm-hcu** | 0.21.0 | torch2100 | 插件 patch 层，需 vLLM 0.21 + torch2100 |
| **sglang** | 0.5.7 / 0.5.10rc0（release）/ 0.5.12/0.5.15（fork） | torch251/290（py3-none） | 框架本体，e2e 见 `dcu-sglang-source` |
| **sglang-router** | 0.3.0 / 0.3.2 | torch2100/290 | ✅ import OK（<device-B>） |
| **triton** | 3.1.0 / 3.3.0 / 3.4.0 / 3.5.1 / 3.6.0 | torch251/271/280/290/2100 | ✅ import + kernel smoke（<device-B>） |
| **onnxruntime** | 1.19.2 / 1.22.2 | torch251/271（1.19.2）/ py3-none（1.22.2） | ✅ import + 推理 smoke（<device-B>） |
| **deepspeed** | 0.15.4 / 0.18.2 | torch251/271/290 | ✅ import（<device-B>，需 hjson 依赖） |
| **megatron-core** | 0.13.2 / 0.15.4 | torch251/271/290 | ✅ import（<device-B>） |
| **dcu-megatron** | 0.13.0 / 0.15.0 | torch251/271/290 | ⚠️ 需 megatron.training 完整包 |
| **apex** | 1.5.0 / 1.7.0 | torch251/271/290 | ✅ import（<device-B>） |
| **transformer-engine** | 2.7.0 / 2.10.0 | torch251/271 | ✅ import（<device-A>, torch271） |

### 1.3 算子/通信包 torch 绑定矩阵（dtk2604）
| 包 | 基础版本 | 可绑 torch 标记 | 仓库 | 验证状态（§3） |
|---|---|---|---|---|
| **flash-attn** | 2.6.1 / 2.8.3 | torch2100/251/271/280/290 | release | 见 `dcu-flashattn-source` |
| **flash-mla** | 1.0.0 / 1.2.0 | torch2100/251/271/290 | release | MHA 解码加速 |
| **deepgemm** | 2.1.0 | torch2100/251/271/290 | release | 见 `dcu-deepgemm-source` |
| **aiter** | 0.1.2 / 0.1.3 | torch2100/290 | release | 见 `dcu-aiter-source` |
| **lightop** | 0.6.0 | torch2100/251/271/290 | release | 见 `dcu-lightop-source` |
| **lmslim** | 0.3.1 | torch2100/251/271/290 | release | 推理/压缩工具链 |
| **deep-ep** | 1.0.0 / 1.1.0 | torch2100/251/271/290 | release | ✅ import（<device-B>） |
| **mooncake-transfer-engine** | 0.3.7.post2 / 0.3.10.post1 | torch251/290/2100 | release | ✅ import（<device-B>） |
| **tilelang** | 0.1.9 | torch2100/290 | release | ⚠️ 需 tvm-ffi 依赖（源里无） |
| **fastsafetensors** | 0.3.2 | torch2100 | release | 权重加载（dlopen 双路径） |
| **causal-conv1d** | 1.5.4 / 1.6.1 | torch2100/271 | **rc4** | ✅ import（<device-A>, torch271） |
| **fast-hadamard-transform-cuda** | 0.1.0 | torch290 | **rc4** | ✅ import + 数值 smoke（<device-B>） |
| **mhc-fused-kernel** | 0.1.0 | torch290（**cp312 only**） | **rc4** | ✅ import + sinkhorn smoke（<device-B>, py3.12） |
| **hip-dsa** | 0.1.0 | torch290（**cp312 only**） | **rc4** | ✅ import + DSA API（<device-B>, py3.12） |
| **hip-dsa-kernels** | 0.1.0 | torch290 | **rc4** | ✅ install（<device-B>，纯 .so） |
| **hipdnn** | 1.0.1 | （cp311） | release | ✅ import（<device-A>, py3.11） |
| **xformers** | 0.0.33 | torch251/271（py3-none） | release | |

> **rc4 独有 14 包**（release/dtk2604 主索引没有，装包用 `customer/dtk2604-rc4` 仓库）：cupy、deep-ep-shca、fast-hadamard-transform-cuda、hip-dsa、hip-dsa-kernels、mhc-fused-kernel、mmcv、mmcv-full、mooncake-transfer-engine-rpc、mori、qwen-asr、spconv-rocm、torch-scatter、transformer-engine-hygon。

### 1.4 推荐组合（按场景）
| 场景 | 推荐组合 | 备注 |
|---|---|---|
| **vLLM 推理（主力）** | DTK26.04 + torch2.9.0 + vllm0.15.1(torch290) + triton3.5.1(torch290) + flash-attn2.8.3(torch290) | <device-B> 基线，最稳 |
| **vLLM 0.21 + vllm-hcu** | DTK26.04 + torch2.10.0 + vllm0.21 + vllm-hcu0.21.0(torch2100) | 新插件 patch 层，需 torch2100 |
| **SGLang 推理** | DTK26.04 + torch2.9.0 + sglang0.5.12/0.5.15(fork) | 见 `dcu-sglang-source` |
| **训练（Megatron）** | DTK26.04 + torch2.9.0 + megatron-core0.15.4 + dcu-megatron0.15.0 + apex1.7.0 + TE2.10.0(torch271 需 torch2.7.1) | TE 注意 torch 标记 |
| **MoE 推理（EP）** | 上述 + deep-ep1.1.0(torch290) + mooncake(torch290) | 专家并行 + KV 传输 |

## 2. 软件能力矩阵（哪些能跑 / 验证状态 / 限制）

> 验证状态图例：✅=真机 import+算子 smoke 通过 / ⚠️=import 通过但有依赖/环境限制 / ❓=仅静态分析未真机验证 / ❌=不可用
> 验证环境：<device-B>（BW100X8, py3.10, torch2.9.0）+ <device-A>（BW150, py3.11, torch2.7.1），2026-09-01。

### 2.1 推理框架
| 软件 | 能力 | 验证状态 | 限制/坑 |
|---|---|---|---|
| **vLLM** | LLM 推理（TP/PP/EP） | ✅（e2e 见 `dcu-vllm-deploy`） | 0.21+ 走 vllm-hcu 插件；page_size 须 64 倍数（gfx936） |
| **vllm-hcu** | vLLM DCU 插件 patch 层 | ❓（静态分析，需 vLLM0.21+torch2100 环境） | 字符串 patch 升级后静默失败，须重跑 apply-patches |
| **SGLang** | LLM 推理（投机解码/PD 分离） | ✅（e2e 见 `dcu-sglang-source`） | 0.5.12 用 is_dcu，0.5.15+ 用 is_hcu；page_size 须 64 倍数 |
| **sglang-router** | 请求路由/负载均衡 | ✅ import（<device-B>） | 纯 CPU 路由，无 GPU kernel |
| **onnxruntime** | ONNX 推理（ROCm/MIGraphX EP） | ✅ import + 推理 smoke（<device-B>） | ROCm EP 链 libgalaxyhip；MIGraphX EP 常更快 |

**vLLM 支持的模型结构（DAS 官方文档口径，2026-10-14，未真机实测）**：Qwen3ForCausalLM（DAS vLLM ≥v0.8.4）、Llama4（≥v0.8.5.post1）、GLM-4-0414；量化列 FP16/BF16/AWQ/GPTQ。新模型结构先对照此表判断「官方已适配/需自行适配」，再走 `dcu-model-portability` 流程。

### 2.2 算子/通信库
| 软件 | 能力 | 验证状态 | 限制/坑 |
|---|---|---|---|
| **triton** | Triton kernel 编译（hcu backend） | ✅ import + kernel smoke（<device-B>） | DTK 用 aillvm + libgalaxyhip；gfx936/938 768-vgprs |
| **flash-attn** | 稠密 attention | ✅（见 `dcu-flashattn-source`） | FA3 不可移植；page_size 约束 |
| **flash-mla** | MHA 解码加速 | ❓ | 见 `dcu-flashattn-source` 关联 |
| **deepgemm** | 通用/分组 GEMM | ❓（见 `dcu-deepgemm-source`） | fp8 需 gfx938；layout 陷阱 |
| **aiter** | MoE 融合算子 | ❓（见 `dcu-aiter-source`） | 四后端调度 |
| **lightop** | 自研融合算子 | ❓（见 `dcu-lightop-source`） | MoE 多量化路径 |
| **deep-ep** | MoE 专家并行通信 | ✅ import（<device-B>） | normal(prefill)/low-latency(decode) 两路径 |
| **mooncake** | KV cache 跨节点传输 | ✅ import（<device-B>） | EP buffer 按 torch 版本动态加载；需 RDMA 设备 |
| **tilelang** | TVM 系 DSL 编译器 | ⚠️ import 需 tvm-ffi（源里无） | kernel cache 命中跳过 lowering |
| **fastsafetensors** | 权重并行加载 | ❓ | dlopen libamdhip64（依赖 DTK 兼容层） |
| **causal-conv1d** | Mamba 因果卷积 | ✅ import（<device-A>, torch271） | 有 HCU 专属接口 causal_conv1d_fn_hcu |
| **fast-hadamard** | Hadamard 变换 | ✅ import + 数值 smoke（<device-B>） | 多尺寸特化（12N/20N/28N/40N） |
| **mhc-fused-kernel** | Megatron MHC 融合 | ✅ import + sinkhorn smoke（<device-B>, py3.12） | 需 py3.12（cp312 only）；顶层 fused_mhc.py（非包）；fused_sinkhorn 是行随机投影（非完整双随机） |
| **hip-dsa** | cudnn.DSA 稀疏注意力 HIP 替代 | ✅ import + DSA API（<device-B>, py3.12） | 需 py3.12（cp312 only）；无 PyTorch fallback；q_causal_offsets 非零报错 |
| **hipdnn** | PReLU 等小算子 | ✅ import（<device-A>, py3.11） | 链 libhipdnn_backend |

### 2.3 训练框架
| 软件 | 能力 | 验证状态 | 限制/坑 |
|---|---|---|---|
| **deepspeed** | ZeRO 训练 + MoE 推理 | ✅ import（<device-B>，需 hjson） | wheel 带 14 .hip 源码 |
| **megatron-core** | Megatron 训练核心 | ✅ import（<device-B>） | 本体几乎无 DCU 代码，特化在 dcu-megatron |
| **dcu-megatron** | Megatron DCU patch 适配层 | ⚠️ 需 megatron.training 完整包 | Patch 类机制（同 vllm-hcu 范式） |
| **apex** | fused kernel（adam/layernorm 等） | ✅ import（<device-B>） | 27 .so，无 C++ 源码在 wheel |
| **transformer-engine** | FP8 GEMM/attention | ✅ import（<device-A>, torch271） | 链 rccl+hipblaslt+hipblas+rocblas |

## 3. 真机验证结果（2026-09-01，<device-A>/<device-B>）

> 验证方法：起专用容器（pytorch 基础镜像 + 挂 hyhal + DCU 设备）→ pip install（--no-deps）→ import → 关键算子 smoke。
> 环境：<device-B>（BW100X8, py3.10, torch2.9.0）+ <device-A>（BW150, py3.11, torch2.7.1）+ <device-B> py3.12 容器（vllm0.18.1 镜像, py3.12.7, torch2.10.0），2026-09-01。

### 3.1 <device-B>（torch2.9.0, py3.10）
| 包 | install | import | 算子 smoke | 备注 |
|---|---|---|---|---|
| triton 3.5.1(torch290) | ✅ | ✅ | ✅ kernel max_err=0 | hcu backend 编译链路通（aillvm+libgalaxyhip） |
| onnxruntime 1.22.2 | ✅ | ✅ | ✅ 推理 max_err=6e-08 | ROCm EP 可用 |
| deep-ep 1.1.0(torch290) | ✅ | ✅ | — | |
| mooncake 0.3.7.post2(torch290) | ✅ | ✅ | — | |
| deepspeed 0.18.2(torch290) | ✅ | ✅（补 hjson） | — | --no-deps 缺 hjson，补后 OK |
| megatron-core 0.15.4(torch290) | ✅ | ✅ | — | |
| apex 1.7.0(torch290) | ✅ | ✅ | — | |
| sglang-router 0.3.2(torch290) | ✅ | ✅ | — | |
| fast-hadamard 0.1.0(torch290, cp310) | ✅ | ✅ | ✅ 数值 max_err=9.5e-06 | 须用 cp310 版（2605261719），cp312 版装不上 |
| hip-dsa-kernels 0.1.0 | ✅ | —（纯 .so） | — | |
| tilelang 0.1.9(torch290) | ✅ | ⚠️ 缺 tvm_ffi | — | tvm-ffi 源里无，wheel 不自带 |
| dcu-megatron 0.15.0(torch290) | ✅ | ⚠️ 缺 megatron.training | — | 需完整 megatron 包（非 core） |
| mhc-fused-kernel 0.1.0 | ❌ | — | — | 只有 cp312 wheel，py3.10 装不上 |
| hip-dsa 0.1.0 | ❌ | — | — | 只有 cp312 wheel，py3.10 装不上 |

### 3.2 <device-A>（torch2.7.1, py3.11）
| 包 | install | import | 备注 |
|---|---|---|---|
| causal-conv1d 1.5.4(torch271) | ✅ | ✅ | rc4 仓库 |
| transformer-engine 2.10.0(torch271) | ✅ | ✅ | |
| hipdnn 1.0.1(cp311) | ✅ | ✅ | |

### 3.3 <device-B> py3.12 容器（vllm0.18.1 镜像, py3.12.7, torch2.10.0）
> mhc/hip-dsa 只有 cp312 wheel，py3.10/3.11 环境装不上，用 vllm0.18.1-py3.12 镜像（torch2.10.0）验证。
| 包 | install | import | 算子 smoke | 备注 |
|---|---|---|---|---|
| mhc-fused-kernel 0.1.0(torch290, cp312) | ✅ | ✅ | ✅ fused_sinkhorn 行归一化 row_dev=1e-6 无 NaN | fused_sinkhorn 是行随机投影（非完整双随机，col 不收敛是预期）；fused_proj_rms 输出有限 |
| hip-dsa 0.1.0(torch290, cp312) | ✅ | ✅ | ✅ DSA API 对齐 cudnn.DSA | indexer_forward_wrapper + sparse_attention_backward_wrapper 都有 |
| hip-dsa-kernels 0.1.0 | ✅ | ✅ | — | 纯 .so 包 |

> 注：torch290 标记的 wheel 在 torch2.10.0 环境 import 成功（ABI 兼容），但严格对齐应找 torch2100 构建（源里 mhc/hip-dsa 暂无 torch2100 版）。

### 3.4 未真机验证（环境限制）
| 包 | 原因 | 怎么验证 |
|---|---|---|
| vllm-hcu | 需 vLLM0.21 + torch2.10.0 环境 | 需 vllm0.21 镜像 |
| fastsafetensors | 仅 torch2100 版 | 需 torch2100 环境（py3.12 容器可试） |
| flash-attn/flash-mla/deepgemm/aiter/lightop | 已有专门 source skill + e2e 验证 | 见各自 source skill |

## 4. 升级 SOP

### 4.1 换 torch 大版本（最常见）
```
1. 定目标 torch 版本（如 2.9.0 → 2.10.0）
2. 查 §1.2/§1.3 矩阵，确认目标 torch 标记下各包有对应版本
   （用 dcu-das-hub-api 复核最新：/api/pypi/package/<repo>/<name>?size=-1）
3. 整体重装一整套（不能只升 torch）：
   pip install torch torchvision triton flash-attn vllm ... \
     -i https://pypi.sourcefind.cn/release/dtk2604/
   （每个包选带目标 torch 标记的版本）
4. 验证：import 各包 + 关键算子 smoke（§3 方法）
5. 检查 torch ABI：import 编译型包无 undefined symbol: c10::...
```
**坑**：
- 部分包**没有目标 torch 标记的版本**（如 mhc/hip-dsa 只有 cp312，TE 2.10.0 只到 torch271）——换 torch 后这些包可能装不上，需找替代或等海光出新构建。
- **rc4 独有包**换 torch 时也要查 rc4 仓库（release 主索引没有）。

### 4.2 升 DTK 版本（如 dtk2604 → dtk26042）
```
1. 装新 DTK（dcu-dtk）
2. 用新 DTK 索引重装整套：-i https://pypi.sourcefind.cn/release/dtk26042/
3. 验证 gfx（rocminfo 宿主机）+ import + 算子 smoke
```
**坑**：DTK 升级可能改变 gfx 标识/库版本，编译型包需重编（hipcc --offload-arch）。

### 4.3 升框架版本（如 vLLM 0.15 → 0.21）
```
1. 确认新框架版本的 DCU 适配形态（fork 还是 vllm-hcu 插件）
   - 0.15/0.18：fork（dcu-vllm-source）
   - 0.21+：vllm-hcu 插件（dcu-vllm-hcu-source）
2. 装新框架 + 对应 vllm-hcu（若插件形态）
3. 重跑 vllm-hcu-apply-patches（字符串 patch 升级后可能静默失败）
4. e2e 验证（起服务跑真实模型）
```
**坑**：vllm-hcu 的字符串 patch 在框架升级后 `old` 子串匹配不到时**不报错**（静默跳过）——升级后必须重跑 apply-patches 看日志。

## 5. 交叉引用

| 场景 | 用哪个 skill |
|---|---|
| 源里有哪些包 / 怎么装 / 装不上排障 | `dcu-env-packages` |
| 怎么适配一个**新**软件（方法论） | `dcu-software-adaptation-guide` |
| 查源里某包版本 / 下 wheel / 跨仓库枚举 | `dcu-das-hub-api` |
| 具体包源码结构（10 个案例） | `dcu-vllm-hcu-source` / `dcu-triton-source` / `dcu-ort-source` / `dcu-deepep-source` / `dcu-mooncake-source` / `dcu-fastsafetensors-source` / `dcu-tilelang-source` / `dcu-mhc-kernels-source` / `dcu-hipdsa-source` / `dcu-training-frameworks-source` |
| vLLM 各版本 fork 源码 | `dcu-vllm-source` |
| SGLang 源码 + e2e | `dcu-sglang-source` |
| vLLM 部署调优 | `dcu-vllm-deploy` |
| DTK 安装/布局 | `dcu-dtk` / `dcu-dtk-layout` |
| 真机验证方法 | `dcu-skill-on-device-verification` |

## 使用须知

- 本 skill 是**查表总览**，版本/验证状态是 **2026-09-01 快照**——装前用 `dcu-das-hub-api` 复核最新（轮子会持续更新）。
- **torch 是基座**：换 torch 大版本要整体重装一整套，不能只升 torch（§4.1）。
- **rc4 独有 14 包**：release 主索引没有，装包用 `customer/dtk2604-rc4` 仓库。
- **验证状态**：✅=真机验证过 / ⚠️=有依赖/环境限制 / ❓=仅静态分析——❓ 的包用前建议真机验证（§3 方法）。
- **不存性能基线**：本 skill 只存"能不能用/怎么配"，性能数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
