---
name: dcu-inference
description: "DCU 推理框架适配技能集。覆盖海光 DCU 推理框架技能集：vLLM/SGLang 部署与调优、精度定位、自定义算子、MIGraphX 与模型部署目录。"
version: 1.5.1
author: DCU Toolkit Contributors
license: MIT
---

# DCU 推理框架适配 — 域级技能包

## 工具介绍

海光 DCU 推理框架技能集：vLLM/SGLang 部署与调优、精度定位、自定义算子、MIGraphX 与模型部署目录。本目录为 Kits Platform Skill Tool `dcu-inference` 的发布单元，包含 25 个子 skill：

| skill | 说明 |
|---|---|
| add-jit-kernel | Add a lightweight upstream SGLang JIT CUDA kernel with bindings, corre… |
| add-sgl-kernel | Add a heavyweight upstream sgl-kernel AOT CUDA/C++ operator with regis… |
| dcu-cosyvoice-opt | CosyVoice2(0.5B TTS)在DCU上的部署与推理优化 |
| dcu-fastsafetensors-source | fastsafetensors DCU 适配源码结构——safetensors 并行加载（gpu_compat.h 运行时 dlopen C… |
| dcu-inference-infra | DCU推理infra子域路由器，分发到各skill |
| dcu-libtorch | DCU上C++LibTorch推理部署：libtorch_hip+dcc编译与TorchScript |
| dcu-lmcache | 海光DCU上用LMCache做KV Cache卸载优化推理的方法论与实战 |
| dcu-migraphx-custom-op | 海光DCU上用MIGraphX写自定义算子（C+++HIP核函数） |
| dcu-migraphx-driver | 海光DCU上MIGraphX命令行工具migraphx-driver源码解析与扩展 |
| dcu-model-deploy-catalog | DCU 模型部署目录：100 个已验证模型（LLM/VLM/Embedding/CV/图像视频生成/时序/科学计算）的部署配方速查——按模型… |
| dcu-model-deploy-verify | DCU模型部署：先真机实测跑通再写文档，含PyTorch范式与Uni-Mol实测 |
| dcu-model-portability | 评估任意PyTorch模型能否即插即用部署到海光DCU并产出部署手册。 |
| dcu-mooncake-source | Mooncake Transfer Engine DCU 适配源码结构——KV cache 跨节点 RDMA 传输（libtransfer_… |
| dcu-ninfer-port | NInfer to DCU(gfx936) port: SIMT-first + 4-layer quant. |
| dcu-ort-source | ONNX Runtime DCU 适配源码结构——ROCm EP 链接 DTK libgalaxyhip + MIGraphX EP + 二… |
| dcu-pytorch-source | DCU源码编译低版本PyTorch与LibTorch提取 |
| dcu-sglang-source | SGLang DCU源码结构与海光适配点索引 |
| dcu-training-frameworks-source | DCU 训练框架适配合集——deepspeed（wheel 带 14 个 .hip 源码）+ megatron-core + dcu-meg… |
| dcu-vllm-custom-op | vLLM DCU版从零开发自定义算子流程 |
| dcu-vllm-deploy | DCU上vLLM部署与调优经验速查 |
| dcu-vllm-hcu-source | vllm-hcu 包源码结构——vLLM 在 DCU/HCU 上的插件式 patch 层（105 个 patch + HCUPlatform… |
| dcu-vllm-op-replace | vLLM DCU版自定义算子替换原生算子流程 |
| dcu-vllm-runtime-patch | DCU 上 vLLM 运行时 monkey-patch 注入专用 kernel 的标准接入范式：环境变量开关 + 形状表 + 上游 fall… |
| dcu-vllm-source | vLLM 各版本 DCU 源码结构与适配点索引（合并 0.15/0.18/0.21.0/0.23.0/0.26 五版） |
| sglang-diffusion-add-model | Integrate an image or video diffusion model from Diffusers or a refere… |

## 安装与使用

- **单 skill 使用**：Agent 按上表选定 skill 后，读取 `skills/<skill>/SKILL.md` 并按其流程执行（子 skill 自包含，含 scripts/ 与 references/）。
- **统一路由**：跨域任务优先读仓库根 `SKILL.md`（包级路由表）或 `dcu-core/skills/dcu-skill-router`（全量索引）。
- **安装**：见仓库根 `README.md` / `install.sh`（支持 Claude Code / OpenCode / Codex 等多框架，项目级或全局）。
- **版本**：与包根 SKILL.md 一致（1.5.1）；升级以包内版本为准。
