# HCU 官方部署 cookbook 上游引用（dcu-model-deploy-catalog 的 P0 更新源）

> 仓库：https://github.com/HYGON-AI/inference-cookbook-das （61⭐，持续更新）
> 本文件是 `dcu-model-deploy-catalog` 的**官方上游指针**：本地 100 篇手册是内部快照（BW1100 为主），
> 官方 cookbook 是持续更新的**模型×框架×卡型（K100_AI/BW1000/BW1100）验证矩阵**，两者互补：
> 本地覆盖 CV/科学计算/时序等全类别，官方覆盖最新 LLM/VLM/视频生成的**三卡型验证状态**。
> 版本锚点（fetched 2026-10-01）见 `references/hygon-upstream/upstream-snapshot.json` 的 `cookbook_matrix`。

## 更新机制（每次使用前确保最新）

```bash
cd <agent-home>/skills/dcu
python3 scripts/hygon-upstream-sync.py --check   # 快检（仓库列表 + 关键版本锚点）
python3 scripts/hygon-upstream-sync.py --full    # 刷新 cookbook 矩阵（解析 README HTML 表格）
```

`--full` 会对比上次快照，打印「cookbook 新增已验证: <模型>@<框架> -> <卡型>」，据此把新模型配方补进本目录 `references/`。
触发时机：**任何涉及模型部署/配方查询的任务，先跑 `--check`；查新模型（尤其最新 SOTA）前跑 `--full`**。建议每周包一次 `--full` 更新矩阵。
数据说明：README 全文缓存于 `<agent-home>/cache/hygon-upstream/inference-cookbook-das.md`；快照只存矩阵元数据。

## 模型验证矩阵（fetched 2026-10-01）

以下为官方 cookbook「模型×框架×卡型」已验证（✅）状态，`/` 代表卡型列。
文档 URL 模板：`docs/model-deployment/<framework>/<model>.md`（在仓库 `main` 分支）。

| 模型 | 框架 | K100_AI | BW1000 | BW1100 |
|---|---|---|---|---|
| Qwen3.8 | vLLM / SGLang | ✅ | ✅ | ✅ |
| Qwen3.6 | vLLM / SGLang | ✅ | ✅ | ✅ |
| Qwen3.5 | vLLM / SGLang | ✅ | ✅ | ✅ |
| Qwen3 | vLLM / SGLang | ✅ | ✅ | ✅ |
| Qwen3-VL | vLLM | 🚧 | ✅ | 🚧 |
| Qwen3-VL | SGLang | ✅ | ✅ | ✅ |
| Qwen3-Coder | vLLM | 🚧 | 🚧 | ✅ |
| Qwen3-TTS | vLLM-Omni | 🚧 | ✅ | ✅ |
| Qwen2.5-VL | vLLM | 🚧 | ✅ | ✅ |
| Qwen2-VL / Qwen2-72B | vLLM | - | ✅ | - |
| QwQ | vLLM | ✅ | ✅ | ✅ |
| DeepSeek-V3.2 | vLLM | - | ✅ | ✅ |
| DeepSeek-V3.2 | SGLang | ✅ | ✅ | ✅ |
| DeepSeek-V3 | vLLM | - | - | ✅ |
| DeepSeek-V3 | SGLang | - | ✅ | ✅ |
| DeepSeek-R1 | vLLM / SGLang | ✅ | ✅ | ✅ |
| DeepSeek-V4 | vLLM / SGLang | 🚧 | ✅ | ✅ |
| GLM-5.2 | vLLM / SGLang | 🚧 | ✅ | ✅ |
| GLM-5.1 | vLLM / SGLang | 🚧 | ✅ | ✅ |
| GLM-5 | vLLM / SGLang | - | ✅ | ✅ |
| GLM-4.7 | vLLM | ✅ | ✅ | ✅ |
| Hy3 (Tencent) | vLLM | - | - | ✅ |
| Hy3 (Tencent) | SGLang | - | ✅ | ✅ |
| Hy4 | vLLM | - | - | ✅ |
| Kimi-K2.6 | SGLang | - | - | ✅ |
| Kimi-K2.5 | vLLM | - | - | ✅ |
| Kimi-K2.5 | SGLang | - | ✅ | ✅ |
| Kimi-K2 / Kimi-K3 | vLLM / SGLang | - | - | ✅ |
| MiniMax-M2.7 | SGLang | - | - | ✅ |
| MiniMax-M2.5 | vLLM / SGLang | ✅ | ✅ | ✅ |
| MiniMax-M3 | SGLang | - | - | ✅ |
| MiniMax-H3 | SGLang Diffusion | 🚧 | ✅ | ✅ |
| MiMo-V2.5-Pro | SGLang | - | - | ✅ |
| MiMo-V2-Flash | SGLang | ✅ | ✅ | ✅ |
| Step-3.5 / Ling-1T | - | 🚧 | 🚧 | 🚧 |
| MAGI-2-preview | vLLM | 🚧 | ✅ | 🚧 |
| Wan2.1-I2V | Wan-DAS | ✅ | ✅ | ✅ |
| Wan2.2-I2V | Wan-DAS | ✅ | ✅ | ✅ |
| Wan2.2-I2V | vLLM-Omni | - | ✅ | ✅ |
| Wan2.2-T2V | Wan-DAS / vLLM-Omni | - | ✅ | ✅ |
| Wan2.2-T2V | SGLang Diffusion | - | - | ✅ |
| Wan2.2-TI2V | Wan-DAS | ✅ | ✅ | ✅ |
| BGE | vLLM / Infinity | - | ✅ | - |

> 完整 85 行矩阵（含 🚧 wip 行）见 `upstream-snapshot.json`；★ 标注的卡型列在某行首次出现 ✅ 即为新增验证。
> 注：表里含本地已覆盖模型（Qwen3/GLM4.7/DeepSeek-R1 等）的**三卡型交叉验证**，可作为本地手册的卡型适配补全参考。

## 与本地 catalog 的差异/使用方法

1. **本地 `references/llm.md` 等** 是内部《海光 DCU 模型部署手册》抄录（含本地验证过的具体命令/坑），官方 cookbook 是**验证状态矩阵**+链接到官方文档。
2. 查询流程：先按本地 catalog 分类文件查配方 → 若模型在官方矩阵且有 ✅ 但本地无详细命令，则打开官方对应 `docs/model-deployment/<框架>/<模型>.md` 补全。
3. 官方矩阵卡型列（K100_AI/BW1000/BW1100）与本地手册常标 BW1100 不一致时，以**官方矩阵的三卡型交叉验证**为准做适配判断。
4. 量化工具 `one-click-quant`（HF 社区组件 llmcompressor 开发）：BF16→W8A8 FP8/INT8、Block FP8→BF16、DeepseekV3/Qwen3.5 MoE 定制、`model_free_ptq`（免校准免 transformers）、slimquant w4a8、MoE-Quant（单机大权重 GPTQ）；可作为本地量化部署的补充工具（对应 `dcu-model-deploy-catalog` 的量化场景）。
