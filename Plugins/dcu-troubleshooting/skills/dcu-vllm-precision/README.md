# dcu-vllm-precision — vLLM 推理精度异常定位 skill

> 定位 vLLM 推理输出与 transformers(HF) 基准对不齐（乱码 / 全 NULL / token 异常 / 精度不齐）到具体算子。指路是天级→小时级：先从模型结构自动探测，逐层对比，再层内钻取到算子，最后出报告。

## 何时用
- 用户抱怨 vLLM 推理输出异常：全 NULL / 乱码 / token id 异常（如全 188）、与 HF 输出对不齐、逐层精度衰减。
- 输入必需：本地模型路径（含 `config.json`）。
- 本 skill 是**算子级定位**，不是框架级排错（部署/性能类走 `dcu-vllm-deploy` 等）。

## 七段流程（SKILL.md 详细）
1. **平台探测** — `lib/probe_platform` 确认 DCU/gfx/HIP；DCU 上 `is_cuda()=False` 改变 vLLM 调度（知识库 hygon_dcu.md 有专节）。
2. **模型配置解析** — `lib/load_model_profile` 读出 arch/MoE/num_layers/自定义 py。
3. **代码路径** — `lib/resolve_code_paths` 给出 vLLM 模型文件/fused_moe 目录/关键算子入口（含 sigmoid+bias 必走 fused_gate 的结构事实）。
4. **粗粒度逐层** — `examples/compare_layers.py` 对比每层 hidden_states，找首个发散层。
5. **层内钻取** — `--op attn/mlp/router` 在该层内拆解，定位具体算子。
6. **绕过验证** — 环境变量对照（`VLLM_ENABLE_MOE_FUSED_GATE=0` 等），确认根因。
7. **报告** — `examples/generate_report.py` 落盘 `report.md` + `verdict.json` + 证据，提示人工 review 回写知识库。

## 前置
- vLLM + transformers + torch 已装；模型已下载到本地。
- 卡由用户 `HIP_VISIBLE_DEVICES` 前置指定（脚本继承、不自动选卡）。
- 起容器须挂 `/opt/hyhal:/opt/hyhal:ro`（DCU 可见）。

## 目录结构
```
dcu-vllm-precision/
├── SKILL.md          # agent 向工作流（七段流程 + 工具/知识指针）
├── README.md         # 本文件（用户向概述）
├── lib/              # 工具库：config_loader / path_resolver / hook_manager / tensor_compare / model_probe / platform_probe / config_patch
├── references/       # 知识库：arch_index / precision_known_issues / vllm_forward_paths / hook_pitfalls / model_config_issues / hygon_dcu / compare·probe·report
├── examples/         # 入口脚本：compare_layers.py / generate_report.py / probe_model.py / quickstart_antangelmed.py
├── run/              # 标准启动：run_vllm.sh / run_hf.sh
└── tests/            # lib 单测（pytest）
```

## 运行前提/脚本运行前提
| 脚本 | 前提 |
|---|---|
| `examples/compare_layers.py` / `generate_report.py` / `probe_model.py` | vLLM + transformers 已装；模型本地路径；用户指定空闲卡（HIP_VISIBLE_DEVICES） |
| `examples/quickstart_antangelmed.py` | 最先跑：冒烟环境 + 端到端 token 对比 |
| `run/run_vllm.sh` | vLLM 离线推理启动（HIP_VISIBLE_DEVICES 前置；自带 ENV=0 绕过 fused_gate 对照） |
| `run/run_hf.sh` | transformers(HF) 基准启动（自动补 config 缺失字段；大模型多卡 device_map=auto） |

## 使用须知
- **知识库历史案例只是个例**，不代表当前问题就是它；根因须用运行时探针独立验证，不预判。
- **不自动回写知识库**：报告后请用户 review，确认结论后再追加到 `references/precision_known_issues.md`。
- 卡由用户指定，脚本继承不自动选卡；挂 `/opt/hyhal` 容器内才见 DCU。
- 默认不设修复环境变量；对照验证用 `--env` 显式开。