# R067 MTP4 — K100AI 27B W8A8 单卡 decode 补丁（Qwen3.6-27B）

## 来源
公开 repo：DocPang/qwen36-27b-k100ai-w8a8-optimization（R067 冠军路径）
验证环境：Hygon K100AI / gfx928 / vLLM 0.18.1 / DTK 26.04 / metax-tech/Qwen3.6-27B-W8A8

## 安全说明
纯代码补丁，无密码 / 密钥 / 凭据。列出的脚本均为通过 `bash -n` 语法检查的
启动/编译脚本；使用前按真实环境核对 GPU/端口/容器名。

## 内容含义
- `sitecustomize.py`：R067 的 vLLM runtime monkey-patch 注入。M=1 的 lm_head 走
  INT32 scaled_mm 快路径（~10% faster），draft M=1 用 K1024 BN64/kpack2 两级
  selector，M=5 的 verifier 走 INT32 scaled_mm 精确路径（bitwise guard）。
  环境开关：`K100_27B_LM_HEAD_W8A8=1` / `K100_27B_LM_HEAD_W8A8_MAX_M=4` /
  `K100_27B_LM_HEAD_W8A8_M5=...`（详见文件头部注释）。
- `candidate_head.py`：R067 selector 的两级候选路径。K_FULL=5120、K_SELECTOR=1024、
  VOCAB=248320、CANDIDATE_K=2048。先 coarse Top2K，再 exact full-K 打分。默认
  `_CFG`（bm32/bn32/bk512/w8/w2/kp1/mmac5-ds6）与 `_SELECTOR_CFG`
  （bm32/bn64/bk512/w8/w2/kp2/mmac5-ds6）位-identical（748/748 验证）。
- `Qwen3.6-27B-W8A8-DCU.config.json`：已验证的 DCU config（compressed-tensors
  W8A8 格式），与上游 fff7ade... 原始 config 不同（上游 lm_head 仍 BF16，本路径
  加载后量化为 INT8）。
- `quickstart_mtp4.sh` / `serve_mtp4.sh`：一键准备/Docker 起服务启动脚本。
  `serve_mtp3.sh` / `serve_nomtp.sh` / `serve_mtp2.sh` 为对照路径版本。
- `benchmark_fixed_512.py`：R067 验收基准（fixed-512 六轮、mean/median、SHA256
  门禁；warmup 前一轮不计入稳态）。R067 六轮 median 42.2978 tok/s。
- `apply_dcu_config.py`：安全应用 DCU config；校验上游/DCU 两 SHA256 后覆盖，
  不改 safetensors。备份在 `config.json.upstream.bak`。
- `build_k100_int8_gemv_v6.py` + `k100_int8_gemv_v6.hip`：K100AI 原生 INT8 GEMV
  源码与本地 hipcc 编译脚本（ABI/编译环境不同，不发布预编译 .so）。

## 使用前提（离开 K100 AI / gfx928 / vLLM 0.18.1 / DTK 26.04 须重新验证）
- 固定镜像：`<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423@sha256:13ce550...`
- 固定模型：`metax-tech/Qwen3.6-27B-W8A8`；repo 的 `-DCU` 目录名只是建议本地目录，
  脚本只改 config.json，不重新量化 safetensors。
- 该补丁的 MTP4 路径依赖 `num_speculative_tokens=4`（`--speculative-config`），
  是否启用 CUDAGraph/compile 由启动脚本控制。
- R067 的两级 selector 是 **draft-only 投机优化**；目标模型仍用完整 248320 词表
  验证最终 token，不是永久词表裁减。

## 关联索引（方法论合并到 dcu-vllm-runtime-patch SKILL.md）
- `OPTIMIZATION_NOTES.md`（repo docs/）关键结论已并入 SKILL.md §10：
  真实 runtime shape 优先、Small-M W8A8 是 K100AI 核心问题、HCU MMAC 非 AMD MFMA、
  bandwidth 测量改变策略、MTP 必须独立 shape family 调优等。
- 脚本工具本身不进 SKILL.md（按纪律只收方法论+坑），留 references 目录供直接取用。