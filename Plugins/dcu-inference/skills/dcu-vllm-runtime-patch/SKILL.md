---
name: dcu-vllm-runtime-patch
description: "DCU 上 vLLM 运行时 monkey-patch 注入专用 kernel 的标准接入范式：环境变量开关 + 形状表 + 上游 fallback 安全网 + bit-identical 一致性门禁，不改 vLLM 源码即可做单卡/单并发 decode 优化"
version: 1.1.0
author: DCU Engineer
license: MIT
---

# DCU vLLM 运行时 monkey-patch 注入范式

> 用途：当你要在 **不改 vLLM 源码** 的前提下，为某款 DCU 机型（如 K100 AI / gfx928）的
> 某个模型（如 Qwen3.6 W8A8）注入**专用 INT8 Linear / lm_head / MoE / 融合 kernel**，
> 并希望**只对特定 decode 形状生效、其余形状安全 fallback 上游、且保证数值一致**时，
> 用本 skill 的范式搭补丁。它来自 K100 AI Qwen3.6 实战并抽象为通用模板。
>
> 为什么不用改源码：monkey-patch（通过 `PYTHONPATH` 注入 `sitecustomize.py`）能 hook
> 任意 vLLM 内部函数，零侵入、可一键开关、可按形状精确控制、天然保留上游 fallback 安全网。
> 这套「补丁默认关、只对 decode 形状开、其余 fallback」是 DCU 推理优化推荐的**标准接入范式**。

---

## 1. 适用场景

- 目标：**单卡 / TP=1 / 单并发 c1 / Agent 交互速度**（decode 是固定小 shape 问题）。
- 手段：针对真实 `(M,K,N)` 形状搜专用 Triton/AITER config，或挂 native HIP kernel，
  替换 vLLM 通用 heuristic 在 DCU 上偏慢的那些小 GEMM / MoE / lm_head / 融合算子。
- 约束：**数值必须与上游 bit-identical**（用 `relative_l2==0` 或固定输出 SHA256 门禁）。

不适用：要改 vLLM 算子注册/后端分发结构的，走 `dcu-vllm-op-replace`（替换算子）或
`dcu-vllm-custom-op`（从零开发）；要写 HIP/TileLang kernel 的，走 `dcu-kernel-dev`。
本 skill 只管「如何把专用 kernel 干净地挂进运行中的 vLLM」。

---

## 2. 核心范式（5 条原则，务必遵守）

1. **环境变量总开关，默认全关**：每个专用路径一个 `DCU_PATCH_*_ENABLED` 与
   `DCU_PATCH_*_MAX_M`；不设置就不注入，行为等同原版 vLLM。
2. **只替换 decode 形状，其余 fallback 上游**：专用 config 只对 `M <= MAX_M`
   （如 1~4）的形状生效；未命中形状一律调用 `_orig_*`（上游 heuristic），保留安全网。
3. **形状表必须 bit-identical 门禁**：填进形状的专用 config 要先用离线微基准验证
   与上游输出 `relative_l2==0`（或固定输入 SHA256 一致）。否则宁可 fallback。
4. **所有 hook 带 `register_fake`**：用 `torch.library.custom_op` 定义的新算子必须
   注册 `register_fake`，否则 `torch.compile` / inductor CUDAGraph 捕获会崩。
5. **先对齐测速口径再比数**：固定输入/输出 token、单并发、热态、统一是否启用
   compile/CUDAGraph 后，才对比优化前后；旧报告冷路径数字不可直接当基线。

---

## 3. 标准接入步骤

1. **探真实形状分布**（patch 关闭时）：用 `scripts/trace_linear_shapes.py` 安装
   (M,K,N) 计数器（SIGUSR1 dump），跑一次真实 decode，找出主导形状
   （如 `(1,2048,12288)`、`(1,2048,9216)`）。决定为哪些形状搜 config。
2. **离线搜专用 config**：对这些 `(M,K,N)` 搜 Triton `bm/bn/bk/warps/waves`（或 AITER
   `BLOCK_SIZE_*`），逐个验证 `relative_l2==0`。把结果填进形状表。
3. **写 `sitecustomize.py`**：复制 `scripts/sitecustomize_skeleton.py`，在
   `_shape_specific_config` 填表，在 `_apply_linear_patch` 里用惰性导入 hook
   目标函数（如 `triton_scaled_mm`），命中形状走专用 kernel、否则 `_orig_*`。
4. **挂启动脚本**：`docker run ... -e PYTHONPATH=/opt/k100-runtime-patch:... -v $PATCH:/opt/k100-runtime-patch:ro ...`，并设 `DCU_PATCH_*_ENABLED=1`、`DCU_PATCH_*_MAX_M=N`。
5. **一致性门禁**：连续同卡测速 3 轮，固定输出 SHA256 一致；仅首次 Triton 编译变慢
   的请求单独标记剔除，不得删后续稳定结果；禁用「不同 GPU 一次性结果替代同卡连续测试」。
6. **回退开关**：出问题时 `DCU_PATCH_*_ENABLED=0` 即恢复上游，无需改镜像。

---

## 4. 目录结构

```
dcu-vllm-runtime-patch/
├── SKILL.md
├── scripts/
│   ├── sitecustomize_skeleton.py   # 通用骨架：env 开关 + 形状表 + fallback + register_fake 模板
│   └── trace_linear_shapes.py       # 诊断：(M,K,N) 调用分布探针，SIGUSR1 dump
└── references/
    ├── r180_nomtp_sitecustomize.py          # 实战完整版：K100 AI Qwen3.6 W8A8 无MTP纯模型补丁(R180)，含 INT8 Linear/lm_head/native MoE 三 hook + K100_* 开关
    ├── r184_mtp3_fused_qkvz_ba_sitecustomize.py  # 实战完整版：上 + QKVZ+BA 融合，MTP3 版(R184)
    ├── start_r180_nomtp_gpu1_speedtest.sh          # 实战启动脚本：R180 无MTP 服务(GPU1/端口7995, cudagraph 1)
    ├── start_final_r184_mtp3_gpu0_speedtest.sh      # 实战启动脚本：R184+MTP3 服务(GPU0/端口7994, cudagraph 1 4, speculative-config)
    ├── k100_channelwise_int8_moe.hip                # 实战 native INT8 MoE kernel 源码(gate_up/down fused GEMM, 固定 template 吃满 decode 形状)
    ├── build_k100_channelwise_int8_moe.py           # 实战编译脚本：patch AITER 模板 group_size_n==1 + hipcc 编译为 .so（绝对路径已脱敏为占位）
    ├── E=256,N=512,device_name=K100_AI.json         # 实战 MoE 调优配置(per-M BLOCK_SIZE 表, 合法 JSON 可被 vLLM 直接挂载)
    └── E=256,N=512,device_name=K100_AI.json.README.md  # 上述 JSON 的 provenance + 使用说明(不污染 JSON 本体)
```

- `scripts/` 两个文件均 `py_compile` 通过（语法校验）；真实 hook 的 torch/vllm 导入在 `_apply_*` 内惰性进行，因此骨架可不依赖运行环境直接编译。
- `references/` 两个文件是**从真实复现工程逐字节提取**的完整补丁（顶部已加 provenance + 安全说明 + 使用前提），用于「直接拿可用」而非只看骨架；两者亦 `py_compile` 通过。离开 K100 AI / gfx928 / vLLM 0.18.1 / DTK 26.04 须重新验证形状表与 native kernel，不可照搬。

**实战完整版索引（references/）**
| 文件 | 对应演进阶段 | 包含的专用 hook | 启用环境变量 |
|---|---|---|---|
| `r180_nomtp_sitecustomize.py` | R180 无 MTP 纯模型 | INT8 Linear（`K100_SC_LINEAR_A8W8`）+ INT8 lm_head（`K100_LM_HEAD_W8A8`）+ native INT8 MoE（`K100_NATIVE_INT8_MOE`） | 三者各设 `=1`，`MAX_M` 默认 4/4/32 |
| `r184_mtp3_fused_qkvz_ba_sitecustomize.py` | R184 + MTP3 | 上三项 + QKVZ+BA 融合（`K100_FUSED_QKVZ_BA=1`） | 四项全开 |
| `r067_mtp4_qwen36_27b_sitecustomize.py` | R067 + MTP4（Qwen3.6-27B） | INT8 lm_head（`K100_27B_LM_HEAD_W8A8`）+ 两级 selector（K1024/B64/kpack2） | `K100_27B_LM_HEAD_W8A8=1`，`MAX_M` 默认 4 |

> 注：各阶段实测 tok/s 属性能基线，不入本 skill；如需引用转 `dcu-perf-cases` Case 10（K100 AI Qwen3.6 W8A8 实战）。

**实战启动脚本（references/）**
| 文件 | 对应阶段 | 关键参数 | 说明 |
|---|---|---|---|
| `start_r180_nomtp_gpu1_speedtest.sh` | R180 无 MTP | `HIP_VISIBLE_DEVICES=1`、端口 7995、`--cudagraph-capture-sizes 1`、`K100_SC_LINEAR_A8W8=1`/`K100_LM_HEAD_W8A8=1`/`K100_NATIVE_INT8_MOE=1`(MAX_M 4/4/32)、无 `--speculative-config` | 单卡纯模型服务，TP=1 |
| `start_final_r184_mtp3_gpu0_speedtest.sh` | R184 + MTP3 | `HIP_VISIBLE_DEVICES=0`、端口 7994、`--cudagraph-capture-sizes 1 4`、`K100_FUSED_QKVZ_BA=1`、附加 `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3,...}'` | 隔离容器名，避免被实验清理脚本误删 |

> ⚠️ 启动脚本内 GPU/端口/容器名是为复现测试隔离设定的；照搬前按需调整，且**勿误停生产服务**。脚本已通过 `bash -n` 语法检查、无密码密钥。

> ⚠️ references 配套 artifact 说明：
> - **native INT8 MoE kernel 现已收为 references**（`k100_channelwise_int8_moe.hip` + `build_k100_channelwise_int8_moe.py`，build 脚本绝对路径已脱敏为 `<PROJECT_ROOT>` / `<AITER_BUILD>` / `<VLLM_SITE_PACKAGES>`，使用前按真实环境回填）。build 脚本负责 patch AITER 模板 `group_size_n==1` 为真 per-output-channel 并 hipcc 编译为 `.so`。
> - **MoE 调优 JSON 现已收为 references**（`E=256,N=512,device_name=K100_AI.json`，逐字节原样、合法 JSON 可被 vLLM 直接挂载；其 provenance/使用说明在同名 `.README.md`，不污染 JSON 本体）。挂载到 vLLM `fused_moe/configs/` 即可，无需再回复现工程。
> - 至此 `references/` 已**完全自包含**：补丁（R180/R184）+ 启动脚本（R180/R184）+ native MoE kernel 源码 + 编译脚本（路径脱敏）+ MoE 调优 JSON，后人无需回本地 `<local_repro_pkg>_2026-08-07.gz` 即可拿到完整可落地材料（仅 build 脚本的 `<PROJECT_ROOT>`/`<AITER_BUILD>`/`<VLLM_SITE_PACKAGES>` 占位需按真实环境回填）。
> - native MoE 的 AITER 模板**必须** patch `group_size_n==1` 为真 per-output-channel，否则数值错（详见 perf-cases Case 10）。

---

## 5. 骨架要点（节选自 `scripts/sitecustomize_skeleton.py`）

- env 开关示例：
  ```python
  _LINEAR_ENABLED = os.getenv("DCU_PATCH_LINEAR_A8W8", "0") == "1"
  _LINEAR_MAX_M  = int(os.getenv("DCU_PATCH_LINEAR_A8W8_MAX_M", "4"))
  ```
- 形状表（只放 bit-identical 验证过的形状）：
  ```python
  def _shape_specific_config(m, k, n):
      table = {
          # (1, 2048, 12288): {"bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2},
      }
      return table.get((m, k, n))
  ```
- hook 结构：`_apply_linear_patch()` 用 `importlib.import_module` 惰性拿到
  `triton_scaled_mm` 与你的 GEMM，定义 `torch.library.custom_op(..., device_types="cuda")`
  并 `register_fake`，命中形状用专用 kernel，否则 `_orig_triton_scaled_mm(...)`。
- 自定义注入算子（如 fused kernel）同理：hook 模型 `forward`
  （如 `Qwen3_5GatedDeltaNet.forward`）分流 fused / 上游两条路径。

---

## 6. 实测环境前提（严格遵守）

本范式在以下环境被真实跑通并验证：

- 机型：海光 **K100 AI（RDNA gfx928，64KB LDS / 120 CU / INT4·INT8 Tensor Core）**
- 框架：vLLM **0.18.1**（镜像 `<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`）
- DTK：**26.04**
- 模型：`Qwen3.6-35B-A3B-W8A8-DCU`（compressed-tensors W8A8）

> ⚠️ 离开 K100 AI / gfx928 / 该 vLLM 版本 / DTK 26.04 后，Triton tile、native HIP
> kernel、MoE 形状表**必须重新验证**才能复用；其他机型（BW 系等）的 LDS/ CU 数不同，
> 形状表与 template 参数不可直接照搬。

---

## 7. 实战样例索引

完整「K100 AI Qwen3.6 W8A8 单卡」演进（原版~20 → 专用算子纯模型~53.5 无 MTP →
叠加 MTP3 ~85/峰值107）、8 个 `K100_*` 环境变量速查、native INT8 MoE kernel 的
AITER 模板 patch 说明，见 **`dcu-perf-cases` Case 10**。本 skill 只抽范式，
不重复案例数字（避免口径污染）。

**本 skill 的 `references/` 已收录该案例的真实完整补丁**（R180 无 MTP / R184+MTP3 两套
`sitecustomize.py`，含完整形状表与 `K100_*` 开关），可直接取其对应阶段文件落地使用；
配套 native MoE kernel 与 MoE 调优 JSON 仍需从复现工程取得（见 §4 索引说明）。

## 8. R067/27B 方法论沉淀（来自 OPTIMIZATION_NOTES.md）

从 `qwen36-27b-k100ai-w8a8-optimization` repo 的 docs/OPTIMIZATION_NOTES.md
提炼的跨项目可迁移结论，直接并入本 skill 的参考方法论：

1. **识别真实 runtime shape，不要只看 checkpoint 名字**：Qwen3.6-27B 是 Dense 模型，
   vLLM 合并多个投影后才执行。只看单个 safetensors 权重名会定错矩阵。
   运行时真实 packed shape 包括：MLP gate/up `5120→34816`、MLP down `17408→5120`、
   GDN qkv+z `5120→16384`、GDN b+a `5120→96`、attention q+k+v `5120→14336`、
   GDN/attention out `6144→5120`、MTP fc `10240→5120`、lm_head `5120→248320`。
   第一个大性能跳变往往发生在这里（重新搜这些 packed shape 的 kernel）。

2. **Small-M W8A8 是 K100AI/gfx928 的核心问题**：单请求 decode 的 M 很小（M=1~5）。
   通用 Triton heuristic 与真实 K100AI 小 M 形状严重错配。代表性本地增益：
   gate_up / GDN qkvz / attention qkv / MTP fc 在 M=3 下均有约 2x 量级提升（具体加速比属性能基线，转 `dcu-perf-cases`）。可复用方法：① 捕获真实 runtime `(M,K,N)`；② 基准这些
   exact 形状；③ 设数值门禁；④ 才做端到端验证。**不要照搬别的模型的 tile 值**。

3. **MTP 必须作为独立 shape family 调优，不能并入 no-MTP**：
   no-MTP decode 主要关心 M=1；MTP2 需要 M=3 验证路径；MTP3 需要 M=4。
   MTP 的 acceptance 是内容依赖的，必须连同 workload/scenario 一起报告。

4. **上游 checkpoint 的 lm_head 仍 BF16**：下载的 W8A8 checkpoint 未把 lm_head
   纳入 compressed-tensors 量化；词表 248320 使该投影是大 decode 热点。
   把 lm_head 在加载时额外量化为 INT8 weight + per-OC scale 是真实提速，
   同时是**显式数值变化**——公布时说明是 performance/precision trade-off，
   不宣称与上游 bitwise identical。

5. **HCU MMAC 不是标准 AMD INT8 MFMA**：K100AI 的快速 INT8 dot 走
   `v_mmac_i32_16x16x32_i8`（`__builtin_hcu_mmac_i32_16x16x32_i8`）。
   强制标准 `v_mfma_i32_16x16x16i8` 能在注入 target-feature 后编译，但在真实
   gfx928 硬件报 Illegal Instruction。因此把上游 AMD MFMA 模板盲移植到 K100AI
   不安全，即使编译通过。这也解释了为何不能随便套用 CK/MFMA 模板。

6. **HCU MMAC/LDS 调度是 shape 依赖的**：M=1 down 项目配 MMAC latency 5 / DS
   latency 10 有收益；lm_head 等其它 shape 会 regress。所以 compiler 调度参数
   必须**逐 shape 选择**，不能全局一开。

7. **native out GEMV 值得专门化**：`6144→5120` out 投影是 native HCU/GFX928
   的好候选。v6 kernel 用 HCU buffer 资源、`buffer_load_dwordx4`、手控 `vmcnt`、
   双 accumulator 链、`sdot4` 累加。源码公开且**本地编译**（不发布预编译 .so，
   因 ABI/编译环境不同）。`apply_dcu_config.py` 校验上游/DCU 两 SHA256 后覆盖
   config.json，不改 safetensors。

8. **若干 attractive 微基准是 false lead**：tested 被拒的有 split-K down（数学
   对但 scratch/atomic/reduction 开销更慢）、普通 GEMV/simple sdot4（不如
   MMAC 路径）、contiguous `[K,N]` transpose copy（多数 main shape 明显变慢）、
   generic CK-Tile INT8 GEMM、async-copy（down shape 严重回归）、WASP warp
   特化（HCU 后端拒 gfx928 为 unsupported）。**不要被局部微基准误导**。

9. **带宽测量改变策略**：K100AI read-only HBM streaming ~810 GB/s。tuned GEMM
   已用相当带宽（尤其 gate_up）。逼近 memory wall 后策略从反复 tile search 转：
   dataflow 变化、投影融合、减重复量化、native load/wait 调度、避免非必要
   BF16 中间物化。这是跨模型通用的方法论。

10. **发布纪律**：更快 kernel **不是 release candidate** 直到完整模型路径通过
   端到端门禁（输出 SHA、config hash、Docker image 固定）。局部微基准收益
   不可直接等同于模型端到端提速。

## 9. R067 补丁 references 索引

`references/r067_mtp4_qwen36_27b_w8a8_k100ai.README.md` 收录 R067 补丁全套：

| 文件 | 说明 |
|---|---|
| `r067_mtp4_qwen36_27b_sitecustomize.py` | R067 MTP4 monkey-patch 注入（lm_head INT32 scaled_mm 快路径、draft K1024/B64 selector、M5 精确 verifier） |
| `r067_mtp4_qwen36_27b_candidate_head.py` | 两级候选 selector 源码（K_SELECTOR=1024、Top2K、full-K exact 打分） |
| `r067_mtp4_qwen36_27b_DCU.config.json` | 已验证 DCU config（compressed-tensors W8A8，上游 lm_head 仍 BF16） |
| `r067_mtp4_qwen36_27b_quickstart_mtp4.sh` / `serve_mtp4.sh` | 一键准备 / 起服务启动脚本（MTP4 默认） |
| `r067_mtp4_qwen36_27b_benchmark_fixed_512.py` | R067 验收基准脚本（fixed-512 六轮、SHA 门禁） |
| `r067_mtp4_qwen36_27b_apply_dcu_config.py` | config 应用脚本（双 SHA256 校验保护） |
| `r067_mtp4_qwen36_27b_k100_int8_gemv_v6.hip` + `build_k100_int8_gemv_v6.py` | 原生 INT8 GEMV 源码 + 本地 hipcc 编译脚本（不发布预编译 .so） |

该 README 同时说明"离开 K100AI/gfx928/vLLM 0.18.1/DTK 26.04 须重新验证"，
并给出与 dcu-vllm-deploy/MTP toolcall drift 等相邻 skill 的关联索引。

---

## 10. 与相邻 skill 的边界（互补不重叠）

| 任务 | 用哪个 |
|---|---|
| 把专用 kernel 干净挂进运行中 vLLM（不改源码/开关/fallback） | **本 skill**（dcu-vllm-runtime-patch） |
| 替换 vLLM 原生算子定义（注册/分发/重编译） | dcu-vllm-op-replace |
| 从零写 vLLM 自定义算子（ROCm 适配 / dispatch） | dcu-vllm-custom-op |
| 写 HIP / TileLang kernel 本体 | dcu-kernel-dev |
| 真实优化数字与演进案例 | dcu-perf-cases |
| vLLM 启动参数 / 部署调优 | dcu-vllm-deploy |
| 端到端压测与口径纪律 | dcu-llm-benchmark |

---

## 11. 坑与反例

- **不要把 M 档形状漏掉**：MTP 的 draft 模型是 3 个串行 M=1 forward，若只搜了
  目标模型 M=4 路径而漏了 draft 的 M=1，draft 层会 fallback 上游（尽管目标路径已调好），
  整体反而慢。**每个会被实际触发的 M 档都要有表项**。
- **native MoE kernel 的 scale 语义**：上游 AITER 模板把多 channel scale 塌缩成标量，
  仅在 128 channel 共享 scale 时正确；`group_size_n==1` 必须 patch 成 true
  per-output-channel，否则数值错（不是慢，是错）。
- **CUDAGraph 捕获**：decode 用 `--cudagraph-capture-sizes 1`，MTP 额外 `1 4`
  （4-token 验证路径）；自定义算子在捕获图内必须行为确定。
- **首编译请求**：Triton 首次编译会拖慢单次请求，必须单独标记剔除，但**不能删除**
  后续稳定结果来「美化」基线。
- **生产隔离**：打补丁做速度测试时，固定专用 GPU、不动生产服务、每次只跑一个候选。

---

> 溯源：范式提炼自 `dcu-perf-cases` Case 10（本地复现工程 `<local_repro_pkg>_2026-08-07.gz` 含 R180/R184 两套 `sitecustomize.py` 与 `native_ext`）；骨架脚本为通用抽象，未捆绑具体模型权重或镜像。
