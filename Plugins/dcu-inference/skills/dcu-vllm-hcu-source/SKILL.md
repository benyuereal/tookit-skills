---
name: dcu-vllm-hcu-source
description: vllm-hcu 包源码结构——vLLM 在 DCU/HCU 上的插件式 patch 层（105 个 patch + HCUPlatform 插件 + 模块交换机制）
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# vllm-hcu 源码适配索引（dcu-vllm-hcu-source）

> 定位：**vllm-hcu 是 vLLM 在 DCU/HCU 上的插件式 patch 层**，不是 vLLM 本体。它通过 vLLM 的 `platform_plugins` / `general_plugins` entry point 注册，在运行时对 vLLM 源码做字符串 patch + 模块交换，让上游 vLLM 在 HCU 上跑起来。
> 与 `dcu-vllm-source` 的关系：`dcu-vllm-source` 讲 vLLM 各版本 fork 的源码结构；本 skill 讲 **vllm-hcu 这个独立包**怎么 patch vLLM。两者互补——vllm-hcu 是 0.21+ 时代海光把适配从「fork 改源码」转向「插件 patch」的产物。
> 实测来源：`vllm_hcu-0.21.0+das.dtk2604.torch2100.2608132222.g386d02-cp310` wheel（release/dtk2604 源），解包 236 文件。

## §0 定位：vllm-hcu 是什么、与 vLLM fork 的关系

vllm-hcu = 海光**独立 pip 包**，装到 vLLM 旁边，通过 entry point 钩进 vLLM 的插件系统。它**不改 vLLM 安装目录的源码文件**（除 post_install 软链一个文件），而是：
1. **字符串 patch**：运行时读 vLLM 模块源码，做 `(old, new)` 子串替换（105 个 `*.patch.py` 文件，每个 target 一个 vLLM 模块）。
2. **模块交换**：把 `vllm.xxx` 官方模块名在 `sys.modules` 里替换成 `vllm_hcu.xxx` 实现（`MODULE_EXCHANGE_MAP`）。
3. **平台插件**：注册 `HCUPlatform`（继承 vLLM `Platform`），接管设备/attention backend/算子分发。

| 维度 | vLLM fork（dcu-vllm-source） | vllm-hcu（本 skill） |
|---|---|---|
| 交付 | 改 vLLM 源码后打 fork wheel | 独立包，装到 vLLM 旁 |
| 适配方式 | 源码级改（csrc/rocm.py/_custom_ops） | 运行时字符串 patch + 模块交换 |
| 版本 | 0.15/0.18/0.21/0.23/0.26 各 fork | 0.21.0（对应 vLLM 0.21） |
| 回滚 | 换回上游 vLLM | 卸载 vllm-hcu 即恢复 |

**演进方向**：0.15（深度 kernel 新增 fork）→ 0.18（轻量 fork）→ 0.21+（**vllm-hcu 插件 patch**，vLLM 本体尽量不动）。

## §1 源码地图（wheel 实测目录）

```
vllm_hcu/
├── __init__.py              # 3 个 entry point 函数（hcu_platform_plugin / register_model / register_ops）
├── patch_utils.py           # ★ patch 引擎：import_hook + 模块交换 + patch_module_class_function
├── post_install.py          # post_install 钩子：软链 modular_kernel.py + apply patches
├── version.py
├── hcu_ops.cpython-310-x86_64-linux-gnu.so   # 编译产物（HCU 自定义算子，链接 libgalaxyhip.so.5）
├── platforms/
│   ├── hcu.py               # ★ HCUPlatform(Platform) + register_attention_backends + flash_attn 模式
│   └── envs.py              # HCU 环境变量（VLLM_HCU_USE_*_FLASH_ATTN 等）
├── patches/                 # ★ 105 个 *.patch.py（每个 target 一个 vLLM 模块）
│   ├── __init__.py          # apply_patches() 引擎 + PATCHES 格式说明（字符串/正则两种）
│   ├── vllm__*.patch.py     # 命名 = vLLM 模块路径（__ 替 /），如 vllm__v1__worker__gpu_input_batch.patch.py
│   └── patch_*.py           # 海光专属 patch（patch_aiter_moe_bf16_fp16 / patch_w8a8_channelwise_blaslt 等）
├── ops/                     # HCU 自定义算子 Python 侧（rms_norm / rotary_embedding / silu_and_mul / topk_topp_sample / fuse_moe_gate / fuse_rms_norm_quant / fuse_silu_mul_quant / gemma_rms_norm / fp8_einsum_fallback）
├── models/                  # 海光专属模型（deepseek_v2/v4/v4_mtp / glm4_moe / hy_v3 / step3p7 + mtp 变体）
├── v1/                      # vLLM v1 引擎 HCU 适配（hcu_model_runner.py / worker.py）
├── model_executor/          # 模块交换目标（parameter.py / layers/fused_moe/modular_kernel.py）
├── distributed/             # 分布式 HCU 适配
├── utils/                   # bf16_fp32_gemm / multi_stream_utils
└── examples/ tests/         # 示例与测试
```

## §2 安装与 entry point（怎么钩进 vLLM）

**entry_points.txt**（wheel 实测）：
```ini
[console_scripts]
vllm-hcu-apply-patches = vllm_hcu.post_install:main

[vllm.general_plugins]
hcu_model = vllm_hcu:hcu_platform_register_model
hcu_ops   = vllm_hcu:hcu_platform_register_ops

[vllm.platform_plugins]
hcu = vllm_hcu:hcu_platform_plugin
```

- **`vllm.platform_plugins.hcu`** → `hcu_platform_plugin()`：调 `import_hook()` 装 import 钩子，返回 `"vllm_hcu.platforms.hcu.HCUPlatform"`（vLLM 据此注册 HCU 平台）。
- **`vllm.general_plugins.hcu_ops`** → `hcu_platform_register_ops()`：`patch_module_class_function()` + `import vllm_hcu.ops`（注册 HCU 算子）。
- **`vllm.general_plugins.hcu_model`** → `hcu_platform_register_model()`：注册海光专属模型。
- **`vllm-hcu-apply-patches`**（console script）→ `post_install.main()`：post_install 阶段跑，软链 `modular_kernel.py` + `_register_patches()`。

**安装**：`pip install vllm_hcu -i https://pypi.sourcefind.cn/release/dtk2604/`（torch 标记须对齐，本版绑 torch2100）。装完 vLLM 启动时自动经 entry point 加载，无需改 vLLM 源码。

## §3 patch 引擎（patch_utils.py，精确落点）

| 函数 | 行号 | 作用 |
|---|---|---|
| `_apply_vllm_patches()` | `patch_utils.py:11` | 调 `patches.apply_patches()`，幂等（`_patches_applied` 标志） |
| `_register_patches()` | `patch_utils.py:26` | 包装 `_apply_vllm_patches`，import 时触发 |
| `MODULE_EXCHANGE_MAP` | `patch_utils.py:42` | 官方模块名 → vllm_hcu 实现名映射（7 条，见 §4） |
| `_import_targets_exchange_candidate()` | `patch_utils.py:61` | 判断本次 import 是否牵涉交换映射 |
| `_maybe_lazy_module_exchanges()` | `patch_utils.py:73` | 首次命中映射相关 import 时惰性加载交换 |
| `module_exchanges_all_ok()` | `patch_utils.py:84` | 加载全部交换条目，**任一失败回滚已注册项** |
| `_custom_import()` | `patch_utils.py:135` | 替换 `builtins.__import__`，拦截 import 做交换 |
| `import_hook()` | `patch_utils.py:185` | 装 import 钩子（`hcu_platform_plugin` 调它） |
| `patch_module_class_function()` | `patch_utils.py:191` | patch 模块/类/函数（`hcu_platform_register_ops` 调它） |
| `patch_fuction_class()` | `patch_utils.py:219` | 单函数 patch（`patched_import` 闭包） |

**patch 文件格式**（`patches/__init__.py` 说明，两种风格）：
1. **字符串 match（默认）**：`PATCHES = [(old, new), ...]`，`old` 是普通 str（非 re），引擎找精确子串替换。例：`("ops.concat_and_cache_mla(", "torch.ops.hcu_ops.concat_and_cache_mla(")`。
2. **正则 match（显式 opt-in）**：用 `re.compile` 或特定标记，仅当需要正则时用。

**patch 文件命名**：`vllm__<path with __ for />.patch.py`，如 `vllm__v1__worker__gpu_input_batch.patch.py` → target `vllm.v1.worker.gpu_input_batch`。`patch_*.py` 是海光专属（非 1:1 对应 vLLM 模块路径）。

## §4 模块交换映射（MODULE_EXCHANGE_MAP，patch_utils.py:42）

把 vLLM 官方模块名在 `sys.modules` 替换成 vllm_hcu 实现（**import 即生效**，惰性触发）：

| 官方模块名 | vllm_hcu 实现 |
|---|---|
| `vllm.model_executor.layers.deepseek_compressor` | `vllm_hcu.model_executor.layers.deepseek_compressor` |
| `vllm.model_executor.layers.mhc` | `vllm_hcu.model_executor.layers.mhc` |
| `vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache` | `vllm_hcu.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache` |
| `vllm.v1.attention.backends.mla.sparse_swa` | `vllm_hcu.v1.attention.backends.mla.sparse_swa` |
| `vllm.model_executor.layers.fused_moe.deep_gemm_utils` | `vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils` |
| `vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe` | `vllm_hcu.model_executor.layers.fused_moe.experts.deep_gemm_moe` |
| `vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe` | `vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe` |

> 注：`modular_kernel` 不走 import_hook 交换（import 时缓存 FusedMoEKernel 不可靠），改用 **post_install 软链**（见 §5）。

## §5 post_install 软链（post_install.py）

`apply_post_install_patches()`（`post_install.py:4`）：
1. 取 `vllm.model_executor.layers.fused_moe.modular_kernel` 的源码路径（`_module_to_source_path`）。
2. 软链到 `vllm_hcu/model_executor/layers/fused_moe/modular_kernel.py`（`link_path.symlink_to(src)`）。
3. `_register_patches()` 应用全部 patch。

> 这是**唯一改 vLLM 安装目录**的地方（软链一个文件），其余全是运行时 patch。回滚：删软链 + 卸载 vllm-hcu。

## §6 HCUPlatform（platforms/hcu.py，精确落点）

| 函数/方法 | 行号 | 作用 |
|---|---|---|
| `get_hcu_flash_attn_mode()` | `hcu.py:26` | 解析 HCU flash attention 子模式（classic/cutlass/custom），读 `VllmConfig.attention_config.hcu_flash_attn_mode` 或 env |
| `_rocm_device_count_stateless()` | `hcu.py:86` | 无状态设备计数 |
| `with_amdsmi_context()` | `hcu.py:123` | amdsmi 上下文装饰器 |
| `flash_attn_triton_available()` | `hcu.py:135` | Triton flash attn 可用性 |
| `_get_gcn_arch_name()` | `hcu.py:148` | 取 GCN arch 名（gfx936/938 等） |
| `on_gfx93x()` / `on_gfx938()` | `hcu.py:157/160` | gfx 判定 |
| `_get_backend_priorities()` | `hcu.py:164` | attention backend 优先级 |
| `register_attention_backends()` | `hcu.py:192` | 注册 HCU attention backends |
| `class HCUPlatform(Platform)` | `hcu.py:216` | ★ 平台类（继承 vLLM Platform） |
| `HCUPlatform.import_kernels()` | `hcu.py:261` | import HCU kernels |
| `HCUPlatform.get_valid_backends()` | `hcu.py:272` | 有效 attention backends |
| `HCUPlatform.get_attn_backend_cls()` | `hcu.py:305` | 取 attention backend 类 |
| `HCUPlatform.get_supported_vit_attn_backends()` | `hcu.py:391` | ViT attention backends |
| `HCUPlatform.get_vit_attn_backend()` | `hcu.py:400` | 取 ViT attention backend |
| `HCUPlatform.use_custom_allreduce()` | `hcu.py:425` | 自定义 allreduce |
| `HCUPlatform.device_count()` | `hcu.py:431` | 设备数 |
| `HCUPlatform.get_device_capability()` | `hcu.py:472` | 设备能力 |

**flash attention 子模式**（`get_hcu_flash_attn_mode`）：`--attention-backend FLASH_ATTN_CLASSIC / FLASH_ATTN_CUTLASS / FLASH_ATTN_CUSTOM`，env 开关 `VLLM_HCU_USE_CUSTOM_FLASH_ATTN` / `VLLM_HCU_USE_FLASH_ATTN_UNIFIED` / `VLLM_HCU_USE_FLASH_ATTN`。

## §7 105 个 patch 文件分类（patches/）

按 target 模块分组（`vllm__*.patch.py` 命名 = vLLM 模块路径）：

| 分组 | 代表 patch 文件 | 作用 |
|---|---|---|
| **config** | `vllm__config__{attention,compilation,kernel,model,parallel,speculative,vllm}.patch.py` | 配置项 HCU 适配 |
| **distributed** | `vllm__distributed__{parallel_state,utils,device_communicators__*,eplb__*,kv_transfer__*}.patch.py` | 分布式/通信/KV transfer（含 mooncake_connector patch） |
| **fused_moe** | `vllm__model_executor__layers__fused_moe__{fused_moe,layer,config,utils,experts__*,oracle__*,prepare_finalize__deepep_{ht,ll},router__*,runner__*}.patch.py` | ★ MoE 全链路（含 DeepEP HT/LL prepare_finalize） |
| **quantization** | `vllm__model_executor__layers__quantization__{*,compressed_tensors__*,utils__*}.patch.py` | 量化（compressed-tensors W8A8 FP8/INT8、fp8_utils、w8a8_utils） |
| **attention** | `vllm__v1__attention__{backend,backends__mla__*,ops__triton_unified_attention}.patch.py` | attention backend（MLA sparse/indexer、triton unified） |
| **models** | `vllm__model_executor__models__{config,gpt_oss,qwen3_5,qwen3_vl,qwen3_vl_moe,step3p5_mtp}.patch.py` | 模型适配（Qwen3.5/3-VL、GPT-OSS、Step3.5） |
| **v1 engine** | `vllm__v1__{core__*,engine__*,executor__*,spec_decode__*,worker__*,outputs,utils}.patch.py` | v1 引擎（scheduler/executor/spec decode/worker） |
| **mamba** | `vllm__model_executor__layers__mamba__{gdn_linear_attn,mamba_mixer2}.patch.py` | Mamba/GDN 线性注意力 |
| **海光专属** | `patch_{aiter_moe_bf16_fp16,w8a8_channelwise_blaslt_apply_scaled_mm,hcu_lora_column_parallel_linear,deepseek_r1_distill_llama_70b_tokenizer,kimi_k25_*,qwen35_lora_disable_piecewise_cudagraph,weight_utils_skip_debug}.py` | 海光专属 patch（AITER MoE BF16/FP16、W8A8 channelwise BLASLt、LoRA、tokenizer） |

> 改某个 vLLM 模块的 HCU 行为：先找对应 `vllm__<path>.patch.py` 看 patch 内容，再决定是改 patch 还是改 vllm_hcu 实现。

## §8 HCU 自定义算子（ops/ + hcu_ops.so）

`ops/` 下 Python 侧算子（经 `torch.ops.hcu_ops.*` 注册，编译产物 `hcu_ops.cpython-310-x86_64-linux-gnu.so` 链接 `libgalaxyhip.so.5`）：
- `rms_norm.py` / `gemma_rms_norm.py` — RMSNorm
- `rotary_embedding.py` — RoPE
- `silu_and_mul.py` — SiLU+Mul
- `topk_topp_sample.py` — top-k/top-p 采样
- `fuse_moe_gate.py` / `fuse_rms_norm_quant.py` / `fuse_silu_mul_quant.py` — MoE 融合
- `fp8_einsum_fallback.py` — FP8 einsum fallback

> patch 里大量 `ops.xxx(` → `torch.ops.hcu_ops.xxx(` 的替换，就是把 vLLM 算子调用重定向到 HCU 编译算子。

## §9 与兄弟 skill 关系

- `dcu-vllm-source`：vLLM 各版本 fork 源码结构（0.15/0.18/0.21/0.23/0.26）→ 本 skill 是 0.21+ 的**插件 patch 层**，两者对照看演进。
- `dcu-vllm-custom-op` / `dcu-vllm-op-replace`：从零造算子 / 替换算子 → vllm-hcu 的 `ops/` + patch 是「批量替换」的工业化版本。
- `dcu-vllm-runtime-patch`：运行时 monkey-patch 注入 kernel（env 开关 + bit-identical 门禁）→ vllm-hcu 是同一思路的**包级**实现（105 patch + 模块交换）。
- `dcu-aiter-source` / `dcu-deepgemm-source`：AITER/DeepGEMM 算子库 → vllm-hcu 的 `patch_aiter_moe_bf16_fp16` / `fused_moe__experts__deep_gemm_moe` 是它们接入 vLLM 的 patch 落点。
- `dcu-inference-infra`：推理 infra 总入口。

## §10 验证入口

```bash
# 安装（torch 标记对齐，本版 torch2100）
pip install vllm_hcu -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 entry point 注册
python3 -c "from importlib.metadata import entry_points; print([e for e in entry_points().get('vllm.platform_plugins',[])])"

# 确认 patch 应用（启动 vLLM 后看日志 "Post-install patches applied successfully"）
vllm-hcu-apply-patches   # 手动重跑 post_install patch

# 看某个 patch 内容
cat $(python3 -c "import vllm_hcu,os;print(os.path.dirname(vllm_hcu.__file__))")/patches/vllm__v1__worker__gpu_input_batch.patch.py
```

## 使用须知

- vllm-hcu 是**独立包**，装到 vLLM 旁，不改 vLLM 源码（除 post_install 软链 1 个文件）。回滚 = 卸载 vllm-hcu + 删软链。
- **torch 标记必须对齐**：本版绑 torch2100，装到 torch 2.9 环境会 ABI 不匹配（见 `dcu-das-hub-api` §5 torch ABI 诊断）。
- patch 是**字符串替换**，vLLM 版本升级后 patch 的 `old` 子串可能匹配不到（静默失败）——升级 vLLM 后必须重跑 `vllm-hcu-apply-patches` 看是否有 "Failed to apply" 日志。
- **`VLLM_PLUGINS=`（空串）会静默禁用插件**（2026-10-08 实测）：envs.py 解析 `os.environ["VLLM_PLUGINS"].split(",")`，空串 → `['']`，而加载逻辑是 `allowed_plugins is None` 才加载全部——`"hcu" in ['']` 为 False，插件被**静默跳过，无任何报错**，日志只少 "Platform plugin hcu is activated" 一行。后果：HCUPlatform 不激活，attention backend/算子分发全变，Qwen3.8-27B decode 掉速 7-8%（曾误判为 fork 代码问题）。A/B 对照时两组 `VLLM_PLUGINS` 必须完全一致（都设或都不设）。排查：`grep "Platform plugin hcu is activated" server.log`。
- **源码编译 vLLM（fork）会覆盖插件 patch 过的文件**（2026-10-08 实测）：插件靠字符串 patch 往 vLLM 源码注入类（如 `vllm/v1/attention/backend.py` 的 `CpCommonAttentionMetadata`），从源码编译/重装 vLLM 会把 patch 过的文件覆盖回原版 → 插件激活时 worker `ImportError: cannot import name 'CpCommonAttentionMetadata'`（patch 失败只 `logger.error` 不致命，易被忽略）。修复：编译后手动 `from vllm_hcu.patches import apply_patches; apply_patches()`（幂等，写盘）或跑 `vllm-hcu-apply-patches`。验证：`grep -c CpCommonAttentionMetadata $VLLM/v1/attention/backend.py` ≥1。
- 模块交换**任一失败回滚**（`module_exchanges_all_ok`），不会半套生效。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 海光 fork 具体行为以 sourcefind `codes/OpenDAS/vllm-hcu` 仓库实时核验。
- 高危操作谨慎，优先测试机验证。
