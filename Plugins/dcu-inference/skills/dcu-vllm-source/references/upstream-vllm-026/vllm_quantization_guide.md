# vLLM 量化（Quantization）模块详解

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖 FP8/GPTQ/AWQ/Marlin/compressed-tensors/slimquant 等量化方案的设计、kernel 选择与权重加载。

---

## 目录

1. [概述](#1-概述)
2. [模块架构与注册机制](#2-模块架构与注册机制)
3. [核心抽象：QuantizationConfig 与 QuantizeMethodBase](#3-核心抽象quantizationconfig-与-quantizemethodbase)
4. [Kernel 选择与工具（utils/）](#4-kernel-选择与工具utils)
5. [FP8 量化（fp8.py）](#5-fp8-量化fp8py)
6. [GPTQ / AWQ / Marlin](#6-gptq--awq--marlin)
7. [compressed-tensors 通用框架](#7-compressed-tensors-通用框架)
8. [SlimQuant W4A8（GLM-5.1 实际使用）](#8-slimquant-w4a8glm-51-实际使用)
9. [其他量化方案](#9-其他量化方案)
10. [KV Cache 量化](#10-kv-cache-量化)
11. [与 FusedMoE 的集成](#11-与-fusedmoe-的集成)
12. [量化方法选择指南](#12-量化方法选择指南)
13. [实际案例：GLM-5.1 slimquant_w4a8 分析](#13-实际案例glm-51-slimquant_w4a8-分析)
14. [配置参数与环境变量](#14-配置参数与环境变量)
15. [常见问题](#15-常见问题)

---

## 1. 概述

量化是大模型推理中**降低显存、提升吞吐**的核心手段。vLLM 的量化模块是代码库中最大的子模块之一（约 118 个文件，2.5M 代码），支持从 INT4 到 FP8、NVFP4 等十余种量化方案，并集成多种高性能 kernel（Marlin、CUTLASS、Triton、DeepGEMM）。

### 量化维度

| 维度 | 含义 | 典型值 |
|------|------|--------|
| **Weight bits** | 权重位宽 | 4 / 8 / 16 |
| **Activation bits** | 激活位宽 | 8 / 16 |
| **Weight scale 粒度** | 权重缩放粒度 | per-tensor / per-channel / per-group(128) / per-block(128×128) |
| **Activation scale 粒度** | 激活缩放粒度 | per-tensor / per-token / per-group(128) |
| **Activation scheme** | 激活量化时机 | static（离线校准）/ dynamic（在线量化） |

### 命名约定

`WxAy`：x = 权重位宽，y = 激活位宽
- **W8A8**：权重 8-bit + 激活 8-bit（FP8/INT8）
- **W4A16**：权重 4-bit + 激活 16-bit（weight-only，GPTQ/AWQ）
- **W4A8**：权重 4-bit + 激活 8-bit（slimquant）
- **W4A4**：权重 4-bit + 激活 4-bit（NVFP4）

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/model_executor/layers/quantization/__init__.py` | 197 | 量化方法注册表 |
| `.../quantization/base_config.py` | 171 | QuantizationConfig / QuantizeMethodBase 抽象基类 |
| `.../quantization/fp8.py` | ~1200 | FP8 量化（最重要） |
| `.../quantization/gptq.py` | ~450 | GPTQ 量化 |
| `.../quantization/awq.py` | ~460 | AWQ 量化 |
| `.../quantization/gptq_marlin.py` | ~1050 | GPTQ Marlin kernel |
| `.../quantization/awq_marlin.py` | ~910 | AWQ Marlin kernel |
| `.../quantization/slimquant_w4a8.py` | ~330 | SlimQuant W4A8（GLM-5.1 使用） |
| `.../quantization/slimquant_w4a8_marlin.py` | ~300 | SlimQuant W4A8 Marlin |
| `.../quantization/compressed_tensors/compressed_tensors.py` | ~1000 | compressed-tensors 通用框架 |
| `.../quantization/modelopt.py` | ~1900 | NVIDIA ModelOpt（FP8/NvFp4） |
| `.../quantization/mxfp4.py` | ~1400 | MXFP4 |
| `.../quantization/bitsandbytes.py` | ~630 | bitsandbytes（NF4/FP4） |
| `.../quantization/kv_cache.py` | ~200 | KV Cache 量化基类 |
| `.../quantization/utils/w8a8_utils.py` | ~300 | W8A8 kernel 后端选择 |
| `.../quantization/utils/quant_utils.py` | ~850 | GroupShape、QuantKey、打包工具 |
| `.../quantization/utils/marlin_utils.py` | — | Marlin kernel 辅助 |

---

## 2. 模块架构与注册机制

### 2.1 量化方法注册表

`__init__.py` 维护所有量化方法的注册表：

```python
QuantizationMethods = Literal[
    "awq", "fp8", "ptpc_fp8", "fbgemm_fp8", "fp_quant",
    "modelopt", "modelopt_fp4", "bitblas", "gguf",
    "gptq_marlin_24", "gptq_marlin", "gptq_bitblas",
    "awq_marlin", "gptq", "compressed-tensors", "bitsandbytes",
    "experts_int8", "ipex", "quark", "moe_wna16", "torchao",
    "inc", "mxfp4", "petit_nvfp4", "cpu_awq", "blockwise_int8",
    "slimquant_w4a8", "slimquant_w4a8_marlin",
    "slimquant_marlin", "slimquant_compressed_tensors_marlin",
]

DEPRECATED_QUANTIZATION_METHODS = [
    "tpu_int8", "ptpc_fp8", "fbgemm_fp8", "fp_quant",
    "bitblas", "gptq_marlin_24", "gptq_bitblas",
    "experts_int8", "ipex", "petit_nvfp4",
]
```

### 2.2 工厂函数

```python
def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    # 懒加载：按需 import 对应 Config 类
    method_to_config = {
        "awq": AWQConfig,
        "fp8": Fp8Config,
        "gptq": GPTQConfig,
        "gptq_marlin": GPTQMarlinConfig,
        "awq_marlin": AWQMarlinConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "slimquant_w4a8": SlimQuantW4A8Int8Config,
        "slimquant_w4a8_marlin": SlimQuantW4A8Int8MarlinConfig,
        "mxfp4": Mxfp4Config,
        "modelopt": ModelOptFp8Config,
        "modelopt_fp4": ModelOptNvFp4Config,
        "bitsandbytes": BitsAndBytesConfig,
        # ... 30+ 种
    }
    return method_to_config[quantization]
```

### 2.3 自定义量化注册

```python
@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    ...
```

### 2.4 整体流程

```
用户指定 --quantization slimquant_w4a8
       │
       ▼
get_quantization_config("slimquant_w4a8") → SlimQuantW4A8Int8Config
       │
       ▼
config.from_config(hf_quant_config) → 实例化 Config
       │
       ▼
对每个模型层:
  config.get_quant_method(layer, prefix) → QuantizeMethodBase
       │
       ├── LinearBase → XxxLinearMethod
       ├── FusedMoE   → XxxMoEMethod
       └── Attention  → XxxKVCacheMethod
       │
       ▼
method.create_weights(layer, ...)        # 创建量化权重参数
       │
       ▼
[权重加载] weight_loader(...)
       │
       ▼
method.process_weights_after_loading(layer)  # 转置/重打包/kernel 预热
       │
       ▼
method.apply(layer, x, ...)              # 推理时前向
```

---

## 3. 核心抽象：QuantizationConfig 与 QuantizeMethodBase

### 3.1 QuantizationConfig（量化配置基类）

```python
class QuantizationConfig(ABC):
    def __init__(self):
        # 模型在初始化时更新此映射（packed_modules_mapping）
        self.packed_modules_mapping: dict[str, list[str]] = {}

    @abstractmethod
    def get_name(self) -> QuantizationMethods: ...        # 方法名
    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...  # 支持的激活 dtype
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int: ...               # 最低 GPU 算力（70=Volta, 75=Turing, 80=Ampere）
    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]: ...          # 量化配置文件名
    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "QuantizationConfig": ...  # 从 HF config 构造

    @abstractmethod
    def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None: ...

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> QuantizationMethods | None:
        """允许某 Config 接管其他量化格式的 checkpoint"""
        return None

    def get_cache_scale(self, name: str) -> str | None: ...  # KV cache scale 名映射
```

### 3.2 QuantizeMethodBase（量化方法基类）

```python
class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(self, layer, *weight_args, **extra_weight_attrs): ...
    @abstractmethod
    def apply(self, layer, *args, **kwargs) -> torch.Tensor: ...

    def embedding(self, layer, *args, **kwargs) -> torch.Tensor: ...  # 可选
    def process_weights_after_loading(self, layer) -> None: ...        # 可选：转置/打包
```

### 3.3 层类型与方法的映射

`get_quant_method(layer, prefix)` 根据层类型返回不同的量化方法：

| 层类型 | 基类 | 量化方法示例 |
|--------|------|------------|
| `LinearBase` | `LinearMethodBase` | `Fp8LinearMethod`, `GPTQLinearMethod` |
| `FusedMoE` | `FusedMoEMethodBase` | `Fp8MoEMethod`, `SlimQuantW4A8Int8MoEMethod` |
| `Attention` | `BaseKVCacheMethod` | `Fp8KVCacheMethod` |

被忽略的层（`ignored_layers`）返回未量化方法 `UnquantizedLinearMethod` / `UnquantizedFusedMoEMethod`。

---

## 4. Kernel 选择与工具（utils/）

### 4.1 GroupShape（缩放粒度描述）

```python
class GroupShape(NamedTuple):
    group_n: int   # 输出维度分组大小（-1=不分组）
    group_k: int   # 输入维度分组大小（-1=不分组）

GroupShape.PER_TENSOR  = GroupShape(-1, -1)   # 每张量一个 scale
GroupShape.PER_TOKEN   = GroupShape(1, -1)    # 每 token 一个 scale
GroupShape.PER_CHANNEL = GroupShape(-1, 1)    # 每输出通道一个 scale
```

### 4.2 QuantKey（量化方案标识）

```python
# 用于 kernel 兼容性匹配
kFp8StaticTensorSym  = QuantKey(FP8_DTYPE, kStaticTensorScale, symmetric=True)
kFp8DynamicTensorSym = QuantKey(FP8_DTYPE, kDynamicTensorScale, symmetric=True)
kFp8DynamicTokenSym  = QuantKey(FP8_DTYPE, kDynamicTokenScale, symmetric=True)
kFp8Static128BlockSym = QuantKey(FP8_DTYPE, kStatic128BlockScale, symmetric=True)  # 128×128 block
kFp8Dynamic128Sym     = QuantKey(FP8_DTYPE, kDynamic128Scale, symmetric=True)     # 1×128 per-token-group

kNvfp4DynamicGroupScale = ScaleDesc(FP8_DTYPE, False, GroupShape(1, 16))
kNvfp4StaticGroupScale  = ScaleDesc(FP8_DTYPE, True, GroupShape(1, 16))
```

### 4.3 W8A8 Kernel 后端选择（w8a8_utils.py）

`apply_int8_linear` 是 INT8 W8A8 的核心，通过 `w8a8_strategy` 选择 kernel 后端：

```python
def apply_int8_linear(input, weight, weight_scale, ..., w8a8_strategy=0, ...):
    # 1. 激活量化（dynamic per-token / static）
    x_q, x_scale = per_token_quant_int8(input)   # 或 ops.scaled_int8_quant

    # 2. 按 strategy 选择 GEMM kernel
    if w8a8_strategy == 1:
        return ops.triton_scaled_mm(x_q, weight, scale_a=x_scale, scale_b=weight_scale, ...)
    elif w8a8_strategy == 2:
        return ops.cutlass_scaled_mm(x_q, weight, scale_a=x_scale, scale_b=weight_scale, ...)
    elif w8a8_strategy == 3:
        return ops.blaslt_scaled_mm(x_q, weight, scale_a=x_scale, scale_b=weight_scale, ...)
    else:
        return ops.rocblas_scaled_mm(...)   # ROCm
```

| strategy | kernel | 适用平台 |
|----------|--------|---------|
| 1 | `triton_scaled_mm` | Triton（带 JSON 调优配置） |
| 2 | `cutlass_scaled_mm` | NVIDIA CUTLASS |
| 3 | `blaslt_scaled_mm` | hipBLASLt（ROCm） |
| 默认 | `rocblas_scaled_mm` | ROCm |

**strategy 1 的调优**：通过 `W8a8GetCacheJSON` 单例缓存 Triton autotuning 结果，按 `(m, n, k)` 形状查找最优 config，避免每次启动重新 autotune。

### 4.4 CUTLASS 能力检测

```python
def cutlass_fp8_supported() -> bool: ...           # SM ≥ 89 (Hopper)
def cutlass_block_fp8_supported() -> bool: ...      # block-wise FP8
def cutlass_group_gemm_supported() -> bool: ...     # group GEMM（MoE）
def sparse_cutlass_supported() -> bool: ...         # 2:4 稀疏
def cutlass_fp4_supported() -> bool: ...            # NVFP4 (Blackwell)
```

### 4.5 Marlin 工具

`marlin_utils.py`：
- `check_marlin_supported(...)` / `verify_marlin_supported(...)`：检查设备/形状是否支持 Marlin
- `get_marlin_input_dtype(prefix)`：确定 Marlin 输入 dtype

`marlin_utils_fp8.py`：
- `prepare_fp8_layer_for_marlin(layer, ...)`：将 FP8 权重重排为 Marlin 布局
- `apply_fp8_marlin_linear(...)`：执行 Marlin FP8 GEMM

---

## 5. FP8 量化（fp8.py）

FP8 是现代大模型推理最重要的量化方案（DeepSeek-V3、Llama 等）。

### 5.1 Fp8Config

```python
class Fp8Config(QuantizationConfig):
    def __init__(self,
        is_checkpoint_fp8_serialized: bool = False,  # checkpoint 是否已 FP8 序列化
        activation_scheme: str = "dynamic",          # "static" / "dynamic"
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,  # [block_n, block_k] 块量化
    ):
```

**关键约束**：
- 块量化（`weight_block_size`）要求 checkpoint 已序列化 + 动态激活
- 仅支持 `float8_e4m3fn` dtype（`torch._scaled_mm` 限制）

### 5.2 方法分派

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, LinearBase):
        if is_layer_skipped(...): return UnquantizedLinearMethod()
        if not self.is_checkpoint_fp8_serialized:
            return Fp8OnlineLinearMethod(self)    # 在线量化（FP16→FP8）
        else:
            return Fp8LinearMethod(self)          # 离线量化（加载 FP8 checkpoint）
    elif isinstance(layer, FusedMoE):
        if is_layer_skipped(...): return UnquantizedFusedMoEMethod(...)
        if self.is_checkpoint_fp8_serialized:
            return Fp8MoEMethod(self, layer)
        else:
            return Fp8OnlineMoEMethod(self, layer)
    elif isinstance(layer, Attention):
        return Fp8KVCacheMethod(self)
```

### 5.3 Fp8LinearMethod

支持两种权重量化粒度：

**Per-tensor / Per-channel**（`weight_block_size=None`）：
```python
# 权重: [N, K] float8_e4m3fn
# weight_scale: [1] 或 [N, 1] (per-channel)
# input_scale: [1] (static) 或 None (dynamic)
```

**Block-wise**（`weight_block_size=[128, 128]`）：
```python
# 权重: [N, K] float8_e4m3fn
# weight_scale_inv: [N//128, K//128] float32  (DeepSeek-V3 风格)
# 激活动态 per-token-group (1×128)
```

**Kernel 选择**：
```python
if self.use_marlin:                    # SM < 89 或强制
    apply_fp8_marlin_linear(...)        # Marlin FP8 (weight-only)
elif self.block_quant:
    self.w8a8_block_fp8_linear.apply(...)  # CUTLASS/DeepGEMM block FP8
else:
    self.fp8_linear.apply_weights(...)     # torch._scaled_mm / CUTLASS
```

### 5.4 Fp8OnlineLinearMethod（在线量化）

加载 FP16/BF16 checkpoint，**加载时即时量化为 FP8**：

```python
def process_weights_after_loading(self, layer):
    qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)
    weight = qweight.t()  # 转置为 [K, N] 供 scaled_mm
    replace_parameter(layer, "weight", weight)
    replace_parameter(layer, "weight_scale", weight_scale)
```

使用 `CopyNumelCounter`（TorchDispatchMode）跟踪加载进度，**流式量化**：最后一个权重块加载完即触发量化，无需等待全部加载。

### 5.5 Fp8MoEMethod

FP8 MoE 集成 FusedMoE 模块化 kernel：

```python
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config, layer):
        # 选择 FP8 MoE 后端
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe, weight_key=..., activation_key=...)

    def get_fused_moe_quant_config(self, layer):
        return make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=..., w2_scale=..., a1_scale=..., a2_scale=...,
            block_shape=self.weight_block_size)

    def apply(self, layer, x, topk_weights, topk_ids, ...):
        return fused_experts(x, layer.w13_weight, layer.w2_weight,
                            quant_config=self.moe_quant_config, ...)
```

**MoE 后端**（`Fp8MoeBackend` enum）：Triton / CUTLASS / DeepGEMM / FlashInfer-TRTLLM / FlashInfer-CUTLASS / Marlin。

**Block 量化权重布局**：
```python
# w13_weight: [E, 2*N, K] float8_e4m3fn
# w13_weight_scale_inv: [E, 2*(N//128), K//128] float32
```

**TRT-LLM 单体路径**（`is_monolithic=True`）：FlashInfer TRTLLM 在单个 kernel 内完成路由+专家计算，不走模块化 kernel：
```python
def apply_monolithic(self, layer, x, router_logits):
    return torch.ops.vllm.flashinfer_fused_moe_blockscale_fp8(
        routing_logits=..., x=x, w13_weight=..., ...)
```

### 5.6 FP8 精度格式

| 格式 | 指数位 | 尾数位 | 范围 | 用途 |
|------|--------|--------|------|------|
| `float8_e4m3fn` | 4 | 3 | ±448 | 权重 + 前向激活（vLLM 主用） |
| `float8_e5m2` | 5 | 2 | ±57344 | 训练梯度（推理少用） |
| `float8_e4m3fnuz` | 4 | 3 | (MI300) | AMD ROCm FNUZ 变体 |

---

## 6. GPTQ / AWQ / Marlin

### 6.1 GPTQ（gptq.py）

INT4 weight-only 量化，基于二阶 Hessian 信息的训练后量化。

```python
class GPTQConfig(QuantizationConfig):
    def __init__(self, weight_bits: int, group_size: int, desc_act: bool, ...):
        # weight_bits: 4 或 8
        # group_size: 分组大小（-1 = per-channel）
        # desc_act: 是否按激活大小重排序（提升精度，降低速度）
```

**GPTQLinearMethod**：
- 权重：INT4 打包（`qweight`）+ 缩放（`qzeros` + `scales`）+ `g_idx`（desc_act 重排序索引）
- Kernel：ExLlamaV2 / Marlin / Triton fallback
- `create_weights` 要求 `input_size_per_partition % group_size == 0`

### 6.2 AWQ（awq.py）

INT4 weight-only，Activation-aware 量化（保护重要权重通道）。

```python
class AWQConfig(QuantizationConfig):
    def __init__(self, weight_bits: int, group_size: int, zero_point: bool, ...):
        # zero_point: True=非对称量化（AWQ），False=对称（GPTQ）
```

**AWQLinearMethod**：
- 权重：INT4 打包 + `qzeros`（zero_point=True 时）+ `scales`
- `AWQShareWorkspace`：共享 workspace 优化显存

### 6.3 Marlin（gptq_marlin.py / awq_marlin.py）

Marlin 是 vLLM 自研的**高性能 INT4 GEMM kernel**，比 ExLlamaV2 快 50%+。

**为何用 Marlin**：
- 预打包权重到特殊布局，消除运行时重排
- 融合反量化 + GEMM
- 高效的 shared memory 访问模式

**GPTQMarlinConfig**：
```python
class GPTQMarlinConfig(QuantizationConfig):
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        # 自动接管 GPTQ checkpoint → 使用更快的 Marlin kernel
        if cls.is_gptq_marlin_compatible(hf_quant_cfg):
            if user_quant in (None, "marlin", "gptq_marlin"):
                logger.info("Detected GPTQ model can run with gptq_marlin, forcing...")
                return "gptq_marlin"
```

**关键逻辑**：用户加载 GPTQ 模型时，vLLM **自动尝试升级为 gptq_marlin**（更快），除非用户显式指定 `--quantization gptq`。

**Marlin MoE**（`gptq_marlin.py` + `fused_marlin_moe.py`）：
```python
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
# Marlin kernel 用于 MoE 的专家 GEMM
```

### 6.4 Marlin 支持检查

```python
check_marlin_supported(min_capability=80, ...)   # SM ≥ 80 (Ampere)
verify_marlin_supported(...)                      # 不支持则 raise
check_moe_marlin_supports_layer(layer, group_size)  # MoE 兼容性
```

### 6.5 MoE WNA16（moe_wna16.py）

MoE 专用的 W<8>A16 量化（Weight N-bit, Activation 16-bit），支持 INT4/INT8 权重 + FP16/BF16 激活。

### 6.6 GPTQ Marlin 24（gptq_marlin_24.py）

2:4 稀疏结构 + Marlin kernel，50% 权重稀疏。

---

## 7. compressed-tensors 通用框架

`compressed-tensors` 是 Neural Magic 的通用量化框架，支持多种精度组合，是现代模型（如 Llama 量化版）的主流格式。

### 7.1 CompressedTensorsConfig

```python
class CompressedTensorsConfig(QuantizationConfig):
    def __init__(self,
        target_scheme_map: dict[str, Any],    # target → 量化方案
        ignore: list[str],                    # 忽略的层
        quant_format: str,                    # pack-quantized / naive-quantized
        sparsity_scheme_map: dict,            # 稀疏方案
        kv_cache_scheme: dict | None,         # KV cache 量化
        transform_config: dict | None,        # 权重变换（如 Hadamard）
    ):
```

### 7.2 支持的 Scheme

通过 `compressed_tensors/schemes/` 实现各精度组合：

| Scheme 类 | 精度 | 说明 |
|-----------|------|------|
| `CompressedTensorsW8A8Int8` | W8A8 INT8 | 权重+激活均 INT8 |
| `CompressedTensorsW8A8Fp8` | W8A8 FP8 | 权重+激活均 FP8 |
| `CompressedTensorsW8A16Fp8` | W8A16 FP8 | 权重 FP8，激活 FP16 |
| `CompressedTensorsW4A8Int` | W4A8 INT | 权重 4-bit，激活 INT8 |
| `CompressedTensorsW4A8Fp8` | W4A8 FP8 | 权重 4-bit，激活 FP8 |
| `CompressedTensorsWNA16` | W4A16 / W8A16 | GPTQ/AWQ 风格 |
| `CompressedTensorsW4A16Mxfp4` | W4A16 MXFP4 | MXFP4 权重 |
| `CompressedTensorsW4A16Fp4` | W4A16 NVFP4 | NVFP4 权重 |
| `CompressedTensorsW4A4Fp4` | W4A4 NVFP4 | 权重+激活均 4-bit |
| `CompressedTensorsW4A16Sparse24` | W4A16 2:4 稀疏 | 2:4 稀疏 + 4-bit |
| `CompressedTensors24` | 2:4 稀疏 | 仅稀疏 |

### 7.3 权重变换（Transform）

`transform_config` 支持 Hadamard 变换等权重预处理，用于 QuIP# 等方法的在线反变换。

### 7.4 CompressedTensorsMoEMethod

MoE 集成通过 `compressed_tensors_moe.py`，支持 FP8/INT8 MoE。

---

## 8. SlimQuant W4A8（GLM-5.1 实际使用）

这是你的 `start.sh` 中 `--quantization slimquant_w4a8` 实际使用的模块。

### 8.1 SlimQuantW4A8Int8Config

```python
class SlimQuantW4A8Int8Config(QuantizationConfig):
    """W4A8 Int8 Quantization.
    - Weight: static, per-channel, symmetric (INT4 打包在 INT8 容器中)
    - Activation: dynamic, per-token, symmetric (INT8)
    """
    @classmethod
    def get_supported_act_dtypes(cls): return [torch.float16, torch.bfloat16]
    @classmethod
    def get_min_capability(cls): return 75   # SM ≥ 75 (Turing)
    @classmethod
    def get_name(cls): return "slimquant_w4a8"

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return SlimQuantW4A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return SlimQuantW4A8Int8MoEMethod(self, layer.moe_config)
```

**特点**：无 config 文件（`get_config_filenames` 返回 `[]`），配置硬编码。

### 8.2 SlimQuantW4A8Int8LinearMethod

```python
class SlimQuantW4A8Int8LinearMethod(LinearMethodBase):
    def __init__(self, quant_config):
        self.w8a8_strategy = envs.VLLM_W8A8_BACKEND   # 1=triton, 2=cutlass, 3=blaslt

    def create_weights(self, layer, ...):
        # 权重: [N, K] int8 (实际是 INT4 打包)
        weight = ModelWeightParameter(data=torch.empty(N, K, dtype=torch.int8), ...)
        # 缩放: [N, 1] float32 (per-channel)
        weight_scale = ChannelQuantScaleParameter(data=torch.empty(N, 1, dtype=torch.float32), ...)

    def process_weights_after_loading(self, layer):
        if self.w8a8_strategy == 1:   # Triton: 预热 + 缓存 autotune config
            ops.triton_int8_gemm_helper(m=..., n=..., k=..., best_config=...)
        elif self.w8a8_strategy == 3: # blaslt: 转置
            layer.weight.data = layer.weight.data.T
        else:                          # 默认: 重排
            layer.weight.data = weight_data.T.contiguous().reshape(n, -1)
        layer.weight = Parameter(layer.weight.t(), requires_grad=False)

    def apply(self, layer, x, bias=None, ...):
        return apply_int8_linear(input=x, weight=layer.weight,
                                weight_scale=layer.weight_scale, bias=bias,
                                w8a8_strategy=self.w8a8_strategy, ...)
```

### 8.3 SlimQuantW4A8Int8MoEMethod

```python
class SlimQuantW4A8Int8MoEMethod:
    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale, a2_scale=layer.w2_input_scale,
            per_act_token_quant=True, per_out_ch_quant=False,
            block_shape=None, weight_dtype='int4')   # ← INT4 权重！

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size, ...):
        # w13_weight: [E, 2*N, K//2] int8 (INT4 打包，K//2 因为 4-bit)
        # w2_weight:  [E, K, N//2] int8
        # w13_weight_scale: [E, 2*N, 1] float32 (per-channel)
        # w2_weight_scale:  [E, K, 1] float32

    def apply(self, layer, x, topk_weights, topk_ids, ...):
        return fused_experts(x, layer.w13_weight, layer.w2_weight,
                            topk_weights=topk_weights, topk_ids=topk_ids,
                            quant_config=self.moe_quant_config, ...)
```

**关键点**：
- 权重 dtype 是 `int8`，但 `weight_dtype='int4'` 告诉 FusedMoE kernel 实际是 4-bit 打包
- `K//2` 和 `N//2` 维度反映了 INT4 打包（2 个 4-bit 值打包成 1 个 int8）
- 使用 `lmslim` 库的 `per_token_quant_int8` 做激活量化
- `process_weights_after_loading` 预热 MoE Triton kernel 的 autotune config

### 8.4 SlimQuantW4A8Int8MarlinConfig（Marlin 变体）

```python
class SlimQuantW4A8Int8MarlinConfig(QuantizationConfig):
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        # 用户用 slimquant_w4a8 checkpoint + 指定 slimquant_w4a8_marlin → 升级为 Marlin kernel
        if hf_quant_cfg.get("quant_method") == "slimquant_w4a8" \
                and user_quant in ("slimquant_w4a8_marlin", "slimquant_marlin"):
            return "slimquant_w4a8_marlin"

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            return SlimQuantW4A8Int8MarlinMoEMethod(self, layer.moe_config)
```

Marlin 变体使用 `fused_experts_impl_w4a8_marlin`（来自 `lmslim`），比 Triton 更快。

---

## 9. 其他量化方案

### 9.1 ModelOpt（modelopt.py）

NVIDIA TensorRT-ModelOpt 量化，支持 FP8 和 NvFp4。

```python
class ModelOptFp8Config(ModelOptQuantConfigBase):   # FP8
    # 子类: ModelOptFp8PcPtLinearMethod, ModelOptFp8PbWoLinearMethod
class ModelOptNvFp4Config(ModelOptQuantConfigBase):  # NVFP4 (Blackwell)
```

支持 PCPT（Per-Channel Per-Tensor）和 PBWO（Per-Block Weight-Only）变体。

### 9.2 MXFP4（mxfp4.py）

Micro-exponent FP4（OCP MX 标准），4-bit 浮点 + block scaling。

```python
class Mxfp4Config(QuantizationConfig):
    # Mxfp4Backend enum: 选择 MoE 后端
class Mxfp4MoEMethod(FusedMoEMethodBase):
    # 权重 dequant_mxfp4 → 实际计算用 FP8/BF16 kernel
```

支持 `IpexMxfp4MoEMethod`（Intel XPU 变体）。

### 9.3 bitsandbytes（bitsandbytes.py）

在线量化（加载时即时量化），NF4/FP4/INT8。

```python
class BitsAndBytesConfig(QuantizationConfig):
    bnb_4bit_quant_type: str = "fp4"   # "fp4" 或 "nf4"
    bnb_4bit_compute_dtype: ...
    bnb_4bit_use_double_quant: bool    # 二次量化
```

### 9.4 GGUF（gguf.py）

llama.cpp 格式，支持多种位宽（Q4_K_M、Q8_0 等）。

### 9.5 Quark（quark/）

AMD Quark 量化框架。

### 9.6 TorchAO（torchao.py）

PyTorch 原生量化（torchao 库）。

### 9.7 BlockInt8（blockwise_int8.py）

块级 INT8 量化。

### 9.8 已废弃方法

`DEPRECATED_QUANTIZATION_METHODS`：`tpu_int8`、`ptpc_fp8`、`fbgemm_fp8`、`fp_quant`、`bitblas`、`gptq_marlin_24`、`gptq_bitblas`、`experts_int8`、`ipex`、`petit_nvfp4`。

---

## 10. KV Cache 量化

### 10.1 BaseKVCacheMethod

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    """为 Attention 层添加 _k_scale 和 _v_scale 属性"""
    def create_weights(self, layer):
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def apply(self, layer):
        # 若 k/v_scale <= 0（未从 checkpoint 加载）:
        #   若 calculate_kv_scales=True: 动态计算 scale
        #   否则: 强制 scale = 1.0（不量化）
```

### 10.2 FP8 KV Cache

`Fp8KVCacheMethod(BaseKVCacheMethod)`：KV Cache 以 FP8 存储，节省 50% KV Cache 显存。

### 10.3 CacheConfig 中的相关参数

```python
CacheConfig:
    cache_dtype: str = "auto"   # auto/fp8/fp8_e4m3/fp8_e5m2/int8/...
    calculate_kv_scales: bool = False  # 动态计算 KV scale（fp8）
    kv_cache_dtype_skip_layers: list[str]  # 跳过 KV 量化的层
```

---

## 11. 与 FusedMoE 的集成

量化与 MoE 的集成通过 `FusedMoEQuantConfig`（见 MoE 文档第 4.3 节）：

```python
# 各量化方法的 MoE 变体提供 get_fused_moe_quant_config()
def get_fused_moe_quant_config(self, layer) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        quant_dtype=torch.int8,           # 或 torch.float8_e4m3fn
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a1_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
        per_act_token_quant=True,
        per_out_ch_quant=False,
        block_shape=None,                 # 或 [128, 128]
        weight_dtype='int4',              # slimquant 用
    )
```

**MoE 量化 kernel 选择**（`oracle/`）：
- `select_fp8_moe_backend(...)` → FP8 后端（Triton/CUTLASS/DeepGEMM/FlashInfer）
- `select_nvfp4_moe_backend(...)` → NVFP4 后端
- `select_unquantized_moe_backend(...)` → 未量化后端

**模块化 kernel 切换**：权重加载后 `maybe_init_modular_kernel()` 将量化方法包装为 `FusedMoEModularMethod`（见 MoE 文档第 7 章）。

---

## 12. 量化方法选择指南

### 12.1 按精度需求

| 场景 | 推荐方案 | 精度损失 | 显存节省 |
|------|---------|---------|---------|
| 最高精度 | BF16（不量化） | 无 | 0% |
| 微损 + 硬件加速 | FP8 W8A8 | <1% | 50% |
| 小损 + 最大压缩 | GPTQ/AWQ INT4 W4A16 | 1-3% | 75% |
| 平衡（你用的） | SlimQuant W4A8 | 1-2% | ~75% 权重 + 50% 激活 |
| Blackwell 专属 | NVFP4 W4A4 | 2-4% | 75% |

### 12.2 按 GPU 架构

| GPU | 算力 | 推荐方案 |
|-----|------|---------|
| V100 | 70 | GPTQ/AWQ INT4 |
| T4/A10 | 75 | GPTQ/AWQ INT4, SlimQuant W4A8 |
| A100 | 80 | FP8 (Marlin), GPTQ Marlin, SlimQuant |
| H100/H200 | 89 | FP8 (CUTLASS/DeepGEMM), block FP8 |
| B200 | 100 | NVFP4, MXFP4, FP8 |

### 12.3 按 Kernel 优先级

对于同一量化格式，vLLM 自动选择最快 kernel：

```
GPTQ checkpoint:
  → GPTQMarlinConfig.override_quantization_method 自动升级为 gptq_marlin
  → 若 Marlin 不支持 → fallback GPTQ (ExLlamaV2/Triton)

FP8 checkpoint:
  → SM ≥ 89: CUTLASS / DeepGEMM
  → SM < 89: Marlin FP8 (weight-only)
  → MoE: select_fp8_moe_backend 选最优

SlimQuant W4A8:
  → slimquant_w4a8: Triton/CUTLASS (VLLM_W8A8_BACKEND)
  → slimquant_w4a8_marlin: Marlin kernel (更快)
```

---

## 13. 实际案例：GLM-5.1 slimquant_w4a8 分析

```bash
vllm serve /models/GLM-5.1-INT4-w4a8 \
    --quantization slimquant_w4a8 \
    -tp 8 --enable-expert-parallel \
    --dtype bfloat16 ...
```

### 13.1 量化配置解析

```python
# 1. get_quantization_config("slimquant_w4a8") → SlimQuantW4A8Int8Config
# 2. from_config({}) → SlimQuantW4A8Int8Config()  (无 config 文件, 硬编码)
# 3. 对每层:
#    LinearBase → SlimQuantW4A8Int8LinearMethod
#    FusedMoE   → SlimQuantW4A8Int8MoEMethod
#    Attention  → (无 KV 量化, get_quant_method 返回 None)
```

### 13.2 权重布局

```
Dense 层 (QKV/O proj, FFN):
  weight: [N, K] int8 (per-channel scale)
  weight_scale: [N, 1] float32
  → apply_int8_linear(w8a8_strategy=VLLM_W8A8_BACKEND)

MoE 层:
  w13_weight: [E, 2*N, K//2] int8 (INT4 打包)
  w2_weight:  [E, K, N//2] int8
  w13_weight_scale: [E, 2*N, 1] float32 (per-channel)
  w2_weight_scale:  [E, K, 1] float32
  → fused_experts(quant_config=FusedMoEQuantConfig(quant_dtype=int8, weight_dtype='int4'))
```

### 13.3 推理流程

```
Dense 层:
  1. x_q, x_scale = per_token_quant_int8(x)        # 动态 per-token INT8 量化
  2. ops.triton_scaled_mm(x_q, weight, x_scale, weight_scale)  # strategy=1
     或 ops.cutlass_scaled_mm(...)                  # strategy=2

MoE 层:
  1. router.select_experts → topk_weights, topk_ids
  2. fused_experts(x, w13, w2, quant_config=...):
     a. moe_kernel_quantize_input(x, quant_dtype=int8)  # per-token INT8
     b. moe_align_block_size(topk_ids)
     c. GEMM1: fused_moe_kernel(q_x, w13, ..., use_int4_w4a8=True)  # INT4 权重
     d. silu_and_mul
     e. GEMM2: fused_moe_kernel(q_a2, w2, ..., use_int4_w4a8=True)
     f. moe_sum (topk 归约)
```

### 13.4 显存节省估算

```
GLM-5.1 假设 300B 参数 (MoE):
  BF16:  300B × 2 bytes = 600 GB
  W4A8:  权重 300B × 0.5 bytes (INT4) = 150 GB
         激活 INT8 (动态, 不占持久显存)
  节省: ~75% 权重显存

TP=8 + EP=8:
  每 GPU: 150GB / 8 ≈ 18.75 GB 权重 (可放入单卡)
```

---

## 14. 配置参数与环境变量

### 14.1 启动参数

| 参数 | 说明 |
|------|------|
| `--quantization <method>` | 量化方法（slimquant_w4a8 / fp8 / gptq / ...） |
| `--kv-cache-dtype <dtype>` | KV Cache dtype（auto / fp8 / int8） |
| `--dtype <dtype>` | 模型计算 dtype（bfloat16 / float16） |

### 14.2 环境变量

```bash
# SlimQuant / W8A8
VLLM_W8A8_BACKEND=1          # 1=triton, 2=cutlass, 3=blaslt

# FP8
VLLM_TEST_FORCE_FP8_MARLIN=0 # 强制 FP8 用 Marlin kernel

# GPTQ / Marlin
VLLM_USE_MOE_W16A16_TRITON=0 # W16A16 用 Triton 而非 Marlin

# DeepGEMM
VLLM_USE_DEEP_GEMM=0         # 启用 DeepGEMM FP8

# Fused MoE
VLLM_FUSED_MOE_CHUNK_SIZE=32768  # MoE 分块大小

# KV Cache
VLLM_FLASHINFER_MOE_BACKEND=throughput  # FlashInfer MoE 后端

# fused RMS + quant (slimquant 相关)
USE_FUSED_RMS_QUANT=1        # 融合 RMSNorm + 量化
```

### 14.3 CacheConfig 相关

```python
CacheConfig:
    cache_dtype: str = "auto"               # KV cache dtype
    calculate_kv_scales: bool = False       # 动态计算 KV scale
    kv_cache_dtype_skip_layers: list[str]   # 跳过 KV 量化的层
```

---

## 15. 常见问题

### 15.1 量化选择

| 问题 | 原因 | 解决 |
|------|------|------|
| `Invalid quantization method` | 方法名拼写错误 | 检查 `QUANTIZATION_METHODS` 列表 |
| 精度异常 | 量化方案不适合模型 | FP8 > INT4；启用 `desc_act`（GPTQ） |
| 显存未减少 | 在线量化未生效 | 确认 `is_checkpoint_fp8_serialized` |
| 自动升级为 marlin | `override_quantization_method` | 显式 `--quantization gptq` 禁用 |

### 15.2 Kernel 问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `Marlin not supported` | SM < 80 或形状不符 | 升级 GPU 或用 Triton fallback |
| CUTLASS 报错 | SM < 89（FP8） | 用 Marlin FP8 或降级到 INT4 |
| DeepGEMM 不可用 | 未安装或形状不对 | `VLLM_USE_DEEP_GEMM=0` |
| Triton autotune 慢 | 首次启动调优 | `W8a8GetCacheJSON` 会缓存，二次启动变快 |
| NVFP4 报错 | 非 Blackwell GPU | 用 MXFP4 或 FP8 替代 |

### 15.3 SlimQuant 特定

| 问题 | 原因 | 解决 |
|------|------|------|
| `Please install lmslim` | 缺少 lmslim 库 | 安装 `lmslim`（slimquant 依赖） |
| W8A8 kernel 选错 | `VLLM_W8A8_BACKEND` 未设 | 设 1（triton）/ 2（cutlass）/ 3（blaslt） |
| MoE INT4 权重形状错误 | `K//2` 打包理解错误 | INT4 打包：2 个 4-bit → 1 个 int8 |
| Marlin 变体不生效 | `override` 条件不符 | 显式 `--quantization slimquant_w4a8_marlin` |

### 15.4 性能调优

| 优化 | 方法 |
|------|------|
| FP8 启用 block 量化 | checkpoint 提供 `weight_block_size=[128,128]` |
| Marlin 加速 INT4 | 让 `override_quantization_method` 自动升级 |
| DeepGEMM FP8 | `VLLM_USE_DEEP_GEMM=1`（Hopper+） |
| 融合 RMS+量化 | `USE_FUSED_RMS_QUANT=1` |
| Triton 调优缓存 | `W8a8GetCacheJSON` 自动缓存 autotune |
| KV Cache FP8 | `--kv-cache-dtype fp8`（节省 50% KV 显存） |

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-08-13  
> **核心文件**：`vllm/model_executor/layers/quantization/__init__.py`, `base_config.py`, `fp8.py`, `gptq.py`, `awq.py`, `gptq_marlin.py`, `slimquant_w4a8.py`, `slimquant_w4a8_marlin.py`, `compressed_tensors/compressed_tensors.py`, `utils/w8a8_utils.py`, `utils/quant_utils.py`, `kv_cache.py`
