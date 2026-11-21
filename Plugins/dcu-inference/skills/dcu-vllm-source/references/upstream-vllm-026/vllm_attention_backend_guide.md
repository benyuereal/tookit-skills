# vLLM V1 Attention Backend 体系详解

> **适用版本**: vLLM v0.15.x+ (V1 Engine)
> **核心目录**: `vllm/v1/attention/`, `vllm/v1/attention/backends/`, `vllm/v1/attention/backends/mla/`, `vllm/platforms/`
> **最后更新**: 2026-07-25

---

## 目录

1. [概述](#1-概述)
2. [核心抽象：AttentionBackend 基类](#2-核心抽象attentionbackend-基类)
3. [Backend 注册表 (Registry)](#3-backend-注册表-registry)
4. [Backend 选择机制 (Selector + Platform)](#4-backend-选择机制-selector--platform)
5. [AttentionMetadataBuilder 元数据构建器](#5-attentionmetadatabuilder-元数据构建器)
6. [AttentionImpl 注意力实现](#6-attentionimpl-注意力实现)
7. [CommonAttentionMetadata 公共元数据](#7-commonattentionmetadata-公共元数据)
8. [KVCacheSpec 体系](#8-kvcachespec-体系)
9. [标准 Attention Backend 详解](#9-标准-attention-backend-详解)
10. [MLA Backend 体系](#10-mla-backend-体系)
11. [ROCm Backend 体系](#11-rocm-backend-体系)
12. [Mamba/线性注意力 Backend](#12-mamba线性注意力-backend)
13. [特殊 Backend](#13-特殊-backend)
14. [CUDA Graph 支持等级](#14-cuda-graph-支持等级)
15. [Cascade Attention 级联注意力](#15-cascade-attention-级联注意力)
16. [Backend 选择决策流程图](#16-backend-选择决策流程图)
17. [配置参数与最佳实践](#17-配置参数与最佳实践)

---

## 1. 概述

### 1.1 Attention Backend 在 vLLM 中的位置

Attention Backend 是 vLLM V1 中**最核心的算子抽象层**，它将不同硬件平台、不同注意力算法（标准 MHA/GQA、MLA、线性注意力等）统一封装在一致的接口之下，让上层 ModelRunner 和模型代码无需关心底层实现细节。

```
┌──────────────────────────────────────────────────────────────┐
│                      ModelRunner                              │
│   _build_attention_metadata()  →  AttentionMetadataBuilder    │
│   _model_forward()             →  Attention layer             │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   Attention Layer (Attention/MLAAttention)    │
│   根据 attn_backend 创建 AttentionImpl                        │
│   forward() → AttentionImpl.forward()                         │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   AttentionBackend (抽象基类)                  │
│   ├─ get_impl_cls()    → AttentionImpl (实际计算)              │
│   ├─ get_builder_cls() → AttentionMetadataBuilder (元数据)     │
│   ├─ get_kv_cache_shape()  (KV cache 张量形状)                │
│   └─ validate_configuration() (能力校验)                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
        ┌──────────┬───────┴────────┬───────────┬──────────────┐
        ▼          ▼                ▼           ▼              ▼
   FlashAttn  FlashInfer      TritonAttn   CUTLASS_MLA    ROCm_Aiter
   (NVIDIA)   (NVIDIA/多)      (通用)       (Blackwell)    (AMD)
```

### 1.2 核心组件

Attention Backend 体系由以下核心组件构成：

| 组件 | 文件 | 职责 |
|------|------|------|
| `AttentionBackend` | `vllm/v1/attention/backend.py` | 抽象基类，定义 backend 契约 |
| `AttentionBackendEnum` | `backends/registry.py` | 所有 backend 的枚举注册表 |
| `AttentionSelectorConfig` | `vllm/v1/attention/selector.py` | backend 选择配置 |
| `Platform.get_attn_backend_cls()` | `vllm/platforms/*.py` | 平台相关的 backend 选择逻辑 |
| `AttentionMetadataBuilder` | `vllm/v1/attention/backend.py` | 构建 per-layer 元数据 |
| `AttentionImpl` | `vllm/v1/attention/backend.py` | 实际的注意力计算 |
| `CommonAttentionMetadata` | `vllm/v1/attention/backend.py` | 跨层/跨 backend 共享的批次元数据 |
| `KVCacheSpec` | `vllm/v1/kv_cache_interface.py` | 描述每层的 KV cache 格式 |
| `AttentionConfig` | `vllm/config/attention.py` | 用户可配置的 attention 参数 |

### 1.3 Backend 数量概览

vLLM V1 注册了 **22 种标准 Attention backend** 和 **6 种 Mamba 类 backend**：

- 标准 backend：FLASH_ATTN, FLASH_ATTN_DIFFKV, TRITON_ATTN, ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, TORCH_SDPA, FLASHINFER, FLEX_ATTENTION, TREE_ATTN, CPU_ATTN, TURBOQUANT, IPEX, NO_ATTENTION, 以及 8 种 MLA 变体
- MLA backend：ROCM_AITER_MLA, ROCM_AITER_TRITON_MLA, ROCM_AITER_MLA_SPARSE, FLASHINFER_MLA, TRITON_MLA, CUTLASS_MLA, FLASHMLA, FLASHMLA_SPARSE, FLASH_ATTN_MLA
- Mamba 类：MAMBA1, MAMBA2, SHORT_CONV, LINEAR, GDN_ATTN

---

## 2. 核心抽象：AttentionBackend 基类

`AttentionBackend` (`vllm/v1/attention/backend.py:46`) 是所有 attention backend 的抽象基类，定义了 backend 必须实现的能力契约。

### 2.1 类属性（能力声明）

```python
class AttentionBackend(ABC):
    # 是否需要预先分配 output tensor (piecewise cudagraph 需要)
    accept_output_buffer: bool = False
    
    # 支持的数据类型
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    
    # 支持的 KV cache 数据类型
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = ["auto", "bfloat16"]
    
    # forward() 是否包含 KV cache 更新 (reshape_and_cache)
    # False 表示 KV cache 更新在 forward 外单独进行
    forward_includes_kv_cache_update: bool = True
```

### 2.2 抽象方法（必须实现）

| 方法 | 说明 |
|------|------|
| `get_name() -> str` | backend 名称（如 "FLASH_ATTN"） |
| `get_impl_cls() -> type[AttentionImpl]` | 返回注意力实现类 |
| `get_builder_cls()` | 返回元数据构建器类 |
| `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)` | 返回 KV cache 张量的形状 |

### 2.3 可选方法（能力查询）

```python
@staticmethod
def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
    """支持的 block size。
    MultipleOf(16) 表示支持 16 的倍数；
    [16, 32, 64] 表示只支持这三个固定值。"""
    return [MultipleOf(1)]

@classmethod
def get_supported_head_sizes(cls) -> list[int]:
    """支持的 head size。空列表表示不限制。"""
    return []

@classmethod
def is_mla(cls) -> bool:
    """是否是 MLA (Multi-head Latent Attention) backend。"""
    return False

@classmethod
def supports_sink(cls) -> bool:
    """是否支持 attention sink。"""
    return False

@classmethod
def supports_mm_prefix(cls) -> bool:
    """是否支持多模态 partial token 的 full attention。"""
    return False

@classmethod
def is_sparse(cls) -> bool:
    """是否是 sparse attention backend。"""
    return False

@classmethod
def supports_attn_type(cls, attn_type: str) -> bool:
    """支持的 attention 类型。默认只支持 DECODER。"""
    return attn_type == AttentionType.DECODER

@classmethod
def supports_compute_capability(cls, capability) -> bool:
    """是否支持某 GPU 架构。"""
    return True

@classmethod
def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
    """要求特定的 KV cache 内存布局 (NHD/HND)。"""
    return None
```

### 2.4 配置校验方法

`validate_configuration()` (`backend.py:217-272`) 是 backend 选择时的核心校验逻辑，逐项检查 backend 是否支持当前配置：

```python
@classmethod
def validate_configuration(cls, head_size, dtype, kv_cache_dtype, block_size,
    use_mla, has_sink, use_sparse, use_mm_prefix, device_capability, attn_type
) -> list[str]:
    invalid_reasons = []
    if not cls.supports_head_size(head_size):
        invalid_reasons.append("head_size not supported")
    if not cls.supports_dtype(dtype):
        invalid_reasons.append("dtype not supported")
    if not cls.supports_kv_cache_dtype(kv_cache_dtype):
        invalid_reasons.append("kv_cache_dtype not supported")
    if not cls.supports_block_size(block_size):
        invalid_reasons.append("block_size not supported")
    if use_mm_prefix and not cls.supports_mm_prefix():
        invalid_reasons.append("partial multimodal token full attention not supported")
    if use_mla != cls.is_mla():
        invalid_reasons.append("MLA not supported" if use_mla else "non-MLA not supported")
    if has_sink and not cls.supports_sink():
        invalid_reasons.append("sink setting not supported")
    if use_sparse != cls.is_sparse():
        invalid_reasons.append("sparse not supported" if use_sparse else "non-sparse not supported")
    if not cls.supports_compute_capability(device_capability):
        invalid_reasons.append("compute capability not supported")
    if not cls.supports_attn_type(attn_type):
        invalid_reasons.append(f"attention type {attn_type} not supported")
    # 组合校验 (如某 backend 不支持 head_size+dtype 的特定组合)
    combination_reason = cls.supports_combination(...)
    if combination_reason is not None:
        invalid_reasons.append(combination_reason)
    return invalid_reasons  # 空列表 = 配置有效
```

### 2.5 AttentionType 枚举

```python
class AttentionType(str, Enum):
    DECODER = "decoder"              # Decoder 自注意力
    ENCODER = "encoder"              # Encoder-decoder 的 encoder 自注意力
    ENCODER_ONLY = "encoder_only"    # Encoder-only 模型 (如 ViT)
    ENCODER_DECODER = "encoder_decoder"  # Decoder Q 对 encoder K/V 的 cross-attention
```

---

## 3. Backend 注册表 (Registry)

### 3.1 AttentionBackendEnum (`backends/registry.py:34`)

所有标准 attention backend 通过枚举注册，枚举值是 backend 类的完整路径：

```python
class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_DIFFKV = "vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend"
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    ROCM_AITER_MLA = "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"
    ROCM_AITER_TRITON_MLA = "vllm.v1.attention.backends.mla.aiter_triton_mla.AiterTritonMLABackend"
    ROCM_AITER_FA = "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
    ROCM_AITER_MLA_SPARSE = "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend"
    TORCH_SDPA = ""  # 仅用于 ViT
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLASHINFER_MLA = "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    FLASHMLA_SPARSE = "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    IPEX = "vllm.v1.attention.backends.ipex.IpexAttentionBackend"
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    TREE_ATTN = "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend"
    ROCM_AITER_UNIFIED_ATTN = "vllm.v1.attention.backends.rocm_aiter_unified_attn.RocmAiterUnifiedAttentionBackend"
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
    TURBOQUANT = "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"
    CUSTOM = None  # 第三方自定义 backend，使用前必须注册
```

### 3.2 枚举的核心方法

```python
def get_path(self, include_classname=True) -> str:
    """获取 backend 类路径（尊重运行时 override）。
    CUSTOM 必须先注册，否则报错。"""
    path = _ATTN_OVERRIDES.get(self, self.value)
    if not path:
        raise ValueError(f"Backend {self.name} must be registered before use.")
    return path

def get_class(self) -> type["AttentionBackend"]:
    """通过 qualname 动态导入并返回 backend 类。"""
    return resolve_obj_by_qualname(self.get_path())

def is_overridden(self) -> bool:
    """是否被运行时 override。"""
    return self in _ATTN_OVERRIDES
```

### 3.3 MambaAttentionBackendEnum (`registry.py:130`)

线性注意力/状态空间模型的 backend 单独注册：

```python
class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    MAMBA1 = "vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend"
    MAMBA2 = "vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend"
    SHORT_CONV = "vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend"
    LINEAR = "vllm.v1.attention.backends.linear_attn.LinearAttentionBackend"
    GDN_ATTN = "vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend"
    CUSTOM = None

MAMBA_TYPE_TO_BACKEND_MAP = {
    "mamba1": "MAMBA1",
    "mamba2": "MAMBA2",
    "short_conv": "SHORT_CONV",
    "linear_attention": "LINEAR",
    "gdn_attention": "GDN_ATTN",
    "custom": "CUSTOM",
}
```

### 3.4 register_backend() 运行时覆盖

支持第三方覆盖已有 backend 或注册自定义 backend：

```python
@register_backend(AttentionBackendEnum.FLASH_ATTN)
class MyCustomFlashAttn:
    ...

@register_backend(AttentionBackendEnum.CUSTOM)
class MyCustomBackend:
    ...

# 或直接注册路径
register_backend(AttentionBackendEnum.CUSTOM, "my.module.MyCustomBackend")
```

覆盖通过 `_ATTN_OVERRIDES: dict[AttentionBackendEnum, str]` 字典实现，`get_path()` 优先返回 override 路径。

---

## 4. Backend 选择机制 (Selector + Platform)

### 4.1 选择入口：get_attn_backend()

`vllm/v1/attention/selector.py:46` 是 backend 选择的统一入口：

```python
def get_attn_backend(head_size, dtype, kv_cache_dtype, block_size,
    use_mla=False, has_sink=False, use_sparse=False,
    use_mm_prefix=False, attn_type=None
) -> type[AttentionBackend]:
    """选择并延迟导入 attention backend。"""
    
    # 1. 校验 kv_cache_dtype
    if kv_cache_dtype is not None:
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes
    
    # 2. 从当前 vllm_config 获取用户指定的 backend
    vllm_config = get_current_vllm_config()
    backend_enum = vllm_config.attention_config.backend
    
    # 3. 构建选择配置
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size, dtype=dtype, kv_cache_dtype=kv_cache_dtype,
        block_size=block_size, use_mla=use_mla, has_sink=has_sink,
        use_sparse=use_sparse, use_mm_prefix=use_mm_prefix,
        attn_type=attn_type or AttentionType.DECODER,
    )
    
    # 4. 带缓存的解析
    return _cached_get_attn_backend(backend=backend_enum, attn_selector_config=attn_selector_config)

@cache  # 相同参数只解析一次
def _cached_get_attn_backend(backend, attn_selector_config):
    # 委托给当前平台的实现
    attention_cls = current_platform.get_attn_backend_cls(backend, attn_selector_config)
    backend = resolve_obj_by_qualname(attention_cls)
    
    # 如果 backend 要求特定 KV cache 布局，设置之
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        set_kv_cache_layout(required_layout)
    
    return backend
```

### 4.2 AttentionSelectorConfig

```python
class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    attn_type: str = AttentionType.DECODER
```

### 4.3 平台层选择逻辑

每个平台 (`CudaPlatform`, `RocmPlatform`, `CpuPlatform`, `XpuPlatform`) 实现 `get_attn_backend_cls()`，决定该平台下用哪个 backend。

#### 4.3.1 CUDA 平台 (`platforms/cuda.py:308`)

CUDA 平台的选择逻辑分两层：**用户指定优先**，否则按**优先级列表**自动选择。

**用户指定 backend** (`selected_backend is not None`)：
```python
if selected_backend is not None:
    try:
        backend_class = selected_backend.get_class()
        invalid_reasons = backend_class.validate_configuration(
            device_capability=device_capability, **attn_selector_config._asdict()
        )
    except ImportError:
        invalid_reasons = ["ImportError"]
    if invalid_reasons:
        raise ValueError(f"Selected backend {selected_backend} is not valid. Reason: {invalid_reasons}")
    return selected_backend.get_path()  # 直接使用用户指定的
```

**自动选择**（无指定 backend）：
```python
valid_backends_priorities, invalid_reasons = cls.get_valid_backends(...)
# 按优先级排序，选最高优先级的有效 backend
sorted_indices = sorted(range(len(valid_backends_priorities)),
                        key=lambda i: valid_backends_priorities[i][1])
selected_backend = valid_backends_priorities[sorted_indices[0]][0]
return selected_backend.get_path()
```

**get_valid_backends()** (`cuda.py:272`)：
```python
def get_valid_backends(cls, device_capability, attn_selector_config):
    # TurboQuant KV cache: 直接路由到 TQ backend
    if kv_cache_dtype.startswith("turboquant_"):
        return [(AttentionBackendEnum.TURBOQUANT, 0)], {}
    
    backend_priorities = _get_backend_priorities(use_mla, device_capability)
    valid_backends = []
    for priority, backend in enumerate(backend_priorities):
        backend_class = backend.get_class()
        invalid_reasons = backend_class.validate_configuration(...)
        if not invalid_reasons:
            valid_backends.append((backend, priority))  # priority 越小越优先
    return valid_backends, invalid_reasons
```

#### 4.3.2 Backend 优先级 (`cuda.py:45`)

`_get_backend_priorities()` 根据 MLA 和 GPU 架构返回优先级列表：

```python
def _get_backend_priorities(use_mla, device_capability):
    if use_mla:
        if device_capability.major == 10:  # Blackwell
            return [FLASHINFER_MLA, CUTLASS_MLA, FLASH_ATTN_MLA, FLASHMLA, TRITON_MLA, FLASHMLA_SPARSE]
        else:  # Hopper/Ampere
            return [FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA, FLASHMLA_SPARSE]
    else:
        if device_capability.major == 10:  # Blackwell
            return [FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION]
        else:  # Hopper/Ampere
            return [FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION]
```

**关键观察**：
- Blackwell (SM100) 上 FlashInfer 优先级最高（TRT-LLM kernel 优化）
- 非 Blackwell 上 FlashAttention 优先级最高
- MLA 场景在 Blackwell 上优先 FlashInfer_MLA，其他上优先 FlashAttn_MLA

#### 4.3.3 MLA 的 block_size 强制调整 (`cuda.py:170-247`)

CUDA 平台会根据 MLA backend 强制调整 block_size：

```python
if model_config.use_mla and cache_config.block_size is not None:
    use_sparse = hasattr(hf_config, "index_topk")
    
    if attention_config.backend is None:  # 自动选择
        if is_blackwell and not use_sparse and qk_nope_head_dim == 128:
            use_flashinfer_mla = True  # Blackwell + DSv3.1 → FlashInfer MLA
            attention_config.backend = FLASHINFER_MLA
        elif is_blackwell and not use_sparse:
            use_cutlass_mla = True  # Blackwell 其他 → CUTLASS MLA
        elif is_flashmla_dense_supported()[0]:
            use_flashmla = True  # 非 Blackwell + FlashMLA 支持
    else:  # 用户指定
        use_flashmla = backend == FLASHMLA
        use_cutlass_mla = backend == CUTLASS_MLA
        use_flashinfer_mla = backend == FLASHINFER_MLA
    
    # 强制 block_size
    if use_flashmla and block_size % 64 != 0:
        block_size = 64  # FlashMLA 要求 64
    if use_cutlass_mla and block_size % 128 != 0:
        block_size = 128  # CUTLASS MLA 要求 128
    if use_flashinfer_mla and block_size % 64 != 0:
        block_size = 64  # FlashInfer MLA 要求 64
    if use_sparse and block_size != 64:
        block_size = 64  # Sparse MLA 要求 64
```

#### 4.3.4 ROCm 平台 (`platforms/rocm.py:257`)

ROCm 平台的选择逻辑更硬编码：

```python
def get_attn_backend_cls(cls, selected_backend, attn_selector_config):
    # TurboQuant 直接路由
    if kv_cache_dtype.startswith("turboquant_"):
        return TURBOQUANT.get_path()
    
    # Sparse MLA
    if use_sparse:
        return FLASHMLA_SPARSE.get_path()
    
    # MLA
    if use_mla:
        use_flashmla = selected_backend == FLASHMLA or envs.VLLM_USE_FLASH_MLA
        use_triton = selected_backend == TRITON_MLA or selected_backend is None
        if use_flashmla and block_size == 64:
            return FLASHMLA.get_path()
        if use_triton:
            return TRITON_MLA.get_path()  # ROCm 默认 Triton MLA
        raise ValueError(...)
    
    # 标准 attention
    if envs.VLLM_USE_FLASH_ATTN_PA and block_size == 64:
        return FLASH_ATTN.get_path()
    else:
        return TRITON_ATTN.get_path()  # ROCm 默认 Triton
```

#### 4.3.5 ViT attention backend

ViT (Vision Transformer) 模型有独立的 backend 选择：
```python
def get_vit_attn_backend(cls, head_size, dtype, backend=None):
    if backend is not None:
        return backend
    if cc.major >= 8:  # Ampere+
        try:
            if FLASH_ATTN.supports_head_size(head_size) and FLASH_ATTN.supports_dtype(dtype):
                return FLASH_ATTN
        except ImportError:
            pass
    return TORCH_SDPA  # 回退到 PyTorch SDPA
```

---

## 5. AttentionMetadataBuilder 元数据构建器

### 5.1 抽象基类 (`backend.py:433`)

`AttentionMetadataBuilder` 负责为每层 attention 构建执行所需的元数据：

```python
class AttentionMetadataBuilder(ABC, Generic[M]):
    # CUDA Graph 支持等级 (默认不支持)
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # 是否需要重排 batch (None=不重排，否则设为要拉到前面的 query 长度)
    reorder_batch_threshold: int | None = None
    # 是否支持更新已有 metadata 的 block table (多 KV cache group 优化)
    supports_update_block_table: bool = False
    
    @abstractmethod
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
    
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        return cls._cudagraph_support
    
    @abstractmethod
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False) -> M:
        """构建 attention metadata。
        common_prefix_len: batch 公共前缀长度 (cascade attention 用)
        fast_build: 优先构建速度而非执行速度 (spec-decode 用)
        """
    
    def update_block_table(self, metadata, blk_table, slot_mapping) -> M:
        """更新已有 metadata 的 block table (supports_update_block_table=True 时)。"""
    
    def build_for_cudagraph_capture(self, common_attn_metadata) -> M:
        """为 CUDA graph capture 构建 metadata (默认调用 build)。"""
        return self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
    
    def build_for_drafting(self, common_attn_metadata, draft_index) -> M:
        """为 draft model 构建 metadata (默认 fast_build=True)。"""
        return self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata, fast_build=True)
    
    def use_cascade_attention(self, ...) -> bool:
        """是否使用 cascade attention (默认 False)。"""
        return False
```

### 5.2 _init_reorder_batch_threshold

```python
def _init_reorder_batch_threshold(self, reorder_batch_threshold=1,
    supports_spec_as_decode=False, supports_dcp_with_varlen=False):
    self.reorder_batch_threshold = reorder_batch_threshold
    # spec-as-decode: 根据 spec token 数调整 threshold
    if self.reorder_batch_threshold is not None and supports_spec_as_decode:
        if speculative_config and speculative_config.num_speculative_tokens:
            self.reorder_batch_threshold = max(
                self.reorder_batch_threshold, 1 + num_speculative_tokens
            )
    # DCP 且不支持 varlen 时强制 threshold=1
    if dcp_world_size > 1 and not supports_dcp_with_varlen:
        self.reorder_batch_threshold = 1
```

### 5.3 build() 的典型实现 (FlashAttention)

`FlashAttentionMetadataBuilder.build()` (`flash_attn.py:384`) 的关键步骤：
1. 从 `common_attn_metadata` 提取 query_start_loc, seq_lens, block_table 等
2. 决定是否使用 AOT (Ahead-Of-Time) 调度（FA3 特性）
3. 处理 sliding window
4. 构建 `FlashAttentionMetadata`（包含 per-req 的 query_lens, seq_lens, block_table 等）
5. 如果使用 cascade attention，构建 cascade 相关元数据

### 5.4 Metadata Builder 在 ModelRunner 中的使用

ModelRunner 的 `_build_attention_metadata()` 为每个 KV cache group × attention group 调用 builder：

```python
# 优化: 相同 (KVCacheSpec, builder_type) 的 metadata 可复用
cached_attn_metadata = {}
for kv_cache_gid, attn_gid in ...:
    builder = attn_group.get_metadata_builder(ubid or 0)
    cache_key = (kv_cache_spec, type(builder))
    
    if cache_key in cached_attn_metadata and builder.supports_update_block_table:
        # 复用: 只更新 block table
        attn_metadata_i = builder.update_block_table(
            cached_attn_metadata[cache_key],
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.slot_mapping,
        )
    else:
        # 完整构建
        attn_metadata_i = builder.build(
            common_prefix_len=cascade_attn_prefix_len,
            common_attn_metadata=common_attn_metadata,
        )
        if builder.supports_update_block_table:
            cached_attn_metadata[cache_key] = attn_metadata_i
```

---

## 6. AttentionImpl 注意力实现

### 6.1 抽象基类 (`backend.py:597`)

`AttentionImpl` 是实际执行注意力计算的类：

```python
class AttentionImpl(ABC, Generic[T]):
    num_heads: int
    head_size: int
    scale: float
    
    # decode 时能否返回 softmax lse (DCP 需要)
    can_return_lse_for_decode: bool = False
    # 是否支持 Prefill Context Parallelism
    supports_pcp: bool = False
    # 是否支持 MTP (cp_kv_cache_interleave_size > 1)
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False
    # 是否实际需要返回 lse (DCP 启用时自动置 True)
    need_to_return_lse_for_decode: bool = False
    # 是否支持预量化 query 输入 (FP8 KV cache)
    supports_quant_query_input: bool = False
    supports_per_head_quant_scales: bool = False
    
    # 并行配置 (在 __new__ 中自动从 parallel_state 获取)
    dcp_world_size: int  # Decode Context Parallelism
    dcp_rank: int
    pcp_world_size: int  # Prefill Context Parallelism
    pcp_rank: int
    total_cp_world_size: int
    total_cp_rank: int
    
    def __new__(cls, *args, **kwargs):
        # 自动初始化 CP 相关属性
        self.dcp_world_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        self.pcp_world_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
    
    @abstractmethod
    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes,
        sliding_window, kv_cache_dtype, logits_soft_cap, attn_type,
        kv_sharing_target_layer_name):
        ...
    
    @abstractmethod
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
        output=None, output_scale=None, output_block_scale=None) -> torch.Tensor:
        """执行注意力计算。
        query: [num_tokens, num_heads, head_size]
        key/value: [num_tokens, num_kv_heads, head_size]
        kv_cache: backend 特定形状
        返回: [num_tokens, num_heads * head_size]
        """
    
    def fused_output_quant_supported(self, quant_key) -> bool:
        """是否支持融合的输出量化 (AttnFusionPass 用)。"""
        return False
    
    def process_weights_after_loading(self, act_dtype):
        """权重加载后的处理 hook。"""
        pass
```

### 6.2 MLAAttentionImpl (`backend.py:711`)

MLA 专属的 impl 基类，扩展了 MLA 特有的 forward 签名：

```python
class MLAAttentionImpl(AttentionImpl[T], Generic[T]):
    @abstractmethod
    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes,
        sliding_window, kv_cache_dtype, logits_soft_cap, attn_type,
        kv_sharing_target_layer_name,
        # MLA 特有参数
        q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
        qk_head_dim, v_head_dim, kv_b_proj, indexer=None):
        ...
    
    @abstractmethod
    def forward(self, layer, hidden_states_or_cq, kv_c_normed, k_pe,
        kv_cache, attn_metadata, output=None, output_scale=None,
        output_block_scale=None) -> torch.Tensor:
        """MLA 特有的 forward:
        hidden_states_or_cq: 压缩 query (cq) 或 hidden states
        kv_c_normed: 压缩 KV
        k_pe: decoupled RoPE key
        """
```

---

## 7. CommonAttentionMetadata 公共元数据

### 7.1 数据结构 (`backend.py:286`)

`CommonAttentionMetadata` 是跨层、跨 backend 共享的批次级元数据，由 ModelRunner 构建，传给各 builder：

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor        # (batch+1,) 每个 request 在 query 中的起始位置
    query_start_loc_cpu: torch.Tensor    # CPU 版本
    
    seq_lens: torch.Tensor               # (batch,) 每个 request 的序列长度
    num_reqs: int                        # 请求数
    num_actual_tokens: int               # 实际 token 数 (可能 padded)
    max_query_len: int                   # 最长 query
    max_seq_len: int                     # 最长 context (可能是上界)
    
    block_table_tensor: torch.Tensor     # block table
    slot_mapping: torch.Tensor           # slot 映射
    
    causal: bool = True                  # 是否 causal
    
    # FastPrefillAttentionBuilder 用
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None
    
    # CrossAttentionBuilder 用
    encoder_seq_lens: torch.Tensor | None = None
    encoder_seq_lens_cpu: np.ndarray | None = None
    
    # Decode Context Parallelism
    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    
    # Deprecated (v0.15.0 移除)
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cache: torch.Tensor | None = None
```

### 7.2 关键方法

```python
def batch_size(self) -> int:
    return self.seq_lens.shape[0]

def naive_query_lens(self) -> torch.Tensor:
    """假设 query 结束于下一个 query 开始位置的 query 长度。"""
    return self.query_start_loc[1:] - self.query_start_loc[:-1]

def compute_num_computed_tokens(self) -> torch.Tensor:
    """在 device 上计算 num_computed_tokens (seq_lens - query_lens)。"""
    if self._num_computed_tokens_cache is None:
        query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
        self._num_computed_tokens_cache = self.seq_lens - query_lens
    return self._num_computed_tokens_cache

def unpadded(self, num_actual_tokens, num_actual_reqs) -> "CommonAttentionMetadata":
    """创建去 padding 的视图 (FULL-CG spec-decode 用)。"""
```

### 7.3 注意：deprecated 字段

`seq_lens_cpu` 和 `num_computed_tokens_cpu` 已 deprecated，因为隐式 H↔D 同步会破坏 async scheduling。推荐直接用 device 上的 `seq_lens`，需要 CPU 副本时显式 `.cpu()`。

---

## 8. KVCacheSpec 体系

### 8.1 KVCacheSpec 基类 (`vllm/kv_cache_interface.py:19`)

`KVCacheSpec` 描述模型每一层的 KV cache 格式，由 Attention layer 在 `get_kv_cache_spec()` 中产出：

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int  # 每个 block 的 token 数
    
    @property
    def page_size_bytes(self) -> int:
        """一个 page 的字节数。"""
        raise NotImplementedError
    
    def max_memory_usage_bytes(self, vllm_config) -> int:
        """该层 KV cache 的最大内存使用 (字节)。"""
        raise NotImplementedError
    
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """合并多个相同类型的 spec (同 KV cache group 内)。"""
```

### 8.2 AttentionSpec 及其子类

```
KVCacheSpec
└── AttentionSpec (block_size, num_kv_heads, head_size, dtype, page_size_padded)
    ├── FullAttentionSpec         # 标准 full attention (含 sliding_window, attention_chunk_size)
    │   ├── TQFullAttentionSpec   # TurboQuant 感知
    │   ├── MLAAttentionSpec      # MLA (DeepSeek)
    │   └── SinkFullAttentionSpec # 带 attention sink
    ├── ChunkedLocalAttentionSpec  # 分块局部注意力
    ├── SlidingWindowSpec          # 滑动窗口注意力
    └── EncoderOnlyAttentionSpec   # Encoder-only (无 KV cache)
    └── CrossAttentionSpec         # Cross-attention (encoder-decoder)

MambaSpec (KVCacheSpec)            # Mamba 状态空间模型
UniformTypeKVCacheSpecs (KVCacheSpec)  # 多层同类型 spec 的合并
```

### 8.3 各 Spec 的内存计算

**FullAttentionSpec.max_memory_usage_bytes()**：
```python
max_model_len = vllm_config.model_config.max_model_len
# DCP/PCP: 每个 rank 只存 max_model_len / (dcp*pcp) 个 token
if dcp_world_size * pcp_world_size > 1:
    max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
return cdiv(max_model_len, block_size) * page_size_bytes
```

**SlidingWindowSpec.max_memory_usage_bytes()**：
```python
# 只需存最后 sliding_window-1 个 computed token + 新调度的 token
num_tokens = min(sliding_window - 1 + max_num_batched_tokens, max_model_len)
# +1 因为 sliding window 可能不从 block 起始
return (cdiv(num_tokens, block_size) + 1) * page_size_bytes
```

**ChunkedLocalAttentionSpec.max_memory_usage_bytes()**：
```python
# 只需存 attention_chunk_size 个 computed token + 新调度的 token
num_tokens = min(attention_chunk_size + max_num_batched_tokens, max_model_len)
return cdiv(num_tokens, block_size) * page_size_bytes
```

**MLAAttentionSpec.real_page_size_bytes()**：
```python
if cache_dtype_str == "fp8_ds_mla":
    return block_size * 656  # 特殊量化格式
return block_size * num_kv_heads * head_size * dtype_size
```

### 8.4 UniformTypeKVCacheSpecs

当多层具有相同类型的 KV cache spec 时，合并为一个 `UniformTypeKVCacheSpecs`，便于统一管理：

```python
@classmethod
def is_uniform_type(cls, kv_cache_specs) -> bool:
    """判断所有层是否同类型。
    sliding_window 不同大小的层不算同类型。"""
    block_sizes = set(spec.block_size for spec in specs.values())
    if len(block_sizes) > 1:
        return False
    one_spec = next(iter(specs.values()))
    if isinstance(one_spec, FullAttentionSpec):
        return all(isinstance(s, FullAttentionSpec) for s in specs.values())
    elif isinstance(one_spec, SlidingWindowSpec):
        return all(isinstance(s, SlidingWindowSpec) 
                   and s.sliding_window == one_spec.sliding_window for s in specs.values())
    # ... 其他类型
```

### 8.5 KVCacheGroupSpec 和 KVCacheConfig

```python
@dataclass
class KVCacheGroupSpec:
    """共享同一 block table 的一组层。"""
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec

@dataclass
class KVCacheConfig:
    num_blocks: int                       # KV cache block 总数
    kv_cache_tensors: list[KVCacheTensor] # 每层的 KV cache tensor 描述
    kv_cache_groups: list[KVCacheGroupSpec]  # KV cache group 列表
```

单类型 attention 模型只有一个 group 包含所有层；混合 attention 模型（如 hybrid Mamba+Attention）有多个 group。

---

## 9. 标准 Attention Backend 详解

### 9.1 FlashAttentionBackend (`flash_attn.py`)

最常用的 NVIDIA backend，基于 Dao-AILab 的 flash-attention 库。

**关键属性**：
```python
class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes = [float16, bfloat16]
    forward_includes_kv_cache_update = False  # KV cache 更新在 forward 外
    # 支持 16 的倍数 (hybrid + float32 mamba 时固定 16/32/64)
    
    @classmethod
    def supports_attn_type(cls, attn_type) -> bool:
        return attn_type in (DECODER, ENCODER, ENCODER_ONLY, ENCODER_DECODER)  # 全支持
```

**KV cache 形状** (ROCm 分支不同)：
```python
# CUDA:
get_kv_cache_shape(...) -> (2, num_blocks, block_size, num_kv_heads, head_size)
# ROCm:
get_kv_cache_shape(...) -> (
    (num_blocks, num_kv_heads, block_size, head_size),       # K
    (num_blocks, num_kv_heads, head_size, block_size),       # V (转置)
)
```

**stride_order** 支持 NHD/HND 布局切换（通过 `get_kv_cache_layout()`）。

**FlashAttentionMetadataBuilder**：
```python
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS if flash_attn_version == 3 or is_rocm()
        else AttentionCGSupport.UNIFORM_BATCH  # FA2 限制
    )
    supports_update_block_table = True  # 支持复用 metadata
    
    # FA3 用 AOT (Ahead-Of-Time) 调度，需要 max_num_splits 上界
    # FA2 用 max_query_len=1 捕获，不支持 mixed prefill-decode
```

**FlashAttentionImpl**：
```python
class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode = True  # 支持 DCP
    supports_quant_query_input = True  # FP8 KV cache
    supports_per_head_quant_scales = (flash_attn_version >= 3)
    
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, ...):
        # 1. reshape_and_cache_flash: 写入 KV cache
        # 2. flash_attn_varlen_func: varlen attention (prefill)
        #    或 flash_attn_with_kvcache: paged decode
        # 3. 可选: cascade attention 路径
        # 4. 可选: DCP 的 lse 聚合
```

### 9.2 FlashInferBackend (`flashinfer.py`)

基于 FlashInfer 库，在 Blackwell 上优先级最高（TRT-LLM kernel）。

**关键属性**：
```python
class FlashInferBackend(AttentionBackend):
    accept_output_buffer = True
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"]  # 支持 FP8
    # Blackwell 上只支持 page size 16/32/64
    
    @staticmethod
    def get_kv_cache_shape(...) -> (num_blocks, 2, block_size, num_kv_heads, head_size)
```

**FlashInferMetadataBuilder**：
```python
class FlashInferMetadataBuilder(...):
    reorder_batch_threshold = 1  # 需要重排 batch
    
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        if can_use_trtllm_attention(num_qo_heads, num_kv_heads):
            return AttentionCGSupport.UNIFORM_BATCH  # TRT-LLM 支持 spec-decode
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    
    # 内部维护:
    # - _prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper (或 DCP 版本)
    # - _decode_wrapper: BatchDecodeWithPagedKVCacheWrapper (按 batch_size 缓存多个 for cudagraph)
```

### 9.3 TritonAttentionBackend (`triton_attn.py`)

纯 Triton 实现的 backend，作为兼容性回退，支持所有架构。

**关键属性**：
```python
class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes = [float16, bfloat16, float32]  # 额外支持 FP32
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"]
    
    @classmethod
    def supports_head_size(cls, head_size) -> bool:
        return head_size >= 32  # 至少 32
    
    @classmethod
    def supports_mm_prefix(cls) -> bool: return True  # 支持多模态 prefix
    @classmethod
    def supports_sink(cls) -> bool: return True       # 支持 sink
    @classmethod
    def supports_alibi_sqrt(cls) -> bool: return True # 支持 ALiBi
    @classmethod
    def supports_compute_capability(cls, capability) -> bool: return True  # 全架构
    @classmethod
    def supports_attn_type(cls, attn_type) -> bool:
        return attn_type in (DECODER, ENCODER, ENCODER_ONLY, ENCODER_DECODER)  # 全支持
    
    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool: return False  # 不支持 cascade
    
    get_kv_cache_shape(...) -> (num_blocks, 2, block_size, num_kv_heads, head_size)
```

**TritonAttentionMetadataBuilder**：
```python
_cudagraph_support = AttentionCGSupport.ALWAYS  # 始终支持 cudagraph (含 mixed prefill-decode)
```

**TritonAttentionImpl**：
```python
class TritonAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key):
        return quant_key == kFp8StaticTensorSym  # 支持融合 FP8 输出量化
    # 使用 triton_prefill_attention / triton_decode_attention
```

### 9.4 FlexAttentionBackend (`flex_attention.py`)

基于 PyTorch 2.5+ 的 `flex_attention`，灵活但性能一般，作为最后回退。

```python
class FlexAttentionBackend(AttentionBackend):
    get_kv_cache_shape(...) -> (2, num_blocks, block_size, num_kv_heads, head_size)
    # 支持任意 head_size (作为 ROCm 非标准 head_size 的回退)
```

### 9.5 TreeAttentionBackend (`tree_attn.py`)

为推测解码的 tree-based attention 设计。

```python
class TreeAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    get_supported_head_sizes() -> [32, 64, 96, 128, 160, 192, 224, 256]
    get_kv_cache_shape(...) -> (2, num_blocks, block_size, num_kv_heads, head_size)
    use_cascade_attention(*args, **kwargs) -> False
    # 包含 tree_attn_bias: 树形注意力掩码
```

### 9.6 FlashAttentionDiffKVBackend (`flash_attn_diffkv.py`)

支持 K 和 V 有不同 head_size 的 FlashAttention 变体。

### 9.7 CPUAttentionBackend (`cpu_attn.py`)

CPU 平台 backend：

```python
class CPUAttentionBackend(AttentionBackend):
    supported_dtypes = [float16, bfloat16, float32]
    get_supported_head_sizes() -> [32, 64, 80, 96, 112, 128, 160, 192, 224, 256]
    supports_attn_type(...) -> (DECODER, ENCODER, ENCODER_ONLY, ENCODER_DECODER)
    get_kv_cache_shape(...) -> (2, num_blocks, num_kv_heads, block_size, head_size)  # 注意维度顺序
    use_cascade_attention(...) -> False
    
    # Builder 会根据 CPU 架构决定:
    # - _CPU_ARCH_PREFER_MIXED_BATCH: 不重排, 统一 kernel
    # - 其他: reorder decode 到前面, prefill 用 SDPA, decode 用 cpu_attention_with_kv_cache
```

### 9.8 TurboQuantAttentionBackend (`turboquant_attn.py`)

2-bit KV cache 压缩 backend（4x 容量提升）：

```python
class TurboQuantAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    forward_includes_kv_cache_update = False  # KV cache 更新在 forward 外
    supported_kv_cache_dtypes = [
        "turboquant_k8v4", "turboquant_4bit_nc",
        "turboquant_k3v4_nc", "turboquant_3bit_nc",
    ]
    get_supported_kernel_block_sizes() -> [16, 32, 64, 128]
    supports_attn_type(...) -> DECODER only
    
    # KV cache 形状特殊: K+V 打包到单个 interleaved slot
    # (num_blocks, block_size, num_kv_heads, slot_size_aligned)
    # 每个 slot = [key_packed | value_packed | padding]
```

**路由**：CUDA 和 ROCm 平台检测到 `kv_cache_dtype.startswith("turboquant_")` 时直接路由到此 backend。

### 9.9 NoAttentionBackend

Attention-free 模型（如纯 Mamba）的占位 backend。

---

## 10. MLA Backend 体系

MLA (Multi-head Latent Attention) 是 DeepSeek-V2/V3 提出的注意力机制，通过压缩 KV 降低显存。vLLM 提供丰富的 MLA backend 实现。

### 10.1 MLACommonBackend 基类 (`mla_attention.py:333`)

所有 MLA backend 继承自 `MLACommonBackend`：

```python
class MLACommonBackend(AttentionBackend):
    accept_output_buffer = True
    
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...):
        # MLA 的 num_kv_heads 假设为 1
        return (num_blocks, block_size, head_size)  # 注意: 无 2 维 (K/V 合并)
    
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]  # MLA 特有的 576 维 (512 nope + 64 rope)
    
    @classmethod
    def is_mla(cls) -> bool:
        return True
```

### 10.2 MLA Backend 列表

| Backend | 架构要求 | block_size | FP8 | 说明 |
|---------|---------|-----------|-----|------|
| **TRITON_MLA** | 全架构 | MultipleOf(16) | ❌ | 纯 Triton，兼容性最好 |
| **CUTLASS_MLA** | SM100 (Blackwell) | 128 | ✅ | CUTLASS kernel，Blackwell 优先 |
| **FLASHINFER_MLA** | Blackwell (qk_nope=128) | 64 | ✅ | TRT-LLM MLA，Blackwell 默认 |
| **FLASHMLA** | 非 Blackwell (支持检测) | 64 | ✅ | FlashMLA kernel，Hopper 优化 |
| **FLASH_ATTN_MLA** | 非指定 | - | ✅ | FlashAttention MLA 变体 |
| **FLASHMLA_SPARSE** | - | 64 | - | Sparse MLA (DeepSeek-V3.2) |
| **ROCM_AITER_MLA** | ROCm | - | - | ROCm AITER MLA |
| **ROCM_AITER_TRITON_MLA** | ROCm | - | - | ROCm Triton MLA |
| **ROCM_AITER_MLA_SPARSE** | ROCm | - | - | ROCm Sparse MLA |

### 10.3 TritonMLABackend (`mla/triton_mla.py`)

```python
class TritonMLABackend(MLACommonBackend):
    supported_dtypes = [float16, bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]  # 不支持 FP8
    
    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True  # 全架构支持

class TritonMLAImpl(MLACommonImpl):
    can_return_lse_for_decode = True
    # 不支持: alibi_slopes, sliding_window, logits_soft_cap
    # 不支持: encoder attention, cross-attention
    # 不支持: FP8 KV cache
    # 使用 decode_attention_fwd (Triton kernel)
```

### 10.4 CutlassMLABackend (`mla/cutlass_mla.py`)

```python
class CutlassMLABackend(MLACommonBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]
    get_supported_kernel_block_sizes() -> [128]  # 固定 128
    
    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major == 10  # 仅 Blackwell

class CutlassMLAImpl(MLACommonImpl):
    can_return_lse_for_decode = True
    # 使用 SM100 cutlass MLA kernel, 需要专门的 workspace
    # block_size 强制 128
```

### 10.5 FlashInferMLABackend (`mla/flashinfer_mla.py`)

Blackwell 上 MLA 的默认选择（当 `qk_nope_head_dim == 128`）：
```python
class FlashInferMLABackend(MLACommonBackend):
    # 使用 FlashInfer 的 MLA kernel
    # block_size 强制 64
```

### 10.6 FlashMLABackend (`mla/flashmla.py`)

非 Blackwell 上 MLA 的优先选择：
```python
class FlashMLABackend(MLACommonBackend):
    # block_size 强制 64
    # 使用 FlashMLA dense kernel
    # 通过 is_flashmla_dense_supported() 检测可用性
```

### 10.7 FlashMLASparseBackend (`mla/flashmla_sparse.py`)

DeepSeek-V3.2 的 sparse MLA：
```python
class FlashMLASparseBackend(MLACommonBackend):
    @classmethod
    def is_sparse(cls) -> bool:
        return True
    # block_size 强制 64
    # FP8 ds_mla: real_page_size_bytes = block_size * 656
```

### 10.8 MLA 的特殊 forward 签名

MLA backend 的 impl 使用 `MLAAttentionImpl.forward()`，签名与标准 attention 不同：

```python
def forward(self, layer, hidden_states_or_cq, kv_c_normed, k_pe,
            kv_cache, attn_metadata, output=None, ...):
    """hidden_states_or_cq: 压缩 query 或 hidden states
    kv_c_normed: 压缩 KV (经 kv_c_norm 归一化)
    k_pe: decoupled RoPE key
    """
```

---

## 11. ROCm Backend 体系

AMD GPU 有一套独立的 backend 体系。

### 11.1 ROCm 标准 Backend

| Backend | 说明 |
|---------|------|
| **ROCM_ATTN** | ROCm 原生 C++ paged attention (prefill-decode 分离 kernel) |
| **ROCM_AITER_FA** | AITER Flash Attention (gfx9 only) |
| **ROCM_AITER_UNIFIED_ATTN** | AITER 统一 attention |
| **TRITON_ATTN** | Triton (ROCm 默认回退) |
| **FLASH_ATTN** | FlashAttention PA (block_size=64 + VLLM_USE_FLASH_ATTN_PA) |

### 11.2 RocmAttentionBackend (`rocm_attn.py`)

```python
class RocmAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_dtypes = [float16, bfloat16, float32]
    get_supported_kernel_block_sizes() -> [16, 32, 544]  # 544 为 qwen3-next 等非标准
    get_supported_head_sizes() -> [32, 64, 96, 128, 160, 192, 224, 256]
    get_kv_cache_shape(...) -> (2, num_blocks, block_size, num_kv_heads, head_size)
    
    # Builder:
    _cudagraph_support = AttentionCGSupport.ALWAYS  # 始终支持 cudagraph
```

### 11.3 ROCm MLA Backend

ROCm 默认使用 `TRITON_MLA`（当 `selected_backend is None`），可选 `FLASHMLA`（block_size=64）或 `FLASHMLA_SPARSE`（sparse 场景）。

### 11.4 ROCm block_size 强制

- 标准 attention: 默认 Triton，`VLLM_USE_FLASH_ATTN_PA=1` + block_size=64 可用 FlashAttention
- MLA: FlashMLA 要求 64，sparse MLA 要求 64

---

## 12. Mamba/线性注意力 Backend

### 12.1 MambaAttentionBackendEnum

线性注意力/状态空间模型通过 `MambaAttentionBackendEnum` 注册，与标准 attention 分离：

```python
class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    MAMBA1 = Mamba1AttentionBackend       # Mamba-1
    MAMBA2 = Mamba2AttentionBackend       # Mamba-2
    SHORT_CONV = ShortConvAttentionBackend # 短卷积
    LINEAR = LinearAttentionBackend        # 线性注意力
    GDN_ATTN = GDNAttentionBackend         # GDN (Gated Delta Network)
    CUSTOM = None
```

### 12.2 选择入口

```python
def get_mamba_attn_backend(mamba_type: str) -> type[AttentionBackend]:
    """根据 mamba_type 字符串选择 backend。"""
    backend_name = MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]  # "mamba2" -> "MAMBA2"
    selected_backend = MambaAttentionBackendEnum[backend_name]
    return selected_backend.get_class()
```

### 12.3 MambaSpec

Mamba 的 KV cache 格式与 attention 不同，使用 `MambaSpec`：

```python
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]    # 多个状态张量的形状
    dtypes: tuple[torch.dtype]              # 对应 dtype
    mamba_type: str = "mamba2"
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0
    
    def max_memory_usage_bytes(self, vllm_config) -> int:
        # "all": 存全部 (cdiv(max_model_len, block_size) * page_size)
        # "align": 存对齐状态 (page_size * (2 + num_speculative_blocks))
        # "none": page_size * (1 + num_speculative_blocks)
```

### 12.4 GDNAttentionBackend (`gdn_attn.py`)

Gated Delta Network 的 attention backend，支持 spec decode：
- `_cudagraph_support`: UNIFORM_BATCH
- 支持 `num_accepted_tokens` 和 `num_decode_draft_tokens` 元数据

---

## 13. 特殊 Backend

### 13.1 IPEX (Intel XPU)

Intel Extension for PyTorch 的 attention backend，用于 XPU 平台。

### 13.2 TORCH_SDPA

PyTorch 原生 Scaled Dot Product Attention，仅用于 ViT（视觉模型）：
```python
TORCH_SDPA = ""  # 空字符串，仅作 tag
```

### 13.3 NO_ATTENTION

Attention-free 模型（纯 Mamba 等）的占位 backend，不执行任何 attention 计算。

### 13.4 子类化工具函数

`backend.py:757` 提供了 backend 子类化的工具：

```python
def subclass_attention_backend(name_prefix, attention_backend_cls, builder_cls):
    """创建一个新子类，其 get_builder_cls() 返回 builder_cls。
    用于在不修改原 backend 的情况下替换 builder。"""
    return type(name_prefix + attention_backend_cls.__name__,
                (attention_backend_cls,),
                {"get_builder_cls": lambda: builder_cls})

def subclass_attention_backend_with_overrides(name_prefix, attention_backend_cls, overrides):
    """创建带任意 override 的子类。"""
    return type(name_prefix + attention_backend_cls.__name__,
                (attention_backend_cls,), overrides)
```

---

## 14. CUDA Graph 支持等级

### 14.1 AttentionCGSupport 枚举 (`backend.py:416`)

```python
class AttentionCGSupport(Enum):
    ALWAYS = 3                 # 始终支持 cudagraph (含 mixed prefill-decode)
    UNIFORM_BATCH = 2          # 支持 query 长度相同的 batch (spec-decode 用)
    UNIFORM_SINGLE_TOKEN_DECODE = 1  # 仅支持 query_len==1 的 decode
    NEVER = 0                  # 不支持 cudagraph
```

### 14.2 各 Backend 的 cudagraph 支持

| Backend | cudagraph_support | 说明 |
|---------|------------------|------|
| FlashAttention (FA3) | ALWAYS | FA3 支持所有场景 |
| FlashAttention (FA2) | UNIFORM_BATCH | FA2 的 max_query_len=1 限制 |
| FlashAttention (ROCm) | ALWAYS | ROCm FA 始终支持 |
| FlashInfer (TRT-LLM) | UNIFORM_BATCH | 支持 spec-decode |
| FlashInfer (普通) | UNIFORM_SINGLE_TOKEN_DECODE | 仅纯 decode |
| TritonAttn | ALWAYS | 始终支持 |
| ROCmAttn | ALWAYS | 始终支持 |
| TurboQuant | UNIFORM_BATCH | 支持 spec-decode |
| FlexAttention | NEVER | 不支持 |
| CPUAttn | NEVER | 不支持 |

### 14.3 与 CudagraphDispatcher 的交互

ModelRunner 的 `CudagraphDispatcher` 会查询 backend 的 `get_cudagraph_support()`，结合 `uniform_decode` 判断决定 CUDA Graph 模式：
- `ALWAYS`: 可用 FULL cudagraph（含 mixed prefill-decode）
- `UNIFORM_BATCH`: 仅 uniform decode batch 可用 FULL，否则降级 PIECEWISE
- `UNIFORM_SINGLE_TOKEN_DECODE`: 仅纯 decode (query_len==1) 可用 FULL
- `NEVER`: 强制 eager 或 PIECEWISE

---

## 15. Cascade Attention 级联注意力

### 15.1 概念

Cascade attention 是 vLLM 的优化：当 batch 内多个 request 共享较长公共前缀（prefix caching 命中）时，先对公共前缀做一次 attention，再对各 request 的差异部分单独 attention，避免重复计算。

### 15.2 启用判断 (`flash_attn.py:1158`)

```python
def use_cascade_attention(common_prefix_len, query_lens, num_query_heads,
    num_kv_heads, use_alibi, use_sliding_window, use_local_attention,
    num_sms, dcp_world_size) -> bool:
    
    # 1. 公共前缀太短 (< 256 token) 不值得
    if common_prefix_len < 256:
        return False
    
    # 2. 这些变体不支持 cascade
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    
    # 3. 请求数太少 (< 8) 不值得
    if len(query_lens) < 8:
        return False
    
    # 4. DCP 不支持
    if dcp_world_size > 1:
        return False
    
    # 5. 启发式: 非 FlashDecoding 时 cascade 更快 (省带宽)
    num_queries_per_kv = num_query_heads // num_kv_heads
    use_flash_decoding = (num_queries_per_kv > 1 
                          and not use_sliding_window 
                          and not use_alibi 
                          and np.all(query_lens == 1))
    if not use_flash_decoding:
        return True
    
    # 6. FlashDecoding 时用性能模型比较 cascade vs FlashDecoding
    # ... (rough performance model)
```

### 15.3 各 Backend 的 cascade 支持

| Backend | cascade 支持 |
|---------|-------------|
| FlashAttention | ✅ |
| FlashInfer | ✅ |
| TritonAttn | ❌ (`use_cascade_attention` 返回 False) |
| TreeAttn | ❌ |
| CPUAttn | ❌ |

### 15.4 在 ModelRunner 中的使用

ModelRunner 的 `_build_attention_metadata()` 调用 `_compute_cascade_attn_prefix_lens()` 预计算公共前缀长度，传给 builder 的 `build(common_prefix_len=...)`。

注意：`_determine_batch_execution_and_padding()` 中 `use_cascade_attn` 为 True 时会 `disable_full=True`，即 cascade attention 不使用 FULL cudagraph。

---

## 16. Backend 选择决策流程图

### 16.1 完整选择流程

```
get_attn_backend(head_size, dtype, kv_cache_dtype, block_size, use_mla, ...)
│
├─ 校验 kv_cache_dtype
│
├─ 从 vllm_config.attention_config.backend 获取用户指定 backend
│
├─ 构建 AttentionSelectorConfig
│
└─ _cached_get_attn_backend(backend, config)  [@cache]
   │
   ├─ current_platform.get_attn_backend_cls(backend, config)
   │  │
   │  ├─ [CUDA 平台]
   │  │  ├─ kv_cache_dtype 以 "turboquant_" 开头? → TURBOQUANT
   │  │  ├─ 用户指定了 backend?
   │  │  │  ├─ validate_configuration() 校验
   │  │  │  │  ├─ 通过 → 返回指定 backend
   │  │  │  │  └─ 失败 → raise ValueError
   │  │  │  └─ 未指定 → 自动选择
   │  │  │     ├─ get_valid_backends() 获取所有有效 backend
   │  │  │     │  └─ _get_backend_priorities(use_mla, device_capability)
   │  │  │     │     ├─ MLA + Blackwell: [FLASHINFER_MLA, CUTLASS_MLA, ...]
   │  │  │     │     ├─ MLA + 其他:     [FLASH_ATTN_MLA, FLASHMLA, ...]
   │  │  │     │     ├─ 标准 + Blackwell: [FLASHINFER, FLASH_ATTN, ...]
   │  │  │     │     └─ 标准 + 其他:     [FLASH_ATTN, FLASHINFER, ...]
   │  │  │     └─ 返回优先级最高的有效 backend
   │  │  └─ MLA 场景额外: 强制调整 block_size (64/128)
   │  │
   │  ├─ [ROCm 平台]
   │  │  ├─ turboquant_ → TURBOQUANT
   │  │  ├─ use_sparse → FLASHMLA_SPARSE
   │  │  ├─ use_mla:
   │  │  │  ├─ VLLM_USE_FLASH_MLA + block_size=64 → FLASHMLA
   │  │  │  └─ 默认 → TRITON_MLA
   │  │  └─ 标准:
   │  │     ├─ VLLM_USE_FLASH_ATTN_PA + block_size=64 → FLASH_ATTN
   │  │     └─ 默认 → TRITON_ATTN
   │  │
   │  ├─ [CPU 平台] → CPU_ATTN
   │  └─ [XPU 平台] → IPEX
   │
   ├─ resolve_obj_by_qualname() 动态导入
   │
   ├─ backend.get_required_kv_cache_layout() → 设置 KV cache 布局 (NHD/HND)
   │
   └─ 返回 backend 类
```

### 16.2 Backend 与其他组件的关系

```
                    ┌─────────────────────┐
                    │  AttentionConfig    │
                    │  (用户配置 backend)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  get_attn_backend() │
                    │  (selector.py)      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Platform           │
                    │  .get_attn_backend_ │
                    │  cls()              │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      AttentionBackend 类        │
              │  (FlashAttn/Triton/FlashInfer) │
              └────────────────────────────────┘
                    │                │
          get_builder_cls    get_impl_cls
                    │                │
                    ▼                ▼
    ┌───────────────────────┐  ┌──────────────────────┐
    │ AttentionMetadataBuilder│  │   AttentionImpl     │
    │  .build()              │  │   .forward()        │
    │  → AttentionMetadata   │  │   → attention 输出   │
    └───────────────────────┘  └──────────────────────┘
              ▲                          ▲
              │                          │
    ┌─────────┴──────────┐    ┌─────────┴──────────┐
    │  ModelRunner       │    │  Attention Layer    │
    │ _build_attn_metadata│   │  (vllm.attention.   │
    │ 传入 CommonAttnMeta│    │   layer.Attention)  │
    └────────────────────┘    └─────────────────────┘
```

---

## 17. 配置参数与最佳实践

### 17.1 AttentionConfig 参数 (`config/attention.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `AttentionBackendEnum \| None` | None | 指定 backend，None 则自动选择 |
| `flash_attn_version` | `Literal[2,3] \| None` | None | 强制 FA 版本 |
| `use_prefill_decode_attention` | bool | False | 用分离的 prefill/decode kernel (ROCm) |
| `flash_attn_max_num_splits_for_cuda_graph` | int | 32 | FA cudagraph decode 的 max splits |
| `tq_max_kv_splits_for_cuda_graph` | int | 32 | TurboQuant cudagraph 的 max splits |
| `use_cudnn_prefill` | bool | False | 是否用 cuDNN prefill |
| `use_trtllm_ragged_deepseek_prefill` | bool | True | TRTLLM ragged DeepSeek prefill |
| `use_trtllm_attention` | `bool \| None` | None | 是否用 TRTLLM attention (FlashInfer) |
| `disable_flashinfer_prefill` | bool | False | 禁用 FlashInfer prefill |
| `disable_flashinfer_q_quantization` | bool | False | FP8 KV 时不禁用 Q 量化 |

### 17.2 关键环境变量

| 变量 | 说明 |
|------|------|
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | FlashInfer workspace buffer 大小 |
| `VLLM_USE_FLASH_MLA` | ROCm 上强制使用 FlashMLA |
| `VLLM_USE_FLASH_ATTN_PA` | ROCm 上启用 FlashAttention PA (block_size=64) |
| `VLLM_ROCM_USE_AITER` | ROCm 启用 AITER |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | ROCm AITER 统一 attention |
| `VLLM_ROCM_USE_AITER_MHA` | ROCm AITER MHA |

### 17.3 Backend 选择最佳实践

**NVIDIA GPU**：
- Hopper (SM90) / Ampere (SM80): 默认 FLASH_ATTN 即可，性能最优
- Blackwell (SM100): 默认会选 FLASHINFER（TRT-LLM 优化）
- MLA 模型 (DeepSeek): Blackwell → FLASHINFER_MLA/CUTLASS_MLA，其他 → FLASH_ATTN_MLA/FLASHMLA
- 不确定时：留空让 vLLM 自动选择

**AMD GPU (ROCm)**：
- 默认 TRITON_ATTN (兼容性最好)
- gfx9 + 需要性能：尝试 ROCM_AITER_FA
- MLA: TRITON_MLA (默认) 或 FLASHMLA (block_size=64)

**CPU**：CPU_ATTN

**特殊场景**：
- 2-bit KV cache 压缩：设置 `kv_cache_dtype=turboquant_*`，自动路由到 TURBOQUANT
- 推测解码 tree attention：TREE_ATTN
- 非标准 head_size (ROCm)：FLEX_ATTENTION 作为回退

### 17.4 性能调优建议

1. **不要强行指定 backend**：除非有明确需求，让 vLLM 自动选择，它会根据 GPU 架构、模型配置选最优
2. **Blackwell 上优先 FlashInfer**：SM100 上 TRT-LLM kernel 性能领先
3. **MLA block_size 注意**：FlashMLA=64, CUTLASS_MLA=128, FlashInfer_MLA=64，自动选择时会强制调整
4. **Cascade attention 自动启用**：公共前缀 ≥256 token + ≥8 请求时自动开启，无需配置
5. **CUDA Graph 兼容性**：FA3/Triton/ROCmAttn 支持 ALWAYS，FA2/FlashInfer 受限，Dispatcher 会自动降级
6. **FP8 KV cache**：FlashInfer/Triton/CUTLASS_MLA/FlashAttn_MLA 支持，TritonMLA 不支持
7. **TurboQuant**：4x KV cache 容量，适合长 context 显存受限场景，但只支持 DECODER attention
8. **DCP (Decode Context Parallelism)**：需要 backend 的 `can_return_lse_for_decode=True`，FA/Triton MLA 支持
9. **hybrid_blocks**：block_size 只需是 kernel 要求的倍数（`MultipleOf`），不必完全匹配
10. **ViT 模型**：走独立的 `get_vit_attn_backend()`，Ampere+ 用 FLASH_ATTN，其他用 TORCH_SDPA

### 17.5 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| "Selected backend X is not valid" | backend 不支持当前配置 | 检查 validate_configuration 返回的 invalid_reasons |
| "No valid attention backend found" | 无 backend 支持当前组合 | 检查 head_size/dtype/block_size/kv_cache_dtype 组合 |
| CUDA Graph 频繁降级 | backend 的 cudagraph_support 低 | 换 FA3/Triton 等 ALWAYS 级 backend |
| MLA 性能差 | 选了 TRITON_MLA 而非专用 kernel | Blackwell 用 CUTLASS_MLA/FLASHINFER_MLA，Hopper 用 FLASHMLA |
| FP8 KV cache 报错 | backend 不支持 FP8 | 用 FlashInfer/CUTLASS_MLA，或换非 FP8 |
| block_size 被强制改变 | MLA backend 的要求 | 正常现象，FlashMLA=64, CUTLASS_MLA=128 |

---

> **相关文档**:
> - [VLLM_PRODUCTION_GUIDE.md](VLLM_PRODUCTION_GUIDE.md) — vLLM 生产级指导手册
> - [VLLM_SCHEDULER_GUIDE.md](VLLM_SCHEDULER_GUIDE.md) — 调度器详解
> - [VLLM_MODELRUNNER_GUIDE.md](VLLM_MODELRUNNER_GUIDE.md) — ModelRunner 详解（含 `_build_attention_metadata()` 流程）
