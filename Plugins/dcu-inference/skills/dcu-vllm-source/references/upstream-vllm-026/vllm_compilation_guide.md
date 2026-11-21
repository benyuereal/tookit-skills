# vLLM 编译优化（Compilation）模块详解

> 基于 vLLM 0.15.x 版本（V1 架构），全面覆盖 torch.compile 集成、CUDA Graph 捕获/重放、piecewise 分段编译、Inductor 自定义 pass 与算子融合。

---

## 目录

1. [概述](#1-概述)
2. [模块架构](#2-模块架构)
3. [CompilationConfig 配置体系](#3-compilationconfig-配置体系)
4. [编译后端与编译流程](#4-编译后端与编译流程)
5. [VllmBackend 与图分割](#5-vllmbackend-与图分割)
6. [Piecewise 分段编译](#6-piecewise-分段编译)
7. [CUDA Graph 捕获与重放](#7-cuda-graph-捕获与重放)
8. [Inductor 自定义 Pass 体系](#8-inductor-自定义-pass-体系)
9. [算子融合 Pass 详解](#9-算子融合-pass-详解)
10. [编译缓存机制](#10-编译缓存机制)
11. [support_torch_compile 装饰器](#11-support_torch_compile-装饰器)
12. [编译统计与监控](#12-编译统计与监控)
13. [实际案例：GLM-5.1 编译流程](#13-实际案例glm-51-编译流程)
14. [配置参数与环境变量](#14-配置参数与环境变量)
15. [调优建议与常见问题](#15-调优建议与常见问题)

---

## 1. 概述

编译优化是 vLLM 性能优化的第三大支柱（前两个是 PagedAttention 和连续批处理）。`vllm/compilation/` 模块负责：

1. **torch.compile 集成**：通过自定义 Inductor backend 编译模型计算图
2. **CUDA Graph 捕获/重放**：消除 kernel launch 开销（decode 阶段提升 20-30%）
3. **Piecewise 分段编译**：将图按 attention 等"splitting ops"切分，分别编译
4. **自定义 Inductor Pass**：在 Inductor 编译流程中插入算子融合优化
5. **编译缓存**：缓存编译产物，二次启动跳过编译

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/compilation/backends.py` | 1015 | VllmBackend、图分割、编译入口 |
| `vllm/compilation/compiler_interface.py` | 673 | CompilerInterface 抽象 + Inductor/Eager 适配器 |
| `vllm/compilation/cuda_graph.py` | 309 | CUDAGraphWrapper（捕获/重放） |
| `vllm/compilation/piecewise_backend.py` | 285 | PiecewiseBackend（分段编译调度） |
| `vllm/compilation/pass_manager.py` | 164 | PostGradPassManager（pass 编排） |
| `vllm/compilation/fusion.py` | 570 | RMSNorm + 量化融合 |
| `vllm/compilation/fusion_attn.py` | 364 | Attention + 量化融合 |
| `vllm/compilation/collective_fusion.py` | 1228 | AllReduce/AsyncTP 融合 |
| `vllm/compilation/activation_quant_fusion.py` | 215 | SiLU + 量化融合 |
| `vllm/compilation/qk_norm_rope_fusion.py` | 244 | QK Norm + RoPE 融合 |
| `vllm/compilation/decorators.py` | 688 | support_torch_compile 装饰器 |
| `vllm/compilation/caching.py` | 516 | AOT 编译产物缓存 |
| `vllm/compilation/inductor_pass.py` | 180 | InductorPass 基类 |
| `vllm/compilation/fix_functionalization.py` | 266 | functionalization 修复 |
| `vllm/compilation/noop_elimination.py` | 130 | no-op 消除 |
| `vllm/compilation/wrapper.py` | 329 | TorchCompileWithNoGuardsWrapper |
| `vllm/config/compilation.py` | — | CompilationConfig / CUDAGraphMode / PassConfig |

---

## 2. 模块架构

### 2.1 编译流程总览

```
torch.compile(model, backend=VllmBackend)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                    VllmBackend                            │
│                                                          │
│  1. Dynamo trace → fx.GraphModule                        │
│  2. split_graph(graph, splitting_ops)                    │
│     → 按 attention/MoE 等 op 切分为多个子图               │
│  3. PiecewiseCompileInterpreter 遍历子图                  │
│     → 对每个子图创建 PiecewiseBackend                     │
│     → wrap_with_cudagraph_if_needed (PIECEWISE)          │
│                                                          │
│  4. PostGradPassManager 注册到 Inductor config            │
│     → 自定义 pass 在 Inductor 编译时执行                  │
└──────────────────────────────────────────────────────────┘
       │
       ▼ (每个子图)
┌──────────────────────────────────────────────────────────┐
│                  PiecewiseBackend                         │
│                                                          │
│  对每个 compile_range (形状范围):                         │
│    CompilerManager.compile(graph, example_inputs, range)  │
│      → CompilerInterface (InductorAdaptor)                │
│      → Inductor 编译 + 自定义 pass                        │
│      → 缓存编译产物                                       │
│                                                          │
│  运行时: __call__(args)                                   │
│    → 按 runtime_shape 查找 range_entry                    │
│    → 首次: 编译; 后续: 直接调用                           │
└──────────────────────────────────────────────────────────┘
       │
       ▼ (CUDA Graph 模式)
┌──────────────────────────────────────────────────────────┐
│                  CUDAGraphWrapper                         │
│                                                          │
│  首次调用 (batch_descriptor):                             │
│    → torch.cuda.graph(cudagraph, pool) 捕获               │
│    → 缓存 CUDAGraphEntry                                 │
│  后续调用:                                                │
│    → cudagraph.replay()                                  │
└──────────────────────────────────────────────────────────┘
```

### 2.2 两层 CUDA Graph

vLLM 的 CUDA Graph 分两层，由 `CUDAGraphMode` 决定：

| 层级 | 模式 | 捕获对象 | 特点 |
|------|------|---------|------|
| **FULL** | 整个模型前向 | `model(input_ids, positions)` | 一个 graph 覆盖整个 forward，最大消除 launch 开销 |
| **PIECEWISE** | 分段子图 | 每个 splitting op 之间的子图 | 按层切分，每个子图独立捕获，灵活但开销略大 |

---

## 3. CompilationConfig 配置体系

### 3.1 CompilationMode

```python
class CompilationMode(enum.IntEnum):
    NONE = 0                # 纯 eager，无 torch.compile
    STOCK_TORCH_COMPILE = 1 # 标准 torch.compile
    DYNAMO_TRACE_ONCE = 2   # 单次 Dynamo trace，避免重编译
    VLLM_COMPILE = 3        # vLLM 自定义 Inductor backend（V1 默认）
```

### 3.2 CUDAGraphMode

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0                    # 不使用 CUDA Graph
    PIECEWISE = 1               # 分段 CUDA Graph
    FULL = 2                    # 全图 CUDA Graph
    FULL_DECODE_ONLY = (FULL, NONE)        # 仅 decode 全图
    FULL_AND_PIECEWISE = (FULL, PIECEWISE) # 全图 + 分段组合（V1 默认）

    # 元组模式 (FULL, PIECEWISE) 表示:
    #   decode (每请求1 token) → FULL 模式
    #   prefill/mixed (多 token) → PIECEWISE 模式
```

**关键方法**：
- `decode_mode()` → 元组第一个元素（FULL）
- `mixed_mode()` → 元组第二个元素（PIECEWISE）
- `has_full_cudagraphs()` → 是否包含 FULL
- `has_piecewise_cudagraphs()` → 是否包含 PIECEWISE

### 3.3 CompilationConfig 核心字段

```python
@dataclass
class CompilationConfig:
    mode: CompilationMode = None           # 编译模式
    backend: str = ""                      # "inductor" / "eager" / 自定义
    splitting_ops: list[str] | None = None # 图分割算子（如 attention）
    compile_sizes: list[int | str] | None  # 显式编译的形状
    cudagraph_mode: CUDAGraphMode = None   # CUDA Graph 模式
    cudagraph_capture_sizes: list[int] | None  # CUDA Graph 捕获的 batch size
    use_inductor_graph_partition: bool     # 用 Inductor 原生分割 vs Dynamo 分割
    pass_config: PassConfig                # 自定义 pass 配置
    inductor_compile_config: dict          # 传给 Inductor 的额外配置
    compile_cache_save_format: str         # 缓存格式
```

### 3.4 PassConfig（算子融合配置）

```python
@dataclass
class PassConfig:
    fuse_norm_quant: bool       # 融合 RMSNorm + 量化
    fuse_act_quant: bool        # 融合 SiLU + 量化
    fuse_attn_quant: bool       # 融合 Attention + 量化
    eliminate_noops: bool       # 消除 no-op
    enable_sp: bool             # 序列并行
    fuse_gemm_comms: bool       # AsyncTP（GEMM + 通信重叠）
    fuse_allreduce_rms: bool    # FlashInfer AllReduce + RMSNorm 融合
    enable_qk_norm_rope_fusion: bool  # QK Norm + RoPE 融合
```

### 3.5 OptimizationLevel 预设

| Level | mode | cudagraph_mode | 适用 |
|-------|------|---------------|------|
| O0 | NONE | NONE | 开发调试 |
| O1 | VLLM_COMPILE | PIECEWISE | 快速优化 |
| O2 | VLLM_COMPILE | FULL_AND_PIECEWISE | 生产默认 |
| O3 | VLLM_COMPILE | FULL_AND_PIECEWISE | 最大优化 |

---

## 4. 编译后端与编译流程

### 4.1 CompilerInterface 抽象

```python
class CompilerInterface:
    name: str  # "inductor" / "eager" / 自定义

    def initialize_cache(self, cache_dir, disable_cache, prefix): ...
    def compute_hash(self, vllm_config) -> str: ...
    def compile(self, graph, example_inputs, compiler_config, compile_range, key)
        -> (compiled_callable, handle): ...
    def load(self, handle, graph, example_inputs, graph_index, compile_range): ...
```

### 4.2 make_compiler 工厂

```python
def make_compiler(compilation_config) -> CompilerInterface:
    if compilation_config.backend == "inductor":
        if envs.VLLM_USE_STANDALONE_COMPILE and torch_version >= "2.8.0":
            return InductorStandaloneAdaptor(...)  # AOT 独立编译
        else:
            return InductorAdaptor()                # 标准 Inductor
    elif compilation_config.backend == "eager":
        return EagerAdaptor()                       # 不编译
    else:
        return resolve_obj_by_qualname(custom_backend)()  # 自定义后端
```

### 4.3 三种适配器

| 适配器 | 说明 |
|--------|------|
| `InductorAdaptor` | 标准 Inductor 编译（JIT，运行时编译） |
| `InductorStandaloneAdaptor` | AOT 独立编译，产物可序列化（`VLLM_USE_STANDALONE_COMPILE=1`） |
| `EagerAdaptor` | 不编译，直接运行 FX 图 |

### 4.4 CompilerManager（编译管理器）

```python
class CompilerManager:
    cache: dict[tuple[Range, int, str], Any]  # (形状范围, 图索引, 后端名) → handle
    compiler: CompilerInterface

    def compile(self, graph, example_inputs, config, compile_range, graph_index, num_graphs):
        # 1. 尝试从缓存加载
        compiled = self.load(graph, example_inputs, graph_index, compile_range)
        if compiled is not None: return compiled

        # 2. 编译
        with self.compile_context(compile_range):  # 设置 pass_context + partition_rule
            compiled, handle = self.compiler.compile(graph, ...)

        # 3. 缓存
        if is_compile_cache_enabled and handle is not None:
            self.cache[(compile_range, graph_index, compiler.name)] = handle
        return compiled
```

---

## 5. VllmBackend 与图分割

### 5.1 VllmBackend

`VllmBackend` 是 `torch.compile` 的自定义 backend，核心工作：

```python
class VllmBackend:
    def __init__(self, vllm_config, prefix="", is_encoder=False):
        self.prefix = prefix or model_tag  # "backbone" / "eagle_head"
        self.compiler_manager = CompilerManager(compilation_config)
        self.inductor_config = compilation_config.inductor_compile_config.copy()
        # 注册 PostGradPassManager 到 Inductor config
        self.inductor_config["post_grad_custom_post_pass"] = PostGradPassManager()

    def __call__(self, graph, example_inputs):
        # torch.compile 调用此方法
        # 1. split_graph 按 splitting_ops 切分
        # 2. PiecewiseCompileInterpreter 遍历子图编译
        # 3. 返回拼接后的 callable
```

### 5.2 split_graph（图分割）

```python
def split_graph(graph, splitting_ops) -> (split_gm, list[SplitItem]):
    # 遍历所有节点，按 splitting_ops（如 "vllm.unified_attention"）切分
    for node in graph.graph.nodes:
        if should_split(node, splitting_ops):
            subgraph_id += 1  # 开启新子图
            # 连续 splitting op 保持在一起
        node_to_subgraph_id[node] = subgraph_id

    # torch.fx.passes.split_module 按子图 ID 切分
    split_gm = split_module(graph, None, lambda node: node_to_subgraph_id[node],
                            keep_original_order=True)  # 保持顺序（mutation 安全）
    return split_gm, [SplitItem(name, graph_id, is_splitting, submod)]
```

**为何分割**：Attention/MoE 等操作有动态控制流或自定义 kernel，不适合 Inductor 全图编译。切分后，可编译的部分交给 Inductor，不可编译的部分保持 eager。

### 5.3 PiecewiseCompileInterpreter

```python
class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """用 fake inputs 遍历切分后的图，对指定子图编译"""

    def call_module(self, target, args, kwargs):
        output = super().call_module(target, args, kwargs)
        if target in self.compile_submod_names:
            # 创建 PiecewiseBackend 编译此子图
            piecewise_backend = PiecewiseBackend(submod, vllm_config, index, ...)
            # 用 CUDAGraphWrapper 包装（PIECEWISE 模式）
            self.module.__dict__[target] = wrap_with_cudagraph_if_needed(
                piecewise_backend, ..., runtime_mode=CUDAGraphMode.PIECEWISE)
        return output
```

---

## 6. Piecewise 分段编译

### 6.1 PiecewiseBackend

```python
class PiecewiseBackend:
    def __init__(self, graph, vllm_config, piecewise_compile_index, total_piecewise_compiles,
                 sym_shape_indices, vllm_backend, returns_tuple, compiled_runnables=None):
        self.compile_ranges = compilation_config.get_compile_ranges()  # 形状范围
        self.range_entries: dict[Range, RangeEntry] = {}               # 每个范围的编译状态
        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total - 1
        self.is_full_graph = total_piecewise_compiles == 1  # 单图=全图编译
```

### 6.2 运行时调度

```python
def __call__(self, *args):
    runtime_shape = args[self.sym_shape_indices[0]]  # batch size
    range_entry = self._find_range_for_shape(runtime_shape)
    self._maybe_compile_for_range_entry(range_entry, args)  # 首次编译
    return range_entry.runnable(*args)                      # 后续直接调用

def _maybe_compile_for_range_entry(self, range_entry, args):
    if not range_entry.compiled:
        # 编译此形状范围的图
        range_entry.runnable = self.vllm_backend.compiler_manager.compile(
            self.graph, args_list, self.vllm_backend.inductor_config,
            self.compilation_config, compile_range=range_entry.compile_range, ...)
        range_entry.compiled = True
        self.to_be_compiled_ranges.remove(range_entry.compile_range)
        self.check_for_ending_compilation()  # 最后一个图编译完 → 保存缓存
```

### 6.3 compile_ranges（形状范围）

`get_compile_ranges()` 将 `cudagraph_capture_sizes` 转换为 `Range` 列表：

```
cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32, 64]
→ compile_ranges = [Range(1,1), Range(2,2), Range(4,4), ..., Range(65, max_tokens)]

运行时 batch_size=5 → 匹配 Range(5,8) → 用该范围编译的图
```

**两种编译策略**：
- **具体 size**（`Range(4,4)`）：静态形状，最优性能
- **范围**（`Range(5,8)`）：动态形状，一次编译覆盖多个 batch size

### 6.4 AOT 预编译模式

`compiled_runnables` 非 None 时，跳过编译直接加载预编译产物（从缓存/AOT artifact）：

```python
if self.compiled_runnables is not None:
    range_entry.runnable = self.get_compiled_graph_wrapper(
        self.compiled_runnables[str(range_entry.compile_range)])
```

---

## 7. CUDA Graph 捕获与重放

### 7.1 CUDAGraphWrapper

```python
class CUDAGraphWrapper:
    def __init__(self, runnable, vllm_config, runtime_mode, cudagraph_options):
        self.runnable = runnable                    # 被包装的可执行对象
        self.runtime_mode = runtime_mode            # FULL 或 PIECEWISE
        self.graph_pool = current_platform.get_global_graph_pool()  # 全局 graph pool
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # 模式不匹配 → 直接 eager 执行
        if cudagraph_runtime_mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        batch_descriptor = forward_context.batch_descriptor
        entry = self.concrete_cudagraph_entries.get(batch_descriptor)

        if entry is None or entry.cudagraph is None:
            # 首次：捕获 CUDA Graph
            cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=current_stream()):
                output = self.runnable(*args, **kwargs)
                if self.cudagraph_options.weak_ref_output:
                    output = weak_ref_tensors(output)  # 释放内存
            entry.cudagraph = cudagraph
            entry.output = weak_ref_tensors(output)
            return output
        else:
            # 后续：重放
            entry.cudagraph.replay()
            return entry.output
```

### 7.2 BatchDescriptor

`BatchDescriptor` 是 CUDA Graph 缓存的 key，包含 batch size 等形状信息。相同 descriptor 复用同一 graph。

### 7.3 CUDAGraphOptions

```python
@dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True    # 日志（仅首个子图）
    gc_disable: bool = False         # 禁用 GC（分段模式加速捕获）
    weak_ref_output: bool = True     # 弱引用输出（仅最后一个图，省内存）
```

分段模式下：
- 第一个图：`debug_log_enable=True, gc_disable=False`（记录日志 + GC）
- 后续图：`gc_disable=True`（避免每层都 GC，大幅加速捕获）
- 最后一个图：`weak_ref_output=True`（输出不被其他 graph 使用，可弱引用）

### 7.4 全局 Graph Pool

```python
self.graph_pool = current_platform.get_global_graph_pool()
```

所有 CUDA Graph 共享同一个 memory pool，避免重复分配显存。这对分段模式（每层一个 graph）至关重要。

### 7.5 输入地址校验

DEBUG 模式下，重放时校验输入 tensor 的 `data_ptr` 与捕获时一致：

```python
if self.is_debugging_mode:
    new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
    assert new_input_addresses == entry.input_addresses
```

CUDA Graph 要求输入 buffer 地址固定，校验帮助排查 buffer 复用问题。

### 7.6 FULL vs PIECEWISE 捕获

**FULL 模式**（GPUModelRunner 的 `CudaGraphManager`，见 ModelRunner 文档第 9 章）：
- 捕获整个 `model(input_ids, positions)` 前向
- 每个 `cudagraph_capture_size` 一个 graph
- 输入数据预先拷贝到静态 buffer

**PIECEWISE 模式**（`CUDAGraphWrapper`）：
- 捕获每个 splitting op 之间的子图
- 每个 `BatchDescriptor` 一个 graph
- 由 `wrap_with_cudagraph_if_needed` 包装 PiecewiseBackend

---

## 8. Inductor 自定义 Pass 体系

### 8.1 PostGradPassManager

```python
class PostGradPassManager(CustomGraphPass):
    def __call__(self, graph: fx.Graph):
        compile_range = get_pass_context().compile_range
        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)  # 执行 pass
        self.post_cleanup(graph)            # 清理
        self.fix_functionalization(graph)    # 最后修复 functionalization
```

**Pass 执行顺序**：
1. 用户自定义 passes（构造参数）
2. 默认 passes（NoOpEliminationPass, FusionPass 等）
3. `config["post_grad_custom_post_pass"]`（若存在）
4. `fix_functionalization`（最后，需要 functional 图）

### 8.2 configure（按配置启用 pass）

```python
def configure(self, config: VllmConfig):
    if self.pass_config.eliminate_noops:
        self.passes += [NoOpEliminationPass(config)]
    if self.pass_config.enable_sp:
        self.passes += [SequenceParallelismPass(config)]
        if self.pass_config.fuse_gemm_comms:
            self.passes += [AsyncTPPass(config)]
    if self.pass_config.fuse_allreduce_rms:
        self.passes += [AllReduceFusionPass(config)]
    if self.pass_config.fuse_norm_quant:
        self.passes += [RMSNormQuantFusionPass(config)]
    if self.pass_config.fuse_act_quant:
        self.passes += [ActivationQuantFusionPass(config)]
    if self.pass_config.fuse_attn_quant:
        self.passes += [AttnFusionPass(config)]
    if self.pass_config.enable_qk_norm_rope_fusion:
        self.passes += [QKNormRoPEFusionPass(config)]
```

### 8.3 InductorPass 基类

```python
class InductorPass:
    def __call__(self, graph: fx.Graph): ...
    def uuid(self) -> str: ...                    # 影响编译缓存的 uuid
    def is_applicable_for_range(self, compile_range) -> bool: ...

class VllmInductorPass(InductorPass):
    dump_prefix: int  # 调试 dump 索引
```

### 8.4 pass_context

```python
@contextmanager
def pass_context(compile_range: Range):
    # 设置当前编译的形状范围，供 pass 查询
    ...
```

---

## 9. 算子融合 Pass 详解

### 9.1 RMSNorm + 量化融合（fusion.py）

```python
class RMSNormQuantFusionPass(VllmPatternMatcherPass):
    # 匹配模式:
    #   RMSNorm(x) → Quant(y)  →  融合为 RMSNormQuant(x)
    #   FusedAddRMSNorm(x, residual) → Quant  →  FusedAddRMSNormQuant

# 支持的量化变体:
RMSNormStaticQuantPattern          # RMSNorm + 静态 FP8 量化
FusedAddRMSNormStaticQuantPattern # 残差+RMSNorm + 静态量化
RMSNormGroupQuantPattern           # RMSNorm + 分组量化
FusedAddRMSNormGroupQuantPattern  # 残差+RMSNorm + 分组量化
RMSNormDynamicQuantPattern         # RMSNorm + 动态量化
FusedAddRMSNormDynamicQuantPattern # 残差+RMSNorm + 动态量化
```

**效果**：减少一次 kernel launch + 中间 tensor 读写。

### 9.2 Attention + 量化融合（fusion_attn.py）

```python
class AttnFusionPass(VllmPatternMatcherPass):
    # 融合 attention 输出后的量化
AttentionFp8StaticQuantPattern   # Attention + FP8 静态量化
AttentionNvfp4QuantPattern       # Attention + NVFP4 量化
```

### 9.3 Activation + 量化融合（activation_quant_fusion.py）

```python
class ActivationQuantFusionPass(VllmPatternMatcherPass):
    # 融合 SiLU*gate 后的量化
SiluMulFp8StaticQuantPattern  # silu_and_mul + FP8 静态量化
SiluMulNvfp4QuantPattern      # silu_and_mul + NVFP4 量化
```

### 9.4 集合通信融合（collective_fusion.py）

```python
class AllReduceFusionPass(VllmPatternMatcherPass):
    # 融合 AllReduce + RMSNorm + 量化
AllReduceRMSNormPattern                    # AllReduce + RMSNorm
AllReduceFusedAddRMSNormPattern            # AllReduce + 残差 + RMSNorm
AllReduceFusedRMSNormStaticQuantFP8Pattern # AllReduce + RMSNorm + FP8
AllReduceFusedRMSNormStaticQuantNVFP4Pattern # AllReduce + RMSNorm + NVFP4

class AsyncTPPass(VllmPatternMatcherPass):
    # AsyncTP: GEMM + 通信重叠
GEMMReduceScatterPattern       # GEMM + ReduceScatter 融合
AllGatherGEMMPattern           # AllGather + GEMM 融合
ScaledMMReduceScatterPattern   # scaled_mm + ReduceScatter
AllGatherScaledMMPattern       # AllGather + scaled_mm
```

### 9.5 QK Norm + RoPE 融合（qk_norm_rope_fusion.py）

```python
class QKNormRoPEFusionPass(VllmPatternMatcherPass):
    # 融合 Q/K 的 RMSNorm + RoPE
QkNormRopePattern  # RMSNorm(Q) + RoPE(Q), RMSNorm(K) + RoPE(K)
```

### 9.6 其他 Pass

| Pass | 文件 | 功能 |
|------|------|------|
| `NoOpEliminationPass` | noop_elimination.py | 消除 no-op 操作（如 shape 为 0 的算子） |
| `SequenceParallelismPass` | sequence_parallelism.py | 序列并行变换 |
| `FixFunctionalizationPass` | fix_functionalization.py | 修复 functionalization 后的图 |
| `PostCleanupPass` | post_cleanup.py | 清理死节点、冗余操作 |
| `RocmAiterRMSNormFusionPass` | rocm_aiter_fusion.py | ROCm AITER RMSNorm 融合 |
| `RocmAiterSiluMulFp8GroupQuantFusionPass` | rocm_aiter_fusion.py | ROCm AITER SiLU+量化融合 |

---

## 10. 编译缓存机制

### 10.1 缓存结构

```
cache_dir=/path/to/hash_str/rank_i_j/prefix/
├── vllm_compile_cache.py          # CompilerManager 缓存 (Range, graph_idx, backend) → handle
├── computation_graph.py           # FX 计算图
└── transformed_code.py            # 编译后的代码
```

### 10.2 缓存 key

```python
cache: dict[tuple[Range, int, str], Any]
# key = (compile_range, graph_index, compiler_name)
# value = handle (Inductor 的编译产物句柄)
```

### 10.3 缓存流程

```python
def compile(self, graph, example_inputs, ...):
    # 1. 尝试加载
    compiled = self.load(graph, example_inputs, graph_index, compile_range)
    if compiled is not None:
        logger.info("Directly load the compiled graph(s) from the cache")
        return compiled

    # 2. 编译
    compiled, handle = self.compiler.compile(graph, ...)

    # 3. 存入缓存
    if is_compile_cache_enabled(config) and handle is not None:
        self.cache[(compile_range, graph_index, compiler.name)] = handle
        self.is_cache_updated = True
    return compiled

def save_to_file(self):
    # 进程结束时保存到 vllm_compile_cache.py
    if not self.disable_cache and self.is_cache_updated:
        pprint.pformat(self.cache) → write to file
```

### 10.4 compute_hash（缓存目录哈希）

`VllmConfig.compute_hash()` 综合模型结构、并行配置、编译配置等生成哈希，作为缓存目录名。配置变化时自动重新编译。

### 10.5 AOT 独立编译（caching.py）

`VLLM_USE_STANDALONE_COMPILE=1` 启用 AOT 编译：

```python
class InductorStandaloneAdaptor:
    # 编译产物可序列化，支持跨进程加载
    # to_bytes() → pickle 编译产物
    # load(handle) → 反序列化加载
```

`VLLM_USE_MEGA_AOT_ARTIFACT=1`：加载预编译的 AOT artifact，完全跳过运行时编译。

---

## 11. support_torch_compile 装饰器

### 11.1 装饰器作用

```python
@support_torch_compile
class MyAttention(nn.Module):
    def forward(self, x):
        ...

@support_torch_compile(dynamic_arg_dims={"x": 0})  # 指定动态维度
class MyLayer(nn.Module):
    ...
```

标记可被 torch.compile 编译的模块，并声明动态形状维度。

### 11.2 ignore_torch_compile

```python
@ignore_torch_compile
class NotCompiled(nn.Module):
    # 即使父类有 support_torch_compile，此类也不编译
    ...
```

### 11.3 机制

`support_torch_compile` 在模块的 `forward` 上注入 `torch.compile` 包装，使 Dynamo 能正确 trace。`dynamic_arg_dims` 告诉 Dynamo 哪些维度是动态的（如 batch 维），避免重编译。

---

## 12. 编译统计与监控

### 12.1 compilation_counter

```python
from vllm.compilation.counter import compilation_counter

compilation_counter.num_backend_compilations      # 后端编译次数
compilation_counter.num_cudagraph_captured         # CUDA Graph 捕获数
compilation_counter.num_piecewise_capturable_graphs_seen  # 可捕获的分段图数
compilation_counter.num_cache_entries_updated      # 缓存更新次数
```

### 12.2 CUDAGraphLogging

```python
@dataclass
class CUDAGraphStat:
    num_unpadded_tokens: int    # 实际 token 数
    num_padded_tokens: int      # padding 后 token 数
    num_paddings: int           # padding 数量
    runtime_mode: str           # FULL / PIECEWISE

class CUDAGraphLogging:
    def observe(self, stat): ...
    def generate_metric_table(self) -> str:  # 生成 Markdown 统计表
        # 按频率降序排列
```

### 12.3 monitor

```python
validate_cudagraph_capturing_enabled()  # 校验当前是否允许捕获 CUDA Graph
end_monitoring_torch_compile(config)     # 编译结束监控
```

---

## 13. 实际案例：GLM-5.1 编译流程

```bash
vllm serve /models/GLM-5.1-INT4-w4a8 \
    -tp 8 --enable-expert-parallel \
    --quantization slimquant_w4a8 ...
```

### 13.1 默认编译配置（V1）

```python
# V1 默认: mode=VLLM_COMPILE, cudagraph_mode=FULL_AND_PIECEWISE
CompilationConfig:
    mode = VLLM_COMPILE              # 自定义 Inductor backend
    backend = "inductor"
    cudagraph_mode = FULL_AND_PIECEWISE  # decode=FULL, prefill=PIECEWISE
    cudagraph_capture_sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ...]
    splitting_ops = ["vllm.unified_attention", ...]  # attention 切分点
    pass_config:
        fuse_norm_quant = True       # RMSNorm + 量化融合
        fuse_act_quant = True        # SiLU + 量化融合
        fuse_attn_quant = True       # Attention + 量化融合
        eliminate_noops = True
```

### 13.2 编译时序

```
1. 模型加载 → load_model()
2. initialize_kv_cache() → KV cache 分配
3. profile_run() → dummy 前向（不编译，cudagraph_mode=NONE）
4. capture_model() → CUDA Graph 捕获
   │
   ├── 首次 forward 触发 torch.compile:
   │   a. Dynamo trace → fx.GraphModule
   │   b. VllmBackend.__call__:
   │      - split_graph(按 attention 切分)
   │      - PiecewiseCompileInterpreter 遍历子图
   │   c. 每个子图:
   │      - PiecewiseBackend 对每个 compile_range 编译
   │      - InductorAdaptor.compile → Inductor 编译
   │      - PostGradPassManager 执行融合 pass
   │      - wrap_with_cudagraph_if_needed (PIECEWISE)
   │
   ├── FULL CUDA Graph 捕获 (GPUModelRunner.CudaGraphManager):
   │   - 对每个 cudagraph_capture_size:
   │     - warmup run
   │     - torch.cuda.graph() 捕获整个 forward
   │
   └── 编译产物缓存到 VLLM_CACHE_ROOT
```

### 13.3 运行时调度

```
execute_model(scheduler_output):
    num_tokens = total_num_scheduled_tokens
    cudagraph_size = get_cudagraph_size(num_tokens, ...)
    use_cudagraph, num_tokens_after_padding, ... = get_cudagraph_and_dp_padding(...)

    if use_cudagraph:
        # FULL 模式: 重放整个 forward 的 CUDA Graph
        hidden_states = cudagraph_manager.run(num_tokens_after_padding)
        # forward_context.cudagraph_runtime_mode = FULL
    else:
        # Eager 或 PIECEWISE 模式
        with set_forward_context(attn_metadata, ..., cudagraph_runtime_mode=PIECEWISE):
            hidden_states = model(input_ids, positions)
            # 每个子图的 CUDAGraphWrapper 按 PIECEWISE 模式重放
```

### 13.4 MoE 层的编译处理

GLM-5.1 的 MoE 层是 splitting op，会被切分到独立子图：
- MoE 的 router + fused_experts 保持在 eager（因有动态控制流）
- MoE 前后的 RMSNorm + Linear 被 Inductor 编译
- `fuse_act_quant` 融合 MoE 输入的 SiLU + 量化
- `fuse_norm_quant` 融合 MoE 后的 RMSNorm + 量化

---

## 14. 配置参数与环境变量

### 14.1 启动参数

| 参数 | 说明 |
|------|------|
| `--compilation-mode <mode>` | NONE / DYNAMO_TRACE_ONCE / VLLM_COMPILE |
| `--cudagraph-mode <mode>` | NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE |
| `--cudagraph-capture-sizes <list>` | CUDA Graph 捕获的 batch size 列表 |
| `-O <level>` | 优化等级 O0-O3 |
| `--enable-sp` | 启用序列并行 |

### 14.2 环境变量

```bash
# 编译后端
VLLM_USE_STANDALONE_COMPILE=0     # AOT 独立编译（需 Torch ≥ 2.8）
VLLM_USE_MEGA_AOT_ARTIFACT=0      # 加载预编译 AOT artifact（需 STANDALONE_COMPILE）

# 编译缓存
VLLM_CACHE_ROOT=~/.cache/vllm     # 缓存根目录
VLLM_DISABLE_COMPILE_CACHE=0      # 禁用编译缓存

# 调试
VLLM_PATTERN_MATCH_DEBUG=         # pattern match 调试
VLLM_LOGGING_LEVEL=INFO           # 日志级别（DEBUG 输出 CUDA Graph 校验）

# CUDA Graph
VLLM_USE_CUDAGRAPH_SIZES="1,2,4,8"  # 自定义捕获大小
```

### 14.3 PassConfig 相关

```python
# 通过 CompilationConfig.pass_config 配置
pass_config.fuse_norm_quant = True
pass_config.fuse_act_quant = True
pass_config.fuse_attn_quant = True
pass_config.fuse_allreduce_rms = True
pass_config.fuse_gemm_comms = True   # AsyncTP
pass_config.enable_sp = True
pass_config.enable_qk_norm_rope_fusion = True
```

---

## 15. 调优建议与常见问题

### 15.1 CUDA Graph 调优

| 问题 | 原因 | 解决 |
|------|------|------|
| 启动时间长 | 首次编译 + CUDA Graph 捕获 | 启用编译缓存，二次启动变快 |
| CUDA Graph 捕获 OOM | 显存不足 | 减少 `cudagraph_capture_sizes`，降低 `gpu_memory_utilization` |
| Eager fallback 频繁 | batch size 未覆盖 | 增加 `cudagraph_capture_sizes` |
| `Input addresses different` | buffer 复用冲突 | DEBUG 模式排查，确保输入 buffer 固定 |
| 捕获失败 | 动态控制流 | 检查 splitting_ops 是否正确切分 |

### 15.2 编译调优

| 优化 | 方法 |
|------|------|
| 减少编译时间 | 启用缓存 `VLLM_CACHE_ROOT`；AOT 编译 `VLLM_USE_STANDALONE_COMPILE=1` |
| 减少重编译 | `DYNAMO_TRACE_ONCE` 模式避免动态形状触发重编译 |
| 最优性能 | `FULL_AND_PIECEWISE`（V1 默认）+ 覆盖常用 batch size |
| 调试编译 | `VLLM_LOGGING_LEVEL=DEBUG` 查看 CUDA Graph 捕获/重放日志 |

### 15.3 算子融合调优

| Pass | 效果 | 启用条件 |
|------|------|---------|
| `fuse_norm_quant` | RMSNorm+量化减少 1 次 launch | FP8/INT8 量化模型 |
| `fuse_act_quant` | SiLU+量化减少 1 次 launch | 量化模型 |
| `fuse_attn_quant` | Attention+量化减少 1 次 launch | 量化模型 |
| `fuse_allreduce_rms` | AllReduce+RMSNorm 融合 | TP > 1 + FlashInfer |
| `fuse_gemm_comms` | GEMM+通信重叠 | AsyncTP |
| `enable_qk_norm_rope_fusion` | QK Norm+RoPE 融合 | 支持的模型（如 Gemma3） |

### 15.4 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 编译后结果错误 | pass 引入 bug | 逐个禁用 pass 排查 |
| 编译 cache 不生效 | 配置 hash 变化 | 检查 `compute_hash`，确认配置未变 |
| 分段编译比全图慢 | splitting ops 太多 | 减少 splitting_ops，用 FULL 模式 |
| `gc_disable` 内存增长 | GC 被禁用 | 仅分段模式首图后禁用，正常现象 |
| Inductor 编译失败 | 不支持的 op | 加入 splitting_ops 排除该 op |
| AOT artifact 加载失败 | Torch 版本不匹配 | `VLLM_USE_STANDALONE_COMPILE` 需 Torch ≥ 2.8 |

---

> **文档版本**：基于 vLLM 0.15.x V1 架构  
> **生成日期**：2026-08-15  
> **核心文件**：`vllm/compilation/backends.py`, `compiler_interface.py`, `cuda_graph.py`, `piecewise_backend.py`, `pass_manager.py`, `fusion.py`, `collective_fusion.py`, `decorators.py`, `vllm/config/compilation.py`
