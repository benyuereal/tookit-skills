# Inductor 在 DCU 上生成 Triton Kernel（实测验证）

> 来源：公众号文章《Torch解析系列四: Inductor原理详解-从FX图翻译IR 到Triton内核》（AI圈的9527，参考 jino-rohit.github.io/blogs/15_inductor.html）
> 验证：2026-09-17 在 <device-A>（BW150/gfx936，torch 2.9.0+hip6.3.26113，triton 3.5.1，DTK 26.04）起容器挂卡 0 实测，全部 PASS。
> 定位：本 skill（dcu-triton-source）讲 hcu backend 结构；本文件补**中间层**——Inductor 如何在 DCU 上把 FX 图变成 Triton kernel，以及文章与 DCU 实际的差异。

## 1. 三阶段流程（文章核心，DCU 同样适用）

```
FX 图（Dynamo 捕获 + AOTAutograd 切前/后向）
  → ① 降级成 Inductor IR：每个节点存 inner_fn「配方」，不执行、不分配显存
  → ② 调度 + 融合：逐点算子配方卷进同一个 inner_fn；归约算子单走 Reduction 表达
  → ③ 代码生成：执行 inner_fn，ops.* 经 OpOverrides 子类返回代码字符串；CSE 去重相同 load
```

最小示例（文章同款，DCU 实测通过）：
```python
@torch.compile(fullgraph=True)
def f(x):
    b = torch.floor(x) + torch.ceil(x)
    c = b.sum(dim=-1)
    d = c + 1
    return d
```

## 2. DCU 实测生成的 kernel（torch 2.9.0 原文）

`floor+ceil+add+sum+add` 五个算子融进**一个** kernel `triton_per_fused_add_ceil_floor_sum_0`：

```python
@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    triton_meta={... 'device': DeviceProperties(type='hip', index=0,
        multi_processor_count=80, cc='gfx936', major=9,
        regs_per_multiprocessor=196608, max_threads_per_multi_processor=2560,
        warp_size=64), ...},
)
@triton.jit
def triton_per_fused_add_ceil_floor_sum_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 16384          # 32*512（归约输出维度）
    r0_numel = 1024          # 归约维度
    R0_BLOCK: tl.constexpr = 1024
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)   # ★ 只 load 一次
    tmp1 = libdevice.floor(tmp0)
    tmp2 = libdevice.ceil(tmp0)                        # ★ 复用 tmp0（CSE 生效）
    tmp3 = tmp1 + tmp2
    tmp6 = tl.sum(tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK]), 1)[:, None].to(tl.float32)
    tmp8 = tmp6 + 1.0
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, None)
```

验证结果：
- 正确性 max_diff = 0.00（vs eager reference）
- `tl.load` 全 kernel 仅 1 次，floor/ceil 各 1 次 → CSE 把两次相同地址的 load 合并
- 归约走 `persistent_reduction`（持久化归约，带 size_hints/reduction_hint）

## 3. 文章 vs DCU 实际差异（排查时别按文章找）

| 维度 | 文章写法 | DCU 实测（torch 2.9.0） |
|---|---|---|
| 数学函数 | `tl.math.floor(x)` | **`libdevice.floor(x)`**（`torch._inductor.runtime.triton_helpers.libdevice`） |
| 归约 kernel | 普通 reduction 表达 | **`persistent_reduction`** 装饰器 + `ReductionHint.INNER` |
| 设备信息 | 未提 | kernel 元数据直接带 **`multi_processor_count=80, cc=gfx936, warp_size=64`**（DCU CU 数/wavefront 可见） |
| CSE 类 | 简化版 `generate(buffer, expr)` | 真实 `CSE.generate` 带 `bounds/write/assignment/dtype/shape` 多参数（`codegen/common.py:2067`） |
| OpOverrides | 简化版 `TritonOverrides` | 真实在 `codegen/triton.py:1328`，`CppOverrides` 在 `codegen/cpp.py:647` |

**排查含义**：在 DCU 上读 Inductor 生成的 Triton 代码，数学函数找 `libdevice.*` 不是 `tl.math.*`；归约 kernel 看 `persistent_reduction` 的 size_hints 判断分块策略。

## 4. 复现脚本

```bash
# <device-A> 起容器（挂卡 0，不碰别人的容器）
docker run -d --name <container> \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v /opt/hyhal:/opt/hyhal:ro -v /tmp:/tmp \
  -e HSA_VISIBLE_DEVICES=0 -e ROCM_PATH=/opt/dtk \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620 \
  sleep 3600

# 跑验证（脚本见下）
docker exec <container> bash -c 'rm -rf /tmp/inductor_dcu_cache && python3 /tmp/verify_inductor_dcu.py'

# 清理
docker rm -f <container>
```

验证脚本要点（`/tmp/verify_inductor_dcu.py`）：
```python
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_dcu_cache"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
import torch, re, triton

# 1. 确认 backend
target = triton.runtime.driver.active.get_current_target()
assert "gfx9" in str(target.arch)

# 2. 编译 + 正确性
@torch.compile(fullgraph=True)
def f(x):
    b = torch.floor(x) + torch.ceil(x)
    c = b.sum(dim=-1)
    return c + 1

x = torch.randn(32, 512, 1024, device="cuda")
y = f(x); torch.cuda.synchronize()
ref = (torch.floor(x) + torch.ceil(x)).sum(dim=-1) + 1
assert (y - ref).abs().max().item() < 1e-4

# 3. 从 cache 找融合 kernel，验证 CSE（主计算段 tl.load 仅 1 次）
for root, _, files in os.walk("/tmp/inductor_dcu_cache"):
    for fn in files:
        if fn.endswith(".py"):
            p = os.path.join(root, fn)
            c = open(p).read()
            if "tl.load" in c and "floor" in c and "ceil" in c:
                m = re.search(r"(tl\.load.*?)(?=tl\.store|return)", c, re.S)
                assert len(re.findall(r"tl\.load\(", m.group(1))) == 1, "CSE FAIL"
```

## 5. 与兄弟 skill 关系

- `dcu-triton-source`（父 skill）：hcu backend 结构 / DTK 工具链 / 调优 CLI → 本文件是它的**代码生成层**补充
- `dcu-triton-kernels`：手写 Triton kernel（DSL 层）→ 本文件讲的是 Inductor **自动生成**的 kernel 长什么样
- 读 Inductor 源码入口：`torch/_inductor/codegen/common.py`（CSE/OpOverrides）、`codegen/triton.py`（TritonOverrides）、`codegen/cpp.py`（CppOverrides）
