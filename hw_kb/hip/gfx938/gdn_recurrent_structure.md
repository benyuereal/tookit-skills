## GDN recurrent kernel 的结构与可用边界（lightop `qwen3.8-27b-perf-gdn` 分支）

**状态**：✅ 已验证（读源码 + 接口确认，2026-11-26）
**来源**：`lightop` 仓库分支 `qwen3.8-27b-perf-gdn`，提交 `bd716f4`
（`feat(gdn): ship the hand-written recurrent kernel`）
**适用**：gfx938

### 它是什么

GDN = Gated Delta Net，线性注意力的 delta-rule 变体。
GLM-5.3 decode 里占 **1.41 ms / 3.2%**（`optimize-glm53/HANDOFF.md`）。

这个分支里有一整套手写 HIP 实现：

```
csrc/gdn/gdn_recurrent.hip    1174 行   手写 HIP kernel
csrc/gdn/gdn_binding.cpp       738 行   C 绑定（含 stride 推导）
csrc/gdn/gdn_recurrent.h        93 行   接口契约（给 ffi.cdef 用）
python/lightop/gdn.py          254 行   Python 语义包装
scripts/build_gdn.sh            73 行   构建
```

**已编译的 `.so` 也提交在仓库里**（529 KB，`gfx938/_gdn.cpython-310-*.so`）。

### 6 个 C 入口（不是 1 个）

| 入口 | 用途 |
|---|---|
| `gdn_recurrent_update` | 基础版：一次扫完 T 步 |
| `gdn_recurrent_update_bplus` | + 存储步/行（快照） |
| `gdn_recurrent_update_bplus_state` | + 状态回写 |
| `gdn_recurrent_update_nofma` | 无 FMA 版（兼容/对照） |
| `gdn_recurrent_update_bplus_state_multi` | 多层一次 launch |
| `gdn_verify_norm_pass` | 只做 q/k l2norm + gating |

**这个结构本身就是一条信息**：有 `nofma` 版说明
「用不用 FMA」在这条 kernel 上被当作可调项；
有 `_multi` 版说明**多层合批能省 launch 开销**。

### Python 层暴露的能力（比 kernel 更值得抄）

```python
gdn_norm_pass(...)          # q/k l2norm（dense fp32）+ eg/beta gating
gdn_verify(...)             # 只写 committed snapshot row（B+）
gdn_verify_full(...)        # 每步都写
gdn_compensate(...)         # 重放被截断的 slot
gdn_compensate_multi(...)   # 多层一次 launch 重放
gdn_conv1d_unsupported_reason(...)   # **返回字符串说明为什么必须留在 Triton**
gdn_conv1d_update(...)      # HIP causal_conv1d
```

### 三个值得吸收的机制

**① 能力降级要有显式的「为什么」**

```python
gdn_conv1d_unsupported_reason(...) -> Optional[str]
```

不是「支持/不支持」的布尔，而是**返回一句原因**（`None` = 能跑）。
调用方可以据此决定 fallback 到 Triton，而不是二元判断 + 猜。

> 对照本仓：我们的 `candidates.yaml` 里 `status: blocked` + `blocker:`
> 是同一个思路。但这条更彻底——**在运行期问，不是一个静默的路由**。

**② 精确到 ULP 的等价性声明**

> "The recurrence is bit-exact with the aiter/Triton reference (`maxULP=0`
> over the recorded sweep)"

**`maxULP=0`** 是极强的声明。本仓的容差判据是「bf16 有 7 位尾数，
ULP 随量级变」——这条做到了 0 ULP，说明计算顺序完全一致。

> 但它只说 "over the recorded sweep"，**扫描范围要问清楚**，
> 不能当成全形状成立。

**③ 一个会「静默算错」的陷阱（值得单独记）**

> "`A_log` and `dt_bias` are read as `const float*` by every entry:
> pass bfloat16 tensors through f32() **or the kernel runs off the buffer
> and every head's decay is wrong**."

**给错精度的张量不会报错，会读越界、结果全错。**
这正是本仓 `uninit-struct` / `wrong-variable` 那一族的问题：
**输入契约不匹配时，现象是「算错」而不是「崩」，最难发现。**

修法写在源码里：调 `f32()` 或直接用 Python 层的 wrapper（它替你转换）。

### 实测：已确认是**延迟受限**，不是带宽受限（2026-11-26）

自己编译了这个 kernel（aicc + 简化 flags），在 H=16/K=128/V=128/HV=32 上实测：

| T（递推步数） | 总耗时 | 每步 |
|---|---|---|
| 16 | 0.1183 ms | **7.39 us** |
| 32 | 0.2510 ms | **7.84 us** |

**每步耗时基本恒定，总耗时随 T 线性增长 —— 这是串行延迟的签名。**

带宽下限推算：T=16 时读 q/k/v 约 0.25 MB，
`0.25 MB / 1680 GB/s = 0.15 us`，而实测 110–118 us，**差约 740 倍**。
**带宽占比 0.1%，完全不是瓶颈。**

### 这意味着优化空间很窄

唯一的杠杆是**缩短单步的依赖链**。而作者已经：
- 把归约 shuffle 从 80 次（G=32）压到 12 次（G=8）
- 状态全放寄存器（`float h[KL][VL]`，64 个 f32，125 VGPR / 0 溢出）
- 试过 DPP 内联汇编，不可靠，放弃

**剩下能试的只有「软流水」**：让 t+1 步的 k/v 加载与 t 步的计算重叠。
但 delta rule 的状态更新是严格递推的，**能重叠的只有「加载」，不是「计算」** ——
而加载只占 0.1%，重叠它收益上限极低。

### 结论：不建议继续优化这条 kernel

三条独立证据：
1. 实测确认是延迟受限，带宽占比 0.1%
2. 作者已扫过 G / WAVES / k 布局三个关键参数，且留了实测数据
3. 唯一的剩余杠杆（软流水加载）天花板只有 0.1%

**真正的优化机会在别处**：比如减少 GDN 的调用次数、
或把相邻算子融进同一趟（省的是 launch 开销和中间的读写）。

### 与本仓的接入点

如果要建 `op-gdn-recurrent` 工坊，注意：

- **不要复制这个 kernel**（它在别人分支上正在用）
- 用 `lightop.gdn` 的 Python 接口作为**候选实现之一**
- 竞品：aiter/Triton 的参考实现（源码注释里提到 bit-exact 对比过）
- 语义不变量：`maxULP=0` 这个基线可以当精度门的锚

### 未验证

- 6 个入口里哪些在生产中被用到（`HANDOFF` 只说 GDN 占 1.41ms，没拆）
- `nofma` 版与普通版的实际性能差
- `_multi` 版省了多少 launch（prefill 的 launch 开销占比已知约 0，decode 未知）
- 那套 `snapshot / compensate` 机制解决的问题（看起来是**投机解码的
  state rollback**，但未向作者确认）
