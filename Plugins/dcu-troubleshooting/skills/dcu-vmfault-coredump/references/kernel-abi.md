# 内核参数 ABI 参考

解 kernel args hex dump 需要知道该 kernel 的参数布局（= ABI）。来源：**框架/算子的 launch 处源码**。下面记录常用布局与真实案例。

## TensorInfo 布局（PyTorch CUDA 侧）

来源：`aten/src/ATen/cuda/detail/TensorInfo.h`（本地 fork `/txdata/zhaosong/gitlab/pytorch`）。本 fork 用 **int32 sizes/strides**：

```
struct TensorInfo<T, IndexType> {
    T*          data;            // @0     8 字节
    int32_t     sizes[25];       // @8     100 字节
    int32_t     strides[25];     // @108   100 字节
    uint32_t    dims;            // @208   4 字节
    // pad -> 216 字节
};
```

- 尺寸：216 字节。
- `sizes`/`strides` 是定长 `[25]` 数组（`MAX_TENSORINFO_DIMS`），只有前 `dims` 个有效。
- IndexType 决定寻址类型：`unsigned int`（uint）或 `long`（int64）。sizes/strides 仍按 4 字节 int32 存。

## 真实案例：`indexSelectSmallIndex`（embed_tokens OOB）

崩溃 kernel（c++filt 后）：
```
at::native::{anonymous}::indexSelectSmallIndex<c10::Half, long, unsigned int, 2, 2, -2>(...)
```

launch 处（`aten/src/ATen/native/cuda/Indexing.cu:1715`），7 参数：
```cpp
indexSelectSmallIndex<<<...>>>(dst, src, indices,
    dstSelectDim, srcSelectDim, innerSize, srcSelectDimSize);
```

参数布局（928 字节，`size` 字段匹配日志里的 `DUMP KERNEL ARGS: size: 928`）：
| 偏移 | 内容 |
|------|------|
| 0    | dst：TensorInfo（216 字节） |
| 216  | src：TensorInfo |
| 432  | indices：TensorInfo |
| 648  | dstSelectDim (int32) |
| 652  | srcSelectDim (int32) |
| 656  | innerSize (uint32) |
| 664  | srcSelectDimSize (int64) |
| ...  | 其余 padding 到 928 |

这就是 `parse_kernel_args.py` 的默认 ABI（`DEFAULT_ABI`）。真实解析结果：
```
dst: data=0x7ef3247aaa00 dims=2 sizes=[4, 2048]
src: data=0x7ef1cb000000 dims=2 sizes=[248320, 2048]
indices: data=0x7ef3cd1e3400 dims=1 sizes=[4]
srcSelectDimSize=248320  innerSize=2048
```

## 如何为一个新 kernel 找 ABI

1. 用 `c++filt` 得到 op 类型 + 模板参数（dtype、维数、索引 dtype）。
2. 在框架 fork 里 grep kernel 名或对应的 launch 函数，找到 `<<<...>>>(...)` 调用。
3. 按源码里参数顺序排布：TensorInfo（216 字节/个）从 @0 起连续排，标量随后（int32/int64 各自对齐）。
4. 把布局写成一个自定义 ABI JSON，喂给 `parse_kernel_args.py` 验证（对照日志 size）。

## 非 PyTorch 内核的 ABI

崩溃不一定来自 PyTorch kernel。Triton / CK / 自定义 HIP 内核没有 TensorInfo，参数是**扁平 struct**（指针 + 标量按自然对齐排布）。

| 内核族 | ABI 形态 | 怎么拿 |
|--------|---------|--------|
| torch.compile Triton（`triton_poi_fused_*`） | 扁平 struct：数据指针 + grid/block int | 生成的 `.ttir` / `.hip` 源码里 kernel 签名 |
| 手写 Triton kernel | 同左（签名里的每个参数按 8 字节对齐） | Triton 编译产物 |
| CK 模板 GEMM（`Cijk_Ailk_Bljk*`） | 大 struct：前几个是 A/B/C 指针，后面全是 stride/tile int | CK 模板实例化参数，源码里模板实参 |
| 自定义 HIP kernel（`hcu_cache_kernel.cu`、flash-attn 等） | 函数签名 → 扁平 struct | 头文件/`.cu` 源码签名 |

**没有 ABI 时的通用策略：盲解码 + 段交叉引用。** 对任何内核族成立：
1. `python parse_kernel_args.py CRASH.log --raw --gpucore gpucore.21292`
2. 盲扫每个 u64 word；落在 gpucore PT_LOAD 段的 = 数据指针（tensor buffer），段 memsz = 该 buffer 大小。
3. 把段大小与模型/框架已知张量形状对号（字节数 = prod(shape) × dtype size），就能识别"哪个 buffer 是哪个"。
4. 非段内的大地址（`0xFFFF...`）往往就是 OOB 目标（越界值本身）。

一个真实特征：**CK GEMM 的 arg struct 里，A/B/C 指针是前几个 u64，后面几十个 u64 全是 stride/tile 常数**——盲扫时前面几个对上段的指针就是输入输出，可直接去 gpucore 读数据。

## 已知坑

- **TensorInfo 尺寸不是固定的**：若 fork 改了 `MAX_TENSORINFO_DIMS`（比如 32），216 → 288 字节，所有偏移跟着变。先确认 `TensorInfo.h`。
- **IndexType 影响的是寻址，不是 sizes/strides 存储宽度**：sizes/strides 总是 int32。
- 标量对齐：int64 对齐到 8 字节边界；若某个标量前边是 int32，可能有 4 字节 pad。
- 盲解码的指针可能不在任何 PT_LOAD 段（host/GTT/未提交内存）：这类地址要么是 OOB 值，要么是没被 coredump 捕获的地址空间——别急着排除，先看它是不是越界计算的结果。
