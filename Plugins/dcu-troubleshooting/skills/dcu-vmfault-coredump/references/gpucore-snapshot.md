# gpucore 现场数据恢复原理

崩溃时 GPU 显存内容其实还在（trap 只中止执行，不清零、不回收）。gpucore 是运行时在崩溃瞬间"拍照"下来的
显存快照。本页讲清"从 GPU 虚拟地址到文件字节"的恢复链路，以及如何确认读到的数据可信。

## 1. 前提：`HSA_ENABLE_COREDUMP=1`

进程带这个环境变量启动时，运行时在 trap 发生时遍历 GPU 地址空间，把每个已分配区域按页写进一个文件
（gpucore，真实案例 56GB）。**没有它就没有快照**，只能停在"指针+形状"这层，拿不到 fault-time 数值。

## 2. gpucore 是标准 ELF，PT_LOAD 段 = 显存区域

每个内存区域对应一个 `PT_LOAD` 段，四个字段定义"GPU 地址空间 ↔ 文件字节"映射：

```
seg va=0x7ef1cb000000  memsz=0x3ca00000  foff=0x...  filesz=0x3ca00000
```

| 字段 | 含义 |
|------|------|
| `p_vaddr`（va） | 这段内存的 **GPU 虚拟地址起点** |
| `p_memsz` | 这段声明的内存大小（含未写的 bss） |
| `p_offset`（foff） | 这段内容在 **ELF 文件里的起点** |
| `p_filesz` | 文件里**实际写进去**的字节数（≤ memsz） |

## 3. VA → 文件偏移换算（核心一步）

kernel args 里解出的 `data` 指针是 **GPU 虚拟地址**。有上面的映射表就能换算成文件偏移：

```
rel  = va - p_vaddr     # 段内偏移
foff = p_offset + rel   # 文件里该指针的位置
```

`decode_gpucore.py` 的 `read_va()` 就是干这个的：扫描所有 PT_LOAD 段 → 找 `va` 落在哪个段 → 按偏移读 N 字节。

**全程 mmap**：56GB 文件不加载进内存，只按需读那几十个字节——这是能跑大现场的关键。

## 4. 读 fault-time 的真实值

第 2 层解 args 得到三个指针（indices / src / dst），第 3 层在这三个地址读内存。真实案例：

```bash
python decode_gpucore.py gpucore.21292 \
    0x7ef3cd1e3400:64  0x7ef1cb000000:96  0x7ef3247aaa00:64
```

读出来 `indices = [40, 0, 29348, -1]` —— 这就是**越界值 -1 的现场**。src 头几行用于与配置对号
（`248320×2048×2 = 0x3CA00000` 与 PT_LOAD 段 memsz 精确匹配 → src = `embed_tokens.weight`）。

## 5. 两个保底机制

- **bss/未写区域**：`rel >= filesz` 时返回 `unwritten/bss`，不把垃圾当现场（崩溃进程可能有些 buffer 只有 memsz 没有 filesz）。
- **确认数据是 fault-time 的**（`compare_buffers.py`）：把 dst 每行与 src 对应 index 行**逐字节比对**。
  案例里 dst row0/1/2 与 src row40/0/29348 逐字节 64/64 匹配，row3 是 NaN（未写）→ 证明这 4 行是崩溃前按顺序写进去的，
  **不是崩溃后内存被复用污染**。字节级匹配是"现场数据可信"的最强证据。

## 一句话总结

> **崩溃 trap 保留显存 → 运行时 dump 成 ELF → PT_LOAD 段给出 VA↔文件偏移映射 → args 里的指针当"路标" →
> mmap 按偏移读出 fault-time 值。**

## 已知坑

- **gpucore 很大**（真实案例 56GB）：一定要 mmap + 按段偏移读，别 f.read() 全量。
- **一个指针可能在多个段的条件之外**：`sva <= va < sva + memsz` 是左闭右开，指针落在 memsz 末尾是越界。
- **`--ptrs` 模式只报告归属，不读内容**：先看指针落在哪个段、memsz 多大、读得到几字节，再决定读多少。
- **同型号 DCU 才能复用 dump**：gpucore 的段布局依赖驱动/运行时版本，跨版本段表可能对不上。
- **没有 gpucore 时降级**：`parse_kernel_args.py --raw` 不带 `--gpucore` 只能得到指针+形状（疑似指针标注），
  无法确认 fault-time 值——这是 skill 把 `HSA_ENABLE_COREDUMP=1` 列为前置的原因。
