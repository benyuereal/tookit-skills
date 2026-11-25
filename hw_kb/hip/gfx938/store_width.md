## 写回向量宽度只有读的一半（`buffer_store_dwordx2` vs `buffer_load_dwordx4`）

**状态**：✅ 已证实（ISA 静态计数，2026-11-24）

**适用**：gfx938 / gfx936

### 现象

对一个已调优的 elementwise kernel 做 ISA 反汇编统计，访存指令的宽度**不对称**：

| 指令 | 出现次数 | 位宽 | 含义 |
|---|---|---|---|
| `buffer_load_dwordx4` | 12 | 128-bit | 读：128-bit 用满 ✅ |
| **`buffer_store_dwordx2`** | **32** | **64-bit** | **写：只有 64-bit** ⚠️ |

**读用满 128-bit，写只有 64-bit。** 32 次 64-bit 写本可以是 16 次 128-bit。

同时可以看到全走 `buffer_load`、`global_load` 计数为 0 —— 这是**正确**的
（`global_load` 是标量路径，`buffer_load` 才走向量单元）。

### 为什么会这样

写回时如果每个线程的输出不是 16 字节对齐的连续块，编译器会保守地退到
`dwordx2`。常见原因：

1. **写回地址的基址没对齐**（例如 `base = row * H + off`，`off` 是 8 的倍数但不是 16）
2. **`VEC` 是奇数或 8 的倍数但不是 16 的倍数**，导致 uint4 写无法成立
3. 用了 `bf16` 标量写而不是打包成 `uint4` 写

### 怎么查

```bash
# 反汇编后数一下，两者应该大致 2:1 的关系（读 128-bit，写 128-bit）
llvm-objdump -d --mcpu=gfx938 kernel.so | grep -oE "buffer_(load|store)_dwordx[0-9]" \
  | sort | uniq -c | sort -rn
```

**判据**：如果 `store_dwordx2` 的次数是 `load_dwordx4` 的 2 倍以上，
说明写回没有用满 128-bit。

### 怎么改

确保写回是 `uint4`（16 字节）而非 `uint2`：

```cpp
// ❌ 编译成 buffer_store_dwordx2（64-bit）
*(uint2*)(y + base + off) = yv;

// ✅ 编译成 buffer_store_dwordx4（128-bit）
*(uint4*)(y + base + off) = yv;   // 要求 off 是 16 的倍数
```

**前提**：`off` 必须是 16 的倍数，否则会越界或错位。
如果 `H` 不是 16 的倍数，需要用尾部处理分支。

### 相关性

- 这条和 [[vec16_uint4]] 是同一个问题的两面：
  那条讲「用 uint4 读」，这条讲「用 uint4 写」
- 和 [[private_spill_trap]] 的区别：那个是寄存器不够，
  这个是**寄存器够但指令选窄了**，症状相似（都慢），改法完全不同

### 未验证

- 写回从 `dwordx2` 提到 `dwordx4` 能快多少 —— **没测过**
  （只在代码里找到了这个模式，还没做 A/B）
- 是否所有 elementwise kernel 都有这个现象，还是只有这一个

> 按本仓的规矩，**没做过 A/B 的优化不算已验证**。
> 这条目前只能作为「值得一看的信号」，不能当作「改了就能快 2 倍」。
