## 这个 kernel 必须用 aicc 编，hipcc 会「静默成功但产出空 .so」

**状态**：✅ 已证实（源码注释 + 构建脚本，2026-11-26）
**来源**：`lightop` 分支 `qwen3.8-27b-perf-gdn`，`scripts/build_gdn.sh`
**适用**：gfx938，以及任何依赖 `-mllvm` 后端选项调优的 kernel

### 现象（两层，都很隐蔽）

**第一层：hipcc 静默失败**

> "hipcc rejects those flags (prints the option help, **exits 0 and leaves
> an empty .so**)"

**编译退出码是 0**，但 .so 里没有任何 kernel。如果不检查符号就发版，
装上去会在运行期找不到入口 —— 而且错误信息指向加载失败，不指向编译。

**第二层：能编过也不行 —— 数值会变**

> "even a plain hipcc build changes the numerics, which drops the speculative
> accept rate **0.95 → 0.89** and costs **~8% end to end**"

这条更贵：hipcc 编出来的 kernel **能跑、不报错、结果看起来合理**，
但数值与 aicc 版有微小差异，导致投机解码的接受率下降，
**端到端慢 8%**。

> **这属于本仓 `verifier-lies` 那一族**：不是「编译失败」，是
> 「编译成功但产物不是你以为的那个」。

### 为什么会有数值差异

`-mllvm` 那批选项里有几个**改变重结合顺序 / 消去中间结果的优化**：

```
-mllvm=-enable-latency-hack=true      # 延迟 hack，会重排指令
-mllvm=-check-valu-data-forward-hazards=0   # 关掉某些 hazard 检查
-mllvm=-hcu-update-wait-by-reverse-search=true  # 反向搜索决定 wait 位置
```

这些不是「更快但同结果」，而是**改变了浮点计算的结合顺序**。
对普通的 elementwise 无所谓（<1 ULP），对**递推 / 累积**（recurrent kernel）
误差会在 T 步上放大，最终改变下游的 argmax → 接受率下降。

### 正确的构建流程（照抄 build_gdn.sh）

```bash
AICC=/opt/dtk/bin/aicc
GDN_FLAGS=(
  -O3 -fPIC --offload-arch=gfx938
  -mllvm=-enable-latency-hack=true
  -mllvm=-mmac-latency=5
  -mllvm=-ds-load-store-latency=6
  -mllvm=-disable-machine-sink=True
  -mllvm=-check-valu-data-forward-hazards=0
  -mllvm=-disable-cluster-lds-memops=true
  -mllvm=-amdgpu-disable-backoff-barrier=false
  -mllvm=-support-768-vgprs=true
  -mllvm=-hcu-update-wait-by-reverse-search=true
  -mllvm=-hcu-pre-emit-load-store-opt=false
)
"$AICC" "${GDN_FLAGS[@]}" -I csrc -c csrc/gdn/gdn_recurrent.hip -o gdn_recurrent.o
```

**注意 `.hip` 用 aicc 编，`.cpp` 用 g++ 编，最后 g++ 链接。**
（binding 是纯 C++，不需要 aicc。）

### 必须做的产物校验（`build_gdn.sh` 里已有，值得抄）

```bash
# A silent compile failure would ship an empty module; assert the entries.
for name in gdn_recurrent_update gdn_recurrent_update_bplus ...; do
    nm -D --defined-only "$OUT" | grep -q " $name\$" || {
        echo "missing $name (the HIP compile failed silently)" >&2; exit 1
    }
done
```

**注释里那句话就是判据**：「A silent compile failure would ship an empty module」。
没有这段校验，空 .so 会一路装到线上。

### 与本仓已有机制的关系

`scripts/env_probe.py` 的 `check_build(src, so, symbols=[...])` 就是做这件事的。
**任何新 kernel 接入时，都应该传 `symbols=` 参与检查。**

但它只查「符号在不在」，查不出上面第二层（数值变了）。
数值那层要靠**端到端指标**（这里是投机接受率）才能发现 ——
**这类问题只能靠「同批 A/B 比接受率」发现，不能靠单看 kernel 耗时。**

### 通用化：什么时候该怀疑这一类问题

| 症状 | 可能原因 |
|---|---|
| 编译退出 0 但 .so 很小 / `nm` 查不到符号 | aicc-only 的 flag 被 hipcc 吞了 |
| kernel 跑得动、结果「看起来对」，但端到端变慢 | 数值被重排，下游指标退化 |
| 换编译器后单 kernel 变快但整体变慢 | 同上，别只看单 kernel |

**判据**：凡是「用 `-mllvm` 调优的 kernel」，换编译器/换 flag 后
**必须跑端到端指标**，不能只比单 kernel 耗时。

### 未验证

- 这 10 个 flag 里具体哪几个影响数值（只知整体有影响，未逐个拆）
- 除 GDN 外还有哪些 kernel 用了 aicc-only flag
- aicc 与 hipcc 在这条 kernel 上的单次耗时差异（未测）
