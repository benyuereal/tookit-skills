# W8G32（int8）转换具体细节 — Qwen3.8-27B

> 2026-08-25 会话沉淀。W8G32_F16S = 8-bit 权重 + group-32 + fp16 scale，引擎已有格式码/布局/kernel/dispatch。

## 背景

- 用户拍板：从原生 bf16 直接转 W8G32（int8），不走 Q5→int8 的 hack。精度更高（8 bit vs 5 bit）、步骤更少、引擎现成支持。
- 27B text core 格式分配（原）：
  - Q4：mlp/gate_up (34816×5120)、gdn/query_key (4096×5120)、attention/query_key (7168×5120)
  - Q5：mlp/down (5120×17408)、gdn/value_z (12288×5120)、gdn/output (5120×6144)、attention/gate_value (7168×5120)、attention/output (5120×4096)
  - W8：embedding、output_head（已是 W8）
- 增量验证策略：先只转 linear_add 的 3 个 tensor（down_proj 5120×17408、gdn/output 5120×6144、attn/output 5120×4096），零 wrapper 改动（linear_add 的 W8 路径已存在），先验证整条链路。

## W8G32 布局（row-split-k128-v1）

- group-32，K pad 到 128，256 对齐。
- codes plane：`rows × groups_per_row × 32` 字节（group-major，1 字节/int8）。
- scale plane：`rows × groups_per_row × 2` 字节（fp16）。
- 布局计算：`src/artifact/storage_layouts.cpp`。

## W8 kernel 现状

- W8 SIMT kernel：N 维并行（`grid=div_up(n,8)`）+ kStages=2 + 向量化（`load_vec<uint2>`）+ int8 cvt（dequant 比 Q5 轻：无位提取/符号扩展/high plane）。
- W8 dispatch M=1 走 `launch_w8_simt_r8_c4`（SIMT，不是 MMA stub）。
- W8 linear_add 有 3 个 kernel：`w8_linear_add_gemm_splitk.cu`、`w8_linear_add_gemm_mma.cu`、`w8_linear_add_gemm_simt.cu`。
- **SIMT kernel 形状通用**（`grid=div_up(rows)×div_up(cols)`），但 plan 准入 + wrapper 校验硬编码 MTP 形状（rows=2048, k∈{4096,6144}）。

## 4 层 patch 内容（linear_add 扩到 rows=5120）

### 层 4a：`src/ops/linear_add/w8/w8_linear_add_plan.cpp` — admits

```cpp
// 原
bool w8_linear_add_admits(const W8LinearAddProblem& problem) noexcept {
    return problem.rows == 2048 && (problem.k == 4096 || problem.k == 6144) &&
           problem.padded_k == problem.k && problem.cols >= 1;
}
// 改
bool w8_linear_add_admits(const W8LinearAddProblem& problem) noexcept {
    if (problem.rows == 2048) {
        return (problem.k == 4096 || problem.k == 6144) &&
               problem.padded_k == problem.k && problem.cols >= 1;
    }
    // 主模型形状 (rows=5120)：走通用 SIMT 路径
    if (problem.rows == 5120) {
        return (problem.k == 4096 || problem.k == 6144 || problem.k == 17408) &&
               problem.padded_k == problem.k && problem.cols >= 1;
    }
    return false;
}
```

### 层 4b：同文件 — resolve

```cpp
// 原
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK4096Routes);
}
// 改
    if (problem.rows == 5120) {
        // 主模型形状：全部走形状通用的 SIMT kernel
        return {W8LinearAddScheduleId::SimtR8C4};
    }
    return problem.k == 6144 ? resolve_from(kK6144Routes) : resolve_from(kK4096Routes);
}
```

### 层 3：`src/ops/wrapper/linear_add.cpp` — 形状校验

```cpp
// 原
        if (w.n != 2048 || (w.k != 4096 && w.k != 6144)) {
            throw std::invalid_argument("linear_add: unsupported W8 shape");
        }
// 改
        if (!((w.n == 2048 && (w.k == 4096 || w.k == 6144)) ||
              (w.n == 5120 && (w.k == 4096 || w.k == 6144 || w.k == 17408)))) {
            throw std::invalid_argument("linear_add: unsupported W8 shape");
        }
```

### 层 2：`src/targets/qwen3_6_27b/impl/load/bindings.cpp` — 3 处 Q5→W8G32_F16S

attention/output、gdn/output、mlp/down 的期望格式从 Q5G64_F16S 改成 W8G32_F16S。

### 层 1：`tools/convert/qwen3_8_27b/inventory.py` — 格式分配

linear_add 的 3 个 tensor 从 Q5 改成 W8（增量验证只改这 3 个；全量转 W8 需改所有 Q4/Q5/Q6）。

## 转换命令

```bash
# CPU 转换（避免 GPU OOM，27B 模型 ~210s）
cd <local_ninfer_src>-hip
python3 tools/convert/qwen3_8_27b/convert.py --device cpu \
  --out /data1/out/qwen3_8_27b_w8.ninfer
# 输出：/data1/out/qwen3_8_27b_w8.ninfer (21.3GB) + .conversion.json
# 注意：输出在 /data1/out/，不是 /data1/models/
```

## 重编 + 重链

```bash
cd <local_ninfer_src>-hip
INC="-Iinclude -Isrc -Ithird_party -Ithird_party/utf8proc"
CXX="-O2 -std=c++20 --offload-arch=gfx936"
# 重编 4 个文件
for f in src/ops/linear_add/w8/w8_linear_add_plan.cpp \
         src/ops/wrapper/linear_add.cpp \
         src/targets/qwen3_6_27b/impl/load/bindings.cpp; do
  base=$(basename $f .cpp)
  /opt/dtk/bin/hipcc $CXX $INC $f -c -o /data1/build/ops/${base}.o
done
# 重链（精确排除：只排除刚重编文件的旧 src_ops_ 版本，别全删）
# 刚重编的 2 个文件的旧 src_ops_ 版本在 cpp/ 下：
#   /data1/build/cpp/src_ops_linear_add_w8_w8_linear_add_plan.o
#   /data1/build/cpp/src_ops_wrapper_linear_add.o
# 用精确排除链接脚本（见 /data1/do_link_precise.sh）：
#   - 遍历 core/ops/cpp 全部 .o + /tmp/gqa_prefill_simt_launcher.o
#   - 只跳过上面 2 个旧 src_ops_ 版本
#   - 其余全链接
# ⚠ 不能 `find /data1/build -name "src_ops_*" -delete`——有些符号只定义在 src_ops_* .o 里，全删会 undefined symbol。
```

## 踩坑记录

1. **GPU 转换 OOM**：27B 模型（52GB）+ 量化 buffer 超 64GB 卡显存。改用 `--device cpu`（主机 1.5TB 内存，量化计算不重，~210s）。
2. **形状白名单**：W8 的 plan 准入 + wrapper 校验硬编码 MTP 形状（rows=2048, k∈{4096,6144}），主模型 rows=5120 不匹配。SIMT kernel 本身形状通用，只需扩白名单路由到 SimtR8C4。
3. **target 契约**：bindings.cpp 硬编码每个 tensor 期望格式，转 W8 后必须同步改，否则 `tensor descriptor does not match target contract`。
4. **输出路径**：转换输出在 `/data1/out/`，不是 `/data1/models/`。
5. **CLI 参数**：`ninfer_cli <model.ninfer>` 是位置参数（不是 `--model`），`--max-new N`（不是 `--max-new-tokens`）。
6. **重链精确排除**：不能删/排除所有 `src_ops_*` .o（有些符号只定义在 src_ops_* .o 里，如 `q5_linear_add_capacity_workspace_bytes` 只在 `cpp/src_ops_linear_add_q5_q5_linear_add_plan.o`）。只排除刚重编文件的旧 src_ops_ 版本。用 `nm -C <.o> | grep "T <symbol>"` 查符号定义在哪个 .o。
7. **GQA launcher .o 容器重启丢**：`/tmp/gqa_prefill_simt_launcher.o` 容器重启后丢，需重编（见 SKILL.md 重编命令）。
8. **hy-smi 解析**：hy-smi 显示 VRAM%（第 6 列），不是空闲 GB。空闲 GB = 64 × (100 - VRAMPCT) / 100。解析：`hy-smi | awk '$1==4 {print $6}' | tr -d '%'`。

## 结果（2026-08-25）

- 4 层 patch 已应用 + 编译 PASS + **重链成功**（精确排除，218 .o，LINK_OK）。
- W8 模型跑通，**精度正确**（"2 + 2 = 4"）。
- **性能：W8（25% tensor）= 18.61 tok/s vs Q5 基线 20.86 tok/s = -11%**。
- 原因：W8 SimtR8C4 kernel 没做 u32 向量化 + 只转 25% tensor。int8 v2 原型（u32 向量化）比 Q5 GEMV 快 1.78x 且无损。
- **下一步**：给 W8 SimtR8C4 kernel 加 u32 向量化（参考 int8 v2 原型），再全量转 W8。

## 待办

- 给 W8 SimtR8C4 kernel 加 u32 向量化（参考 int8 v2 原型 `/tmp/gdn_probe/int8_gemv_v2.cu`）。
- 全量转 W8（所有 Q4/Q5/Q6）需扩展 linear_swiglu（34816×5120）、gdn_input_proj（4096×5120、12288×5120）、attn_input_proj（7168×5120）的 W8 形状白名单。
