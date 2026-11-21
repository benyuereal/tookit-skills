# 整引擎链接 unblocker：mma.cuh no-op 化 + 链接期 undefined 符号 stub

2026-08 NInfer→DCU（gfx936 / DTK 26.04）实测。目标：tensor-core kernel 太多、逐个重写不现实时，
**先让整引擎链接通过**（部分路径运行时 no-op/throw），再逐路径补真实现。

## A. 中枢 PTX 头 no-op 化（mma.cuh stub）

中枢头（`ops/common/mma.cuh`）集中了所有 `mma.sync`/`ldmatrix` PTX 内联汇编。DTK 发不了这些 PTX
（mai-insts 被拒，见 mfma-availability.md）→ 所有 include 它的 .cu 全编译失败。

**手法**：保留每个 PTX 函数的**签名**，函数体改成 **no-op**（空实现 / 写 0）。
- 效果：所有 include 它的文件**都能编译**，MMA kernel 也能编译（运行时不计算）。
- 一次改一个头，覆盖几十个 .cu（NInfer 实测 mma stub 后 ops 编译 53→77 pass）。

⚠️ **no-op ≠ 正确**：MMA 路径运行时**不崩溃但静默出垃圾值**（不写输出/写 0）。
no-op 化只用于"让链接过 + 跑通不碰 MMA 的路径"；**碰 MMA 的路径必须换真 SIMT 实现**，否则结果错但不报错。

配套 DTK API 差异（MMA kernel 的 launch 代码）：
- `hipFuncSetAttribute(kernel_name, attr, val)`：DTK 要 `const void*`，直接传模板 kernel 名不行 → 加 `(const void*)` cast。
- `hipLaunchKernelEx`：DTK 签名是 `hipLaunchKernelEx(const hipDrvLaunchConfig_t*, ...)`，和 CUDA 的 `cudaLaunchKernelEx(&config, kernel, args...)` 不同。cooperative launch 路径要么改 cast，要么函数体改 throw（若目标模型不走该路径）。

## B. 链接期 undefined 符号 stub

砍掉 NVFP4/FP8/MoE/vision/media 等路径后，它们的 launcher 符号仍被 dispatch/plan 的 .cpp 引用
→ 链接报 undefined。逐个重写不现实，**写一个 stub .cpp 定义这些符号（运行时 throw）**。

### 步骤
1. **一次拿全 undefined**（别分批）：
   ```bash
   hipcc ... $OBJS -o app -Wl,--error-limit=0 2> link.err
   grep -oE "undefined symbol: [^ ]+" link.err | sed 's/undefined symbol: //' | sort -u
   ```
   默认 lld 只报前 20 个就停，`--error-limit=0` 拿全量（NInfer 实测一次拿到 40 个）。

2. **从 .h 自动提取签名**（Python 脚本，别手抄）：
   - 对每个 undefined 符号，在 `src/` + `include/` 的 .h 里正则找 `ret name(`。
   - 配对括号取完整参数列表，生成 `name(...) { throw std::logic_error("..."); }`。
   - 坑：
     - 声明带默认参数（`= {}`）→ 定义里**去掉默认值**（否则 `redefinition of default argument`）。
     - 跨 namespace 符号（`ninfer::media::decode::x`）→ 按 namespace 分组包 `namespace ... { }`。
     - 签名用到的类型（`PagedKVBatchLayerView`/`SwaPlan`/`Fp8A8Workspace`/`W8PairScheduleId` 等）→ include 对应头（`grep -rln "struct SwaPlan" src/` 找定义处）。
     - 提取时别多/少闭括号：参数列表本身含 `)`，拼接时别重复追加。

3. **重编 stub + 重链接**，循环直到 LINK_OK。

### 两个必踩的坑
- **★stale .o**：stub .cpp 改了但没重编，链接器用旧 .o（不含新符号）→ 还是 undefined。
  每次改 stub 后**先 `rm` 旧 .o 再重编**，并 `nm -C stub.o | grep <sym>` 确认符号真在 .o 里。
- **★duplicate symbol**：同一 .cpp 被两个构建脚本各编一次（build/core 和 build/cpp 都编 `src/core/*.cpp`）
  → 链接报 duplicate symbol。**通用去重**：链接前，任何 .o 若 build/core 里已有同名 .o 就跳过。

### C 文件
third_party 纯 C 库（utf8proc.c 等）用 `gcc -O2 -c` 编，**别用 hipcc**
（hipcc 对 .c 报 `invalid argument '-std=c++20' not allowed with 'C'`）。

## 验证
- `nm -C stub.o | grep -c ' T '` = stub 定义的符号数。
- 链接 `LINK_OK` + 二进制生成。
- 跑起来：不碰 stub 路径的 forward 正常；碰 stub 路径的会 throw（预期，后续补真实现）。
