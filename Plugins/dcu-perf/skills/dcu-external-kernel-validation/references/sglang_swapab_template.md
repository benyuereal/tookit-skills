# SGLang DCU 后端 patch 模板（CSR 布局 swap-AB split-K decode kernel）

> 来源：2026-10-09 <device-B> gfx936 实测案例的最终版（已清理 probe/调试代码）。用于 SGLang triton 后端的 decode attention 替换。
>
> **结论先行**：本模板在 SGLang 0.5.18 DCU 版上 E2E 是**负收益**（SGLang 的 `decode_attention_fwd_grouped` 已内置 GQA 打包 + stage 拆分，跑不过它）。保留它作为：① CSR 布局 kernel 的写法参考 ② sitecustomize 注入法的完整模板 ③ 负收益验证案例。

## 关键适配点（相对 vLLM block_table 版）

- SGLang KV 是 **CSR 格式**：`kv_indptr (B+1,)` + `kv_indices (total,)`，直接索引扁平 token 池，比 vLLM 的 block_table 少一层块间接（`tok = kv_indices[k0 + n]`）
- **HND 4D 布局**：K 是 `(pages, H, ps, D)`、V 是 `(pages, H, D, ps)`（shape 不同是**正常的**，gate 里只能比 ndim 不能比 shape）；ps=1 时 per-token 步长 = stride(0)、per-head 步长 = stride(1)，和 3D 布局偏移等价
- **decode 实际 batch 是 CUDA graph padding 值**（单请求也跑 B=8），L=0 的 padding 请求必须有显式零输出分支（combine 里的 `if L == 0` 早退）
- **patch 注入**：SGLang scheduler 是 spawn 子进程，import-time patch 只进主进程。用 `sitecustomize.py` + `PYTHONPATH=<dir>` + 环境变量门控（`SGLAB_PATCH=1`），所有 python 进程启动时都会执行
- **graph-safe**：workspace 在 `install_patch` 时一次性预分配（worst-case 尺寸），`_run` 热路径零分配、零 `.item()`
- **k_scale/v_scale 是 Python float（1.0）不是 tensor**，gate 要兼容两种（`_scale_ok`）

## 用法

```
cp sgl_swapab.py sitecustomize.py <dir>/   # 两个文件同目录
cd <dir> && PYTHONPATH=. SGLAB_PATCH=1 python bench.py
```

## 验证 patch 生效的方法

gate 不过时别猜——打印前 N 次调用的真实参数一次定位（本案例 gate 连错三次：buffer ndim 4≠3、scale 类型、shape 比较）。加一段：

```python
if _CALLS["n"] < 5:
    print(f"[probe] q{tuple(q.shape)} kbuf{tuple(k_buffer.shape)} "
          f"ndim={k_buffer.ndim} ps={ps} k_scale={k_scale!r} v_scale={v_scale!r} "
          f"OK={ok}", flush=True)
```
