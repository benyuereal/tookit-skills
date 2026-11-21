# dcu-sglang-source — 海光 DCU 版 SGLang 源码结构与适配索引

> agent 向规范见 [`SKILL.md`](./SKILL.md)。本文件面向使用者。

## 它解决什么

在 **海光 DCU 版 SGLang** 上做算子/backend 开发与适配时，快速定位：
- SGLang 的三套 kernel 扩展体系（Triton / sgl-kernel AOT / jit_kernel JIT）
- attention backend 注册机制（`@register_attention_backend` / `AttentionBackend` 基类）
- **海光实测适配点**（基于 <device-A> 设备 DTK26.04 sglang 0.5.12 容器实测源码）：`dcu_mla_backend.py`（海光版权）、`dsv4/` 子目录（含 TileLang kernel）、`--attention-backend dcu_mla` 开关、`is_dcu()` 分发模式
- profiling / patch 纪律（与 `dcu-op-optimization` 闭环一致）

## 与 vLLM skill 的差异（重要）

SGLang **没有** vLLM 那种 `_custom_ops.py` 统一 C++ 注册体系，而是 **attention backend 插件化 + Triton 为主 + sgl-kernel/jit_kernel 下沉**。平台判断函数是 **`is_dcu()`**（不是 vLLM 的 `is_rocm`），写 DCU 分支时认准它。

## 实测关键事实（<device-A> 容器，sglang 0.5.12+g9a3a3e5f9）

- `srt/layers/attention/dcu_mla_backend.py` —— `Copyright 2026 Hygon`，海光官方 MLA backend
- `srt/layers/attention/dsv4/` —— DSV4 完整适配（compressor/indexer/flash_mla_triton/tilelang_kernel/sparse_prefill）
- `server_args.py` 原生支持 `DCU_ATTENTION_BACKEND_CHOICES={"dcu_mla",...}`

## 快速使用

```bash
# 起 DTK26.04 sglang 容器（必挂 hyhal）
docker run -dit --name sgl --device=/dev/kfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro <sglang-dtk26.04-img> bash -c "sleep infinity"
SGDIR=$(docker exec sgl python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))")
docker exec sgl grep -rlE "is_dcu" $SGDIR/srt | head   # 看 DCU 适配入口
```

## 离线参考代码

海光 SGLang 0.5.12(+g9a3a3e5f9) fork 的 DCU 专属适配源码位于 DTK26.04 镜像内 `/usr/local/lib/python3.10/dist-packages/sglang`（含 `dcu_mla_backend.py` 海光官方 MLA backend、`dsv4/` 子目录含 `tilelang_kernel.py`、`forward_mla_fused_rope_rocm.py`），**未随本包分发**；需要时从容器 `docker cp` 自行抽取。详见 SKILL.md「离线参考代码」节。

## DSpark 移植参考

**DSpark（置信度驱动投机解码）上游已集成（PR #30261），但海光 fork 0.5.12 不支持**（grep `dspark` 0 命中）。上游源码快照 [`scripts/dspark_upstream/`](./scripts/dspark_upstream/)（sglang main，23 文件，py_compile 通过）+ 算法级验证 [`scripts/dspark_sandbox_verify.py`](./scripts/dspark_sandbox_verify.py)（纯逻辑跑通）+ 完整移植指南 [`DSPARK_PORTING.md`](./DSPARK_PORTING.md)（含依赖树 / 海光 fork 缺失模块 / is_dcu 适配点 / 版本错配警告）。**移植需海光 fork 先 rebase 上游 main 对应 commit，非加桩可跑。**

## 适用版本

海光 SGLang 以 **fork 构建**推进（版本号带 `+<commit>`），不按大版本拆独立 skill。跨版本请先 `python3 -c "import sglang;print(sglang.__version__)"` 确认 fork 状态。
