# dcu-hicache — SGLang HiCache 三层缓存 DCU 部署

SGLang **内置** HiCache（hierarchical cache）在海光 DCU 上做三层 KV/Mamba 缓存卸载（L1 GPU 显存 / L2 host 内存 / L3 磁盘 file 后端）的部署、补丁与验证。缓解 HBM 显存墙、降长前缀 TTFT、提并发吞吐。

> **与 `dcu-lmcache` 的区别**：LMCache 是独立 pip 库（vLLM/SGLang 都接，需编译安装）；HiCache 是 SGLang 内置（`--enable-hierarchical-cache` 开关）。本 skill 只讲 SGLang 原生 HiCache 路线，两者不要混。

## 目录

```
dcu-hicache/
├── SKILL.md                 # agent 向：补丁 + 启动配置 + 坑 + 验证
├── references/hicache_deploy.md   # 补丁 anchor 行号、错误签名与验证命令
├── references/hicache-dsv4-deepseek-multiturn.md  # DSV4+DSPARK 多轮变长场景（完整脚本/patch/容量公式/性能对比）
└── templates/hicache_adapt_dcu.sh # 一键适配脚本（5 步幂等）
```

## 快速用法 / 要点

- **必打两个补丁**（DCU 版 SGLang 0.5.12）：Bug B（`memory_pool_host.py` 分页布局 D2H/H2D 回退）+ Bug C（`hi_mamba_radix_cache.py` load_back protect/release）。镜像重建后需重打。
  - ⚠ Bug B 用 `def v_buffer` property（全文件唯一）做 anchor 插入，**不要**用 `def load_to_device_per_layer(`——它在文件里出现 9 次，第一次在基类 decorator 处，插错会让 `_dcu_paged_device` 变抽象方法、启动崩。
- **容器必带 `--security-opt seccomp=unconfined`**（TP≥2 必需，否则 `mbind` 被拦截启不动）+ 必挂 `/opt/hyhal:ro`。
- **启动关键**：`--numa-node 0 1 2 3` `--enable-hierarchical-cache --hicache-ratio 1` `--hicache-io-backend direct` `--hicache-storage-backend file --hicache-storage-prefetch-policy timeout` `--attention-backend fa3 --page-size 64`。
  - L3 目录用环境变量 `export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=<路径>`（**不是** `--hicache-storage-path`，该参数在 0.5.12 不存在）。
  - 超模型派生 context_length 需 `export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`。
- **`--hicache-ratio` 按节点内存格局选**：node 空闲多可 ratio 2，被占则 ratio 1（单 rank host 池减半，稳定优先）。
- **Mamba 混合模型**（如 Qwen3.8-27B）三层命中"错位"是回载机制的正常行为，看 **fact_ok 全 True** 而非命中层细节。
- **DeepSeek-V4（DSV4）+ DSPARK 多轮变长场景**（纯 KV/MLA 模型，8× BW1100，SGLang 0.5.15 安装版）：`--hicache-io-backend direct` + `--hicache-mem-layout layer_first` 必须；`--hicache-size` 不支持 DSV4 只能用 `--hicache-ratio`；主动 `--max-total-tokens` 压池制造驱逐是让 HiCache 有意义的前置；**HSA stream 卡死**（宿主池初始化 `alloc_mmap`+register 坏 DMA/HSA → 改 `torch.empty(device="cpu")`）是此场景最严重 bug。详见 `references/hicache-dsv4-deepseek-multiturn.md`。
- 不存性能基线/逐行日志；明细见 `SKILL.md`。

> 详细内容（补丁、启动配置、seccomp/NUMA/SSH 坑、验证方法）见 `SKILL.md`。
