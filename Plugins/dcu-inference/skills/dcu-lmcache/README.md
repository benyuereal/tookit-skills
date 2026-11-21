# dcu-lmcache — 海光 DCU LMCache KV Cache 卸载优化

LMCache 在海光 DCU 上做 KV Cache 卸载（offload）优化 vLLM 推理的方法论与实战：三级缓存架构、CPU/NVMe offload 选型、完整配置参数、DCU 部署编译、推理启动脚本、GDS/xDS 待研究方案。

## 目录

```
dcu-lmcache/
├── SKILL.md              # agent 向：完整方法 + 参数表 + 脚本引用
└── scripts/
    ├── lmcache_compile.sh        # LMCache 海光 DCU 编译安装（BUILD_WITH_HIP=1）
    ├── qwen3_32b_serve.sh       # Qwen3-32B 4 卡推理启动（NUMA 绑核 + LMCACHE_*）
    ├── glm47_w8a8_serve.sh      # GLM4.7-W8A8 8 卡推理启动（slimquant_marlin）
    ├── lmcache_config_cpu.yaml  # CPU offload 配置
    └── lmcache_config_nvme.yaml # NVMe offload 配置
```

## 快速用法 / 要点

- 长上下文/多轮/高并发场景，用 LMCache 把 KV 卸载到 CPU/NVMe，降 TTFT、提吞吐
- 小模型选 CPU offload，千亿模型选 NVMe offload
- DCU 编译关键：`BUILD_WITH_HIP=1` + `PYTORCH_ROCM_ARCH=gfx936` + `CXX=hipcc`
- 启动必设：`VLLM_NUMA_BIND=1` 绑核、`PYTHONHASHSEED=0`（KV 缓存键一致）、`kv-transfer-config` 接 `LMCacheConnectorV1`
- 性能数字仅收录主文档已文字化结论；明细 ksheet 权限受限不编造

> 详细内容（配置表、部署脚本、GDS 已知坑、溯源纪律）见 `SKILL.md`。
