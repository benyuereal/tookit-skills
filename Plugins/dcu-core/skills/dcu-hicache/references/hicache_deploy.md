# HiCache DCU 部署细节（2026-09 <device-A> 实测）

本文件是 `dcu-hicache` SKILL.md 的 session 级细节补充：补丁 anchor 行号、错误签名、验证命令、脚本结构。

## 1. 补丁 anchor 与行号（DCU 版 SGLang 0.5.12）

### Bug B — memory_pool_host.py
- 文件：`/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/memory_pool_host.py`
- 基类 `HostKVCache` 从 line 291 之前开始；其 `load_to_device_per_layer` 在 **line 222**（line 221 是 `@abc.abstractmethod`）。
- MHA 类 `MHATokenToKVPoolHost` 从 **line 291** 开始；其 `load_to_device_per_layer` 在 **line 397**。
- **正确 anchor**：`def v_buffer` property（**line 394**，全文件唯一，在 MHA 类内）。
- 错误 anchor：`def load_to_device_per_layer(`（出现 **9 次**，第一次在基类 line 222）。
- 打错后的错误签名：
  ```
  TypeError: Can't instantiate abstract class MHATokenToKVPoolHost
  with abstract method _dcu_paged_device
  ```
- 验证命令：
  ```bash
  # 方法落在 MHA 类内（line 406/412/426）
  grep -n "_dcu_paged_device" memory_pool_host.py
  # 无残留抽象方法
  python3 -c "import inspect; from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost as C; print([m for m in dir(C) if getattr(getattr(C,m,None),'__isabstractmethod__',False)])"
  # 应为 []
  ```

### Bug C — hi_mamba_radix_cache.py
- 文件：`/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/hi_mamba_radix_cache.py`
- load_back 路径：protect 1 处 + release 3 处。
- anchor 有 assert（插错会报错，不会静默打错）。
- 备份：`.bak_dcu`（打补丁前 `cp file file.bak_dcu`）。

## 2. 启动脚本（serve_hicache_dcu_tp4.sh 最终版）

```bash
#!/bin/bash
MODEL_PATH=/public/models/Qwen3.8-27B
# 性能环境变量（8 个，来自参考脚本）
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_MARLIN=1
export SGLANG_ENABLE_CAUSAL_CONV1D=1
# ... 线性注意力等
python3 -u -m sglang.launch_server \
  --numa-node 0 1 2 4 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 64 \
  --kv-cache-dtype fp8_e5m2 \
  --model-path $MODEL_PATH \
  --host 0.0.0.0 --port 8008 \
  --served-model-name Qwen3.8-27B \
  --mm-attention-backend fa3 \
  --enable-piecewise-cuda-graph \
  --tp-size 4 --pp-size 1 \
  --chunked-prefill-size 8192 \
  --watchdog-timeout 3000 \
  --attention-backend fa3 --page-size 64 \
  --mem-fraction-static 0.83 \
  --keep-mm-feature-on-device \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-cache-report \
  --json-model-override-args '{"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}' \
  --context-length 1000000 \
  --enable-hierarchical-cache --hicache-ratio 1 \
  --hicache-io-backend direct \
  --hicache-mem-layout layer_first \
  --hicache-write-policy write_through \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy timeout
# L3 磁盘目录用环境变量（--hicache-storage-path 在 sglang 0.5.12 不存在，写了启动即失败）
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/public/hicache_l3
mkdir -p /public/hicache_l3
# --context-length 超模型派生值时需放行
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
```

> 注意：`--numa-node 0 1 2 3`（ratio 1 下任何 node 都够，见 SKILL.md 坑 2）；`--hicache-ratio 1`（稳定值）；**无 `--max-total-tokens`**（会限死 KV 池）。

## 3. HiCache 初始化确认值（ratio 1 最终配置）

- GPU KV 池：2,622,144 tokens（每卡 20GB，K 10GB + V 10GB）
- L2 host KV cache：21.48 GB × 4 ranks
- L2 host Mamba cache：19.36 GB × 4 ranks
- L3 storage backend：'file'（HiCacheFile），路径 /public/hicache_l3
- enable_cache_report：True
- max_running_requests：100
- 显存占用：~53GB/64GB ≈ 84%（avail 10-11GB）

## 4. 验证脚本结构

### verify_l123.py（L1/L2/L3 三层缓存验证）
- Phase 1：fresh 请求（无缓存），看 `cached_tokens_details` 全 0
- Phase 2：同前缀重发，看哪层命中（device/host/storage）
- Phase 3：长前缀埋事实，驱逐后重问，fact_ok=True
- 注意：max_tokens 给够（≥64），推理模型 thinking 会吃掉小 max_tokens 的输出空间

### load_test.py（并发工作集压测）
- 16 并发 worker 各持独立 8000-token 前缀（工作集 128K）
- 三阶段：A 预热 / B 稳态 / C 压力
- 看缓存加速比 + 成功率
- 注意：`out = msg.get("content") or msg.get("reasoning_content") or ""`（reasoning-parser 在 max_tokens 小时把输出放进 reasoning_content 导致 content=None）

## 5. 诊断命令速查

```bash
# NUMA 策略 OOM 诊断
dmesg -T | grep -iE "oom|killed process" | tail -20
# 看 constraint=CONSTRAINT_MEMORY_POLICY + nodemask=X

# 各 NUMA node 空闲内存
for n in 0 1 2 3 4 5 6 7; do
  grep MemFree /sys/devices/system/node/node$n/meminfo 2>/dev/null
done

# seccomp mbind 诊断
grep -i "mbind" /var/log/hylog/*.log 2>/dev/null | tail -10

# L3 磁盘状态
ls /public/hicache_l3 | wc -l
du -sh /public/hicache_l3

# 服务状态
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8008/health
docker exec <c> ps aux | grep -E "sglang|launch_server" | grep -v grep | wc -l
```

## 6. 容器创建（<container>）

```bash
docker run -d --name <container> \
  --network host --ipc host --shm-size 256g \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE --group-add video \
  -v /opt/hyhal:/opt/hyhal:ro -v /public:/public -v /data0:/data0 \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620
```

- **DCU 设备节点必须用 `--device`（不是 `-v` 绑定挂载）**：`-v /dev/kfd` 绑定挂载字符设备时 cgroup 设备白名单没加，新容器打开报 EPERM（`Unable to open /dev/kfd read-write: Operation not permitted`）；`--device` 让 docker 自动加白名单。旧容器能用 `-v` 是创建时 cgroup 配置不同，rebuild 全新容器后必踩。
- 4 卡场景用 `HIP_VISIBLE_DEVICES=0,1,2,3` 指定卡
- 补丁打在容器内 `dist-packages`，镜像重建后需重打
- **防环境污染**：怀疑容器脏状态（bashrc 塞了 DTK 变量、cgroup 配置不同、之前手动操作过）时用 `bash hicache_adapt_dcu.sh --rebuild` 强制 `docker rm -f` + 全新 `docker run`。旧容器的环境污染会掩盖脚本自身的 bug（2026-09-03 干净环境 rebuild 抓到 4 个 bug：参数解析 / 验证依赖 DCU 环境 / DCU 设备 EPERM / pkill 自杀，详见 `dcu-sglang-source` HiCache 章节）。
