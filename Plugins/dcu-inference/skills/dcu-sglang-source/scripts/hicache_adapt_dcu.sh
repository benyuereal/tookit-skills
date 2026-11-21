#!/bin/bash
# =============================================================================
# hicache_adapt_dcu.sh — SGLang HiCache 在 DCU 上的一键适配脚本
#
# 功能：预检 → 建容器 → 打补丁+验证(Bug B/C) → 起服务 → 验服务
# 特性：幂等（可重复执行，已完成的步骤自动跳过）；任一步失败即停并报错
# 用法：
#   1) 修改下方 [配置区] 的变量（镜像/模型/卡数/端口等）
#   2) bash hicache_adapt_dcu.sh            # 全流程（幂等：容器已存在则跳过）
#   3) bash hicache_adapt_dcu.sh --rebuild  # 强制重建容器（删旧容器+全新创建，防环境污染）
#   4) bash hicache_adapt_dcu.sh --step 2   # 只跑第 2 步（打补丁+验证）
#   5) bash hicache_adapt_dcu.sh --dry-run  # 只打印将执行的动作，不实际执行
#
# 依赖：宿主机已装 docker + sshpass 不需要（脚本在宿主机本地跑）
# =============================================================================
set -euo pipefail

# ============================ [配置区] 按需修改 ============================
IMAGE="harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620"
MODEL_PATH="/public/models/Qwen3.8-27B"          # 模型目录（宿主机路径，会被挂进容器）
SERVED_NAME="Qwen3.8-27B"
TP_SIZE=4                                        # 张量并行卡数
GPUS="0,1,2,3"                                   # 用哪几张卡（HIP_VISIBLE_DEVICES）
NUMA_NODES="0 1 2 3"                             # 按卡数分散到不同 NUMA node
MEM_FRACTION=0.83                                # TP=2 用 0.5，TP=4 用 0.83
HICACHE_RATIO=1                                  # 节点内存被占用用 1；内存充足可用 2
L3_PATH="/public/hicache_l3"                     # L3 磁盘缓存目录
PORT=8008
CONTAINER_NAME="<container>"
SHM_SIZE="256g"
# 可选：rope override（Qwen3.8-27B 长上下文需要；其他模型留空即可）
ROPE_OVERRIDE='{"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}'
CONTEXT_LENGTH=1000000
# ============================================================================

# ---- 参数解析 ----
DRY_RUN=0
ONLY_STEP=0
REBUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --rebuild) REBUILD=1; shift ;;
    --step) shift; ONLY_STEP="${1:-0}"; shift ;;
    *) shift ;;
  esac
done

log()  { echo -e "\n\033[1;32m[STEP $1]\033[0m $2"; }
info() { echo -e "\033[0;36m  ·\033[0m $*"; }
die()  { echo -e "\033[1;31m  ✗ $*\033[0m" >&2; exit 1; }
run()  {  # run <desc> <cmd...>  —— dry-run 时只打印
  if [ "$DRY_RUN" = 1 ]; then echo -e "\033[0;33m  [dry-run]\033[0m $*"; else eval "$*"; fi
}

# 判断某步是否需要执行
do_step() { [ "$ONLY_STEP" = 0 ] || [ "$ONLY_STEP" = "$1" ]; }

# ============================ 步骤 0：预检 ============================
if do_step 0; then
log 0 "预检（docker / 模型 / DCU 可见性）"
command -v docker >/dev/null 2>&1 || die "未找到 docker，请先安装"
[ -d "$MODEL_PATH" ] || die "模型目录不存在：$MODEL_PATH"
info "镜像：$IMAGE"
info "模型：$MODEL_PATH"
info "TP=$TP_SIZE  卡=$GPUS  端口=$PORT  容器=$CONTAINER_NAME"
# 检查宿主机 DCU 设备节点
for dev in /dev/kfd /dev/dri; do
  [ -e "$dev" ] || die "缺少 DCU 设备节点 $dev（驱动未装？）"
done
info "DCU 设备节点 OK"
fi

# ============================ 步骤 1：建容器 ============================
if do_step 1; then
log 1 "创建容器（含 --security-opt seccomp=unconfined，必须）"
if [ "$REBUILD" = 1 ]; then
  info "--rebuild：删除旧容器 $CONTAINER_NAME 并全新创建（防环境污染）"
  run "docker rm -f $CONTAINER_NAME"
  run "docker run -d --name $CONTAINER_NAME \
    --network host --ipc host --shm-size $SHM_SIZE \
    --security-opt seccomp=unconfined \
    --cap-add SYS_PTRACE --group-add video \
    -v /opt/hyhal:/opt/hyhal:ro -v /public:/public -v /data0:/data0 \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    $IMAGE sleep infinity"
  info "容器已重建（全新环境，sglang 文件为镜像原始状态，后续步骤会重新打补丁）"
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  info "容器 $CONTAINER_NAME 已存在，跳过创建（如需干净环境：bash $0 --rebuild）"
else
  run "docker run -d --name $CONTAINER_NAME \
    --network host --ipc host --shm-size $SHM_SIZE \
    --security-opt seccomp=unconfined \
    --cap-add SYS_PTRACE --group-add video \
    -v /opt/hyhal:/opt/hyhal:ro -v /public:/public -v /data0:/data0 \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    $IMAGE sleep infinity"
  info "容器已创建（sleep infinity 保活，服务用 docker exec 起）"
fi
fi

# ============================ 步骤 2：打补丁 + 验证（Bug B + Bug C） ============================
if do_step 2; then
log 2 "应用代码补丁（Bug B 分页布局回退 + Bug C load_back protect/release，幂等）并验证"
# 补丁脚本通过 heredoc 传入容器内 python 执行；幂等（已打则跳过）
PATCH_PY='
import sys
# ===== Bug B: memory_pool_host.py (direct 后端分页 KV 布局) =====
p = "/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/memory_pool_host.py"
s = open(p).read()
if "_dcu_d2h_paged" in s:
    print("Bug B patch already applied, skipping")
else:
    mha_anchor = """    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,"""
    assert s.count(mha_anchor) == 1, "MHA anchor not unique"
    methods = """    # ---- DCU paged-layout fallback for the \"direct\" io backend ----
    def _dcu_paged_device(self, device_pool):
        try:
            return device_pool.k_buffer[0].dim() == 4
        except Exception:
            return False

    def _dcu_d2h_paged(self, device_pool, host_indices, device_indices):
        ps = self.page_size
        dev = device_pool.k_buffer[0].device
        pg = (device_indices // ps).to(dev)
        off = (device_indices % ps).to(dev)
        hidx = host_indices.to(self.device)
        for lid in range(self.layer_num):
            self.k_data_refs[lid][hidx] = device_pool.k_buffer[lid][
                pg, :, off, :
            ].to(self.device)
            self.v_data_refs[lid][hidx] = device_pool.v_buffer[lid][
                pg, :, :, off
            ].to(self.device)

    def _dcu_h2d_paged(self, device_pool, host_indices, device_indices, layer_id):
        ps = self.page_size
        dev = device_pool.k_buffer[layer_id].device
        pg = (device_indices // ps).to(dev)
        off = (device_indices % ps).to(dev)
        hidx = host_indices.to(self.device)
        device_pool.k_buffer[layer_id][pg, :, off, :] = self.k_data_refs[layer_id][
            hidx
        ].to(dev)
        device_pool.v_buffer[layer_id][pg, :, :, off] = self.v_data_refs[layer_id][
            hidx
        ].to(dev)

"""
    s = s.replace(mha_anchor, """    @property
    def v_buffer(self):
        return self.kv_buffer[1]

""" + methods + """    def load_to_device_per_layer(
        self,
        device_pool,""", 1)
    h2d_old = """        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],"""
    h2d_new = """        elif io_backend == "direct":
            if self.layout == "layer_first" and self._dcu_paged_device(device_pool):
                self._dcu_h2d_paged(
                    device_pool, host_indices, device_indices, layer_id
                )
                return
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],"""
    assert h2d_old in s, "H2D call site not found"
    s = s.replace(h2d_old, h2d_new, 1)
    d2h_old = """        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,"""
    d2h_new = """        elif io_backend == "direct":
            if self.layout == "layer_first" and self._dcu_paged_device(device_pool):
                self._dcu_d2h_paged(device_pool, host_indices, device_indices)
                return
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,"""
    assert d2h_old in s, "D2H call site not found"
    s = s.replace(d2h_old, d2h_new, 1)
    open(p, "w").write(s)
    print("Bug B patch applied OK")

# ===== Bug C: hi_mamba_radix_cache.py (load_back protect_host/release_host) =====
p2 = "/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/hi_mamba_radix_cache.py"
s2 = open(p2).read()
if "n.protect_host()" in s2:
    print("Bug C patch already applied, skipping")
else:
    r1_old = """        else:
            ancestor_node = node

        mamba_restore_nodes = []"""
    r1_new = """        else:
            ancestor_node = node

        for n in nodes_to_load:
            n.protect_host()

        mamba_restore_nodes = []"""
    assert r1_old in s2, "R1 anchor not found"
    s2 = s2.replace(r1_old, r1_new, 1)
    r2_old = """            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancestor_node)
            return None"""
    r2_new = """            # skip loading back if the total size is too small or exceeding the memory quota
            for n in nodes_to_load:
                n.release_host()
            self.dec_lock_ref(ancestor_node)
            return None"""
    assert r2_old in s2, "R2 anchor not found"
    s2 = s2.replace(r2_old, r2_new, 1)
    r3_old = """        self.dec_lock_ref(ancestor_node)
        if full_device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None"""
    r3_new = """        self.dec_lock_ref(ancestor_node)
        if full_device_indices is None:
            # no sufficient GPU memory to load back KV caches
            for n in nodes_to_load:
                n.release_host()
            return None"""
    assert r3_old in s2, "R3 anchor not found"
    s2 = s2.replace(r3_old, r3_new, 1)
    r4_old = """        self._update_leaf_status(ancestor_node)

        self.inc_lock_ref(last_hit_node)"""
    r4_new = """        self._update_leaf_status(ancestor_node)

        for n in nodes_to_load:
            n.release_host()
        self.inc_lock_ref(last_hit_node)"""
    assert r4_old in s2, "R4 anchor not found"
    s2 = s2.replace(r4_old, r4_new, 1)
    open(p2, "w").write(s2)
    print("Bug C patch applied OK")
print("All patches applied")
'
# 先备份（仅首次，避免覆盖已备份的原始文件）
run "docker exec $CONTAINER_NAME bash -c '
  for f in memory_pool_host.py hi_mamba_radix_cache.py; do
    p=/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/\$f
    [ -f \$p.bak_dcu ] || cp \$p \$p.bak_dcu
  done'"
# 写入补丁脚本到容器再执行（避免 heredoc 转义问题）
run "docker exec -i $CONTAINER_NAME bash -c 'cat > /tmp/_dcu_patch.py' <<'PYEOF'
$PATCH_PY
PYEOF"
run "docker exec $CONTAINER_NAME python3 /tmp/_dcu_patch.py"
# 立即验证补丁（用 ast 静态解析，不 import sglang，避免依赖 DCU 环境）
# 检查 _dcu_* 方法落在 MHATokenToKVPoolHost 类内，且类内无 abstractmethod
VERIFY_PY='
import ast
p = "/usr/local/lib/python3.10/dist-packages/sglang/srt/mem_cache/memory_pool_host.py"
tree = ast.parse(open(p).read())
cls = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "MHATokenToKVPoolHost":
        cls = node
        break
assert cls, "MHATokenToKVPoolHost class not found"
methods, abstract = set(), set()
for node in cls.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        methods.add(node.name)
        for dec in node.decorator_list:
            name = dec.id if isinstance(dec, ast.Name) else (dec.attr if isinstance(dec, ast.Attribute) else None)
            if name == "abstractmethod":
                abstract.add(node.name)
dcu = {m for m in methods if m.startswith("_dcu_")}
print("_dcu_ methods:", sorted(dcu))
assert dcu == {"_dcu_d2h_paged","_dcu_h2d_paged","_dcu_paged_device"}, "Bug B methods missing or in wrong class"
print("abstract methods in class:", sorted(abstract))
assert not abstract, "abstract methods present -> patch may be in wrong class"
print("PATCH VERIFY OK")
'
run "docker exec -i $CONTAINER_NAME bash -c 'cat > /tmp/_dcu_verify.py' <<'VEOF'
$VERIFY_PY
VEOF"
run "docker exec $CONTAINER_NAME python3 /tmp/_dcu_verify.py"
info "补丁应用并验证完成"
fi

# ============================ 步骤 3：起服务 ============================
if do_step 3; then
log 3 "启动 SGLang HiCache 服务（先清理旧进程）"
# 清理旧服务进程
# 注意：pkill -f 的 pattern 用方括号技巧 [s]glang.launch_server，
# 否则 bash -c 自己的命令行里含 "sglang.launch_server" 字符串会被 pkill 匹配到，
# 导致 pkill 杀掉执行它的那个 bash 进程自己（自杀，返回 143，set -e 退出脚本）。
# 正则 [s]glang 匹配字面 "sglang"，但命令行里的 "[s]glang" 不匹配（[s] 要求字面 s）。
run "docker exec $CONTAINER_NAME bash -c 'pkill -f \"[s]glang.launch_server\" 2>/dev/null; sleep 3; true'"
# 生成启动脚本
SERVE_SH='
#!/bin/bash
source /opt/dtk/env.sh
export ROCM_PATH=/opt/dtk
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HIP_VISIBLE_DEVICES=__GPUS__
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=__L3__
mkdir -p __L3__
MODEL_PATH=__MODEL__
python3 -u -m sglang.launch_server \
  --numa-node __NUMA__ --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 64 --kv-cache-dtype fp8_e5m2 \
  --model-path $MODEL_PATH --host 0.0.0.0 --port __PORT__ --served-model-name __NAME__ \
  --mm-attention-backend fa3 --enable-piecewise-cuda-graph \
  --tp-size __TP__ --pp-size 1 --chunked-prefill-size 8192 --watchdog-timeout 3000 \
  --attention-backend fa3 --page-size 64 --mem-fraction-static __MF__ \
  --keep-mm-feature-on-device --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-cache-report \
  __ROPE_FLAG__ \
  --enable-hierarchical-cache --hicache-ratio __RATIO__ \
  --hicache-io-backend direct \
  --hicache-mem-layout layer_first --hicache-write-policy write_through \
  --hicache-storage-backend file --hicache-storage-prefetch-policy timeout
'
# 计算 ROPE 参数（JSON 含空格/花括号/引号，用单引号包裹避免 word split）
if [ -n "$ROPE_OVERRIDE" ]; then
  ROPE_FLAG="--json-model-override-args '$ROPE_OVERRIDE' --context-length $CONTEXT_LENGTH"
else
  ROPE_FLAG=""
fi
# 替换占位符
SERVE_SH="${SERVE_SH//__GPUS__/$GPUS}"
SERVE_SH="${SERVE_SH//__MODEL__/$MODEL_PATH}"
SERVE_SH="${SERVE_SH//__ROPE_FLAG__/$ROPE_FLAG}"
SERVE_SH="${SERVE_SH//__NUMA__/$NUMA_NODES}"
SERVE_SH="${SERVE_SH//__PORT__/$PORT}"
SERVE_SH="${SERVE_SH//__NAME__/$SERVED_NAME}"
SERVE_SH="${SERVE_SH//__TP__/$TP_SIZE}"
SERVE_SH="${SERVE_SH//__MF__/$MEM_FRACTION}"
SERVE_SH="${SERVE_SH//__RATIO__/$HICACHE_RATIO}"
SERVE_SH="${SERVE_SH//__L3__/$L3_PATH}"
# 写入容器
run "docker exec -i $CONTAINER_NAME bash -c 'cat > /root/serve_hicache_dcu.sh' <<'SHEOF'
$SERVE_SH
SHEOF"
run "docker exec $CONTAINER_NAME bash -c 'chmod +x /root/serve_hicache_dcu.sh'"
# 后台启动（</dev/null 断开管道，避免 SSH 会话被占住）
run "docker exec -d $CONTAINER_NAME bash -c 'cd /root && nohup bash serve_hicache_dcu.sh > /root/serve.log 2>&1 < /dev/null'"
info "服务已后台启动，日志：docker exec $CONTAINER_NAME tail -f /root/serve.log"
fi

# ============================ 步骤 4：验服务 ============================
if do_step 4; then
log 4 "等待服务就绪并验证（health + 三层缓存命中信号）"
info "等待 health=200（最多 600s，模型加载需数分钟）..."
if [ "$DRY_RUN" = 0 ]; then
  ok=0
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then ok=1; break; fi
    # 服务崩了（进程没了）就提前退出
    if ! docker exec "$CONTAINER_NAME" pgrep -f sglang.launch_server >/dev/null 2>&1; then
      echo "  ✗ 服务进程已退出，查看日志：docker exec $CONTAINER_NAME tail -50 /root/serve.log" >&2
      exit 1
    fi
    sleep 5
  done
  [ "$ok" = 1 ] || die "600s 内服务未就绪，查看日志：docker exec $CONTAINER_NAME tail -50 /root/serve.log"
  info "health=200 ✓"
  # 发一个带缓存明细的请求，验证三层命中信号
  info "验证 cached_tokens_details（三层命中信号）..."
  curl -s "http://127.0.0.1:$PORT/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"Hello, DCU HiCache! 请回复 FACT-TEST-OMEGA\", \"sampling_params\": {\"max_new_tokens\": 16}, \"return_cached_tokens_details\": true}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  cached_tokens_details:", d.get("sglext",{}).get("cached_tokens_details","(无)"))'
  info "验证完成。三层命中判读：device>0=L1 / host>0=L2 / storage>0=L3"
else
  info "[dry-run] 跳过实际等待"
fi
fi

echo -e "\n\033[1;32m全部步骤完成。\033[0m"
echo "  服务地址：http://<host>:$PORT"
echo "  查看日志：docker exec $CONTAINER_NAME tail -f /root/serve.log"
echo "  停止服务：docker exec $CONTAINER_NAME pkill -f \"[s]glang.launch_server\""
