

import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

try:
    from .model_analyzer import MODEL_CONFIG_MAP, _PROJECT_ROOT, model_analyzer
except ImportError:
    from model_analyzer import MODEL_CONFIG_MAP, _PROJECT_ROOT, model_analyzer



@dataclass
class PDSweepConfig:
    """PD 分离遍历的输入配置"""
    total_nodes: int
    gpus_per_node: int = 8
    nvlink_domain_size: int = 8      # NVLink 域大小: 8卡节点=8, 超节点=40
    intra_node_bw_gbs: float = 140.0  # 节点内带宽 (GB/s)
    inter_node_bw_gbs: float = 50.0   # 节点间带宽 (GB/s)

    model: str = "minimax_m2.5_w8a8"
    batch_size: int = 256
    seq_len: int = 4096
    output_length: int = 1500
    chunk_prefill_size: int = 4096
    gpu: str = "NMZ"
    tbo: bool = False
    fuse_moe: bool = True
    dp_attention: bool = False
    moe_dp_size: int = 1
    moe_dense_tp_size: Optional[int] = None     # None | 1
    moe_a2a_backend: str = "none"                # {none,deepep,mooncake,nixl,mori,ascend_fuseep,flashinfer,megamoe}
    mtp_param: str = ""
    prefix_cache_hit_rate: float = 0.85
    model_json_path: Optional[str] = None

    memory_ratio: float = 0.9

    max_tp: int = 8
    max_pp: int = 4
    include_unified: bool = False  # 是否包含 unified 模式（全节点先 Prefill 后 Decode）
    ep_strategy: str = "ep_eq_tp"  # EP 策略 (deepep 后端): "ep_eq_tp" (EP=TP) | "ep_eq_1" (EP=1)

    num_workers: int = 0   # 0 = min(cpu_count(), 8)

    output_dir: str = "raw_data/pd_sweep"


@dataclass
class PDSweepEntry:
    """单次 (prefill_cfg, decode_cfg) 的运行结果"""
    prefill_nodes: int = 0
    prefill_tp: int = 1
    prefill_dp: int = 1
    prefill_pp: int = 1
    prefill_ep: int = 1
    prefill_cp: int = 1

    decode_nodes: int = 0
    decode_tp: int = 1
    decode_dp: int = 1
    decode_pp: int = 1
    decode_ep: int = 1

    dp_attention: bool = False         # 该配置是否启用 DPA

    ttft_ms: float = 0.0
    prefill_tps: float = 0.0      # 单实例: BS × SeqLen / (TTFT/1000)，tokens/s
    tbt_ms: float = 0.0           # Time Between Tokens，单步 decode 延迟
    decode_tps: float = 0.0       # 单实例: BS / (TBT/1000)，tokens/s
    e2e_ms: float = 0.0           # 单请求 E2E: TTFT + TBT × output_length

    sys_prefill_tps: float = 0.0  # prefill_tps × p_dp (系统 prefill 吞吐)
    sys_decode_tps: float = 0.0   # decode_tps × d_dp  (系统 decode 吞吐)
    sys_throughput_tps: float = 0.0  # 流水线均衡吞吐 (考虑 P/D 瓶颈)
    bottleneck: str = ""          # "Prefill" / "Decode" / "Balanced"

    prefill_mem_gb: float = 0.0
    decode_mem_gb: float = 0.0

    status: str = "OK"            # OK / OOM-P / OOM-D / OOM-PD
    detail_excel: str = ""        # 详细 Excel 文件名（相对于 output_dir）



_MODEL_CP_SUPPORT = {
    "deepseek_v3":       True,   # NSA CP (MLA)
    "deepseek_v3_w8a8":  True,
    "glm5":              True,   # NSA CP (DSA)
    "glm5_w8a8":         True,
    "glm5_fp8":          True,
    "glm5-fp8":          True,
    "qwen3moe":          True,   # Dense CP
    "minimax_m2.5_w8a8": True,
    # MiMo-V2-Flash (FP8-w8a8): GQA (full + SWA hybrid) Dense CP
    "mimo_v2_flash_channel_fp8_w8a8": True,
    "mimo_v2_channel_fp8_w8a8":       True,
    # Step-3.5/3.7-Flash: GQA (full + SWA 混合) + MoE
    "step_3_5_flash":     True,
    "step_3_5_flash_fp8": True,
    "step_3_7_flash":     True,
    "step_3_7_flash_fp8": True,
    "mimo2_channel_fp8_w8a8":         True,
    # Not in the table (MiniMax, LLaMA, ...) = unsupported
}


def model_supports_cp(model_name: str) -> bool:
    """检查模型是否支持 CP (基于 SGLang 实现)"""
    return _MODEL_CP_SUPPORT.get(model_name.lower(), False)



def enumerate_parallel_configs(
    total_gpus: int,
    num_layers: int,
    batch_size: int,
    max_tp: int = 8,
    max_pp: int = 4,
    is_decode: bool = False,
    cp_mode: str = "power_of_2",
    dp_attention: bool = False,
    ep_strategy: str = "ep_eq_tp",
    allow_cp: bool = True,
) -> List[Tuple[int, int, int, int, int]]:
    """
    枚举 total_gpus 个 GPU 下所有合法的 (tp, dp, pp, ep, cp) 组合。

    约束:
      - TP × DP × PP = total_gpus   (EP 不计入 world_size, SGLang 语义)
      - EP 策略 (deepep/mooncake 后端硬约束 EP ∈ {1, TP}):
          "ep_eq_tp" → EP = TP
          "ep_eq_1"  → EP = 1
      - TP ∈ {1,2,4,8}（2 的幂）且 TP ≤ max_tp
      - PP ∈ {1,2,4,...} 且 PP ≤ max_pp
      - DPA 硬约束: dp_attention=True 且 dp>1 时要求 tp % dp == 0
      - Prefill CP: 仅在 dp_attention=True 且 allow_cp=True 时枚举
                    cp_mode="power_of_2" → 2^k, 上界 = dp (不再要求 cp | dp)
                    cp_mode="all_factors" → 1..dp 之间所有整数 (不再限 dp 的因子)
                    规则依据: per_rank_attn_tokens = batch×seq / dp / cp
                              TP/PP 不切 token, DP/CP 独立均分 token
      - Decode  CP: 固定 1

    Args:
        batch_size: 系统总 batch。用于过滤 BS 小于有效 DP 的退化配置。
        cp_mode: CP 搜索模式。
            "power_of_2" — 仅搜索 2 的幂（默认）
            "all_factors" — 搜索 DP 的所有因子（用于超节点等非 2 幂拓扑）
        dp_attention: 当前配置是否启用 DP Attention。影响 CP 枚举和 DPA 拓扑检查。
        ep_strategy: EP 赋值策略 (deepep 后端下仅支持以下两项)。
            "ep_eq_tp" — EP=TP (默认)
            "ep_eq_1"  — EP=1  (禁用专家并行)
        allow_cp: 模型级 CP 开关。False 时强制 cp=1。
    """
    configs: List[Tuple[int, int, int, int, int]] = []
    skipped = 0

    for tp in [1, 2, 4, 8]:
        if tp > max_tp or tp > total_gpus:
            continue
        if total_gpus % tp != 0:
            continue
        remaining = total_gpus // tp

        for pp in range(1, max_pp + 1):
            if pp > remaining:
                continue
            if remaining % pp != 0:
                continue
            dp = remaining // pp

            #   "ep_eq_tp": EP = TP
            if ep_strategy == "ep_eq_tp":
                ep = tp
            elif ep_strategy == "ep_eq_1":
                ep = 1
            else:
                raise ValueError(
                    f"ep_strategy={ep_strategy!r} 不受支持 (deepep 后端下仅允许 "
                    f"'ep_eq_tp' / 'ep_eq_1')"
                )

            if is_decode:
                if dp > batch_size:
                    skipped += 1
                    continue
                cp_list = [1]
            elif not allow_cp or not dp_attention:
                cp_list = [1]
            elif cp_mode == "all_factors":
                cp_list = list(range(1, dp + 1))
            else:
                k_max = int(math.log2(dp)) if dp >= 1 else 0
                cp_list = [2**k for k in range(0, k_max + 1)]

            if dp_attention and dp > 1 and tp % dp != 0:
                skipped += 1
                continue

            if ep != 1 and ep != tp:
                skipped += 1
                continue

            for cp in cp_list:
                if not is_decode:
                    #   per_rank_attn_tokens = batch × seq / dp / cp
                    if dp > batch_size:
                        skipped += 1
                        continue
                configs.append((tp, dp, pp, ep, cp))

    if skipped > 0:
        stage = "Decode" if is_decode else "Prefill"
        print(f"  [enumerate] {stage} GPUs={total_gpus} BS={batch_size}: "
              f"保留 {len(configs)} 个, 跳过 {skipped} 个 (BS<有效DP 退化配置)")

    return configs



def _load_num_layers(config: PDSweepConfig) -> int:
    """从模型 JSON 读取 num_hidden_layers（PP 约束用）"""
    json_path = config.model_json_path
    if json_path is None:
        config_name = MODEL_CONFIG_MAP.get(config.model.lower())
        if config_name is None:
            raise ValueError(f"Unknown model: {config.model}")
        wheel_config_dir = Path(__file__).resolve().parent / "data" / "model_configs"
        config_dir = wheel_config_dir if wheel_config_dir.is_dir() else _PROJECT_ROOT / "configs"
        json_path = str(config_dir / config_name)
    with open(json_path, "r") as f:
        data = json.load(f)
    return int(data.get("num_hidden_layers", 1))


def _compute_status(result: dict) -> str:
    """根据 后端 返回的细粒度内存字段计算状态标签"""
    p_ok = result.get("prefill_memory_feasible", result.get("memory_feasible", False))
    d_ok = result.get("decode_memory_feasible", result.get("memory_feasible", False))
    if p_ok and d_ok:
        return "OK"
    if not p_ok and not d_ok:
        return "OOM-PD"
    if not p_ok:
        return "OOM-P"
    return "OOM-D"


def _fmt_parallel(tp: int, dp: int, pp: int, ep: int, cp: int, include_cp: bool = True) -> str:
    """格式化并行配置为简短字符串，如 tp8dp4pp1ep4cp2"""
    s = f"tp{tp}dp{dp}pp{pp}ep{ep}"
    if include_cp:
        s += f"cp{cp}"
    return s


def _build_detail_excel_name(
    model: str,
    prefill_nodes: int, p_tp: int, p_dp: int, p_pp: int, p_ep: int, p_cp: int,
    decode_nodes: int,  d_tp: int, d_dp: int, d_pp: int, d_ep: int,
    batch_size: int, seq_len: int, output_length: int,
    dp_attention: bool = False,
) -> str:
    """
    生成详细 Excel 文件名。
    示例: minimax_P3n_tp8dp3pp1ep3cp3_D5n_tp8dp5pp1ep5_dpa_bs256_in4096_out1500.xlsx
    """
    p_str = _fmt_parallel(p_tp, p_dp, p_pp, p_ep, p_cp, include_cp=True)
    d_str = _fmt_parallel(d_tp, d_dp, d_pp, d_ep, 1, include_cp=False)
    dpa_tag = "dpa" if dp_attention else "nodpa"
    return (
        f"{model}_P{prefill_nodes}n_{p_str}"
        f"_D{decode_nodes}n_{d_str}"
        f"_{dpa_tag}"
        f"_bs{batch_size}_in{seq_len}_out{output_length}.xlsx"
    )


def _print_entry(entry: PDSweepEntry, idx: int, total: int) -> None:
    """单行进度打印"""
    p_str = _fmt_parallel(entry.prefill_tp, entry.prefill_dp, entry.prefill_pp,
                          entry.prefill_ep, entry.prefill_cp)
    d_str = _fmt_parallel(entry.decode_tp, entry.decode_dp, entry.decode_pp,
                          entry.decode_ep, 1, include_cp=False)
    dpa_str = "DPA" if entry.dp_attention else "noDPA"
    if entry.status == "OK":
        print(
            f"  [{idx:>4}/{total}] {dpa_str} P{entry.prefill_nodes}n({p_str}) "
            f"D{entry.decode_nodes}n({d_str}) | "
            f"TTFT={entry.ttft_ms:8.2f}ms  TBT={entry.tbt_ms:7.3f}ms  "
            f"SysTPS={entry.sys_throughput_tps:10.0f}  [{entry.bottleneck}]  [{entry.status}]"
        )
    else:
        print(
            f"  [{idx:>4}/{total}] {dpa_str} P{entry.prefill_nodes}n({p_str}) "
            f"D{entry.decode_nodes}n({d_str}) | "
            f"P_mem={entry.prefill_mem_gb:.1f}GB  D_mem={entry.decode_mem_gb:.1f}GB  "
            f"[{entry.status}]"
        )



def _sweep_worker(task: tuple) -> "PDSweepEntry":
    """
    进程池工作函数（必须为模块级函数，保证 pickle 序列化）。

    task = (config, prefill_nodes, p_cfg, decode_nodes, d_cfg, output_dir_str, dpa)
      p_cfg = (tp, dp, pp, ep, cp)
      d_cfg = (tp, dp, pp, ep, cp)  -- cp 固定 1
      dpa   = bool, 该任务的 dp_attention 状态
    """
    import sys as _sys

    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_stdout_fd = os.dup(1)
    _saved_stderr_fd = os.dup(2)
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)
    _old_stdout = _sys.stdout
    _old_stderr = _sys.stderr
    _sys.stdout = open(os.devnull, 'w')
    _sys.stderr = open(os.devnull, 'w')

    try:
        config, prefill_nodes, p_cfg, decode_nodes, d_cfg, output_dir_str, dpa = task
        p_tp, p_dp, p_pp, p_ep, p_cp = p_cfg
        d_tp, d_dp, d_pp, d_ep, _    = d_cfg
        return _run_single_config(
            config,
            prefill_nodes, p_tp, p_dp, p_pp, p_ep, p_cp,
            decode_nodes,  d_tp, d_dp, d_pp, d_ep,
            Path(output_dir_str),
            dp_attention=dpa,
        )
    finally:
        _sys.stdout.flush()
        _sys.stdout.close()
        _sys.stderr.flush()
        _sys.stderr.close()
        _sys.stdout = _old_stdout
        _sys.stderr = _old_stderr
        os.dup2(_saved_stdout_fd, 1)
        os.dup2(_saved_stderr_fd, 2)
        os.close(_saved_stdout_fd)
        os.close(_saved_stderr_fd)
        os.close(_devnull_fd)


def _run_single_config(
    config: PDSweepConfig,
    prefill_nodes: int, p_tp: int, p_dp: int, p_pp: int, p_ep: int, p_cp: int,
    decode_nodes: int,  d_tp: int, d_dp: int, d_pp: int, d_ep: int,
    output_dir: Path,
    dp_attention: bool = False,
) -> PDSweepEntry:
    """执行单次 model_analyzer() 并包装为 PDSweepEntry"""

    output_dir.mkdir(parents=True, exist_ok=True)

    detail_excel_name = _build_detail_excel_name(
        config.model,
        prefill_nodes, p_tp, p_dp, p_pp, p_ep, p_cp,
        decode_nodes,  d_tp, d_dp, d_pp, d_ep,
        config.batch_size, config.seq_len, config.output_length,
        dp_attention=dp_attention,
    )
    detail_excel_path = str(output_dir / detail_excel_name)

    result = model_analyzer(
        prefill_tp=p_tp, prefill_dp=p_dp, prefill_pp=p_pp,
        prefill_ep=p_ep, prefill_cp=p_cp,
        decode_tp=d_tp,  decode_dp=d_dp,  decode_pp=d_pp,
        decode_ep=d_ep,  decode_cp=1,
        prefill_nodes=prefill_nodes, decode_nodes=decode_nodes,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        output_length=config.output_length,
        chunk_prefill_size=config.chunk_prefill_size,
        gpu=config.gpu,
        model=config.model,
        tbo=config.tbo,
        fuse_moe=config.fuse_moe,
        dp_attention=dp_attention,
        moe_dp_size=config.moe_dp_size,
        moe_dense_tp_size=config.moe_dense_tp_size,
        moe_a2a_backend=config.moe_a2a_backend,
        mtp_param=config.mtp_param,
        prefix_cache_hit_rate=config.prefix_cache_hit_rate,
        generate_csv=True,
        verbose=False,
        model_json_path=config.model_json_path,
        gpus_per_node=config.gpus_per_node,
        nvlink_domain_size=config.nvlink_domain_size,
        intra_node_bw_gbs=config.intra_node_bw_gbs,
        inter_node_bw_gbs=config.inter_node_bw_gbs,
        memory_ratio=config.memory_ratio,
    )

    src_excel = result.get("excel_path", "")
    if src_excel and os.path.exists(src_excel) and src_excel != detail_excel_path:
        shutil.move(src_excel, detail_excel_path)

    entry = PDSweepEntry(
        prefill_nodes=prefill_nodes,
        prefill_tp=p_tp, prefill_dp=p_dp, prefill_pp=p_pp,
        prefill_ep=p_ep, prefill_cp=p_cp,
        decode_nodes=decode_nodes,
        decode_tp=d_tp, decode_dp=d_dp, decode_pp=d_pp,
        decode_ep=d_ep,
        dp_attention=dp_attention,
        detail_excel=detail_excel_name,
    )

    entry.status = _compute_status(result)
    entry.prefill_mem_gb = result.get("prefill_memory_gb", 0.0)
    entry.decode_mem_gb  = result.get("decode_memory_gb",  0.0)

    if entry.status == "OK":
        prefill_batch_ms = result["prefill_latency_ms"]
        ttft = result.get("ttft_ms", prefill_batch_ms)
        tbt  = result["decode_latency_ms"]
        ttft_s = prefill_batch_ms / 1000.0    # 供给率用整批耗时
        tbt_s  = tbt / 1000.0

        tbt_bottleneck = result.get("decode_pp_bottleneck_ms", 0.0)
        decode_batch_per_dp = max(1, config.batch_size // d_dp) if d_dp > 0 else config.batch_size
        can_pipeline = (tbt_bottleneck > 0 and decode_batch_per_dp >= d_pp)
        tbt_bottleneck_s = tbt_bottleneck / 1000.0 if can_pipeline else tbt_s

        entry.ttft_ms     = ttft
        entry.tbt_ms      = tbt
        entry.e2e_ms      = ttft + tbt * config.output_length

        entry.prefill_tps = config.batch_size * config.seq_len / ttft_s
        entry.decode_tps  = config.batch_size / tbt_s

        entry.sys_prefill_tps = entry.prefill_tps
        entry.sys_decode_tps  = config.batch_size / tbt_bottleneck_s

        prefill_rps = config.batch_size / ttft_s
        decode_rps  = config.batch_size / (config.output_length * tbt_bottleneck_s)
        balanced_rps = min(prefill_rps, decode_rps)
        entry.sys_throughput_tps = balanced_rps * config.output_length

        if prefill_rps < decode_rps * 0.95:
            entry.bottleneck = "Prefill"
        elif decode_rps < prefill_rps * 0.95:
            entry.bottleneck = "Decode"
        else:
            entry.bottleneck = "Balanced"

    return entry


def run_pd_sweep(config: PDSweepConfig) -> List[PDSweepEntry]:
    """
    主遍历函数（多进程并行 + 进度条）。

    遍历所有 (prefill_nodes, prefill_cfg, decode_nodes, decode_cfg) 组合，
    返回完整结果列表（含 OOM 条目）。
    """
    num_layers  = _load_num_layers(config)
    output_dir  = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = config.num_workers if config.num_workers > 0 else max(cpu_count(), 8)

    dpa_values = [False, True] if config.dp_attention else [False]

    print(f"\n{'='*70}")
    print(f"  PD Sweep: {config.model}  N={config.total_nodes}nodes  GPU={config.gpu}")
    print(f"  BS={config.batch_size}  SeqLen={config.seq_len}  OutLen={config.output_length}")
    print(f"  ChunkSize={config.chunk_prefill_size}  max_tp={config.max_tp}  max_pp={config.max_pp}")
    print(f"  DPA遍历={dpa_values}  num_hidden_layers={num_layers}  workers={num_workers}")
    print(f"{'='*70}")

    all_tasks: List[tuple] = []

    for dpa in dpa_values:
        allow_cp = model_supports_cp(config.model) and dpa

        if config.include_unified:
            unified_gpus = config.total_nodes * config.gpus_per_node
            unified_cfgs = enumerate_parallel_configs(
                unified_gpus, num_layers, config.batch_size,
                config.max_tp, config.max_pp, is_decode=False,
                dp_attention=dpa, allow_cp=allow_cp,
                ep_strategy=config.ep_strategy)
            unified_skipped = 0
            for p_cfg in unified_cfgs:
                p_tp, p_dp, p_pp, p_ep, p_cp = p_cfg
                if p_dp > config.batch_size:
                    unified_skipped += 1
                    continue
                d_cfg = (p_tp, p_dp, p_pp, p_ep, 1)  # 复用相同并行配置，CP=1
                all_tasks.append((
                    config, config.total_nodes, p_cfg,
                    config.total_nodes, d_cfg, str(output_dir),
                    dpa,
                ))
            if unified_skipped > 0:
                dpa_tag = "DPA" if dpa else "noDPA"
                print(f"  [enumerate] Unified GPUs={unified_gpus} BS={config.batch_size} {dpa_tag}: "
                      f"跳过 {unified_skipped} 个 (decode 复用后 DP>BS 退化配置)")

        for prefill_nodes in range(1, config.total_nodes):
            decode_nodes = config.total_nodes - prefill_nodes
            prefill_cfgs = enumerate_parallel_configs(
                prefill_nodes * config.gpus_per_node,
                num_layers, config.batch_size,
                config.max_tp, config.max_pp, is_decode=False,
                dp_attention=dpa, allow_cp=allow_cp,
                ep_strategy=config.ep_strategy)
            decode_cfgs  = enumerate_parallel_configs(
                decode_nodes  * config.gpus_per_node,
                num_layers, config.batch_size,
                config.max_tp, config.max_pp, is_decode=True,
                dp_attention=dpa,
                ep_strategy=config.ep_strategy)
            for p_cfg in prefill_cfgs:
                for d_cfg in decode_cfgs:
                    all_tasks.append((
                        config, prefill_nodes, p_cfg,
                        decode_nodes, d_cfg, str(output_dir),
                        dpa,
                    ))

    total = len(all_tasks)
    print(f"  总配置数: {total}  开始并行遍历...\n")

    entries: List[PDSweepEntry] = []
    ok_cnt   = 0
    oom_cnt  = 0
    best_sys_tps = 0.0
    t_start  = time.time()

    with Pool(processes=num_workers) as pool:
        imap_iter = pool.imap_unordered(_sweep_worker, all_tasks)

        if _HAS_TQDM:
            pbar = _tqdm(
                total=total,
                desc="PD Sweep",
                unit="cfg",
                ncols=100,
                dynamic_ncols=True,
            )
        else:
            pbar = None
            print(f"  提示: 安装 tqdm 可获得更好的进度显示 (pip install tqdm)")

        for entry in imap_iter:
            entries.append(entry)
            if entry.status == "OK":
                ok_cnt += 1
                if entry.sys_throughput_tps > best_sys_tps:
                    best_sys_tps = entry.sys_throughput_tps
            else:
                oom_cnt += 1

            if pbar is not None:
                best_str = f"{best_sys_tps:.0f}" if best_sys_tps > 0 else "N/A"
                pbar.update(1)
                pbar.set_postfix(OK=ok_cnt, OOM=oom_cnt, best_SysTPS=best_str, refresh=False)
            else:
                completed = len(entries)
                if completed % max(1, total // 20) == 0 or completed == total:
                    elapsed_s = time.time() - t_start
                    speed = completed / elapsed_s if elapsed_s > 0 else 0
                    eta   = (total - completed) / speed if speed > 0 else 0
                    best_str = f"{best_sys_tps:.0f}" if best_sys_tps > 0 else "N/A"
                    print(f"  [{completed:>5}/{total}] OK={ok_cnt} OOM={oom_cnt} "
                          f"best_SysTPS={best_str}  速度={speed:.1f}cfg/s  ETA={eta:.0f}s")

        if pbar is not None:
            pbar.close()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  遍历完成: 共 {len(entries)} 种配置，"
          f"OK={ok_cnt}，OOM={oom_cnt}，耗时 {elapsed:.1f}s  "
          f"({total/elapsed:.1f} cfg/s)")
    print(f"{'='*70}\n")

    return entries



_BLUE_FILL  = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
_GREEN_FILL = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
_GOLD_FILL  = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
_SILVER_FILL= PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")
_BRONZE_FILL= PatternFill(start_color="CD7F32", end_color="CD7F32", fill_type="solid")
_RED_FILL   = PatternFill(start_color="FF6B6B", end_color="FF6B6B", fill_type="solid")
_THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"),  bottom=Side(style="thin"),
)

_LEADERBOARD_COLS = [
    ("Rank",              "Rank"),
    ("DPA",               "DP\nAttn"),
    ("Prefill_Nodes",     "Prefill\nNodes"),
    ("Prefill_GPUs",      "Prefill\nGPUs"),
    ("P_TP",              "P\nTP"),
    ("P_DP",              "P\nDP"),
    ("P_PP",              "P\nPP"),
    ("P_EP",              "P\nEP"),
    ("P_CP",              "P\nCP"),
    ("Decode_Nodes",      "Decode\nNodes"),
    ("Decode_GPUs",       "Decode\nGPUs"),
    ("D_TP",              "D\nTP"),
    ("D_DP",              "D\nDP"),
    ("D_PP",              "D\nPP"),
    ("D_EP",              "D\nEP"),
    ("TTFT_ms",           "TTFT\n(ms)"),
    ("Prefill_TPS",       "P TPS\n(单实例)"),
    ("Sys_Prefill_TPS",   "Sys P TPS\n(×P_DP)"),
    ("TBT_ms",            "TBT\n(ms)"),
    ("Decode_TPS",        "D TPS\n(单实例)"),
    ("Sys_Decode_TPS",    "Sys D TPS\n(×D_DP)"),
    ("Sys_Throughput_TPS","Sys Throughput\n(tokens/s)  ★"),
    ("Bottleneck",        "Bottleneck"),
    ("E2E_Time_ms",       "E2E Time\n(ms)"),
    ("P_Mem_GB",          "P Mem\n(GB/GPU)"),
    ("D_Mem_GB",          "D Mem\n(GB/GPU)"),
    ("Detail_Excel",      "Detail Excel"),
]

_OOM_COLS = [
    ("DPA",            "DP\nAttn"),
    ("Prefill_Nodes",  "Prefill\nNodes"),
    ("Prefill_GPUs",   "Prefill\nGPUs"),
    ("P_TP",           "P TP"),
    ("P_DP",           "P DP"),
    ("P_PP",           "P PP"),
    ("P_EP",           "P EP"),
    ("P_CP",           "P CP"),
    ("Decode_Nodes",   "Decode\nNodes"),
    ("Decode_GPUs",    "Decode\nGPUs"),
    ("D_TP",           "D TP"),
    ("D_DP",           "D DP"),
    ("D_PP",           "D PP"),
    ("D_EP",           "D EP"),
    ("P_Mem_GB",       "P Mem\n(GB/GPU)"),
    ("D_Mem_GB",       "D Mem\n(GB/GPU)"),
    ("Status",         "Status"),
]


def _write_header(ws, cols, fill):
    """写入表头行"""
    for col_idx, (_, label) in enumerate(cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=label)
        cell.font = Font(bold=True, size=10, color="FFFFFF")
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _THIN_BORDER
    ws.row_dimensions[1].height = 36


def _entry_to_leaderboard_row(entry: PDSweepEntry, rank: int) -> dict:
    return {
        "Rank":               rank,
        "DPA":                "ON" if entry.dp_attention else "OFF",
        "Prefill_Nodes":      entry.prefill_nodes,
        "Prefill_GPUs":       entry.prefill_tp * entry.prefill_dp * entry.prefill_pp,
        "P_TP":               entry.prefill_tp,
        "P_DP":               entry.prefill_dp,
        "P_PP":               entry.prefill_pp,
        "P_EP":               entry.prefill_ep,
        "P_CP":               entry.prefill_cp,
        "Decode_Nodes":       entry.decode_nodes,
        "Decode_GPUs":        entry.decode_tp * entry.decode_dp * entry.decode_pp,
        "D_TP":               entry.decode_tp,
        "D_DP":               entry.decode_dp,
        "D_PP":               entry.decode_pp,
        "D_EP":               entry.decode_ep,
        "TTFT_ms":            round(entry.ttft_ms, 3),
        "Prefill_TPS":        round(entry.prefill_tps, 1),
        "Sys_Prefill_TPS":    round(entry.sys_prefill_tps, 1),
        "TBT_ms":             round(entry.tbt_ms, 4),
        "Decode_TPS":         round(entry.decode_tps, 1),
        "Sys_Decode_TPS":     round(entry.sys_decode_tps, 1),
        "Sys_Throughput_TPS": round(entry.sys_throughput_tps, 1),
        "Bottleneck":         entry.bottleneck,
        "E2E_Time_ms":        round(entry.e2e_ms, 1),
        "P_Mem_GB":           round(entry.prefill_mem_gb, 2),
        "D_Mem_GB":           round(entry.decode_mem_gb, 2),
        "Detail_Excel":       entry.detail_excel,
    }


def _entry_to_oom_row(entry: PDSweepEntry) -> dict:
    return {
        "DPA":           "ON" if entry.dp_attention else "OFF",
        "Prefill_Nodes": entry.prefill_nodes,
        "Prefill_GPUs":  entry.prefill_tp * entry.prefill_dp * entry.prefill_pp,
        "P_TP":          entry.prefill_tp,
        "P_DP":          entry.prefill_dp,
        "P_PP":          entry.prefill_pp,
        "P_EP":          entry.prefill_ep,
        "P_CP":          entry.prefill_cp,
        "Decode_Nodes":  entry.decode_nodes,
        "Decode_GPUs":   entry.decode_tp * entry.decode_dp * entry.decode_pp,
        "D_TP":          entry.decode_tp,
        "D_DP":          entry.decode_dp,
        "D_PP":          entry.decode_pp,
        "D_EP":          entry.decode_ep,
        "P_Mem_GB":      round(entry.prefill_mem_gb, 2),
        "D_Mem_GB":      round(entry.decode_mem_gb, 2),
        "Status":        entry.status,
    }


def _auto_col_width(ws, cols, start_row: int = 2):
    """自动设置列宽"""
    for col_idx, (key, label) in enumerate(cols, 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(len(line) for line in label.split("\n"))
        for row in ws.iter_rows(
            min_row=start_row, max_row=min(ws.max_row, start_row + 50),
            min_col=col_idx, max_col=col_idx
        ):
            for cell in row:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 3, 30)


def save_sweep_summary(
    entries: List[PDSweepEntry],
    config: PDSweepConfig,
    output_path: str,
) -> None:
    """
    生成总表 Excel，包含两个 sheet：
      - leaderboard: OK 配置，按 E2E_Time_ms 升序排列
      - oom_configs:  OOM 配置，按 (prefill_nodes, decode_nodes) 排列
    """
    ok_entries  = sorted([e for e in entries if e.status == "OK"],
                         key=lambda e: -e.sys_throughput_tps)  # 按系统吞吐降序
    oom_entries = sorted([e for e in entries if e.status != "OK"],
                         key=lambda e: (e.prefill_nodes, e.decode_nodes))

    wb = Workbook()

    ws_lb = wb.active
    ws_lb.title = "leaderboard"

    meta = (
        f"Model={config.model}  N={config.total_nodes}nodes  GPU={config.gpu}  "
        f"BS={config.batch_size}  SeqLen={config.seq_len}  OutLen={config.output_length}  "
        f"ChunkSize={config.chunk_prefill_size}  "
        f"OK={len(ok_entries)}  OOM={len(oom_entries)  }"
    )
    ws_lb.merge_cells(start_row=1, start_column=1,
                      end_row=1, end_column=len(_LEADERBOARD_COLS))
    meta_cell = ws_lb.cell(row=1, column=1, value=meta)
    meta_cell.font = Font(bold=True, size=10, color="FFFFFF")
    meta_cell.fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    meta_cell.alignment = Alignment(horizontal="left", vertical="center")
    ws_lb.row_dimensions[1].height = 20

    for col_idx, (_, label) in enumerate(_LEADERBOARD_COLS, 1):
        cell = ws_lb.cell(row=2, column=col_idx, value=label)
        cell.font = Font(bold=True, size=10, color="FFFFFF")
        cell.fill = _BLUE_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _THIN_BORDER
    ws_lb.row_dimensions[2].height = 36

    for row_idx, entry in enumerate(ok_entries, 3):
        rank = row_idx - 2
        row_data = _entry_to_leaderboard_row(entry, rank)

        rank_fill = None
        if rank == 1:
            rank_fill = _GOLD_FILL
        elif rank == 2:
            rank_fill = _SILVER_FILL
        elif rank == 3:
            rank_fill = _BRONZE_FILL

        for col_idx, (key, _) in enumerate(_LEADERBOARD_COLS, 1):
            cell = ws_lb.cell(row=row_idx, column=col_idx, value=row_data[key])
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if rank_fill and key in ("Rank", "Sys_Throughput_TPS"):
                cell.fill = rank_fill
                cell.font = Font(bold=True)
            if key == "Sys_Throughput_TPS" and rank_fill is None:
                cell.font = Font(bold=True)

    _auto_col_width(ws_lb, _LEADERBOARD_COLS, start_row=3)
    ws_lb.freeze_panes = "A3"

    ws_oom = wb.create_sheet(title="oom_configs")
    _write_header(ws_oom, _OOM_COLS, _RED_FILL)

    for row_idx, entry in enumerate(oom_entries, 2):
        row_data = _entry_to_oom_row(entry)
        for col_idx, (key, _) in enumerate(_OOM_COLS, 1):
            cell = ws_oom.cell(row=row_idx, column=col_idx, value=row_data[key])
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")

    _auto_col_width(ws_oom, _OOM_COLS, start_row=2)

    wb.save(output_path)

    print(f"\n总表已保存: {output_path}")
    print(f"  - leaderboard: {len(ok_entries)} 条有效配置（按系统吞吐降序）")
    print(f"  - oom_configs:  {len(oom_entries)} 条 OOM 配置")

    _print_top_results(ok_entries[:10], config)


def _print_top_results(entries: List[PDSweepEntry], config: PDSweepConfig) -> None:
    """控制台打印 Top-N 结果"""
    if not entries:
        print("  (无有效配置)")
        return

    print(f"\n{'='*140}")
    print(f"  Top-{len(entries)} 配置排名（按系统吞吐降序）"
          f"  BS={config.batch_size}  SeqLen={config.seq_len}  OutLen={config.output_length}")
    print(f"{'='*140}")
    header = (
        f"  {'Rank':>4}  {'DPA':>5}  {'P-Nodes':>7}  {'Prefill Config':<26}  "
        f"{'D-Nodes':>7}  {'Decode Config':<22}  "
        f"{'TTFT(ms)':>10}  {'P-TPS':>10}  {'SysP-TPS':>10}  "
        f"{'TBT(ms)':>8}  {'D-TPS':>10}  {'SysD-TPS':>10}  "
        f"{'SysThrpt':>10}  {'BtlNeck':>8}  {'E2E(ms)':>10}"
    )
    print(header)
    print(f"  {'-'*144}")
    for i, e in enumerate(entries, 1):
        p_cfg = _fmt_parallel(e.prefill_tp, e.prefill_dp, e.prefill_pp, e.prefill_ep, e.prefill_cp)
        d_cfg = _fmt_parallel(e.decode_tp, e.decode_dp, e.decode_pp, e.decode_ep, 1, include_cp=False)
        dpa_str = "ON" if e.dp_attention else "OFF"
        print(
            f"  {i:>4}  {dpa_str:>5}  {e.prefill_nodes:>7}  {p_cfg:<26}  "
            f"{e.decode_nodes:>7}  {d_cfg:<22}  "
            f"{e.ttft_ms:>10.2f}  {e.prefill_tps:>10.0f}  {e.sys_prefill_tps:>10.0f}  "
            f"{e.tbt_ms:>8.3f}  {e.decode_tps:>10.0f}  {e.sys_decode_tps:>10.0f}  "
            f"{e.sys_throughput_tps:>10.0f}  {e.bottleneck:>8}  {e.e2e_ms:>10.1f}"
        )
    print(f"{'='*148}\n")



def pd_sweep(
    total_nodes: int = 2,
    model: str = "glm5",
    batch_size: int = 32,
    seq_len: int = 50000,
    output_length: int = 200,
    chunk_prefill_size: int = 16384,
    gpu: str = "NMZ",
    gpus_per_node: int = 8,
    nvlink_domain_size: int = 8,      # NVLink 域大小: 8卡节点=8, 超节点=40
    intra_node_bw_gbs: float = 140.0,  # 节点内带宽 (GB/s)
    inter_node_bw_gbs: float = 50.0,   # 节点间带宽 (GB/s)
    tbo: bool = False,
    fuse_moe: bool = True,
    dp_attention: bool = True,
    mtp_param: str = "",
    max_tp: int = 8,
    max_pp: int = 4,
    include_unified: bool = True,
    prefix_cache_hit_rate = 0.2,
    num_workers: int = 0,
    output_dir: str = "raw_data/pd_sweep_mmw8a8_2N_glm",
    model_json_path: Optional[str] = None,
    ep_strategy: str = "ep_eq_tp",
    memory_ratio: float = 0.9,
) -> Tuple[List[PDSweepEntry], str]:
    """
    PD 遍历分析器主入口。

    支持两种模式：
      - unified:  全节点先 Prefill 后 Decode（共享权重分片，TP/DP/PP/EP 相同）
      - pd_split: PD 分离（不同节点分别做 Prefill 和 Decode）

    Args:
        total_nodes:       总节点数（每节点 8 卡）
        model:             模型名称
        batch_size:        批大小
        seq_len:           输入序列长度
        output_length:     输出序列长度
        chunk_prefill_size: Chunked Prefill 块大小
        gpu:               GPU 类型
        gpus_per_node:     每节点GPU数（默认8）
        nvlink_domain_size: NVLink域大小：8卡节点=8，超节点=40
        intra_node_bw_gbs: 节点内带宽 (GB/s)
        inter_node_bw_gbs: 节点间带宽 (GB/s)
        tbo:               是否启用 TBO
        fuse_moe:          是否融合 MoE
        dp_attention:      是否启用 DPA 遍历 (True=遍历[False,True], False=仅[False])
        mtp_param:         MTP 参数（暂预留）
        max_tp:            TP 最大值（默认 8）
        max_pp:            PP 最大值（默认 4）
        include_unified:   是否包含 unified 模式（默认 True）
        num_workers:       进程数（0 = min(cpu_count, 8)）
        output_dir:        输出目录
        model_json_path:   直接指定模型 JSON 路径
        ep_strategy:       EP 策略 "ep_eq_tp" (EP=TP, deepep 默认) | "ep_eq_1" (EP=1, 禁用专家并行)

    Returns:
        (entries, summary_excel_path)
    """
    config = PDSweepConfig(
        total_nodes=total_nodes,
        model=model,
        batch_size=batch_size,
        seq_len=seq_len,
        output_length=output_length,
        chunk_prefill_size=chunk_prefill_size,
        gpu=gpu,
        gpus_per_node=gpus_per_node,
        nvlink_domain_size=nvlink_domain_size,
        intra_node_bw_gbs=intra_node_bw_gbs,
        inter_node_bw_gbs=inter_node_bw_gbs,
        tbo=tbo,
        fuse_moe=fuse_moe,
        dp_attention=dp_attention,
        include_unified=include_unified,
        mtp_param=mtp_param,
        prefix_cache_hit_rate = prefix_cache_hit_rate,
        max_tp=max_tp,
        max_pp=max_pp,
        ep_strategy=ep_strategy,
        num_workers=num_workers,
        output_dir=output_dir,
        model_json_path=model_json_path,
        memory_ratio=memory_ratio,
    )

    entries = run_pd_sweep(config)

    summary_name = (
        f"pd_sweep_summary_{model}_N{total_nodes}n_{gpu}"
        f"_bs{batch_size}_in{seq_len}_out{output_length}.xlsx"
    )
    summary_path = str(Path(output_dir) / summary_name)
    save_sweep_summary(entries, config, summary_path)

    return entries, summary_path

if __name__ == "__main__":
    pd_sweep(dp_attention=True)