import os
import shutil
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
    from .model_analyzer import (MODEL_CONFIG_MAP, _PROJECT_ROOT, model_analyzer,
                                 run_cli, _cli_str_list)
    from .pd_sweep import (
        enumerate_parallel_configs,
        model_supports_cp,
        _load_num_layers,
        _compute_status,
        _fmt_parallel,
        PDSweepConfig,
    )
except ImportError:
    from model_analyzer import MODEL_CONFIG_MAP, _PROJECT_ROOT, model_analyzer
    from pd_sweep import (
        enumerate_parallel_configs,
        model_supports_cp,
        _load_num_layers,
        _compute_status,
        _fmt_parallel,
        PDSweepConfig,
    )


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SupernodeTopology:
    """超节点拓扑描述"""
    gpus_per_tray: int = 4
    num_trays: int = 10
    intra_tray_bw_gbs: float = 140   
    inter_tray_bw_gbs: float = 140  

    @property
    def total_gpus(self) -> int:
        return self.gpus_per_tray * self.num_trays


@dataclass
class SupernodeSweepConfig:
    """超节点遍历配置"""
    # 拓扑
    topology: SupernodeTopology = field(default_factory=SupernodeTopology)

    # 推理参数
    model: str = "minimax_m2.5_w8a8"
    batch_size: int = 256
    seq_len: int = 70000
    output_length: int = 200
    chunk_prefill_size: int = 4096
    gpu: str = "NMZ"
    tbo: bool = False
    sbo: bool = False                            # SBO (Single-Batch Overlap), 与 tbo 组合见 model_analyzer()
    sequence_parallel: bool = False              # Megatron-SP (norm/residual ÷sp, AR→RS+AG); 见 model_analyzer()
    fuse_moe: bool = False
    dp_attention: bool = False
    # PP 调度: pp_loop_size = pp_size + pp_async_batch_depth (0=经典 1F1B)
    pp_async_batch_depth: int = 0
    #   典型 LMSYS 风格 (超节点大 EP): moe_dense_tp_size=1, moe_a2a_backend="deepep", moe_dp_size=1
    moe_dp_size: int = 1
    moe_dense_tp_size: Optional[int] = None    
    moe_a2a_backend: str = "none"               # {none,deepep,mooncake,nixl,mori,ascend_fuseep,flashinfer,megamoe}
    mtp_param: str = ""
    kv_cache_dtype: str = "fp8"                  # KV cache dtype (fp8/bf16/...), 影响 attention KV 口径
    attention_compute_dtype: str = "fp8"         # attn QK/PV 算力 dtype (fp8默认强制/bf16/auto)
    # 视觉输入 (Kimi-K2.5 / Qwen3-VL-MoE); num_images=0 → 纯文本
    num_images: int = 0
    image_height: int = 448
    image_width: int = 448
    model_json_path: Optional[str] = None

    # Prefix Cache
    prefix_cache_hit_rate: float = 0.85

    memory_ratio: float = 0.9

    # 遍历模式: "unified", "pd_split"
    sweep_modes: List[str] = field(default_factory=lambda: ["unified", "pd_split"])

    # 搜索空间控制
    max_pp: int = 4
    cp_mode: str = "all_factors"  # "power_of_2" | "all_factors"
    ep_strategy: str = "auto"  # EP 策略: "auto" (DPA=OFF→{1,TP}; DPA=ON→{1,TP,TP*DP}) | "ep_eq_tp" | "ep_eq_1"

    # 并行执行
    num_workers: int = 0

    # 输出
    output_dir: str = "raw_data/supernode_sweep"
    generate_detail_excel: bool = True  # 是否生成每种配置的详细算子性能 Excel


@dataclass
class SupernodeSweepEntry:
    """超节点遍历结果条目"""
    # 模式
    sweep_mode: str = "unified"        # "unified" | "pd_split"

    # Tray 分配
    prefill_trays: int = 0
    decode_trays: int = 0
    prefill_gpus: int = 0
    decode_gpus: int = 0

    # Prefill 并行配置
    prefill_tp: int = 1
    prefill_dp: int = 1
    prefill_pp: int = 1
    prefill_ep: int = 1
    prefill_cp: int = 1

    # Decode 并行配置
    decode_tp: int = 1
    decode_dp: int = 1
    decode_pp: int = 1
    decode_ep: int = 1

    # DP Attention 状态 (P/D 独立控制)
    prefill_dp_attention: bool = False
    decode_dp_attention: bool = False

    # 性能指标
    ttft_ms: float = 0.0
    prefill_tps: float = 0.0
    tbt_ms: float = 0.0
    decode_tps: float = 0.0
    e2e_ms: float = 0.0

    # 系统级吞吐 (多实例)
    system_prefill_tps: float = 0.0   # prefill_tps * prefill_dp
    system_decode_tps: float = 0.0    # decode_tps * decode_dp
    sys_throughput_tps: float = 0.0   # 流水线均衡吞吐 (考虑 P/D 瓶颈)
    bottleneck: str = ""              # "Prefill" / "Decode" / "Balanced"

    # 显存
    prefill_mem_gb: float = 0.0
    decode_mem_gb: float = 0.0

    # 状态
    status: str = "OK"
    detail_excel: str = ""
    bandwidth_warning: str = ""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _build_detail_excel_name(
    model: str, mode: str,
    p_trays: int, p_tp: int, p_dp: int, p_pp: int, p_ep: int, p_cp: int,
    d_trays: int, d_tp: int, d_dp: int, d_pp: int, d_ep: int,
    batch_size: int, seq_len: int, output_length: int,
    chunk_prefill_size: int,
    prefill_dp_attention: bool = False,
    decode_dp_attention: bool = False,
) -> str:
    p_str = _fmt_parallel(p_tp, p_dp, p_pp, p_ep, p_cp, include_cp=True)
    d_str = _fmt_parallel(d_tp, d_dp, d_pp, d_ep, 1, include_cp=False)
    p_dpa_tag = "Pdpa" if prefill_dp_attention else "Pnodpa"
    d_dpa_tag = "Ddpa" if decode_dp_attention else "Dnodpa"
    return (
        f"{model}_{mode}_P{p_trays}t_{p_str}"
        f"_D{d_trays}t_{d_str}"
        f"_{p_dpa_tag}_{d_dpa_tag}"
        f"_bs{batch_size}_in{seq_len}_out{output_length}"
        f"_cs{chunk_prefill_size}.xlsx"
    )


def _load_num_layers_from_config(config: SupernodeSweepConfig) -> int:
    """从模型 JSON 读取 num_hidden_layers"""
    tmp = PDSweepConfig(
        total_nodes=1,
        model=config.model,
        model_json_path=config.model_json_path,
    )
    return _load_num_layers(tmp)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _supernode_worker(task: tuple) -> SupernodeSweepEntry:
    """进程池工作函数"""
    import sys as _sys

    # 子进程静默
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
        (config, mode,
         p_trays, p_gpus, p_cfg,
         d_trays, d_gpus, d_cfg,
         output_dir_str, p_dpa, d_dpa, generate_detail) = task
        p_tp, p_dp, p_pp, p_ep, p_cp = p_cfg
        d_tp, d_dp, d_pp, d_ep, _    = d_cfg
        return _run_single(
            config, mode,
            p_trays, p_gpus, p_tp, p_dp, p_pp, p_ep, p_cp,
            d_trays, d_gpus, d_tp, d_dp, d_pp, d_ep,
            Path(output_dir_str),
            prefill_dp_attention=p_dpa,
            decode_dp_attention=d_dpa,
            generate_detail_excel=generate_detail,
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


def _run_single(
    config: SupernodeSweepConfig, mode: str,
    p_trays: int, p_gpus: int,
    p_tp: int, p_dp: int, p_pp: int, p_ep: int, p_cp: int,
    d_trays: int, d_gpus: int,
    d_tp: int, d_dp: int, d_pp: int, d_ep: int,
    output_dir: Path,
    prefill_dp_attention: bool = False,
    decode_dp_attention: bool = False,
    generate_detail_excel: bool = True,
) -> SupernodeSweepEntry:
    """执行单次 model_analyzer() 并包装为 SupernodeSweepEntry"""
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_name = _build_detail_excel_name(
        config.model, mode,
        p_trays, p_tp, p_dp, p_pp, p_ep, p_cp,
        d_trays, d_tp, d_dp, d_pp, d_ep,
        config.batch_size, config.seq_len, config.output_length,
        config.chunk_prefill_size,
        prefill_dp_attention=prefill_dp_attention,
        decode_dp_attention=decode_dp_attention,
    )
    detail_path = str((output_dir / detail_name).resolve())

    result = model_analyzer(
        prefill_tp=p_tp, prefill_dp=p_dp, prefill_pp=p_pp,
        prefill_ep=p_ep, prefill_cp=p_cp,
        decode_tp=d_tp, decode_dp=d_dp, decode_pp=d_pp,
        decode_ep=d_ep, decode_cp=1,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        output_length=config.output_length,
        chunk_prefill_size=config.chunk_prefill_size,
        gpu=config.gpu,
        model=config.model,
        tbo=config.tbo,
        sbo=config.sbo,
        sequence_parallel=config.sequence_parallel,
        pp_async_batch_depth=config.pp_async_batch_depth,
        attention_compute_dtype=config.attention_compute_dtype,
        fuse_moe=config.fuse_moe,
        prefill_dp_attention=prefill_dp_attention,
        decode_dp_attention=decode_dp_attention,
        moe_dp_size=config.moe_dp_size,
        moe_dense_tp_size=config.moe_dense_tp_size,
        moe_a2a_backend=config.moe_a2a_backend,
        mtp_param=config.mtp_param,
        kv_cache_dtype=config.kv_cache_dtype,
        num_images=config.num_images,
        image_height=config.image_height,
        image_width=config.image_width,
        generate_csv=generate_detail_excel,
        verbose=False,
        model_json_path=config.model_json_path,
        prefix_cache_hit_rate=config.prefix_cache_hit_rate,
        gpus_per_node=config.topology.gpus_per_tray,
        nvlink_domain_size=config.topology.total_gpus,
        intra_node_bw_gbs=config.topology.intra_tray_bw_gbs,
        inter_node_bw_gbs=config.topology.inter_tray_bw_gbs,
        memory_ratio=config.memory_ratio,
    )

    # 移动生成的 Excel
    src_excel = result.get("excel_path", "")
    if generate_detail_excel and src_excel and os.path.exists(src_excel) and src_excel != detail_path:
        shutil.move(src_excel, detail_path)

    # 带宽提示
    topo = config.topology
    bw_warn = (
        f"TP AllReduce 走 PCIe ~{topo.intra_tray_bw_gbs:.0f} GB/s, "
        f"CP/EP 走 xHCL ~{topo.inter_tray_bw_gbs:.0f} GB/s, "
        f"工具使用统一带宽模型，通信延迟估算偏乐观"
    )

    entry = SupernodeSweepEntry(
        sweep_mode=mode,
        prefill_trays=p_trays,
        decode_trays=d_trays,
        prefill_gpus=p_gpus,
        decode_gpus=d_gpus,
        prefill_tp=p_tp, prefill_dp=p_dp, prefill_pp=p_pp,
        prefill_ep=p_ep, prefill_cp=p_cp,
        decode_tp=d_tp, decode_dp=d_dp, decode_pp=d_pp,
        decode_ep=d_ep,
        prefill_dp_attention=prefill_dp_attention,
        decode_dp_attention=decode_dp_attention,
        detail_excel=detail_path if generate_detail_excel else "",
        bandwidth_warning=bw_warn,
    )

    entry.status = _compute_status(result)
    entry.prefill_mem_gb = result.get("prefill_memory_gb", 0.0)
    entry.decode_mem_gb = result.get("decode_memory_gb", 0.0)

    if entry.status == "OK":
        prefill_batch_ms = result["prefill_latency_ms"]
        ttft = result.get("ttft_ms", prefill_batch_ms)
        tbt = result["decode_latency_ms"]
        ttft_s = prefill_batch_ms / 1000.0    # 供给率用整批耗时
        tbt_s = tbt / 1000.0

    
        tbt_bottleneck = result.get("decode_pp_bottleneck_ms", 0.0)
        decode_batch_per_dp = max(1, config.batch_size // d_dp) if d_dp > 0 else config.batch_size
        can_pipeline = (tbt_bottleneck > 0 and decode_batch_per_dp >= d_pp)
        tbt_bottleneck_s = tbt_bottleneck / 1000.0 if can_pipeline else tbt_s

        entry.ttft_ms = ttft
        entry.tbt_ms = tbt
        entry.e2e_ms = ttft + tbt * config.output_length

       
        entry.prefill_tps = config.batch_size * config.seq_len / ttft_s
        entry.decode_tps = config.batch_size / tbt_s
        entry.system_prefill_tps = entry.prefill_tps
        entry.system_decode_tps = config.batch_size / tbt_bottleneck_s

     
        prefill_rps = config.batch_size / ttft_s
        decode_rps = config.batch_size / (config.output_length * tbt_bottleneck_s)
        balanced_rps = min(prefill_rps, decode_rps)
        entry.sys_throughput_tps = balanced_rps * config.output_length

        if prefill_rps < decode_rps * 0.95:
            entry.bottleneck = "Prefill"
        elif decode_rps < prefill_rps * 0.95:
            entry.bottleneck = "Decode"
        else:
            entry.bottleneck = "Balanced"

    return entry


def run_supernode_sweep(config: SupernodeSweepConfig) -> List[SupernodeSweepEntry]:
    """
    超节点并行策略遍历（多进程并行 + 进度条）。
    返回完整结果列表（含 OOM 条目）。
    """
    topo = config.topology
    num_layers = _load_num_layers_from_config(config)


    if not config.output_dir: 
        summary_name = (
            f"supernode_sweep_{config.model}_{topo.num_trays}T{topo.gpus_per_tray}G_{config.gpu}"
            f"_bs{config.batch_size}_in{config.seq_len}_out{config.output_length}"
            f"_cs{config.chunk_prefill_size}"
        )
        config.output_dir = f"raw_data/{summary_name}"

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = config.num_workers if config.num_workers > 0 else min(cpu_count(), 8)
    max_tp = topo.gpus_per_tray  # TP 不跨 Tray


    if config.dp_attention:
        dpa_combos = [(False, False), (True, False), (False, True), (True, True)]
    else:
        dpa_combos = [(False, False)]

    print(f"\n{'='*70}")
    print(f"  Supernode Sweep: {config.model}  "
          f"Topo={topo.num_trays}Tray×{topo.gpus_per_tray}GPU={topo.total_gpus}卡")
    print(f"  PCIe={topo.intra_tray_bw_gbs:.0f}GB/s  "
          f"xHCL={topo.inter_tray_bw_gbs:.0f}GB/s  GPU={config.gpu}")
    print(f"  BS={config.batch_size}  SeqLen={config.seq_len}  "
          f"OutLen={config.output_length}  ChunkSize={config.chunk_prefill_size}"
          f"  PrefixCache={config.prefix_cache_hit_rate}")
    print(f"  max_tp={max_tp}  max_pp={config.max_pp}  "
          f"cp_mode={config.cp_mode}  modes={config.sweep_modes}")
    dpa_label = [(f"P{'ON' if p else 'OFF'}_D{'ON' if d else 'OFF'}") for p, d in dpa_combos]
    print(f"  DPA遍历={dpa_label}  num_hidden_layers={num_layers}  workers={num_workers}")
    print(f"{'='*70}")

    all_tasks: List[tuple] = []

    for p_dpa, d_dpa in dpa_combos:
        allow_cp = model_supports_cp(config.model) and p_dpa

        if "unified" in config.sweep_modes:
            if p_dpa == d_dpa:
                total_gpus = topo.total_gpus
                prefill_cfgs = enumerate_parallel_configs(
                    total_gpus, num_layers, config.batch_size,
                    max_tp, config.max_pp,
                    is_decode=False, cp_mode=config.cp_mode,
                    dp_attention=p_dpa, allow_cp=allow_cp,
                    ep_strategy=config.ep_strategy)
                unified_skipped = 0
                for p_cfg in prefill_cfgs:
                    p_tp, p_dp, p_pp, p_ep, p_cp = p_cfg
                    if p_dp > config.batch_size:
                        unified_skipped += 1
                        continue
                    d_cfg = (p_tp, p_dp, p_pp, p_ep, 1)
                    all_tasks.append((
                        config, "unified",
                        topo.num_trays, total_gpus, p_cfg,
                        topo.num_trays, total_gpus, d_cfg,
                        str(output_dir),
                        p_dpa, d_dpa, config.generate_detail_excel,
                    ))
                if unified_skipped > 0:
                    tag = f"P_DPA={'ON' if p_dpa else 'OFF'}_D_DPA={'ON' if d_dpa else 'OFF'}"
                    print(f"  [enumerate] Unified GPUs={total_gpus} BS={config.batch_size} {tag}: "
                          f"跳过 {unified_skipped} 个 (decode 复用后 DP>BS 退化配置)")

        if "pd_split" in config.sweep_modes:
            for p_trays in range(1, topo.num_trays):
                d_trays = topo.num_trays - p_trays
                p_gpus = p_trays * topo.gpus_per_tray
                d_gpus = d_trays * topo.gpus_per_tray
                prefill_cfgs = enumerate_parallel_configs(
                    p_gpus, num_layers, config.batch_size,
                    max_tp, config.max_pp,
                    is_decode=False, cp_mode=config.cp_mode,
                    dp_attention=p_dpa, allow_cp=allow_cp,
                    ep_strategy=config.ep_strategy)
                decode_cfgs = enumerate_parallel_configs(
                    d_gpus, num_layers, config.batch_size,
                    max_tp, config.max_pp,
                    is_decode=True, cp_mode=config.cp_mode,
                    dp_attention=d_dpa,
                    ep_strategy=config.ep_strategy)
                for p_cfg in prefill_cfgs:
                    for d_cfg in decode_cfgs:
                        all_tasks.append((
                            config, "pd_split",
                            p_trays, p_gpus, p_cfg,
                            d_trays, d_gpus, d_cfg,
                            str(output_dir),
                            p_dpa, d_dpa, config.generate_detail_excel,
                        ))

    total = len(all_tasks)
    unified_cnt = sum(1 for t in all_tasks if t[1] == "unified")
    pd_cnt = total - unified_cnt
    print(f"  总配置数: {total}（unified={unified_cnt}, pd_split={pd_cnt}）")
    print(f"  开始并行遍历...\n")

    entries: List[SupernodeSweepEntry] = []
    ok_cnt = 0
    oom_cnt = 0
    best_sys_tps = 0.0
    t_start = time.time()

    with Pool(processes=num_workers) as pool:
        imap_iter = pool.imap_unordered(_supernode_worker, all_tasks)

        if _HAS_TQDM:
            pbar = _tqdm(
                total=total, desc="Supernode Sweep",
                unit="cfg", ncols=100, dynamic_ncols=True,
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
                    eta = (total - completed) / speed if speed > 0 else 0
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


# ---------------------------------------------------------------------------
# Excel 总表
# ---------------------------------------------------------------------------

_BLUE_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
_GREEN_FILL = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
_GOLD_FILL = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
_SILVER_FILL = PatternFill(start_color="C0C0C0", end_color="C0C0C0", fill_type="solid")
_BRONZE_FILL = PatternFill(start_color="CD7F32", end_color="CD7F32", fill_type="solid")
_RED_FILL = PatternFill(start_color="FF6B6B", end_color="FF6B6B", fill_type="solid")
_THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)

_LEADERBOARD_COLS = [
    ("Rank",           "Rank"),
    ("P_DPA",          "P\nDPA"),
    ("D_DPA",          "D\nDPA"),
    ("Mode",           "Mode"),
    ("Prefill_Trays",  "Prefill\nTrays"),
    ("Prefill_GPUs",   "Prefill\nGPUs"),
    ("P_TP",           "P\nTP"),
    ("P_DP",           "P\nDP"),
    ("P_PP",           "P\nPP"),
    ("P_EP",           "P\nEP"),
    ("P_CP",           "P\nCP"),
    ("Decode_Trays",   "Decode\nTrays"),
    ("Decode_GPUs",    "Decode\nGPUs"),
    ("D_TP",           "D\nTP"),
    ("D_DP",           "D\nDP"),
    ("D_PP",           "D\nPP"),
    ("D_EP",           "D\nEP"),
    ("TTFT_ms",        "TTFT\n(ms)"),
    ("Prefill_TPS",    "Prefill TPS\n(tokens/s)"),
    ("TBT_ms",         "TBT\n(ms)"),
    ("Decode_TPS",     "Decode TPS\n(tokens/s)"),
    ("Sys_Prefill_TPS","Sys Prefill TPS\n(tokens/s)"),
    ("Sys_Decode_TPS", "Sys Decode TPS\n(tokens/s)"),
    ("Sys_Throughput_TPS","Sys Throughput\n(tokens/s)  ★"),
    ("Bottleneck",     "Bottleneck"),
    ("E2E_Time_ms",    "E2E Time\n(ms)"),
    ("P_Mem_GB",       "P Mem\n(GB/GPU)"),
    ("D_Mem_GB",       "D Mem\n(GB/GPU)"),
    ("BW_Warning",     "BW\nWarning"),
    ("Detail_Excel",   "Detail Excel"),
]

_OOM_COLS = [
    ("P_DPA",          "P\nDPA"),
    ("D_DPA",          "D\nDPA"),
    ("Mode",           "Mode"),
    ("Prefill_Trays",  "Prefill\nTrays"),
    ("Prefill_GPUs",   "Prefill\nGPUs"),
    ("P_TP",           "P TP"),
    ("P_DP",           "P DP"),
    ("P_PP",           "P PP"),
    ("P_EP",           "P EP"),
    ("P_CP",           "P CP"),
    ("Decode_Trays",   "Decode\nTrays"),
    ("Decode_GPUs",    "Decode\nGPUs"),
    ("D_TP",           "D TP"),
    ("D_DP",           "D DP"),
    ("D_PP",           "D PP"),
    ("D_EP",           "D EP"),
    ("P_Mem_GB",       "P Mem\n(GB/GPU)"),
    ("D_Mem_GB",       "D Mem\n(GB/GPU)"),
    ("Status",         "Status"),
]


def _entry_to_row(entry: SupernodeSweepEntry, rank: int) -> dict:
    return {
        "Rank":          rank,
        "P_DPA":         "ON" if entry.prefill_dp_attention else "OFF",
        "D_DPA":         "ON" if entry.decode_dp_attention else "OFF",
        "Mode":          entry.sweep_mode,
        "Prefill_Trays": entry.prefill_trays,
        "Prefill_GPUs":  entry.prefill_gpus,
        "P_TP":          entry.prefill_tp,
        "P_DP":          entry.prefill_dp,
        "P_PP":          entry.prefill_pp,
        "P_EP":          entry.prefill_ep,
        "P_CP":          entry.prefill_cp,
        "Decode_Trays":  entry.decode_trays,
        "Decode_GPUs":   entry.decode_gpus,
        "D_TP":          entry.decode_tp,
        "D_DP":          entry.decode_dp,
        "D_PP":          entry.decode_pp,
        "D_EP":          entry.decode_ep,
        "TTFT_ms":       round(entry.ttft_ms, 3),
        "Prefill_TPS":   round(entry.prefill_tps, 1),
        "TBT_ms":        round(entry.tbt_ms, 4),
        "Decode_TPS":    round(entry.decode_tps, 1),
        "Sys_Prefill_TPS": round(entry.system_prefill_tps, 1),
        "Sys_Decode_TPS":  round(entry.system_decode_tps, 1),
        "Sys_Throughput_TPS": round(entry.sys_throughput_tps, 1),
        "Bottleneck":      entry.bottleneck,
        "E2E_Time_ms":   round(entry.e2e_ms, 1),
        "P_Mem_GB":      round(entry.prefill_mem_gb, 2),
        "D_Mem_GB":      round(entry.decode_mem_gb, 2),
        "BW_Warning":    entry.bandwidth_warning,
        "Detail_Excel":  entry.detail_excel,
    }


def _entry_to_oom_row(entry: SupernodeSweepEntry) -> dict:
    return {
        "P_DPA":         "ON" if entry.prefill_dp_attention else "OFF",
        "D_DPA":         "ON" if entry.decode_dp_attention else "OFF",
        "Mode":          entry.sweep_mode,
        "Prefill_Trays": entry.prefill_trays,
        "Prefill_GPUs":  entry.prefill_gpus,
        "P_TP":          entry.prefill_tp,
        "P_DP":          entry.prefill_dp,
        "P_PP":          entry.prefill_pp,
        "P_EP":          entry.prefill_ep,
        "P_CP":          entry.prefill_cp,
        "Decode_Trays":  entry.decode_trays,
        "Decode_GPUs":   entry.decode_gpus,
        "D_TP":          entry.decode_tp,
        "D_DP":          entry.decode_dp,
        "D_PP":          entry.decode_pp,
        "D_EP":          entry.decode_ep,
        "P_Mem_GB":      round(entry.prefill_mem_gb, 2),
        "D_Mem_GB":      round(entry.decode_mem_gb, 2),
        "Status":        entry.status,
    }


def _write_header(ws, cols, fill):
    for col_idx, (_, label) in enumerate(cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=label)
        cell.font = Font(bold=True, size=10, color="FFFFFF")
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _THIN_BORDER
    ws.row_dimensions[1].height = 36


def _auto_col_width(ws, cols, start_row: int = 2):
    for col_idx, (key, label) in enumerate(cols, 1):
        col_letter = get_column_letter(col_idx)
        max_len = max(len(line) for line in label.split("\n"))
        for row in ws.iter_rows(
            min_row=start_row, max_row=min(ws.max_row, start_row + 50),
            min_col=col_idx, max_col=col_idx,
        ):
            for cell in row:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 3, 30)


def _write_leaderboard_sheet(wb, title: str, ok_entries: List[SupernodeSweepEntry],
                              config: SupernodeSweepConfig, meta_extra: str = ""):
    """写入一个排行榜 sheet"""
    ws = wb.create_sheet(title=title)
    topo = config.topology

    # 元信息行
    meta = (
        f"Model={config.model}  Topo={topo.num_trays}T×{topo.gpus_per_tray}G={topo.total_gpus}卡  "
        f"GPU={config.gpu}  BS={config.batch_size}  SeqLen={config.seq_len}  "
        f"OutLen={config.output_length}  ChunkSize={config.chunk_prefill_size}  "
        f"PrefixCache={config.prefix_cache_hit_rate}  "
        f"OK={len(ok_entries)}"
    )
    if meta_extra:
        meta += f"  {meta_extra}"
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=len(_LEADERBOARD_COLS))
    meta_cell = ws.cell(row=1, column=1, value=meta)
    meta_cell.font = Font(bold=True, size=10, color="FFFFFF")
    meta_cell.fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    meta_cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 20

    # 表头（第 2 行）
    for col_idx, (_, label) in enumerate(_LEADERBOARD_COLS, 1):
        cell = ws.cell(row=2, column=col_idx, value=label)
        cell.font = Font(bold=True, size=10, color="FFFFFF")
        cell.fill = _BLUE_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _THIN_BORDER
    ws.row_dimensions[2].height = 36

    # 数据行
    for row_idx, entry in enumerate(ok_entries, 3):
        rank = row_idx - 2
        row_data = _entry_to_row(entry, rank)

        rank_fill = None
        if rank == 1:
            rank_fill = _GOLD_FILL
        elif rank == 2:
            rank_fill = _SILVER_FILL
        elif rank == 3:
            rank_fill = _BRONZE_FILL

        for col_idx, (key, _) in enumerate(_LEADERBOARD_COLS, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=row_data[key])
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if rank_fill and key in ("Rank", "Sys_Throughput_TPS"):
                cell.fill = rank_fill
                cell.font = Font(bold=True)
            if key == "Sys_Throughput_TPS" and rank_fill is None:
                cell.font = Font(bold=True)

    _auto_col_width(ws, _LEADERBOARD_COLS, start_row=3)
    ws.freeze_panes = "A3"


def save_supernode_summary(
    entries: List[SupernodeSweepEntry],
    config: SupernodeSweepConfig,
    output_path: str,
) -> None:
    """生成超节点遍历总表 Excel"""
    ok_entries = sorted([e for e in entries if e.status == "OK"],
                        key=lambda e: -e.sys_throughput_tps)
    oom_entries = sorted([e for e in entries if e.status != "OK"],
                         key=lambda e: (e.sweep_mode, e.prefill_trays))

    # 按模式拆分
    ok_unified = [e for e in ok_entries if e.sweep_mode == "unified"]
    ok_pd = [e for e in ok_entries if e.sweep_mode == "pd_split"]

    wb = Workbook()
    # 删除默认 sheet
    wb.remove(wb.active)

    # Sheet 1: 合并排行榜
    _write_leaderboard_sheet(wb, "leaderboard_combined", ok_entries, config,
                              meta_extra=f"(unified={len(ok_unified)}, pd_split={len(ok_pd)})")

    # Sheet 2: unified 排行榜
    if ok_unified:
        _write_leaderboard_sheet(wb, "leaderboard_unified", ok_unified, config,
                                  meta_extra="mode=unified")

    # Sheet 3: pd_split 排行榜
    if ok_pd:
        _write_leaderboard_sheet(wb, "leaderboard_pd_split", ok_pd, config,
                                  meta_extra="mode=pd_split")

    # Sheet 4: OOM 配置
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
    print(f"  - leaderboard_combined: {len(ok_entries)} 条有效配置（按系统吞吐降序）")
    if ok_unified:
        print(f"  - leaderboard_unified:  {len(ok_unified)} 条 unified 配置")
    if ok_pd:
        print(f"  - leaderboard_pd_split: {len(ok_pd)} 条 pd_split 配置")
    print(f"  - oom_configs:          {len(oom_entries)} 条 OOM 配置")

    # 控制台打印 Top-10
    _print_top_results(ok_entries[:10], config)


def _print_top_results(entries: List[SupernodeSweepEntry],
                       config: SupernodeSweepConfig) -> None:
    if not entries:
        print("  (无有效配置)")
        return

    print(f"\n{'='*148}")
    print(f"  Top-{len(entries)} 配置排名（按系统吞吐降序）"
          f"  BS={config.batch_size}  SeqLen={config.seq_len}  OutLen={config.output_length}")
    print(f"{'='*148}")
    header = (
        f"  {'Rank':>4}  {'P_DPA':>5}  {'D_DPA':>5}  {'Mode':<9}  {'P-Trays':>7}  {'Prefill Config':<26}  "
        f"{'D-Trays':>7}  {'Decode Config':<22}  "
        f"{'TTFT(ms)':>10}  {'P-TPS':>10}  "
        f"{'TBT(ms)':>8}  {'D-TPS':>10}  "
        f"{'SysThrpt':>10}  {'BtlNeck':>8}  {'E2E(ms)':>10}"
    )
    print(header)
    print(f"  {'-'*152}")
    for i, e in enumerate(entries, 1):
        p_cfg = _fmt_parallel(e.prefill_tp, e.prefill_dp, e.prefill_pp,
                              e.prefill_ep, e.prefill_cp)
        d_cfg = _fmt_parallel(e.decode_tp, e.decode_dp, e.decode_pp,
                              e.decode_ep, 1, include_cp=False)
        p_dpa_str = "ON" if e.prefill_dp_attention else "OFF"
        d_dpa_str = "ON" if e.decode_dp_attention else "OFF"
        print(
            f"  {i:>4}  {p_dpa_str:>5}  {d_dpa_str:>5}  {e.sweep_mode:<9}  {e.prefill_trays:>7}  {p_cfg:<26}  "
            f"{e.decode_trays:>7}  {d_cfg:<22}  "
            f"{e.ttft_ms:>10.2f}  {e.prefill_tps:>10.0f}  "
            f"{e.tbt_ms:>8.3f}  {e.decode_tps:>10.0f}  "
            f"{e.sys_throughput_tps:>10.0f}  {e.bottleneck:>8}  {e.e2e_ms:>10.1f}"
        )
    print(f"{'='*148}\n")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def supernode_sweep(
    # 拓扑参数
    gpus_per_tray: int = 4,
    num_trays: int = 10,
    intra_tray_bw_gbs: float = 140,
    inter_tray_bw_gbs: float = 140,
    # 推理参数
    model: str = "minimax_m2.5_w8a8",
    batch_size: int = 1,
    seq_len: int = 70000,
    output_length: int = 1500,
    chunk_prefill_size: int = 70000000,
    gpu: str = "NMZ",
    # 遍历控制
    sweep_modes: Optional[List[str]] =  ["pd_split"],   # 默认只跑 PD 分离
    max_pp: int = 4,
    cp_mode: str = "all_factors",
    prefix_cache_hit_rate = 0.85,
    # 执行控制
    num_workers: int = 256,
    output_dir: str = "raw_data/supernode_sweep_",
    generate_detail_excel: bool = True,
    # 透传选项
    tbo: bool = False,
    sbo: bool = False,
    sequence_parallel: bool = False,   # Megatron-SP: norm/residual ÷sp, o_proj/mlp_down AR→RS+AG
    pp_async_batch_depth: int = 0,
    attention_compute_dtype: str = "fp8",
    fuse_moe: bool = True,
    dp_attention: bool = True,
    mtp_param: str = "",
    model_json_path: Optional[str] = None,
    ep_strategy: str = "auto",
    memory_ratio: float = 0.9,
    # ★ 2026-05-22 新增 (集成文档 §4.5/§4.6)
    moe_dp_size: int = 1,
    moe_dense_tp_size: Optional[int] = None,    # None | 1
    moe_a2a_backend: str = "none",
    # KV cache dtype + 视觉输入 (与 model_analyzer() / shape_collector 对齐)
    kv_cache_dtype: str = "fp8",
    num_images: int = 0,
    image_height: int = 448,
    image_width: int = 448,
) -> Tuple[List[SupernodeSweepEntry], str]:
    """
    40 卡超节点并行策略遍历主入口。

    Args:
        gpus_per_tray:      每个 Tray 的 GPU 数（默认 4）
        num_trays:          Tray 总数（默认 10）
        intra_tray_bw_gbs:  Tray 内 PCIe 单向带宽 (GB/s)
        inter_tray_bw_gbs:  Tray 间 xHCL 单向带宽 (GB/s)
        model:              模型名称
        batch_size:         批大小
        seq_len:            输入序列长度
        output_length:      输出序列长度
        chunk_prefill_size: Chunked Prefill 块大小
        gpu:                GPU 类型
        sweep_modes:        遍历模式列表，默认 ["unified", "pd_split"]
        max_pp:             PP 最大值
        cp_mode:            CP 搜索模式: "power_of_2" | "all_factors"
        num_workers:        进程数（0 = min(cpu_count, 8)）
        output_dir:         输出目录。如果为空字符串，将自动在 raw_data/ 下创建目录，目录名基于模型和参数生成。
        generate_detail_excel: 是否生成每种配置的详细算子性能 Excel（默认 True）。
                               设为 False 只生成总表，不生成详细表格，大幅减少文件数量。
        model_json_path:    直接指定模型 JSON 路径
        ep_strategy:        EP 策略
                            "auto"     (默认) DPA=OFF→{1,TP}; DPA=ON→{1,TP,TP×DP} (支持 LMSYS 大 EP)
                            "ep_eq_tp" EP=TP (向后兼容旧脚本)
                            "ep_eq_1"  EP=1 (禁用专家并行)

    Returns:
        (entries, summary_excel_path)
    """
    if sweep_modes is None:
        sweep_modes = ["pd_split"]

    topology = SupernodeTopology(
        gpus_per_tray=gpus_per_tray,
        num_trays=num_trays,
        intra_tray_bw_gbs=intra_tray_bw_gbs,
        inter_tray_bw_gbs=inter_tray_bw_gbs,
    )

    config = SupernodeSweepConfig(
        topology=topology,
        model=model,
        batch_size=batch_size,
        seq_len=seq_len,
        output_length=output_length,
        chunk_prefill_size=chunk_prefill_size,
        gpu=gpu,
        tbo=tbo,
        sbo=sbo,
        sequence_parallel=sequence_parallel,
        pp_async_batch_depth=pp_async_batch_depth,
        attention_compute_dtype=attention_compute_dtype,
        fuse_moe=fuse_moe,
        dp_attention=dp_attention,
        mtp_param=mtp_param,
        sweep_modes=sweep_modes,
        prefix_cache_hit_rate = prefix_cache_hit_rate,
        max_pp=max_pp,
        cp_mode=cp_mode,
        num_workers=num_workers,
        output_dir=output_dir,
        generate_detail_excel=generate_detail_excel,
        model_json_path=model_json_path,
        ep_strategy=ep_strategy,
        memory_ratio=memory_ratio,
        # ★ 2026-05-22 透传
        moe_dp_size=moe_dp_size,
        moe_dense_tp_size=moe_dense_tp_size,
        moe_a2a_backend=moe_a2a_backend,
        kv_cache_dtype=kv_cache_dtype,
        num_images=num_images,
        image_height=image_height,
        image_width=image_width,
    )

    entries = run_supernode_sweep(config)

    topo = config.topology
    summary_name = (
        f"supernode_sweep_{model}_{topo.num_trays}T{topo.gpus_per_tray}G_{gpu}"
        f"_bs{batch_size}_in{seq_len}_out{output_length}"
        f"_cs{chunk_prefill_size}.xlsx"
    )
    summary_path = str(Path(config.output_dir) / summary_name)
    save_supernode_summary(entries, config, summary_path)

    return entries, summary_path

def _print_result(title, entries, summary_path):
    """打印结果统计"""
    ok_count = sum(1 for e in entries if e.status == "OK")
    oom_count = sum(1 for e in entries if e.status != "OK")

    print(f"\n{'='*70}")
    print(f"  {title} - 结果汇总")
    print(f"{'='*70}")
    print(f"总表路径: {summary_path}")
    print(f"共 {len(entries)} 种配置，有效 {ok_count}，OOM {oom_count}")

    # 按模式分组统计
    for mode in ["unified", "pd_split"]:
        mode_entries = [e for e in entries if e.sweep_mode == mode]
        if not mode_entries:
            continue
        mode_ok = [e for e in mode_entries if e.status == "OK"]
        print(f"\n  [{mode}] 共 {len(mode_entries)} 配置, 有效 {len(mode_ok)}")

        ok_sorted = sorted(mode_ok, key=lambda e: -e.sys_throughput_tps)
        for i, e in enumerate(ok_sorted[:5], 1):
            p_dpa_tag = "PDPA" if e.prefill_dp_attention else "PnoDPA"
            d_dpa_tag = "DDPA" if e.decode_dp_attention else "DnoDPA"
            dpa_tag = f"{p_dpa_tag}_{d_dpa_tag}"
            if mode == "unified":
                print(f"    #{i}: {dpa_tag} {e.prefill_gpus}GPU "
                      f"tp{e.prefill_tp}dp{e.prefill_dp}pp{e.prefill_pp}"
                      f"ep{e.prefill_ep}cp{e.prefill_cp} "
                      f"SysTPS={e.sys_throughput_tps:.0f} [{e.bottleneck}] "
                      f"E2E={e.e2e_ms:.1f}ms TTFT={e.ttft_ms:.2f}ms TBT={e.tbt_ms:.3f}ms")
            else:
                print(f"    #{i}: {dpa_tag} P{e.prefill_trays}tray(tp{e.prefill_tp}dp{e.prefill_dp}"
                      f"pp{e.prefill_pp}ep{e.prefill_ep}cp{e.prefill_cp}) "
                      f"D{e.decode_trays}tray(tp{e.decode_tp}dp{e.decode_dp}"
                      f"pp{e.decode_pp}ep{e.decode_ep}) "
                      f"SysTPS={e.sys_throughput_tps:.0f} [{e.bottleneck}] "
                      f"E2E={e.e2e_ms:.1f}ms TTFT={e.ttft_ms:.2f}ms TBT={e.tbt_ms:.3f}ms")
    print()
    
    
def run_minimax_w8a8_10tray_unified_only_prefix_cache():
    """MiniMax M2.5 W8A8 - 10 Tray x 4 DCU, 仅 unified 模式"""
    entries, summary_path = supernode_sweep(
        gpus_per_tray=8,
        num_trays=12,
        intra_tray_bw_gbs=1600*0.8,
        inter_tray_bw_gbs=1600*0.8,
        model="deepseek_v4",
        batch_size=4,
        seq_len=128*1024,
        output_length=1500,
        chunk_prefill_size=700000000,
        gpu="NMZ",
        sweep_modes=["pd_split"],
        prefix_cache_hit_rate = 0.8,
        max_pp=12,
        cp_mode="all_factors",
        num_workers=192,
        output_dir="",
        tbo=False,
        fuse_moe=False,
        dp_attention=True,
    )
    _print_result("MiniMax W8A8 10Tray (unified) prefix_cache", entries, summary_path)


def _supernode_main_cli():
    """命令行入口: `python -m model_analyzer.supernode_sweep --model ... --gpus_per_tray ...`。

    参数由 supernode_sweep() 签名自动生成; sweep_modes 用逗号分隔
    (如 `--sweep_modes unified,pd_split`)。
    """
    entries, summary_path = run_cli(
        supernode_sweep,
        type_overrides={"sweep_modes": _cli_str_list, "moe_dense_tp_size": int},
        prog="python -m model_analyzer.supernode_sweep",
        description="超节点 (多 Tray) 并行策略遍历: 枚举 TP/DP/EP/CP/PP, 输出 SysTPS 排行")
    _print_result("Supernode Sweep", entries, summary_path)
    return entries, summary_path


if __name__ == "__main__":
    # 防 import-time 副作用: wheel 装到 site-packages 后, `from model_analyzer import ...`
    # 会触发本文件 import, 顶层调用会跑实际任务并因 hardcode 路径找不到 configs 而报错。
    _supernode_main_cli()
