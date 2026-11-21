import math
import os
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
from openpyxl.styles import Alignment, Font, PatternFill

try:
    from .model_analyzer import model_analyzer, run_cli
    from .pd_sweep import (
        enumerate_parallel_configs, model_supports_cp, _load_num_layers,
        _fmt_parallel, _write_header, _auto_col_width,
        _BLUE_FILL, _GREEN_FILL, _GOLD_FILL, _SILVER_FILL, _BRONZE_FILL,
        _RED_FILL, _THIN_BORDER,
    )
except ImportError:
    from model_analyzer import model_analyzer, run_cli
    from pd_sweep import (
        enumerate_parallel_configs, model_supports_cp, _load_num_layers,
        _fmt_parallel, _write_header, _auto_col_width,
        _BLUE_FILL, _GREEN_FILL, _GOLD_FILL, _SILVER_FILL, _BRONZE_FILL,
        _RED_FILL, _THIN_BORDER,
    )

_GREY_FILL = PatternFill(start_color="A6A6A6", end_color="A6A6A6", fill_type="solid")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    """单个阶段 (prefill 或 decode) 的独立推理参数 + 搜索/过滤设置。"""
    nodes_budget: int = 1            # 该阶段节点预算上限, 遍历 nodes ∈ [1, budget]
    batch_size: int = 32
    seq_len: int = 8192              # prefill=输入长; decode=上下文长 (KV)
    output_length: int = 1024        # decode 生成长度; prefill 端基本无意义 (取小值即可)
    chunk_prefill_size: int = 16384
    dp_attention: bool = False       # True → 该阶段遍历 [False, True]; False → 仅 [False]
    tbo: bool = False
    sbo: bool = False
    moe_dp_size: int = 1
    moe_dense_tp_size: Optional[int] = None
    moe_a2a_backend: str = "none"
    # MTP: 仅 decode 端生效 (prefill 端本脚本强制不传)
    mtp_param: str = ""
    mtp_acceptance_rate: float = 0.8
    # SLO 过滤阈值 (None = 不过滤): prefill→TTFT(ms), decode→TBT(ms)
    slo_ms: Optional[float] = None
    bs_search: str = "slo"
    bs_min: int = 1
    bs_max: int = 512
    bs_step: int = 2                
    bs_list: Optional[List[int]] = None


@dataclass
class CommonConfig:
    """P/D 共享的环境/模型参数。"""
    model: str = "minimax_m2.5_w8a8"
    gpu: str = "NMZ"
    device_type: Optional[str] = None    # None=跟随 gpu; 显式传则覆盖 (透传 model_analyzer)
    gpus_per_node: int = 8
    nvlink_domain_size: int = 8
    intra_node_bw_gbs: float = 140.0
    inter_node_bw_gbs: float = 50.0
    memory_ratio: float = 0.9
    sequence_parallel: bool = False
    pp_async_batch_depth: int = 0
    fuse_moe: bool = True
    attention_compute_dtype: str = "fp8"
    kv_cache_dtype: str = "fp8"
    num_images: int = 0
    image_height: int = 448
    image_width: int = 448
    prefix_cache_hit_rate: float = 0.0
    ep_strategy: str = "auto"
    max_tp: int = 8
    max_pp: int = 8
    num_workers: int = 0             # 0 = max(cpu_count(), 8)
    output_dir: str = ""
    model_json_path: Optional[str] = None
    util_profile: Optional[str] = None     # 逐算子/分阶段利用率 profile JSON 路径 (透传 model_analyzer)
    # Roofline 全局利用率标量 (-1.0 = 用 device JSON utilization_factors, 透传 model_analyzer)
    util_tensor_core: float = -1.0
    util_vector: float = -1.0
    util_sfu: float = -1.0
    util_bandwidth: float = -1.0
    generate_detail_excel: bool = False   # 每个候选是否落详细 Excel (默认否, 加速)


@dataclass
class StageEntry:
    """单个 (nodes, 并行配置, dpa) 候选的结果行。"""
    stage: str = "prefill"           # "prefill" / "decode"
    nodes: int = 0
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    dp_attention: bool = False
    batch_size: int = 0              # 该候选选出的 BS (BS 搜索胜出值; bs_search="off" 时=spec.batch_size)

    # 指标
    latency_ms: float = 0.0          # prefill=TTFT; decode=TBT(per-token)
    sys_tps: float = 0.0             # 系统(单实例)吞吐 tokens/s
    per_card_tps: float = 0.0        # 单卡吞吐 = sys_tps / (tp×pp)
    request_rate: float = 0.0        # 请求率 req/s/实例 (排序键)
    mem_gb: float = 0.0
    mtp_tps: float = 0.0             # decode MTP 加速后吞吐 (无 MTP=0)

    # 状态: OK / OOM / SLO (显存超 = OOM; 显存 OK 但超 SLO = SLO)
    status: str = "OK"

    @property
    def gpus(self) -> int:
        return self.tp * self.pp


# ---------------------------------------------------------------------------
# 单候选执行 (worker, 必须模块级以便 pickle)
# ---------------------------------------------------------------------------

def _silence_fds():
    """重定向 fd 1/2 + sys.stdout/stderr 到 /dev/null, 返回还原所需句柄。"""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2), sys.stdout, sys.stderr)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    return devnull_fd, saved


def _restore_fds(handles):
    devnull_fd, (s1, s2, old_out, old_err) = handles
    sys.stdout.flush(); sys.stdout.close()
    sys.stderr.flush(); sys.stderr.close()
    sys.stdout, sys.stderr = old_out, old_err
    os.dup2(s1, 1); os.dup2(s2, 2)
    os.close(s1); os.close(s2); os.close(devnull_fd)


def _stage_worker(task: tuple) -> StageEntry:
    """
    task = (common, spec, stage, nodes, cfg, dpa, bs_fixed)
      cfg = (tp, dp, pp, ep, cp)
      stage ∈ {"prefill","decode"}
    bs_fixed=None → 单 worker 内沿 BS 轴求胜出工作点 (二分/网格/固定);
    bs_fixed=int  → bs_list 模式, 固定该 BS 跑一次 (每个 BS 独立成榜单一行)。
    """
    handles = _silence_fds()
    try:
        common, spec, stage, nodes, cfg, dpa, bs_fixed = task
        if bs_fixed is not None:
            return _run_stage_candidate(common, spec, stage, nodes, cfg, dpa, bs_fixed)
        return _search_bs_for_candidate(common, spec, stage, nodes, cfg, dpa)
    finally:
        _restore_fds(handles)


def _search_bs_for_candidate(
    common: CommonConfig, spec: StageSpec, stage: str,
    nodes: int, cfg: Tuple[int, int, int, int, int], dpa: bool,
) -> StageEntry:
    """对单个 (nodes, cfg, dpa) 沿 BS 轴求胜出工作点。返回带 batch_size 的 StageEntry。

    BS 与并行枚举的唯一耦合: dp ≤ bs (DPA 实例数 = dp)。故该 cfg 的 BS 下界 = max(bs_min, dp)。
    """
    dp = cfg[1]
    if spec.bs_search == "off":
        return _run_stage_candidate(common, spec, stage, nodes, cfg, dpa, spec.batch_size)

    lo = max(spec.bs_min, dp)
    hi = max(lo, spec.bs_max)
    if spec.bs_search == "maxrps":
        return _grid_max_rps_bs(common, spec, stage, nodes, cfg, dpa, lo, hi)
    return _bisect_max_bs(common, spec, stage, nodes, cfg, dpa, lo, hi)


def _bisect_max_bs(
    common: CommonConfig, spec: StageSpec, stage: str,
    nodes: int, cfg: Tuple[int, int, int, int, int], dpa: bool,
    lo: int, hi: int,
) -> StageEntry:
    """二分找 SLO∧显存 双约束下的最大 BS (两约束对 BS 均单调)。

    best = 满足约束的最大 BS 的 entry; 若连 lo 都不满足, 返回 lo 那次 (非 OK 状态, 供诊断)。
    """
    best: Optional[StageEntry] = None
    fallback: Optional[StageEntry] = None      # lo 那次, 用于全程失败时如实展示
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        e = _run_stage_candidate(common, spec, stage, nodes, cfg, dpa, mid)
        if fallback is None or mid == lo:
            fallback = e
        if e.status == "OK":
            best = e
            a = mid + 1                          # 还能更大
        else:
            b = mid - 1                          # OOM 或超 SLO, 退
    return best if best is not None else fallback


def _grid_max_rps_bs(
    common: CommonConfig, spec: StageSpec, stage: str,
    nodes: int, cfg: Tuple[int, int, int, int, int], dpa: bool,
    lo: int, hi: int,
) -> StageEntry:
    """网格扫 BS 取 rps 最高 (rps-vs-BS 非单调)。显存到顶 (OOM) 即提前停 (显存对 BS 单调)。"""
    best: Optional[StageEntry] = None
    last: Optional[StageEntry] = None
    step = max(1, spec.bs_step)
    for bs in range(lo, hi + 1, step):
        e = _run_stage_candidate(common, spec, stage, nodes, cfg, dpa, bs)
        last = e
        if e.status == "OOM":
            break                                # 显存单调: 再大必 OOM
        if e.status == "OK" and (best is None or e.request_rate > best.request_rate):
            best = e
    return best if best is not None else last


def _run_stage_candidate(
    common: CommonConfig, spec: StageSpec, stage: str,
    nodes: int, cfg: Tuple[int, int, int, int, int], dpa: bool, bs: int,
) -> StageEntry:
    """对一个阶段的单个 (nodes, 并行配置, bs) 跑一次 model_analyzer, 只取本阶段指标。"""
    tp, dp, pp, ep, cp = cfg
    is_prefill = (stage == "prefill")

    # 镜像另一侧 (保证 world_size 合法; decode 侧 cp 固定 1)
    mirror = (tp, dp, pp, ep, 1)
    if is_prefill:
        p_cfg, d_cfg = cfg, mirror
        mtp_param = ""                       # prefill 端永不用 MTP
    else:
        p_cfg, d_cfg = mirror, cfg
        mtp_param = spec.mtp_param

    p_tp, p_dp, p_pp, p_ep, p_cp = p_cfg
    d_tp, d_dp, d_pp, d_ep, d_cp = d_cfg

    result = model_analyzer(
        prefill_tp=p_tp, prefill_dp=p_dp, prefill_pp=p_pp, prefill_ep=p_ep, prefill_cp=p_cp,
        decode_tp=d_tp,  decode_dp=d_dp,  decode_pp=d_pp,  decode_ep=d_ep,  decode_cp=d_cp,
        prefill_nodes=nodes, decode_nodes=nodes,
        batch_size=bs,
        seq_len=spec.seq_len,
        output_length=spec.output_length,
        chunk_prefill_size=spec.chunk_prefill_size,
        gpu=common.gpu,
        device_type=common.device_type,
        model=common.model,
        tbo=spec.tbo,
        sbo=spec.sbo,
        sequence_parallel=common.sequence_parallel,
        pp_async_batch_depth=common.pp_async_batch_depth,
        attention_compute_dtype=common.attention_compute_dtype,
        kv_cache_dtype=common.kv_cache_dtype,
        fuse_moe=common.fuse_moe,
        dp_attention=dpa,
        prefill_dp_attention=dpa, decode_dp_attention=dpa,
        moe_dp_size=spec.moe_dp_size,
        prefill_moe_dp_size=spec.moe_dp_size, decode_moe_dp_size=spec.moe_dp_size,
        moe_dense_tp_size=spec.moe_dense_tp_size,
        prefill_moe_dense_tp_size=spec.moe_dense_tp_size,
        decode_moe_dense_tp_size=spec.moe_dense_tp_size,
        moe_a2a_backend=spec.moe_a2a_backend,
        prefill_moe_a2a_backend=spec.moe_a2a_backend,
        decode_moe_a2a_backend=spec.moe_a2a_backend,
        mtp_param=mtp_param,
        mtp_acceptance_rate=spec.mtp_acceptance_rate,
        num_images=common.num_images,
        image_height=common.image_height,
        image_width=common.image_width,
        prefix_cache_hit_rate=common.prefix_cache_hit_rate,
        memory_ratio=common.memory_ratio,
        gpus_per_node=common.gpus_per_node,
        nvlink_domain_size=common.nvlink_domain_size,
        intra_node_bw_gbs=common.intra_node_bw_gbs,
        inter_node_bw_gbs=common.inter_node_bw_gbs,
        model_json_path=common.model_json_path,
        util_profile=common.util_profile,
        util_tensor_core=common.util_tensor_core,
        util_vector=common.util_vector,
        util_sfu=common.util_sfu,
        util_bandwidth=common.util_bandwidth,
        generate_csv=common.generate_detail_excel,
        verbose=False,
        skip_memory_check=True,    # 镜像侧绝不能让整跑提前返回
    )

    entry = StageEntry(
        stage=stage, nodes=nodes,
        tp=tp, dp=dp, pp=pp, ep=ep, cp=cp, dp_attention=dpa, batch_size=bs,
    )

    if is_prefill:
        _fill_prefill_metrics(entry, result, spec)
    else:
        _fill_decode_metrics(entry, result, spec)
    return entry


def _fill_prefill_metrics(entry: StageEntry, result: dict, spec: StageSpec) -> None:
    feasible = result.get("prefill_memory_feasible",
                          result.get("memory_feasible", False))
    entry.mem_gb = result.get("prefill_memory_gb", 0.0)
    entry.latency_ms = result.get("ttft_ms", result.get("prefill_latency_ms", 0.0))
    # 系统(单实例) prefill 吞吐: 用 effective_tps (含原始 seq_len), req/s = tps / 输入长 = batch/latency
    entry.sys_tps = result.get("prefill_effective_tps", 0.0)
    entry.per_card_tps = entry.sys_tps / entry.gpus if entry.gpus > 0 else 0.0
    entry.request_rate = (entry.sys_tps / spec.seq_len) if spec.seq_len > 0 else 0.0
    entry.status = _classify(feasible, entry.latency_ms, spec.slo_ms,
                             tps_ok=entry.sys_tps > 0)


def _fill_decode_metrics(entry: StageEntry, result: dict, spec: StageSpec) -> None:
    feasible = result.get("decode_memory_feasible",
                          result.get("memory_feasible", False))
    entry.mem_gb = result.get("decode_memory_gb", 0.0)

    use_mtp = bool(spec.mtp_param) and result.get("mtp_overall_tps", 0.0) > 0
    if use_mtp:
        entry.sys_tps = result.get("mtp_overall_tps", 0.0)
        entry.mtp_tps = entry.sys_tps
        tokens_per_step = result.get("mtp_expected_tokens_per_step", 1.0) or 1.0
        entry.latency_ms = result.get("mtp_step_latency_ms", 0.0) / tokens_per_step
    else:
        entry.sys_tps = result.get("overall_tps", 0.0)
        entry.latency_ms = result.get("decode_latency_ms", 0.0)   # TBT (单 token)

    entry.per_card_tps = entry.sys_tps / entry.gpus if entry.gpus > 0 else 0.0
    entry.request_rate = (entry.sys_tps / spec.output_length) if spec.output_length > 0 else 0.0
    entry.status = _classify(feasible, entry.latency_ms, spec.slo_ms,
                             tps_ok=entry.sys_tps > 0)


def _classify(feasible: bool, latency_ms: float, slo_ms: Optional[float],
              tps_ok: bool) -> str:
    if not feasible:
        return "OOM"
    if not tps_ok:
        return "OOM"   # 显存 OK 但吞吐为 0 (后端异常 fallback), 当不可用处理
    if slo_ms is not None and latency_ms > slo_ms:
        return "SLO"
    return "OK"


# ---------------------------------------------------------------------------
# 单阶段遍历
# ---------------------------------------------------------------------------

def _run_stage_sweep(common: CommonConfig, spec: StageSpec, stage: str,
                     num_layers: int) -> List[StageEntry]:
    is_decode = (stage == "decode")
    dpa_values = [False, True] if spec.dp_attention else [False]
    allow_cp_base = model_supports_cp(common.model)

    # 并行枚举用的 BS (dp≤bs 约束取最松): bs_list 取最大值; 搜索模式取 bs_max; off 用固定 batch_size。
    bs_list = sorted(set(spec.bs_list)) if spec.bs_list else None
    if bs_list:
        enum_bs = bs_list[-1]
    elif spec.bs_search == "off":
        enum_bs = spec.batch_size
    else:
        enum_bs = spec.bs_max

    tasks: List[tuple] = []
    for nodes in range(1, spec.nodes_budget + 1):
        total_gpus = nodes * common.gpus_per_node
        for dpa in dpa_values:
            allow_cp = allow_cp_base and dpa and (not is_decode)
            cfgs = enumerate_parallel_configs(
                total_gpus, num_layers, enum_bs,
                common.max_tp, common.max_pp, is_decode=is_decode,
                dp_attention=dpa, allow_cp=allow_cp,
                ep_strategy=common.ep_strategy)
            for cfg in cfgs:
                if bs_list:
                    for bs in bs_list:
                        if bs >= cfg[1]:           # dp ≤ bs (DPA 实例数 = dp)
                            tasks.append((common, spec, stage, nodes, cfg, dpa, bs))
                else:
                    tasks.append((common, spec, stage, nodes, cfg, dpa, None))

    total = len(tasks)
    num_workers = common.num_workers if common.num_workers > 0 else max(cpu_count(), 8)
    if bs_list:
        bs_desc = f"bs=list{bs_list}"
    elif spec.bs_search == "off":
        bs_desc = f"bs={spec.batch_size}(fixed)"
    else:
        bs_desc = f"bs={spec.bs_search}[{spec.bs_min}..{spec.bs_max}]"
    print(f"\n  [{stage}] 预算 {spec.nodes_budget} 节点  {bs_desc} "
          f"in={spec.seq_len} out={spec.output_length} "
          f"dpa={dpa_values} mtp={'on' if (is_decode and spec.mtp_param) else 'off'}  "
          f"→ {total} 个候选 × BS搜索, workers={num_workers}")

    entries: List[StageEntry] = []
    ok = oom = slo = 0
    best_rps = 0.0
    t0 = time.time()
    with Pool(processes=num_workers) as pool:
        it = pool.imap_unordered(_stage_worker, tasks)
        pbar = _tqdm(total=total, desc=f"{stage} sweep", unit="cfg",
                     ncols=100, dynamic_ncols=True) if _HAS_TQDM else None
        for entry in it:
            entries.append(entry)
            if entry.status == "OK":
                ok += 1
                best_rps = max(best_rps, entry.request_rate)
            elif entry.status == "SLO":
                slo += 1
            else:
                oom += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(OK=ok, SLO=slo, OOM=oom,
                                 best_rps=f"{best_rps:.3f}", refresh=False)
        if pbar is not None:
            pbar.close()

    dt = time.time() - t0
    print(f"  [{stage}] 完成: OK={ok} SLO={slo} OOM={oom}  耗时 {dt:.1f}s "
          f"({total/dt:.1f} cfg/s)" if dt > 0 else f"  [{stage}] 完成")
    return entries


# ---------------------------------------------------------------------------
# Excel 输出
# ---------------------------------------------------------------------------

_PREFILL_COLS = [
    ("Rank", "Rank"), ("DPA", "DP\nAttn"), ("Nodes", "Nodes"), ("GPUs", "GPUs"),
    ("TP", "TP"), ("DP", "DP"), ("PP", "PP"), ("EP", "EP"), ("CP", "CP"),
    ("BS", "BS"), ("In", "In\nLen"),
    ("TTFT_ms", "TTFT\n(ms)"),
    ("Sys_TPS", "Sys Prefill\nTPS"),
    ("PerCard_TPS", "Per-Card\nTPS"),
    ("Req_Rate", "Req Rate\n(req/s) ★"),
    ("Mem_GB", "Mem\n(GB/GPU)"),
    ("Status", "Status"),
]

_DECODE_COLS = [
    ("Rank", "Rank"), ("DPA", "DP\nAttn"), ("Nodes", "Nodes"), ("GPUs", "GPUs"),
    ("TP", "TP"), ("DP", "DP"), ("PP", "PP"), ("EP", "EP"),
    ("BS", "BS"), ("Ctx", "Ctx\nLen"), ("Out", "Out\nLen"), ("MTP", "MTP"),
    ("TBT_ms", "TBT\n(ms)"),
    ("Sys_TPS", "Sys Decode\nTPS"),
    ("PerCard_TPS", "Per-Card\nTPS"),
    ("Req_Rate", "Req Rate\n(req/s) ★"),
    ("Mem_GB", "Mem\n(GB/GPU)"),
    ("Status", "Status"),
]

_BALANCE_COLS = [
    ("Pair", "P/D\nTop"),
    ("P_Cfg", "Prefill Config"),
    ("P_GPUs", "P GPUs\n/inst"),
    ("P_ReqRate", "P Req Rate\n(req/s/inst)"),
    ("D_Cfg", "Decode Config"),
    ("D_GPUs", "D GPUs\n/inst"),
    ("D_ReqRate", "D Req Rate\n(req/s/inst)"),
    ("Ratio", "理想 P:D\n实例配比"),
    ("Ratio_Int", "整数近似\nP:D"),
    ("Note", "说明"),
]


def _prefill_row(e: StageEntry, rank) -> dict:
    return {
        "Rank": rank, "DPA": "ON" if e.dp_attention else "OFF",
        "Nodes": e.nodes, "GPUs": e.gpus,
        "TP": e.tp, "DP": e.dp, "PP": e.pp, "EP": e.ep, "CP": e.cp,
        "BS": e._bs, "In": e._inlen,
        "TTFT_ms": round(e.latency_ms, 3),
        "Sys_TPS": round(e.sys_tps, 1),
        "PerCard_TPS": round(e.per_card_tps, 1),
        "Req_Rate": round(e.request_rate, 4),
        "Mem_GB": round(e.mem_gb, 2),
        "Status": e.status,
    }


def _decode_row(e: StageEntry, rank) -> dict:
    return {
        "Rank": rank, "DPA": "ON" if e.dp_attention else "OFF",
        "Nodes": e.nodes, "GPUs": e.gpus,
        "TP": e.tp, "DP": e.dp, "PP": e.pp, "EP": e.ep,
        "BS": e._bs, "Ctx": e._inlen, "Out": e._outlen,
        "MTP": e._mtp or "-",
        "TBT_ms": round(e.latency_ms, 4),
        "Sys_TPS": round(e.sys_tps, 1),
        "PerCard_TPS": round(e.per_card_tps, 1),
        "Req_Rate": round(e.request_rate, 4),
        "Mem_GB": round(e.mem_gb, 2),
        "Status": e.status,
    }


def _write_leaderboard(ws, entries: List[StageEntry], cols, row_fn, meta: str):
    """写一个阶段榜 sheet: 第1行 meta, 第2行表头, 数据从第3行。OK 在前, SLO/OOM 灰显在后。"""
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(cols))
    mc = ws.cell(row=1, column=1, value=meta)
    mc.font = Font(bold=True, size=10, color="FFFFFF")
    mc.fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    mc.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 20

    for ci, (_, label) in enumerate(cols, 1):
        c = ws.cell(row=2, column=ci, value=label)
        c.font = Font(bold=True, size=10, color="FFFFFF")
        c.fill = _BLUE_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = _THIN_BORDER
    ws.row_dimensions[2].height = 36

    ok = sorted([e for e in entries if e.status == "OK"],
                key=lambda e: -e.request_rate)
    rest = sorted([e for e in entries if e.status != "OK"],
                  key=lambda e: (e.status, -e.request_rate))

    r = 3
    for rank, e in enumerate(ok, 1):
        data = row_fn(e, rank)
        fill = _GOLD_FILL if rank == 1 else _SILVER_FILL if rank == 2 else _BRONZE_FILL if rank == 3 else None
        for ci, (key, _) in enumerate(cols, 1):
            cell = ws.cell(row=r, column=ci, value=data[key])
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if fill and key in ("Rank", "Req_Rate"):
                cell.fill = fill; cell.font = Font(bold=True)
            elif key == "Req_Rate":
                cell.font = Font(bold=True)
        r += 1

    for e in rest:
        data = row_fn(e, "-")
        for ci, (key, _) in enumerate(cols, 1):
            cell = ws.cell(row=r, column=ci, value=data[key])
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.fill = _GREY_FILL if e.status == "SLO" else _RED_FILL
        r += 1

    _auto_col_width(ws, cols, start_row=3)
    ws.freeze_panes = "A3"


def _ratio_int(d_per_p: float) -> str:
    """把 1 : d_per_p (P:D) 近似成最简单的整数比 (1:N 或 N:1)。"""
    if d_per_p <= 0 or math.isinf(d_per_p) or math.isnan(d_per_p):
        return "-"
    if d_per_p >= 1:
        return f"1 : {max(1, round(d_per_p))}"      # 1 个 P 配 ~N 个 D
    return f"{max(1, round(1.0 / d_per_p))} : 1"     # P 比 D 慢 (罕见)


def _write_balance(ws, p_top: List[StageEntry], d_top: List[StageEntry]):
    _write_header(ws, _BALANCE_COLS, _GREEN_FILL)
    r = 2
    k = min(len(p_top), len(d_top))
    for i in range(k):
        p, d = p_top[i], d_top[i]
        p_rps, d_rps = p.request_rate, d.request_rate
        d_per_p = (p_rps / d_rps) if d_rps > 0 else float("inf")   # 1 个 P 配几个 D
        if d_rps <= 0:
            ratio_str, note = "-", "D 请求率为 0"
        else:
            ratio_str = f"1 : {d_per_p:.2f}"
            note = "P 供给 = D 消化 时配比 (n_P×P_rps = n_D×D_rps)"
        data = {
            "Pair": f"Top{i+1}",
            "P_Cfg": f"P{p.nodes}n_" + _fmt_parallel(p.tp, p.dp, p.pp, p.ep, p.cp),
            "P_GPUs": p.gpus,
            "P_ReqRate": round(p_rps, 4),
            "D_Cfg": f"D{d.nodes}n_" + _fmt_parallel(d.tp, d.dp, d.pp, d.ep, 1, include_cp=False),
            "D_GPUs": d.gpus,
            "D_ReqRate": round(d_rps, 4),
            "Ratio": ratio_str,
            "Ratio_Int": _ratio_int(d_per_p),
            "Note": note,
        }
        for ci, (key, _) in enumerate(_BALANCE_COLS, 1):
            cell = ws.cell(row=r, column=ci, value=data[key])
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
        r += 1
    _auto_col_width(ws, _BALANCE_COLS, start_row=2)


def _bs_meta(spec: StageSpec) -> str:
    """meta 行的 BS 口径描述: 显式列表 / 固定值 / 搜索区间。"""
    if spec.bs_list:
        return "list" + str(sorted(set(spec.bs_list)))
    if spec.bs_search == "off":
        return str(spec.batch_size)
    return f"{spec.bs_search}[{spec.bs_min}..{spec.bs_max}]"


def save_separate_summary(
    common: CommonConfig,
    p_spec: StageSpec, d_spec: StageSpec,
    p_entries: List[StageEntry], d_entries: List[StageEntry],
    output_path: str, balance_topk: int = 1,
    validate_des: bool = True, validate_top_k: int = 1,
    validate_rate_factors: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    validate_num_requests: int = 500,
) -> None:
    # 给每个 entry 挂上展示用的 bs/len/mtp (bs 取每候选 BS 搜索胜出值, 不再是固定 spec.batch_size)
    for e in p_entries:
        e._bs, e._inlen, e._outlen, e._mtp = e.batch_size, p_spec.seq_len, p_spec.output_length, ""
    for e in d_entries:
        e._bs, e._inlen, e._outlen, e._mtp = e.batch_size, d_spec.seq_len, d_spec.output_length, d_spec.mtp_param

    wb = Workbook()

    ws_p = wb.active
    ws_p.title = "prefill_leaderboard"
    p_meta = (f"PREFILL  model={common.model}  gpu={common.gpu}  budget={p_spec.nodes_budget}n  "
              f"bs={_bs_meta(p_spec)}  in={p_spec.seq_len}  chunk={p_spec.chunk_prefill_size}  "
              f"SLO(TTFT)={p_spec.slo_ms}  prefix={common.prefix_cache_hit_rate}")
    _write_leaderboard(ws_p, p_entries, _PREFILL_COLS, _prefill_row, p_meta)

    ws_d = wb.create_sheet("decode_leaderboard")
    d_meta = (f"DECODE  model={common.model}  gpu={common.gpu}  budget={d_spec.nodes_budget}n  "
              f"bs={_bs_meta(d_spec)}  ctx={d_spec.seq_len}  out={d_spec.output_length}  "
              f"mtp={d_spec.mtp_param or '-'}  SLO(TBT)={d_spec.slo_ms}")
    _write_leaderboard(ws_d, d_entries, _DECODE_COLS, _decode_row, d_meta)

    ws_b = wb.create_sheet("balance")
    p_ok = sorted([e for e in p_entries if e.status == "OK"], key=lambda e: -e.request_rate)
    d_ok = sorted([e for e in d_entries if e.status == "OK"], key=lambda e: -e.request_rate)
    _write_balance(ws_b, p_ok[:balance_topk], d_ok[:balance_topk])

    des_rows: List[dict] = []
    if validate_des and (p_ok or d_ok):
        print(f"\n  [DES] 排队精校 Top{validate_top_k} (num_requests={validate_num_requests}) ...")
        t0 = time.time()
        des_rows = _run_des_validation(common, p_spec, d_spec, p_ok, d_ok,
                                       validate_top_k, validate_rate_factors,
                                       validate_num_requests)
        print(f"  [DES] 完成, 耗时 {time.time()-t0:.1f}s")
        _write_des(wb.create_sheet("des_validation"), des_rows)

    wb.save(output_path)
    print(f"\n总表已保存: {output_path}")
    print(f"  - prefill_leaderboard: {len(p_ok)} OK / {len(p_entries)} 总")
    print(f"  - decode_leaderboard:  {len(d_ok)} OK / {len(d_entries)} 总")
    _print_balance(p_ok[:balance_topk], d_ok[:balance_topk])
    _print_des(des_rows)


def _print_balance(p_top, d_top):
    if not p_top or not d_top:
        print("  (P 或 D 无 OK 配置, 无法配平)")
        return
    print(f"\n{'='*96}")
    print("  配平 (理想 P:D 实例配比, 1 个 Prefill 实例可喂几个 Decode 实例)")
    print(f"{'='*96}")
    for i in range(min(len(p_top), len(d_top))):
        p, d = p_top[i], d_top[i]
        p_cfg = f"P{p.nodes}n_" + _fmt_parallel(p.tp, p.dp, p.pp, p.ep, p.cp)
        d_cfg = f"D{d.nodes}n_" + _fmt_parallel(d.tp, d.dp, d.pp, d.ep, 1, include_cp=False)
        d_per_p = (p.request_rate / d.request_rate) if d.request_rate > 0 else float("inf")
        print(f"  Top{i+1}: {p_cfg} (P_rps={p.request_rate:.4f}) : "
              f"{d_cfg} (D_rps={d.request_rate:.4f})  →  1 : {d_per_p:.2f}  "
              f"[{_ratio_int(d_per_p)}]")
    print(f"{'='*96}\n")


# ---------------------------------------------------------------------------
# DES 排队精校 (复用 serving_simulator; 见设计文档 §10)
# ---------------------------------------------------------------------------

# 单池 mixed DES 角色纯化: prefill 池 output_len=1 → prefill-bound (TTFT p99 准);
#   decode 池 output_len=真实 → 长输出摊薄 prefill, 近 decode-bound (TPOT p99 准, 饱和 QPS 偏保守)。

def validate_entry_with_des(
    common: CommonConfig, spec: StageSpec, stage: str, entry: StageEntry,
    rate_factors: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    num_requests: int = 500,
) -> dict:
    """对单个选定工作点跑角色纯化 DES, 返回校验行 dict。

    判据 (设计文档 §10.2):
      - 取 req_rate 最接近 predicted_rps 的那点 p99 → SLO PASS/FAIL;
      - sat_qps = 饱和前最高 achieved_qps; sat_qps ≥ predicted_rps → 静态保守可信, 否则高估。
    """
    try:
        from .serving_simulator import ServingConfig, sweep_req_rate
    except ImportError:
        from serving_simulator import ServingConfig, sweep_req_rate

    is_p = (stage == "prefill")
    predicted = max(entry.request_rate, 1e-9)
    base = ServingConfig(
        req_rate=predicted,
        model=common.model, gpu=common.gpu,
        tp=entry.tp, dp=entry.dp, pp=entry.pp, ep=entry.ep, cp=entry.cp,
        nodes=entry.nodes, dp_attention=entry.dp_attention,
        input_len=spec.seq_len,
        output_len=1 if is_p else spec.output_length,
        max_running_requests=max(1, entry.batch_size),
        chunked_prefill_size=spec.chunk_prefill_size,
        kv_cache_dtype=common.kv_cache_dtype,
        attention_compute_dtype=common.attention_compute_dtype,
        fuse_moe=common.fuse_moe,
        gpus_per_node=common.gpus_per_node,
        nvlink_domain_size=common.nvlink_domain_size,
        intra_node_bw_gbs=common.intra_node_bw_gbs,
        inter_node_bw_gbs=common.inter_node_bw_gbs,
        num_requests=num_requests,
        analyzer_kwargs={
            "device_type": common.device_type,
            "util_tensor_core": common.util_tensor_core,
            "util_vector": common.util_vector,
            "util_sfu": common.util_sfu,
            "util_bandwidth": common.util_bandwidth,
        },
    )
    rates = sorted({round(f * predicted, 6) for f in rate_factors if f > 0})
    metrics = sweep_req_rate(base, rates, stop_on_saturate=True, verbose=False)

    # 取最接近 predicted 的那点 p99
    at = min(metrics, key=lambda m: abs(m.req_rate - predicted)) if metrics else None
    p99 = (at.ttft_ms_p99 if is_p else at.tpot_ms_p99) if at else 0.0
    # 饱和 QPS: 饱和前最高 achieved_qps (无非饱和点则取整体最高)
    unsat = [m.achieved_qps for m in metrics if not m.saturated]
    sat_qps = max(unsat) if unsat else (max((m.achieved_qps for m in metrics), default=0.0))

    slo = spec.slo_ms
    slo_ok = (slo is None) or (p99 <= slo)
    sat_ok = sat_qps >= predicted
    if not slo_ok:
        verdict = "FAIL(p99 超SLO/单点高估)"
    elif not sat_ok:
        verdict = "WARN(饱和QPS<预测)"
    else:
        verdict = "PASS"

    return {
        "stage": stage,
        "cfg": _fmt_parallel(entry.tp, entry.dp, entry.pp, entry.ep,
                             entry.cp, include_cp=is_p),
        "nodes": entry.nodes, "bs": entry.batch_size,
        "predicted_rps": round(predicted, 4),
        "metric": "TTFT_p99" if is_p else "TPOT_p99",
        "p99_ms": round(p99, 3),
        "slo_ms": slo,
        "sat_qps": round(sat_qps, 4),
        "verdict": verdict,
    }


_DES_COLS = [
    ("stage", "Stage"), ("cfg", "Parallel"), ("nodes", "Nodes"), ("bs", "BS*"),
    ("predicted_rps", "Predicted\nReq Rate"),
    ("metric", "Metric"), ("p99_ms", "DES p99\n(ms)"), ("slo_ms", "SLO\n(ms)"),
    ("sat_qps", "Sat QPS\n(DES)"), ("verdict", "Verdict"),
]


def _write_des(ws, rows: List[dict]):
    _write_header(ws, _DES_COLS, _GREEN_FILL)
    r = 2
    for data in rows:
        for ci, (key, _) in enumerate(_DES_COLS, 1):
            val = data.get(key)
            cell = ws.cell(row=r, column=ci, value=("-" if val is None else val))
            cell.border = _THIN_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if key == "verdict":
                if data["verdict"].startswith("PASS"):
                    cell.fill = _GREEN_FILL
                elif data["verdict"].startswith("WARN"):
                    cell.fill = _GREY_FILL
                else:
                    cell.fill = _RED_FILL
        r += 1
    _auto_col_width(ws, _DES_COLS, start_row=2)


def _print_des(rows: List[dict]):
    if not rows:
        return
    print(f"\n{'='*100}")
    print("  DES 排队精校 (角色纯化单池仿真; 验证单点估算选出的 BS 在真实并发下扛不扛得住)")
    print(f"{'='*100}")
    for d in rows:
        slo_s = f"{d['slo_ms']}" if d['slo_ms'] is not None else "—"
        print(f"  [{d['stage']:>7}] {d['cfg']}  {d['nodes']}n bs={d['bs']:>4}  "
              f"pred_rps={d['predicted_rps']:.4f}  {d['metric']}={d['p99_ms']:.2f}ms "
              f"(SLO={slo_s})  sat_qps={d['sat_qps']:.4f}  → {d['verdict']}")
    print(f"{'='*100}\n")


def _run_des_validation(common, p_spec, d_spec, p_ok, d_ok,
                        top_k, rate_factors, num_requests) -> List[dict]:
    """对 P/D 各 Top-K OK 工作点跑 DES, 返回校验行列表。"""
    rows: List[dict] = []
    for e in p_ok[:top_k]:
        rows.append(validate_entry_with_des(common, p_spec, "prefill", e,
                                             rate_factors, num_requests))
    for e in d_ok[:top_k]:
        rows.append(validate_entry_with_des(common, d_spec, "decode", e,
                                             rate_factors, num_requests))
    return rows


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _parse_bs_list(v) -> Optional[List[int]]:
    """接受 None / int 列表 / 逗号(或空格)分隔字符串 "8,16,32" → 排序去重 int 列表。"""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        vals = [int(x) for x in v.replace(",", " ").split()]
    else:
        vals = [int(x) for x in v]
    return sorted(set(vals)) or None


def pd_separate_sweep(
    # 节点预算 (分开给定, 各自遍历 1..budget)
    prefill_nodes: int = 2,
    decode_nodes: int = 2,
    model: str = "minimax_m2.5_w8a8",
    gpu: str = "NMZ",
    # --- Prefill 独立参数 ---
    prefill_batch_size: int = 256,
    prefill_seq_len: int = 4096,
    prefill_output_length: int = 1,
    prefill_chunk_prefill_size: int = 4096,
    prefill_dp_attention: bool = False,
    prefill_tbo: bool = False,
    prefill_sbo: bool = False,
    prefill_moe_dp_size: int = 1,
    prefill_moe_dense_tp_size: Optional[int] = None,
    prefill_moe_a2a_backend: str = "none",
    prefill_ttft_slo_ms: Optional[float] = None,
    prefill_bs_search: str = "slo",
    prefill_bs_min: int = 1,
    prefill_bs_max: int = 512,
    prefill_bs_step: int = 2,
    # 显式 BS 列表 (非空时优先于 bs_search): "8,16,32" 或 Python list; 每个 BS 独立成榜单一行
    prefill_bs_list: Optional[str] = None,
    # --- Decode 独立参数 ---
    decode_batch_size: int = 256,
    decode_seq_len: int = 4096,
    decode_output_length: int = 1500,
    decode_chunk_prefill_size: int = 4096,
    decode_dp_attention: bool = False,
    decode_tbo: bool = False,
    decode_sbo: bool = False,
    decode_moe_dp_size: int = 1,
    decode_moe_dense_tp_size: Optional[int] = None,
    decode_moe_a2a_backend: str = "none",
    decode_mtp_param: str = "",
    decode_mtp_acceptance_rate: float = 0.8,
    decode_tbt_slo_ms: Optional[float] = None,
    decode_bs_search: str = "slo",
    decode_bs_min: int = 1,
    decode_bs_max: int = 512,
    decode_bs_step: int = 2,
    decode_bs_list: Optional[str] = None,   # 同 prefill_bs_list
    # --- 共享参数 ---
    device_type: Optional[str] = None,   # None = 跟随 gpu; 显式传则覆盖 (透传 model_analyzer)
    gpus_per_node: int = 8,
    nvlink_domain_size: int = 8,
    intra_node_bw_gbs: float = 140.0,
    inter_node_bw_gbs: float = 50.0,
    memory_ratio: float = 0.9,
    sequence_parallel: bool = False,
    pp_async_batch_depth: int = 0,
    fuse_moe: bool = True,
    attention_compute_dtype: str = "fp8",
    kv_cache_dtype: str = "fp8",
    num_images: int = 0,
    image_height: int = 448,
    image_width: int = 448,
    prefix_cache_hit_rate: float = 0.0,
    ep_strategy: str = "auto",
    max_tp: int = 8,
    max_pp: int = 8,
    num_workers: int = 0,
    output_dir: str = "",
    model_json_path: Optional[str] = None,
    util_profile: Optional[str] = None,
    # Roofline 全局利用率标量 (-1.0 = 用 device JSON utilization_factors, 透传 model_analyzer)
    util_tensor_core: float = -1.0,
    util_vector: float = -1.0,
    util_sfu: float = -1.0,
    util_bandwidth: float = -1.0,
    generate_detail_excel: bool = False,
    balance_topk: int = 1,
    # --- DES 排队精校 (默认开) ---
    validate_des: bool = True,
    validate_top_k: int = 1,
    validate_num_requests: int = 500,
) -> Tuple[List[StageEntry], List[StageEntry], str]:
    """
    PD 分离独立遍历主入口 (P / D 各自寻优, 按请求率排序, 末尾给理想 P:D 实例配比)。

    Returns: (prefill_entries, decode_entries, summary_excel_path)
    """
    common = CommonConfig(
        model=model, gpu=gpu, device_type=device_type, gpus_per_node=gpus_per_node,
        nvlink_domain_size=nvlink_domain_size,
        intra_node_bw_gbs=intra_node_bw_gbs, inter_node_bw_gbs=inter_node_bw_gbs,
        memory_ratio=memory_ratio, sequence_parallel=sequence_parallel,
        pp_async_batch_depth=pp_async_batch_depth, fuse_moe=fuse_moe,
        attention_compute_dtype=attention_compute_dtype, kv_cache_dtype=kv_cache_dtype,
        num_images=num_images, image_height=image_height, image_width=image_width,
        prefix_cache_hit_rate=prefix_cache_hit_rate, ep_strategy=ep_strategy,
        max_tp=max_tp, max_pp=max_pp, num_workers=num_workers,
        output_dir=output_dir, model_json_path=model_json_path,
        util_profile=util_profile,
        util_tensor_core=util_tensor_core, util_vector=util_vector,
        util_sfu=util_sfu, util_bandwidth=util_bandwidth,
        generate_detail_excel=generate_detail_excel,
    )
    p_spec = StageSpec(
        nodes_budget=prefill_nodes, batch_size=prefill_batch_size,
        seq_len=prefill_seq_len, output_length=prefill_output_length,
        chunk_prefill_size=prefill_chunk_prefill_size,
        dp_attention=prefill_dp_attention, tbo=prefill_tbo, sbo=prefill_sbo,
        moe_dp_size=prefill_moe_dp_size, moe_dense_tp_size=prefill_moe_dense_tp_size,
        moe_a2a_backend=prefill_moe_a2a_backend, mtp_param="", slo_ms=prefill_ttft_slo_ms,
        bs_search=prefill_bs_search, bs_min=prefill_bs_min,
        bs_max=prefill_bs_max, bs_step=prefill_bs_step,
        bs_list=_parse_bs_list(prefill_bs_list),
    )
    d_spec = StageSpec(
        nodes_budget=decode_nodes, batch_size=decode_batch_size,
        seq_len=decode_seq_len, output_length=decode_output_length,
        chunk_prefill_size=decode_chunk_prefill_size,
        dp_attention=decode_dp_attention, tbo=decode_tbo, sbo=decode_sbo,
        moe_dp_size=decode_moe_dp_size, moe_dense_tp_size=decode_moe_dense_tp_size,
        moe_a2a_backend=decode_moe_a2a_backend,
        mtp_param=decode_mtp_param, mtp_acceptance_rate=decode_mtp_acceptance_rate,
        slo_ms=decode_tbt_slo_ms,
        bs_search=decode_bs_search, bs_min=decode_bs_min,
        bs_max=decode_bs_max, bs_step=decode_bs_step,
        bs_list=_parse_bs_list(decode_bs_list),
    )

    num_layers = _load_num_layers(_FakePDConfig(common))

    if not output_dir:
        name = (f"pd_separate_{model}_P{prefill_nodes}n_D{decode_nodes}n_{gpu}"
                f"_Pbs{prefill_batch_size}in{prefill_seq_len}"
                f"_Dbs{decode_batch_size}out{decode_output_length}")
        output_dir = f"raw_data/{name}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    common.output_dir = str(out)

    print(f"\n{'='*72}")
    print(f"  PD Separate Sweep: {model}  GPU={gpu}")
    print(f"  Prefill 预算={prefill_nodes}n  Decode 预算={decode_nodes}n  "
          f"num_hidden_layers={num_layers}")
    print(f"{'='*72}")

    p_entries = _run_stage_sweep(common, p_spec, "prefill", num_layers)
    d_entries = _run_stage_sweep(common, d_spec, "decode", num_layers)

    summary_path = str(out / (
        f"pd_separate_summary_{model}_P{prefill_nodes}n_D{decode_nodes}n_{gpu}.xlsx"))
    save_separate_summary(common, p_spec, d_spec, p_entries, d_entries,
                          summary_path, balance_topk=balance_topk,
                          validate_des=validate_des, validate_top_k=validate_top_k,
                          validate_num_requests=validate_num_requests)
    return p_entries, d_entries, summary_path


class _FakePDConfig:
    """适配 pd_sweep._load_num_layers (它只读 .model 与 .model_json_path)。"""
    def __init__(self, common: CommonConfig):
        self.model = common.model
        self.model_json_path = common.model_json_path


if __name__ == "__main__":
    run_cli(
        pd_separate_sweep,
        type_overrides={
            "prefill_moe_dense_tp_size": int, "decode_moe_dense_tp_size": int,
            "prefill_ttft_slo_ms": float, "decode_tbt_slo_ms": float,
            "prefix_cache_hit_rate": float,
        },
        prog="python -m model_analyzer.pd_separate_sweep",
        description="PD 分离独立遍历 (P/D 各自寻优 + 请求率配平表)")
