import os
import sys
import argparse
import inspect
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("model_analyzer")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(os.environ.get("LLM_PERF_LOG_LEVEL", "INFO").upper())

# add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

try:
    import llm_perf_core
    from llm_perf_core import analyze_model_binding
except ImportError as e:
    # 保留原始报错: 除"未编译"外, 还可能是 op_analyzer SDK ABI 不匹配 (pybind init
    # 的 abi_version 校验) 或 libop_analyzer.so/libonnxruntime.so 加载失败。
    raise ImportError(
        f"llm_perf_core 后端加载失败 (原始错误: {e})。"
        "若为 '模块未找到' 请先运行: bash build.sh; "
        "若为 'ABI mismatch' 说明 libop_analyzer.so 与编译时的 op_analyzer_api.h 版本不一致, "
        "需重编 model_analyzer。"
    ) from e

try:
    from .summary_generator import generate_excel
except ImportError:
    from summary_generator import generate_excel


# model name -> JSON config mapping
MODEL_CONFIG_MAP = {
    "deepseekv3": "deepseek_v3.json",
    "deepseek_v3": "deepseek_v3.json",
    "deepseekr1": "deepseek_r1.json",
    "deepseek_r1": "deepseek_r1.json",
    "deepseek-r1": "deepseek_r1.json",
    "deepseekv3_2": "deepseek_v3_2.json",
    "deepseek_v3_2": "deepseek_v3_2.json",
    "deepseek_v3.2": "deepseek_v3_2.json",
    "deepseek-v3.2": "deepseek_v3_2.json",
    "deepseekv4": "deepseek_v4.json",
    "deepseek_v4": "deepseek_v4.json",
    "deepseek_v4_5t": "deepseek_v4_5T.json",
    "deepseek_v4_10t": "deepseek_v4_10T.json",
    "deepseek_v4_flash": "deepseek_v4_flash.json",
    "deepseekv4_flash": "deepseek_v4_flash.json",
    "deepseek-v4-flash": "deepseek_v4_flash.json",
    "v4_flash": "deepseek_v4_flash.json",
    "deepseek_v4_pro_channel_fp8_w8a8": "deepseek_v4_pro_fp8_w8a8.json",
    "deepseek_v4_pro_fp8_w8a8": "deepseek_v4_pro_fp8_w8a8.json",
    "deepseek-v4-pro-channel-fp8-w8a8": "deepseek_v4_pro_fp8_w8a8.json",
    "deepseek-v4-pro-fp8-w8a8": "deepseek_v4_pro_fp8_w8a8.json",
    "llama2": "llama2_7b.json",
    "llama2_7b": "llama2_7b.json",
    "minimax_m2": "minimax_m2.json",
    "minimax_m2.5": "minimax_m2.json",
    "minimax_m2_w8a8": "minimax_m2_w8a8.json",
    "minimax_m2.5_w8a8": "minimax_m2_w8a8.json",
    "mimo_v2_flash_channel_fp8_w8a8": "mimo_v2_flash.json",
    "mimo_v2_channel_fp8_w8a8": "mimo_v2_flash.json",
    "mimo2_channel_fp8_w8a8": "mimo_v2_flash.json",
    "mimo_v2_5_pro": "mimo_v2_5_pro.json",
    "mimo_v2.5_pro": "mimo_v2_5_pro.json",
    "mimo_v2_5_pro_fp8": "mimo_v2_5_pro.json",
    "mimo_v2.5-pro": "mimo_v2_5_pro.json",
    "mimo2_5_pro": "mimo_v2_5_pro.json",
    "mimo_v2_5_pro_channel_fp8_w8a8": "mimo_v2_5_pro_channel_fp8_w8a8.json",
    "mimo_v2.5_pro_channel_fp8_w8a8": "mimo_v2_5_pro_channel_fp8_w8a8.json",
    "mimo2_5_pro_channel_fp8_w8a8": "mimo_v2_5_pro_channel_fp8_w8a8.json",
    "step_3_5_flash": "step3p5_flash.json",
    "step_3_5_flash_fp8": "step3p5_flash_fp8.json",
    "step_3_7_flash": "step3p7_flash.json",
    "step_3_7_flash_fp8": "step3p7_flash_fp8.json",
    "qwen2.5_72b": "Qwen2.5-72B.json",
    "qwen2.5-72b": "Qwen2.5-72B.json",
    "qwen2": "Qwen2.5-72B.json",
    "qwen3_32b": "Qwen3-32B.json",
    "qwen3-32b": "Qwen3-32B.json",
    "qwen3": "Qwen3-32B.json",
    "gpt_moe": "gpt_moe.json",
    "gpt-moe": "gpt_moe.json",
    "gpt2_moe": "gpt_moe.json",
    "gpt_moe_fp32": "gpt_moe_fp32.json",
    "gpt2_moe_fp32": "gpt_moe_fp32.json",
    "gpt_moe_w8w8": "gpt_moe_int8_w8a8.json",
    "gpt_moe_int8_w8a8": "gpt_moe_int8_w8a8.json",
    "gpt2_moe_int8_w8a8": "gpt_moe_int8_w8a8.json",
    "qwen3_moe_30b_a3b": "qwen3_moe_30b_a3b.json",
    "qwen3-moe-30b-a3b": "qwen3_moe_30b_a3b.json",
    "qwen3_moe": "qwen3_moe_30b_a3b.json",
    "qwen3-30b": "qwen3_moe_30b_a3b.json",
    "qwen3_vl_moe_235b": "qwen3_vl_moe_235b.json",
    "qwen3-vl-235b": "qwen3_vl_moe_235b.json",
    "qwen3_vl_moe": "qwen3_vl_moe_235b.json",
    "qwen3-vl-moe": "qwen3_vl_moe_235b.json",
    "qwen3-vl-235b-a22b": "qwen3_vl_moe_235b.json",
    "qwen35_397b": "qwen35_397b.json",
    "qwen3.5_397b": "qwen35_397b.json",
    "qwen3.5-397b": "qwen35_397b.json",
    "qwen3_5_moe": "qwen35_397b.json",
    "qwen35_397b_fp8": "qwen35_397b_fp8.json",
    "qwen3.5_397b_fp8": "qwen35_397b_fp8.json",
    "qwen3.5-397b-a17b-fp8": "qwen35_397b_fp8.json",
    "qwen35_27b": "qwen35_27b.json",
    "qwen3.5_27b": "qwen35_27b.json",
    "qwen3.5-27b": "qwen35_27b.json",
    "qwen3_5": "qwen35_27b.json",          # "qwen3_5" = 27B dense; "qwen3_5_moe" = 397B
    "qwen35_35b_a3b": "qwen35_35b_a3b.json",
    "qwen3.5_35b_a3b": "qwen35_35b_a3b.json",
    "qwen3.5-35b-a3b": "qwen35_35b_a3b.json",
    "qwen35_35b": "qwen35_35b_a3b.json",
    "qwen3.5-35b": "qwen35_35b_a3b.json",
    "qwen36_35b_a3b": "qwen36_35b_a3b.json",
    "qwen3.6_35b_a3b": "qwen36_35b_a3b.json",
    "qwen3.6-35b-a3b": "qwen36_35b_a3b.json",
    "qwen36_35b": "qwen36_35b_a3b.json",
    "qwen3.6-35b": "qwen36_35b_a3b.json",
    "glm5": "glm5.json",
    "glm_moe_dsa": "glm5.json",
    "glm5_fp8": "glm5_fp8.json",
    "glm5-fp8": "glm5_fp8.json",
    "kimi_k25": "kimi25.json",
    "kimi25": "kimi25.json",
    "kimi-k2.5": "kimi25.json",
    "kimi_k2_6_w4a16": "kimi_k2_6_w4a16.json",
    "kimi26_w4a16": "kimi_k2_6_w4a16.json",
    "kimi-k2.6-w4a16": "kimi_k2_6_w4a16.json",
    # Kimi-K2.7 W4A16 与 K2.6 同模型同 config (alias)
    "kimi_k2_7_w4a16": "kimi_k2_6_w4a16.json",
    "kimi27_w4a16": "kimi_k2_6_w4a16.json",
    "kimi-k2.7-w4a16": "kimi_k2_6_w4a16.json",
    "kimi_k2_6_w8a8": "kimi_k2_6_w8a8.json",
    "kimi26_w8a8": "kimi_k2_6_w8a8.json",
    "kimi-k2.6-w8a8": "kimi_k2_6_w8a8.json",
    "kimi_k2": "kimi_k2.json",
    "kimi2": "kimi_k2.json",
    "kimi-k2": "kimi_k2.json",
    "hunyuan_v3": "hunyuan_v3.json",
    "hunyuan-v3": "hunyuan_v3.json",
    "hy_v3": "hunyuan_v3.json",
    "hy3": "hunyuan_v3.json",
    "hyv3": "hunyuan_v3.json",
    "hunyuan_v3_fp8": "hunyuan_v3_fp8.json",
    "hunyuan-v3-fp8": "hunyuan_v3_fp8.json",
    "hy_v3_fp8": "hunyuan_v3_fp8.json",
    "hy3_fp8": "hunyuan_v3_fp8.json",
    "hyv3_fp8": "hunyuan_v3_fp8.json",
    "hunyuan_v3_preview_w8a8": "hunyuan_v3_int8_w8a8.json",
    "hunyuan-v3-preview-w8a8": "hunyuan_v3_int8_w8a8.json",
    "hy3_preview_w8a8": "hunyuan_v3_int8_w8a8.json",
    "hyv3_preview_w8a8": "hunyuan_v3_int8_w8a8.json",
    "hy3_int8_w8a8": "hunyuan_v3_int8_w8a8.json",
    "hunyuan_v3_int8_w8a8": "hunyuan_v3_int8_w8a8.json",
}


def resolve_model_json(model: str, model_json_path: Optional[str] = None) -> str:
    """
    把模型名 (别名 / model_type / 配置文件路径) 统一解析为 JSON 绝对路径。

    优先级:
      1. 显式 model_json_path 非空 → 直接用 (跳过 MAP)
      2. model 本身是已存在的文件路径 → 直接用
      3. MODEL_CONFIG_MAP[model.lower()] 命中 → 拼接 wheel/源码 configs 目录

    """
    if model_json_path:
        path = model_json_path
    elif model and os.path.isfile(model):
        path = model
    else:
        wheel_config_dir = Path(__file__).resolve().parent / "data" / "model_configs"
        config_dir = wheel_config_dir if wheel_config_dir.is_dir() else _PROJECT_ROOT / "configs"
        config_name = MODEL_CONFIG_MAP.get(model.lower()) if model else None
        if config_name is None:
            raise ValueError(
                f"Unknown model: {model!r}. "
                f"Available aliases: {sorted(MODEL_CONFIG_MAP.keys())}. "
                f"Or pass model_json_path to specify a custom config."
            )
        path = str(config_dir / config_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model config not found: {path}")
    return path


def _compute_avg_ttft_ns(
    prefill_latency_ns: float,
    t_chunk_ns: float,
    t_remainder_ns: float,
    num_chunks_full: int,
    remainder_size: int,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    dp_size: int = 1,
    pp_size: int = 1,
    attn_slices_latency_ns: float = 0.0,
) -> float:
    dp = max(1, dp_size)
    per_worker_reqs = max(1, (batch_size + dp - 1) // dp)

  
    warmup_ns = max(0, pp_size - 1) * max(t_chunk_ns, t_remainder_ns)

    def _pp_fold_attn(base_ns: float) -> float:
  
        if pp_size <= 1 or base_ns <= 0:
            return base_ns
        n_mb = max(1, num_chunks_full + (1 if remainder_size > 0 else 0))
        return base_ns / pp_size + (pp_size - 1) * (attn_slices_latency_ns / n_mb) / pp_size

   
    if num_chunks_full == 0 and t_remainder_ns == 0:
        return float(prefill_latency_ns)
    if num_chunks_full == 0:
        return float(t_remainder_ns)

   
    avg_reqs_factor = (per_worker_reqs + 1) / 2.0
    attn_ttft_ns = (avg_reqs_factor * attn_slices_latency_ns / per_worker_reqs
                    if attn_slices_latency_ns > 0 else 0.0)
    attn_ttft_ns = _pp_fold_attn(attn_ttft_ns)

    # 场景 B: chunk_size < seq_len, 单请求跨多 chunk 才能产出首 token
    if chunk_size < seq_len:
        chunks_per_req = (seq_len + chunk_size - 1) // chunk_size
        return float(avg_reqs_factor * chunks_per_req * t_chunk_ns + attn_ttft_ns + warmup_ns)

    # 场景 A: chunk_size >= seq_len
    K = chunk_size // seq_len              # 每个满 chunk 装下的完整请求数
    R = remainder_size // seq_len          # remainder chunk 装下的完整请求数

 
    chunk_struct_reqs = K * num_chunks_full + R
    if chunk_struct_reqs != per_worker_reqs and chunk_struct_reqs > 0:
        per_worker_reqs = chunk_struct_reqs
        # per_worker_reqs 变化 → 重算注意力分摊 (保持 A_total × (R+1)/(2R) 口径), 再按 pp 折算
        attn_ttft_ns = ((per_worker_reqs + 1) / 2.0 * attn_slices_latency_ns / per_worker_reqs
                        if attn_slices_latency_ns > 0 else 0.0)
        attn_ttft_ns = _pp_fold_attn(attn_ttft_ns)

    sum_full = K * num_chunks_full * (num_chunks_full + 1) // 2 * t_chunk_ns
    sum_rem  = R * (num_chunks_full * t_chunk_ns + t_remainder_ns)
    return float(sum_full + sum_rem) / max(1, per_worker_reqs) + attn_ttft_ns + warmup_ns


def _print_config_banner(
    *, model, device_type, batch_size, seq_len, output_length,
    tbo, mtp_param, fuse_moe, p_dpa, d_dpa,
    prefix_cache_hit_rate, s_new, prefill_kv_seq_len,
    prefill_nodes, decode_nodes,
    prefill_tp, prefill_dp, prefill_pp, prefill_ep, prefill_cp,
    decode_tp, decode_dp, decode_pp, decode_ep, decode_cp,
    filename_prefix,
) -> None:
    """打印配置 banner (Model / GPU / Batch&Seq / Parallel / File prefix 那段).

    verbose=True 路径下被 model_analyzer() 主流程调用 (替代原 _log 逐行打);
    verbose=False 路径下由 _print_memory_summary() 在输出 memory 详情前再调用一次,
    给 pd_sweep 等批量场景的日志加上下文标识.
    """
    print(f"{'='*60}")
    print(f"  LLM Performance Analyzer")
    print(f"{'='*60}")
    print(f"  Model: {model}")
    print(f"  GPU: {device_type}")
    print(f"  Batch Size: {batch_size}, Seq Len: {seq_len}, Output Length: {output_length}")
    if mtp_param:
        print(f"  TBO: {tbo}, MTP: {mtp_param}, Fuse MoE: {fuse_moe}, P_DPA: {p_dpa}, D_DPA: {d_dpa}")
    else:
        print(f"  TBO: {tbo}, MTP: disabled, Fuse MoE: {fuse_moe}, P_DPA: {p_dpa}, D_DPA: {d_dpa}")
    if prefix_cache_hit_rate > 0:
        print(f"  Prefix Cache: hit_rate={prefix_cache_hit_rate:.1%}, S_new={s_new}, kv_seq_len={prefill_kv_seq_len}")
    print(f"  PD Nodes: {prefill_nodes}P{decode_nodes}D")
    print(f"  Prefill Parallel: TP={prefill_tp}, DP={prefill_dp}, PP={prefill_pp}, EP={prefill_ep}, CP={prefill_cp}")
    print(f"  Decode Parallel:  TP={decode_tp}, DP={decode_dp}, PP={decode_pp}, EP={decode_ep}, CP={decode_cp}")
    print(f"  File prefix: {filename_prefix}")
    print(f"{'='*60}")


def _print_memory_section_preflight(result: dict, tag: str) -> None:
    """打印 preflight 单段显存细分, 1:1 对齐 后端 print_memory_result."""
    model_gb    = result.get(f"{tag.lower()}_model_memory_gb", 0.0)
    expert_gb   = result.get(f"{tag.lower()}_expert_memory_gb", 0.0)
    kv_cache_gb = result.get(f"{tag.lower()}_kv_cache_memory_gb", 0.0)
    total_gb    = result.get(f"{tag.lower()}_memory_gb", 0.0)
    limit_gb    = result.get("gpu_memory_limit_gb", 0.0)
    is_valid    = result.get(f"{tag.lower()}_memory_feasible", False)
    pass_tag    = "[PASS]" if is_valid else "[FAIL - 超限!]"

    print(f"  [{tag}] 单卡显存估算:")
    print(f"    ├─ 非Expert模型参数: {model_gb:.2f} GB")
    if expert_gb > 0:
        print(f"    ├─ Expert参数(TP+EP分割): {expert_gb:.2f} GB")
    print(f"    ├─ KV Cache:          {kv_cache_gb:.2f} GB")
    print(f"    ├─ 总计:              {total_gb:.2f} GB")
    print(f"    └─ 限制:              {limit_gb:.2f} GB  {pass_tag}")


def _print_memory_section_with_activation(result: dict, tag: str) -> None:
    """打印含 activation peak 的单段显存, 1:1 对齐 后端 print_memory_result_with_activation."""
    key = tag.lower()
    model_gb    = result.get(f"{key}_model_memory_gb", 0.0)
    expert_gb   = result.get(f"{key}_expert_memory_gb", 0.0)
    kv_cache_gb = result.get(f"{key}_kv_cache_memory_gb", 0.0)
    total_gb    = result.get(f"{key}_total_memory_with_act_gb", 0.0)
    limit_gb    = result.get("gpu_memory_limit_gb", 0.0)
    is_valid    = result.get(f"{key}_memory_feasible_with_act", False)
    pass_tag    = "[PASS]" if is_valid else "[FAIL - OOM 超限!]"

    has_act     = result.get(f"{key}_has_activation_peak", False)
    act_gb      = result.get(f"{key}_activation_peak_gb", 0.0)
    act_module  = result.get(f"{key}_activation_peak_module", "")
    act_op      = result.get(f"{key}_activation_peak_op", "")
    act_desc    = result.get(f"{key}_activation_peak_desc", "")

    print(f"  [{tag}] 单卡显存 (含激活峰值):")
    print(f"    ├─ 非Expert模型参数:  {model_gb:.2f} GB")
    if expert_gb > 0:
        print(f"    ├─ Expert参数(TP+EP):  {expert_gb:.2f} GB")
    print(f"    ├─ KV Cache:          {kv_cache_gb:.2f} GB")
    if has_act:
        print(f"    ├─ Activation Peak:   {act_gb:.2f} GB")
        loc = act_module
        if act_op:
            if loc:
                loc += "/"
            loc += act_op
        if loc:
            print(f"    │     └─ 位置: {loc}")
        if act_desc:
            print(f"    │     └─ shape: {act_desc}")
    else:
        print(f"    ├─ Activation Peak:   N/A (analyze 未跑或聚合失败)")
    print(f"    ├─ 总计:              {total_gb:.2f} GB")
    print(f"    └─ 限制:              {limit_gb:.2f} GB  {pass_tag}")
    if limit_gb > 0:
        util = total_gb / limit_gb * 100.0
        remaining = limit_gb - total_gb
        print(f"       占用率: {util:.2f}% (余量 {remaining:.2f} GB)")


def _print_memory_summary(
    result: dict,
    *,
    model, device_type, batch_size, seq_len, output_length,
    tbo, mtp_param, fuse_moe, p_dpa, d_dpa,
    prefix_cache_hit_rate, s_new, prefill_kv_seq_len,
    prefill_nodes, decode_nodes,
    prefill_tp, prefill_dp, prefill_pp, prefill_ep, prefill_cp,
    decode_tp, decode_dp, decode_pp, decode_ep, decode_cp,
    filename_prefix,
) -> None:
    """verbose=False 路径下补打配置 banner + 显存详情 (preflight + 含 activation).

    格式严格 1:1 对齐 后端 check_memory_feasibility / print_memory_result*.
    """
    _print_config_banner(
        model=model, device_type=device_type,
        batch_size=batch_size, seq_len=seq_len, output_length=output_length,
        tbo=tbo, mtp_param=mtp_param, fuse_moe=fuse_moe, p_dpa=p_dpa, d_dpa=d_dpa,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
        s_new=s_new, prefill_kv_seq_len=prefill_kv_seq_len,
        prefill_nodes=prefill_nodes, decode_nodes=decode_nodes,
        prefill_tp=prefill_tp, prefill_dp=prefill_dp, prefill_pp=prefill_pp,
        prefill_ep=prefill_ep, prefill_cp=prefill_cp,
        decode_tp=decode_tp, decode_dp=decode_dp, decode_pp=decode_pp,
        decode_ep=decode_ep, decode_cp=decode_cp,
        filename_prefix=filename_prefix,
    )

    model_type     = result.get("model_type", "")
    is_mla         = result.get("is_mla", False)
    is_v4          = result.get("is_v4", False)
    is_moe         = result.get("is_moe", False)
    limit_gb       = result.get("gpu_memory_limit_gb", 0.0)
    est_params     = result.get("estimated_params", 0)
    attn_label = "V4-sparse" if is_v4 else ("MLA" if is_mla else "MHA/GQA")
    arch_label = "MoE" if is_moe else "Dense"

    print(f"\n{'='*40}")
    print(f"GPU 显存可行性检查")
    if device_type:
        print(f"  Device: {device_type} (limit={limit_gb:.2f} GB)")
    print(f"{'='*40}")
    print(f"  模型: {model_type}")
    print(f"  类型: {arch_label} + {attn_label}")
    print(f"  估算参数量: {est_params / 1e9:.1f}B")
    print(f"  Batch Size: {batch_size}, Input Length: {seq_len}, Output Length: {output_length}")

    _print_memory_section_preflight(result, "Prefill")
    _print_memory_section_preflight(result, "Decode")
    print(f"{'-'*40}")

    prefill_ok = result.get("prefill_memory_feasible", False)
    decode_ok  = result.get("decode_memory_feasible", False)
    if prefill_ok and decode_ok:
        print(f"  显存检查通过!")
    else:
        print(f"  [ERROR] 显存检查失败!")
        if not prefill_ok:
            print(f"  [ERROR] Prefill 配置 (TP={prefill_tp}, EP={prefill_ep}) "
                  f"需要 {result.get('prefill_memory_gb', 0.0):.2f} GB, "
                  f"超过限制 {limit_gb:.2f} GB")
        if not decode_ok:
            print(f"  [ERROR] Decode 配置 (TP={decode_tp}, EP={decode_ep}) "
                  f"需要 {result.get('decode_memory_gb', 0.0):.2f} GB, "
                  f"超过限制 {limit_gb:.2f} GB")
    print(f"{'='*40}\n")

    if "prefill_total_memory_with_act_gb" not in result:
        return

    print(f"{'='*40}")
    print(f"最终显存使用估算 (含激活峰值, 基于真实算子 shape 聚合)")
    print(f"{'='*40}")
    _print_memory_section_with_activation(result, "Prefill")
    _print_memory_section_with_activation(result, "Decode")
    print(f"{'-'*40}")
    if result.get("memory_feasible_with_activation", True):
        print(f"  OOM 风险: 否")
    else:
        print(f"  ⚠ OOM 风险: 是 (含 activation peak 后超过限制)")
    print(f"{'='*40}\n")


def _build_filename_prefix(
    model: str, tbo: bool,
    prefill_tp: int, prefill_dp: int, prefill_pp: int, prefill_ep: int, prefill_cp: int,
    decode_tp: int, decode_dp: int, decode_pp: int, decode_ep: int, decode_cp: int,
    batch_size: int, seq_len: int, output_length: int, chunk_prefill_size: int, device_type: str,
    p_dpa: bool = True, d_dpa: bool = True,
    prefix_cache_hit_rate: float = 0.0,
    mtp_param: str = "",
) -> str:
    """
    生成文件名前缀: 模型名称_并行方式_bs_输入大小_输出大小
    例: deepseekv3_tp8dp1ep1_tp1dp8ep8_tbo_bs256_4096_1500
    """
    mode = "tbo" if tbo else "notbo"

    # Prefill 并行简写
    prefill_parts = []
    if prefill_tp > 1: prefill_parts.append(f"tp{prefill_tp}")
    if prefill_dp > 1: prefill_parts.append(f"dp{prefill_dp}")
    if prefill_pp > 1: prefill_parts.append(f"pp{prefill_pp}")
    if prefill_ep > 1: prefill_parts.append(f"ep{prefill_ep}")
    if prefill_cp > 1: prefill_parts.append(f"cp{prefill_cp}")
    prefill_str = "".join(prefill_parts) if prefill_parts else "tp1"

    # Decode 并行简写
    decode_parts = []
    if decode_tp > 1: decode_parts.append(f"tp{decode_tp}")
    if decode_dp > 1: decode_parts.append(f"dp{decode_dp}")
    if decode_pp > 1: decode_parts.append(f"pp{decode_pp}")
    if decode_ep > 1: decode_parts.append(f"ep{decode_ep}")
    if decode_cp > 1: decode_parts.append(f"cp{decode_cp}")
    decode_str = "".join(decode_parts) if decode_parts else "tp1"

    extras = [f"dpaP{int(bool(p_dpa))}D{int(bool(d_dpa))}"]
    if prefix_cache_hit_rate and prefix_cache_hit_rate > 0:
        extras.append(f"pc{int(round(prefix_cache_hit_rate * 100))}")
    if mtp_param:
        extras.append(f"mtp{mtp_param}")
    extra_str = "_".join(extras)

    return (
        f"{model}_{prefill_str}_{decode_str}_{mode}"
        f"_bs{batch_size}_{seq_len}_{output_length}"
        f"_chunk_prefill_size{chunk_prefill_size}"
        f"_{extra_str}_{device_type}"
    )


def model_analyzer(
    # Prefill 并行配置
    prefill_tp: int = 8,
    prefill_dp: int = 8,
    prefill_pp: int = 1,
    prefill_ep: int = 8,
    prefill_cp: int = 1,
    # Decode 并行配置
    decode_tp: int = 8,
    decode_dp: int = 8,
    decode_pp: int = 1,
    decode_ep: int = 8,
    decode_cp: int = 1,
    # PD 节点数（xPyD）
    prefill_nodes: int = 1,
    decode_nodes: int = 1,
    # 推理参数
    batch_size: int = 32,
    seq_len: int = 8192,
    output_length: int = 1024,
    chunk_prefill_size: int = 16384,
    # 模型/硬件
    gpu: str = "nmz",
    model: str = "deepseekv4",
    # 功能开关
    tbo: bool = False,
    sbo: bool = False,
    # Sequence Parallel (Megatron-SP, 对齐 SGLang ScatterMode.SCATTERED):
    #   开启且 attn_tp_size_eff()>1 时, norm/residual 沿序列维 ÷sp,
    #   o_proj/mlp_down 的 AllReduce → ReduceScatter, 下一个 GEMM 前补 AllGather.
    sequence_parallel: bool = False,
    fuse_moe: bool = True,   # remote overlay: 默认 True (clean 为 False), 同步勿动
    # DPA 开关 (Data Parallel Attention)
    dp_attention: bool = True,
    prefill_dp_attention: Optional[bool] = None,
    decode_dp_attention: Optional[bool] = None,
    moe_dp_size: int = 1,
    prefill_moe_dp_size: Optional[int] = None,
    decode_moe_dp_size: Optional[int] = None,
    moe_dense_tp_size: Optional[int] = None,  # None / 1
    prefill_moe_dense_tp_size: Optional[int] = None,
    decode_moe_dense_tp_size: Optional[int] = None,
    moe_a2a_backend: str = "none",
    prefill_moe_a2a_backend: Optional[str] = None,
    decode_moe_a2a_backend: Optional[str] = None,
    mtp_param: str = "",
    mtp_acceptance_rate: float = 0.8,   # P0-6: MTP 接受率 (与 C++ SpeculativeConfig 默认值一致)
    generate_csv: bool = True,
    verbose: bool = True,
    skip_memory_check: bool = False,  # 算子建模验证时绕过 OOM 早返回, 仍输出 raw_data
    memory_ratio: float = 0.9,         # 显存预留比例 (gpu_memory_gb × ratio = limit), 0.9=留 10%
    # 视觉输入 (Kimi-K2.5)
    num_images: int = 0,
    image_height: int = 448,
    image_width: int = 448,
    # Prefix Cache
    prefix_cache_hit_rate: float = 0,
    # 集群拓扑
    device_type: Optional[str] = None,   # None = 跟随 gpu 参数 (大小写不敏感, 后端 端归一化); 显式传则覆盖 gpu
    gpus_per_node: int = 4,
    nvlink_domain_size: int = 40,      # NVLink domain size: 8-card node=8, supernode=40
    intra_node_bw_gbs: float = 110.0,
    inter_node_bw_gbs: float = 110.0,
    # Roofline utilization (-1.0 = use device JSON utilization_factors defaults).
    # Only affects the Roofline fallback path. See doc/utilization_parameterization_design.md.
    util_tensor_core: float = -1.0,
    util_vector: float = -1.0,
    util_sfu: float = -1.0,
    util_bandwidth: float = -1.0,
    util_profile: Optional[str] = None,
    # KV cache dtype: "fp8"(默认, FP8 KV 省一半显存) / "bf16" / "auto"(跟随模型架构)
    kv_cache_dtype: str = "fp8",
    # attention QK/PV 张量核算力 dtype: "fp8"(默认, 部署侧 attention 量化至 fp8 计算)
    # / "bf16" / "auto"(跟随输入 dtype 位宽)。仅影响 Roofline 的 TC 峰值选择,
    attention_compute_dtype: str = "fp8",
    pp_async_batch_depth: int = 0,
    # 高级选项
    model_json_path: Optional[str] = None,
    # 利用率扫描内部参数 (util_sweep.py 使用, 普通调用勿传):
    _sweep_label: Optional[str] = None,
) -> dict:
    """
    LLM 性能分析主接口

    Args:
        prefill_tp/dp/pp/ep/cp: Prefill 阶段并行配置
        decode_tp/dp/pp/ep/cp: Decode 阶段并行配置
        prefill_nodes: Prefill 节点数 (0=从并行配置自动推算: tp×dp×pp/gpus_per_node)
        decode_nodes: Decode 节点数 (0=从并行配置自动推算)
        batch_size: 批大小
        seq_len: 输入序列长度
        output_length: 输出序列长度
        gpu: GPU 类型 (NMZ 等)
        model: 模型名称 (deepseekv3, llama2 等)
        tbo: 是否启用 TBO (Two-Batch Overlap)
        mtp_param: MTP 参数字符串，如 '314'
            --speculative-algorithm EAGLE
            --speculative-num-steps 3
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 4
        prefix_cache_hit_rate: Prefix Cache 命中率 (0.0~1.0)，0.0 表示无缓存，1.0 表示全部命中
            命中的 token 完全跳过 prefill 前向计算，未命中 token 仍需 attend 到完整序列
        generate_csv: 是否生成详细的算子性能 Excel (含 raw_data + summary 两个 sheet)
        verbose: 是否打印详细日志，False 时静默运行（pd_sweep 遍历模式使用）
        model_json_path: 直接指定模型 JSON 配置文件路径（优先于 model 参数）

    Returns:
        dict: 包含 latency, TPS, flops 等性能指标
              单实例指标: prefill_tps, decode_tps_per_step, overall_tps
              系统级指标: sys_prefill_tps, sys_decode_tps, sys_overall_tps, sys_throughput_tps
    """
    def _log(msg: str):
        if 1:#verbose
            print(msg)

    p_dpa = prefill_dp_attention if prefill_dp_attention is not None else dp_attention
    d_dpa = decode_dp_attention if decode_dp_attention is not None else dp_attention

    p_moe_dp = prefill_moe_dp_size if prefill_moe_dp_size is not None else moe_dp_size
    d_moe_dp = decode_moe_dp_size if decode_moe_dp_size is not None else moe_dp_size
    _mdts_unified = -1 if moe_dense_tp_size is None else moe_dense_tp_size
    p_mdts = -1 if prefill_moe_dense_tp_size is None else prefill_moe_dense_tp_size
    d_mdts = -1 if decode_moe_dense_tp_size is None else decode_moe_dense_tp_size
    if prefill_moe_dense_tp_size is None:
        p_mdts = _mdts_unified
    if decode_moe_dense_tp_size is None:
        d_mdts = _mdts_unified
    p_a2a = prefill_moe_a2a_backend if prefill_moe_a2a_backend is not None else moe_a2a_backend
    d_a2a = decode_moe_a2a_backend if decode_moe_a2a_backend is not None else moe_a2a_backend

    # a2a backend 字符串白名单 (集成文档 §4.5)
    _VALID_A2A = {"none", "deepep", "mooncake", "nixl", "mori",
                  "ascend_fuseep", "flashinfer", "megamoe"}
    if p_a2a not in _VALID_A2A:
        raise ValueError(f"prefill_moe_a2a_backend={p_a2a!r} 非法, 必须 ∈ {sorted(_VALID_A2A)}")
    if d_a2a not in _VALID_A2A:
        raise ValueError(f"decode_moe_a2a_backend={d_a2a!r} 非法, 必须 ∈ {sorted(_VALID_A2A)}")
    if p_mdts not in (-1, 1):
        raise ValueError(f"prefill_moe_dense_tp_size={p_mdts} 非法, 仅支持 None/-1 或 1")
    if d_mdts not in (-1, 1):
        raise ValueError(f"decode_moe_dense_tp_size={d_mdts} 非法, 仅支持 None/-1 或 1")

    def _check_moe_dp(name: str, tp: int, ep: int, mdp: int):
        if mdp <= 1:
            return
        if tp % mdp != 0:
            raise ValueError(f"{name}: tp={tp} % moe_dp={mdp} != 0 (SGLang 约束)")
        if ep * mdp > tp:
            raise ValueError(f"{name}: ep*moe_dp={ep*mdp} > tp={tp} (SGLang 约束)")
        if ep > 1 and ep * mdp != tp:
            raise ValueError(f"{name}: ep>1 时强约束 ep*moe_dp == tp; got ep={ep}, moe_dp={mdp}, tp={tp}")
    _check_moe_dp("prefill", prefill_tp, prefill_ep, p_moe_dp)
    _check_moe_dp("decode",  decode_tp,  decode_ep,  d_moe_dp)

    # CP × DP × TP 物理合法性 (P1-5, 2026-05-24)
    def _check_cp_dp(name: str, tp: int, dp: int, cp: int):
        if cp <= 1:
            return
        if dp * cp > tp:
            raise ValueError(
                f"{name}: dp×cp={dp*cp} > tp={tp} (SGLang 约束: cp 在 attn_tp 内切 seq, "
                f"dp×cp ≤ tp). got dp={dp}, cp={cp}, tp={tp}"
            )
        if tp % (dp * cp) != 0:
            raise ValueError(
                f"{name}: tp={tp} % (dp×cp)={dp*cp} != 0 (SGLang 约束). "
                f"got dp={dp}, cp={cp}, tp={tp}"
            )
    _check_cp_dp("prefill", prefill_tp, prefill_dp, prefill_cp)
    _check_cp_dp("decode",  decode_tp,  decode_dp,  decode_cp)

    # 并行配置硬断言 (SGLang + deepep/mooncake 后端语义)
    def _legal_ep(ep: int, tp: int, dp: int, dpa: bool) -> set:
        return {1, tp, tp * dp} if dpa else {1, tp}
    p_ep_legal = _legal_ep(prefill_ep, prefill_tp, prefill_dp, p_dpa)
    d_ep_legal = _legal_ep(decode_ep, decode_tp, decode_dp, d_dpa)
    if prefill_ep not in p_ep_legal:
        raise ValueError(
            f"prefill_ep={prefill_ep} 非法: DPA={p_dpa} 下 EP 必须 ∈ "
            f"{sorted(p_ep_legal)} (tp={prefill_tp}, dp={prefill_dp})"
        )
    if decode_ep not in d_ep_legal:
        raise ValueError(
            f"decode_ep={decode_ep} 非法: DPA={d_dpa} 下 EP 必须 ∈ "
            f"{sorted(d_ep_legal)} (tp={decode_tp}, dp={decode_dp})"
        )

    # 推算节点数 (0 表示从并行配置自动计算)
    if prefill_nodes <= 0:
        prefill_gpus = prefill_tp * prefill_dp * prefill_pp
        prefill_nodes = max(1, (prefill_gpus + gpus_per_node - 1) // gpus_per_node)
    if decode_nodes <= 0:
        decode_gpus = decode_tp * decode_dp * decode_pp
        decode_nodes = max(1, (decode_gpus + gpus_per_node - 1) // gpus_per_node)

    # 计算 Prefix Cache 参数
    prefix_cache_hit_rate = max(0.0, min(1.0, prefix_cache_hit_rate))
    if prefix_cache_hit_rate > 0:
        s_new = max(1, int(seq_len * (1 - prefix_cache_hit_rate)))
        prefill_kv_seq_len = seq_len  # Attention 仍需 attend 到完整序列
        # Prefix cache + chunked prefill 交互修正:
        #
        if chunk_prefill_size > 0 and chunk_prefill_size < s_new:
            chunk_prefill_size = s_new
    else:
        s_new = seq_len
        prefill_kv_seq_len = 0  # 0 表示不启用，后端 端回退到 seq_len

    from contextlib import contextmanager
    import os as _os

    @contextmanager
    def _suppress_stdout():
        """
        同时压制 后端 和 Python 的 stdout 输出：
          - fd 级重定向：覆盖 后端 printf / std::cout（直接写 fd 1）
          - sys.stdout 替换：覆盖 Python print()（经过 Python 缓冲层）
        """
        import sys as _sys
        _sys.stdout.flush()
        _old_py_stdout = _sys.stdout
        _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        _saved_fd = _os.dup(1)
        try:
            _os.dup2(_devnull_fd, 1)
            _sys.stdout = open(_os.devnull, 'w')
            yield
        finally:
            _sys.stdout.flush()
            _sys.stdout.close()
            _sys.stdout = _old_py_stdout
            _os.dup2(_saved_fd, 1)
            _os.close(_saved_fd)
            _os.close(_devnull_fd)

    model_json_path = resolve_model_json(model, model_json_path)

    # 2. 确定 device_type (GPU 类型映射)
    if device_type is None:
        device_type = gpu

    # 3. 生成文件名前缀
    filename_prefix = _build_filename_prefix(
        model, tbo,
        prefill_tp, prefill_dp, prefill_pp, prefill_ep, prefill_cp,
        decode_tp, decode_dp, decode_pp, decode_ep, decode_cp,
        batch_size, seq_len, output_length, chunk_prefill_size, device_type,
        p_dpa=p_dpa, d_dpa=d_dpa,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
        mtp_param=mtp_param,
    )

    # 利用率扫描: 每个场景用独立文件名前缀, 避免多场景 raw CSV 互相覆盖
    if _sweep_label is not None:
        filename_prefix = f"{filename_prefix}_util{_sweep_label}"

    csv_output_path = ""
    if generate_csv:
        raw_data_dir = Path.cwd() / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)
        csv_output_path = str(raw_data_dir / f"{filename_prefix}.csv")

    # 5. 调用 后端 绑定
    _print_config_banner(
        model=model, device_type=device_type,
        batch_size=batch_size, seq_len=seq_len, output_length=output_length,
        tbo=tbo, mtp_param=mtp_param, fuse_moe=fuse_moe, p_dpa=p_dpa, d_dpa=d_dpa,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
        s_new=s_new, prefill_kv_seq_len=prefill_kv_seq_len,
        prefill_nodes=prefill_nodes, decode_nodes=decode_nodes,
        prefill_tp=prefill_tp, prefill_dp=prefill_dp, prefill_pp=prefill_pp,
        prefill_ep=prefill_ep, prefill_cp=prefill_cp,
        decode_tp=decode_tp, decode_dp=decode_dp, decode_pp=decode_pp,
        decode_ep=decode_ep, decode_cp=decode_cp,
        filename_prefix=filename_prefix,
    )

    _prefill_util_profile: dict = {}
    _decode_util_profile: dict = {}
    if util_profile:
        import json as _json
        with open(util_profile, "r") as _f:
            _up = _json.load(_f)
        _prefill_util_profile = _up.get("prefill", {}) or {}
        _decode_util_profile = _up.get("decode", {}) or {}

    if verbose:
        result = analyze_model_binding(
            model_json_path=model_json_path,
            prefill_tp=prefill_tp, prefill_dp=prefill_dp,
            prefill_pp=prefill_pp, prefill_ep=prefill_ep,
            prefill_cp=prefill_cp,
            decode_tp=decode_tp, decode_dp=decode_dp,
            decode_pp=decode_pp, decode_ep=decode_ep,
            decode_cp=decode_cp,
            batch_size=batch_size, seq_len=s_new,
            output_length=output_length,
            chunk_prefill_size=chunk_prefill_size,
            enable_tbo=tbo,
            enable_sbo=sbo,
            enable_sequence_parallel=sequence_parallel,
            fuse_moe=fuse_moe,
            prefill_dp_attention=p_dpa,
            decode_dp_attention=d_dpa,
            # ★ 2026-05-22 新增
            prefill_moe_dp_size=p_moe_dp,
            decode_moe_dp_size=d_moe_dp,
            prefill_moe_dense_tp_size=p_mdts,
            decode_moe_dense_tp_size=d_mdts,
            prefill_moe_a2a_backend=p_a2a,
            decode_moe_a2a_backend=d_a2a,
            mtp_param=mtp_param,
            mtp_acceptance_rate=mtp_acceptance_rate,
            csv_output_path=csv_output_path,
            device_type=device_type,
            num_images=num_images,
            image_height=image_height,
            image_width=image_width,
            gpus_per_node=gpus_per_node,
            nvlink_domain_size=nvlink_domain_size,
            intra_node_bw_gbs=intra_node_bw_gbs,
            inter_node_bw_gbs=inter_node_bw_gbs,
            prefill_kv_seq_len=prefill_kv_seq_len,
            skip_memory_check=skip_memory_check,
            memory_ratio=memory_ratio,
            util_tensor_core=util_tensor_core,
            util_vector=util_vector,
            util_sfu=util_sfu,
            util_bandwidth=util_bandwidth,
            kv_cache_dtype=kv_cache_dtype,
            pp_async_batch_depth=pp_async_batch_depth,
            attention_compute_dtype=attention_compute_dtype,
            prefill_util_profile=_prefill_util_profile,
            decode_util_profile=_decode_util_profile,
        )
    else:
        with _suppress_stdout():
            result = analyze_model_binding(
                model_json_path=model_json_path,
                prefill_tp=prefill_tp, prefill_dp=prefill_dp,
                prefill_pp=prefill_pp, prefill_ep=prefill_ep,
                prefill_cp=prefill_cp,
                decode_tp=decode_tp, decode_dp=decode_dp,
                decode_pp=decode_pp, decode_ep=decode_ep,
                decode_cp=decode_cp,
                batch_size=batch_size, seq_len=s_new,
                output_length=output_length,
                chunk_prefill_size=chunk_prefill_size,
                enable_tbo=tbo,
                enable_sbo=sbo,
                enable_sequence_parallel=sequence_parallel,
                fuse_moe=fuse_moe,
                prefill_dp_attention=p_dpa,
                decode_dp_attention=d_dpa,
                # ★ 2026-05-22 新增
                prefill_moe_dp_size=p_moe_dp,
                decode_moe_dp_size=d_moe_dp,
                prefill_moe_dense_tp_size=p_mdts,
                decode_moe_dense_tp_size=d_mdts,
                prefill_moe_a2a_backend=p_a2a,
                decode_moe_a2a_backend=d_a2a,
                mtp_param=mtp_param,
                csv_output_path=csv_output_path,
                device_type=device_type,
                num_images=num_images,
                image_height=image_height,
                image_width=image_width,
                gpus_per_node=gpus_per_node,
                nvlink_domain_size=nvlink_domain_size,
                intra_node_bw_gbs=intra_node_bw_gbs,
                inter_node_bw_gbs=inter_node_bw_gbs,
                prefill_kv_seq_len=prefill_kv_seq_len,
                skip_memory_check=skip_memory_check,
                memory_ratio=memory_ratio,
                util_tensor_core=util_tensor_core,
                util_vector=util_vector,
                util_sfu=util_sfu,
                util_bandwidth=util_bandwidth,
                kv_cache_dtype=kv_cache_dtype,
                pp_async_batch_depth=pp_async_batch_depth,
                attention_compute_dtype=attention_compute_dtype,
                prefill_util_profile=_prefill_util_profile,
                decode_util_profile=_decode_util_profile,
            )
        _print_memory_summary(
            result,
            model=model, device_type=device_type,
            batch_size=batch_size, seq_len=seq_len, output_length=output_length,
            tbo=tbo, mtp_param=mtp_param, fuse_moe=fuse_moe, p_dpa=p_dpa, d_dpa=d_dpa,
            prefix_cache_hit_rate=prefix_cache_hit_rate,
            s_new=s_new, prefill_kv_seq_len=prefill_kv_seq_len,
            prefill_nodes=prefill_nodes, decode_nodes=decode_nodes,
            prefill_tp=prefill_tp, prefill_dp=prefill_dp, prefill_pp=prefill_pp,
            prefill_ep=prefill_ep, prefill_cp=prefill_cp,
            decode_tp=decode_tp, decode_dp=decode_dp, decode_pp=decode_pp,
            decode_ep=decode_ep, decode_cp=decode_cp,
            filename_prefix=filename_prefix,
        )

    if not result.get("memory_feasible", False):
        if skip_memory_check:
            if verbose:
                logger.warning("显存不足，但 skip_memory_check=True，继续生成建模数据")
        else:
            if verbose:
                logger.error("显存不足，无法运行分析")
            else:
                logger.debug("显存不足，无法运行分析")
            return result

    # 6. 从底层返回的 latency 计算 Python 端指标
    result = dict(result)  # copy 避免改到底层返回的 dict
    result["filename_prefix"] = filename_prefix
    result["prefix_cache_hit_rate"] = prefix_cache_hit_rate
    result["prefill_seq_len"] = s_new
    result["original_seq_len"] = seq_len

    prefill_latency_ns = result["prefill_latency_ns"]
    decode_latency_ns = result["decode_latency_ns"]
    decode_pp_bottleneck_ns = result.get("decode_pp_bottleneck_ns", 0)
    nextn_latency_ns = result.get("nextn_decode_latency_ns", 0)

    prefill_chunk_latency_ns = result.get("prefill_chunk_latency_ns", 0)
    prefill_remainder_latency_ns = result.get("prefill_remainder_latency_ns", 0)
    num_prefill_chunks = int(result.get("num_prefill_chunks", 0))
    prefill_remainder_size = int(result.get("prefill_remainder_size", 0))
    effective_chunk_size = int(result.get("prefill_effective_chunk_size", 0))
    if effective_chunk_size <= 0:
        effective_chunk_size = chunk_prefill_size
    # 方案② (chunked prefill 注意力核切片): 注意力核已从 per-chunk backbone 剥离,
    #   t_chunk_ns 不再含注意力. 单独取切片总延迟, 折算进 per-request TTFT (下方).
    prefill_attn_slices_latency_ns = result.get("prefill_attn_slices_latency_ns", 0)

    avg_ttft_ns = _compute_avg_ttft_ns(
        prefill_latency_ns=prefill_latency_ns,
        t_chunk_ns=prefill_chunk_latency_ns,
        t_remainder_ns=prefill_remainder_latency_ns,
        num_chunks_full=num_prefill_chunks,
        remainder_size=prefill_remainder_size,
        batch_size=batch_size,
        seq_len=s_new,
        chunk_size=effective_chunk_size,
        dp_size=prefill_dp,
        pp_size=prefill_pp,   
        attn_slices_latency_ns=prefill_attn_slices_latency_ns,  
    )
    result["ttft_ms"] = avg_ttft_ns / 1e6
    result["prefill_total_latency_ms"] = prefill_latency_ns / 1e6
    # Chunked prefill 细粒度延迟换算 (单块 / 尾块), 供展示与脚本读取
    result["prefill_chunk_latency_ms"] = prefill_chunk_latency_ns / 1e6
    result["prefill_remainder_latency_ms"] = prefill_remainder_latency_ns / 1e6

    # 基础指标
    result["prefill_latency_ms"] = prefill_latency_ns / 1e6
    result["decode_latency_ms"] = decode_latency_ns / 1e6
    result["decode_pp_bottleneck_ms"] = decode_pp_bottleneck_ns / 1e6
    result["nextn_decode_latency_ms"] = nextn_latency_ns / 1e6

    # effective_batch 在 TBO 下不再 //2:
    effective_batch = batch_size

    if prefill_latency_ns > 0:
        result["prefill_compute_tps"] = effective_batch * s_new / (prefill_latency_ns / 1e9)
        result["prefill_effective_tps"] = effective_batch * seq_len / (prefill_latency_ns / 1e9)
        result["prefill_tps"] = result["prefill_compute_tps"]
    else:
        result["prefill_compute_tps"] = 0
        result["prefill_effective_tps"] = 0
        result["prefill_tps"] = 0

    if prefill_latency_ns > 0:
        per_rank_batch = max(1, (batch_size + prefill_dp - 1) // prefill_dp)
        per_instance_tps = per_rank_batch * s_new / (prefill_latency_ns / 1e9)
        result["per_instance_tps"] = per_instance_tps
        result["system_tps"] = prefill_dp * per_instance_tps
    else:
        result["per_instance_tps"] = 0
        result["system_tps"] = 0
    result["tps_semantics_note"] = (
        "tps / prefill_tps = per_instance_tps (单 DP 实例 throughput, 与历史 baseline 兼容); "
        "system_tps = dp × per_instance_tps (集群总 throughput, 对齐 SGLang benchmark)"
    )

    # Decode 单步吞吐 (主模型，不含 MTP)
    if decode_latency_ns > 0:
        result["decode_tps_per_step"] = effective_batch / (decode_latency_ns / 1e9)
    else:
        result["decode_tps_per_step"] = 0

    # 总 decode 时间 (不含 MTP)
    total_decode_time_ns = decode_latency_ns * output_length
    result["total_decode_time_ms"] = total_decode_time_ns / 1e6

    # 总时间 (prefill + decode)
    result["total_time_ms"] = (prefill_latency_ns + total_decode_time_ns) / 1e6

    # 总体 TPS (不含 MTP)
    if total_decode_time_ns > 0:
        result["overall_tps"] = (effective_batch * output_length) / (total_decode_time_ns / 1e9)
    else:
        result["overall_tps"] = 0

  
    tbt_throughput_ns = decode_latency_ns

   
    result["prefill_nodes"] = prefill_nodes
    result["decode_nodes"] = decode_nodes
    result["prefill_dp"] = prefill_dp
    result["decode_dp"] = decode_dp
    result["sys_prefill_tps"] = result["prefill_tps"]
    if tbt_throughput_ns > 0:
        result["sys_decode_tps"] = effective_batch / (tbt_throughput_ns / 1e9)
    else:
        result["sys_decode_tps"] = result["decode_tps_per_step"]
    result["sys_overall_tps"] = result["overall_tps"]

    if prefill_latency_ns > 0 and decode_latency_ns > 0:
        prefill_batch_time_s = prefill_latency_ns / 1e9
        tbt_s  = tbt_throughput_ns / 1e9
        prefill_rps = effective_batch / prefill_batch_time_s
        decode_rps  = effective_batch / (output_length * tbt_s)
        balanced_rps = min(prefill_rps, decode_rps)
        result["sys_throughput_tps"] = balanced_rps * output_length
        if prefill_rps < decode_rps * 0.95:
            result["bottleneck"] = "Prefill"
        elif decode_rps < prefill_rps * 0.95:
            result["bottleneck"] = "Decode"
        else:
            result["bottleneck"] = "Balanced"
    else:
        result["sys_throughput_tps"] = 0
        result["bottleneck"] = ""

    if mtp_param and nextn_latency_ns > 0 and decode_latency_ns > 0:
        spec_cfg = llm_perf_core.SpeculativeConfig.from_mtp_param(
            mtp_param, float(mtp_acceptance_rate))
        D                        = spec_cfg.num_draft_tokens
        accept                   = mtp_acceptance_rate
        mtp_step_latency_ns      = decode_latency_ns + nextn_latency_ns
        expected_tokens_per_step = spec_cfg.expected_tokens_per_step()
        mtp_decode_steps         = output_length / expected_tokens_per_step
        mtp_total_decode_ns      = mtp_step_latency_ns * mtp_decode_steps

        result["mtp_draft_tokens"]             = D
        result["mtp_acceptance_rate"]          = accept
        result["mtp_expected_tokens_per_step"] = expected_tokens_per_step
        result["mtp_layers"]                   = D                   # 向后兼容旧字段名
        result["mtp_tokens_per_step"]          = expected_tokens_per_step
        result["mtp_step_latency_ns"]          = mtp_step_latency_ns
        result["mtp_step_latency_ms"]          = mtp_step_latency_ns / 1e6
        result["mtp_effective_tpot_ms"]        = (
            mtp_step_latency_ns / expected_tokens_per_step / 1e6)
        result["mtp_total_decode_time_ms"]     = mtp_total_decode_ns / 1e6
        result["mtp_total_time_ms"]            = (prefill_latency_ns + mtp_total_decode_ns) / 1e6

        if mtp_total_decode_ns > 0:
            result["mtp_overall_tps"] = (effective_batch * output_length) / (mtp_total_decode_ns / 1e9)
        else:
            result["mtp_overall_tps"] = 0
        result["mtp_sys_overall_tps"] = result["mtp_overall_tps"]

    # 8. 生成 Excel (raw_data + summary 合并为一个文件)
    if generate_csv and csv_output_path:
        result["raw_csv_path"] = csv_output_path
        if _sweep_label is not None:
            _log(f"\n[util sweep] 场景 {_sweep_label} raw CSV: {csv_output_path}")
        else:
            num_dense_layers = result.get("num_dense_layers", 3)
            num_moe_layers = result.get("num_moe_layers", 58)
            excel_path = str(Path(csv_output_path).parent / f"{filename_prefix}.xlsx")
            if verbose:
                generate_excel(csv_output_path, excel_path, num_dense_layers, num_moe_layers)
            else:
                with _suppress_stdout():
                    generate_excel(csv_output_path, excel_path, num_dense_layers, num_moe_layers)
            result["excel_path"] = excel_path
            # 删除中间 CSV 文件，只保留 Excel
            try:
                os.remove(csv_output_path)
            except OSError:
                pass
            _log(f"\nExcel 已保存到: {excel_path}")

    # 9. 打印结果
    _log(f"\n{'='*60}")
    _log(f"  性能分析结果")
    _log(f"{'='*60}")
    _log(f"  device 种类: {device_type}")
    _log(f"  Prefill 延迟 (整批):  {result['prefill_latency_ms']:.4f} ms")
    if num_prefill_chunks > 0:
        _log(f"  Prefill 单块延迟:     {result['prefill_chunk_latency_ms']:.4f} ms × {num_prefill_chunks} 块"
             f"  + 尾块 {result['prefill_remainder_latency_ms']:.4f} ms (rem_size={prefill_remainder_size})")
    _log(f"  平均 TTFT (每请求):   {result['ttft_ms']:.4f} ms  [chunks={num_prefill_chunks}, rem_size={prefill_remainder_size}]")
    _log(f"  Prefill TPS (单实例): {result['prefill_tps']:.2f} tokens/s")
    _log(f"  Decode 单步延迟: {result['decode_latency_ms']:.4f} ms")
    _log(f"  Decode TPS (单实例): {result['decode_tps_per_step']:.2f} tokens/s")
    _log(f"  总 Decode 时间: {result['total_decode_time_ms']:.2f} ms")
    _log(f"  总时间 (Prefill + Decode): {result['total_time_ms']:.2f} ms")
    _log(f"  Overall TPS (单实例): {result['overall_tps']:.2f} tokens/s")

    if prefill_dp > 1 or decode_dp > 1:
        _log(f"\n  [系统级 TPS] ({prefill_nodes}P{decode_nodes}D, P_DP={prefill_dp}, D_DP={decode_dp})")
        _log(f"  Sys Prefill TPS: {result['sys_prefill_tps']:.2f} tokens/s")
        _log(f"  Sys Decode TPS:  {result['sys_decode_tps']:.2f} tokens/s")
        _log(f"  Sys Throughput:   {result['sys_throughput_tps']:.2f} tokens/s  [{result['bottleneck']}]")
        _log(f"  Sys Overall TPS: {result['sys_overall_tps']:.2f} tokens/s")

    if "mtp_overall_tps" in result:
        _log(f"\n  [MTP 加速] (理想接受率)")
        _log(f"  MTP 层数: {result['mtp_layers']}")
        _log(f"  每步产出 token: {result['mtp_tokens_per_step']}")
        _log(f"  MTP 单步延迟: {result['mtp_step_latency_ms']:.4f} ms")
        _log(f"  MTP 有效 TPOT (单步/每步产出): {result['mtp_effective_tpot_ms']:.4f} ms/token")
        _log(f"  MTP 总 Decode 时间: {result['mtp_total_decode_time_ms']:.2f} ms")
        _log(f"  MTP Overall TPS (单实例): {result['mtp_overall_tps']:.2f} tokens/s")
        if decode_dp > 1:
            _log(f"  MTP Sys Overall TPS: {result['mtp_sys_overall_tps']:.2f} tokens/s")

    _log(f"{'='*60}")

    return result


# ============================================================================
# 命令行支持: 从函数签名自动生成 argparse, 让 `python -m <module>` 可直接传参。
# build_cli_parser / run_cli 被 pd_sweep / supernode_sweep / util_sweep 复用,
# 新增函数参数时 CLI 自动覆盖, 无需手工维护 add_argument。
# ============================================================================
def _cli_str2bool(v):
    """argparse 用: 把 "true/false/1/0/yes/no" 解析成 bool。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"无法解析为布尔值: {v!r}")


def _cli_str_list(v):
    """argparse 用: 逗号分隔字符串 → list[str](去空白, 跳过空项)。"""
    return [x.strip() for x in str(v).split(",") if x.strip()]


def build_cli_parser(func, prog=None, exclude=(), type_overrides=None, description=None):
    """根据 func 的签名自动生成 argparse.ArgumentParser。

    类型推断优先级: type_overrides > bool > 注解(float/int/str) > 默认值类型 > str。
    - bool 参数 → `--name true/false`(默认沿用函数默认)
    - 无默认值的参数 → required=True
    - 默认值为 None 的参数 → 默认 None(不传则从 kwargs 剔除, 由函数用自身默认)
    `*args` / `**kwargs` 形参自动跳过。
    """
    type_overrides = dict(type_overrides or {})
    p = argparse.ArgumentParser(
        prog=prog, description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name, prm in inspect.signature(func).parameters.items():
        if name in exclude or prm.kind in (prm.VAR_POSITIONAL, prm.VAR_KEYWORD):
            continue
        default = prm.default
        required = default is inspect.Parameter.empty
        ann = prm.annotation
        if name in type_overrides:
            t = type_overrides[name]
        elif isinstance(default, bool) or ann is bool:
            t = bool
        elif ann is float or isinstance(default, float):
            t = float
        elif ann is int or (isinstance(default, int) and not isinstance(default, bool)):
            t = int
        else:
            t = str
        kw = {}
        if t is bool:
            kw["type"] = _cli_str2bool
            kw["metavar"] = "{true,false}"
        else:
            kw["type"] = t
        if required:
            kw["required"] = True
        else:
            kw["default"] = default
        p.add_argument("--" + name, **kw)
    return p


def run_cli(func, exclude=(), type_overrides=None, prog=None, description=None):
    """从命令行解析参数并调用 func; 未显式传(且默认 None)的参数从 kwargs 剔除。"""
    parser = build_cli_parser(func, prog=prog, exclude=exclude,
                              type_overrides=type_overrides, description=description)
    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    return func(**kwargs)


# model_analyzer 里默认值为 None 但需要真实类型的参数 (Optional[bool]/Optional[int])
_MODEL_ANALYZER_CLI_TYPES = {
    "prefill_dp_attention": bool, "decode_dp_attention": bool,
    "prefill_moe_dp_size": int, "decode_moe_dp_size": int,
    "moe_dense_tp_size": int,
    "prefill_moe_dense_tp_size": int, "decode_moe_dense_tp_size": int,
}


def _main_cli():
    result = run_cli(
        model_analyzer,
        exclude=("_sweep_label",),
        type_overrides=_MODEL_ANALYZER_CLI_TYPES,
        prog="python -m model_analyzer.model_analyzer",
        description="单次 LLM 推理性能分析 (prefill/decode 延迟、吞吐、TTFT、显存)")
    print("\n" + "=" * 50)
    print(f"  Prefill: {result.get('prefill_latency_ms', 0):.2f} ms"
          f"  |  Decode TBT: {result.get('decode_latency_ms', 0):.2f} ms")
    print(f"  System TPS: {result.get('sys_throughput_tps', 0):.0f} tokens/s"
          f"  |  Bottleneck: {result.get('bottleneck', '-')}")
    if result.get("excel_path"):
        print(f"  Excel: {result['excel_path']}")
    print("=" * 50)
    return result


if __name__ == "__main__":
    _main_cli()


