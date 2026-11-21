

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

_PACKAGE_DIR = Path(__file__).resolve().parent          # python/model_analyzer
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent              # model_analyzer/
_WHEEL_DEVICE_DIR = _PACKAGE_DIR / "data" / "op_analyzer_configs" / "devices"
_DEFAULT_DEVICE_DIR = _WHEEL_DEVICE_DIR if _WHEEL_DEVICE_DIR.is_dir() \
                                        else _PROJECT_ROOT / "configs" / "devices"
_DEVICE_DIR = Path(
    os.environ.get(
        "MODEL_ANALYZER_DEVICE_CONFIG_ROOT",
        _DEFAULT_DEVICE_DIR,
    )
)


def _normalize(name: str) -> str:
    return "".join(name.strip().lower().split())


@lru_cache(maxsize=None)
def load_device(name: str) -> dict:
    """按 device 名 (大小写不敏感) 加载并缓存 JSON。文件不存在抛 FileNotFoundError。"""
    key = _normalize(name)
    if not key:
        raise ValueError("device name is empty")
    path = _DEVICE_DIR / f"{key}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Device config not found: {path}. "
            f"Available: {available_devices()}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gpu_memory_gb(name: str) -> float:
    """单卡显存 (GB)。"""
    return float(load_device(name)["gpu_memory_gb"])


def memory_bandwidth_gbs(name: str) -> float:
    """显存带宽 (GB/s)。"""
    return float(load_device(name)["memory_bandwidth_gbs"])


def nvlink_bidirectional_bw_gbs(name: str) -> float:
    """NVLink (或同等卡间互联) 双向带宽 (GB/s)。"""
    return float(load_device(name)["interconnect"]["nvlink"]["bidirectional_bw_gbs"])


def tensor_core_bf16_tflops(name: str) -> float:
    return float(load_device(name).get("tensor_core_bf16_tflops", 0.0))


def tensor_core_fp8_tflops(name: str) -> float:
    """注: 设备不支持 FP8 时返回 0.0; 调用方应据此判断 (op_analyzer 侧请求会抛错)。"""
    return float(load_device(name).get("tensor_core_fp8_tflops", 0.0))


def available_devices() -> List[str]:
    """返回 configs/devices/*.json 的设备名 (小写, 不含 .json 后缀)。"""
    if not _DEVICE_DIR.exists():
        return []
    return sorted(p.stem for p in _DEVICE_DIR.glob("*.json"))


def normalize_device_name(name: str) -> str:
    """对外暴露的小写化函数 (与 op_analyzer::normalize_device_name 行为一致)。"""
    return _normalize(name)
