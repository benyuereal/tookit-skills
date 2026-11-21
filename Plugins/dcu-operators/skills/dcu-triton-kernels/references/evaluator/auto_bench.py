#!/usr/bin/env python3
"""
DCU 适配版官方评测器
源自：KernelSwift 2026 官方评测器 auto_bench.py
适配：Hygon BW1000 (gcu 后端) / C500 (gcu 后端) / Ascend 910B (npu 后端)

用法：
  # BW1000/C500
  DEVICE_ID=0 python3 auto_bench.py \
    --v0_file reference.py --v1_file solution.py --warmup 200 --repeat 500

  # Ascend 910B
  CANN_DEVICE_ID=0 python3 auto_bench.py \
    --v0_file reference.py --v1_file solution.py --warmup 200 --repeat 500

与官方评测器差异：
- 后端自动检测 gcu/cuda/npu/mlu，DCU 上自动适配
- 移除了上游路径强依赖，保持引用路径可配置
- 保留了完整的精度比较（atol=1e-2, rtol=1e-2）和中位数计时逻辑
"""

import argparse
import ast
import importlib.util
import statistics
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

import torch

# 尝试导入 NPU 后端（Ascend 910B）
try:
    import torch_npu  # noqa: F401 - 注册 Ascend torch.npu 后端
except ImportError:
    pass


class KsCompareError(Exception):
    pass


@dataclass
class CaseResult:
    name: str
    passed: bool
    v0_ms: float | None = None
    v1_ms: float | None = None
    speedup: float | None = None
    message: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DCU 适配版评测器：比较 PyTorch reference 与 Triton 优化实现的精度和性能。"
            "v0 文件须定义 Model/get_init_inputs/get_inputs，"
            "v1 文件须定义 ModelNew/get_init_inputs/get_inputs。"
        )
    )
    parser.add_argument("--v0_file", type=Path, required=True, help="PyTorch 基准实现 .py 文件")
    parser.add_argument("--v1_file", type=Path, required=True, help="Triton 优化实现 .py 文件")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--fail-fast", action="store_true", help="首个失败后停止")
    parser.add_argument("--full-traceback", action="store_true", help="打印完整 Python traceback")
    return parser.parse_args()


def _is_safe_literal(node):
    """判断 AST 节点是否为安全字面量（不含函数调用等副作用）"""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_safe_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _is_safe_literal(key)) and _is_safe_literal(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_literal(node.operand)
    return False


def _filter_module_ast(tree):
    """过滤 AST，移除可能产生副作用的语句"""
    kept_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            kept_nodes.append(node)
        elif isinstance(node, ast.Assign) and _is_safe_literal(node.value):
            kept_nodes.append(node)
        elif isinstance(node, ast.AnnAssign) and node.value is not None and _is_safe_literal(node.value):
            kept_nodes.append(node)
    tree.body = kept_nodes
    ast.fix_missing_locations(tree)
    return tree


def _iter_accelerators():
    """枚举所有可用加速器后端"""
    for name in ("gcu", "cuda", "npu", "mlu"):
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            if mod.is_available():
                yield name, mod
        except Exception:
            continue


def set_seed(seed):
    torch.manual_seed(seed)
    for _name, mod in _iter_accelerators():
        try:
            mod.manual_seed_all(seed)
        except Exception:
            pass


def sync_devices():
    for _name, mod in _iter_accelerators():
        try:
            mod.synchronize()
        except Exception:
            pass


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return [clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def load_ks_module(path: Path) -> types.ModuleType:
    if not path.exists():
        raise KsCompareError(f"file does not exist: {path}")
    if path.suffix != ".py":
        raise KsCompareError(f"expected a .py file, got: {path}")

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text()
    except OSError as exc:
        raise KsCompareError(f"failed to read {path}: {exc}") from exc

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise KsCompareError(f"syntax error in {path}:{exc.lineno}: {exc.msg}") from exc

    module_name = f"_ks_compare_{path.stem}_{abs(hash(path.resolve()))}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
    sys.modules[module_name] = module
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        code = compile(_filter_module_ast(tree), filename=str(path), mode="exec")
        exec(code, module.__dict__)
    except Exception as exc:
        raise KsCompareError(f"failed to load definitions from {path}: {exc}") from exc
    finally:
        sys.path[:] = old_sys_path
        sys.modules.pop(module_name, None)
    return module


def require_attr(module, attr_name, path: Path):
    if not hasattr(module, attr_name):
        raise KsCompareError(f"{path} must define `{attr_name}`.")
    return getattr(module, attr_name)


def as_args(value, description):
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise KsCompareError(
        f"{description} must return a list or tuple, got {type(value).__name__}."
    )


def compare_values(v0, v1, path, atol, rtol):
    if isinstance(v0, torch.Tensor) or isinstance(v1, torch.Tensor):
        if not isinstance(v0, torch.Tensor) or not isinstance(v1, torch.Tensor):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if v0.shape != v1.shape:
            raise KsCompareError(
                f"{path}: tensor shape mismatch: {v0.shape} vs {v1.shape}"
            )
        if v0.dtype.is_floating_point or v1.dtype.is_floating_point or v0.is_complex() or v1.is_complex():
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True):
                diff = (lhs - rhs).abs() if not (lhs.is_complex() or rhs.is_complex()) else (lhs - rhs).abs()
                diff_summary = (
                    f"empty tensor"
                    if diff.numel() == 0
                    else f"max_abs_diff={diff.max().item():.6e}, mean_abs_diff={diff.mean().item():.6e}"
                )
                raise KsCompareError(
                    f"{path}: tensor values differ; {diff_summary}, atol={atol}, rtol={rtol}"
                )
        else:
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            if not torch.equal(lhs, rhs):
                mismatch = (lhs != rhs).sum().item()
                raise KsCompareError(
                    f"{path}: tensor values differ; mismatched_elements={mismatch}"
                )
        return

    if isinstance(v0, (tuple, list)):
        if not isinstance(v0, type(v1)) or len(v0) != len(v1):
            raise KsCompareError(f"{path}: length mismatch")
        for i, (item0, item1) in enumerate(zip(v0, v1)):
            compare_values(item0, item1, f"{path}[{i}]", atol, rtol)
        return

    if isinstance(v0, dict):
        if set(v0) != set(v1):
            raise KsCompareError(f"{path}: dict keys mismatch")
        for key in sorted(v0):
            compare_values(v0[key], v1[key], f"{path}[{key!r}]", atol, rtol)
        return

    if v0 != v1:
        raise KsCompareError(f"{path}: values differ: {v0!r} vs {v1!r}")


def build_case(v0_path: Path, v1_path: Path, seed: int):
    v0_module = load_ks_module(v0_path)
    v1_module = load_ks_module(v1_path)

    model_cls = require_attr(v0_module, "Model", v0_path)
    model_new_cls = require_attr(v1_module, "ModelNew", v1_path)
    v0_get_init = require_attr(v0_module, "get_init_inputs", v0_path)
    v1_get_init = require_attr(v1_module, "get_init_inputs", v1_path)
    v0_get_inp = require_attr(v0_module, "get_inputs", v0_path)
    v1_get_inp = require_attr(v1_module, "get_inputs", v1_path)

    set_seed(seed)
    v0_init_args = as_args(v0_get_init(), f"{v0_path}: get_init_inputs()")
    set_seed(seed)
    v1_init_args = as_args(v1_get_init(), f"{v1_path}: get_init_inputs()")

    model = model_cls(*v0_init_args)
    model_new = model_new_cls(*v1_init_args)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model_new, "eval"):
        model_new.eval()

    set_seed(seed)
    v0_inputs = as_args(v0_get_inp(), f"{v0_path}: get_inputs()")
    set_seed(seed)
    v1_inputs = as_args(v1_get_inp(), f"{v1_path}: get_inputs()")

    if len(v0_inputs) != len(v1_inputs):
        raise KsCompareError(
            f"get_inputs argument count mismatch: {len(v0_inputs)} vs {len(v1_inputs)}"
        )
    return model, model_new, v0_inputs, v1_inputs


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        pass
    return None


def _first_input_device(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            d = _first_input_device(item)
            if d is not None:
                return d
    if isinstance(inputs, dict):
        for item in inputs.values():
            d = _first_input_device(item)
            if d is not None:
                return d
    return None


def _detect_target_device(model, model_new, v0_inputs, v1_inputs):
    """优先使用模型设备，其次输入设备，最后自动检测"""
    for m in (model, model_new):
        d = _get_model_device(m)
        if d is not None and d.type != "cpu":
            return d
    for inputs in (v0_inputs, v1_inputs):
        d = _first_input_device(inputs)
        if d is not None and d.type != "cpu":
            return d
    for name, _ in _iter_accelerators():
        return torch.device(name)
    raise KsCompareError(
        "no accelerator device available (gcu/cuda/npu/mlu); cannot run on CPU."
    )


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def compare_case(name, v0_path, v1_path, args):
    model, model_new, v0_inputs, v1_inputs = build_case(v0_path, v1_path, args.seed)
    target_device = _detect_target_device(model, model_new, v0_inputs, v1_inputs)

    # 尝试用参考实现的 state_dict 初始化优化模型
    try:
        model_new.load_state_dict(model.state_dict())
    except Exception:
        pass

    if hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model_new, "to"):
        model_new = model_new.to(target_device)

    v0_inputs = _move_to_device(v0_inputs, target_device)
    v1_inputs = _move_to_device(v1_inputs, target_device)
    # 优化模型的输入使用参考实现的克隆（保证相同数据）
    v1_inputs = clone_value(v0_inputs)

    # 精度校验
    with torch.no_grad():
        v0_output = model.forward(*v0_inputs)
    with torch.no_grad():
        v1_output = model_new.forward(*v1_inputs)
    compare_values(v0_output, v1_output, "output", args.atol, args.rtol)

    # 性能测速
    def time_model_forward(model, inputs):
        for _ in range(args.warmup):
            model.forward(*inputs)
        sync_devices()
        samples = []
        for _ in range(args.repeat):
            set_seed(args.seed)
            start = time.perf_counter()
            model.forward(*inputs)
            sync_devices()
            samples.append((time.perf_counter() - start) * 1000.0)
        return statistics.median(samples)

    v0_ms = time_model_forward(model, v0_inputs)
    v1_ms = time_model_forward(model_new, v1_inputs)
    speedup = v0_ms / v1_ms if v1_ms > 0 else float("inf")
    return CaseResult(name=name, passed=True, v0_ms=v0_ms, v1_ms=v1_ms, speedup=speedup)


def main():
    args = parse_args()
    v0_path = args.v0_file.resolve()
    v1_path = args.v1_file.resolve()
    if not v0_path.is_file():
        raise SystemExit(f"v0_file is not a file: {v0_path}")
    if not v1_path.is_file():
        raise SystemExit(f"v1_file is not a file: {v1_path}")

    name = str(v0_path)
    try:
        result = compare_case(name, v0_path, v1_path, args)
        print(
            f"PASS accuracy; v0={result.v0_ms:.6f} ms, "
            f"v1={result.v1_ms:.6f} ms, speedup={result.speedup:.3f}x"
        )
        passed, failed = 1, 0
    except Exception as exc:
        if args.full_traceback:
            traceback.print_exc()
        result = CaseResult(name=name, passed=False, message=str(exc))
        print(f"FAIL {exc}")
        passed, failed = 0, 1

    print(f"\nSummary: {passed} passed, {failed} failed, 1 total.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()