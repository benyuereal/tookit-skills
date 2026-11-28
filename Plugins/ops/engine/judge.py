#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""judge 对象层：从卡片/描述生成「可执行的判据 + 冻结容差」。

契约见 `engine/JUDGE-CONTRACT.md` 与 lead 的
`updates/handoff/HANDOFF-2026-11-28-object-layer-plan.md`。

    python3 -I Plugins/ops/engine/judge.py generate \
        --card Plugins/ops/skills/op-silu-mul --out .artifacts/arm-C/op-silu-mul_x/judge

产物（<out>/）：
    judge.py          # 可执行，唯一被允许的调用方式
    tolerance.json    # 冻结容差（含 sha256 / calibration_cmd / evidence_kind）
    CALIBRATION.md    # 人读的校准记录

**fail-closed**：生成不出 judge 或校准不出容差 → rc=2，且**不写** tolerance.json。
宁可让 run 跑不了，也不许拿未校准的容差进账本。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

MAX_ULP_DEFAULT = 2.0
COSINE_DEFAULT = 0.99999
RTOL_DEFAULT = 0.02
ATOL_DEFAULT = 0.01

VENDOR_ROLES = {"competitor", "primary", "fallback"}

_JUDGE_TMPL = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""judge（生成物）。用法唯一：

    python3 judge.py --shape <name> --json <out.json> [--variant <name>]
"""
import argparse
import json
import sys
from pathlib import Path

CARD = Path(r"{card}")
ENGINE = Path(r"{engine}")
sys.path.insert(0, str(CARD))
# engine/ 放在末尾：卡片自带的同名模块仍然优先，只在卡片没有 `_loader.py` 时
# 才命中引擎那一份（正常情况下一定命中）。
if str(ENGINE) not in sys.path:
    sys.path.append(str(ENGINE))

import torch  # noqa: E402


def _load_common():
    """加载卡的 `_common.py`（key `_judge_common`，带 `sys.modules` 缓存）。"""
    import _loader
    return _loader.load_module("_judge_common", CARD / "_common.py")


def _params(shape_key):
    try:
        import yaml
        data = yaml.safe_load((CARD / "shapes.yaml").read_text(encoding="utf-8")) or {{}}
    except Exception:
        import sys as _s
        _s.path.insert(0, str(Path(r"{repo}") / "scripts"))
        from opverify import parse_yaml
        data = parse_yaml((CARD / "shapes.yaml").read_text(encoding="utf-8")) or {{}}
    key = shape_key if shape_key in data else next(iter(data))
    return key, dict((data.get(key) or {{}}).get("params") or {{}})


def _ulp(a, b):
    """bf16 量级的 ULP 误差（粗略但可复现：|a-b| / spacing(ref)）。"""
    ref = a.float()
    eps = torch.finfo(torch.bfloat16).eps
    spacing = ref.abs().clamp_min(1e-30) * eps
    return float((a.float() - b.float()).abs().div(spacing).max().item()) if a.numel() else 0.0


def _cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    na = a.norm()
    nb = b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="s0")
    ap.add_argument("--json", default=None)
    ap.add_argument("--variant", default=None, help="被测变体；缺省用 ref-heavy 自检")
    a = ap.parse_args(argv)
    C = _load_common()
    key, params = _params(a.shape)
    try:
        inp = C.build_inputs(params, seed=0, device="cpu")
    except TypeError:
        inp = C.build_inputs(params, seed=0)
    args = inp[:-1] if (inp and isinstance(inp[-1], dict)) else inp
    ref = C.reference(*args, params=params) if hasattr(C, "reference") else None
    if ref is None:
        raise SystemExit("卡片没有 reference()，judge 无法校准")
    ref = ref[0] if isinstance(ref, tuple) else ref
    if a.variant:
        runner = C.get_runner(a.variant, **(C.runner_kwargs(params)
                                            if hasattr(C, "runner_kwargs") else {{}}))
        got = runner(*args)
    else:
        got = ref
    got = got[0] if isinstance(got, tuple) else got
    nonfinite = int((~torch.isfinite(got)).sum().item())
    d = {{"shape": key,
         "max_ulp": 0.0 if got is ref else _ulp(got, ref),
         "cosine": 1.0 if got is ref else _cos(got, ref),
         "nonfinite": nonfinite,
         "ref_absmax": float(ref.abs().max().item()) if ref.numel() else 0.0,
         "evidence_kind": r"{evidence}"}}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(d, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _sha256_normalized(obj: dict) -> str:
    """除 sha256 字段外的规范化 sha256。"""
    d = {k: v for k, v in obj.items() if k != "sha256"}
    blob = json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict:
    """优先 PyYAML；并发 YAML 有 map/list 混用，opverify.parse_yaml 会报错
    （实测 2026-11-28：op-fused 的 variants.yaml 解析失败 → evidence_kind
    被误判成 self-consistent-only）。"""
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:                                        # noqa: BLE001
        try:
            import sys as _s
            _s.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
            from opverify import parse_yaml
            return parse_yaml(path.read_text(encoding="utf-8")) or {}
        except Exception:                                    # noqa: BLE001
            return {}


def _evidence_kind(card: Path) -> str:
    """按 variants 的角色判证据种类。

    competitor/primary/fallback 可用作独立核验 → vendor_library；
    只有 correctness_only 的参考 → self-consistent-only（claim 必须 FAIL）。
    """
    p = card / "variants.yaml"
    if not p.is_file():
        return "self-consistent-only"
    d = _read_yaml(p)
    for v in d.get("variants") or []:
        if isinstance(v, dict) and str(v.get("role") or "") in VENDOR_ROLES:
            return "vendor_library"
    return "self-consistent-only"


def _smallest_shape(card: Path) -> str:
    d = _read_yaml(card / "shapes.yaml")
    best, best_n = "s0", None
    for k, v in d.items():
        params = dict((v or {}).get("params") or {})
        n = 1
        for x in params.values():
            if isinstance(x, int) and x > 0:
                n *= x
        if best_n is None or n < best_n:
            best, best_n = k, n
    return best


def generate(card: Path, out: Path) -> int:
    card = card.resolve()
    out = out.resolve()
    if not (card / "_common.py").is_file():
        print("✗ 卡片没有 _common.py：无法生成 judge", file=sys.stderr)
        return 2
    # ---- 先验证 reference 能跑（CPU、最小形状）。跑不了就不写任何产物。
    if Path(__file__).resolve().parent.joinpath("nextstep.py").is_file():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_j_gen", str(card / "_common.py"))
        m = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(card))
        spec.loader.exec_module(m)
        shape = _smallest_shape(card)
        params = {}
        try:
            import sys as _s
            _s.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
            from opverify import parse_yaml
            data = parse_yaml((card / "shapes.yaml").read_text(encoding="utf-8")) or {}
            params = dict((data.get(shape) or {}).get("params") or {})
        except Exception:                                    # noqa: BLE001
            pass
        try:
            inp = m.build_inputs(params, seed=0, device="cpu")
        except TypeError:
            inp = m.build_inputs(params, seed=0)
        args = inp[:-1] if (inp and isinstance(inp[-1], dict)) else inp
        ref = m.reference(*args, params=params)
        ref = ref[0] if isinstance(ref, tuple) else ref
        if getattr(ref, "numel", lambda: 0)() == 0:
            raise RuntimeError("reference 返回空")
    except NotImplementedError:
        print("✗ reference()/build_inputs() 还是占位符 —— 第 2 步未完成，不能生成 judge",
              file=sys.stderr)
        return 2
    except Exception as e:                                   # noqa: BLE001
        print("✗ reference 跑不起来：%s: %s —— fail-closed，不写容差" % (type(e).__name__, e),
              file=sys.stderr)
        return 2

    evidence = _evidence_kind(card)
    out.mkdir(parents=True, exist_ok=True)
    judge_py = out / "judge.py"
    judge_py.write_text(_JUDGE_TMPL.format(
        card=str(card), repo=str(Path(__file__).resolve().parents[3]),
        engine=str(Path(__file__).resolve().parent),
        evidence=evidence), encoding="utf-8")
    judge_py.chmod(0o755)

    tol = {
        "max_ulp": MAX_ULP_DEFAULT,
        "cosine": COSINE_DEFAULT,
        "rtol": RTOL_DEFAULT,
        "atol": ATOL_DEFAULT,
        "evidence_kind": evidence,
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # 校准覆盖的形状：目前只对最小形状做参考自检，明确写出来，
        # 避免下游误以为所有形状都校准过（lead 2026-11-28 遗留项 2）。
        "shapes": [shape],
        "calibration_cmd": ("python3 -I %s generate --card %s --out %s"
                            % (Path(__file__).resolve(), card, out)),
    }
    tol["sha256"] = _sha256_normalized(tol)
    (out / "tolerance.json").write_text(
        json.dumps(tol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cal = out / "CALIBRATION.md"
    cal.write_text(
        "# 校准记录\n\n"
        "- 卡片：`%s`\n- 形状：`%s`\n- 证据种类：`%s`\n"
        "- 生成时间：%s\n- 命令：`%s`\n\n"
        "%s\n"
        % (card, shape, evidence, tol["calibrated_at"], tol["calibration_cmd"],
           "> **无独立核验**：只有卡片自身的参考实现。`claim` 会判 FAIL，"
           "不能声称数值正确。\n" if evidence == "self-consistent-only"
           else "> 有独立核验（vendor/独立实现），`claim` 可放行数值正确。\n"),
        encoding="utf-8")

    print("✓ judge 生成：%s" % out)
    print("  evidence_kind=%s  shape=%s" % (evidence, shape))
    print("  tolerance.json sha256=%s" % tol["sha256"][:16])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="judge")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="从卡片生成 judge 与冻结容差")
    g.add_argument("--card", required=True)
    g.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "generate":
        return generate(Path(a.card), Path(a.out))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
