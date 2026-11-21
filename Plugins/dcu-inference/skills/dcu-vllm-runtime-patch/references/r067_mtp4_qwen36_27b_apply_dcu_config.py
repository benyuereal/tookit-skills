#!/usr/bin/env python3
"""Safely apply the validated K100AI/DCU config to Qwen3.6-27B-W8A8.

The upstream ModelScope checkpoint already contains W8A8 weights. This script
only replaces config.json with the exact compressed-tensors/DCU configuration
used for the published K100AI results. Safetensors are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

UPSTREAM_CONFIG_SHA256 = "fff7ade801ff7d51965ee2dccf36fb330faeb28a5f731a6f529d3f0f6245f61d"
DCU_CONFIG_SHA256 = "481eaef70106fb335db108707b0fb79db0efe543d8413b6dd3fa9ebb92990724"
CONFIG_NAME = "Qwen3.6-27B-W8A8-DCU.config.json"
EXPECTED_MODEL_TYPE = "qwen3_5"
EXPECTED_ARCH = "Qwen3_5ForConditionalGeneration"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow an unexpected upstream config hash after manual verification.",
    )
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    target = model_dir / "config.json"
    source = Path(__file__).resolve().parents[1] / "configs" / CONFIG_NAME

    if not target.is_file():
        raise SystemExit(f"missing model config: {target}")
    if not source.is_file():
        raise SystemExit(f"missing repository DCU config: {source}")

    current_hash = sha256(target)
    source_hash = sha256(source)
    if source_hash != DCU_CONFIG_SHA256:
        raise SystemExit(
            f"repository DCU config hash mismatch: {source_hash} != {DCU_CONFIG_SHA256}"
        )

    if current_hash == DCU_CONFIG_SHA256:
        print("DCU config is already applied; nothing to do.")
        return

    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("model_type") != EXPECTED_MODEL_TYPE:
        raise SystemExit(f"unexpected model_type: {data.get('model_type')!r}")
    if EXPECTED_ARCH not in data.get("architectures", []):
        raise SystemExit(f"unexpected architectures: {data.get('architectures')!r}")

    q = data.get("quantization_config") or data.get("compression_config")
    if not isinstance(q, dict) or q.get("format") != "int-quantized":
        raise SystemExit("checkpoint does not look like the expected INT8 W8A8 model")

    if current_hash != UPSTREAM_CONFIG_SHA256 and not args.force:
        raise SystemExit(
            "refusing to overwrite an unexpected config.json. "
            f"got {current_hash}, expected upstream {UPSTREAM_CONFIG_SHA256}. "
            "Use --force only after manually verifying that this is "
            "metax-tech/Qwen3.6-27B-W8A8."
        )

    backup = model_dir / "config.json.upstream.bak"
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)

    final_hash = sha256(target)
    if final_hash != DCU_CONFIG_SHA256:
        raise SystemExit(f"post-copy hash mismatch: {final_hash}")

    final = json.loads(target.read_text(encoding="utf-8"))
    fq = final.get("quantization_config") or {}
    print(f"applied DCU config: {target}")
    print(f"sha256: {final_hash}")
    print(
        "quantization: "
        f"{fq.get('quant_method')} / {fq.get('format')} / "
        f"ignore={len(fq.get('ignore', []))} entries"
    )
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
