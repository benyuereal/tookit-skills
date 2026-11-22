#!/usr/bin/env python3
"""op-moe-align-permute 微基准（獨立入口；A/B 由 bench/ab.py 提供）。"""
import argparse, sys
from pathlib import Path
CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))
sys.path.insert(0, str(CARD.parents[3] / "scripts"))
from opbench import bench_fn  # noqa: E402
import _common as C  # noqa: E402

def main(argv=None):
    p = argparse.ArgumentParser(description="op-moe-align-permute 微基准")
    p.add_argument("--shape", default="decode-m32")
    p.add_argument("--variants", default="ref-heavy")
    a = p.parse_args(argv)
    shape = C.load_shape(a.shape); params = dict(shape.get("params") or {})
    inp = C.build_inputs(params, device="cpu")
    kw = C.runner_kwargs(params)
    for tag in [x.strip() for x in a.variants.split(",") if x.strip()]:
        fn = C.get_runner(tag, **kw)
        st = bench_fn(lambda: fn(*inp[:-1], **kw))
        print("%-16s median=%.4fms mad=%.4fms" % (tag, st["median_ms"], st["mad_ms"]))
    return 0

if __name__ == "__main__":
    sys.exit(main())
