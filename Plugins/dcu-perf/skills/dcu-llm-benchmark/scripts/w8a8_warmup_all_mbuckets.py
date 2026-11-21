#!/usr/bin/env python3
"""Pre-compile ALL W8A8 m_ buckets (A' warmup) to eliminate first-request Triton JIT.

After a W8A8 INT8 tune cache is filled (see SKILL.md §3.11), the engine's built-in
warmup usually covers only a few shapes (e.g. 1024/2048/2230 -> m_={1024,2048});
every OTHER m_ bucket still pays a one-off Triton JIT on its first request (incl. the
6000-16383 -> m_=8192 bucket). This probe sends one prefill per m_ bucket so each
tuned config is compiled BEFORE real traffic.

Usage:  w8a8_warmup_all_mbuckets.py [PORT]     (or env Q38_WARMUP_PORT; default 8068)
Posts to the sglang /generate endpoint. Wire it as a post-ready hook: wait for the
port to listen -> run this -> then mark healthy / admit traffic.
Verify: after it finishes, the first request at a given length should equal steady.
"""
import json, time, sys, os, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("Q38_WARMUP_PORT", "8068")
GEN = f"http://127.0.0.1:{PORT}/generate"

# m_ buckets = everything the W8A8 rounding table can produce (SKILL.md §3.11.3),
# INCLUDING 8192 (the prefill cap for 6000-16383 inputs). Warm ALL of them.
M_BUCKETS = (
    [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    + [20,24,28,32,36,40,44,48,52,56,60,64]
    + [72,80,88,96,104,112,120,128,136,144,152,160]
    + [256,512,1024,2048,4096,8192]
)

def one(m, seed, max_new=4):
    # independent random tokens per bucket -> no shared prefix -> real prefill (no radix hit)
    ids = [(seed*7919 + i*131) % 149000 + 100 for i in range(m)]
    body = json.dumps({"input_ids": ids, "stream": True,
                       "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}).encode()
    req = urllib.request.Request(GEN, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                if first is None:
                    first = time.perf_counter() - t0
                break
    return first

print(f"=== A' warmup @ :{PORT}  pre-compile {len(M_BUCKETS)} m_ buckets (2..8192) ===", flush=True)
t0 = time.time()
for i, m in enumerate(M_BUCKETS):
    first = one(m, seed=2000 + i)
    if first is not None and first > 0.5:   # only print the slow (JIT) ones
        print(f"  m_={m:5d}: first={first:.2f}s  <- JIT compiled this bucket", flush=True)
    del first
print(f"=== done, {len(M_BUCKETS)} buckets, {(time.time()-t0)/60.0:.1f} min ===", flush=True)
print("All m_ buckets pre-compiled; first request at each length no longer pays cold JIT.", flush=True)
