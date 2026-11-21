#!/bin/bash
# Verify a rebuilt DCU vLLM image against the original container.
# Usage: verify_rebuild.sh <orig-container> <rebuilt-image> <test-container-name>
# Run on the source device (has docker + the original container).
set -e
ORIG="$1"; REBUILT="$2"; TESTC="$3"
KEYS="vllm torch torchvision triton flash_attn flash_mla lightop lmslim \
tensorizer setuptools msgspec safetensors numpy transformers xgrammar amdsmi \
cmake outlines_core llguidance gguf mistral_common numba grpcio timm \
runai-model-streamer cinotificate citest opencompass h5py"

docker start "$ORIG" >/dev/null 2>&1; sleep 2
docker exec "$ORIG" pip list --format=freeze 2>/dev/null | sort > /tmp/orig.txt
timeout 120 docker run --rm "$REBUILT" pip list --format=freeze 2>/dev/null | sort > /tmp/reb.txt

echo "=== DCU / key package version diff (rebuild vs original) ==="
python3 - "$KEYS" <<'PY'
import sys
keys = set(sys.argv[1].split())
def load(p):
    d={}
    for l in open(p):
        l=l.strip()
        if '==' in l:
            n,v=l.split('==',1); d[n.strip().lower()]=v
    return d
o=load('/tmp/orig.txt'); r=load('/tmp/reb.txt')
found=0
for k in sorted(set(o)|set(r)):
    if k in o and k in r and o[k]!=r[k] and k in keys:
        print(f"  {k}: orig={o[k]}  rebuild={r[k]}"); found+=1
if found==0: print("  all key packages identical or series-aligned (no mismatch flagged)")
print("  extra in rebuild:", [k for k in r if k not in o] or "none")
print("  missing in rebuild:", [k for k in o if k not in r] or "none")
PY

echo "=== vllm serve -h on a GPU-enabled container (lightop needs real DCU) ==="
docker rm -f "$TESTC" >/dev/null 2>&1 || true
docker run -d --privileged --name "$TESTC" \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --shm-size=200gb -v /opt/hyhal:/opt/hyhal:ro "$REBUILT" tail -f /dev/null
sleep 5
docker exec "$TESTC" bash -c 'vllm serve -h >/tmp/h.log 2>&1; echo EXIT=$?; tail -3 /tmp/h.log'
docker exec "$TESTC" hy-smi 2>/dev/null | grep -E 'HCU|Normal' | head -3 || echo "hy-smi unavailable"
