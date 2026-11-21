# Rebuild Playbook — worked example (vLLM 0.15.1 / DTK 26.04-rc4)

## Source image
`<internal_registry>/dcu/admin/base/custom:vllm015-ubuntu22.04-dtk26.04-0409-modelzoo`
running as container `<container_vllm015>` on device <device-B> (<<internal-ip>>).

## Step 1 — recover structure (on source device)
```bash
docker history --no-trunc <image> > img_history.txt
docker inspect <image> > img_inspect.json
docker start <container>
docker exec <container> cat /install_dep.sh        # original install script
docker exec <container> cat /etc/profile.d/env.sh  # DTK env injection
docker exec <container> cat /opt/dtk/.dtk_version  # -> DTK-26.04-rc4
```
Reproduce every RUN/ENV/CMD 1:1 from `docker history`. Three ENV names
(LD_LIBRARY_PATH/PATH/C_INCLUDE_PATH) get overwritten — write only final values
(from `docker inspect` Config.Env).

## Step 2 — build-context gaps
- `/opt/dtk` (15GB): PREFER external URL (user-approved replacement). Default in Dockerfile:
  ```dockerfile
  # DTK-26.04 Ubuntu22.04 x86_64; .dtk_version=DTK-26.04-rc4 (rc in file, not visible version)
  RUN wget -q --tries=0 --timeout=60 -O /tmp/dtk.tar.gz \
      "https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz" \
      && mkdir -p /opt/dtk && tar -xzf /tmp/dtk.tar.gz -C /opt/dtk --strip-components=1 \
      && rm -f /tmp/dtk.tar.gz
  # Fallback if build host cannot reach the source: COPY dtk /opt/dtk  (prepare dtk/ into ctx first)
  ```
  NOTE: the `/file/1/...` link downloads fine INSIDE containers (incl. build RUN stage) even
  though it times out from the machine-room host — verified on build #26.
- Small `COPY` files (`install_dep.sh`, `env.sh`): extract real content, COPY them.
- `/usr/share/misc/magic` etc.: usually redundant (file pkg symlink), skip.

## Step 3 — source layering
`~/.pip/pip.conf` written in Dockerfile:
```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
extra-index-url = https://pypi.sourcefind.cn/nightly/dtk/
```
Install script (install_dep.sh) two stages:
```bash
PIP_OPT="--timeout 120 --retries 10 --resume-retries 50"
VLLM_INDEX="https://pypi.sourcefind.cn/nightly/dtk/"
TSINGHUA="https://pypi.tuna.tsinghua.edu.cn/simple"
# STAGE 1 — lock ALL DCU packages (incl torch 2.9.0+dcu) before any public pkg
pip install $PIP_OPT "torch==2.9.0+..." -i $VLLM_INDEX
pip install $PIP_OPT "torchvision==0.24.0+..." -i $VLLM_INDEX
pip install $PIP_OPT "triton==3.3.0+..." -i $VLLM_INDEX
pip install $PIP_OPT "flash_attn==2.8.3+..." -i $VLLM_INDEX
pip install $PIP_OPT "flash_mla==1.2.0+..." -i $VLLM_INDEX
pip install $PIP_OPT "lightop==0.6.0+..." -i $VLLM_INDEX
pip install $PIP_OPT "lmslim==0.3.1+..." -i $VLLM_INDEX
pip install $PIP_OPT "amdsmi==24.5.3+..." -i $VLLM_INDEX
# STAGE 2 — public pkgs (torch already satisfied, won't pull official torch)
pip install $PIP_OPT "opencompass==0.5.1" -i $TSINGHUA
pip install $PIP_OPT "citest==0.0.8" -i $VLLM_INDEX
pip install $PIP_OPT "cinotificate==0.0.3" -i $VLLM_INDEX
# vLLM: exact wheel, no-deps, then fill deps excluding cmake
VLLM_SPEC="vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba"
pip download $PIP_OPT "$VLLM_SPEC" --no-deps -d /tmp/vwhl -i $VLLM_INDEX
pip install --no-deps /tmp/vwhl/*.whl
# parse METADATA Requires-Dist, skip cmake + torch, pip install the rest (--resume-retries 50)
# finally re-pin numpy at the very end (opencompass may have bumped it to 2.x)
pip install $PIP_OPT "numpy==1.25.0" -i $TSINGHUA
```

## Step 4 — build + verify
```bash
docker build -f Dockerfile.vllm015 -t vllm015-rebuild-final .   # never prune mid-build
# verify (GPU container!):
docker run -d --privileged --name test \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --shm-size=200gb -v /opt/hyhal:/opt/hyhal:ro vllm015-rebuild-final tail -f /dev/null
docker exec test bash -c 'vllm serve -h'      # must exit 0 (lightop needs real GPU)
docker exec test hy-smi                       # 8 HCU visible
```
See `../scripts/verify_rebuild.sh` for the automated freeze-diff + serve check.
See `version-pins.md` (same directory) for the exact verified nightly versions.
