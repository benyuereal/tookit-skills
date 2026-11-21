---
name: dcu-offline-container-deps
description: "Use when an offline DCU container lacks a Python package."
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# Offline DCU Container: Installing Missing Python Deps

A model repo's container on a no-external-network DCU device is missing a package
(omegaconf, an old pinned antlr4, a sdist-only dep, ...). The container has no PyPI
reach, so every install goes: **wheel on the networked machine → `docker exec -i cat >`
transfer → `pip install --no-index --no-deps` in the container → import-verify.**

## Procedure

1. **Identify the missing set** inside the container, one interpreter check at a time
   (`docker exec <c> python -c "import X"`): do not trust the repo's `requirements.txt`
   blindly — the base image already carries most of it, and the base image's
   transformers/torch versions are paired and must not be touched.

2. **Check the version pin before downloading.** Read the dependent's
   `requires_dist` from its PyPI JSON (`https://pypi.org/pypi/<dep>/<ver>/json`).
   Installing the *newest* of a pinned family can fail at **import time**, not install
   time — e.g. omegaconf 2.3.0 pins `antlr4-python3-runtime==4.9.*`, and 4.11.1 dies with
   `Could not deserialize ATN with version 3 (expected 4)`. Install the pinned older
   version.

3. **Obtain the wheel on the networked machine**:
   - has a wheel: take the `py3-none-any` URL from the PyPI JSON and curl it.
   - **sdist-only** (the antlr4 4.9.x case): `pip wheel <pkg>==<ver> --no-deps -w wheels/ <sdist-url>` —
     build the wheel where pip works; the offline container cannot build sdists.

4. **Transfer with the correct filename.**
   ```bash
   ssh <host> "docker exec -i <c> bash -c 'cat > /tmp/<name>-<ver>-py3-none-any.whl'" < local.whl
   ```
   The in-container filename MUST be PEP 427 compliant — pip rejects `omegaconf.whl` /
   `antlr4.whl` with `Invalid wheel filename (wrong number of parts)`. (Prefer this over
   `docker cp`, which on some hosts hits overlay quota and reports `no space left on
device` while disk is fine — see dcu-docker.)

5. **Install offline, dep-free:**
   ```bash
   docker exec <c> pip install --no-index --no-deps [--force-reinstall] /tmp/<wheel>.whl
   ```
   `--no-deps` is mandatory: `--no-index` cannot resolve transitive deps, so every dep
   is its own wheel step. Repeat per wheel. `--force-reinstall` when replacing a wrong
   version already present.

6. **Verify by import, never by install exit code** — and import the *full chain*
   (the dependent package, not just the dep), because the failures above surface at
   import: `python -c "import omegaconf"` after installing the wrong antlr4 is the
   smoke test.

## Pitfalls

- **`requires-python >=3.11` in a repo's pyproject is often a conservative declaration.**
  Before concluding you need a different python image, run
  `python3.10 -m compileall -q src/ scripts/` with the container's actual interpreter —
  exit 0 means the code is 3.10-compatible and you can use the image you already have.
- **Do not attempt `pip install <pkg>` inside the offline container at all.** It has no
  index to reach; the wheel-transfer route is the only path on an offline device.
- **Offline HF loading:** with weights staged locally, set
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and point every loader at the local path
  (`from_pretrained(<local_dir>)`, or the repo's config-override mechanism, e.g.
  `--set text_encoder.pretrained=<local_dir>` when the script defaults to a hub id).
  The hub id will otherwise try the network and fail with a confusing connection error.
- **Small-file verification after any cleanup/re-pull:** tokenizer/config files are the
  first thing to be lost and the first thing to fail; re-verify they exist and import
  before the big-model run.
- For the weight download itself onto the offline device, see
  `china-network-model-download` (probe, resume, integrity check); this skill is only
  the in-container dependency step.
