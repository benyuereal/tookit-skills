import os
"""sitecustomize: install decode-attention patch in EVERY python process.
SGLang's scheduler/detokenizer are spawn subprocesses; an import-time patch
in the main script never reaches them. sitecustomize runs at interpreter
startup in each process, before sglang is imported, so the backend's
`from ... import decode_attention_fwd` binds the patched function.

Gate with an env var so normal interpreters stay untouched:
    PYTHONPATH=<dir-with-this-file> PATCH_ME=1 python ...
"""
if os.environ.get("PATCH_ME") == "1":
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import my_kernel_patch  # the module with install_patch()
        my_kernel_patch.install_patch(verbose=True)
    except Exception as e:
        print(f"[patch] sitecustomize install failed: {e}", flush=True)
