# Verified nightly/dtk version pins (vLLM 0.15.1 / DTK 26.04-rc4)

All confirmed AVAILABLE on `https://pypi.sourcefind.cn/nightly/dtk/` (not just
`INSTALLED` — verified via `pip index versions` showing the version line WITHOUT
the `INSTALLED` prefix). Use `scripts/probe_nightly.py` to re-check before a build.

## Exact-match (byte-identical with original image)
- vllm: `0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba`
- torchvision: `0.24.0+das.opt1.dtk2604.torch290.2604091152.g66069c`
- triton: `3.3.0+das.opt2.dtk2604.torch290.20260331.g31542e`
- lightop: `0.6.0+das.dtk2604.torch290.2604091144.ge8aebd`
- lmslim: `0.3.1+das.opt4.dtk2604.torch290.2604091256.gafc115`
- amdsmi: `24.5.3+02cbffb.dirty`
- transformers: `5.2.0`
- xgrammar: `0.1.29`
- cmake: `3.29.0.1` (vLLM metadata wants 3.29.0; only 3.29.0.1 exists)

## Series-aligned (original exact build number NOT on nightly)
- torch: original `2.9.0+...20260331.g4e3c1e7` -> pin
  `2.9.0+das.opt1.dtk2604.20260331.g4e3c1e7` (may fall back to another 2.9.0+
  nightly build on retry; acceptable per user rule "pick higher when not exact").
- flash_attn: original `2.8.3+...202604031932` NOT on nightly -> use
  `2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be` (torch290 series).
- flash_mla: original `1.0.0+...202604031618` NOT on nightly -> use
  `1.2.0+das.opt1.dtk2604.torch290.2605281449.ge3aed6` (1.2.0 torch290 series,
  version even higher — acceptable).

## Public packages (Tsinghua mirror)
- opencompass: `0.5.1` (general public pkg; sourcefind /open mirror unstable)
- numpy: `1.25.0` (re-pin at END of install script)
- setuptools: `79.0.1`

## DTK
- Original container `.dtk_version`: `DTK-26.04-rc4` (built Wed Mar 18 2026).
- External URL (z-file `/file/1/...` direct link):
  `https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz`
  VERIFIED 2026-08-06 (build #26): downloads fine INSIDE containers incl. the `docker build`
  RUN stage (~13MB/s, 3.3GB compressed -> 15GB extracted), and the extracted DTK's
  `.dtk_version` is `DTK-26.04-rc4` — matching the original (rc4, 1 week newer build).
  NOTE: `/1/main/...` returns a 200 SPA shell (not a file); the real file URL is `/file/1/...`.
  Host-side curl/wget TIMES OUT on this source (machine-room <device-B>) — do NOT probe from the
  host. Default: `RUN wget -q --tries=0 --timeout=60 -O /tmp/dtk.tar.gz <link> && tar -xzf ...
  --strip-components=1`; keep `COPY dtk /opt/dtk` only as fallback for build hosts whose whole
  network cannot reach the source.
