# Cleanup of failed `docker build` leftovers on the source device

Rebuilding an image from scratch (esp. a 30+ GB DCU image) generates MANY
intermediate layers / images across failed attempts. Leaving them wastes
50–200 GB+ per session. The naive `docker image prune -f` does NOT fully
clean them — here is the gotcha and the correct order.

## Gotcha (why `prune -f` leaves dangling behind)
A failed `docker build` leaves anonymous intermediate **containers** (docker
auto-names them, e.g. `naughty_burnell`, `epic_cori`; status `Exited (1)`).
Those containers still *reference* the build's intermediate image layers,
so the layers are NOT "unreferenced" and `docker image prune -f` skips them.
`docker images -f dangling=true` shows them, but `docker image rm <id>`
fails with:
```
conflict: unable to delete <id> (must be forced) - image is being used by stopped container <cid>
```

## Correct cleanup order
1. Find the anonymous stopped containers holding the layers:
   ```bash
   docker ps -a --filter status=exited --format '{{.ID}}\t{{.Names}}\t{{.Status}}' \
     | grep -vE '<internal-user>|<internal-user>|<internal-user>|<other known-good name prefixes>'
   ```
   Anonymous build-residue containers have docker-generated names (adjective_noun)
   and `Exited (1)`; named containers (<internal-user>-*, etc.) are real — DO NOT touch.
2. Remove those anonymous containers:
   ```bash
   for c in <cid...>; do docker rm -f "$c"; done
   ```
3. Now prune dangling images (this will now actually delete the layers):
   ```bash
   docker image prune -f
   docker images -f dangling=true   # should be empty
   ```

## Safety rules (do NOT delete)
- Never `docker image rm -f` a layer that `docker history <good-image>` lists
  (it belongs to a surviving image). Check with:
  ```bash
  docker history <good-image-id> | grep <layer-id>   # skip if it appears
  ```
- Only delete images with NO repo tag AND not referenced by any named/running
  container. `prune -f` is safe for this; the container-first step above just
  frees the references prune missed.
- NEVER run `docker image prune` (or `-a`) DURING an in-flight build — it can
  delete layers the running build still needs (caused a build #18 "no such
  image" failure). Only prune after the build process has fully exited.
- Leave other users' containers (<internal-user>-*, <internal-user>, etc.) strictly alone — only
  clean YOUR session's build residue (anonymous / your own container names).

## Result from a real session (2026-08-06, device <device-B>)
Failed vLLM rebuild attempts left ~10 dangling images + 6 anonymous exited
containers holding 18.4 GB layers each. After container-rm + prune,
**~200 GB reclaimed**, dangling count → 0, while the success image
`vllm015-rebuild-final` (f9e2a57c9290), the original container, and the
validation container were all left intact.
