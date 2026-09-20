# Testing Airgapped-Readiness Locally (Network Off)

The most honest way to find out whether something *actually* needs the
internet is to turn the internet off and try it — this walks through
doing exactly that on this machine, before ever touching a real airgapped
box. If something breaks with the network off, it would have broken
airgapped too; better to find that here.

## What this does and doesn't prove

**Proves**: whether a `docker build`/`docker run`/`docker compose up`
sequence has a hidden live-network dependency (an `apt-get`/`pip install`
that isn't actually pinned to a local wheelhouse/`.deb` bundle, a
`--find-links` path that's wrong, a base image that isn't actually cached
locally yet).

**Doesn't prove**: that the *real* airgapped machine will behave
identically — different OS patch level, different Docker version,
different hardware (GPU or none) can still surface machine-specific
issues this local test can't catch. This is a strong pre-check, not a
substitute for a real airgapped run.

## Prerequisites — must all be true BEFORE you disconnect

Everything needed must already be sitting on disk / in Docker's local
image cache. Disconnecting first and discovering something's missing
just means reconnecting, fetching it, and starting over — check first:

```bash
# 1. Required images already pulled/built locally
docker images
#    Need at least: multi-model-inference:latest, and/or
#    spark-lean:latest + ner-translate-server:latest, and/or
#    ubuntu:22.04 / python:3.11-slim-bookworm (bases for building on-box)

# 2. Wheelhouse and .deb bundle already built (if testing the offline
#    dependency-update path — docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md)
ls wheels/ner_translate/ | wc -l      # expect ~63 files
ls debs/ner_translate/ | wc -l        # expect ~50 files

# 3. Model weights already downloaded
ls models/weights/gliner-multi models/weights/nllb-200-distilled-600M

# 4. Docker Desktop itself is healthy and NOT mid-restart
docker info
```

If any of these come up empty, fetch them now, while you still have
internet — that's the whole point of the wheelhouse/`.deb`-bundle
pattern: do the internet-dependent part once, ahead of time, on your
own schedule.

## Disconnecting

Any of these work — pick whichever is easiest to reverse on this machine:

- **Airplane mode** (Windows Settings → Network & Internet → Airplane
  mode) — simplest, one toggle, one toggle back.
- **Disable the network adapter**: `Get-NetAdapter | Disable-NetAdapter -Confirm:$false`
  (PowerShell, run as the adapter's actual name, e.g. `Wi-Fi` or
  `Ethernet`) — re-enable with `Enable-NetAdapter`.
- **Unplug the Ethernet cable / turn off Wi-Fi** at the hardware level —
  most literal, zero chance of a background process quietly reconnecting.

Docker Desktop's own WSL2 VM doesn't need to be restarted for this —
already-running containers and the local image cache aren't affected by
the host losing its network route to the internet; only NEW pulls or
live package-index calls are.

## What to actually run, once disconnected

Pick the path matching what you pre-staged above.

### A) Offline dependency-update path (reuse existing base image)

```bash
cd pytorch-spark-inference-platform
docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate .
```
This should succeed purely from `wheels/ner_translate/` (`--no-index
--find-links`) and `debs/ner_translate/` (`dpkg -i`) — no `apt-get
update`, no live PyPI resolution. If it fails here with a network-related
error, that pinpoints exactly which step still has a live dependency
that needs fixing before shipping this to a real airgapped machine.

### B) Waiter/kitchen split

```bash
bash deploy/scripts/setup_ner_translate_server.sh
docker exec ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"
```
Both `spark-lean:latest` and `ner-translate-server:latest` must already
be built (step 1 of the prerequisites) — this path doesn't build
anything new, only runs already-built images.

### C) Concurrency/UDF enhancements (no new images needed at all)

```bash
docker run --rm -v "$(pwd -W 2>/dev/null || pwd):/app" -w /app multi-model-inference:latest \
  python submit_job.py --model resnet18 --samples 32 --mode cpu_only --partitions 2 --engine udf
```
Everything this needs (`pandas`, `pyarrow`, `torch`) is already inside
`multi-model-inference:latest` — confirmed in
`docs/TEST_RESULTS_AND_TRADEOFFS.md` §4a. If this fails offline, something
about the image itself has a live dependency that wasn't supposed to be
there.

## Reading the failure, if there is one

| Symptom | Likely cause |
|---|---|
| `Could not resolve host` / `Temporary failure in name resolution` | Something is trying to hit a real DNS name — either a live `apt-get`/`pip` call that isn't `--no-index`, or a stray `docker pull` for an image that wasn't actually cached locally |
| `dpkg: dependency problems` | Not a network issue — see `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md`'s gotcha section (`.deb` bundle built against the wrong base image version) |
| Container ↔ container communication fails (Spark master/worker/kitchen can't reach each other) | Not a network issue either — this is Docker's own bridge network, works identically with or without internet; check `docker compose` service names / `NER_TRANSLATE_ROOT` path setup instead |
| Everything just works | This is the goal — reconnect and move on |

## Reconnecting afterward

Airplane mode off / `Enable-NetAdapter` / cable back in. Nothing about
this test needs cleanup — no state changes based on being offline vs
online, since none of the tested paths write anything network-dependent.
