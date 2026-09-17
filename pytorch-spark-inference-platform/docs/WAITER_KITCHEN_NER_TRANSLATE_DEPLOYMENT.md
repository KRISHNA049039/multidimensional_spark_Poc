# `ner_translate` — Waiter/Kitchen Architecture & Airgapped Deployment

**Status: implemented and verified end-to-end on real GPU infrastructure
(AWS `g4dn.xlarge`, Tesla T4) as of 2026-09-17.** This is
`docs/MODEL_CONTAINER_ISOLATION.md`'s **Option B**, made real for the
`ner_translate` pipeline. Read that doc first for the design rationale —
this one is the concrete "what to build, what to download, how to run it"
reference.

## Why this exists

Two separate problems pushed toward this design:

1. **Dependency isolation ("BYOM")** — every model/pipeline should be able
   to bring its own dependency set (torch version, CUDA version, transformer
   library versions) without any risk of one model's `pip install` silently
   changing a version another model relies on. See
   `docs/MODEL_CONTAINER_ISOLATION.md` for the numpy-version incident that
   motivated this.
2. **Airgapped shipping cost** — this framework is built on an
   internet-connected machine, then physically transferred (optical media /
   secure transfer) to an airgapped system. Before this change, *any*
   dependency update meant re-shipping the entire multi-GB image. The
   architecture below separates "things that rarely change" (base
   Spark/Java layer) from "things that change per model" (a model's own
   container) from "things that change often" (application code, via a
   wheelhouse — see `docs/internet_to_airgapped_transfer.md`'s Phase 6 for
   the code-only-update pattern this pipeline still supports).

## Architecture: waiter and kitchen

```
                     ┌─────────────────────────────┐
                     │   spark-lean:latest ("waiter") │
                     │   Ubuntu + Java 17 + Spark 3.5.1 │
                     │   + pyspark + requests ONLY      │
                     │   NO torch, NO CUDA, NO model deps│
                     └───────────────┬─────────────────┘
                                     │ Spark standalone protocol (port 7077)
                     ┌───────────────┴─────────────────┐
                     │   ner-translate-master  (driver)  │
                     │   ner-translate-worker  (executor) │
                     └───────────────┬─────────────────┘
                                     │ HTTP  POST /predict
                                     ▼
                     ┌─────────────────────────────────┐
                     │ ner-translate-server:latest ("kitchen")│
                     │ python:3.11-slim + torch/CUDA cu126    │
                     │ + gliner/transformers/tesseract/etc.   │
                     │ Loads GLiNER + NLLB once at startup    │
                     └─────────────────────────────────┘
```

- **Waiter** (`spark-lean:latest`) — runs the Spark master and worker(s).
  Knows nothing about any specific model. `mapPartitions` closures in
  `inference/text_pipeline_engine.py`'s `run_text_pipeline_job_via_service()`
  just POST batches of file paths to the kitchen's `/predict` endpoint over
  HTTP and collect the JSON response. **This exact image is reused for every
  future kitchen-backed pipeline** — adding a tenth model never touches this
  image.
- **Kitchen** (`ner-translate-server:latest`) — a plain FastAPI/uvicorn
  service (`models/pipelines/ner_translate/serve.py`). Owns 100% of this
  pipeline's dependencies (torch, CUDA, gliner, transformers, tesseract).
  Loads GLiNER + NLLB once at container startup, not per Spark task. One
  kitchen image per model/pipeline.
- Neither image bakes in application code or model weights. Both are
  supplied at container start via the same filesystem mount
  (`../models:/app/models`, `../data:/app/data`) — matching how this
  framework already treats weights as filesystem-provided
  (`docs/BRING_YOUR_OWN_MODEL.md`), not baked into any image.

## What changed to make this work (2026-09-17)

| # | File(s) | Change |
|---|---|---|
| 1 | `models/pipelines/ner_translate/serve.py` (new) | FastAPI wrapper around the pipeline's existing `load()`/`run()` — one `/predict` endpoint, model loaded once at startup |
| 2 | `inference/text_pipeline_engine.py` | New `run_text_pipeline_job_via_service()` — same job shape as the in-process `run_text_pipeline_job()`, but each Spark partition POSTs to the kitchen instead of importing the model |
| 3 | `deploy/Dockerfile` | New `lean` target (`FROM spark-base`, no torch) alongside the existing `final` target |
| 4 | `deploy/Dockerfile.ner_translate_server` (new) | The kitchen image — torch/CUDA + this pipeline's deps only, no Spark/Java, no application code |
| 5 | `deploy/docker-compose.ner_translate_server.yml` (new) | Wires `ner-translate-master` + `ner-translate-worker` (both `spark-lean:latest`) to `ner-translate-server` (the kitchen) on one Docker network |
| 6 | `models/pipelines/manifest.json` | Added `"service_url"` per pipeline entry (e.g. `http://ner-translate-server:8000`) alongside the existing `"master_url"` |
| 7 | `submit_pipeline_job.py` | Auto-detects whether the service URL resolves and routes through HTTP if so, otherwise falls back to in-process (Option A) or `local[4]` |

**Three image-bloat bugs found and fixed while validating this on AWS**
(the images went from **11.9GB combined → 3.82GB combined** as a result):

| Bug | Root cause | Fix |
|---|---|---|
| Lean image carried a 3GB dead wheelhouse | `deploy/Dockerfile`'s `lean` stage did `COPY . .`, which also copied `wheels/ner_translate/` (torch/CUDA `.whl` files it never installs) | Replaced `COPY . .` with `RUN --mount=type=bind,source=.,target=/build-context find ... ! -name wheels ! -name .git -exec cp -a {} /app/ \;` — a bind mount is never committed as an image layer |
| Kitchen image shipped the wheelhouse twice | `COPY wheels/ner_translate/ /tmp/wheels/` then `RUN pip install ... && rm -rf /tmp/wheels` — Docker layers are additive, so the `COPY` layer's 3GB stayed in the image even after the `rm -rf` | Same bind-mount fix: `RUN --mount=type=bind,source=wheels/ner_translate,target=/tmp/wheels pip install ...` — the wheelhouse is never a layer at all |
| A rebuild picked up its own prior export | A `docker save \| gzip > *.tar.gz` output sitting in the build directory got copied into the *next* build via the same bind-mount-copy | Added `*.tar.gz` to `.dockerignore` |

If you're adding a new pipeline's Dockerfile, copy the bind-mount pattern
above rather than `COPY wheels/<pipeline>/ ...` + `rm -rf` — the latter
looks like it removes the weight but doesn't.

## How many images you need on the airgapped system

**Two, total — not one per model.**

| Image | Size (gzip) | Rebuilt when |
|---|---|---|
| `spark-lean.tar.gz` | **837 MB** | Only if the Spark/Java/pyspark version changes, or `submit_pipeline_job.py`/`inference/*.py` driver code changes — shared across **every** kitchen-backed pipeline |
| `ner-translate-server.tar.gz` | **3.0 GB** | Only if `ner_translate`'s own dependencies (torch, gliner, transformers, tesseract) change |

Adding an 11th pipeline later means building **one new kitchen image**
(`ner-translate-server` for a `sentiment-server`, `ocr-server`, etc.) — the
waiter image is untouched and doesn't need re-downloading.

**Besides the two images**, the airgapped system also needs (neither is
baked into either image, both are filesystem-mounted at runtime):

| Artifact | Size | Source |
|---|---|---|
| Application code (`inference/`, `models/`, `deploy/`, `submit_pipeline_job.py`, etc.) | A few MB | `git bundle` or `tar czf` of the repo, excluding `weights/`, `wheels/`, `.git/`, `node_modules/` |
| Model weights (`models/weights/gliner-multi/`, `models/weights/nllb-200-distilled-600M/`) | **~4.5 GB** (2.2GB + 2.4GB) | Downloaded once on the internet-connected machine (see `mt_ner_all_formats.py`'s module docstring for the exact `from_pretrained(...).save_pretrained(...)` commands) |
| Test data (optional, `data/ner_samples/`) | ~50 KB | Repo |

**Total transfer for a first-time `ner_translate` deployment: ~8.4 GB**
(837 MB + 3.0 GB images + 4.5 GB weights + negligible code/data).

## Complete setup flow

### Phase 1 — Internet-connected build machine

```bash
git clone <repo-url> && cd pytorch-spark-inference-platform

# 1. Download model weights once (see mt_ner_all_formats.py's docstring
#    for the exact from_pretrained().save_pretrained() commands)
python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi-v2.1').save_pretrained('./models/weights/gliner-multi')"
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
m = 'facebook/nllb-200-distilled-600M'
AutoTokenizer.from_pretrained(m).save_pretrained('./models/weights/nllb-200-distilled-600M')
AutoModelForSeq2SeqLM.from_pretrained(m).save_pretrained('./models/weights/nllb-200-distilled-600M')
"

# 2. Build this pipeline's wheelhouse (pinned, offline-installable deps)
bash deploy/scripts/build_ner_translate_wheelhouse.sh

# 3. Build both images
export DOCKER_BUILDKIT=1
docker build --target lean -t spark-lean:latest -f deploy/Dockerfile .
docker build -t ner-translate-server:latest -f deploy/Dockerfile.ner_translate_server .

# 4. Verify locally (see docs/TESTING_BEFORE_EXPORT.md for the full tiered checklist)
docker compose -f deploy/docker-compose.ner_translate_server.yml up -d
docker exec ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"

# 5. Export images
docker save spark-lean:latest | gzip > spark-lean.tar.gz              # ~837 MB
docker save ner-translate-server:latest | gzip > ner-translate-server.tar.gz  # ~3.0 GB

# 6. Package code + weights separately (small, changes more often than the images)
tar czf ner-translate-code.tar.gz \
  inference/ models/ deploy/ submit_pipeline_job.py data/ner_samples/ \
  --exclude=models/weights
tar czf ner-translate-weights.tar.gz models/weights/gliner-multi models/weights/nllb-200-distilled-600M
```

### Phase 2 — Physical transfer

Same options as `docs/internet_to_airgapped_transfer.md`'s Phase 2
(encrypted USB, data diode, optical media, approved transfer system).
Transfer all four artifacts: the two `.tar.gz` images, the code tarball,
and the weights tarball. Verify with `sha256sum` on both sides.

### Phase 3 — Airgapped system

```bash
# 1. Load both images
gunzip -c spark-lean.tar.gz | docker load
gunzip -c ner-translate-server.tar.gz | docker load
docker images   # confirm spark-lean:latest and ner-translate-server:latest

# 2. Unpack code + weights into the SAME folder — models/weights/ from the
#    second tarball must merge into models/ from the first, not sit
#    somewhere else
mkdir -p /opt/ner-translate && cd /opt/ner-translate
tar xzf /path/to/ner-translate-code.tar.gz -C .
tar xzf /path/to/ner-translate-weights.tar.gz -C .

# 3. Bring the cluster up — don't call `docker compose` by hand here.
#    setup_ner_translate_server.sh validates every path above actually
#    exists (and that both images are loaded) BEFORE starting anything, and
#    fails with a specific "what's missing" message instead of containers
#    coming up against an empty bind mount. This is what makes the same
#    command safe to run identically on an internet-connected dev machine
#    and on an airgapped box — it never touches the network, only checks
#    the local filesystem and sets the mount path explicitly.
bash deploy/scripts/setup_ner_translate_server.sh /opt/ner-translate
# (omit the path argument if you're running it from inside the repo root
# itself, e.g. during internet-side testing — it defaults to its own
# location's repo root)

# 4. Submit a job (the script prints this exact command at the end, with
#    NER_TRANSLATE_ROOT already filled in)
NER_TRANSLATE_ROOT=/opt/ner-translate docker exec ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"
```

If `setup_ner_translate_server.sh` reports a `[MISSING]` line, that's the
whole diagnosis — it names the exact path it expected and didn't find. The
most common cause is the two tarballs landing in different folders (so
`models/weights/` never merges into `models/`), or running the script from
a location other than where you actually extracted things.

### Updating later (no image rebuild needed)

- **Code change only** (bug fix in `inference/`, `models/pipelines/ner_translate/`,
  `submit_pipeline_job.py`): re-run the code tarball step, re-transfer just
  that (~few MB), unpack over the existing `/opt/ner-translate/` — both
  images already bind-mount this directory, so a container restart
  (`docker compose restart`) picks it up.
- **New/updated dependency for this pipeline only**: rebuild just the
  wheelhouse and the kitchen image (`ner-translate-server.tar.gz`, ~3GB) —
  the waiter image and every other pipeline's kitchen are untouched.
- **New pipeline**: build one new kitchen image following
  `docs/MODEL_CONTAINER_ISOLATION.md`'s "Adding the next pipeline" steps
  (adapted for Option B — new `Dockerfile.<name>_server`, new `serve.py`,
  new `service_url` manifest entry). The waiter image doesn't change.

## Known gotchas

- **`Py4JNetworkError: TimeoutError` on `JavaSparkContext` construction** —
  if this ever resurfaces, check that `submit_pipeline_job.py`'s
  `_host_resolvable()` still restores `socket.setdefaulttimeout()` in a
  `finally` block. Leaving the global default timeout at 1s (its DNS-lookup
  value) silently poisons py4j's own gateway socket for the rest of the
  process — this was the hardest bug in this session's validation and looks
  completely unrelated to sockets when it fails (a bare `TimeoutError` deep
  inside `JavaSparkContext.__init__`).
- **Worker core count must be ≥ `executor_cores`** requested by
  `create_cluster_session` (default 4) — `start-worker.sh ... -c 4` in the
  compose file, not `-c 2`, or jobs hang at "Stage 0: 0/N tasks" with no
  error.
- **Never `COPY wheels/<pipeline>/ ...` then `rm -rf` in a later `RUN`** in
  any new Dockerfile — see the bind-mount fix above. The wheelhouse bytes
  ship anyway.
- **Don't let `docker save` output land inside the build context directory**
  before rebuilding — `*.tar.gz` is now `.dockerignore`d, but if you export
  to a different path pattern, make sure it's still excluded.
