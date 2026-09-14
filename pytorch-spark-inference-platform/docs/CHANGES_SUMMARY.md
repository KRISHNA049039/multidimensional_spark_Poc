# Changes Summary — Spark/Docker/NER Pipeline Version Mismatch & Dependency Isolation

One consolidated reference for everything implemented across this work.
Full technical diffs live in `CHANGELOG_20260913.md` and
`CHANGELOG_20260914.md` — this doc pulls both together, states current git
status accurately, and separates **implemented** from **discussed but not
yet built**.

---

## Part 1 — Version mismatch fixes (2026-09-13)

**Problem:** `deploy/Dockerfile` pinned `torch==2.6.0`, but `requirements.txt`
separately pinned `torchvision==0.17.0` — which hard-requires `torch==2.2.0`,
so every `pip install -r requirements.txt` silently downgraded torch and
undid the CUDA 12.6/sm_120 (RTX 5060) fix. Separately, the `ner_translate`
pipeline's dependencies (`gliner`, `transformers`, etc.) were declared in
`models/pipelines/manifest.json` but nothing ever actually installed them,
and `Dockerfile.worker` was pinned to torch versions that don't exist on
either wheel index.

| File | Change |
|---|---|
| `requirements.txt` | Removed the conflicting `torchvision==0.17.0` pin; bumped `numpy`/`pandas`/`pyarrow`/`ultralytics` for numpy-2 compatibility; added `ner_translate`'s deps (later moved out again — see Part 2) |
| `deploy/Dockerfile` | Added `tesseract-ocr` + 10 language packs + `poppler-utils` (later moved to `Dockerfile.ner_translate` — see Part 2) |
| `Dockerfile.worker` | Repinned both build stages (CPU + GPU) from non-existent torch versions to the real, stable `torch==2.9.1`/`torchvision==0.24.1` pair (cp314 wheels, sm_120/Blackwell support) |
| `docs/CHANGELOG_20260913.md` **(new)** | Full root-cause writeup, before/after tables, the `cudnn-runtime` vs `cudnn-devel` base-image decision (kept `runtime` — nothing here compiles CUDA code) |

Every version pin was checked against PyPI/Apache/Ubuntu-archive before
committing, not guessed — see that changelog for the verification commands.

**Git status: committed.** Branch `fix/docker-ner-pipeline-version-mismatch`,
commit `538741f`.

---

## Part 2 — Dependency isolation: `ner_translate` gets its own image (2026-09-14)

**Problem:** even after Part 1's fix, the repo still had one shared image +
one `requirements.txt` for every model. `docs/MODEL_CONTAINER_ISOLATION.md`
already documented a real incident (installing `gliner`/`transformers` once
silently bumped `numpy` for every model sharing the environment) — this
makes that doc's "Option A" (one Spark worker image per model) real for
`ner_translate`, instead of it staying a proposal.

| File | Change |
|---|---|
| `requirements.txt` | Stripped back to CORE deps only (`pyspark`, `numpy`, `pandas`, `pyarrow`, `ultralytics`, `matplotlib`, `tabulate`, `boto3`) |
| `deploy/Dockerfile` | Now purely the shared **base** image (Python 3.11 + Java 17 + Spark 3.5.1 + torch/torchvision + core requirements). OCR system packages removed — moved to `Dockerfile.ner_translate` |
| `models/pipelines/ner_translate/requirements.txt` **(new)** | `ner_translate`'s own deps — single source of truth, installed by both its Dockerfile and local (non-Docker) dev |
| `deploy/Dockerfile.ner_translate` **(new)** | `FROM multi-model-inference:latest` (the base above) + tesseract-ocr/poppler + this pipeline's own `pip install` |
| `deploy/docker-compose.ner_translate.yml` **(new)** | `ner_translate`'s own master+worker Spark cluster, independent of `docker-compose.cluster.yml`. Ports offset (+1) so both can run side by side |
| `models/pipelines/manifest.json` | `"extra_requirements"` (declarative only, never consumed — that's *why* bug #2 in Part 1 happened) replaced with `"requirements_file"` (actually installed) and `"master_url": "spark://ner-translate-master:7077"` |
| `submit_pipeline_job.py` | Added `_resolve_master_url()`: `--master` flag > `SPARK_MASTER_URL`/`SPARK_MASTER` env var > manifest `master_url` **only if that host resolves** (`socket.getaddrinfo`) > `local[4]`. The resolvability check is what keeps the exact same command working in both plain local dev and inside the pipeline's own container — verified with 4 test cases locally before shipping (an earlier version of this logic would have broken local `local[4]` testing; caught and fixed before it landed) |
| `docs/MODEL_CONTAINER_ISOLATION.md` | Marked Option A implemented; added a 5-step "adding the next pipeline" checklist |
| `docs/REPO_STRUCTURE.md`, `docs/FRAMEWORK_OVERVIEW.md` | Updated references from the old `extra_requirements`/merged `requirements.txt` to the new per-pipeline file + manifest schema |
| `docs/CHANGELOG_20260914.md` **(new)** | Full writeup of the above, including the master-URL bug caught pre-ship |

**Behavior change:** the shared cluster (`docker-compose.cluster.yml`) can
no longer run `ner_translate` — that's the point of isolation, but it's a
real change from Part 1's state. Use `docker-compose.ner_translate.yml`.

**Verified, not assumed:** dry-run resolved the split requirements files
together — identical 73-package resolution to the merged version, torch
still pinned at `2.6.0`.

**Git status: NOT yet committed.** Sitting as working-tree changes on the
same `fix/docker-ner-pipeline-version-mismatch` branch:
```
modified:   deploy/Dockerfile
modified:   docs/FRAMEWORK_OVERVIEW.md
modified:   docs/MODEL_CONTAINER_ISOLATION.md
modified:   docs/REPO_STRUCTURE.md
modified:   models/pipelines/manifest.json
modified:   requirements.txt
modified:   submit_pipeline_job.py
new file:   deploy/Dockerfile.ner_translate
new file:   deploy/docker-compose.ner_translate.yml
new file:   docs/CHANGELOG_20260914.md
new file:   models/pipelines/ner_translate/requirements.txt
```

---

## Discussed, NOT implemented — pending your decision

Two things came up after Part 2 that are **design discussion only** — no
files changed for either yet:

1. **Model 2 vs. Model 3 architecture pivot.** You raised a genuinely
   stronger alternative to Part 2's "separate image + separate Spark
   cluster per pipeline": one shared image with per-pipeline Python venvs
   baked in (`/opt/venvs/ner_translate`, etc.), selected per-job via
   `spark.executorEnv.PYSPARK_PYTHON`. Better fit for the air-gapped
   transfer cost you flagged (one image instead of N tarballs), same
   dependency isolation strength, but keeps one Spark cluster instead of
   N. I asked which way to go; you interrupted that question to ask about
   wheel files instead — the architecture decision is still open.
2. **Wheel-vendoring for build reproducibility.** Recommended switching
   `pip install -r requirements.txt` (hits the live index at build time) to
   `pip install --no-index --find-links=<wheelhouse>` against a
   `pip download`-generated, gitignored wheelhouse directory — motivated by
   the fact `Dockerfile.worker`'s nightly torch pin had already gone stale
   once, and the DRDO/classified context named in `docs/air_gapped_dep.md`
   generally wants pinned, auditable binaries rather than a live index
   resolve. Not implemented — waiting on a go-ahead.

---

## How to verify everything above without an NVIDIA GPU

(Full detail in `CHANGELOG_20260913.md`'s testing section.)

```bash
# Fastest: no Docker at all — pipeline auto-falls-back to CPU
pip install -r requirements.txt -r models/pipelines/ner_translate/requirements.txt
python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 1

# Multi-container cluster, CPU-only, exercises the actual isolation
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
docker compose -f deploy/docker-compose.ner_translate.yml build
docker compose -f deploy/docker-compose.ner_translate.yml up
docker exec -it ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 2"
```

Requires `models/weights/gliner-multi/` and
`models/weights/nllb-200-distilled-600M/` populated first — see
`models/weights/README.md`.

---

## Next steps

- Decide Model 2 vs. Model 3 (or a hybrid) for the architecture question above
- Decide whether to build the wheel-vendoring pattern, and for which images
- Commit Part 2's changes (currently uncommitted on `fix/docker-ner-pipeline-version-mismatch`)
- Push the branch + open a PR against `main`, or fast-forward directly — your call, not yet done either way
