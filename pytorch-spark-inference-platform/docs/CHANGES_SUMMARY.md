# Changes Summary — Spark/Docker/NER Pipeline Version Mismatch & Dependency Isolation

One consolidated reference for everything implemented across this work.
Full technical diffs for Parts 1-2 live in `CHANGELOG_20260913.md` and
`CHANGELOG_20260914.md`; Part 3 is documented only here. This doc states
current git status accurately and separates **implemented** from
**discussed but not yet built**.

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

**Git status: committed and pushed.** Branch
`fix/docker-ner-pipeline-version-mismatch`, commit `adb2bed`, pushed to
`origin`. PR not yet opened (`gh` CLI isn't installed in this environment —
GitHub's compare link is
`https://github.com/KRISHNA049039/multidimensional_spark_Poc/pull/new/fix/docker-ner-pipeline-version-mismatch`).

---

## Part 3 — Dependency wheelhouse + runtime hotfix mount (2026-09-14, continued)

**Question raised:** should dependencies install into the image at build
time, or be mountable via Docker volumes at runtime? Answer landed on:
build-time wheelhouse as the primary mechanism (also resolves Part 2's open
"wheel-vendoring" item), with runtime volume-mounted wheels added ONLY as
an explicit hotfix path — not the default. Runtime-as-primary was rejected
because this repo has already hit a real driver/worker Python-environment
mismatch bug once (`docs/FRAMEWORK_OVERVIEW.md`), and volume-mounted deps
that can silently drift between the master and worker containers (or
between nodes) reproduce exactly that failure mode; it also breaks the
airgapped deployment's self-containment (`docs/air_gapped_dep.md` — a
`docker load`d image should need nothing else to run).

| File | Change |
|---|---|
| `deploy/scripts/build_ner_translate_wheelhouse.sh` **(new)** | Downloads `ner_translate`'s full dependency closure (including transitive deps) into `wheels/ner_translate/` via `pip download`. Two-pass: strict cross-platform wheel-only download for most packages, plus a separate unrestricted pass for `odfpy`/`ebooklib` (pure-Python, sdist-only — pip refuses cross-platform sdist downloads with `--only-binary` unset, confirmed while testing). Run on an internet-connected machine before `docker build` |
| `deploy/Dockerfile.ner_translate` | Now installs via `pip install --no-index --find-links=/tmp/wheels -r ...` against the wheelhouse above, instead of hitting the live index — build fails loudly if the wheelhouse is missing/stale rather than silently resolving a different version. Also copies in the hotfix script below |
| `deploy/apply_wheels_hotfix.sh` **(new)** | Runs at container start (before Spark launches); installs anything found in `/wheels-hotfix`, no-op if empty. Its own header comment explains why this isn't the primary mechanism |
| `deploy/docker-compose.ner_translate.yml` | Both services mount `../wheels-hotfix/ner_translate:/wheels-hotfix` and call the hotfix script before `start-master.sh`/`start-worker.sh` |
| `wheels-hotfix/ner_translate/README.md` **(new)** | Explains when to use the hotfix path, how, and the "apply to every node, not just one" warning |
| `.gitignore` | Added `wheels/` (the generated wheelhouse — regenerable, never committed) and `**/wheels-hotfix/**/*.whl` (dropped hotfix files — the README stays tracked since it doesn't match `*.whl`). Both patterns verified with `git check-ignore` after an initial version of the negation silently failed (git can't re-include a file under an excluded parent directory — switched to ignoring by file pattern instead of directory+negation) |

**Verified, not assumed:** actually ran the wheelhouse script end-to-end (50
files produced, including the two sdists), actually ran `apply_wheels_hotfix.sh`
for both the empty and wheel-present cases, validated the compose file's
YAML, and confirmed with `git check-ignore -v` that the wheelhouse is fully
ignored while `wheels-hotfix/*/README.md` stays tracked and `.whl` files
under it don't.

**Git status: NOT yet committed.**

---

## Discussed, NOT implemented — pending your decision

1. **Model 2 vs. Model 3 architecture pivot.** You raised a genuinely
   stronger alternative to Part 2's "separate image + separate Spark
   cluster per pipeline": one shared image with per-pipeline Python venvs
   baked in (`/opt/venvs/ner_translate`, etc.), selected per-job via
   `spark.executorEnv.PYSPARK_PYTHON`. Better fit for the air-gapped
   transfer cost you flagged (one image instead of N tarballs), same
   dependency isolation strength, but keeps one Spark cluster instead of
   N. I asked which way to go; you interrupted that question to ask about
   wheel files instead — the architecture decision is still open, and Part
   3 doesn't resolve it (it's orthogonal: Part 3 is about *how* a pip
   install happens, not *how many* images/venvs exist).

---

## How to verify everything above without an NVIDIA GPU

(Full detail in `CHANGELOG_20260913.md`'s testing section.)

```bash
# Fastest: no Docker at all — pipeline auto-falls-back to CPU
pip install -r requirements.txt -r models/pipelines/ner_translate/requirements.txt
python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 1

# Multi-container cluster, CPU-only, exercises the actual isolation +
# wheelhouse build (do this once, or whenever ner_translate's requirements.txt changes)
bash deploy/scripts/build_ner_translate_wheelhouse.sh
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
docker compose -f deploy/docker-compose.ner_translate.yml build
docker compose -f deploy/docker-compose.ner_translate.yml up
docker exec -it ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"
```
(Absolute path on the cluster invocation — a real distributed executor
runs from its own work directory, not `/app`; a relative path here
"succeeds" while erroring on every file. Local `local[4]` above is
unaffected — driver and "executor" share one process/CWD there.)

Requires `models/weights/gliner-multi/` and
`models/weights/nllb-200-distilled-600M/` populated first — see
`models/weights/README.md`.

---

## Next steps

- Decide Model 2 vs. Model 3 (or a hybrid) for the architecture question above
- Commit Part 3's changes (currently uncommitted, same branch)
- Open the PR (link above) — nothing has merged to `main` yet at any point in this work
