# Changelog — September 14, 2026

Implementing `docs/MODEL_CONTAINER_ISOLATION.md` Option A for real: giving
`ner_translate` its own isolated dependency set and its own Spark cluster,
instead of sharing one image/`requirements.txt` with every other model.

---

## Why

Following up on `CHANGELOG_20260913.md`'s version-mismatch fixes, the
question came up: is one shared image + one `requirements.txt` the right
long-term way to add more AI pipelines with their own, possibly conflicting,
dependencies? No — `docs/MODEL_CONTAINER_ISOLATION.md` already documents a
real incident where installing `gliner`/`transformers` silently bumped
`numpy` for every model sharing the environment. This change makes that
doc's "Option A" (one Spark worker image per model/family) real for
`ner_translate`, so the next pipeline added follows an established pattern
instead of getting hand-merged into the shared `requirements.txt` again.

---

## Modified / New Files

| File | Change |
|---|---|
| `requirements.txt` | Stripped back to CORE deps only (`pyspark`, `numpy`, `pandas`, `pyarrow`, `ultralytics`, `matplotlib`, `tabulate`, `boto3`). The `ner_translate`-specific packages added in `CHANGELOG_20260913.md` were removed from here and moved to the file below. |
| `models/pipelines/ner_translate/requirements.txt` **(new)** | `ner_translate`'s own deps (`gliner`, `transformers<5`, `sentencepiece`, `py3langid`, plus the full document-extraction stack). Single source of truth — installed by both its Dockerfile and a local (non-Docker) dev before running against `local[4]`. |
| `deploy/Dockerfile` | Now purely the shared **base** image (Python 3.11 + Java 17 + Spark 3.5.1 + torch/torchvision + core `requirements.txt`). Tesseract/poppler system packages removed — they moved to `Dockerfile.ner_translate` since only that pipeline needs OCR. |
| `deploy/Dockerfile.ner_translate` **(new)** | `FROM multi-model-inference:latest` (the base image above) + `tesseract-ocr` + language packs + `poppler-utils` + `pip install -r models/pipelines/ner_translate/requirements.txt`. Nothing else. |
| `deploy/docker-compose.ner_translate.yml` **(new)** | `ner_translate`'s own master+worker pair, built from `Dockerfile.ner_translate`. Ports offset (+1: `7078`/`8081`/`4041`) from the shared cluster so both can run side by side. Independent from `docker-compose.cluster.yml` — starting/stopping one never touches the other. |
| `models/pipelines/manifest.json` | `"extra_requirements"` (declarative only — nothing had ever consumed it, which is how bug #2 in `CHANGELOG_20260913.md` happened) replaced with `"requirements_file"` (actually installed, see above) and `"master_url": "spark://ner-translate-master:7077"`. |
| `submit_pipeline_job.py` | Added `_resolve_master_url()`: `--master` flag > `SPARK_MASTER_URL`/`SPARK_MASTER` env var > manifest `master_url` **if that host resolves** (`socket.getaddrinfo`) > `local[4]`. The resolvability check is the important part — it's what keeps `python submit_pipeline_job.py --pipeline ner_translate ...` working unmodified in plain local dev (unresolvable host -> `local[4]`) *and* automatically targeting the dedicated cluster when run inside its container (compose service name resolves). Verified locally with all four precedence cases before shipping. |
| `docs/MODEL_CONTAINER_ISOLATION.md` | Marked Option A implemented; changed the "changes required" table to reflect what actually shipped; added a "adding the next pipeline" walkthrough. |
| `docs/REPO_STRUCTURE.md`, `docs/FRAMEWORK_OVERVIEW.md` | Updated references to the old `extra_requirements` field / merged `requirements.txt` to point at the new per-pipeline `requirements.txt` + manifest schema. |

---

## A bug caught and fixed before it shipped

The first version of `_resolve_master_url()` unconditionally used the
manifest's `master_url` as the default whenever `--master` wasn't passed.
That would have **broken plain local dev** — `ner-translate-master` isn't a
resolvable hostname outside that pipeline's own Docker network, so
`python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples`
run on bare `local[4]` (the Tier-1 test path recommended in the previous
session) would fail to connect instead of running locally. Fixed by adding
the `socket.getaddrinfo()` resolvability check, so the manifest's
`master_url` is only used when it's actually reachable — verified with four
test cases (unresolvable host, resolvable host, explicit `--master`
override, and env-var precedence) before committing.

---

## Behavior change worth knowing

**The shared cluster (`docker-compose.cluster.yml`) can no longer run
`ner_translate`** — its image only has the base image now, not `gliner`/
`transformers`/OCR. This is intentional (that's the whole point of
isolation), but it's a real change from `CHANGELOG_20260913.md`'s state,
where the shared image *could* run `ner_translate` after the version fixes.
Use `deploy/docker-compose.ner_translate.yml` for that pipeline going
forward.

---

## How to run `ner_translate` now

```bash
# Build (base image first, then the pipeline's own image on top of it)
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
docker compose -f deploy/docker-compose.ner_translate.yml build

# Run its own cluster
docker compose -f deploy/docker-compose.ner_translate.yml up

# Submit a job — driver runs inside the master container (it imports
# gliner/transformers too, so it needs this image, not the host or the
# shared cluster's containers). No --master needed; manifest master_url
# resolves automatically inside this network.
docker exec -it ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 2"
```

Local, no-Docker dev is unchanged:
```bash
pip install -r requirements.txt -r models/pipelines/ner_translate/requirements.txt
python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 1
```
