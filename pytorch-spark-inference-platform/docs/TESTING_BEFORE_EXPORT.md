# Testing `ner_translate` Before Exporting for Air-Gapped Transfer

An ordered checklist — each tier only worth running once the one before it
passes. Mirrors `docs/air_gapped_dep.md`'s own Phase 1 "build → verify →
export" structure, applied to the new per-pipeline image from
`docs/CHANGES_SUMMARY.md` Parts 2-3. Stop and fix before moving to the next
tier — no point building/exporting an image whose pipeline logic hasn't
even been proven locally.

## Tier 0 — Blockers (checked 2026-09-14, still true)

| Blocker | Status | Fix |
|---|---|---|
| Model weights not populated | `models/weights/` has only `README.md` | Run the two `from_pretrained(...).save_pretrained(...)` commands in `models/weights/README.md` — needs internet, ~2-3GB total |
| Docker Desktop not running | Confirmed via `docker info` | Start Docker Desktop before Tier 2+ |
| OCR/PDF path never exercised | `data/ner_samples/` was text-only | **Fixed** — added `data/ner_samples/sample_scan6.png`, a synthetic image with person/org/location/phone entities, verified legible |

Nothing below Tier 1 is worth running until weights are populated.

## Tier 1 — Local, no Docker (fastest feedback)

Exercises: language detection, direct-path (English), translated-path
(Hindi `sample_text4.txt`, Marathi `sample_text5.txt` — already non-English
in the existing samples, so translation was already covered), phone-number
regex, entity dedup. Does **not** exercise OCR (Windows has no tesseract by
default) or the Docker/Spark-cluster machinery.

```bash
pip install -r requirements.txt -r models/pipelines/ner_translate/requirements.txt
python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 1
```

**Pass criteria:** all 5 `.txt` files produce entities (check
`results/ner_translate_<timestamp>.json`); Hindi/Marathi samples show
`"translated": true`; `sample_text3.txt`'s phone number
(`1234567890`) appears as a `phone number` entity. `sample_scan6.png` will
either be skipped (no OCR locally) or error — that's expected here, not a
failure; it's covered in Tier 3.

## Tier 2 — Docker build verification

```bash
# Base image
docker build -t multi-model-inference:latest -f deploy/Dockerfile .

# Wheelhouse (skip if already built — verify with: ls wheels/ner_translate | wc -l, want 50)
bash deploy/scripts/build_ner_translate_wheelhouse.sh

# Pipeline image
docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate .
```

**Pass criteria — run each of these, all should succeed:**
```bash
docker run --rm ner-translate-worker:latest python --version
# Python 3.11.x

docker run --rm ner-translate-worker:latest python -c "import gliner, transformers, sentencepiece, py3langid; print('deps OK')"
# deps OK — proves the --no-index --find-links wheelhouse install actually worked

docker run --rm ner-translate-worker:latest tesseract --list-langs
# should list: eng, mar, hin, tel, tam, kan, ben, guj, pan, mal, urd

docker run --rm ner-translate-worker:latest python -c "import torch; print(torch.__version__)"
# 2.6.0 — confirms the base image's torch wasn't touched by the pipeline image's install
```

If the wheelhouse install step in the Dockerfile fails here, it's cheaper
to find out now than after `docker compose up`.

## Tier 3 — Cluster + real job submission

```bash
docker compose -f deploy/docker-compose.ner_translate.yml up -d
```

**Pass criteria:**
1. `docker compose -f deploy/docker-compose.ner_translate.yml logs ner-translate-master | grep wheels-hotfix` shows
   `No wheels found in /wheels-hotfix — running with the image's baked-in dependencies.`
   (proves the hotfix mechanism is correctly a no-op by default, not silently active)
2. Spark UI reachable at `http://localhost:8081` (offset port — the shared
   cluster's UI, if also running, stays on `8080`), worker shows as
   `ALIVE` under it
3. Submit the job — this now includes the OCR image. **Use an absolute
   path** (`/app/data/ner_samples`, not `data/ner_samples`): on a real
   distributed cluster each executor task runs from its own per-application
   work directory, not `/app` — a relative path here "succeeds" (no
   exception) while producing nothing but a `[Errno 2] No such file or
   directory` error for every single file. Caught only by reading the job's
   actual per-file results, not by it merely completing without crashing.
   ```bash
   docker exec -it ner-translate-master bash -c \
     "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"
   ```
4. Check `_resolve_master_url()` actually picked the dedicated cluster
   automatically (no `--master` flag was passed above) — the job output's
   `partition_details` should show `hostname` values matching the
   container, and it should complete without trying to reach `local[4]`
5. `sample_scan6.png` in the results JSON shows non-empty `entities_unique`
   containing something like `Rakesh Verma` (person), `Indian Army` /
   `5th Mountain Brigade` (organization/military unit), `Leh` (location),
   `9876543210` (phone number) — this is the actual proof tesseract +
   poppler + pytesseract are wired correctly end-to-end, not just installed
6. Nothing in the master/worker logs mentions a Python or package version
   mismatch between them (the failure mode `docs/FRAMEWORK_OVERVIEW.md`
   already documented once)

```bash
docker compose -f deploy/docker-compose.ner_translate.yml down
```

## Tier 4 — only now, export

Once Tiers 1-3 all pass, follow `docs/air_gapped_dep.md` §3.5's pattern,
applied to this image:

```bash
docker save ner-translate-worker:latest -o ner-translate-worker.tar
gzip ner-translate-worker.tar
```

Transfer `ner-translate-worker.tar.gz` + `deploy/docker-compose.ner_translate.yml`
+ `wheels-hotfix/ner_translate/` (empty — the hotfix mechanism, present in
case it's needed later, not because it's needed now) to the air-gapped
target, same transfer method as the base image. `docs/air_gapped_dep.md`
itself still documents the base/shared image's own export — this doc only
covers the new `ner_translate`-specific image on top of it.
