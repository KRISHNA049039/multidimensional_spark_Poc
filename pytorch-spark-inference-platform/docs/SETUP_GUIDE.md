# Setup Guide

Practical steps to get this repo running, from a bare checkout to a working
pipeline job. Read `docs/PROJECT_OVERVIEW_AND_CONCEPTS.md` first if you
haven't — this doc assumes you know *why* things are structured this way,
not just the commands.

## Prerequisites

| Need | Why |
|---|---|
| Docker Desktop (or Docker Engine on Linux) | Every path below except pure local dev uses containers |
| Git Bash, if on Windows | The `.sh` scripts under `deploy/scripts/` need a POSIX shell — plain PowerShell/cmd can't run them |
| Python 3.11 | Matches the container images' interpreter version; only needed locally if you're running Tier 0 (below) without Docker |
| An NVIDIA GPU + driver, only if you want real (not just built) GPU inference | Building images never needs a GPU present — only running `--gpus all` does |

Verify Docker is actually usable before anything else:
```bash
docker info
```
If this fails on Windows, Docker Desktop's engine isn't running yet — start
the app and wait ~30-60s.

## Path 1 — Fastest local iteration (no Docker, no cluster)

For working on pipeline/plugin logic itself, not packaging:
```bash
pip install -r requirements.txt
pip install -r models/pipelines/ner_translate/requirements.txt   # only for this pipeline
python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 2
```
No `--master` needed — `submit_pipeline_job.py` falls back to `local[4]`
when nothing cluster-specific resolves. This is Tier 0 in the testing
discipline (`docs/TESTING_BEFORE_EXPORT.md`) — the cheapest place to catch
a logic bug, always try here first.

## Path 2 — One pipeline, its own small Docker cluster (Option A)

Builds a per-pipeline image `FROM` a shared base — see
`docs/MODEL_CONTAINER_ISOLATION.md`'s Option A.

```bash
# 1. Base image (Spark + Java + torch/CUDA + core deps) — build once,
#    reused by every pipeline
docker build -t multi-model-inference:latest -f deploy/Dockerfile .

# 2. This pipeline's own dependency wheelhouse (pinned, reproducible)
bash deploy/scripts/build_ner_translate_wheelhouse.sh

# 3. This pipeline's own image, FROM the base
docker build -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate .

# 4. Bring up its own small cluster and submit a job
docker compose -f deploy/docker-compose.ner_translate.yml up -d
docker exec -it ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"
```

## Path 3 — Waiter/kitchen split (Option B)

Spark runs on a model-agnostic image; the model lives behind its own HTTP
server. See `docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md` for the full
picture — short version:

```bash
export DOCKER_BUILDKIT=1   # needed for the --mount=type=bind steps below

# 1. The "waiter" — Spark master/worker, zero model dependencies, reused
#    across every future model
docker build --target lean -t spark-lean:latest -f deploy/Dockerfile .

# 2. This model's "kitchen" — its own wheelhouse, then its own HTTP-server image
bash deploy/scripts/build_ner_translate_wheelhouse.sh
docker build -t ner-translate-server:latest -f deploy/Dockerfile.ner_translate_server .

# 3. Bring both up together and submit a job — prefer the validating setup
#    script over calling docker compose directly (see its own header for why)
bash deploy/scripts/setup_ner_translate_server.sh
docker exec ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"
```

## Model weights

Neither path above bakes model weights into an image — they're mounted
from the filesystem at container start, same principle as
`docs/MODEL_CONTAINER_ISOLATION.md`'s "images are deps only." Populate them
once:
```bash
python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi-v2.1').save_pretrained('./models/weights/gliner-multi')"
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
m = 'facebook/nllb-200-distilled-600M'
AutoTokenizer.from_pretrained(m).save_pretrained('./models/weights/nllb-200-distilled-600M')
AutoModelForSeq2SeqLM.from_pretrained(m).save_pretrained('./models/weights/nllb-200-distilled-600M')
"
```
See `models/weights/README.md` for the full list across every model, not
just `ner_translate`.

## Airgapped / no internet on the target machine

Two different starting points, two different docs — pick based on what's
already on the target machine:

- **Target has nothing yet**: full image export/transfer/load, Phase-by-phase,
  in `docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md`.
- **Target already has a compatible base image** (e.g.
  `multi-model-inference:latest`) loaded from an earlier transfer: skip
  re-shipping images entirely — `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md`
  walks through building the final image *on* the target machine from a
  small wheelhouse + `.deb` bundle + code transfer instead.

## Multi-node / AWS

For a real multi-machine Spark cluster (on-prem or cloud):
`docs/CLUSTER_SETUP_GUIDE.md` (general), `docs/AWS_CDK_DEPLOYMENT.md` and
`docs/DEPLOY_WINDOWS_CLUSTER_AWS.md` (AWS-specific, CDK-driven). The GPU
benchmarking path used throughout this session's testing is
`docs/GPU_BENCHMARK_MANUAL.md`.

## Adding a new model

Not a setup step for THIS repo's infra — see `docs/BRING_YOUR_OWN_MODEL.md`
for the plugin/pipeline authoring contract, then
`docs/MODEL_CONTAINER_ISOLATION.md`'s "Adding the next pipeline" section
for wiring a new model into whichever isolation option you're using.

## Troubleshooting

- Airgapped-specific issues: `docs/AIRGAPPED_TROUBLESHOOTING.md`
- `ModuleNotFoundError: No module named 'models'` inside a container: your
  bind mount is pointing at the wrong/empty directory — see
  `docs/WAITER_KITCHEN_NER_TRANSLATE_DEPLOYMENT.md`'s "Known gotchas."
- `py4j.protocol.Py4JNetworkError` / `TimeoutError` constructing a
  `SparkContext` in standalone-cluster mode: same doc, same section — a
  global `socket.setdefaulttimeout()` left unset is the usual cause.
- `dpkg: dependency problems` installing a `.deb` bundle:
  `docs/NER_TRANSLATE_OFFLINE_DEPENDENCY_UPDATES.md`'s gotcha section.
