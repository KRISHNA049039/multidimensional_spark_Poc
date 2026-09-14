# Framework Overview — Start Here

This repo has accumulated a lot of docs. This one is the single entry point:
what this actually is, how data flows through it, what's required for it to
work, and how to run it three different ways. Everything else in `docs/` is a
deep-dive on one specific piece — pointers to the right one are in §6.

## 1. What this is, in one paragraph

A framework for running an arbitrary PyTorch model (or a whole multi-model
pipeline) as a **distributed Spark job** across CPU and/or GPU workers. You
drop your model in, register it in a manifest, and submit it with one CLI
command — the framework handles partitioning the input, loading your model
once per executor, running it in parallel, and getting results back.

## 2. Architecture — the two plugin families

There are **two parallel tracks**, because "a model" means two different
things depending on what you're running. Forcing both through one contract
would make both worse, so they don't share one.

```
                    ┌─────────────────────────┐    ┌─────────────────────────┐
                    │   models/plugins/       │    │   models/pipelines/     │
                    │   (simple tensor model) │    │   (complex pipeline)    │
                    ├─────────────────────────┤    ├─────────────────────────┤
Input               │ fixed-shape float32     │    │ file paths / raw text,  │
                    │ numpy array             │    │ any format              │
Load contract       │ ModelClass() +          │    │ load() -> anything      │
                    │ load_state_dict(.pt)    │    │ (HF from_pretrained,    │
                    │                         │    │  tokenizers, multiple   │
                    │                         │    │  models, ...)           │
Run contract        │ model(batch_tensor)     │    │ run(loaded, paths)      │
                    │                         │    │ -> real structured      │
                    │                         │    │    results              │
Execution engine    │ inference/              │    │ inference/              │
                    │ cluster_engine.py       │    │ text_pipeline_engine.py │
CLI                 │ submit_job.py           │    │ submit_pipeline_job.py  │
Example             │ ew_classifier,          │    │ ner_translate           │
                    │ example_mlp             │    │ (NLLB + GLiNER)         │
                    └─────────────────────────┘    └─────────────────────────┘
```

Pick `models/plugins/` if your model takes one fixed-shape tensor in and
gives one tensor out (classifiers, detectors, regressors). Pick
`models/pipelines/` if it needs its own loading logic, multiple models, or
text/file input (translation, NER, OCR, anything HuggingFace-`from_pretrained`
shaped).

## 3. Input → output flow

Both tracks share the same shape, just with different payloads:

```
1. CLI parses args (--model/--pipeline, --input, --partitions, --master)
        │
2. Registry lookup (models/plugins/manifest.json or models/pipelines/manifest.json)
   -> dynamically imports your file, no code elsewhere needs to change
        │
3. create_cluster_session() opens a SparkSession against:
     - local[4]                      (dev machine, no cluster)
     - spark://<master-ip>:7077      (real Standalone cluster - Docker or EC2)
        │
4. Input is split into N partitions:
     - plugins:   numpy array sliced into N chunks
     - pipelines: list of file paths divided into N chunks
        │
5. Spark ships each partition to an executor. mapPartitions() runs once per
   executor (not once per item) — this is where your model actually loads:
     - plugins:   class_map lookup -> ModelClass() -> load_state_dict(broadcast weights)
     - pipelines: your load() function runs, however it wants
        │
6. Every item in the partition runs through the now-loaded model:
     - plugins:   model(batch_tensor)  -- output is currently DISCARDED,
                  only a sample count comes back (see docs/CONCURRENT_JOBS_AND_COMPLETING_THE_FRAMEWORK.md)
     - pipelines: run(loaded, paths)   -- REAL results come back, merged
                  across all partitions on the driver
        │
7. Driver collects partition results, prints a summary, writes
   results/<name>_<timestamp>.json (and uploads to S3 if ARTIFACTS_BUCKET is set)
```

## 4. What's actually required for this to work as a framework

Minimum load-bearing pieces — if any of these is missing or broken, "submit
a job" stops working:

| Piece | File(s) | Why it's required |
|---|---|---|
| Spark + Java | `requirements.txt` (`pyspark==3.5.1`), a JVM (Java 17) on every node | PySpark cannot start without a JVM — this is the #1 thing that silently breaks on a fresh machine |
| PyTorch | `requirements.txt` (`torchvision`; `torch` comes in transitively) + a CUDA build for GPU nodes | every model, tensor or pipeline, ultimately runs on torch |
| Tensor plugin registry | `models/model_registry.py`, `models/plugin_loader.py`, `models/plugins/manifest.json` | resolves a model name to a class + weights at both submit time and inside each executor |
| Pipeline registry | `models/pipelines/manifest.json` | same, for pipeline-style plugins |
| Weight storage | `models/weights/` (gitignored — never commit weights) | plugins/pipelines load from here; empty repo checkout = models fail to load, not a bug |
| Tensor execution engine | `inference/cluster_engine.py` | Spark session creation + the `mapPartitions` loop for tensor models |
| Pipeline execution engine | `inference/text_pipeline_engine.py` | same, for pipeline models |
| CLIs | `submit_job.py`, `submit_pipeline_job.py` | the actual thing a user runs |
| A cluster to submit to | one of: nothing (`local[4]`), `deploy/docker-compose.cluster.yml`, or an AWS CDK stack under `deploy/aws-cdk/` | Spark needs *some* master, even if it's a fake single-machine one |

Everything else in the repo (benchmarks, dashboards, monitoring publishers,
the older `pytorch-spark-ew-poc` sibling project) is not required for the
framework itself to function — it's tooling built on top of it.

## 5. How to run it

### 5a. Native, no Docker (what's been used for all local testing)

Fastest way to try a model, and the only way that works on a machine without
Docker at all (e.g. Windows Server EC2 — see
`docs/DEPLOY_WINDOWS_CLUSTER_AWS.md`).

**Dependencies to install:**
- Java 17 (Temurin or any JDK) — set `JAVA_HOME`
- Python 3.12 with `pip install -r requirements.txt`
- For pipeline plugins specifically: that pipeline's own requirements file,
  named in its manifest entry's `requirements_file` (e.g. `ner_translate` ->
  `pip install -r models/pipelines/ner_translate/requirements.txt`) — kept
  separate from the base `requirements.txt` on purpose, see
  `docs/MODEL_CONTAINER_ISOLATION.md`
- If more than one Python is on `PATH`, pin the one PySpark should use:
  `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` env vars — a real bug hit during
  development when a stray second Python caused a driver/worker version
  mismatch

**Run:**
```powershell
# tensor model
python submit_job.py --model example_mlp --samples 2000 --mode cpu_only

# pipeline
python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 1
```
No `--master` needed — defaults to `local[4]` (4 threads on this machine
simulating a cluster). Results land in `results/`.

### 5b. Docker Compose (Linux cluster, single machine or LAN)

For an actual multi-process master + worker(s) topology without cloud.

```powershell
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
docker compose -f deploy/docker-compose.cluster.yml up
docker exec -it spark-master bash -c "python submit_job.py --model example_mlp --master spark://spark-master:7077"
```
Scale workers: `docker compose -f deploy/docker-compose.cluster.yml up --scale spark-cpu-worker=3 --scale spark-gpu-worker=2`.

Full detail (multi-machine LAN topology, firewall ports, automation scripts):
`docs/WINDOWS_CLUSTER_SETUP.md`.

### 5c. AWS (real GPU hardware, no local Docker needed)

Two options exist, both under `deploy/aws-cdk/`:

- **`WindowsSparkClusterStack`** — native (no Docker) Windows Server master +
  GPU worker; this is what most of this project's AWS testing has used.
  Full walkthrough: `docs/DEPLOY_WINDOWS_CLUSTER_AWS.md`.
- **`SparkClusterStack`** — the original Linux/Docker EC2 cluster. Full
  walkthrough: `docs/AWS_CDK_DEPLOYMENT.md`.

Both create real, billed AWS resources (`cdk deploy`) — confirm cost/region
before running either.

## 6. Where to go for more detail

This doc is the map, not the territory. Once you know which piece you're
touching, go here:

| Topic | Doc |
|---|---|
| Writing a new tensor plugin | `docs/BRING_YOUR_OWN_MODEL.md` |
| Repo layout convention (where new files go) | `docs/REPO_STRUCTURE.md` |
| Concurrent job submission, and what's missing for a "complete" framework | `docs/CONCURRENT_JOBS_AND_COMPLETING_THE_FRAMEWORK.md` |
| Deploying/testing the Windows AWS cluster | `docs/DEPLOY_WINDOWS_CLUSTER_AWS.md` |
| Deploying the Linux AWS cluster + CloudWatch metrics | `docs/AWS_CDK_DEPLOYMENT.md` |
| Multi-machine Windows LAN cluster (no cloud) | `docs/WINDOWS_CLUSTER_SETUP.md` |
| Airgapped/offline deployment (Linux/Docker only — see caveat below) | `docs/AIRGAPPED_5NODE_DEPLOYMENT.md`, `docs/AIRGAPPED_TROUBLESHOOTING.md` |

**Airgapped caveat:** the airgapped docs cover the Linux/Docker path only
(build the image with internet, transfer the tarball, load offline). The
Windows AWS stack pulls Java/Python/Spark/drivers live from the internet on
every boot and is not airgapped-compatible as built — see
`docs/DEPLOY_WINDOWS_CLUSTER_AWS.md` §12 for what would need to change.
