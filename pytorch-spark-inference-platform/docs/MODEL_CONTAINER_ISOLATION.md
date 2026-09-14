# Running Each Model in Its Own Docker Container

## Why this comes up

Right now every plugin and pipeline shares **one Python environment** —
whatever's installed in `multi-model-inference:latest` (Docker) or on the
Windows cluster nodes (native install). This already bit us once during
development: installing `gliner`/`transformers` for the `ner_translate`
pipeline silently upgraded `numpy` from `1.26.4` to `2.5.3` for *every*
model, tensor plugins included, because pip resolves one shared dependency
set for the whole environment. It happened to not break anything this time —
it easily could next time, especially between two pipelines with genuinely
incompatible pinned versions (e.g. one needs `transformers<4.40`, another
needs `>=5.0`).

Per-model containers fix this by giving each model its own isolated
dependency set. Here's what that actually requires.

## The constraint that shapes everything

Spark's `mapPartitions` runs your Python code **inside whatever container
the executor's JVM + Python worker were launched from**. There's no
per-task "run this one on a different container" knob in vanilla Spark —
an executor is a fixed environment for the lifetime of that worker process.
So "one container per model" can't mean "Spark transparently picks a
container per task." It means one of two real architectures:

## Option A — One Spark worker image per model (stays Spark-native)

> **Status: implemented for `ner_translate`, as of 2026-09-14.** See
> `docs/CHANGELOG_20260914.md` for the concrete diff. The pattern below is
> now real, not a proposal — follow it for the next pipeline you add.

Instead of one `multi-model-inference:latest` image running everything,
build a separate worker image per model (or per model family), each with
only that model's dependencies, and run each as its own Spark
master+worker pair (or at minimum its own worker pool joined to a shared
master via a custom resource label).

**Changes required (✅ = done for `ner_translate`):**

| File | Change |
|---|---|
| ✅ `deploy/Dockerfile` | Stays the shared **base** image — Python/Java/Spark/torch + `requirements.txt` trimmed to CORE deps only (no model-specific packages) |
| ✅ `deploy/Dockerfile.<model>` (one per model/family) | `FROM` the base image above, adds only that model's own system packages + `models/pipelines/<model>/requirements.txt` (e.g. `Dockerfile.ner_translate` adds `tesseract-ocr`/`poppler-utils` + `gliner transformers sentencepiece py3langid` + doc-extraction libs) |
| ✅ `models/pipelines/<model>/requirements.txt` (new, one per pipeline) | That pipeline's own deps, single source of truth — installed by both its Dockerfile and a local (non-Docker) dev before running against `local[4]` |
| ✅ `deploy/docker-compose.<model>.yml` (new, one per model/family) | That model's own master+worker pair, built from its own Dockerfile — kept as a separate compose file (not folded into `docker-compose.cluster.yml`) so it has an independent up/down lifecycle, per the trade-off noted below |
| ✅ `models/pipelines/manifest.json` | `extra_requirements` (was purely declarative — nothing ever consumed it) replaced with `"requirements_file"` (path, actually installed) and `"master_url"` (e.g. `"spark://ner-translate-master:7077"`) |
| ✅ `submit_pipeline_job.py` | Resolves `--master` as: explicit flag > `SPARK_MASTER_URL`/`SPARK_MASTER` env var > manifest `master_url` **if that host actually resolves** (`socket.getaddrinfo`) > `local[4]`. The resolvability check matters: it's what lets the exact same command work unmodified both in plain local dev (host doesn't exist -> falls through to `local[4]`) and inside the pipeline's own container (compose service name resolves -> uses its dedicated cluster) |
| Not yet done | `models/plugins/manifest.json` / `submit_job.py` — tensor plugins don't have per-model dependency conflicts yet (`example_mlp` is trivial), so this hasn't been needed. Follow the same pattern if that changes. |

**Trade-off:** simplest mental model, no new runtime dependency (no HTTP
server to write), stays entirely within Spark's own execution model. The
downside is granularity — you get isolation per *image build*, not per
running task, so if you truly want independent up/down lifecycle per model
(scale one model's workers without touching another's), you're running N
separate small Spark clusters, which is more infrastructure to manage than
today's single cluster.

## Option B — Model runs behind an HTTP endpoint, Spark executors become thin clients

Instead of loading the model directly inside `mapPartitions`, the executor
sends each batch over the network to a small, separate model-serving
container, and gets results back. This is the standard "model server /
sidecar" pattern (what TorchServe, Triton, and a hand-rolled FastAPI service
all do).

**Changes required:**

| File | Change |
|---|---|
| `models/plugins/<name>/serve.py` or `models/pipelines/<name>/serve.py` (new, one per model) | A small FastAPI/Flask app wrapping the existing `load()`/`run()` (pipelines) or `ModelClass()`/`forward()` (plugins) behind one `/predict` endpoint. Model loads once at container startup, not per Spark partition. |
| `deploy/Dockerfile.<model>-server` (new, one per model) | `FROM python:3.12-slim` (or `-cuda` for GPU models) + that model's deps + `serve.py` + its weights — no Spark/Java needed at all in this image |
| `deploy/docker-compose.cluster.yml` | Add one lightweight service per model server, on the same Docker network as the Spark workers, reachable by hostname |
| `inference/cluster_engine.py` (`process_partition`) | Replace the direct `model(batch_tensor)` call with an HTTP request to the model's service URL, sending the batch and parsing the response |
| `inference/text_pipeline_engine.py` (`run_fn` call) | Same swap: call the pipeline's HTTP endpoint instead of importing and calling `run()` in-process |
| `models/plugins/manifest.json`, `models/pipelines/manifest.json` | Add a `"service_url"` field per entry (e.g. `"http://ner-translate-server:8080"`) |

**The real win here:** the Spark worker image itself no longer needs torch,
transformers, gliner, or any model-specific dependency at all — it just
needs `requests` (or `httpx`) to make the call. All the heavy,
conflict-prone dependencies move into small, single-purpose server
containers that can be rebuilt/redeployed/scaled independently, with zero
risk of one model's dependency bleeding into another's, or into Spark's own
Python environment.

**Trade-off:** network hop + (de)serialization per batch instead of an
in-process function call — added latency, and you lose Spark's data
locality benefit (the whole point of `mapPartitions` loading the model
"next to" the data). For small batches or latency-sensitive workloads this
matters; for throughput-oriented batch inference (which is what this
framework has been built and tested for) it's usually a small fraction of
total time compared to model inference itself.

## Which one to pick

- **Small number of models, occasional dependency conflicts, want to stay
  simple**: Option A. It's a smaller diff from what exists today, and it's
  what's implemented now (see above).
- **Models with genuinely incompatible dependencies, want to scale/redeploy
  each independently, or want the Spark workers themselves to stay generic
  and lightweight**: Option B. This is the one that would have fully
  prevented the numpy version bump from ever being a shared-environment risk
  in the first place.

## Adding the next pipeline (following Option A)

1. `models/pipelines/<name>/requirements.txt` — that pipeline's own deps only.
   Never add them to the root `requirements.txt`.
2. `deploy/Dockerfile.<name>` — copy `deploy/Dockerfile.ner_translate` as a
   template: `FROM multi-model-inference:latest`, add any system packages
   (apt) the pipeline needs, then `pip install -r` its own requirements file.
3. `deploy/docker-compose.<name>.yml` — copy
   `deploy/docker-compose.ner_translate.yml`, renaming the service names and
   offsetting the host ports (each new pipeline needs its own free port
   triplet, e.g. `7079:7077`/`8082:8080`/`4042:4040` for a third pipeline).
4. In `models/pipelines/manifest.json`, set `"requirements_file"` and
   `"master_url"` (matching the master service name from step 3) for the new
   entry.
5. Nothing in `submit_pipeline_job.py` needs to change — `--pipeline <name>`
   already knows how to pick up the new manifest entry's `master_url`.

Either way, **the user-facing CLI surface doesn't change** — `submit_job.py
--model X` / `submit_pipeline_job.py --pipeline X` still work the same way.
All the added complexity (which cluster or which service URL to talk to)
lives in the manifest and the engine files, not in anything a user types.

## What doesn't need to change either way

- `models/plugin_loader.py` / the plugin manifest *shape* — still name →
  metadata lookup, just with one new field added.
- `submit_job.py` / `submit_pipeline_job.py`'s argument parsing.
- The plugin/pipeline authoring contract in `docs/BRING_YOUR_OWN_MODEL.md` —
  a plugin author still just writes `forward()` or `load()`/`run()`; whether
  that code ends up called in-process or wrapped in an HTTP server is a
  packaging decision made later, not something the plugin author needs to
  think about.
