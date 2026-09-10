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

Instead of one `multi-model-inference:latest` image running everything,
build a separate worker image per model (or per model family), each with
only that model's dependencies, and run each as its own Spark
master+worker pair (or at minimum its own worker pool joined to a shared
master via a custom resource label).

**Changes required:**

| File | Change |
|---|---|
| `deploy/Dockerfile.<model>` (new, one per model/family) | Same base as today's `Dockerfile`, but `requirements.txt` trimmed to only that model's deps (e.g. a `Dockerfile.ner_translate` installing `gliner transformers sentencepiece py3langid pyarrow`, nothing else) |
| `deploy/docker-compose.cluster.yml` | One worker service block per model image, each pointing at its own `Dockerfile.<model>`, instead of the current single `spark-cpu-worker`/`spark-gpu-worker` pair built from one shared image |
| `models/plugins/manifest.json`, `models/pipelines/manifest.json` | Add a `"master_url"` field per entry (e.g. `"spark://ner-worker:7077"`) so the CLI knows which cluster to submit to for that model |
| `submit_job.py`, `submit_pipeline_job.py` | Read `master_url` from the manifest entry as the default (instead of always falling back to `local[4]`/env var) when `--master` isn't explicitly passed |

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
  simple**: Option A. It's a smaller diff from what exists today.
- **Models with genuinely incompatible dependencies, want to scale/redeploy
  each independently, or want the Spark workers themselves to stay generic
  and lightweight**: Option B. This is the one that would have fully
  prevented the numpy version bump from ever being a shared-environment risk
  in the first place.

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
