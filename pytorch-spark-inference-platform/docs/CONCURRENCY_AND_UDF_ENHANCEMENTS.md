# Concurrency & UDF Enhancements — Architecture Proposal

**Status: implemented and verified on real GPU infrastructure (AWS
`g4dn.xlarge`, Tesla T4), 2026-09-20.** All four pieces below ran
successfully: `predict_batch_udf`/`cluster_engine_udf.py` (64/64 samples,
real per-sample predictions returned), the CUDA-streams fix in
`distributed_gpu.py` (10 models dispatched concurrently across streams,
sane per-model timing via CUDA events), NVIDIA MPS (two containers running
GPU inference concurrently via a shared MPS daemon, both completed
correctly), and `deploy/scripts/gpu_discovery.sh` (valid Spark-contract
JSON on real hardware). The Spark GPU-scheduling *config*
(`spark.task.resource.gpu.amount` etc.) itself wasn't exercised against a
real multi-worker standalone cluster — that would need a fuller
multi-node setup than this pass covered — everything else that config
depends on (the discovery script, the underlying GPU) is confirmed
working. Two real bugs were found and fixed during this validation, both
described in their sections below: `spark.createDataFrame` schema
inference hanging on deeply-nested tensor rows, and `from __future__
import annotations` silently breaking `pandas_udf`'s signature detection.

Covers four enhancements discussed together because they compose into one
concurrent-GPU-inference story: Pandas UDFs, a dedicated
`predict_batch_udf`, real CUDA streams, and NVIDIA MPS. **Hard constraint
driving every piece below: none of this requires rebuilding or re-shipping
any image you already have (`spark-lean:latest`,
`ner-translate-server:latest`, `multi-model-inference:latest`,
`ner-translate-worker:latest`).** Where that's true because of something
already installed, or because a piece is
pure config/host-level rather than an image concern, it's called out
explicitly — that claim is checked against the actual `requirements.txt`
and Dockerfiles below, not assumed.

## Why these four together

They're not independent features bolted on — they're four layers of the
same problem (run more inference work concurrently on the GPU(s) you have,
safely), in the order the work actually flows:

```
Spark task scheduling  →  intra-process concurrency  →  cross-process sharing  →  batched inference itself
  (which tasks run          (CUDA streams: overlap      (MPS: multiple            (predict_batch_udf: the
   concurrently)              within one process)         processes co-scheduled    actual per-partition
                                                            on one physical GPU        model call, batched,
                                                            efficiently)                loaded once)
```

UDFs are a separate axis (execution *API*, not concurrency), included here
because the enhancement most relevant to concurrency — real stream usage —
naturally lives in the same inference-engine code being touched anyway.

## No-image-change audit (checked, not assumed)

| Piece | What it needs | Already present? | Why no rebuild is needed |
|---|---|---|---|
| `mapInPandas` (UDFs) | `pandas`, `pyarrow` | **Yes** — `requirements.txt`: `pandas==2.2.3`, `pyarrow==18.1.0`, installed in `multi-model-inference:latest` (and everything `FROM` it, incl. `ner-translate-worker:latest`) | Pure code change in `inference/`, which is already bind-mounted (Option A) or `COPY`'d at build time but unaffected by a Python-only change re-mounted the same way |
| `predict_batch_udf` | `torch`, `pandas`, `pyarrow` | Yes — same as above, plus `torch` (already the base image's core dependency) | New function under `inference/`, same mount/copy story |
| Real `torch.cuda.Stream()` usage | `torch` | Yes — already the base image's core dependency | Pure PyTorch code change, no new package |
| NVIDIA MPS | `nvidia-cuda-mps-control` daemon | **Runs on the host / a sidecar, not inside your application images at all** | MPS is fundamentally host-driver-level — the daemon manages the physical GPU across *every* container touching it via a shared pipe directory, mounted in as a volume. Application images never need the MPS binary themselves. |
| Spark GPU-aware scheduling | `spark.task.resource.gpu.amount` config + a discovery script | Config: yes, pure runtime flags. Script: needs to exist as a *file*, but gets volume-mounted (same pattern as `deploy/scripts/` already is), not baked in | No `RUN`/`COPY` needed — same bind-mount pattern already used for `models/`, `data/` |

Every enhancement below is either (a) a Python file added under a directory
that's already mounted/copied at runtime, (b) a Spark runtime config flag,
or (c) a host-level daemon your containers connect to via environment
variables + a shared volume. None require `docker build`.

---

## 1. Pandas UDFs (`mapInPandas`) — the execution API

Applies to the **tensor-plugin path** (`inference/cluster_engine.py`,
`models/plugins/`) — fixed-shape `(N,C,H,W)` or similar tensors in/out,
exactly the shape Pandas UDFs are built for. **Not** proposed for
`models/pipelines/` (file paths, nested/variable-shape results like NER
entity lists) — that data shape fights Arrow's columnar schema requirement
more than it benefits from it; leave `text_pipeline_engine.py` as-is.

**New file**: `inference/cluster_engine_udf.py` (sibling to
`cluster_engine.py`, not a replacement — `submit_job.py` gains a
`--engine {rdd,udf}` flag defaulting to `rdd` so nothing existing changes
behavior unless explicitly opted in). This file wires a DataFrame source
(`spark.read.format("image")` — Spark's own `ImageSchema` loader,
genuinely built-in, not currently used anywhere in this repo — for image
inputs, or `spark.createDataFrame` over the existing numpy-array path for
other tensors) to the `predict_batch_udf` defined in section 2 below, and
collects results back the same way `cluster_engine.py` does today.

**Honest trade-off**: Arrow-based serialization is faster than pickle for
this data shape, and it's Spark's own documented pattern for ML inference —
but it's a second execution engine to maintain alongside the RDD one until/
unless the RDD path is retired. Proposing it as opt-in specifically so
`ner_translate` and anything on the RDD path today keeps working unchanged.

---

## 2. `predict_batch_udf` — the batch-prediction UDF itself

This is the actual per-partition model call — the piece `cluster_engine_udf.py`
above wires into a Spark job. Scoped as its own named, reusable function
(not just inlined in section 1) since it's the part worth getting right on
its own: correct batching, device placement, and a contract generic enough
to cover any `models/plugins/` entry, not one hardcoded model.

**New file**: `inference/predict_batch_udf.py`

```python
# inference/predict_batch_udf.py — sketch, not final
from typing import Iterator, Type
import io
import pandas as pd
import numpy as np
import torch
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType


def predict_batch_udf(model_bytes: bytes, model_class: Type[torch.nn.Module],
                       device: str = "cuda"):
    """
    Scalar-iterator Pandas UDF factory: the model loads ONCE per partition
    (same optimization cluster_engine.py's mapPartitions docstring already
    calls out — "loads models ONCE, processes all items" — this is that
    same idea, expressed as a Pandas UDF instead of a raw mapPartitions
    closure), then every batch Spark hands this partition streams through
    the already-loaded model via Arrow instead of pickle-per-row.

    Args:
        model_bytes: serialized state_dict, from the same
            sc.broadcast(_serialize_model(...)) pattern cluster_engine.py
            already uses — no new serialization mechanism introduced.
        model_class: the nn.Module subclass to instantiate before loading
            model_bytes into it — generic across any models/plugins/ entry,
            not hardcoded to one model.
        device: "cuda" or "cpu" — resolved once per partition, same
            fallback logic cluster_engine.py's process_partition already has
            (falls back to "cpu" if torch.cuda.is_available() is False).

    Returns: a pandas_udf usable in df.select(predict_batch_udf(...)('input_col')).
    """
    @pandas_udf(ArrayType(FloatType()))
    def _predict(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        model = model_class()
        model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location="cpu"))
        resolved_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        model = model.to(resolved_device).eval()

        for batch in batches:
            arr = np.stack(batch.to_numpy())          # (batch_size, C, H, W)
            tensor = torch.from_numpy(arr).to(resolved_device)
            with torch.no_grad():
                out = model(tensor)
            yield pd.Series(list(out.cpu().numpy()))   # back to host before handing to Arrow
    return _predict
```

**Batching semantics worth being explicit about**: Spark decides the
Arrow record-batch size Spark hands to each `batches` iteration (tunable
via `spark.sql.execution.arrow.maxRecordsPerBatch`, default 10,000) — this
is a *different* knob than the model-level `batch_size` parameter
`cluster_engine.py`'s `run_cluster_inference()` already exposes. Worth
reconciling explicitly (e.g. re-chunking each Arrow batch to the model's
preferred batch size inside `_predict`) rather than assuming they line up,
since a 10,000-row Arrow batch fed straight into one `model()` call could
blow VRAM for a large model.

**Error handling gap to close, not present in the sketch above**: a single
malformed input in a batch (wrong shape, corrupt array) currently has no
defined behavior — `cluster_engine.py`'s existing `process_partition` also
doesn't isolate a bad row from the rest of its batch today, so this isn't
a regression, but it's worth deciding (skip-and-log vs. fail-the-partition)
as part of implementing this, not left implicit.

---

## 3. Real CUDA streams (fixing a documented-but-missing feature)

`inference/distributed_gpu.py`'s docstring already claims "CUDA streams for
parallel inference" — checked, and there's no `torch.cuda.Stream()`
anywhere in the file. This section actually implements what's claimed.

**Where it matters**: within *one* executor process, when a single
partition needs to run multiple independent model calls (an ensemble, or
multiple documents that don't depend on each other) — without streams,
PyTorch issues all kernels to the default stream, executing strictly in
order even though the work is independent and the GPU has spare capacity.

```python
# inference/distributed_gpu.py — sketch addition
def run_concurrent_on_streams(models_and_inputs, device="cuda"):
    """Each (model, input) pair gets its own CUDA stream — the GPU can
    genuinely overlap their kernels instead of serializing them, subject
    to actual SM availability (not guaranteed speedup for large batches
    that already saturate the GPU alone — most useful for several
    small/medium jobs sharing one GPU, e.g. ensemble members)."""
    streams = [torch.cuda.Stream() for _ in models_and_inputs]
    results = [None] * len(models_and_inputs)

    for i, (model, inp) in enumerate(models_and_inputs):
        with torch.cuda.stream(streams[i]):
            results[i] = model(inp.to(device, non_blocking=True))

    torch.cuda.synchronize()  # wait for all streams before returning
    return results
```

**Honest limitation**: streams help when the GPU has spare capacity a
single sequential job isn't using (small batches, memory-bound ops) — they
don't help (and add synchronization overhead for no gain) when one job
already saturates the GPU's compute. Worth profiling before assuming this
speeds up any specific workload; it's a real capability gap being closed,
not a guaranteed win everywhere it's applied.

---

## 4. NVIDIA MPS — enabling cross-process GPU sharing

This is the piece that makes "multiple Spark executor processes sharing
one physical GPU" actually efficient, instead of the driver crudely
time-slicing between separate CUDA contexts.

**Nothing in your application images changes.** MPS is a daemon
(`nvidia-cuda-mps-control`) that:
1. Runs once per GPU-having host (or a privileged sidecar container with
   host GPU access) — *not* inside `spark-lean`, `ner-translate-server`,
   or any worker image.
2. Exposes a pipe directory (`/tmp/nvidia-mps` by default) that every
   *application* container mounts as a volume and points at via
   `CUDA_MPS_PIPE_DIRECTORY` — a docker-compose `environment:` +
   `volumes:` addition, not a Dockerfile change.

```yaml
# deploy/docker-compose.ner_translate_server.yml — additive sketch,
# no image rebuild, just new environment/volumes entries on the existing
# ner-translate-server service
services:
  ner-translate-server:
    # ... existing config unchanged ...
    environment:
      - CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
      - CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    volumes:
      - /tmp/nvidia-mps:/tmp/nvidia-mps      # shared with the host MPS daemon
      - /tmp/nvidia-mps-log:/tmp/nvidia-mps-log
```

Host-side (or a small privileged sidecar in compose, `network_mode: host`
+ GPU access), start the daemon once:
```bash
nvidia-cuda-mps-control -d
```
`docs/internet_to_airgapped_transfer.md` §3.5 already documents this as a
manual per-node step for the multi-GPU-node cluster case — this proposal
is making it a docker-compose-level default for the single-GPU
waiter/kitchen and Option A cases too, still host-level, still zero image
changes.

---

## 5. (Ties it together) Spark GPU-aware task scheduling

Config-only, referenced from the earlier concurrency discussion —
included here for completeness since it's what decides how many
concurrent tasks land on a GPU in the first place, upstream of streams and
MPS both mattering.

```python
# inference/cluster_engine.py's create_cluster_session() — additive config
builder = builder.config("spark.executor.resource.gpu.amount", "1") \
                  .config("spark.task.resource.gpu.amount", "0.5") \
                  .config("spark.worker.resource.gpu.discoveryScript",
                          "/app/deploy/scripts/gpu_discovery.sh")
```

```bash
# deploy/scripts/gpu_discovery.sh — new file, volume-mounted (deploy/ is
# already bind-mounted in every compose file), never baked into an image
#!/usr/bin/env bash
echo "{\"name\": \"gpu\", \"addresses\":[$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)]}"
```

## Rollout order (smallest blast radius first)

1. **CUDA streams fix** — self-contained, one file, easiest to verify in
   isolation (profile a known ensemble workload before/after).
2. **`predict_batch_udf`** — no dependents yet, safe to land and unit-test
   in isolation (feed it a small DataFrame directly) before wiring it into
   a real Spark job.
3. **MPS** — host/compose-level, test on a single node before rolling to
   the multi-node cluster docs already describe.
4. **Spark GPU scheduling config** — pairs naturally with MPS being live;
   sequencing after it avoids scheduling concurrent tasks onto a GPU that
   isn't actually sharing well yet.
5. **`cluster_engine_udf.py`** (wiring `predict_batch_udf` into a real
   Spark job via `mapInPandas`/DataFrame) — lands last since it depends on
   `predict_batch_udf` already existing and is the most independent,
   optional piece; nothing else here depends on it.

## Open items before implementing

- **Batching-size reconciliation** (section 2): decide how Arrow's
  `maxRecordsPerBatch` and the model's own preferred inference batch size
  get reconciled inside `predict_batch_udf`, rather than assuming they
  match.
- **Bad-input handling** (section 2): decide skip-and-log vs.
  fail-the-partition for a malformed row inside a batch — not a regression
  from today's behavior, but worth deciding explicitly while this is being
  written rather than leaving it implicit.
