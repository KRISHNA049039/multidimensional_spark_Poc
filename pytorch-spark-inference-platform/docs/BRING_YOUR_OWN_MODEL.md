# Bring-Your-Own-Model (BYOM) Framework Design

## 1. Short answer

Yes. `pytorch-spark-inference-platform` already *is* a working distributed CPU/GPU
inference engine — it just hardcodes its own 10 demo models instead of accepting
arbitrary ones. Turning it into a general framework where you drop in a model
file and it runs as a parallel Spark job across CPU and GPU workers requires:

- **1 new folder** (`models/plugins/`) for your model files + weights
- **1 new manifest file** describing each plugin
- **2 new small helper files** (a plugin loader + a CLI submit script)
- **1 small, additive edit** to an existing file (`inference/cluster_engine.py`)

Nothing else moves. No existing model file, docker-compose file, or directory
is renamed/restructured.

## 2. What the repo already does (verified in code)

| Capability | Where | Notes |
|---|---|---|
| Spark cluster session (master + CPU/GPU workers) | `inference/cluster_engine.py:43-86` | `master_url` from `SPARK_MASTER_URL` env, falls back to `local[4]` |
| Data partitioning across executors | `cluster_engine.py:147-161` | driver slices numpy arrays into `num_partitions` chunks |
| Model-loaded-once-per-executor pattern | `cluster_engine.py:163-256` (`mapPartitions`) | avoids reloading weights per task |
| Weight broadcast | `cluster_engine.py:141-145` | `state_dict()` → bytes → `sc.broadcast()` |
| CPU/GPU device selection | `cluster_engine.py:180-191` | `device_mode`: `cpu_only` / `gpu_only` / `hybrid` (auto-detect `torch.cuda.is_available()`) |
| Cluster topology | `deploy/docker-compose.cluster.yml` | 1 master, N CPU workers (`spark-cpu-worker`, scalable), N GPU workers (`spark-gpu-worker`, scalable) |
| Live code sync (no rebuild for code/model changes) | same compose file, `volumes:` | `../models`, `../inference`, `../data`, `../benchmark` are **bind-mounted**, not baked into the image |
| Model metadata registry | `models/model_registry.py` (`ModelRegistry`) | register/load/serialize — but populated by a hardcoded call list in `models/__init__.py:get_default_registry()` |

**The only real coupling point:** `_get_class_map()` in `cluster_engine.py:95-114`
is a hardcoded `{name: ModelClass}` dict, re-imported *inside* the executor
closure so each executor can rehydrate the broadcast weights into a live
`nn.Module`. Today, a model can only run on a worker if it's in that dict.
Everything else (session setup, partitioning, broadcast, device routing,
result collection) is already generic — it just calls `model(batch_tensor)`
and doesn't care what the model is.

Because the bind mounts already sync `models/`, `inference/`, and `data/`
live into every container, **dropping a new file into `models/` requires no
image rebuild** as long as its Python dependencies (torch, torchvision, etc.)
are already in the image. This matters because the platform is explicitly
airgapped (`docs/TECHNICAL_ARCHITECTURE.md`: "No runtime downloads, all
weights baked into the image") — new plugin weights must ship as local files,
not `torch.hub`/pretrained downloads.

## 3. Proposed additions (all new, nothing existing renamed)

```
pytorch-spark-inference-platform/
├── models/
│   ├── plugins/                  # NEW — drop your model files here
│   │   ├── manifest.json         # NEW — declares each plugin model
│   │   ├── weights/               # NEW — .pt state_dict files (optional)
│   │   └── my_model.py            # your file, added by you
│   └── plugin_loader.py          # NEW — dynamic import + registry glue
├── data/
│   └── generic_loader.py         # NEW — loads .npy/.npz into the {name: ndarray} shape run_cluster_inference expects
└── submit_job.py                 # NEW — CLI wrapper around the existing cluster_engine flow
```

### 3.1 `models/plugins/manifest.json`

```json
{
  "my_model": {
    "module": "models.plugins.my_model",
    "class_name": "MyModel",
    "weights_path": "models/plugins/weights/my_model.pt",
    "input_shape": [128],
    "category": "custom",
    "estimated_memory_mb": 100
  }
}
```

### 3.2 `models/plugin_loader.py` (new file)

```python
import importlib, json, os

def _load_manifest(manifest_path="models/plugins/manifest.json"):
    with open(manifest_path) as f:
        return json.load(f)

def get_plugin_class_map(manifest_path="models/plugins/manifest.json"):
    """{name: ModelClass} for every plugin — merged into cluster_engine's class map."""
    classes = {}
    for name, spec in _load_manifest(manifest_path).items():
        mod = importlib.import_module(spec["module"])
        classes[name] = getattr(mod, spec["class_name"])
    return classes

def register_plugins(registry, manifest_path="models/plugins/manifest.json"):
    """Adds plugin entries into an existing ModelRegistry — no ModelRegistry changes needed."""
    for name, spec in _load_manifest(manifest_path).items():
        mod = importlib.import_module(spec["module"])
        registry.register(
            name=name,
            model_class=getattr(mod, spec["class_name"]),
            input_shape=tuple(spec["input_shape"]),
            output_desc=spec.get("output_desc", ""),
            category=spec.get("category", "custom"),
            estimated_memory_mb=spec.get("estimated_memory_mb", 100),
        )
```

Weight loading reuses `ModelRegistry.load_model()`/`deserialize_model()`
unchanged — if `weights_path` is set, load it via `state_dict()` before the
first broadcast; if not, the model runs with random init (fine for pure
throughput benchmarking).

### 3.3 The one required edit — `inference/cluster_engine.py`

`_get_class_map()` (lines 95-114) needs to merge plugin classes in, since
that's the only place executors resolve `name → class` to rehydrate broadcast
weights. This is a 3-line addition, not a rewrite:

```python
def _get_class_map():
    from models.plugin_loader import get_plugin_class_map
    class_map = {
        "ew_classifier": EWSignalClassifier,
        # ...existing 10 entries, unchanged...
    }
    class_map.update(get_plugin_class_map())   # <-- new line
    return class_map
```

(`inference/distributed_gpu.py` has its own duplicate class map, but the
code's own header comment says it's superseded by `cluster_engine.py` — leave
it untouched; it isn't on the path `benchmark/cluster_benchmark.py` actually
uses.)

### 3.4 `submit_job.py` (new, repo-root CLI)

Thin wrapper — does not reimplement anything, just sequences the existing
building blocks so you don't have to hand-edit a benchmark script per model:

```python
# python submit_job.py --model my_model --input data/my_inputs.npy \
#     --mode hybrid --partitions 8 --batch-size 256
import argparse, numpy as np
from models import get_default_registry
from models.plugin_loader import register_plugins
from inference.cluster_engine import create_cluster_session, run_cluster_inference

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)          # .npy file, shape (N, *input_shape)
    p.add_argument("--mode", default="hybrid", choices=["cpu_only", "gpu_only", "hybrid"])
    p.add_argument("--partitions", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    registry = get_default_registry()
    register_plugins(registry)                         # adds anything in manifest.json

    device = "cuda" if args.mode != "cpu_only" else "cpu"
    model = registry.load_model(args.model, device="cpu")  # broadcast always starts from CPU state_dict

    data = {args.model: np.load(args.input).astype("float32")}
    spark = create_cluster_session(app_name=f"byom-{args.model}")
    result = run_cluster_inference(
        spark, data, {args.model: model},
        num_partitions=args.partitions, batch_size=args.batch_size,
        device_mode=args.mode,
    )
    print(result)

if __name__ == "__main__":
    main()
```

### 3.5 `data/generic_loader.py` (new)

Only needed because `data/image_generator.py`/`signal_generator.py` hardcode
shapes for the 10 built-in demo models. For a plugin model, `submit_job.py`
above already handles the common case (a single `.npy` file) directly with
`np.load`; add `generic_loader.py` only if you need to combine multiple
`.npy`/`.npz` shards from a directory into one array before partitioning.

## 4. Contract your model file must follow

Matches the convention every existing model already uses — no new abstraction
to learn:

- Plain `torch.nn.Module` subclass.
- `__init__(self)` takes **no required arguments** (the class map instantiates
  it as `ModelClass()` on the executor). If you need config, hardcode it in
  `__init__` or read from a small constant at module import time.
- `forward(self, x: torch.Tensor) -> torch.Tensor` — batch-first, same dtype
  path as the rest of the platform (`float32`).
- No network calls in `__init__`/`forward` (airgapped constraint) — if you
  need pretrained weights, save them as a local `state_dict` `.pt` file and
  point `weights_path` at it in the manifest.
- Any new pip dependency your model needs must be added to
  `requirements.txt` and baked into the `multi-model-inference` image once
  (`docker build`) — this is the one case that *does* need a rebuild, since
  bind mounts sync code/data but not installed packages.

## 5. CPU/GPU distribution — unchanged

Once your class is resolvable via the merged class map, it automatically
gets the same treatment as the 10 built-in models:

- `--mode cpu_only` / `gpu_only` / `hybrid` controls `device_mode`, broadcast
  to every executor (`cluster_engine.py:180-191`).
- Scale workers with `docker compose -f deploy/docker-compose.cluster.yml up
  --scale spark-cpu-worker=N --scale spark-gpu-worker=M` — no code change.
- Note the GPU worker is currently disabled in the compose file pending an
  image rebuild for `sm_120` support (see comment in
  `docker-compose.cluster.yml`); this is a pre-existing platform limitation,
  not something the plugin layer changes.

## 6. End-to-end usage

1. Add `models/plugins/my_model.py` with your `nn.Module`.
2. (Optional) drop pretrained weights at `models/plugins/weights/my_model.pt`.
3. Add one entry for it in `models/plugins/manifest.json`.
4. If your model needs new pip packages, add them to `requirements.txt` and
   rebuild the image once: `docker build -t multi-model-inference:latest .`
   — otherwise skip this step entirely.
5. Start (or reuse) the cluster:
   `docker compose -f deploy/docker-compose.cluster.yml up -d`
6. Submit the job:
   `docker exec -it spark-master bash -c "python submit_job.py --model my_model --input /app/data/my_inputs.npy --mode hybrid --partitions 8"`
7. Watch progress at `http://localhost:8080` (cluster UI) and
   `http://localhost:4040` (application UI); results land under
   `./results/` on the host (already bind-mounted).

## 7. What stays exactly as-is

- All 10 existing models, `model_registry.py`'s public API, `models/__init__.py`.
- `benchmark/cluster_benchmark.py` and every existing benchmark/report script.
- `docker-compose.yml`, `docker-compose.cluster.yml` topology and volumes.
- `inference/hybrid_cpu_gpu.py`, `single_gpu.py`, `distributed_gpu.py`,
  `cuda_streams_engine.py`, `gpu_memory_manager.py` — untouched.
- Directory layout — only additive files/folders as listed in §3.

## 8. Known limitations to call out

- Spark's *native* GPU resource scheduling
  (`spark.task.resource.gpu.amount`) is referenced in docs but never actually
  wired in — GPU placement today is via `CUDA_VISIBLE_DEVICES` per worker
  container, not per-task Spark scheduling. A plugin model inherits this same
  limitation (fine for one-GPU-per-worker-container topologies, not true
  fractional GPU scheduling).
- `submit_job.py` as sketched assumes a single input array per job (one model
  at a time). If you want to submit several plugin models in one Spark job
  (like the existing multi-model benchmark does), extend the `--model`
  argument to accept a comma-separated list and build the `data`/`models`
  dicts accordingly — the underlying `run_cluster_inference()` already
  supports multiple models per call.
- Heterogeneous input shapes across plugins in the same job need one `.npy`
  array per model (matches existing `data: Dict[str, np.ndarray]` contract);
  there's no automatic batching across models of different shapes.
