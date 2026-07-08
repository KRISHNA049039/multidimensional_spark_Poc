# Multi-Model Distributed Inference Platform — Technical Architecture

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Models — Input/Output Specification](#3-models--inputoutput-specification)
4. [Core Components](#4-core-components)
5. [Three Inference Modes — Code Flow](#5-three-inference-modes--code-flow)
6. [GPU Sharing Implementation](#6-gpu-sharing-implementation)
7. [Data Flow End-to-End](#7-data-flow-end-to-end)
8. [Benchmark Runner](#8-benchmark-runner)
9. [Deployment Flow](#9-deployment-flow)
10. [Metrics Output Format](#10-metrics-output-format)

---

## 1. Overview

This platform runs **10 PyTorch models in parallel** across 3 different execution
strategies, benchmarking throughput for electronic warfare (EW) signal processing,
image classification, and object detection — all coordinated through Apache Spark
for distributed execution.

### Design Goals

| Goal | How Achieved |
|------|---------------|
| Run 7-10 models concurrently | CUDA Streams (single GPU) + MPS (cluster) |
| Work on local Docker AND real cluster | Same code, `master()` URL is the only difference |
| Efficient GPU sharing | GPUMemoryManager with budget-aware placement |
| Airgapped deployable | No runtime downloads, all weights baked into Docker image |
| Publishable metrics | JSON + Markdown report generator |

---

## 2. Project Structure

```
pytorch-spark-inference-platform/
├── models/
│   ├── model_registry.py       # Central registry: metadata, load, serialize
│   ├── ew_signal_model.py      # EW Signal Classifier (MLP, 8 classes)
│   ├── signal_models.py        # Denoiser, Prioritizer, Fingerprinter, Anomaly
│   ├── image_models.py         # ResNet-18, MobileNetV3, EfficientNet-B0
│   ├── yolo_model.py           # YOLOv8 Nano/Small wrappers
│   └── __init__.py             # get_default_registry() — wires all 10 models
├── data/
│   ├── signal_generator.py     # Synthetic IQ signal generator (128-dim)
│   └── image_generator.py      # Synthetic image generator (224² / 640²)
├── inference/
│   ├── cuda_streams_engine.py  # Core: multi-model parallel execution engine
│   ├── gpu_memory_manager.py   # Budget-aware GPU/CPU placement planner
│   ├── single_gpu.py           # Mode 2: all models on 1 GPU
│   ├── hybrid_cpu_gpu.py       # Mode 3: split GPU/CPU by memory budget
│   └── distributed_gpu.py      # Mode 1: Spark RDD across GPU cluster
├── benchmark/
│   └── run_benchmark.py        # Unified runner — all models × all modes
├── deploy/
│   ├── Dockerfile               # CUDA 12.1 + Python 3.11 + Java 17 + deps
│   ├── docker-compose.yml       # Local dev (volume-mounted, single GPU)
│   └── docker-compose.cluster.yml  # Multi-node Spark + GPU workers
├── docs/
│   └── TECHNICAL_ARCHITECTURE.md   # This file
├── results/
│   ├── metrics_report.md        # Generated after each benchmark run
│   └── raw_results.json         # Machine-readable results
└── requirements.txt
```

### Module Dependency Graph

```
run_benchmark.py
    ├── models/__init__.py (get_default_registry)
    │     └── model_registry.py ← ew_signal_model.py, signal_models.py,
    │                              image_models.py, yolo_model.py
    ├── data/image_generator.py
    │     └── data/signal_generator.py
    ├── inference/single_gpu.py
    │     └── inference/cuda_streams_engine.py
    ├── inference/hybrid_cpu_gpu.py
    │     ├── inference/cuda_streams_engine.py
    │     └── inference/gpu_memory_manager.py
    └── inference/distributed_gpu.py
          └── (imports models/ dynamically inside Spark worker function)
```

---

## 3. Models — Input/Output Specification

### Signal Models (5) — Input: 128-dim float32 IQ vector

| Model | Input Shape | Output Shape | Output Meaning | Params | Est. GPU MB |
|-------|------------|---------------|-----------------|--------|-------------|
| `ew_classifier` | (N, 128) | (N, 8) | Class logits (CW/Pulsed/FMCW/Jammer/Comm) | 340K | 50 |
| `signal_denoiser` | (N, 128) | (N, 128) | Denoised IQ vector (autoencoder reconstruction) | 1.2M | 100 |
| `threat_prioritizer` | (N, 128) | (N,) | Priority score [0,1] via self-attention | 8M | 350 |
| `rf_fingerprinter` | (N, 128) | (N, 32) | L2-normalized emitter embedding (1D-CNN) | 2.5M | 120 |
| `anomaly_detector` | (N, 128) | (N,) | Anomaly score (VAE reconstruction error) | 1.8M | 100 |

### Image Classification Models (3) — Input: 3×224×224 RGB float32 [0,1]

| Model | Input Shape | Output Shape | Output Meaning | Params | Est. GPU MB |
|-------|------------|---------------|-----------------|--------|-------------|
| `resnet18` | (N, 3, 224, 224) | (N, 1000) | ImageNet class logits | 11.7M | 300 |
| `mobilenetv3` | (N, 3, 224, 224) | (N, 1000) | ImageNet class logits | 5.4M | 150 |
| `efficientnet_b0` | (N, 3, 224, 224) | (N, 1000) | ImageNet class logits | 5.3M | 200 |

### Object Detection Models (2) — Input: 3×640×640 RGB float32 [0,1]

| Model | Input Shape | Output Shape | Output Meaning | Params | Est. GPU MB |
|-------|------------|---------------|-----------------|--------|-------------|
| `yolov8_nano` | (N, 3, 640, 640) | list[Results] or (N,) fallback | Bounding boxes, class, confidence | 3.2M | 200 |
| `yolov8_small` | (N, 3, 640, 640) | list[Results] or (N,) fallback | Bounding boxes, class, confidence | 11.2M | 400 |

**Total registry: 10 models, ~1,970 MB estimated GPU memory**

### Unified Input Dictionary (used across all inference modes)

```python
data = {
    "ew_classifier":       np.ndarray (N1, 128)  float32,
    "signal_denoiser":     np.ndarray (N1, 128)  float32,  # same signal batch
    "threat_prioritizer":  np.ndarray (N1, 128)  float32,
    "rf_fingerprinter":    np.ndarray (N1, 128)  float32,
    "anomaly_detector":    np.ndarray (N1, 128)  float32,
    "resnet18":            np.ndarray (N2, 3, 224, 224) float32,
    "mobilenetv3":         np.ndarray (N2, 3, 224, 224) float32,
    "efficientnet_b0":     np.ndarray (N2, 3, 224, 224) float32,
    "yolov8_nano":         np.ndarray (N3, 3, 640, 640) float32,
    "yolov8_small":        np.ndarray (N3, 3, 640, 640) float32,
}
```
`N1`, `N2`, `N3` are independently configurable (`--signal-samples`, `--image-samples`, `--detection-samples`).

---

## 4. Core Components

### 4.1 `ModelRegistry` (`models/model_registry.py`)

Central catalog. Does NOT load model weights until requested (lazy loading).

```python
registry = get_default_registry()          # registers metadata for 10 models
registry.summary()                           # prints table: name, category, memory, shape
models = registry.load_all(device="cpu")     # instantiates all 10 nn.Module objects
model_bytes = registry.serialize_model("resnet18")   # torch.save() to bytes, for Spark broadcast
model = registry.deserialize_model("resnet18", model_bytes, device="cuda")
```

Key methods:
| Method | Purpose |
|--------|---------|
| `register(name, class, shape, desc, category, mem_mb)` | Add model metadata |
| `load_model(name, device)` | Instantiate + cache + place on device |
| `load_all(device)` | Load every registered model |
| `serialize_model(name)` / `serialize_all()` | `state_dict` → bytes for broadcast |
| `deserialize_model(name, bytes, device)` | bytes → model on target device |
| `total_memory_estimate_mb()` | Sum of all `estimated_memory_mb` |

### 4.2 `CUDAStreamsEngine` (`inference/cuda_streams_engine.py`)

The parallel execution core used by **all three modes**.

```python
engine = CUDAStreamsEngine(models_dict, device_map_dict, batch_size=256)
outputs = engine.infer_all_parallel(inputs_dict)   # runs ALL models concurrently
```

Internal mechanics:
1. On init: creates one `torch.cuda.Stream()` per GPU-assigned model, and a
   `ThreadPoolExecutor` for CPU-assigned models.
2. `infer_all_parallel()`:
   - **Phase 1**: For each GPU model, enters `torch.cuda.stream(stream)` context,
     moves input non-blocking, calls `model(x)`. This call returns immediately —
     the actual kernel is queued on that model's stream.
   - CPU models are submitted to the thread pool at the same time (`.submit()`),
     so GPU queuing and CPU execution start together.
   - **Phase 2**: `torch.cuda.synchronize()` — blocks until ALL GPU streams finish.
     Because streams are independent, the GPU scheduler interleaves their kernels,
     so total GPU time ≈ max(individual model times), not sum.
   - **Phase 3**: `.cpu()` on GPU outputs, `.result()` on CPU futures.

This is literally how the "10 models running in parallel" requirement is met on
one GPU: independent CUDA streams + GPU's own kernel scheduler.

### 4.3 `GPUMemoryManager` (`inference/gpu_memory_manager.py`)

Plans WHERE each model goes (which GPU, or CPU) before any inference runs.

```python
mgr = GPUMemoryManager(reserve_mb=500, strategy="priority")
device_map = mgr.plan_placement(
    models_with_sizes={"yolov8_small": 400, "signal_denoiser": 100, ...},
    priorities={"yolov8_small": 10, "signal_denoiser": 2, ...},
)
mgr.report()   # prints per-GPU allocation table + CPU fallback list
```

Strategies:
| Strategy | Sort order | Use case |
|----------|-----------|----------|
| `priority` (default) | By caller-supplied priority, descending | Business-critical models get GPU first |
| `largest_first` | By memory size, descending | Maximize compute payoff per GPU byte |
| `balanced` | Round-robin across multiple GPUs by free space | Multi-GPU nodes |
| `greedy` | Registration order | Simple deterministic fallback |

Placement loop: for each model (in sorted order), scan GPUs for one with enough
`available_mb`; if none fit, model → `"cpu"`. This is exactly how Mode 3 (Hybrid)
decides its split.

---

## 5. Three Inference Modes — Code Flow

### Mode 2: Single GPU (`inference/single_gpu.py`) — start here, it's the simplest

```
run_single_gpu_inference(models, data, batch_size, device="cuda")
  │
  ├─ 1. If CUDA unavailable → device="cpu" (graceful fallback)
  ├─ 2. Move all 10 models to device, set .eval()
  ├─ 3. Create CUDAStreamsEngine(models, device_map, batch_size)
  ├─ 4. Warmup: 3 dummy calls to engine.infer_all_parallel() (stabilizes cuDNN)
  ├─ 5. Loop over batches (max_samples // batch_size):
  │       for each model: slice its numpy array → torch tensor
  │       engine.infer_all_parallel(batch_inputs)   ← all 10 run concurrently
  ├─ 6. elapsed_time = wall clock across all batches
  └─ 7. Return {mode, elapsed_time, total_throughput, per_model_processed, ...}
```

**Input:** `models: Dict[str, nn.Module]`, `data: Dict[str, np.ndarray]`
**Output:** single result dict (see §10)

### Mode 3: Hybrid CPU+GPU (`inference/hybrid_cpu_gpu.py`)

```
run_hybrid_inference(models, model_sizes_mb, data, batch_size, gpu_memory_limit_mb, priorities, strategy)
  │
  ├─ 1. GPUMemoryManager.plan_placement(sizes, priorities) → device_map
  │       e.g. {"yolov8_small":"cuda:0", "resnet18":"cuda:0",
  │             "signal_denoiser":"cpu", "anomaly_detector":"cpu", ...}
  ├─ 2. Split models dict into gpu_models{} / cpu_models{} per device_map
  ├─ 3. gpu_engine = CUDAStreamsEngine(gpu_models, ...)   (only if any GPU models)
  ├─ 4. cpu_pool = ThreadPoolExecutor(max_workers=len(cpu_models))
  ├─ 5. Warmup GPU engine only
  ├─ 6. Loop over batches:
  │       gpu_inputs{}, cpu_inputs{} ← split incoming batch by device_map
  │       gpu_engine.infer_all_parallel(gpu_inputs)   ← streams, non-blocking-ish
  │       cpu_pool.submit(...) for each cpu model      ← threads, truly parallel
  │       wait on cpu futures
  └─ 7. Return result dict + which models landed on gpu vs cpu
```

**Key difference from Mode 2:** not all models get a CUDA stream — small models
(denoiser, anomaly detector) intentionally run on CPU threads to leave GPU memory
for the large image/detection models. This matters directly for the GTX 1650 (4GB) case.

### Mode 1: Distributed GPU (`inference/distributed_gpu.py`) — Spark-based

```
run_distributed_gpu_inference(spark, data, models, model_classes, num_partitions, batch_size)
  │
  ├─ DRIVER SIDE:
  │   1. Serialize every model's state_dict → bytes (torch.save to BytesIO)
  │   2. sc.broadcast(model_bytes_map)      ← sent once to all executors
  │   3. For each model's data array: split into num_partitions numpy chunks,
  │      sc.broadcast(chunks) per model      ← avoids parallelize() OOM (see §7)
  │   4. index_rdd = sc.parallelize(range(num_partitions), num_partitions)
  │
  ├─ EXECUTOR SIDE (inside infer_on_partition(partition_idx), runs on each worker):
  │   5. device = "cuda" if (CUDA available AND SPARK_EXECUTOR_GPU=1) else "cpu"
  │      ← local Docker mode defaults to CPU (thread-safety, see note below)
  │   6. Deserialize all 10 models from the broadcast bytes onto `device`
  │   7. For each model: slice this partition's chunk, run batched forward pass
  │   8. Return {model_name: samples_processed} for this partition
  │
  └─ DRIVER SIDE (after .collect()):
      9. Sum samples_processed across all partitions → total_processed{}
      10. throughput = total_all / elapsed_time
```

**Why CPU by default in local mode:** Spark `local[*]` runs partitions as
**threads inside one JVM/Python process**. If all threads call `.to("cuda")`
simultaneously on a single physical GPU, they contend for the same CUDA context
and can deadlock (observed and fixed during development — see
`pytorch-spark-ew-poc/docs` troubleshooting notes). On a **real cluster**, each
executor is a **separate OS process** on its own machine/GPU — set
`SPARK_EXECUTOR_GPU=1` there and it's safe and fast (especially with MPS, §6).

---

## 6. GPU Sharing Implementation

| Layer | Mechanism | Where in code |
|-------|-----------|----------------|
| Intra-process, single GPU | CUDA Streams (kernel interleaving) | `cuda_streams_engine.py` |
| Intra-process, mixed devices | CUDA Streams (GPU) + ThreadPoolExecutor (CPU) | `hybrid_cpu_gpu.py` |
| Inter-process, single machine | NVIDIA MPS daemon (see `docs/GPU_SHARING_AND_MPP_ARCHITECTURE.md` in the EW PoC repo) | Deployment-level, no code change |
| Inter-machine, cluster | Spark task scheduling + broadcast + MPS per node | `distributed_gpu.py` + `spark.task.resource.gpu.amount=0.1` |

### Why streams alone are not enough on a cluster

CUDA Streams only help **within one process**. On a Spark cluster, each executor
is its own OS process. Without MPS, executor processes on the *same* GPU node
still serialize at the CUDA context level (same failure mode we hit in local mode).
**MPS is the piece that makes multi-process GPU sharing on a node actually parallel.**
Enable it once per GPU node (systemd service), and `distributed_gpu.py` needs zero
code changes — it's an infrastructure switch, not an application switch.

### Fractional GPU scheduling (cluster spark-defaults.conf)

```properties
spark.executor.resource.gpu.amount=1
spark.task.resource.gpu.amount=0.1   # 10 tasks (models) can share one GPU
spark.executorEnv.CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
spark.executorEnv.CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
```

---

## 7. Data Flow End-to-End

```
                    generate_mixed_data(signal_n, image_n, detection_n)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     signal_generator.py   image_generator.py     image_generator.py
     (128-dim IQ vectors)  (224×224 RGB)           (640×640 RGB)
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                    data: Dict[str, np.ndarray]   ← one array per model
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  single_gpu.py               hybrid_cpu_gpu.py          distributed_gpu.py
  (Mode 2)                    (Mode 3)                   (Mode 1)
        │                           │                           │
        ▼                           ▼                           ▼
  CUDAStreamsEngine      GPUMemoryManager.plan_placement   Spark broadcast +
  (all 10 on 1 GPU)      → split → CUDAStreamsEngine       partition chunks
                          (GPU subset) + ThreadPool         → per-executor
                          (CPU subset)                        CUDAStreams-like
                                                               loop (CPU/GPU)
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                    result dict: {mode, elapsed_time, total_throughput,
                                   per_model_processed, ...}
                                    │
                                    ▼
                    run_benchmark.py collects all 3 mode results
                                    │
                                    ▼
                    generate_report() → results/metrics_report.md
                    json.dump()       → results/raw_results.json
```

### Why data is pre-chunked/broadcast instead of `sc.parallelize(full_array)`

Sending large numpy arrays directly through `sc.parallelize()` serializes them
through the JVM driver's Py4J socket — for arrays in the 100s-of-MB range this
overflows driver memory and hangs (`[Stage 0: (0+8)/8]` never completing — a
real failure encountered and fixed during this project). The fix used throughout
`distributed_gpu.py`: pre-slice the numpy array into `num_partitions` chunks on
the driver, `sc.broadcast()` each chunk once, and have the RDD only carry
lightweight partition **indices** (`sc.parallelize(range(num_partitions), ...)`).
Each task looks up its chunk from the broadcast variable instead of receiving it
via the parallelize path. Same pattern is used for the EW-only PoC in
`pytorch-spark-ew-poc/inference/spark_inference.py`.

---

## 8. Benchmark Runner (`benchmark/run_benchmark.py`)

### CLI

```bash
python benchmark/run_benchmark.py                     # all 3 modes, defaults
python benchmark/run_benchmark.py --mode single_gpu    # just Mode 2
python benchmark/run_benchmark.py --mode hybrid --batch-size 512
python benchmark/run_benchmark.py --mode distributed --partitions 8
python benchmark/run_benchmark.py --signal-samples 50000 --image-samples 2000 --detection-samples 500
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `all` | `all` \| `single_gpu` \| `hybrid` \| `distributed` |
| `--signal-samples` | 10000 | Count for the 5 signal models |
| `--image-samples` | 500 | Count for the 3 image classifiers |
| `--detection-samples` | 100 | Count for the 2 YOLO models |
| `--batch-size` | 256 | Per-model batch size in all modes |
| `--partitions` | 4 | Spark partitions (Mode 1 only) |

### Execution sequence inside `main()`

1. `get_system_info()` — platform, python/torch version, GPU name/memory/count
2. `get_default_registry()` + `registry.summary()` — prints the 10-model table
3. `registry.load_all("cpu")` — instantiate all models (weights on CPU initially;
   each mode moves them to the right device itself)
4. `generate_mixed_data(...)` — builds the unified `data` dict (§7)
5. Depending on `--mode`, call one or more of:
   `run_mode_single_gpu`, `run_mode_hybrid`, `run_mode_distributed`
6. `generate_report(results, sys_info)` → markdown string
7. Write `results/metrics_report.md` and `results/raw_results.json`

---

## 9. Deployment Flow

### Local Docker (development)

```bash
cd pytorch-spark-inference-platform
docker compose -f deploy/docker-compose.yml up --build
```
- `runtime: nvidia` passes the local GPU through
- `volumes: - ..:/app` mounts source live — edit code, no rebuild needed
- Default command runs `--mode all`

### Cluster Docker (multi-node, on-prem/DRDO)

```bash
docker compose -f deploy/docker-compose.cluster.yml up --build
```
- `spark-master` service: driver + UI on ports 7077/8080/4040
- `spark-worker` service: `deploy.replicas: 2`, each with `runtime: nvidia` and
  `SPARK_EXECUTOR_GPU=1` — this is the flag `distributed_gpu.py` checks to decide
  GPU vs CPU placement inside `infer_on_partition()`
- For airgapped: build once with internet, `docker save`/`docker load` per
  `pytorch-spark-ew-poc/AIRGAPPED_DEPLOYMENT.md` (same procedure applies here)

### AWS test path (before DRDO)

- `g4dn.xlarge` spot (T4 16GB) for worker(s), `t3.medium` for master
- Same Docker images, same compose files — only the instance/network layer differs
- Detailed in `pytorch-spark-ew-poc/docs/EXPANDED_PROJECT_PLAN.md`, Phase 2

---

## 10. Metrics Output Format

### `results/raw_results.json` (machine-readable)

```json
{
  "system_info": {"platform": "...", "gpu_name": "...", "cuda": true, ...},
  "config": {"signal_samples": 10000, "image_samples": 500, "batch_size": 256, ...},
  "models": {"ew_classifier": {"category": "signal", "memory_mb": 50}, ...},
  "single_gpu": {
    "mode": "single_gpu", "elapsed_time": 4.21, "total_throughput": 6912.4,
    "per_model_processed": {"ew_classifier": 10000, "resnet18": 500, ...}
  },
  "hybrid_cpu_gpu": {
    "mode": "hybrid_cpu_gpu", "gpu_models": ["yolov8_small", "resnet18", ...],
    "cpu_models": ["signal_denoiser", "anomaly_detector"], ...
  },
  "distributed_gpu": {
    "mode": "distributed_gpu", "num_partitions": 4, "total_throughput": 5100.2, ...
  },
  "total_benchmark_time": 38.7
}
```

### `results/metrics_report.md` (human-readable)

Generated by `generate_report()`:
1. **System** table (platform, GPU, CUDA)
2. **Models (10)** table (name, category, memory)
3. **Inference Mode Comparison** table (throughput, time, GPU/CPU split per mode)
4. **Per-Model Processing** table (samples handled by each model, per mode)
5. **Recommendations** (when to use which mode)

This is the file you hand to a reviewer or include in a DRDO deliverable —
regenerate it any time with `python benchmark/run_benchmark.py`.
