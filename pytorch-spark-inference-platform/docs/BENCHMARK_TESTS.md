# Benchmark Test Phases — Complete Reference

This document describes each benchmark phase executed by `run_benchmarks_cloud.ps1` (and the local `run_all_cpu_tests.ps1`).

---

## Phase 1: Mode Comparison

**Purpose:** Compare the 3 inference execution modes at different data scales.

| Mode | Description |
|------|-------------|
| `single_gpu` | All 10 models loaded on a single GPU, inference via CUDA streams in parallel |
| `hybrid` | Memory-aware placement — models split between GPU and CPU based on VRAM budget |
| `distributed` | Spark RDD-based distribution — data is partitioned across cluster executors |

**Tests:**
- 1.1: All modes, small load (1K signals, 20 images, 5 detections)
- 1.2: All modes, medium load (5K signals, 50 images, 10 detections)
- 1.3: Distributed only, large load (10K signals, 100 images, 20 detections)

**What it measures:** Throughput (samples/sec), elapsed time, per-model processing counts.

**Key insight:** At small data sizes, single GPU wins due to zero distribution overhead. As data grows, distributed mode scales better by parallelizing across workers.

---

## Phase 2: Partition Scaling

**Purpose:** Measure how Spark partition count affects distributed inference throughput.

**Tests:** Partitions = 2, 4, 6, 8, 12, 16 (fixed data: 5K signals, 50 images, 10 detections)

**What it measures:**
- Throughput vs. partition count
- Task scheduling overhead
- Optimal partition-to-worker ratio

**Key insight:** More partitions = more parallelism, but beyond (cores × 2) you hit diminishing returns from scheduling overhead and smaller per-partition data chunks.

---

## Phase 3: Data Size Scaling

**Purpose:** Observe how throughput changes as data volume increases.

**Tests:**

| Label | Signals | Images | Detections | Partitions |
|-------|---------|--------|------------|------------|
| Tiny | 500 | 10 | 5 | 4 |
| Small | 1,000 | 20 | 5 | 4 |
| Medium | 5,000 | 50 | 10 | 4 |
| Large | 8,000 | 50 | 10 | 8 |
| XLarge | 10,000 | 80 | 15 | 8 |

**What it measures:**
- Throughput scaling with data volume
- Memory pressure effects
- Point where distributed mode amortizes model load overhead

**Key insight:** Distributed mode has fixed overhead (model loading on each executor). Larger datasets amortize this cost, making throughput asymptotically approach the theoretical max.

---

## Phase 4: Batch Size Impact

**Purpose:** Find the optimal batch size for inference throughput.

**Tests:** Batch sizes = 16, 32, 64, 128, 256, 512 (fixed: 5K signals, 50 images, 10 detections, 4 partitions)

**What it measures:**
- Throughput vs. batch size
- GPU utilization (larger batches = better GPU saturation)
- Memory vs. speed tradeoff

**Key insight:** Larger batches generally improve GPU throughput (better tensor parallelism) up to a point where memory becomes the bottleneck. On CPU, the sweet spot is typically 64-128.

---

## Phase 5: Worker Scaling

**Purpose:** Measure horizontal scaling — does adding more Spark workers improve throughput linearly?

**Tests:** Workers = 1, 2, 3, 4, 6 (each with proportional partitions: workers × 2)

**Method:** Dynamically starts/stops Docker worker containers on the master node, waits 15s for registration, then runs the benchmark.

**What it measures:**
- Throughput vs. worker count
- Linear scaling efficiency
- Spark scheduling overhead per worker

**Key insight:** Ideal is linear scaling (2× workers = 2× throughput). In practice you see sub-linear scaling due to network overhead, model load duplication on each executor, and data serialization costs.

---

## Phase 6: Cluster Benchmark — CPU Modes

**Purpose:** Test the `cluster_benchmark.py` engine with CPU-only device mode at varying partition counts.

**Tests:**
- cpu_only, 2 partitions, 3000 signals
- cpu_only, 4 partitions, 3000 signals
- cpu_only, 8 partitions, 3000 signals

**What it measures:**
- Per-executor metrics (hostname, device, model load time, inference time)
- Per-partition task timing
- Spark REST API executor metrics (cores, tasks completed, GC time, memory)

**Output format:** Detailed table showing each partition's executor assignment, device, timing breakdown.

---

## Phase 7: GPU Benchmark Tests

**Purpose:** Test GPU-accelerated inference using `gpu_only` and `hybrid` device modes.

**Tests:**

| Test | Mode | Partitions | Signals | Images | Detections |
|------|------|-----------|---------|--------|------------|
| 7.1 | gpu_only | 2 | 1,000 | 50 | 20 |
| 7.2 | gpu_only | 4 | 3,000 | 100 | 30 |
| 7.3 | gpu_only | 4 | 5,000 | 200 | 50 |
| 7.4 | hybrid | 2 | 1,000 | 50 | 20 |
| 7.5 | hybrid | 4 | 3,000 | 100 | 30 |
| 7.6 | hybrid | 8 | 5,000 | 200 | 50 |

**Device modes:**
- `gpu_only` — Forces all executors to use CUDA. Fails gracefully on CPU-only workers.
- `hybrid` — Auto-detects: executors with GPU use CUDA, others use CPU. Best for mixed clusters.

**Key insight:** GPU mode shows 5-20× throughput improvement for CNN models (image/detection). Signal models (small FC networks) see less benefit since they're already fast on CPU.

---

## Phase 8: GPU Batch Size Scaling

**Purpose:** Find optimal batch size specifically for GPU inference.

**Tests:** Batch sizes = 32, 64, 128, 256, 512 (gpu_only mode, 4 partitions, 3K signals)

**Key insight:** GPUs benefit more from larger batches than CPUs due to CUDA kernel launch overhead amortization. T4 GPUs typically peak at batch size 128-256 for these model sizes.

---

## Phase 8b: Hybrid Worker Scaling

**Purpose:** Test how adding CPU workers alongside a GPU worker affects hybrid mode throughput.

**Tests:** 1 GPU worker + 0, 1, 2, 4 CPU workers (hybrid mode)

**What it measures:**
- Whether CPU workers meaningfully contribute when a GPU is present
- Optimal CPU:GPU worker ratio
- Task distribution fairness between fast (GPU) and slow (CPU) executors

**Key insight:** In hybrid mode, Spark distributes partitions evenly — but GPU executors finish faster. Adding too many CPU workers can actually slow down the overall job if Spark waits for the slowest task.

---

## Phase 9: Incremental Load Test

**Purpose:** Progressively increase data size to find the scaling curve.

**Tests:** 5 runs with increasing data:
1. 500 signals, 20 images, 10 detections (2 partitions)
2. 1,000 signals, 50 images, 20 detections (2 partitions)
3. 2,000 signals, 100 images, 30 detections (2 partitions)
4. 5,000 signals, 200 images, 50 detections (4 partitions)
5. 10,000 signals, 400 images, 80 detections (4 partitions)

**What it measures:**
- Throughput curve as data grows
- Spark UI job/stage/executor metrics captured via REST API
- Point where setup time becomes negligible vs inference time

---

## Phase 10: Full Incremental (All Modes × 3 Loads)

**Purpose:** Comprehensive comparison of all 3 device modes across 3 load levels.

**Tests:** 9 runs total (gpu_only × 3 + cpu_only × 3 + hybrid × 3):

| Load | Partitions | Signals | Images | Detections |
|------|-----------|---------|--------|------------|
| Small | 2 | 1,000 | 50 | 20 |
| Medium | 2 | 3,000 | 100 | 30 |
| Large | 4 | 5,000 | 200 | 50 |

**Output:** Summary comparison table showing throughput, time, and device used per run. Saved as `incremental_all_modes_<timestamp>.json`.

---

## Data Types Processed

All benchmarks process 3 types of synthetic sensor data through 10 ML models simultaneously:

| Data Type | Shape | Models | Use Case |
|-----------|-------|--------|----------|
| EW Signals | (N, 128) float32 | 5 signal classifiers (FC networks) | Electronic warfare signal identification |
| Images | (N, 3, 224, 224) float32 | 3 image classifiers (ResNet18, MobileNetV3, EfficientNet-B0) | Platform/vehicle recognition |
| Detection Images | (N, 3, 640, 640) float32 | 2 object detectors (YOLOv8-based) | Threat detection in imagery |

---

## Output Files

Each benchmark run produces:
- `results_<run_name>_<mode>_<config>_<timestamp>.json` — Raw metrics, per-model breakdown, executor details
- `report_<run_name>_<mode>_<config>_<timestamp>.md` — Human-readable markdown report
- `cluster_benchmark_<mode>_<signals>_<timestamp>.json` — Cluster benchmark specific results
- `incremental_results.json` — Incremental load test accumulator
