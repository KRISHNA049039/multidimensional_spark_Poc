# Architecture & Code Flow
## Distributed Multi-Model Inference Platform

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT / DRIVER MACHINE                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  benchmark/cluster_benchmark.py  or  benchmark/run_benchmark.py      │    │
│  │    → Loads 10 model classes                                          │    │
│  │    → Generates synthetic EW data (signals, images, detections)       │    │
│  │    → Creates SparkSession (connects to master)                       │    │
│  │    → Submits distributed inference job                               │    │
│  └──────────────────────────┬───────────────────────────────────────────┘    │
└─────────────────────────────┼────────────────────────────────────────────────┘
                              │ SparkSession.master("spark://master:7077")
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SPARK MASTER (port 7077)                              │
│                                                                              │
│  ┌────────────────────────────────────────────────────────┐                  │
│  │  DAG Scheduler → Task Scheduler → Cluster Manager       │                  │
│  │    • Receives job: parallelize(partition_data, N)        │                  │
│  │    • Creates Stage 0 with N tasks                        │                  │
│  │    • Assigns tasks to executors (data-local preferred)   │                  │
│  └────────────────────────────┬───────────────────────────┘                  │
│                               │                                              │
└───────────────────────────────┼──────────────────────────────────────────────┘
          ┌─────────────────────┼─────────────────────────┐
          ▼                     ▼                         ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   EXECUTOR 0     │  │   EXECUTOR 1     │  │   EXECUTOR N     │
│   (Worker 1)     │  │   (Worker 1)     │  │   (Worker 2)     │
│                  │  │                  │  │                  │
│  Partition 0     │  │  Partition 1     │  │  Partition N     │
│  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │
│  │ Load 10    │  │  │  │ Load 10    │  │  │  │ Load 10    │  │
│  │ models →   │  │  │  │ models →   │  │  │  │ models →   │  │
│  │ Infer →    │  │  │  │ Infer →    │  │  │  │ Infer →    │  │
│  │ Return     │  │  │  │ Return     │  │  │  │ Return     │  │
│  └────────────┘  │  │  └────────────┘  │  │  └────────────┘  │
│  Device: cuda/cpu│  │  Device: cuda/cpu│  │  Device: cuda/cpu│
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                     │                         │
          └─────────────────────┼─────────────────────────┘
                                ▼
                    ┌────────────────────┐
                    │  .collect()        │
                    │  Results → Driver  │
                    └────────────────────┘
```

---

## 2. Detailed Code Flow (Job Submission to Completion)

### Phase A: Driver Setup (runs once)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Load Model Registry                                          │
│     models/model_registry.py → get_default_registry()            │
│     Returns: 10 model instances (on CPU, uninitialized)          │
│                                                                  │
│  2. Generate Synthetic Data                                      │
│     data/image_generator.py → generate_mixed_data()              │
│     Returns: {                                                   │
│       "ew_classifier": ndarray(5000, 128),      # IQ signals     │
│       "signal_denoiser": ndarray(5000, 128),                     │
│       "resnet18": ndarray(200, 3, 224, 224),    # images         │
│       "yolov8_nano": ndarray(50, 3, 640, 640),  # detections     │
│       ...                                                        │
│     }                                                            │
│                                                                  │
│  3. Create SparkSession                                          │
│     inference/cluster_engine.py → create_cluster_session()       │
│     Connects to: spark://master-ip:7077                          │
│     Config: executor.memory=4g, executor.cores=4, task.cpus=2    │
└─────────────────────────────────────────────────────────────────┘
```

### Phase B: Data Preparation & Broadcast

```
┌─────────────────────────────────────────────────────────────────┐
│  4. Serialize Model Weights                                      │
│     For each of 10 models:                                       │
│       model.state_dict() → torch.save() → bytes                  │
│     Total: ~75 MB (all models combined)                          │
│                                                                  │
│  5. Broadcast Model Bytes                                        │
│     sc.broadcast(model_bytes_map)                                 │
│     → Spark sends 75MB once to each worker node                  │
│     → Workers cache it in memory (not per-task)                  │
│                                                                  │
│  6. Partition Data into N Chunks                                 │
│     For N=4 partitions, 5000 signals:                            │
│       Partition 0: signals[0:1250], images[0:50], dets[0:12]     │
│       Partition 1: signals[1250:2500], images[50:100], ...       │
│       Partition 2: signals[2500:3750], ...                       │
│       Partition 3: signals[3750:5000], ...                       │
│                                                                  │
│  7. Create RDD                                                   │
│     sc.parallelize(partition_data, num_partitions=N)              │
│     Each element: (partition_idx, {model_name: numpy_chunk})      │
└─────────────────────────────────────────────────────────────────┘
```

### Phase C: Distributed Execution (runs on each executor in parallel)

```
┌─────────────────────────────────────────────────────────────────┐
│  data_rdd.map(infer_on_partition).collect()                      │
│                                                                  │
│  For EACH partition (runs in parallel across executors):          │
│  ════════════════════════════════════════════════════════         │
│                                                                  │
│  8. Device Detection (per executor)                              │
│     if torch.cuda.is_available():                                │
│       device = "cuda"    # GPU worker                            │
│       gpu_name = torch.cuda.get_device_name(0)  # "Tesla T4"    │
│     else:                                                        │
│       device = "cpu"     # CPU worker                            │
│                                                                  │
│  9. Deserialize Models (1.2 sec, cached per executor)            │
│     model_bytes = broadcast_variable.value                       │
│     For each of 10 models:                                       │
│       buf = BytesIO(model_bytes[name])                           │
│       model = ModelClass()                                       │
│       model.load_state_dict(torch.load(buf))                     │
│       model.to(device)   # Move to GPU if available              │
│       model.eval()       # Set to inference mode                 │
│                                                                  │
│  10. Run Inference (per model, batched)                          │
│      with torch.no_grad():                                       │
│        for batch in chunks(data, batch_size=256):                │
│          tensor = torch.from_numpy(batch).float().to(device)     │
│          output = model(tensor)                                  │
│                                                                  │
│  11. Return Results                                              │
│      {                                                           │
│        "partition_idx": 0,                                       │
│        "executor": {"hostname", "device", "gpu_name"},           │
│        "timing": {"model_load": 1.2s, "inference": 4.1s},       │
│        "per_model": {model: {samples, throughput, shapes}}       │
│      }                                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Phase D: Result Collection & Aggregation

```
┌─────────────────────────────────────────────────────────────────┐
│  12. .collect() — Spark gathers all partition results            │
│      Driver receives: List[partition_result_dict]                 │
│                                                                  │
│  13. Aggregate Metrics                                           │
│      total_throughput = total_samples / elapsed_time             │
│      per_model_totals = sum across all partitions                │
│                                                                  │
│  14. Capture Spark UI Stats (REST API on port 4040)              │
│      /api/v1/applications/{id}/executors                         │
│      /api/v1/applications/{id}/jobs                              │
│      /api/v1/applications/{id}/stages                            │
│                                                                  │
│  15. Save Results                                                │
│      → results/cluster_benchmark_{mode}_{signals}_{ts}.json      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Partitioning Strategy

```
INPUT DATA (Driver)
═══════════════════════════════════════════════════════════════════

Signal Data:   [████████████████████████████████████████] 5,000 × 128-dim
Image Data:    [████████] 200 × 3×224×224
Detection Data:[███] 50 × 3×640×640

                    ┌─── SPLIT BY num_partitions = 4 ───┐
                    ▼                                    ▼

PARTITION 0              PARTITION 1              PARTITION 2              PARTITION 3
signals[0:1250]          signals[1250:2500]       signals[2500:3750]       signals[3750:5000]
images[0:50]             images[50:100]           images[100:150]          images[150:200]
dets[0:12]               dets[12:25]              dets[25:37]              dets[37:50]

Each partition: ~22 MB    Each partition: ~22 MB   Each partition: ~22 MB   Each partition: ~22 MB
───────────────────────────────────────────────────────────────────────────────────────────────

DATA FLOW PER PARTITION:
  numpy array → sc.parallelize → executor JVM → Python worker → torch.from_numpy → model(tensor)
```

---

## 4. Executor-Level Execution Detail

```
┌─────────────── EXECUTOR (Python Worker Process) ───────────────────┐
│                                                                      │
│  ┌─── Model Loading (1.2 sec, ONE TIME per executor) ────────────┐  │
│  │                                                                │  │
│  │  broadcast.value → 75 MB model bytes                           │  │
│  │                                                                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │  │
│  │  │ew_class  │ │sig_denoi │ │threat_pri│ │rf_finger │  ...×10  │  │
│  │  │ 50 MB    │ │ 100 MB   │ │ 350 MB   │ │ 120 MB   │         │  │
│  │  │ →cuda/cpu│ │ →cuda/cpu│ │ →cuda/cpu│ │ →cuda/cpu│         │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─── Inference Loop (per model, sequential) ────────────────────┐  │
│  │                                                                │  │
│  │  for model_name, model in loaded_models:                       │  │
│  │    data_chunk = partition_data[model_name]  # numpy slice      │  │
│  │                                                                │  │
│  │    with torch.no_grad():                                       │  │
│  │      for batch_start in range(0, N, batch_size):               │  │
│  │        ┌──────────────────────────────────────────────┐        │  │
│  │        │ numpy[start:end] → torch.tensor → .to(device)│        │  │
│  │        │ output = model(tensor)  # forward pass        │        │  │
│  │        │ GPU: CUDA kernel launch + compute             │        │  │
│  │        │ CPU: sequential matrix multiply               │        │  │
│  │        └──────────────────────────────────────────────┘        │  │
│  │                                                                │  │
│  │  Record: samples_processed, inference_time, throughput         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Return: partition_result → Spark → Driver                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Device Mode Routing

```
┌─────────────────────────────────────────────────────────────────┐
│  DEVICE MODE SELECTION (per executor)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  mode = "cpu_only"                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CUDA_VISIBLE_DEVICES='' → torch.cuda.is_available()=False │   │
│  │ ALL 10 models run on CPU                                  │   │
│  │ Signal models: 74K-175K/sec ✓ (fast)                      │   │
│  │ CNN models: 14-21/sec ✗ (slow)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  mode = "gpu_only"                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ NVIDIA_VISIBLE_DEVICES=all → torch.cuda.is_available()=True│   │
│  │ ALL 10 models loaded on GPU (T4 16GB VRAM)                │   │
│  │ Signal models: ~74K/sec (no speedup, kernel overhead)     │   │
│  │ CNN models: 150-400/sec ✓ (10-20× speedup)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  mode = "hybrid"                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Auto-detect per executor:                                 │   │
│  │   if cuda available → use cuda (for all models on that    │   │
│  │   executor)                                               │   │
│  │   else → use cpu                                          │   │
│  │ In multi-node: GPU workers get cuda, CPU workers get cpu  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Optimizations Implemented

### 6.1 Model Weight Broadcasting

```
PROBLEM: 10 models × 4 partitions = 40 model loads from disk?
SOLUTION: Broadcast once, deserialize per-executor

Driver                          Worker Node
┌──────────┐    broadcast      ┌──────────────────┐
│ Serialize │ ──── 75 MB ────→ │ Cached in memory  │
│ 10 models │    (one time)    │                   │
│ to bytes  │                  │ Executor 0: deser │
└──────────┘                   │ Executor 1: deser │ (from RAM, not network)
                               └──────────────────┘

Savings: 75 MB × (N-1) network transfer avoided
```

### 6.2 Batch Inference with torch.no_grad()

```
WITHOUT optimization:           WITH optimization:
for sample in data:             with torch.no_grad():
  output = model(sample)          for batch in chunks(data, 256):
  # N forward passes                output = model(batch)
  # N grad graph allocations         # N/256 forward passes
  # O(N) memory                      # Zero grad memory
                                     # Vectorized on GPU

Savings: 256× fewer kernel launches, 0 gradient memory
```

### 6.3 Data Embedding in RDD (vs File-Based)

```
APPROACH A (file-based):        APPROACH B (embedded, used):
  Save to /tmp/part_0.npy       sc.parallelize([(0, numpy_chunk), ...])
  Save paths to RDD             → Data lives IN the RDD element
  Worker: np.load(path)         → No disk I/O on executor
  ✗ Requires shared filesystem  → spark.rpc.message.maxSize=512MB
                                ✓ Works on any cluster

Trade-off: Higher task serialization cost, but simpler + no NFS needed
```

### 6.4 Executor Memory Tuning

```
PROBLEM: 10 models = 2.1 GB total → OOM with 2 GB executor memory
SOLUTION: executor.memory=4g + task.cpus=2

Per Executor (4 GB):
┌──────────────────────────────────────────────────────────┐
│  JVM Heap: 1.5 GB (Spark internals)                       │
│  Python Worker: 2.5 GB available                          │
│    └── 10 model weights: 2.1 GB                           │
│    └── Batch tensors: ~100 MB                             │
│    └── Python overhead: ~300 MB                           │
└──────────────────────────────────────────────────────────┘

task.cpus=2 ensures only 1 task per executor
(avoids 2 tasks each loading 2.1 GB = OOM)
```

### 6.5 Model Caching Across Tasks (spark.python.worker.reuse=true)

```
WITHOUT reuse:                  WITH reuse:
Task 1 → Load 10 models        Task 1 → Load 10 models (1.2 sec)
Task 2 → Load 10 models        Task 2 → Models already in memory (0 sec)
Task 3 → Load 10 models        Task 3 → Models already in memory (0 sec)
                                ← Same Python process handles multiple tasks

Savings: (N_tasks - 1) × 1.2 sec model load time per executor
Requires: spark.python.worker.reuse=true (enabled)
```

---

## 7. Cluster Topology

```
┌─────────────────────── DOCKER HOST (EC2 g4dn.xlarge) ──────────────────────┐
│                                                                              │
│  ┌─── spark-master container ────────────────────────────────────────────┐  │
│  │  Spark Master (port 7077)                                              │  │
│  │  Spark Web UI (port 8080)                                              │  │
│  │  Driver Process (submits jobs here)                                    │  │
│  │  Spark App UI (port 4040) — per-job metrics                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─── spark-gpu-worker container ────────────────────────────────────────┐  │
│  │  Spark Worker (4 cores, 12 GB, registers with master)                  │  │
│  │  NVIDIA T4 GPU (16 GB VRAM) — --gpus all                              │  │
│  │  Executors spawned here by Spark (Python processes)                    │  │
│  │  Each executor: loads 10 models → runs inference → returns results     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Shared: --network host (all containers on same network namespace)           │
│  Volume: /opt/benchmark/app/results:/app/results (results persist on host)   │
│  GPU:    --gpus all --shm-size=4g (full T4 access)                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Performance Bottleneck Analysis

```
TIME BREAKDOWN (typical 5K signal, 4 partition run):

Total elapsed: ~12 sec
├── Spark job overhead (scheduling, serialization): 0.5 sec (4%)
├── Data transfer (driver → executors via RDD): 0.3 sec (2.5%)
├── Model deserialization (per executor, from broadcast): 1.2 sec (10%)
├── Inference (all 10 models × data chunk):
│   ├── Signal models (5): 0.7 sec total (6%)
│   ├── Image models (3): 7.5 sec total (62%) ← BOTTLENECK
│   └── Detection models (2): 3.0 sec total (25%)
└── Result collection (.collect()): 0.1 sec (1%)

GPU IMPACT (on CNN models only):
  ResNet18:      CPU 21/sec → GPU 200/sec  = 10× speedup
  EfficientNet:  CPU 25/sec → GPU 250/sec  = 10× speedup
  YOLOv8:        CPU 14/sec → GPU 150/sec  = 10× speedup
  Signals:       CPU 74K/sec → GPU 74K/sec = 1× (no benefit)
```

---

## 9. File Structure & Module Responsibilities

```
pytorch-spark-inference-platform/
├── benchmark/
│   ├── run_benchmark.py          # Unified benchmark (all 3 modes)
│   ├── cluster_benchmark.py      # Cluster-specific (gpu_only/cpu_only/hybrid)
│   └── incremental_load_test.py  # Progressive scaling test
│
├── inference/
│   ├── distributed_gpu.py        # Core: Spark RDD + model broadcast + GPU inference
│   ├── cluster_engine.py         # SparkSession factory + cluster inference wrapper
│   ├── single_gpu.py             # Non-distributed: all models on 1 GPU
│   └── hybrid_cpu_gpu.py         # Memory-aware CPU/GPU split (non-distributed)
│
├── models/
│   ├── model_registry.py         # Registry: lists all 10 models with metadata
│   ├── ew_signal_model.py        # EW classifier (FC network, 128→8)
│   ├── signal_models.py          # Denoiser, Prioritizer, Fingerprinter, Anomaly
│   ├── image_models.py           # ResNet18, MobileNetV3, EfficientNet-B0
│   └── yolo_model.py             # YOLOv8 nano + small wrappers
│
├── data/
│   └── image_generator.py        # Synthetic data generator (signals + images)
│
├── deploy/
│   ├── Dockerfile                # nvidia/cuda base + Spark + PyTorch + models
│   ├── docker-compose.cluster.yml# Multi-container cluster (master + workers)
│   ├── aws-cdk/                  # CDK infrastructure (EC2 + S3 + IAM)
│   └── scripts/                  # Setup/benchmark shell scripts for EC2
│
└── results/                      # Benchmark outputs (JSON + reports)
```

---

## 10. Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Broadcast model weights (not data) | Models are same for all tasks; data differs per partition | 75 MB broadcast vs per-task serialization |
| Embed data in RDD elements | No shared filesystem needed (works on any EC2 cluster) | Higher task serialization cost |
| Sequential model execution (per executor) | Avoids GPU memory fragmentation from parallel model loads | Leaves GPU idle between models |
| Single Python worker per executor (task.cpus=2) | 10 models need 2.1 GB → can't fit 2 concurrent tasks | Reduces parallelism but prevents OOM |
| torch.no_grad() everywhere | Inference only — no backprop needed | Cannot do fine-tuning in this path |
| batch_size=256 | Balances GPU utilization vs memory for largest models (YOLO 640×640) | Smaller batches would underutilize GPU |


---

## 11. Model Predictions — What Each Model Does

### Signal Processing Pipeline (5 models, input: 128-dim IQ vector)

```
              ┌──────────── RAW EW SIGNAL ────────────┐
              │  128 IQ samples (64 In-phase +         │
              │  64 Quadrature channel values)          │
              │  Example: [0.28, 0.54, 0.45, -0.30,    │
              │            -1.70, 0.02, 0.65, ...]     │
              └────────────────┬───────────────────────┘
                               │
          ┌────────────────────┼────────────────────┬──────────────────────┐
          ▼                    ▼                    ▼                      ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  EW CLASSIFIER   │ │ SIGNAL DENOISER  │ │THREAT PRIORITIZER│ │  RF FINGERPRINTER │
│                  │ │                  │ │                  │ │                  │
│ Architecture:    │ │ Architecture:    │ │ Architecture:    │ │ Architecture:    │
│  FC Network      │ │  Autoencoder     │ │  Multi-head      │ │  1D CNN          │
│  128→256→512→    │ │  128→256→128→32  │ │  Attention       │ │  Conv1d layers   │
│  256→128→64→8    │ │  →128→256→128    │ │  128→512→attn    │ │  128→32→64→128   │
│                  │ │                  │ │  →256→128→1      │ │  →256→32         │
│ Output: 8 logits │ │ Output: 128 vals │ │ Output: 1 score  │ │ Output: 32-dim   │
│ (class scores)   │ │ (clean signal)   │ │ (0-1 priority)   │ │ (embedding)      │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ "Pulsed_Radar"   │ │ Clean signal for │ │ Priority: 0.53   │ │ Embedding vector │
│  confidence: 85% │ │ downstream use   │ │ (medium threat)  │ │ for DB matching  │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘

                               │
                               ▼
                    ┌──────────────────┐
                    │ ANOMALY DETECTOR │
                    │                  │
                    │ Architecture:    │
                    │  VAE (Variational│
                    │  Autoencoder)    │
                    │  128→256→128→    │
                    │  μ,σ→z→128→     │
                    │  256→128         │
                    │                  │
                    │ Output: scalar   │
                    │ (recon. error)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Score: 130.07    │
                    │ (high = unknown  │
                    │  signal type)    │
                    └──────────────────┘
```

### Image Classification Pipeline (3 models, input: 3×224×224 RGB)

```
              ┌──────────── RGB IMAGE ────────────────┐
              │  3 channels × 224 × 224 pixels         │
              │  Normalized [0,1] range                 │
              │  Example: surveillance camera frame     │
              └────────────────┬───────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│    RESNET-18     │ │  MOBILENET-V3    │ │ EFFICIENTNET-B0  │
│                  │ │                  │ │                  │
│ 11.7M params    │ │ 5.4M params      │ │ 5.3M params      │
│ 300 MB GPU      │ │ 150 MB GPU       │ │ 200 MB GPU       │
│                  │ │                  │ │                  │
│ Deep residual   │ │ Inverted         │ │ Compound-scaled  │
│ blocks (18      │ │ residuals +      │ │ CNN with squeeze │
│ layers)         │ │ squeeze-excite   │ │ -and-excitation  │
│                  │ │                  │ │                  │
│ Pretrained:     │ │ Pretrained:      │ │ Pretrained:      │
│ ImageNet 1000   │ │ ImageNet 1000    │ │ ImageNet 1000    │
│                  │ │                  │ │                  │
│ Output: 1000    │ │ Output: 1000     │ │ Output: 1000     │
│ class logits    │ │ class logits     │ │ class logits     │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│ Top predictions (after softmax):                              │
│   Class 817: "military vehicle" — 92% confidence              │
│   Class 424: "pickup truck"     — 4% confidence               │
│   Class 656: "van"              — 2% confidence               │
│                                                               │
│ Use case: Identify platforms/vehicles in surveillance imagery  │
└──────────────────────────────────────────────────────────────┘
```

### Object Detection Pipeline (2 models, input: 3×640×640 RGB)

```
              ┌──────────── HIGH-RES IMAGE ───────────┐
              │  3 channels × 640 × 640 pixels         │
              │  Full surveillance frame                │
              └────────────────┬───────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐  ┌──────────────────┐
          │  YOLOV8-NANO     │  │  YOLOV8-SMALL    │
          │                  │  │                  │
          │ 3.2M params      │  │ 11.2M params     │
          │ 200 MB GPU       │  │ 400 MB GPU       │
          │                  │  │                  │
          │ Fastest variant  │  │ Better accuracy  │
          │ 640→320→160→80   │  │ 640→320→160→80   │
          │ Multi-scale      │  │ Multi-scale      │
          │ feature pyramid  │  │ feature pyramid  │
          │                  │  │                  │
          │ Output: up to    │  │ Output: up to    │
          │ 400 detections   │  │ 400 detections   │
          └────────┬─────────┘  └────────┬─────────┘
                   │                     │
                   ▼                     ▼
          ┌────────────────────────────────────────┐
          │ Per detection:                          │
          │   [x_center, y_center, width, height,   │
          │    confidence, class_id]                 │
          │                                         │
          │ Example output:                         │
          │   Box 1: [320, 240, 80, 60, 0.94, 0]   │
          │          → Vehicle at center, 94% conf  │
          │   Box 2: [100, 400, 40, 30, 0.87, 1]   │
          │          → Person at bottom-left, 87%   │
          │   Box 3: [500, 100, 120, 90, 0.72, 2]  │
          │          → Building at top-right, 72%   │
          │                                         │
          │ Use case: Detect and localize threats    │
          │ in real-time imagery                    │
          └────────────────────────────────────────┘
```

---

## 12. EW Signal Classes (ew_classifier predictions)

| Class ID | Signal Type | Description |
|----------|-------------|-------------|
| 0 | CW_Radar | Continuous wave radar (tracking/illumination) |
| 1 | Pulsed_Radar | Traditional pulse radar (search/surveillance) |
| 2 | FMCW_Radar | Frequency-modulated continuous wave (automotive/altimeter) |
| 3 | Phase_Coded_Radar | Phase-coded pulse radar (military LPI) |
| 4 | Noise_Jammer | Broadband noise jamming signal |
| 5 | Spot_Jammer | Narrowband spot jamming (targeted) |
| 6 | Sweep_Jammer | Swept frequency jammer (barrage) |
| 7 | Comm_Signal | Communications signal (friend/foe) |

---

## 13. End-to-End EW Processing Scenario

```
REAL-WORLD SCENARIO: A sensor array intercepts 5,000 unknown signals

Step 1: Raw IQ data collected (5000 × 128 values each)
        └── Fed into Spark cluster as input batch

Step 2: PARALLEL inference on ALL 5 signal models simultaneously:
        ├── ew_classifier:      "What TYPE of signal is this?"
        │                       → "Pulsed_Radar" (85% confidence)
        │
        ├── signal_denoiser:    "Clean this signal for analysis"
        │                       → 128 denoised IQ values
        │
        ├── threat_prioritizer: "How urgent is this threat?"
        │                       → Priority score 0.87 (HIGH)
        │
        ├── rf_fingerprinter:   "Which specific emitter?"
        │                       → 32-dim vector → DB lookup → "SA-11 Gadfly"
        │
        └── anomaly_detector:   "Is this something we've never seen?"
                                → Score 12.3 (normal) or 450.7 (ANOMALY!)

Step 3: PARALLEL inference on image/detection models:
        ├── resnet18:           "What's in this surveillance image?"
        │                       → "military_vehicle" (92%)
        │
        ├── yolov8_small:       "Where are the threats in frame?"
        │                       → 3 bounding boxes with classes
        │
        └── Combined:           Full situational awareness picture

Step 4: Results aggregated → threat assessment → operator display
        Total time: ~12 seconds for 25,700 inferences across 10 models
        Throughput: 2,083 samples/sec (with GPU)
```

---

## 14. Model Parameter Summary

| Model | Parameters | Memory (GPU) | Architecture | Speed (GPU) | Speed (CPU) |
|-------|-----------|-------------|--------------|-------------|-------------|
| ew_classifier | ~500K | 50 MB | FC 5-layer | 74K/s | 74K/s |
| signal_denoiser | ~1.2M | 100 MB | Autoencoder | 175K/s | 175K/s |
| threat_prioritizer | ~8M | 350 MB | Multi-head Attn | 29K/s | 29K/s |
| rf_fingerprinter | ~2.5M | 120 MB | 1D-CNN | 4.7K/s | 4.7K/s |
| anomaly_detector | ~1.8M | 100 MB | VAE | 150K/s | 150K/s |
| resnet18 | 11.7M | 300 MB | ResNet (pretrained) | ~200/s | 21/s |
| mobilenetv3 | 5.4M | 150 MB | Inverted Residual | ~1000/s | 170/s |
| efficientnet_b0 | 5.3M | 200 MB | Compound Scaling | ~250/s | 25/s |
| yolov8_nano | 3.2M | 200 MB | YOLO Backbone | ~150/s | 45/s |
| yolov8_small | 11.2M | 400 MB | YOLO Backbone | ~100/s | 14/s |
| **TOTAL** | **~51M** | **~2.1 GB** | — | — | — |
