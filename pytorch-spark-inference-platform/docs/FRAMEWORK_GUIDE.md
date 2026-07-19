# PyTorch-Spark Inference Platform — Framework Guide

**For AIA Team Use**

This document covers the complete code flow, architecture decisions, data distribution best practices, and how to extend this framework with your own models and data.

---

## 1. Repository Structure

```
pytorch-spark-inference-platform/
├── models/                    # Model definitions + registry
│   ├── __init__.py            # get_default_registry() — entry point
│   ├── model_registry.py      # Central ModelRegistry class
│   ├── ew_signal_model.py     # EW signal classifier (MLP)
│   ├── signal_models.py       # Denoiser, Prioritizer, Fingerprinter, Anomaly
│   ├── image_models.py        # ResNet18, MobileNetV3, EfficientNet-B0
│   └── yolo_model.py          # YOLOv8 Nano/Small (with fallback)
├── inference/                 # 3 inference modes + support engines
│   ├── single_gpu.py          # Mode 2: All models on 1 GPU (CUDA streams)
│   ├── hybrid_cpu_gpu.py      # Mode 3: Memory-aware GPU/CPU split
│   ├── distributed_gpu.py     # Mode 1: Spark cluster distribution
│   ├── cuda_streams_engine.py # Core engine: parallel model execution
│   └── gpu_memory_manager.py  # VRAM budget planner
├── data/                      # Data generators
│   ├── signal_generator.py    # 128-dim IQ signal synthesis (8 classes)
│   └── image_generator.py     # Image + detection data + mixed generator
├── benchmark/                 # Benchmarking tools
│   ├── run_benchmark.py       # Main benchmark runner (all modes)
│   └── incremental_load_test.py # Scaling test
├── monitoring/                # Metrics publishers
│   ├── cloudwatch_publisher.py # Shared CloudWatch client
│   ├── spark_metrics_publisher.py   # Cluster + executor metrics
│   ├── gpu_metrics_publisher.py     # nvidia-smi → CloudWatch
│   └── benchmark_metrics_publisher.py # Results → CloudWatch + S3
├── deploy/                    # Deployment tooling
│   ├── Dockerfile             # Container image (CUDA + Spark + PyTorch)
│   ├── docker-compose.yml     # Local multi-container development
│   ├── deploy_to_cluster.ps1  # Automated AWS deploy script
│   ├── scripts/               # Remote setup scripts (master/worker)
│   └── aws-cdk/              # Infrastructure as Code
├── results/                   # Benchmark outputs + report generator
└── docs/                      # Documentation
```

---

## 2. Data Flow — End to End

```
┌──────────────┐     ┌──────────────┐     ┌────────────────────┐
│ Data Source   │────▶│ Model        │────▶│ Inference Engine    │
│              │     │ Registry     │     │ (Mode 1/2/3)       │
│ signal_gen   │     │              │     │                    │
│ image_gen    │     │ 10 models    │     │ CUDA Streams       │
│ (or real)    │     │ registered   │     │ + GPU Mem Manager  │
└──────────────┘     └──────────────┘     └────────┬───────────┘
                                                    │
                                                    ▼
                                           ┌────────────────┐
                                           │ Results + Metrics│
                                           │ JSON + MD + CW  │
                                           └────────────────┘
```

### Step-by-step:

1. **Model Registry** (`models/__init__.py` → `get_default_registry()`)
   - Registers 10 models with metadata (class, input shape, category, memory estimate)
   - Models are NOT loaded into memory at registration time (lazy)

2. **Load Models** (`registry.load_all("cpu")`)
   - Instantiates all 10 model classes
   - Loads pretrained weights (from cache or downloads)
   - Returns `{model_name: nn.Module}` dict on CPU

3. **Generate Data** (`data/image_generator.py` → `generate_mixed_data()`)
   - Creates numpy arrays keyed by model name
   - Signal models: (N, 128) float32
   - Image classifiers: (N, 3, 224, 224) float32
   - Object detectors: (N, 3, 640, 640) float32

4. **Run Inference** (one of 3 modes)
   - Each mode takes `models` dict + `data` dict → returns results dict

5. **Collect Results** → JSON + Markdown + CloudWatch metrics

---

## 3. The Three Inference Modes

### Mode 1: Distributed GPU (Spark)
**File:** `inference/distributed_gpu.py`
**When to use:** Multiple machines, data too large for one node

```
Driver (master):
  1. Serialize model weights → broadcast to all executors (~75MB)
  2. Partition data into N chunks → create RDD
  3. Each RDD element = (partition_idx, {model_name: data_chunk})

Executor (worker):
  1. Receive broadcast model weights
  2. Receive one data chunk via RDD
  3. Deserialize models onto local GPU (or CPU)
  4. Run inference batch-by-batch
  5. Return {model_name: num_processed}

Driver collects results → aggregates → reports
```

### Mode 2: Single GPU (CUDA Streams)
**File:** `inference/single_gpu.py`
**When to use:** One machine with GPU, all models fit in VRAM

```
1. Move all models to GPU
2. Create CUDAStreamsEngine (1 stream per model)
3. For each batch:
   - Launch all 10 models on separate streams (truly parallel)
   - GPU kernel scheduler interleaves execution
   - Synchronize
4. Collect outputs
```

### Mode 3: Hybrid CPU+GPU
**File:** `inference/hybrid_cpu_gpu.py`
**When to use:** GPU VRAM too small for all models

```
1. GPUMemoryManager plans placement (priority-based):
   - High-priority models → GPU
   - Overflow → CPU
2. GPU models: CUDAStreamsEngine (parallel on streams)
3. CPU models: ThreadPoolExecutor (parallel on threads)
4. Both groups run simultaneously
5. Merge results
```

---

## 4. Key Components — Deep Dive

### 4.1 ModelRegistry (`models/model_registry.py`)

The backbone of the system. Responsibilities:
- **Registration:** `register(name, class, input_shape, category, memory_mb)`
- **Lazy loading:** `load_model(name, device)` — instantiates only when needed
- **Serialization:** `serialize_model(name)` → bytes (for Spark broadcast)
- **Deserialization:** `deserialize_model(name, bytes, device)` → nn.Module
- **Memory planning:** `total_memory_estimate_mb()` for GPU budget decisions

**How to add a new model:**
```python
# In models/__init__.py, add to get_default_registry():
registry.register(
    name="my_new_model",
    model_class=MyNewModel,
    input_shape=(3, 512, 512),
    output_desc="segmentation mask",
    category="segmentation",
    estimated_memory_mb=800,
)
```

### 4.2 CUDAStreamsEngine (`inference/cuda_streams_engine.py`)

The core parallel execution engine used by ALL modes:
- Assigns 1 CUDA stream per GPU model
- Uses ThreadPoolExecutor for CPU models
- `infer_all_parallel(inputs)` — launches all models concurrently
- `infer_all_batched(data_batches)` — processes multiple batches

**Why CUDA streams work:**
- GPU has thousands of cores → can run multiple kernels simultaneously
- Each stream is an independent queue of GPU operations
- Different models on different streams execute concurrently
- Synchronize once after all streams finish

### 4.3 GPUMemoryManager (`inference/gpu_memory_manager.py`)

Budget-aware placement with 4 strategies:

| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| `priority` | Place highest-priority models on GPU first | Known critical models |
| `largest_first` | Place biggest models on GPU first | Max compute benefit |
| `greedy` | Fill GPU in registration order | Simple default |
| `balanced` | Spread across multiple GPUs evenly | Multi-GPU nodes |

**Decision flow:**
```
For each model (sorted by strategy):
  For each GPU (sorted by available memory):
    If model fits in remaining budget:
      Place on GPU → update budget
      break
  If not placed:
    Place on CPU
```

---

## 5. Data Distribution — Best Practices

### 5.1 What to Broadcast vs What to Partition

| Data Type | Size | Strategy | Why |
|-----------|------|----------|-----|
| Model weights | ~75 MB total | **Broadcast** | Same weights needed by ALL executors, small |
| Batch size config | 4 bytes | **Broadcast** | Tiny, same everywhere |
| Input data (signals) | 10K×128×4 = 5 MB | **RDD embed** | Small per partition |
| Input data (images) | 500×3×224×224×4 = 300 MB | **RDD embed** | Per partition, within limits |
| Input data (large, >1GB) | Variable | **RDD + more partitions** | Keep per-partition < 450MB |
| Input data (>10GB) | Very large | **Read from storage (S3/NFS)** | Don't fit in driver memory |

### 5.2 Broadcast (Good for: model weights, configs)

```python
# Driver
model_bytes = serialize_all_models()  # ~75MB
bc_model_bytes = sc.broadcast(model_bytes)

# Executor
weights = bc_model_bytes.value  # Downloaded once, cached on worker
model = deserialize(weights)
```

**Rules:**
- Use for data < 200MB that ALL executors need identically
- Broadcast happens once, cached for the entire SparkSession lifetime
- Do NOT broadcast input data (wastes memory — every executor gets ALL data)

### 5.3 RDD Partitioning (Good for: input data)

```python
# Driver: split data into per-partition chunks
partition_data = []
for i in range(num_partitions):
    chunk = data[start:end]  # Only this partition's slice
    partition_data.append((i, chunk))

data_rdd = sc.parallelize(partition_data, num_partitions)
results = data_rdd.map(infer_on_partition).collect()
```

**Rules:**
- Each executor only receives ITS partition (1/N of total data)
- Keep per-partition size < `spark.rpc.message.maxSize` (default 128MB, we use 512MB)
- More partitions = smaller messages but more scheduling overhead
- Ideal: 1-2 partitions per executor

### 5.4 Dynamic Strategy Selection

```python
total_data_bytes = sum(arr.nbytes for arr in data.values())
per_partition_bytes = total_data_bytes / num_partitions

if per_partition_bytes < 400 * 1024 * 1024:  # < 400MB
    # Embed directly in RDD — simplest, fast for small/medium data
    strategy = "rdd_embed"
elif total_data_bytes < driver_memory * 0.6:
    # Data fits in driver — use more partitions to stay under 400MB
    num_partitions = math.ceil(total_data_bytes / (400 * 1024 * 1024))
    strategy = "rdd_embed_more_partitions"
else:
    # Data too large for driver memory — read from shared storage
    # Save to NFS/S3, pass paths to executors
    strategy = "storage_backed"
```

### 5.5 When to Use Each Approach

```
Data < 500MB total → Single GPU mode (no Spark needed)
Data 500MB - 5GB   → Distributed, RDD embed, 2-8 partitions
Data 5GB - 50GB    → Distributed, RDD embed, 16-64 partitions (large driver needed)
Data > 50GB        → Distributed, storage-backed (S3/NFS/HDFS)
```

---

## 6. How to Add Your Own Model

### Step 1: Define the model class

```python
# models/my_model.py
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 128),
        )
    
    def forward(self, x):
        return self.layers(x)
```

### Step 2: Register in the registry

```python
# models/__init__.py — add to get_default_registry()
from models.my_model import MyCustomModel

registry.register(
    name="my_custom_model",
    model_class=MyCustomModel,
    input_shape=(256,),
    output_desc="128-dim embedding",
    category="signal",  # or "image_classification", "object_detection", etc.
    estimated_memory_mb=150,
)
```

### Step 3: Add data generation (or real data loading)

```python
# data/image_generator.py — add to generate_mixed_data()
return {
    ...existing models...,
    "my_custom_model": np.random.rand(num_signal_samples, 256).astype(np.float32),
}
```

### Step 4: Add to distributed executor class map

```python
# inference/distributed_gpu.py — inside infer_on_partition()
from models.my_model import MyCustomModel

class_map = {
    ...existing models...,
    "my_custom_model": MyCustomModel,
}
```

### Step 5: Run benchmark

```bash
python benchmark/run_benchmark.py --mode all
```

---

## 7. How to Use Real Data (Not Synthetic)

Replace `generate_mixed_data()` with your data loading:

```python
# data/real_data_loader.py
import numpy as np

def load_real_data(data_dir: str) -> dict:
    """Load real EW signals and imagery from disk."""
    return {
        # Signal models expect (N, 128) float32
        "ew_classifier": np.load(f"{data_dir}/ew_signals.npy"),
        "signal_denoiser": np.load(f"{data_dir}/ew_signals.npy"),
        ...
        # Image models expect (N, 3, 224, 224) float32 [0,1]
        "resnet18": np.load(f"{data_dir}/images_224.npy"),
        ...
        # Detection models expect (N, 3, 640, 640) float32 [0,1]
        "yolov8_nano": np.load(f"{data_dir}/images_640.npy"),
    }
```

---

## 8. Monitoring Architecture

```
┌────────────────────────────────────────────┐
│           CloudWatch Dashboard              │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │ Host CPU │ │ Spark    │ │ GPU       │  │
│  │ Memory   │ │ Workers  │ │ Util/Temp │  │
│  │ Disk     │ │ Tasks    │ │ Memory    │  │
│  └──────────┘ └──────────┘ └───────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │ Benchmark Throughput / Elapsed Time  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
        ▲              ▲              ▲
        │              │              │
  CloudWatch     Spark REST     nvidia-smi
  Agent          API poll       subprocess
  (host rpm)     (:8080,:4040)  (every 15s)
        │              │              │
  ┌─────┴──────┐ ┌────┴─────┐ ┌─────┴──────┐
  │ CW Agent   │ │spark_     │ │gpu_metrics_│
  │ (system)   │ │metrics_   │ │publisher.py│
  │            │ │publisher  │ │            │
  └────────────┘ └──────────┘ └────────────┘
```

**Publishers are independent daemons** — they run alongside Spark, not inside it. Each uses `CloudWatchPublisher` which gracefully degrades when offline (logs instead of crashing).

---

## 9. Best Practices Summary

### Model Management
- Register models in the central registry — don't instantiate directly
- Always provide `estimated_memory_mb` for GPU placement planning
- Use `model.eval()` and `torch.no_grad()` during inference (saves 50%+ memory)
- Serialize once, broadcast once — don't re-serialize per partition

### GPU Memory
- Reserve 500MB for CUDA context/buffers (never fill to 100%)
- Use priority-based placement — critical models get GPU first
- Monitor with `GPUMemoryManager.get_current_gpu_usage()`
- If models don't fit: Hybrid mode auto-spills to CPU

### Distributed Mode
- Broadcast model weights (small, ~75MB) — never broadcast data
- Partition data via RDD — each executor gets only its slice
- Use 1-2 partitions per executor (more = scheduling overhead)
- Set `spark.rpc.message.maxSize=512` for medium data
- For large data (>10GB): read from shared storage on executors

### Performance
- Warmup GPU before timing (first inference is slow — JIT compilation)
- Use batch sizes 128-512 for GPU (balance throughput vs latency)
- Signal models are memory-light but compute-fast — batch them together
- Image models are memory-heavy — limit concurrent loading

### Docker
- Bake model weights into the image (avoid runtime downloads)
- Use `--network host` for Spark containers (avoids port mapping complexity)
- Use `--gpus all --shm-size=8g` for GPU containers
- Use `--restart unless-stopped` for production

---

## 10. Extending for AIA Team

### Adding a new inference mode

1. Create `inference/my_new_mode.py`
2. Accept `models: Dict[str, nn.Module]` + `data: Dict[str, np.ndarray]`
3. Return a dict with `mode`, `elapsed_time`, `total_throughput`, `per_model_processed`
4. Add to `benchmark/run_benchmark.py` in `main()`

### Adding a new model category

1. Define models in `models/` with standard `nn.Module` interface
2. Register with a new category string (e.g., `"segmentation"`, `"tracking"`)
3. Add data generator in `data/` matching the input shape
4. Add to `generate_mixed_data()` keyed by model name
5. Add to `infer_on_partition()` class_map in distributed mode

### Adding a new monitoring metric

1. Create publisher in `monitoring/` following `gpu_metrics_publisher.py` pattern
2. Use `CloudWatchPublisher(namespace="SparkInference/YourMetric", node_role="...")`
3. Add to CDK dashboard widgets in `spark_cluster_stack.py`

### Adding a new deployment target

1. Write setup script in `deploy/scripts/`
2. Create Docker Compose or CDK stack as needed
3. The Docker image is deployment-agnostic — same image works everywhere

---

## 11. Quick Start (For New Team Members)

```bash
# 1. Local development (CPU only, no Docker)
pip install -r requirements.txt
python benchmark/run_benchmark.py --mode single_gpu  # falls back to CPU

# 2. Local with GPU (Docker)
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
docker run --gpus all multi-model-inference:latest

# 3. Distributed cluster (Docker)
# Master:
docker run -d --name spark-master --network host multi-model-inference:latest \
  bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
# Worker:
docker run -d --name spark-worker --network host --gpus all multi-model-inference:latest \
  bash -c "start-worker.sh spark://<MASTER-IP>:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
# Run benchmark:
docker exec -it spark-master bash -c \
  "SPARK_MASTER_URL=spark://<MASTER-IP>:7077 python benchmark/run_benchmark.py --mode distributed"
```
