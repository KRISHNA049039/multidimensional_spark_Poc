# GPU Resource Sharing & Massively Parallel Processing (MPP) Architecture

## Problem Statement

Run **7-10 PyTorch models simultaneously** on shared GPU resources across a Spark cluster, efficiently sharing GPU memory and compute without contention or OOM errors.

### Models to Run in Parallel

| # | Model | Type | Size (params) | GPU Memory | Use Case |
|---|-------|------|---------------|------------|----------|
| 1 | EW Signal Classifier | MLP | 340K | ~50 MB | Radar/jammer classification |
| 2 | YOLOv8-nano | CNN | 3.2M | ~200 MB | Real-time object detection |
| 3 | YOLOv8-small | CNN | 11.2M | ~400 MB | Higher accuracy detection |
| 4 | ResNet-18 | CNN | 11.7M | ~300 MB | Image classification |
| 5 | MobileNetV3 | CNN | 5.4M | ~150 MB | Edge image classification |
| 6 | EfficientNet-B0 | CNN | 5.3M | ~200 MB | Balanced accuracy/speed |
| 7 | Signal Denoiser | Autoencoder | 1.2M | ~100 MB | Signal preprocessing |
| 8 | Threat Prioritizer | Transformer | 8M | ~350 MB | Multi-modal fusion |
| 9 | RF Fingerprinting | 1D-CNN | 2.5M | ~120 MB | Emitter identification |
| 10 | Anomaly Detector | VAE | 1.8M | ~100 MB | Unknown signal detection |

**Total GPU memory needed if all loaded simultaneously: ~2-2.5 GB**
(Fits on GTX 1650 4GB / T4 16GB / A100 40GB)

---

## GPU Sharing Strategies

### Strategy 1: NVIDIA Multi-Process Service (MPS) — RECOMMENDED

```
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA MPS                                 │
│                                                              │
│  Without MPS (default):                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐                                │
│  │Proc 1│ │Proc 2│ │Proc 3│   Each process gets exclusive   │
│  │Model1│ │Model2│ │Model3│   GPU access → context switch   │
│  └──┬───┘ └──┬───┘ └──┬───┘   overhead is HUGE             │
│     │        │        │                                      │
│     ▼ wait   ▼ wait   ▼ wait  (serial execution!)           │
│  ┌──────────────────────────────────────────┐               │
│  │          GPU (one at a time)              │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  With MPS:                                                   │
│  ┌──────┐ ┌──────┐ ┌──────┐                                │
│  │Proc 1│ │Proc 2│ │Proc 3│   All share GPU simultaneously │
│  │Model1│ │Model2│ │Model3│   through MPS server            │
│  └──┬───┘ └──┬───┘ └──┬───┘                                │
│     │        │        │                                      │
│     └────────┼────────┘                                      │
│              ▼                                                │
│  ┌──────────────────────────────────────────┐               │
│  │       MPS Server (multiplexes)            │               │
│  │  Shares CUDA context across processes     │               │
│  └──────────────────────┬───────────────────┘               │
│                          ▼                                    │
│  ┌──────────────────────────────────────────┐               │
│  │    GPU (parallel kernels from all procs)  │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**How MPS works:**
- Single CUDA context shared across multiple processes
- GPU kernels from different processes execute concurrently
- No context-switch overhead between processes
- Each process thinks it has exclusive GPU access

**Setup:**
```bash
# Start MPS daemon on each GPU node (before launching Spark workers)
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d

# Set memory limits per client (optional)
echo "set_default_active_thread_percentage 12" | nvidia-cuda-mps-control
# 12% per client × 8 clients ≈ 96% GPU utilization

# Verify MPS is running
echo "get_server_list" | nvidia-cuda-mps-control
```

**Spark integration:**
```python
# In spark-submit or spark-defaults.conf:
# spark.executorEnv.CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
# spark.executorEnv.CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

# Each Spark executor (process) shares the GPU via MPS automatically
# No code changes needed in PyTorch!
```

**Pros:** Best throughput, true parallel execution, no code changes
**Cons:** Requires NVIDIA driver support (Volta+ for full features), all processes must use same GPU

---

### Strategy 2: CUDA Streams (Single Process, Multiple Models)

```
┌─────────────────────────────────────────────────────────────┐
│              CUDA STREAMS (in-process parallelism)            │
│                                                              │
│  Single Python Process:                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │  Stream 0: [Model 1 inference]─────────────────►   │    │
│  │  Stream 1: [Model 2 inference]─────────────────►   │    │
│  │  Stream 2: [Model 3 inference]─────────────────►   │    │
│  │  Stream 3: [Model 4 inference]─────────────────►   │    │
│  │  ...                                                │    │
│  │  Stream 9: [Model 10 inference]────────────────►   │    │
│  │                                                     │    │
│  │  All streams execute concurrently on GPU!           │    │
│  │  GPU scheduler interleaves kernels from streams     │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
import torch
import torch.cuda

class MultiModelInferenceEngine:
    """Run multiple models in parallel using CUDA streams."""
    
    def __init__(self, models, device="cuda"):
        self.device = torch.device(device)
        self.models = []
        self.streams = []
        
        for model in models:
            model = model.to(self.device).eval()
            self.models.append(model)
            self.streams.append(torch.cuda.Stream())
    
    def infer_all(self, inputs_per_model):
        """
        Run all models concurrently on their respective inputs.
        
        Args:
            inputs_per_model: list of tensors, one per model
            
        Returns:
            list of output tensors
        """
        outputs = [None] * len(self.models)
        
        # Launch all models on separate streams
        for i, (model, stream, inp) in enumerate(
            zip(self.models, self.streams, inputs_per_model)
        ):
            with torch.cuda.stream(stream):
                inp_gpu = inp.to(self.device, non_blocking=True)
                with torch.no_grad():
                    outputs[i] = model(inp_gpu)
        
        # Synchronize all streams
        torch.cuda.synchronize()
        
        return outputs
```

**Pros:** Single process (no IPC overhead), fine-grained control, works everywhere
**Cons:** GIL limits Python-side parallelism (but CUDA kernels run truly parallel), all models share one memory pool

---

### Strategy 3: NVIDIA Time-Slicing (Kubernetes/Docker)

```
┌─────────────────────────────────────────────────────────────┐
│             GPU TIME-SLICING                                  │
│                                                              │
│  Each container gets a "virtual GPU" slice:                  │
│                                                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                      │
│  │Pod 1 │ │Pod 2 │ │Pod 3 │ │Pod 4 │                      │
│  │Model │ │Model │ │Model │ │Model │                      │
│  │ 1,2  │ │ 3,4  │ │ 5,6  │ │ 7-10 │                      │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                      │
│     │        │        │        │                            │
│     ▼        ▼        ▼        ▼                            │
│  ┌──────────────────────────────────────────┐               │
│  │  GPU Time-Slice Scheduler                 │               │
│  │  [Pod1][Pod2][Pod3][Pod4][Pod1][Pod2]...  │               │
│  │                                           │               │
│  │  Each pod gets fair share of GPU time     │               │
│  │  Memory is NOT partitioned (oversubscribe)│               │
│  └──────────────────────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Setup (Kubernetes):**
```yaml
# nvidia-device-plugin-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-plugin-configs
data:
  config: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 10    # 1 physical GPU → 10 virtual GPUs
```

**Pros:** No code changes, works with any framework, Kubernetes-native
**Cons:** Not true parallelism (time-division), higher latency per model, memory oversubscription risk

---

### Strategy 4: Multi-Instance GPU (MIG) — A100/H100 Only

```
┌─────────────────────────────────────────────────────────────┐
│              NVIDIA MIG (A100/H100 only)                      │
│                                                              │
│  Physical A100 80GB → Split into isolated GPU instances:     │
│                                                              │
│  ┌─────────────┬─────────────┬─────────────┐               │
│  │ MIG Slice 1 │ MIG Slice 2 │ MIG Slice 3 │               │
│  │ 3g.20gb     │ 3g.20gb     │ 2g.20gb     │               │
│  │             │             │             │               │
│  │ Models 1-3  │ Models 4-6  │ Models 7-10 │               │
│  │ 20GB VRAM   │ 20GB VRAM   │ 20GB VRAM   │               │
│  │ Isolated!   │ Isolated!   │ Isolated!   │               │
│  └─────────────┴─────────────┴─────────────┘               │
│                                                              │
│  Each slice is a fully isolated GPU with guaranteed          │
│  memory and compute. No interference between slices.         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Pros:** Hardware-level isolation, guaranteed resources, no noisy neighbor
**Cons:** Only A100/H100, coarse granularity (max 7 slices), requires specific hardware

---

## Recommended Architecture: Spark + MPS + CUDA Streams

For your use case (7-10 models, Spark cluster, DRDO airgapped):

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    SPARK DRIVER                               │    │
│  │                                                              │    │
│  │  1. Load all 10 model weights                                │    │
│  │  2. Serialize each model → broadcast                         │    │
│  │  3. Partition input data                                     │    │
│  │  4. Schedule tasks across executors                          │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
│                              │                                        │
│         ┌────────────────────┼────────────────────┐                  │
│         ▼                    ▼                    ▼                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  EXECUTOR 1  │    │  EXECUTOR 2  │    │  EXECUTOR 3  │          │
│  │  (Worker 1)  │    │  (Worker 2)  │    │  (Worker 3)  │          │
│  │              │    │              │    │              │          │
│  │  GPU: A100   │    │  GPU: A100   │    │  GPU: A100   │          │
│  │  MPS: enabled│    │  MPS: enabled│    │  MPS: enabled│          │
│  │              │    │              │    │              │          │
│  │  ┌────────┐ │    │  ┌────────┐ │    │  ┌────────┐ │          │
│  │  │Stream 0│ │    │  │Stream 0│ │    │  │Stream 0│ │          │
│  │  │Model 1 │ │    │  │Model 1 │ │    │  │Model 1 │ │          │
│  │  ├────────┤ │    │  ├────────┤ │    │  ├────────┤ │          │
│  │  │Stream 1│ │    │  │Stream 1│ │    │  │Stream 1│ │          │
│  │  │Model 2 │ │    │  │Model 2 │ │    │  │Model 2 │ │          │
│  │  ├────────┤ │    │  ├────────┤ │    │  ├────────┤ │          │
│  │  │  ...   │ │    │  │  ...   │ │    │  │  ...   │ │          │
│  │  ├────────┤ │    │  ├────────┤ │    │  ├────────┤ │          │
│  │  │Stream 9│ │    │  │Stream 9│ │    │  │Stream 9│ │          │
│  │  │Model 10│ │    │  │Model 10│ │    │  │Model 10│ │          │
│  │  └────────┘ │    │  └────────┘ │    │  └────────┘ │          │
│  │              │    │              │    │              │          │
│  │ Each executor│    │ Processes    │    │ its own     │          │
│  │ runs ALL 10  │    │ data chunk   │    │ data chunk  │          │
│  │ models on its│    │ through all  │    │ through all │          │
│  │ data chunk   │    │ 10 models    │    │ 10 models   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### MPP Data Flow for 10 Models

```
Input: 1,000,000 signals/images

Step 1: Partition across workers (data parallelism)
  Worker 1: 333,333 samples
  Worker 2: 333,333 samples
  Worker 3: 333,334 samples

Step 2: Each worker runs ALL 10 models on its chunk (model parallelism)
  Per worker, using CUDA streams:
    Stream 0: EW Classifier     → 333K predictions
    Stream 1: YOLOv8 Detection  → 333K detections
    Stream 2: ResNet-18         → 333K classifications
    Stream 3: MobileNet         → 333K classifications
    Stream 4: EfficientNet      → 333K classifications
    Stream 5: Signal Denoiser   → 333K denoised signals
    Stream 6: Threat Prioritizer→ 333K priority scores
    Stream 7: RF Fingerprinter  → 333K fingerprints
    Stream 8: Anomaly Detector  → 333K anomaly scores
    Stream 9: (spare/ensemble)  → 333K ensemble outputs

Step 3: Collect and merge results from all workers
  Total: 1M samples × 10 model outputs = 10M predictions
```

---

## GPU Memory Budget Planning

### Memory Estimation Formula

```
Total GPU Memory Required =
    Σ(model_size × 1.2)          # Model weights + optimizer overhead
  + Σ(batch_size × input_size)    # Input tensors (per stream)
  + Σ(batch_size × output_size)   # Output tensors (per stream)
  + CUDA_context_overhead          # ~300MB fixed
  + MPS_overhead                   # ~100MB if using MPS
```

### Budget for Our 10 Models on Different GPUs

| GPU | VRAM | Models Fit? | Batch Size (per model) | Utilization |
|-----|------|-------------|----------------------|-------------|
| GTX 1650 | 4 GB | 5-6 max | 32 | ~85% |
| T4 | 16 GB | All 10 | 256 | ~60% |
| A100 40GB | 40 GB | All 10 | 1024 | ~40% |
| A100 80GB | 80 GB | All 10 | 4096 | ~25% |

### Memory-Efficient Loading

```python
class GPUMemoryManager:
    """Manages GPU memory budget across multiple models."""
    
    def __init__(self, max_memory_gb=4.0, reserve_gb=0.5):
        self.max_bytes = int((max_memory_gb - reserve_gb) * 1e9)
        self.allocated = 0
        self.models_loaded = []
    
    def can_fit(self, model_size_bytes):
        return (self.allocated + model_size_bytes) < self.max_bytes
    
    def load_model(self, model, model_name):
        """Load model to GPU if memory available, else keep on CPU."""
        model_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        
        if self.can_fit(model_size):
            model = model.to("cuda")
            self.allocated += model_size
            self.models_loaded.append((model_name, "cuda", model_size))
            return model, "cuda"
        else:
            # Fallback to CPU (hybrid mode)
            self.models_loaded.append((model_name, "cpu", model_size))
            return model, "cpu"
    
    def report(self):
        print(f"GPU Memory: {self.allocated/1e9:.2f} / {self.max_bytes/1e9:.2f} GB")
        for name, device, size in self.models_loaded:
            print(f"  {name}: {device} ({size/1e6:.1f} MB)")
```

---

## Three Inference Modes Implementation

### Mode 1: Distributed GPU (Cluster)

```python
def run_distributed_gpu(spark, data, models, num_workers=3):
    """
    Each worker has its own GPU. All 10 models run on each worker's GPU.
    Data is partitioned across workers.
    
    Best for: Large datasets, cluster with multiple GPU nodes
    """
    # Broadcast all model weights
    bc_models = {}
    for name, model in models.items():
        bc_models[name] = spark.sparkContext.broadcast(serialize_model(model))
    
    # Partition data
    rdd = partition_data(spark, data, num_workers)
    
    def infer_on_partition(partition_data):
        # Load ALL models onto this worker's GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_models = {}
        streams = {}
        
        for name, bc_model in bc_models.items():
            model = deserialize_model(bc_model.value).to(device).eval()
            loaded_models[name] = model
            streams[name] = torch.cuda.Stream() if device.type == "cuda" else None
        
        # Run all models using CUDA streams
        results = run_multi_model_streams(loaded_models, streams, partition_data, device)
        return results
    
    return rdd.map(infer_on_partition).collect()
```

### Mode 2: Single GPU

```python
def run_single_gpu(data, models, batch_size=256):
    """
    All 10 models on one GPU with CUDA streams for parallelism.
    Sequential data processing, parallel model execution.
    
    Best for: Single workstation, moderate data size
    """
    device = torch.device("cuda")
    engine = MultiModelInferenceEngine(models, device)
    
    all_results = {name: [] for name in models}
    
    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start:batch_start + batch_size]
        
        # Prepare inputs for each model (may differ in preprocessing)
        inputs = prepare_inputs_per_model(batch, models)
        
        # Run all 10 models concurrently via CUDA streams
        outputs = engine.infer_all(inputs)
        
        for name, output in zip(models.keys(), outputs):
            all_results[name].append(output.cpu())
    
    return all_results
```

### Mode 3: CPU + GPU Hybrid

```python
def run_hybrid_cpu_gpu(data, models, gpu_memory_limit_gb=3.5):
    """
    Large models on GPU, small models on CPU, running in parallel.
    Uses threading for CPU models, CUDA streams for GPU models.
    
    Best for: Limited GPU memory, mixed model sizes
    """
    memory_mgr = GPUMemoryManager(max_memory_gb=gpu_memory_limit_gb)
    
    gpu_models = {}
    cpu_models = {}
    
    # Sort models by size (largest first → fill GPU with biggest models)
    sorted_models = sorted(models.items(), key=lambda x: model_size(x[1]), reverse=True)
    
    for name, model in sorted_models:
        model, device = memory_mgr.load_model(model, name)
        if device == "cuda":
            gpu_models[name] = model
        else:
            cpu_models[name] = model
    
    memory_mgr.report()
    # Example output:
    # GPU Memory: 2.80 / 3.50 GB
    #   yolov8_small: cuda (400.0 MB)
    #   resnet18: cuda (300.0 MB)
    #   threat_prioritizer: cuda (350.0 MB)
    #   ...
    #   signal_denoiser: cpu (100.0 MB)
    #   anomaly_detector: cpu (100.0 MB)
    
    # Run GPU models with CUDA streams
    gpu_results = run_with_streams(gpu_models, data)
    
    # Run CPU models with thread pool
    cpu_results = run_with_threads(cpu_models, data, num_threads=4)
    
    return {**gpu_results, **cpu_results}
```

---

## Spark GPU Scheduling Configuration

### Fractional GPU Assignment

```properties
# spark-defaults.conf
# Allow multiple tasks to share one GPU:
spark.executor.resource.gpu.amount=1
spark.task.resource.gpu.amount=0.1    # 10 tasks share 1 GPU

# This means each executor gets 1 GPU, and can run 10 tasks on it
# Combined with MPS, all 10 tasks execute truly in parallel
```

### Resource Scheduling Diagram

```
Executor (1 GPU, 4 cores, 8GB RAM):
┌─────────────────────────────────────────────────────────┐
│ Core 0: Task (Model 1+2+3 inference on Stream 0,1,2)    │
│ Core 1: Task (Model 4+5+6 inference on Stream 3,4,5)    │
│ Core 2: Task (Model 7+8 inference on Stream 6,7)        │
│ Core 3: Task (Model 9+10 inference on Stream 8,9)       │
│                                                          │
│ GPU (shared via MPS):                                    │
│   Stream 0-9 all active simultaneously                   │
│   Kernels interleaved by GPU scheduler                   │
└─────────────────────────────────────────────────────────┘
```

---

## Performance Expectations

### Throughput Projections (1M samples)

| Mode | Hardware | Expected Throughput | Wall Time |
|------|----------|--------------------| ----------|
| Single GPU (streams) | 1× T4 | ~50K samples/sec (all models) | ~20s |
| Distributed GPU (3 nodes) | 3× T4 | ~140K samples/sec | ~7s |
| Hybrid CPU+GPU | 1× T4 + 8 cores | ~35K samples/sec | ~29s |
| CPU only (baseline) | 8 cores | ~8K samples/sec | ~125s |

### Scaling Factor

```
Distributed GPU scaling:
  1 GPU:  1.0x (baseline)
  2 GPUs: 1.9x (near-linear, small overhead)
  4 GPUs: 3.7x (network overhead starts)
  8 GPUs: 7.0x (good efficiency at 87%)
  
  Efficiency = actual_speedup / ideal_speedup
  For inference (embarrassingly parallel): efficiency > 85% typical
```

---

## AWS Test Plan (Low Cost)

### Cheapest GPU Cluster Setup

```
Option A: Spot Instances (~$0.50/hour total)
  1× t3.medium (master, no GPU)     = $0.04/hr
  2× g4dn.xlarge spot (T4 workers)  = $0.16/hr × 2 = $0.32/hr
  Total: ~$0.36/hour
  
  Budget $5 → ~14 hours of testing

Option B: Single GPU (minimal test, ~$0.16/hr)
  1× g4dn.xlarge spot               = $0.16/hr
  
  Budget $2 → ~12 hours of testing
```

### AWS Setup Script

```bash
# Launch with AWS CLI (spot instances)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Deep Learning AMI (CUDA pre-installed)
  --instance-type g4dn.xlarge \
  --spot-instance-type one-time \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.20"}}' \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --count 2 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spark-worker}]'
```

---

## DRDO Deployment Considerations

| Requirement | Solution |
|-------------|----------|
| No internet | Docker image with all deps pre-baked |
| GPU sharing | MPS daemon started at boot via systemd |
| Multi-model | CUDA streams within each executor |
| Fault tolerance | Spark auto-retry on task failure |
| Security | No external network calls, model weights on-premise |
| Monitoring | Spark UI on internal network |
| Scale | Add workers → linear throughput increase |

### MPS Startup Service (on each GPU node)

```bash
# /etc/systemd/system/nvidia-mps.service
[Unit]
Description=NVIDIA MPS Control Daemon
After=nvidia-persistenced.service

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-cuda-mps-control -d
ExecStop=/bin/echo quit | /usr/bin/nvidia-cuda-mps-control
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## Summary: Which Strategy When?

| Scenario | Strategy | Why |
|----------|----------|-----|
| Single GPU, 7-10 small models | CUDA Streams | Lowest overhead, true parallel |
| Cluster, 1 GPU per node | MPS + Streams | Best multi-process sharing |
| Kubernetes cluster | Time-slicing + Streams | K8s native, no MPS needed |
| A100/H100 cluster | MIG + Streams | Hardware isolation, guaranteed perf |
| Limited GPU memory (4GB) | Hybrid CPU+GPU | Fit critical models on GPU, rest on CPU |
| AWS spot testing | MPS + Streams | Same as production, cheap to test |

### Our Implementation Priority

1. **CUDA Streams** (works everywhere, no infra dependency)
2. **MPS** (add on cluster, transparent to code)
3. **Hybrid CPU+GPU** (fallback for memory-constrained environments)
4. **Time-slicing** (if using K8s on DRDO)
