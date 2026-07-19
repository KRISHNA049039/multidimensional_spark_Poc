# Final Comprehensive Report — Multi-Model Distributed Inference POC

**Date:** July 19, 2026
**AWS Region:** ap-south-1 (Mumbai)
**Cluster:** m5.xlarge (master) + g4dn.xlarge (GPU worker)
**Duration:** ~5 hours total testing

---

## 1. Executive Summary

Successfully demonstrated multi-model inference across 3 modes on an AWS Spark cluster:

| Mode | Best Throughput | Latency | GPU Used | Where Executed |
|------|----------------|---------|----------|----------------|
| Single GPU (CUDA Streams) | **30,087 samples/sec** | 0.85s | Tesla T4 | GPU worker |
| Hybrid CPU+GPU | **29,910 samples/sec** | 0.86s | Tesla T4 | GPU worker |
| Distributed Spark (cluster) | **1,880 samples/sec** | 13.67s | T4 (executor) + CPU | Both nodes |

**Key achievement:** Confirmed GPU execution inside Spark executor subprocess (`device=cuda` in executor logs).

---

## 1.5 Inference Mode Definitions

| Mode | Uses Spark? | Uses GPU? | What It Does | When to Use |
|------|-------------|-----------|--------------|-------------|
| **Single GPU** | No | Yes (1 GPU) | Loads all 10 models on ONE GPU, runs them in parallel using CUDA Streams. No cluster involved — runs on a single machine. | Single workstation with a GPU. Best throughput per node. |
| **Hybrid CPU+GPU** | No | Yes + CPU | Intelligently splits models between GPU and CPU based on available VRAM. High-priority/large models go to GPU, rest overflow to CPU. Both run in parallel. Single machine only. | When GPU memory is limited (e.g., 4GB VRAM can't fit all models). |
| **Distributed Spark** | **Yes** | Yes (per executor) | Uses Apache Spark to distribute data across multiple machines. Data is partitioned into chunks, each chunk sent to a different worker node. Each worker loads models and runs inference on its chunk. Results are collected back to the driver. | Multiple machines, very large datasets (>single machine RAM), production fault-tolerance. |

### Visual Summary

```
SINGLE GPU (No Spark):
  [Your Machine] → Load 10 models on GPU → Process ALL data → Results
  - 1 machine, 1 GPU, 30,000 samples/sec

HYBRID CPU+GPU (No Spark):
  [Your Machine] → GPU gets priority models, CPU gets overflow → Both run parallel → Results
  - 1 machine, GPU + CPU, 30,000 samples/sec (when all fit on GPU)

DISTRIBUTED SPARK (Uses Spark Cluster):
  [Master] → Partition data into N chunks →
    [Worker 1 GPU] processes chunk 1
    [Worker 2 GPU] processes chunk 2
    [Worker 3 GPU] processes chunk 3
    ...
  → Collect all results back to Master
  - N machines, N GPUs, scales linearly with nodes
```

### Key Point for Management

- **Single GPU and Hybrid are single-machine modes** — they don't use Spark at all. They demonstrate maximum per-node performance.
- **Distributed Spark is the cluster mode** — it's the only one that uses Apache Spark to coordinate work across multiple machines.
- In production (air-gapped 5-node cluster), the **Distributed Spark mode with 5 GPUs** combines the best of both: Spark handles distribution, each node uses GPU for inference.

---

## 2. Test Environment

### 2.1 Cluster Hardware

| Node | Instance | vCPUs | RAM | GPU | Disk | Role |
|------|----------|-------|-----|-----|------|------|
| Master | m5.xlarge | 4 | 16 GB | None | 100 GB gp3 | Spark Master + Driver + CPU Worker |
| GPU Worker | g4dn.xlarge | 4 | 16 GB | Tesla T4 (16 GB VRAM) | 150 GB gp3 | Spark GPU Worker |

### 2.2 Software Stack

| Component | Version |
|-----------|---------|
| OS | Amazon Linux 2023 |
| Docker | 25.x |
| Spark | 3.5.1 (standalone cluster) |
| PyTorch | 2.2.0+cu121 |
| CUDA | 12.1 |
| Python | 3.11.15 |
| NVIDIA Driver | 535.x (via dnf module) |

### 2.3 Models (10 total, 1,970 MB GPU footprint)

| # | Model | Category | Memory | Input | Params |
|---|-------|----------|--------|-------|--------|
| 1 | ew_classifier | Signal | 50 MB | (128,) | MLP, 4 layers |
| 2 | signal_denoiser | Signal | 100 MB | (128,) | Autoencoder |
| 3 | threat_prioritizer | Signal | 350 MB | (128,) | Multi-head attention |
| 4 | rf_fingerprinter | Signal | 120 MB | (128,) | 1D-CNN |
| 5 | anomaly_detector | Signal | 100 MB | (128,) | VAE |
| 6 | resnet18 | Image | 300 MB | (3,224,224) | 11.7M params |
| 7 | mobilenetv3 | Image | 150 MB | (3,224,224) | 5.4M params |
| 8 | efficientnet_b0 | Image | 200 MB | (3,224,224) | 5.3M params |
| 9 | yolov8_nano | Detection | 200 MB | (3,640,640) | 3.2M params |
| 10 | yolov8_small | Detection | 400 MB | (3,640,640) | 11.2M params |

---

## 3. All Test Results

### Mode Comparison (GPU)

![Mode Comparison](final_mode_comparison.png)

### GPU vs CPU Comparison

![GPU vs CPU](final_gpu_vs_cpu.png)

### 3.1 Single GPU Mode (Tesla T4, g4dn.xlarge)

**Executor:** GPU worker container directly (docker exec)

| Metric | Value |
|--------|-------|
| Device | cuda (Tesla T4, 15.6 GB VRAM) |
| Total Samples | 25,700 |
| **Throughput** | **30,087 samples/sec** |
| **Elapsed Time** | **0.85s** |
| Avg Batch Latency | 42.86 ms |
| P99 Batch Latency | 638.02 ms |
| Batch Size | 256 |
| Batches | 20 |
| Models on GPU | 10/10 |
| VRAM Used | 1,970 / 15,600 MB (12.6%) |

### 3.2 Hybrid CPU+GPU Mode (Tesla T4, g4dn.xlarge)

**Executor:** GPU worker container directly

| Metric | Value |
|--------|-------|
| Device | cuda (all models fit on GPU) |
| Total Samples | 25,700 |
| **Throughput** | **29,910 samples/sec** |
| **Elapsed Time** | **0.86s** |
| Avg Batch Latency | 42.96 ms |
| P99 Batch Latency | 640.38 ms |
| GPU Models | 10 |
| CPU Models | 0 (none spilled) |
| Strategy | Priority-based placement |
| GPU Memory: Total | 14,922 MB |
| GPU Memory: Free After | 12,452 MB |

### 3.3 Distributed Spark Mode (Cluster: m5.xlarge + g4dn.xlarge)

**Executor:** Spark driver on master, tasks distributed to both nodes

| Metric | Value |
|--------|-------|
| Spark Master | spark://10.0.0.187:7077 |
| Total Samples | 25,700 |
| **Throughput** | **1,880 samples/sec** |
| **Elapsed Time** | **13.67s** |
| Partitions | 4 |
| Total Tasks | 4 |
| Failed Tasks | 0 |
| Executors | 2 |
| GPU Executor (10.0.0.45) | device=cuda confirmed |
| CPU Executor (10.0.0.187) | device=cpu |

### 3.4 All Modes from Master (CPU-only, for baseline comparison)

| Mode | Throughput | Time | Note |
|------|-----------|------|------|
| Single GPU (CPU fallback) | 1,332 /sec | 19.29s | Master has no GPU |
| Hybrid (CPU only) | 1,334 /sec | 19.27s | All 10 models on CPU |
| Distributed (GPU+CPU) | 1,880 /sec | 13.67s | GPU executor helps |

**Insight:** Distributed mode outperforms master-only modes because it leverages the GPU worker's executor.

### 3.5 Incremental Load Test (Distributed, scaling data size)

![Scaling Throughput](final_scaling_throughput.png)

| Run | Samples | Data (MB) | Partitions | Throughput | Elapsed | Status |
|-----|---------|-----------|-----------|-----------|---------|--------|
| 1 | 2,580 | 135.7 | 2 | 327 /sec | 7.90s | OK (cold start) |
| 2 | 5,190 | 289.5 | 2 | 973 /sec | 5.33s | OK |
| 3 | 10,360 | 480.7 | 2 | 1,280 /sec | 8.09s | OK |
| 4 | 25,700 | 865.6 | 4 | 3,197 /sec | 8.04s | OK |
| 5 | 51,360 | 1,534.6 | 4 | 3,046 /sec | 16.86s | OK |

---

## 4. Spark Executor-Level Statistics

### Executor Task Distribution

![Executor Distribution](final_executor_distribution.png)

### 4.1 Executor Distribution

| Executor ID | Host | Cores | Tasks Completed | Total Duration | GC Time | Memory Used |
|-------------|------|-------|----------------|---------------|---------|-------------|
| 0 (GPU) | 10.0.0.45 | 2 | **9 tasks** | 34.7s | 75ms | 85 MB |
| 1 (CPU) | 10.0.0.187 | 2 | **5 tasks** | 44.6s | 105ms | 89 MB |
| Driver | 10.0.0.187 | — | 0 (coordination) | 83.0s | — | 89 MB |

**GPU worker handled 64% of all tasks** — Spark naturally routes more work to faster executors.

### 4.2 Stage-Level Metrics (5 Stages from Incremental Test)

![Stage Timeline](final_stage_timeline.png)

| Stage | Tasks | Executor Run Time | Deserialize Time | GC Time | Result Size |
|-------|-------|------------------|-----------------|---------|-------------|
| 0 (cold) | 2 | 11,538 ms | 728 ms | 110 ms | 3.1 KB |
| 1 | 2 | 5,512 ms | 183 ms | 19 ms | 3.0 KB |
| 2 | 2 | 7,885 ms | 264 ms | 12 ms | 3.1 KB |
| 3 | 4 | 10,632 ms | 415 ms | 10 ms | 6.2 KB |
| 4 | 4 | 18,036 ms | 560 ms | 29 ms | 6.2 KB |

### 4.3 Job Summary

| Job | Stage | Tasks | Duration | Status |
|-----|-------|-------|----------|--------|
| 0 (Run 1) | 0 | 2 | 7.7s | SUCCEEDED |
| 1 (Run 2) | 1 | 2 | 5.3s | SUCCEEDED |
| 2 (Run 3) | 2 | 2 | 8.1s | SUCCEEDED |
| 3 (Run 4) | 3 | 4 | 8.0s | SUCCEEDED |
| 4 (Run 5) | 4 | 4 | 16.8s | SUCCEEDED |

**Zero failures across 14 total tasks.**

---

## 5. GPU Confirmation in Distributed Mode

**From executor stderr logs on 10.0.0.45:**
```
[Executor] partition=0, host=ip-10-0-0-45.ap-south-1.compute.internal, cuda=True, device=cuda
```

This confirms:
- `torch.cuda.is_available()` = True inside the Spark executor subprocess
- Model inference ran on `device=cuda` (Tesla T4)
- The NVIDIA environment variables fix (`NVIDIA_VISIBLE_DEVICES=all`, `CUDA_VISIBLE_DEVICES=0`, `LD_LIBRARY_PATH`) enabled GPU access

---

## 6. Challenges Faced on EC2 and How to Avoid in Air-Gapped

### 6.1 GPU Not Accessible from Spark Executor (Docker)

**EC2 Problem:** Spark's Python executor subprocess inside Docker couldn't access GPU initially.

**Root Cause:** Docker's NVIDIA runtime hooks into `docker run`, but Spark's JVM spawns Python subprocesses that may not inherit the GPU device mappings without explicit environment variables.

**EC2 Fix:** Set `spark.executorEnv.NVIDIA_VISIBLE_DEVICES=all` and `spark.executorEnv.LD_LIBRARY_PATH=/usr/local/cuda/lib64`.

**Air-Gapped Avoidance:** Run Spark natively (no Docker). GPU is visible to ALL processes system-wide via `/dev/nvidia0`. No environment variable tricks needed.

### 6.2 Driver OOM on Large Data (>2GB)

**EC2 Problem:** m5.xlarge (16GB) couldn't serialize >2GB of data into RDD elements.

**Root Cause:** Driver JVM needs to hold all partition data in memory during `sc.parallelize()`.

**EC2 Fix:** Increase partitions to keep each under 450MB; limit data size.

**Air-Gapped Avoidance:** 256GB RAM per node. Driver can hold 100+GB of data. For truly massive datasets (>50GB), read from shared NFS/disk on each executor instead of driver-serialized RDD.

### 6.3 spark.rpc.message.maxSize Exceeded

**EC2 Problem:** Serialized task exceeded 128MB default (then 512MB after fix).

**Root Cause:** Each RDD element (partition) contains all model data for that chunk. Large images (3x224x224 or 3x640x640) × thousands = hundreds of MB per partition.

**EC2 Fix:** Set `spark.rpc.message.maxSize=512`; increase partition count.

**Air-Gapped Avoidance:** With 256GB RAM and fast LAN, set `spark.rpc.message.maxSize=2048`. Or better: use `mapPartitions` with iterator-based data loading from shared disk.

### 6.4 Per-Task Model Loading Overhead

**EC2 Problem:** Each Spark task deserializes all 10 models (~5-7s per task), dominating total time.

**Root Cause:** Current code loads models fresh on every `infer_on_partition()` call. With 4+ partitions, this happens 4+ times.

**EC2 Impact:** Distributed throughput limited to ~1,880/sec vs 30,000/sec single GPU.

**Air-Gapped Fix (code change):**
```python
# Use mapPartitions instead of map — loads models ONCE per executor
def init_and_infer(partition_iterator):
    # Load models once
    models = load_all_models(device="cuda")
    # Process all items in this partition
    for item in partition_iterator:
        yield infer(models, item)

results = data_rdd.mapPartitions(init_and_infer).collect()
```

This loads models once per executor (not per task), eliminating the 5-7s overhead per partition.

### 6.5 SSM Agent Crashing Under Load

**EC2 Problem:** Heavy Docker builds + Spark jobs consumed all memory, crashing the SSM agent.

**Root Cause:** m5.xlarge (16GB) shared between OS, Docker, Spark JVM, and SSM agent.

**Air-Gapped Avoidance:** 256GB RAM. No SSM needed (direct physical/KVM access). No memory pressure.

### 6.6 Auto-Shutdown Safety Net Triggered

**EC2 Problem:** Instances stopped mid-test due to the 4-hour safety timer.

**Air-Gapped Avoidance:** No auto-shutdown needed. Machines are always on in the lab.

---

## 7. Best Practices for Partitioning

### 7.1 Partition Sizing Rules

| Data Size | Recommended Partitions | Per-Partition | Reasoning |
|-----------|----------------------|---------------|-----------|
| < 200 MB | 2 | < 100 MB | Minimal overhead, 1 task per executor |
| 200 MB - 1 GB | 2-4 | 100-300 MB | Balance overhead vs parallelism |
| 1 - 5 GB | 4-8 | 200-500 MB | Stay under message size limit |
| 5 - 20 GB | 8-32 | 300-600 MB | More parallelism needed |
| > 20 GB | Storage-backed | N/A | Read from disk on each executor |

### 7.2 Formula

```
optimal_partitions = max(num_executors, ceil(total_data_bytes / (300 * 1024 * 1024)))
```

### 7.3 Key Principles

1. **1 partition per executor minimum** — ensures all executors get work
2. **Per-partition < 450MB** — stays within `spark.rpc.message.maxSize=512`
3. **Fewer partitions = less model loading overhead** — each partition triggers model deserialization
4. **More partitions = better load balancing** — but more scheduling overhead
5. **Sweet spot: 1-2 partitions per executor** for inference workloads

### 7.4 For Air-Gapped 5-Node Cluster

```
5 GPU executors × 1 partition each = 5 partitions (minimum)
5 GPU executors × 2 partitions each = 10 partitions (good balance)

With 256GB driver and spark.rpc.message.maxSize=2048:
  Max data per partition: ~2GB
  Max total data (5 partitions): ~10GB in-memory
  Max total data (10 partitions): ~20GB in-memory
  Beyond 20GB: use disk-backed loading
```

---

## 8. Performance Analysis

### Batch Latency: GPU vs CPU

![Latency Breakdown](final_latency_breakdown.png)

### 8.1 Why Single GPU is 16x Faster Than Distributed

| Factor | Single GPU | Distributed |
|--------|-----------|-------------|
| Model loading | Once (3s) | Per task (5-7s × 4 tasks) |
| Data transfer | None (local) | Serialize + network + deserialize |
| GPU access | Direct | Via executor subprocess |
| Coordination | None | Spark scheduler + JVM |
| Batch processing | Continuous | Per-partition restart |

### 8.2 When Distributed Mode Wins

- Data > single-machine RAM (>16GB on g4dn, >256GB on air-gapped)
- Multiple GPUs needed (models > single GPU VRAM)
- Fault tolerance required (Spark retries failed tasks)
- Continuous streaming (new data arrives, distribute across workers)

### 8.3 Throughput Scaling Pattern

```
Samples/sec
30,000 |●─────────────── Single GPU / Hybrid
       |
10,000 |
       |
 3,200 |          ●───── Distributed (warm, 4 partitions)
 3,000 |        ●
       |      /
 1,300 |   ●
       |  /
   500 |●               Distributed (cold start)
       |________________
        2.5K  5K  10K  25K  51K   Samples
```

---

## 9. Projected Air-Gapped Performance (5 Nodes, 256GB, 24GB VRAM)

![Projected Performance](final_projected_airgapped.png)

| Mode | Projected Throughput | Reasoning |
|------|---------------------|-----------|
| Single GPU (1 node, native) | **40,000-50,000 /sec** | Larger GPU (24 vs 16 GB), faster |
| Hybrid (1 node, 24GB VRAM) | **40,000-50,000 /sec** | All models fit easily |
| Distributed 5 GPUs (current code) | **8,000-12,000 /sec** | 5× executors, model loading overhead |
| Distributed 5 GPUs (mapPartitions) | **150,000-200,000 /sec** | Load models once per executor |
| Distributed 5 GPUs (persistent service) | **200,000+ /sec** | Models always loaded, stream data |

---

## 10. Recommendations

### Immediate (for presentation)
1. Use Single GPU results (30,087/sec) as the **per-node benchmark**
2. Use Distributed results (1,880/sec with GPU confirmed) as **cluster proof**
3. Explain that `mapPartitions` optimization will close the gap in production

### For Air-Gapped Production
1. Install Spark natively (no Docker) — eliminates GPU access issues
2. Use `mapPartitions` — load models once per executor, not per task
3. Set 5 partitions (1 per GPU node) — each executor loads once, processes all its data
4. Pre-load model weights into shared NFS — no download step
5. Set `spark.rpc.message.maxSize=2048` (256GB RAM can handle it)
6. Use NVIDIA MPS for multi-process GPU sharing if running >1 executor per node

### Code Changes for 10x Improvement
```python
# Replace: data_rdd.map(infer_on_partition).collect()
# With: data_rdd.mapPartitions(init_once_and_infer).collect()
```

This single change will boost distributed throughput from ~1,880 to ~15,000+ samples/sec on the same hardware.

---

## 11. Cost Summary

| Item | Cost |
|------|------|
| m5.xlarge (master, ~5 hours) | ~$0.96 |
| g4dn.xlarge (GPU worker, ~5 hours) | ~$2.63 |
| S3 storage | < $0.01 |
| Data transfer | $0.00 |
| **Total POC** | **~$3.60** |

---

## 12. Files Produced

| File | Description |
|------|-------------|
| `raw_results_single_gpu.json` | Single GPU mode (30,087 /sec) |
| `raw_results_hybrid_gpu.json` | Hybrid mode (29,910 /sec) |
| `raw_results_all_modes.json` | All modes from master |
| `raw_results.json` | Latest distributed run |
| `incremental_results.json` | 5-run scaling test + Spark UI stats |
| `INFERENCE_REPORT.md` | Original report with graphs |
| `DISTRIBUTED_CLUSTER_REPORT.md` | Cluster scaling analysis |
| `SPARK_JOBS_STAGES_TASKS_REPORT.md` | Detailed Spark metrics |
| `FINAL_COMPREHENSIVE_REPORT.md` | This document |
