# POC Results — Multi-Model Distributed Inference on AWS Spark Cluster

**Date:** July 19, 2026
**Author:** Auto-generated from benchmark runs
**Platform:** AWS EC2 (ap-south-1, Mumbai)
**Cluster:** 1x t3.large (master/CPU) + 1x g4dn.xlarge (GPU worker)

---

## 1. Cluster Configuration

| Component | Master | GPU Worker |
|-----------|--------|------------|
| Instance Type | t3.large | g4dn.xlarge |
| vCPUs | 2 | 4 |
| RAM | 8 GB | 16 GB |
| GPU | None | NVIDIA Tesla T4 (16 GB VRAM) |
| Storage | 100 GB gp3 | 150 GB gp3 |
| OS | Amazon Linux 2023 | Amazon Linux 2023 |
| Docker Image | multi-model-inference:latest | multi-model-inference:latest |
| Spark Role | Master + CPU Worker | GPU Worker |
| Spark Version | 3.5.1 | 3.5.1 |
| PyTorch | 2.2.0+cu121 | 2.2.0+cu121 |
| CUDA | N/A (CPU only) | 12.1 |

## 2. Models (10 Total)

| Model | Category | Est. Memory | Input Shape |
|-------|----------|-------------|-------------|
| ew_classifier | Signal | 50 MB | (128,) |
| signal_denoiser | Signal | 100 MB | (128,) |
| threat_prioritizer | Signal | 350 MB | (128,) |
| rf_fingerprinter | Signal | 120 MB | (128,) |
| anomaly_detector | Signal | 100 MB | (128,) |
| resnet18 | Image Classification | 300 MB | (3, 224, 224) |
| mobilenetv3 | Image Classification | 150 MB | (3, 224, 224) |
| efficientnet_b0 | Image Classification | 200 MB | (3, 224, 224) |
| yolov8_nano | Object Detection | 200 MB | (3, 640, 640) |
| yolov8_small | Object Detection | 400 MB | (3, 640, 640) |

**Total estimated GPU memory:** 1,970 MB (1.9 GB) — fits easily in T4's 16GB VRAM.

---

## 3. Benchmark Results

### 3.1 Mode Comparison (25,700 samples, batch_size=256)

| Mode | Throughput | Elapsed Time | Speedup vs Distributed |
|------|-----------|-------------|------------------------|
| **Single GPU** (CUDA Streams) | 29,980 samples/sec | 0.86s | 18.3x |
| **Hybrid CPU+GPU** | 30,029 samples/sec | 0.86s | 18.3x |
| **Distributed** (Spark, 2 partitions) | 1,639 samples/sec | 15.68s | 1.0x (baseline) |

### 3.2 Single GPU — Detailed Metrics

```
System: Linux-6.1.176-221.360.amzn2023.x86_64
GPU: Tesla T4 (15.6 GB VRAM)
CUDA: True
Samples: 25,700 total (5000 signals/model + 200 images + 50 detections)
Batch Size: 256
Batches: 20

Throughput: 29,980 samples/sec
Elapsed: 0.86s
Avg Batch Latency: 42.86 ms
P99 Batch Latency: 638.02 ms
Total Benchmark Time: 3.36s
```

### 3.3 Hybrid CPU+GPU — Detailed Metrics

```
GPU Models: 10 (all fit in VRAM)
CPU Models: 0 (none spilled)
Strategy: priority (fit all on GPU)

Throughput: 30,029 samples/sec
Elapsed: 0.86s
Avg Batch Latency: 42.79 ms
P99 Batch Latency: 636.76 ms
Total Benchmark Time: 3.38s
```

### 3.4 Distributed GPU (Spark) — Detailed Metrics

```
Spark Master: spark://10.0.0.187:7077
Workers: 2 (1 CPU @ t3.large, 1 GPU @ g4dn.xlarge)
Partitions: 2
Executor Memory: 8 GB

Throughput: 1,639 samples/sec
Elapsed: 15.68s
Total Samples Processed: 25,700
Total Benchmark Time: 28.55s (includes Spark session setup)
```

### 3.5 Incremental Load Test (Distributed Mode from Master)

| Run | Samples | Data Size | Partitions | Throughput | Time | Status |
|-----|---------|-----------|-----------|-----------|------|--------|
| 1 | 5,190 | 289.5 MB | 2 | 26 samples/sec | 198.9s | OK (cold start) |
| 2 | 25,700 | 865.6 MB | 2 | 3,019 samples/sec | 8.51s | OK |
| 3 | 51,700 | 1,911 MB | 2 | 1,280 samples/sec | 40.4s | OK |
| 4 | 103,400 | 3,824 MB | 2 | — | — | FAILED (driver OOM) |
| 5 | 257,000 | 8,656 MB | 4 | — | — | FAILED (driver OOM) |

**Run 1 notes:** First run is slow (198.9s) because model weights (resnet18=44MB, mobilenet=9MB, efficientnet=20MB) are downloaded from PyTorch Hub on the executor for the first time. Subsequent runs use cached weights.

### 3.6 Single GPU — Large Scale (from GPU Worker)

```
Signals: 50,000 | Images: 2,000 | Detections: 500
Total benchmark time: 16.7s
Status: SUCCESS
```

---

## 4. Analysis and Findings

### 4.1 Why Single GPU Mode is Fastest

Single GPU mode achieves 18x higher throughput than distributed mode because:
- Zero serialization overhead (no pickling model weights or data)
- Zero network transfer (everything stays in GPU VRAM)
- CUDA streams enable concurrent execution of all 10 models
- All models fit in T4's 16GB VRAM (only 1.97GB needed)

### 4.2 Why Hybrid Matches Single GPU

The hybrid scheduler places all 10 models on GPU because total memory (1.97GB) fits within the T4's 16GB budget. Zero models spill to CPU. In effect, hybrid mode degenerates to single GPU mode. Hybrid would outperform single GPU only when VRAM is insufficient (e.g., on a 4GB GPU).

### 4.3 When Distributed Mode is Advantageous

Distributed mode's overhead (Spark session, serialization, network, task scheduling) makes it slower for small-to-medium datasets. It becomes advantageous when:
- Dataset exceeds single-machine RAM (>16GB on g4dn.xlarge)
- Multiple GPU workers are available (linear scaling with N GPUs)
- Models exceed single GPU VRAM (partition models across nodes)
- Fault tolerance is needed (Spark retries failed tasks)

### 4.4 Scaling Limits Hit During POC

| Limit | Cause | Resolution |
|-------|-------|-----------|
| Driver OOM (Java heap) | 8GB master trying to serialize 2+GB data | Use larger instance (m5.xlarge+) or read from S3 |
| spark.rpc.message.maxSize | RDD elements > 128MB default | Set to 512MB |
| Broadcast OOM | Broadcasting multi-GB data to all executors | Use RDD partitioning instead |
| Executor ModuleNotFoundError | Python path not set on remote workers | Set spark.executorEnv.PYTHONPATH |
| File-based /tmp approach | /tmp not shared across nodes | Not applicable for multi-node |

---

## 5. Architecture Validated

```
┌─────────────────────────────────────────────────────────┐
│                    SPARK CLUSTER                          │
├─────────────────────┬───────────────────────────────────┤
│   Master (t3.large) │   GPU Worker (g4dn.xlarge)        │
│   - Spark Master    │   - Spark Worker                  │
│   - CPU Worker      │   - NVIDIA T4 GPU                 │
│   - Driver          │   - Executor (runs inference)     │
│   - Port 7077 RPC   │   - Port 8081 Worker UI           │
│   - Port 8080 UI    │   - CUDA 12.1 + PyTorch 2.2      │
│   - Port 4040 App   │   - nvidia-container-toolkit      │
├─────────────────────┴───────────────────────────────────┤
│   Infrastructure: AWS CDK (Python)                       │
│   VPC: Public subnet, 3 AZs, Internet Gateway           │
│   Security: Security Group (inter-node + admin UI)       │
│   Monitoring: CloudWatch Dashboard                       │
│   Storage: S3 (code, results)                            │
│   Access: SSM Session Manager (no SSH keys)              │
└─────────────────────────────────────────────────────────┘
```

---

## 6. All Challenges Encountered (18 Total)

| # | Challenge | Root Cause | Resolution |
|---|-----------|-----------|-----------|
| 1 | Node.js v14 can't run CDK | CDK needs Node 18+ | Install Node 20 |
| 2 | AWS credentials not found | No CLI configured | Set $env vars |
| 3 | CDK bootstrap missing | First deploy to region | `cdk bootstrap` |
| 4 | Non-ASCII in security group | Em-dash in description | Use plain ASCII |
| 5 | S3 bucket delete failure | CloudFormation rollback bug | `--retain-resources` |
| 6 | Ubuntu AMI not resolvable | SSM parameter not available | Use Amazon Linux 2023 |
| 7 | Disk space exhaustion (30GB) | Docker image too large | Increased to 100GB |
| 8 | g4dn.xlarge not in us-east-1a | Single AZ VPC | Multi-AZ VPC |
| 9 | GPU vCPU quota = 0 | Default account limit | Switch to ap-south-1 |
| 10 | NVIDIA driver compile failed | Kernel header mismatch | Use dnf RPM packages |
| 11 | Docker --gpus all silent fail | nvidia-container-toolkit missing | Install + configure |
| 12 | Spark not in Docker image | Dockerfile missing Spark | Add Spark 3.5.1 install |
| 13 | Benchmark only ran CPU | Driver has no GPU | Run GPU modes on worker |
| 14 | Spark hardcoded local[4] | Ignored cluster master URL | Read SPARK_MASTER_URL env |
| 15 | Clock skew (signature error) | System time drifted | Sync time |
| 16 | $ARTIFACTS_BUCKET empty | Env vars don't persist across sessions | Set manually |
| 17 | Executor ModuleNotFoundError | Python path not set on workers | Set PYTHONPATH + sys.path |
| 18 | Data broadcast OOM / task size | Broadcasting GBs of data | RDD partitioning + maxSize=512 |

---

## 7. Cost Summary

| Duration | Cost |
|----------|------|
| t3.large (master, ~4 hours) | ~$0.33 |
| g4dn.xlarge (GPU worker, ~4 hours) | ~$2.10 |
| S3 storage | < $0.01 |
| Data transfer (intra-region) | $0.00 |
| **Total POC cost** | **~$2.50** |

---

## 8. Recommendations for Production

1. **Driver instance:** Use m5.2xlarge (32GB RAM) or larger for production workloads
2. **Data ingestion:** Read from S3/HDFS instead of generating in-memory
3. **Multiple GPUs:** Add 4+ g4dn.xlarge workers for linear scaling
4. **Model caching:** Pre-bake model weights into Docker image
5. **NVIDIA MPS:** Enable for multi-process GPU sharing across executors
6. **Monitoring:** Deploy CloudWatch Agent + custom metrics publishers
7. **Auto-shutdown:** Keep safety net to prevent forgotten GPU instances
