# Airgapped Deployment Guide — Multi-Model Inference Platform

Complete guide for deploying the pytorch-spark-inference-platform on systems with **no internet access** (DRDO, classified networks, isolated infrastructure).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites on Target System](#2-prerequisites-on-target-system)
3. [Phase 1: Build on Internet-Connected Machine](#3-phase-1-build-on-internet-connected-machine)
4. [Phase 2: Transfer to Airgapped System](#4-phase-2-transfer-to-airgapped-system)
5. [Phase 3: Deploy Single-Node (One GPU)](#5-phase-3-deploy-single-node-one-gpu)
6. [Phase 4: Deploy Multi-Node Cluster](#6-phase-4-deploy-multi-node-cluster)
7. [Code Updates Without Rebuilding](#7-code-updates-without-rebuilding)
8. [NVIDIA MPS Setup for GPU Sharing](#8-nvidia-mps-setup-for-gpu-sharing)
9. [Verification Checklist](#9-verification-checklist)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

### What Gets Transferred

| Item | Size | Contents |
|---|---|---|
| Docker image (`multi-model-inference.tar.gz`) | ~4 GB | CUDA 12.1 + Python 3.11 + Java 17 + PyTorch 2.2 + all deps + model weights + source code |
| (Optional) Source code tarball | ~100 KB | For code-only updates without Docker rebuild |

### What Runs at Runtime (Zero Internet Required)

- All 10 models with pre-baked weights
- Synthetic data generation (no external data needed)
- All 3 inference modes (single_gpu, hybrid, distributed)
- Results generation (metrics_report.md + raw_results.json)

---

## 2. Prerequisites on Target System

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any NVIDIA GPU with 4+ GB VRAM | Tesla T4 / V100 / A100 (16+ GB) |
| RAM | 8 GB | 16+ GB |
| Storage | 10 GB free | 20+ GB |
| CPU | 2 cores | 4+ cores |

### Software (must be installed on target BEFORE going airgapped)

| Software | Version | Purpose |
|---|---|---|
| Docker Engine | 20.10+ | Container runtime |
| NVIDIA Driver | 525+ | GPU access |
| NVIDIA Container Toolkit | 1.13+ | `--gpus all` flag support |
| (Optional) Docker Compose | 2.0+ | Multi-node orchestration |

### Verify Prerequisites on Target

```bash
# Docker installed?
docker --version

# NVIDIA driver installed?
nvidia-smi

# NVIDIA Container Toolkit working?
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If the last command shows your GPU inside the container, you're ready.

---

## 3. Phase 1: Build on Internet-Connected Machine

### 3.1 Clone and Prepare

```bash
git clone <repo-url>
cd pytorch-spark-inference-platform
```

### 3.2 Pre-Download Pretrained Weights

The image models and YOLO auto-download weights on first use. For airgapped, bake them into the image:

```bash
# Create weights directory
mkdir -p models/weights

# Download torchvision weights
python -c "
import torch, os
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

torch.save(resnet18(weights=ResNet18_Weights.DEFAULT).state_dict(),
           'models/weights/resnet18.pth')
torch.save(mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).state_dict(),
           'models/weights/mobilenetv3.pth')
torch.save(efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).state_dict(),
           'models/weights/efficientnet_b0.pth')
print('TorchVision weights saved to models/weights/')
"

# Download YOLO weights
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"
mv yolov8n.pt yolov8s.pt models/weights/

# Verify
ls -la models/weights/
# Should show: resnet18.pth, mobilenetv3.pth, efficientnet_b0.pth, yolov8n.pt, yolov8s.pt
```

### 3.3 Build Docker Image

```bash
docker build -f deploy/Dockerfile -t multi-model-inference:latest .
```

This takes 10-15 minutes (downloads CUDA base, PyTorch wheels, all pip packages).

### 3.4 Verify Build

```bash
# Quick test — should print 10 models
docker run --rm multi-model-inference:latest \
  python -c "from models import get_default_registry; r=get_default_registry(); r.summary()"
```

### 3.5 Export Docker Image

```bash
# Save as tar (uncompressed: ~6-8 GB)
docker save multi-model-inference:latest -o multi-model-inference.tar

# Compress (reduces to ~3-4 GB, takes a few minutes)
gzip multi-model-inference.tar

# Final file
ls -lh multi-model-inference.tar.gz
# ~3-4 GB
```

### 3.6 (Optional) Package Source Code Separately

For future code-only updates without full image rebuild:

```bash
tar czf inference-platform-source.tar.gz \
  benchmark/ data/ inference/ models/ requirements.txt
# ~100 KB
```

---

## 4. Phase 2: Transfer to Airgapped System

### Transfer Package

```
Files to transfer:
├── multi-model-inference.tar.gz     (~4 GB)  — Full Docker image
├── docker-compose.cluster.yml       (~1 KB)  — Cluster config (optional)
├── inference-platform-source.tar.gz (~100 KB) — Source for future updates (optional)
└── AIRGAPPED_DEPLOYMENT.md          (~10 KB) — This file

Total: ~4 GB
```

### Transfer Methods

| Method | Use When |
|---|---|
| USB drive (encrypted) | Standard secure transfer |
| Approved file transfer system | Network-connected secure transfer boundary |
| Optical media (DVD/Blu-ray) | Maximum security, write-once |
| Air-gap data diode | One-way transfer systems |

### Transfer to Multiple Nodes

If deploying a cluster, transfer the image to **every node** that will run containers.

---

## 5. Phase 3: Deploy Single-Node (One GPU)

### 5.1 Load Docker Image

```bash
# Decompress and load (takes 2-3 minutes)
gunzip -c multi-model-inference.tar.gz | docker load

# Verify
docker images | grep multi-model-inference
# REPOSITORY               TAG       SIZE
# multi-model-inference    latest    ~7 GB
```

### 5.2 Run Full Benchmark

```bash
docker run --rm --gpus all --network host --shm-size=4g \
  -e PYSPARK_PYTHON=python \
  -e PYSPARK_DRIVER_PYTHON=python \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/results:/app/results \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode single_gpu \
  --signal-samples 50000 --image-samples 1000 --detection-samples 200 \
  --batch-size 64
```

### 5.3 Run Specific Modes

```bash
# Single GPU (fastest on one machine)
docker run --rm --gpus all --shm-size=4g \
  -v $(pwd)/results:/app/results \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode single_gpu \
  --signal-samples 50000 --batch-size 64

# Hybrid CPU+GPU (for memory-constrained GPUs)
docker run --rm --gpus all --shm-size=4g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/results:/app/results \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode hybrid \
  --signal-samples 50000 --image-samples 1000 --detection-samples 200 \
  --batch-size 32

# Spark Distributed (local mode, proves Spark pipeline works)
docker run --rm --gpus all --shm-size=4g \
  -e PYSPARK_PYTHON=python \
  -e PYSPARK_DRIVER_PYTHON=python \
  -v $(pwd)/results:/app/results \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode distributed \
  --signal-samples 5000 --image-samples 50 --detection-samples 10 \
  --batch-size 64 --partitions 4
```

### 5.4 View Results

```bash
cat results/metrics_report.md
cat results/raw_results.json
```

---

## 6. Phase 4: Deploy Multi-Node Cluster

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Master Node    │     │  Worker Node 1  │     │  Worker Node 2  │
│  (no GPU req.)  │     │  (GPU required) │     │  (GPU required) │
│                 │     │                 │     │                 │
│  Spark Driver   │◄───►│  Spark Executor │     │  Spark Executor │
│  Port 7077      │     │  + CUDA Streams │     │  + CUDA Streams │
│  Port 8080 (UI) │     │  + MPS daemon   │     │  + MPS daemon   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 6.1 Load Image on All Nodes

```bash
# On every node:
gunzip -c multi-model-inference.tar.gz | docker load
```

### 6.2 Install Apache Spark Standalone (Alternative to Docker-based Spark)

Since our Docker image doesn't include Spark standalone scripts, use PySpark in local mode on each worker independently, or install Spark standalone:

```bash
# Download Spark (do this on internet machine, transfer the tar)
wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
# Transfer spark-3.5.1-bin-hadoop3.tgz to airgapped system

# On each node:
tar xzf spark-3.5.1-bin-hadoop3.tgz -C /opt/
export SPARK_HOME=/opt/spark-3.5.1-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH
```

### 6.3 Start Spark Master

```bash
# On master node:
$SPARK_HOME/sbin/start-master.sh
# Master URL: spark://<MASTER_IP>:7077
# Web UI: http://<MASTER_IP>:8080
```

### 6.4 Start Spark Workers

```bash
# On each worker node:
$SPARK_HOME/sbin/start-worker.sh spark://<MASTER_IP>:7077 -c 4 -m 12g
```

### 6.5 Submit Job to Cluster

```bash
# From master node:
docker run --rm --network host \
  -e PYSPARK_PYTHON=python \
  -e PYSPARK_DRIVER_PYTHON=python \
  -v $(pwd)/results:/app/results \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode distributed \
  --signal-samples 10000 --image-samples 100 --detection-samples 20 \
  --batch-size 64 --partitions 8
```

### 6.6 Independent Worker Mode (Simplest)

If setting up Spark standalone is too complex, run each worker independently and compare results:

```bash
# On each GPU node: run the benchmark independently
docker run --rm --gpus all --shm-size=4g \
  -v $(pwd)/results:/app/results \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode single_gpu \
  --signal-samples 50000 --image-samples 1000 --detection-samples 200 \
  --batch-size 64

# Each node produces its own metrics_report.md
# In production, a data splitter feeds different signal data to each node
```

---

## 7. Code Updates Without Rebuilding

For iterative code changes (bug fixes, parameter tweaks, new models) without rebuilding the entire 4GB Docker image:

### 7.1 On Internet Machine: Package Changes

```bash
# Only package the source code (~100 KB)
cd pytorch-spark-inference-platform
tar czf code-update-v2.tar.gz benchmark/ data/ inference/ models/
```

### 7.2 Transfer to Airgapped System

Transfer `code-update-v2.tar.gz` (~100 KB) via approved method.

### 7.3 Deploy with Bind Mount

```bash
# Extract new code
mkdir -p /opt/inference-platform
tar xzf code-update-v2.tar.gz -C /opt/inference-platform/

# Run with new code overlaid on old Docker image
docker run --rm --gpus all --shm-size=4g \
  -v /opt/inference-platform:/app \
  -v $(pwd)/results:/app/results \
  -e PYSPARK_PYTHON=python \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode single_gpu \
  --signal-samples 50000 --batch-size 64
```

This uses the **new Python code** with the **old Docker image's runtime** (CUDA, PyTorch, Java). No rebuild needed for code-only changes.

### When You DO Need a Full Rebuild

| Change Type | Rebuild Needed? |
|---|---|
| Python code changes (.py files) | No — use bind mount |
| New pip package added to requirements.txt | Yes |
| PyTorch version upgrade | Yes |
| New model weights added | No — bind mount the weights dir |
| Dockerfile changes | Yes |

---

## 8. NVIDIA MPS Setup for GPU Sharing

MPS (Multi-Process Service) enables multiple Spark executor processes to share one GPU truly in parallel. Without MPS, processes serialize at the CUDA context level.

### 8.1 Enable MPS (Run Once Per Boot)

```bash
# On each GPU worker node:
sudo nvidia-cuda-mps-control -d

# Verify
echo get_default_active_thread_percentage | nvidia-cuda-mps-control
# Expected output: 100
```

### 8.2 Persistent MPS via Systemd

Create `/etc/systemd/system/nvidia-mps.service`:

```ini
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

```bash
sudo systemctl enable nvidia-mps
sudo systemctl start nvidia-mps
```

### 8.3 Spark Config for Fractional GPU

Add to `spark-defaults.conf` on the cluster:

```properties
spark.executor.resource.gpu.amount=1
spark.task.resource.gpu.amount=0.1
spark.executorEnv.CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
spark.executorEnv.CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
```

This allows 10 Spark tasks (one per model) to share one physical GPU via MPS.

---

## 9. Verification Checklist

Run these checks after deploying on the airgapped system:

| # | Check | Command | Pass Criteria |
|---|---|---|---|
| 1 | Image loaded | `docker images \| grep multi-model` | Shows `latest` tag |
| 2 | GPU visible | `docker run --rm --gpus all multi-model-inference:latest nvidia-smi` | GPU name + memory shown |
| 3 | Python works | `docker run --rm multi-model-inference:latest python --version` | `Python 3.11.x` |
| 4 | PyTorch works | `docker run --rm multi-model-inference:latest python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| 5 | Models load | `docker run --rm multi-model-inference:latest python -c "from models import get_default_registry; r=get_default_registry(); print(len(r.list_models()))"` | `10` |
| 6 | Inference runs | Run `--mode single_gpu --signal-samples 1000 --batch-size 64` | Throughput > 0 |
| 7 | Results generated | `ls results/` | Both `.md` and `.json` files exist |
| 8 | Spark works | Run `--mode distributed --signal-samples 1000 --partitions 2` | Completes without error |

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker: Error response from daemon: could not select device driver` | NVIDIA Container Toolkit not installed | Install `nvidia-container-toolkit` package |
| `nvidia-smi: command not found` inside container | Missing `--gpus all` | Add `--gpus all` to `docker run` |
| YOLO models output random numbers (fallback CNN) | Weights not pre-baked | Rebuild image with weights in `models/weights/` |
| `CUDA out of memory` | Batch size too large for available VRAM | Reduce `--batch-size` (try 32 or 16) |
| Spark broadcast error (pickle/OOM) | Data too large for driver memory | Reduce `--image-samples` and `--detection-samples` |
| `No module named 'models'` with bind mount | Volume mount path wrong | Ensure `-v` mounts to `/app` exactly |
| Slow first inference batch | GPU warming up (cuDNN autotuning) | Normal — subsequent batches are faster |
| `Permission denied` writing results | Docker user vs host user mismatch | `chmod 777 results/` on host, or run with `--user $(id -u)` |
| MPS not working | `nvidia-cuda-mps-control -d` not run | Run MPS daemon before starting Spark workers |

---

## Quick Reference Card

```
# Load image
gunzip -c multi-model-inference.tar.gz | docker load

# Run benchmark (single GPU, fastest)
docker run --rm --gpus all --shm-size=4g \
  -v $(pwd)/results:/app/results \
  multi-model-inference:latest \
  python /app/benchmark/run_benchmark.py --mode single_gpu \
  --signal-samples 50000 --batch-size 64

# View results
cat results/metrics_report.md

# Update code without rebuild
tar xzf code-update.tar.gz -C /opt/platform/
docker run --rm --gpus all -v /opt/platform:/app ...
```
