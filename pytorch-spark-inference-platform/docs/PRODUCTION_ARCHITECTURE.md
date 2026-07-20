# Production Architecture — 10 Concurrent Models in Air-Gapped Spark Cluster

## Document Purpose

This document captures:
1. All issues encountered during Windows lab cluster setup and their solutions
2. Recommended production architecture for running 10 ML models concurrently in an air-gapped environment
3. GPU distribution strategy for maximum throughput

---

## Part 1: Issues Encountered During Lab Testing & Solutions

### Issue 1: PowerShell Backtick Syntax

**Problem:** `docker run` with backtick-wrapped image name caused `invalid reference format`

**Root Cause:** PowerShell interprets backtick (`` ` ``) as escape character. Wrapping image name in backticks corrupts the reference.

**Solution:** Run as single line or ensure backtick is only at line end (as line continuation):
```powershell
# WRONG
docker run -d `multi-model-inference:latest`

# RIGHT
docker run -d multi-model-inference:latest bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
```

---

### Issue 2: Container Name Conflict

**Problem:** `container name "/spark-master" is already in use`

**Root Cause:** Previous container (from compose or manual run) wasn't removed.

**Solution:**
```powershell
docker rm -f spark-master
```

---

### Issue 3: start-master.sh with `-h <host-ip>` Fails

**Problem:** Netty binding error when passing `-h 192.168.1.100` to `start-master.sh`

**Root Cause:** The container's network namespace doesn't have the host IP. Spark tries to bind its socket to that IP inside the container and fails because only Docker's internal IP (172.x.x.x) exists inside.

**Solution:** Don't pass `-h` flag. Let Spark bind to `0.0.0.0` inside the container. Port mapping (`-p 7077:7077`) handles external access:
```powershell
docker run -d --name spark-master -p 7077:7077 -p 8080:8080 -p 4040:4040 multi-model-inference:latest bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
```

---

### Issue 4: Worker Netty Binding Error

**Problem:** Worker crashes with same Netty error when `SPARK_LOCAL_IP` or `SPARK_WORKER_HOST` is set to host's real IP.

**Root Cause:** Same as Issue 3 — container can't bind to an IP that exists only on the host.

**Solution:** Don't set `SPARK_LOCAL_IP` or `SPARK_WORKER_HOST`. For same-network machines, workers connect to master's host IP (port-forwarded into container):
```powershell
docker run -d --name spark-worker -p 8081:8081 multi-model-inference:latest bash -c "start-worker.sh spark://192.168.4.100:7077 -c 4 -m 8g && tail -f /opt/spark/logs/*worker*"
```

---

### Issue 5: Machines on Different Subnets Can't Communicate

**Problem:** Master on `192.168.1.100`, worker on `10.181.22.233` — connection timed out.

**Root Cause:** Different subnets (different physical networks). No route between them.

**Solution:** Found both machines had a common Ethernet interface (`192.168.4.x`). Used that subnet instead. Always verify connectivity first:
```powershell
# From worker machine
Test-NetConnection 192.168.4.100 -Port 7077
```

**Lesson:** Run `ipconfig | findstr "IPv4 Subnet"` on both machines and find a common subnet.

---

### Issue 6: `--network host` Doesn't Work on Windows Docker Desktop

**Problem:** `network_mode: host` in docker-compose doesn't expose ports on Windows.

**Root Cause:** Docker Desktop runs containers inside a Linux VM (WSL2). `host` mode shares the VM's network, not the Windows host's network.

**Solution:** Use explicit port mapping (`-p`) and Docker's bridge network with service names:
```yaml
services:
  spark-master:
    ports:
      - "7077:7077"
      - "8080:8080"
  spark-worker:
    command: bash -c "start-worker.sh spark://spark-master:7077 ..."
```

---

### Issue 7: GPU Not Detected Inside Container

**Problem:** `WARNING: The NVIDIA Driver was not detected. GPU functionality will not be available.`

**Root Cause:** Container started without `--gpus all` flag.

**Solution:**
```powershell
docker run -d --name spark-master --gpus all --shm-size=4g -p 7077:7077 -p 8080:8080 -p 4040:4040 multi-model-inference:latest bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
```

**Prerequisites:**
- NVIDIA driver installed on host (`nvidia-smi` must work)
- Docker Desktop → Settings → Resources → enable GPU support (WSL2 backend)

---

### Issue 8: New Files Not Found in Container

**Problem:** `python: can't open file '/app/benchmark/quick_compare.py': No such file or directory`

**Root Cause:** Docker image was built before the new files were created. Image is frozen at build time.

**Solution:** Copy files into running container or rebuild:
```powershell
# Quick fix
docker cp benchmark/quick_compare.py spark-master:/app/benchmark/quick_compare.py

# Permanent fix
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
```

---

### Issue 9: `rsync: command not found` Warning

**Problem:** Workers log `rsync: command not found` during startup.

**Root Cause:** Spark daemon script tries to rsync configuration from master. Not needed since all containers use the same image.

**Solution:** Ignore — purely cosmetic, doesn't affect functionality.

---

### Issue 10: Spark UI Port 4040 Only Available During Job

**Problem:** `http://localhost:4040` doesn't work after benchmark completes.

**Root Cause:** Port 4040 is the Application UI, only alive while a Spark application is running. When the job finishes, the port closes.

**Solution:** Use port 8080 (Master UI, always alive) for cluster status. Capture 4040 stats programmatically during the run using `capture_spark_stats.py`.

---

## Part 2: Working Lab Setup (Proven)

### Single Machine (Docker Compose)

```powershell
cd D:\Spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform
docker compose -f deploy/docker-compose.cluster.yml up --scale spark-cpu-worker=3
```

### Multi-Machine (2 nodes, same Ethernet subnet 192.168.4.x)

**Master (192.168.4.100):**
```powershell
docker run -d --name spark-master --gpus all --shm-size=4g -p 7077:7077 -p 8080:8080 -p 4040:4040 multi-model-inference:latest bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
```

**Worker (192.168.4.102):**
```powershell
docker run -d --name spark-worker --gpus all --shm-size=4g -p 8081:8081 multi-model-inference:latest bash -c "start-worker.sh spark://192.168.4.100:7077 -c 4 -m 8g && tail -f /opt/spark/logs/*worker*"
```

---

## Part 3: Production Architecture for Air-Gapped Deployment

### 3.1 Hardware Specification

| Component | Per Node | 5-Node Total |
|-----------|----------|--------------|
| GPU | 1× 24GB VRAM (A5000/A6000/L40) | 120 GB VRAM |
| RAM | 256 GB | 1.28 TB |
| Storage | 4 TB NVMe SSD | 20 TB |
| CPU | 16+ cores | 80+ cores |
| Network | 10 GbE (minimum) | Low-latency LAN |
| OS | Ubuntu 22.04 / RHEL 8+ | — |

### 3.2 Cluster Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AIR-GAPPED LAN (10 GbE)                               │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │    NODE 1         │   │    NODE 2         │   │    NODE 3         │        │
│  │  SPARK MASTER     │   │  SPARK WORKER     │   │  SPARK WORKER     │        │
│  │  + Worker         │   │  + GPU Executor   │   │  + GPU Executor   │        │
│  │  + Driver         │   │                   │   │                   │        │
│  │                   │   │  Models:           │   │  Models:           │        │
│  │  Models:          │   │  - ew_classifier   │   │  - ew_classifier   │        │
│  │  ALL 10 (local)   │   │  - signal_denoiser │   │  - signal_denoiser │        │
│  │                   │   │  - threat_prior.   │   │  - threat_prior.   │        │
│  │  24GB VRAM        │   │  - rf_fingerprint  │   │  - rf_fingerprint  │        │
│  │  :7077 :8080      │   │  - anomaly_detect  │   │  - anomaly_detect  │        │
│  │  :4040            │   │  - resnet18        │   │  - resnet18        │        │
│  │                   │   │  - mobilenetv3     │   │  - mobilenetv3     │        │
│  └──────────────────┘   │  - efficientnet_b0 │   │  - efficientnet_b0 │        │
│                          │  - yolov8_nano     │   │  - yolov8_nano     │        │
│  ┌──────────────────┐   │  - yolov8_small    │   │  - yolov8_small    │        │
│  │    NODE 4         │   │                   │   │                   │        │
│  │  SPARK WORKER     │   │  24GB VRAM        │   │  24GB VRAM        │        │
│  │  + GPU Executor   │   └──────────────────┘   └──────────────────┘        │
│  │                   │                                                       │
│  │  Models: ALL 10   │   ┌──────────────────┐                               │
│  │  24GB VRAM        │   │    NODE 5         │                               │
│  │                   │   │  SPARK WORKER     │                               │
│  └──────────────────┘   │  + GPU Executor   │                               │
│                          │  Models: ALL 10   │                               │
│                          │  24GB VRAM        │                               │
│                          └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Model Memory Budget (All 10 Models)

| # | Model | Category | Est. VRAM | Input Shape |
|---|-------|----------|-----------|-------------|
| 1 | ew_classifier | Signal | 50 MB | (128,) |
| 2 | signal_denoiser | Signal | 100 MB | (128,) |
| 3 | threat_prioritizer | Signal | 350 MB | (128,) |
| 4 | rf_fingerprinter | Signal | 120 MB | (128,) |
| 5 | anomaly_detector | Signal | 100 MB | (128,) |
| 6 | resnet18 | Image Classification | 300 MB | (3,224,224) |
| 7 | mobilenetv3 | Image Classification | 150 MB | (3,224,224) |
| 8 | efficientnet_b0 | Image Classification | 200 MB | (3,224,224) |
| 9 | yolov8_nano | Object Detection | 200 MB | (3,640,640) |
| 10 | yolov8_small | Object Detection | 400 MB | (3,640,640) |
| | **TOTAL** | | **~2 GB** | |

**Conclusion:** All 10 models fit in ~2GB VRAM. A 24GB GPU can hold all 10 models + batch data with room to spare. No CPU overflow needed.

---

### 3.4 Recommended Architecture: GPU-First with Spark Distribution

```
                    ┌──────────────────────────────────┐
                    │         INCOMING DATA             │
                    │  EW Signals + Imagery + Video     │
                    └───────────────┬──────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────┐
                    │      SPARK DRIVER (Node 1)        │
                    │                                    │
                    │  1. Receives sensor data stream   │
                    │  2. Creates RDD partitions        │
                    │  3. Distributes to executors      │
                    │  4. Collects results              │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │  EXECUTOR 1   │ │  EXECUTOR 2   │ │  EXECUTOR N   │
        │  (Node 2 GPU) │ │  (Node 3 GPU) │ │  (Node 5 GPU) │
        │               │ │               │ │               │
        │ Load 10 models│ │ Load 10 models│ │ Load 10 models│
        │ onto GPU once │ │ onto GPU once │ │ onto GPU once │
        │               │ │               │ │               │
        │ Process batch:│ │ Process batch:│ │ Process batch:│
        │ Signal→5 models│ │ Signal→5 models│ │ Signal→5 models│
        │ Image→3 models│ │ Image→3 models│ │ Image→3 models│
        │ Detect→2 models│ │ Detect→2 models│ │ Detect→2 models│
        │               │ │               │ │               │
        │ Return results│ │ Return results│ │ Return results│
        └───────────────┘ └───────────────┘ └───────────────┘
```

**How it works:**
1. Data arrives at the driver (Node 1)
2. Driver partitions data into N chunks (N ≥ number of workers)
3. Each executor receives a partition, loads all 10 models onto its local GPU (cached after first load)
4. Each model runs inference on its relevant data type from the partition
5. Results stream back to the driver
6. Driver aggregates and outputs final classifications/detections

### 3.5 Why This Architecture

| Design Choice | Reason |
|---------------|--------|
| All 10 models on every GPU | Only 2GB total — fits easily in 24GB. Avoids network transfer of model weights. |
| Spark distributes DATA, not models | Data is the bottleneck, not model size. Partitioning data across 5 GPUs gives ~5× throughput. |
| GPU-only mode (no CPU inference) | 24GB VRAM is sufficient. GPU is 10-50× faster than CPU for neural network inference. |
| `--network host` on Linux | Eliminates Docker NAT overhead. Direct IP communication. Required for Spark's bidirectional RPC. |
| Models loaded once per executor | First task loads models; subsequent tasks reuse them. Amortizes the 30-60s load time. |
| `--shm-size=4g` | PyTorch uses /dev/shm for IPC between DataLoader workers. Default 64MB causes crashes. |
| NVIDIA MPS (optional) | Multi-Process Service allows multiple Spark executors to share one GPU efficiently. |

### 3.6 Performance Expectations (5-Node Cluster)

| Metric | Single GPU | 5-Node Cluster | Speedup |
|--------|-----------|----------------|---------|
| Signal throughput (5 models) | ~50,000 samples/sec | ~250,000 samples/sec | ~5× |
| Image classification (3 models) | ~500 images/sec | ~2,500 images/sec | ~5× |
| Object detection (2 models) | ~100 frames/sec | ~500 frames/sec | ~5× |
| Total VRAM available | 24 GB | 120 GB | 5× |
| Fault tolerance | None | N-1 workers can fail | — |

---

## Part 4: Production Deployment Steps (Air-Gapped Linux)

### 4.1 Preparation (Internet Machine)

```bash
# Build Docker image
docker build -t multi-model-inference:latest -f deploy/Dockerfile .

# Save image as tarball
docker save multi-model-inference:latest | gzip > multi-model-inference.tar.gz

# Download NVIDIA driver + container toolkit offline packages
# (see docs/AIRGAPPED_5NODE_DEPLOYMENT.md for full package list)
```

Transfer to air-gapped environment via USB/external drive.

### 4.2 Setup on Each Node

```bash
# Load Docker image
docker load < multi-model-inference.tar.gz

# Install NVIDIA driver (offline)
sudo dpkg -i nvidia-driver-*.deb

# Install nvidia-container-toolkit (offline)
sudo dpkg -i nvidia-container-toolkit*.deb
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
nvidia-smi
docker run --rm --gpus all multi-model-inference:latest nvidia-smi
```

### 4.3 Start Cluster

**Node 1 (Master) — e.g., 10.0.0.1:**
```bash
docker run -d --name spark-master --network host --gpus all --shm-size=4g \
  multi-model-inference:latest \
  bash -c "start-master.sh && start-worker.sh spark://10.0.0.1:7077 -c 8 -m 200g && tail -f /opt/spark/logs/*master*"
```

**Nodes 2-5 (Workers) — e.g., 10.0.0.2:**
```bash
docker run -d --name spark-worker --network host --gpus all --shm-size=4g \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://10.0.0.1:7077 -c 16 -m 200g && tail -f /opt/spark/logs/*worker*"
```

Note: `--network host` works properly on Linux (unlike Windows Docker Desktop).

### 4.4 Run Production Inference

```bash
# From Node 1 (master)
docker exec -it spark-master bash -c \
  "SPARK_MASTER_URL=spark://10.0.0.1:7077 python benchmark/cluster_benchmark.py \
    --device-mode gpu_only \
    --partitions 5 \
    --signal-samples 50000 \
    --image-samples 2000 \
    --detection-samples 500 \
    --batch-size 512"
```

### 4.5 Automation Script (Linux)

```bash
#!/bin/bash
# start_cluster.sh — Run on Node 1
MASTER_IP="10.0.0.1"
WORKER_IPS=("10.0.0.2" "10.0.0.3" "10.0.0.4" "10.0.0.5")
IMAGE="multi-model-inference:latest"

# Start master + local worker
docker run -d --name spark-master --network host --gpus all --shm-size=4g \
  $IMAGE bash -c "start-master.sh && start-worker.sh spark://$MASTER_IP:7077 -c 8 -m 200g && tail -f /opt/spark/logs/*master*"

sleep 10

# Start remote workers
for ip in "${WORKER_IPS[@]}"; do
  echo "Starting worker on $ip..."
  ssh $ip "docker run -d --name spark-worker --network host --gpus all --shm-size=4g \
    $IMAGE bash -c 'start-worker.sh spark://$MASTER_IP:7077 -c 16 -m 200g && tail -f /opt/spark/logs/*worker*'"
done

echo "Cluster running. UI at http://$MASTER_IP:8080"
```

---

## Part 5: Optimization for 10 Concurrent Models

### 5.1 CUDA Memory Optimization

```python
# In production, pre-load all models with CUDA graphs for fastest inference
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
torch.cuda.empty_cache()  # Clear unused memory before loading

# Load all 10 models in eval mode with no_grad context
with torch.no_grad():
    for model in models.values():
        model.eval().cuda()
```

### 5.2 CUDA Streams for Parallel Model Execution

```python
# Run independent models in parallel using CUDA streams
streams = [torch.cuda.Stream() for _ in range(10)]

for i, (name, model) in enumerate(models.items()):
    with torch.cuda.stream(streams[i]):
        output = model(batch_data)

# Synchronize all streams
torch.cuda.synchronize()
```

### 5.3 Spark Configuration for GPU Workloads

```python
spark = SparkSession.builder \
    .master("spark://10.0.0.1:7077") \
    .config("spark.executor.memory", "200g") \
    .config("spark.executor.cores", "16") \
    .config("spark.task.cpus", "4") \
    .config("spark.default.parallelism", "20") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "1g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()
```

### 5.4 NVIDIA MPS for Multi-Executor GPU Sharing

On each GPU node:
```bash
# Enable MPS (allows multiple processes to share GPU efficiently)
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
```

With MPS enabled, multiple Spark executors on the same node can share the GPU without context-switching overhead.

---

## Part 6: Monitoring in Production

### 6.1 Metrics to Track

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| GPU Utilization | `nvidia-smi` | < 50% (underutilized) |
| GPU Memory Used | `nvidia-smi` | > 90% (risk of OOM) |
| GPU Temperature | `nvidia-smi` | > 85°C |
| Inference Throughput | Application metrics | < baseline × 0.8 |
| Spark Task Failures | Spark UI / REST API | > 0 |
| Executor Heartbeat | Spark Master | Missing > 60s |
| Network Throughput | `iftop` / `nload` | Saturated 10GbE |

### 6.2 Monitoring Commands

```bash
# GPU status (all nodes)
watch -n 1 nvidia-smi

# Spark cluster status
curl -s http://10.0.0.1:8080/json/ | python -m json.tool

# Capture full benchmark stats
docker exec spark-master python benchmark/capture_spark_stats.py
```

---

## Part 7: Summary of Correct Commands

### Windows Lab (Docker Desktop, port mapping)

```powershell
# Build
docker build -t multi-model-inference:latest -f deploy/Dockerfile .

# Master (with GPU)
docker run -d --name spark-master --gpus all --shm-size=4g -p 7077:7077 -p 8080:8080 -p 4040:4040 multi-model-inference:latest bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"

# Worker on different machine (with GPU)
docker run -d --name spark-worker --gpus all --shm-size=4g -p 8081:8081 multi-model-inference:latest bash -c "start-worker.sh spark://192.168.4.100:7077 -c 4 -m 8g && tail -f /opt/spark/logs/*worker*"

# Run benchmark
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://192.168.4.100:7077 python benchmark/quick_compare.py"
```

### Linux Production (Air-Gapped, --network host)

```bash
# Master
docker run -d --name spark-master --network host --gpus all --shm-size=4g multi-model-inference:latest bash -c "start-master.sh && start-worker.sh spark://10.0.0.1:7077 -c 8 -m 200g && tail -f /opt/spark/logs/*master*"

# Workers
docker run -d --name spark-worker --network host --gpus all --shm-size=4g multi-model-inference:latest bash -c "start-worker.sh spark://10.0.0.1:7077 -c 16 -m 200g && tail -f /opt/spark/logs/*worker*"

# Benchmark
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://10.0.0.1:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 5 --signal-samples 50000"
```

---

## Part 8: Intricacies & Pitfalls — How to Avoid in Air-Gapped Production

These are the subtle issues that cause multi-hour debugging in air-gapped environments where you can't Google solutions. Each was encountered during lab testing.

---

### Pitfall 1: Docker Networking — The #1 Source of Failures

**The Intricacy:** Spark uses bidirectional RPC. The master must reach the worker AND the worker must reach the driver. If either direction fails, tasks hang silently at `(0 + 0) / N`.

**Windows Docker Desktop behavior:**
- `--network host` does NOT work (containers run inside a Linux VM)
- Containers get internal IPs (172.17.x.x) invisible to other physical machines
- `-h <host-ip>` in start-master.sh causes Netty bind failures (IP doesn't exist in container)
- `SPARK_LOCAL_IP=<host-ip>` also causes Netty bind failures for same reason

**Linux (air-gapped production) behavior:**
- `--network host` works perfectly — container shares host's actual network stack
- No port mapping needed, no NAT, no routing confusion
- ALL the Windows issues disappear on Linux with `--network host`

**Air-Gapped Rule:**
```bash
# ALWAYS use --network host on Linux production. Period.
docker run -d --network host --gpus all ...
```

---

### Pitfall 2: Executor Memory vs Worker Memory Mismatch

**The Intricacy:** Spark requests `spark.executor.memory` (default 1GB + overhead ≈ 12GB in our config). If the worker's advertised memory (`-m` flag) is less than this, the app enters `WAITING` state forever. No error message — just silent hang.

**Symptom:** Spark UI shows app state as `"WAITING"`, `coresused: 0`

**Air-Gapped Rule:**
```bash
# Worker memory MUST exceed executor memory request
# Our config requests 12GB per executor, so offer at least 16GB:
start-worker.sh spark://MASTER:7077 -c 16 -m 200g

# With 256GB RAM nodes, offer 200g (leave 56GB for OS + driver)
```

**Verification:** After starting, check Spark UI → Workers tab. Memory column must show > executor memory request.

---

### Pitfall 3: Driver Hostname Resolution (Container ID Routing)

**The Intricacy:** When the driver runs inside a container, Spark advertises the driver's address using the container hostname (a random hex ID like `686832736e42`). Workers on other machines can't resolve this hostname, so executors crash with `exit code 1` in a retry loop.

**Symptom:** Worker logs show executor launching then immediately `finished with state EXITED message Command exited with code 1` repeatedly.

**Air-Gapped Rule (Linux with --network host):**
```bash
# With --network host, the container's hostname IS the machine's hostname.
# Ensure /etc/hosts on every node maps all hostnames to IPs:
echo "10.0.0.1 node1" >> /etc/hosts
echo "10.0.0.2 node2" >> /etc/hosts
echo "10.0.0.3 node3" >> /etc/hosts
echo "10.0.0.4 node4" >> /etc/hosts
echo "10.0.0.5 node5" >> /etc/hosts
```

Or explicitly set hostname:
```bash
docker run --hostname node1 --network host ...
```

Or set the Spark property:
```bash
docker run -e SPARK_LOCAL_IP=10.0.0.1 --network host ...
# This works on Linux --network host because the IP exists on the host's interface
```

---

### Pitfall 4: Firewall Blocks Between Nodes

**The Intricacy:** Spark uses dynamic high ports for executor↔driver communication. Even if port 7077 is open, the cluster will fail if random ephemeral ports (30000-65535) are blocked.

**Symptom:** Worker registers fine (port 7077 works), but tasks never complete (executor can't reach driver on high port).

**Air-Gapped Rule:**
```bash
# On EVERY node, disable firewall (air-gapped = no external threat):
sudo ufw disable
# OR
sudo systemctl stop firewalld && sudo systemctl disable firewalld

# If firewall is mandatory (compliance), open entire range between cluster nodes:
sudo ufw allow from 10.0.0.0/24 to any
```

---

### Pitfall 5: NVIDIA Driver + Container Toolkit Versions Must Match

**The Intricacy:** In air-gapped environments, you download packages offline. If the nvidia-container-toolkit version doesn't match the driver version, `--gpus all` silently fails or the container can't access the GPU.

**Air-Gapped Rule:**
```bash
# BEFORE transferring to air-gap, verify the exact combo works:
nvidia-smi                    # Note driver version (e.g., 535.183.01)
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

# Transfer SAME versions of:
#   - nvidia-driver-535 (exact version)
#   - nvidia-container-toolkit (matching version)
#   - Docker CE (tested version)
#   - Docker image (built with matching CUDA base)

# Pin versions in your offline package list:
nvidia-driver-535=535.183.01-0ubuntu1
nvidia-container-toolkit=1.14.3-1
docker-ce=5:24.0.7-1~ubuntu.22.04~jammy
```

---

### Pitfall 6: PyTorch CUDA Version Must Match Driver CUDA

**The Intricacy:** The Docker image uses `nvidia/cuda:12.1.1-runtime` and installs `torch==2.2.0+cu121`. If the host NVIDIA driver only supports CUDA 11.x, PyTorch will report `cuda is not available` inside the container even with `--gpus all`.

Additionally, newer GPUs (RTX 50-series / Blackwell, sm_120) require PyTorch 2.6+ for kernel support. Older PyTorch versions report `CUDA error: no kernel image is available for execution on the device` even though CUDA is detected as available.

**Verification:**
```bash
# Check max CUDA version supported by driver:
nvidia-smi   # Look at "CUDA Version" in top-right

# Check GPU compute capability:
nvidia-smi --query-gpu=compute_cap --format=csv
# sm_86 (A5000/A6000) → torch 2.2.0 works
# sm_89 (L40/L4/RTX 4090) → torch 2.2.0 works
# sm_90 (H100) → torch 2.2.0 works
# sm_120 (RTX 5060/5070/5090) → needs torch 2.6+
```

**Air-Gapped Rule:** Match the chain: Driver CUDA ≥ Container CUDA ≥ PyTorch CUDA build. Also verify GPU compute capability is in PyTorch's supported list.

**Workaround for unsupported GPUs (lab testing):**
```bash
# Hide GPU entirely, force CPU execution:
CUDA_VISIBLE_DEVICES='' python benchmark/quick_compare.py
```

---

### Pitfall 7: Shared Memory (`/dev/shm`) Too Small

**The Intricacy:** Docker default shared memory is 64MB. PyTorch DataLoader with `num_workers > 0` uses shared memory for IPC. With 10 models and batch processing, 64MB fills instantly → crash with `Bus error (core dumped)`.

**Symptom:** Executor runs for a few seconds then crashes with no clear Python error.

**Air-Gapped Rule:**
```bash
# ALWAYS set --shm-size on GPU inference containers:
docker run --shm-size=4g ...

# For 256GB RAM nodes, can go higher:
docker run --shm-size=16g ...
```

---

### Pitfall 8: Model Weight Serialization Over Spark

**The Intricacy:** Spark serializes everything sent to executors. If you pass PyTorch model objects through Spark RDD operations, it serializes multi-GB model weights for every task. This causes massive network traffic and OOM errors.

**Air-Gapped Rule:**
```python
# WRONG — serializes models for every partition:
rdd.map(lambda data: model(data))

# RIGHT — load models INSIDE the executor (once per executor lifecycle):
def run_on_partition(partition_data):
    # This runs on the worker — loads models locally
    models = load_all_models(device="cuda")  # From local Docker image
    results = []
    for batch in partition_data:
        results.append(inference(models, batch))
    return results

rdd.mapPartitions(run_on_partition)
```

The models are already baked into the Docker image on every node. Load them inside the executor — never send them over the network.

---

### Pitfall 9: Stale Docker Containers on Restart

**The Intricacy:** If a node reboots or Docker restarts, old containers remain in `Exited` state. Running `docker run --name spark-worker` fails with "name already in use". In an air-gapped environment without remote monitoring, this can go unnoticed.

**Air-Gapped Rule:**
```bash
# Use --rm flag for auto-cleanup on exit:
docker run -d --rm --name spark-worker --network host --gpus all ...

# OR use a systemd service for auto-restart:
# /etc/systemd/system/spark-worker.service
[Unit]
Description=Spark Worker
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker rm -f spark-worker
ExecStart=/usr/bin/docker run --rm --name spark-worker --network host --gpus all --shm-size=4g multi-model-inference:latest bash -c "start-worker.sh spark://10.0.0.1:7077 -c 16 -m 200g && tail -f /opt/spark/logs/*worker*"
ExecStop=/usr/bin/docker stop spark-worker

[Install]
WantedBy=multi-user.target
```

---

### Pitfall 10: No DNS in Air-Gapped Network

**The Intricacy:** Air-gapped networks often have no DNS server. Spark relies on hostname resolution for task routing. Without DNS, hostnames resolve to `127.0.0.1` or fail entirely.

**Air-Gapped Rule:**
```bash
# On EVERY node, add all cluster nodes to /etc/hosts:
cat >> /etc/hosts << EOF
10.0.0.1 node1 spark-master
10.0.0.2 node2 spark-worker-1
10.0.0.3 node3 spark-worker-2
10.0.0.4 node4 spark-worker-3
10.0.0.5 node5 spark-worker-4
EOF

# Also set hostname properly:
hostnamectl set-hostname node1  # (on node 1)
```

---

### Pitfall 11: Clock Skew Between Nodes

**The Intricacy:** Air-gapped systems can't reach NTP servers. If clocks drift between nodes, Spark's heartbeat timeouts and task scheduling can behave erratically. Logs become impossible to correlate.

**Air-Gapped Rule:**
```bash
# Set up one node as local NTP server, others sync to it:
# On node1 (master):
sudo apt install chrony
echo "local stratum 10" >> /etc/chrony/chrony.conf
echo "allow 10.0.0.0/24" >> /etc/chrony/chrony.conf
sudo systemctl restart chrony

# On all other nodes:
echo "server 10.0.0.1 iburst" > /etc/chrony/chrony.conf
sudo systemctl restart chrony
```

---

### Pitfall 12: Disk Space Exhaustion from Docker Layers

**The Intricacy:** Docker images are large (~8GB). Building, loading, and running containers creates layers. On 4TB drives this seems fine, but if /var/lib/docker is on a small root partition, it fills up silently.

**Air-Gapped Rule:**
```bash
# Check Docker storage location:
docker info | grep "Docker Root Dir"

# If root partition is small, move Docker to the large drive:
sudo systemctl stop docker
sudo mv /var/lib/docker /data/docker
sudo ln -s /data/docker /var/lib/docker
sudo systemctl start docker

# Periodic cleanup:
docker system prune -f  # Remove unused images/containers/volumes
```

---

### Summary Checklist for Air-Gapped Deployment

| # | Check | Command to Verify |
|---|-------|-------------------|
| 1 | `--network host` on all containers | `docker inspect spark-master \| grep NetworkMode` |
| 2 | Worker memory > executor memory | Spark UI → Workers tab |
| 3 | `/etc/hosts` has all nodes | `cat /etc/hosts` on each node |
| 4 | Firewall disabled or open | `sudo ufw status` / `firewall-cmd --state` |
| 5 | NVIDIA driver matches toolkit | `nvidia-smi` + `docker run --gpus all nvidia-smi` |
| 6 | `--shm-size=4g` or higher | `docker inspect spark-worker \| grep ShmSize` |
| 7 | Clock sync between nodes | `chronyc sources` on each node |
| 8 | Docker root on large partition | `df -h /var/lib/docker` |
| 9 | GPU visible in container | `docker exec spark-worker nvidia-smi` |
| 10 | Models load inside executor (not serialized) | Check inference code uses `mapPartitions` |
| 11 | Containers use `--rm` or systemd | `systemctl status spark-worker` |
| 12 | Port 7077 reachable between nodes | `nc -zv 10.0.0.1 7077` from each worker |

---

## Key Takeaways

1. **Windows Docker Desktop** — never use `--network host` or `-h <ip>` in start-master.sh. Use port mapping.
2. **Linux production** — always use `--network host`. It's simpler and faster.
3. **All 10 models fit in 2GB VRAM** — no need for CPU fallback on 24GB GPUs.
4. **Spark distributes DATA across nodes** — each node has all models loaded locally.
5. **Scale linearly** — 5 GPUs ≈ 5× throughput for embarrassingly parallel inference.
6. **Air-gapped** — transfer Docker image tarball + NVIDIA packages via USB. No internet needed at runtime.
7. **Most failures are networking** — `--network host` + `/etc/hosts` + firewall off eliminates 90% of issues.
8. **Silent hangs mean resource mismatch** — check executor memory vs worker memory in Spark UI.
9. **Use systemd** for auto-restart resilience in production.
10. **Sync clocks** — one node as NTP master, others sync to it.
