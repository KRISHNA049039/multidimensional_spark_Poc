# Air-Gapped System Troubleshooting Guide

**System:** 2 nodes (expanding to 5), 256 GB RAM, 24 GB VRAM, 4 TB HDD, NVIDIA GPU
**Issues encountered and solutions**

---

## Issue 1: Container Cannot Identify GPUs

### Symptom
```bash
docker run --gpus all multi-model-inference:latest nvidia-smi
# ERROR: could not select device driver "" with capabilities: [[gpu]]
```
Or inside container:
```python
torch.cuda.is_available()  # Returns False
```

### Root Cause Chain

There are 4 things that must ALL be working for Docker to see GPUs:

```
1. NVIDIA Driver (kernel module)
   └── 2. nvidia-smi works on HOST
        └── 3. nvidia-container-toolkit installed
             └── 4. Docker daemon configured with nvidia runtime
                  └── 5. Container started with --gpus all
```

If ANY step fails, the container can't see the GPU.

### Diagnosis Steps

Run these **on the host** (not inside container):

```bash
# Step 1: Does the driver work?
nvidia-smi
# If fails: driver not installed or wrong version

# Step 2: Is nvidia-container-toolkit installed?
dpkg -l nvidia-container-toolkit 2>/dev/null || rpm -q nvidia-container-toolkit
# If fails: toolkit not installed

# Step 3: Is Docker configured?
cat /etc/docker/daemon.json
# Should contain "nvidia" runtime or "default-runtime": "nvidia"

# Step 4: Test GPU in container
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# If this fails: toolkit/daemon config issue
# If this works but YOUR image fails: image-specific issue
```

### Fix

```bash
# 1. Install NVIDIA driver (if nvidia-smi fails)
# Ubuntu:
apt-get install -y nvidia-driver-535
# RHEL/Amazon Linux:
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf module install -y nvidia-driver:latest-dkms

# 2. Install nvidia-container-toolkit
# Ubuntu:
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
# RHEL:
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  tee /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit

# 3. Configure Docker runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 4. Verify
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Air-Gapped Fix (No Internet)

Pre-download these on an internet machine:
```bash
# Ubuntu: download .deb packages
apt-get download nvidia-container-toolkit nvidia-container-toolkit-base \
  libnvidia-container1 libnvidia-container-tools

# Transfer to air-gapped, then install:
dpkg -i libnvidia-container1_*.deb libnvidia-container-tools_*.deb \
  nvidia-container-toolkit-base_*.deb nvidia-container-toolkit_*.deb
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### Common Gotchas

| Problem | Cause | Fix |
|---------|-------|-----|
| nvidia-smi works but Docker can't see GPU | nvidia-container-toolkit missing | Install toolkit |
| Toolkit installed but `--gpus all` fails | Docker daemon not configured | Run `nvidia-ctk runtime configure --runtime=docker` + restart Docker |
| Works with `nvidia/cuda` base image but not yours | Your image missing CUDA libs | Ensure Dockerfile uses `FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04` |
| Driver version mismatch | Host driver too old for container CUDA | Host driver must be >= container CUDA version |
| "NVIDIA driver was not detected" warning | Non-fatal warning in older images | Ignore if `torch.cuda.is_available()=True` |

---

## Issue 2: Java OOM (Out of Memory) with 256 GB RAM

### Symptom
```
java.lang.OutOfMemoryError: Java heap space
```
Or:
```
ERROR TorrentBroadcast: Store broadcast fail
```

### Root Cause

The JVM (Spark driver/executor) has a FIXED heap size that's much smaller than your physical RAM. Even with 256 GB, if `spark.driver.memory=10g`, the JVM only uses 10 GB.

```
Your system: 256 GB RAM
  ├── OS + Docker: ~4 GB
  ├── Spark Driver JVM: spark.driver.memory (default: 1g, ours: 10g)
  ├── Spark Executor JVM: spark.executor.memory (ours: 12g)
  ├── Python worker memory: spark.python.worker.memory (ours: 2g)
  └── Remaining: 228 GB UNUSED by Spark!
```

### Fix: Increase JVM Memory

Edit `inference/cluster_engine.py` (or pass via spark-submit):

```python
# For 256 GB RAM system:
.config("spark.driver.memory", "64g")       # Was 10g
.config("spark.executor.memory", "100g")    # Was 12g
.config("spark.driver.maxResultSize", "10g") # Was 2g
.config("spark.rpc.message.maxSize", "2048") # Was 512
.config("spark.python.worker.memory", "8g")  # Was 2g
```

Or via command line:
```bash
spark-submit \
  --driver-memory 64g \
  --executor-memory 100g \
  --conf spark.rpc.message.maxSize=2048 \
  --conf spark.driver.maxResultSize=10g \
  benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4
```

### Memory Budget for 256 GB System

```
Total: 256 GB
  ├── OS + system:       8 GB
  ├── Docker overhead:   4 GB
  ├── Spark Driver:     64 GB  (handles data serialization + RDD creation)
  ├── Spark Executor:  100 GB  (model loading + inference buffers)
  ├── Python workers:   16 GB  (PyTorch tensors in worker subprocess)
  ├── GPU VRAM:         24 GB  (separate, not from system RAM)
  └── Reserve:          64 GB  (headroom for spikes)
```

### Which OOM? (Driver vs Executor)

| Error Location | Cause | Fix |
|---------------|-------|-----|
| `broadcast` / `parallelize` | Driver OOM — data too large for driver heap | Increase `spark.driver.memory` |
| `Task X failed` + OOM | Executor OOM — models + data don't fit | Increase `spark.executor.memory` |
| `Python worker` + MemoryError | Python subprocess out of memory | Increase `spark.python.worker.memory` |
| `maxResultSize` exceeded | Results too large to collect back | Increase `spark.driver.maxResultSize` |
| `rpc.message.maxSize` exceeded | Single task serialized data > limit | Increase `spark.rpc.message.maxSize` |

---

## Issue 3: Which Script to Run — run_benchmark.py vs cluster_benchmark.py

### Decision Matrix

| What You Want | Script | Run From | Example |
|---------------|--------|----------|---------|
| Quick single-mode test (GPU/CPU/hybrid) | `run_benchmark.py` | Any node directly | `python benchmark/run_benchmark.py --mode single_gpu` |
| Distributed test across cluster | `cluster_benchmark.py` | Master (driver) | `SPARK_MASTER_URL=spark://master:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only` |
| All 3 distributed modes, increasing load | `cluster_benchmark.py --incremental` | Master | `SPARK_MASTER_URL=spark://master:7077 python benchmark/cluster_benchmark.py --incremental` |
| Single GPU + Hybrid (no Spark) | `run_benchmark.py` | GPU node directly | `python benchmark/run_benchmark.py --mode all` |

### When to Use Each

```
run_benchmark.py:
  ├── --mode single_gpu    → Runs on local GPU, no Spark, fastest
  ├── --mode hybrid        → Runs on local GPU+CPU, no Spark
  ├── --mode distributed   → Uses Spark (needs SPARK_MASTER_URL)
  └── --mode all           → Runs all 3 modes sequentially

cluster_benchmark.py:
  ├── --device-mode gpu_only    → Distributed, forces GPU on executors
  ├── --device-mode cpu_only    → Distributed, forces CPU on executors
  ├── --device-mode hybrid      → Distributed, auto-detects per executor
  └── --incremental             → All 3 modes × 3 load levels (9 runs)
```

### Key Difference

| | run_benchmark.py | cluster_benchmark.py |
|---|---|---|
| Uses Spark? | Only in `--mode distributed` | Always (all modes use Spark) |
| mapPartitions? | No (old code uses `map`) | Yes (optimized, models loaded once) |
| Detailed executor stats? | No | Yes (partition timing, device info) |
| Captures Spark UI? | No | Yes (jobs, stages, executors JSON) |
| CLI flexibility? | Basic (mode, samples) | Full (device-mode, partitions, batch-size) |

### Recommendation

- **For presentation/benchmarking:** Use `cluster_benchmark.py` (better stats)
- **For quick validation:** Use `run_benchmark.py --mode single_gpu` (no cluster needed)
- **For comparing GPU vs CPU in cluster:** Use `cluster_benchmark.py` with different `--device-mode`

---

## Issue 4: Port 4040 Not Opening

### What Port 4040 Is

Port 4040 is the **Spark Application UI**. It's different from port 8080 (Master UI):

| Port | What | When Active | Shows |
|------|------|-------------|-------|
| 8080 | Spark Master UI | Always (while master runs) | Workers, apps list |
| 4040 | Application UI | **Only during a running job** | Jobs, stages, tasks, executors |
| 18080 | History Server | After enabling event logging | Past completed jobs |

### Why It's Not Available

**4040 only exists while a SparkSession is active.** The moment `spark.stop()` is called (or the benchmark finishes), port 4040 disappears.

```
Timeline:
  benchmark starts → SparkSession created → port 4040 OPENS
  benchmark running → port 4040 ACTIVE (can view jobs/stages)
  benchmark finishes → spark.stop() → port 4040 CLOSES immediately
```

### Fix: Access During Execution

**Option A: Keep the session open longer (for interactive debugging)**

```python
# At the end of your benchmark, BEFORE spark.stop():
input("Press Enter to stop Spark (port 4040 is available now)...")
spark.stop()
```

**Option B: Use the History Server (persists after job completes)**

```bash
# Before running the benchmark, create events dir:
mkdir -p /tmp/spark-events

# Add to Spark config (in cluster_engine.py or spark-defaults.conf):
.config("spark.eventLog.enabled", "true")
.config("spark.eventLog.dir", "/tmp/spark-events")

# After benchmark completes, start History Server:
start-history-server.sh
# Access at port 18080 — shows ALL completed jobs permanently
```

**Option C: Our cluster_benchmark.py already captures the stats**

The `_capture_spark_ui_stats()` function grabs all executor/job/stage data via REST API BEFORE `spark.stop()`. Results are saved in the JSON output file.

### Firewall/Network Issues (Port Accessible but Not Loading)

```bash
# Check if port is listening:
ss -tlnp | grep 4040

# If listening but browser can't reach:
iptables -F    # Flush firewall (air-gapped LAN, safe to do)

# Or specific rule:
iptables -A INPUT -p tcp --dport 4040 -j ACCEPT
```

---

## Issue 5: Job Hanging at Stage 0 [0+0/12]

### What `[0+0/12]` Means

```
[Stage 0:>     (0 + 0) / 12]
               │   │     │
               │   │     └── Total tasks in this stage
               │   └── Tasks currently running
               └── Tasks completed
```

`(0 + 0) / 12` = 0 completed, 0 running, 12 total = **NO tasks are being scheduled**

### Root Causes

| Cause | How to Diagnose | Fix |
|-------|----------------|-----|
| No executors connected | Check Spark Master UI (8080) → Workers = 0 | Start worker containers |
| Executors not accepting tasks | Workers shown but "Cores Used = 0" | Check executor.cores vs task.cpus config |
| Data serialization stuck | Driver creating RDD takes forever | Reduce data size or increase driver memory |
| Network partition | Worker can't reach master | Check firewall, ping between nodes |
| Executor OOM on first task | Executor crashed immediately | Check worker logs, increase executor.memory |

### Diagnosis Steps

```bash
# 1. Check if workers are alive
curl -s http://localhost:8080/json/ | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'Workers: {d.get(\"aliveworkers\",0)}')
print(f'Cores: {d.get(\"coresinuse\",0)}/{d.get(\"cores\",0)}')
for w in d.get('workers',[]):
    print(f'  {w[\"host\"]}:{w[\"port\"]} state={w[\"state\"]} cores={w[\"cores\"]}')
"

# 2. Check executor logs on worker
docker logs spark-gpu-worker --tail 20

# 3. Check if tasks.cpus > executor.cores (prevents scheduling)
# If executor.cores=2 and task.cpus=4, NO task can ever run!

# 4. Network check from worker to master
docker exec spark-gpu-worker bash -c "curl -s http://MASTER_IP:8080/json/ | head -1"
```

### Common Fixes

**Fix 1: Executor cores mismatch**
```python
# WRONG: executor has 2 cores but task needs 4 → stuck
.config("spark.executor.cores", "2")
.config("spark.task.cpus", "4")  # No executor has enough cores!

# RIGHT: task.cpus <= executor.cores
.config("spark.executor.cores", "4")
.config("spark.task.cpus", "2")   # 2 concurrent tasks per executor
```

**Fix 2: Worker not started or crashed**
```bash
# Check worker container status
docker ps -a --filter name=spark

# If exited/restarting, check logs:
docker logs spark-gpu-worker

# Common: GPU access fails → container exits → no executor
# Fix: ensure --gpus all and nvidia-container-toolkit configured
```

**Fix 3: Too many partitions for available resources**
```
12 partitions but only 2 executor slots → 6 rounds needed
If each round takes 10s → 60s total
Appears "stuck" but is actually just slow

Fix: reduce partitions to match executor count × 2
     e.g., 2 executors × 2 = 4 partitions max for responsive UI
```

**Fix 4: Data too large to serialize (stuck at creation)**
```
If [0+0/12] and NO progress for minutes:
  The driver is stuck CREATING the RDD (serializing data)
  This happens before any task is dispatched

Fix: Reduce --signal-samples or increase --partitions
     or increase spark.driver.memory
```

---

## Issue 6: General Performance Troubleshooting

### Throughput Lower Than Expected

| Expected | Actual | Likely Cause |
|----------|--------|-------------|
| 30,000 /sec | 2,000 /sec | Spark overhead dominates (small data) |
| 30,000 /sec | 1,500 /sec | Running on CPU not GPU |
| 30,000 /sec | 500 /sec | Model downloading weights at runtime |
| 26,000 /sec executor | 3,000 /sec overall | CPU executor bottleneck |

### Checklist for Air-Gapped System

```bash
# 1. Verify GPU accessible in container
docker exec spark-gpu-worker python -c "import torch; print(torch.cuda.is_available())"
# Must print: True

# 2. Verify model weights pre-loaded (no download)
docker exec spark-gpu-worker ls /root/.cache/torch/hub/checkpoints/
# Must show: resnet18-f37072fd.pth, mobilenet_v3_small-047dcff4.pth, efficientnet_b0_rwightman-7f5810bc.pth

# 3. Verify Spark cluster connectivity
curl -s http://MASTER_IP:8080/json/ | python3 -c "import sys,json; print(json.load(sys.stdin)['aliveworkers'])"
# Must show: 2 (or however many workers)

# 4. Check executor device in logs during run
docker logs spark-gpu-worker 2>&1 | grep "Executor"
# Must show: device=cuda

# 5. Memory config adequate
docker exec spark-master bash -c "cat /opt/spark/conf/spark-defaults.conf" 2>/dev/null || echo "Using inline config"
```

---

## Quick Reference: Recommended Settings for 256 GB / 24 GB VRAM System

```python
# cluster_engine.py settings for air-gapped 256GB nodes:
.config("spark.driver.memory", "64g")
.config("spark.executor.memory", "100g")
.config("spark.executor.cores", "8")
.config("spark.task.cpus", "2")
.config("spark.rpc.message.maxSize", "2048")
.config("spark.driver.maxResultSize", "10g")
.config("spark.python.worker.memory", "8g")
.config("spark.network.timeout", "600s")
.config("spark.executor.heartbeatInterval", "120s")
```

Docker run command:
```bash
docker run -d --name spark-gpu-worker \
  --network host \
  --gpus all \
  --shm-size=16g \
  -v /opt/model-weights:/root/.cache/torch/hub/checkpoints \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://MASTER_IP:7077 -c 8 -m 200g && tail -f /opt/spark/logs/*worker*"
```

---

## Summary: Decision Tree

```
Problem: "Container can't see GPU"
  → Run nvidia-smi on host → fails? → Install driver
  → nvidia-smi works? → Check nvidia-container-toolkit
  → Toolkit installed? → Run nvidia-ctk runtime configure && restart docker
  → Still fails? → Check Dockerfile base image (needs nvidia/cuda:*)

Problem: "Java OOM"
  → Check which OOM (driver vs executor)
  → Increase the corresponding memory config
  → For 256GB system: driver=64g, executor=100g

Problem: "Which script to run?"
  → Single node test? → run_benchmark.py --mode single_gpu
  → Cluster test with stats? → cluster_benchmark.py --device-mode gpu_only

Problem: "Port 4040 not opening"
  → Job running? → Yes → firewall issue (iptables -F)
  → Job finished? → Port 4040 closes after spark.stop(). Use History Server.

Problem: "Stuck at (0+0/N)"
  → Check alive workers on :8080
  → Check task.cpus <= executor.cores
  → Check executor logs for OOM/crash
  → Reduce data size or increase partitions
```
