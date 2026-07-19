# Air-Gapped 5-Node Spark Inference Cluster — Deployment Guide

**Target Environment:** 5 identical workstations, no internet access
**Hardware per node:** 256 GB RAM, 4 TB HDD/SSD, 24 GB VRAM GPU
**Network:** Isolated LAN (air-gapped), nodes can communicate with each other
**OS:** Linux (Ubuntu 22.04 or RHEL 8/9 recommended)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIR-GAPPED NETWORK (LAN)                       │
├─────────────┬─────────────┬─────────────┬───────────┬───────────┤
│   Node 1    │   Node 2    │   Node 3    │  Node 4   │  Node 5   │
│   MASTER    │   WORKER    │   WORKER    │  WORKER   │  WORKER   │
│  + Worker   │             │             │           │           │
│             │             │             │           │           │
│ Spark Master│ Spark Worker│ Spark Worker│ Spark Wkr │ Spark Wkr │
│ Spark Worker│ GPU Executor│ GPU Executor│ GPU Exec  │ GPU Exec  │
│ Driver      │             │             │           │           │
│ Spark UI    │ Worker UI   │ Worker UI   │ Worker UI │ Worker UI │
│ :8080,:4040 │ :8081       │ :8081       │ :8081     │ :8081     │
│             │             │             │           │           │
│ 256GB RAM   │ 256GB RAM   │ 256GB RAM   │ 256GB RAM │ 256GB RAM │
│ 24GB VRAM   │ 24GB VRAM   │ 24GB VRAM   │ 24GB VRAM │ 24GB VRAM │
│ 4TB Disk    │ 4TB Disk    │ 4TB Disk    │ 4TB Disk  │ 4TB Disk  │
└─────────────┴─────────────┴─────────────┴───────────┴───────────┘
```

**Roles:**
- Node 1: Spark Master + Driver + Worker (submits and coordinates jobs)
- Nodes 2-5: Spark Workers (execute distributed inference tasks)
- All 5 nodes contribute GPU compute (5x 24GB = 120GB total VRAM)

---

## 2. What You Need to Transfer (Internet Machine → Air-Gapped)

All dependencies must be prepared on an internet-connected machine and transferred
via USB drive, external HDD, or one-way data diode.

### 2.1 Transfer Checklist

| Item | Size (approx) | Purpose |
|------|---------------|---------|
| Docker image tarball | ~8 GB | Complete runtime environment |
| NVIDIA driver RPM/deb | ~300 MB | GPU kernel driver |
| nvidia-container-toolkit packages | ~50 MB | Docker GPU support |
| Docker CE packages | ~100 MB | Container runtime |
| Project source (project.zip) | ~1 MB | Application code |
| PyTorch model weights | ~75 MB | Pre-trained model checkpoints |
| OS packages (if needed) | ~500 MB | Java, Python, utilities |

**Total transfer size: ~9 GB**

---

## 3. Preparation Phase (Internet-Connected Machine)

### 3.1 Build the Docker Image

```bash
cd pytorch-spark-inference-platform
docker build -t multi-model-inference:latest -f deploy/Dockerfile .

# Pre-download model weights into the image
docker run --rm -v $(pwd)/model_cache:/root/.cache/torch multi-model-inference:latest \
  python -c "
from torchvision.models import resnet18, mobilenet_v3_small, efficientnet_b0
resnet18(weights='DEFAULT')
mobilenet_v3_small(weights='DEFAULT')
efficientnet_b0(weights='DEFAULT')
print('Weights cached')
"

# Rebuild with cached weights baked in
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
```

### 3.2 Save Docker Image as Tarball

```bash
docker save multi-model-inference:latest | gzip > multi-model-inference.tar.gz
# Size: ~7-8 GB
```

### 3.3 Download NVIDIA Driver Packages

For Ubuntu 22.04:
```bash
# Download driver
wget https://us.download.nvidia.com/tesla/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run

# Download nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get download nvidia-container-toolkit nvidia-container-toolkit-base \
  libnvidia-container1 libnvidia-container-tools
# Collects ~5 .deb files
```

For RHEL/Amazon Linux:
```bash
dnf download --resolve nvidia-container-toolkit
# Collects RPM files
```

### 3.4 Download Docker CE (if not pre-installed)

```bash
# Ubuntu
apt-get download docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# RHEL
dnf download --resolve docker-ce docker-ce-cli containerd.io
```

### 3.5 Download Java (for Spark without Docker)

```bash
# If running Spark natively (not in Docker)
apt-get download openjdk-17-jre-headless
```

### 3.6 Package Everything for Transfer

```bash
mkdir -p /transfer/airgapped-bundle
cp multi-model-inference.tar.gz /transfer/airgapped-bundle/
cp NVIDIA-Linux-x86_64-535.183.01.run /transfer/airgapped-bundle/
cp *.deb /transfer/airgapped-bundle/nvidia-container-toolkit/
cp *.deb /transfer/airgapped-bundle/docker/
cp project.zip /transfer/airgapped-bundle/

# Create manifest
ls -la /transfer/airgapped-bundle/ > /transfer/airgapped-bundle/MANIFEST.txt
sha256sum /transfer/airgapped-bundle/* > /transfer/airgapped-bundle/checksums.sha256

# Final bundle
tar czf airgapped-bundle.tar.gz -C /transfer airgapped-bundle/
```

---

## 4. Network Configuration (Air-Gapped LAN)

### 4.1 Assign Static IPs

| Node | Hostname | IP Address | Role |
|------|----------|------------|------|
| 1 | spark-master | 192.168.1.10 | Master + Worker |
| 2 | spark-worker-1 | 192.168.1.11 | Worker |
| 3 | spark-worker-2 | 192.168.1.12 | Worker |
| 4 | spark-worker-3 | 192.168.1.13 | Worker |
| 5 | spark-worker-4 | 192.168.1.14 | Worker |

### 4.2 Configure /etc/hosts on ALL nodes

```bash
cat >> /etc/hosts << 'EOF'
192.168.1.10  spark-master
192.168.1.11  spark-worker-1
192.168.1.12  spark-worker-2
192.168.1.13  spark-worker-3
192.168.1.14  spark-worker-4
EOF
```

### 4.3 Firewall Rules (or disable firewall for LAN)

```bash
# Option A: Disable firewall (simplest for isolated LAN)
systemctl stop firewalld && systemctl disable firewalld  # RHEL
ufw disable  # Ubuntu

# Option B: Open specific ports
# Spark Master RPC: 7077
# Spark Master UI: 8080
# Spark Worker UI: 8081
# Spark App UI: 4040
# Spark Block Manager: 7078
# Spark Executor: random high ports (or set spark.port.maxRetries)
firewall-cmd --permanent --add-port=7077/tcp
firewall-cmd --permanent --add-port=7078/tcp
firewall-cmd --permanent --add-port=8080/tcp
firewall-cmd --permanent --add-port=8081/tcp
firewall-cmd --permanent --add-port=4040/tcp
firewall-cmd --permanent --add-port=30000-65535/tcp  # executor dynamic ports
firewall-cmd --reload
```

### 4.4 Verify Connectivity

From each node:
```bash
ping spark-master
ping spark-worker-1
# ... etc
```

---

## 5. Installation on Each Air-Gapped Node

### 5.1 Copy Bundle to Each Node

```bash
# From USB/external drive mounted at /mnt/usb
cp /mnt/usb/airgapped-bundle.tar.gz /opt/
cd /opt && tar xzf airgapped-bundle.tar.gz
cd /opt/airgapped-bundle

# Verify checksums
sha256sum -c checksums.sha256
```

### 5.2 Install Docker (if not present)

```bash
# Ubuntu
dpkg -i docker/*.deb || apt-get install -f -y
systemctl enable docker && systemctl start docker

# RHEL
rpm -ivh docker/*.rpm
systemctl enable docker && systemctl start docker
```

### 5.3 Install NVIDIA Driver

```bash
# Method A: .run file (works on any Linux)
bash NVIDIA-Linux-x86_64-535.183.01.run --silent
nvidia-smi  # verify

# Method B: Package manager (Ubuntu)
dpkg -i nvidia-driver-*.deb || apt-get install -f -y
nvidia-smi  # verify
```

### 5.4 Install nvidia-container-toolkit

```bash
# Ubuntu
dpkg -i nvidia-container-toolkit/*.deb || apt-get install -f -y

# RHEL
rpm -ivh nvidia-container-toolkit/*.rpm

# Configure Docker runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Verify
docker run --rm --gpus all multi-model-inference:latest nvidia-smi
```

### 5.5 Load Docker Image

```bash
gunzip -c /opt/airgapped-bundle/multi-model-inference.tar.gz | docker load
# Output: Loaded image: multi-model-inference:latest
docker images | grep multi-model-inference
```

---

## 6. Spark Cluster Startup

### 6.1 Start Master (Node 1 only)

```bash
docker run -d --name spark-master --network host --gpus all --shm-size=8g \
  --restart unless-stopped \
  multi-model-inference:latest \
  bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"

# Also start a worker on the master (it has a GPU too)
sleep 10
docker run -d --name spark-worker --network host --gpus all --shm-size=8g \
  --restart unless-stopped \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://192.168.1.10:7077 -c 8 -m 200g && tail -f /opt/spark/logs/*worker*"
```

### 6.2 Start Workers (Nodes 2-5)

Run on each worker node:
```bash
docker run -d --name spark-worker --network host --gpus all --shm-size=8g \
  --restart unless-stopped \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://192.168.1.10:7077 -c 8 -m 200g && tail -f /opt/spark/logs/*worker*"
```

**Parameters explained:**
- `-c 8`: Offer 8 CPU cores (adjust based on your CPU count, leave some for OS)
- `-m 200g`: Offer 200GB RAM to Spark (leave 56GB for OS + Docker + GPU buffers)
- `--shm-size=8g`: Shared memory for PyTorch DataLoader
- `--gpus all`: Expose GPU to container

### 6.3 Verify Cluster

From Node 1, open browser: `http://192.168.1.10:8080`

You should see:
- Alive Workers: 5
- Cores in use: 40 Total
- Memory in use: 1000.0 GiB Total

---

## 7. Running Benchmarks

### 7.1 Distributed Mode (All 5 GPUs)

From Node 1:
```bash
docker exec -it spark-master bash -c \
  "SPARK_MASTER_URL=spark://192.168.1.10:7077 python benchmark/run_benchmark.py --mode distributed --partitions 5 --signal-samples 100000 --image-samples 5000 --detection-samples 1000"
```

### 7.2 Incremental Load Test

```bash
docker exec -it spark-master bash -c \
  "SPARK_MASTER_URL=spark://192.168.1.10:7077 python benchmark/incremental_load_test.py"
```

### 7.3 Single GPU Test (any node)

```bash
docker exec -it spark-worker python benchmark/run_benchmark.py --mode single_gpu \
  --signal-samples 100000 --image-samples 5000 --detection-samples 1000
```

### 7.4 Hybrid Test (any node)

```bash
docker exec -it spark-worker python benchmark/run_benchmark.py --mode hybrid \
  --signal-samples 100000 --image-samples 5000 --detection-samples 1000
```

---

## 8. Expected Performance (5-Node, 256GB RAM, 24GB VRAM)

### 8.1 Capacity Estimates

| Resource | Per Node | Total Cluster |
|----------|----------|---------------|
| GPU VRAM | 24 GB | 120 GB |
| RAM | 256 GB | 1,280 GB |
| CPU Cores (est. 16-32) | 16-32 | 80-160 |
| Disk | 4 TB | 20 TB |
| Models in VRAM | All 10 (1.97 GB) | All 10 × 5 copies |

### 8.2 Performance Projections (based on AWS POC results)

| Mode | Estimated Throughput | Reasoning |
|------|---------------------|-----------|
| Single GPU (24GB vs T4 16GB) | ~35,000-45,000 samples/sec | Larger VRAM, faster GPU likely |
| Distributed (5 GPUs) | ~15,000-25,000 samples/sec | 5x executors, overhead amortized |
| Hybrid (24GB VRAM) | ~35,000-45,000 samples/sec | All models fit, same as single |

### 8.3 Max Data Capacity

With 200GB RAM per worker available to Spark:
- **Max dataset in driver memory:** ~200 GB (limited by driver node)
- **Max dataset with Spark DataFrames (out-of-core):** Limited by disk = 4 TB per node
- **Recommended:** For datasets > 50GB, use Spark DataFrames reading from disk (Parquet/NPY files)

---

## 9. Potential Issues and Solutions (Air-Gapped Specific)

### Issue 1: DNS Resolution Fails Inside Docker

**Symptom:** `docker build` or Python code fails with `Name resolution failed`

**Cause:** Air-gapped network has no DNS server. Docker containers use host DNS by default.

**Solution:**
```bash
# Add to /etc/docker/daemon.json
{
  "dns": ["192.168.1.10"],
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Or disable DNS entirely — use IPs in all configs
# Add --add-host entries to docker run:
docker run --add-host spark-master:192.168.1.10 \
           --add-host spark-worker-1:192.168.1.11 ...
```

Or just use `--network host` (recommended, which we already do).

### Issue 2: Docker Image Layer Corruption During Transfer

**Symptom:** `docker load` fails with checksum errors

**Cause:** USB drive filesystem corruption, incomplete copy, or wrong compression

**Solution:**
```bash
# Always verify checksums after transfer
sha256sum multi-model-inference.tar.gz
# Compare with checksum from source machine

# If corrupt, re-transfer. Use tar (not zip) for large files
# Alternatively, split into smaller chunks:
split -b 2G multi-model-inference.tar.gz image_part_
# On air-gapped side:
cat image_part_* > multi-model-inference.tar.gz
```

### Issue 3: NVIDIA Driver Version Mismatch

**Symptom:** `nvidia-smi` works but Docker `--gpus all` fails, or CUDA errors inside container

**Cause:** Driver version on host doesn't match CUDA version in container image

**Solution:**
```
Host driver must be >= container CUDA toolkit version:
- Container uses CUDA 12.1 → needs driver >= 530.xx
- We use driver 535.183.01 → compatible ✓

Verify compatibility:
  nvidia-smi  # shows "CUDA Version: 12.x" — this is the MAX supported
  # The container's CUDA toolkit must be <= this version
```

If your air-gapped GPUs need a different driver:
```bash
# Check GPU model
lspci | grep -i nvidia

# Download the correct driver for your GPU on the internet machine
# NVIDIA driver archive: https://www.nvidia.com/Download/index.aspx
```

### Issue 4: Spark Worker Can't Connect to Master

**Symptom:** Worker logs show `Retrying connection to master` indefinitely

**Cause:**
- Firewall blocking port 7077
- Wrong IP address
- Master container not running

**Solution:**
```bash
# From worker node, test connectivity
nc -zv 192.168.1.10 7077
# If fails: check firewall, check master is running

# On master, verify port is listening
ss -tlnp | grep 7077

# Check master container
docker logs spark-master --tail 5
```

### Issue 5: Executor OOM (Out of Memory)

**Symptom:** Tasks fail with `java.lang.OutOfMemoryError: Java heap space`

**Cause:** Too many models × too many partitions × too much data per executor

**Solution:**
With 256GB RAM, you have plenty of headroom:
```bash
# Increase executor memory (in create_spark_session or spark-submit)
--conf spark.executor.memory=100g
--conf spark.driver.memory=50g
--conf spark.driver.maxResultSize=10g
--conf spark.rpc.message.maxSize=1024
```

### Issue 6: GPU Not Detected by Executor (CUDA Available = False)

**Symptom:** `[Executor] cuda=False, device=cpu` in logs

**Cause:** Spark executor subprocess doesn't inherit GPU access from Docker

**Solution:**
```bash
# Verify GPU is visible inside container
docker exec spark-worker nvidia-smi
docker exec spark-worker python -c "import torch; print(torch.cuda.is_available())"

# If False inside container but nvidia-smi works:
# Check nvidia-container-toolkit configuration
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
# Recreate the container
```

### Issue 7: Clock Skew Between Nodes

**Symptom:** Spark tasks fail with timeout errors, inconsistent log timestamps

**Cause:** Air-gapped nodes can't sync to NTP servers

**Solution:**
```bash
# Set up one node as local NTP server (chrony)
# On Node 1 (master):
yum install chrony
cat > /etc/chrony.conf << 'EOF'
local stratum 10
allow 192.168.1.0/24
EOF
systemctl restart chronyd

# On all other nodes:
cat > /etc/chrony.conf << 'EOF'
server 192.168.1.10 iburst
EOF
systemctl restart chronyd

# Verify sync
chronyc tracking
```

### Issue 8: Model Weights Not Found (First Run)

**Symptom:** `urllib.error.URLError: Name resolution failed` when downloading model weights

**Cause:** Model weights are downloaded from PyTorch Hub on first run, which requires internet

**Solution:** Pre-bake weights into the Docker image (done in Step 3.1):
```bash
# On internet machine, download weights into a cache directory
docker run --rm -v /path/to/cache:/root/.cache/torch multi-model-inference:latest \
  python -c "
from torchvision.models import resnet18, mobilenet_v3_small, efficientnet_b0
resnet18(weights='DEFAULT')
mobilenet_v3_small(weights='DEFAULT')
efficientnet_b0(weights='DEFAULT')
"

# Add to Dockerfile before CMD:
COPY --from=cache /root/.cache/torch /root/.cache/torch
```

Or mount the cache directory:
```bash
docker run --gpus all -v /opt/model-cache:/root/.cache/torch ...
```

### Issue 9: Disk Space Fills Up During Long Runs

**Symptom:** Spark shuffle or result writes fail with `No space left on device`

**Cause:** Spark temp files, Docker layers, or results accumulate

**Solution:**
```bash
# Point Spark local dirs to the large 4TB disk
docker run ... -v /data/spark-tmp:/tmp/spark ...
# Or set in spark-defaults.conf:
spark.local.dir=/data/spark-tmp

# Periodic cleanup
docker system prune -f
rm -rf /data/spark-tmp/blockmgr-*
```

### Issue 10: Updating Code Without Internet

**Symptom:** Need to update application code after deployment

**Solution:** Transfer only the project zip (1MB), not the full Docker image:
```bash
# On internet machine:
zip -r project.zip . -x '.git/*' '__pycache__/*' '.venv/*'
# Transfer project.zip to air-gapped node

# On each air-gapped node:
docker cp project.zip spark-worker:/app/
docker exec spark-worker bash -c "cd /app && unzip -o project.zip"
docker restart spark-worker

# Or rebuild image (if Dockerfile changed):
unzip project.zip -d /opt/app-source
cd /opt/app-source
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
docker rm -f spark-worker
# Re-run docker run command
```

---

## 10. Operational Procedures

### 10.1 Startup Sequence

1. Power on all 5 nodes
2. Verify network connectivity (`ping` between all nodes)
3. Start Spark master on Node 1
4. Start Spark workers on all 5 nodes
5. Verify cluster via Spark UI (`http://192.168.1.10:8080`)
6. Run benchmark or workload

### 10.2 Shutdown Sequence

1. Stop any running Spark applications
2. Stop worker containers on Nodes 2-5: `docker stop spark-worker`
3. Stop worker container on Node 1: `docker stop spark-worker`
4. Stop master container on Node 1: `docker stop spark-master`
5. Power off nodes

### 10.3 Health Checks

```bash
# On any node — check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check GPU health
nvidia-smi

# Check Spark cluster health (from master)
curl -s http://192.168.1.10:8080/json/ | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Workers: {d.get(\"aliveworkers\", 0)}')
print(f'Cores: {d.get(\"cores\", 0)}')
print(f'Memory: {d.get(\"memory\", 0) / 1024:.0f} GB')
"
```

### 10.4 Log Collection

```bash
# Spark master logs
docker logs spark-master > /data/logs/spark-master-$(date +%Y%m%d).log 2>&1

# Worker logs (on each node)
docker logs spark-worker > /data/logs/spark-worker-$(hostname)-$(date +%Y%m%d).log 2>&1

# GPU metrics snapshot
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,temperature.gpu \
  --format=csv >> /data/logs/gpu-metrics-$(hostname).csv
```

---

## 11. Automation Script (setup_all_nodes.sh)

Run this on each node after copying the bundle:

```bash
#!/bin/bash
# setup_node.sh — Run on each air-gapped node
# Usage: sudo bash setup_node.sh <role> <master_ip>
#   role: "master" or "worker"
#   master_ip: IP of the master node (e.g., 192.168.1.10)

set -e
ROLE=${1:-worker}
MASTER_IP=${2:-192.168.1.10}
BUNDLE_DIR=/opt/airgapped-bundle

echo "=== Setting up node as: $ROLE ==="
echo "=== Master IP: $MASTER_IP ==="

# Install NVIDIA driver (if not present)
if ! command -v nvidia-smi &>/dev/null; then
    echo "Installing NVIDIA driver..."
    bash $BUNDLE_DIR/NVIDIA-Linux-x86_64-535.183.01.run --silent
fi
nvidia-smi

# Install Docker (if not present)
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    dpkg -i $BUNDLE_DIR/docker/*.deb 2>/dev/null || \
    rpm -ivh $BUNDLE_DIR/docker/*.rpm 2>/dev/null || true
    systemctl enable docker && systemctl start docker
fi

# Install nvidia-container-toolkit
if ! dpkg -l nvidia-container-toolkit &>/dev/null && ! rpm -q nvidia-container-toolkit &>/dev/null; then
    echo "Installing nvidia-container-toolkit..."
    dpkg -i $BUNDLE_DIR/nvidia-container-toolkit/*.deb 2>/dev/null || \
    rpm -ivh $BUNDLE_DIR/nvidia-container-toolkit/*.rpm 2>/dev/null || true
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# Load Docker image
echo "Loading Docker image..."
gunzip -c $BUNDLE_DIR/multi-model-inference.tar.gz | docker load

# Disable firewall for LAN
systemctl stop firewalld 2>/dev/null || ufw disable 2>/dev/null || true

# Start containers
docker rm -f spark-master spark-worker 2>/dev/null || true

if [ "$ROLE" = "master" ]; then
    echo "Starting Spark Master..."
    docker run -d --name spark-master --network host --gpus all --shm-size=8g \
      --restart unless-stopped \
      multi-model-inference:latest \
      bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
    sleep 10
fi

echo "Starting Spark Worker..."
docker run -d --name spark-worker --network host --gpus all --shm-size=8g \
  --restart unless-stopped \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://${MASTER_IP}:7077 -c 8 -m 200g && tail -f /opt/spark/logs/*worker*"

sleep 5
docker logs spark-worker --tail 3
echo "=== Node setup complete ($ROLE) ==="
```

**Usage:**
```bash
# On Node 1 (master):
sudo bash setup_node.sh master 192.168.1.10

# On Nodes 2-5 (workers):
sudo bash setup_node.sh worker 192.168.1.10
```

---

## 12. Comparison: AWS POC vs Air-Gapped Production

| Aspect | AWS POC | Air-Gapped 5-Node |
|--------|---------|-------------------|
| GPU VRAM | 16 GB (T4) × 1 | 24 GB × 5 = 120 GB |
| RAM | 8 + 16 GB | 256 GB × 5 = 1,280 GB |
| GPU Workers | 1 | 5 |
| Max parallel models | 10 (on 1 GPU) | 10 × 5 = 50 instances |
| Max dataset (in-memory) | ~1 GB (driver limited) | ~200 GB (driver: 256GB) |
| Max dataset (disk-backed) | 100-150 GB | 4 TB per node |
| Network | 25 Gbps (AWS VPC) | Depends on LAN (1-10 Gbps) |
| Throughput (projected) | 2,686 dist / 29,980 single | 15,000+ dist / 45,000+ single |
| Fault tolerance | None (POC) | Spark retry + --restart |
| Monitoring | CloudWatch | Local Spark UI + nvidia-smi |

---

## 13. Security Considerations (Air-Gapped)

1. **No outbound internet:** Docker images cannot pull updates. All dependencies must be pre-baked.
2. **USB transfer verification:** Always verify checksums after transfer to detect corruption.
3. **No package manager updates:** Pin all package versions. Document exact versions used.
4. **Log retention:** Logs stay on local disk. Set up log rotation to prevent disk fill.
5. **No cloud monitoring:** Use local Spark UI + custom scripts for monitoring.
6. **Physical security:** USB drives used for transfer must be wiped after use per your org's policy.
7. **Audit trail:** Keep a manifest of all software versions transferred and installed.
