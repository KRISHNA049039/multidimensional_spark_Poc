# Windows Spark Cluster Setup Guide

How to deploy and run the multi-model inference benchmark on a Windows lab cluster using Docker Desktop.

---

## Architecture

```
Machine A (Master)                Machine B (Worker)              Machine C (Worker)
192.168.1.100                     192.168.1.101                   192.168.1.102
┌─────────────────────┐           ┌─────────────────────┐        ┌─────────────────────┐
│ spark-master        │◄──────────│ spark-cpu-worker    │        │ spark-gpu-worker    │
│ Port 7077 (RPC)     │◄──────────────────────────────────────────│ --gpus all          │
│ Port 8080 (UI)      │           │ -c 4 -m 8g          │        │ -c 4 -m 12g         │
│ Port 4040 (App UI)  │           │                     │        │ --shm-size=4g       │
└─────────────────────┘           └─────────────────────┘        └─────────────────────┘
         │                                 │                              │
         └─────────── Same LAN / Subnet ───┴──────────────────────────────┘
```

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| Docker Desktop (each machine) | Run Spark containers |
| NVIDIA GPU Driver (GPU machines) | GPU compute |
| Docker Desktop GPU support / WSL2 backend (GPU machines) | `--gpus all` flag |
| All machines on same subnet | Workers must reach master on port 7077 |
| `multi-model-inference:latest` image on each machine | Spark + PyTorch + project code |

---

## Phase 1: Build Docker Image

Build on one machine and distribute to others:

```powershell
cd D:\Spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
```

Transfer to other machines:

```powershell
# Export on build machine
docker save multi-model-inference:latest -o multi-model-inference.tar

# Load on each worker machine (copy tar via USB / network share)
docker load -i multi-model-inference.tar
```

---

## Phase 2: Single-Machine Cluster (Quick Test)

Use Docker Compose to start master + workers on one machine:

```powershell
cd D:\Spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform
docker compose -f deploy/docker-compose.cluster.yml up
```

Scale workers:

```powershell
docker compose -f deploy/docker-compose.cluster.yml up --scale spark-cpu-worker=3 --scale spark-gpu-worker=2
```

Teardown:

```powershell
docker compose -f deploy/docker-compose.cluster.yml down
```

---

## Phase 3: Multi-Machine Cluster

### 3.1 Start Master (Machine A — 192.168.1.100)

```powershell
docker run -d --name spark-master `
  -p 7077:7077 `
  -p 8080:8080 `
  -p 4040:4040 `
  multi-model-inference:latest `
  bash -c "start-master.sh -h 192.168.1.100 && tail -f /opt/spark/logs/*master*"
```

The `-h 192.168.1.100` flag tells Spark to advertise the machine's real LAN IP (not the container hostname).

Verify: open `http://192.168.1.100:8080` from any machine on the network.

### 3.2 Register CPU Worker (Machine B — 192.168.1.101)

```powershell
docker run -d --name spark-cpu-worker `
  -p 8081:8081 `
  -e SPARK_WORKER_HOST=192.168.1.101 `
  multi-model-inference:latest `
  bash -c "SPARK_LOCAL_IP=192.168.1.101 start-worker.sh spark://192.168.1.100:7077 -c 4 -m 8g && tail -f /opt/spark/logs/*worker*"
```

### 3.3 Register GPU Worker (Machine C — 192.168.1.102)

```powershell
docker run -d --name spark-gpu-worker `
  -p 8081:8081 `
  --gpus all `
  --shm-size=4g `
  -e SPARK_WORKER_HOST=192.168.1.102 `
  multi-model-inference:latest `
  bash -c "SPARK_LOCAL_IP=192.168.1.102 start-worker.sh spark://192.168.1.100:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
```

### 3.4 Add More Workers

Repeat Step 3.2 or 3.3 on any additional machines. Each worker self-registers with the master automatically.

---

## Environment Variables Explained

| Variable | Where | Purpose |
|----------|-------|---------|
| `SPARK_LOCAL_IP=<worker-ip>` | Worker container | Advertises the host machine's real IP to the master |
| `SPARK_WORKER_HOST=<worker-ip>` | Worker container | Ensures master can route tasks back to this worker |
| `-h <master-ip>` (start-master.sh flag) | Master container | Master advertises its real IP to workers |

Without these, containers advertise their internal Docker IPs (e.g., `172.17.0.2`) which other machines can't reach.

---

## Phase 4: Running Benchmarks

### Device Modes

| Mode | `--device-mode` | Behavior |
|------|-----------------|----------|
| CPU Only | `cpu_only` | All executors force `torch.device('cpu')` |
| GPU Only | `gpu_only` | All executors force `torch.device('cuda')` — fails if no GPU |
| Hybrid | `hybrid` | Each executor auto-detects: CUDA where available, CPU elsewhere |

### Run from Master Container

```powershell
# CPU-only test
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://192.168.1.100:7077 python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 5000"

# GPU-only test
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://192.168.1.100:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 2 --signal-samples 1000"

# Hybrid test (CPU + GPU mixed)
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://192.168.1.100:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 5000"

# Full incremental test (all 3 modes × 3 load levels = 9 runs)
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://192.168.1.100:7077 python benchmark/cluster_benchmark.py --incremental"
```

### Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device-mode` | hybrid | `gpu_only`, `cpu_only`, or `hybrid` |
| `--partitions` | 2 | Number of data chunks (= parallel tasks) |
| `--signal-samples` | 5000 | Signal samples per model (128-dim IQ vectors) |
| `--image-samples` | 200 | Image classification samples (3×224×224) |
| `--detection-samples` | 50 | Object detection samples (3×640×640) |
| `--batch-size` | 256 | Batch size for inference |
| `--incremental` | — | Run all modes with increasing load |

---

## Phase 5: Verification

### Check Spark Master UI

Open `http://192.168.1.100:8080` — shows registered workers, running/completed applications.

### Check Workers via CLI

```powershell
docker exec spark-master curl -s http://localhost:8080/json/ | python -c "import sys,json; d=json.load(sys.stdin); print(f'Alive Workers: {d[\"aliveworkers\"]}'); [print(f'  {w[\"host\"]}:{w[\"port\"]} - {w[\"cores\"]} cores, {w[\"memoryfree\"]//1024//1024}MB free') for w in d.get('workers',[])]"
```

### Check GPU Inside Container

```powershell
docker exec spark-gpu-worker nvidia-smi
```

---

## Firewall Configuration

These ports must be open between all cluster machines:

| Port | Direction | Purpose |
|------|-----------|---------|
| 7077 | Worker → Master | Spark RPC (worker registration) |
| 8080 | Any → Master | Master Web UI |
| 4040 | Any → Master | Application UI (active during jobs) |
| 8081 | Any → Worker | Worker Web UI |
| Dynamic high ports | Master ↔ Worker | Executor shuffle/task communication |

### Quick Fix for Lab (disable firewall on private network):

```powershell
# Run as Administrator on each machine
Set-NetFirewallProfile -Profile Private -Enabled False
```

### Or allow specific ports:

```powershell
New-NetFirewallRule -DisplayName "Spark Master" -Direction Inbound -LocalPort 7077,8080,4040 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Spark Worker" -Direction Inbound -LocalPort 8081 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Spark Executors" -Direction Inbound -LocalPort 30000-40000 -Protocol TCP -Action Allow
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `invalid reference format` | Backticks around image name in PowerShell | Run command as single line or ensure backtick is only at line end |
| `container name already in use` | Leftover container from previous run | `docker rm -f spark-master` |
| Worker can't connect to master | Firewall blocking port 7077 | Open port or disable private firewall |
| Worker registers with internal IP | Missing `SPARK_LOCAL_IP` | Add `-e SPARK_WORKER_HOST=<real-ip>` and `SPARK_LOCAL_IP=<real-ip>` |
| `NVIDIA Driver was not detected` | No GPU or Docker GPU support not configured | Use `--device-mode cpu_only`; for GPU, enable WSL2 GPU passthrough |
| `rsync: command not found` | Harmless warning from Spark daemon script | Ignore — doesn't affect functionality |
| Tasks not distributing evenly | Fewer partitions than workers | Increase `--partitions` to match or exceed worker count |

---

## Phase 6: Automated Cluster Management (Recommended)

Instead of manually running `docker run` on each machine, use the automation scripts for one-click start/stop.

### Prerequisites for Remote Automation

Enable WinRM on each worker machine (run as Administrator once):

```powershell
Enable-PSRemoting -Force
```

This allows the master machine to start/stop containers on workers remotely via PowerShell Remoting.

### Configure Your Lab Topology

Edit the `$WORKERS` array in both scripts (`deploy/start_cluster.ps1` and `deploy/stop_cluster.ps1`):

```powershell
$MASTER_IP = "192.168.1.100"       # Your master machine's LAN IP

$WORKERS = @(
    @{ IP = $MASTER_IP;      Type = "cpu"; Cores = 2; Memory = "4g";  Location = "local";  Name = "spark-cpu-worker-local" },
    @{ IP = "192.168.1.101"; Type = "cpu"; Cores = 4; Memory = "8g";  Location = "remote"; Name = "spark-cpu-worker-1" },
    @{ IP = "192.168.1.102"; Type = "gpu"; Cores = 4; Memory = "12g"; Location = "remote"; Name = "spark-gpu-worker-1" }
)
```

Add or remove entries as your lab grows. Set `Type` to `"cpu"` or `"gpu"`, and `Location` to `"local"` (same machine as master) or `"remote"`.

### Start the Cluster

```powershell
cd D:\Spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform

# Start full cluster (master + all workers)
.\deploy\start_cluster.ps1

# Start master only (add workers later)
.\deploy\start_cluster.ps1 -MasterOnly

# Start master + local workers only (skip remote machines)
.\deploy\start_cluster.ps1 -SkipRemote

# Force restart (stops existing containers first)
.\deploy\start_cluster.ps1 -Force
```

The script:
1. Validates Docker is running and image exists
2. Starts the Spark master and waits for it to be healthy
3. Starts local workers on the master machine
4. SSHs into remote machines via WinRM and starts workers there
5. Prints a summary with the Spark UI URL and benchmark commands

### Stop the Cluster

```powershell
# Stop everything (workers first for graceful deregistration, then master)
.\deploy\stop_cluster.ps1

# Stop workers only, keep master running (for adding different workers)
.\deploy\stop_cluster.ps1 -WorkersOnly

# Stop local containers only (skip remote machines)
.\deploy\stop_cluster.ps1 -SkipRemote

# Full cleanup (stop + remove orphan containers + prune networks)
.\deploy\stop_cluster.ps1 -Cleanup
```

### Typical Workflow

```powershell
# 1. Start cluster
.\deploy\start_cluster.ps1

# 2. Run benchmarks
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://192.168.1.100:7077 python benchmark/cluster_benchmark.py --incremental"

# 3. Stop cluster when done
.\deploy\stop_cluster.ps1
```

### Scaling Workers Without Restarting

```powershell
# Stop workers, keep master
.\deploy\stop_cluster.ps1 -WorkersOnly

# Edit $WORKERS in start_cluster.ps1 (change cores, memory, add machines)

# Restart workers with new config
.\deploy\start_cluster.ps1 -MasterOnly  # skipped since master is running
.\deploy\start_cluster.ps1              # starts workers, detects master already healthy
```

---

## Cleanup

### Stop Single-Machine Compose Cluster

```powershell
docker compose -f deploy/docker-compose.cluster.yml down
```

### Stop Multi-Machine Cluster (Manual)

On each machine:

```powershell
docker stop spark-master spark-cpu-worker spark-gpu-worker 2>$null
docker rm spark-master spark-cpu-worker spark-gpu-worker 2>$null
```

### Stop Multi-Machine Cluster (Automated)

```powershell
.\deploy\stop_cluster.ps1 -Cleanup
```

---

## Industry Approaches Comparison

| Approach | Complexity | Best For | What We Use |
|----------|-----------|----------|-------------|
| Manual `docker run` | High (error-prone) | One-off testing | — |
| Automation scripts (PowerShell) | Low | Lab / small team | **start_cluster.ps1 / stop_cluster.ps1** |
| Docker Compose | Low | Single-machine testing | **docker-compose.cluster.yml** |
| Docker Swarm | Medium | Multi-node without K8s | Future option |
| Kubernetes + Spark Operator | High | Production, auto-scaling | When moving to production |
| Managed (AWS EMR, Databricks) | Low (ops) | Cloud-native teams | When moving to cloud |

---

## Quick Reference

| Action | Command |
|--------|---------|
| Build image | `docker build -t multi-model-inference:latest -f deploy/Dockerfile .` |
| Start master | `docker run -d --name spark-master -p 7077:7077 -p 8080:8080 -p 4040:4040 multi-model-inference:latest bash -c "start-master.sh -h <MASTER_IP> && tail -f /opt/spark/logs/*master*"` |
| Start CPU worker | `docker run -d --name spark-cpu-worker -p 8081:8081 -e SPARK_WORKER_HOST=<WORKER_IP> multi-model-inference:latest bash -c "SPARK_LOCAL_IP=<WORKER_IP> start-worker.sh spark://<MASTER_IP>:7077 -c 4 -m 8g && tail -f /opt/spark/logs/*worker*"` |
| Start GPU worker | `docker run -d --name spark-gpu-worker -p 8081:8081 --gpus all --shm-size=4g -e SPARK_WORKER_HOST=<WORKER_IP> multi-model-inference:latest bash -c "SPARK_LOCAL_IP=<WORKER_IP> start-worker.sh spark://<MASTER_IP>:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"` |
| Run CPU benchmark | `docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://<MASTER_IP>:7077 python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 5000"` |
| Run GPU benchmark | `docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://<MASTER_IP>:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 2 --signal-samples 1000"` |
| Run hybrid benchmark | `docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://<MASTER_IP>:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 5000"` |
| Scale (compose) | `docker compose -f deploy/docker-compose.cluster.yml up --scale spark-cpu-worker=3 --scale spark-gpu-worker=2` |
| View Spark UI | `http://<MASTER_IP>:8080` |
| Teardown | `docker compose -f deploy/docker-compose.cluster.yml down` |
