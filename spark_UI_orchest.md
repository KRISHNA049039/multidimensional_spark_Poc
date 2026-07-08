# Spark UI Monitoring & Docker Cluster Orchestration

## Table of Contents

1. [Spark UI — What It Is](#1-spark-ui--what-it-is)
2. [Enabling Spark UI in Docker](#2-enabling-spark-ui-in-docker)
3. [Navigating the Spark UI](#3-navigating-the-spark-ui)
4. [Key Metrics to Monitor for EW Inference](#4-key-metrics-to-monitor-for-ew-inference)
5. [Running Docker Containers Across a Cluster](#5-running-docker-containers-across-a-cluster)
6. [Docker Swarm Approach (Simplest)](#6-docker-swarm-approach-simplest)
7. [Kubernetes Approach (Production)](#7-kubernetes-approach-production)
8. [Manual Multi-Node Docker (Airgapped)](#8-manual-multi-node-docker-airgapped)
9. [Auto-Start Containers on Session Init](#9-auto-start-containers-on-session-init)
10. [Complete Example — 4 Node Cluster](#10-complete-example--4-node-cluster)

---

## 1. Spark UI — What It Is

Spark automatically launches a web-based monitoring UI when a SparkSession is created.

```
┌─────────────────────────────────────────────────────────────┐
│                    SPARK WEB UIs                              │
│                                                              │
│  Port 4040 — Application UI (per-job metrics)                │
│    → Jobs, Stages, Tasks, Storage, Environment               │
│    → Active while your application runs                      │
│                                                              │
│  Port 8080 — Master UI (cluster overview)                    │
│    → Worker status, running apps, completed apps             │
│    → Always active while master is running                   │
│                                                              │
│  Port 8081 — Worker UI (per-worker details)                  │
│    → Executors running on this worker                        │
│    → Log access                                              │
│                                                              │
│  Port 18080 — History Server (past jobs)                     │
│    → Browse completed application metrics                    │
│    → Requires spark.eventLog.enabled=true                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Enabling Spark UI in Docker

### docker-compose.yml with UI ports exposed

```yaml
services:
  benchmark:
    build: .
    container_name: ew-pytorch-spark-poc
    runtime: nvidia
    volumes:
      - .:/app
      - ./results:/app/results
    ports:
      # Spark Application UI
      - "4040:4040"
      # Spark Master UI (if running as master)
      - "8080:8080"
      # Spark Worker UI
      - "8081:8081"
    environment:
      - PYSPARK_PYTHON=python
      - PYSPARK_DRIVER_PYTHON=python
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    shm_size: '2g'
    command: python benchmark/run_benchmark.py
```

### Spark Config for UI

Add to `create_spark_session()` in code:

```python
spark = (
    SparkSession.builder
    .appName("EW_PyTorch_Inference")
    .master("local[*]")
    .config("spark.ui.enabled", "true")
    .config("spark.ui.port", "4040")
    .config("spark.eventLog.enabled", "true")
    .config("spark.eventLog.dir", "/app/results/spark-events")
    .getOrCreate()
)
```

### Access the UI

```
While the benchmark is running, open in browser:
  http://localhost:4040

If running on remote server:
  http://<server-ip>:4040
```

---

## 3. Navigating the Spark UI

### Jobs Tab (http://localhost:4040/jobs/)

```
┌─────────────────────────────────────────────────────────────┐
│ JOBS                                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Job ID │ Description        │ Duration │ Stages │ Tasks     │
│────────┼────────────────────┼──────────┼────────┼───────────│
│ 0      │ collect at run_.py │ 2.5s     │ 1      │ 8/8       │
│ 1      │ collect at run_.py │ 4.1s     │ 1      │ 8/8       │
│ 2      │ collect at run_.py │ 8.3s     │ 1      │ 8/8       │
│                                                              │
│ Each "collect" = one scale of our benchmark                  │
│ (10K, 50K, 100K signals)                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Stages Tab (http://localhost:4040/stages/)

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE DETAIL                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Stage 0: map at spark_inference.py:130                       │
│                                                              │
│ Summary Metrics:                                             │
│ ┌────────────────┬───────┬───────┬───────┬───────┐          │
│ │ Metric         │ Min   │ 25%   │ Med   │ Max   │          │
│ ├────────────────┼───────┼───────┼───────┼───────┤          │
│ │ Task Duration  │ 0.8s  │ 0.9s  │ 1.0s  │ 1.2s  │          │
│ │ GC Time        │ 0ms   │ 0ms   │ 0ms   │ 12ms  │          │
│ │ Result Size    │ 48B   │ 48B   │ 48B   │ 48B   │          │
│ │ Input Size     │ 0B    │ 0B    │ 0B    │ 0B    │          │
│ └────────────────┴───────┴───────┴───────┴───────┘          │
│                                                              │
│ DAG Visualization:                                           │
│ [parallelize] → [map(infer_partition)] → [collect]           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**What to look for:**
- **Task Duration spread**: If max >> min, one partition is a straggler
- **GC Time**: High GC = executor needs more memory
- **Skew**: If one task takes 10x longer, data is unevenly split

### Tasks Tab (within a stage)

```
┌─────────────────────────────────────────────────────────────┐
│ TASKS (8 total for our 8 partitions)                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Task │ Executor │ Duration │ Status  │ Locality  │ Errors   │
│──────┼──────────┼──────────┼─────────┼───────────┼──────────│
│ 0    │ driver   │ 1.0s     │ SUCCESS │ PROCESS   │ 0        │
│ 1    │ driver   │ 0.9s     │ SUCCESS │ PROCESS   │ 0        │
│ 2    │ driver   │ 1.1s     │ SUCCESS │ PROCESS   │ 0        │
│ 3    │ driver   │ 0.8s     │ SUCCESS │ PROCESS   │ 0        │
│ 4    │ driver   │ 1.0s     │ SUCCESS │ PROCESS   │ 0        │
│ 5    │ driver   │ 0.9s     │ SUCCESS │ PROCESS   │ 0        │
│ 6    │ driver   │ 1.2s     │ SUCCESS │ PROCESS   │ 0        │
│ 7    │ driver   │ 1.0s     │ SUCCESS │ PROCESS   │ 0        │
│                                                              │
│ On cluster: Executor column shows which worker ran each task │
│ PROCESS_LOCAL = data is on same machine as executor          │
│ NODE_LOCAL = data is on same node but different executor     │
│ RACK_LOCAL = data is on same rack                            │
│ ANY = data transferred over network                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Executors Tab (http://localhost:4040/executors/)

```
┌─────────────────────────────────────────────────────────────┐
│ EXECUTORS                                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ID     │ Address         │ Cores │ Memory │ Active │ Tasks  │
│────────┼─────────────────┼───────┼────────┼────────┼────────│
│ driver │ 172.17.0.2:4040 │ 8     │ 2.0 GB │ 0      │ 24     │
│ 0      │ 172.17.0.3:5555 │ 4     │ 8.0 GB │ 2      │ 16     │
│ 1      │ 172.17.0.4:5555 │ 4     │ 8.0 GB │ 2      │ 16     │
│                                                              │
│ On cluster: shows all workers + their resource usage         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Key Metrics to Monitor for EW Inference

| Metric | Where to Find | Good Value | Problem Indicator |
|--------|--------------|------------|-------------------|
| Task duration (max/min ratio) | Stages → Tasks | < 1.5x | > 3x means data skew |
| GC time per task | Stages → Summary | < 5% of duration | > 20% = need more memory |
| Shuffle read/write | Stages | 0 (for our RDD map) | High = unnecessary shuffle |
| Task failures | Jobs | 0 | > 0 = check executor logs |
| Executor memory used | Executors | < 80% of allocated | > 90% = risk of OOM |
| Active tasks | Executors | = num_partitions | < means resource bottleneck |
| Scheduler delay | Stages → Tasks | < 50ms | > 500ms = scheduler overloaded |

### Interpreting for EW workload

```
Healthy inference run:
  ✓ 8 tasks all complete in ~1s each (balanced)
  ✓ 0 GC time (data fits in memory)
  ✓ 0 shuffle (pure map operation)
  ✓ 0 failures
  ✓ Total time ≈ max(task durations) + scheduling overhead

Unhealthy run:
  ✗ 7 tasks finish in 1s, 1 task takes 10s (straggler)
  ✗ High GC → executor memory too small
  ✗ Task failures → check executor stderr logs
  ✗ Total time >> expected → check if partitions are too large
```

---

## 5. Running Docker Containers Across a Cluster

### The Challenge

```
You have 4 physical machines. You need:
  - Spark master running on Node 1
  - Spark worker + PyTorch on Nodes 2, 3, 4
  - All using the SAME Docker image
  - All connected on a network

How do you start containers on ALL machines simultaneously?
```

### Three Options

| Approach | Complexity | Requires | Best For |
|----------|-----------|----------|----------|
| Manual SSH | Low | SSH access to all nodes | Testing, small cluster |
| Docker Swarm | Medium | Docker installed on all nodes | Airgapped, no K8s |
| Kubernetes | High | K8s cluster | Production, large scale |

---

## 6. Docker Swarm Approach (Simplest for Airgapped)

Docker Swarm is Docker's built-in clustering. No extra software needed — just Docker itself.

### What is Docker Swarm?

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCKER SWARM                               │
│                                                              │
│  Node 1 (Manager/Master):                                    │
│    - Runs swarm manager                                      │
│    - Schedules services across nodes                         │
│    - Runs Spark master container                             │
│                                                              │
│  Node 2, 3, 4 (Workers):                                    │
│    - Join the swarm                                          │
│    - Docker automatically starts containers on them          │
│    - Run Spark worker containers with GPU                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step Setup

#### 1. Initialize Swarm on Manager Node

```bash
# On Node 1 (will be manager):
docker swarm init --advertise-addr 192.168.1.10

# Output:
# docker swarm join --token SWMTKN-1-abc123... 192.168.1.10:2377
```

#### 2. Join Worker Nodes

```bash
# On Node 2:
docker swarm join --token SWMTKN-1-abc123... 192.168.1.10:2377

# On Node 3:
docker swarm join --token SWMTKN-1-abc123... 192.168.1.10:2377

# On Node 4:
docker swarm join --token SWMTKN-1-abc123... 192.168.1.10:2377
```

#### 3. Load Docker Image on ALL Nodes

```bash
# On each node (airgapped — load from tar):
docker load -i ew-pytorch-spark.tar
```

#### 4. Deploy Stack (starts containers on all nodes automatically)

Create `docker-stack.yml`:

```yaml
version: "3.8"

services:
  spark-master:
    image: ew-pytorch-spark:latest
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager    # Run master on manager node
    ports:
      - "8080:8080"    # Master Web UI
      - "7077:7077"    # Master connection port
      - "4040:4040"    # Application UI
    environment:
      - SPARK_MODE=master
    command: >
      bash -c "
        $$SPARK_HOME/sbin/start-master.sh &&
        tail -f $$SPARK_HOME/logs/*master*
      "
    networks:
      - spark-net

  spark-worker:
    image: ew-pytorch-spark:latest
    deploy:
      replicas: 3                   # One per worker node
      placement:
        constraints:
          - node.role == worker     # Only on worker nodes
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER=spark://spark-master:7077
      - SPARK_WORKER_CORES=4
      - SPARK_WORKER_MEMORY=8g
    command: >
      bash -c "
        $$SPARK_HOME/sbin/start-worker.sh spark://spark-master:7077 &&
        tail -f $$SPARK_HOME/logs/*worker*
      "
    depends_on:
      - spark-master
    networks:
      - spark-net

networks:
  spark-net:
    driver: overlay    # Overlay network spans all swarm nodes
```

#### 5. Deploy the Stack

```bash
# On manager node:
docker stack deploy -c docker-stack.yml ew-spark

# Check status:
docker stack services ew-spark
docker stack ps ew-spark
```

#### 6. Submit Your Job

```bash
# Connect to master container and submit:
docker exec -it $(docker ps -q -f name=ew-spark_spark-master) bash

# Inside container:
spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  --num-executors 3 \
  --executor-cores 4 \
  --executor-memory 8g \
  /app/benchmark/run_benchmark.py
```

### Viewing Spark UI

```
From any machine on the network:
  Master UI:      http://192.168.1.10:8080
  Application UI: http://192.168.1.10:4040

The Master UI shows:
  - Worker nodes connected
  - Running applications
  - Completed applications
  - Cluster resources (cores, memory)
```

---

## 7. Kubernetes Approach (Production)

### Architecture

```
┌──── Kubernetes Cluster ────────────────────────────────────┐
│                                                             │
│  ┌─────────────────┐                                       │
│  │ spark-submit    │  (runs as a K8s pod)                   │
│  │ (driver pod)    │                                        │
│  └────────┬────────┘                                       │
│           │ creates executor pods dynamically               │
│     ┌─────┼─────┬─────────┐                                │
│     ▼     ▼     ▼         ▼                                │
│  ┌──────┐┌──────┐┌──────┐┌──────┐                         │
│  │Exec 1││Exec 2││Exec 3││Exec 4│  (auto-created pods)    │
│  │GPU   ││GPU   ││GPU   ││GPU   │                         │
│  └──────┘└──────┘└──────┘└──────┘                         │
│                                                             │
│  When job finishes → executor pods auto-deleted             │
│  (no wasted resources between jobs)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Submit to K8s

```bash
spark-submit \
  --master k8s://https://k8s-api-server:6443 \
  --deploy-mode cluster \
  --conf spark.kubernetes.container.image=ew-pytorch-spark:latest \
  --conf spark.kubernetes.namespace=ew-inference \
  --conf spark.executor.instances=4 \
  --conf spark.executor.cores=4 \
  --conf spark.executor.memory=8g \
  --conf spark.kubernetes.executor.resources.nvidia.com/gpu.amount=1 \
  --conf spark.kubernetes.driver.pod.name=ew-driver \
  local:///app/benchmark/run_benchmark.py
```

### Access Spark UI on K8s

```bash
# Port-forward the driver pod:
kubectl port-forward ew-driver 4040:4040

# Then open: http://localhost:4040
```

---

## 8. Manual Multi-Node Docker (Airgapped, No Swarm)

If you can't use Swarm or K8s — just SSH and docker run on each machine.

### Setup Script (run from master node)

```bash
#!/bin/bash
# deploy_cluster.sh
# Run this from the master node. Requires SSH access to all workers.

MASTER_IP="192.168.1.10"
WORKERS=("192.168.1.11" "192.168.1.12" "192.168.1.13")
IMAGE="ew-pytorch-spark:latest"
SPARK_HOME="/opt/spark"

# ─── Start Spark Master on this machine ───
echo "Starting Spark Master on $MASTER_IP..."
docker run -d \
  --name spark-master \
  --network host \
  -p 7077:7077 \
  -p 8080:8080 \
  -p 4040:4040 \
  $IMAGE \
  bash -c "$SPARK_HOME/sbin/start-master.sh && tail -f $SPARK_HOME/logs/*master*"

echo "Master UI: http://$MASTER_IP:8080"
sleep 5

# ─── Start Spark Workers on remote machines ───
for WORKER_IP in "${WORKERS[@]}"; do
  echo "Starting Spark Worker on $WORKER_IP..."
  ssh $WORKER_IP "
    docker run -d \
      --name spark-worker \
      --network host \
      --gpus all \
      $IMAGE \
      bash -c '$SPARK_HOME/sbin/start-worker.sh spark://$MASTER_IP:7077 && \
               tail -f $SPARK_HOME/logs/*worker*'
  "
done

echo ""
echo "Cluster ready. Submit jobs with:"
echo "  docker exec spark-master spark-submit \\"
echo "    --master spark://$MASTER_IP:7077 \\"
echo "    /app/benchmark/run_benchmark.py"
```

### Teardown Script

```bash
#!/bin/bash
# stop_cluster.sh

WORKERS=("192.168.1.11" "192.168.1.12" "192.168.1.13")

# Stop workers
for WORKER_IP in "${WORKERS[@]}"; do
  echo "Stopping worker on $WORKER_IP..."
  ssh $WORKER_IP "docker stop spark-worker && docker rm spark-worker"
done

# Stop master
echo "Stopping master..."
docker stop spark-master && docker rm spark-master

echo "Cluster stopped."
```

---

## 9. Auto-Start Containers on Session Init

### What "Session Initiated" Means

When you run `spark-submit` or `SparkSession.builder.getOrCreate()`:
1. Driver connects to master
2. Master allocates executors on workers
3. Executors start automatically (JVM processes on worker nodes)
4. Your code runs

**You don't need to manually start Docker containers per session.** Docker containers run the Spark daemons (master/worker), which are long-running services. Your jobs use them repeatedly.

### Pattern: Long-Running Cluster + On-Demand Jobs

```
┌─────────────────────────────────────────────────────────────┐
│                    LIFECYCLE                                  │
│                                                              │
│  ONCE (cluster setup):                                       │
│    Start master container  ─── always running ───────────►   │
│    Start worker containers ─── always running ───────────►   │
│                                                              │
│  PER JOB (when you want to run inference):                   │
│    spark-submit ... run_benchmark.py                          │
│      ↓                                                       │
│    [Spark allocates executors on existing workers]            │
│      ↓                                                       │
│    [Job runs, results collected]                             │
│      ↓                                                       │
│    [Executors released — workers still running]              │
│                                                              │
│  NEVER needed:                                               │
│    Starting new Docker containers for each job               │
│    (containers stay up, Spark handles sessions internally)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Auto-Start on Machine Boot (systemd)

To make containers start automatically when machines boot:

```bash
# /etc/systemd/system/spark-worker.service
[Unit]
Description=Spark Worker Container
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=5
ExecStartPre=-/usr/bin/docker stop spark-worker
ExecStartPre=-/usr/bin/docker rm spark-worker
ExecStart=/usr/bin/docker run \
  --name spark-worker \
  --network host \
  --gpus all \
  ew-pytorch-spark:latest \
  bash -c "/opt/spark/sbin/start-worker.sh spark://192.168.1.10:7077 && tail -f /opt/spark/logs/*worker*"
ExecStop=/usr/bin/docker stop spark-worker

[Install]
WantedBy=multi-user.target
```

```bash
# Enable auto-start:
sudo systemctl enable spark-worker
sudo systemctl start spark-worker
```

### Windows Service (if cluster runs Windows)

```cmd
:: Create a scheduled task that runs at startup:
schtasks /create /tn "SparkWorker" /sc onstart /ru SYSTEM /tr ^
  "docker run --name spark-worker --gpus all ew-pytorch-spark:latest bash -c '/opt/spark/sbin/start-worker.sh spark://master:7077'"
```

---

## 10. Complete Example — 4 Node Cluster

### Network Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Network: 192.168.1.0/24 (airgapped, no internet)            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Node 1       │  │ Node 2       │                        │
│  │ 192.168.1.10 │  │ 192.168.1.11 │                        │
│  │              │  │              │                        │
│  │ Role: Master │  │ Role: Worker │                        │
│  │ + Driver     │  │ GPU: A100    │                        │
│  │ No GPU       │  │ RAM: 64GB    │                        │
│  │ RAM: 32GB    │  │ Cores: 16    │                        │
│  │              │  │              │                        │
│  │ Ports:       │  │ Container:   │                        │
│  │  7077 (Spark)│  │  spark-worker│                        │
│  │  8080 (UI)   │  │              │                        │
│  │  4040 (App)  │  │              │                        │
│  └──────────────┘  └──────────────┘                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Node 3       │  │ Node 4       │                        │
│  │ 192.168.1.12 │  │ 192.168.1.13 │                        │
│  │              │  │              │                        │
│  │ Role: Worker │  │ Role: Worker │                        │
│  │ GPU: A100    │  │ GPU: A100    │                        │
│  │ RAM: 64GB    │  │ RAM: 64GB    │                        │
│  │ Cores: 16    │  │ Cores: 16    │                        │
│  │              │  │              │                        │
│  │ Container:   │  │ Container:   │                        │
│  │  spark-worker│  │  spark-worker│                        │
│  └──────────────┘  └──────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Steps

```bash
# ═══ STEP 1: Load image on ALL nodes (one time) ═══

# Transfer ew-pytorch-spark.tar to all nodes via USB/network
# On each node:
docker load -i ew-pytorch-spark.tar


# ═══ STEP 2: Start Master (Node 1) ═══

# On 192.168.1.10:
docker run -d \
  --name spark-master \
  --network host \
  --restart unless-stopped \
  -v /data/ew-poc:/app \
  ew-pytorch-spark:latest \
  bash -c "/opt/spark/sbin/start-master.sh && tail -f /opt/spark/logs/*"

# Verify: open http://192.168.1.10:8080 — should show "Spark Master"


# ═══ STEP 3: Start Workers (Nodes 2, 3, 4) ═══

# On 192.168.1.11:
docker run -d \
  --name spark-worker \
  --network host \
  --gpus all \
  --restart unless-stopped \
  --shm-size=4g \
  ew-pytorch-spark:latest \
  bash -c "/opt/spark/sbin/start-worker.sh spark://192.168.1.10:7077 -c 16 -m 48g && tail -f /opt/spark/logs/*"

# On 192.168.1.12 (same command):
docker run -d \
  --name spark-worker \
  --network host \
  --gpus all \
  --restart unless-stopped \
  --shm-size=4g \
  ew-pytorch-spark:latest \
  bash -c "/opt/spark/sbin/start-worker.sh spark://192.168.1.10:7077 -c 16 -m 48g && tail -f /opt/spark/logs/*"

# On 192.168.1.13 (same command):
docker run -d \
  --name spark-worker \
  --network host \
  --gpus all \
  --restart unless-stopped \
  --shm-size=4g \
  ew-pytorch-spark:latest \
  bash -c "/opt/spark/sbin/start-worker.sh spark://192.168.1.10:7077 -c 16 -m 48g && tail -f /opt/spark/logs/*"

# Verify: http://192.168.1.10:8080 should show 3 workers


# ═══ STEP 4: Submit Job ═══

# On Node 1 (or any node with network access to master):
docker exec spark-master spark-submit \
  --master spark://192.168.1.10:7077 \
  --deploy-mode client \
  --num-executors 3 \
  --executor-cores 8 \
  --executor-memory 32g \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=0.5 \
  /app/benchmark/run_benchmark.py

# View live progress: http://192.168.1.10:4040


# ═══ STEP 5: View Results ═══

# Results are in /data/ew-poc/results/ on Node 1 (mounted volume)
cat /data/ew-poc/results/metrics_report.md
```

### What You See in Spark UI During Execution

```
http://192.168.1.10:8080 (Master UI):
┌─────────────────────────────────────────────────────────────┐
│ Spark Master at spark://192.168.1.10:7077                    │
│                                                              │
│ Workers (3):                                                 │
│  ┌────────────┬───────┬────────┬──────────────┐             │
│  │ Worker     │ Cores │ Memory │ State        │             │
│  ├────────────┼───────┼────────┼──────────────┤             │
│  │ 192.168.1.11│ 16    │ 48 GB  │ ALIVE        │             │
│  │ 192.168.1.12│ 16    │ 48 GB  │ ALIVE        │             │
│  │ 192.168.1.13│ 16    │ 48 GB  │ ALIVE        │             │
│  └────────────┴───────┴────────┴──────────────┘             │
│                                                              │
│ Running Applications (1):                                    │
│  ┌───────────────────────┬──────┬────────┬─────────┐        │
│  │ App Name              │ Cores│ Memory │ State   │        │
│  ├───────────────────────┼──────┼────────┼─────────┤        │
│  │ EW_PyTorch_Inference  │ 24   │ 96 GB  │ RUNNING │        │
│  └───────────────────────┴──────┴────────┴─────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘

http://192.168.1.10:4040 (Application UI):
┌─────────────────────────────────────────────────────────────┐
│ EW_PyTorch_Inference                                         │
│                                                              │
│ Active Jobs: 1                                               │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ Job 3: 500,000 signals                                │    │
│ │ [████████████████████░░░░░░░░] 75%                    │    │
│ │ 6/8 tasks complete | Duration: 3.2s                   │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                              │
│ Executor Summary:                                            │
│  Worker 11: 2 active tasks, GPU util 85%                     │
│  Worker 12: 2 active tasks, GPU util 82%                     │
│  Worker 13: 2 active tasks (completed), idle                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│              QUICK REFERENCE                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ VIEW SPARK UI:                                               │
│   1. Expose port 4040 in docker-compose                      │
│   2. Run benchmark                                           │
│   3. Open http://localhost:4040 in browser                   │
│                                                              │
│ DOCKER ON CLUSTER:                                           │
│   1. Load image on all nodes (docker load -i)                │
│   2. Start master container (node 1, --network host)         │
│   3. Start worker containers (nodes 2-N, --gpus all)         │
│   4. Submit job (spark-submit --master spark://node1:7077)   │
│   5. View progress at http://node1:8080 and :4040            │
│                                                              │
│ KEY INSIGHT:                                                 │
│   Docker containers are LONG-RUNNING services.               │
│   You start them ONCE. Spark sessions come and go.           │
│   Each spark-submit creates a new session inside             │
│   the already-running containers.                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
