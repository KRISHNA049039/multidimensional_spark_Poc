# Spark Cluster Architecture — Conceptual Guide for EW Inference

## Table of Contents

1. [What is a Spark Cluster?](#1-what-is-a-spark-cluster)
2. [Master vs Worker — Roles Explained](#2-master-vs-worker--roles-explained)
3. [How Spark Decides Master and Worker](#3-how-spark-decides-master-and-worker)
4. [Data Splitting and Partitioning](#4-data-splitting-and-partitioning)
5. [Our EW Inference — Local vs Cluster](#5-our-ew-inference--local-vs-cluster)
6. [Cluster Setup — What's Needed on Each Node](#6-cluster-setup--whats-needed-on-each-node)
7. [spark-submit — Running on Cluster](#7-spark-submit--running-on-cluster)
8. [Data Flow Example — 1M Signals on 4 Nodes](#8-data-flow-example--1m-signals-on-4-nodes)
9. [Fault Tolerance and Task Retry](#9-fault-tolerance-and-task-retry)
10. [Cluster Modes Comparison](#10-cluster-modes-comparison)

---

## 1. What is a Spark Cluster?

A Spark cluster is a group of machines (nodes) that work together to process data in parallel.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SPARK CLUSTER                                     │
│                                                                          │
│  ┌───────────────────────────────┐                                       │
│  │        MASTER NODE            │                                       │
│  │                               │                                       │
│  │  ┌─────────────────────────┐  │                                       │
│  │  │    Cluster Manager      │  │  Decides who does what                │
│  │  │  (resource allocation)  │  │  Monitors worker health               │
│  │  └─────────────────────────┘  │  Restarts failed tasks                │
│  │  ┌─────────────────────────┐  │                                       │
│  │  │    Driver Program       │  │  Your Python script runs here         │
│  │  │  (your run_benchmark.py)│  │  Sends tasks to workers               │
│  │  └─────────────────────────┘  │                                       │
│  └───────────────┬───────────────┘                                       │
│                  │                                                        │
│     ┌────────────┼────────────┬──────────────┐                           │
│     ▼            ▼            ▼              ▼                           │
│  ┌──────┐   ┌──────┐    ┌──────┐       ┌──────┐                        │
│  │Worker│   │Worker│    │Worker│       │Worker│                        │
│  │Node 1│   │Node 2│    │Node 3│       │Node N│                        │
│  │      │   │      │    │      │       │      │                        │
│  │[GPU] │   │[GPU] │    │[GPU] │       │[GPU] │                        │
│  │      │   │      │    │      │       │      │                        │
│  │Exec 1│   │Exec 2│    │Exec 3│       │Exec N│                        │
│  │Exec 2│   │Exec 3│    │Exec 4│       │      │                        │
│  └──────┘   └──────┘    └──────┘       └──────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Insight

In our local Docker setup: `local[*]` means the **master, driver, and all workers are the same machine**.
On a real cluster: they're separate physical/virtual machines connected by network.

---

## 2. Master vs Worker — Roles Explained

### The Master Node

| Responsibility | What It Does | In Our PoC |
|----------------|-------------|------------|
| Resource Management | Allocates CPU, memory, GPU to jobs | Decides how many executors to launch |
| Task Scheduling | Breaks job into tasks, assigns to workers | Splits our 1M signals into 8 partitions |
| Driver Execution | Runs your main Python script | Runs `run_benchmark.py` |
| Broadcast Variables | Sends shared data (model bytes) to all workers | Broadcasts our 1.4MB model to all executors |
| Result Collection | Gathers results from all workers | Collects `(count, correct)` tuples |
| Monitoring | Tracks worker health, restarts failed tasks | Web UI at port 8080 |

**The master does NOT process data.** It orchestrates.

### Worker Nodes

| Responsibility | What It Does | In Our PoC |
|----------------|-------------|------------|
| Execute Tasks | Run the actual computation | Deserialize model, run PyTorch inference |
| Store Data | Hold partitions of the dataset in memory | Hold chunk of signal features |
| Report Status | Send heartbeats and results back to master | Return `(n_samples, n_correct)` |
| GPU Computation | Run model forward pass on GPU | `model.to("cuda")`, process batch |

**Workers do ALL the heavy lifting.** Each worker runs one or more "executors" (JVM processes that run your Python code).

### The Driver

The driver is your Python script (`run_benchmark.py`). It can run on:
- The master node (client mode) — you see output in your terminal
- A worker node (cluster mode) — output goes to logs

```
┌─────────────────────────────────────────────────────────┐
│                    DRIVER PROGRAM                         │
│              (run_benchmark.py)                           │
│                                                          │
│  1. spark = create_spark_session()                        │
│  2. model_bytes = serialize_model(model)                  │
│  3. bc_model = sc.broadcast(model_bytes)    ──────────►  │ sent to all workers
│  4. rdd = sc.parallelize(chunks, 8)         ──────────►  │ distributed to workers
│  5. results = rdd.map(infer).collect()      ◄──────────  │ gathered from workers
│  6. print(throughput)                                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. How Spark Decides Master and Worker

Spark does NOT auto-discover who is master and who is worker. **You configure it explicitly.**

### Option A: Spark Standalone Cluster (simplest)

You manually start master on one machine and workers on others:

```bash
# On machine-1 (designated master):
$SPARK_HOME/sbin/start-master.sh
# Master starts listening on spark://machine-1:7077

# On machine-2 (worker):
$SPARK_HOME/sbin/start-worker.sh spark://machine-1:7077

# On machine-3 (worker):
$SPARK_HOME/sbin/start-worker.sh spark://machine-1:7077

# On machine-4 (worker):
$SPARK_HOME/sbin/start-worker.sh spark://machine-1:7077
```

The master URL (`spark://machine-1:7077`) is what connects everything. Workers register themselves with the master.

### Option B: YARN (Hadoop clusters)

YARN is the resource manager in Hadoop. You don't start Spark master/workers manually — YARN handles it:

```bash
# YARN decides which nodes run your executors
spark-submit --master yarn --deploy-mode cluster your_script.py
```

### Option C: Kubernetes

K8s scheduler decides where pods (executors) run:

```bash
spark-submit --master k8s://https://k8s-api:6443 your_script.py
```

### How Master Assignment Works

```
┌──────────────────────────────────────────────────────────────┐
│ YOU DECIDE (via configuration, not auto-discovery):           │
│                                                              │
│  ┌────────────────────┐     ┌────────────────────┐          │
│  │ spark-defaults.conf│     │ spark-submit cmd   │          │
│  │                    │     │                    │          │
│  │ spark.master=      │ OR  │ --master spark://  │          │
│  │  spark://host:7077 │     │   master-ip:7077   │          │
│  └────────────────────┘     └────────────────────┘          │
│                                                              │
│  Common values:                                              │
│  • local[*]              → everything on one machine         │
│  • local[8]              → 8 threads on one machine          │
│  • spark://master:7077   → Standalone cluster                │
│  • yarn                  → YARN manages resources            │
│  • k8s://...             → Kubernetes manages resources      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### In Our Code

```python
# Current (local mode — for Docker testing):
spark = SparkSession.builder.master("local[*]").getOrCreate()

# Cluster mode — change one line:
spark = SparkSession.builder.master("spark://master-node:7077").getOrCreate()

# Or better — don't hardcode, pass via spark-submit:
spark = SparkSession.builder.getOrCreate()
# Then: spark-submit --master spark://master:7077 benchmark/run_benchmark.py
```

---

## 4. Data Splitting and Partitioning

### What is a Partition?

A partition is a **chunk of data** that one task processes. Spark's parallelism = number of partitions being processed simultaneously.

```
1,000,000 signals ÷ 8 partitions = 125,000 signals per partition

┌──────────────────────────────────────────────────────────┐
│              ORIGINAL DATASET: 1M signals                 │
│                                                          │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐    │
│  │Part │Part │Part │Part │Part │Part │Part │Part │    │
│  │  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │    │
│  │125K │125K │125K │125K │125K │125K │125K │125K │    │
│  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘    │
│     │     │     │     │     │     │     │                │
└─────┼─────┼─────┼─────┼─────┼─────┼─────┼────────────────┘
      │     │     │     │     │     │     │
      ▼     ▼     ▼     ▼     ▼     ▼     ▼
   Worker1  W1    W2    W2    W3    W3    W4   W4
   (2 tasks each on 4 workers with 2 cores each)
```

### How Spark Decides Partitioning

| Method | How partitions are created | In our code |
|--------|---------------------------|-------------|
| `sc.parallelize(data, N)` | You specify N partitions | `sc.parallelize(range(8), 8)` |
| `df.repartition(N)` | Reshuffles data into N partitions | When using DataFrames |
| Reading from HDFS | 1 partition per HDFS block (128MB default) | In cluster with HDFS |
| Reading from Kafka | 1 partition per Kafka partition | In streaming mode |

### Our PoC Chunking Logic

```python
# We manually chunk data into num_partitions pieces:
num_partitions = 8
chunk_size = len(features) // num_partitions  # 1M / 8 = 125K per chunk

chunks = []
for i in range(num_partitions):
    start = i * chunk_size
    end = start + chunk_size if i < num_partitions - 1 else len(features)
    chunks.append((
        features[start:end].tobytes(),   # 125K × 128 × 4 bytes = ~64MB
        labels[start:end].tobytes(),
        end - start
    ))

# Distribute chunks across workers
rdd = spark.sparkContext.parallelize(range(num_partitions), num_partitions)
```

### How Many Partitions Should You Use?

```
Rule of thumb:
  partitions = 2-4 × total_cores_across_cluster

Example:
  4 worker nodes × 8 cores each = 32 cores
  → Use 64-128 partitions

Why more partitions than cores?
  - Load balancing (some partitions finish faster)
  - Fault tolerance (smaller retry unit if task fails)
  - Memory management (smaller chunks per task)
```

### Partition → Task Mapping

```
┌────────────────────────────────────────────────────┐
│ Spark Scheduler decides which worker runs which    │
│ partition based on:                                 │
│                                                    │
│  1. Data locality (run task where data lives)      │
│  2. Available resources (CPU/GPU slots free)       │
│  3. Fair scheduling (spread work evenly)           │
│  4. Speculation (re-run slow tasks elsewhere)      │
│                                                    │
│ For our inference workload:                        │
│  - Data locality doesn't matter (broadcast model)  │
│  - Resource availability is the main factor        │
│  - Each partition = 1 task = 1 PyTorch inference   │
└────────────────────────────────────────────────────┘
```

---

## 5. Our EW Inference — Local vs Cluster

### Local Mode (our Docker)

```
┌─── Single Machine (Docker container) ───────────────────┐
│                                                          │
│  Driver + Master + 8 Worker Threads = 1 process         │
│                                                          │
│  sc.parallelize creates 8 partitions                     │
│  → 8 Python threads run infer_partition()                │
│  → All share 1 GPU (contention!)                        │
│  → All share same RAM                                    │
│                                                          │
│  Limitation: GIL limits true parallelism                │
│  (but PyTorch releases GIL during forward pass)          │
└──────────────────────────────────────────────────────────┘
```

### Cluster Mode (your airgapped target)

```
┌─── Master Node ─────────────────────────────────────────┐
│  Driver: run_benchmark.py                                │
│  Broadcasts model (1.4MB) to all workers                 │
│  Broadcasts data chunks (via sc.broadcast)               │
│  Collects results: [(125000, 118750), ...]               │
└────────────────┬─────────────────────────────────────────┘
                 │ Network (10GbE / InfiniBand)
     ┌───────────┼───────────┬───────────┐
     ▼           ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker 4 │
│         │ │         │ │         │ │         │
│GPU: A100│ │GPU: A100│ │GPU: A100│ │GPU: A100│
│RAM: 64GB│ │RAM: 64GB│ │RAM: 64GB│ │RAM: 64GB│
│         │ │         │ │         │ │         │
│Process: │ │Process: │ │Process: │ │Process: │
│ 250K    │ │ 250K    │ │ 250K    │ │ 250K    │
│ signals │ │ signals │ │ signals │ │ signals │
│ (2 part)│ │ (2 part)│ │ (2 part)│ │ (2 part)│
└─────────┘ └─────────┘ └─────────┘ └─────────┘

Total: 1M signals processed in parallel across 4 GPUs
Throughput: ~4x single-node (linear scaling)
```

### Code Change: Local → Cluster

```python
# The ONLY change in your code:

# LOCAL (current):
spark = SparkSession.builder.master("local[*]").getOrCreate()

# CLUSTER (target):
spark = SparkSession.builder.master("spark://master:7077").getOrCreate()
# Or simply remove .master() and let spark-submit handle it

# Everything else stays IDENTICAL:
bc_model = sc.broadcast(model_bytes)
rdd = sc.parallelize(range(num_partitions), num_partitions)
results = rdd.map(infer_partition).collect()
```

---

## 6. Cluster Setup — What's Needed on Each Node

### Master Node Requirements

| Component | Required | Purpose |
|-----------|----------|---------|
| Java 17+ | Yes | Spark runtime |
| Spark binaries | Yes | Master daemon + driver |
| Python 3.11+ | Yes | Driver script execution |
| PyTorch | Yes | Model training (if training on master) |
| Network access to workers | Yes | Task scheduling, result collection |
| GPU | No | Master doesn't run inference |
| Large RAM | Recommended | Holds broadcast variables, collects results |

### Worker Node Requirements

| Component | Required | Purpose |
|-----------|----------|---------|
| Java 17+ | Yes | Executor JVM |
| Spark binaries | Yes | Worker daemon |
| Python 3.11+ | Yes | Task execution (PySpark) |
| PyTorch + CUDA | Yes | Model inference on GPU |
| NVIDIA GPU + drivers | Yes | Accelerated inference |
| Network access to master | Yes | Heartbeats, task results |
| RAM ≥ 8GB | Yes | Hold data partition + model in memory |

### File Structure on Each Node

```
/opt/spark/                    # Spark installation
├── bin/
│   ├── spark-submit
│   ├── pyspark
│   └── spark-shell
├── sbin/
│   ├── start-master.sh        # Run on master only
│   └── start-worker.sh        # Run on workers only
├── conf/
│   ├── spark-defaults.conf    # Default configs
│   ├── spark-env.sh           # Environment variables
│   └── workers                # List of worker hostnames
└── jars/                      # Spark JARs

/opt/ew-poc/                   # Our application
├── models/
├── inference/
├── benchmark/
├── data/
└── results/

/usr/local/lib/python3.11/     # Python packages
└── site-packages/
    ├── torch/
    ├── pyspark/
    ├── numpy/
    └── pandas/
```

### Configuration Files

**`/opt/spark/conf/spark-defaults.conf`** (on all nodes):
```properties
spark.executor.memory=8g
spark.executor.cores=4
spark.executor.resource.gpu.amount=1
spark.task.resource.gpu.amount=0.5
spark.network.timeout=600s
spark.executor.heartbeatInterval=120s
spark.driver.extraJavaOptions=--add-opens=java.base/java.nio=ALL-UNNAMED
```

**`/opt/spark/conf/workers`** (on master only):
```
worker-node-1
worker-node-2
worker-node-3
worker-node-4
```

**`/opt/spark/conf/spark-env.sh`** (on all nodes):
```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PYSPARK_PYTHON=/usr/bin/python3.11
export SPARK_WORKER_CORES=8
export SPARK_WORKER_MEMORY=32g
export SPARK_WORKER_GPU_AMOUNT=1
```

---

## 7. spark-submit — Running on Cluster

### Basic Command

```bash
spark-submit \
  --master spark://master-node:7077 \
  --deploy-mode client \
  --num-executors 8 \
  --executor-cores 4 \
  --executor-memory 8g \
  --driver-memory 4g \
  --conf spark.executor.resource.gpu.amount=1 \
  /opt/ew-poc/benchmark/run_benchmark.py
```

### What Each Flag Does

| Flag | Value | Meaning |
|------|-------|---------|
| `--master` | `spark://master:7077` | Connect to this Spark master |
| `--deploy-mode` | `client` | Run driver on submitting machine (see output) |
| `--deploy-mode` | `cluster` | Run driver on a worker (output goes to logs) |
| `--num-executors` | `8` | Request 8 executor processes across cluster |
| `--executor-cores` | `4` | Each executor gets 4 CPU cores |
| `--executor-memory` | `8g` | Each executor gets 8GB RAM |
| `--driver-memory` | `4g` | Driver process gets 4GB RAM |
| `--conf spark...gpu` | `1` | Each executor gets 1 GPU |

### Client vs Cluster Deploy Mode

```
CLIENT MODE (--deploy-mode client):
┌───────────────────────────────────────────────┐
│ Your terminal on master/submit machine:        │
│   $ spark-submit ... run_benchmark.py          │
│   [prints output here — you can see results]   │
│                                                │
│   Driver runs HERE (your machine)              │
│   Workers run on cluster nodes                 │
└───────────────────────────────────────────────┘

CLUSTER MODE (--deploy-mode cluster):
┌───────────────────────────────────────────────┐
│ Your terminal:                                 │
│   $ spark-submit ... run_benchmark.py          │
│   [returns immediately — job submitted]        │
│                                                │
│   Driver runs on some worker node              │
│   Output goes to Spark UI / log files          │
│   (used in production, CI/CD)                  │
└───────────────────────────────────────────────┘
```

---

## 8. Data Flow Example — 1M Signals on 4 Nodes

Step-by-step execution of our benchmark on a real cluster:

### Step 1: Driver Starts

```
Master Node (driver):
  → Loads run_benchmark.py
  → Creates SparkSession connected to master at :7077
  → Trains model locally (10K samples, 20 epochs)
  → Generates 1M synthetic signals (numpy array, ~512MB)
```

### Step 2: Broadcast Model

```
Driver broadcasts model_bytes (1.4MB) to all workers:

  Master ──── 1.4MB ────► Worker 1 (caches in memory)
         ──── 1.4MB ────► Worker 2 (caches in memory)
         ──── 1.4MB ────► Worker 3 (caches in memory)
         ──── 1.4MB ────► Worker 4 (caches in memory)

  Broadcast is a one-time transfer. Workers cache it.
  Even if a worker runs 4 tasks, it only receives the model ONCE.
```

### Step 3: Broadcast Data Chunks

```
Driver chunks 1M signals into 8 pieces (125K each):
  Chunk 0: signals[0:125000]        → broadcast_chunk_0
  Chunk 1: signals[125000:250000]   → broadcast_chunk_1
  ...
  Chunk 7: signals[875000:1000000]  → broadcast_chunk_7

Each chunk = 125000 × 128 × 4 bytes = ~64MB
Total broadcast: 8 × 64MB = 512MB distributed across workers

Note: On a real cluster with HDFS, data would ALREADY be
distributed — no broadcast needed. Workers read local files.
```

### Step 4: Task Scheduling

```
Driver creates RDD with 8 partitions:
  rdd = sc.parallelize(range(8), 8)

Spark scheduler assigns tasks to workers:
  Worker 1: Task 0 (partition 0), Task 1 (partition 1)
  Worker 2: Task 2 (partition 2), Task 3 (partition 3)
  Worker 3: Task 4 (partition 4), Task 5 (partition 5)
  Worker 4: Task 6 (partition 6), Task 7 (partition 7)

2 tasks per worker (8 partitions ÷ 4 workers)
```

### Step 5: Parallel Inference

```
All workers execute simultaneously:

Worker 1 (GPU 0):                    Worker 2 (GPU 1):
┌─────────────────────┐             ┌─────────────────────┐
│ Task 0:             │             │ Task 2:             │
│  load model from    │             │  load model from    │
│    broadcast cache  │             │    broadcast cache  │
│  model.to("cuda:0")│             │  model.to("cuda:0")│
│  for batch in 0..122:│            │  for batch in 0..122:│
│    125K/1024 batches│             │    125K/1024 batches│
│    tensor.to(gpu)   │             │    tensor.to(gpu)   │
│    predictions =    │             │    predictions =    │
│      model(batch)   │             │      model(batch)   │
│  return (125000,    │             │  return (125000,    │
│          118750)    │             │          118700)    │
│                     │             │                     │
│ Task 1: (same)     │             │ Task 3: (same)     │
└─────────────────────┘             └─────────────────────┘

Worker 3 (GPU 2):                    Worker 4 (GPU 3):
  [same pattern]                       [same pattern]

All 8 tasks run in parallel → 1M signals in ~2.5 seconds
vs sequential: 1M signals in ~10+ seconds
```

### Step 6: Collect Results

```
Workers send results back to driver:

Worker 1 → (125000, 118750), (125000, 118800)
Worker 2 → (125000, 118700), (125000, 118650)
Worker 3 → (125000, 118900), (125000, 118750)
Worker 4 → (125000, 118800), (125000, 118850)

Driver aggregates:
  total_count = sum of first elements = 1,000,000
  total_correct = sum of second elements = 950,200
  accuracy = 950,200 / 1,000,000 = 95.02%
  throughput = 1,000,000 / 2.5sec = 400,000 samples/sec
```

### Timeline

```
Time ──────────────────────────────────────────────────────►
  0s        1s        2s        3s        4s        5s

Driver:
  [train]────[broadcast]──[schedule]──────────────[collect]─

Worker 1:
  ............[load model]─[infer 250K]───────────..........

Worker 2:
  ............[load model]─[infer 250K]───────────..........

Worker 3:
  ............[load model]─[infer 250K]───────────..........

Worker 4:
  ............[load model]─[infer 250K]───────────..........

Total wall time: ~5s (with overhead)
Inference time: ~2.5s (just the parallel compute)
```

---

## 9. Fault Tolerance and Task Retry

### What Happens When a Worker Fails?

```
Scenario: Worker 3 crashes mid-inference (GPU OOM, hardware fault, etc.)

Before failure:
  Worker 1: ████████████ Done (Task 0, 1)
  Worker 2: ████████████ Done (Task 2, 3)
  Worker 3: ██████░░░░░░ CRASHED (Task 4 lost, Task 5 not started)
  Worker 4: ████████████ Done (Task 6, 7)

Spark automatic recovery:
  1. Master detects Worker 3 heartbeat timeout (120s)
  2. Master marks Task 4, Task 5 as FAILED
  3. Master re-schedules Task 4 on Worker 1 (which has free slot)
  4. Master re-schedules Task 5 on Worker 2 (which has free slot)
  5. Tasks re-run from scratch (read from broadcast cache)

After recovery:
  Worker 1: ████████████ Done (Task 0, 1) + ████ Re-run (Task 4)
  Worker 2: ████████████ Done (Task 2, 3) + ████ Re-run (Task 5)
  Worker 3: ✗ Dead
  Worker 4: ████████████ Done (Task 6, 7)

Result: Job completes (slower, but no data loss)
```

### Speculative Execution

```
Scenario: Worker 3 is slow (thermal throttling, background load)

  Worker 1: ████████████ Done
  Worker 2: ████████████ Done
  Worker 3: ██████░░░░░░ Still running (straggler)
  Worker 4: ████████████ Done

With spark.speculation=true:
  1. Master notices Worker 3 is 50% slower than median
  2. Master launches DUPLICATE of Task 4 on Worker 1
  3. Whichever finishes first wins (other is killed)
  4. Job completes without waiting for slow node

Critical for EW: mission latency cannot depend on one slow node.
```

---

## 10. Cluster Modes Comparison

| Feature | Standalone | YARN | Kubernetes |
|---------|-----------|------|------------|
| Complexity | Simple | Medium | Complex |
| Setup | Manual start-master/worker | Part of Hadoop | K8s cluster required |
| Resource sharing | Spark-only | Multi-tenant (with Hadoop) | Multi-tenant (with other apps) |
| Airgapped friendly | Very (just Spark binaries) | Needs Hadoop ecosystem | Needs K8s infrastructure |
| GPU support | Manual config | YARN GPU scheduler | K8s device plugin |
| Best for | Dedicated Spark cluster | Existing Hadoop infra | Cloud/container environments |
| Our recommendation | **Yes (simplest for airgapped)** | If you already have Hadoop | If you already have K8s |

### Standalone Cluster — Startup Script

```bash
#!/bin/bash
# start_cluster.sh — run on master node

# Start master
$SPARK_HOME/sbin/start-master.sh

# Start workers (reads from $SPARK_HOME/conf/workers file)
$SPARK_HOME/sbin/start-workers.sh

# Or start individual workers manually:
# ssh worker-1 "$SPARK_HOME/sbin/start-worker.sh spark://$(hostname):7077"
# ssh worker-2 "$SPARK_HOME/sbin/start-worker.sh spark://$(hostname):7077"
```

### Docker-Based Cluster (for airgapped with Docker)

```yaml
# docker-compose.cluster.yml
services:
  master:
    image: ew-pytorch-spark:latest
    container_name: spark-master
    command: >
      bash -c "
        $SPARK_HOME/sbin/start-master.sh &&
        tail -f $SPARK_HOME/logs/*master*
      "
    ports:
      - "7077:7077"   # Spark master port
      - "8080:8080"   # Web UI
    environment:
      - SPARK_MODE=master

  worker-1:
    image: ew-pytorch-spark:latest
    container_name: spark-worker-1
    runtime: nvidia
    command: >
      bash -c "
        $SPARK_HOME/sbin/start-worker.sh spark://master:7077 &&
        tail -f $SPARK_HOME/logs/*worker*
      "
    depends_on:
      - master
    environment:
      - SPARK_MODE=worker
      - SPARK_WORKER_CORES=4
      - SPARK_WORKER_MEMORY=8g

  worker-2:
    image: ew-pytorch-spark:latest
    container_name: spark-worker-2
    runtime: nvidia
    command: >
      bash -c "
        $SPARK_HOME/sbin/start-worker.sh spark://master:7077 &&
        tail -f $SPARK_HOME/logs/*worker*
      "
    depends_on:
      - master
    environment:
      - SPARK_MODE=worker
      - SPARK_WORKER_CORES=4
      - SPARK_WORKER_MEMORY=8g
```

---

## Summary: Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  YOU write:    Python code (model, inference logic)           │
│  SPARK does:   Distribution, scheduling, fault tolerance     │
│  GPU does:     Matrix multiplication (model forward pass)    │
│                                                              │
│  You don't manage:                                           │
│    ✗ Which machine runs which chunk                          │
│    ✗ What happens when a node fails                          │
│    ✗ How data moves between nodes                            │
│    ✗ Load balancing                                          │
│                                                              │
│  You DO decide:                                              │
│    ✓ How many partitions (num_partitions)                    │
│    ✓ Batch size for PyTorch (batch_size)                     │
│    ✓ How many executors to request (--num-executors)         │
│    ✓ Resources per executor (--executor-memory, --gpu)       │
│                                                              │
│  Scaling is LINEAR:                                          │
│    2 workers → 2x throughput                                 │
│    4 workers → 4x throughput                                 │
│    8 workers → 8x throughput                                 │
│    (for embarrassingly parallel workloads like inference)    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
