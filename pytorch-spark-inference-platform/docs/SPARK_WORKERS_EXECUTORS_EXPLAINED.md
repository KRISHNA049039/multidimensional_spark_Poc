# Spark Workers, Executors & Cores — How They Relate

This document explains the relationship between Spark Workers, Executors, Cores, Tasks, and Partitions for the AIA team.

---

## 1. The Hierarchy

```
CLUSTER
 └── Master (1 per cluster)
      ├── Worker Node 1 (1 worker process per physical/virtual machine)
      │    └── Executor 1 (1 or more per worker, depending on config)
      │         ├── Task A (runs on 1 core slot)
      │         ├── Task B (runs on 1 core slot)
      │         └── ... (up to executor.cores / task.cpus tasks in parallel)
      ├── Worker Node 2
      │    └── Executor 2
      │         ├── Task C
      │         └── Task D
      └── Worker Node N
           └── Executor N
                └── ...
```

---

## 2. Definitions

| Term | What it is | Analogy |
|------|-----------|---------|
| **Cluster** | All machines managed by one Spark Master | The whole office building |
| **Master** | Coordinator daemon. Tracks workers, schedules apps | Building manager |
| **Worker** | A daemon on each machine that registers with Master | A floor in the building |
| **Executor** | A JVM process inside a Worker that actually runs tasks | A desk on that floor |
| **Task** | A unit of work (1 partition = 1 task) | A document to process at the desk |
| **Core (slot)** | A "slot" in an executor that can run 1 task at a time | A person sitting at the desk |
| **Partition** | A chunk of data. 1 partition creates 1 task | A pile of documents |

---

## 3. How Many of Each?

### Workers per Node

**Answer: 1 Worker per physical machine** (always).

A Worker is just a management daemon. You don't create multiple workers on one machine — that would just waste overhead. The Worker's job is to:
- Register with the Master ("I have X cores and Y GB RAM available")
- Launch Executors when an application is submitted
- Report health back to Master

### Executors per Worker

**Answer: Usually 1 Executor per Worker** (in standalone mode).

In Spark standalone mode (what we use), each application gets **1 Executor per Worker**. The executor gets all the cores and memory that the Worker advertised.

```
Worker started with: -c 4 -m 12g
  → When an app is submitted, 1 Executor is created with 4 cores, 12 GB

You CANNOT have 2 executors on 1 worker in standalone mode (unlike YARN).
```

**Exception (YARN/Kubernetes):** On YARN, you can configure multiple executors per node:
```
spark.executor.instances=3     # 3 executors total
spark.executor.cores=4         # each gets 4 cores
spark.executor.memory=8g       # each gets 8GB
# If node has 16 cores, 32GB → can fit 3-4 executors
```

### Cores per Executor

**Answer: Configurable.** Controls how many tasks run in parallel on that executor.

```
executor.cores = 4, task.cpus = 1  → 4 tasks run simultaneously on this executor
executor.cores = 4, task.cpus = 2  → 2 tasks run simultaneously
executor.cores = 4, task.cpus = 4  → 1 task at a time (our current config)
```

### Tasks per Executor

**Formula:** `concurrent_tasks = executor.cores / task.cpus`

| executor.cores | task.cpus | Concurrent Tasks | Best For |
|---------------|-----------|-----------------|----------|
| 4 | 1 | 4 | Lightweight tasks (SQL, simple transforms) |
| 4 | 2 | 2 | Medium tasks (our distributed inference) |
| 4 | 4 | 1 | Heavy tasks (full GPU model loading) |
| 8 | 2 | 4 | Multi-threaded inference, good parallelism |

---

## 4. Our Current Setup (AWS POC)

```
┌────────────────────────────────────────────────────────┐
│ Spark Master (coordinator only, no compute)             │
│ URL: spark://10.0.0.187:7077                            │
└────────────────────────────────────────────────────────┘
         │
         ├── Worker 1 (on master machine, 10.0.0.187)
         │    Registered with: -c 4 -m 12g
         │    └── Executor 0: 2 cores, 12 GB, device=cpu
         │         └── max 1 task at a time (cores=2, task.cpus=2)
         │
         └── Worker 2 (GPU machine, 10.0.0.45)
              Registered with: -c 4 -m 12g
              └── Executor 1: 2 cores, 12 GB, device=cuda
                   └── max 1 task at a time (cores=2, task.cpus=2)

Total capacity: 2 concurrent tasks
```

### Why executor.cores=2 and task.cpus=2?

We set `task.cpus=2` because our inference task:
- Loads 10 heavy models (needs memory headroom)
- Uses PyTorch (internally multi-threaded for matrix ops)
- Benefits from NOT sharing the CPU with another task

With `task.cpus=1` and `executor.cores=2`, we'd get 2 tasks per executor running simultaneously — but they'd compete for GPU memory and CPU cache, likely reducing per-task throughput.

---

## 5. Air-Gapped 5-Node System

### Recommended Config

```
Each node: 256 GB RAM, 24 GB VRAM, ~16-32 CPU cores

┌──────────────────────────────────────────────────────────────────┐
│ Spark Master (Node 1, also runs a Worker)                         │
│ URL: spark://192.168.1.10:7077                                    │
└──────────────────────────────────────────────────────────────────┘
         │
         ├── Worker 1 (Node 1): -c 8 -m 200g
         │    └── Executor: 8 cores, 200 GB, device=cuda
         │         └── 4 concurrent tasks (cores=8, task.cpus=2)
         │
         ├── Worker 2 (Node 2): -c 8 -m 200g
         │    └── Executor: 8 cores, 200 GB, device=cuda
         │         └── 4 concurrent tasks
         │
         ├── Worker 3 (Node 3): -c 8 -m 200g
         │    └── Executor: 8 cores, 200 GB, device=cuda
         │         └── 4 concurrent tasks
         │
         ├── Worker 4 (Node 4): -c 8 -m 200g
         │    └── Executor: 8 cores, 200 GB, device=cuda
         │         └── 4 concurrent tasks
         │
         └── Worker 5 (Node 5): -c 8 -m 200g
              └── Executor: 8 cores, 200 GB, device=cuda
                   └── 4 concurrent tasks

Total capacity: 5 executors × 4 tasks = 20 concurrent tasks
```

### Why 8 cores and task.cpus=2?

- Each node has 16-32 cores
- Offer 8 to Spark (leave rest for OS, Docker, monitoring)
- `task.cpus=2` gives 4 concurrent tasks per executor
- Each task runs 10 models with batch inference
- 4 tasks per GPU is fine because models share GPU via CUDA MPS

### Optimal Partitions

```
If each executor runs 4 concurrent tasks:
  5 executors × 4 = 20 total task slots
  Set partitions = 20 for full utilization
  Each partition gets 1/20th of the data
```

---

## 6. The Relationship Chain

```
1 Cluster → 1 Master + N Worker Nodes
1 Worker Node → 1 Worker daemon → 1 Executor (standalone) or M Executors (YARN)
1 Executor → C/T concurrent tasks (C=executor.cores, T=task.cpus)
1 Task → processes 1 Partition of data
1 Job → 1+ Stages → many Tasks

So:
  partitions = total tasks created
  concurrent_tasks = sum(executor_cores / task_cpus) across all executors
  rounds = ceil(partitions / concurrent_tasks)
  wall_time ≈ rounds × time_per_task + spark_overhead
```

---

## 7. Can We Have Multiple Executors Per Node?

### Standalone Mode (what we use): **No**

1 Worker → 1 Executor per application. Period.

If you need more parallelism, increase `executor.cores` (more concurrent tasks in the single executor).

### YARN Mode: **Yes**

```yaml
# YARN example: 3 executors per node
spark.executor.instances: 15    # 5 nodes × 3 executors each
spark.executor.cores: 4         # each executor gets 4 cores
spark.executor.memory: 60g      # each executor gets 60GB (node has 200GB)
```

### Kubernetes Mode: **Yes**

Each executor is a separate pod. K8s schedules pods across nodes.

### Multiple Workers Hack (Standalone): **Yes, but not recommended**

You CAN start multiple Worker daemons on one machine with different ports:
```bash
# Worker 1 on port 8081
start-worker.sh spark://master:7077 -c 4 -m 100g --webui-port 8081
# Worker 2 on port 8082
start-worker.sh spark://master:7077 -c 4 -m 100g --webui-port 8082
```
This creates 2 executors on 1 node. But they'd compete for GPU → not useful for inference.

---

## 8. GPU and Executors

### The GPU Problem

```
1 GPU per machine → 1 executor should own it exclusively
If 2 executors share 1 GPU → OOM or contention

Solutions:
1. 1 executor per GPU (recommended) — our approach
2. NVIDIA MPS — allows safe multi-process GPU sharing
3. GPU resource scheduling (Spark 3.0+) — assigns GPUs to executors
```

### Spark GPU Resource Config (for air-gapped)

```properties
# spark-defaults.conf
spark.executor.resource.gpu.amount=1
spark.task.resource.gpu.amount=1
spark.worker.resource.gpu.amount=1
spark.worker.resource.gpu.discoveryScript=/opt/spark/scripts/getGpusResources.sh
```

This tells Spark "each executor needs 1 GPU" — ensures no two executors fight over the same GPU.

---

## 9. Quick Reference

| Question | Answer |
|----------|--------|
| Workers per node? | **1** (always, it's a daemon) |
| Executors per worker? | **1** in standalone, configurable in YARN/K8s |
| Tasks per executor? | `executor.cores / task.cpus` |
| Partitions = Tasks? | **Yes** (1 partition = 1 task) |
| How to increase parallelism? | More nodes OR increase executor.cores OR decrease task.cpus |
| How to use GPU in distributed? | 1 executor per GPU node, set CUDA env vars |
| Dynamic scaling? | Not in standalone. Use YARN/K8s for auto-scaling |
| Our POC | 2 executors, 1 task each, 2 tasks total |
| Air-gapped target | 5 executors, 4 tasks each, 20 tasks total |

---

## 10. Visual: How a Job Flows

```
You run: python cluster_benchmark.py --partitions 6

                    ┌─────────────────────┐
                    │   DRIVER (Master)    │
                    │                     │
                    │ 1. Create 6 chunks  │
                    │ 2. Create RDD(6)    │
                    │ 3. Submit Job       │
                    └─────────┬───────────┘
                              │
                    ┌─────────┴───────────┐
                    │   SPARK SCHEDULER    │
                    │                     │
                    │ "I have 6 tasks and │
                    │  2 executor slots"  │
                    └─────────┬───────────┘
                              │
              Round 1:        │
              ┌───────────────┼───────────────┐
              ▼                               ▼
     ┌─────────────────┐            ┌─────────────────┐
     │ Executor 0 (CPU)│            │ Executor 1 (GPU)│
     │ Task: Partition 1│            │ Task: Partition 0│
     │ Time: 9.14s      │            │ Time: 0.69s     │
     └─────────────────┘            └────────┬────────┘
                                              │ Done! Get next task
              Round 2:                        ▼
                                    ┌─────────────────┐
     (CPU still running P1)         │ Executor 1 (GPU)│
                                    │ Task: Partition 2│
                                    │ Time: 0.32s     │
                                    └────────┬────────┘
              Round 3:                        ▼
                                    ┌─────────────────┐
     (CPU still running P1)         │ Executor 1 (GPU)│
                                    │ Task: Partition 3│
                                    │ Time: 0.33s     │
                                    └────────┬────────┘
              Round 4:                        ▼
                                    ┌─────────────────┐
     (CPU still running P1)         │ Executor 1 (GPU)│
                                    │ Task: Partition 4│
                                    │ Time: 0.33s     │
                                    └────────┬────────┘
              Round 5:                        ▼
                                    ┌─────────────────┐
     (CPU FINALLY done at 9.14s)    │ Executor 1 (GPU)│
                                    │ Task: Partition 5│
                                    │ Time: 0.34s     │
                                    └─────────────────┘

Total: GPU did 5 tasks in 2.0s, CPU did 1 task in 9.14s
Wall clock = 9.14s + Spark overhead = 21.1s
```

---

## 11. Summary

- **1 machine = 1 Worker = 1 Executor** (in standalone Spark)
- **Executor cores / task cpus = concurrent tasks** per executor
- **Partitions = total tasks** distributed across all executors
- **Spark automatically sends more tasks to faster executors** (work-stealing)
- **GPU node finishes tasks 13-29x faster** → gets 5x more tasks than CPU node
- **For production:** Set partitions = num_executors × concurrent_tasks_per_executor
