# Changelog — July 20-24, 2026

All changes made during the Windows lab cluster setup, multi-node testing, and production architecture documentation.

---

## July 20, 2026 — Initial Cluster Setup & Documentation

### New Files Created

| File | Purpose |
|------|---------|
| `deploy/docker-compose.cluster.yml` | Windows Docker Desktop cluster (master + CPU workers + GPU workers) |
| `deploy/docker-compose.cluster.linux.yml` | Linux production cluster (--network host, nvidia runtime) |
| `deploy/start_cluster.ps1` | One-click PowerShell script to start cluster across multiple machines via WinRM |
| `deploy/stop_cluster.ps1` | Graceful cluster shutdown script (workers first, then master) |
| `docs/WINDOWS_CLUSTER_SETUP.md` | Full Windows lab setup guide (single-machine compose + multi-machine manual) |
| `docs/PRODUCTION_ARCHITECTURE.md` | Production architecture for 10 concurrent models in air-gapped 5-node cluster |
| `benchmark/quick_compare.py` | Fast 2-mode comparison (gpu_only + hybrid) with minimal data |
| `benchmark/capture_spark_stats.py` | Captures Spark UI stats (jobs/stages/executors) from port 4040 REST API |

---

## July 24, 2026 — Fixing Cluster Issues & Enhancing Benchmarks

### Modified Files

#### `deploy/Dockerfile`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| Base image | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` | `nvidia/cuda:12.6.3-runtime-ubuntu22.04` | CUDA 12.6 needed for RTX 5060 (sm_120) |
| PyTorch version | `torch==2.2.0 torchvision==0.17.0 --index-url .../cu121` | `torch==2.6.0 torchvision==0.21.0 --index-url .../cu126` | PyTorch 2.6+ needed for sm_120 (Blackwell) GPU support |

---

#### `inference/distributed_gpu.py`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| Executor memory | `spark.executor.memory = 12g` | `spark.executor.memory = 2g` | 12GB didn't fit in 4g/8g workers causing silent WAITING state |
| Driver memory | `spark.driver.memory = 10g` | `spark.driver.memory = 2g` | Reduced to match lab machine resources |
| Task CPUs | `spark.task.cpus = 2` | `spark.task.cpus = 1` | Allow more tasks per executor |
| Executor return data | Simple `{model: count}` dict | Detailed dict with executor info, per-model timing, input/output shapes | User requested full statistics |
| Spark UI capture | Not captured | Captures jobs/stages/executors from 4040 REST API after inference | Persist cluster metrics |

**Detailed changes to `infer_on_partition` function:**
- Now returns `executor` info (hostname, PID, device, GPU name, CPU count)
- Now returns `timing` breakdown (model_load_time, inference_time, total_task_time)
- Now returns per-model `input_shape` and `output_shape`
- Now returns per-model `throughput` and `batches` count

---

#### `benchmark/run_benchmark.py`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| Output filenames | Fixed `metrics_report.md` / `raw_results.json` (overwritten each run) | Timestamped: `report_{mode}_{config}_{timestamp}.md` + `results_{mode}_{config}_{timestamp}.json` | Results were being overridden across runs |
| Latest symlink | N/A | Also writes `metrics_report_latest.md` and `raw_results_latest.json` | Quick access to most recent run |
| Custom run name | N/A | `RUN_NAME` env var prepended to filename | Organize runs by experiment |
| Distributed mode output | Basic 3 lines (throughput, partitions, time) | Full detailed printout: job composition, executor/worker detail, model I/O, Spark metrics | User requested complete statistics |

**New output sections in distributed mode:**
- JOB COMPOSITION — what the job does (stages, tasks per partition)
- EXECUTOR / WORKER DETAIL — hostname, device, GPU, model load time, inference time per partition
- WORKER SUMMARY — unique workers, tasks handled, samples processed
- MODEL INPUT/OUTPUT & THROUGHPUT — input shape, output shape, per-model throughput
- SPARK EXECUTOR METRICS — from REST API (cores, completed tasks, duration)
- SPARK JOBS — status, stages completed, tasks completed

---

#### `deploy/docker-compose.cluster.yml`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| Volume mounts | None | `../results`, `../benchmark`, `../inference`, `../models`, `../data` mounted | Results appear on host; code changes picked up without rebuild |
| `CUDA_VISIBLE_DEVICES=` | Set on all containers | Removed (GPU now supported with PyTorch 2.6) | Enable GPU inference |
| GPU reservation | Removed (disabled) | Re-enabled `deploy.resources.reservations.devices` on master + gpu-worker | Pass GPU into containers |
| Comments | Pointed to old benchmark commands | Updated usage examples | Reflect current workflow |

---

#### `benchmark/quick_compare.py`

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| Modes tested | cpu_only + gpu_only + hybrid | gpu_only + hybrid only | User requested removing cpu_only mode |

---

### Issues Encountered & Resolved

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `invalid reference format` | PowerShell backticks around image name | Run as single line |
| 2 | `container name already in use` | Leftover container from previous run | `docker rm -f spark-master` |
| 3 | Netty bind failure with `-h <host-ip>` | Container can't bind to host's external IP | Don't pass `-h` flag to start-master.sh |
| 4 | Worker Netty bind failure | `SPARK_LOCAL_IP` set to unreachable host IP | Don't set `SPARK_LOCAL_IP` on Windows Docker Desktop |
| 5 | Machines on different subnets | Master on 192.168.1.x, worker on 10.181.x.x | Found common Ethernet subnet (192.168.4.x) |
| 6 | Worker registered but executor exit code 1 | Driver hostname (`686832736e42`) unresolvable from worker | Used Docker network with service names instead |
| 7 | App stuck in WAITING state | Executor requests 12GB, worker only has 4-8GB | Reduced `spark.executor.memory` to 2GB |
| 8 | `no kernel image available for execution on device` | RTX 5060 (sm_120) not supported by PyTorch 2.2.0 | Upgraded to PyTorch 2.6.0 + CUDA 12.6 |
| 9 | `docker cp` file not found | New scripts created after image was built | Volume mounts solve this permanently |
| 10 | Results overwritten each run | Fixed filename `metrics_report.md` | Timestamped filenames per mode/config |
| 11 | `${PWD}` not working in CMD | Bash/PowerShell syntax in CMD terminal | Use absolute paths in CMD |
| 12 | `--network host` not working on Windows | Docker Desktop NAT limitation | Use Docker bridge network with service names |

---

### Working Commands (as of July 24)

**Start cluster:**
```cmd
cd D:\Spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform
docker compose -f deploy/docker-compose.cluster.yml down
docker build -t multi-model-inference:latest -f deploy/Dockerfile .
docker compose -f deploy/docker-compose.cluster.yml up
```

**Run distributed benchmark (GPU):**
```cmd
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://spark-master:7077 python benchmark/run_benchmark.py --mode distributed --signal-samples 10000 --image-samples 100 --detection-samples 20 --batch-size 64 --partitions 4"
```

**Run cluster benchmark (device mode control):**
```cmd
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://spark-master:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 5000"
```

**Run incremental load test:**
```cmd
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://spark-master:7077 python benchmark/incremental_load_test.py"
```

**Run all modes comparison:**
```cmd
docker exec -it spark-master bash -c "SPARK_MASTER_URL=spark://spark-master:7077 python benchmark/run_benchmark.py --mode all --signal-samples 5000 --partitions 4"
```

**Check results (on host, no docker cp needed):**
```cmd
dir results\
type results\metrics_report_latest.md
```

---

### File Tree After Changes

```
pytorch-spark-inference-platform/
├── benchmark/
│   ├── capture_spark_stats.py      [NEW] Spark UI stats capture
│   ├── cluster_benchmark.py        [existing] Device mode cluster test
│   ├── incremental_load_test.py    [existing] Scaling load test
│   ├── quick_compare.py            [NEW] Fast 2-mode comparison
│   ├── run_benchmark.py            [MODIFIED] Timestamped output + detailed stats
│   └── __init__.py
├── deploy/
│   ├── docker-compose.cluster.yml  [MODIFIED] Volumes + GPU support + CUDA fix
│   ├── docker-compose.cluster.linux.yml [NEW] Linux production version
│   ├── docker-compose.yml          [existing] Dev mode
│   ├── Dockerfile                  [MODIFIED] PyTorch 2.6 + CUDA 12.6
│   ├── start_cluster.ps1           [NEW] Multi-machine startup automation
│   └── stop_cluster.ps1            [NEW] Multi-machine shutdown automation
├── docs/
│   ├── CHANGELOG_20260720.md       [NEW] This file
│   ├── PRODUCTION_ARCHITECTURE.md  [NEW] Full production architecture + pitfalls
│   ├── WINDOWS_CLUSTER_SETUP.md    [NEW] Windows lab setup guide
│   └── ... (existing docs)
├── inference/
│   └── distributed_gpu.py          [MODIFIED] Reduced memory + detailed executor stats
├── results/
│   ├── report_{mode}_{config}_{timestamp}.md   [per-run reports]
│   ├── results_{mode}_{config}_{timestamp}.json [per-run data]
│   ├── metrics_report_latest.md    [latest run]
│   └── raw_results_latest.json     [latest run]
└── ...
```
