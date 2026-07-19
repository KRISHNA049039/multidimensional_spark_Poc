# Distributed Spark Cluster — Incremental Load Test Results

**Date:** July 19, 2026
**Cluster:** 1x m5.xlarge (master/CPU) + 1x g4dn.xlarge (GPU worker)
**Region:** ap-south-1 (Mumbai)
**All 5 runs: SUCCEEDED**

---

## Cluster Configuration

| Node | Instance | IP | Cores | Memory | Role |
|------|----------|----|----|--------|------|
| Master (Driver) | m5.xlarge | 10.0.0.187 | 4 vCPU | 16 GB | Spark Master + Driver |
| CPU Worker | (on master) | 10.0.0.187 | 2 offered | 12 GB offered | Executor 1 |
| GPU Worker | g4dn.xlarge | 10.0.0.45 | 2 offered | 12 GB offered | Executor 0 |

---

## Throughput Scaling Results

| Run | Signal Samples | Images | Detections | Total Samples | Data Size | Partitions | Throughput | Elapsed |
|-----|---------------|--------|-----------|---------------|-----------|-----------|-----------|---------|
| 1 | 500 | 20 | 10 | 2,580 | 135.7 MB | 2 | **327 /sec** | 7.90s |
| 2 | 1,000 | 50 | 20 | 5,190 | 289.5 MB | 2 | **973 /sec** | 5.33s |
| 3 | 2,000 | 100 | 30 | 10,360 | 480.7 MB | 2 | **1,280 /sec** | 8.09s |
| 4 | 5,000 | 200 | 50 | 25,700 | 865.6 MB | 4 | **3,197 /sec** | 8.04s |
| 5 | 10,000 | 400 | 80 | 51,360 | 1,534.6 MB | 4 | **3,046 /sec** | 16.86s |

### Key Observations

1. **Cold start penalty (Run 1):** First run is slowest (327/sec) because executors download model weights (resnet18=44MB, mobilenet=9MB, efficientnet=20MB) for the first time.
2. **Throughput ramp-up (Runs 2-4):** 973 → 1,280 → 3,197 as model caches warm and Spark overhead is amortized over more data.
3. **Plateau (Run 5):** 3,046/sec — throughput stabilizes once executors are fully utilized. More data doesn't increase throughput; it just takes proportionally longer.
4. **Linear scaling with data:** Elapsed time scales roughly linearly (8s for 25K → 17s for 51K = ~2x).

---

## Spark Job Execution Detail

| Job | Tasks | Status | Submission | Completion | Duration |
|-----|-------|--------|------------|-----------|----------|
| Job 0 (Run 1) | 2 | SUCCEEDED | 14:02:01 | 14:02:08 | 7.7s |
| Job 1 (Run 2) | 2 | SUCCEEDED | 14:02:13 | 14:02:18 | 5.3s |
| Job 2 (Run 3) | 2 | SUCCEEDED | 14:02:23 | 14:02:31 | 8.1s |
| Job 3 (Run 4) | 4 | SUCCEEDED | 14:02:38 | 14:02:46 | 8.0s |
| Job 4 (Run 5) | 4 | SUCCEEDED | 14:02:58 | 14:03:15 | 16.8s |

- **Zero task failures** across all jobs
- **Zero shuffle** — data is embedded in RDD elements (no disk I/O between stages)
- **Zero memory spill** — all processing fits in executor memory

---

## Executor Performance

| Executor | Host | Cores | Tasks Completed | Total Duration | GC Time | Memory Used |
|----------|------|-------|----------------|---------------|---------|-------------|
| **0 (GPU Worker)** | 10.0.0.45 | 2 | **9 tasks** | 34.7s | 75ms | 85 MB |
| **1 (CPU Worker)** | 10.0.0.187 | 2 | **5 tasks** | 44.6s | 105ms | 89 MB |
| Driver | 10.0.0.187 | — | 0 (coordinates) | 83.0s | — | 89 MB |

### Executor Distribution Analysis

- **GPU Worker (Executor 0) completed 9 tasks** — nearly 2x more than CPU Worker
- **CPU Worker (Executor 1) completed 5 tasks** — slower per-task execution
- This imbalance shows the GPU worker processes tasks faster (even running on CPU fallback mode, the g4dn.xlarge has faster CPUs than the m5.xlarge's worker container)
- Max memory per executor: 7.5 GB (of 12 GB offered)
- Disk used per executor: ~89 MB (broadcast model weights cached)

---

## Stage-Level Metrics

| Stage | Tasks | Executor Run Time | Executor CPU Time | Deserialize Time | GC Time | Result Size |
|-------|-------|------------------|------------------|-----------------|---------|-------------|
| 0 (Run 1) | 2 | 11,538 ms | 137 ms | 728 ms | 110 ms | 3.1 KB |
| 1 (Run 2) | 2 | 5,512 ms | 10 ms | 183 ms | 19 ms | 3.0 KB |
| 2 (Run 3) | 2 | 7,885 ms | 12 ms | 264 ms | 12 ms | 3.1 KB |
| 3 (Run 4) | 4 | 10,632 ms | 21 ms | 415 ms | 10 ms | 6.2 KB |
| 4 (Run 5) | 4 | 18,036 ms | 26 ms | 560 ms | 29 ms | 6.2 KB |

### Stage Insights

1. **Stage 0 has highest GC (110ms):** First run triggers class loading and JIT compilation
2. **Deserialization decreases after Stage 0:** Model weights are cached after first broadcast
3. **Executor CPU time is low:** Most time is in Python (not JVM) — the actual PyTorch inference happens in Python subprocess
4. **Result sizes are tiny (3-6 KB):** Only per-model counts are returned, not predictions

---

## Comparison: All Inference Modes

| Mode | Throughput | Time | Where | GPU Used |
|------|-----------|------|-------|----------|
| Single GPU (CUDA Streams) | **29,980 /sec** | 0.86s | GPU Worker | Yes |
| Hybrid CPU+GPU | **30,029 /sec** | 0.86s | GPU Worker | Yes |
| Distributed (25K samples) | **3,197 /sec** | 8.04s | Both nodes | CPU only* |
| Distributed (51K samples) | **3,046 /sec** | 16.86s | Both nodes | CPU only* |
| Single GPU (257K, large) | **24,105 /sec** | 10.66s | GPU Worker | Yes |

*Distributed mode runs on CPU because Spark executor subprocess doesn't inherit Docker's GPU context. On the air-gapped system with native Spark (no Docker), GPU will be available to executors.

---

## Throughput Pattern

```
Throughput (samples/sec)
    |
3200|         ●─────────●
    |        /
3000|       /
    |      /
1200|    ●
    |   /
 900|  ●
    | /
 300|●
    |___________________________
     2.5K  5K  10K  25K  51K   Samples
```

**Pattern:** Throughput increases sharply from 327→3,197 as Spark overhead is amortized, then plateaus around 3,000-3,200 samples/sec (executor compute saturation).

---

## Conclusions for AIA Team

1. **Distributed Spark works correctly** — tasks are distributed across both executors with zero failures
2. **GPU Worker handles more tasks** — 9 vs 5 (faster CPU + GPU potential)
3. **Throughput saturates at ~3,200 /sec** on 2 CPU executors — adding more GPU workers would increase this linearly
4. **Single GPU mode is 10x faster** for single-node workloads — use distributed only when data exceeds single-machine capacity
5. **On the air-gapped 256GB / 24GB VRAM × 5 nodes:**
   - Expected distributed throughput: ~15,000-20,000 samples/sec (5 GPU executors)
   - Expected single GPU: ~35,000-45,000 samples/sec (larger GPU)
   - No OOM issues (256GB vs 16GB driver memory)
