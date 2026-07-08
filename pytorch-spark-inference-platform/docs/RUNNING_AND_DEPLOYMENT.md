# Multi-Model Inference Platform — Running & Deployment Guide

Companion to `docs/TECHNICAL_ARCHITECTURE.md` (architecture/code flow). This doc
covers **how to actually run and deploy** the platform: local native, local
Docker, multi-node cluster, AWS test, and DRDO airgapped.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Running Natively (No Docker)](#2-running-natively-no-docker)
3. [Running with Docker (Local Dev)](#3-running-with-docker-local-dev)
4. [Deployment Config Reference](#4-deployment-config-reference)
5. [Running on a Cluster (Multi-Node)](#5-running-on-a-cluster-multi-node)
6. [AWS Test Deployment](#6-aws-test-deployment)
7. [Airgapped / DRDO Deployment](#7-airgapped--drdo-deployment)
8. [Environment Variables Reference](#8-environment-variables-reference)
9. [Verifying a Run / Reading Results](#9-verifying-a-run--reading-results)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Local Native | Local Docker | Cluster | Airgapped |
|-------------|:---:|:---:|:---:|:---:|
| Python 3.11 | Yes | (in image) | (in image) | (in image) |
| Java 17 (for PySpark) | Yes | (in image) | (in image) | (in image) |
| Docker + NVIDIA Container Toolkit | No | Yes | Yes | Yes |
| NVIDIA GPU + driver | Optional | Optional | Recommended | Recommended |
| Internet access | Yes (pip/torch/yolo weights) | Yes (build time only) | Yes (build time only) | **No** — pre-built image required |

**Note on YOLO/torchvision weights:** `models/yolo_model.py` and
`models/image_models.py` try to auto-download pretrained weights on first use
(`ultralytics` fetches `yolov8n.pt`/`yolov8s.pt`; `torchvision` fetches ImageNet
weights). This requires internet **once**. For airgapped deployment, download
these ahead of time and bake them into the Docker image — see §7.

---

## 2. Running Natively (No Docker)

Useful for quick iteration on a machine that already has Python/Java/CUDA set up.

```bash
cd pytorch-spark-inference-platform

# 1. Install PyTorch matching your CUDA version (or CPU-only)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
# CPU-only alternative:
# pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest
pip install -r requirements.txt

# 3. Run the full benchmark (all 3 modes)
python benchmark/run_benchmark.py

# 4. Or run a single mode
python benchmark/run_benchmark.py --mode single_gpu
python benchmark/run_benchmark.py --mode hybrid
python benchmark/run_benchmark.py --mode distributed
```

Requires `JAVA_HOME` pointing at a Java 17 install for the `distributed` mode
(PySpark needs it even in local mode). All other modes (`single_gpu`, `hybrid`)
do not touch Spark at all and will run without Java installed.

### CLI Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `all` | `all` \| `single_gpu` \| `hybrid` \| `distributed` |
| `--signal-samples` | 10000 | Sample count for the 5 signal models (128-dim input) |
| `--image-samples` | 500 | Sample count for the 3 image classifiers (224×224) |
| `--detection-samples` | 100 | Sample count for the 2 YOLO models (640×640) |
| `--batch-size` | 256 | Per-model inference batch size, all modes |
| `--partitions` | 4 | Spark partition count, `distributed` mode only |

Example — larger signal-heavy benchmark:
```bash
python benchmark/run_benchmark.py --mode all --signal-samples 100000 --image-samples 2000 --detection-samples 500 --batch-size 512
```

---

## 3. Running with Docker (Local Dev)

This is the recommended path on Windows (avoids native PyTorch/CUDA DLL issues)
and matches what will eventually run on the cluster.

```bash
cd pytorch-spark-inference-platform

# Build + run (all 3 modes by default, per docker-compose.yml command)
docker compose -f deploy/docker-compose.yml up --build
```

What happens:
- `deploy/Dockerfile` builds a CUDA 12.1 + Python 3.11 + Java 17 image (~6-8 GB)
- `runtime: nvidia` passes your local GPU into the container
- `volumes: - ..:/app` mounts the **entire project source** live — edit any
  `.py` file on your host, re-run `docker compose up` (no `--build` needed)
- Results land in `./results/metrics_report.md` and `./results/raw_results.json`
  on your host machine (volume-mounted)

### Running a single mode / custom args in Docker

Override the default command:
```bash
docker compose -f deploy/docker-compose.yml run inference \
  python benchmark/run_benchmark.py --mode single_gpu --batch-size 512
```

### Interactive shell inside the container

```bash
docker compose -f deploy/docker-compose.yml run inference bash
# then inside:
python benchmark/run_benchmark.py --mode hybrid
```

### Viewing Spark UI (distributed mode only)

Ports `4040` (application UI) and `8080` (master UI, cluster mode only) are
already exposed in `docker-compose.yml`. While a `distributed` run is active,
open `http://localhost:4040` in your browser. If the run finishes and stops the
SparkSession before you can look, add a short pause after `spark.stop()` is
skipped (or interrupt with Ctrl+C to inspect while running) — see the EW PoC's
`SPARK_UI_AND_DOCKER_CLUSTER.md` for the same pattern in more detail.

### Rebuild vs no-rebuild — quick rule

| Change | Rebuild needed? |
|--------|-----------------|
| Edit any `.py` file | No (volume-mounted) |
| Edit `requirements.txt` | Yes |
| Edit `deploy/Dockerfile` | Yes |
| Edit `deploy/docker-compose*.yml` | No, just re-run `up` |

---

## 4. Deployment Config Reference

### `deploy/Dockerfile` — what it builds

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04    # CUDA runtime base
# ... installs Python 3.11, Java 17 (openjdk-17-jre-headless), procps
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url .../cu121
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "benchmark/run_benchmark.py"]
```

Key points:
- Base image already includes NVIDIA CUDA 12.1 runtime libraries
- Java 17 (not 21) is used deliberately — Java 21 blocks `sun.misc.Unsafe`
  access that Arrow/PySpark relies on unless extra `--add-opens` JVM flags are
  set (see EW PoC troubleshooting docs for the full story)
- `procps` is installed so PySpark's `load-spark-env.sh` (`ps` command) doesn't
  warn/fail inside the container
- PyTorch is installed **before** `requirements.txt` with an explicit CUDA
  wheel index — installing it via `requirements.txt` alone would pull a
  CPU-only build by default

### `requirements.txt`

```
pyspark==3.5.1
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
torchvision==0.17.0
ultralytics==8.2.0
matplotlib==3.9.0
tabulate==0.9.0
```
`torch` itself is intentionally **not** listed here — it's installed separately
in the Dockerfile with the correct CUDA index URL to avoid pip resolving a
mismatched/CPU build.

### `deploy/docker-compose.yml` (local dev, single node/GPU)

```yaml
services:
  inference:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
    runtime: nvidia
    volumes:
      - ..:/app                    # live source mount
      - ../results:/app/results    # results land on host
    ports:
      - "4040:4040"   # Spark application UI
      - "8080:8080"   # Spark master UI (only relevant in cluster compose)
    environment:
      - PYSPARK_PYTHON=python
      - PYSPARK_DRIVER_PYTHON=python
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    shm_size: '4g'
    command: python benchmark/run_benchmark.py --mode all
```

### `deploy/docker-compose.cluster.yml` (multi-node)

```yaml
services:
  spark-master:
    image: multi-model-inference:latest   # pre-built, not built here
    network_mode: host
    environment: [SPARK_MODE=master]
    ports: ["7077:7077", "8080:8080", "4040:4040"]
    command: start-master.sh + tail logs

  spark-worker:
    image: multi-model-inference:latest
    runtime: nvidia
    network_mode: host
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER=spark://localhost:7077
      - SPARK_EXECUTOR_GPU=1        # ← tells distributed_gpu.py to use CUDA
    shm_size: '4g'
    deploy:
      replicas: 2                    # 2 GPU worker containers on this host
    command: start-worker.sh spark://localhost:7077 -c 4 -m 8g + tail logs
```

Important: this compose file uses `image:`, not `build:` — you must build and
tag the image first (`docker build -f deploy/Dockerfile -t multi-model-inference:latest ..`)
before `docker compose -f deploy/docker-compose.cluster.yml up` will find it.

The `SPARK_EXECUTOR_GPU=1` environment variable is read inside
`inference/distributed_gpu.py`'s worker function:
```python
if torch.cuda.is_available() and os.environ.get("SPARK_EXECUTOR_GPU", "0") == "1":
    device = "cuda"
else:
    device = "cpu"
```
This is the single switch between "local mode plays it safe on CPU" and
"cluster mode uses the executor's GPU."

---

## 5. Running on a Cluster (Multi-Node)

### Step 1 — Build and tag the image once (on a machine with internet)

```bash
cd pytorch-spark-inference-platform
docker build -f deploy/Dockerfile -t multi-model-inference:latest .
```

### Step 2 — Distribute the image to all nodes

```bash
docker save multi-model-inference:latest -o multi-model-inference.tar
# copy multi-model-inference.tar to every node (scp / USB / internal registry)
docker load -i multi-model-inference.tar     # on each node
```

### Step 3 — Start the master (on the designated master node)

```bash
docker compose -f deploy/docker-compose.cluster.yml up spark-master -d
```

### Step 4 — Start workers (on each GPU worker node)

Edit `SPARK_MASTER=spark://localhost:7077` in `docker-compose.cluster.yml` to
point at the **master node's real hostname/IP** if workers are on separate
machines (the file as shipped assumes single-host testing with `network_mode: host`
and multiple worker replicas on the same box). For true multi-machine:

```bash
# On each worker machine:
docker run -d --name spark-worker --network host --gpus all --shm-size=4g \
  -e SPARK_MODE=worker \
  -e SPARK_MASTER=spark://<MASTER_IP>:7077 \
  -e SPARK_EXECUTOR_GPU=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  multi-model-inference:latest \
  bash -c "$SPARK_HOME/sbin/start-worker.sh spark://<MASTER_IP>:7077 -c 4 -m 8g && tail -f $SPARK_HOME/logs/*worker*"
```

### Step 5 — Submit the benchmark job

```bash
docker exec spark-master spark-submit \
  --master spark://<MASTER_IP>:7077 \
  --deploy-mode client \
  --num-executors 2 \
  --executor-cores 4 \
  --executor-memory 8g \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=0.1 \
  /app/benchmark/run_benchmark.py --mode distributed --partitions 8
```

### Step 6 — Monitor

- Master UI: `http://<MASTER_IP>:8080` — shows connected workers, running app
- Application UI: `http://<MASTER_IP>:4040` — jobs, stages, tasks, executors

Full conceptual background (how Spark decides master/worker, partitioning,
fault tolerance, MPS for multi-process GPU sharing) is in the companion EW PoC
docs: `pytorch-spark-ew-poc/docs/SPARK_CLUSTER_CONCEPTS.md` and
`GPU_SHARING_AND_MPP_ARCHITECTURE.md` — same concepts apply unchanged here.

---

## 6. AWS Test Deployment

Cheap way to validate real multi-node/multi-GPU behavior before touching DRDO
hardware. Full walkthrough is in
`pytorch-spark-ew-poc/docs/EXPANDED_PROJECT_PLAN.md` (Phase 2) — summary here:

| Component | Instance | Spot $/hr | Role |
|-----------|----------|-----------|------|
| Master | `t3.medium` | ~$0.04 | Driver, no GPU |
| Worker ×2 | `g4dn.xlarge` (T4 16GB) | ~$0.16 each | GPU executors |

```bash
# Launch (repeat for each worker, adjust tags)
aws ec2 run-instances \
  --image-id <deep-learning-ami-id> \
  --instance-type g4dn.xlarge \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.20"}}' \
  --key-name <your-key> --security-group-ids <sg-id> --count 2

# On each instance: docker load the pre-built image, then follow §5 steps 3-6
```

Budget estimate for a 2-3 hour test session: **under $2**. Always terminate
instances immediately after collecting `results/metrics_report.md`.

---

## 7. Airgapped / DRDO Deployment

Full generic workflow (image export/import, git bundles, offline pip wheels)
is documented in `pytorch-spark-ew-poc/AIRGAPPED_DEPLOYMENT.md` and
`pytorch-spark-ew-poc/docs/AIRGAPPED_CODE_UPDATE_WORKFLOW.md`. The one thing
**specific to this platform** that needs extra attention: pretrained weights.

### Pre-baking pretrained weights (do this BEFORE going airgapped)

`models/image_models.py` and `models/yolo_model.py` fall back to downloading
weights on first use. On an airgapped target this fails silently into the
"fallback CNN" path in `yolo_model.py` (still runs, but isn't real YOLO). Fix:
download once on an internet-connected machine and save under `models/weights/`:

```python
# run this once, with internet, before building the final image
import torch, os
os.makedirs("models/weights", exist_ok=True)

from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, \
    MobileNet_V3_Small_Weights, efficientnet_b0, EfficientNet_B0_Weights

torch.save(resnet18(weights=ResNet18_Weights.DEFAULT).state_dict(),
           "models/weights/resnet18_imagenet.pth")
torch.save(mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).state_dict(),
           "models/weights/mobilenetv3_small.pth")
torch.save(efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).state_dict(),
           "models/weights/efficientnet_b0.pth")

from ultralytics import YOLO
YOLO("yolov8n.pt")   # downloads to CWD
YOLO("yolov8s.pt")   # downloads to CWD
# then move yolov8n.pt / yolov8s.pt into models/weights/
```

Then pass `pretrained_path=` when constructing the image models (currently the
constructors default to `pretrained_path=None` and attempt auto-download —
update `models/__init__.py`'s `get_default_registry()` to pass the local paths,
or set them after `registry.load_all()` before deploying airgapped).

### Build → export → transfer → load (same pattern as EW PoC)

```bash
# Internet-connected build machine
docker build -f deploy/Dockerfile -t multi-model-inference:latest .
docker save multi-model-inference:latest -o multi-model-inference.tar
# Windows: docker save multi-model-inference:latest -o multi-model-inference.tar (same, no gzip pipe)

# Transfer multi-model-inference.tar via USB / approved secure transfer

# On DRDO airgapped node(s)
docker load -i multi-model-inference.tar
docker compose -f deploy/docker-compose.cluster.yml up -d
```

### Airgapped code updates (no rebuild)

For iterative changes to `.py` files only (no new pip packages), use the
volume-mount pattern instead of rebuilding — identical procedure to the EW PoC:
tar up just the source folders, transfer (~KB, not GB), extract on the airgapped
box, and re-run with a bind mount over `/app`. See
`pytorch-spark-ew-poc/docs/AIRGAPPED_CODE_UPDATE_WORKFLOW.md` for the exact
commands (Approach A) — apply verbatim to this project's folder layout.

### MPS for multi-model GPU sharing on DRDO nodes

```bash
# On each GPU worker node, before starting Spark workers:
nvidia-cuda-mps-control -d
```
Enables true parallel execution of the 10 models across Spark executor
processes on the same physical GPU. See
`pytorch-spark-ew-poc/docs/GPU_SHARING_AND_MPP_ARCHITECTURE.md` §"NVIDIA MPS"
for the systemd service definition to make this persistent across reboots.

---

## 8. Environment Variables Reference

| Variable | Used By | Values | Effect |
|----------|---------|--------|--------|
| `SPARK_EXECUTOR_GPU` | `inference/distributed_gpu.py` | `"1"` / unset | `"1"` → executors use CUDA (cluster); unset → CPU (safe for local threads) |
| `PYSPARK_PYTHON` | PySpark | `python` | Which interpreter Spark workers invoke |
| `PYSPARK_DRIVER_PYTHON` | PySpark | `python` | Which interpreter the driver uses |
| `NVIDIA_VISIBLE_DEVICES` | NVIDIA Container Toolkit | `all` or GPU indices | Which GPUs are passed into the container |
| `NVIDIA_DRIVER_CAPABILITIES` | NVIDIA Container Toolkit | `compute,utility` | Required capabilities for CUDA + `nvidia-smi` inside container |
| `JAVA_HOME` | PySpark | path | Must point to Java 17 install |
| `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY` | MPS + Spark executors | path | Only needed when running MPS on cluster nodes |

---

## 9. Verifying a Run / Reading Results

After any run (native, Docker, or cluster), two files appear under `results/`:

```
results/
├── metrics_report.md    # human-readable — open directly
└── raw_results.json     # machine-readable — for further analysis/CI
```

Quick sanity check that a run succeeded:
```bash
python -c "import json; d=json.load(open('results/raw_results.json')); print(list(d.keys()))"
# Expect: ['system_info', 'config', 'models', 'single_gpu', 'hybrid_cpu_gpu', 'distributed_gpu', 'total_benchmark_time']
```

If a mode key contains `{"error": "..."}` instead of throughput numbers (only
possible for `distributed_gpu`, which is wrapped in try/except in
`run_mode_distributed`), that mode failed — check the printed traceback in the
console output for the underlying Spark/Java error.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `DLL load failed` importing torch (Windows native) | Broken local CUDA/torch install | Use Docker instead (§3) |
| `sun.misc.Unsafe ... not available` (distributed mode) | Java 21 blocks Arrow memory access | Dockerfile already pins Java 17; if running natively, install Java 17 not 21 |
| Spark job stuck at `[Stage 0: (0+N)/N]` forever | Too many partitions all hitting one GPU simultaneously in `local[*]` mode | Reduce `--partitions`, or leave `SPARK_EXECUTOR_GPU` unset so partitions use CPU locally |
| `CUDA out of memory` in `single_gpu` mode | All 10 models + batch tensors exceed VRAM | Switch to `--mode hybrid` (offloads small models to CPU) or reduce `--batch-size` |
| YOLO models silently using fallback CNN, not real YOLOv8 | `ultralytics` couldn't download weights (no internet / airgapped) | Pre-bake weights per §7 before going airgapped |
| `docker compose -f deploy/docker-compose.cluster.yml up` fails: image not found | That compose file uses `image:` not `build:` | Run `docker build -f deploy/Dockerfile -t multi-model-inference:latest .` first |
| Port 4040 already bound / falls back to 4041 | A previous SparkSession didn't shut down cleanly | `pkill -f java` inside the container, or just use whichever port it reports |
| `no such option: --break-system-packages` during pip install | Older pip in the base image doesn't support that flag | Already removed from `deploy/Dockerfile`; don't add it back |

For deeper Spark-specific and Docker-specific troubleshooting (fault tolerance,
speculative execution, volume performance, GPU passthrough details), see the
companion docs in `pytorch-spark-ew-poc/docs/`:
- `DOCKER_CONCEPTS.md`
- `SPARK_CLUSTER_CONCEPTS.md`
- `SPARK_UI_AND_DOCKER_CLUSTER.md`
- `GPU_SHARING_AND_MPP_ARCHITECTURE.md`
