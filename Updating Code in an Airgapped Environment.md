# Updating Code in an Airgapped Environment

## The Problem

Your airgapped workstation/cluster has:
- No internet access
- Docker installed with GPU support
- The pre-built Docker image (transferred earlier via USB/diode)

You want to:
- Edit Python code on your local (internet-connected) machine
- Transfer **only the code changes** to the airgapped system
- Run the updated code **without rebuilding the full Docker image** (which takes 30+ minutes and needs internet for pip)

---

## Two Approaches

| Approach | Transfer Size | Rebuild Needed? | Best For |
|----------|--------------|-----------------|----------|
| **A: Volume mount (recommended)** | ~500 KB (just source code) | No | Iterative development, frequent changes |
| **B: Rebuild image offline** | ~6-8 GB (full image) | Yes | Final deployment, CI/CD |

---

## Approach A: Volume Mount (No Rebuild)

### How it Works

```
┌─────────────────────────────────────────────────────────────┐
│ AIRGAPPED WORKSTATION                                        │
│                                                              │
│  Docker Image (pre-built, has all dependencies)              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  CUDA 12.1 + Python 3.11 + Java 17              │       │
│  │  PyTorch + PySpark + numpy + pandas              │       │
│  │  (all pip packages pre-installed)                 │       │
│  │                                                   │       │
│  │  /app/ ← OVERRIDDEN by volume mount             │       │
│  └──────────────────────────────────────────────────┘       │
│            ▲                                                 │
│            │ volume mount: -v /path/to/code:/app             │
│            │                                                 │
│  /home/user/ew-poc/  ← YOUR UPDATED SOURCE CODE             │
│  ├── models/                                                 │
│  ├── inference/                                              │
│  ├── benchmark/                                              │
│  ├── data/                                                   │
│  └── results/  (output written here)                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The image has all heavy dependencies (CUDA, PyTorch, PySpark, Java). Your source code is mounted on top — Docker reads your local files instead of the baked-in copy.

### Step-by-Step

#### On Your Local Machine (internet-connected, Windows)

```cmd
:: 1. Make your code changes in Kiro/VS Code
::    Edit spark_inference.py, run_benchmark.py, etc.

:: 2. Package ONLY the source code (tiny file)
cd D:\pyspark_torch_poc_dist\pytorch-spark-ew-poc
tar -czf ew-poc-source.tar.gz ^
    models/ ^
    inference/ ^
    benchmark/ ^
    data/ ^
    requirements.txt ^
    docs/

:: Result: ew-poc-source.tar.gz (~50-500 KB)
```

#### Transfer

Transfer `ew-poc-source.tar.gz` to airgapped system via:
- USB drive
- Cross-domain file transfer
- Data diode
- Approved secure transfer mechanism

#### On the Airgapped Workstation (Linux)

```bash
# 3. Extract source code
mkdir -p ~/ew-poc
cd ~/ew-poc
tar -xzf /path/to/ew-poc-source.tar.gz

# 4. Run with volume mount (uses pre-loaded Docker image)
docker run --gpus all \
    -v $(pwd):/app \
    -v $(pwd)/results:/app/results \
    -e PYSPARK_PYTHON=python \
    -e PYSPARK_DRIVER_PYTHON=python \
    --shm-size=2g \
    ew-pytorch-spark:latest \
    python benchmark/run_benchmark.py

# Results appear in ~/ew-poc/results/
```

#### On the Airgapped Workstation (Windows)

```cmd
:: 3. Extract source code
mkdir C:\ew-poc
cd C:\ew-poc
tar -xzf D:\transfer\ew-poc-source.tar.gz

:: 4. Run with volume mount
docker run --gpus all ^
    -v "%cd%":/app ^
    -v "%cd%\results":/app/results ^
    -e PYSPARK_PYTHON=python ^
    -e PYSPARK_DRIVER_PYTHON=python ^
    --shm-size=2g ^
    ew-pytorch-spark:latest ^
    python benchmark/run_benchmark.py
```

### With docker-compose (recommended)

Create a `docker-compose.yml` on the airgapped machine:

```yaml
# ~/ew-poc/docker-compose.yml
services:
  benchmark:
    image: ew-pytorch-spark:latest   # Uses pre-loaded image (no build)
    container_name: ew-pytorch-spark-poc
    runtime: nvidia
    volumes:
      - .:/app                        # Mount local code over /app
      - ./results:/app/results
    environment:
      - PYSPARK_PYTHON=python
      - PYSPARK_DRIVER_PYTHON=python
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    shm_size: '2g'
    command: python benchmark/run_benchmark.py
```

Then:
```bash
cd ~/ew-poc
docker compose up
```

### Update Cycle

```
┌─ LOCAL MACHINE ─────────────────────────────────────────┐
│                                                          │
│  Edit code → tar -czf ew-poc-source.tar.gz ...          │
│                         │                                │
└─────────────────────────┼────────────────────────────────┘
                          │ USB / secure transfer
                          ▼
┌─ AIRGAPPED WORKSTATION ─────────────────────────────────┐
│                                                          │
│  tar -xzf ew-poc-source.tar.gz                          │
│  docker compose up                                       │
│                                                          │
│  (runs in seconds — no rebuild, no download)             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### What If You Need a New pip Package?

If your code change requires a new Python package not in the original image:

```bash
# Option 1: pip install at runtime (temporary, lost when container stops)
docker run --gpus all -v $(pwd):/app ew-pytorch-spark:latest \
    bash -c "pip install new_package && python benchmark/run_benchmark.py"

# Option 2: Transfer the wheel and install from file
#   On local: pip download new_package -d ./wheels/
#   Transfer wheels/ folder
#   On airgapped:
docker run --gpus all -v $(pwd):/app ew-pytorch-spark:latest \
    bash -c "pip install --no-index --find-links=/app/wheels new_package && python benchmark/run_benchmark.py"

# Option 3: Rebuild image (see Approach B below)
```

---

## Approach B: Rebuild Image Offline

Use this when you need to:
- Change `requirements.txt` (add/update dependencies)
- Change `Dockerfile` (different base image, system packages)
- Create a clean production image with code baked in

### Strategy: Two-Layer Image Build

```
┌──────────────────────────────────────────────────┐
│ BASE IMAGE (rarely changes, ~6-8 GB)             │
│ ew-pytorch-spark-base:latest                     │
│                                                   │
│  - CUDA 12.1 runtime                             │
│  - Python 3.11                                    │
│  - Java 17                                        │
│  - PyTorch, PySpark, numpy, pandas                │
│  - All pip dependencies                           │
└──────────────────────┬───────────────────────────┘
                       │ FROM ew-pytorch-spark-base
                       ▼
┌──────────────────────────────────────────────────┐
│ APP IMAGE (changes with code, ~50 KB layer)      │
│ ew-pytorch-spark:latest                          │
│                                                   │
│  - Your Python source code                        │
│  - Config files                                   │
└──────────────────────────────────────────────────┘
```

### One-Time Setup (on internet-connected machine)

#### 1. Create base image Dockerfile

```dockerfile
# Dockerfile.base — heavy dependencies, rarely changes
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common wget gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        openjdk-17-jre-headless && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt
```

#### 2. Create app image Dockerfile

```dockerfile
# Dockerfile.app — just source code, tiny
FROM ew-pytorch-spark-base:latest

WORKDIR /app
COPY . .
RUN mkdir -p results

CMD ["python", "benchmark/run_benchmark.py"]
```

#### 3. Build and export base image (one time)

```cmd
:: Build base (takes 30+ minutes, needs internet)
docker build -f Dockerfile.base -t ew-pytorch-spark-base:latest .

:: Export base image
docker save ew-pytorch-spark-base:latest -o ew-pytorch-spark-base.tar
```

#### 4. Transfer base image to airgapped system (one time)

```cmd
:: ~6-8 GB, transfer via USB
:: On airgapped machine:
docker load -i ew-pytorch-spark-base.tar
```

### Ongoing Updates (on airgapped system)

```bash
# 1. Transfer updated source code (tar.gz or git bundle)
tar -xzf ew-poc-source.tar.gz -C ~/ew-poc/

# 2. Rebuild app layer LOCALLY (no internet needed, takes ~5 seconds)
cd ~/ew-poc
docker build -f Dockerfile.app -t ew-pytorch-spark:latest .

# 3. Run
docker run --gpus all -v $(pwd)/results:/app/results ew-pytorch-spark:latest
```

This rebuild takes **seconds** because:
- Base image is cached locally (already loaded)
- Only the `COPY . .` layer changes
- No pip install, no apt-get, no downloads

---

## Approach C: Git Bundle (for version control)

If you use git on both machines:

### On Local Machine

```cmd
:: Create a git bundle containing all commits since last transfer
cd D:\pyspark_torch_poc_dist\pytorch-spark-ew-poc
git bundle create ew-poc-update.bundle --all
:: Or just recent changes:
git bundle create ew-poc-update.bundle main~5..main
```

### On Airgapped Machine

```bash
cd ~/ew-poc
git pull /path/to/ew-poc-update.bundle main
docker compose up
```

---

## Decision Matrix

| Scenario | Approach | Transfer Size | Downtime |
|----------|----------|---------------|----------|
| Changed 1 Python file | A (volume mount) | ~500 KB tar | 0 (instant run) |
| Changed multiple files, testing | A (volume mount) | ~500 KB tar | 0 |
| Added new pip dependency | B (rebuild) or A+wheel | ~50 KB wheel or 6GB image | 5 sec or 30 min |
| New CUDA version needed | B (rebuild base) | ~8 GB image | 30 min |
| First deployment | B (full image) | ~8 GB image | Transfer time |
| Iterative development on airgapped | A (volume mount) | ~500 KB per iteration | 0 |

---

## Complete File List for Transfer

### Minimal (code only — for Approach A)

```
Transfer: ew-poc-source.tar.gz (~500 KB)
Contains:
  models/ew_signal_model.py
  inference/spark_inference.py
  inference/baseline_inference.py
  benchmark/run_benchmark.py
  data/generate_signals.py
  requirements.txt
```

### First-Time Setup (full image — for Approach B)

```
Transfer: ew-pytorch-spark-base.tar (~6-8 GB, one time)
Transfer: ew-poc-source.tar.gz (~500 KB, every update)
Transfer: Dockerfile.app (~200 bytes, one time)
Transfer: docker-compose.yml (~300 bytes, one time)
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              AIRGAPPED UPDATE CHEAT SHEET                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  LOCAL (with internet):                                      │
│    1. Edit code                                              │
│    2. tar -czf ew-poc-source.tar.gz models/ inference/      │
│       benchmark/ data/ requirements.txt                      │
│    3. Transfer tar.gz to airgapped system                    │
│                                                              │
│  AIRGAPPED:                                                  │
│    4. tar -xzf ew-poc-source.tar.gz -C ~/ew-poc/           │
│    5. cd ~/ew-poc && docker compose up                       │
│                                                              │
│  That's it. No rebuild. No internet. No waiting.             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
