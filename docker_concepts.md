# Docker Concepts — Explained with Our EW PyTorch-Spark Project

## Table of Contents

1. [Quick Answer: How to Edit Files Without Rebuilding](#1-quick-answer-volumes-for-live-editing)
2. [Docker Core Concepts](#2-docker-core-concepts)
3. [Dockerfile Explained (Our Project)](#3-dockerfile-explained)
4. [Docker Compose Explained (Our Project)](#4-docker-compose-explained)
5. [Volumes — Deep Dive](#5-volumes--deep-dive)
6. [Networking](#6-networking)
7. [GPU in Docker](#7-gpu-in-docker)
8. [Common Commands Cheat Sheet](#8-common-commands-cheat-sheet)
9. [Development vs Production Patterns](#9-development-vs-production-patterns)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Quick Answer: Volumes for Live Editing

**Problem:** You edit code on your Windows machine but Docker uses the copy baked into the image (`COPY . .` in Dockerfile). You have to rebuild every time.

**Solution:** Mount your source code as a volume. Docker reads files directly from your disk:

### Updated docker-compose.yml (dev mode)

```yaml
services:
  benchmark:
    build: .
    container_name: ew-pytorch-spark-poc
    runtime: nvidia
    volumes:
      # Mount ENTIRE source code — edits are instantly visible in container
      - .:/app
      # Mount results output
      - ./results:/app/results
    environment:
      - PYSPARK_PYTHON=python
      - PYSPARK_DRIVER_PYTHON=python
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    shm_size: '2g'
    command: python benchmark/run_benchmark.py
```

**Key change:** `- .:/app` mounts your project directory into `/app` inside the container. Now:
- Edit `spark_inference.py` in VS Code/Kiro on Windows
- Run `docker compose up` — container sees your latest code immediately
- **No rebuild needed** for code changes

### When do you still need `--build`?

| Change | Rebuild needed? |
|--------|----------------|
| Edit `.py` source files | No (volume mount) |
| Change `requirements.txt` | **Yes** (dependencies installed in image) |
| Change `Dockerfile` | **Yes** |
| Add new Python package | **Yes** |
| Edit `docker-compose.yml` | No (just restart) |

---

## 2. Docker Core Concepts

### Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR WINDOWS MACHINE                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              DOCKER ENGINE (Linux VM in WSL2)         │   │
│  │                                                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ Container 1│  │ Container 2│  │ Container 3│    │   │
│  │  │ (our poc)  │  │ (database) │  │ (redis)    │    │   │
│  │  │            │  │            │  │            │    │   │
│  │  │ Python 3.11│  │ PostgreSQL │  │ Redis      │    │   │
│  │  │ CUDA 12.1  │  │            │  │            │    │   │
│  │  │ Java 17    │  │            │  │            │    │   │
│  │  │ PySpark    │  │            │  │            │    │   │
│  │  │ PyTorch    │  │            │  │            │    │   │
│  │  └─────┬──────┘  └────────────┘  └────────────┘    │   │
│  │        │                                             │   │
│  │        │ Volumes (shared folders)                    │   │
│  └────────┼─────────────────────────────────────────────┘   │
│           │                                                  │
│           ▼                                                  │
│  D:\pyspark_torch_poc_dist\pytorch-spark-ew-poc\            │
│  (your source code on Windows)                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Terms

| Term | What it is | In our project |
|------|-----------|----------------|
| **Image** | A read-only template (like a class) | Built from our `Dockerfile` — contains CUDA, Python, Java, all pip packages |
| **Container** | A running instance of an image (like an object) | `ew-pytorch-spark-poc` — runs our benchmark |
| **Dockerfile** | Recipe to build an image | Installs CUDA, Python 3.11, Java 17, PyTorch, PySpark |
| **docker-compose.yml** | Orchestration config for one or more containers | Defines volumes, GPU access, environment vars |
| **Volume** | Shared folder between host and container | `./results:/app/results` maps Windows folder into container |
| **Layer** | Each Dockerfile instruction creates a cached layer | `RUN pip install ...` is cached — speeds up rebuilds |
| **Registry** | Image storage (like GitHub for images) | Docker Hub, or your airgapped internal registry |
| **Build context** | Files sent to Docker during build | Everything in `pytorch-spark-ew-poc/` (minus `.dockerignore`) |

---

## 3. Dockerfile Explained

Our `Dockerfile` annotated:

```dockerfile
# ─── BASE IMAGE ───────────────────────────────────────────────
# Start from NVIDIA's CUDA runtime image (has GPU drivers pre-installed)
# This is Layer 0 — pulled from Docker Hub (~3GB)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ─── ENVIRONMENT ──────────────────────────────────────────────
# Prevent apt from asking interactive questions (timezone, etc.)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ─── SYSTEM PACKAGES ─────────────────────────────────────────
# Each RUN creates a new layer (cached independently)
# Combines multiple commands with && to keep layers small
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common wget gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        openjdk-17-jre-headless && \
    # Create 'python' symlink pointing to python3.11
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    # Clean up apt cache to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Tell Java where to find itself (PySpark needs this)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# ─── WORKING DIRECTORY ────────────────────────────────────────
# All subsequent commands run from /app
# Created automatically if it doesn't exist
WORKDIR /app

# ─── PYTHON DEPENDENCIES ─────────────────────────────────────
# Copy requirements FIRST (before source code)
# Why? Docker caches layers. If requirements.txt doesn't change,
# this expensive pip install step is cached and skipped on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ─── SOURCE CODE ──────────────────────────────────────────────
# Copy everything from build context into /app
# This layer changes every time you edit code → rebuilds from here
COPY . .

# ─── SETUP ────────────────────────────────────────────────────
RUN mkdir -p results

# ─── DEFAULT COMMAND ──────────────────────────────────────────
# What runs when you do: docker run <image>
# Can be overridden: docker run <image> python inference/spark_inference.py
CMD ["python", "benchmark/run_benchmark.py"]
```

### Layer Caching Concept

```
Layer 1: FROM nvidia/cuda:12.1.0     ← Cached (never changes)
Layer 2: RUN apt-get install ...      ← Cached (rarely changes)
Layer 3: COPY requirements.txt        ← Cached until you change requirements.txt
Layer 4: RUN pip install ...          ← Cached (most expensive layer, ~5min)
Layer 5: COPY . .                     ← INVALIDATED every code edit
Layer 6: RUN mkdir -p results         ← Rebuilds (after invalidation)

This is why we COPY requirements.txt separately before COPY . .
If we did COPY . . first, pip install would re-run on every code change.
```

---

## 4. Docker Compose Explained

```yaml
# docker-compose.yml

services:
  # Service name — you reference this in commands: docker compose run benchmark
  benchmark:
    # Build image from Dockerfile in current directory
    build: .

    # Custom container name (otherwise Docker generates random name)
    container_name: ew-pytorch-spark-poc

    # Use NVIDIA runtime (passes GPU into container)
    runtime: nvidia

    # ─── VOLUMES ────────────────────────────────────────────
    volumes:
      # Syntax: <host_path>:<container_path>
      # Host (Windows): ./results → Container: /app/results
      # Files written to /app/results inside container appear on your disk
      - ./results:/app/results

    # ─── ENVIRONMENT VARIABLES ──────────────────────────────
    environment:
      - PYSPARK_PYTHON=python         # Tell Spark which Python to use
      - PYSPARK_DRIVER_PYTHON=python   # Same for driver
      - NVIDIA_VISIBLE_DEVICES=all     # Expose all GPUs
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

    # Shared memory size (Spark/PyTorch need this for IPC)
    shm_size: '2g'

    # Override CMD from Dockerfile (or same command)
    command: python benchmark/run_benchmark.py
```

---

## 5. Volumes — Deep Dive

### Types of Volumes

```
┌─────────────────────────────────────────────────────────┐
│                    VOLUME TYPES                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. BIND MOUNT (what we use)                            │
│     Host path ←→ Container path                         │
│     - .:/app                                            │
│     - ./results:/app/results                            │
│     - Changes on either side are instantly visible       │
│                                                          │
│  2. NAMED VOLUME (managed by Docker)                    │
│     - my_data:/app/data                                 │
│     - Stored inside Docker's VM                         │
│     - Survives container deletion                        │
│     - Better performance on macOS/Windows                │
│                                                          │
│  3. ANONYMOUS VOLUME (temporary)                        │
│     - /app/temp                                         │
│     - Deleted when container is removed                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Volume Examples for Our Project

```yaml
volumes:
  # ─── Development: mount all source code ───
  - .:/app
  # Now you can edit on Windows, run in container

  # ─── Output: get results on your disk ───
  - ./results:/app/results
  # metrics_report.md appears in your Windows folder

  # ─── Read-only: mount data without allowing writes ───
  - ./data/signals:/app/input_data:ro
  # Container can read but not modify your signal data

  # ─── Named volume: persist model cache between runs ───
  - model_cache:/app/model_cache
  # Survives container restart, managed by Docker
```

### Volume Mount Syntax

```
<host_path>:<container_path>[:options]

Options:
  ro    = read-only (container can't write)
  rw    = read-write (default)

Windows paths in docker-compose.yml:
  - relative: ./results:/app/results     (relative to docker-compose.yml location)
  - absolute: D:/data:/app/data          (use forward slashes)
```

### Performance Note (Windows/WSL2)

| Volume location | Performance |
|----------------|-------------|
| Files in WSL2 filesystem (`/home/...`) | Fast (native Linux) |
| Files on Windows drive (`/mnt/d/...` or `D:\...`) | Slower (cross-filesystem) |
| Named Docker volumes | Fast (stored in WSL2 VM) |

For our PoC, the speed difference is negligible. For production with large datasets, prefer keeping data in WSL2 or named volumes.

---

## 6. Networking

### How Spark Works in Our Container

```
┌────────────────────────── Container ──────────────────────────┐
│                                                                │
│  ┌──────────────┐         ┌─────────────────────────────┐    │
│  │ Python Driver│◄───────►│ JVM (Spark Driver + Workers)│    │
│  │ (our code)   │  Py4J   │ localhost:4040 (Spark UI)   │    │
│  └──────────────┘         └─────────────────────────────┘    │
│         │                           │                         │
│         ▼                           ▼                         │
│  ┌──────────────┐         ┌──────────────┐                   │
│  │ PyTorch/CUDA │         │ Worker threads│                   │
│  │ (GPU)        │         │ (local[*])    │                   │
│  └──────────────┘         └──────────────┘                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         ▲
         │ No external ports needed for local mode
         │ (everything runs inside one container)
```

### Exposing Spark UI (optional)

```yaml
services:
  benchmark:
    build: .
    ports:
      - "4040:4040"   # Spark UI accessible at http://localhost:4040
      - "8080:8080"   # Spark Master UI (cluster mode)
    ...
```

### Multi-Container Spark Cluster (advanced)

```yaml
services:
  spark-master:
    image: ew-pytorch-spark:latest
    command: /opt/spark/sbin/start-master.sh
    ports:
      - "7077:7077"
      - "8080:8080"

  spark-worker-1:
    image: ew-pytorch-spark:latest
    command: /opt/spark/sbin/start-worker.sh spark://spark-master:7077
    runtime: nvidia
    depends_on:
      - spark-master

  spark-worker-2:
    image: ew-pytorch-spark:latest
    command: /opt/spark/sbin/start-worker.sh spark://spark-master:7077
    runtime: nvidia
    depends_on:
      - spark-master
```

---

## 7. GPU in Docker

### How GPU Passthrough Works

```
┌─────────────────────────────────────────────────────┐
│ Windows Host                                         │
│  └─ NVIDIA Driver (610.62)                          │
│      └─ WSL2 Linux Kernel                           │
│          └─ Docker Engine                           │
│              └─ nvidia-container-toolkit             │
│                  └─ Container (CUDA 12.1 runtime)   │
│                      └─ PyTorch cu121               │
│                          └─ model.to("cuda")        │
│                              └─ GTX 1650 (4GB)      │
└─────────────────────────────────────────────────────┘
```

### Three Ways to Enable GPU

```yaml
# Method 1: runtime (Docker Desktop)
services:
  app:
    runtime: nvidia

# Method 2: deploy (Docker Compose v2)
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

# Method 3: docker run flag (command line)
# docker run --gpus all <image>
```

### Verify GPU in Container

```bash
# Check if GPU is visible
docker run --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# Check PyTorch sees GPU
docker compose run benchmark python -c "import torch; print(torch.cuda.is_available())"
```

---

## 8. Common Commands Cheat Sheet

### Building

```cmd
:: Build image from Dockerfile
docker compose build

:: Build with no cache (fresh install)
docker compose build --no-cache

:: Build specific service
docker compose build benchmark
```

### Running

```cmd
:: Run the default command (CMD in Dockerfile)
docker compose up

:: Run in background (detached)
docker compose up -d

:: Build + run in one command
docker compose up --build

:: Run a different command
docker compose run benchmark python inference/spark_inference.py

:: Run interactive shell inside container
docker compose run benchmark bash

:: Run with GPU explicitly
docker run --gpus all -v "%cd%":/app ew-pytorch-spark:latest python benchmark/run_benchmark.py
```

### Stopping

```cmd
:: Stop running containers
docker compose down

:: Stop and remove volumes
docker compose down -v

:: Force stop
docker compose kill
```

### Debugging

```cmd
:: See logs
docker compose logs

:: Follow logs in real-time
docker compose logs -f

:: Shell into running container
docker exec -it ew-pytorch-spark-poc bash

:: Check container status
docker compose ps

:: Check image size
docker images | findstr pytorch-spark
```

### Image Management

```cmd
:: List images
docker images

:: Remove unused images (free disk space)
docker image prune

:: Tag for export
docker tag pytorch-spark-ew-poc-benchmark:latest ew-pytorch-spark:latest

:: Export for airgapped transfer
docker save ew-pytorch-spark:latest -o ew-pytorch-spark.tar

:: Load on another machine
docker load -i ew-pytorch-spark.tar
```

---

## 9. Development vs Production Patterns

### Development Mode (what you want now)

```yaml
# docker-compose.dev.yml
services:
  benchmark:
    build: .
    container_name: ew-pytorch-spark-dev
    runtime: nvidia
    volumes:
      # Mount ALL source code — edit locally, run in container
      - .:/app
      - ./results:/app/results
    environment:
      - PYSPARK_PYTHON=python
      - PYSPARK_DRIVER_PYTHON=python
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    shm_size: '2g'
    command: python benchmark/run_benchmark.py
```

**Usage:**
```cmd
:: First time only (installs dependencies in image)
docker compose -f docker-compose.dev.yml build

:: Edit code in VS Code/Kiro, then run:
docker compose -f docker-compose.dev.yml up

:: Edit more code... run again (no rebuild!)
docker compose -f docker-compose.dev.yml up
```

### Production Mode (airgapped cluster)

```yaml
# docker-compose.yml (production)
services:
  benchmark:
    build: .
    container_name: ew-pytorch-spark-poc
    runtime: nvidia
    volumes:
      # Only mount output directory — code is baked into image
      - ./results:/app/results
    environment:
      - PYSPARK_PYTHON=python
      - PYSPARK_DRIVER_PYTHON=python
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    shm_size: '2g'
    command: python benchmark/run_benchmark.py
```

### Comparison

| Aspect | Dev Mode | Production Mode |
|--------|----------|-----------------|
| Code location | Host filesystem (volume mount) | Baked into image (COPY) |
| Edit workflow | Edit → run (instant) | Edit → rebuild → run |
| Reproducibility | Depends on host files | Self-contained image |
| Image size | Same | Same |
| Performance | Slightly slower (cross-filesystem) | Native speed |
| Airgapped transfer | Can't (needs host files) | Works (image has everything) |

---

## 10. Troubleshooting

### Common Issues with Our Project

| Problem | Cause | Fix |
|---------|-------|-----|
| `no such option: --break-system-packages` | Old pip version | Remove the flag from Dockerfile |
| `openjdk-17 not available` | Debian Trixie only has 21 | Use Adoptium PPA or Java 21 with --add-opens |
| `sun.misc.Unsafe not available` | Java 21 blocks Arrow memory access | Add `--add-opens` JVM flags in Spark config |
| `DLL load failed` (Windows) | Local PyTorch install is broken | Use Docker instead (isolates dependencies) |
| JVM OOM / Py4J disconnect | Spark sending too much data through driver | Use broadcast variables, increase `shm_size` |
| `timezone` prompt during build | Ubuntu interactive package install | Set `DEBIAN_FRONTEND=noninteractive` |
| GPU not detected in container | Missing runtime or toolkit | Add `runtime: nvidia` + check `nvidia-smi` |
| Slow volume performance | Cross-filesystem Windows → WSL2 | Use named volumes for large data |

### Useful Debug Commands

```cmd
:: Check what's inside the container
docker compose run benchmark ls -la /app

:: Verify Python version
docker compose run benchmark python --version

:: Check if CUDA works
docker compose run benchmark python -c "import torch; print(torch.cuda.get_device_name(0))"

:: Check Java version
docker compose run benchmark java -version

:: View container resource usage
docker stats ew-pytorch-spark-poc

:: View build layers (find what's large)
docker history pytorch-spark-ew-poc-benchmark:latest
```

---

## Summary: Docker Lifecycle for Our Project

```
┌──────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT CYCLE                           │
│                                                               │
│  1. Write Dockerfile + docker-compose.yml  (one time)         │
│                    │                                          │
│  2. docker compose build  (installs all dependencies)         │
│                    │                                          │
│  3. Edit Python code on Windows ◄──────────┐                 │
│                    │                        │                 │
│  4. docker compose up  (runs instantly      │                 │
│     with volume mount — no rebuild)         │                 │
│                    │                        │                 │
│  5. View results in ./results/              │                 │
│                    │                        │                 │
│  6. Found a bug? ──────────────────────────►┘                 │
│                                                               │
│                                                               │
│                    DEPLOYMENT CYCLE                            │
│                                                               │
│  7. docker compose build  (final image with code baked in)    │
│                    │                                          │
│  8. docker tag + docker save -o file.tar                      │
│                    │                                          │
│  9. Transfer to airgapped cluster                             │
│                    │                                          │
│  10. docker load -i file.tar                                  │
│                    │                                          │
│  11. docker run --gpus all ew-pytorch-spark:latest            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```
