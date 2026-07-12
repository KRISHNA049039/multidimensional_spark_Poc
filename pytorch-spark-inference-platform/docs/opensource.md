# Code-to-Open-Source Mapping

Mapping every code file in this platform to the open-source libraries, repos, and techniques it uses. Intended for PoC review/presentation.

---

## Summary

This platform uses **7 open-source frameworks** to build a multi-model GPU inference system. No proprietary libraries or licensed code. Everything is Apache 2.0 / MIT / BSD licensed.

| Layer | Open-Source Dependency | License |
|-------|----------------------|---------|
| Neural Networks | PyTorch 2.2.0 | BSD-3 |
| Pretrained Vision Models | TorchVision 0.17.0 | BSD-3 |
| Object Detection | Ultralytics YOLOv8 8.2.0 | AGPL-3.0 |
| Distributed Compute | Apache Spark (PySpark) 3.5.1 | Apache-2.0 |
| GPU Runtime | NVIDIA CUDA 12.1 | Proprietary (free) |
| Containers | Docker + NVIDIA Container Toolkit | Apache-2.0 |
| Numerical Compute | NumPy 1.26.4 | BSD-3 |

---

## Data Layer (`data/`)

### `data/signal_generator.py` — Synthetic IQ Signal Generator

| Code Pattern | What It Does | Open-Source Reference |
|---|---|---|
| `np.cos(2π * freq * t)` / `np.sin(...)` | Generates I/Q channels for CW radar | Standard DSP — NumPy |
| Pulsed envelope `(t % PRI) < PW` | Simulates pulsed radar | Skolnik, *Intro to Radar Systems* (textbook technique) |
| `np.cumsum(freq_sweep)` → chirp phase | FMCW radar waveform | Standard chirp signal generation |
| Binary phase code `[-1, 1]` × carrier | Phase-coded radar | Barker code modulation |
| Gaussian `np.random.normal(0, 1)` | Noise jammer | White noise generation (NumPy) |
| QAM constellation `[-3,-1,1,3]` | Communications signal | Standard 16-QAM modulation |
| L2 normalization `vec / np.linalg.norm(vec)` | Input normalization | NumPy |

**Key point for manager:** No real classified EW data is used. All signals are synthetically generated using textbook radar/comms formulas. The 128-dim IQ vector format matches what a real SDR sensor would produce.

**Open datasets with similar format (for reference):**
- DeepSig RadioML 2016/2018: https://www.deepsig.ai/datasets
- RFDataFactory: https://github.com/brysef/rfml

---

### `data/image_generator.py` — Synthetic Image Generator

| Code Pattern | What It Does | Open-Source Reference |
|---|---|---|
| `np.random.rand(N, 3, 224, 224)` | Random RGB images for benchmarking | NumPy |
| `np.random.rand(N, 3, 640, 640)` | Random 640×640 for YOLO | NumPy |

**Key point:** Images are random noise for throughput benchmarking. In production, replace with real camera/satellite feeds. The tensor format (NCHW, float32, [0,1]) matches TorchVision conventions.

---

## Model Layer (`models/`)

### `models/model_registry.py` — Central Model Registry

| Code Pattern | Open-Source API Used | Repo |
|---|---|---|
| `torch.save(model.state_dict(), buf)` | PyTorch serialization | https://github.com/pytorch/pytorch |
| `torch.load(buf, map_location="cpu")` | PyTorch deserialization | https://github.com/pytorch/pytorch |
| `io.BytesIO()` for serialization | Python stdlib | N/A |
| `model.to(device).eval()` | PyTorch device placement | https://github.com/pytorch/pytorch |

**Pattern:** Registry pattern (GoF) — not from any specific repo, standard software engineering.

---

### `models/ew_signal_model.py` — EW Signal Classifier (Custom)

| Code Pattern | Open-Source API Used | Reference |
|---|---|---|
| `nn.Linear(128, 256)` | PyTorch fully-connected layer | https://pytorch.org/docs/stable/generated/torch.nn.Linear.html |
| `nn.BatchNorm1d(256)` | Batch normalization | Ioffe & Szegedy 2015, https://arxiv.org/abs/1502.03167 |
| `nn.ReLU()` | Activation function | Standard, in PyTorch |
| `nn.Dropout(0.3)` | Regularization | Srivastava et al. 2014 |
| `nn.CrossEntropyLoss()` | Training loss | PyTorch |
| `torch.optim.Adam(lr=0.001)` | Optimizer | Kingma & Ba 2014, https://arxiv.org/abs/1412.6980 |

**Architecture inspiration:** O'Shea & Corgan 2016, "Convolutional Radio Modulation Recognition Networks" — https://arxiv.org/abs/1602.04105

---

### `models/signal_models.py` — 4 Signal Processing Models

#### SignalDenoiser (Autoencoder)

| Code Pattern | Open-Source API | Academic Reference |
|---|---|---|
| Encoder: `Linear→ReLU` × 3 (128→256→128→32) | PyTorch `nn.Sequential` | Vincent et al. 2010, "Stacked Denoising Autoencoders" |
| Decoder: `Linear→ReLU` × 2 + `Tanh` (32→128→256→128) | PyTorch | Same paper |
| Symmetric encoder-decoder | Standard autoencoder pattern | https://www.jmlr.org/papers/v11/vincent10a.html |

#### ThreatPrioritizer (Attention)

| Code Pattern | Open-Source API | Academic Reference |
|---|---|---|
| `nn.MultiheadAttention(512, num_heads=4)` | PyTorch MHA | Vaswani et al. 2017, https://arxiv.org/abs/1706.03762 |
| `nn.Sigmoid()` output → [0,1] score | PyTorch | Standard |
| Input projection `Linear(128, 512)` | PyTorch | Standard embedding projection |

#### RFFingerprinter (1D-CNN)

| Code Pattern | Open-Source API | Academic Reference |
|---|---|---|
| `nn.Conv1d(1, 32, kernel_size=7)` | PyTorch 1D convolution | Riyaz et al. 2018, https://ieeexplore.ieee.org/document/8454327 |
| `nn.BatchNorm1d(32)` | PyTorch | Ioffe & Szegedy 2015 |
| `nn.AdaptiveAvgPool1d(8)` | PyTorch pooling | Standard CNN pattern |
| `F.normalize(embedding, p=2, dim=1)` | L2 normalization for metric learning | FaceNet (Schroff 2015) style embedding |

#### AnomalyDetector (VAE)

| Code Pattern | Open-Source API | Academic Reference |
|---|---|---|
| `fc_mu` / `fc_logvar` (dual-head encoder) | PyTorch | Kingma & Welling 2013, https://arxiv.org/abs/1312.6114 |
| Reparameterization trick `mu + eps*std` | PyTorch | Same paper (§2.4) |
| `F.mse_loss(recon, x)` as anomaly score | PyTorch | Standard VAE reconstruction loss |

---

### `models/image_models.py` — 3 Pretrained Image Classifiers

| Model | Open-Source Origin | Weights Source | Paper |
|---|---|---|---|
| ResNet-18 | `torchvision.models.resnet18` | `ResNet18_Weights.DEFAULT` (ImageNet-1K) | He et al. 2016, https://arxiv.org/abs/1512.03385 |
| MobileNetV3-Small | `torchvision.models.mobilenet_v3_small` | `MobileNet_V3_Small_Weights.DEFAULT` | Howard et al. 2019, https://arxiv.org/abs/1905.02244 |
| EfficientNet-B0 | `torchvision.models.efficientnet_b0` | `EfficientNet_B0_Weights.DEFAULT` | Tan & Le 2019, https://arxiv.org/abs/1905.11946 |

**Repo:** https://github.com/pytorch/vision (all three)

**Code pattern:** Wrap torchvision model in `nn.Module`, support local weight loading for airgapped, fall back to download.

---

### `models/yolo_model.py` — 2 Object Detection Models

| Model | Open-Source Origin | Weights | Paper/Repo |
|---|---|---|---|
| YOLOv8-Nano | `ultralytics.YOLO("yolov8n.pt")` | https://github.com/ultralytics/assets | Jocher et al. 2023 |
| YOLOv8-Small | `ultralytics.YOLO("yolov8s.pt")` | https://github.com/ultralytics/assets | Same |

**Repo:** https://github.com/ultralytics/ultralytics

**Code pattern:** Wrapper that tries Ultralytics first, falls back to a lightweight CNN with same I/O shape for benchmarking without weights.

---

## Inference Layer (`inference/`)

### `inference/cuda_streams_engine.py` — Parallel Execution Core

| Code Pattern | Open-Source API | Reference |
|---|---|---|
| `torch.cuda.Stream()` | PyTorch CUDA Streams | https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html |
| `with torch.cuda.stream(stream):` | Stream context manager | https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams |
| `inp.to(device, non_blocking=True)` | Async H2D transfer | PyTorch CUDA best practices |
| `torch.cuda.synchronize()` | Stream synchronization | NVIDIA CUDA Programming Guide |
| `ThreadPoolExecutor` for CPU models | Python stdlib `concurrent.futures` | N/A |
| `torch.no_grad()` inference context | PyTorch autograd | Standard inference pattern |

**NVIDIA reference:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams

**Key insight:** Each model gets its own CUDA stream → GPU interleaves their kernels → effective parallelism on one GPU without needing multiple GPUs.

---

### `inference/gpu_memory_manager.py` — Memory Budget Planner

| Code Pattern | Open-Source API | Reference |
|---|---|---|
| `torch.cuda.get_device_properties(i).total_memory` | PyTorch CUDA introspection | https://pytorch.org/docs/stable/cuda.html |
| `torch.cuda.memory_allocated(i)` | Runtime memory tracking | PyTorch |
| Priority-sorted bin-packing placement | Algorithm: First-Fit Decreasing | Standard CS algorithm (no library) |
| `@dataclass` for `ModelPlacement`, `GPUBudget` | Python stdlib | N/A |

**Pattern:** First-Fit Decreasing bin-packing — a classical algorithm for resource allocation. Not from any specific repo.

---

### `inference/single_gpu.py` — Mode 2: Single GPU

| Code Pattern | Open-Source API | What It Does |
|---|---|---|
| `model.to("cuda").eval()` | PyTorch | Move model to GPU |
| `CUDAStreamsEngine(models, device_map)` | Custom (wraps PyTorch streams) | Parallel execution |
| `torch.from_numpy(arr).float()` | PyTorch-NumPy bridge | Convert numpy→tensor |
| Warmup loop (3 iterations before timing) | Standard ML benchmarking practice | Stabilize cuDNN autotune |

---

### `inference/hybrid_cpu_gpu.py` — Mode 3: Hybrid

| Code Pattern | Open-Source API | What It Does |
|---|---|---|
| `GPUMemoryManager.plan_placement()` | Custom (uses PyTorch CUDA) | Decide GPU/CPU split |
| `CUDAStreamsEngine` for GPU subset | Custom (wraps PyTorch streams) | GPU parallel inference |
| `ThreadPoolExecutor` for CPU subset | Python stdlib | CPU parallel inference |
| Concurrent GPU+CPU execution | Threading + CUDA overlap | Both run simultaneously |

---

### `inference/distributed_gpu.py` — Mode 1: Spark Distributed

| Code Pattern | Open-Source API | Reference |
|---|---|---|
| `SparkSession.builder.master("local[4]")` | PySpark | https://spark.apache.org/docs/3.5.1/api/python/ |
| `sc.broadcast(model_bytes_map)` | Spark broadcast variables | https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables |
| `sc.parallelize(range(N), N)` | Spark RDD creation | Same doc |
| `index_rdd.map(infer_on_partition).collect()` | Spark map-reduce | Core Spark pattern |
| `torch.save(state_dict) → bytes → broadcast` | PyTorch + Spark | Model distribution pattern |
| `os.environ.get("SPARK_EXECUTOR_GPU")` | Env-based device selection | Custom (for local vs cluster) |
| `--add-opens=java.base/...` JVM flags | Java 17 module access | Required for Arrow/PySpark on Java 17+ |

**Spark GPU scheduling reference:** https://spark.apache.org/docs/3.5.1/configuration.html#custom-resource-scheduling

**Key pattern:** Broadcast model weights once → partition data → each executor runs inference independently. This is the standard Spark ML inference pattern.

---

## Benchmark Layer (`benchmark/`)

### `benchmark/run_benchmark.py` — CLI Runner

| Code Pattern | Open-Source API | Reference |
|---|---|---|
| `argparse.ArgumentParser` | Python stdlib | CLI interface |
| `json.dump(results)` | Python stdlib | Results output |
| `platform.platform()` / `os.cpu_count()` | Python stdlib | System info |
| `torch.cuda.get_device_name(0)` | PyTorch | GPU detection |
| Markdown report generation | String formatting | Custom |

---

## Deploy Layer (`deploy/`)

### `deploy/Dockerfile`

| Component | Source | Reference |
|---|---|---|
| `nvidia/cuda:12.1.0-runtime-ubuntu22.04` | NVIDIA Docker Hub | https://hub.docker.com/r/nvidia/cuda |
| Python 3.11 via deadsnakes PPA | https://github.com/deadsnakes | Ubuntu Python builds |
| `openjdk-17-jre-headless` | Ubuntu apt | Required for PySpark |
| `--index-url https://download.pytorch.org/whl/cu121` | PyTorch CUDA wheels | https://pytorch.org/get-started/locally/ |

### `deploy/docker-compose.yml` / `docker-compose.cluster.yml`

| Pattern | Reference |
|---|---|
| `runtime: nvidia` | NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-container-toolkit |
| `shm_size: '4g'` | Docker shared memory for PyTorch DataLoader |
| `network_mode: host` | Standard Docker networking for Spark |
| `deploy.replicas: 2` | Docker Compose scaling |

---

## What's Custom vs What's Open-Source

| Category | Custom Code (ours) | Open-Source (theirs) |
|---|---|---|
| Signal models (5) | Architecture design, training loop | PyTorch `nn.Module` API |
| Image models (3) | Wrapper + fallback logic | TorchVision pretrained models |
| Detection models (2) | Wrapper + fallback CNN | Ultralytics YOLOv8 |
| CUDA streams engine | Orchestration logic | PyTorch `torch.cuda.Stream` |
| GPU memory manager | Placement algorithm | PyTorch `torch.cuda` introspection |
| Spark distribution | Broadcast+partition pattern | PySpark RDD API |
| Signal data generation | IQ waveform formulas | NumPy math functions |
| Benchmark runner | CLI + report generation | argparse, json (stdlib) |
| Docker deployment | Compose configs | Docker, NVIDIA runtime |

---

## License Summary for PoC Presentation

| Dependency | License | Commercial Use? |
|---|---|---|
| PyTorch | BSD-3-Clause | Yes |
| TorchVision | BSD-3-Clause | Yes |
| PySpark | Apache-2.0 | Yes |
| NumPy | BSD-3-Clause | Yes |
| Ultralytics YOLOv8 | AGPL-3.0 | Requires open-sourcing OR enterprise license |
| Docker | Apache-2.0 | Yes |
| NVIDIA CUDA | Proprietary (free to use) | Yes (redistribution rules apply) |

**Note on AGPL:** Ultralytics YOLOv8 is AGPL-3.0. For internal PoC/research this is fine. For production deployment, either open-source your code or purchase an Ultralytics Enterprise license. The YOLO fallback CNN in `yolo_model.py` avoids the Ultralytics dependency entirely if needed.
