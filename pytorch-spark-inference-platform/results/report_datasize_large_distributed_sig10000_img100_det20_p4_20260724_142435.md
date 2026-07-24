# Multi-Model Distributed Inference — Performance Metrics

**Generated:** 2026-07-24T14:24:21.961464

## System

| Component | Value |
|-----------|-------|
| Platform | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35 |
| Python | 3.11.15 |
| PyTorch | 2.2.0+cu121 |
| CPU Cores | 20 |
| GPU | N/A (N/A GB) |
| GPU Count | 0 |
| CUDA | False |

## Models (10)

| # | Model | Category | Est. Memory |
|---|-------|----------|-------------|
| 1 | ew_classifier | signal | 50 MB |
| 2 | signal_denoiser | signal | 100 MB |
| 3 | threat_prioritizer | signal | 350 MB |
| 4 | rf_fingerprinter | signal | 120 MB |
| 5 | anomaly_detector | signal | 100 MB |
| 6 | resnet18 | image_classification | 300 MB |
| 7 | mobilenetv3 | image_classification | 150 MB |
| 8 | efficientnet_b0 | image_classification | 200 MB |
| 9 | yolov8_nano | object_detection | 200 MB |
| 10 | yolov8_small | object_detection | 400 MB |

## Inference Mode Comparison

| Mode | Total Throughput | Time (sec) | Models on GPU | Models on CPU |
|------|-----------------|------------|---------------|---------------|
| distributed_gpu | 6,175 samples/sec | 8.15 | 10 | 0 |

## Per-Model Processing (samples)

| Model | Single GPU | Hybrid | Distributed |
|-------|-----------|--------|-------------|
| anomaly_detector | - | - | 10000 |
| efficientnet_b0 | - | - | 100 |
| ew_classifier | - | - | 10000 |
| mobilenetv3 | - | - | 100 |
| resnet18 | - | - | 100 |
| rf_fingerprinter | - | - | 10000 |
| signal_denoiser | - | - | 10000 |
| threat_prioritizer | - | - | 10000 |
| yolov8_nano | - | - | 20 |
| yolov8_small | - | - | 20 |

## Recommendations

- **Single GPU mode** is best for workstation inference with sufficient VRAM
- **Hybrid mode** is best when GPU memory is limited (models overflow to CPU)
- **Distributed mode** scales linearly with cluster size for production EW systems
- Enable **NVIDIA MPS** on cluster workers for true multi-process GPU sharing
- Use **CUDA streams** within each executor for intra-node model parallelism
