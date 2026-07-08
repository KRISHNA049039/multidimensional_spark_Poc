# Expanded Project Plan: Multi-Model Distributed Inference

## Overview

Expand the EW PoC into a **production-ready multi-model inference platform** supporting:
- 7-10 models running in parallel
- 3 inference modes (distributed GPU, single GPU, hybrid CPU+GPU)
- Image classification + YOLO object detection + EW signal models
- AWS testing → DRDO airgapped deployment

---

## Phase 1: Multi-Model Engine (Local Docker, 1-2 days)

### Deliverables
- [ ] Multi-model inference engine with CUDA streams
- [ ] Add YOLOv8 (nano + small) for object detection
- [ ] Add ResNet-18 / MobileNetV3 for image classification
- [ ] Add 3 inference mode implementations
- [ ] GPU memory manager for hybrid mode
- [ ] Unified benchmark runner across all models
- [ ] Metrics report (throughput per model, GPU utilization, latency)

### New Project Structure

```
pytorch-spark-inference-platform/
├── models/
│   ├── ew_signal_model.py          # Existing EW classifier
│   ├── yolo_model.py               # YOLOv8 wrapper
│   ├── image_classifier.py         # ResNet-18, MobileNetV3, EfficientNet
│   ├── signal_denoiser.py          # Autoencoder for preprocessing
│   └── model_registry.py           # Load/manage all models
├── inference/
│   ├── cuda_streams_engine.py      # Multi-model CUDA streams
│   ├── distributed_gpu.py          # Spark + multi-GPU cluster
│   ├── single_gpu.py               # All models on 1 GPU
│   ├── hybrid_cpu_gpu.py           # Memory-aware CPU+GPU split
│   └── gpu_memory_manager.py       # Memory budget tracking
├── data/
│   ├── generate_signals.py         # EW signals (existing)
│   ├── generate_images.py          # Synthetic images for testing
│   └── data_loader.py              # Unified data loading
├── benchmark/
│   ├── run_benchmark.py            # All modes, all models
│   ├── metrics_collector.py        # Throughput, latency, GPU util
│   └── report_generator.py         # Markdown metrics report
├── deploy/
│   ├── Dockerfile                  # GPU image with all models
│   ├── docker-compose.yml          # Local dev
│   ├── docker-compose.cluster.yml  # Multi-node cluster
│   ├── aws/
│   │   ├── launch_cluster.sh       # AWS spot instance setup
│   │   ├── teardown.sh             # Cleanup
│   │   └── user_data.sh            # Instance bootstrap
│   └── airgapped/
│       ├── export_image.sh         # Docker save for transfer
│       ├── deploy_cluster.sh       # Start master + workers
│       └── mps_setup.sh            # NVIDIA MPS configuration
├── docs/
│   └── ... (existing + new)
├── results/
│   ├── metrics_report.md
│   └── raw_results.json
└── requirements.txt
```

---

## Phase 2: AWS Testing (1 day, ~$5-10 budget)

### Infrastructure

| Component | Instance Type | Spot Price | Role |
|-----------|--------------|-----------|------|
| Master | t3.medium | $0.04/hr | Spark driver, no GPU |
| Worker 1 | g4dn.xlarge | $0.16/hr | T4 GPU, inference |
| Worker 2 | g4dn.xlarge | $0.16/hr | T4 GPU, inference |

### Steps
1. Launch spot instances with Deep Learning AMI
2. Load Docker image (or install deps via user_data script)
3. Start Spark cluster
4. Run full benchmark (all 3 modes × 10 models × 5 scales)
5. Collect metrics
6. Tear down (save results first)

### Budget Estimate
```
Setup + transfer: 30 min = $0.18
Benchmark runs: 2 hours = $0.72
Debugging buffer: 1 hour = $0.36
─────────────────────────────────
Total estimate: ~$1.26

With safety margin: $5 budget is plenty.
```

---

## Phase 3: DRDO Airgapped Deployment (1 day)

### Pre-deployment Checklist
- [ ] Docker image tested on AWS (same architecture as target)
- [ ] All model weights included in image (no download at runtime)
- [ ] MPS setup script tested
- [ ] Spark cluster startup/teardown scripts tested
- [ ] Metrics collection verified
- [ ] No external network calls in any code path

### Transfer Package
```
ew-inference-platform.tar (Docker image): ~8-10 GB
deploy-scripts.tar.gz: ~10 KB
README-DEPLOYMENT.md: deployment instructions
```

---

## Model Details

### Models to Implement

| Model | Framework | Input | Output | Pretrained? |
|-------|-----------|-------|--------|-------------|
| EW Signal Classifier | PyTorch (custom) | 128-dim float vector | 8 classes | Trained in PoC |
| YOLOv8-nano | Ultralytics | 640×640 RGB image | Bounding boxes | Yes (COCO) |
| YOLOv8-small | Ultralytics | 640×640 RGB image | Bounding boxes | Yes (COCO) |
| ResNet-18 | torchvision | 224×224 RGB image | 1000 classes | Yes (ImageNet) |
| MobileNetV3-small | torchvision | 224×224 RGB image | 1000 classes | Yes (ImageNet) |
| EfficientNet-B0 | torchvision | 224×224 RGB image | 1000 classes | Yes (ImageNet) |
| Signal Denoiser | PyTorch (custom) | 128-dim vector | 128-dim denoised | Trained in PoC |
| Threat Prioritizer | PyTorch (custom) | Multi-modal input | Priority score | Trained in PoC |
| RF Fingerprinter | PyTorch (custom) | 128-dim vector | Emitter ID | Trained in PoC |
| Anomaly Detector | PyTorch (custom) | 128-dim vector | Anomaly score | Trained in PoC |

### Pretrained Model Handling for Airgapped

```python
# Download once (on internet machine), save to models/ directory:
import torchvision.models as models
import torch

# ResNet-18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
torch.save(resnet.state_dict(), "models/weights/resnet18_imagenet.pth")

# MobileNetV3
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
torch.save(mobilenet.state_dict(), "models/weights/mobilenetv3_small.pth")

# YOLOv8 (ultralytics saves to .pt file)
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Downloads and saves
# Copy yolov8n.pt and yolov8s.pt to models/weights/
```

All weights are packaged into the Docker image — no downloads needed at runtime.

---

## Metrics to Publish

### Per-Model Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| Throughput | samples/sec | How fast each model processes data |
| Latency (avg) | ms | Average time per sample |
| Latency (P99) | ms | 99th percentile latency |
| GPU Memory | MB | Memory consumed by model |
| GPU Utilization | % | SM occupancy during inference |
| Accuracy | % | Model accuracy on test data |

### System Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| Total throughput | samples/sec | All models combined |
| GPU Memory Used | GB | Total across all models |
| Scaling efficiency | % | actual_speedup / ideal_speedup |
| Mode comparison | ratio | Distributed vs Single vs Hybrid |

### Comparison Report Format

```markdown
## Results: 10 Models × 3 Modes × 5 Scales

| Scale | Mode | Total Throughput | GPU Util | Wall Time |
|-------|------|-----------------|----------|-----------|
| 100K | Distributed (3 GPU) | 420K/sec | 85% | 0.24s |
| 100K | Single GPU | 150K/sec | 92% | 0.67s |
| 100K | Hybrid CPU+GPU | 95K/sec | 78% | 1.05s |
| 1M | Distributed (3 GPU) | 1.2M/sec | 88% | 0.83s |
| ...
```

---

## Implementation Priority

```
Week 1:
  ├── Day 1-2: Multi-model engine + CUDA streams + 3 modes
  ├── Day 3: YOLO + image models integration
  ├── Day 4: Unified benchmark + metrics report
  └── Day 5: AWS scripts + test run

Week 2:
  ├── Day 1: Fix issues found in AWS testing
  ├── Day 2: Package for airgapped transfer
  └── Day 3: Deploy + validate on DRDO cluster
```

---

## Next Steps

Tell me which to build first:
1. **Multi-model CUDA streams engine** (core of everything)
2. **YOLO + image model integration** (new models)
3. **AWS launch scripts** (test infrastructure)
4. **All of the above** (I'll build it sequentially)
