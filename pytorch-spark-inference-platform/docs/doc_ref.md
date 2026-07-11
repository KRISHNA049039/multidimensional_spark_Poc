# References & External Dependencies

## Core Frameworks

| Library | Version | Repository | Documentation | Role in Platform |
|---------|---------|-----------|---------------|-----------------|
| PyTorch | 2.2.0 | https://github.com/pytorch/pytorch | https://pytorch.org/docs/stable/ | Neural network models, CUDA streams, tensor operations |
| TorchVision | 0.17.0 | https://github.com/pytorch/vision | https://pytorch.org/vision/stable/ | Pretrained image classification models (ResNet-18, MobileNetV3, EfficientNet-B0) |
| Apache Spark (PySpark) | 3.5.1 | https://github.com/apache/spark | https://spark.apache.org/docs/3.5.1/ | Distributed inference orchestration (RDD, broadcast, executors) |
| Ultralytics YOLOv8 | 8.2.0 | https://github.com/ultralytics/ultralytics | https://docs.ultralytics.com/ | YOLOv8-Nano and YOLOv8-Small object detection |

## Data & Utility Libraries

| Library | Version | Repository | Role |
|---------|---------|-----------|------|
| NumPy | 1.26.4 | https://github.com/numpy/numpy | Synthetic data generation, array manipulation |
| Pandas | 2.2.2 | https://github.com/pandas-dev/pandas | Benchmark result tabulation |
| PyArrow | 16.1.0 | https://github.com/apache/arrow | Spark ↔ Pandas columnar interop |
| Matplotlib | 3.9.0 | https://github.com/matplotlib/matplotlib | Results visualization |
| Tabulate | 0.9.0 | https://github.com/astanin/python-tabulate | Terminal table formatting |

## Infrastructure & Docker

| Component | Reference | Role |
|-----------|-----------|------|
| NVIDIA CUDA Runtime 12.1 (Ubuntu 22.04) | https://hub.docker.com/r/nvidia/cuda | GPU base image |
| OpenJDK 17 | https://openjdk.org/ | JVM runtime for PySpark |
| Python 3.11 (deadsnakes PPA) | https://github.com/deadsnakes | Python runtime |
| NVIDIA MPS (Multi-Process Service) | https://docs.nvidia.com/deploy/mps/index.html | Multi-process GPU sharing on cluster nodes |
| Docker Compose | https://docs.docker.com/compose/ | Local and cluster deployment orchestration |

## Pretrained Model Weight Sources

| Model | Source Code | Weights Origin | Params |
|-------|------------|----------------|--------|
| ResNet-18 | `torchvision.models.resnet18` | `ResNet18_Weights.DEFAULT` (ImageNet-1K) | 11.7M |
| MobileNetV3-Small | `torchvision.models.mobilenet_v3_small` | `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet-1K) | 5.4M |
| EfficientNet-B0 | `torchvision.models.efficientnet_b0` | `EfficientNet_B0_Weights.DEFAULT` (ImageNet-1K) | 5.3M |
| YOLOv8-Nano | `ultralytics.YOLO("yolov8n.pt")` | https://github.com/ultralytics/assets/releases | 3.2M |
| YOLOv8-Small | `ultralytics.YOLO("yolov8s.pt")` | https://github.com/ultralytics/assets/releases | 11.2M |

TorchVision pretrained weights reference: https://pytorch.org/vision/stable/models.html

## EW/IQ Signal Data — Source & Generation Approach

The IQ (In-phase/Quadrature) signal data used in this platform is **synthetically generated** — there is no external dataset dependency. The generator is at `data/signal_generator.py`.

### Signal Generation Method

Each 128-dim feature vector is composed of **64 I-channel + 64 Q-channel** samples, L2-normalized. The generation follows standard EW signal modelling from radar/comms textbook references:

| Class ID | Signal Type | Generation Technique | Real-World Basis |
|----------|-------------|---------------------|-----------------|
| 0 | CW Radar | Constant-frequency cosine/sine + Gaussian noise | Continuous wave radar transmitter |
| 1 | Pulsed Radar | Rectangular pulse envelope × carrier + noise | Pulsed radar (PRI/PW modulation) |
| 2 | FMCW Radar | Linear frequency sweep (chirp) via cumulative phase | Frequency-modulated CW radar |
| 3 | Phase Coded Radar | Barker/random binary phase code × carrier | Phase-coded pulse compression radar |
| 4 | Noise Jammer | Wideband Gaussian white noise (I and Q) | Barrage noise jamming |
| 5 | Spot Jammer | High-amplitude narrowband CW + noise | Spot/targeted jamming |
| 6 | Sweep Jammer | Sinusoidal frequency modulation (FM sweep) | Sweep/responsive jamming |
| 7 | Comm Signal | 16-QAM symbol constellation + AWGN | Digital communications (QAM modulation) |

### Academic & Technical References for EW Signal Modelling

| Topic | Reference |
|-------|-----------|
| Radar signal fundamentals (CW, pulsed, FMCW) | Skolnik, M. *Introduction to Radar Systems*, 3rd Ed. McGraw-Hill, 2001 |
| Electronic warfare & jamming | Adamy, D. *EW 101: A First Course in Electronic Warfare*. Artech House, 2001 |
| IQ signal representation | Haykin, S. *Communication Systems*, 4th Ed. Wiley, 2001 |
| Phase-coded waveforms | Richards, M. *Fundamentals of Radar Signal Processing*, 2nd Ed. McGraw-Hill, 2014 |
| QAM modulation | Proakis, J. *Digital Communications*, 5th Ed. McGraw-Hill, 2007 |
| Radar signal classification with deep learning | O'Shea, T. & Corgan, J. "Convolutional Radio Modulation Recognition Networks." arXiv:1602.04105, 2016 — https://arxiv.org/abs/1602.04105 |
| RF fingerprinting via deep learning | Riyaz, S. et al. "Deep Learning Convolutional Neural Networks for Radio Identification." IEEE COMST, 2018 — https://ieeexplore.ieee.org/document/8454327 |
| DeepSig RadioML dataset (similar IQ approach) | https://www.deepsig.ai/datasets — RadioML 2016.10A / 2018.01A |

### Why Synthetic (Not Real Datasets)?

1. **Security** — Real EW signal recordings are classified/restricted
2. **Reproducibility** — Deterministic generation with `seed` parameter
3. **Benchmarking focus** — The platform benchmarks inference throughput, not model accuracy; synthetic data exercises the full pipeline identically to real data
4. **Drop-in replacement** — In production, replace `generate_ew_signals()` with real sensor feed; the 128-dim IQ format stays the same

### Related Open Datasets (for reference, not used here)

| Dataset | URL | Notes |
|---------|-----|-------|
| DeepSig RadioML 2016.10A | https://www.deepsig.ai/datasets | 11 modulations, SNR-varied, IQ format |
| DeepSig RadioML 2018.01A | https://www.deepsig.ai/datasets | 24 modulations, 1024 IQ samples/example |
| RFDataFactory | https://github.com/brysef/rfml | RF ML utilities and signal generation |
| ADS-B Signals (OpenSky) | https://opensky-network.org/ | Real ADS-B IQ recordings |

---

## Custom Model Architecture References

The 5 signal models are custom architectures. Here's the design basis for each:

| Model | File | Architecture | Design Reference |
|-------|------|-------------|-----------------|
| EW Signal Classifier | `models/ew_signal_model.py` | 4-layer MLP (128→256→512→256→128→8) with BatchNorm + Dropout | Standard MLP classifier; similar to O'Shea 2016 fully-connected baseline |
| Signal Denoiser | `models/signal_models.py` | Symmetric autoencoder (128→256→128→32→128→256→128) | Denoising autoencoder — Vincent et al. "Stacked Denoising Autoencoders" JMLR 2010 |
| Threat Prioritizer | `models/signal_models.py` | Linear projection + `nn.MultiheadAttention` (4 heads, 512-dim) → MLP scorer | Vaswani et al. "Attention Is All You Need" 2017 — https://arxiv.org/abs/1706.03762 |
| RF Fingerprinter | `models/signal_models.py` | 1D-CNN (Conv1d 3 layers + AdaptiveAvgPool + FC → 32-dim L2-normalized embedding) | Riyaz et al. 2018 (1D-CNN for RF fingerprinting); metric learning embeddings |
| Anomaly Detector | `models/signal_models.py` | VAE (encoder→μ/σ→reparameterize→decoder, anomaly = MSE reconstruction error) | Kingma & Welling "Auto-Encoding Variational Bayes" 2013 — https://arxiv.org/abs/1312.6114 |

### Key Papers Behind the Model Designs

| Paper | arXiv / URL | Relevance |
|-------|-------------|-----------|
| O'Shea & Corgan 2016 — CNN Radio Modulation Recognition | https://arxiv.org/abs/1602.04105 | Basis for IQ-based signal classification approach |
| Vaswani et al. 2017 — Attention Is All You Need | https://arxiv.org/abs/1706.03762 | Multi-head attention used in ThreatPrioritizer |
| Kingma & Welling 2013 — VAE | https://arxiv.org/abs/1312.6114 | VAE architecture for AnomalyDetector |
| Vincent et al. 2010 — Stacked Denoising Autoencoders | https://www.jmlr.org/papers/v11/vincent10a.html | Autoencoder design for SignalDenoiser |
| Riyaz et al. 2018 — Deep Learning for Radio ID | https://ieeexplore.ieee.org/document/8454327 | 1D-CNN design for RFFingerprinter |
| He et al. 2016 — Deep Residual Learning | https://arxiv.org/abs/1512.03385 | ResNet-18 architecture |
| Howard et al. 2019 — MobileNetV3 | https://arxiv.org/abs/1905.02244 | MobileNetV3-Small architecture |
| Tan & Le 2019 — EfficientNet | https://arxiv.org/abs/1905.11946 | EfficientNet-B0 architecture |
| Jocher et al. 2023 — YOLOv8 | https://github.com/ultralytics/ultralytics | YOLOv8 Nano/Small detection |

---

## Spark + GPU Architecture References

| Topic | Reference |
|-------|-----------|
| Spark GPU-aware scheduling | https://spark.apache.org/docs/3.5.1/configuration.html#custom-resource-scheduling |
| CUDA Streams programming guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams |
| NVIDIA MPS documentation | https://docs.nvidia.com/deploy/mps/index.html |
| Fractional GPU sharing in Spark | https://spark.apache.org/docs/latest/running-on-yarn.html#resource-allocation |
| PyTorch CUDA best practices | https://pytorch.org/docs/stable/notes/cuda.html |
| Spark broadcast variables | https://spark.apache.org/docs/3.5.1/rdd-programming-guide.html#broadcast-variables |

