# Philosophy Behind Models & Data Generation — Reference Guide

A non-technical overview for stakeholders explaining *why* we chose these models, *how* the data is generated, and *where* the ideas come from.

---

## 1. Why These Models?

This PoC demonstrates **real-time multi-model inference** for Electronic Warfare (EW) — classifying, denoising, prioritizing, and fingerprinting radar/jammer signals while simultaneously running image classification and object detection. The 10 models represent a realistic EW sensor fusion workload.

### The Signal Processing Models (5)

| Model | Real-World Purpose | Why It Exists |
|---|---|---|
| EW Classifier | "What type of signal is this?" (radar vs jammer vs comms) | First step in any EW receiver — identify what you're seeing |
| Signal Denoiser | Clean up noisy sensor input before downstream processing | Real sensors have noise; denoising improves accuracy of everything else |
| Threat Prioritizer | "How urgent is this signal?" Score from 0 to 1 | Operators can't look at 1000 signals — system must rank them |
| RF Fingerprinter | "Which specific emitter sent this?" Match to known devices | Track individual radar units across time/space |
| Anomaly Detector | "Is this something we've never seen before?" | Detect new/unknown threats that don't match any trained class |

**Philosophy:** These 5 models form a pipeline that mirrors how a real EW system processes intercepted signals — classify → clean → prioritize → identify → flag unknowns.

### The Image/Detection Models (5)

| Model | Real-World Purpose | Why It Exists |
|---|---|---|
| ResNet-18 | General image classification (proven, reliable) | Baseline vision model — widely deployed, well-understood |
| MobileNetV3 | Lightweight classification for edge/constrained hardware | Shows the platform handles small efficient models too |
| EfficientNet-B0 | Best accuracy-per-FLOP image classifier | Demonstrates modern efficient architecture |
| YOLOv8-Nano | Real-time object detection (fast, small) | Detect vehicles/assets in surveillance imagery |
| YOLOv8-Small | Higher-accuracy object detection | Better accuracy when compute budget allows |

**Philosophy:** Running all 10 in parallel demonstrates GPU sharing — the real challenge in EW systems where latency matters and you can't process models one at a time.

---

## 2. How & Why Data Is Generated Synthetically

### The IQ Signal Format

Real EW receivers capture radio signals as **IQ (In-phase / Quadrature)** pairs — two channels that together represent amplitude and phase of the received waveform. Our synthetic generator creates 128-dimensional vectors: 64 I-channel samples + 64 Q-channel samples.

**Why synthetic, not real data?**

1. Real EW intercepts are **classified/restricted** — can't use them in a PoC
2. Synthetic data lets us **control exactly what signals look like** — reproducible benchmarks
3. The **throughput benchmark** (what this PoC measures) doesn't depend on data content — random vs real data exercises the GPU pipeline identically
4. In production, you **swap in real sensor feeds** with zero code changes — same 128-dim format

### How Each Signal Type Is Generated

| Signal | Real-World Source | Generation Technique | Physics Behind It |
|---|---|---|---|
| CW Radar | Continuous wave radar (speed guns, simple trackers) | Constant-frequency sine/cosine | Single tone at fixed frequency |
| Pulsed Radar | Military search/track radar | Carrier gated by rectangular pulses | Transmit in bursts, listen between pulses |
| FMCW Radar | Automotive radar, altimeters | Linear frequency sweep (chirp) | Frequency ramps up linearly over time |
| Phase-Coded | Pulse compression radar (long range) | Binary code flips carrier phase | Phase modulation for range resolution |
| Noise Jammer | Hostile jamming (deny radar use) | Wideband Gaussian noise | Flood the spectrum with random energy |
| Spot Jammer | Targeted jamming (specific radar) | High-power narrowband tone | Concentrate energy on one frequency |
| Sweep Jammer | Responsive jamming (follows frequency) | FM-modulated sweep | Track and jam across frequency range |
| Comm Signal | Digital communications (radios) | QAM symbol constellation | Standard digital modulation (16-QAM) |

**Key references your manager can read:**

| Topic | Resource | Link |
|---|---|---|
| What is IQ data? (5-min intro) | Ettus/NI "What is I/Q Data?" | https://www.ettus.com/tech-notes/iq-data-explained/ |
| EW signal types (overview) | Electronic Warfare Fundamentals (USAF) | https://www.esd.whs.mil/portals/54/documents/dd/issuances/dodi/510002p.pdf |
| How radar signals work | MIT OpenCourseWare — Intro to Radar | https://ocw.mit.edu/courses/res-ll-003-build-a-small-radar-system-capable-of-sensing-range-doppler-and-synthetic-aperture-radar-imaging-january-iap-2011/ |
| Modulation recognition with DL | O'Shea & Corgan 2016 (seminal paper) | https://arxiv.org/abs/1602.04105 |
| DeepSig (company doing this commercially) | RadioML dataset & research | https://www.deepsig.ai/datasets |

---

## 3. Why These Specific Neural Network Architectures?

### MLP for Classification (EW Classifier)

- Simplest effective architecture for tabular/vector input
- 128 input features → hidden layers → 8 output classes
- Same approach used in O'Shea 2016 as baseline for RF signal classification
- **Reference:** https://arxiv.org/abs/1602.04105

### Autoencoder for Denoising

- Input a noisy signal, output a clean version
- Network learns to compress signal to essential features, then reconstruct
- Classic technique from Vincent et al. 2010
- **Reference:** https://www.jmlr.org/papers/v11/vincent10a.html

### Attention for Priority Scoring

- Multi-head attention lets the model "focus" on which signal features matter most
- Same mechanism used in ChatGPT/transformers — proven to capture complex relationships
- **Reference:** https://arxiv.org/abs/1706.03762 (Vaswani et al. "Attention Is All You Need")

### 1D-CNN for Fingerprinting

- Convolutional networks excel at finding patterns in sequential data
- Applied to 1D signals (instead of 2D images) to identify unique emitter "signatures"
- Proven in RF fingerprinting research by Riyaz et al.
- **Reference:** https://ieeexplore.ieee.org/document/8454327

### VAE for Anomaly Detection

- Variational Autoencoder learns what "normal" signals look like
- When a signal doesn't fit → high reconstruction error → anomaly flag
- No labels needed for unknown/new threats — learns from normal traffic only
- **Reference:** https://arxiv.org/abs/1312.6114 (Kingma & Welling 2013)

### Pretrained Vision Models (ResNet, MobileNet, EfficientNet)

- Industry-standard models with ImageNet weights — not custom
- Demonstrate that the platform handles standard vision workloads alongside EW
- Available off-the-shelf from PyTorch/TorchVision
- **References:**
  - ResNet: https://arxiv.org/abs/1512.03385
  - MobileNetV3: https://arxiv.org/abs/1905.02244
  - EfficientNet: https://arxiv.org/abs/1905.11946

### YOLOv8 for Object Detection

- State-of-the-art real-time object detection (2023)
- Used for detecting vehicles, assets, personnel in surveillance imagery
- Ultralytics maintains the most popular open-source implementation
- **Reference:** https://github.com/ultralytics/ultralytics

---

## 4. Why Parallel Inference on GPU? (The Core Innovation)

Traditional approach: run models **one after another** → total time = sum of all model times.

Our approach: run models **simultaneously** using CUDA Streams → total time ≈ slowest model's time.

| Approach | 10 models × 50ms each | Result |
|---|---|---|
| Sequential | 10 × 50ms = 500ms | Too slow for real-time EW |
| Parallel (CUDA Streams) | ~50-80ms total | Meets real-time requirements |

**References your manager can share:**

| Topic | Resource | Link |
|---|---|---|
| CUDA Streams explained (NVIDIA) | CUDA C Programming Guide §3.2.5 | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams |
| PyTorch CUDA best practices | PyTorch official docs | https://pytorch.org/docs/stable/notes/cuda.html |
| Multi-model serving at scale | NVIDIA Triton Inference Server (similar concept, enterprise) | https://github.com/triton-inference-server/server |
| Spark for ML inference | Databricks blog — distributed inference | https://www.databricks.com/blog/2023/04/18/distributed-inference-spark.html |

---

## 5. Similar Open-Source Projects (Comparable Work)

| Project | What It Does | How We Compare | Link |
|---|---|---|---|
| NVIDIA Triton Inference Server | Multi-model serving with GPU sharing | We do the same thing at smaller scale, without the infra complexity | https://github.com/triton-inference-server/server |
| TorchServe | PyTorch model serving | Serves models via HTTP; we run batch inference via Spark | https://github.com/pytorch/serve |
| DeepSig (commercial) | RF signal classification with deep learning | Same domain (RF/IQ signals); our models are custom-built | https://www.deepsig.ai/ |
| Ray Serve | Distributed model serving | Similar distributed concept; we use Spark instead of Ray | https://github.com/ray-project/ray |
| Spark Rapids (NVIDIA) | GPU-accelerated Spark | Lower level (DataFrame ops); we do model inference | https://github.com/NVIDIA/spark-rapids |

---

## 6. One-Paragraph Summary for Your Manager

> This PoC demonstrates that 10 AI models (5 for electronic warfare signal processing, 5 for image/object recognition) can run **simultaneously on a single GPU** using CUDA streams, achieving near-real-time throughput. The signal models are based on published academic research (neural networks for radar classification, anomaly detection, RF fingerprinting) and operate on standard IQ radio data format. The platform scales from a single laptop GPU to a multi-node Spark cluster for production deployment. All code uses open-source frameworks (PyTorch, Spark, Docker) with no proprietary dependencies. Synthetic data is used for benchmarking; production deployment swaps in real sensor feeds with zero code changes.

---

## 7. Key Links to Send

**For understanding the signal processing approach:**
- https://arxiv.org/abs/1602.04105 — "How deep learning classifies radio signals" (foundational paper)
- https://www.deepsig.ai/datasets — RadioML datasets (shows this is an established field)
- https://www.ettus.com/tech-notes/iq-data-explained/ — "What is IQ data?" (non-technical explainer)

**For understanding the GPU parallelism:**
- https://pytorch.org/docs/stable/notes/cuda.html — PyTorch CUDA streams
- https://github.com/triton-inference-server/server — NVIDIA's enterprise version of multi-model GPU serving

**For understanding the distributed architecture:**
- https://spark.apache.org/docs/3.5.1/ — Apache Spark (the distributed compute framework)
- https://www.databricks.com/blog/2023/04/18/distributed-inference-spark.html — Distributed inference patterns

**Core open-source repos we build on:**
- https://github.com/pytorch/pytorch — Neural network framework
- https://github.com/pytorch/vision — Pretrained image models
- https://github.com/ultralytics/ultralytics — YOLOv8 object detection
- https://github.com/apache/spark — Distributed compute engine
