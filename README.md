# Model Architecture, Features & Training — Technical Reference

## 1. Signal Representation: IQ (In-phase / Quadrature) Data

### What is IQ Data?

Every radar receiver and electronic warfare (EW) system captures signals in **IQ format** — a standard baseband representation of a radio frequency signal as two orthogonal components:

```
s(t) = I(t) + j·Q(t)

Where:
  I(t) = In-phase component (real part)
  Q(t) = Quadrature component (imaginary part)
  j = √(-1)
```

This is how ALL modern digital receivers (ELINT systems, ESM sensors, SDRs) represent intercepted signals internally. It preserves both amplitude and phase information of the original RF waveform after downconversion to baseband.

### Why IQ?

- Captures full signal information (amplitude, frequency, phase)
- Hardware-native format — no preprocessing needed from receiver output
- Standard across military and commercial RF systems
- Directly amenable to neural network processing

### Textbook References

| Reference | Relevance |
|-----------|-----------|
| Skolnik, M.I. *Introduction to Radar Systems*, 3rd Edition, McGraw-Hill, 2001 | Standard radar engineering textbook; Ch. 3 covers radar receiver signal representation including I/Q demodulation |
| Richards, M.A. *Fundamentals of Radar Signal Processing*, 2nd Edition, McGraw-Hill, 2014 | Ch. 2 covers baseband signal models, IQ sampling, and digital receiver architectures |
| Wiley, R.G. *ELINT: The Interception and Analysis of Radar Signals*, Artech House, 2006 | The definitive ELINT reference; covers signal interception, pulse analysis, and emitter classification |
| Pace, P.E. *Detecting and Classifying Low Probability of Intercept Radar*, 2nd Edition, Artech House, 2009 | Covers LPI radar waveforms (FMCW, phase-coded) and their detection/classification |

---

## 2. Feature Vector Schema

### Layout

```
Feature Vector: 128 dimensions (float32)
┌────────────────────────────────────────────────────────────────────┐
│  I_0, I_1, I_2, ..., I_63  │  Q_0, Q_1, Q_2, ..., Q_63          │
│  ← 64 In-phase samples →   │  ← 64 Quadrature samples →          │
└────────────────────────────────────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| Total dimensions | 128 |
| I-channel samples | 64 (indices 0-63) |
| Q-channel samples | 64 (indices 64-127) |
| Data type | float32 |
| Normalization | L2-normalized (unit vector) |
| Time window | 64 samples representing a signal observation window |
| Sampling scheme | Uniform time sampling over observation interval |

### Why 128 Dimensions?

- 64 complex IQ samples = 128 real values (standard split)
- 64 time steps provides sufficient bandwidth for classifying modulation types
- Matches the RadioML 2016.10A format (128 samples per observation)
- Small enough for high-throughput inference, rich enough for classification

### L2 Normalization

```python
feature_vec = feature_vec / np.linalg.norm(feature_vec)
```

This forces the classifier to learn **signal shape** (modulation pattern) rather than **signal strength** (which varies with range, antenna gain, propagation). In real EW systems, signal amplitude is unreliable for classification — a CW radar at 10km looks the same as one at 100km, just louder.

---

## 2.1 Training Dataset Schema

### Overview

| Property | Value |
|----------|-------|
| Total training samples | 10,000 |
| Samples per class | 1,250 (balanced) |
| Feature matrix shape | `(10000, 128)` — dtype: float32 |
| Label vector shape | `(10000,)` — dtype: int64 |
| Total memory | ~5 MB |
| Format | In-memory numpy arrays (generated on-the-fly) |

### Dataset Structure

```
Training Dataset
├── features: numpy.ndarray
│     shape: (10000, 128)
│     dtype: float32
│     range: [-1.0, 1.0] (L2-normalized)
│     
│     Column layout:
│     ┌───────────────────────────────────────────────────────────┐
│     │ Col 0-63:   I-channel time samples (In-phase)            │
│     │ Col 64-127: Q-channel time samples (Quadrature)          │
│     └───────────────────────────────────────────────────────────┘
│
└── labels: numpy.ndarray
      shape: (10000,)
      dtype: int64
      values: [0, 1, 2, 3, 4, 5, 6, 7]
      distribution: uniform (1250 per class)
```

### Column Definitions

| Column Index | Name | Description | Unit | Range |
|-------------|------|-------------|------|-------|
| 0 | I_0 | In-phase sample at time t=0 | normalized amplitude | [-1, 1] |
| 1 | I_1 | In-phase sample at time t=1/64 | normalized amplitude | [-1, 1] |
| 2 | I_2 | In-phase sample at time t=2/64 | normalized amplitude | [-1, 1] |
| ... | ... | ... | ... | ... |
| 63 | I_63 | In-phase sample at time t=63/64 | normalized amplitude | [-1, 1] |
| 64 | Q_0 | Quadrature sample at time t=0 | normalized amplitude | [-1, 1] |
| 65 | Q_1 | Quadrature sample at time t=1/64 | normalized amplitude | [-1, 1] |
| ... | ... | ... | ... | ... |
| 127 | Q_63 | Quadrature sample at time t=63/64 | normalized amplitude | [-1, 1] |

### Label Mapping

| Label Value | Class Name | Signal Type | Threat Category |
|-------------|-----------|-------------|-----------------|
| 0 | CW_Radar | Continuous Wave radar | Threat emitter |
| 1 | Pulsed_Radar | Pulsed radar | Threat emitter |
| 2 | FMCW_Radar | Frequency Modulated CW radar | Threat emitter |
| 3 | Phase_Coded_Radar | Phase-coded pulse compression | Threat emitter |
| 4 | Noise_Jammer | Broadband noise jammer | Electronic attack |
| 5 | Spot_Jammer | Narrowband spot jammer | Electronic attack |
| 6 | Sweep_Jammer | Swept frequency jammer | Electronic attack |
| 7 | Comm_Signal | Communication signal (16-QAM) | Non-threat |

### Sample Data (first 3 rows, first 8 features)

```
features[0] = [ 0.0821,  0.0953,  0.1074,  0.1180,  0.1266,  0.1330,  0.1368,  0.1381, ... ]
features[1] = [-0.0234,  0.0412,  0.0891, -0.0567,  0.1023, -0.0890,  0.0345,  0.0678, ... ]
features[2] = [ 0.1456,  0.1389,  0.1298,  0.1185,  0.1054,  0.0908,  0.0749,  0.0582, ... ]

labels[0] = 3  (Phase_Coded_Radar)
labels[1] = 4  (Noise_Jammer)
labels[2] = 0  (CW_Radar)
```

### Equivalent Tabular Schema (if stored as CSV/Parquet)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Column Name │ Type    │ Description                                          │
├─────────────┼─────────┼──────────────────────────────────────────────────────┤
│ sample_id   │ int32   │ Unique row identifier (0 to N-1)                     │
│ i_0         │ float32 │ In-phase sample 0                                    │
│ i_1         │ float32 │ In-phase sample 1                                    │
│ ...         │ ...     │ ...                                                  │
│ i_63        │ float32 │ In-phase sample 63                                   │
│ q_0         │ float32 │ Quadrature sample 0                                  │
│ q_1         │ float32 │ Quadrature sample 1                                  │
│ ...         │ ...     │ ...                                                  │
│ q_63        │ float32 │ Quadrature sample 63                                 │
│ label       │ int64   │ Signal class (0-7)                                   │
│ class_name  │ string  │ Human-readable class label                           │
│ category    │ string  │ Threat category (threat/jammer/non-threat)            │
└─────────────┴─────────┴──────────────────────────────────────────────────────┘

Total columns: 131 (1 id + 128 features + 1 label + 1 class_name)
```

### Inference Dataset Schema

The inference dataset is identical to training features, without labels:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Column Name │ Type    │ Description                                          │
├─────────────┼─────────┼──────────────────────────────────────────────────────┤
│ signal_id   │ int32   │ Unique signal intercept identifier                   │
│ i_0         │ float32 │ In-phase sample 0                                    │
│ ...         │ float32 │ (64 I-channel columns)                               │
│ i_63        │ float32 │ In-phase sample 63                                   │
│ q_0         │ float32 │ Quadrature sample 0                                  │
│ ...         │ float32 │ (64 Q-channel columns)                               │
│ q_63        │ float32 │ Quadrature sample 63                                 │
│ timestamp   │ int64   │ Signal intercept time (optional, for real data)       │
│ sensor_id   │ string  │ Originating sensor/receiver ID (optional)             │
└─────────────┴─────────┴──────────────────────────────────────────────────────┘
```

### Data Generation Parameters per Class

| Class | Carrier Freq Range | Noise σ | Amplitude | Special Parameters |
|-------|-------------------|---------|-----------|-------------------|
| CW_Radar | [0.1, 0.3] | 0.05 | 1.0 | — |
| Pulsed_Radar | [0.2, 0.4] | 0.10 | 1.0 | pulse_width ∈ [0.1, 0.3] |
| FMCW_Radar | [0.05→0.45] sweep | 0.08 | 1.0 | f_start, f_end random |
| Phase_Coded | [0.2, 0.3] | 0.10 | 1.0 | 8-chip Barker-like code |
| Noise_Jammer | N/A (wideband) | 1.00 | — | Pure Gaussian |
| Spot_Jammer | [0.2, 0.25] | 0.30 | [2.0, 4.0] | High amplitude |
| Sweep_Jammer | [0.05→0.49] | 0.20 | 1.5 | sweep_rate ∈ [2, 5] |
| Comm_Signal | N/A (baseband) | 0.15 | 1/3 | 16-QAM, 8 symbols |

---

## 3. Signal Classes (8 categories)

### Class Definitions

| ID | Label | Physical Description | Generation Method |
|----|-------|---------------------|-------------------|
| 0 | CW_Radar | Continuous Wave — single-frequency transmission | `cos(2πft) + jsin(2πft) + noise` |
| 1 | Pulsed_Radar | Rectangular pulse envelope modulating a carrier | `rect(t/τ) × carrier + noise` |
| 2 | FMCW_Radar | Frequency Modulated Continuous Wave — linear chirp | `cos(2π∫f(t)dt)` where `f(t)` sweeps linearly |
| 3 | Phase_Coded_Radar | Pulse compression using binary phase codes (Barker-like) | `code[n] × carrier`, code ∈ {-1, +1} |
| 4 | Noise_Jammer | Broadband Gaussian noise (barrage jamming) | `N(0,σ²)` for both I and Q |
| 5 | Spot_Jammer | High-power narrowband interference at a specific frequency | `A·cos(2πf₀t) + noise`, A >> 1 |
| 6 | Sweep_Jammer | Swept-frequency jammer covering a band | `cos(2π∫f(t)dt)`, f(t) sinusoidally varying |
| 7 | Comm_Signal | Digital communication (16-QAM constellation) | Symbol sequence mapped to I/Q constellation |

### EW Taxonomy Basis

This classification scheme follows the standard ELINT/ESM taxonomy:

1. **Threat emitters** (Classes 0-3): Radar signals that need to be identified for threat assessment
2. **Electronic attack** (Classes 4-6): Jammer signals that need to be recognized to apply countermeasures
3. **Non-threat** (Class 7): Communication signals that should be ignored by threat warning systems

This is consistent with how real EW systems categorize intercepted emissions for prioritization.

### Signal Generation Details

Each class is generated using physics-based models:

**CW Radar (Class 0):**
```python
freq = random(0.1, 0.3)  # Normalized frequency
I = cos(2π·freq·t) + N(0, 0.05)
Q = sin(2π·freq·t) + N(0, 0.05)
```
Real examples: Police speed radar, some missile seekers, Doppler-only surveillance

**Pulsed Radar (Class 1):**
```python
freq = random(0.2, 0.4)
pulse_width = random(0.1, 0.3)  # Duty cycle
envelope = (t mod PRI) < pulse_width
I = envelope × cos(2π·freq·t) + N(0, 0.1)
Q = envelope × sin(2π·freq·t) + N(0, 0.1)
```
Real examples: AN/SPY-1 (Aegis), AN/APG-68, most surveillance radars

**FMCW Radar (Class 2):**
```python
f_start, f_end = random ranges
f(t) = f_start + (f_end - f_start)·t  # Linear chirp
phase = 2π·∫f(t)dt
I = cos(phase) + N(0, 0.08)
Q = sin(phase) + N(0, 0.08)
```
Real examples: Automotive radar (77GHz), radar altimeters, some targeting radars

**Phase-Coded Radar (Class 3):**
```python
code = random_barker_like(length=8)  # Binary {-1, +1}
phase_code = repeat_each(code, 8)  # 8 chips × 8 samples each
I = phase_code × cos(2π·freq·t) + N(0, 0.1)
Q = phase_code × sin(2π·freq·t) + N(0, 0.1)
```
Real examples: Pulse compression radars, LPI radars, AN/APG-77

**Noise Jammer (Class 4):**
```python
I = N(0, 1.0)  # Wideband Gaussian
Q = N(0, 1.0)
```
Real examples: AN/ALQ-99 barrage mode, SLQ-32 noise jamming

**Spot Jammer (Class 5):**
```python
freq = random(0.2, 0.25)  # Narrow band
amplitude = random(2.0, 4.0)  # High power
I = amplitude × cos(2π·freq·t) + N(0, 0.3)
Q = amplitude × sin(2π·freq·t) + N(0, 0.3)
```
Real examples: Targeted CW jamming on known threat radar frequency

**Sweep Jammer (Class 6):**
```python
f(t) = f_start + (f_end - f_start) × (0.5 + 0.5·sin(2π·rate·t))
phase = 2π·∫f(t)dt
I = 1.5 × cos(phase) + N(0, 0.2)
Q = 1.5 × sin(phase) + N(0, 0.2)
```
Real examples: Swept-spot jammer covering a frequency band

**Communication Signal (Class 7):**
```python
symbols_I = random_choice([-3,-1,1,3], size=8) / 3  # 16-QAM
symbols_Q = random_choice([-3,-1,1,3], size=8) / 3
I = repeat_each(symbols_I, 8) + N(0, 0.15)
Q = repeat_each(symbols_Q, 8) + N(0, 0.15)
```
Real examples: Tactical datalinks, communication radios

---

## 4. Model Architecture

### Network Design

```
EWSignalClassifier (339,912 parameters)
═══════════════════════════════════════════════════════

Input Layer:        128 features (IQ vector)
                         │
Feature Extractor:       │
    ┌────────────────────┼────────────────────────┐
    │  Linear(128 → 256)                          │
    │  BatchNorm1d(256)                           │
    │  ReLU                                       │
    │  Dropout(0.3)                               │
    │                                             │
    │  Linear(256 → 512)                          │
    │  BatchNorm1d(512)                           │
    │  ReLU                                       │
    │  Dropout(0.3)                               │
    │                                             │
    │  Linear(512 → 256)                          │
    │  BatchNorm1d(256)                           │
    │  ReLU                                       │
    │  Dropout(0.2)                               │
    │                                             │
    │  Linear(256 → 128)                          │
    │  BatchNorm1d(128)                           │
    │  ReLU                                       │
    │  Dropout(0.2)                               │
    └────────────────────┼────────────────────────┘
                         │
Classifier:              │
    ┌────────────────────┼────────────────────────┐
    │  Linear(128 → 64)                           │
    │  ReLU                                       │
    │  Linear(64 → 8)    ← 8 signal classes       │
    └────────────────────┼────────────────────────┘
                         │
Output:             8 logits → argmax → class prediction
```

### Design Choices

| Choice | Rationale |
|--------|-----------|
| MLP (not CNN/LSTM) | IQ vectors are fixed-length 1D; MLP is simplest and fastest for throughput benchmarking |
| Expansion then compression (128→256→512→256→128) | Learns hierarchical feature combinations in higher dimensions, then compresses to discriminative representation |
| BatchNorm after every linear layer | Stabilizes training, allows higher learning rates, acts as regularizer |
| Dropout (0.3→0.3→0.2→0.2) | Prevents overfitting; decreasing rate as representations become more refined |
| ReLU activation | Standard, fast, no vanishing gradient issues |
| No softmax in forward() | CrossEntropyLoss includes LogSoftmax internally (PyTorch convention) |
| 339K parameters | Lightweight for high-throughput inference; production EW systems need speed over marginal accuracy gains |

### Why Not CNN or Transformer?

For this PoC, the goal is **throughput benchmarking**, not state-of-the-art accuracy. An MLP:
- Has predictable, consistent inference time per sample
- No variable-length input complications
- Serialization/deserialization is simple and fast
- Represents a realistic "deployed model" size for edge/embedded EW systems

For production, you'd likely use 1D-CNN (better feature extraction from time-series IQ) or a Transformer (captures long-range temporal dependencies). The Spark distribution architecture remains identical — only the model bytes change.

---

## 5. Training Process

### Configuration

| Parameter | Value |
|-----------|-------|
| Training samples | 10,000 (synthetic, balanced) |
| Samples per class | 1,250 |
| Optimizer | Adam (lr=0.001) |
| Loss function | CrossEntropyLoss |
| Batch size | 256 |
| Epochs | 20 |
| Seed | 42 (reproducible) |

### Training Loop

```python
for epoch in range(20):
    for batch_x, batch_y in DataLoader(train_data, batch_size=256, shuffle=True):
        optimizer.zero_grad()
        output = model(batch_x)
        loss = CrossEntropyLoss(output, batch_y)
        loss.backward()
        optimizer.step()
```

### Why Quick Training?

The model achieves ~95% accuracy in 20 epochs because:
1. Synthetic classes are designed to be **separable** in feature space
2. We only need **realistic weights** for inference benchmarking — not SOTA accuracy
3. Fast training allows the benchmark to run quickly end-to-end

On real intercepted data with noise, interference, and edge cases, you'd train much longer with data augmentation.

---

## 6. Inference Pipeline

### Single-Node Flow

```
Input numpy array (N × 128)
    → torch.FloatTensor
    → model.to(device)  [CPU or CUDA]
    → model.eval()
    → torch.no_grad()
    → Forward pass (batches of 1024)
    → torch.argmax(logits, dim=1)
    → predictions.cpu().numpy()
```

### Spark Distributed Flow

```
Driver:
    1. Serialize model → bytes (torch.save state_dict)
    2. Broadcast model bytes to all executors
    3. Partition data into N chunks
    4. Create RDD of partition indices

Each Executor/Partition:
    1. Receive model bytes from broadcast
    2. Deserialize model (torch.load)
    3. Move model to GPU (if available)
    4. Run batched inference on local data chunk
    5. Return (count, correct) tuple

Driver:
    6. Collect results
    7. Compute aggregate throughput and accuracy
```

### Key Optimization: Model Broadcasting

Instead of shipping the model N times (once per task), Spark's broadcast mechanism sends it once per executor and caches it. For a 339K-parameter model (~1.4MB serialized), this saves negligible network bandwidth. But for large models (BERT, ResNet-50, etc.), broadcasting saves significant transfer time.

---

## 7. Open-Source Datasets & Tools (for future enhancement)

### Datasets

| Dataset | Samples | Classes | Format | Link |
|---------|---------|---------|--------|------|
| RadioML 2016.10A | 220K | 11 modulations | IQ (128×2) | [DeepSig Datasets](https://www.deepsig.ai/datasets) |
| RadioML 2018.01A | 2.5M | 24 modulations | IQ (1024×2) | [DeepSig Datasets](https://www.deepsig.ai/datasets) |
| DeepRadar | Radar signals | Multiple radar types | IQ | [Kaggle: DeepRadar](https://www.kaggle.com/datasets/khilian/deepradar) |

Content was rephrased for compliance with licensing restrictions. The RadioML datasets contain 24 modulation types across 26 SNR levels with 4096 frames per combination, as described by [DeepSig](https://www.deepsig.ai/datasets) and [cyclostationary.blog](https://cyclostationary.blog/2020/09/24/deepsigs-2018-data-set-2018-01-osc-0001_1024x2m-h5-tar-gz/).

For a comprehensive comparison of available RF datasets, see [panoradio-sdr.de](https://panoradio-sdr.de/overview-of-open-datasets-for-rf-signal-classification/).

### Open-Source Tools

| Tool | Description | Link |
|------|-------------|------|
| TorchSig | PyTorch signal processing ML toolkit, 50+ modulations | [github.com/TorchDSP/torchsig](https://github.com/TorchDSP/torchsig) |
| rfml (RFDataFactory) | Synthetic RF signal generation library | [github.com/brysef/rfml](https://github.com/brysef/rfml) |
| GNU Radio | Open-source SDR framework (can generate real signals) | [gnuradio.org](https://www.gnuradio.org/) |
| SigMF | Signal Metadata Format standard | [github.com/sigmf/sigmf](https://github.com/sigmf/sigmf) |

### Research Papers

| Paper | Year | Relevance |
|-------|------|-----------|
| O'Shea & Corgan, "Convolutional Radio Modulation Recognition Networks" | 2016 | Foundational work on DNN-based AMC using IQ data |
| O'Shea, Roy, Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification" | 2018 | RadioML 2018 dataset and CNN architectures |
| [Multi-task Learning for Radar Signal Characterisation](https://arxiv.org/html/2306.13105) | 2023 | Radar signal classification as multi-task learning problem |
| [LSTM Framework for Classification of Radar and Communications Signals](https://arxiv.org/html/2305.03192v2) | 2023 | LSTM-based classification of radar vs comms with DeepRadar dataset |
| [Large Scale Radio Frequency Signal Classification](https://arxiv.org/abs/2207.09918) | 2022 | TorchSig paper — large-scale RF signal classification framework |
| [Recognition of Noisy Radar Emitter Signals Using Deep Residual Shrinkage Network](https://mdpi.com/1424-8220/21/23/7973/htm) | 2021 | 1D deep residual network for radar emitter recognition |
| [Radar Emitter Signal Recognition Based on 1D-CNN with Attention](https://www.mdpi.com/1424-8220/20/21/6350/htm) | 2020 | CNN + attention for radar emitter identification |

---

## 8. Summary

| Aspect | This PoC | Production Alternative |
|--------|----------|----------------------|
| Data source | Synthetic (physics-based generation) | RadioML 2018.01A, real intercepts, TorchSig |
| Feature representation | 128-dim IQ vector (64 I + 64 Q) | 1024+ samples, spectrogram, or time-frequency |
| Model architecture | 5-layer MLP (339K params) | 1D-CNN, ResNet-1D, or Transformer |
| Training | 20 epochs, 10K samples | 100+ epochs, millions of samples, augmentation |
| Accuracy | ~95% (synthetic, well-separated classes) | 85-93% (real data, challenging SNR) |
| Inference focus | **Throughput benchmarking** (Spark vs single-node) | Accuracy + latency in operational conditions |
| Deployment | Docker / spark-submit | Kubernetes, YARN, edge devices |

The model and data are **purpose-built for this PoC** — they demonstrate that the Spark-distributed inference architecture works correctly and at scale. Swapping in a real dataset and more complex model requires only changing the model class and data loader; the entire Spark distribution, broadcasting, and inference pipeline remains unchanged.
