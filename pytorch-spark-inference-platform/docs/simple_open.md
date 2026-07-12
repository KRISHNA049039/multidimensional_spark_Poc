# EW Signal PoC — Datasets & Models Reference (For Manager)

Quick-reference guide explaining the philosophy behind the model and data in `pytorch-spark-ew-poc`.

---

## What This PoC Does (One Sentence)

Classifies intercepted electronic warfare signals (radar, jammers, comms) in real-time using a neural network distributed across a GPU cluster via Apache Spark.

---

## The Model

**What:** A 5-layer neural network (MLP) that takes a 128-dimensional radio signal vector and outputs one of 8 signal types.

**Why this architecture:**
- Simple MLP is the fastest architecture for fixed-length input — critical for real-time EW
- Same approach used in the seminal paper that started this field (O'Shea 2016)
- 340K parameters — small enough to run at scale, large enough to classify accurately
- In production you'd upgrade to 1D-CNN or Transformer; the Spark pipeline stays identical

**Open-source repos behind it:**

| Component | Repo | What We Use |
|---|---|---|
| Neural network framework | https://github.com/pytorch/pytorch | `nn.Linear`, `nn.BatchNorm1d`, `nn.ReLU`, `nn.Dropout` |
| Training optimizer (Adam) | Same (PyTorch) | `torch.optim.Adam` — from Kingma & Ba 2014 |
| Model serialization for Spark | Same (PyTorch) | `torch.save` / `torch.load` state dictionaries |

**Key papers (the "philosophy"):**

| Paper | Why It Matters | Link |
|---|---|---|
| O'Shea & Corgan 2016 — "Convolutional Radio Modulation Recognition Networks" | **Foundational paper** — proved deep learning can classify radio signals from raw IQ data. Our approach is based on this. | https://arxiv.org/abs/1602.04105 |
| O'Shea, Roy, Clancy 2018 — "Over-the-Air Deep Learning Based Radio Signal Classification" | Created the RadioML 2018 dataset (2.5M samples); validated CNN-based approaches at scale | https://arxiv.org/abs/1712.04578 |
| Ioffe & Szegedy 2015 — Batch Normalization | Why we use `BatchNorm` between layers (stabilizes training, allows faster learning) | https://arxiv.org/abs/1502.03167 |
| Kingma & Ba 2014 — Adam Optimizer | Why we use Adam (adaptive learning rate, standard for DL) | https://arxiv.org/abs/1412.6980 |

---

## The Data (IQ Signals)

**What:** 128-dimensional vectors representing radio signal snapshots. 64 In-phase (I) + 64 Quadrature (Q) samples — the standard format all modern digital radio receivers produce.

**Why synthetic:**

| Reason | Explanation |
|---|---|
| Real EW intercepts are classified/restricted | Can't put them in a PoC |
| Benchmarking throughput, not accuracy | Random vs real data exercises the GPU pipeline identically |
| Reproducible | Same seed → same data every time |
| Drop-in replacement | Swap `generate_signals.py` for real sensor feed; zero code changes |

**How it works (physics-based generation):**

Each of the 8 signal classes is generated from the real physics of that signal type:

| Signal | Real-World Example | How We Generate It |
|---|---|---|
| CW Radar | Speed guns, simple missile seekers | Constant-frequency sine wave + noise |
| Pulsed Radar | AN/SPY-1 (Aegis), surveillance radar | Carrier gated by rectangular pulses |
| FMCW Radar | Automotive radar, altimeters | Linear frequency sweep (chirp) |
| Phase-Coded Radar | AN/APG-77, LPI radars | Binary phase code modulating carrier |
| Noise Jammer | AN/ALQ-99 barrage mode | Wideband Gaussian white noise |
| Spot Jammer | Targeted single-freq jamming | High-amplitude narrowband tone |
| Sweep Jammer | Responsive broadband jammer | Sinusoidal frequency modulation |
| Comm Signal | Tactical radios, datalinks | 16-QAM digital modulation |

**References for understanding IQ data:**

| Resource | What It Explains | Link |
|---|---|---|
| Ettus/NI "What is I/Q Data?" | Non-technical 5-min explainer of IQ format | https://www.ettus.com/tech-notes/iq-data-explained/ |
| Skolnik, *Introduction to Radar Systems* (3rd Ed) | The radar engineering bible — signal representation, pulse types | Textbook (McGraw-Hill, 2001) |
| Wiley, *ELINT: Interception and Analysis of Radar Signals* | How real EW systems intercept and classify signals | Textbook (Artech House, 2006) |
| Adamy, *EW 101: A First Course in Electronic Warfare* | Accessible intro to EW concepts for non-specialists | Textbook (Artech House, 2001) |

---

## Similar Public Datasets (We Don't Use Them, But They Validate Our Approach)

| Dataset | What It Is | Why It Matters | Link |
|---|---|---|---|
| RadioML 2016.10A | 220K IQ signal samples, 11 modulation types | Proves DL on IQ data works — same 128-dim format as ours | https://www.deepsig.ai/datasets |
| RadioML 2018.01A | 2.5M samples, 24 modulations, SNR-varied | Industry standard benchmark for signal classification | https://www.deepsig.ai/datasets |
| DeepRadar (Kaggle) | Radar-specific signals | Validates radar classification with neural nets | https://www.kaggle.com/datasets/khilian/deepradar |
| TorchSig | PyTorch toolkit with 50+ signal types | Production-grade signal ML framework (MIT Lincoln Lab) | https://github.com/TorchDSP/torchsig |

**Key point:** The fact that RadioML uses the exact same IQ format (128×2 → 256 samples, or in their case 1024×2) validates our data representation approach. We're using the same philosophy as the field's standard datasets.

---

## Similar Open-Source Projects (Industry Context)

| Project | What They Do | How We Compare | Link |
|---|---|---|---|
| DeepSig (commercial) | RF signal classification products using DL | Same domain; they commercialized O'Shea's research | https://www.deepsig.ai/ |
| TorchSig (MIT Lincoln Lab) | Open-source signal classification toolkit | More mature; we use similar approach at smaller scope | https://github.com/TorchDSP/torchsig |
| rfml (RFDataFactory) | Synthetic RF data generation | Similar synthetic generation approach | https://github.com/brysef/rfml |
| GNU Radio | Open-source SDR framework | Can generate/receive real signals; our data mimics its output | https://www.gnuradio.org/ |

---

## Why This Matters (Business Context)

1. **The field is proven** — O'Shea's 2016 paper has 2000+ citations. DL-based signal classification is not experimental, it's deployed.

2. **Our approach matches industry** — Same IQ format, same neural network family, same classification taxonomy as RadioML/DeepSig/TorchSig.

3. **The innovation is the distributed pipeline** — Many people can classify signals with a neural net. Our value-add is running it at scale via Spark across a GPU cluster in real-time.

4. **Drop-in upgrade path** — The synthetic data and simple MLP prove the pipeline works. Swapping in real sensor data and a more complex model (1D-CNN, Transformer) requires changing one file, not redesigning the system.

---

## Quick Links to Send

**"What is IQ data?"** → https://www.ettus.com/tech-notes/iq-data-explained/

**"How does DL classify radio signals?"** → https://arxiv.org/abs/1602.04105

**"What datasets exist for this?"** → https://www.deepsig.ai/datasets

**"Who else is doing this?"** → https://www.deepsig.ai/ and https://github.com/TorchDSP/torchsig

**"What is Spark's role?"** → https://www.databricks.com/blog/2023/04/18/distributed-inference-spark.html

**Our core open-source dependency:** → https://github.com/pytorch/pytorch
