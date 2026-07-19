"""
Generate inference benchmark report with graphs.
Reads results from raw JSON files and produces charts + summary markdown.

Usage:
    pip install matplotlib
    python results/generate_report.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results():
    """Load all result files."""
    results = {}
    with open(os.path.join(RESULTS_DIR, "raw_results.json")) as f:
        results["distributed"] = json.load(f)
    with open(os.path.join(RESULTS_DIR, "raw_results_single_gpu.json")) as f:
        results["single_gpu"] = json.load(f)
    with open(os.path.join(RESULTS_DIR, "raw_results_hybrid.json")) as f:
        results["hybrid"] = json.load(f)
    return results


def plot_throughput_comparison(results):
    """Bar chart comparing throughput across modes."""
    modes = ["Single GPU\n(CUDA Streams)", "Hybrid\n(CPU+GPU)", "Distributed\n(Spark Cluster)"]
    throughputs = [
        results["single_gpu"]["single_gpu"]["total_throughput"],
        results["hybrid"]["hybrid_cpu_gpu"]["total_throughput"],
        results["distributed"]["distributed_gpu"]["total_throughput"],
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(modes, throughputs, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("Inference Throughput Comparison — 10 Models, 25,700 Samples", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(throughputs) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "throughput_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: throughput_comparison.png")


def plot_latency_comparison(results):
    """Bar chart comparing elapsed time across modes."""
    modes = ["Single GPU", "Hybrid", "Distributed"]
    times = [
        results["single_gpu"]["single_gpu"]["elapsed_time"],
        results["hybrid"]["hybrid_cpu_gpu"]["elapsed_time"],
        results["distributed"]["distributed_gpu"]["elapsed_time"],
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(modes, times, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Elapsed Time (seconds)", fontsize=12)
    ax.set_title("Inference Elapsed Time — Lower is Better", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: latency_comparison.png")


def plot_per_model_samples(results):
    """Horizontal bar chart showing samples processed per model."""
    distributed = results["distributed"]["distributed_gpu"]["per_model_processed"]
    models = list(distributed.keys())
    samples = list(distributed.values())

    # Color by category
    model_info = results["distributed"]["models"]
    category_colors = {
        "signal": "#2ecc71",
        "image_classification": "#3498db",
        "object_detection": "#e74c3c",
    }
    colors = [category_colors.get(model_info[m]["category"], "#95a5a6") for m in models]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(models, samples, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, samples):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=10)

    ax.set_xlabel("Samples Processed", fontsize=12)
    ax.set_title("Samples Processed Per Model (All Modes — Same Data)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    patches = [mpatches.Patch(color=c, label=cat.replace("_", " ").title())
               for cat, c in category_colors.items()]
    ax.legend(handles=patches, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_model_samples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: per_model_samples.png")


def plot_speedup_factor(results):
    """Chart showing speedup of GPU modes vs distributed baseline."""
    distributed_tp = results["distributed"]["distributed_gpu"]["total_throughput"]
    single_gpu_tp = results["single_gpu"]["single_gpu"]["total_throughput"]
    hybrid_tp = results["hybrid"]["hybrid_cpu_gpu"]["total_throughput"]

    modes = ["Distributed\n(Baseline)", "Single GPU", "Hybrid"]
    speedups = [1.0, single_gpu_tp / distributed_tp, hybrid_tp / distributed_tp]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(modes, speedups, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}x", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylabel("Speedup Factor", fontsize=12)
    ax.set_title("Speedup vs Distributed Mode (Same Data, Same Models)", fontsize=14, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(speedups) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "speedup_factor.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: speedup_factor.png")


def plot_batch_latency(results):
    """Chart showing batch latency for GPU modes."""
    single_gpu = results["single_gpu"]["single_gpu"]
    hybrid = results["hybrid"]["hybrid_cpu_gpu"]

    modes = ["Single GPU", "Hybrid"]
    avg_latencies = [single_gpu["avg_batch_latency_ms"], hybrid["avg_batch_latency_ms"]]
    p99_latencies = [single_gpu["p99_batch_latency_ms"], hybrid["p99_batch_latency_ms"]]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, avg_latencies, width, label="Avg Latency", color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, p99_latencies, width, label="P99 Latency", color="#e74c3c", edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars1, avg_latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=10)
    for bar, val in zip(bars2, p99_latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.1f}ms", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Batch Latency — Avg vs P99 (batch_size=256)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "batch_latency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: batch_latency.png")


def plot_model_memory(results):
    """Pie chart of estimated model memory allocation."""
    models = results["distributed"]["models"]
    names = list(models.keys())
    memory = [models[n]["memory_mb"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        memory, labels=names, autopct="%1.0f%%",
        textprops={"fontsize": 9}, pctdistance=0.8,
        colors=plt.cm.Set3(np.linspace(0, 1, len(names)))
    )
    ax.set_title(f"Model Memory Distribution (Total: {sum(memory)} MB / {sum(memory)/1024:.1f} GB)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_memory.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_memory.png")


def generate_summary_report(results):
    """Generate final markdown report with embedded image references."""
    single = results["single_gpu"]["single_gpu"]
    hybrid = results["hybrid"]["hybrid_cpu_gpu"]
    distributed = results["distributed"]["distributed_gpu"]
    sys_gpu = results["single_gpu"]["system_info"]
    sys_master = results["distributed"]["system_info"]

    report = f"""# Multi-Model Inference Benchmark Report

**Date:** {sys_gpu['timestamp'][:10]}
**Platform:** Amazon Linux 2023 (AWS EC2)
**Cluster:** 1x t3.large (master/CPU) + 1x g4dn.xlarge (GPU worker)

---

## System Configuration

| Component | Master (Driver) | GPU Worker |
|-----------|----------------|------------|
| Instance | t3.large | g4dn.xlarge |
| CPU Cores | {sys_master['cpu_cores']} | {sys_gpu['cpu_cores']} |
| GPU | N/A | {sys_gpu['gpu_name']} ({sys_gpu['gpu_memory_gb']} GB) |
| CUDA | No | Yes |
| PyTorch | {sys_gpu['torch']} | {sys_gpu['torch']} |
| Spark | 3.5.1 (standalone cluster) | 3.5.1 (worker) |

## Models (10)

| Model | Category | Est. Memory |
|-------|----------|-------------|
| ew_classifier | Signal | 50 MB |
| signal_denoiser | Signal | 100 MB |
| threat_prioritizer | Signal | 350 MB |
| rf_fingerprinter | Signal | 120 MB |
| anomaly_detector | Signal | 100 MB |
| resnet18 | Image Classification | 300 MB |
| mobilenetv3 | Image Classification | 150 MB |
| efficientnet_b0 | Image Classification | 200 MB |
| yolov8_nano | Object Detection | 200 MB |
| yolov8_small | Object Detection | 400 MB |

**Total GPU Memory Required:** 1,970 MB (1.9 GB)

---

## Benchmark Results Summary

| Mode | Throughput | Elapsed Time | Speedup |
|------|-----------|-------------|---------|
| Single GPU (CUDA Streams) | **{single['total_throughput']:,.0f}** samples/sec | {single['elapsed_time']:.2f}s | {single['total_throughput']/distributed['total_throughput']:.1f}x |
| Hybrid CPU+GPU | **{hybrid['total_throughput']:,.0f}** samples/sec | {hybrid['elapsed_time']:.2f}s | {hybrid['total_throughput']/distributed['total_throughput']:.1f}x |
| Distributed (Spark, 2 partitions) | **{distributed['total_throughput']:,.0f}** samples/sec | {distributed['elapsed_time']:.2f}s | 1.0x (baseline) |

**Data:** {distributed['total_samples_processed']:,} total samples (5,000 signals/model + 200 images + 50 detections)
**Batch Size:** 256

---

## Throughput Comparison

![Throughput Comparison](throughput_comparison.png)

## Elapsed Time

![Latency Comparison](latency_comparison.png)

## Speedup Factor (vs Distributed Baseline)

![Speedup Factor](speedup_factor.png)

## Batch Latency (GPU Modes)

![Batch Latency](batch_latency.png)

| Mode | Avg Batch Latency | P99 Batch Latency |
|------|-------------------|-------------------|
| Single GPU | {single['avg_batch_latency_ms']:.2f} ms | {single['p99_batch_latency_ms']:.2f} ms |
| Hybrid | {hybrid['avg_batch_latency_ms']:.2f} ms | {hybrid['p99_batch_latency_ms']:.2f} ms |

## Per-Model Sample Distribution

![Per Model Samples](per_model_samples.png)

## Model Memory Allocation

![Model Memory](model_memory.png)

---

## Analysis

### Why Single GPU is Fastest
- All 10 models fit in the T4's 15.6 GB VRAM (only 1.97 GB needed)
- CUDA streams enable concurrent model execution without serialization overhead
- No network transfer, no data partitioning, no Spark coordination

### Why Hybrid Matches Single GPU
- All 10 models fit in GPU memory, so the hybrid scheduler places everything on GPU
- 0 models spill to CPU — effectively becomes single GPU mode
- Hybrid shines when GPU VRAM is limited (e.g., only 4 GB available)

### When Distributed Mode Wins
- Distributed mode's overhead (serialization, data partitioning, network) makes it slower for small datasets
- It becomes advantageous with:
  - Datasets > 100K samples (amortizes Spark startup cost)
  - Multiple GPU workers (linear throughput scaling)
  - Models too large for a single GPU (spread across nodes)

### Recommendations for Production EW Systems
1. **Single workstation with 1 GPU:** Use Single GPU mode (CUDA Streams)
2. **GPU memory constrained:** Use Hybrid mode (auto-spills large models to CPU)
3. **Multi-node cluster, large-scale data:** Use Distributed mode with 1 partition per GPU worker
4. **Real-time streaming:** Single GPU + TensorRT for lowest latency

---

## Environment Details

- Spark Master: `spark://10.0.0.187:7077`
- Workers: 2 (1 CPU @ t3.large, 1 GPU @ g4dn.xlarge)
- Docker Image: `multi-model-inference:latest` (CUDA 12.1 + PyTorch 2.2 + Spark 3.5.1)
- Region: ap-south-1 (Mumbai)
"""
    report_path = os.path.join(RESULTS_DIR, "INFERENCE_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: INFERENCE_REPORT.md")


def main():
    print("\n  Generating Inference Benchmark Report...")
    print("  " + "=" * 50)

    results = load_results()

    plot_throughput_comparison(results)
    plot_latency_comparison(results)
    plot_per_model_samples(results)
    plot_speedup_factor(results)
    plot_batch_latency(results)
    plot_model_memory(results)
    generate_summary_report(results)

    print("  " + "=" * 50)
    print("  Done! Open results/INFERENCE_REPORT.md for the full report.")
    print("")


if __name__ == "__main__":
    main()
