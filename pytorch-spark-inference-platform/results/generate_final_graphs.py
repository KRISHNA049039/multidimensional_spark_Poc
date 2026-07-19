"""Generate graphs for the Final Comprehensive Report."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_mode_comparison():
    """All 3 modes side by side — GPU results."""
    modes = ["Single GPU\n(CUDA Streams)", "Hybrid\n(CPU+GPU)", "Distributed\n(Spark Cluster)"]
    throughputs = [30087, 29910, 1880]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(modes, throughputs, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{val:,}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("Inference Mode Comparison — 10 Models, 25,700 Samples", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 35000)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_mode_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: final_mode_comparison.png")


def plot_scaling_throughput():
    """Incremental load test — throughput vs data size."""
    samples = [2580, 5190, 10360, 25700, 51360]
    throughputs = [327, 973, 1280, 3197, 3046]
    data_mb = [135.7, 289.5, 480.7, 865.6, 1534.6]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = "#2ecc71"
    ax1.plot(samples, throughputs, 'o-', color=color1, linewidth=2, markersize=8, label="Throughput")
    ax1.set_xlabel("Total Samples", fontsize=12)
    ax1.set_ylabel("Throughput (samples/sec)", fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 3800)

    for i, (x, y, d) in enumerate(zip(samples, throughputs, data_mb)):
        ax1.annotate(f"{y:,}/s\n({d:.0f}MB)", (x, y), textcoords="offset points",
                     xytext=(0, 15), ha='center', fontsize=9)

    ax1.set_title("Distributed Mode — Throughput Scaling with Data Size", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_scaling_throughput.png"), dpi=150)
    plt.close()
    print("  Saved: final_scaling_throughput.png")


def plot_executor_distribution():
    """Pie chart of task distribution across executors."""
    labels = ["GPU Worker\n(10.0.0.45)", "CPU Worker\n(10.0.0.187)"]
    tasks = [9, 5]
    colors = ["#2ecc71", "#3498db"]
    explode = (0.05, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(tasks, labels=labels, autopct="%1.0f%%",
                                       colors=colors, explode=explode,
                                       textprops={"fontsize": 12},
                                       pctdistance=0.75, startangle=90)
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(14)
    ax.set_title("Task Distribution Across Executors\n(14 total tasks, 5 jobs)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_executor_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: final_executor_distribution.png")


def plot_stage_timeline():
    """Horizontal bar chart showing stage durations."""
    stages = ["Stage 0\n(Run 1, 2.5K)", "Stage 1\n(Run 2, 5.2K)", "Stage 2\n(Run 3, 10K)",
              "Stage 3\n(Run 4, 25K)", "Stage 4\n(Run 5, 51K)"]
    durations = [7.7, 5.3, 8.1, 8.0, 16.8]
    tasks_per_stage = [2, 2, 2, 4, 4]
    colors = ["#e74c3c" if t == 4 else "#3498db" for t in tasks_per_stage]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(stages, durations, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val, t in zip(bars, durations, tasks_per_stage):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}s ({t} tasks)", va="center", fontsize=10)

    ax.set_xlabel("Duration (seconds)", fontsize=12)
    ax.set_title("Spark Stage Duration — Distributed Incremental Test", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color="#3498db", label="2 partitions"),
               mpatches.Patch(color="#e74c3c", label="4 partitions")]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_stage_timeline.png"), dpi=150)
    plt.close()
    print("  Saved: final_stage_timeline.png")


def plot_gpu_vs_cpu():
    """Compare GPU vs CPU execution for same workload."""
    categories = ["Single GPU\n(GPU Worker)", "Hybrid\n(GPU Worker)", "Single GPU\n(CPU Fallback)",
                  "Hybrid\n(CPU Fallback)", "Distributed\n(GPU+CPU)"]
    throughputs = [30087, 29910, 1332, 1334, 1880]
    colors = ["#2ecc71", "#2ecc71", "#95a5a6", "#95a5a6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, throughputs, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("GPU vs CPU — Same Data, Same Models (25,700 samples)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 35000)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color="#2ecc71", label="GPU (Tesla T4)"),
               mpatches.Patch(color="#95a5a6", label="CPU only (m5.xlarge)"),
               mpatches.Patch(color="#e74c3c", label="Distributed (GPU+CPU)")]
    ax.legend(handles=patches, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_gpu_vs_cpu.png"), dpi=150)
    plt.close()
    print("  Saved: final_gpu_vs_cpu.png")


def plot_latency_breakdown():
    """Batch latency comparison: GPU vs CPU."""
    modes = ["GPU\n(Single)", "GPU\n(Hybrid)", "CPU\n(Single)", "CPU\n(Hybrid)"]
    avg_lat = [42.86, 42.96, 964.35, 963.31]
    p99_lat = [638.02, 640.38, 14912.17, 14921.79]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, avg_lat, width, label="Avg Latency", color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, p99_lat, width, label="P99 Latency", color="#e74c3c", edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars1, avg_lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, p99_lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Batch Latency — GPU vs CPU (batch_size=256)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_latency_breakdown.png"), dpi=150)
    plt.close()
    print("  Saved: final_latency_breakdown.png")


def plot_projected_airgapped():
    """Projected performance on air-gapped 5-node cluster."""
    scenarios = ["POC\nSingle GPU\n(T4, 16GB)", "POC\nDistributed\n(2 nodes)", 
                 "Air-Gapped\nSingle GPU\n(24GB)", "Air-Gapped\nDistributed\n(5 GPUs, current)",
                 "Air-Gapped\nDistributed\n(5 GPUs, optimized)"]
    throughputs = [30087, 1880, 45000, 12000, 180000]
    colors = ["#3498db", "#3498db", "#2ecc71", "#2ecc71", "#2ecc71"]
    hatches = ["", "", "", "", "//"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(scenarios, throughputs, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    for bar, val in zip(bars, throughputs):
        label = f"{val:,}" if val < 100000 else f"{val/1000:.0f}K"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("Performance Projection — POC vs Air-Gapped 5-Node Cluster", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 210000)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color="#3498db", label="AWS POC (measured)"),
               mpatches.Patch(color="#2ecc71", label="Air-Gapped (projected)"),
               mpatches.Patch(facecolor="#2ecc71", hatch="//", label="With mapPartitions optimization")]
    ax.legend(handles=patches, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_projected_airgapped.png"), dpi=150)
    plt.close()
    print("  Saved: final_projected_airgapped.png")


if __name__ == "__main__":
    print("\n  Generating Final Report Graphs...")
    print("  " + "=" * 50)
    plot_mode_comparison()
    plot_scaling_throughput()
    plot_executor_distribution()
    plot_stage_timeline()
    plot_gpu_vs_cpu()
    plot_latency_breakdown()
    plot_projected_airgapped()
    print("  " + "=" * 50)
    print("  Done! 7 graphs generated.")
