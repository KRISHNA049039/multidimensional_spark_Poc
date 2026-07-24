"""
Unified Benchmark Runner — All Models × All Modes × Metrics

Runs the full multi-model inference benchmark:
  Mode 1: Distributed GPU (Spark RDD partitions)
  Mode 2: Single GPU (CUDA streams, all models parallel)
  Mode 3: Hybrid CPU+GPU (memory-aware placement)

Publishes metrics to results/metrics_report.md and results/raw_results.json.

Usage:
  python benchmark/run_benchmark.py
  python benchmark/run_benchmark.py --mode single_gpu
  python benchmark/run_benchmark.py --mode all --scale 10000
"""

import sys
import os
import time
import json
import platform
import argparse
from datetime import datetime
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_default_registry
from data.image_generator import generate_mixed_data
from inference.single_gpu import run_single_gpu_inference
from inference.hybrid_cpu_gpu import run_hybrid_inference


def get_system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_cores": os.cpu_count(),
        "cuda": torch.cuda.is_available(),
        "gpu_name": "N/A",
        "gpu_memory_gb": "N/A",
        "gpu_count": 0,
        "timestamp": datetime.now().isoformat(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
        info["gpu_count"] = torch.cuda.device_count()
    return info


def run_mode_single_gpu(models, data, batch_size=256) -> dict:
    """Run Mode 2: Single GPU with all models in parallel."""
    print("\n  [MODE 2] Single GPU — CUDA Streams...")
    result = run_single_gpu_inference(models, data, batch_size=batch_size)
    print(f"    Throughput: {result['total_throughput']:,.0f} total samples/sec")
    print(f"    Time: {result['elapsed_time']:.2f}s")
    return result


def run_mode_hybrid(models, registry, data, batch_size=256) -> dict:
    """Run Mode 3: Hybrid CPU+GPU."""
    print("\n  [MODE 3] Hybrid CPU+GPU — Memory-Aware Split...")
    model_sizes = {name: info.estimated_memory_mb
                   for name, info in registry.list_models().items()
                   if name in models}
    result = run_hybrid_inference(models, model_sizes, data, batch_size=batch_size)
    print(f"    Throughput: {result['total_throughput']:,.0f} total samples/sec")
    print(f"    GPU models: {result['gpu_models']}")
    print(f"    CPU models: {result['cpu_models']}")
    print(f"    Time: {result['elapsed_time']:.2f}s")
    return result


def run_mode_distributed(models, model_classes, data, num_partitions=4, batch_size=256) -> dict:
    """Run Mode 1: Distributed GPU via Spark."""
    print("\n  [MODE 1] Distributed GPU — Spark RDD...")
    try:
        from inference.distributed_gpu import create_spark_session, run_distributed_gpu_inference
        # Use cluster master if SPARK_MASTER env var is set (e.g. from spark-submit)
        spark = create_spark_session(num_cores="4")
        master_used = spark.sparkContext.master
        print(f"    Spark master: {master_used}")
        result = run_distributed_gpu_inference(
            spark, data, models, model_classes,
            num_partitions=num_partitions, batch_size=batch_size,
        )
        spark.stop()

        # Print detailed results
        print(f"\n{'─' * 70}")
        print(f"  DISTRIBUTED INFERENCE RESULTS")
        print(f"{'─' * 70}")
        print(f"  Total Throughput : {result['total_throughput']:,.0f} samples/sec")
        print(f"  Total Samples    : {result['total_samples_processed']:,}")
        print(f"  Elapsed Time     : {result['elapsed_time']:.2f}s")
        print(f"  Partitions       : {result['num_partitions']}")
        print(f"  Models           : {result['num_models']}")

        # Job description
        print(f"\n  JOB COMPOSITION:")
        print(f"  ─────────────────────────────────────────────────────────")
        print(f"  Job: Distributed inference of {result['num_models']} ML models")
        print(f"  Stages: 1 (parallelize → map → collect)")
        print(f"  Tasks: {result['num_partitions']} (one per partition)")
        print(f"  Each task: Load 10 models → run inference on data chunk → return results")

        # Executor / Worker details
        if result.get("partition_details"):
            print(f"\n  EXECUTOR / WORKER DETAIL:")
            print(f"  {'─' * 70}")
            print(f"  {'Part':<5} {'Hostname':<20} {'Device':<6} {'GPU':<20} "
                  f"{'ModelLoad':<10} {'Inference':<10} {'Samples'}")
            print(f"  {'─' * 70}")
            for pd in result["partition_details"]:
                ex = pd["executor"]
                tm = pd["timing"]
                print(f"  {pd['partition_idx']:<5} {ex['hostname']:<20} {ex['device']:<6} "
                      f"{ex['gpu_name'][:18]:<20} "
                      f"{tm['model_load_time_sec']:<10.2f} "
                      f"{tm['total_inference_time_sec']:<10.2f} "
                      f"{pd['total_samples_processed']}")
            print(f"  {'─' * 70}")

            # Unique executors summary
            executors = {}
            for pd in result["partition_details"]:
                eid = pd["executor"]["hostname"]
                if eid not in executors:
                    executors[eid] = {
                        "device": pd["executor"]["device"],
                        "gpu": pd["executor"]["gpu_name"],
                        "cpu_count": pd["executor"]["cpu_count"],
                        "partitions_handled": 0,
                        "total_samples": 0,
                    }
                executors[eid]["partitions_handled"] += 1
                executors[eid]["total_samples"] += pd["total_samples_processed"]

            print(f"\n  WORKER SUMMARY ({len(executors)} unique workers):")
            print(f"  {'─' * 70}")
            print(f"  {'Worker':<20} {'Device':<6} {'CPUs':<5} {'Tasks':<6} {'Samples':<10} {'GPU'}")
            print(f"  {'─' * 70}")
            for host, info in executors.items():
                print(f"  {host:<20} {info['device']:<6} {info['cpu_count']:<5} "
                      f"{info['partitions_handled']:<6} {info['total_samples']:<10,} "
                      f"{info['gpu']}")
            print(f"  {'─' * 70}")

        # Per-model input/output and throughput
        if result.get("partition_details"):
            # Get model info from first partition
            first_partition = result["partition_details"][0]
            print(f"\n  MODEL INPUT/OUTPUT & THROUGHPUT:")
            print(f"  {'─' * 70}")
            print(f"  {'Model':<22} {'Input Shape':<15} {'Output Shape':<15} "
                  f"{'Samples':<8} {'Throughput'}")
            print(f"  {'─' * 70}")
            for model_name, info in first_partition["per_model"].items():
                # Aggregate across all partitions
                total_samples = sum(
                    pd["per_model"].get(model_name, {}).get("samples_processed", 0)
                    for pd in result["partition_details"]
                )
                total_time = sum(
                    pd["per_model"].get(model_name, {}).get("inference_time_sec", 0)
                    for pd in result["partition_details"]
                )
                avg_throughput = total_samples / total_time if total_time > 0 else 0
                in_shape = str(info["input_shape"])
                out_shape = str(info["output_shape"])
                print(f"  {model_name:<22} {in_shape:<15} {out_shape:<15} "
                      f"{total_samples:<8,} {avg_throughput:,.0f}/s")
            print(f"  {'─' * 70}")

        # Spark UI stats
        if result.get("spark_ui_stats", {}).get("executors"):
            print(f"\n  SPARK EXECUTOR METRICS (from REST API):")
            print(f"  {'─' * 70}")
            print(f"  {'ID':<8} {'Host':<25} {'Cores':<6} {'Tasks':<6} {'Duration'}")
            print(f"  {'─' * 70}")
            for ex in result["spark_ui_stats"]["executors"]:
                if ex.get("id") == "driver":
                    continue
                host = ex.get("hostPort", "?").split(":")[0]
                print(f"  {ex.get('id','?'):<8} {host:<25} "
                      f"{ex.get('totalCores',0):<6} "
                      f"{ex.get('completedTasks',0):<6} "
                      f"{ex.get('totalDuration',0)/1000:.1f}s")
            print(f"  {'─' * 70}")

        if result.get("spark_ui_stats", {}).get("jobs"):
            print(f"\n  SPARK JOBS:")
            print(f"  {'─' * 70}")
            for job in result["spark_ui_stats"]["jobs"]:
                print(f"  Job {job.get('jobId',0)}: {job.get('status','?')} | "
                      f"Stages: {job.get('numCompletedStages',0)}/{job.get('numStages',0)} | "
                      f"Tasks: {job.get('numCompletedTasks',0)}/{job.get('numTasks',0)}")
            print(f"  {'─' * 70}")

        return result
    except Exception as e:
        print(f"    [ERROR] Spark mode failed: {e}")
        import traceback
        traceback.print_exc()
        return {"mode": "distributed_gpu", "error": str(e)}


def generate_report(results: dict, sys_info: dict) -> str:
    """Generate markdown metrics report."""
    md = []
    md.append("# Multi-Model Distributed Inference — Performance Metrics\n")
    md.append(f"**Generated:** {sys_info['timestamp']}\n")

    # System
    md.append("## System\n")
    md.append("| Component | Value |")
    md.append("|-----------|-------|")
    md.append(f"| Platform | {sys_info['platform']} |")
    md.append(f"| Python | {sys_info['python']} |")
    md.append(f"| PyTorch | {sys_info['torch']} |")
    md.append(f"| CPU Cores | {sys_info['cpu_cores']} |")
    md.append(f"| GPU | {sys_info['gpu_name']} ({sys_info['gpu_memory_gb']} GB) |")
    md.append(f"| GPU Count | {sys_info['gpu_count']} |")
    md.append(f"| CUDA | {sys_info['cuda']} |")
    md.append("")

    # Models
    md.append("## Models (10)\n")
    md.append("| # | Model | Category | Est. Memory |")
    md.append("|---|-------|----------|-------------|")
    if "models" in results:
        for i, (name, info) in enumerate(results["models"].items(), 1):
            md.append(f"| {i} | {name} | {info['category']} | {info['memory_mb']} MB |")
    md.append("")

    # Mode results
    md.append("## Inference Mode Comparison\n")
    md.append("| Mode | Total Throughput | Time (sec) | Models on GPU | Models on CPU |")
    md.append("|------|-----------------|------------|---------------|---------------|")

    for mode_key in ["single_gpu", "hybrid_cpu_gpu", "distributed_gpu"]:
        if mode_key in results:
            r = results[mode_key]
            if "error" in r:
                md.append(f"| {mode_key} | ERROR | - | - | - |")
            else:
                tp = r.get("total_throughput", 0)
                t = r.get("elapsed_time", 0)
                n_gpu = r.get("num_on_gpu", r.get("num_models", "-"))
                n_cpu = r.get("num_on_cpu", 0)
                md.append(f"| {mode_key} | {tp:,.0f} samples/sec | {t:.2f} | {n_gpu} | {n_cpu} |")
    md.append("")

    # Per-model throughput
    md.append("## Per-Model Processing (samples)\n")
    md.append("| Model | Single GPU | Hybrid | Distributed |")
    md.append("|-------|-----------|--------|-------------|")

    all_models = set()
    for mode_key in ["single_gpu", "hybrid_cpu_gpu", "distributed_gpu"]:
        if mode_key in results and "per_model_processed" in results.get(mode_key, {}):
            all_models.update(results[mode_key]["per_model_processed"].keys())

    for model_name in sorted(all_models):
        sg = results.get("single_gpu", {}).get("per_model_processed", {}).get(model_name, "-")
        hy = results.get("hybrid_cpu_gpu", {}).get("per_model_processed", {}).get(model_name, "-")
        dist = results.get("distributed_gpu", {}).get("per_model_processed", {}).get(model_name, "-")
        md.append(f"| {model_name} | {sg} | {hy} | {dist} |")
    md.append("")

    # Recommendations
    md.append("## Recommendations\n")
    md.append("- **Single GPU mode** is best for workstation inference with sufficient VRAM")
    md.append("- **Hybrid mode** is best when GPU memory is limited (models overflow to CPU)")
    md.append("- **Distributed mode** scales linearly with cluster size for production EW systems")
    md.append("- Enable **NVIDIA MPS** on cluster workers for true multi-process GPU sharing")
    md.append("- Use **CUDA streams** within each executor for intra-node model parallelism")
    md.append("")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Inference Benchmark")
    parser.add_argument("--mode", choices=["all", "single_gpu", "hybrid", "distributed"],
                        default="all", help="Which inference mode to benchmark")
    parser.add_argument("--signal-samples", type=int, default=5000,
                        help="Number of signal samples per model")
    parser.add_argument("--image-samples", type=int, default=200,
                        help="Number of image samples (224x224)")
    parser.add_argument("--detection-samples", type=int, default=50,
                        help="Number of detection images (640x640)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for inference")
    parser.add_argument("--partitions", type=int, default=2,
                        help="Spark partitions for distributed mode")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  MULTI-MODEL DISTRIBUTED INFERENCE BENCHMARK")
    print("=" * 70)

    sys_info = get_system_info()
    print(f"\n  System: {sys_info['platform']}")
    print(f"  Python: {sys_info['python']} | PyTorch: {sys_info['torch']}")
    print(f"  GPU: {sys_info['gpu_name']} | CUDA: {sys_info['cuda']}")
    print(f"  Signals: {args.signal_samples} | Images: {args.image_samples} | Detections: {args.detection_samples}")

    # --- Load models ---
    print("\n  [SETUP] Loading model registry...")
    registry = get_default_registry()
    registry.summary()

    print("\n  [SETUP] Instantiating models...")
    models = registry.load_all("cpu")
    model_classes = {name: info.model_class for name, info in registry.list_models().items()}
    print(f"    Loaded {len(models)} models")

    # --- Generate data ---
    print("\n  [SETUP] Generating synthetic data...")
    data = generate_mixed_data(
        num_signal_samples=args.signal_samples,
        num_image_samples=args.image_samples,
        num_detection_samples=args.detection_samples,
    )
    total_data = sum(arr.nbytes for arr in data.values())
    print(f"    Total data: {total_data / 1e6:.1f} MB")

    # --- Run benchmarks ---
    results = {
        "system_info": sys_info,
        "config": {
            "signal_samples": args.signal_samples,
            "image_samples": args.image_samples,
            "detection_samples": args.detection_samples,
            "batch_size": args.batch_size,
            "partitions": args.partitions,
        },
        "models": {name: {"category": info.category, "memory_mb": info.estimated_memory_mb}
                   for name, info in registry.list_models().items()},
    }

    start_total = time.time()

    if args.mode in ("all", "single_gpu"):
        results["single_gpu"] = run_mode_single_gpu(models, data, args.batch_size)

    if args.mode in ("all", "hybrid"):
        results["hybrid_cpu_gpu"] = run_mode_hybrid(models, registry, data, args.batch_size)

    if args.mode in ("all", "distributed"):
        results["distributed_gpu"] = run_mode_distributed(
            models, model_classes, data, args.partitions, args.batch_size
        )

    total_time = time.time() - start_total
    results["total_benchmark_time"] = round(total_time, 2)

    # --- Save results ---
    print(f"\n{'='*70}")
    print("  SAVING RESULTS")
    print(f"{'='*70}")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate unique filename with mode, config, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = os.environ.get("RUN_NAME", "")
    mode_tag = args.mode
    config_tag = f"sig{args.signal_samples}_img{args.image_samples}_det{args.detection_samples}_p{args.partitions}"

    if run_name:
        file_prefix = f"{run_name}_{mode_tag}_{config_tag}_{timestamp}"
    else:
        file_prefix = f"{mode_tag}_{config_tag}_{timestamp}"

    report_path = os.path.join(results_dir, f"report_{file_prefix}.md")
    report_md = generate_report(results, sys_info)
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"  Report: {report_path}")

    json_path = os.path.join(results_dir, f"results_{file_prefix}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Raw data: {json_path}")

    # Also write a latest symlink/copy for quick access
    latest_report = os.path.join(results_dir, "metrics_report_latest.md")
    latest_json = os.path.join(results_dir, "raw_results_latest.json")
    with open(latest_report, "w") as f:
        f.write(report_md)
    with open(latest_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Files saved:")
    print(f"    {report_path}")
    print(f"    {json_path}")
    print(f"    {latest_report} (latest)")
    print(f"    {latest_json} (latest)")
    print(f"\n  Total time: {total_time:.1f}s")
    print("")


if __name__ == "__main__":
    main()
