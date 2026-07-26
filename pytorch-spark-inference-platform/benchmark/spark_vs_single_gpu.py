"""
SPARK vs SINGLE GPU — Scalability Proof Benchmark
==================================================

Purpose: Prove to leadership that Spark cluster computing outperforms
single-GPU inference for multi-model workloads at scale.

Tests:
  1. BASELINE: 10 models on single GPU (sequential, no Spark)
  2. BASELINE: 10 models on single GPU (CUDA streams, parallel)
  3. SPARK: 1 worker, 1 executor (minimal cluster)
  4. SPARK: 1 worker, 2 executors
  5. SPARK: 2 workers (horizontal scaling)
  6. SPARK: varying executor memory (2g, 4g, 8g)
  7. SPARK: varying partitions (2, 4, 8, 16)
  8. SPARK: CPU+GPU hybrid cluster

Output: JSON results + comparison table showing crossover point
where Spark beats single GPU.

Usage (on GPU instance):
  SPARK_MASTER_URL=spark://localhost:7077 python benchmark/spark_vs_single_gpu.py
  python benchmark/spark_vs_single_gpu.py --local  # single-GPU only tests
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_default_registry
from data.image_generator import generate_mixed_data


# =============================================================================
# TEST 1: Single GPU Baseline (No Spark — just PyTorch)
# =============================================================================

def run_single_gpu_sequential(models, data, batch_size=256):
    """
    Baseline: Run all 10 models sequentially on a single GPU.
    This is how most teams do inference without Spark.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  [SINGLE-GPU SEQUENTIAL] Device: {device}")

    # Move all models to device
    for name, model in models.items():
        models[name] = model.to(device).eval()

    results = {}
    total_start = time.time()

    for model_name, model in models.items():
        if model_name not in data:
            continue
        arr = data[model_name]
        n = len(arr)
        model_start = time.time()
        processed = 0

        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = torch.from_numpy(arr[i:i+batch_size]).float().to(device)
                output = model(batch)
                processed += len(batch)

                # Capture sample I/O for first batch
                if i == 0:
                    results[model_name] = {
                        "input_shape": list(batch.shape),
                        "input_sample": batch[0].flatten()[:8].cpu().tolist(),
                        "output_shape": list(output.shape),
                        "output_sample": output[0].flatten()[:10].cpu().tolist(),
                    }

        model_elapsed = time.time() - model_start
        results[model_name]["samples"] = processed
        results[model_name]["time_sec"] = round(model_elapsed, 4)
        results[model_name]["throughput"] = round(processed / model_elapsed, 1)

    total_elapsed = time.time() - total_start
    total_samples = sum(r["samples"] for r in results.values())

    return {
        "mode": "single_gpu_sequential",
        "device": device,
        "elapsed_time": round(total_elapsed, 4),
        "total_samples": total_samples,
        "total_throughput": round(total_samples / total_elapsed, 1),
        "per_model": results,
    }


def run_single_gpu_parallel_streams(models, data, batch_size=256):
    """
    Optimized single GPU: use CUDA streams to overlap model inference.
    Still single GPU, but models run concurrently via streams.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  [SINGLE-GPU PARALLEL STREAMS] Device: {device}")

    if device == "cpu":
        print("    No CUDA — falling back to sequential")
        return run_single_gpu_sequential(models, data, batch_size)

    for name, model in models.items():
        models[name] = model.to(device).eval()

    # Create a CUDA stream per model
    streams = {name: torch.cuda.Stream() for name in models}
    results = {}
    total_start = time.time()

    # Launch all models in parallel streams
    for model_name, model in models.items():
        if model_name not in data:
            continue
        arr = data[model_name]
        stream = streams[model_name]

        with torch.cuda.stream(stream):
            n = len(arr)
            processed = 0
            model_start = time.time()

            with torch.no_grad():
                for i in range(0, n, batch_size):
                    batch = torch.from_numpy(arr[i:i+batch_size]).float().to(device)
                    output = model(batch)
                    processed += len(batch)

                    if i == 0:
                        results[model_name] = {
                            "input_shape": list(batch.shape),
                            "input_sample": batch[0].flatten()[:8].cpu().tolist(),
                            "output_shape": list(output.shape),
                            "output_sample": output[0].flatten()[:10].cpu().tolist(),
                        }

            model_elapsed = time.time() - model_start
            results[model_name]["samples"] = processed
            results[model_name]["time_sec"] = round(model_elapsed, 4)
            results[model_name]["throughput"] = round(processed / model_elapsed, 1)

    # Synchronize all streams
    torch.cuda.synchronize()
    total_elapsed = time.time() - total_start
    total_samples = sum(r["samples"] for r in results.values())

    return {
        "mode": "single_gpu_parallel_streams",
        "device": device,
        "elapsed_time": round(total_elapsed, 4),
        "total_samples": total_samples,
        "total_throughput": round(total_samples / total_elapsed, 1),
        "per_model": results,
    }


# =============================================================================
# TEST 2: Spark Cluster Configurations
# =============================================================================

def run_spark_config(models, model_classes, data, config, label):
    """Run Spark distributed inference with specific configuration."""
    from inference.distributed_gpu import create_spark_session, run_distributed_gpu_inference

    print(f"\n  [SPARK: {label}]")
    print(f"    Config: partitions={config['partitions']}, batch_size={config['batch_size']}")

    spark = create_spark_session(num_cores=str(config.get("cores", "4")))
    print(f"    Master: {spark.sparkContext.master}")

    start = time.time()
    result = run_distributed_gpu_inference(
        spark, data, models, model_classes,
        num_partitions=config["partitions"],
        batch_size=config["batch_size"],
    )
    elapsed = time.time() - start
    spark.stop()

    # Add I/O samples from partition details
    io_samples = {}
    if result.get("partition_details"):
        for pm_name, pm_info in result["partition_details"][0].get("per_model", {}).items():
            io_samples[pm_name] = {
                "input_shape": pm_info.get("input_shape"),
                "output_shape": pm_info.get("output_shape"),
                "samples": pm_info.get("samples_processed", 0),
                "throughput": pm_info.get("throughput", 0),
            }

    return {
        "mode": f"spark_{label}",
        "config": config,
        "elapsed_time": round(elapsed, 4),
        "total_samples": result.get("total_samples_processed", 0),
        "total_throughput": result.get("total_throughput", 0),
        "per_model_io": io_samples,
        "partition_count": config["partitions"],
        "partition_details": result.get("partition_details", []),
    }


# =============================================================================
# TEST 3: Data Scaling — Find the crossover point
# =============================================================================

def run_scaling_comparison(models, model_classes, batch_size=256):
    """
    Run both single-GPU and Spark at increasing data volumes.
    Find where Spark overtakes single GPU.
    """
    scales = [
        {"signals": 1000, "images": 50, "detections": 20, "label": "1K"},
        {"signals": 5000, "images": 200, "detections": 50, "label": "5K"},
        {"signals": 10000, "images": 400, "detections": 100, "label": "10K"},
        {"signals": 20000, "images": 800, "detections": 200, "label": "20K"},
        {"signals": 50000, "images": 1000, "detections": 300, "label": "50K"},
    ]

    results = []

    for scale in scales:
        print(f"\n{'='*60}")
        print(f"  SCALE TEST: {scale['label']} signals")
        print(f"{'='*60}")

        data = generate_mixed_data(
            num_signal_samples=scale["signals"],
            num_image_samples=scale["images"],
            num_detection_samples=scale["detections"],
        )
        total_data_mb = sum(arr.nbytes for arr in data.values()) / 1e6
        print(f"  Data: {total_data_mb:.1f} MB")

        # Single GPU
        single_result = run_single_gpu_sequential(models, data, batch_size)

        # Spark (4 partitions)
        spark_result = None
        try:
            spark_result = run_spark_config(
                models, model_classes, data,
                {"partitions": 4, "batch_size": batch_size, "cores": "4"},
                f"4-part_{scale['label']}"
            )
        except Exception as e:
            print(f"    Spark failed: {e}")

        entry = {
            "scale": scale["label"],
            "signals": scale["signals"],
            "data_mb": round(total_data_mb, 1),
            "single_gpu_throughput": single_result["total_throughput"],
            "single_gpu_time": single_result["elapsed_time"],
            "spark_throughput": spark_result["total_throughput"] if spark_result else 0,
            "spark_time": spark_result["elapsed_time"] if spark_result else 0,
            "speedup": round(
                (spark_result["total_throughput"] / single_result["total_throughput"]), 2
            ) if spark_result and single_result["total_throughput"] > 0 else 0,
        }
        results.append(entry)

        # Print comparison
        print(f"\n  COMPARISON @ {scale['label']}:")
        print(f"    Single GPU: {single_result['total_throughput']:,.0f} samples/sec ({single_result['elapsed_time']:.2f}s)")
        if spark_result:
            print(f"    Spark (4p): {spark_result['total_throughput']:,.0f} samples/sec ({spark_result['elapsed_time']:.2f}s)")
            print(f"    Speedup:   {entry['speedup']}×")

    return results


# =============================================================================
# MAIN
# =============================================================================

def print_io_table(result):
    """Print input/output table for each model."""
    print(f"\n  {'─'*75}")
    print(f"  MODEL INPUT/OUTPUT DETAILS")
    print(f"  {'─'*75}")
    print(f"  {'Model':<22} {'Input Shape':<18} {'Output Shape':<18} {'Throughput':<12} {'Device'}")
    print(f"  {'─'*75}")

    per_model = result.get("per_model", result.get("per_model_io", {}))
    device = result.get("device", "cluster")

    for name, info in per_model.items():
        in_shape = str(info.get("input_shape", "?"))
        out_shape = str(info.get("output_shape", "?"))
        tp = info.get("throughput", 0)
        print(f"  {name:<22} {in_shape:<18} {out_shape:<18} {tp:>8,.0f}/s   {device}")

    print(f"  {'─'*75}")


def print_comparison_table(all_results):
    """Print final comparison table."""
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON: SINGLE GPU vs SPARK CLUSTER")
    print(f"{'='*80}")
    print(f"\n  {'Config':<35} {'Throughput':<15} {'Time':<10} {'Speedup':<10} {'Verdict'}")
    print(f"  {'─'*35} {'─'*15} {'─'*10} {'─'*10} {'─'*12}")

    baseline = all_results[0]["total_throughput"] if all_results else 1

    for r in all_results:
        mode = r.get("mode", "?")
        tp = r.get("total_throughput", 0)
        elapsed = r.get("elapsed_time", 0)
        speedup = round(tp / baseline, 2) if baseline > 0 else 0
        verdict = "★ BEST" if tp == max(x["total_throughput"] for x in all_results) else ""
        print(f"  {mode:<35} {tp:>10,.0f}/s   {elapsed:>6.2f}s   {speedup:>5.2f}×   {verdict}")

    print(f"  {'─'*80}")


def main():
    parser = argparse.ArgumentParser(description="Spark vs Single GPU — Scalability Proof")
    parser.add_argument("--local", action="store_true", help="Run only single-GPU tests (no Spark)")
    parser.add_argument("--signals", type=int, default=5000, help="Signal samples")
    parser.add_argument("--images", type=int, default=200, help="Image samples")
    parser.add_argument("--detections", type=int, default=50, help="Detection samples")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--scaling-test", action="store_true", help="Run full scaling comparison")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("  SPARK vs SINGLE GPU — SCALABILITY PROOF BENCHMARK")
    print("="*80)
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  CPU cores: {os.cpu_count()}")

    # Load models
    print("\n  Loading 10 models...")
    registry = get_default_registry()
    models = registry.load_all("cpu")
    model_classes = {name: info.model_class for name, info in registry.list_models().items()}
    print(f"  Loaded {len(models)} models")

    # Generate data
    print(f"  Generating data: {args.signals} signals, {args.images} images, {args.detections} detections")
    data = generate_mixed_data(
        num_signal_samples=args.signals,
        num_image_samples=args.images,
        num_detection_samples=args.detections,
    )
    total_mb = sum(arr.nbytes for arr in data.values()) / 1e6
    print(f"  Total data: {total_mb:.1f} MB")

    all_results = []

    # ─── Single GPU Tests ───
    print("\n" + "="*80)
    print("  PHASE 1: SINGLE GPU BASELINES (No Spark)")
    print("="*80)

    r1 = run_single_gpu_sequential(models, data, args.batch_size)
    print_io_table(r1)
    all_results.append(r1)

    r2 = run_single_gpu_parallel_streams(models, data, args.batch_size)
    all_results.append(r2)

    if args.local:
        print_comparison_table(all_results)
        return

    # ─── Spark Tests ───
    print("\n" + "="*80)
    print("  PHASE 2: SPARK CLUSTER CONFIGURATIONS")
    print("="*80)

    spark_configs = [
        {"partitions": 2, "batch_size": 256, "cores": "4", "label": "2-partitions"},
        {"partitions": 4, "batch_size": 256, "cores": "4", "label": "4-partitions"},
        {"partitions": 8, "batch_size": 256, "cores": "4", "label": "8-partitions"},
        {"partitions": 4, "batch_size": 64, "cores": "4", "label": "4p-batch64"},
        {"partitions": 4, "batch_size": 512, "cores": "4", "label": "4p-batch512"},
    ]

    for cfg in spark_configs:
        try:
            r = run_spark_config(models, model_classes, data, cfg, cfg["label"])
            all_results.append(r)
        except Exception as e:
            print(f"    FAILED: {e}")

    # ─── Scaling Test ───
    if args.scaling_test:
        print("\n" + "="*80)
        print("  PHASE 3: SCALING COMPARISON (find crossover point)")
        print("="*80)
        scaling_results = run_scaling_comparison(models, model_classes, args.batch_size)

        print(f"\n  {'─'*60}")
        print(f"  CROSSOVER ANALYSIS")
        print(f"  {'─'*60}")
        print(f"  {'Scale':<8} {'Single GPU':<15} {'Spark':<15} {'Speedup':<10} {'Winner'}")
        print(f"  {'─'*8} {'─'*15} {'─'*15} {'─'*10} {'─'*10}")
        for s in scaling_results:
            winner = "SPARK ★" if s["speedup"] > 1.0 else "Single GPU"
            print(f"  {s['scale']:<8} {s['single_gpu_throughput']:>10,.0f}/s   "
                  f"{s['spark_throughput']:>10,.0f}/s   {s['speedup']:>6.2f}×   {winner}")

    # ─── Final Comparison ───
    print_comparison_table(all_results)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"spark_vs_single_gpu_{ts}.json")

    output = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "cpu_cores": os.cpu_count(),
        },
        "config": {
            "signals": args.signals,
            "images": args.images,
            "detections": args.detections,
            "batch_size": args.batch_size,
        },
        "results": all_results,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {filepath}")


if __name__ == "__main__":
    main()
