"""
Cluster Benchmark CLI — Run distributed inference tests from command line.

Supports 3 cluster device modes:
  --device-mode gpu_only    → All executors use GPU (cuda)
  --device-mode cpu_only    → All executors use CPU
  --device-mode hybrid      → Auto-detect (cuda where available, cpu elsewhere)

Submit incrementally increasing load via command line parameters.

Usage:
  # GPU cluster test (small)
  python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 2 --signal-samples 1000

  # CPU cluster test (medium)
  python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 5000

  # Hybrid cluster test (large)
  python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 10000

  # Full incremental test (all modes, increasing load)
  python benchmark/cluster_benchmark.py --incremental

Output:
  - Prints detailed statistics per executor, task, partition
  - Saves JSON results to results/cluster_benchmark_<mode>_<timestamp>.json
  - Explains what each metric means

What is measured:
  INPUT:  Synthetic EW signals (128-dim), images (3x224x224), detections (3x640x640)
  THROUGHPUT: total_samples / elapsed_time = samples processed per second
  OUTPUT: Per-model inference results (classification logits, embeddings, scores)
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_default_registry
from data.image_generator import generate_mixed_data
from inference.cluster_engine import create_cluster_session, run_cluster_inference


def print_header(device_mode, partitions, signal_samples, image_samples, detection_samples):
    print("\n" + "=" * 80)
    print("  SPARK CLUSTER BENCHMARK")
    print("=" * 80)
    print(f"""
  CONFIGURATION:
  ─────────────────────────────────────────────────────────────
  Device Mode     : {device_mode.upper()}
  Partitions      : {partitions}
  Signal Samples  : {signal_samples:,} (per model, 128-dim IQ vectors)
  Image Samples   : {image_samples:,} (3x224x224 RGB tensors)
  Detection Imgs  : {detection_samples:,} (3x640x640 RGB tensors)
  ─────────────────────────────────────────────────────────────

  WHAT THIS MEASURES:
  ─────────────────────────────────────────────────────────────
  INPUT:    Synthetic sensor data fed into 10 ML models simultaneously
  PROCESS:  Spark distributes data chunks to executors across the cluster
            Each executor loads models → runs batch inference → returns count
  OUTPUT:   Throughput (samples/sec) = how fast the cluster processes data
  ─────────────────────────────────────────────────────────────
""")


def print_results(result):
    """Print detailed results with explanations."""
    print(f"\n{'─' * 80}")
    print(f"  RESULTS — {result['device_mode'].upper()} MODE")
    print(f"{'─' * 80}")

    print(f"""
  THROUGHPUT:
    Total Samples Processed : {result['total_samples_processed']:,}
    Elapsed Time            : {result['elapsed_time']:.2f} seconds
    ► Throughput            : {result['total_throughput']:,.0f} samples/second

  EXPLANATION:
    "Throughput" = total samples processed by ALL models across ALL executors
    divided by wall-clock time. Higher = better cluster utilization.
""")

    # Per-partition detail
    print(f"  PARTITION / TASK DETAIL:")
    print(f"  {'─' * 76}")
    print(f"  {'Part':<5} {'Executor':<35} {'Device':<6} {'Samples':<8} "
          f"{'ModelLoad':<10} {'Inference':<10} {'Total':<8}")
    print(f"  {'─' * 76}")

    for p in result["partition_details"]:
        print(f"  {p['partition_idx']:<5} {p['hostname'][:30]:<35} {p['device']:<6} "
              f"{p['samples_processed']:<8} {p['model_load_time_sec']:<10.2f} "
              f"{p['inference_time_sec']:<10.2f} {p['total_task_time_sec']:<8.2f}")

    print(f"  {'─' * 76}")

    # Executor summary
    executors = {}
    for p in result["partition_details"]:
        eid = p["executor_id"]
        if eid not in executors:
            executors[eid] = {
                "hostname": p["hostname"], "device": p["device"],
                "tasks": 0, "samples": 0, "total_time": 0,
                "model_load_time": p["model_load_time_sec"],
            }
        executors[eid]["tasks"] += 1
        executors[eid]["samples"] += p["samples_processed"]
        executors[eid]["total_time"] += p["inference_time_sec"]

    print(f"\n  EXECUTOR SUMMARY:")
    print(f"  {'─' * 76}")
    print(f"  {'Executor':<35} {'Device':<6} {'Tasks':<6} {'Samples':<10} "
          f"{'ModelLoad':<10} {'InferTime':<10} {'Throughput'}")
    print(f"  {'─' * 76}")

    for eid, ex in executors.items():
        tp = ex["samples"] / ex["total_time"] if ex["total_time"] > 0 else 0
        print(f"  {ex['hostname'][:30]:<35} {ex['device']:<6} {ex['tasks']:<6} "
              f"{ex['samples']:<10,} {ex['model_load_time']:<10.2f} "
              f"{ex['total_time']:<10.2f} {tp:,.0f}/sec")

    print(f"  {'─' * 76}")

    # Per-model breakdown
    print(f"\n  PER-MODEL SAMPLES PROCESSED:")
    print(f"  {'─' * 40}")
    for name, count in sorted(result["per_model_processed"].items()):
        print(f"    {name:<25} : {count:,}")
    print(f"  {'─' * 40}")
    print(f"    {'TOTAL':<25} : {result['total_samples_processed']:,}")

    # Spark UI stats
    if result.get("spark_ui_stats", {}).get("executors"):
        print(f"\n  SPARK EXECUTOR METRICS (from REST API):")
        print(f"  {'─' * 76}")
        print(f"  {'ID':<5} {'Host':<35} {'Cores':<6} {'Tasks':<6} "
              f"{'Duration':<10} {'GC Time':<8} {'Memory'}")
        print(f"  {'─' * 76}")
        for ex in result["spark_ui_stats"]["executors"]:
            if ex["id"] == "driver":
                continue
            host = ex.get("hostPort", "?").split(":")[0]
            print(f"  {ex['id']:<5} {host:<35} {ex.get('totalCores',0):<6} "
                  f"{ex.get('completedTasks',0):<6} "
                  f"{ex.get('totalDuration',0)/1000:<10.1f}s "
                  f"{ex.get('totalGCTime',0):<8}ms "
                  f"{ex.get('memoryUsed',0)/1e6:<.0f}MB")
        print(f"  {'─' * 76}")

    print(f"""
  KEY METRICS EXPLAINED:
  ─────────────────────────────────────────────────────────────
  ModelLoad  : Time to deserialize 10 model weights onto device (once per executor)
  Inference  : Time to run forward passes on all data batches
  Throughput : samples_processed / inference_time (per executor)
  Partitions : Data split into N chunks, 1 task per chunk
  Tasks      : Spark units of work, distributed across executors
  ─────────────────────────────────────────────────────────────
""")


def run_single_test(args):
    """Run a single benchmark with given parameters."""
    print_header(args.device_mode, args.partitions, args.signal_samples,
                 args.image_samples, args.detection_samples)

    # Load models
    print("  [1/4] Loading model registry...")
    registry = get_default_registry()
    models = registry.load_all("cpu")
    print(f"         Loaded {len(models)} models")

    # Generate data
    print(f"  [2/4] Generating synthetic data...")
    data = generate_mixed_data(
        num_signal_samples=args.signal_samples,
        num_image_samples=args.image_samples,
        num_detection_samples=args.detection_samples,
    )
    total_data_mb = sum(arr.nbytes for arr in data.values()) / 1e6
    total_samples = sum(len(arr) for arr in data.values())
    print(f"         {total_samples:,} samples, {total_data_mb:.1f} MB")
    print(f"         Per-partition: ~{total_data_mb/args.partitions:.0f} MB")

    # Create Spark session
    print(f"  [3/4] Connecting to Spark cluster...")
    spark = create_cluster_session(
        app_name=f"ClusterBench_{args.device_mode}_{args.signal_samples}",
    )
    print(f"         Master: {spark.sparkContext.master}")

    # Run inference
    print(f"  [4/4] Running distributed inference ({args.device_mode} mode)...")
    result = run_cluster_inference(
        spark, data, models,
        num_partitions=args.partitions,
        batch_size=args.batch_size,
        device_mode=args.device_mode,
    )

    spark.stop()

    # Print results
    print_results(result)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cluster_benchmark_{args.device_mode}_{args.signal_samples}_{timestamp}.json"
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Results saved: {filepath}")

    return result


def run_incremental(args):
    """Run incremental tests across all 3 modes with increasing load."""
    configs = [
        # (device_mode, partitions, signals, images, detections)
        ("gpu_only", 2, 1000, 50, 20),
        ("cpu_only", 2, 1000, 50, 20),
        ("hybrid", 2, 1000, 50, 20),
        ("gpu_only", 2, 3000, 100, 30),
        ("cpu_only", 2, 3000, 100, 30),
        ("hybrid", 2, 3000, 100, 30),
        ("gpu_only", 4, 5000, 200, 50),
        ("cpu_only", 4, 5000, 200, 50),
        ("hybrid", 4, 5000, 200, 50),
    ]

    all_results = []

    for i, (mode, parts, sigs, imgs, dets) in enumerate(configs):
        print(f"\n{'=' * 80}")
        print(f"  INCREMENTAL RUN {i+1}/{len(configs)}")
        print(f"  Mode={mode}, Partitions={parts}, Signals={sigs}")
        print(f"{'=' * 80}")

        args.device_mode = mode
        args.partitions = parts
        args.signal_samples = sigs
        args.image_samples = imgs
        args.detection_samples = dets

        try:
            result = run_single_test(args)
            result["run_number"] = i + 1
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({
                "run_number": i + 1,
                "device_mode": mode,
                "status": "failed",
                "error": str(e),
            })

        time.sleep(3)  # Pause between runs

    # Save all results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    filepath = os.path.join(results_dir,
                            f"incremental_all_modes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("  INCREMENTAL TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'#':<3} {'Mode':<12} {'Parts':<6} {'Samples':<10} {'Throughput':<15} "
          f"{'Time':<8} {'Device'}")
    print(f"  {'─'*3} {'─'*12} {'─'*6} {'─'*10} {'─'*15} {'─'*8} {'─'*8}")

    for r in all_results:
        if "error" in r:
            print(f"  {r.get('run_number','?'):<3} {r['device_mode']:<12} "
                  f"{'—':<6} {'—':<10} {'FAILED':<15} {'—':<8} {'—'}")
        else:
            devices = set(p["device"] for p in r.get("partition_details", []))
            print(f"  {r['run_number']:<3} {r['device_mode']:<12} "
                  f"{r['num_partitions']:<6} {r['total_samples_processed']:<10,} "
                  f"{r['total_throughput']:<15,.0f} {r['elapsed_time']:<8.2f} "
                  f"{','.join(devices)}")

    print(f"\n  Results saved: {filepath}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Spark Cluster Benchmark — Test GPU/CPU/Hybrid distributed inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick GPU test
  python benchmark/cluster_benchmark.py --device-mode gpu_only --signal-samples 1000

  # CPU baseline
  python benchmark/cluster_benchmark.py --device-mode cpu_only --signal-samples 5000

  # Hybrid (auto-detect GPU)
  python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 5000

  # Full incremental test (all modes × 3 load levels)
  python benchmark/cluster_benchmark.py --incremental
        """)

    parser.add_argument("--device-mode", choices=["gpu_only", "cpu_only", "hybrid"],
                        default="hybrid",
                        help="Cluster device mode: gpu_only, cpu_only, or hybrid")
    parser.add_argument("--partitions", type=int, default=2,
                        help="Number of data partitions (= number of tasks)")
    parser.add_argument("--signal-samples", type=int, default=5000,
                        help="Signal samples per model (128-dim IQ vectors)")
    parser.add_argument("--image-samples", type=int, default=200,
                        help="Image classification samples (3x224x224)")
    parser.add_argument("--detection-samples", type=int, default=50,
                        help="Object detection samples (3x640x640)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for inference")
    parser.add_argument("--incremental", action="store_true",
                        help="Run all 3 modes with increasing load (ignores other params)")

    args = parser.parse_args()

    if args.incremental:
        run_incremental(args)
    else:
        run_single_test(args)


if __name__ == "__main__":
    main()
