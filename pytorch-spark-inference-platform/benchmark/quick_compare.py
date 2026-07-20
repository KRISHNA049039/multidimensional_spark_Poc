"""
Quick 3-Mode Comparison — Minimal load, captures stats fast.

Runs gpu_only and hybrid modes with SMALL data to get first-cut statistics quickly.

Usage (from master container):
    SPARK_MASTER_URL=spark://192.168.4.100:7077 python benchmark/quick_compare.py

Saves results + Spark UI stats to results/quick_compare_<timestamp>.json
"""

import sys
import os
import time
import json
import platform
import urllib.request
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── CONFIG (keep small for speed) ───────────────────────────────────────────
SIGNAL_SAMPLES = 200
IMAGE_SAMPLES = 10
DETECTION_SAMPLES = 5
BATCH_SIZE = 64
PARTITIONS = 2
# ─────────────────────────────────────────────────────────────────────────────


def get_system_info():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cpu_cores": os.cpu_count(),
        "timestamp": datetime.now().isoformat(),
    }


def capture_spark_ui():
    """Capture whatever is available from Spark UI."""
    stats = {}
    try:
        apps = json.loads(urllib.request.urlopen("http://localhost:4040/api/v1/applications", timeout=3).read())
        if apps:
            app_id = apps[0]["id"]
            stats["app"] = apps[0]
            try:
                stats["jobs"] = json.loads(urllib.request.urlopen(
                    f"http://localhost:4040/api/v1/applications/{app_id}/jobs", timeout=3).read())
            except:
                pass
            try:
                stats["stages"] = json.loads(urllib.request.urlopen(
                    f"http://localhost:4040/api/v1/applications/{app_id}/stages", timeout=3).read())
            except:
                pass
            try:
                stats["executors"] = json.loads(urllib.request.urlopen(
                    f"http://localhost:4040/api/v1/applications/{app_id}/allexecutors", timeout=3).read())
            except:
                pass
    except:
        stats["note"] = "Spark UI not available (app may have exited)"
    return stats


def run_distributed_mode(spark, data, models, model_classes, device_mode):
    """Run distributed inference with specified device mode."""
    from inference.distributed_gpu import run_distributed_gpu_inference

    start = time.time()
    result = run_distributed_gpu_inference(
        spark, data, models, model_classes,
        num_partitions=PARTITIONS,
        batch_size=BATCH_SIZE,
    )
    elapsed = time.time() - start
    result["wall_time"] = round(elapsed, 2)
    result["device_mode_requested"] = device_mode
    return result


def main():
    print("\n" + "=" * 70)
    print("  QUICK 3-MODE COMPARISON (minimal load)")
    print("=" * 70)

    sys_info = get_system_info()
    print(f"  Platform: {sys_info['platform']}")
    print(f"  CUDA: {sys_info['cuda_available']} | GPU: {sys_info['gpu']}")
    print(f"  Config: signals={SIGNAL_SAMPLES}, images={IMAGE_SAMPLES}, "
          f"detections={DETECTION_SAMPLES}, partitions={PARTITIONS}")
    print(f"  This should take 1-2 minutes total.\n")

    # ─── Load models ─────────────────────────────────────────────────────────
    print("  [1/5] Loading models...")
    t0 = time.time()
    from models import get_default_registry
    from data.image_generator import generate_mixed_data

    registry = get_default_registry()
    models = registry.load_all("cpu")
    model_classes = {name: info.model_class for name, info in registry.list_models().items()}
    model_load_time = time.time() - t0
    print(f"         {len(models)} models loaded in {model_load_time:.1f}s")

    # ─── Generate data ───────────────────────────────────────────────────────
    print("  [2/5] Generating data...")
    data = generate_mixed_data(
        num_signal_samples=SIGNAL_SAMPLES,
        num_image_samples=IMAGE_SAMPLES,
        num_detection_samples=DETECTION_SAMPLES,
    )
    total_samples = sum(len(arr) for arr in data.values())
    total_mb = sum(arr.nbytes for arr in data.values()) / 1e6
    print(f"         {total_samples} samples, {total_mb:.1f} MB")

    # ─── Create Spark session ────────────────────────────────────────────────
    print("  [3/5] Connecting to Spark...")
    from inference.distributed_gpu import create_spark_session
    spark = create_spark_session(num_cores="4")
    print(f"         Master: {spark.sparkContext.master}")

    all_results = {
        "system_info": sys_info,
        "config": {
            "signal_samples": SIGNAL_SAMPLES,
            "image_samples": IMAGE_SAMPLES,
            "detection_samples": DETECTION_SAMPLES,
            "batch_size": BATCH_SIZE,
            "partitions": PARTITIONS,
            "total_samples": total_samples,
            "total_data_mb": round(total_mb, 1),
        },
        "model_load_time_sec": round(model_load_time, 2),
        "modes": {},
    }

    # ─── Mode 1: GPU Only ────────────────────────────────────────────────────
    print("\n  [4/5] Running GPU_ONLY mode...")
    if torch.cuda.is_available():
        os.environ["FORCE_DEVICE"] = "cuda"
    else:
        os.environ["FORCE_DEVICE"] = "cpu"
        print("         (No GPU — running on CPU as fallback, labeled gpu_only)")
    try:
        result_gpu = run_distributed_mode(spark, data, models, model_classes, "gpu_only")
        all_results["modes"]["gpu_only"] = {
            "throughput": result_gpu.get("total_throughput", 0),
            "elapsed_time": result_gpu.get("elapsed_time", 0),
            "wall_time": result_gpu.get("wall_time", 0),
            "total_processed": result_gpu.get("total_samples_processed", 0),
            "partitions": PARTITIONS,
            "per_model": result_gpu.get("per_model_processed", {}),
            "actual_device": "cuda" if torch.cuda.is_available() else "cpu (no GPU)",
            "status": "success",
        }
        print(f"         Throughput: {result_gpu.get('total_throughput', 0):,.0f} samples/sec")
        print(f"         Time: {result_gpu.get('elapsed_time', 0):.2f}s")
    except Exception as e:
        all_results["modes"]["gpu_only"] = {"status": "failed", "error": str(e)}
        print(f"         FAILED: {e}")

    # ─── Mode 3: Hybrid ──────────────────────────────────────────────────────
    print("\n  [5/5] Running HYBRID mode...")
    os.environ.pop("FORCE_DEVICE", None)
    try:
        result_hybrid = run_distributed_mode(spark, data, models, model_classes, "hybrid")
        all_results["modes"]["hybrid"] = {
            "throughput": result_hybrid.get("total_throughput", 0),
            "elapsed_time": result_hybrid.get("elapsed_time", 0),
            "wall_time": result_hybrid.get("wall_time", 0),
            "total_processed": result_hybrid.get("total_samples_processed", 0),
            "partitions": PARTITIONS,
            "per_model": result_hybrid.get("per_model_processed", {}),
            "status": "success",
        }
        print(f"         Throughput: {result_hybrid.get('total_throughput', 0):,.0f} samples/sec")
        print(f"         Time: {result_hybrid.get('elapsed_time', 0):.2f}s")
    except Exception as e:
        all_results["modes"]["hybrid"] = {"status": "failed", "error": str(e)}
        print(f"         FAILED: {e}")

    # ─── Capture Spark UI ────────────────────────────────────────────────────
    print("\n  Capturing Spark UI stats...")
    all_results["spark_ui"] = capture_spark_ui()

    spark.stop()

    # ─── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Mode':<12} {'Throughput':<20} {'Time':<10} {'Samples':<12} {'Status'}")
    print(f"  {'─'*12} {'─'*20} {'─'*10} {'─'*12} {'─'*8}")

    for mode, r in all_results["modes"].items():
        if r["status"] == "success":
            print(f"  {mode:<12} {r['throughput']:>15,.0f}/s {r['elapsed_time']:>8.2f}s "
                  f"{r['total_processed']:>10,} OK")
        else:
            print(f"  {mode:<12} {'—':>15} {'—':>8} {'—':>10} FAILED")

    print(f"\n  Model load time: {model_load_time:.1f}s")
    print(f"  Data: {total_samples} samples, {total_mb:.1f} MB")

    # ─── Save ────────────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"quick_compare_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {filepath}")
    print("")


if __name__ == "__main__":
    main()
