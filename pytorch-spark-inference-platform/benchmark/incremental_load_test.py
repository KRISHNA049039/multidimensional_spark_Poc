"""
Incremental Load Test — Scales data size and observes Spark distribution.

Runs the distributed benchmark with increasing data sizes to observe:
- How throughput scales with data volume
- How Spark distributes work across executors
- When distributed mode becomes advantageous over single-GPU

Usage (from master container):
    SPARK_MASTER_URL=spark://10.0.0.187:7077 python benchmark/incremental_load_test.py

Each run produces a JSON result that accumulates into results/incremental_results.json.
Watch the Spark UI at http://<master-ip>:8080 while this runs to see task distribution.
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_default_registry
from data.image_generator import generate_mixed_data
from inference.distributed_gpu import create_spark_session, run_distributed_gpu_inference


def run_incremental_test():
    print("\n" + "=" * 70)
    print("  INCREMENTAL LOAD TEST — Distributed Spark Mode")
    print("=" * 70)

    # Load configurations: smaller sizes that fit within m5.xlarge memory
    # All runs < 1GB total to avoid OOM, observe scaling patterns
    load_configs = [
        {"signals": 500, "images": 20, "detections": 10, "partitions": 2},
        {"signals": 1000, "images": 50, "detections": 20, "partitions": 2},
        {"signals": 2000, "images": 100, "detections": 30, "partitions": 2},
        {"signals": 5000, "images": 200, "detections": 50, "partitions": 4},
        {"signals": 10000, "images": 400, "detections": 80, "partitions": 4},
    ]

    # Load models once
    print("\n  [SETUP] Loading models...")
    registry = get_default_registry()
    models = registry.load_all("cpu")
    model_classes = {name: info.model_class for name, info in registry.list_models().items()}
    print(f"    Loaded {len(models)} models")

    # Create Spark session (uses SPARK_MASTER_URL env var for cluster mode)
    print("\n  [SETUP] Creating Spark session...")
    spark = create_spark_session(num_cores="4")
    master_url = spark.sparkContext.master
    print(f"    Connected to: {master_url}")
    print(f"    Open Spark UI to watch task distribution!")
    print(f"    → http://localhost:8080 (or http://<master-public-ip>:8080)")

    all_results = []

    for i, config in enumerate(load_configs):
        print(f"\n{'─' * 70}")
        print(f"  RUN {i+1}/{len(load_configs)}: "
              f"signals={config['signals']}, images={config['images']}, "
              f"detections={config['detections']}, partitions={config['partitions']}")
        print(f"{'─' * 70}")

        # Generate data for this run
        data = generate_mixed_data(
            num_signal_samples=config["signals"],
            num_image_samples=config["images"],
            num_detection_samples=config["detections"],
        )
        total_samples = sum(len(arr) for arr in data.values())
        total_data_mb = sum(arr.nbytes for arr in data.values()) / 1e6
        print(f"    Data: {total_samples:,} samples, {total_data_mb:.1f} MB")

        # Run distributed inference
        start = time.time()
        try:
            result = run_distributed_gpu_inference(
                spark, data, models, model_classes,
                num_partitions=config["partitions"],
                batch_size=256,
            )
            elapsed = time.time() - start

            run_result = {
                "run": i + 1,
                "config": config,
                "total_samples": total_samples,
                "total_data_mb": round(total_data_mb, 1),
                "throughput": result["total_throughput"],
                "elapsed_time": result["elapsed_time"],
                "total_time_with_setup": round(elapsed, 2),
                "partitions": config["partitions"],
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }

            print(f"    Throughput: {result['total_throughput']:,.0f} samples/sec")
            print(f"    Elapsed: {result['elapsed_time']:.2f}s")
            print(f"    Total (incl. setup): {elapsed:.2f}s")

        except Exception as e:
            run_result = {
                "run": i + 1,
                "config": config,
                "total_samples": total_samples,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            print(f"    FAILED: {e}")

        all_results.append(run_result)

        # Brief pause between runs to let Spark UI update
        time.sleep(3)

    # Capture Spark UI stats before stopping (port 4040 REST API)
    spark_stats = {}
    try:
        import urllib.request
        # Applications
        apps_resp = urllib.request.urlopen("http://localhost:4040/api/v1/applications", timeout=5)
        apps = json.loads(apps_resp.read().decode())
        if apps:
            app_id = apps[0]["id"]
            # Jobs
            jobs_resp = urllib.request.urlopen(f"http://localhost:4040/api/v1/applications/{app_id}/jobs", timeout=5)
            spark_stats["jobs"] = json.loads(jobs_resp.read().decode())
            # Stages
            stages_resp = urllib.request.urlopen(f"http://localhost:4040/api/v1/applications/{app_id}/stages", timeout=5)
            spark_stats["stages"] = json.loads(stages_resp.read().decode())
            # Executors
            exec_resp = urllib.request.urlopen(f"http://localhost:4040/api/v1/applications/{app_id}/executors", timeout=5)
            spark_stats["executors"] = json.loads(exec_resp.read().decode())
            print(f"\n  Captured Spark UI stats: {len(spark_stats['jobs'])} jobs, "
                  f"{len(spark_stats['stages'])} stages, {len(spark_stats['executors'])} executors")
    except Exception as e:
        print(f"  [WARN] Could not capture Spark UI stats: {e}")

    spark.stop()

    # Save results
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "incremental_results.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"runs": all_results, "spark_ui_stats": spark_stats}, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Run':<5} {'Samples':<10} {'Data MB':<10} {'Throughput':<15} {'Time':<10} {'Status'}")
    print(f"  {'─'*5} {'─'*10} {'─'*10} {'─'*15} {'─'*10} {'─'*8}")
    for r in all_results:
        if r["status"] == "success":
            print(f"  {r['run']:<5} {r['total_samples']:<10,} {r['total_data_mb']:<10} "
                  f"{r['throughput']:<15,.0f} {r['elapsed_time']:<10.2f} OK")
        else:
            print(f"  {r['run']:<5} {r['total_samples']:<10,} {'—':<10} {'—':<15} {'—':<10} FAIL")

    print(f"\n  Results saved: {results_path}")
    print("")


if __name__ == "__main__":
    run_incremental_test()
