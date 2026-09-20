"""
BYOM submit CLI — run any registered (built-in or plugin) model as a
distributed Spark inference job across CPU/GPU workers.

Usage:
    python submit_job.py --model example_mlp --input path/to/data.npy \
        --mode hybrid --partitions 8 --batch-size 256

    # generate random input matching the model's registered input_shape
    # instead of loading a file:
    python submit_job.py --model example_mlp --samples 2000 --mode cpu_only
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

from models import get_default_registry
from models.plugin_loader import register_plugins
from inference.cluster_engine import create_cluster_session, run_cluster_inference


def _write_results(model_name: str, summary: dict, results_dir: str = "results") -> str:
    """Writes the run summary to results/<model>_<timestamp>.json (creating
    the directory if needed) and, if ARTIFACTS_BUCKET is set (present on the
    AWS cluster nodes' bootstrap env, absent locally), also uploads it to
    s3://$ARTIFACTS_BUCKET/results/ so deploy/aws-cdk/pull_results.ps1 can
    sync it back to a local machine.
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{model_name}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    bucket = os.environ.get("ARTIFACTS_BUCKET")
    if bucket:
        import boto3
        boto3.client("s3").upload_file(path, bucket, f"results/{os.path.basename(path)}")

    return path


def main():
    parser = argparse.ArgumentParser(description="Submit a model as a distributed Spark inference job")
    parser.add_argument("--model", required=True, help="Registered model name (built-in or plugin)")
    parser.add_argument("--input", default=None, help=".npy file, shape (N, *input_shape), dtype float32")
    parser.add_argument("--samples", type=int, default=None,
                         help="If --input is omitted, generate this many random samples instead")
    parser.add_argument("--mode", default="hybrid", choices=["cpu_only", "gpu_only", "hybrid"])
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--master", default=None, help="Spark master URL override (default: env or local[4])")
    parser.add_argument("--engine", default="rdd", choices=["rdd", "udf"],
                         help="rdd (default, unchanged): cluster_engine.py's mapPartitions path. "
                              "udf: Pandas-UDF path (docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md §1) — "
                              "opt-in, only affects this run, no other behavior changes.")
    args = parser.parse_args()

    registry = get_default_registry()
    plugin_names = register_plugins(registry)
    if args.model not in registry.list_models():
        available = ", ".join(sorted(registry.list_models().keys()))
        raise SystemExit(f"Unknown model '{args.model}'. Registered models: {available}")

    info = registry.get_info(args.model)

    if args.input:
        data_arr = np.load(args.input).astype("float32")
    else:
        n = args.samples or 512
        data_arr = np.random.randn(n, *info.input_shape).astype("float32")

    model = registry.load_model(args.model, device="cpu")

    spark = create_cluster_session(app_name=f"byom-{args.model}", master_url=args.master)
    try:
        if args.engine == "udf":
            from inference.cluster_engine_udf import run_cluster_inference_udf
            result = run_cluster_inference_udf(
                spark,
                data={args.model: data_arr},
                models={args.model: model},
                num_partitions=args.partitions,
                batch_size=args.batch_size,
                device_mode=args.mode,
            )
        else:
            result = run_cluster_inference(
                spark,
                data={args.model: data_arr},
                models={args.model: model},
                num_partitions=args.partitions,
                batch_size=args.batch_size,
                device_mode=args.mode,
            )
    finally:
        spark.stop()

    summary = {k: v for k, v in result.items() if k not in ("partition_details", "spark_ui_stats")}
    print(json.dumps(summary, indent=2, default=str))
    saved_path = _write_results(args.model, summary)
    print(f"Results written to {saved_path}")


if __name__ == "__main__":
    main()
