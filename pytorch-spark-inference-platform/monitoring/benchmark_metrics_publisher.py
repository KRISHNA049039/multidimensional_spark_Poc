"""
Benchmark metrics publisher — bridges the existing benchmark output
(results/raw_results.json, produced by benchmark/run_benchmark.py, see its
`generate_report()`/`main()`) into CloudWatch, and optionally archives both
report files to S3.

This does NOT change run_benchmark.py's output format — it treats
raw_results.json as an external contract and reads it as-is, so existing local
workflows (native run, docker compose local dev) are unaffected.

Two modes:
  --once            Publish whatever is currently in results/raw_results.json, then exit.
  --watch           Poll for changes (mtime) and publish whenever a new run completes.
                     This is what the CDK master bootstrap runs in the background so that
                     every `spark-submit`/`run_benchmark.py` invocation is automatically
                     reflected on the CloudWatch dashboard without operator action.

Metrics published (namespace "SparkInference/Benchmark"), dimensioned by Mode
(single_gpu / hybrid_cpu_gpu / distributed_gpu):
  - ThroughputSamplesPerSec
  - ElapsedTimeSec
  - TotalSamplesProcessed
  - NumModelsOnGpu / NumModelsOnCpu (hybrid mode only, when present)

Usage:
    python monitoring/benchmark_metrics_publisher.py --once
    python monitoring/benchmark_metrics_publisher.py --watch --interval 30
    python monitoring/benchmark_metrics_publisher.py --watch --s3-bucket my-bucket --s3-prefix results/
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring.cloudwatch_publisher import CloudWatchPublisher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [benchmark-metrics] %(message)s")
logger = logging.getLogger("benchmark_metrics_publisher")

NAMESPACE = "SparkInference/Benchmark"
PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RESULTS_PATH = os.path.join(PLATFORM_ROOT, "results", "raw_results.json")
REPORT_PATH = os.path.join(PLATFORM_ROOT, "results", "metrics_report.md")

MODE_KEYS = ["single_gpu", "hybrid_cpu_gpu", "distributed_gpu"]


def load_results(path: str = RAW_RESULTS_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def publish_results(publisher: CloudWatchPublisher, results: dict) -> None:
    batch = []
    for mode_key in MODE_KEYS:
        r = results.get(mode_key)
        if not r or "error" in r:
            continue
        dims = {"Mode": mode_key}
        batch.append({"metric_name": "ThroughputSamplesPerSec", "value": r.get("total_throughput", 0),
                       "unit": "Count/Second", "extra_dimensions": dims})
        batch.append({"metric_name": "ElapsedTimeSec", "value": r.get("elapsed_time", 0),
                       "unit": "Seconds", "extra_dimensions": dims})
        batch.append({"metric_name": "TotalSamplesProcessed", "value": r.get("total_samples_processed", 0),
                       "unit": "Count", "extra_dimensions": dims})
        if "num_on_gpu" in r:
            batch.append({"metric_name": "NumModelsOnGpu", "value": r["num_on_gpu"], "unit": "Count", "extra_dimensions": dims})
        if "num_on_cpu" in r:
            batch.append({"metric_name": "NumModelsOnCpu", "value": r["num_on_cpu"], "unit": "Count", "extra_dimensions": dims})

    total_time = results.get("total_benchmark_time")
    if total_time is not None:
        batch.append({"metric_name": "TotalBenchmarkTimeSec", "value": total_time, "unit": "Seconds"})

    if not batch:
        logger.info("No completed mode results found in raw_results.json yet.")
        return

    publisher.put_metrics(batch)
    logger.info("Published %d benchmark metric datapoints (modes: %s)",
                len(batch), [m for m in MODE_KEYS if m in results])


def upload_to_s3(bucket: str, prefix: str) -> None:
    try:
        import boto3
        from datetime import datetime, timezone
    except ImportError:
        logger.warning("boto3 not available — skipping S3 upload")
        return

    s3 = boto3.client("s3")
    run_prefix = f"{prefix.rstrip('/')}/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    for local_path, filename in [(RAW_RESULTS_PATH, "raw_results.json"), (REPORT_PATH, "metrics_report.md")]:
        if os.path.exists(local_path):
            key = f"{run_prefix}/{filename}"
            try:
                s3.upload_file(local_path, bucket, key)
                logger.info("Uploaded %s -> s3://%s/%s", local_path, bucket, key)
            except Exception as exc:
                logger.warning("Failed to upload %s to S3: %s", local_path, exc)


def main():
    parser = argparse.ArgumentParser(description="Publish benchmark results (results/raw_results.json) -> CloudWatch/S3")
    parser.add_argument("--once", action="store_true", help="Publish current results once and exit")
    parser.add_argument("--watch", action="store_true", help="Poll for new/updated results indefinitely")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds for --watch")
    parser.add_argument("--s3-bucket", default=os.environ.get("ARTIFACTS_BUCKET", ""),
                         help="If set, uploads results/raw_results.json and metrics_report.md to this bucket "
                              "on every new run (defaults to $ARTIFACTS_BUCKET env var).")
    parser.add_argument("--s3-prefix", default="results", help="Key prefix for S3 uploads")
    args = parser.parse_args()

    if not args.once and not args.watch:
        args.once = True  # default behavior

    publisher = CloudWatchPublisher(namespace=NAMESPACE, node_role="master")
    logger.info("Publishing benchmark metrics to CloudWatch namespace '%s' from %s", NAMESPACE, RAW_RESULTS_PATH)

    last_mtime = None
    while True:
        if os.path.exists(RAW_RESULTS_PATH):
            mtime = os.path.getmtime(RAW_RESULTS_PATH)
            if last_mtime is None or mtime > last_mtime:
                results = load_results()
                if results:
                    publish_results(publisher, results)
                    if args.s3_bucket:
                        upload_to_s3(args.s3_bucket, args.s3_prefix)
                last_mtime = mtime
        else:
            logger.debug("No results file yet at %s", RAW_RESULTS_PATH)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
