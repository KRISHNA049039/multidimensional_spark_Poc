"""
Spark master/driver + executor level metrics publisher.

Runs on the master node (started by the CDK worker/master bootstrap via
`docker exec -d spark-master python monitoring/spark_metrics_publisher.py --role master`).
Polls two Spark REST endpoints that are already exposed by the cluster (no extra
Spark config needed):

1. Master JSON state — http://localhost:8080/json/
   -> cluster-level: alive/dead workers, total/used cores, total/used memory,
      active/completed application counts.

2. Application UI REST API — http://localhost:4040/api/v1/applications/<id>/executors
   -> executor-level: active/completed/failed tasks, memory used, disk used,
      total duration, per executor (dimensioned by ExecutorId + Host). This is
      the "executor level" metric the platform's benchmark modes map onto Spark
      executors (see inference/distributed_gpu.py).

Publishes to CloudWatch namespace "SparkInference/Spark".

Usage:
    python monitoring/spark_metrics_publisher.py --role master --interval 30
    python monitoring/spark_metrics_publisher.py --once   # single poll, for cron/testing
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring.cloudwatch_publisher import CloudWatchPublisher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [spark-metrics] %(message)s")
logger = logging.getLogger("spark_metrics_publisher")

NAMESPACE = "SparkInference/Spark"
MASTER_JSON_URL = "http://localhost:8080/json/"
APP_UI_BASE_URL = "http://localhost:4040/api/v1/applications"


def _get_json(url: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        logger.debug("GET %s failed: %s", url, exc)
        return None


def poll_master_metrics(publisher: CloudWatchPublisher) -> None:
    """Cluster/master-level metrics from the Spark standalone master's JSON endpoint."""
    state = _get_json(MASTER_JSON_URL)
    if state is None:
        logger.warning("Master JSON endpoint unreachable at %s — is spark-master running?", MASTER_JSON_URL)
        return

    workers = state.get("workers", [])
    alive_workers = [w for w in workers if w.get("state") == "ALIVE"]

    metrics = [
        {"metric_name": "TotalWorkers", "value": len(workers), "unit": "Count"},
        {"metric_name": "ActiveWorkers", "value": len(alive_workers), "unit": "Count"},
        {"metric_name": "TotalCores", "value": sum(w.get("cores", 0) for w in workers), "unit": "Count"},
        {"metric_name": "CoresUsed", "value": sum(w.get("coresused", 0) for w in workers), "unit": "Count"},
        {"metric_name": "TotalMemoryMb", "value": sum(w.get("memory", 0) for w in workers), "unit": "Megabytes"},
        {"metric_name": "MemoryUsedMb", "value": sum(w.get("memoryused", 0) for w in workers), "unit": "Megabytes"},
        {"metric_name": "ActiveApps", "value": len(state.get("activeapps", [])), "unit": "Count"},
        {"metric_name": "CompletedApps", "value": len(state.get("completedapps", [])), "unit": "Count"},
    ]
    publisher.put_metrics(metrics)
    logger.info("Master: %d/%d workers alive, %d active apps",
                len(alive_workers), len(workers), len(state.get("activeapps", [])))


def poll_executor_metrics(publisher: CloudWatchPublisher) -> None:
    """
    Executor-level metrics from the running Spark application's REST API (port 4040).
    Only available while a benchmark/inference job is actively running a SparkSession
    (see benchmark/run_benchmark.py -> run_mode_distributed -> create_spark_session()).
    """
    apps = _get_json(APP_UI_BASE_URL)
    if not apps:
        logger.debug("No active Spark application on port 4040 (job not running right now).")
        return

    # The application UI REST API only serves the currently running application
    # in local/standalone client mode; take the most recent entry.
    app = apps[0]
    app_id = app.get("id")
    executors = _get_json(f"{APP_UI_BASE_URL}/{app_id}/executors")
    if not executors:
        return

    batch = []
    total_active_tasks = 0
    total_completed_tasks = 0
    for ex in executors:
        executor_id = ex.get("id", "unknown")
        host = (ex.get("hostPort") or "unknown").split(":")[0]
        dims = {"ExecutorId": str(executor_id), "Host": host}

        active_tasks = ex.get("activeTasks", 0)
        completed_tasks = ex.get("completedTasks", 0)
        total_active_tasks += active_tasks
        total_completed_tasks += completed_tasks

        batch.extend([
            {"metric_name": "ExecutorActiveTasks", "value": active_tasks, "unit": "Count", "extra_dimensions": dims},
            {"metric_name": "ExecutorCompletedTasks", "value": completed_tasks, "unit": "Count", "extra_dimensions": dims},
            {"metric_name": "ExecutorFailedTasks", "value": ex.get("failedTasks", 0), "unit": "Count", "extra_dimensions": dims},
            {"metric_name": "ExecutorTotalDurationMs", "value": ex.get("totalDuration", 0), "unit": "Milliseconds", "extra_dimensions": dims},
            {"metric_name": "ExecutorMemoryUsedMb", "value": ex.get("memoryUsed", 0) / 1e6, "unit": "Megabytes", "extra_dimensions": dims},
            {"metric_name": "ExecutorDiskUsedMb", "value": ex.get("diskUsed", 0) / 1e6, "unit": "Megabytes", "extra_dimensions": dims},
            {"metric_name": "ExecutorMaxMemoryMb", "value": ex.get("maxMemory", 0) / 1e6, "unit": "Megabytes", "extra_dimensions": dims},
            {"metric_name": "ExecutorTotalCores", "value": ex.get("totalCores", 0), "unit": "Count", "extra_dimensions": dims},
        ])

    publisher.put_metrics(batch)
    logger.info("App %s: %d executors, %d active tasks, %d completed tasks",
                app_id, len(executors), total_active_tasks, total_completed_tasks)


def main():
    parser = argparse.ArgumentParser(description="Spark master/driver + executor metrics -> CloudWatch")
    parser.add_argument("--role", default="master", choices=["master"],
                        help="Node role (only 'master' polls the Spark REST APIs; worker-level "
                             "GPU metrics are handled separately by gpu_metrics_publisher.py)")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Poll once and exit (for cron/testing)")
    args = parser.parse_args()

    publisher = CloudWatchPublisher(namespace=NAMESPACE, node_role="master")
    logger.info("Publishing Spark master/executor metrics to CloudWatch namespace '%s' every %ds",
                NAMESPACE, args.interval)

    while True:
        poll_master_metrics(publisher)
        poll_executor_metrics(publisher)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
