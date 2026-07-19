"""
GPU-level metrics publisher — runs on each Spark GPU worker node.

Polls `nvidia-smi` (already available inside the container via
`--gpus all` + NVIDIA Container Toolkit, see deploy/Dockerfile and
deploy/aws-cdk/spark_cluster/spark_cluster_stack.py's worker bootstrap) and
publishes per-GPU utilization/memory/temperature/power to CloudWatch, plus
PyTorch-visible allocated/reserved memory via
inference.gpu_memory_manager.GPUMemoryManager.get_current_gpu_usage() when a
CUDA context is available in this process (falls back gracefully otherwise —
the two sources are complementary: nvidia-smi sees the whole device including
other processes/executors sharing it under MPS, while
torch.cuda.memory_allocated() sees only what *this* process's tensors hold).

Publishes to CloudWatch namespace "SparkInference/GPU", dimensioned by
GpuIndex (in addition to NodeRole/InstanceId from CloudWatchPublisher).

Usage:
    python monitoring/gpu_metrics_publisher.py --interval 15
    python monitoring/gpu_metrics_publisher.py --once
"""
import argparse
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring.cloudwatch_publisher import CloudWatchPublisher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [gpu-metrics] %(message)s")
logger = logging.getLogger("gpu_metrics_publisher")

NAMESPACE = "SparkInference/GPU"

NVIDIA_SMI_FIELDS = [
    "index", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "temperature.gpu", "power.draw",
]


def query_nvidia_smi():
    """Return a list of per-GPU dicts, or [] if nvidia-smi is unavailable (e.g. CPU-only node)."""
    query = ",".join(NVIDIA_SMI_FIELDS)
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi unavailable on this node: %s", exc)
        return []

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(NVIDIA_SMI_FIELDS):
            continue
        record = dict(zip(NVIDIA_SMI_FIELDS, parts))
        gpus.append(record)
    return gpus


def query_torch_memory():
    """Best-effort per-process CUDA memory via torch, if torch/CUDA is importable here."""
    try:
        from inference.gpu_memory_manager import GPUMemoryManager
        return GPUMemoryManager.get_current_gpu_usage()
    except Exception as exc:
        logger.debug("torch CUDA memory query unavailable: %s", exc)
        return {}


def poll_gpu_metrics(publisher: CloudWatchPublisher) -> None:
    gpus = query_nvidia_smi()
    if not gpus:
        return

    torch_usage = query_torch_memory()

    batch = []
    for g in gpus:
        idx = g["index"]
        dims = {"GpuIndex": idx}
        try:
            batch.extend([
                {"metric_name": "GpuUtilizationPercent", "value": float(g["utilization.gpu"]), "unit": "Percent", "extra_dimensions": dims},
                {"metric_name": "GpuMemoryUtilizationPercent", "value": float(g["utilization.memory"]), "unit": "Percent", "extra_dimensions": dims},
                {"metric_name": "GpuMemoryUsedMb", "value": float(g["memory.used"]), "unit": "Megabytes", "extra_dimensions": dims},
                {"metric_name": "GpuMemoryTotalMb", "value": float(g["memory.total"]), "unit": "Megabytes", "extra_dimensions": dims},
                {"metric_name": "GpuTemperatureC", "value": float(g["temperature.gpu"]), "unit": "None", "extra_dimensions": dims},
            ])
            if g["power.draw"] not in ("", "[N/A]", "N/A"):
                batch.append({"metric_name": "GpuPowerDrawWatts", "value": float(g["power.draw"]), "unit": "None", "extra_dimensions": dims})
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed nvidia-smi row for GPU %s: %s", idx, exc)
            continue

        if int(idx) in torch_usage:
            u = torch_usage[int(idx)]
            batch.extend([
                {"metric_name": "TorchAllocatedMb", "value": u["allocated_mb"], "unit": "Megabytes", "extra_dimensions": dims},
                {"metric_name": "TorchReservedMb", "value": u["reserved_mb"], "unit": "Megabytes", "extra_dimensions": dims},
            ])

    publisher.put_metrics(batch)
    logger.info("Published metrics for %d GPU(s)", len(gpus))


def main():
    parser = argparse.ArgumentParser(description="GPU-level metrics (nvidia-smi) -> CloudWatch")
    parser.add_argument("--interval", type=int, default=15, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Poll once and exit (for cron/testing)")
    args = parser.parse_args()

    publisher = CloudWatchPublisher(namespace=NAMESPACE, node_role="worker")
    logger.info("Publishing GPU metrics to CloudWatch namespace '%s' every %ds", NAMESPACE, args.interval)

    while True:
        poll_gpu_metrics(publisher)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
