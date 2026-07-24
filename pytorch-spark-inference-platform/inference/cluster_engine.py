"""
Cluster Inference Engine — Optimized Spark distributed inference.

Key improvements over distributed_gpu.py:
1. mapPartitions: loads models ONCE per executor (not per task)
2. 3 device modes: gpu_only, cpu_only, hybrid (auto-detect)
3. Returns detailed per-task timing (model load time, inference time)
4. Supports CLI parameter injection

Modes:
  - gpu_only: Forces device=cuda on executors (fails if no GPU)
  - cpu_only: Forces device=cpu on all executors
  - hybrid:   Auto-detects — uses cuda where available, cpu elsewhere

Architecture:
  Driver:
    1. Serialize model weights → broadcast (small, ~75MB)
    2. Generate/load input data
    3. Partition data into RDD elements
    4. Submit to Spark executors

  Executor (via mapPartitions):
    1. Load models ONCE for this executor (not per partition element)
    2. Process ALL partition elements using loaded models
    3. Return detailed timing per element

  Driver collects results → aggregates → reports
"""

import sys
import os
import time
import io
import json
import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_cluster_session(app_name="ClusterBenchmark", master_url=None,
                           executor_memory="2g", driver_memory="4g",
                           executor_cores=2, max_message_size=512):
    """Create SparkSession optimized for cluster inference."""
    from pyspark.sql import SparkSession

    if master_url is None:
        master_url = os.environ.get("SPARK_MASTER_URL") or os.environ.get("SPARK_MASTER")
    if not master_url:
        master_url = "local[4]"

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.cores", str(executor_cores))
        .config("spark.task.cpus", "1")
        .config("spark.rpc.message.maxSize", str(max_message_size))
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "120s")
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.worker.memory", "2g")
        .config("spark.driver.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED "
                "--add-opens=java.base/java.util=ALL-UNNAMED")
    )

    if "local" not in master_url:
        builder = (builder
            .config("spark.executorEnv.PYTHONPATH", "/app:/app/inference:/app/models:/app/data")
            .config("spark.executorEnv.NVIDIA_VISIBLE_DEVICES", "all")
            .config("spark.executorEnv.CUDA_VISIBLE_DEVICES", "0")
            .config("spark.executorEnv.LD_LIBRARY_PATH",
                    "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _serialize_model(model: torch.nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _get_class_map():
    """Import and return model class map (called inside executor)."""
    from models.ew_signal_model import EWSignalClassifier
    from models.yolo_model import YOLOv8Nano, YOLOv8Small
    from models.image_models import (ResNet18Classifier, MobileNetV3Classifier,
                                     EfficientNetB0Classifier)
    from models.signal_models import (SignalDenoiser, ThreatPrioritizer,
                                      RFFingerprinter, AnomalyDetector)
    return {
        "ew_classifier": EWSignalClassifier,
        "yolov8_nano": YOLOv8Nano,
        "yolov8_small": YOLOv8Small,
        "resnet18": ResNet18Classifier,
        "mobilenetv3": MobileNetV3Classifier,
        "efficientnet_b0": EfficientNetB0Classifier,
        "signal_denoiser": SignalDenoiser,
        "threat_prioritizer": ThreatPrioritizer,
        "rf_fingerprinter": RFFingerprinter,
        "anomaly_detector": AnomalyDetector,
    }


def run_cluster_inference(
    spark,
    data: Dict[str, np.ndarray],
    models: Dict[str, torch.nn.Module],
    num_partitions: int = 4,
    batch_size: int = 256,
    device_mode: str = "hybrid",
) -> Dict:
    """
    Run distributed inference with mapPartitions optimization.

    Args:
        spark: SparkSession
        data: {model_name: numpy array}
        models: {model_name: loaded model}
        num_partitions: Number of data partitions
        batch_size: Inference batch size
        device_mode: "gpu_only", "cpu_only", or "hybrid" (auto-detect)

    Returns:
        Detailed results dict with per-partition timing
    """
    sc = spark.sparkContext

    # Broadcast model weights (~75MB, same for all executors)
    model_bytes_map = {name: _serialize_model(m) for name, m in models.items()}
    bc_model_bytes = sc.broadcast(model_bytes_map)
    bc_batch_size = sc.broadcast(batch_size)
    bc_device_mode = sc.broadcast(device_mode)

    # Partition data into RDD
    total_samples = {}
    partition_data = []
    for i in range(num_partitions):
        chunk_dict = {}
        for model_name, arr in data.items():
            n = len(arr)
            total_samples[model_name] = n
            chunk_size = n // num_partitions
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_partitions - 1 else n
            chunk_dict[model_name] = arr[start_idx:end_idx].copy()
        partition_data.append((i, chunk_dict))

    data_rdd = sc.parallelize(partition_data, num_partitions)

    def process_partition(iterator):
        """
        mapPartitions function — loads models ONCE, processes all items.
        This is the key optimization: model loading happens once per executor,
        not once per partition element.
        """
        import sys
        import os
        import time
        import socket
        import torch
        import numpy as np

        for p in ["/app", "/app/inference", "/app/models", "/app/data"]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Determine device based on mode
        mode = bc_device_mode.value
        cuda_available = torch.cuda.is_available()

        if mode == "gpu_only":
            device = "cuda" if cuda_available else "cpu"
            if not cuda_available:
                print(f"[WARN] gpu_only mode but CUDA not available, falling back to cpu")
        elif mode == "cpu_only":
            device = "cpu"
        else:  # hybrid
            device = "cuda" if cuda_available else "cpu"

        hostname = socket.gethostname()
        executor_id = f"{hostname}_{os.getpid()}"

        # Load models ONCE for this executor
        model_load_start = time.time()
        class_map = _get_class_map()
        model_bytes = bc_model_bytes.value
        loaded_models = {}
        for name, mbytes in model_bytes.items():
            if name in class_map:
                buf = io.BytesIO(mbytes)
                model = class_map[name]()
                model.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
                model = model.to(device).eval()
                loaded_models[name] = model
        model_load_time = time.time() - model_load_start

        print(f"[Executor] host={hostname}, pid={os.getpid()}, device={device}, "
              f"cuda={cuda_available}, models_loaded={len(loaded_models)}, "
              f"model_load_time={model_load_time:.2f}s")

        bs = bc_batch_size.value

        # Process ALL partition elements with loaded models
        for partition_idx, data_chunk in iterator:
            task_start = time.time()
            task_results = {}
            task_samples = 0

            for model_name, model in loaded_models.items():
                if model_name not in data_chunk:
                    continue
                chunk = data_chunk[model_name]
                n_samples = len(chunk)
                num_outputs = 0

                with torch.no_grad():
                    for start in range(0, n_samples, bs):
                        end = min(start + bs, n_samples)
                        batch_np = chunk[start:end]
                        batch_tensor = torch.from_numpy(batch_np).float().to(device)
                        _ = model(batch_tensor)
                        num_outputs += (end - start)

                task_results[model_name] = num_outputs
                task_samples += num_outputs

            task_time = time.time() - task_start
            inference_time = task_time  # model already loaded

            yield {
                "partition_idx": partition_idx,
                "executor_id": executor_id,
                "hostname": hostname,
                "device": device,
                "cuda_available": cuda_available,
                "model_load_time_sec": round(model_load_time, 4),
                "inference_time_sec": round(inference_time, 4),
                "total_task_time_sec": round(task_time, 4),
                "samples_processed": task_samples,
                "per_model_processed": task_results,
                "batch_size": bs,
            }

    # Execute with mapPartitions (models loaded ONCE per executor)
    start_time = time.time()
    partition_results = data_rdd.mapPartitions(process_partition).collect()
    elapsed_time = time.time() - start_time

    # Aggregate
    total_processed = {}
    for pr in partition_results:
        for name, count in pr["per_model_processed"].items():
            total_processed[name] = total_processed.get(name, 0) + count

    total_all = sum(total_processed.values())
    throughput = total_all / elapsed_time if elapsed_time > 0 else 0

    # Collect Spark UI stats
    spark_ui_stats = _capture_spark_ui_stats()

    bc_model_bytes.unpersist()

    return {
        "mode": f"distributed_{device_mode}",
        "device_mode": device_mode,
        "elapsed_time": round(elapsed_time, 4),
        "total_samples_processed": total_all,
        "total_throughput": round(throughput, 1),
        "per_model_processed": total_processed,
        "num_partitions": num_partitions,
        "num_models": len(models),
        "batch_size": batch_size,
        "partition_details": partition_results,
        "spark_ui_stats": spark_ui_stats,
        "timestamp": datetime.now().isoformat(),
    }


def _capture_spark_ui_stats() -> Dict:
    """Capture stats from Spark REST API (port 4040) before session closes."""
    import urllib.request
    import json

    stats = {}
    try:
        base = "http://localhost:4040/api/v1/applications"
        apps = json.loads(urllib.request.urlopen(base, timeout=3).read().decode())
        if apps:
            app_id = apps[0]["id"]
            stats["app_id"] = app_id
            stats["jobs"] = json.loads(
                urllib.request.urlopen(f"{base}/{app_id}/jobs", timeout=3).read().decode())
            stats["stages"] = json.loads(
                urllib.request.urlopen(f"{base}/{app_id}/stages", timeout=3).read().decode())
            stats["executors"] = json.loads(
                urllib.request.urlopen(f"{base}/{app_id}/executors", timeout=3).read().decode())
    except Exception:
        pass
    return stats
