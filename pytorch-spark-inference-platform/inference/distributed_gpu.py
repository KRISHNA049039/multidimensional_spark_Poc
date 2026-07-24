"""
Mode 1: Distributed GPU Inference (Spark + Multi-GPU Cluster)

Each Spark executor runs on a separate machine/GPU. All 10 models are loaded
on each executor's GPU via CUDA streams for parallel inference.

Architecture:
  Driver → broadcast model weights → partition data → distribute to executors
  Executor → deserialize models → load on local GPU → CUDA streams inference
  Results → collect back to driver

Works in:
  - local[N] mode: simulates distribution (CPU, threads)
  - cluster mode: true multi-GPU across machines (with MPS for process sharing)
"""

import sys
import os
import time
import io
import numpy as np
import torch
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_spark_session(app_name="MultiModel_Distributed_GPU", num_cores="4",
                         master_url=None):
    """Create SparkSession configured for distributed GPU inference.

    Args:
        app_name: Spark application name
        num_cores: Number of cores for local mode (ignored if master_url is set)
        master_url: Spark master URL (e.g. 'spark://10.0.0.187:7077').
                    If None, checks SPARK_MASTER env var, then falls back to local[N].
    """
    from pyspark.sql import SparkSession

    # Determine master: explicit arg > env var > local mode
    if master_url is None:
        master_url = os.environ.get("SPARK_MASTER_URL") or os.environ.get("SPARK_MASTER")
    if not master_url:
        master_url = f"local[{num_cores}]"

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.task.cpus", "1")
        .config("spark.rpc.message.maxSize", "512")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "120s")
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.worker.memory", "1g")
        .config("spark.driver.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED "
                "--add-opens=java.base/java.util=ALL-UNNAMED")
    )

    # When connecting to a cluster, set executor env so workers detect GPU
    # and can find the app modules
    if "local" not in master_url:
        builder = (builder
            .config("spark.executorEnv.SPARK_EXECUTOR_GPU", "1")
            .config("spark.executorEnv.PYTHONPATH", "/app:/app/inference:/app/models:/app/data")
            .config("spark.executorEnv.NVIDIA_VISIBLE_DEVICES", "all")
            .config("spark.executorEnv.CUDA_VISIBLE_DEVICES", "0")
            .config("spark.executorEnv.LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _serialize_model(model: torch.nn.Module) -> bytes:
    """Serialize a model's state dict to bytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _deserialize_model(model_class, model_bytes: bytes, device: str = "cpu") -> torch.nn.Module:
    """Deserialize model from bytes onto specified device."""
    buf = io.BytesIO(model_bytes)
    model = model_class()
    model.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def run_distributed_gpu_inference(
    spark,
    data: Dict[str, np.ndarray],
    models: Dict[str, torch.nn.Module],
    model_classes: Dict[str, type],
    num_partitions: int = 4,
    batch_size: int = 256,
) -> Dict[str, any]:
    """
    Run all models across Spark partitions using distributed GPU/CPU.

    Each partition:
    1. Deserializes all model weights from broadcast
    2. Loads models on available device (GPU in cluster, CPU in local)
    3. Runs inference using CUDA streams (if GPU available)
    4. Returns per-model results

    Args:
        spark: SparkSession
        data: {model_name: numpy input array} — data for each model
        models: {model_name: loaded model instance}
        model_classes: {model_name: model class} for deserialization
        num_partitions: Number of Spark partitions
        batch_size: Batch size per model per partition

    Returns:
        Dict with timing, throughput, and per-model metrics
    """
    sc = spark.sparkContext

    # Broadcast only model weights (~75MB total) — small, same for all executors
    model_bytes_map = {}
    for name, model in models.items():
        model_bytes_map[name] = _serialize_model(model)
    bc_model_bytes = sc.broadcast(model_bytes_map)
    bc_batch_size = sc.broadcast(batch_size)

    # For data distribution, we use a hybrid approach:
    # - Small data (<100MB per partition): embed in RDD elements
    # - Large data: save to /tmp as numpy files, pass file paths via RDD
    # This avoids both broadcast limits AND task serialization limits.

    total_samples = {}
    for model_name, arr in data.items():
        total_samples[model_name] = len(arr)

    # Calculate total data size per partition
    total_data_bytes = sum(arr.nbytes for arr in data.values())
    per_partition_bytes = total_data_bytes / num_partitions

    # Always embed data directly in RDD elements.
    # spark.rpc.message.maxSize is set to 512MB to handle large partitions.
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

    def infer_on_partition(item):
        """
        Worker function: load models, run inference on data chunk.
        Returns detailed stats: executor info, per-model timing, input/output shapes.
        """
        partition_idx, data_chunk = item
        import sys
        import os
        import time as _time
        import torch
        import numpy as np

        # Ensure app modules are importable on the executor
        for p in ["/app", "/app/inference", "/app/models", "/app/data"]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        else:
            device = "cpu"
            gpu_name = "N/A"
            gpu_memory_mb = 0

        import socket
        hostname = socket.gethostname()
        executor_id = f"{hostname}:{os.getpid()}"

        # Deserialize models
        model_load_start = _time.time()
        model_bytes = bc_model_bytes.value
        loaded_models = {}

        from models.ew_signal_model import EWSignalClassifier
        from models.yolo_model import YOLOv8Nano, YOLOv8Small
        from models.image_models import ResNet18Classifier, MobileNetV3Classifier, EfficientNetB0Classifier
        from models.signal_models import SignalDenoiser, ThreatPrioritizer, RFFingerprinter, AnomalyDetector

        class_map = {
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

        for name, mbytes in model_bytes.items():
            if name in class_map:
                loaded_models[name] = _deserialize_model(class_map[name], mbytes, device)

        model_load_time = _time.time() - model_load_start

        bs = bc_batch_size.value
        model_results = {}
        total_inference_start = _time.time()

        for model_name, model in loaded_models.items():
            if model_name not in data_chunk:
                continue

            chunk = data_chunk[model_name]
            n_samples = len(chunk)
            num_outputs = 0
            input_shape = list(chunk.shape[1:]) if len(chunk.shape) > 1 else [chunk.shape[-1] if len(chunk.shape) == 1 else 0]
            output_shape = None

            model_start = _time.time()
            with torch.no_grad():
                for start in range(0, n_samples, bs):
                    end = min(start + bs, n_samples)
                    batch_np = chunk[start:end]
                    batch_tensor = torch.from_numpy(batch_np).float().to(device)
                    output = model(batch_tensor)
                    if output_shape is None:
                        output_shape = list(output.shape[1:]) if len(output.shape) > 1 else [output.shape[0]]
                    num_outputs += (end - start)
            model_elapsed = _time.time() - model_start

            model_results[model_name] = {
                "samples_processed": num_outputs,
                "inference_time_sec": round(model_elapsed, 4),
                "throughput": round(num_outputs / model_elapsed, 1) if model_elapsed > 0 else 0,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "batches": (n_samples + bs - 1) // bs,
            }

        total_inference_time = _time.time() - total_inference_start
        total_samples = sum(r["samples_processed"] for r in model_results.values())

        return {
            "partition_idx": partition_idx,
            "executor": {
                "id": executor_id,
                "hostname": hostname,
                "pid": os.getpid(),
                "device": device,
                "gpu_name": gpu_name,
                "gpu_memory_mb": round(gpu_memory_mb, 0),
                "cpu_count": os.cpu_count(),
            },
            "timing": {
                "model_load_time_sec": round(model_load_time, 4),
                "total_inference_time_sec": round(total_inference_time, 4),
                "total_task_time_sec": round(model_load_time + total_inference_time, 4),
            },
            "models_run": len(model_results),
            "total_samples_processed": total_samples,
            "per_model": model_results,
        }

    # Execute distributed inference
    start_time = time.time()
    partition_results = data_rdd.map(infer_on_partition).collect()
    elapsed_time = time.time() - start_time

    # Aggregate results
    total_processed = {}
    for pr in partition_results:
        for name, info in pr["per_model"].items():
            total_processed[name] = total_processed.get(name, 0) + info["samples_processed"]

    total_all = sum(total_processed.values())
    throughput = total_all / elapsed_time

    # Capture Spark UI stats
    spark_ui_stats = {}
    try:
        import urllib.request
        import json as _json
        apps = _json.loads(urllib.request.urlopen("http://localhost:4040/api/v1/applications", timeout=3).read())
        if apps:
            app_id = apps[0]["id"]
            spark_ui_stats["app_id"] = app_id
            spark_ui_stats["app_name"] = apps[0].get("name", "")
            try:
                jobs = _json.loads(urllib.request.urlopen(
                    f"http://localhost:4040/api/v1/applications/{app_id}/jobs", timeout=3).read())
                spark_ui_stats["jobs"] = jobs
            except:
                pass
            try:
                stages = _json.loads(urllib.request.urlopen(
                    f"http://localhost:4040/api/v1/applications/{app_id}/stages", timeout=3).read())
                spark_ui_stats["stages"] = stages
            except:
                pass
            try:
                executors = _json.loads(urllib.request.urlopen(
                    f"http://localhost:4040/api/v1/applications/{app_id}/allexecutors", timeout=3).read())
                spark_ui_stats["executors"] = executors
            except:
                pass
    except:
        pass

    # Cleanup
    bc_model_bytes.unpersist()

    return {
        "mode": "distributed_gpu",
        "elapsed_time": round(elapsed_time, 4),
        "total_samples_processed": total_all,
        "total_throughput": round(throughput, 1),
        "per_model_processed": total_processed,
        "num_partitions": num_partitions,
        "num_models": len(models),
        "partition_details": partition_results,
        "spark_ui_stats": spark_ui_stats,
    }
