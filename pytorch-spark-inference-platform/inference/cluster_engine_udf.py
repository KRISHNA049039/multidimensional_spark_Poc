"""
Cluster Inference Engine — Pandas UDF variant (docs/CONCURRENCY_AND_UDF_
ENHANCEMENTS.md §1). Sibling to cluster_engine.py, NOT a replacement —
opt-in via submit_job.py's --engine udf flag. Everything on the RDD path
(cluster_engine.py's run_cluster_inference, and all of models/pipelines/
via text_pipeline_engine.py) keeps working completely unchanged; this file
is only reachable when explicitly requested.

Only targets the tensor-plugin path (models/plugins/, fixed-shape arrays)
— see predict_batch_udf.py's own module docstring for why models/pipelines/
stays on the RDD engine.

One real, honest difference from cluster_engine.py's RDD path: this engine
actually returns per-sample predictions (via predict_batch_udf), where the
RDD path's process_partition() only counts samples and discards the actual
model output. Not a regression — a capability the RDD path never had.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict

from inference.cluster_engine import _serialize_model, _get_class_map, _capture_spark_ui_stats
from inference.predict_batch_udf import predict_batch_udf


def run_cluster_inference_udf(
    spark,
    data: Dict,
    models: Dict,
    num_partitions: int = 4,
    batch_size: int = 256,
    device_mode: str = "hybrid",
) -> Dict:
    """
    Pandas-UDF equivalent of cluster_engine.run_cluster_inference() — same
    call signature and same top-level return keys (submit_job.py's summary
    construction works unchanged regardless of which engine ran), so this
    is a genuine drop-in alternative, not a parallel API to learn.

    Args:
        spark: SparkSession
        data: {model_name: numpy array}
        models: {model_name: loaded model} — only used to look up each
            model's class (via _get_class_map()) and serialize its weights,
            same as the RDD path; the actual forward pass happens inside
            predict_batch_udf's per-partition UDF, not here.
        num_partitions: number of DataFrame partitions
        batch_size: passed through as predict_batch_udf's
            inference_batch_size — the MODEL's batch size, reconciled
            against Arrow's own (larger, Spark-controlled) batch size
            inside predict_batch_udf itself.
        device_mode: "gpu_only", "cpu_only", or "hybrid" — same semantics
            as the RDD path; resolved per-partition inside the UDF, not
            trusted from the driver (a driver without a GPU shouldn't
            decide executors don't have one either).
    """
    import numpy as np
    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, ArrayType, FloatType

    class_map = _get_class_map()
    start_time = time.time()

    per_model_processed = {}
    per_model_predictions = {}  # model_name -> list of prediction rows (for callers that want them)

    for model_name, model in models.items():
        if model_name not in data:
            continue
        if model_name not in class_map:
            raise ValueError(f"run_cluster_inference_udf: '{model_name}' not in _get_class_map() — "
                              f"same registration class_map cluster_engine.py's RDD path uses.")

        arr = data[model_name]
        model_bytes = _serialize_model(model)
        model_class = class_map[model_name]

        # device_mode resolution mirrors cluster_engine.py's process_partition:
        # "gpu_only"/"hybrid" both prefer cuda, actual availability is
        # re-checked per-partition inside predict_batch_udf itself (a
        # driver-side check here would be meaningless for a real cluster).
        requested_device = "cpu" if device_mode == "cpu_only" else "cuda"

        # Rows travel through Spark as FLAT 1-D float arrays with an
        # EXPLICIT schema — not spark.createDataFrame(rows) inferring a
        # schema from nested (3,224,224)-shaped Python lists, which hung
        # for minutes in real local testing (schema inference over deeply
        # nested objects is a known-slow PySpark path). predict_batch_udf
        # reshapes each row back to sample_shape right before the model
        # call — see its own docstring for the same story from the other
        # side.
        sample_shape = arr.shape[1:]
        rows = [Row(input=row.reshape(-1).tolist()) for row in arr]
        schema = StructType([StructField("input", ArrayType(FloatType()), False)])
        df = spark.createDataFrame(rows, schema=schema).repartition(num_partitions)

        udf = predict_batch_udf(model_bytes, model_class,
                                 sample_shape=sample_shape,
                                 device=requested_device,
                                 inference_batch_size=batch_size)
        result_df = df.select(udf(df.input).alias("prediction"))
        collected = result_df.collect()

        predictions = [r.prediction for r in collected]
        per_model_predictions[model_name] = predictions
        per_model_processed[model_name] = sum(1 for p in predictions if p is not None)

    elapsed_time = time.time() - start_time
    total_all = sum(per_model_processed.values())
    throughput = total_all / elapsed_time if elapsed_time > 0 else 0

    spark_ui_stats = _capture_spark_ui_stats()

    return {
        "mode": f"distributed_{device_mode}_udf",
        "device_mode": device_mode,
        "engine": "udf",
        "elapsed_time": round(elapsed_time, 4),
        "total_samples_processed": total_all,
        "total_throughput": round(throughput, 1),
        "per_model_processed": per_model_processed,
        "num_partitions": num_partitions,
        "num_models": len(models),
        "batch_size": batch_size,
        "predictions": per_model_predictions,
        "partition_details": [],  # not tracked per-partition on this engine — see module docstring
        "spark_ui_stats": spark_ui_stats,
        "timestamp": datetime.now().isoformat(),
    }
