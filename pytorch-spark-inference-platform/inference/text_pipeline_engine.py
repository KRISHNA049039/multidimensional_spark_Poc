"""
Text/file pipeline execution engine — Spark mapPartitions runner for
"pipeline" style plugins (multi-model, non-tensor input/output: e.g. the
NER+translation pipeline), as opposed to cluster_engine.py's fixed-shape
tensor engine.

A pipeline plugin module (see models/pipelines/manifest.json) must expose:
    load() -> Any                              # loaded once per executor
    run(loaded, paths, **kwargs) -> Dict[str, Any]   # real results, not just counts

Unlike run_cluster_inference() in cluster_engine.py (which only counts
samples processed - see docs/CONCURRENT_JOBS_AND_COMPLETING_THE_FRAMEWORK.md
section 4), this engine collects and returns the pipeline's actual output.
"""
import os
import socket
import time
from typing import Any, Dict, List, Optional


def run_text_pipeline_job(
    spark,
    file_paths: List[str],
    load_fn,
    run_fn,
    num_partitions: int = 4,
    run_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Args:
        spark: SparkSession (from inference.cluster_engine.create_cluster_session)
        file_paths: list of input file paths to distribute across partitions
        load_fn: zero-arg callable, called once per executor (loads models)
        run_fn: callable(loaded, paths, **run_kwargs) -> {filename: result}
        num_partitions: number of Spark partitions (= parallel tasks)
        run_kwargs: extra kwargs forwarded to run_fn on every call

    Returns:
        {
            "elapsed_time": float,
            "num_files": int,
            "num_partitions": int,
            "partition_details": [{hostname, pid, num_paths, load_time_sec, run_time_sec}, ...],
            "results": {filename: result, ...}   # merged across all partitions
        }
    """
    sc = spark.sparkContext
    run_kwargs = run_kwargs or {}
    bc_run_kwargs = sc.broadcast(run_kwargs)

    effective_partitions = max(1, min(num_partitions, len(file_paths)))
    paths_rdd = sc.parallelize(file_paths, effective_partitions)

    def process_partition(iterator):
        paths = list(iterator)
        if not paths:
            return
        hostname = socket.gethostname()

        load_start = time.time()
        loaded = load_fn()
        load_time = time.time() - load_start

        run_start = time.time()
        results = run_fn(loaded, paths, **bc_run_kwargs.value)
        run_time = time.time() - run_start

        yield {
            "hostname": hostname,
            "pid": os.getpid(),
            "num_paths": len(paths),
            "load_time_sec": round(load_time, 3),
            "run_time_sec": round(run_time, 3),
            "results": results,
        }

    start = time.time()
    partition_results = paths_rdd.mapPartitions(process_partition).collect()
    elapsed = time.time() - start

    merged_results: Dict[str, Any] = {}
    for pr in partition_results:
        merged_results.update(pr["results"])

    bc_run_kwargs.unpersist()

    return {
        "elapsed_time": round(elapsed, 4),
        "num_files": len(file_paths),
        "num_partitions": effective_partitions,
        "partition_details": [
            {k: v for k, v in pr.items() if k != "results"} for pr in partition_results
        ],
        "results": merged_results,
    }
