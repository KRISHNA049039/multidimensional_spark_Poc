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


def run_text_pipeline_job_via_service(
    spark,
    file_paths: List[str],
    service_url: str,
    num_partitions: int = 4,
    labels: Optional[List[str]] = None,
    timeout: float = 600.0,
) -> Dict:
    """
    Sibling to run_text_pipeline_job() for the "waiter/kitchen" split (see
    docs/MODEL_CONTAINER_ISOLATION.md Option B) — instead of importing a
    pipeline module and calling load()/run() inside the Spark executor
    process (which requires torch/CUDA/the pipeline's own deps to be
    installed in the SAME Python environment Spark uses), each partition
    just POSTs its file paths to an already-running model server and gets
    back the identical {filename: result} shape run_fn() always returned.

    Requires: the executor's own Python environment needs nothing but
    `requests` — no torch, no transformers, no model-specific deps at all.
    Requires: file_paths must be readable from the SAME mounted path on
    the model-server container as on the Spark worker (e.g. both mount
    the same host data/ directory to /app/data) — paths are sent as-is,
    not the file contents, matching how the in-process version already
    expects absolute paths on a real cluster (see submit_pipeline_job.py's
    own note on this).
    """
    sc = spark.sparkContext
    bc_service_url = sc.broadcast(service_url)
    bc_labels = sc.broadcast(labels)
    bc_timeout = sc.broadcast(timeout)

    effective_partitions = max(1, min(num_partitions, len(file_paths)))
    paths_rdd = sc.parallelize(file_paths, effective_partitions)

    def process_partition(iterator):
        # Imported here, not at module/outer-function scope: this closure
        # is pickled and shipped to a separate executor process, which
        # re-imports its own free variables on unpickling — importing
        # inside the closure itself is the standard, unambiguous way to
        # guarantee `requests` is resolved fresh in that process rather
        # than relying on however cloudpickle happens to serialize a
        # module reference captured from the driver's enclosing scope.
        import requests

        paths = list(iterator)
        if not paths:
            return
        hostname = socket.gethostname()

        call_start = time.time()
        resp = requests.post(
            bc_service_url.value.rstrip("/") + "/predict",
            json={"paths": paths, "labels": bc_labels.value},
            timeout=bc_timeout.value,
        )
        resp.raise_for_status()
        results = resp.json()
        call_time = time.time() - call_start

        yield {
            "hostname": hostname,
            "pid": os.getpid(),
            "num_paths": len(paths),
            "load_time_sec": 0.0,  # no per-executor model load in this path
            "run_time_sec": round(call_time, 3),
            "results": results,
        }

    start = time.time()
    partition_results = paths_rdd.mapPartitions(process_partition).collect()
    elapsed = time.time() - start

    merged_results: Dict[str, Any] = {}
    for pr in partition_results:
        merged_results.update(pr["results"])

    bc_service_url.unpersist()
    bc_labels.unpersist()
    bc_timeout.unpersist()

    return {
        "elapsed_time": round(elapsed, 4),
        "num_files": len(file_paths),
        "num_partitions": effective_partitions,
        "partition_details": [
            {k: v for k, v in pr.items() if k != "results"} for pr in partition_results
        ],
        "results": merged_results,
    }
