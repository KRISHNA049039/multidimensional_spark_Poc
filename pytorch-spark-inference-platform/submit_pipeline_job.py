"""
Pipeline submit CLI — run a "pipeline" style plugin (multi-model, file/text
input and output, e.g. the NER+translation pipeline) as a distributed Spark
job. Sibling to submit_job.py, which is for single-tensor-model plugins.

Usage:
    python submit_pipeline_job.py --pipeline ner_translate --input data/ner_samples --partitions 2
"""

import argparse
import glob
import importlib
import json
import os
import socket
from datetime import datetime
from urllib.parse import urlparse

from inference.cluster_engine import create_cluster_session
from inference.text_pipeline_engine import run_text_pipeline_job, run_text_pipeline_job_via_service

MANIFEST_PATH = os.path.join("models", "pipelines", "manifest.json")


def _load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _host_resolvable(spark_url: str, timeout: float = 1.0) -> bool:
    """Best-effort check that a spark://host:port URL's host actually
    resolves. Lets us default to a pipeline's own dedicated cluster
    (manifest "master_url") only when we're really running inside that
    cluster's Docker network, and fall back to local[4] everywhere else
    (bare local dev, a shell on the host, the shared cluster's containers)
    without the job just failing to connect.

    Uses socket.setdefaulttimeout() to bound the DNS lookup — but that's a
    GLOBAL, process-wide default, not scoped to this one call. Left set,
    it silently poisons every socket created afterward for the rest of the
    process — including py4j's own Python<->JVM gateway socket, whose reads
    during real standalone-cluster SparkContext initialization can
    legitimately take longer than this function's 1s default. Manifested
    as SparkContext() hanging then failing with a raw socket
    `TimeoutError: timed out` deep inside py4j, with no obvious connection
    to this function at all — restoring the previous default afterward
    (whatever it was, usually None/blocking) is what actually matters here,
    not the specific 1s value.
    """
    previous_timeout = socket.getdefaulttimeout()
    try:
        host, port = urlparse(spark_url).hostname, urlparse(spark_url).port
        if not host:
            return False
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(host, port or 7077)
        return True
    except OSError:
        return False
    finally:
        socket.setdefaulttimeout(previous_timeout)


def _resolve_master_url(cli_master: str | None, manifest_entry: dict) -> str | None:
    """Precedence: explicit --master > SPARK_MASTER_URL/SPARK_MASTER env var
    (create_cluster_session's own fallback — respected here so it still wins,
    matching how the shared cluster's docs already pass it via `docker exec
    ... bash -c "SPARK_MASTER_URL=... python ..."`) > this pipeline's own
    dedicated cluster (manifest "master_url"), but only if that host actually
    resolves > local[4].
    """
    if cli_master:
        return cli_master
    if os.environ.get("SPARK_MASTER_URL") or os.environ.get("SPARK_MASTER"):
        return None  # let create_cluster_session pick up the env var itself
    manifest_master = manifest_entry.get("master_url")
    if manifest_master and _host_resolvable(manifest_master):
        return manifest_master
    return None  # create_cluster_session's own default: local[4]


def _collect_files(input_path: str):
    if os.path.isdir(input_path):
        return sorted(
            p for p in glob.glob(os.path.join(input_path, "**", "*"), recursive=True)
            if os.path.isfile(p)
        )
    if any(c in input_path for c in "*?["):
        return sorted(glob.glob(input_path))
    return [input_path]


def main():
    parser = argparse.ArgumentParser(description="Submit a pipeline plugin as a distributed Spark job")
    parser.add_argument("--pipeline", required=True, help="Registered pipeline name (models/pipelines/manifest.json)")
    parser.add_argument("--input", required=True, help="File, directory, or glob of input documents")
    parser.add_argument("--partitions", type=int, default=2)
    parser.add_argument("--mode", default="hybrid", choices=["cpu_only", "gpu_only", "hybrid"],
                         help="Informational only today - pipeline plugins pick their own device internally")
    parser.add_argument("--master", default=None,
                         help="Spark master URL override (default: this pipeline's manifest "
                              "master_url if set, else env var, else local[4])")
    parser.add_argument("--driver-memory", default="6g",
                         help="Spark driver JVM heap (default 6g, sized for a real cluster "
                              "node — lower this on a small/shared instance, e.g. a single "
                              "g4dn.xlarge running master+worker+kitchen together, where the "
                              "default can starve the JVM gateway during SparkContext init)")
    parser.add_argument("--executor-memory", default="4g",
                         help="Spark executor JVM heap (default 4g, see --driver-memory)")
    args = parser.parse_args()

    manifest = _load_manifest()
    if args.pipeline not in manifest:
        available = ", ".join(sorted(manifest.keys()))
        raise SystemExit(f"Unknown pipeline '{args.pipeline}'. Available: {available}")

    paths = _collect_files(args.input)
    if not paths:
        raise SystemExit(f"No input files found at {args.input}")

    master_url = _resolve_master_url(args.master, manifest[args.pipeline])
    spark = create_cluster_session(
        app_name=f"pipeline-{args.pipeline}", master_url=master_url,
        driver_memory=args.driver_memory, executor_memory=args.executor_memory,
    )

    # waiter/kitchen split (docs/MODEL_CONTAINER_ISOLATION.md Option B): if
    # this pipeline has a "service_url" in the manifest AND that host
    # actually resolves (same pattern _resolve_master_url already uses for
    # master_url — lets the identical command work unmodified both in plain
    # local dev, where the service doesn't exist, and inside the
    # server-enabled compose cluster, where it does), route through the
    # HTTP model server instead of importing the pipeline module in-process.
    # Deliberately NOT importing the pipeline module until we know we need
    # it (the else branch below): on the lean/waiter image the module's own
    # import chain pulls in torch/transformers, which that image doesn't
    # have installed on purpose — importing it unconditionally here would
    # crash the driver before it ever got a chance to use the service path.
    service_url = manifest[args.pipeline].get("service_url")
    use_service = bool(service_url) and _host_resolvable(service_url)
    print(f"[submit_pipeline_job] execution path: "
          f"{'HTTP model server at ' + service_url if use_service else 'in-process (mod.load/mod.run)'}")
    try:
        if use_service:
            result = run_text_pipeline_job_via_service(
                spark, paths, service_url,
                num_partitions=args.partitions,
            )
        else:
            mod = importlib.import_module(manifest[args.pipeline]["module"])
            result = run_text_pipeline_job(
                spark, paths, mod.load, mod.run,
                num_partitions=args.partitions,
            )
    finally:
        spark.stop()

    summary = {k: v for k, v in result.items() if k != "results"}
    print(json.dumps(summary, indent=2, default=str))

    print(f"\nProcessed {len(result['results'])} document(s):")
    for name, data in result["results"].items():
        if data.get("error"):
            print(f"  {name}: ERROR {data['error']}")
        else:
            print(f"  {name}: lang={data['language']} translated={data['translated']} "
                  f"entities={len(data['entities_unique'])}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{args.pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nFull results (including extracted entities) written to {out_path}")


if __name__ == "__main__":
    main()
