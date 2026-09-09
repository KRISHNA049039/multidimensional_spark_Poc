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
from datetime import datetime

from inference.cluster_engine import create_cluster_session
from inference.text_pipeline_engine import run_text_pipeline_job

MANIFEST_PATH = os.path.join("models", "pipelines", "manifest.json")


def _load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


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
    parser.add_argument("--master", default=None, help="Spark master URL override (default: env or local[4])")
    args = parser.parse_args()

    manifest = _load_manifest()
    if args.pipeline not in manifest:
        available = ", ".join(sorted(manifest.keys()))
        raise SystemExit(f"Unknown pipeline '{args.pipeline}'. Available: {available}")

    mod = importlib.import_module(manifest[args.pipeline]["module"])

    paths = _collect_files(args.input)
    if not paths:
        raise SystemExit(f"No input files found at {args.input}")

    spark = create_cluster_session(app_name=f"pipeline-{args.pipeline}", master_url=args.master)
    try:
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
