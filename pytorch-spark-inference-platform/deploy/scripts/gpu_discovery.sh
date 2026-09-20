#!/usr/bin/env bash
# Spark GPU resource discovery script — referenced by
# spark.worker.resource.gpu.discoveryScript (see
# inference/cluster_engine.py's create_cluster_session(),
# gpu_aware_scheduling=True path, and
# docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md §5).
#
# Spark's standalone worker runs this once at startup and expects exactly
# the JSON shape below on stdout — this is Spark's own documented contract
# for a GPU discovery script, not something specific to this repo.
#
# Volume-mounted (deploy/ is already bind-mounted in every compose file
# that uses it), never baked into any image — this file changing doesn't
# require a rebuild.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo '{"name": "gpu", "addresses": []}'
    exit 0
fi

# Spark expects each address as a JSON STRING (e.g. "0", not 0) — quote
# each nvidia-smi index line before joining.
INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed 's/.*/"&"/' | paste -sd, -)
echo "{\"name\": \"gpu\", \"addresses\": [${INDICES:-}]}"
