#!/usr/bin/env bash
# =============================================================================
# One-command, path-independent bring-up for the ner_translate waiter/kitchen
# cluster (deploy/docker-compose.ner_translate_server.yml). Written because a
# plain `cd deploy && docker compose up` broke on a real airgapped deployment:
# the compose file's `../models` mount is relative to wherever you happen to
# run it from, and when that's wrong Docker silently bind-mounts an empty
# directory instead of erroring — which then fails three layers deep inside
# Python as `ModuleNotFoundError: No module named 'models'`, with nothing in
# the error pointing back to "you extracted the tarballs into the wrong
# place." This script checks the actual layout BEFORE starting any
# container, and refuses to proceed with a specific, actionable message if
# anything's missing.
#
# Works identically on an internet-connected dev machine and on an airgapped
# system — it never touches the network itself, only validates the local
# filesystem, sets NER_TRANSLATE_ROOT absolutely (see the compose file's own
# header comment), and calls `docker compose up`.
#
# Usage:
#   bash deploy/scripts/setup_ner_translate_server.sh [ROOT_DIR]
#
# ROOT_DIR defaults to this script's own repo root (two levels up from
# deploy/scripts/) — pass it explicitly if you extracted the code/weights
# tarballs somewhere else and don't want to move them.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${1:-$DEFAULT_ROOT}"

fail=0
check() {
    local desc="$1" path="$2"
    if [ -e "$path" ]; then
        echo "  [OK]   $desc"
    else
        echo "  [MISSING] $desc"
        echo "         expected at: $path"
        fail=1
    fi
}

echo "Checking layout under: $ROOT"
echo ""
echo "-- from ner-translate-code.tar.gz --"
check "deploy/docker-compose.ner_translate_server.yml" "$ROOT/deploy/docker-compose.ner_translate_server.yml"
check "models/__init__.py"                              "$ROOT/models/__init__.py"
check "models/pipelines/ner_translate/serve.py"          "$ROOT/models/pipelines/ner_translate/serve.py"
check "models/pipelines/manifest.json"                   "$ROOT/models/pipelines/manifest.json"
check "inference/text_pipeline_engine.py"                "$ROOT/inference/text_pipeline_engine.py"
check "submit_pipeline_job.py"                            "$ROOT/submit_pipeline_job.py"
check "data/ner_samples/"                                 "$ROOT/data/ner_samples"
echo ""
echo "-- from ner-translate-weights.tar.gz (must be merged into the SAME models/ tree above) --"
check "models/weights/gliner-multi/"                      "$ROOT/models/weights/gliner-multi"
check "models/weights/nllb-200-distilled-600M/"           "$ROOT/models/weights/nllb-200-distilled-600M"
echo ""
echo "-- docker images (docker load -i <name>.tar.gz) --"
if docker image inspect spark-lean:latest >/dev/null 2>&1; then
    echo "  [OK]   spark-lean:latest"
else
    echo "  [MISSING] spark-lean:latest — run: docker load -i spark-lean.tar.gz"
    fail=1
fi
if docker image inspect ner-translate-server:latest >/dev/null 2>&1; then
    echo "  [OK]   ner-translate-server:latest"
else
    echo "  [MISSING] ner-translate-server:latest — run: docker load -i ner-translate-server.tar.gz"
    fail=1
fi

echo ""
if [ "$fail" -ne 0 ]; then
    echo "One or more required paths/images are missing — see [MISSING] lines above."
    echo "Not starting any container. Fix the layout, then re-run this script."
    exit 1
fi

mkdir -p "$ROOT/results"
echo "Layout OK. Starting the cluster (NER_TRANSLATE_ROOT=$ROOT)..."
NER_TRANSLATE_ROOT="$ROOT" docker compose -f "$ROOT/deploy/docker-compose.ner_translate_server.yml" up -d

echo ""
echo "Waiting for the kitchen (ner-translate-server) to report healthy..."
for i in $(seq 1 15); do
    status="$(docker inspect --format='{{.State.Health.Status}}' ner-translate-server 2>/dev/null || echo unknown)"
    echo "  [$i/15] ner-translate-server: $status"
    [ "$status" = "healthy" ] && break
    sleep 5
done

echo ""
echo "Submit a job with:"
echo "  NER_TRANSLATE_ROOT=$ROOT docker exec ner-translate-master bash -c \\"
echo "    \"python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2\""
