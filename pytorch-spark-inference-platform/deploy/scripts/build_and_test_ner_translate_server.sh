#!/bin/bash
# =============================================================================
# build_and_test_ner_translate_server.sh — build + verify the waiter/kitchen
# split (docs/MODEL_CONTAINER_ISOLATION.md Option B) on a real GPU instance,
# then export both images for transfer back to local/air-gapped.
#
# Sibling to build_and_test_ner_translate_gpu.sh (Option A, in-process) but
# builds spark-lean (no torch/CUDA at all) + ner-translate-server (the
# kitchen — torch/CUDA/model deps, no Spark/Java) instead of the single
# fat ner-translate-worker image. Logs to /opt/benchmark/ner_translate_server.log.
#
# Expects: BUCKET env var set in /etc/environment (done by CDK UserData)
# =============================================================================
set +e
exec > >(tee -a /opt/benchmark/ner_translate_server.log) 2>&1

echo "============================================================"
echo "  ner_translate SERVER SPLIT — BUILD + VERIFY + EXPORT ON GPU"
echo "  Started: $(date)"
echo "============================================================"

source /etc/environment
REGION=${AWS_DEFAULT_REGION:-us-east-1}
if [ -z "$BUCKET" ]; then
    BUCKET=$(grep BUCKET /etc/environment 2>/dev/null | cut -d= -f2)
fi

fail() { echo "FATAL: $1"; exit 1; }

( while true; do aws s3 cp /opt/benchmark/ner_translate_server.log s3://$BUCKET/ner_translate_server.log --region $REGION >/dev/null 2>&1; sleep 20; done ) &
LOG_SYNC_PID=$!
trap "kill $LOG_SYNC_PID 2>/dev/null" EXIT

# =============================================================================
# STEP 1: Docker + GPU
# =============================================================================
echo "" ; echo "=== STEP 1: Docker + GPU ==="
systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true
docker --version || fail "Docker not available"
nvidia-smi || fail "No GPU"

echo "" ; echo "=== STEP 1b: Pulling code from S3 ==="
mkdir -p /opt/benchmark/app
aws s3 cp s3://$BUCKET/project.zip /opt/benchmark/project.zip --region $REGION || fail "project.zip not in S3"
cd /opt/benchmark && rm -rf app/* && unzip -o project.zip -d app
cd /opt/benchmark/app
echo "  Code extracted: $(ls | wc -l) top-level items"

# =============================================================================
# STEP 2: Model weights — still needed even though images no longer bake
# them in, since the "kitchen" mounts models/weights/ from this filesystem
# at container start (docs/MODEL_CONTAINER_ISOLATION.md Option B).
# =============================================================================
echo "" ; echo "=== STEP 2: Model weights ==="
if [ ! -d "models/weights/gliner-multi" ]; then
    pip install -q gliner transformers torch sentencepiece py3langid protobuf tiktoken 2>&1 | tail -3
    python3 -c "
from gliner import GLiNER
GLiNER.from_pretrained('urchade/gliner_multi-v2.1').save_pretrained('models/weights/gliner-multi')
" || fail "GLiNER weights download failed"
fi
if [ ! -d "models/weights/nllb-200-distilled-600M" ]; then
    python3 -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
m='facebook/nllb-200-distilled-600M'
AutoTokenizer.from_pretrained(m).save_pretrained('models/weights/nllb-200-distilled-600M')
AutoModelForSeq2SeqLM.from_pretrained(m).save_pretrained('models/weights/nllb-200-distilled-600M')
" || fail "NLLB weights download failed"
fi
echo "  Weights ready: $(du -sh models/weights/gliner-multi models/weights/nllb-200-distilled-600M 2>/dev/null)"

# =============================================================================
# STEP 3: Build the lean waiter image (no torch/CUDA at all — .dockerignore
# keeps models/weights/ out of every image regardless of what's on disk
# at build time).
# =============================================================================
echo "" ; echo "=== STEP 3: Building spark-lean (waiter) image ==="
docker build --network host --target lean -t spark-lean:latest -f deploy/Dockerfile . || fail "spark-lean build failed"
echo "  Confirming torch is NOT installed in spark-lean (that's the point)..."
docker run --rm spark-lean:latest python -c "import torch" 2>&1 | grep -q "ModuleNotFoundError" \
    && echo "  Confirmed: spark-lean has no torch installed" \
    || fail "spark-lean unexpectedly has torch installed — waiter/kitchen split is broken"

# =============================================================================
# STEP 4: Wheelhouse + the kitchen image
# =============================================================================
echo "" ; echo "=== STEP 4: Wheelhouse + ner-translate-server (kitchen) image ==="
bash deploy/scripts/build_ner_translate_wheelhouse.sh || fail "wheelhouse build failed"
docker build --network host -t ner-translate-server:latest -f deploy/Dockerfile.ner_translate_server . || fail "ner-translate-server build failed"

echo "" ; echo "--- kitchen pass criteria checks ---"
docker run --rm ner-translate-server:latest python3 --version
docker run --rm ner-translate-server:latest python3 -c "import gliner, transformers, fastapi, uvicorn; print('deps OK')" || fail "kitchen deps not installed correctly"
KITCHEN_GPU=$(docker run --rm --gpus all ner-translate-server:latest python3 -c "import torch; print(f'CUDA:{torch.cuda.is_available()} torch:{torch.__version__}')")
echo "  $KITCHEN_GPU"
echo "$KITCHEN_GPU" | grep -q "CUDA:True" || fail "kitchen image cannot see the GPU"

echo "" ; echo "--- image sizes ---"
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "spark-lean|ner-translate-server"

# =============================================================================
# STEP 5: Bring up the split cluster (docker compose build + up)
# =============================================================================
echo "" ; echo "=== STEP 5: docker compose up (waiter + kitchen) ==="
docker compose -f deploy/docker-compose.ner_translate_server.yml down 2>/dev/null || true
docker compose -f deploy/docker-compose.ner_translate_server.yml up -d --build || fail "compose up failed"

echo "  Waiting for kitchen health check (model load can take a minute)..."
for i in $(seq 1 24); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' ner-translate-server 2>/dev/null)
    echo "  [$i/24] ner-translate-server health: $STATUS"
    [ "$STATUS" = "healthy" ] && break
    sleep 10
done
[ "$STATUS" = "healthy" ] || fail "kitchen never became healthy — check docker logs ner-translate-server"

sleep 5
WORKERS=$(docker exec ner-translate-master curl -s http://localhost:8080/json/ | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('workers',[])))" 2>/dev/null || echo "0")
echo "  Workers registered: $WORKERS"

# =============================================================================
# STEP 6: Real job submission — must run through the SERVICE path, since
# the master/worker containers (spark-lean) literally have no torch
# installed; the job can only succeed by calling the kitchen over HTTP.
# =============================================================================
echo "" ; echo "=== STEP 6: Job submission (must route through the kitchen) ==="
docker exec ner-translate-master bash -c \
  "python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2" \
  || fail "job submission failed"

docker cp ner-translate-master:/app/results/. /opt/benchmark/app/results/ 2>/dev/null || true

# =============================================================================
# STEP 7: Export + upload both images
# =============================================================================
echo "" ; echo "=== STEP 7: Exporting verified images ==="
docker save spark-lean:latest | gzip > /opt/benchmark/spark-lean.tar.gz
ls -lh /opt/benchmark/spark-lean.tar.gz
aws s3 cp /opt/benchmark/spark-lean.tar.gz s3://$BUCKET/images/spark-lean.tar.gz --region $REGION

docker save ner-translate-server:latest | gzip > /opt/benchmark/ner-translate-server.tar.gz
ls -lh /opt/benchmark/ner-translate-server.tar.gz
aws s3 cp /opt/benchmark/ner-translate-server.tar.gz s3://$BUCKET/images/ner-translate-server.tar.gz --region $REGION

echo "" ; echo "=== STEP 8: Syncing results + log ==="
aws s3 sync /opt/benchmark/app/results/ s3://$BUCKET/results/ --region $REGION
aws s3 cp /opt/benchmark/ner_translate_server.log s3://$BUCKET/ner_translate_server.log --region $REGION

echo ""
echo "============================================================"
echo "  DONE"
echo "  spark-lean:            s3://$BUCKET/images/spark-lean.tar.gz"
echo "  ner-translate-server:  s3://$BUCKET/images/ner-translate-server.tar.gz"
echo "  Finished: $(date)"
echo "============================================================"
