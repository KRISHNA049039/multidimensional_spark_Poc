#!/bin/bash
# =============================================================================
# build_and_test_ner_translate_gpu.sh — Build + verify ner_translate on a real
# GPU instance, then export the image for transfer back to local/air-gapped.
#
# Sibling to deploy/scripts/setup_and_run_gpu.sh (same S3/SSM pattern, same
# GpuBenchmarkStack) but for docs/TESTING_BEFORE_EXPORT.md's Tiers 2-3
# instead of benchmark/run_benchmark.py. Upload this to S3 and run via SSM,
# same as the existing script. Logs to /opt/benchmark/ner_translate.log.
#
# Expects: BUCKET env var set in /etc/environment (done by CDK UserData)
# =============================================================================
set +e
exec > >(tee -a /opt/benchmark/ner_translate.log) 2>&1

echo "============================================================"
echo "  ner_translate — BUILD + VERIFY + EXPORT ON GPU"
echo "  Started: $(date)"
echo "============================================================"

source /etc/environment
REGION=${AWS_DEFAULT_REGION:-us-east-1}
if [ -z "$BUCKET" ]; then
    BUCKET=$(grep BUCKET /etc/environment 2>/dev/null | cut -d= -f2)
fi

fail() { echo "FATAL: $1"; exit 1; }

# Sync the log to S3 every 20s in the background, so
# `aws s3 cp s3://.../ner_translate.log -` actually shows live progress
# instead of only appearing once the whole script finishes.
( while true; do aws s3 cp /opt/benchmark/ner_translate.log s3://$BUCKET/ner_translate.log --region $REGION >/dev/null 2>&1; sleep 20; done ) &
LOG_SYNC_PID=$!
trap "kill $LOG_SYNC_PID 2>/dev/null" EXIT

# =============================================================================
# STEP 1: Docker + GPU (same as setup_and_run_gpu.sh)
# =============================================================================
echo "" ; echo "=== STEP 1: Docker + GPU ==="
systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true
docker --version || fail "Docker not available"
apt-get install -y unzip jq tesseract-ocr 2>/dev/null || true   # tesseract here too, for a native (non-Docker) sanity check if needed
if ! docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi 2>/dev/null; then
    nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
    systemctl restart docker 2>/dev/null || true
    sleep 3
fi
nvidia-smi || fail "No GPU"

echo "" ; echo "=== STEP 1b: Pulling code from S3 ==="
mkdir -p /opt/benchmark/app
aws s3 cp s3://$BUCKET/project.zip /opt/benchmark/project.zip --region $REGION || fail "project.zip not in S3"
cd /opt/benchmark && rm -rf app/* && unzip -o project.zip -d app
cd /opt/benchmark/app
echo "  Code extracted: $(ls | wc -l) top-level items"

# =============================================================================
# STEP 2: Model weights — download HERE (EC2 has faster/more reliable
# internet than most dev laptops, and models/weights/ is gitignored so
# project.zip never carries them regardless of what's local).
# =============================================================================
echo "" ; echo "=== STEP 2: Model weights ==="
if [ ! -d "models/weights/gliner-multi" ]; then
    # protobuf/tiktoken: not hard deps of gliner/transformers, but loading
    # gliner_multi-v2.1's tokenizer needs one of them — see the same note
    # in models/pipelines/ner_translate/requirements.txt (caught on the
    # first real run of this exact script: "tiktoken is required...").
    pip install -q gliner transformers torch sentencepiece py3langid protobuf tiktoken 2>&1 | tail -3
    # NOT `python` — this bare Deep Learning AMI host only has python3, no
    # `python` alias (that alias is set up inside the Docker image via
    # update-alternatives, not on the host). Caught this exact failure on
    # the first real run: "python: command not found".
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
# STEP 3: Build base image (Tier 2 start)
# =============================================================================
echo "" ; echo "=== STEP 3: Building base image ==="
docker build --network host -t multi-model-inference:latest -f deploy/Dockerfile . || fail "base image build failed"

echo "  Testing GPU inside base image..."
BASE_GPU=$(docker run --rm --gpus all multi-model-inference:latest python -c "import torch; print(f'CUDA:{torch.cuda.is_available()} GPU:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')")
echo "  $BASE_GPU"
echo "$BASE_GPU" | grep -q "CUDA:True" || echo "  WARNING: CUDA not available in base image — continuing anyway (ner_translate falls back to CPU)"

# =============================================================================
# STEP 4: Wheelhouse + ner_translate image
# =============================================================================
echo "" ; echo "=== STEP 4: Wheelhouse + ner_translate image ==="
# Reuse the SAME wheelhouse script Dockerfile.ner_translate is built and
# tested against locally — a simplified inline `pip download` here (this
# script's first version) was WRONG: it assumed "native Linux build" meant
# no cross-platform targeting was needed, but the bare EC2 host's Python
# (3.10) differs from the Docker image's Python (3.11, via deadsnakes) —
# "native to the host" isn't "native to the container". That mismatch
# silently produced a wheelhouse missing sentencepiece==0.2.0 (no matching
# wheel for the host's Python), and pip download's own exit code stayed 0,
# so the broken wheelhouse wasn't caught until the actual Docker build's
# pip install step failed. One shared, already-correct script now, not two.
bash deploy/scripts/build_ner_translate_wheelhouse.sh || fail "wheelhouse build failed"

docker build --network host -t ner-translate-worker:latest -f deploy/Dockerfile.ner_translate . || fail "ner_translate image build failed"

echo "" ; echo "--- Tier 2 pass criteria checks ---"
docker run --rm ner-translate-worker:latest python --version
docker run --rm ner-translate-worker:latest python -c "import gliner, transformers, sentencepiece, py3langid; print('deps OK')" || fail "ner_translate deps not installed correctly"
docker run --rm ner-translate-worker:latest tesseract --list-langs
NER_GPU=$(docker run --rm --gpus all ner-translate-worker:latest python -c "import torch; print(f'CUDA:{torch.cuda.is_available()} torch:{torch.__version__}')")
echo "  $NER_GPU"

# =============================================================================
# STEP 5: Cluster + real job submission (Tier 3)
# =============================================================================
echo "" ; echo "=== STEP 5: Cluster + job submission ==="
docker rm -f ner-translate-master ner-translate-worker 2>/dev/null || true
MASTER_IP=$(hostname -I | awk '{print $1}')

docker run -d --name ner-translate-master --network host --gpus all --shm-size=4g \
  -e SPARK_MODE=master -e SPARK_MASTER_HOST=$MASTER_IP \
  -v /opt/benchmark/app/results:/app/results \
  ner-translate-worker:latest \
  bash -c "/usr/local/bin/apply_wheels_hotfix.sh && \$SPARK_HOME/sbin/start-master.sh -h $MASTER_IP && tail -f \$SPARK_HOME/logs/*master*"

sleep 10
echo "  --- hotfix no-op check ---"
docker logs ner-translate-master 2>&1 | grep "wheels-hotfix"

# -c 4, not 2: create_cluster_session() (inference/cluster_engine.py)
# defaults executor_cores=4 — a worker offering fewer cores than an
# executor demands can never satisfy that resource request, and Spark
# standalone doesn't error out when that happens, it just waits forever
# with 0 executors (exactly what "Stage 0: 0/2" hanging indefinitely with
# no crash and no error message turned out to mean). g4dn.xlarge has 4
# vCPUs, so this isn't over-provisioning, just matching what's actually
# being asked for.
docker run -d --name ner-translate-worker --network host --gpus all --shm-size=4g \
  -e SPARK_MODE=worker -e SPARK_MASTER=spark://$MASTER_IP:7077 \
  -v /opt/benchmark/app/results:/app/results \
  ner-translate-worker:latest \
  bash -c "/usr/local/bin/apply_wheels_hotfix.sh && \$SPARK_HOME/sbin/start-worker.sh spark://$MASTER_IP:7077 -c 4 -m 8g && tail -f \$SPARK_HOME/logs/*worker*"

sleep 10
WORKERS=$(docker exec ner-translate-master curl -s http://localhost:8080/json/ | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('workers',[])))" 2>/dev/null || echo "0")
echo "  Workers registered: $WORKERS"

echo "" ; echo "  Submitting real job (includes OCR test image sample_scan6.png) ..."
# Absolute path, not "data/ner_samples" — on a REAL distributed cluster
# (unlike local[4] testing, where driver and "executor" share one process
# and CWD), each Spark executor task runs from its own per-application work
# directory (/opt/spark/work/app-.../<id>/), not /app. A relative path
# resolves against THAT directory and fails with "No such file or
# directory" for every single file — the job "succeeds" at the Spark level
# (no crash) but produces nothing but per-file errors. The image bakes the
# whole repo into /app at build time, so the absolute path always exists
# regardless of which directory a task happens to run from. Caught only by
# actually reading the job's per-file results, not by the job merely
# completing without an exception.
docker exec ner-translate-master bash -c \
  "SPARK_MASTER_URL=spark://$MASTER_IP:7077 python submit_pipeline_job.py --pipeline ner_translate --input /app/data/ner_samples --partitions 2"

docker cp ner-translate-master:/app/results/. /opt/benchmark/app/results/ 2>/dev/null || true

# =============================================================================
# STEP 6: Export + upload the verified image
# =============================================================================
echo "" ; echo "=== STEP 6: Exporting verified image ==="
docker save ner-translate-worker:latest | gzip > /opt/benchmark/ner-translate-worker.tar.gz
ls -lh /opt/benchmark/ner-translate-worker.tar.gz
aws s3 cp /opt/benchmark/ner-translate-worker.tar.gz s3://$BUCKET/images/ner-translate-worker.tar.gz --region $REGION

echo "" ; echo "=== STEP 7: Syncing results + log ==="
aws s3 sync /opt/benchmark/app/results/ s3://$BUCKET/results/ --region $REGION
aws s3 cp /opt/benchmark/ner_translate.log s3://$BUCKET/ner_translate.log --region $REGION

echo ""
echo "============================================================"
echo "  DONE — image at s3://$BUCKET/images/ner-translate-worker.tar.gz"
echo "  Finished: $(date)"
echo "============================================================"
