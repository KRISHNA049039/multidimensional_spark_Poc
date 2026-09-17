# syntax=docker/dockerfile:1
# =============================================================================
# Multi-stage: `spark-base` (Java+Spark+Python, no torch/CUDA at all) is the
# stage that "just exposes CPU/GPU resources to containers" — the Spark
# master, and any future purely-orchestration worker role, never needs torch
# loaded and can build/run from `--target spark-base` alone (small, no GPU
# deps). The default final stage layers torch/CUDA on top of it and is what
# every model/pipeline image (`deploy/Dockerfile.<pipeline>`) FROMs — see
# docs/MODEL_CONTAINER_ISOLATION.md (Option A) for why pipelines get their
# own image on top of this one.
#
# No `nvidia/cuda` base anywhere in this file, by design: the `cu126` wheel
# below (`--index-url https://download.pytorch.org/whl/cu126`) already
# bundles its own CUDA *runtime* shared libraries as pip deps
# (nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, etc.) — an
# `nvidia/cuda` base would just duplicate several GB of the same libraries.
# GPU access at container-run time (`--gpus all`) comes from the NVIDIA
# Container Toolkit injecting the host's driver (libcuda.so + device nodes),
# which is orthogonal to what CUDA runtime bits the image itself carries.
# =============================================================================
FROM ubuntu:22.04 AS spark-base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11, Java 17, and utilities. software-properties-common/
# gnupg are only needed to register the deadsnakes PPA — purged in the same
# layer once python3.11 is installed so they don't sit in any image built
# from this stage.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common wget gnupg curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        openjdk-17-jre-headless procps && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    apt-get purge -y --auto-remove software-properties-common gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install Apache Spark 3.5.1
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
ENV PYSPARK_PYTHON=python
ENV PYSPARK_DRIVER_PYTHON=python

RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
        -O /tmp/spark.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm /tmp/spark.tgz

WORKDIR /app

# =============================================================================
# AI/model base — FROM spark-base, adds torch + this framework's CORE
# requirements.txt only. This is the "AI image owns the CUDA deps" tier:
# `deploy/Dockerfile.<pipeline>` FROMs the image built by THIS stage, never
# spark-base directly, since every pipeline needs torch.
# =============================================================================
FROM spark-base AS final

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements.txt

# Copy source — not `COPY . .`; see the `lean` stage's comment below for why
# wheels/ and debs/ (each pipeline's wheelhouse and .deb bundle) must be
# skipped here too.
RUN --mount=type=bind,source=.,target=/build-context \
    find /build-context -mindepth 1 -maxdepth 1 ! -name wheels ! -name debs ! -name .git \
        -exec cp -a {} /app/ \;
RUN mkdir -p results

# Download pre-trained model weights to torch cache
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q https://download.pytorch.org/models/resnet18-f37072fd.pth \
        -O /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth && \
    wget -q https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth \
        -O /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth && \
    wget -q https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth \
        -O /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth

CMD ["python", "benchmark/run_benchmark.py"]

# =============================================================================
# "lean" — the waiter/kitchen split (docs/MODEL_CONTAINER_ISOLATION.md
# Option B): FROM spark-base directly, NOT from `final` above — this stage
# never installs torch, CUDA, or any pipeline's model deps at all. It's
# Spark + just enough Python (pyspark, for both the driver and the
# py4j-based worker task runner, plus requests) to call a model server over
# HTTP instead of loading a model in-process. Build with:
#   docker build --target lean -t spark-lean:latest -f deploy/Dockerfile .
# This is the image every "kitchen"-backed pipeline's Spark master/worker
# containers should run on — it's the SAME image regardless of how many
# model servers exist, since it has no model-specific content to isolate.
# =============================================================================
FROM spark-base AS lean

RUN pip install --no-cache-dir pyspark==3.5.1 requests==2.34.2

# Not `COPY . .`: every pipeline's wheelhouse (wheels/<pipeline>/, GBs of
# torch/CUDA .whl files this stage never installs) and .deb bundle
# (debs/<pipeline>/) live in the same build context and can't be excluded
# via .dockerignore without also hiding them from the targeted bind-mounts
# in deploy/Dockerfile.ner_translate{,_server} that DO need them (see
# .dockerignore's note). A bind-mount + explicit copy here skips both
# without ever materializing them as an image layer either way — unlike
# `COPY . .` followed by `RUN rm -rf wheels debs`, which would still ship
# those bytes in the COPY layer.
RUN --mount=type=bind,source=.,target=/build-context \
    find /build-context -mindepth 1 -maxdepth 1 ! -name wheels ! -name debs ! -name .git \
        -exec cp -a {} /app/ \;
RUN mkdir -p results

CMD ["bash"]
