FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# This is the BASE image for every model/pipeline in the framework — Python,
# Java, Spark, torch, and the CORE requirements.txt only. Pipeline-specific
# system/Python deps (e.g. ner_translate's tesseract-ocr + gliner/transformers)
# live in their own deploy/Dockerfile.<pipeline>, which FROMs the image built
# here — see docs/MODEL_CONTAINER_ISOLATION.md (Option A) for why.
#
# Install Python 3.11, Java 17, and utilities.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common wget gnupg curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        openjdk-17-jre-headless procps && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
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

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .
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
