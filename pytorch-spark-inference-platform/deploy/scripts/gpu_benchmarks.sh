#!/bin/bash
# GPU Benchmark Phases 6-10 — Static script (avoids PowerShell escaping)
# This file is uploaded to S3 and run on EC2 via SSM.
# MASTER_IP is replaced by sed before upload.

MASTER_IP=__MASTER_IP__
BUCKET=__BUCKET__
REGION=__REGION__

echo '=== PHASE 6: Cluster Benchmark CPU ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 CUDA_VISIBLE_DEVICES='' python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 3000 --batch-size 128" || echo 'P6.1 done'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 CUDA_VISIBLE_DEVICES='' python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 8 --signal-samples 3000 --batch-size 128" || echo 'P6.2 done'
sleep 3

echo '=== PHASE 7: GPU Tests ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 2 --signal-samples 1000 --image-samples 50 --detection-samples 20 --batch-size 128" || echo 'P7.1 done'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 256" || echo 'P7.2 done'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 5000 --image-samples 200 --detection-samples 50 --batch-size 256" || echo 'P7.3 done'
sleep 3

echo '=== PHASE 7b: Hybrid Tests ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 256" || echo 'P7b.1 done'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 8 --signal-samples 5000 --image-samples 200 --detection-samples 50 --batch-size 256" || echo 'P7b.2 done'
sleep 3

echo '=== PHASE 8: GPU Batch Size ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 64" || echo 'P8 bs=64 done'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 128" || echo 'P8 bs=128 done'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 256" || echo 'P8 bs=256 done'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 512" || echo 'P8 bs=512 done'
sleep 3

echo '=== PHASE 9: Incremental Load ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/incremental_load_test.py" || echo 'P9 done'
sleep 3

echo '=== PHASE 10: Full Incremental ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --incremental" || echo 'P10 done'

echo '=== Syncing results ==='
docker cp spark-master:/app/results/. /opt/spark-inference/app/results/
aws s3 sync /opt/spark-inference/app/results/ s3://${BUCKET}/results/ --region ${REGION}
echo '=== ALL GPU BENCHMARKS COMPLETE ==='
