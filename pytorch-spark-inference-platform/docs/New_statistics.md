spark-master
root@ip-10-0-0-175:/opt/benchmark/app# docker run --rm --gpus all --shm-size=4g \
  -v /opt/benchmark/app/results:/app/results \
  multi-model-inference:latest \
  python benchmark/spark_vs_single_gpu.py --local --signals 10000 --images 200 --detections 50 --batch-size 32

==========
== CUDA ==
==========

CUDA Version 12.6.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.


================================================================================
  SPARK vs SINGLE GPU — SCALABILITY PROOF BENCHMARK
================================================================================
  CUDA: True
  GPU: Tesla T4
  VRAM: 15.6 GB
  CPU cores: 4

  Loading 10 models...
  Loaded 10 models
  Generating data: 10000 signals, 200 images, 50 detections
  Total data: 878.4 MB

================================================================================
  PHASE 1: SINGLE GPU BASELINES (No Spark)
================================================================================

  [SINGLE-GPU SEQUENTIAL] Device: cuda

  ───────────────────────────────────────────────────────────────────────────
  MODEL INPUT/OUTPUT DETAILS
  ───────────────────────────────────────────────────────────────────────────
  Model                  Input Shape        Output Shape       Throughput   Device
  ───────────────────────────────────────────────────────────────────────────
  ew_classifier          [32, 128]          [32, 8]              31,720/s   cuda
  signal_denoiser        [32, 128]          [32, 128]           109,331/s   cuda
  threat_prioritizer     [32, 128]          [32]                 36,737/s   cuda
  rf_fingerprinter       [32, 128]          [32, 32]             14,943/s   cuda
  anomaly_detector       [32, 128]          [32]                 84,938/s   cuda
  resnet18               [32, 3, 224, 224]  [32, 1000]              909/s   cuda
  mobilenetv3            [32, 3, 224, 224]  [32, 1000]              842/s   cuda
  efficientnet_b0        [32, 3, 224, 224]  [32, 1000]              665/s   cuda
  yolov8_nano            [32, 3, 640, 640]  [32, 400]               691/s   cuda
  yolov8_small           [32, 3, 640, 640]  [32, 400]               571/s   cuda
  ───────────────────────────────────────────────────────────────────────────

  [SINGLE-GPU PARALLEL STREAMS] Device: cuda

================================================================================
  FINAL COMPARISON: SINGLE GPU vs SPARK CLUSTER
================================================================================

  Config                              Throughput      Time       Speedup    Verdict
  ─────────────────────────────────── ─────────────── ────────── ────────── ────────────
  single_gpu_sequential                   21,265/s     2.38s    1.00×
  single_gpu_parallel_streams             38,365/s     1.32s    1.80×   ★ BEST
  ────────────────────────────────────────