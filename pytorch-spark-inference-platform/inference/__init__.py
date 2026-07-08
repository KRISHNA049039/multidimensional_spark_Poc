"""
Inference package — 3 execution modes for multi-model parallel inference.

Modes:
  1. distributed_gpu: Spark + multi-GPU cluster (production)
  2. single_gpu: CUDA streams, all models on one GPU (workstation)
  3. hybrid_cpu_gpu: Memory-aware GPU/CPU split (limited VRAM)
"""
