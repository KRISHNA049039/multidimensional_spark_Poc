"""
Mode 2: Single GPU Inference (All Models Parallel via CUDA Streams)

All 10 models loaded on one GPU. Uses CUDA streams to execute them
concurrently. Best throughput per GPU — no IPC or network overhead.

Architecture:
  1. Load all models onto single GPU
  2. Create N CUDA streams (one per model)
  3. For each data batch: launch all models on separate streams
  4. GPU kernel scheduler interleaves execution
  5. Synchronize and collect outputs

Works in:
  - Local mode with single GPU (primary use case)
  - Cluster mode: one executor = one GPU = this function
"""

import sys
import os
import time
import numpy as np
import torch
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.cuda_streams_engine import CUDAStreamsEngine
from inference.gpu_memory_manager import GPUMemoryManager


def run_single_gpu_inference(
    models: Dict[str, torch.nn.Module],
    data: Dict[str, np.ndarray],
    batch_size: int = 256,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Run all models in parallel on a single GPU using CUDA streams.

    If GPU not available, falls back to sequential CPU execution.

    Args:
        models: {model_name: model_instance} (can be on any device, will be moved)
        data: {model_name: numpy input array}
        batch_size: Samples per batch per model
        device: Target device ("cuda" or "cuda:0" or "cpu")

    Returns:
        Dict with timing, throughput, per-model metrics
    """
    # Fallback to CPU if CUDA not available
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
        print("  [WARN] CUDA not available, falling back to CPU")

    # Move all models to device
    device_map = {}
    placed_models = {}
    for name, model in models.items():
        model = model.to(device).eval()
        placed_models[name] = model
        device_map[name] = device

    # Create CUDA streams engine
    engine = CUDAStreamsEngine(placed_models, device_map, batch_size)

    # Determine number of batches (based on largest dataset)
    max_samples = max(len(arr) for arr in data.values())
    num_batches = (max_samples + batch_size - 1) // batch_size

    # Warmup (important for accurate GPU timing)
    warmup_inputs = {}
    for name, arr in data.items():
        warmup_size = min(batch_size, len(arr))
        warmup_inputs[name] = torch.from_numpy(arr[:warmup_size].copy()).float()

    for _ in range(3):
        _ = engine.infer_all_parallel(warmup_inputs)

    # Timed inference
    total_processed = {name: 0 for name in models}
    batch_latencies = []

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = time.time()

        # Prepare batch inputs for each model
        batch_inputs = {}
        for name, arr in data.items():
            start = batch_idx * batch_size
            end = min(start + batch_size, len(arr))
            if start >= len(arr):
                continue
            batch_inputs[name] = torch.from_numpy(arr[start:end].copy()).float()
            total_processed[name] += (end - start)

        if not batch_inputs:
            break

        # Run all models in parallel
        _ = engine.infer_all_parallel(batch_inputs)

        batch_latencies.append(time.time() - batch_start)

    elapsed_time = time.time() - start_time

    # Metrics
    total_all = sum(total_processed.values())
    throughput = total_all / elapsed_time if elapsed_time > 0 else 0

    engine.shutdown()

    return {
        "mode": "single_gpu",
        "device": device,
        "elapsed_time": round(elapsed_time, 4),
        "total_samples_processed": total_all,
        "total_throughput": round(throughput, 1),
        "per_model_processed": total_processed,
        "num_models": len(models),
        "num_batches": num_batches,
        "batch_size": batch_size,
        "avg_batch_latency_ms": round(np.mean(batch_latencies) * 1000, 2) if batch_latencies else 0,
        "p99_batch_latency_ms": round(np.percentile(batch_latencies, 99) * 1000, 2) if batch_latencies else 0,
    }


def run_single_gpu_benchmark(
    models: Dict[str, torch.nn.Module],
    data: Dict[str, np.ndarray],
    batch_sizes: List[int] = [64, 128, 256, 512, 1024],
) -> List[Dict]:
    """
    Benchmark single GPU mode across multiple batch sizes.
    Returns list of result dicts for each batch size.
    """
    results = []
    for bs in batch_sizes:
        print(f"  [Single GPU] batch_size={bs}...", end=" ", flush=True)
        r = run_single_gpu_inference(models, data, batch_size=bs)
        print(f"{r['total_throughput']:,.0f} samples/sec")
        results.append(r)
    return results
