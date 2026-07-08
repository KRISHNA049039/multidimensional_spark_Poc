"""
Mode 3: Hybrid CPU + GPU Inference (Memory-Aware Split)

Intelligently places models across GPU and CPU based on:
- Available GPU memory budget
- Model priority (critical models get GPU)
- Model size (large models benefit most from GPU)

GPU models run on CUDA streams (parallel).
CPU models run on ThreadPoolExecutor (parallel).
Both execute concurrently.

Architecture:
  1. GPUMemoryManager decides placement
  2. High-priority/large models → GPU + CUDA streams
  3. Remaining models → CPU + thread pool
  4. Both groups run simultaneously
  5. Results merged

Works in:
  - Limited GPU memory (GTX 1650, 4GB)
  - Mixed workloads where some models are lightweight
  - Cluster mode: per-executor hybrid placement
"""

import sys
import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.cuda_streams_engine import CUDAStreamsEngine
from inference.gpu_memory_manager import GPUMemoryManager


# Default model priorities (higher = prefer GPU placement)
DEFAULT_PRIORITIES = {
    "yolov8_small": 10,       # Largest compute benefit from GPU
    "yolov8_nano": 9,
    "resnet18": 8,
    "efficientnet_b0": 7,
    "mobilenetv3": 6,
    "threat_prioritizer": 5,
    "ew_classifier": 4,
    "rf_fingerprinter": 3,
    "signal_denoiser": 2,     # Small model, OK on CPU
    "anomaly_detector": 1,    # Small model, OK on CPU
}


def run_hybrid_inference(
    models: Dict[str, torch.nn.Module],
    model_sizes_mb: Dict[str, float],
    data: Dict[str, np.ndarray],
    batch_size: int = 256,
    gpu_memory_limit_mb: Optional[float] = None,
    priorities: Optional[Dict[str, int]] = None,
    strategy: str = "priority",
) -> Dict[str, any]:
    """
    Run models on a mix of GPU and CPU based on memory budget.

    Args:
        models: {model_name: model_instance}
        model_sizes_mb: {model_name: estimated GPU memory in MB}
        data: {model_name: numpy input array}
        batch_size: Samples per batch
        gpu_memory_limit_mb: Max GPU memory to use (None = auto-detect)
        priorities: {model_name: priority} — higher = prefer GPU
        strategy: "priority", "largest_first", "greedy", "balanced"

    Returns:
        Dict with timing, placement info, per-model metrics
    """
    if priorities is None:
        priorities = DEFAULT_PRIORITIES

    # --- Step 1: Plan placement ---
    mem_manager = GPUMemoryManager(
        reserve_mb=500 if torch.cuda.is_available() else 0,
        strategy=strategy,
    )

    # Override total memory if limit specified
    if gpu_memory_limit_mb and mem_manager.gpu_budgets:
        for budget in mem_manager.gpu_budgets:
            budget.total_mb = min(budget.total_mb, gpu_memory_limit_mb + budget.reserved_mb)

    device_map = mem_manager.plan_placement(model_sizes_mb, priorities)
    mem_manager.report()

    # --- Step 2: Place models on assigned devices ---
    gpu_models = {}
    cpu_models = {}
    gpu_device_map = {}
    cpu_device_map = {}

    for name, model in models.items():
        device = device_map.get(name, "cpu")
        if "cuda" in device and torch.cuda.is_available():
            model = model.to(device).eval()
            gpu_models[name] = model
            gpu_device_map[name] = device
        else:
            model = model.to("cpu").eval()
            cpu_models[name] = model
            cpu_device_map[name] = "cpu"

    # --- Step 3: Create engines ---
    gpu_engine = None
    if gpu_models:
        gpu_engine = CUDAStreamsEngine(gpu_models, gpu_device_map, batch_size)

    cpu_pool = ThreadPoolExecutor(max_workers=max(len(cpu_models), 1))

    # --- Step 4: Run inference ---
    max_samples = max(len(arr) for arr in data.values()) if data else 0
    num_batches = (max_samples + batch_size - 1) // batch_size

    # Warmup GPU
    if gpu_engine:
        warmup_inputs = {}
        for name in gpu_models:
            if name in data:
                warmup_size = min(batch_size, len(data[name]))
                warmup_inputs[name] = torch.from_numpy(data[name][:warmup_size].copy()).float()
        if warmup_inputs:
            for _ in range(3):
                _ = gpu_engine.infer_all_parallel(warmup_inputs)

    total_processed = {name: 0 for name in models}
    batch_latencies = []

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = time.time()

        # Prepare batch inputs
        gpu_inputs = {}
        cpu_inputs = {}

        for name, arr in data.items():
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(arr))
            if start_idx >= len(arr):
                continue
            batch_tensor = torch.from_numpy(arr[start_idx:end_idx].copy()).float()

            if name in gpu_models:
                gpu_inputs[name] = batch_tensor
            elif name in cpu_models:
                cpu_inputs[name] = batch_tensor
            total_processed[name] += (end_idx - start_idx)

        # Launch GPU models (non-blocking via streams)
        gpu_future = None
        if gpu_engine and gpu_inputs:
            gpu_results = gpu_engine.infer_all_parallel(gpu_inputs)

        # Launch CPU models in parallel threads
        cpu_futures = {}
        for name, inp in cpu_inputs.items():
            model = cpu_models[name]

            def _run(m=model, x=inp):
                with torch.no_grad():
                    return m(x)

            cpu_futures[name] = cpu_pool.submit(_run)

        # Wait for CPU models
        for name, future in cpu_futures.items():
            _ = future.result()

        batch_latencies.append(time.time() - batch_start)

    elapsed_time = time.time() - start_time

    # Cleanup
    if gpu_engine:
        gpu_engine.shutdown()
    cpu_pool.shutdown(wait=False)

    # Metrics
    total_all = sum(total_processed.values())
    throughput = total_all / elapsed_time if elapsed_time > 0 else 0

    return {
        "mode": "hybrid_cpu_gpu",
        "elapsed_time": round(elapsed_time, 4),
        "total_samples_processed": total_all,
        "total_throughput": round(throughput, 1),
        "per_model_processed": total_processed,
        "num_models": len(models),
        "gpu_models": list(gpu_models.keys()),
        "cpu_models": list(cpu_models.keys()),
        "num_on_gpu": len(gpu_models),
        "num_on_cpu": len(cpu_models),
        "strategy": strategy,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "avg_batch_latency_ms": round(np.mean(batch_latencies) * 1000, 2) if batch_latencies else 0,
        "p99_batch_latency_ms": round(np.percentile(batch_latencies, 99) * 1000, 2) if batch_latencies else 0,
    }
