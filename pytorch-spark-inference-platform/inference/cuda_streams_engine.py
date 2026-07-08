"""
CUDA Streams Multi-Model Inference Engine.

Runs N models in parallel on a single GPU using CUDA streams.
Each model gets its own stream — GPU kernel scheduler interleaves them.

This is the core building block used by all 3 inference modes:
- Mode 1 (Distributed): Each Spark executor runs this engine on its GPU
- Mode 2 (Single GPU): One process runs all models via streams
- Mode 3 (Hybrid): GPU models use streams, CPU models use ThreadPool
"""

import torch
import torch.cuda
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor


class CUDAStreamsEngine:
    """
    Multi-model parallel inference using CUDA streams.

    Each model is assigned a dedicated CUDA stream. Forward passes on
    different streams execute concurrently on the GPU (kernel interleaving).

    For CPU models, uses a ThreadPoolExecutor for parallel execution.
    """

    def __init__(self, models: Dict[str, torch.nn.Module],
                 device_map: Dict[str, str],
                 batch_size: int = 256):
        """
        Args:
            models: {model_name: model_instance} (already on correct device)
            device_map: {model_name: "cuda:0" or "cpu"}
            batch_size: Default batch size for inference
        """
        self.models = models
        self.device_map = device_map
        self.batch_size = batch_size

        # Create CUDA streams for GPU models
        self.streams: Dict[str, Optional[torch.cuda.Stream]] = {}
        for name, device in device_map.items():
            if "cuda" in device:
                self.streams[name] = torch.cuda.Stream(
                    device=torch.device(device)
                )
            else:
                self.streams[name] = None

        # Thread pool for CPU model parallelism
        cpu_count = len([d for d in device_map.values() if d == "cpu"])
        self.cpu_pool = ThreadPoolExecutor(max_workers=max(cpu_count, 1))

        # Ensure all models are in eval mode
        for model in self.models.values():
            model.eval()

    def infer_single(self, model_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on a single model."""
        model = self.models[model_name]
        device = self.device_map[model_name]
        stream = self.streams[model_name]

        with torch.no_grad():
            if stream is not None:
                with torch.cuda.stream(stream):
                    x = input_tensor.to(device, non_blocking=True)
                    output = model(x)
            else:
                x = input_tensor.to(device)
                output = model(x)

        return output

    def infer_all_parallel(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run ALL models in parallel on their respective inputs.

        GPU models: launched on CUDA streams (truly parallel on GPU)
        CPU models: launched on thread pool (parallel via threads)

        Args:
            inputs: {model_name: input_tensor} — tensors can be on CPU,
                    will be moved to correct device non-blocking

        Returns:
            {model_name: output_tensor} — outputs on CPU
        """
        gpu_outputs: Dict[str, torch.Tensor] = {}
        cpu_futures = {}

        # Phase 1: Launch GPU models on streams (non-blocking)
        for name, inp in inputs.items():
            if name not in self.models:
                continue
            device = self.device_map.get(name, "cpu")
            model = self.models[name]
            stream = self.streams.get(name)

            if stream is not None and "cuda" in device:
                with torch.cuda.stream(stream):
                    with torch.no_grad():
                        x = inp.to(device, non_blocking=True)
                        gpu_outputs[name] = model(x)
            else:
                # Submit CPU models to thread pool
                def _cpu_infer(m=model, x=inp):
                    with torch.no_grad():
                        return m(x)
                cpu_futures[name] = self.cpu_pool.submit(_cpu_infer)

        # Phase 2: Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Phase 3: Collect results
        results = {}
        for name, output in gpu_outputs.items():
            results[name] = output.cpu()

        for name, future in cpu_futures.items():
            results[name] = future.result()

        return results

    def infer_all_batched(self, data_batches: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        """
        Process multiple batches through all models.

        For large datasets: iterate over batches, running all models
        in parallel per batch.

        Args:
            data_batches: {model_name: [batch1, batch2, ...]}

        Returns:
            {model_name: [output1, output2, ...]}
        """
        # Determine number of batches (should be same for all models)
        num_batches = max(len(batches) for batches in data_batches.values())

        all_results: Dict[str, List[torch.Tensor]] = {name: [] for name in data_batches}

        for batch_idx in range(num_batches):
            # Prepare inputs for this batch
            batch_inputs = {}
            for name, batches in data_batches.items():
                if batch_idx < len(batches):
                    batch_inputs[name] = batches[batch_idx]

            # Run all models in parallel on this batch
            outputs = self.infer_all_parallel(batch_inputs)

            for name, output in outputs.items():
                all_results[name].append(output)

        return all_results

    def benchmark_single_model(self, model_name: str, input_tensor: torch.Tensor,
                                num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark a single model's throughput and latency."""
        device = self.device_map[model_name]
        model = self.models[model_name]

        # Warmup
        with torch.no_grad():
            x = input_tensor.to(device)
            for _ in range(10):
                _ = model(x)
        if "cuda" in device:
            torch.cuda.synchronize()

        # Timed run
        latencies = []
        start_total = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                t0 = time.time()
                x = input_tensor.to(device)
                _ = model(x)
                if "cuda" in device:
                    torch.cuda.synchronize()
                latencies.append(time.time() - t0)
        total_time = time.time() - start_total

        batch_size = input_tensor.shape[0]
        return {
            "model": model_name,
            "device": device,
            "throughput_per_sec": (num_iterations * batch_size) / total_time,
            "avg_latency_ms": np.mean(latencies) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "total_time_sec": total_time,
        }

    def benchmark_all_parallel(self, inputs: Dict[str, torch.Tensor],
                                num_iterations: int = 50) -> Dict[str, any]:
        """Benchmark all models running in parallel."""
        # Warmup
        for _ in range(5):
            _ = self.infer_all_parallel(inputs)

        # Timed run
        latencies = []
        start_total = time.time()
        for _ in range(num_iterations):
            t0 = time.time()
            _ = self.infer_all_parallel(inputs)
            latencies.append(time.time() - t0)
        total_time = time.time() - start_total

        total_samples = sum(inp.shape[0] for inp in inputs.values())
        return {
            "num_models": len(inputs),
            "total_throughput_per_sec": (num_iterations * total_samples) / total_time,
            "avg_batch_latency_ms": np.mean(latencies) * 1000,
            "p99_batch_latency_ms": np.percentile(latencies, 99) * 1000,
            "total_time_sec": total_time,
            "iterations": num_iterations,
        }

    def shutdown(self):
        """Cleanup resources."""
        self.cpu_pool.shutdown(wait=False)
