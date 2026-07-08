"""
GPU Memory Manager — Budget-aware model placement across GPU and CPU.

Decides which models fit on GPU based on available VRAM, places remaining
models on CPU. Supports dynamic reallocation and memory monitoring.

Works in:
- Local mode: single GPU memory budget
- Cluster mode: per-executor GPU memory budget
"""

import torch
import torch.cuda
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ModelPlacement:
    """Records where a model is placed and its memory footprint."""
    name: str
    device: str  # "cuda:0", "cuda:1", "cpu"
    memory_mb: float
    priority: int  # Higher = placed on GPU first


@dataclass
class GPUBudget:
    """Tracks memory budget for a single GPU device."""
    device_id: int
    total_mb: float
    reserved_mb: float  # CUDA context, buffers, etc.
    allocated_mb: float = 0.0
    models: List[str] = field(default_factory=list)

    @property
    def available_mb(self) -> float:
        return self.total_mb - self.reserved_mb - self.allocated_mb

    @property
    def utilization_pct(self) -> float:
        used = self.reserved_mb + self.allocated_mb
        return (used / self.total_mb) * 100 if self.total_mb > 0 else 0


class GPUMemoryManager:
    """
    Manages GPU memory allocation across multiple models.

    Strategies:
    1. Priority-based: Place high-priority models on GPU first
    2. Size-based: Place largest models on GPU (best compute/transfer ratio)
    3. Greedy-fit: Fill GPU until full, rest goes to CPU
    4. Balanced: Spread models across available GPUs evenly
    """

    def __init__(self, reserve_mb: float = 500.0, strategy: str = "priority"):
        """
        Args:
            reserve_mb: MB to reserve per GPU for CUDA context + buffers
            strategy: "priority", "largest_first", "greedy", "balanced"
        """
        self.reserve_mb = reserve_mb
        self.strategy = strategy
        self.placements: Dict[str, ModelPlacement] = {}
        self.gpu_budgets: List[GPUBudget] = []
        self._init_gpu_budgets()

    def _init_gpu_budgets(self):
        """Detect available GPUs and initialize budgets."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory / (1024 * 1024)
                self.gpu_budgets.append(GPUBudget(
                    device_id=i,
                    total_mb=total_mb,
                    reserved_mb=self.reserve_mb,
                ))

    @property
    def num_gpus(self) -> int:
        return len(self.gpu_budgets)

    @property
    def total_gpu_memory_mb(self) -> float:
        return sum(b.total_mb for b in self.gpu_budgets)

    def plan_placement(self, models_with_sizes: Dict[str, float],
                       priorities: Optional[Dict[str, int]] = None) -> Dict[str, str]:
        """
        Plan model placement across available devices.

        Args:
            models_with_sizes: {model_name: estimated_memory_mb}
            priorities: {model_name: priority} (higher = prefer GPU)

        Returns:
            {model_name: device_string} e.g. {"resnet18": "cuda:0", "denoiser": "cpu"}
        """
        if priorities is None:
            priorities = {name: 5 for name in models_with_sizes}

        # Sort models by strategy
        if self.strategy == "priority":
            sorted_models = sorted(models_with_sizes.items(),
                                   key=lambda x: priorities.get(x[0], 0), reverse=True)
        elif self.strategy == "largest_first":
            sorted_models = sorted(models_with_sizes.items(),
                                   key=lambda x: x[1], reverse=True)
        elif self.strategy == "balanced":
            sorted_models = sorted(models_with_sizes.items(),
                                   key=lambda x: x[1], reverse=True)
        else:  # greedy
            sorted_models = list(models_with_sizes.items())

        placement_map = {}

        for name, size_mb in sorted_models:
            placed = False

            if self.strategy == "balanced" and self.num_gpus > 1:
                # Place on GPU with most available memory
                budgets_sorted = sorted(self.gpu_budgets,
                                        key=lambda b: b.available_mb, reverse=True)
            else:
                budgets_sorted = self.gpu_budgets

            for budget in budgets_sorted:
                if budget.available_mb >= size_mb:
                    device = f"cuda:{budget.device_id}"
                    budget.allocated_mb += size_mb
                    budget.models.append(name)
                    placement_map[name] = device
                    self.placements[name] = ModelPlacement(
                        name=name, device=device,
                        memory_mb=size_mb, priority=priorities.get(name, 5)
                    )
                    placed = True
                    break

            if not placed:
                placement_map[name] = "cpu"
                self.placements[name] = ModelPlacement(
                    name=name, device="cpu",
                    memory_mb=size_mb, priority=priorities.get(name, 5)
                )

        return placement_map

    def get_device(self, model_name: str) -> str:
        """Get the assigned device for a model."""
        if model_name in self.placements:
            return self.placements[model_name].device
        return "cpu"

    def get_gpu_models(self) -> List[str]:
        """Return names of models placed on GPU."""
        return [p.name for p in self.placements.values() if "cuda" in p.device]

    def get_cpu_models(self) -> List[str]:
        """Return names of models placed on CPU."""
        return [p.name for p in self.placements.values() if p.device == "cpu"]

    def report(self) -> str:
        """Generate human-readable placement report."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  GPU MEMORY MANAGER — {self.num_gpus} GPU(s) detected")
        lines.append(f"{'='*60}")

        for budget in self.gpu_budgets:
            lines.append(f"\n  GPU {budget.device_id}: {budget.total_mb:.0f} MB total, "
                         f"{budget.available_mb:.0f} MB free, "
                         f"{budget.utilization_pct:.1f}% used")
            for model_name in budget.models:
                p = self.placements[model_name]
                lines.append(f"    └─ {model_name}: {p.memory_mb:.0f} MB")

        cpu_models = self.get_cpu_models()
        if cpu_models:
            lines.append(f"\n  CPU ({len(cpu_models)} models):")
            for name in cpu_models:
                p = self.placements[name]
                lines.append(f"    └─ {name}: {p.memory_mb:.0f} MB")

        gpu_total = sum(p.memory_mb for p in self.placements.values() if "cuda" in p.device)
        cpu_total = sum(p.memory_mb for p in self.placements.values() if p.device == "cpu")
        lines.append(f"\n  Summary: {len(self.get_gpu_models())} on GPU ({gpu_total:.0f} MB), "
                     f"{len(cpu_models)} on CPU ({cpu_total:.0f} MB)")

        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    @staticmethod
    def get_current_gpu_usage() -> Dict[int, Dict[str, float]]:
        """Get real-time GPU memory usage (requires CUDA)."""
        usage = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                usage[i] = {
                    "allocated_mb": allocated,
                    "reserved_mb": reserved,
                    "total_mb": total,
                    "free_mb": total - reserved,
                }
        return usage
