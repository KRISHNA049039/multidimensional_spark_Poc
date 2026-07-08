"""
Model Registry — Central registry for all inference models.

Manages loading, serialization, and metadata for 10 models:
- EW Signal Classifier (custom MLP)
- YOLOv8 Nano/Small (object detection)
- ResNet-18, MobileNetV3, EfficientNet-B0 (image classification)
- Signal Denoiser (autoencoder)
- Threat Prioritizer (multi-head)
- RF Fingerprinter (1D-CNN)
- Anomaly Detector (VAE)
"""

import torch
import io
from typing import Dict, Tuple, Optional


class ModelInfo:
    """Metadata about a registered model."""

    def __init__(self, name, model_class, input_shape, output_desc,
                 category, estimated_memory_mb):
        self.name = name
        self.model_class = model_class
        self.input_shape = input_shape
        self.output_desc = output_desc
        self.category = category
        self.estimated_memory_mb = estimated_memory_mb


class ModelRegistry:
    """
    Central registry for all models in the platform.

    Handles:
    - Lazy loading (models created on demand)
    - Serialization for Spark broadcasting
    - Memory estimation for GPU budget planning
    - Device placement decisions
    """

    def __init__(self):
        self._registry: Dict[str, ModelInfo] = {}
        self._loaded_models: Dict[str, torch.nn.Module] = {}
        self._model_bytes: Dict[str, bytes] = {}

    def register(self, name: str, model_class, input_shape: tuple,
                 output_desc: str, category: str, estimated_memory_mb: float):
        """Register a model definition (does not load weights yet)."""
        self._registry[name] = ModelInfo(
            name=name,
            model_class=model_class,
            input_shape=input_shape,
            output_desc=output_desc,
            category=category,
            estimated_memory_mb=estimated_memory_mb,
        )

    def list_models(self) -> Dict[str, ModelInfo]:
        """Return all registered model metadata."""
        return self._registry

    def get_info(self, name: str) -> ModelInfo:
        """Get metadata for a specific model."""
        return self._registry[name]

    def load_model(self, name: str, device: str = "cpu") -> torch.nn.Module:
        """Load and return a model (cached after first load)."""
        if name not in self._loaded_models:
            info = self._registry[name]
            model = info.model_class()
            model.eval()
            self._loaded_models[name] = model

        model = self._loaded_models[name]
        model = model.to(device)
        return model

    def load_all(self, device: str = "cpu") -> Dict[str, torch.nn.Module]:
        """Load all registered models."""
        models = {}
        for name in self._registry:
            models[name] = self.load_model(name, device)
        return models

    def serialize_model(self, name: str) -> bytes:
        """Serialize model to bytes for Spark broadcasting."""
        if name not in self._model_bytes:
            model = self.load_model(name, "cpu")
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            self._model_bytes[name] = buf.getvalue()
        return self._model_bytes[name]

    def serialize_all(self) -> Dict[str, bytes]:
        """Serialize all models."""
        return {name: self.serialize_model(name) for name in self._registry}

    def deserialize_model(self, name: str, model_bytes: bytes,
                          device: str = "cpu") -> torch.nn.Module:
        """Deserialize model from bytes (used on Spark workers)."""
        info = self._registry[name]
        model = info.model_class()
        buf = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
        model = model.to(device)
        model.eval()
        return model

    def total_memory_estimate_mb(self) -> float:
        """Estimate total GPU memory needed for all models."""
        return sum(info.estimated_memory_mb for info in self._registry.values())

    def get_models_by_category(self, category: str) -> Dict[str, ModelInfo]:
        """Filter models by category."""
        return {k: v for k, v in self._registry.items() if v.category == category}

    def summary(self):
        """Print registry summary."""
        print(f"\n{'='*60}")
        print(f"  MODEL REGISTRY — {len(self._registry)} models")
        print(f"{'='*60}")
        print(f"{'Name':<25} {'Category':<15} {'Memory (MB)':<12} {'Input Shape'}")
        print(f"{'-'*25} {'-'*15} {'-'*12} {'-'*20}")
        for name, info in self._registry.items():
            print(f"{name:<25} {info.category:<15} {info.estimated_memory_mb:<12.0f} {str(info.input_shape)}")
        total = self.total_memory_estimate_mb()
        print(f"\n  Total estimated GPU memory: {total:.0f} MB ({total/1024:.1f} GB)")
