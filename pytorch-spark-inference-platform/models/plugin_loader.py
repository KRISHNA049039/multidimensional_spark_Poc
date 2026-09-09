"""
Plugin loader — dynamic import + registry glue for user-supplied models.

Lets a model file dropped into models/plugins/ (declared in
models/plugins/manifest.json) run through the exact same Spark
mapPartitions / broadcast / device-routing path as the 10 built-in models,
without editing model_registry.py or models/__init__.py.
"""

import importlib
import json
import os

DEFAULT_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "plugins", "manifest.json")


def _load_manifest(manifest_path: str = DEFAULT_MANIFEST_PATH) -> dict:
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def get_plugin_class_map(manifest_path: str = DEFAULT_MANIFEST_PATH) -> dict:
    """{name: ModelClass} for every plugin — merged into cluster_engine's class map."""
    classes = {}
    for name, spec in _load_manifest(manifest_path).items():
        mod = importlib.import_module(spec["module"])
        classes[name] = getattr(mod, spec["class_name"])
    return classes


def register_plugins(registry, manifest_path: str = DEFAULT_MANIFEST_PATH) -> list:
    """Registers every plugin in the manifest into an existing ModelRegistry.

    Returns the list of plugin names registered.
    """
    names = []
    for name, spec in _load_manifest(manifest_path).items():
        mod = importlib.import_module(spec["module"])
        model_class = getattr(mod, spec["class_name"])
        registry.register(
            name=name,
            model_class=model_class,
            input_shape=tuple(spec["input_shape"]),
            output_desc=spec.get("output_desc", ""),
            category=spec.get("category", "custom"),
            estimated_memory_mb=spec.get("estimated_memory_mb", 100),
        )
        weights_path = spec.get("weights_path")
        if weights_path and os.path.exists(weights_path):
            import torch
            model = registry.load_model(name, device="cpu")
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        names.append(name)
    return names
