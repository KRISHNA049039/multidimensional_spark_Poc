"""
Models package — 10 models for multi-model distributed inference.

Usage:
    from models import get_default_registry
    registry = get_default_registry()
    registry.summary()
"""

from models.model_registry import ModelRegistry, ModelInfo
from models.ew_signal_model import EWSignalClassifier
from models.yolo_model import YOLOv8Nano, YOLOv8Small
from models.image_models import ResNet18Classifier, MobileNetV3Classifier, EfficientNetB0Classifier
from models.signal_models import SignalDenoiser, ThreatPrioritizer, RFFingerprinter, AnomalyDetector


def get_default_registry() -> ModelRegistry:
    """
    Create and return the default model registry with all 10 models registered.
    Models are registered (metadata only) — not loaded into memory until requested.
    """
    registry = ModelRegistry()

    # --- EW Signal Models (128-dim IQ input) ---
    registry.register(
        name="ew_classifier",
        model_class=EWSignalClassifier,
        input_shape=(128,),
        output_desc="8-class logits (radar/jammer/comms)",
        category="signal",
        estimated_memory_mb=50,
    )
    registry.register(
        name="signal_denoiser",
        model_class=SignalDenoiser,
        input_shape=(128,),
        output_desc="128-dim denoised signal",
        category="signal",
        estimated_memory_mb=100,
    )
    registry.register(
        name="threat_prioritizer",
        model_class=ThreatPrioritizer,
        input_shape=(128,),
        output_desc="scalar priority score [0,1]",
        category="signal",
        estimated_memory_mb=350,
    )
    registry.register(
        name="rf_fingerprinter",
        model_class=RFFingerprinter,
        input_shape=(128,),
        output_desc="32-dim emitter embedding",
        category="signal",
        estimated_memory_mb=120,
    )
    registry.register(
        name="anomaly_detector",
        model_class=AnomalyDetector,
        input_shape=(128,),
        output_desc="scalar anomaly score",
        category="signal",
        estimated_memory_mb=100,
    )

    # --- Image Classification Models (3×224×224 input) ---
    registry.register(
        name="resnet18",
        model_class=ResNet18Classifier,
        input_shape=(3, 224, 224),
        output_desc="1000-class ImageNet logits",
        category="image_classification",
        estimated_memory_mb=300,
    )
    registry.register(
        name="mobilenetv3",
        model_class=MobileNetV3Classifier,
        input_shape=(3, 224, 224),
        output_desc="1000-class ImageNet logits",
        category="image_classification",
        estimated_memory_mb=150,
    )
    registry.register(
        name="efficientnet_b0",
        model_class=EfficientNetB0Classifier,
        input_shape=(3, 224, 224),
        output_desc="1000-class ImageNet logits",
        category="image_classification",
        estimated_memory_mb=200,
    )

    # --- Object Detection Models (3×640×640 input) ---
    registry.register(
        name="yolov8_nano",
        model_class=YOLOv8Nano,
        input_shape=(3, 640, 640),
        output_desc="detection boxes (x,y,w,h,conf,cls)",
        category="object_detection",
        estimated_memory_mb=200,
    )
    registry.register(
        name="yolov8_small",
        model_class=YOLOv8Small,
        input_shape=(3, 640, 640),
        output_desc="detection boxes (x,y,w,h,conf,cls)",
        category="object_detection",
        estimated_memory_mb=400,
    )

    return registry
