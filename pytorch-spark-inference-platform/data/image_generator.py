"""
Synthetic Image Data Generator.

Generates random image tensors for benchmarking image classification
and object detection models. Two formats:
- 224×224 (ResNet, MobileNet, EfficientNet)
- 640×640 (YOLOv8)

For real deployment, replace with actual image loading pipeline.
"""

import numpy as np
from typing import Tuple


def generate_classification_images(num_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 224×224 RGB images for classification benchmarking.

    In production this would be replaced by:
    - Camera/sensor feed
    - Satellite imagery
    - Stored image files (PNG/JPG decoded to numpy)

    Args:
        num_samples: Number of images to generate
        seed: Random seed

    Returns:
        (images, labels)
        images: shape (num_samples, 3, 224, 224) float32 [0, 1]
        labels: shape (num_samples,) int64 [0, 999]
    """
    np.random.seed(seed)

    # Generate random images (normalized [0,1] as torchvision expects)
    images = np.random.rand(num_samples, 3, 224, 224).astype(np.float32)

    # Random labels (1000-class ImageNet)
    labels = np.random.randint(0, 1000, size=num_samples, dtype=np.int64)

    return images, labels


def generate_detection_images(num_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic 640×640 RGB images for YOLO object detection.

    Args:
        num_samples: Number of images
        seed: Random seed

    Returns:
        images: shape (num_samples, 3, 640, 640) float32 [0, 1]
    """
    np.random.seed(seed)
    images = np.random.rand(num_samples, 3, 640, 640).astype(np.float32)
    return images


def generate_mixed_data(num_signal_samples: int, num_image_samples: int,
                        num_detection_samples: int, seed: int = 42) -> dict:
    """
    Generate all data needed for benchmarking all 10 models.

    Returns:
        Dict mapping model_name → numpy input array
    """
    from data.signal_generator import generate_ew_signals

    np.random.seed(seed)

    # Signal data (128-dim) for signal models
    signal_features, _ = generate_ew_signals(num_signal_samples, seed=seed)

    # Image data (3×224×224) for image classifiers
    class_images, _ = generate_classification_images(num_image_samples, seed=seed + 1)

    # Detection data (3×640×640) for YOLO
    det_images = generate_detection_images(num_detection_samples, seed=seed + 2)

    return {
        # Signal models all share the same input format
        "ew_classifier": signal_features,
        "signal_denoiser": signal_features,
        "threat_prioritizer": signal_features,
        "rf_fingerprinter": signal_features,
        "anomaly_detector": signal_features,
        # Image classifiers share 224×224 images
        "resnet18": class_images,
        "mobilenetv3": class_images,
        "efficientnet_b0": class_images,
        # YOLO models share 640×640 images
        "yolov8_nano": det_images,
        "yolov8_small": det_images,
    }
