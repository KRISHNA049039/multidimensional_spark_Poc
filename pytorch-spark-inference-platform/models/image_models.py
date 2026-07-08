"""
Image Classification Models — ResNet-18, MobileNetV3, EfficientNet-B0.

These use torchvision pretrained weights for ImageNet 1000-class classification.
For airgapped deployment, weights are saved locally in models/weights/.

Input: (N, 3, 224, 224) RGB image tensor (normalized)
Output: (N, 1000) class logits
"""

import torch
import torch.nn as nn
import os

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


class ResNet18Classifier(nn.Module):
    """ResNet-18 image classifier (11.7M params, ~300MB GPU)."""

    def __init__(self, num_classes=1000, pretrained_path=None):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained_path and os.path.exists(pretrained_path):
            self.model = resnet18(weights=None)
            self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
        else:
            try:
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                self.model = resnet18(weights=None)

        if num_classes != 1000:
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3-Small (5.4M params, ~150MB GPU)."""

    def __init__(self, num_classes=1000, pretrained_path=None):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        if pretrained_path and os.path.exists(pretrained_path):
            self.model = mobilenet_v3_small(weights=None)
            self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
        else:
            try:
                self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            except Exception:
                self.model = mobilenet_v3_small(weights=None)

        if num_classes != 1000:
            self.model.classifier[-1] = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 (5.3M params, ~200MB GPU)."""

    def __init__(self, num_classes=1000, pretrained_path=None):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        if pretrained_path and os.path.exists(pretrained_path):
            self.model = efficientnet_b0(weights=None)
            self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
        else:
            try:
                self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            except Exception:
                self.model = efficientnet_b0(weights=None)

        if num_classes != 1000:
            self.model.classifier[-1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)
