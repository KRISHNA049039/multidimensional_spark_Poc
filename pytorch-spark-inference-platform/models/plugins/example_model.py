"""
Example BYOM plugin — a stand-in for a user-supplied model file.

Demonstrates the only contract a plugin model needs to follow:
  - plain torch.nn.Module subclass
  - __init__(self) takes no required arguments
  - forward(x) is batch-first, returns a tensor

Drop your own file next to this one and add an entry to manifest.json —
nothing else in the platform needs to change.
"""

import torch
import torch.nn as nn

INPUT_FEATURES = 64
NUM_CLASSES = 4


class ExampleMLP(nn.Module):
    def __init__(self, input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
