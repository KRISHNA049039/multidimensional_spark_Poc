"""
EW Signal Classifier — 8-class radar/jammer/comms classification.
Input: 128-dim IQ feature vector (64 I-channel + 64 Q-channel)
Output: 8 class logits
"""

import torch
import torch.nn as nn
import numpy as np

NUM_CLASSES = 8
INPUT_FEATURES = 128

SIGNAL_CLASSES = [
    "CW_Radar", "Pulsed_Radar", "FMCW_Radar", "Phase_Coded_Radar",
    "Noise_Jammer", "Spot_Jammer", "Sweep_Jammer", "Comm_Signal",
]


class EWSignalClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


def create_trained_ew_model(seed=42):
    """Create and quick-train the EW classifier."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EWSignalClassifier()

    # Quick synthetic training
    from data.signal_generator import generate_ew_signals
    features, labels = generate_ew_signals(10000, seed=seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(features), torch.LongTensor(labels)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(20):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model
