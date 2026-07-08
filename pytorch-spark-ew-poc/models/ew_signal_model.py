"""
Electronic Warfare Signal Classifier - PyTorch Model

This module defines a neural network for classifying intercepted electronic warfare
signals (radar pulses, jammers, communication signals) based on their IQ (In-phase/Quadrature)
features and signal characteristics.

Signal Feature Vector (128 features):
- IQ samples (64 complex values -> 128 real values representing I and Q channels)
- Derived features: pulse width, PRI, carrier frequency, bandwidth, modulation indices

Classification targets:
- 8 signal classes: CW Radar, Pulsed Radar, FMCW, Phase-coded, Noise Jammer,
                     Spot Jammer, Sweep Jammer, Communication Signal
"""

import torch
import torch.nn as nn
import numpy as np
import io


# Signal class labels for EW classification
SIGNAL_CLASSES = [
    "CW_Radar",          # Continuous Wave radar
    "Pulsed_Radar",      # Traditional pulsed radar
    "FMCW_Radar",        # Frequency Modulated CW radar
    "Phase_Coded_Radar", # Phase-coded pulse compression radar
    "Noise_Jammer",      # Broadband noise jammer
    "Spot_Jammer",       # Narrowband spot jammer
    "Sweep_Jammer",      # Swept frequency jammer
    "Comm_Signal",       # Communication signal (non-threat)
]

NUM_CLASSES = len(SIGNAL_CLASSES)
INPUT_FEATURES = 128  # IQ sample feature vector length


class EWSignalClassifier(nn.Module):
    """
    Multi-layer neural network for EW signal classification.

    Architecture:
    - Input: 128-dim feature vector (IQ samples + signal parameters)
    - Hidden layers with BatchNorm and Dropout for robustness
    - Output: 8-class softmax classification

    Designed to be lightweight enough for high-throughput inference while
    maintaining accuracy on distinct signal modulation types.
    """

    def __init__(self, input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES):
        super(EWSignalClassifier, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Layer 1: Initial feature expansion
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: Feature refinement
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: Compression
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 4: Final feature representation
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Forward pass through feature extractor and classifier."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def predict(self, x):
        """Run inference and return predicted class indices."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x):
        """Run inference and return class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


def create_trained_model(seed=42):
    """
    Create and train a model on synthetic EW signal data.

    For the PoC, we train on synthetic data that mimics real signal characteristics.
    This gives the model realistic weights for benchmarking inference performance.

    Returns:
        Trained EWSignalClassifier model in eval mode.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EWSignalClassifier()

    # Generate synthetic training data
    train_data, train_labels = generate_training_data(num_samples=10000, seed=seed)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Quick training (enough to get meaningful weights for inference benchmarking)
    model.train()
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data),
        torch.LongTensor(train_labels)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(20):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model


def generate_training_data(num_samples=10000, seed=42):
    """
    Generate synthetic EW signal feature vectors for training.

    Each signal class has distinct characteristics in IQ space:
    - CW Radar: Constant amplitude, single frequency
    - Pulsed Radar: Periodic bursts with specific PRIs
    - FMCW: Linear fr
    equency sweep patterns
    - Phase Coded: BPSK/Barker code modulated
    - Noise Jammer: Wideband Gaussian noise
    - Spot Jammer: Narrowband interference
    - Sweep Jammer: Time-varying frequency
    - Comm Signal: Digital modulation patterns
    """
    np.random.seed(seed)

    samples_per_class = num_samples // NUM_CLASSES
    data = []
    labels = []

    for class_idx in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            feature_vec = _generate_signal_features(class_idx)
            data.append(feature_vec)
            labels.append(class_idx)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(data))
    return data[perm], labels[perm]


def _generate_signal_features(class_idx):
    """Generate a 128-dim feature vector for a given signal class."""
    t = np.linspace(0, 1, 64)  # Time axis for 64 IQ samples

    if class_idx == 0:  # CW Radar
        freq = np.random.uniform(0.1, 0.3)
        i_channel = np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.05, 64)
        q_channel = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, 64)

    elif class_idx == 1:  # Pulsed Radar
        freq = np.random.uniform(0.2, 0.4)
        pulse_width = np.random.uniform(0.1, 0.3)
        envelope = (t % 0.5) < pulse_width
        i_channel = envelope * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)
        q_channel = envelope * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)

    elif class_idx == 2:  # FMCW Radar
        f_start = np.random.uniform(0.05, 0.15)
        f_end = np.random.uniform(0.35, 0.45)
        freq_sweep = f_start + (f_end - f_start) * t
        phase = 2 * np.pi * np.cumsum(freq_sweep) / 64
        i_channel = np.cos(phase) + np.random.normal(0, 0.08, 64)
        q_channel = np.sin(phase) + np.random.normal(0, 0.08, 64)

    elif class_idx == 3:  # Phase Coded Radar
        freq = np.random.uniform(0.2, 0.3)
        # Barker-like code
        code = np.random.choice([-1, 1], size=8)
        phase_code = np.repeat(code, 8)
        i_channel = phase_code * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)
        q_channel = phase_code * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)

    elif class_idx == 4:  # Noise Jammer
        i_channel = np.random.normal(0, 1.0, 64)
        q_channel = np.random.normal(0, 1.0, 64)

    elif class_idx == 5:  # Spot Jammer
        freq = np.random.uniform(0.2, 0.25)
        amplitude = np.random.uniform(2.0, 4.0)
        i_channel = amplitude * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.3, 64)
        q_channel = amplitude * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.3, 64)

    elif class_idx == 6:  # Sweep Jammer
        f_start = np.random.uniform(0.05, 0.1)
        f_end = np.random.uniform(0.4, 0.49)
        sweep_rate = np.random.uniform(2, 5)
        freq_t = f_start + (f_end - f_start) * (0.5 + 0.5 * np.sin(2 * np.pi * sweep_rate * t))
        phase = 2 * np.pi * np.cumsum(freq_t) / 64
        i_channel = 1.5 * np.cos(phase) + np.random.normal(0, 0.2, 64)
        q_channel = 1.5 * np.sin(phase) + np.random.normal(0, 0.2, 64)

    else:  # Communication Signal (class 7)
        # QAM-like modulation
        symbols = np.random.choice([-3, -1, 1, 3], size=8)
        i_symbols = np.repeat(symbols, 8) / 3.0
        symbols_q = np.random.choice([-3, -1, 1, 3], size=8)
        q_symbols = np.repeat(symbols_q, 8) / 3.0
        i_channel = i_symbols + np.random.normal(0, 0.15, 64)
        q_channel = q_symbols + np.random.normal(0, 0.15, 64)

    # Concatenate I and Q channels into 128-dim feature vector
    feature_vec = np.concatenate([i_channel, q_channel]).astype(np.float32)

    # Normalize
    norm = np.linalg.norm(feature_vec)
    if norm > 0:
        feature_vec = feature_vec / norm

    return feature_vec


def serialize_model(model):
    """Serialize model to bytes for Spark broadcasting."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_model(model_bytes):
    """Deserialize model from bytes (used on Spark workers)."""
    buffer = io.BytesIO(model_bytes)
    model = EWSignalClassifier()
    model.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    # Quick validation
    print("Creating and training EW Signal Classifier...")
    model = create_trained_model()

    # Test inference
    test_data, test_labels = generate_training_data(num_samples=1000, seed=99)
    test_tensor = torch.FloatTensor(test_data)

    predictions = model.predict(test_tensor)
    accuracy = (predictions.numpy() == test_labels).mean()
    print(f"Model accuracy on test data: {accuracy:.2%}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Signal classes: {SIGNAL_CLASSES}")

    # Test serialization round-trip
    model_bytes = serialize_model(model)
    restored_model = deserialize_model(model_bytes)
    restored_preds = restored_model.predict(test_tensor)
    assert torch.equal(predictions, restored_preds), "Serialization round-trip failed!"
    print(f"Model serialization: OK ({len(model_bytes):,} bytes)")
