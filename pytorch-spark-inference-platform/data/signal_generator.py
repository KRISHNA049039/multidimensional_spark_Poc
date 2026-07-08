"""
Synthetic EW Signal Data Generator.

Generates IQ signal feature vectors (128-dim) for 8 EW signal classes.
Used for training the EW classifier and benchmarking all signal models.
"""

import numpy as np

NUM_CLASSES = 8
INPUT_FEATURES = 128


def generate_ew_signals(num_samples: int, seed: int = 42) -> tuple:
    """
    Generate synthetic EW signal dataset.

    Args:
        num_samples: Total samples to generate
        seed: Random seed for reproducibility

    Returns:
        (features, labels) — numpy arrays
        features: shape (num_samples, 128) float32
        labels: shape (num_samples,) int64
    """
    np.random.seed(seed)
    features = np.zeros((num_samples, INPUT_FEATURES), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        class_idx = np.random.randint(0, NUM_CLASSES)
        features[i] = _generate_signal(class_idx)
        labels[i] = class_idx

    return features, labels


def _generate_signal(class_idx: int) -> np.ndarray:
    """Generate a 128-dim IQ feature vector for a signal class."""
    t = np.linspace(0, 1, 64)

    if class_idx == 0:  # CW Radar
        freq = np.random.uniform(0.1, 0.3)
        i_ch = np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.05, 64)
        q_ch = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, 64)
    elif class_idx == 1:  # Pulsed Radar
        freq = np.random.uniform(0.2, 0.4)
        pw = np.random.uniform(0.1, 0.3)
        env = (t % 0.5) < pw
        i_ch = env * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)
        q_ch = env * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)
    elif class_idx == 2:  # FMCW
        f0, f1 = np.random.uniform(0.05, 0.15), np.random.uniform(0.35, 0.45)
        phase = 2 * np.pi * np.cumsum(f0 + (f1 - f0) * t) / 64
        i_ch = np.cos(phase) + np.random.normal(0, 0.08, 64)
        q_ch = np.sin(phase) + np.random.normal(0, 0.08, 64)
    elif class_idx == 3:  # Phase Coded
        freq = np.random.uniform(0.2, 0.3)
        code = np.repeat(np.random.choice([-1, 1], size=8), 8)
        i_ch = code * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)
        q_ch = code * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, 64)
    elif class_idx == 4:  # Noise Jammer
        i_ch = np.random.normal(0, 1.0, 64)
        q_ch = np.random.normal(0, 1.0, 64)
    elif class_idx == 5:  # Spot Jammer
        freq = np.random.uniform(0.2, 0.25)
        amp = np.random.uniform(2.0, 4.0)
        i_ch = amp * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.3, 64)
        q_ch = amp * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.3, 64)
    elif class_idx == 6:  # Sweep Jammer
        f0, f1 = np.random.uniform(0.05, 0.1), np.random.uniform(0.4, 0.49)
        rate = np.random.uniform(2, 5)
        freq_t = f0 + (f1 - f0) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
        phase = 2 * np.pi * np.cumsum(freq_t) / 64
        i_ch = 1.5 * np.cos(phase) + np.random.normal(0, 0.2, 64)
        q_ch = 1.5 * np.sin(phase) + np.random.normal(0, 0.2, 64)
    else:  # Comm Signal (QAM)
        sym_i = np.repeat(np.random.choice([-3, -1, 1, 3], size=8), 8) / 3.0
        sym_q = np.repeat(np.random.choice([-3, -1, 1, 3], size=8), 8) / 3.0
        i_ch = sym_i + np.random.normal(0, 0.15, 64)
        q_ch = sym_q + np.random.normal(0, 0.15, 64)

    vec = np.concatenate([i_ch, q_ch]).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
