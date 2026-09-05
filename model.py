"""
PyTorch model for gas-class classification.

The 128 features are 8 response descriptors for each of 16 chemical sensors
(per overview.md), not an arbitrary flat vector. The SensorBlock below
encodes that structure directly: the same small MLP is applied to every
sensor's 8-value slice (weight-shared across sensors, like a depthwise
1D conv), which both respects the known layout and uses far fewer
parameters than a flat 128-input first layer -- important given there are
only ~10k training rows to fit a "complex" network with.
"""

import torch
import torch.nn as nn

N_SENSORS = 16
DESCRIPTORS_PER_SENSOR = 8
N_CLASSES = 6


class SensorBlock(nn.Module):
    """Shared per-sensor embedding: (batch, 16, 8) -> (batch, 16, embed_dim)."""

    def __init__(self, descriptors_per_sensor=DESCRIPTORS_PER_SENSOR, hidden=32, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(descriptors_per_sensor, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        # x: (batch, n_sensors, descriptors_per_sensor)
        # nn.Linear broadcasts over the middle "sensor" dimension, applying
        # the SAME weights to every sensor -- that's the weight-sharing.
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class DriftRobustMLP(nn.Module):
    def __init__(
        self,
        n_sensors=N_SENSORS,
        descriptors_per_sensor=DESCRIPTORS_PER_SENSOR,
        sensor_embed_dim=16,
        trunk_dim=128,
        n_residual_blocks=2,
        dropout=0.4,
        n_classes=N_CLASSES,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.descriptors_per_sensor = descriptors_per_sensor

        self.sensor_block = SensorBlock(descriptors_per_sensor, hidden=32, embed_dim=sensor_embed_dim)

        # +1 for log_concentration, concatenated onto the pooled sensor embeddings.
        trunk_input_dim = n_sensors * sensor_embed_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(trunk_input_dim, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(trunk_dim, dropout) for _ in range(n_residual_blocks)]
        )
        self.head = nn.Linear(trunk_dim, n_classes)

    def forward(self, features, log_concentration):
        # features: (batch, 128) -> reshape into (batch, 16 sensors, 8 descriptors)
        batch_size = features.shape[0]
        sensors = features.view(batch_size, self.n_sensors, self.descriptors_per_sensor)

        sensor_embeds = self.sensor_block(sensors)  # (batch, 16, embed_dim)
        pooled = sensor_embeds.flatten(start_dim=1)  # (batch, 16 * embed_dim)

        x = torch.cat([pooled, log_concentration.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = self.residual_blocks(x)
        return self.head(x)  # raw logits, shape (batch, n_classes)
