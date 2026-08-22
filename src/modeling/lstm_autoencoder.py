from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape: batch_size, sequence_length, num_features.")
        if x.size(-1) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x.size(-1)}.")

        _, (hidden, _) = self.encoder(x)
        sequence_length = x.size(1)
        latent = hidden[-1].unsqueeze(1).repeat(1, sequence_length, 1)
        decoded, _ = self.decoder(latent)
        return self.output_layer(decoded)


def reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(reconstructed, target)


def create_sliding_windows(
    data,
    window_size: int,
    stride: int = 1,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError("Expected data shape: num_rows, num_features.")

    num_rows = tensor.size(0)
    if num_rows < window_size:
        return torch.empty((0, window_size, tensor.size(1)), dtype=tensor.dtype, device=tensor.device)

    windows = [
        tensor[start : start + window_size]
        for start in range(0, num_rows - window_size + 1, stride)
    ]
    return torch.stack(windows)
