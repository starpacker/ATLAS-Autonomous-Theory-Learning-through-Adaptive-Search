"""SciNet encoder-decoder model."""
from __future__ import annotations
import torch
import torch.nn as nn


class SciNet(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, output_dim: int,
                 encoder_hidden: list[int] | None = None,
                 decoder_hidden: list[int] | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim

        if encoder_hidden is None:
            encoder_hidden = [128, 64]
        if decoder_hidden is None:
            decoder_hidden = [64, 128]

        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in encoder_hidden:
            enc_layers.append(nn.Linear(prev_dim, h))
            enc_layers.append(nn.ReLU())
            prev_dim = h
        enc_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev_dim = bottleneck_dim
        for h in decoder_hidden:
            dec_layers.append(nn.Linear(prev_dim, h))
            dec_layers.append(nn.ReLU())
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
