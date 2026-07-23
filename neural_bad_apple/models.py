from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SpatialAttention(nn.Module):
    """A small CBAM-style gate that learns where bottleneck features matter."""

    def __init__(self) -> None:
        super().__init__()
        self.to_attention = nn.Conv2d(
            2, 1, kernel_size=7, stride=1, padding=3, bias=False
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        average = features.mean(dim=1, keepdim=True)
        maximum = features.amax(dim=1, keepdim=True)
        attention = torch.sigmoid(
            self.to_attention(torch.cat((average, maximum), dim=1))
        )
        return features * attention, attention


class PixelActivationAutoencoder(nn.Module):
    """Convolutional autoencoder whose output logits are pixel-neuron activations."""

    def __init__(
        self,
        base_channels: int = 16,
        latent_channels: int = 64,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.encoder = nn.Sequential(
            ConvBlock(1, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, latent_channels),
        )
        self.attention = SpatialAttention() if use_attention else None
        self.decoder = nn.Sequential(
            UpBlock(latent_channels, base_channels * 4),
            UpBlock(base_channels * 4, base_channels * 2),
            UpBlock(base_channels * 2, base_channels),
            nn.ConvTranspose2d(
                base_channels, 1, kernel_size=4, stride=2, padding=1
            ),
        )

    def forward(
        self, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        height, width = pixels.shape[-2:]
        pad_bottom = (-height) % 16
        pad_right = (-width) % 16
        if pad_bottom or pad_right:
            pixels = F.pad(pixels, (0, pad_right, 0, pad_bottom))

        latent = self.encoder(pixels)
        extras: dict[str, torch.Tensor] = {}
        if self.attention is not None:
            latent, attention = self.attention(latent)
            extras["attention"] = attention

        logits = self.decoder(latent)
        logits = logits[..., :height, :width]
        return logits, extras


MODEL_NAMES = ("basic", "attention")


def build_model(
    name: str,
    base_channels: int = 16,
    latent_channels: int = 64,
) -> tuple[nn.Module, dict[str, Any]]:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose one of {MODEL_NAMES}.")
    kwargs = {
        "base_channels": base_channels,
        "latent_channels": latent_channels,
    }
    model = PixelActivationAutoencoder(
        **kwargs,
        use_attention=name == "attention",
    )
    return model, kwargs
