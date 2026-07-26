from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class NeuralModelError(RuntimeError):
    """Raised when a governed neural model cannot be constructed safely."""


@dataclass(frozen=True)
class NeuralModelIdentity:
    algorithm_id: str
    parameter_count: int
    context_length: int
    input_dimension: int
    horizon: int


def configure_deterministic_cpu(seed: int) -> None:
    """Apply the deterministic CPU controls frozen by Gate 6C1."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


class _DenseResidualBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.block(values))


class _NHiTSBlock(nn.Module):
    def __init__(
        self,
        *,
        context_length: int,
        input_dimension: int,
        hidden_size: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        if context_length % pool_size != 0:
            raise NeuralModelError("N-HiTS context length must divide by pool size")
        self.context_length = context_length
        self.input_dimension = input_dimension
        self.pool_size = pool_size
        pooled_length = context_length // pool_size
        flattened = pooled_length * input_dimension
        self.encoder = nn.Sequential(
            nn.Linear(flattened, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden_size, flattened)
        self.forecast = nn.Linear(hidden_size, 1)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.avg_pool1d(
            residual.transpose(1, 2),
            kernel_size=self.pool_size,
            stride=self.pool_size,
        ).transpose(1, 2)
        hidden = self.encoder(pooled.flatten(start_dim=1))
        backcast = self.backcast(hidden).view(
            residual.shape[0],
            self.context_length // self.pool_size,
            self.input_dimension,
        )
        if self.pool_size > 1:
            backcast = backcast.repeat_interleave(self.pool_size, dim=1)
        return residual - backcast, self.forecast(hidden)


class CompactNHiTS(nn.Module):
    """Compact hierarchical interpolation model for a one-step horizon."""

    def __init__(
        self,
        *,
        context_length: int,
        input_dimension: int,
        hidden_size: int,
        stack_count: int,
        block_count_per_stack: int,
    ) -> None:
        super().__init__()
        if stack_count != 2 or block_count_per_stack != 1:
            raise NeuralModelError("Gate 6C compact N-HiTS geometry changed")
        self.blocks = nn.ModuleList(
            [
                _NHiTSBlock(
                    context_length=context_length,
                    input_dimension=input_dimension,
                    hidden_size=hidden_size,
                    pool_size=1,
                ),
                _NHiTSBlock(
                    context_length=context_length,
                    input_dimension=input_dimension,
                    hidden_size=hidden_size,
                    pool_size=4,
                ),
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        forecast = torch.zeros((values.shape[0], 1), dtype=values.dtype, device=values.device)
        for block in self.blocks:
            residual, block_forecast = block(residual)
            forecast = forecast + block_forecast
        return forecast.squeeze(-1)


class CompactTiDE(nn.Module):
    """Compact dense encoder-decoder with a known-covariate skip path."""

    def __init__(
        self,
        *,
        context_length: int,
        input_dimension: int,
        hidden_size: int,
        encoder_layers: int,
        decoder_layers: int,
    ) -> None:
        super().__init__()
        if encoder_layers != 2 or decoder_layers != 2:
            raise NeuralModelError("Gate 6C compact TiDE layer count changed")
        self.history_projection = nn.Linear(context_length * input_dimension, hidden_size)
        self.known_covariate_projection = nn.Linear(input_dimension, hidden_size)
        self.encoder = nn.Sequential(
            *[_DenseResidualBlock(hidden_size) for _ in range(encoder_layers)]
        )
        self.decoder = nn.Sequential(
            *[_DenseResidualBlock(hidden_size) for _ in range(decoder_layers)]
        )
        self.direct_skip = nn.Linear(input_dimension, 1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        latest = values[:, -1, :]
        hidden = self.history_projection(values.flatten(start_dim=1))
        hidden = self.encoder(hidden)
        hidden = hidden + self.known_covariate_projection(latest)
        hidden = self.decoder(hidden)
        return (self.output(hidden) + self.direct_skip(latest)).squeeze(-1)


class CompactPatchTST(nn.Module):
    """Compact patch-based transformer for multivariate temporal contexts."""

    def __init__(
        self,
        *,
        context_length: int,
        input_dimension: int,
        patch_length: int,
        stride: int,
        hidden_size: int,
        attention_heads: int,
        encoder_layers: int,
    ) -> None:
        super().__init__()
        if hidden_size % attention_heads != 0:
            raise NeuralModelError("PatchTST hidden size must divide by attention heads")
        if context_length < patch_length:
            raise NeuralModelError("PatchTST patch length exceeds context length")
        self.patch_length = patch_length
        self.stride = stride
        patch_count = 1 + (context_length - patch_length) // stride
        self.patch_projection = nn.Linear(patch_length * input_dimension, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, patch_count, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.normalization = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        patches = values.unfold(1, self.patch_length, self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        embedded = self.patch_projection(patches.flatten(start_dim=2))
        encoded = self.encoder(embedded + self.position)
        pooled = self.normalization(encoded.mean(dim=1))
        return self.output(pooled).squeeze(-1)


def build_neural_model(
    algorithm_id: str,
    *,
    context_length: int,
    input_dimension: int,
    configuration: dict[str, Any],
) -> nn.Module:
    """Build one frozen Gate 6C neural candidate."""

    if algorithm_id == "nhits_compact":
        return CompactNHiTS(
            context_length=context_length,
            input_dimension=input_dimension,
            hidden_size=int(configuration["hidden_size"]),
            stack_count=int(configuration["stack_count"]),
            block_count_per_stack=int(configuration["block_count_per_stack"]),
        )
    if algorithm_id == "tide_compact":
        return CompactTiDE(
            context_length=context_length,
            input_dimension=input_dimension,
            hidden_size=int(configuration["hidden_size"]),
            encoder_layers=int(configuration["encoder_layers"]),
            decoder_layers=int(configuration["decoder_layers"]),
        )
    if algorithm_id == "patchtst_compact":
        return CompactPatchTST(
            context_length=context_length,
            input_dimension=input_dimension,
            patch_length=int(configuration["patch_length"]),
            stride=int(configuration["stride"]),
            hidden_size=int(configuration["hidden_size"]),
            attention_heads=int(configuration["attention_heads"]),
            encoder_layers=int(configuration["encoder_layers"]),
        )
    raise NeuralModelError(f"Unsupported Gate 6C algorithm: {algorithm_id}")


def model_identity(
    algorithm_id: str,
    model: nn.Module,
    *,
    context_length: int,
    input_dimension: int,
) -> NeuralModelIdentity:
    return NeuralModelIdentity(
        algorithm_id=algorithm_id,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        context_length=context_length,
        input_dimension=input_dimension,
        horizon=1,
    )
