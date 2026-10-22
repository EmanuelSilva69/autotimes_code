"""residual_adapters - Decoders residuais leves sobre MLP congelado.

3 variantes de adapter que adicionam um residuo treinavel ao decoder MLP
pre-treinado (congelado), preservando a memoria geometrica:

    y_final = y_mlp_congelado + alpha * y_adapter

Variantes:
    LinearResidualAdapter: projecao linear residual (Opcao B)
    TaylorResidualAdapter: serie de Taylor residual (Opcao A)
    MoLEResidualAdapter: mixture of linear experts residual (Opcao C)

Todas seguem o padrao:
    1. Copiam o MLP congelado como atributo (frozen_mlp)
    2. Constroem o adapter leve por cima
    3. Congelam o frozen_mlp explicitamente
    4. Inicializam o residuo em zero (y_final = y_mlp no step 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .taylor_kan import TaylorLayer


class LinearResidualAdapter(nn.Module):
    """MLP congelado + Linear residual treinavel (Opcao B).

    adapter(x) = Linear(768, 96), zero-initialized.
    alpha escala o residuo (comeca em 0.1).

    Params: 73,824 (adapter) + 1 (alpha) = 73,825
    """

    def __init__(
        self,
        frozen_mlp: nn.Module,
        hidden: int,
        token_len: int,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.frozen_mlp = frozen_mlp
        self.adapter = nn.Linear(hidden, token_len)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Zero-init: residuo comeca em 0, y_final = y_mlp no step 1
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

        # Congelar MLP interno
        for p in self.frozen_mlp.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_base = self.frozen_mlp(x)
        y_resid = self.adapter(x)
        return y_base + self.alpha * y_resid


class TaylorResidualAdapter(nn.Module):
    """MLP congelado + TaylorKAN residual treinavel (Opcao A).

    adapter(x) = LayerNorm -> TaylorLayer(order=2, tanh).
    Mais expressivo que Linear, menos instavel que ordem 3.

    Params: ~148K (adapter) + 1 (alpha)
    """

    def __init__(
        self,
        frozen_mlp: nn.Module,
        hidden: int,
        token_len: int,
        order: int = 2,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.frozen_mlp = frozen_mlp
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden),
            TaylorLayer(hidden, token_len, order=order, use_tanh=True),
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        for p in self.frozen_mlp.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_base = self.frozen_mlp(x)
        y_resid = self.adapter(x)
        return y_base + self.alpha * y_resid


class MoLEResidualAdapter(nn.Module):
    """MLP congelado + Mixture of Linear Experts residual (Opcao C).

    2 experts lineares (cada um Linear(768, 96)) com gating learned.
    y_adapter = g1(x) * expert1(x) + g2(x) * expert2(x)
    O gating pondera qual expert e mais relevante para cada token.

    Params: ~148K (2 experts + gate) + 1 (alpha)
    """

    def __init__(
        self,
        frozen_mlp: nn.Module,
        hidden: int,
        token_len: int,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.frozen_mlp = frozen_mlp

        # 2 experts lineares
        self.expert1 = nn.Linear(hidden, token_len)
        self.expert2 = nn.Linear(hidden, token_len)

        # Gating network: decide qual expert usar por token
        self.gate = nn.Linear(hidden, 2)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Zero-init dos experts (residuo comeca em 0)
        nn.init.zeros_(self.expert1.weight)
        nn.init.zeros_(self.expert1.bias)
        nn.init.zeros_(self.expert2.weight)
        nn.init.zeros_(self.expert2.bias)

        for p in self.frozen_mlp.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_base = self.frozen_mlp(x)

        # Gating: (B, T, 2) -> softmax -> pesos por expert
        gates = torch.softmax(self.gate(x), dim=-1)  # (B, T, 2)
        e1 = self.expert1(x)  # (B, T, 96)
        e2 = self.expert2(x)  # (B, T, 96)
        y_resid = gates[..., 0:1] * e1 + gates[..., 1:2] * e2

        return y_base + self.alpha * y_resid
