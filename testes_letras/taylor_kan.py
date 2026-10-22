"""TaylorKAN Layer — Série de Taylor como camada neural.

Inspirado em: https://github.com/CUC-Chen/KAN4IQA
Versão fast (Horner) com tanh para estabilidade numérica.

tanh[-1,1] garante achatamento suave preservando derivadas contínuas,
sem matar gradientes como o clamp rígido. Companion com LayerNorm
antes da camada para normalizar latentes do GPT-2 (média 0, var 1).
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class TaylorLayer(nn.Module):
    """Camada que calcula série de Taylor até 'order' termos.

    y_j = bias_j + sum_{i=0}^{order-1} coeffs[j,:,i] * x^i

    Implementação fast via método de Horner (evita potências explícitas).
    Clamp [-1, 1] via tanh protege contra divergência da série.
    """

    def __init__(self, input_dim: int, out_dim: int, order: int = 3,
                 addbias: bool = True, use_tanh: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias
        self.use_tanh = use_tanh

        self.coeffs = nn.Parameter(torch.empty(out_dim, input_dim, order))
        init.kaiming_normal_(self.coeffs, a=0, mode='fan_in',
                             nonlinearity='linear')
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim) — aceita 2D ou 3D
        orig_shape = x.shape
        x = x.reshape(-1, self.input_dim)               # (N_flat, input_dim)

        # tanh [-1, 1]: achatamento suave preservando derivadas contínuas
        if self.use_tanh:
            x = torch.tanh(x)

        # Horner: y = c0 + x*(c1 + x*(c2 + ...))
        # Começa do coeficiente de ordem mais alta
        y = self.coeffs[:, :, -1]                        # (out_dim, input_dim)
        for i in reversed(range(self.order - 1)):
            y = y * x.unsqueeze(1) + self.coeffs[:, :, i]
        y = y.sum(dim=-1)                                # (N_flat, out_dim)

        if self.addbias:
            y = y + self.bias

        out_shape = orig_shape[:-1] + (self.out_dim,)
        return y.reshape(out_shape)
