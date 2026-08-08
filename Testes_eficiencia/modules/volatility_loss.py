"""
VolatilityAwareLogCoshLoss - Perda Log-Cosh ciente da volatilidade.

A perda Log-Cosh combina a suavidade do Huber com diferenciabilidade total:
    L_logcosh(u) = log(cosh(u)),  u = y_pred - y_true.

Implementacao numerica estavel (evita overflow em |u| grande):
    log(cosh(u)) = u + softplus(-2u) - ln(2).

Atenuacao por volatilidade: o resultado e multiplicado por um peso
inversamente proporcional a variancia da janela:
    weight = exp(-beta * variance),
ou seja, janelas de alta volatilidade contribuem menos para o gradiente,
evitando que regimes turbulentos dominem a otimizacao.

Shapes:
    - forward(y_pred (B, L, V), y_true (B, L, V), variance (B,) | (B, 1) | (B, V)) -> scalar.
    O tensor `variance` e tratado como feature injetada e portanto
    destacado do grafo computacional (.detach()) antes do peso.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VolatilityAwareLogCoshLoss(nn.Module):
    """Perda Log-Cosh com peso de atenuacao exponencial por variancia.

    Args:
        beta: Coeficiente de atenuacao (>= 0). Quanto maior, mais agressiva
            a supressao das janelas de alta volatilidade.
        reduction: 'mean' (default) ou 'sum'.
        eps: Constante de estabilidade para clamp da variancia.
    """

    def __init__(self, beta: float = 0.5, reduction: str = "mean", eps: float = 1e-6) -> None:
        super().__init__()
        if beta < 0:
            raise ValueError("beta deve ser >= 0.")
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction deve ser 'mean' ou 'sum'.")
        self.beta: float = beta
        self.reduction: str = reduction
        self.eps: float = eps

    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_cosh(u: Tensor) -> Tensor:
        """Log-Cosh estavel: u + softplus(-2u) - ln(2).

        Args:
            u: Diferenca (y_pred - y_true), shape arbitraria.

        Returns:
            Tensor: log(cosh(u)), mesma shape de `u`.
        """
        return u + F.softplus(-2.0 * u) - math.log(2.0)

    def _broadcast_variance(self, variance: Tensor, pred_ndim: int) -> Tensor:
        """Redimensiona `variance` para broadcast contra (B, L, V).

        Aceita (B,), (B, 1) ou (B, V); converte para (B, 1, 1) ou (B, 1, V).

        Args:
            variance: Tensor de variancia por instancia, shape (B,) | (B, 1) | (B, V).
            pred_ndim: Numero de dimensoes de y_pred (2 ou 3).

        Returns:
            Tensor: Variance com shape broadcastavel.
        """
        if variance.ndim == 1:
            shape = (-1,) + (1,) * (pred_ndim - 1)
            return variance.reshape(*shape)
        if variance.ndim == 2 and pred_ndim == 3:
            # (B, 1) -> (B, 1, 1)  |  (B, V) -> (B, 1, V)
            return variance.unsqueeze(1)
        return variance

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        variance: Optional[Tensor] = None,
    ) -> Tensor:
        """Computa a perda Log-Cosh ponderada pela volatilidade.

        Args:
            y_pred: Predicoes, shape (B, L, V).
            y_true: Alvos, shape (B, L, V).
            variance: Feature estatistica da janela (variancia da instancia),
                shape (B,) | (B, 1) | (B, V). Se None, vira Log-Cosh puro.

        Returns:
            Tensor: Perda escalar (reduction='mean') ou soma.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred {tuple(y_pred.shape)} e y_true {tuple(y_true.shape)} "
                "devem ter a mesma shape."
            )

        u: Tensor = y_pred - y_true                                  # (B, L, V)
        log_cosh: Tensor = self._log_cosh(u)                        # (B, L, V)

        if variance is not None:
            # Feature injetada: destacada do grafo (nao deve receber gradiente).
            var = variance.detach().clamp(min=0.0)                  # (B, ...)
            weight: Tensor = torch.exp(-self.beta * var)            # (B, ...)
            weight = self._broadcast_variance(weight, y_pred.ndim)  # (B, 1, 1)/(B, 1, V)
            log_cosh = log_cosh * weight

        if self.reduction == "mean":
            return log_cosh.mean()
        return log_cosh.sum()


class HybridVolatilityLoss(nn.Module):
    """Loss hibrida: VolatilityAwareLogCosh + penalidade robusta (L1/Huber).

    Combina a atenuacao por volatilidade do Log-Cosh com um termo de regressao
    robusto no residuo (L1 ou Huber), reduzindo o impacto de outliers sem
    perder o foco nas janelas de baixa volatilidade.

    Args:
        beta: Coeficiente de atenuacao da volatilidade (repassa ao termo vol).
        lambda_reg: Peso do termo robusto (0 = soh o termo Log-Cosh volatil).
        robust: 'l1' (nn.L1Loss) ou 'huber' (nn.HuberLoss) para o termo robusto.
        huber_delta: delta do Huber quando robust='huber'.
        reduction: 'mean' (default) aplicado a cada termo separadamente.
        eps: Constante de estabilidade (repassada ao termo volatil).
    """

    def __init__(
        self,
        beta: float = 0.5,
        lambda_scale: float = 0.5,
        robust: str = "huber",
        huber_delta: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if robust not in ("l1", "huber"):
            raise ValueError("robust deve ser 'l1' ou 'huber'.")
        self.lambda_scale: float = lambda_scale
        self.robust: str = robust
        self.volatility = VolatilityAwareLogCoshLoss(beta=beta, reduction=reduction, eps=eps)
        if robust == "huber":
            self.robust_fc = nn.HuberLoss(reduction=reduction, delta=huber_delta)
        else:
            self.robust_fc = nn.L1Loss(reduction=reduction)

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        variance: Optional[Tensor] = None,
    ) -> Tensor:
        """Base volatil (Log-Cosh) + lambda * termo robusto (L1/Huber).

        Args:
            y_pred: Predicoes, shape (B, L, V).
            y_true: Alvos, shape (B, L, V).
            variance: Feature estatistica da janela, (B,) | (B, 1) | (B, V).

        Returns:
            Tensor: Perda escalar (reduction='mean') ou soma.
        """
        base: Tensor = self.volatility(y_pred, y_true, variance)
        robust_loss: Tensor = self.robust_fc(y_pred, y_true)
        return base + self.lambda_scale * robust_loss


if __name__ == "__main__":  # pragma: no cover - smoke test manual
    torch.manual_seed(0)
    loss_fn = VolatilityAwareLogCoshLoss(beta=0.5)
    pred = torch.randn(8, 96, 7, requires_grad=True)
    target = torch.randn(8, 96, 7)
    var = torch.tensor([0.1, 0.5, 2.0, 5.0, 0.3, 1.0, 4.0, 0.8])  # (B,)

    loss = loss_fn(pred, target, var)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    # Log-Cosh puro deve coincidir com a formula direta (precisao dupla).
    pure = loss_fn(pred.detach(), target, None)
    direct = torch.log(torch.cosh(pred.detach() - target)).mean()
    print("Log-Cosh puro - close:", torch.allclose(pure, direct, atol=1e-4))
    # Janela mais volatil deve ter peso menor.
    w_lo = math.exp(-0.5 * 0.1)
    w_hi = math.exp(-0.5 * 5.0)
    print("Pesos: baixa vol =", round(w_lo, 4), "| alta vol =", round(w_hi, 4))
    print("Smoke test VolatilityAwareLogCoshLoss OK")
