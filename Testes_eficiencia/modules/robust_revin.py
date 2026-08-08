"""
RobustRevIN - Normalizacao Reversivel Robusta (Mediana + MAD).

Substitui a media/desvio-padrao da RevIN classica por estatisticas robustas:
    - Centro: Mediana das instancias no eixo temporal.
    - Escala: MAD (Median Absolute Deviation) = mediana(|x - mediana|) * 1.4826,
      um estimador robusto do desvio-padrao sob presenca de outliers
      (fator 1.4826 ajusta a consistencia para distribuicoes gaussianas).

Mantem a capacidade de reversao afin (denorm): apos a predicao, o sinal
normalizado e deslocado/escalado de volta para o dominio original usando a
transformacao inversa, incluindo os parametros afin learnable (weight/bias).

Referencia conceitual: RevIN (Kim et al., 2022) com estatisticas robustas.

Convencoes de shape (documentadas em cada metodo):
    - forward(mode="norm"):   (B, L, V) -> (B, L, V)
    - forward(mode="denorm"): (B, L, V) -> (B, L, V)
    onde B = batch size, L = comprimento temporal, V = numero de variaveis.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn


class RobustRevIN(nn.Module):
    """Normalizacao de instancia robusta com reversao afin.

    Args:
        num_vars: Numero de variaveis (features) do sinal multivariado.
        affine: Se True, aplica transformacao afin learnable pos-normalizacao
            (gamma * x_hat + beta), como na RevIN original.
        eps: Pequena constante para estabilidade numerica da divisao.
        scale_factor: Multiplicador de consistencia do MAD (default 1.4826).
    """

    def __init__(
        self,
        num_vars: int,
        affine: bool = True,
        eps: float = 1e-5,
        scale_factor: float = 1.4826,
    ) -> None:
        super().__init__()
        self.num_vars: int = num_vars
        self.affine: bool = affine
        self.eps: float = eps
        self.scale_factor: float = scale_factor

        # Parametros afin learnable por variavel, como na RevIN original.
        if affine:
            self.affine_weight: nn.Parameter = nn.Parameter(torch.ones(num_vars))
            self.affine_bias: nn.Parameter = nn.Parameter(torch.zeros(num_vars))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        # Cache das estatisticas da ultima normalizacao (usado no denorm).
        self._cache: Tuple[Tensor, Tensor] | None = None

    # ------------------------------------------------------------------ #
    # Helpers internos                                                    #
    # ------------------------------------------------------------------ #
    def _robust_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Computa mediana e MAD por instancia ao longo do eixo temporal.

        Args:
            x: Sinal normalizado ou bruto, shape (B, L, V).

        Returns:
            median: shape (B, 1, V) - mediana ao longo do tempo.
            mad:    shape (B, 1, V) - mediana do desvio absoluto * 1.4826 + eps.
        """
        if x.ndim == 2:
            # Suporta entrada univariada (B, L): vista como (B, L, 1).
            x = x.unsqueeze(-1)
        median: Tensor = x.median(dim=1, keepdim=True).values          # (B, 1, V)
        mad_raw: Tensor = (x - median).abs().median(dim=1, keepdim=True).values  # (B, 1, V)
        mad: Tensor = mad_raw * self.scale_factor + self.eps           # (B, 1, V)
        return median, mad

    def _apply_affine(self, x: Tensor) -> Tensor:
        """Aplica (ou reverte) a transformacao afin learnable por variavel.

        Args:
            x: Shape (B, L, V).

        Returns:
            Tensor: Shape (B, L, V) apos o affine.
        """
        if not self.affine:
            return x
        weight: Tensor = self.affine_weight.view(1, 1, self.num_vars)
        bias: Tensor = self.affine_bias.view(1, 1, self.num_vars)
        return x * weight + bias

    def _undo_affine(self, x: Tensor) -> Tensor:
        """Inverso da transformacao afin: (x - bias) / weight.

        Args:
            x: Shape (B, L, V).

        Returns:
            Tensor: Shape (B, L, V) com o affine revertido.
        """
        if not self.affine:
            return x
        weight: Tensor = self.affine_weight.view(1, 1, self.num_vars)
        bias: Tensor = self.affine_bias.view(1, 1, self.num_vars)
        # Piso de magnitude que preserva o sinal: clamp_min() puro viraria um
        # peso negativo em +eps e quebraria a inversa afin.
        denom: Tensor = weight.sign() * weight.abs().clamp_min(self.eps)
        return (x - bias) / denom

    # ------------------------------------------------------------------ #
    # API publica                                                         #
    # ------------------------------------------------------------------ #
    def normalize(self, x: Tensor) -> Tensor:
        """Normaliza as instancias com mediana/MAD e aplica o affine.

        Args:
            x: Sinal bruto, shape (B, L, V).

        Returns:
            Tensor: Sinal normalizado, shape (B, L, V).
        """
        median, mad = self._robust_stats(x)
        self._cache = (median.detach().clone(), mad.detach().clone())
        x_hat: Tensor = (x - median) / mad
        return self._apply_affine(x_hat)

    def denormalize(self, x_hat: Tensor) -> Tensor:
        """Reconstrói o sinal original a partir do normalizado.

        Requer que `normalize` tenha sido chamado antes no mesmo batch
        (as estatisticas ficam em cache). Lanca RuntimeError caso contrario.

        Args:
            x_hat: Sinal normalizado, shape (B, L, V).

        Returns:
            Tensor: Sinal no dominio original, shape (B, L, V).
        """
        if self._cache is None:
            raise RuntimeError(
                "RobustRevIN.denormalize() chamado sem normalize() previa "
                "no mesmo batch (cache de estatisticas vazio)."
            )
        median, mad = self._cache
        if x_hat.ndim == 2:
            x_hat = x_hat.unsqueeze(-1)
        x_affine_undone: Tensor = self._undo_affine(x_hat)   # (B, L, V)
        return x_affine_undone * mad + median                # (B, L, V)

    def forward(self, x: Tensor, mode: str = "norm") -> Tensor:
        """Interface unificada de normalizacao/reversao.

        Args:
            x: Shape (B, L, V).
            mode: "norm" para normalizar, "denorm" para reverter.

        Returns:
            Tensor: Shape (B, L, V).
        """
        if mode == "norm":
            return self.normalize(x)
        if mode == "denorm":
            return self.denormalize(x)
        raise ValueError(f"Modo desconhecido '{mode}'. Use 'norm' ou 'denorm'.")

    def reset_cache(self) -> None:
        """Limpa o cache de estatisticas (util entre batches/inferencia)."""
        self._cache = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_vars={self.num_vars}, "
            f"affine={self.affine}, eps={self.eps}, "
            f"scale_factor={self.scale_factor})"
        )

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, eps={self.eps}"


if __name__ == "__main__":  # pragma: no cover - smoke test manual
    torch.manual_seed(0)
    revin = RobustRevIN(num_vars=3, affine=True)
    x = torch.randn(4, 672, 3) * torch.tensor([1.0, 5.0, 0.1]) + 2.0
    x_norm = revin.normalize(x)
    x_rec = revin.denormalize(x_norm)
    assert torch.allclose(x, x_rec, atol=1e-5), "Ciclo norm/denorm falhou!"
    # Injeta um outlier e verifica que a escala robusta nao explodiu.
    x_out = x.clone()
    x_out[0, 100, 1] = 1e4
    x_out[1, 300, 2] = -1e4
    x_norm_out = revin.normalize(x_out)
    print("Mediana/MAD estavel com outliers:", torch.isfinite(x_norm_out).all().item())
    print("Smoke test RobustRevIN OK - max |x_norm|:", x_norm_out.abs().max().item())
