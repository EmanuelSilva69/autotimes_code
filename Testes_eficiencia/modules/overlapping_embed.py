"""
OverlappingPatchEmbedding - Patch Embedding com sobreposicao via torch.unfold.

Diferente da particao estatica do AutoTimes (stride == patch_len), este
modulo fatia a serie temporal usando `Tensor.unfold` nativo do PyTorch com
sobreposicao entre janelas:
    Configuracao alvo: patch_len = 96, stride = 48 (50% de overlap).

O numero de patches para uma entrada de comprimento L:
    T = (L - patch_len) // stride + 1 = (672 - 96) // 48 + 1 = 13.

A reconstrucao (de-embedding) reverte a sobreposicao com weighted overlap-add
via `F.fold`: cada posicao coberta por mais de um patch recebe a media das
contribuicoes, eliminando costuras artificials entre janelas.

Shapes:
    - forward(   (N, L)             -> (N, T, D)  )  : tokenizacao + embedding.
    - deembed(   (N, T, D)          -> (N, C)     )  : decoding + re-folding,
      onde C = (T - 1) * stride + patch_len (comprimento coberto).
    `N = B * V` no caso multivariado (variaveis concatenadas no batch).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OverlappingPatchEmbedding(nn.Module):
    """Embedding por patches sobrepostos com restore por overlap-averaging.

    Args:
        patch_len: Tamanho de cada patch (janela de fatiamento).
        stride: Passo entre centros de patches consecutivos (<= patch_len).
        d_model: Dimensao do espaco latente de saida (ex: n_embd do GPT-2).
        dropout: Probabilidade de dropout no embedding linear.
    """

    def __init__(
        self,
        patch_len: int = 96,
        stride: int = 48,
        d_model: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if stride <= 0 or stride > patch_len:
            raise ValueError("stride deve satisfazer 0 < stride <= patch_len.")
        self.patch_len: int = patch_len
        self.stride: int = stride
        self.d_model: int = d_model

        self.embed_proj: nn.Linear = nn.Linear(patch_len, d_model)      # Fatura patch -> latente.
        self.deembed_proj: nn.Linear = nn.Linear(d_model, patch_len)   # Reverte latente->patch.
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def num_patches(self, length: int) -> int:
        """Numero de patches gerados para um sinal de comprimento `length`.

        Args:
            length: Comprimento temporal da sequencia de entrada.

        Returns:
            int: (length - patch_len) // stride + 1.
        """
        return (length - self.patch_len) // self.stride + 1

    def forward(self, x: Tensor) -> Tensor:
        """Tokeniza com torch.unfold e projeta para o espaco latente.

        Args:
            x: Sinal flat (N, L), onde L = seq_len e N = B * V.

        Returns:
            Tensor: Embeddings (N, T, d_model).
        """
        # unfold: (N, L) -> (N, T, patch_len). View diferenciavel nativa.
        patches: Tensor = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.dropout(self.embed_proj(patches))

    def deembed(self, tokens: Tensor, output_len: Optional[int] = None) -> Tensor:
        """Reconstroi a serie temporal a partir dos tokens com overlap-averaging.

        Args:
            tokens: (N, T, d_model) - saida do backbone (LLM).
            output_len: Comprimento final desejado. Se None, retorna o
                comprimento totalmente coberto C = (T - 1) * stride + patch_len.

        Returns:
            Tensor: (N, output_len) no dominio da serie (ja nao normalizado).
        """
        n, t, _ = tokens.shape
        patches: Tensor = self.deembed_proj(tokens)                       # (N, T, patch_len)
        # F.fold espera a representacao col2im (N, C * prod(kernel), L):
        # com kernel (1, patch_len) e C=1, ficamos com (n, patch_len, T).
        patches = patches.transpose(1, 2)                                 # (n, patch_len, T)

        covered: int = (t - 1) * self.stride + self.patch_len
        # Soma das contribuicoes de todas as janelas.
        folded: Tensor = F.fold(
            patches,
            output_size=(1, covered),
            kernel_size=(1, self.patch_len),
            stride=(1, self.stride),
        )                                                              # (n, 1, 1, covered)

        # Denominador: quantas janelas cobrem cada posicao temporal.
        counts: Tensor = torch.ones_like(patches)
        denominator: Tensor = F.fold(
            counts,
            output_size=(1, covered),
            kernel_size=(1, self.patch_len),
            stride=(1, self.stride),
        )                                                              # (n, 1, 1, covered)

        out: Tensor = (folded / denominator.clamp_min(1.0)).squeeze(1) # (n, 1, covered)
        out = out.reshape(n, covered)                                  # (n, covered)

        if output_len is not None:
            if output_len > covered:
                raise ValueError(
                    f"output_len {output_len} excede o comprimento coberto {covered}."
                )
            out = out[:, -output_len:]                                # (n, output_len)
        return out


if __name__ == "__main__":  # pragma: no cover - smoke test manual
    torch.manual_seed(0)
    n, l, d = 4, 672, 384
    emb = OverlappingPatchEmbedding(patch_len=96, stride=48, d_model=d)
    x = torch.randn(n, l)
    tok = emb(x)
    assert tok.shape == (n, emb.num_patches(l), d), tok.shape
    seq = emb.deembed(tok, output_len=96)
    assert seq.shape == (n, 96), seq.shape
    # No caso stride == patch_len (sem overlap), a reconstructao eh exata.
    emb_static = OverlappingPatchEmbedding(patch_len=96, stride=96, d_model=d)
    rec = emb_static.deembed(emb_static(x), output_len=None)
    print("Cobertura total (sem crop):", rec.shape[-1], "| esperado:", l)
    # Gradientes fluem pela reconstrucao com overlap.
    seq.sum().backward()
    assert emb.embed_proj.weight.grad is not None
    print("Smoke test OverlappingPatchEmbedding OK")