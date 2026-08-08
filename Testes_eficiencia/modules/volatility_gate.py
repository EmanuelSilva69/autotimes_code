"""
volatility_gate - VolatilityScaledFusionGate.

Mecanismo de fusao por Cross-Attention entre o latente do LLM e as features
estatisticas da janela (Media, Variancia, Tendencia, Range).

Arquitetura:
    1. Query:  o vetor latente do LLM (pooled sobre os tokens) e projetado.
    2. Key/Value: as features estatisticas (B, V, 4) sao projetadas para o
       espaco do latente e atendidas pela Query (cross-attention por variavel).
    3. Sub-rede de volatilidade: uma MLP linear le o escalar isolado da
       Variancia (indice `variance_idx`) e produz um multiplicador positivo.
    4. Gate: ativacao Sigmoide escalada por esse multiplicador, forçando a
       "abertura" (gate ~ 1) sob regimes de alta volatilidade e permitindo
       que a informacao estatistica flua no blend com o latente.

Saida (blend) para cada token do latente:
    out = (1 - gate) * latent + gate * context
onde `context` e o resultado do cross-attention. Quando a volatilidade e alta,
`gate -> 1` e o modelo confia no contexto estatistico; quando baixa, o latente
do LLM domina.

Shapes:
    - latent: (B, V, T, D)  - hidden states do LLM reorganizados por variavel.
    - stats:  (B, V, S)     - features estatisticas (S=4).
    - out:    (B, V, T, D).
    - gate:   (B, V, 1).
    - attn:   (B, V, V)     - pesos de cross-attention (diagnostico).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VolatilityScaledFusionGate(nn.Module):
    """Gate de fusao com escala de amplitude por volatilidade.

    Args:
        d_model: Dimensao do latente do LLM (por token).
        n_stats: Numero de features estatisticas (default 4).
        n_heads: Numero de cabeca do cross-attention.
        variance_idx: Indice da variacao dentro de `stats` (default 1:
            layout [mean, var, trend, range]).
        hidden: Largura da sub-rede MLP da variancia.
    """

    def __init__(
        self,
        n_model: int = 384,
        n_stats: int = 4,
        n_heads: int = 8,
        variance_idx: int = 1,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if n_model % n_heads != 0:
            raise ValueError("n_model deve ser divisivel por n_heads.")
        self.n_model: int = n_model
        self.n_heads: int = n_heads
        self.head_dim: int = n_model // n_heads
        self.variance_idx: int = variance_idx
        self.scale: float = self.head_dim ** (-0.5)

        # Projecoes da cross-attention (latente do LLM -> Query; stats -> K/V).
        self.q_proj: nn.Linear = nn.Linear(n_model, n_model)
        self.k_proj: nn.Linear = nn.Linear(n_stats, n_model)
        self.v_proj: nn.Linear = nn.Linear(n_stats, n_model)
        self.out_proj: nn.Linear = nn.Linear(n_model, n_model)

        # Logito do portao (sigmoid) sobre o contexto agregado.
        self.gate_logit: nn.Linear = nn.Linear(n_model, 1)

        # Sub-rede linear que ler o escalar isolado da variancia e produz
        # o multiplicador que escala a amplitude do sigmoides.
        self.var_net: nn.Sequential = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _head_reshape(self, x: Tensor) -> Tensor:
        """Reorganiza (B, V, D) em (B, H, V, d_head) para attention multi-head.

        Args:
            x: (B, V, D) - projetado.

        Returns:
            Tensor: (B H, V, head_dim).
        """
        b, v, d = x.shape
        x = x.view(b, v, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        return x.reshape(b * self.n_heads, v, self.head_dim)

    def forward(
        self,
        latent: Tensor,
        stats: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Aplica o portao de fusao ciente de volatilidade.

        Args:
            latent: Hidden states do LLM, shape (B, V, T, n_model).
            stats: Features estatisticas, shape (B, V, n_stats).

        Returns:
            (out, gate, attention_weights):
                out    - (B, V, T, n_model), latente blendado com o contexto.
                gate   - (B, V, 1), valor do portao apos o scaling por var.
                attn_w - (B, V, V), cross-attention (diagnostico).
        """
        if latent.ndim != 4:
            raise ValueError(f"latent deve ser (B, V, T, D); recebido {tuple(latent.shape)}.")
        if stats.ndim != 3 or stats.shape[2] != self.k_proj.in_features:
            raise ValueError(
                f"stats deve ser (B, V, n_stats={self.k_proj.in_features}); "
                f"recebido {tuple(stats.shape)}."
            )

        b, v, t, d = latent.shape

        # Query: pool dos tokens do LLM por variavel e projecao.
        query_in: Tensor = latent.mean(dim=2)              # (B, V, d)
        query: Tensor = self._head_reshape(self.q_proj(query_in))  # (B*H, V, h)

        # Features estatisticas como Key/Value.
        key: Tensor = self._head_reshape(self.k_proj(stats))       # (B*H, V, h)
        value: Tensor = self._head_reshape(self.v_proj(stats))     # (B*H, V, h)

        attn_logits: Tensor = torch.matmul(
            query, key.transpose(-2, -1)
        ) * self.scale                                         # (B*H, V, V)
        attn_weights: Tensor = attn_logits.softmax(dim=-1)     # (B*H, V, V)
        context_mh: Tensor = torch.matmul(attn_weights, value) # (B*H, V, h)

        context = context_mh.view(b, self.n_heads, v, self.head_dim) \
            .permute(0, 2, 1, 3).reshape(b, v, d)             # (B, V, d)
        context = self.out_proj(context)                       # (B, V, d)

        # Sub-rede de volatilidade -> multiplicador de escala.
        variance: Tensor = stats[:, :, self.variance_idx : self.variance_idx + 1]  # (B, V, 1)
        multiplier: Tensor = 1.0 + F.softplus(self.var_net(variance))    # (B, V, 1), > 1

        # Gate: amplitude do sigmoide escalada pelo multiplicador.
        gate_logit: Tensor = self.gate_logit(context)          # (B, V, 1)
        gate: Tensor = (multiplier * torch.sigmoid(gate_logit)).clamp(0.0, 1.0)  # (B, V, 1)

        # Blend residual: o gate decide o peso entre latente e contexto.
        out: Tensor = (1.0 - gate).unsqueeze(-1) * latent + gate.unsqueeze(-1) * context.unsqueeze(-2)  # (B, V, T, d)

        attn_w = attn_weights.view(b, self.n_heads, v, v).mean(dim=1)  # (B, V, V)
        return out, gate, attn_w


if __name__ == "__main__":  # pragma: no cover - smoke test manual
    torch.manual_seed(0)
    gate = VolatilityScaledFusionGate(n_model=384, n_stats=4, n_heads=8)
    latent = torch.randn(4, 7, 13, 384, requires_grad=True)
    stats = torch.randn(4, 7, 4)
    out, g, w = gate(latent, stats)
    assert out.shape == latent.shape and g.shape == (4, 7, 1) and w.shape == (4, 7, 7)
    out.sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    # Semantica do mecanismo: com var_net monotona crescente e logit zerado,
    # alta volatilidade => multiplicador maior => o portao abre mais.
    # (Com pesos aleatorios de init o MLP nao eh monotono, por isso fixamos.)
    gate.var_net = nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(gate.var_net.weight, 1.0)
    gate.gate_logit = nn.Linear(384, 1, bias=True)
    torch.nn.init.zeros_(gate.gate_logit.weight)
    torch.nn.init.zeros_(gate.gate_logit.bias)
    stats_hi = torch.zeros_like(stats).index_fill_(2, torch.tensor(1), 50.0)
    stats_lo = torch.zeros_like(stats).index_fill_(2, torch.tensor(1), 0.01)
    _, g_hi, _ = gate(latent, stats_hi)
    _, g_lo, _ = gate(latent, stats_lo)
    print("Gate alto-vol medio:", g_hi.mean().item(), "| baixo-vol medio:", g_lo.mean().item())
    assert g_hi.mean() > g_lo.mean(), "O gate deveria abrir mais em alta volatilidade."
    print("Smoke test VolatilityScaledFusionGate OK")