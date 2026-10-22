"""
modelos_dual - AutoTimesDualModel com decomposicao Tendencia-Oscilacao.

Extends o pipeline textual do AutoTimes com Dual Embeddings numericos:
    1) Extrai Tendencia via regressao linear ao longo do eixo temporal;
    2) Isola Oscilacao = Original - Tendencia;
    3) Dois embeddings independentes (trend + oscillatory) processam cada
       componente separadamente;
    4) Soma dos embeddings vai para a fusao multimodal com o texto.

Restricao de seguranca: modulo 100% isolado dentro de testes_letras/.
Nao altera nenhum arquivo original do AutoTimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from . import caminhos
from .modelos import CKPTS, carregar_checkpoint


# --------------------------------------------------------------------- #
#  Componentes copiados de run_benchmarks.py (fiel ao original)          #
# --------------------------------------------------------------------- #
class _MLP(nn.Module):
    """MLP identico ao layers/mlp.py (linear + tanh + dropout)."""

    def __init__(self, f_in: int, f_out: int, hidden_dim: int = 256,
                 hidden_layers: int = 2, dropout: float = 0.1,
                 activation: str = "tanh") -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}.get(
            activation, nn.Tanh
        )()
        layers = [nn.Linear(f_in, hidden_dim), act, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class _StandardRevIN(nn.Module):
    """RevIN classica (media/desvio-padrao), identical a Run 0."""

    def __init__(self, num_vars: int, affine: bool = True,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.affine = affine
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_vars))
            self.affine_bias = nn.Parameter(torch.zeros(num_vars))
        self._cache: Optional[Tuple[Tensor, Tensor]] = None

    def normalize(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        self._cache = (mean, std)
        if self.affine:
            w = self.affine_weight.view(1, 1, self.num_vars)
            b = self.affine_bias.view(1, 1, self.num_vars)
            return (x - mean) / std * w + b
        return (x - mean) / std

    def denormalize(self, x_hat: Tensor) -> Tensor:
        if self._cache is None:
            raise RuntimeError("denormalize() sem normalize() previa")
        mean, std = self._cache
        if self.affine:
            w = self.affine_weight.view(1, 1, self.num_vars)
            b = self.affine_bias.view(1, 1, self.num_vars)
            return (x_hat - b) / w.clamp_min(self.eps) * std + mean
        return x_hat * std + mean


class _AdaptiveFusion(nn.Module):
    """Porta de fusao adaptativa do SMETimes."""

    def __init__(self, hidden_dim: int, n_stat_feats: int = 4) -> None:
        super().__init__()
        self.stat_proj = nn.Linear(n_stat_feats, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, times_embeds: Tensor, stat_feats: Tensor) -> Tensor:
        stat_embeds = self.stat_proj(stat_feats)
        combined = torch.cat([times_embeds, stat_embeds], dim=-1)
        gate = self.gate(combined)
        fused = gate * times_embeds + (1 - gate) * stat_embeds
        return self.norm(fused)


def _compute_token_stats(patches: Tensor) -> Tensor:
    """Estatisticas por patch: media, desvio, min, max -> (B, T, 4)."""
    return torch.stack([
        patches.mean(dim=-1),
        patches.std(dim=-1),
        patches.min(dim=-1).values,
        patches.max(dim=-1).values,
    ], dim=-1)


# --------------------------------------------------------------------- #
#  Decomposicao Tendencia-Oscilacao (diferenciavel, semloops)           #
# --------------------------------------------------------------------- #
class _MediaMovel(nn.Module):
    """Decomposicao Autoformer-style: trend = AvgPool1d(kernel, stride=1,
    padding replicado nas bordas."""

    def __init__(self, kernel: int = 25):
        super().__init__()
        assert kernel % 2 == 1, "kernel precisa ser impar"
        self.kernel = kernel

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L) ou (B, L, C) -> trend (B, L, C)."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)                              # (B,L,1)
        pad = (self.kernel - 1) // 2
        front = x[:, :1, :].repeat_interleave(pad, dim=1)
        end = x[:, -1:, :].repeat_interleave(pad, dim=1)
        xp = torch.cat([front, x, end], dim=1)               # (B,L+2p,C)
        win = xp.unfold(dimension=1, size=self.kernel, step=1)  # (B,L,C,k)
        return win.mean(dim=-1)                               # (B,L,C)


def _extrair_tendencia(x: Tensor) -> Tensor:
    """Regressao linear ao longo do eixo temporal (diferenciavel).

    Calcula a tendencia linear t = a*idx + b para cada posicao temporal,
    onde a e b sao os minimos quadrados ao longo de L.

    Args:
        x: (B, L, 1) tensor normalizado.

    Returns:
        trend: (B, L, 1) tendencia linear.
    """
    b, L, _ = x.shape
    idx = torch.arange(L, dtype=x.dtype, device=x.device)  # (L,)
    idx_mean = idx.mean()
    idx_centered = idx - idx_mean                           # (L,)
    idx_var = (idx_centered ** 2).sum()                     # escalar

    x_flat = x.squeeze(-1)                                 # (B, L)
    x_mean = x_flat.mean(dim=1, keepdim=True)              # (B, 1)
    x_centered = x_flat - x_mean                           # (B, L)

    # a = cov(idx, x) / var(idx) -> (B, 1)
    a = (x_centered @ idx_centered) / (idx_var + 1e-12)    # (B,)
    a = a.unsqueeze(1)                                      # (B, 1)
    b_coeff = x_mean - a * idx_mean                         # (B, 1)

    trend = a * idx_centered.unsqueeze(0) + b_coeff        # (B, L)
    return trend.unsqueeze(-1)                              # (B, L, 1)


# --------------------------------------------------------------------- #
#  AutoTimesDualModel                                                     #
# --------------------------------------------------------------------- #
class AutoTimesDualModel(nn.Module):
    """AutoTimes com Dual Embeddings (Tendencia + Oscilacao).

    Arquitetura:
        1) RevIN normalize sobre x_enc;
        2) Decomposicao: trend = linear_regression(x), osc = x - trend;
        3) trend_embedding(trend) -> trend_tokens (B, T, D);
        4) oscillatory_embedding(osc) -> osc_tokens (B, T, D);
        5) dual_gate GLU: num = g*trend + (1-g)*osc, bias init 1.0;
        6) txt_tokens = gpt2.wte(x_ids) + pooling;
        7) fusion(cat([txt, num])) -> GPT-2 -> decoder -> RevIN denorm.

    GPT-2 permanece 100% congelado. Apenas fusion, trend_embedding,
    oscillatory_embedding e decoder sao treinaveis.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        label_len: int,
        mlp_hidden_dim: int = 256,
        mlp_hidden_layers: int = 2,
        mlp_activation: str = "tanh",
        dropout: float = 0.1,
        gpt2_dir: Optional[Path] = None,
        torch_dtype: torch.dtype = torch.float32,
        use_robust: bool = False,
        use_overlap: bool = False,
        use_vol_gate: bool = False,
        unfreeze_layers: int = 0,
        textual: bool = False,
        block_max: int = 128,
        fusao: str = "glu",
        decomp: str = "linear",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.token_len = seq_len - label_len
        self.use_overlap = use_overlap
        self.use_vol_gate = use_vol_gate
        self.textual = textual
        self.block_max = int(block_max)
        if fusao not in ("glu", "soma"):
            raise ValueError(f"fusao invalida: {fusao!r} (use 'glu' ou 'soma')")
        self.fusao = fusao
        if decomp not in ("linear", "stl_simple"):
            raise ValueError(
                f"decomp invalida: {decomp!r} (use 'linear' ou 'stl_simple')"
            )
        self.decomp = decomp
        # MA-k25 Autoformer-style (sem parametros); usada so em stl_simple
        self._media_movel = _MediaMovel(kernel=25)

        # ---- GPT-2 (100% congelado, fp32 + sdpa) ----
        gpt2_dir = Path(gpt2_dir or caminhos.ROOT_DIR / "gpt2")
        if not gpt2_dir.exists():
            raise FileNotFoundError(
                f"GPT-2 nao encontrado: {gpt2_dir}. "
                "Baixe config.json + model.safetensors."
            )
        self.gpt2 = GPT2Model.from_pretrained(
            str(gpt2_dir),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        for p in self.gpt2.parameters():
            p.requires_grad = False
        hidden = int(self.gpt2.config.n_embd)

        # Unfreeze parcial (se solicitado)
        self.unfreeze_layers = unfreeze_layers
        if unfreeze_layers > 0:
            n_blocks = len(self.gpt2.h)
            start = max(n_blocks - unfreeze_layers, 0)
            for idx in range(start, n_blocks):
                for p in self.gpt2.h[idx].parameters():
                    p.requires_grad = True
            for p in self.gpt2.ln_f.parameters():
                p.requires_grad = True
            print(f"  [dual] descongeladas {n_blocks - start} ultimas "
                  f"camadas do GPT-2 (idx {start}..{n_blocks - 1}) + ln_f")

        # ---- RevIN ----
        self.revin = (
            _StandardRevIN(num_vars=1, affine=True)
            if use_robust
            else _StandardRevIN(num_vars=1, affine=False)
        )

        # ---- Dual Embeddings (NOVOS, treinaveis) ----
        # trend_embedding: projeta patches de tendencia para D
        # oscillatory_embedding: projeta patches de oscilacao para D
        if use_overlap:
            self.trend_embedding = _OverlappingPatchProjector(
                patch_len=self.token_len,
                stride=self.token_len // 2,
                d_model=hidden,
                dropout=dropout,
            )
            self.oscillatory_embedding = _OverlappingPatchProjector(
                patch_len=self.token_len,
                stride=self.token_len // 2,
                d_model=hidden,
                dropout=dropout,
            )
            self.n_patches = self.trend_embedding.num_patches(seq_len)
            self.decoder = None
        elif mlp_hidden_layers > 0:
            self.n_patches = seq_len // self.token_len
            self.trend_embedding = _MLPProjector(
                self.token_len, hidden, mlp_hidden_dim,
                mlp_hidden_layers, dropout, mlp_activation,
            )
            self.oscillatory_embedding = _MLPProjector(
                self.token_len, hidden, mlp_hidden_dim,
                mlp_hidden_layers, dropout, mlp_activation,
            )
            self.decoder = _MLP(
                hidden, self.token_len, mlp_hidden_dim,
                mlp_hidden_layers, dropout, mlp_activation,
            )
        else:
            self.n_patches = seq_len // self.token_len
            self.trend_embedding = _LinearProjector(self.token_len, hidden)
            self.oscillatory_embedding = _LinearProjector(self.token_len, hidden)
            self.decoder = nn.Linear(hidden, self.token_len)

        # ---- Fusao multimodal (concat -> Linear -> LayerNorm) ----
        self.n_blocks_text = seq_len // self.token_len
        if textual:
            self.fusion = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.LayerNorm(hidden),
            )
        else:
            self.fusion = None

        self.gate = _AdaptiveFusion(hidden, n_stat_feats=4) if use_vol_gate else None

        # ---- Dual Gate GLU (trend vs oscillation) ----
        # Apenas no braço 'glu'; 'soma' combina por adição (mesma decomposição,
        # sem params extras) para ablation gating vs decomposicao.
        if self.fusao == "glu":
            self.dual_gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.Sigmoid(),
            )
            nn.init.constant_(self.dual_gate[0].bias, 1.0)
        else:
            self.dual_gate = None

        # dtype do backbone congelado, cacheado (evita next(parameters()) por forward)
        self._gpt2_dtype: torch.dtype = next(self.gpt2.parameters()).dtype

    def _decompor(self, sinal: Tensor) -> Tuple[Tensor, Tensor]:
        """Decompoem sinal (N, L) em (trend, osc), ambos destacados do grafo.

        decomp='linear': regressao linear global por janela.
        decomp='stl_simple': media movel convolucional k=25, padding
            replicado (Autoformer-style); osc = x - trend.
        """
        if self.decomp == "stl_simple":
            trend = self._media_movel(sinal).squeeze(-1)          # (N,L)
        else:
            trend = _extrair_tendencia(sinal.unsqueeze(-1)).squeeze(-1)
        osc = sinal - trend
        # Decomposicao parameter-free: grafo desnecessario a jusante
        return trend.detach(), osc.detach()

    def forward(
        self,
        x_enc: Optional[Tensor] = None,
        x_ids: Optional[Tensor] = None,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        if self.textual:
            return self._forward_multimodal(x_enc, x_ids, x_lens)
        # Modo numerico puro (sem texto)
        b, l, v = x_enc.shape
        x_norm = self.revin.normalize(x_enc)
        flat = x_norm.permute(0, 2, 1).reshape(b * v, l)

        trend, osc = self._decompor(flat)                           # (N,L)x2

        if self.use_overlap:
            trend_tokens = self.trend_embedding(trend)
            osc_tokens = self.oscillatory_embedding(osc)
        else:
            t_patches = trend.unfold(-1, self.token_len, self.token_len)
            o_patches = osc.unfold(-1, self.token_len, self.token_len)
            trend_tokens = self.trend_embedding(t_patches)
            osc_tokens = self.oscillatory_embedding(o_patches)

        # Combinacao trend/osc: GLU gate ou soma simples (ablation)
        if self.dual_gate is not None:
            t_cat = torch.cat([trend_tokens, osc_tokens], dim=-1)
            t_gate = self.dual_gate(t_cat)
            num_tokens = t_gate * trend_tokens + (1 - t_gate) * osc_tokens
        else:
            num_tokens = trend_tokens + osc_tokens

        if self.gate is not None:
            num_tokens = self.gate(num_tokens, _compute_token_stats(
                flat.unfold(-1, self.token_len, self.token_len)
                if not self.use_overlap else
                flat.unfold(-1, self.token_len, self.token_len // 2)
            ))

        gpt2_dtype = self._gpt2_dtype
        latent = self.gpt2(
            inputs_embeds=num_tokens.to(gpt2_dtype).contiguous(), use_cache=False
        ).last_hidden_state

        if self.use_overlap:
            rec = self.trend_embedding.deembed(latent, output_len=self.seq_len)
        else:
            dec = self.decoder(latent)
            rec = dec.reshape(b * v, self.n_patches * self.token_len)

        rec = rec.view(b, v, self.seq_len).permute(0, 2, 1)
        return self.revin.denormalize(rec)

    def _forward_multimodal(
        self, x_enc: Tensor, x_ids: Tensor, x_lens: Tensor,
    ) -> Tensor:
        """Pipeline multimodal com Dual Embeddings.

        Passo A: RevIN -> decomposicao trend/osc -> dual embeddings
        Passo B: texto via wte congelado + pooling
        Passo C: fusion(cat([txt, trend+osc])) -> GPT-2 -> decoder -> denorm
        """
        b = x_ids.shape[0]
        if x_enc is None:
            raise ValueError("Modo textual exige x_enc numerico")

        # ---- Passo A: numerica com decomposicao ----
        x_norm = self.revin.normalize(x_enc)                     # (B,L,1)
        flat = x_norm.permute(0, 2, 1).reshape(b, self.seq_len) # (B,L)

        # Decomposicao (linear ou stl_simple, ver _decompor)
        trend, osc = self._decompor(flat)                        # (B,L)x2

        # Dual embeddings
        if self.use_overlap:
            trend_tokens = self.trend_embedding(trend)
            osc_tokens = self.oscillatory_embedding(osc)
        else:
            t_patches = trend.unfold(-1, self.token_len, self.token_len)
            o_patches = osc.unfold(-1, self.token_len, self.token_len)
            trend_tokens = self.trend_embedding(t_patches)
            osc_tokens = self.oscillatory_embedding(o_patches)

        # Combinacao trend/osc: GLU gate ou soma simples (ablation)
        if self.dual_gate is not None:
            t_cat = torch.cat([trend_tokens, osc_tokens], dim=-1)
            t_gate = self.dual_gate(t_cat)
            num_tokens = t_gate * trend_tokens + (1 - t_gate) * osc_tokens  # (B,T,D)
        else:
            num_tokens = trend_tokens + osc_tokens                          # (B,T,D)

        # ---- Passo B: texto (wte congelado + pooling chunked) ----
        txt_parts = []
        for pg in range(self.n_blocks_text):
            e_pg = self.gpt2.wte(x_ids[:, pg * self.block_max:(pg + 1) * self.block_max])
            m_pg = (torch.arange(self.block_max, device=e_pg.device)[None, :, None]
                    < x_lens[:, pg:pg + 1, None]).to(e_pg.dtype)
            txt_parts.append(
                (e_pg * m_pg).sum(dim=1)
                / x_lens[:, pg:pg + 1].clamp(min=1).to(e_pg.dtype)
            )
        txt_tokens = torch.stack(txt_parts, dim=1)                   # (B,Tt,D)

        # Alinha tokens de texto e numerico (overlap pode gerar mais)
        if txt_tokens.shape[1] != num_tokens.shape[1]:
            txt_tokens = torch.nn.functional.interpolate(
                txt_tokens.permute(0, 2, 1), size=num_tokens.shape[1],
                mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        if self.gate is not None:
            patches = flat.unfold(
                -1, self.token_len,
                self.token_len // 2 if self.use_overlap else self.token_len,
            )
            num_tokens = self.gate(num_tokens, _compute_token_stats(patches))

        # ---- Passo C: fusao + GPT-2 + decoder ----
        fused = torch.cat([txt_tokens, num_tokens], dim=-1)      # (B,T,2D)
        fused = self.fusion(fused)                                # (B,T,D)
        latent = self.gpt2(
            inputs_embeds=fused.to(self._gpt2_dtype).contiguous(), use_cache=False
        ).last_hidden_state                                       # (B,T,D)

        if self.use_overlap:
            rec = self.trend_embedding.deembed(latent, output_len=self.seq_len)
        else:
            dec = self.decoder(latent.to(
                next(self.decoder.parameters()).dtype))
            rec = dec.reshape(b, self.n_patches * self.token_len)

        rec = rec.view(b, 1, self.seq_len).permute(0, 2, 1)
        return self.revin.denormalize(rec)


# --------------------------------------------------------------------- #
#  Projetores (reuso de padroes do AutoTimes)                            #
# --------------------------------------------------------------------- #
class _LinearProjector(nn.Module):
    """Projecao linear simples: token_len -> hidden."""

    def __init__(self, token_len: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(token_len, hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)

    def deembed(self, latent: Tensor, output_len: int) -> Tensor:
        return self.proj(latent)


class _MLPProjector(nn.Module):
    """MLP projector: token_len -> hidden (reutiliza _MLP)."""

    def __init__(self, token_len: int, hidden: int, hidden_dim: int,
                 n_layers: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.mlp = _MLP(token_len, hidden, hidden_dim, n_layers,
                        dropout, activation)
        self.token_len = token_len

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def deembed(self, latent: Tensor, output_len: int) -> Tensor:
        # latent (B, T, D) -> rec (B, T, token_len) -> flatten
        rec = self.mlp(latent)
        return rec.reshape(latent.shape[0], -1)


class _OverlappingPatchProjector(nn.Module):
    """Overlapping patch embedding + projetor linear."""

    def __init__(self, patch_len: int, stride: int, d_model: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def num_patches(self, seq_len: int) -> int:
        return max(1, (seq_len - self.patch_len) // self.stride + 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L) -> unfold -> (B, T, patch_len) -> proj -> (B, T, D)
        patches = x.unfold(-1, self.patch_len, self.stride)
        return self.dropout(self.proj(patches))

    def deembed(self, latent: Tensor, output_len: int) -> Tensor:
        # Decodificacao simples: projeta de D para patch_len e reconstrui
        # Usando abordagem overlap-add simplificada
        B, T, D = latent.shape
        rec = torch.einsum("btd,dp->btp", latent, self.proj.weight.T)
        # Sobreposicao simples: media dos patches que se sobrepõem
        output = torch.zeros(B, output_len, device=latent.device,
                             dtype=rec.dtype)
        count = torch.zeros(B, output_len, device=latent.device,
                            dtype=rec.dtype)
        for t in range(T):
            start = t * self.stride
            end = min(start + self.patch_len, output_len)
            if end > start:
                output[:, start:end] += rec[:, t, :end - start]
                count[:, start:end] += 1
        output = output / count.clamp(min=1)
        return output


# --------------------------------------------------------------------- #
#  Funcao de montagem (drop-in de modelos.montar_modelo)                 #
# --------------------------------------------------------------------- #
def montar_modelo_dual(
    ckpt_path: Path,
    device: torch.device,
    use_amp: bool = False,
    gpt2_dir: Optional[Path] = None,
    block_max: int = 128,
    fusao: str = "glu",
    decomp: str = "linear",
) -> Tuple[torch.nn.Module, Dict]:
    """Reconstroi o AutoTimesDualModel a partir do checkpoint de ablacao.

    Carrega os pesos numericos (encoder, decoder, gate, revin) do checkpoint
    original. Os novos modulos (trend_embedding, oscillatory_embedding, fusion)
    nascem com pesos aleatorios e serao treinados do zero.

    Args:
        ckpt_path: checkpoint Run-8 (smetimes).
        device: cuda ou cpu.
        use_amp: mixed precision.
        gpt2_dir: diretorio do GPT-2 base.
        block_max: tokens por bloco.
        fusao: 'glu' (dual gate enviesado) ou 'soma' (adicao simples,
            mesmo numero de params treinaveis que o original).

    Returns:
        (model, info): modelo em device (modo eval) e dict de metadados.
    """
    ckpt = carregar_checkpoint(ckpt_path)
    flags: Dict = ckpt["flags"]
    args: Dict = ckpt["args"]

    torch_dtype = (
        torch.float16 if (use_amp and device.type == "cuda") else torch.float32
    )

    model = AutoTimesDualModel(
        seq_len=args["seq_len"],
        pred_len=args["pred_len"],
        n_vars=args["n_vars"],
        label_len=args["label_len"],
        mlp_hidden_dim=args["mlp_hidden_dim"],
        mlp_hidden_layers=args["mlp_hidden_layers"],
        mlp_activation=args["mlp_activation"],
        dropout=args["dropout"],
        gpt2_dir=Path(gpt2_dir or args["gpt2_dir"]),
        torch_dtype=torch_dtype,
        use_robust=bool(flags["use_robust"]),
        use_overlap=bool(flags["use_overlap"]),
        use_vol_gate=bool(flags["use_vol_gate"]),
        unfreeze_layers=0,
        textual=True,
        block_max=block_max,
        fusao=fusao,
        decomp=decomp,
    ).to(device)

    # Carrega pesos do checkpoint (ignora chaves novas = dual embeddings)
    old_state = ckpt["model_state_dict"]
    new_state = model.state_dict()
    # Mantem apenas chaves que existem nos dois e tenham mesmo shape
    to_load = {}
    for k in old_state:
        if k in new_state and old_state[k].shape == new_state[k].shape:
            to_load[k] = old_state[k]
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    model.eval()

    # Congela GPT-2
    for p in model.gpt2.parameters():
        p.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    gpt2_ok = all(not p.requires_grad for p in model.gpt2.parameters())

    print(f"  [dual] Parametros Treinaveis: {trainable / 1e6:.3f} milhoes | "
          f"Congelados: {frozen / 1e6:.3f} milhoes "
          f"({100.0 * trainable / max(total, 1):.2f}%)")
    print(f"  [dual] backbone GPT-2 100% congelado: {gpt2_ok}")
    if trainable > 3.5e6:
        print("  [dual] ATENCAO: treinaveis > 3.5M -> verificar congelamento!")

    info = {
        "checkpoint": str(ckpt_path),
        "flags": flags,
        "best_val_loss": ckpt.get("best_val_loss"),
        "best_epoch": ckpt.get("best_epoch"),
        "params_total": total,
        "params_treinaveis": trainable,
        "params_congelados": frozen,
        "gpt2_backbone_congelado": bool(gpt2_ok),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "arquitetura": f"AutoTimesDualModel (fusao={fusao}, decomp={decomp})",
    }
    return model, info
