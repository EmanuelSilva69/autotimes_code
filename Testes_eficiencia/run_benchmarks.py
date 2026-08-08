"""
run_benchmarks.py - Ablation Study do AutoTimes, fiel a bateria (rodar_bateria.ps1).

O baseline (Run 0) replica o pipeline exato do run.py long_term_forecast
com AutoTimes_Gpt2:
  - Dataset_ETT_hour / Dataset_Custom: janelas UNIVARIADAS por variavel
    (feat_id = index // tot_len), seq_x = [s_begin, s_begin+seq_len),
    seq_y = [s_end-label_len, s_end+pred_len), z-score ajustado no treino;
  - RevIN z-score, tokenizer/detokenizer MLP (256x2, tanh), GPT-2 base REAL
    congelado (torch_dtype=float16 com AMP, float32 sem), reconstrucao do
    comprimento completo (seq_len);
  - Adam lr=0.001 + CosineAnnealingLR(T_max=10, eta_min=1e-8);
  - AMP (autocast + GradScaler), early stopping (patience=3) com reload do
    melhor checkpoint antes do teste (ultimos pred_len pontos).

As 7 runs do ablation injetam os modulos de melhoria sobre esse baseline:
    Run 0 Padrao | Run 1 RobustRevIN | Run 2 VolatilityLoss
    Run 3 OverlapEmbed | Run 4 DiscriminativeOptimizer
    Run 5 VolatilityGate (== SMETimes: fusao de features estatisticas)
    Run 6 Frankenstein

Saidas:
    - log/training_history.log     (linhas por epoca)
    - log/benchmark_results.json   (metricas finais e curvas de convergencia)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm opcional
    _tqdm = None

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
LOG_DIR = BASE_DIR / "log"
PLOTS_DIR = BASE_DIR / "plots"
CKPT_DIR = BASE_DIR / "checkpoints"
TEXT_CACHE_DIR = BASE_DIR / "text_cache"
GPT2_DIR = ROOT_DIR / "gpt2"

_SESSION_STAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")


def _session_stamp() -> str:
    """Timestamp unico da sessaoo (mesmo arquivo de log em todos os runs)."""
    return _SESSION_STAMP

for _p in (str(BASE_DIR), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import utils.metrics as _metrics  # noqa: E402

from modules import (  # noqa: E402
    RobustRevIN,
    VolatilityAwareLogCoshLoss,
    HybridVolatilityLoss,
    OverlappingPatchEmbedding,
    build_cosine_warm_scheduler,
    build_discriminative_optimizer,
)
from modules import textual as _textual  # noqa: E402

# Versao do schema de serializacao textual. Bump ao mudar a formatacao dos
# textos (pt-BR) para invalidar caches antigos em text_cache\.
_TEXTUAL_VERSION: int = 2

SEED: int = 2021


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("  [device] CUDA/ROCm indisponivel -> fallback para CPU")
    return torch.device("cpu")


def _amp_ctx(device: torch.device, enabled: bool):
    """Contexto de autocast (AMP) para GPU; nullcontext em CPU."""
    if device.type == "cuda" and enabled:
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def verify_device_placement(model: nn.Module, device: torch.device) -> None:
    for name, p in model.named_parameters():
        if p.device.type != device.type:
            raise AssertionError(f"parametro '{name}' em {p.device}, esperado {device}")
    if torch.cuda.is_available():
        print(f"  [device] {torch.cuda.get_device_name(0)} | parametros em cuda ok")


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    mae, mse, _, _, _ = _metrics.metric(pred, true)
    return float(mse), float(mae)


def check_gradients(model: nn.Module) -> Dict[str, int]:
    stats = {"trainable": 0, "missing": 0, "nan": 0, "inf": 0}
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        stats["trainable"] += 1
        if p.grad is None:
            stats["missing"] += 1
            continue
        if torch.isnan(p.grad).any():
            stats["nan"] += 1
        if torch.isinf(p.grad).any():
            stats["inf"] += 1
    return stats


# --------------------------------------------------------------------- #
# Tokenizer/detokenizer MLP identico ao layers/mlp.py da bateria          #
# --------------------------------------------------------------------- #
class MLP(nn.Module):
    """MLP identico ao layers/mlp.py (linear + tanh + dropout, 2 camadas)."""

    def __init__(
        self,
        f_in: int,
        f_out: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError(f"ativacao desconhecida: {activation}")

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for _ in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class StandardRevIN(nn.Module):
    """RevIN classica (media/desvio-padrao), baseline da Run 0.

    Com affine=False corresponde EXATAMENTE ao z-score do AutoTimes_Gpt2
    (means/stdev + 1e-5). Shapes: (B,L,V)->(B,L,V).
    """

    def __init__(self, num_vars: int, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.affine = affine
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_vars))
            self.affine_bias = nn.Parameter(torch.zeros(num_vars))
        self._cache: Optional[Tuple[Tensor, Tensor]] = None

    def _apply_affine(self, x: Tensor) -> Tensor:
        if not self.affine:
            return x
        w = self.affine_weight.view(1, 1, self.num_vars)
        b = self.affine_bias.view(1, 1, self.num_vars)
        return x * w + b

    def _undo_affine(self, x: Tensor) -> Tensor:
        if not self.affine:
            return x
        w = self.affine_weight.view(1, 1, self.num_vars)
        b = self.affine_bias.view(1, 1, self.num_vars)
        return (x - b) / w.clamp_min(self.eps)

    def normalize(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        self._cache = (mean, std)
        return self._apply_affine((x - mean) / std)

    def denormalize(self, x_hat: Tensor) -> Tensor:
        if self._cache is None:
            raise RuntimeError("denormalize() sem normalize() previa no batch")
        mean, std = self._cache
        return self._undo_affine(x_hat) * std + mean

    def forward(self, x: Tensor, mode: str = "norm") -> Tensor:
        return self.normalize(x) if mode == "norm" else self.denormalize(x)


# --------------------------------------------------------------------- #
# SMETimes: features estatisticas por token + AdaptiveFusion (bateria)   #
# --------------------------------------------------------------------- #
def compute_token_stats(patches: Tensor) -> Tensor:
    """Features estatisticas por token (media, var, tendencia OLS, range).

    Mesmas features do AutoTimes_Gpt2_SMETimes da bateria, calculadas sobre
    os patches (janelas) de cada token. Force FP32 para estabilidade sob AMP.

    Args:
        patches: (N, T, token_len).

    Returns:
        Tensor: (N, T, 4) no dtype original do input.
    """
    p32 = patches.float()
    mean = p32.mean(dim=-1, keepdim=True)
    var = p32.var(dim=-1, unbiased=False, keepdim=True)
    rng = p32.max(dim=-1, keepdim=True).values - p32.min(dim=-1, keepdim=True).values
    t = torch.arange(patches.shape[-1], dtype=torch.float32, device=patches.device)
    tc = t - t.mean()
    trend = ((p32 - mean) * tc).sum(dim=-1, keepdim=True) / (tc ** 2).sum()
    return torch.cat([mean, var, trend, rng], dim=-1).to(patches.dtype)


class AdaptiveFusion(nn.Module):
    """Porta de fusao adaptativa do SMETimes (bateria AutoTimes_Gpt2_SMETimes)."""

    def __init__(self, hidden_dim: int, n_stat_feats: int = 4) -> None:
        super().__init__()
        self.stat_proj = nn.Linear(n_stat_feats, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, times_embeds: Tensor, stat_feats: Tensor) -> Tensor:
        stat_embeds = self.stat_proj(stat_feats)
        combined = torch.cat([times_embeds, stat_embeds], dim=-1)
        gate = self.gate(combined)
        fused = gate * times_embeds + (1 - gate) * stat_embeds
        return self.norm(fused)


# --------------------------------------------------------------------- #
# Modelo: replica fiel do AutoTimes_Gpt2 (+ modulos do ablation)         #
# --------------------------------------------------------------------- #
class AutoTimesModel(nn.Module):
    """Modelo fiel ao AutoTimes_Gpt2 do run.py com injecao dos modulos.

    Args:
        seq_len: Janela de entrada (672 na bateria).
        pred_len: Horizonte (96 na bateria).
        n_vars: Variaveis do dataset (meta; batch e univariado por janela).
        label_len: Contexto (576 na bateria); token_len = seq_len - label_len.
        mlp_hidden_dim/mlp_hidden_layers/mlp_activation: tokenizer MLP.
        dropout: Dropout do MLP.
        gpt2_dir: Diretorio local do GPT-2 base.
        torch_dtype: float16 com AMP (bateria) ou float32.
        use_robust/use_overlap/use_vol_gate: flags de injecao.
        unfreeze_layers: Numero de ULTIMAS camadas do GPT-2 a descongelar
            (0 = backbone completamente congelado, como a bateria).
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

        gpt2_dir = Path(gpt2_dir or GPT2_DIR)
        if not gpt2_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint local do GPT-2 nao encontrado: {gpt2_dir}. "
                "Baixe o GPT-2 base (config.json + model.safetensors) para esse diretorio."
            )
        self.gpt2 = GPT2Model.from_pretrained(
            str(gpt2_dir),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        for p in self.gpt2.parameters():  # congelado, como AutoTimes_Gpt2.py
            p.requires_grad = False
        hidden = int(self.gpt2.config.n_embd)

        # Descongelamento parcial (finetune): as ULTIMAS camadas do GPT-2
        # tornam-se treinaveis (plus ln_f) com uma LR muito baixa (lr_llm,
        # ex: 1e-6), via build_discriminative_optimizer quando use_disc_optim.
        self.unfreeze_layers = unfreeze_layers
        if unfreeze_layers > 0:
            n_blocks = len(self.gpt2.h)
            start = max(n_blocks - unfreeze_layers, 0)
            for idx in range(start, n_blocks):
                for p in self.gpt2.h[idx].parameters():
                    p.requires_grad = True
            for p in self.gpt2.ln_f.parameters():
                p.requires_grad = True
            print(f"  [model] descongeladas {n_blocks - start} ultimas camadas "
                  f"do GPT-2 (idx {start}..{n_blocks - 1}) + ln_f")

        self.revin = (
            RobustRevIN(num_vars=1, affine=True)
            if use_robust
            else StandardRevIN(num_vars=1, affine=False)  # z-score da bateria
        )

        if textual:
            # Modo textual: o encoder é o embedding textual (wte) do GPT-2;
            # o decoder MLP mapeia cada bloco textual de volta para valores.
            # Textual é mutuamente exclusivo com overlap (que opera na
            # representação numerica): aqui ele tem precedência.
            self.use_overlap = False
            self.n_patches = seq_len // self.token_len
            self.encoder = None
            self.decoder = (
                MLP(hidden, self.token_len, mlp_hidden_dim,
                    mlp_hidden_layers, dropout, mlp_activation)
                if mlp_hidden_layers > 0
                else nn.Linear(hidden, self.token_len)
            )
            self.revin = None  # valores reais; sem RevIN no fluxo textual
        elif use_overlap:
            self.encoder = OverlappingPatchEmbedding(
                patch_len=self.token_len, stride=self.token_len // 2,
                d_model=hidden, dropout=dropout,
            )
            self.n_patches = self.encoder.num_patches(seq_len)
            self.decoder = None
        elif mlp_hidden_layers > 0:
            self.n_patches = seq_len // self.token_len
            self.encoder = MLP(self.token_len, hidden, mlp_hidden_dim,
                               mlp_hidden_layers, dropout, mlp_activation)
            self.decoder = MLP(hidden, self.token_len, mlp_hidden_dim,
                               mlp_hidden_layers, dropout, mlp_activation)
        else:
            self.n_patches = seq_len // self.token_len
            self.encoder = nn.Linear(self.token_len, hidden)
            self.decoder = nn.Linear(hidden, self.token_len)

        self.gate = AdaptiveFusion(hidden, n_stat_feats=4) if use_vol_gate else None

    def forward(self, x_enc: Optional[Tensor] = None, x_lens: Optional[Tensor] = None) -> Tensor:
        """Pipeline fiel a forecast() do AutoTimes_Gpt2.

        Args:
            x_enc: Sinal bruto (B, seq_len, 1) - janela univariada, OU no
                modo textual: (B, n_patches*block_max) com os tokens GPT-2.
            x_lens: (B, n_patches) comprimento real de cada bloco textual.

        Returns:
            Tensor: Reconstrucao completa (B, seq_len, 1), desnormalizada
            (ou na escala real no modo textual).
        """
        if self.textual:
            return self._forward_textual(x_enc, x_lens)
        b, l, v = x_enc.shape
        x_norm = self.revin.normalize(x_enc)                            # (B,L,1)

        flat = x_norm.permute(0, 2, 1).reshape(b * v, l)                # (N,L)
        if self.use_overlap:
            patches = flat.unfold(-1, self.token_len, self.token_len // 2)  # (N,T,96)
            tokens = self.encoder(flat)                                 # (N,T,D) overlap
        else:
            patches = flat.unfold(-1, self.token_len, self.token_len)   # (N,T,96)
            tokens = self.encoder(patches)                              # (N,T,D)
        if self.gate is not None:
            tokens = self.gate(tokens, compute_token_stats(patches))    # SMETimes

        latent = self.gpt2(
            inputs_embeds=tokens.contiguous(), use_cache=False
        ).last_hidden_state                                             # (N,T,D)

        if self.use_overlap:
            rec = self.encoder.deembed(latent, output_len=self.seq_len)  # (N,L)
        else:
            dec = self.decoder(latent)                                   # (N,T,96)
            rec = dec.reshape(b * v, self.n_patches * self.token_len)    # (N,L)

        rec = rec.view(b, v, self.seq_len).permute(0, 2, 1)              # (B,L,1)
        return self.revin.denormalize(rec)

    def _forward_textual(self, x_ids: Tensor, x_lens: Tensor) -> Tensor:
        """Modo textual: wte(word_ids) -> pooling por bloco -> GPT-2(blocos)
        -> decoder MLP -> reconstrucao (B, seq_len, 1).

        Em vez de alimentar o GPT-2 com os 896 tokens textuais (inviável:
        atencao O(T^2) com T=896), cada bloco de 96h e resumido pelo
        EMBEDDING MEAN ponderado por mask dos seus tokens reais (wte, congelado)
        -> (B, n_blocos, D), e o GPT-2 atende sobre esses 7 vetores de bloco
        (mesmo custo do modo numerico).

        Args:
            x_ids: (B, n_patches*block_max) tokens GPT-2 (ids numericos dos
                textos descritivos).
            x_lens: (B, n_patches) comprimento real (sem pad) de cada bloco.

        Returns:
            Tensor: (B, seq_len, 1) na escala real (sem RevIN).
        """
        b = x_ids.shape[0]
        t = self.n_patches
        block = self.block_max

        emb = self.gpt2.wte(x_ids)                                     # (B, T*block, D)
        emb = emb.view(b, t, block, -1)                   # (B, T, block, D)

        # Mascara dos tokens reais (exclui pads) + pooling mean por bloco.
        # arange: (1, 1, block, 1) ; x_lens[:, :, None, None]: (B, T, 1, 1)
        arange = torch.arange(block, device=emb.device)[None, None, :, None]
        mask = (arange < x_lens[:, :, None, None]).to(emb.dtype)  # (B, T, block, 1)
        lens = x_lens.clamp(min=1).to(emb.dtype).unsqueeze(-1)      # (B, T, 1)
        pooled = (emb * mask).sum(dim=2) / lens                    # (B, T, D)

        # GPT-2 sobre os T = 7 vetores de bloco (um por 96h).
        hidden = self.gpt2(
            inputs_embeds=pooled.contiguous(), use_cache=False
        ).last_hidden_state                              # (B, T, D)

        dec = self.decoder(hidden.to(next(self.decoder.parameters()).dtype))  # (B, T, 96)
        rec = dec.reshape(b, self.n_patches * self.token_len)          # (B, L)
        return rec.unsqueeze(-1)                         # (B, L, 1)


# --------------------------------------------------------------------- #
# Dados: replica do Dataset_ETT_hour / Dataset_Custom (bateria)          #
# --------------------------------------------------------------------- #
def _compute_split_bounds(name: str, n_total: int, seq_len: int, pred_len: int) -> Tuple[List[int], List[int]]:
    """Limites exactos (b1s, b2s) usados pelos loaders numerico e textual."""
    if name.startswith("ett"):
        freq = 4 if name.startswith("ettm") else 1   # ETTm=15min; ETTh=1h
        month = 30 * 24 * freq
        b1s = [0, 12 * month - seq_len, 16 * month - seq_len]
        b2s = [12 * month, 16 * month, 20 * month]
    else:
        num_train = int(n_total * 0.7)
        num_test = int(n_total * 0.2)
        b1s = [0, num_train - seq_len, n_total - num_test - seq_len]
        b2s = [num_train, n_total - num_test, n_total]
    b1s = [max(0, min(b, n_total - seq_len - pred_len)) for b in b1s]
    b2s = [max(0, min(b, n_total)) for b in b2s]
    return b1s, b2s


def _text_csv_path(data_dir: str, data_path: str, suffix: str = "_text") -> Path:
    """Caminho do CSV em texto puro para um dataset numerico (ETTh1_text.csv)."""
    return Path(data_dir) / (Path(data_path).stem + suffix + ".csv")


def build_textoval_split(
    data_dir: str,
    data_path: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    window_stride: int,
    text_step: int = 24,
    force_rebuild: bool = False,
) -> Tuple[int, Path]:
    """Converte UMA vez o dataset numerico em texto puro e salva <data_path>_text.csv.

    Cada linha = um bloco descritivo (por extenso, pt-BR) de uma janela:
      split,feat,start,block,text
    O arquivo fica na mesma pasta dos dados (ex.: dataset/ETT-small/ETTh1_text.csv)
    e, nas execucoes seguintes, eh apenas LIDO (cache), nunca regenerado.

    Args:
        data_dir: Pasta com o CSV numerico (ex: dataset/ETT-small).
        data_path: Arquivo numerico original (ex: ETTh1.csv).
        seq_len: Janela (672).
        label_len: Contexto (576).
        pred_len: Horizonte (96).
        window_stride: 1 = todasa janelas; >1 subamostra.
        text_step: Amostragem dos valores por bloco no texto.
        force_rebuild: Regenera mesmo se o CSV-texto ja existir.

    Returns:
        (n_vars, text_csv_path).
    """
    src = Path(data_dir) / data_path
    if not src.exists():
        raise FileNotFoundError(f"Dataset numerico nao encontrado: {src}")

    out = _text_csv_path(data_dir, data_path)
    if out.exists() and not force_rebuild:
        print(f"  [textual] CSV-texto em cache carregado: {out}")
        # n_vars = qtd colunas de mustras (colunas 1..n no CSV numerico).
        dfn = pd.read_csv(src, nrows=1)
        return int(len(dfn.columns) - 1), out

    print(f"  [textual] gerando CSV-texto (1x): {out.name} ...")
    df_raw = pd.read_csv(src)
    dates = df_raw[df_raw.columns[0]].to_numpy().astype("datetime64[ns]")
    values = df_raw[df_raw.columns[1:]].to_numpy(dtype=np.float32)
    feat_names = list(df_raw.columns[1:])
    n_total, n_vars = values.shape
    b1s, b2s = _compute_split_bounds(data_path.lower(), n_total, seq_len, pred_len)

    rows: List[List] = []
    split_names = ["split0", "split1", "split2"]
    for _si, (b1, b2) in enumerate(zip(b1s, b2s)):
        tot_len = len(values[b1:b2]) - seq_len - pred_len + 1
        for feat in range(n_vars):
            for s_begin in range(0, tot_len, window_stride):
                start = b1 + s_begin
                textos = _textual.gerar_texto_janela(
                    dates, values, feat, start, seq_len, seq_len - label_len,
                    feat_names[feat], text_step,
                )
                for block, texto in enumerate(textos):
                    rows.append([split_names[_si], start, feat, block, texto])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["split", "start", "feat", "block", "text"])
        w.writerows(rows)
    print(f"  [textual] CSV-texto salvo ({len(rows)} linhas): {out}")
    return n_vars, out


def load_real_split(
    data_dir: str,
    data_path: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    window_stride: int,
) -> Tuple[int, TensorDataset, TensorDataset, TensorDataset]:
    """Replica o Dataset_ETT_hour / Dataset_Custom do data_provider.

    Janelas UNIVARIADAS por variavel, na ordem da bateria
    (feat_id = index // tot_len, s_begin = index % tot_len):
        seq_x = data[s_begin:s_begin+seq_len, feat]
        seq_y = data[s_end-label_len:s_end+pred_len, feat]
    com z-score (StandardScaler) ajustado apenas no split de treino.

    Args:
        data_dir: Pasta com o CSV (ex: dataset/ETT-small).
        data_path: Nome do arquivo (ex: ETTh1.csv).
        seq_len: Janela de entrada (672).
        label_len: Contexto (576) -> token_len = seq_len - label_len.
        pred_len: Horizonte (96).
        window_stride: 1 = TODAS as janelas (bateria); >1 subamostra.

    Returns:
        (n_vars, train_ds, val_ds, test_ds) com x:(n, seq_len, 1), y:(n, seq_len, 1).
    """
    csv_path = Path(data_dir) / data_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    values = df_raw[df_raw.columns[1:]].to_numpy(dtype=np.float32)
    n_total, n_vars = values.shape

    b1s, b2s = _compute_split_bounds(data_path.lower(), n_total, seq_len, pred_len)

    scaler = StandardScaler().fit(values[b1s[0]: b2s[0]])
    data = scaler.transform(values).astype(np.float32)

    def _make_split(b1: int, b2: int, stride: int) -> TensorDataset:
        tot_len = len(data[b1:b2]) - seq_len - pred_len + 1
        xs, ys = [], []
        # Ordem da bateria: para cada variavel, todas as janelas temporais.
        for feat in range(n_vars):
            for s_begin in range(0, tot_len, stride):
                s_end = s_begin + seq_len
                r_begin = s_end - label_len
                r_end = r_begin + label_len + pred_len
                xs.append(data[b1 + s_begin: b1 + s_end, feat: feat + 1])
                ys.append(data[b1 + r_begin: b1 + r_end, feat: feat + 1])
        return TensorDataset(
            torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
        )

    train_ds = _make_split(b1s[0], b2s[0], window_stride)
    val_ds = _make_split(b1s[1], b2s[1], window_stride)
    test_ds = _make_split(b1s[2], b2s[2], window_stride)
    return n_vars, train_ds, val_ds, test_ds


def load_textual_split(
    data_dir: str,
    data_path: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    window_stride: int,
    tokenizer,
    text_step: int = 24,
    block_max: int = 128,
    force_rebuild: bool = False,
) -> Tuple[int, TensorDataset, TensorDataset, TensorDataset]:
    """Split textual lendo SEMPRE do CSV-texto em cache (<data_path>_text.csv).

    A conversao numerico -> descricao textual acontece UMA vez
    (build_textoval_split) e fica salva na pasta de dados. Aqui apenas:
        1) garante o CSV-texto (gera 1x se nao existir);
        2) tokeniza os blocos (wte GPT-2) com pad por bloco em block_max;
        3) monta TensorDatasets (x:(n, T*B), x_len:(n, T), y:(n, L, 1) real).

    As targets y sao os valores NUMERICOS reais da janela (label+pred), lidas
    do CSV numerico original - o modelo textual RECONSTROI valores reais.

    O tensor tokenizado ainda e cacheado em TEXT_CACHE_DIR (<chave>.pt) para
    nao repetir a tokenizacao a cada run.

    Args:
        data_dir: Pasta com o CSV (ex: dataset/ETT-small).
        data_path: Nome do arquivo NUMERICO (ex: ETTh1.csv); o texto fica em
            <stem>_text.csv na mesma pasta.
        seq_len: Janela de entrada (672).
        label_len: Contexto (576) -> token_len = seq_len - label_len.
        pred_len: Horizonte (96).
        window_stride: 1 = TODAS as janelas; >1 subamostra.
        tokenizer: AutoTokenizer do GPT-2 (pad = eos/50256).
        text_step: amostra cada N pontos por bloco (deve bater com o CSV).
        block_max: max de tokens por bloco (pad ao final).
        force_rebuild: regenera CSV-texto e tensor cache.

    Returns:
        (n_vars, train_ds, val_ds, test_ds) com x:(n, T*B), x_len:(n, T),
        y:(n, seq_len, 1) - na escala REAL.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    pad = int(tokenizer.pad_token_id)

    src = Path(data_dir) / data_path
    if not src.exists():
        raise FileNotFoundError(f"Dataset numerico nao encontrado: {src}")

    text_csv = _text_csv_path(data_dir, data_path)
    n_vars, text_csv = build_textoval_split(
        data_dir, data_path, seq_len, label_len, pred_len,
        window_stride, text_step, force_rebuild=force_rebuild,
    )
    if not text_csv.exists():
        raise FileNotFoundError(f"CSV-texto nao gerado: {text_csv}")

    cache_key = (f"{data_path}.s{seq_len}.ll{label_len}.pl{pred_len}"
                 f".st{window_stride}.t{text_step}.b{block_max}"
                 f".v{_TEXTUAL_VERSION}.pt")
    cache_file = TEXT_CACHE_DIR / cache_key
    if cache_file.exists() and not force_rebuild:
        _cached = torch.load(cache_file, weights_only=False)
        train_ds = TensorDataset(_cached["train_x"], _cached["train_lens"],
                                 _cached["train_y"])
        val_ds = TensorDataset(_cached["val_x"], _cached["val_lens"],
                               _cached["val_y"])
        test_ds = TensorDataset(_cached["test_x"], _cached["test_lens"],
                                _cached["test_y"])
        print(f"  [textual] tensor cache carregado: {cache_file}")
    else:
        df_raw = pd.read_csv(src)
        values = df_raw[df_raw.columns[1:]].to_numpy(dtype=np.float32)
        n_total, _ = values.shape
        b1s, b2s = _compute_split_bounds(data_path.lower(), n_total, seq_len, pred_len)

        _df = pd.read_csv(text_csv)
        _df["split"] = _df["split"].astype(str)

        def _build_csv(split: str, b1: int, b2: int, stride: int) -> TensorDataset:
            sub = _df[_df["split"] == split].sort_values(
                ["feat", "start", "block"], kind="mergesort"
            )
            xs, ls, ys = [], [], []
            for (feat, start), grp in sub.groupby(["feat", "start"], sort=False):
                textos = grp["text"].tolist()
                ids_blocos: List[int] = []
                lens: List[int] = []
                for texto in textos:
                    ids = tokenizer(texto, add_special_tokens=False,
                                    truncation=True, max_length=block_max)["input_ids"]
                    lens.append(len(ids))
                    ids_blocos.extend(ids[:block_max])
                    ids_blocos.extend([pad] * (block_max - len(ids)))
                # y: valores reais da janela (label+pred) do CSV numerico.
                s_end = start + seq_len
                r_begin = s_end - label_len
                r_end = r_begin + label_len + pred_len
                xs.append(torch.tensor(ids_blocos, dtype=torch.int64))
                ls.append(torch.tensor(lens, dtype=torch.int64))
                ys.append(values[r_begin:r_end, feat:feat + 1].astype(np.float32))
            return TensorDataset(torch.stack(xs), torch.stack(ls),
                                 torch.stack([torch.from_numpy(v) for v in ys]))

        train_ds = _build_csv("split0", b1s[0], b2s[0], window_stride)
        val_ds = _build_csv("split1", b1s[1], b2s[1], window_stride)
        test_ds = _build_csv("split2", b1s[2], b2s[2], window_stride)
        try:
            TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "train_x": train_ds.tensors[0], "train_lens": train_ds.tensors[1],
                    "train_y": train_ds.tensors[2],
                    "val_x": val_ds.tensors[0], "val_lens": val_ds.tensors[1],
                    "val_y": val_ds.tensors[2],
                    "test_x": test_ds.tensors[0], "test_lens": test_ds.tensors[1],
                    "test_y": test_ds.tensors[2],
                },
                cache_file,
            )
            print(f"  [textual] tensor cache salvo: {cache_file}")
        except Exception as _e:  # pragma: no cover - cache é só otimização
            print(f"  [textual] aviso: nao deu para salvar tensor cache ({_e})")
    return n_vars, train_ds, val_ds, test_ds


def build_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoaders no padrao da bateria (shuffle no treino e val, drop_last no treino)."""
    workers = max(num_workers, 4)  # data_factory: workers = max(args.num_workers, 4)
    kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=True, drop_last=False, **kwargs)   # val_set_shuffle=True
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, test_loader


class _QuietProgress:
    """Progresso compacto para consoles nao interativos (sem barra tqdm)."""

    def __init__(self, loader: DataLoader, desc: str) -> None:
        self._it = iter(loader)
        self._total = len(loader)
        self._desc = desc
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self._it)
        except StopIteration:
            print(f"    [{self._desc}] batches concluidos: {self._total}/{self._total}")
            raise
        self._i += 1
        if self._i % 25 == 0 or self._i == self._total:
            print(f"    [{self._desc}] batch {self._i}/{self._total}")
        return item


def _progress(loader: DataLoader, desc: str):
    """Barra de progresso 'Batch i/total' a cada epoca (tqdm ou linhas curtas)."""
    is_tty = bool(sys.stdout) and sys.stdout.isatty()
    if not is_tty:
        return _QuietProgress(loader, desc)
    if _tqdm is not None:
        return _tqdm(loader, total=len(loader), desc=desc, unit="B", ncols=110, leave=False)
    return loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_vol_loss: bool,
    grad_tracker: Optional[Dict] = None,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    desc: str = "treino",
) -> float:
    """Treina UMA epoca completa, iterando por 100% dos batches.

    Forward -> loss -> backward -> step, com AMP (GradScaler) quando ativo,
    exatamente como a bateria (exp_long_term_forecasting.train).

    Args:
        model: Modelo fiel AutoTimes_Gpt2.
        loader: DataLoader de treino.
        optimizer: Adam (bateria) ou AdamW discriminativo.
        criterion: MSE ou VolatilityAwareLogCoshLoss.
        device: Device alvo.
        use_vol_loss: Injeta a feature de variancia na perda.
        grad_tracker: Dict mutavel {"done": bool, "stats": Dict}.
        use_amp: Ativa autocast (mixed precision).
        scaler: GradScaler (criado quando use_amp).
        desc: Rotulo da barra de progresso.

    Returns:
        float: Loss media da epoca (media simples sobre batches).
    """
    model.train()
    running = 0.0
    count = 0
    pbar = _progress(loader, desc)
    for batch in pbar:
        xs, y = batch[:-1], batch[-1]
        x = xs[0].to(device, non_blocking=True)
        sargs = [t.to(device, non_blocking=True) for t in xs[1:]]
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_ctx(device, use_amp):
            pred = model(x, *sargs)
            if use_vol_loss:
                variance = x.var(dim=(1, 2), unbiased=False)        # (B,)
                loss = criterion(pred, y, variance)
            else:
                loss = criterion(pred, y)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf")
        )
        print(
            f"loss={loss.item():.6f} "
            f"grad_norm={total_norm.item():.6f}"
        )
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True
        )
        if grad_tracker is not None and not grad_tracker["done"]:
            grad_tracker["stats"] = check_gradients(model)
            grad_tracker["done"] = True
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        b = x.size(0)
        running += loss.item() * b
        count += b
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{loss.item():.6f}", tl=f"{running / count:.6f}", refresh=False)
    if hasattr(pbar, "close"):
        pbar.close()
    return running / max(count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_vol_loss: bool,
    use_amp: bool = False,
    last_only: bool = False,
    desc: str = "eval",
) -> Tuple[float, float, float]:
    """Avalia loss media + MSE + MAE sobre o loader COMPLETO.

    Args:
        model: Modelo.
        loader: Val ou Test loader.
        criterion: Perda.
        device: Device.
        use_vol_loss: Injeta a feature de variancia na perda.
        use_amp: Ativa autocast (mixed precision).
        last_only: Se True, compara apenas os ultimos pred_len pontos
            (equivalente ao is_test=True da bateria no split de teste).
        desc: Rotulo da barra de progresso.

    Returns:
        (avg_loss, mse, mae).
    """
    model.eval()
    running = 0.0
    count = 0
    preds, trues = [], []
    pbar = _progress(loader, desc)
    for batch in pbar:
        xs, y = batch[:-1], batch[-1]
        x = xs[0].to(device, non_blocking=True)
        sargs = [t.to(device, non_blocking=True) for t in xs[1:]]
        y = y.to(device, non_blocking=True)
        with _amp_ctx(device, use_amp):
            pred = model(x, *sargs)
            if last_only:
                p, t = pred[:, -model.pred_len:, :], y[:, -model.pred_len:, :]
            else:
                p, t = pred, y
            if use_vol_loss:
                variance = x.var(dim=(1, 2), unbiased=False)
                loss = criterion(p, t, variance)
            else:
                loss = criterion(p, t)
        running += loss.item() * x.size(0)
        count += x.size(0)
        preds.append(p.float().cpu().numpy())
        trues.append(t.float().cpu().numpy())
    if hasattr(pbar, "close"):
        pbar.close()
    avg_loss = running / max(count, 1)
    mse, mae = compute_metrics(np.concatenate(preds, 0), np.concatenate(trues, 0))
    return avg_loss, mse, mae


def log_epoch_row(filepath: Path, run: str, epoch: int, ep_time: float, tl: float, vl: float, tll: float) -> None:
    """Append de uma linha tabular do historico de treino."""
    line = f"{run:<34s} | {epoch:4d} | {ep_time:8.3f}s | {tl:10.6f} | {vl:10.6f} | {tll:10.6f}\n"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line)


def log_run_header(filepath: Path, run: str, config: Dict, total_params: int) -> None:
    """Escreve cabecalho da corrida no training_history.log."""
    sep = "=" * 100
    header = (
        f"{sep}\n"
        f"RUN: {run} | {datetime.now().isoformat(timespec='seconds')} | "
        f"config: {json.dumps(config['flags'])} | params_total: {total_params}\n"
        f"{'-' * 100}\n"
    )
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(header)


def update_results_json(filepath: Path, run: str, entry: Dict) -> None:
    """Salva/sobrescreve o dicionario estruturado em benchmark_results.json."""
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {"meta": {}, "runs": {}}
    else:
        data = {"meta": {}, "runs": {}}
    data["meta"] = entry.get("meta", data.get("meta", {}))
    runs_data = {k: v for k, v in entry.items() if k != "meta"}
    data["runs"][run] = runs_data
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


RUN_CONFIGS: List[Dict] = [
    {"name": "Run 0_Padrao", "flags": {"use_robust": False, "use_vol_loss": False, "use_overlap": False, "use_disc_optim": False, "use_vol_gate": False, "use_unfreeze": False, "use_hybrid_loss": False}},
    {"name": "Run 1_RobustRevIN", "flags": {"use_robust": True, "use_vol_loss": False, "use_overlap": False, "use_disc_optim": False, "use_vol_gate": False, "use_unfreeze": False, "use_hybrid_loss": False}},
    {"name": "Run 2_VolatilityLoss", "flags": {"use_robust": False, "use_vol_loss": True, "use_overlap": False, "use_disc_optim": False, "use_vol_gate": False, "use_unfreeze": False, "use_hybrid_loss": False}},
    {"name": "Run 3_OverlappingEmbed", "flags": {"use_robust": False, "use_vol_loss": False, "use_overlap": True, "use_disc_optim": False, "use_vol_gate": False, "use_unfreeze": False, "use_hybrid_loss": False}},
    {"name": "Run 4_DiscrimOptimizer", "flags": {"use_robust": False, "use_vol_loss": False, "use_overlap": False, "use_disc_optim": True, "use_vol_gate": False, "use_unfreeze": False, "use_hybrid_loss": False}},
    {"name": "Run 5_VolatilityGate", "flags": {"use_robust": False, "use_vol_loss": False, "use_overlap": False, "use_disc_optim": False, "use_vol_gate": True, "use_unfreeze": False, "use_hybrid_loss": False}},
    {"name": "Run 6_Frankenstein", "flags": {"use_robust": True, "use_vol_loss": True, "use_overlap": True, "use_disc_optim": True, "use_vol_gate": True, "use_unfreeze": False, "use_hybrid_loss": False}},
]

# --------------------------------------------------------------------- #
# Selecao de runs (--run-filter) e combos customizados (--combos)        #
# --------------------------------------------------------------------- #
MODULE_TOKENS: Dict[str, str] = {
    "robust": "use_robust", "revin": "use_robust",
    "vol": "use_vol_loss", "volatility": "use_vol_loss", "loss": "use_vol_loss",
    "overlap": "use_overlap", "patch": "use_overlap",
    "disc": "use_disc_optim", "optim": "use_disc_optim", "optimizer": "use_disc_optim",
    "gate": "use_vol_gate", "volgate": "use_vol_gate", "smetimes": "use_vol_gate",
    "unfreeze": "use_unfreeze", "finetune": "use_unfreeze", "unfrozen": "use_unfreeze",
    "hybrid": "use_hybrid_loss", "volhuber": "use_hybrid_loss", "vol_l1": "use_hybrid_loss",
    "volrobust": "use_hybrid_loss",
}
ALL_FLAG_KEYS: Tuple[str, ...] = (
    "use_robust", "use_vol_loss", "use_overlap", "use_disc_optim", "use_vol_gate",
    "use_unfreeze", "use_hybrid_loss",
)
TOKEN_LABELS: Dict[str, str] = {
    "use_robust": "RobustRevIN",
    "use_vol_loss": "VolLoss",
    "use_overlap": "Overlap",
    "use_disc_optim": "DiscOptim",
    "use_vol_gate": "SMETimes",
    "use_unfreeze": "Unfreeze",
    "use_hybrid_loss": "HybridLoss",
}
OTIMIZED_TOKENS: Tuple[str, ...] = ("otimizado", "optimizado", "full")


def _flags_from_combo(combo: str) -> Dict[str, bool]:
    """Traduz um combo textual (ex: 'robust+overlap') em dict de flags.

    O token 'otimizado' (alias: full/optimizado) liga TODAS as melhorias
    (robust + vol + overlap + disc + gate + unfreeze + hybrid), usado pelo
    perfil 'Teste Otimizado' do rodar_ablation.ps1.
    """
    flags = {key: False for key in ALL_FLAG_KEYS}
    tokens = [t.strip().lower() for t in combo.split("+") if t.strip()]
    for tok in tokens:
        if tok in OTIMIZED_TOKENS:
            for key in ALL_FLAG_KEYS:
                flags[key] = True
            continue
        if tok not in MODULE_TOKENS:
            raise ValueError(
                f"Token de combo desconhecido: '{tok}'. Validos: "
                f"{sorted(set(MODULE_TOKENS.values()))} ou 'otimizado'."
            )
        flags[MODULE_TOKENS[tok]] = True
    return flags


def expand_run_set(args: argparse.Namespace) -> List[Dict]:
    """Resolve a lista final de runs a partir de --run-filter e --combos.

    - --run-filter: lista separada por virgula com indices ('0,1,4') ou
      substrings do nome ('Overlap,Frankenstein'). 'all'/vazio = as 7 padroes.
      A Run 0 (baseline) e sempre preservada (referencia dos graficos).
    - --combos: lista separada por virgula de combos ('robust+overlap,vol+gate').
      Cada combo gera uma run nova; combos duplicados sao ignorados.

    Args:
        args: Namespace com .run_filter e .combos.

    Returns:
        Lista de dicts {name, flags} na ordem de execucao.
    """
    filter_str = (args.run_filter or "").strip().lower()
    if filter_str and filter_str != "all":
        wanted = [t.strip() for t in filter_str.split(",") if t.strip()]
        selected = []
        for i, cfg in enumerate(RUN_CONFIGS):
            name = cfg["name"].lower()
            if any(tok == str(i) or tok in name for tok in wanted):
                selected.append(cfg)
        if not selected:
            raise ValueError(f"--run-filter '{args.run_filter}' nao casou nenhuma run.")
        if not any(c["name"] == "Run 0_Padrao" for c in selected):
            selected = [RUN_CONFIGS[0]] + selected
    else:
        selected = list(RUN_CONFIGS)

    seen_sigs = {tuple(sorted(c["flags"].items())) for c in RUN_CONFIGS}
    combos = [c.strip() for c in (args.combos or "").split(",") if c.strip()]
    combo_index = len(RUN_CONFIGS)
    for combo in combos:
        flags = _flags_from_combo(combo)
        sig = tuple(sorted(flags.items()))
        if sig in seen_sigs:
            print(f"  [combos] ignorando duplicado: {combo}")
            continue
        if combo.strip().lower() in OTIMIZED_TOKENS:
            label = "TesteOtimizado"
        else:
            label = "+".join(TOKEN_LABELS[k] for k in ALL_FLAG_KEYS if flags[k])
        selected.append({"name": f"Run {combo_index}_{label}", "flags": flags})
        seen_sigs.add(sig)
        combo_index += 1
    return selected


def run_single_experiment(
    cfg: Dict,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    args: argparse.Namespace,
    device: torch.device,
    history_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
) -> Dict:
    """Executa uma corrida completa retornando o dicionario de resultados.

    Protocolo da bateria: treino com early stopping + melhor checkpoint
    (por val_loss), reload do melhor estado e teste final nos ultimos
    pred_len pontos.

    Args:
        cfg: Configuracao do run (name + flags).
        train_ds/val_ds/test_ds: Splits reais (janelas univariadas).
        args: Argumentos CLI.
        device: Device alvo.

    Returns:
        Dict com todas as metricas e curvas para o benchmark_results.json.
    """
    flags = cfg["flags"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if history_path is None:
        history_path = LOG_DIR / f"training_history_{_session_stamp()}.log"
    if results_path is None:
        results_path = LOG_DIR / f"benchmark_results_{_session_stamp()}.json"

    # Mesma semente em todos os runs -> inicializacao justa (mesma topologia).
    seed_everything(SEED)

    # Com descongelamento parcial mantemos o GPT-2 em FP32 (autocast ainda faz
    # a matmul em FP16 sob AMP); com backbone congelado usamos armazenamento
    # FP16 como a bateria (metade da memoria).
    unfreeze_on = flags["use_unfreeze"] and args.unfreeze_layers > 0
    torch_dtype = torch.float32 if unfreeze_on else (
        torch.float16 if args.use_amp else torch.float32
    )

    model = AutoTimesModel(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_vars=args.n_vars,
        label_len=args.label_len,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_hidden_layers=args.mlp_hidden_layers,
        mlp_activation=args.mlp_activation,
        dropout=args.dropout,
        gpt2_dir=Path(args.gpt2_dir),
        torch_dtype=torch_dtype,
        use_robust=flags["use_robust"],
        use_overlap=flags["use_overlap"],
        use_vol_gate=flags["use_vol_gate"],
        unfreeze_layers=args.unfreeze_layers if unfreeze_on else 0,
        textual=args.textual,
        block_max=args.block_max,
    ).to(device)
    verify_device_placement(model, device)
    total_params, trainable_params = count_params(model)
    print(f"  [model] parametros: total={total_params:,} treinaveis={trainable_params:,}")

    if flags["use_disc_optim"]:
        optimizer, groups = build_discriminative_optimizer(
            model, lr_llm=args.lr_llm, lr_new=args.lr_new, weight_decay=args.weight_decay
        )
        scheduler = build_cosine_warm_scheduler(optimizer, t_0=args.t0, t_mult=args.t_mult)
        print(f"  [optim] AdamW discriminativo grupos={groups}")
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_base,
            weight_decay=args.weight_decay,
        )
        # CosineAnnealingLR identico a bateria (run.py --cosine --tmax 10).
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.tmax, eta_min=1e-8
            )
            if args.cosine
            else None
        )

    # AMP/mixed precision (autocast + GradScaler), como a bateria com --use_amp.
    amp_on = bool(args.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    if args.use_amp and not amp_on:
        print("  [amp] device nao-CUDA -> mixed precision desabilitado")
    elif amp_on:
        print("  [amp] mixed precision ATIVO (autocast fp16 + GradScaler, GPT-2 fp16)")

    # Loss: MSELoss (bateria), VolatilityAwareLogCosh (Run 2) ou
    # HybridVolatilityLoss (vol loss + termo robusto L1/Huber - perfil otimizado).
    hybrid_on = flags["use_hybrid_loss"]
    if hybrid_on:
        criterion = HybridVolatilityLoss(
            beta=args.beta, lambda_scale=args.hybrid_lambda,
            robust=args.hybrid_robust, huber_delta=args.hybrid_delta,
        )
        print(f"  [criterion] HybridVolatilityLoss (robust={args.hybrid_robust}, "
              f"lambda={args.hybrid_lambda}, delta={args.hybrid_delta})")
    elif flags["use_vol_loss"]:
        criterion = VolatilityAwareLogCoshLoss(beta=args.beta)
        print(f"  [criterion] VolatilityAwareLogCoshLoss (beta={args.beta})")
    else:
        criterion = nn.MSELoss()

    # Feature de variancia e passada a perda sempre que o criterio precisar.
    # No modo textual nao ha sinal numerico normalizado para a variancia.
    needs_variance = bool((flags["use_vol_loss"] or hybrid_on) and not args.textual)

    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds, args.batch_size, args.num_workers
    )

    history = {"epoch": [], "time_s": [], "train_loss": [], "val_loss": [], "test_loss": [],
               "test_mse": [], "test_mae": []}
    grad_tracker: Dict = {"done": False, "stats": {}}
    run_start = time.perf_counter()

    log_run_header(history_path, cfg["name"], cfg, total_params)

    # Estado do Early Stopping + melhor checkpoint (protocolo da bateria).
    best_val: float = float("inf")
    best_state: Optional[Dict] = None
    best_epoch: Optional[int] = None
    waits: int = 0
    early_stop_epoch: Optional[int] = None

    for epoch in range(1, args.epochs + 1):
        ep_start = time.perf_counter()
        run_tag = f"[{cfg['name']}] ep {epoch}/{args.epochs}"
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            needs_variance, grad_tracker, use_amp=amp_on,
            scaler=scaler if amp_on else None, desc=f"{run_tag} treino",
        )
        val_loss, _, _ = evaluate(model, val_loader, criterion, device,
                                  needs_variance, use_amp=amp_on,
                                  last_only=False, desc=f"{run_tag} val")
        test_loss, test_mse, test_mae = evaluate(model, test_loader, criterion, device,
                                                 needs_variance, use_amp=amp_on,
                                                 last_only=True, desc=f"{run_tag} teste")
        ep_time = time.perf_counter() - ep_start

        history["epoch"].append(epoch)
        history["time_s"].append(round(ep_time, 4))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["test_mse"].append(test_mse)
        history["test_mae"].append(test_mae)

        log_epoch_row(history_path, cfg["name"], epoch, ep_time, train_loss, val_loss, test_loss)

        # Scheduler ao final da epoca: CosineAnnealingLR (--cosine) ou
        # CAWR (discriminativo), ambos aceitam o indice de epoca.
        if scheduler is not None:
            scheduler.step(epoch)

        print(f"    [run {cfg['name']}] epoca {epoch}/{args.epochs} "
              f"train={train_loss:.6f} val={val_loss:.6f} test_mse={test_mse:.6f} "
              f"test_mae={test_mae:.6f} ({ep_time:.1f}s/epoca)")

        # ---- Early Stopping por val_loss (patience=3, como a bateria) ----
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waits = 0
        else:
            waits += 1
            if args.early_stopping and waits >= args.patience:
                early_stop_epoch = epoch
                print(f"    [Early Stopping acionado!] epoca {epoch}: val_loss nao caiu por "
                      f"{args.patience} epocas (melhor val_loss = {best_val:.6f}).")
                break

    grad_stats = grad_tracker["stats"] or check_gradients(model)
    if grad_stats.get("nan") or grad_stats.get("inf"):
        print(f"    [AVISO] gradientes invalidos: {grad_stats}")
    else:
        print(f"    [OK] fluxo de gradientes valido: {grad_stats['trainable']} parametros")

    # Reload do melhor checkpoint + teste final (protocolo da bateria).
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print(f"    [best] restaurado melhor checkpoint (epoch {best_epoch}, val={best_val:.6f})")

    # Persiste o melhor estado em checkpoints/ (nao sobrescreve; nome unico).
    ckpt_path: Optional[Path] = None
    if best_state is not None and not args.no_save_checkpoint:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_slug = cfg["name"].replace(" ", "-").replace("/", "_")
        data_slug = Path(args.data_path).stem
        ckpt_path = CKPT_DIR / f"{run_slug}__{data_slug}__{stamp}.pt"
        save_state = {
            "model_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "flags": flags,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        }
        torch.save(save_state, ckpt_path)
        print(f"    [ckpt] salvo: {ckpt_path}")

    final_loss, final_mse, final_mae = evaluate(
        model, test_loader, criterion, device, needs_variance,
        use_amp=amp_on, last_only=True, desc="[final] teste",
    )

    total_time = time.perf_counter() - run_start
    entry = {
        "meta": {
            "seq_len": args.seq_len,
            "label_len": args.label_len,
            "pred_len": args.pred_len,
            "n_vars": args.n_vars,
            "token_len": model.token_len,
            "d_model": int(model.gpt2.config.n_embd),
            "backbone": f"gpt2-base (transformers, local: {args.gpt2_dir})",
            "epochs": args.epochs,
            "lr_base": args.lr_base,
            "lr_llm": args.lr_llm,
            "lr_new": args.lr_new,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "mlp_hidden_layers": args.mlp_hidden_layers,
            "mlp_activation": args.mlp_activation,
            "unfreeze_layers": args.unfreeze_layers,
            "hybrid_robust": args.hybrid_robust,
            "hybrid_lambda": args.hybrid_lambda,
            "hybrid_delta": args.hybrid_delta,
            "textual": args.textual,
            "text_step": args.text_step,
            "block_max": args.block_max,
            "use_amp": args.use_amp,
            "cosine": args.cosine,
            "tmax": args.tmax,
            "device": str(device),
            "torch": torch.__version__,
            "seed": SEED,
            "data": f"{args.data_dir}/{args.data_path}" if args.data_dir else "dummy",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "name": cfg["name"],
        "flags": flags,
        "epochs": args.epochs,
        "total_time_s": round(total_time, 3),
        "time_per_epoch_s": history["time_s"],
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "test_loss": history["test_loss"],
        "test_mse_per_epoch": history["test_mse"],
        "test_mae_per_epoch": history["test_mae"],
        "final_test_mse": final_mse,
        "final_test_mae": final_mae,
        "final_test_loss": final_loss,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "early_stop": args.early_stopping,
        "patience": args.patience,
        "early_stop_epoch": early_stop_epoch,
        "grad_stats": grad_stats,
    }
    update_results_json(results_path, cfg["name"], entry)
    print(f"  [done] {cfg['name']} | MSE={entry['final_test_mse']:.6f} "
          f"MAE={entry['final_test_mae']:.6f} | {total_time:.1f}s")
    return entry


def parse_args() -> argparse.Namespace:
    """Argumentos da CLI do benchmark (alinhados a rodar_bateria.ps1)."""
    parser = argparse.ArgumentParser(description="Ablation Study AutoTimes (fiel a bateria)")
    parser.add_argument("--epochs", type=int, default=10, help="Epocas por run")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=256,
                        help="Batch size (256 = padrao da bateria)")
    parser.add_argument("--seq-len", type=int, default=672, help="Janela de entrada")
    parser.add_argument("--pred-len", type=int, default=96, help="Horizonte predito")
    parser.add_argument("--label-len", type=int, default=576,
                        help="Contexto; token_len = seq_len - label_len (bateria)")
    parser.add_argument("--n-vars", type=int, default=7, help="Numero de variaveis (features)")
    parser.add_argument("--gpt2-dir", type=str, default=str(GPT2_DIR),
                        help="Diretorio local do GPT-2 base (config.json + model.safetensors)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256, help="Dim oculta do MLP tokenizer")
    parser.add_argument("--mlp-hidden-layers", type=int, default=2, help="Camadas ocultas do MLP (0 = linear)")
    parser.add_argument("--mlp-activation", type=str, default="tanh",
                        help="Ativacao do MLP (tanh/relu/gelu)")
    parser.add_argument("--lr-base", type=float, default=1e-3, help="LR baseline (Adam, bateria 0.001)")
    parser.add_argument("--lr-llm", type=float, default=2e-5, help="LR grupo gpt2/llm")
    parser.add_argument("--lr-new", type=float, default=1e-3, help="LR novos modulos")
    parser.add_argument("--use-amp", "--use_amp", dest="use_amp", action="store_true",
                        help="AMP/mixed precision (autocast + GradScaler + GPT-2 fp16), como a bateria")
    parser.add_argument("--cosine", action="store_true",
                        help="CosineAnnealingLR (T_max=tmax) nas runs nao-discriminativas, como a bateria")
    parser.add_argument("--tmax", type=int, default=10, help="T_max do CosineAnnealingLR")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.5, help="Beta da loss volatil")
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="Num. das ULTIMAS camadas do GPT-2 a descongelar (perfil otimizado; 0 = totalmente congelado)")
    parser.add_argument("--hybrid-robust", type=str, default="huber", choices=["l1", "huber"],
                        help="Termo robusto da loss hibrida (L1 ou Huber)")
    parser.add_argument("--hybrid-lambda", type=float, default=0.5,
                        help="Peso do termo robusto na loss hibrida (0 = so Log-Cosh volatil)")
    parser.add_argument("--hybrid-delta", type=float, default=1.0,
                        help="delta do Huber na loss hibrida (usado quando --hybrid-robust=huber)")
    parser.add_argument("--t0", type=int, default=5, help="T_0 do warm restarts")
    parser.add_argument("--t-mult", type=int, default=2, help="T_mult do warm restarts")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Workers do DataLoader (bateria usa max(num_workers, 4))")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 epoca, MAS com todas as janelas do CSV (sem truncar dados)")
    parser.add_argument("--data-dir", type=str, default="", help="Pasta do dataset real (ex: ../dataset/ETT-small)")
    parser.add_argument("--data-path", type=str, default="ETTh1.csv",
                       help="Arquivo CSV do dataset real no --data-dir")
    parser.add_argument("--stride", "--window-stride", dest="stride", type=int, default=1,
                        help="Passo do janelamento (1 = TODAS as janelas, igual a bateria; >1 subamostra)")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Para o treino se a val_loss nao melhorar por 'patience' epocas")
    parser.add_argument("--patience", type=int, default=3,
                        help="Epocas sem melhora na val_loss antes de acionar o early stopping")
    parser.add_argument("--run-filter", type=str, default="",
                        help="Runs a executar (ex: '0,1,4' ou 'Overlap,Frankenstein'). 'all' = todas")
    parser.add_argument("--combos", type=str, default="",
                        help="Combos customizados (ex: 'robust+overlap,vol+gate'). Tokens: robust,vol,overlap,disc,gate,smetimes,unfreeze,hybrid; 'otimizado' = todos")
    parser.add_argument("--textual", action="store_true",
                        help="Modo TEXTUAL: os blocos de 96h viram textos descritivos (pt-BR) "
                             "embeddados pelo wte do GPT-2, em vez de numeros. Desativa RevIN e "
                             "variancia; y fica na escala real.")
    parser.add_argument("--text-step", type=int, default=24,
                        help="Amostra cada N pontos por bloco no texto descritivo (default 24: "
                             "4 valores/bloco, ~100 tokens). Reduza se truncar (max_length=block-max).")
    parser.add_argument("--text-mode", dest="textual", action="store_true",
                        help="Atalho para --textual (Selecao explicita Texto Puro vs Normal). "
                             "Tambem detecta SOZINHO quando --data-path termina em _text.csv.")
    parser.add_argument("--force-rebuild-text", dest="force_rebuild_text", action="store_true",
                        help="Reconstrói o cache da chave atual (CSV-texto + tensor .pt) do zero")
    parser.add_argument("--block-max", type=int, default=128,
                        help="Max de tokens por bloco textual (pad ao final de cada bloco)")
    parser.add_argument("--sanity", action="store_true",
                        help="Somente VALIDACAO: verifica descongelamento parcial (lr_llm=1e-6), "
                             "loss hibrida VolLoss+L1 e shapes/textos sem treinar; sai sem rodar.")
    parser.add_argument("--no-save-checkpoint", dest="no_save_checkpoint", action="store_true",
                        help="NAO grava o melhor checkpoint por run em checkpoints/ (por padrao salva).")
    args = parser.parse_args()
    return args


def print_summary(results: List[Dict], baseline: Dict) -> None:
    """Imprime tabela comparativa e % relativo ao Run 0."""
    print("\n" + "=" * 84)
    print("RESUMO - Ablation Study (Test MSE / Test MAE) - fiel a bateria")
    print("=" * 84)
    header = f"{'Run':<34s} {'MSE':>12s} {'MAE':>12s} {'MSE d%':>10s} {'Status':>10s}"
    print(header)
    print("-" * 84)
    for r in results:
        d = (r["final_test_mse"] - baseline["final_test_mse"]) / baseline["final_test_mse"] * 100.0
        status = "MELHOR" if d < 0 else ("pior" if d > 0 else "=")
        print(f"{r['name']:<34s} {r['final_test_mse']:>12.6f} {r['final_test_mae']:>12.6f} "
              f"{d:>12.2f}% {status:>10s}")
    print("=" * 84, "\n")


def run_sanity_checks(args: argparse.Namespace, device: torch.device) -> None:
    """Validacao de sanidade (--sanity), SEM treinar.

    Confirma, quando acionados:
      1) Descongelamento parcial das ULTIMAS camadas do GPT-2 (`unfreeze_layers`)
         + lr_llm dedicado no otimizador discriminativo (lr_llm=1e-6 no perfil);
      2) Loss hibrida VolLoss + L1/Huber (HybridVolatilityLoss) aplicada quando
         `use_hybrid_loss`; feature de variancia ligada apenas quando existir;
      3) Formato textual ponta a ponta: CSV-texto + tokenizacao + forward +
         backward com shapes/dtypes corretos (sem erros de tipagem).
    """
    seed_everything(SEED)
    torch.set_float32_matmul_precision("high" if args.use_amp else "highest")
    torch.backends.cudnn.benchmark = args.use_amp
    ok = True

    def _pass(msg: str) -> None:
        print(f"  [SANITY OK] {msg}")

    def _fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"  [SANITY FAIL] {msg}")

    # -- 1) Descongelamento parcial ------------------------------------ #
    if args.unfreeze_layers > 0:
        unfreeze_on = True
        torch_dtype = torch.float32  # unfreeze exige FP32 no backbone (perfil)
        model_ck = AutoTimesModel(
            seq_len=args.seq_len, pred_len=args.pred_len, n_vars=max(args.n_vars, 1),
            label_len=args.label_len, mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden_layers=args.mlp_hidden_layers, mlp_activation=args.mlp_activation,
            dropout=args.dropout, gpt2_dir=Path(args.gpt2_dir),
            torch_dtype=torch_dtype, unfreeze_layers=args.unfreeze_layers,
            textual=args.textual, block_max=args.block_max,
        ).to(device)
        unfrozen = [n for n, p in model_ck.gpt2.named_parameters() if p.requires_grad]
        if not unfrozen:
            _fail("unfreeze_layers>0 mas nenhum parametro do GPT-2 descongelado")
        else:
            n_blk = len(model_ck.gpt2.h)
            start = max(n_blk - args.unfreeze_layers, 0)
            _pass(f"unfreeze: {len(unfrozen)} tensores treinaveis no GPT-2 "
                  f"(ultimas {n_blk - start} camadas + ln_f)")
        opt_ck, _ = build_discriminative_optimizer(
            model_ck, lr_llm=args.lr_llm, lr_new=args.lr_new, weight_decay=args.weight_decay)
        llm_lrs = {round(g["lr"], 10) for g in opt_ck.param_groups if any(
            p.requires_grad for p in g["params"])}
        if not any(abs(lr - args.lr_llm) < 1e-12 for lr in llm_lrs):
            _fail(f"lr_llm={args.lr_llm} nao presente entre as LRs do otimizador ({sorted(llm_lrs)})")
        else:
            _pass(f"lr_llm={args.lr_llm} confirmado no grupo discriminativo do GPT-2")
        del model_ck, opt_ck
        torch.cuda.empty_cache()
    else:
        _pass("unfreeze desligado (unfreeze_layers=0): GPT-2 totalmente congelado")

    # -- 2) Loss hibrida VolLoss + L1/Huber ------------------------------------
    hybrid = HybridVolatilityLoss(
        beta=args.beta, lambda_scale=args.hybrid_lambda,
        robust=args.hybrid_robust, huber_delta=args.hybrid_delta,
    )
    fake_pred = torch.randn(4, 96, 1, device=device)
    fake_true = fake_pred + 0.1
    fake_var = torch.rand(4, device=device)
    v = hybrid(fake_pred, fake_true, fake_var)
    if torch.isnan(v) or torch.isinf(v):
        _fail("HybridVolatilityLoss retornou NaN/Inf mesmo em dados validos")
    else:
        _pass(f"HybridVolatilityLoss (robust={args.hybrid_robust}, "
              f"lambda={args.hybrid_lambda}) forward OK -> {float(v):.4f}")
    plain = VolatilityAwareLogCoshLoss(beta=args.beta)(fake_pred, fake_true, fake_var)
    if torch.isnan(plain) or torch.isinf(plain):
        _fail("VolatilityAwareLogCoshLoss retornou NaN/Inf")
    else:
        _pass(f"VolatilityAwareLogCoshLoss (beta={args.beta}) forward OK -> {float(plain):.4f}")

    # -- 3) Texto puro: CSV-texto + tokenizacao + forward/backward -------------
    if args.textual:
        _tok = AutoTokenizer.from_pretrained(str(Path(args.gpt2_dir)))
        _tok.pad_token_id = _tok.eos_token_id or 50256
        _, tr, va, te = load_textual_split(
            args.data_dir, args.data_path, args.seq_len, args.label_len,
            args.pred_len, args.stride, _tok,
            text_step=args.text_step, block_max=args.block_max,
            force_rebuild=args.force_rebuild_text,
        )
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            _fail("splits textuais vazios")
        else:
            _pass(f"splits textuais prontos: train={len(tr)} val={len(va)} test={len(te)}")
        model_t = AutoTimesModel(
            seq_len=args.seq_len, pred_len=args.pred_len, label_len=args.label_len,
            n_vars=1, mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden_layers=args.mlp_hidden_layers, mlp_activation=args.mlp_activation,
            dropout=args.dropout, gpt2_dir=Path(args.gpt2_dir),
            torch_dtype=torch.float16 if args.use_amp else torch.float32,
            unfreeze_layers=0, textual=True, block_max=args.block_max,
        ).to(device).eval()
        with torch.no_grad():
            xi, xl, y = tr[0]
            xi, xl, y = xi.unsqueeze(0).to(device), xl.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            out = model_t(xi, xl)
        if out.shape != (1, args.seq_len, 1):
            _fail(f"forward textual shape={tuple(out.shape)} != (1, {args.seq_len}, 1)")
        else:
            _pass(f"forward textual shape OK {tuple(out.shape)} e dtype {out.dtype}")
        if not torch.isfinite(out).all():
            _fail("forward textual tensorial com NaN/Inf (tipagem LLM)")
        else:
            _pass("forward textual finito (sem NaN/Inf)")

    print("")
    if ok:
        print("[SANITY] VALIDACAO CONCLUIDA - tudo OK." )
    else:
        print("[SANITY] VALIDACAO CONCLUIDA - PROBLEMAS ENCONTRADAS.")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    args = parse_args()
    device = get_device()
    seed_everything(SEED)

    # Otimizacoes da bateria (AutoTimes_Gpt2.py: TF32 'high' + cudnn benchmark).
    torch.set_float32_matmul_precision("high" if args.use_amp else "highest")
    torch.backends.cudnn.benchmark = args.use_amp

    # Selecao TEXT MODE vs NORMAL: --text-mode/--textual ligam o modo textual;
    # --data-path terminando em _text.csv escolhe o TEXT como origem sozinho.
    if args.data_path.lower().endswith("_text.csv"):
        args.textual = True
        print("[text-mode] --data-path termina em _text.csv -> modo TEXTO PURO ativo")
    # Os loaders sempre partem do CSV NUMERICO (bases/limites/y); o CSV-texto
    # e derivado (ETTh1_text.csv) e gerado/consumido pelo loader textual.
    args.text_csv_src = args.data_path.replace("_text.csv", ".csv")
    args.data_path = args.text_csv_src

    if args.sanity:
        run_sanity_checks(args, device)
        return

    # ------------------------------------------------------------------ #
    # Dados REAIS obrigatorios: o benchmark nunca usa dados sinteticos.   #
    # --quick apenas reduz epocas para 1 (smoke); as janelas continuam    #
    # 100% do CSV, sem fatiar nem truncar o arquivo.                      #
    # ------------------------------------------------------------------ #
    if not args.data_dir:
        raise SystemExit(
            "[ERRO] benchmark real exige --data-dir (ex: ..\\dataset\\ETT-small) "
            "e o CSV via --data-path (default ETTh1.csv)."
        )
    if args.quick:
        args.epochs = 1
        print("[quick] smoke test: 1 epoca, porem TODAS as janelas do CSV "
              "(nenhum truncamento de dados).")

    n_vars, train_ds, val_ds, test_ds = load_real_split(
        args.data_dir, args.data_path, args.seq_len, args.label_len,
        args.pred_len, args.stride,
    )
    args.n_vars = n_vars
    if args.textual:
        _tok = AutoTokenizer.from_pretrained(str(Path(args.gpt2_dir)))
        print(f"[textual] tokenizer GPT-2 OK (vocab={len(_tok):,}, "
              f"pad_token={_tok.pad_token_id})")
        n_vars, train_ds, val_ds, test_ds = load_textual_split(
            args.data_dir, args.data_path, args.seq_len, args.label_len,
            args.pred_len, args.stride, _tok,
            text_step=args.text_step, block_max=args.block_max,
            force_rebuild=args.force_rebuild_text,
        )
        args.n_vars = n_vars
    n_train_batches = max(len(train_ds) // args.batch_size, 1)
    n_val_batches = max(len(val_ds) // args.batch_size, 1)
    n_test_batches = max(len(test_ds) // args.batch_size, 1)
    print(f"[dados] reais: {args.data_path} | train={len(train_ds)} val={len(val_ds)} "
          f"test={len(test_ds)} janelas (univariadas) | vars={n_vars} "
          f"seq={args.seq_len} label={args.label_len} pred={args.pred_len}")
    print(f"[dados] batches/epoca: train={n_train_batches} val={n_val_batches} "
          f"test={n_test_batches} (batch={args.batch_size}, stride={args.stride}, workers={args.num_workers})")

    # Cabecalho global do arquivo de historico (nome unico por sessao).
    hist_path = LOG_DIR / f"training_history_{_session_stamp()}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "a", encoding="utf-8") as f:
        f.write(f"# AutoTimes ablation study iniciado em "
                f"{datetime.now().isoformat(timespec='seconds')} | device={device}\n")
        f.write(f"# {'Run':<34s} | {'Epoca':>5s} | {'Tempo(s)':>8s} | {'TrainLoss':>10s} | "
                f"{'ValLoss':>10s} | {'TestLoss':>10s}\n")

    runs_to_execute = expand_run_set(args)
    print(f"[selecao] {len(runs_to_execute)} runs: " + ", ".join(c["name"] for c in runs_to_execute))
    session_results_path = LOG_DIR / f"benchmark_results_{_session_stamp()}.json"

    results: List[Dict] = []
    for cfg in runs_to_execute:
        print(f"\n=== {cfg['name']} ===")
        try:
            entry = run_single_experiment(
                cfg, train_ds, val_ds, test_ds, args, device,
                history_path=hist_path, results_path=session_results_path,
            )
            results.append(entry)
        except KeyboardInterrupt:
            print(f"  [interrompido] antes de completar {cfg['name']}")
            break
        except Exception as exc:  # nao abortar a bateria por falha isolada
            print(f"  [ERRO em {cfg['name']}]: {exc!r}")
            if args.quick:
                raise
            continue
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if results:
        print_summary(results, results[0])


if __name__ == "__main__":
    main()
