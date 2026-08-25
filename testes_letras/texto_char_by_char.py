"""
texto_char_by_char - Nova hipotese de serializacao (dígito a dígito).

Cada ponto da janela e normalizado pelo max_abs LOCAL da janela (RevIN
local, SEM leakage global), truncado em 6 casas por aritmetica de inteiros
e serializado caractere a caractere em palavras isoladas separadas por
espaco:

    -0.002340 -> "menos zero ponto zero zero dois tres quatro zero"

Entre um ponto e o outro entra a palavra "sep"; no fim do documento, "fim".
Nenhum token especial [EOS]/[SEP]/[UNK] e registrado no tokenizer (o BPE do
GPT-2 fragmenta colchetes em tokens instaveis; palavras comuns fragmentam de
forma estavel e reproduzivel). NaN/Inf sao saneados ANTES do max_abs
(substituidos por 0) e, defensivamente, viram "desconhecido" no texto.

A paginacao e OBRIGATORIA (672 pontos x ate 9 palavras ~= 6.400 tokens nao
cabem em 896 tokens de contexto): reutiliza TestadorPaginacao (estrategia
"janela", overlap 15%, bloco_max=128, n_paginas_max=7) de texto_puro.paginacao.

CONTRATO do loader (TensorDataset de 5 tensores, o max_abs SEMPRE viaja na
amostra - NUNCA em cache de classe, pois o DataLoader embaralha as janelas):

    0. x_numerico (B, 672, 1)  - janela REAL de entrada (sem z-score)
    1. ids        (B, 7*128)   - tokens GPT-2 paginados (pad=50256)
    2. lens       (B, 7)       - comprimento real por pagina (0 = vazio)
    3. y          (B, 672, 1)  - alvo REAL (label+pred, escala original)
    4. max_abs    (B, 1, 1)    - max_abs local de cada janela (para decode)

Substituto drop-in:
    - serie_para_texto_char_by_char(x_window) -> (texto, max_abs)
      no lugar de extenso.serie_para_texto_extenso(...)
    - construir_dataset_char_by_char(pool, ...) -> TensorDataset(5 tensores)
      no lugar de run_etth1_texto.construir_dataset(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .texto_puro.paginacao import TestadorPaginacao

# --------------------------------------------------------------------- #
#  Constantes do contrato (bateria ETTh1)                                #
# --------------------------------------------------------------------- #
SEQ_LEN: int = 672
LABEL_LEN: int = 576
PRED_LEN: int = 96
TOKEN_LEN: int = SEQ_LEN - LABEL_LEN  # 96
BLOCK_MAX: int = 128
N_PAGINAS_MAX: int = 7
OVERLAP: float = 0.15

# Delimitadores como palavras comuns (BPE estavel).
SEP_PALAVRA: str = "sep"
FIM_PALAVRA: str = "fim"
UNK_PALAVRA: str = "desconhecido"

_DIGITOS: np.ndarray = np.array(
    ["zero", "um", "dois", "tres", "quatro", "cinco",
     "seis", "sete", "oito", "nove"],
    dtype=object,
)
_MAX_DECIMAIS: int = 6
_FATOR: int = 10 ** _MAX_DECIMAIS  # 1_000_000


def _sancar_e_normalizar(x_window: np.ndarray) -> Tuple[np.ndarray, float]:
    """Passo 1: sanea NaN/Inf, max_abs LOCAL e normaliza (sem leakage).

    A normalizacao usa o `raw_max` EXATO (sem o +1e-12): se o max_abs
    carregasse o epsilon, o pico da janela viraria ~0.99999999957 e o
    truncamento em 6 casas corromperia o digito mais significativo
    (vira "zero ponto nove nove nove nove nove nove" em vez de
    "um ponto zero zero zero zero zero zero"). O +1e-12 so protege a
    divisao por zero (janela toda zero) e e devolvido no max_abs para o
    contrato/decode (erro ~1e-12 na escala real, irrelevante para o MSE).

    Args:
        x_window: (L,) ou (L, 1) valores reais da janela.

    Returns:
        (x_norm, max_abs): x_norm em [-1, 1] (float64), max_abs = raw + 1e-12.
    """
    x = np.asarray(x_window, dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.where(x == 0.0, 0.0, x)  # corrige -0.0
    raw_max = float(np.max(np.abs(x))) if x.size else 0.0
    max_abs = raw_max + 1e-12
    if raw_max == 0.0:
        return np.zeros_like(x), max_abs
    return x / raw_max, max_abs


def _escalar_para_inteiros(x_norm: np.ndarray) -> np.ndarray:
    """Truncamento em 6 casas por aritmetica de inteiros (deterministico).

    Nunca usa str(float): evita notacao cientifica e artefatos de
    arredondamento. Retorna int64 com o valor truncado para 1e-6
    (ex.: -0.002340 -> -2340).

    Um epsilon (1e-9 na escala de 1e6) e somado NA DIREÇÃO do zero para
    corrigir o erro de representacao binaria do float64: 0.00234 armazena
    0.002339999... e `trunc(0.00234*1e6)` daria 2339 sem o epsilon. O valor
    continua truncation (nao rounding): um 0.002339 real vira 2339.
    """
    scale = x_norm * _FATOR
    direcao = np.where(scale >= 0, 1.0, -1.0)
    scale = scale + direcao * 1e-9
    return np.trunc(scale).astype(np.int64)


def _montar_texto(scaled: np.ndarray) -> str:
    """Passos 2-4: digito -> palavra isolada, "sep" entre pontos, "fim" no final.

    Vetorizado: as palavras sao montadas como matriz (N, 9) via numpy
    (colunas: sinal, parte inteira, "ponto", 6 digitos fracionarios) e o
    texto e unido em UMA passada com " ".join. Sem loops por ponto.
    """
    neg = scaled < 0
    abs_scaled = np.abs(scaled)
    int_part = abs_scaled // _FATOR
    frac = abs_scaled % _FATOR
    frac_dig = np.stack(
        [(frac // (10 ** k)) % 10 for k in range(_MAX_DECIMAIS - 1, -1, -1)],
        axis=1,
    )  # (N, 6)

    n = scaled.shape[0]
    cols: List[np.ndarray] = [
        np.where(neg, "menos", ""),
        _DIGITOS[int_part],
        np.full(n, "ponto", dtype=object),
    ]
    for k in range(_MAX_DECIMAIS):
        cols.append(_DIGITOS[frac_dig[:, k]])
    mat = np.stack(cols, axis=1)  # (N, 9)

    linhas = [" ".join(w for w in linha if w) for linha in mat.tolist()]
    return f" {SEP_PALAVRA} ".join(linhas) + f" {FIM_PALAVRA}"


def texto_char_por_char(x_window: np.ndarray) -> Tuple[str, float]:
    """Passos 1-4 completos, sem paginacao.

    Args:
        x_window: (L,) ou (L, 1) janela real.

    Returns:
        (texto, max_abs): documento char-by-char + max_abs local da janela.
    """
    x_norm, max_abs = _sancar_e_normalizar(x_window)
    scaled = _escalar_para_inteiros(x_norm)
    if not np.all(np.isfinite(scaled)):
        texto = UNK_PALAVRA
    else:
        texto = _montar_texto(scaled)
    return texto, max_abs


def serie_para_texto_char_by_char(x_window: np.ndarray) -> Tuple[str, float]:
    """Drop-in de extenso.serie_para_texto_extenso no dominio char-by-char.

    Diferenca essencial: a normalizacao e POR JANELA (max_abs local), entao
    o documento depende da janela numerica - por isso a tupla (texto,
    max_abs) e retornada em vez de apenas o texto.

    Args:
        x_window: (L,) ou (L, 1) janela real.

    Returns:
        (texto, max_abs).
    """
    return texto_char_por_char(x_window)


def _resolver_tokenizer(tokenizer):
    """Aceita AutoTokenizer pronto ou caminho/pasta do tokenizer local."""
    from transformers import AutoTokenizer

    if not isinstance(tokenizer, (str, Path)):
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
        return tokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 50256
    return tok


class CharByCharPipeline:
    """Pipeline completo: encode (janela->texto+max_abs) e decode (->escala real).

    Args:
        tokenizer: AutoTokenizer ou caminho (usa caminhos.descobrir_tokenizador
            se None).
        block_max: tokens por pagina (padrao 128).
        n_paginas_max: paginas por janela (padrao 7 -> 896 tokens).
        overlap_frac: sobreposicao da estrategia "janela" (padrao 0.15).
    """

    def __init__(
        self,
        tokenizer=None,
        block_max: int = BLOCK_MAX,
        n_paginas_max: int = N_PAGINAS_MAX,
        overlap_frac: float = OVERLAP,
    ) -> None:
        if tokenizer is None:
            tokenizer = caminhos.descobrir_tokenizador()
        self.tokenizer = _resolver_tokenizer(tokenizer)
        self.block_max = int(block_max)
        self.n_paginas_max = int(n_paginas_max)
        self.overlap_frac = float(overlap_frac)
        self.paginador = TestadorPaginacao(
            self.tokenizer,
            block_max=self.block_max,
            overlap_frac=self.overlap_frac,
            n_paginas_max=self.n_paginas_max,
        )

    # ---- encode -------------------------------------------------------- #
    def encode(self, x_window: np.ndarray) -> Tuple[str, float]:
        """Passos 1-4: janela -> (documento char-by-char, max_abs local).

        Args:
            x_window: (L,) ou (L, 1) janela real (sem z-score).

        Returns:
            (texto, max_abs): texto paginavel + max_abs da janela.
        """
        return texto_char_por_char(x_window)

    def paginar(self, x_window: np.ndarray) -> Dict:
        """Encode + paginacao "janela" (overlap 15%) + para_modelo().

        Returns:
            Dict com ids (n_max*block_max), lens (n_max), n_paginas,
            paginas_descartadas e max_abs da janela.
        """
        texto, max_abs = self.encode(x_window)
        r = self.paginador.paginar("janela", texto)
        m = self.paginador.para_modelo(r)
        return {
            "ids": m["ids"],
            "lens": m["lens"],
            "n_paginas": r["n_paginas"],
            "paginas_descartadas": m["paginas_descartadas"],
            "tokens_originais": r.get("tokens_originais", 0),
            "tokens_retidos": r.get("tokens_retidos", 0),
            "max_abs": max_abs,
        }

    # ---- decode -------------------------------------------------------- #
    def decode(self, y_pred_norm, max_abs):
        """Desfaz a normalizacao: y_real = y_pred_norm * max_abs (broadcast).

        Aceita torch.Tensor ou numpy em ambos os argumentos. max_abs pode ser
        (B,), (B, 1, 1) ou escalar; a saida herda o shape de y_pred_norm.

        Nota: no pipeline multimodal atual o modelo ja devolve a
        reconstrucao denormalizada pelo RevIN interno (escala real). Este
        metodo so e necessario se a variante usada prever no espaco
        normalizado por max_abs.
        """
        y = (
            torch.as_tensor(y_pred_norm, dtype=torch.float32)
            if not isinstance(y_pred_norm, torch.Tensor)
            else y_pred_norm
        )
        m = (
            torch.as_tensor(max_abs, dtype=torch.float32)
            if not isinstance(max_abs, torch.Tensor)
            else max_abs
        )
        if m.dim() == 0:
            m = m.reshape(1, 1, 1)
        elif m.dim() == 1:
            m = m.reshape(-1, 1, 1)
        return y * m

    # ---- documentacao -------------------------------------------------- #
    def get_loader_contract(self) -> Dict:
        """Documenta o TensorDataset de 5 tensores esperado pelo loader."""
        return {
            "tensores": [
                ("x_numerico", (None, SEQ_LEN, 1), "float32",
                 "janela REAL de entrada (sem z-score)"),
                ("ids", (None, self.n_paginas_max * self.block_max), "int64",
                 "tokens GPT-2 paginados (pad="
                 f"{int(self.tokenizer.pad_token_id)})"),
                ("lens", (None, self.n_paginas_max), "int64",
                 "comprimento real por pagina (0 = vazio)"),
                ("y", (None, SEQ_LEN, 1), "float32",
                 "alvo REAL (label+pred, escala original)"),
                ("max_abs", (None, 1, 1), "float32",
                 "max_abs local de cada janela (NUNCA cache de classe)"),
            ],
            "paginacao": {
                "estrategia": "janela",
                "block_max": self.block_max,
                "overlap_frac": self.overlap_frac,
                "n_paginas_max": self.n_paginas_max,
                "contexto_total_tokens": self.n_paginas_max * self.block_max,
            },
            "unpack": "x, ids, lens, y, max_abs = batch",
        }


def construir_dataset_char_by_char(
    pool: List[Tuple[np.ndarray, np.ndarray]],
    tokenizer=None,
    block_max: int = BLOCK_MAX,
    n_paginas_max: int = N_PAGINAS_MAX,
    overlap_frac: float = OVERLAP,
    return_stats: bool = False,
):
    """Dataset de 5 tensores, drop-in de run_etth1_texto.construir_dataset.

    Cada janela gera seu PROPRIO documento char-by-char (normalizacao local),
    paginado e acompanhado do max_abs que viaja na amostra.

    Args:
        pool: lista de (xw, yw) com xw/yw (672, 1) na escala REAL (ex:
            saida de run_etth1_texto.pool_janelas_etth1).
        tokenizer: AutoTokenizer ou caminho (default: descoberta local).
        block_max/n_paginas_max/overlap_frac: parametros de paginacao.
        return_stats: se True, retorna (ds, stats) com as estatisticas
            agregadas de paginacao e normalizacao do pool.

    Returns:
        TensorDataset(x_numerico, ids, lens, y, max_abs) ou
        (TensorDataset, stats).
    """
    pipeline = CharByCharPipeline(
        tokenizer, block_max=block_max,
        n_paginas_max=n_paginas_max, overlap_frac=overlap_frac,
    )
    xs, ys = [], []
    ids_linhas, lens_linhas, max_abs_linhas = [], [], []
    stats = {
        "tokens_originais": [], "tokens_retidos": [], "paginas": [],
        "paginas_descartadas": [], "max_abs": [],
    }
    for xw, yw in pool:
        res = pipeline.paginar(xw)
        xs.append(np.asarray(xw, dtype=np.float32))
        ys.append(np.asarray(yw, dtype=np.float32))
        ids_linhas.append(res["ids"])
        lens_linhas.append(res["lens"])
        max_abs_linhas.append(np.asarray(res["max_abs"], dtype=np.float32))
        stats["tokens_originais"].append(res["tokens_originais"])
        stats["tokens_retidos"].append(res["tokens_retidos"])
        stats["paginas"].append(res["n_paginas"])
        stats["paginas_descartadas"].append(res["paginas_descartadas"])
        stats["max_abs"].append(res["max_abs"])
    ds = TensorDataset(
        torch.from_numpy(np.stack(xs)),
        torch.tensor(np.stack(ids_linhas), dtype=torch.int64),
        torch.tensor(np.stack(lens_linhas), dtype=torch.int64),
        torch.from_numpy(np.stack(ys)),
        torch.from_numpy(np.stack(max_abs_linhas).reshape(-1, 1, 1)),
    )
    if return_stats:
        agg = {
            k: (float(np.mean(v)) if k != "max_abs" else float(np.mean(v)))
            for k, v in stats.items()
        }
        agg["max_abs_media"] = float(np.mean(stats["max_abs"]))
        agg["max_abs_desvio"] = float(np.std(stats["max_abs"]))
        agg["tokens_originais_media"] = float(np.mean(stats["tokens_originais"]))
        agg["tokens_retidos_media"] = float(np.mean(stats["tokens_retidos"]))
        agg["paginas_media"] = float(np.mean(stats["paginas"]))
        agg["paginas_descartadas_media"] = float(np.mean(stats["paginas_descartadas"]))
        return ds, agg
    return ds