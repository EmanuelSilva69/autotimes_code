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

from joblib import Parallel, delayed

from . import caminhos
from .texto_puro.formato import (
    FORMATO_PADRAO,
    FormatoTexto,
    definir_formato,
    formato_atual,
    resolver_formato,
)
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


def _decimais_de(casas: Optional[int] = None) -> int:
    """Casas decimais a escrever: explicitas ou as do formato atual (padrao 6)."""
    if casas is None:
        return int(formato_atual().casas)
    return int(casas)


def _escalar_para_inteiros(x_norm: np.ndarray,
                           casas: Optional[int] = None,
                           arredondamento: Optional[str] = None) -> np.ndarray:
    """Truncamento/arredondamento em `casas` casas por aritmetica de inteiros.

    Nunca usa str(float): evita notacao cientifica e artefatos de
    arredondamento binario. Retorna int64 com o valor na escala 10**casas
    (ex.: -0.002340 com 6 casas -> -2340; com 3 casas -> -23).

    Um epsilon (1e-9 na escala de 10**casas) e somado NA DIRECAO do zero
    para corrigir o erro de representacao binaria do float64: 0.00234
    armazena 0.002339999... e `trunc(0.00234*1e6)` daria 2339 sem o
    epsilon.

    Modos (FormatoTexto.arredondamento, padrao 'truncar' = comportamento
    historico):

        truncar    -> corta a cauda (vies de ate 10**-casas no sentido do
                      zero; historico do projeto)
        arredondar -> metade para longe do zero, TAMBEM por aritmetica de
                      inteiros (np.trunc(x + 0.5*sinal)), sem np.round
                      (que usa regra par-e-par e nao e o que queremos aqui)

    Args:
        x_norm: valores normalizados em [-1, 1] (float64).
        casas: casas decimais (None = formato atual, padrao 6).
        arredondamento: 'truncar'|'arredondar' (None = formato atual).

    Returns:
        int64 na escala de 10**casas.
    """
    if arredondamento is None:
        arredondamento = formato_atual().arredondamento
    fator = 10 ** _decimais_de(casas)
    scale = x_norm * fator
    direcao = np.where(scale >= 0, 1.0, -1.0)
    scale = scale + direcao * 1e-9
    if arredondamento == "arredondar":
        return np.trunc(scale + direcao * 0.5).astype(np.int64)
    return np.trunc(scale).astype(np.int64)


def _montar_texto(scaled: np.ndarray,
                  casas: Optional[int] = None,
                  encoding: Optional[str] = None,
                  zeros: Optional[str] = None) -> str:
    """Passos 2-4: digito -> palavra isolada, "sep" entre pontos, "fim" no final.

    Vetorizado: as palavras sao montadas como matriz (N, 3+casas) via numpy
    (colunas: sinal, parte inteira, "ponto", `casas` digitos fracionarios) e o
    texto e unido em UMA passada com " ".join. Sem loops por ponto.

    Politica de zeros (FormatoTexto.zeros):

        manter  -> `casas` casas SEMPRE escritas (padrao = texto atual;
                   -0.000000 vira "menos zero ponto zero zero zero zero
                   zero zero")
        remover -> corta zeros a direita da parte fracionaria (deterministico;
                   -0.000000 vira "menos zero"; -0.500000 vira "menos zero
                   ponto cinco"). O VALOR escrito e o mesmo; economia de
                   1..(casas+1) tokens por ponto "redondo".

    Args:
        scaled: inteiros de _escalar_para_inteiros (escala 10**casas).
        casas: casas decimais (None = formato atual, padrao 6).
        encoding: 'extenso' (palavras, padrao) ou 'digitos' (algarismos
            crus, ex.: -12.500000) - so muda a ESCRITA; normalizacao local,
            max_abs e alvo continuam iguais.
        zeros: 'manter'|'remover' (None = formato atual).

    Returns:
        Documento char-by-char sem notacao cientifica; no padrao, com
        `casas` casas sempre escritas (truncamento).
    """
    casas = _decimais_de(casas)
    if encoding is None:
        encoding = formato_atual().encoding
    if zeros is None:
        zeros = formato_atual().zeros
    fator = 10 ** casas

    neg = scaled < 0
    abs_scaled = np.abs(scaled)
    int_part = abs_scaled // fator
    frac = abs_scaled % fator
    frac_dig = np.stack(
        [(frac // (10 ** k)) % 10 for k in range(casas - 1, -1, -1)],
        axis=1,
    )  # (N, casas)

    n = scaled.shape[0]

    if encoding == "digitos":
        # Algarismos crus: "menos"/"ponto" viram '-' e '.' (ex.: -12.500000).
        if zeros == "remover":
            linhas = [
                ("-" if eh_neg else "") + str(int(inteiro))
                + (("." + "".join(map(str, digs)).rstrip("0")) if any(digs)
                   else "")
                for inteiro, digs, eh_neg in zip(
                    int_part.tolist(), frac_dig.tolist(), neg.tolist()
                )
            ]
        else:
            linhas = [
                ("-" if eh_neg else "") + str(int(inteiro)) + "." + "".join(
                    map(str, digs)
                )
                for inteiro, digs, eh_neg in zip(
                    int_part.tolist(), frac_dig.tolist(), neg.tolist()
                )
            ]
        return f" {SEP_PALAVRA} ".join(linhas) + f" {FIM_PALAVRA}"

    if zeros == "remover":
        # Por linha: corta os "zero" finais; se sobrar fracao vazia, some
        # ate o "ponto" (mesma regra do encoding digitos).
        linhas = []
        for inteiro, digs, eh_neg in zip(int_part.tolist(), frac_dig.tolist(),
                                         neg.tolist()):
            frac_palavras = [_DIGITOS[d] for d in digs]
            while frac_palavras and frac_palavras[-1] == "zero":
                frac_palavras.pop()
            palavras: List[str] = []
            if eh_neg:
                palavras.append("menos")
            palavras.append(_DIGITOS[int(inteiro)])
            if frac_palavras:
                palavras.append("ponto")
                palavras.extend(frac_palavras)
            linhas.append(" ".join(palavras))
        return f" {SEP_PALAVRA} ".join(linhas) + f" {FIM_PALAVRA}"

    cols: List[np.ndarray] = [
        np.where(neg, "menos", ""),
        _DIGITOS[int_part],
        np.full(n, "ponto", dtype=object),
    ]
    for k in range(casas):
        cols.append(_DIGITOS[frac_dig[:, k]])
    mat = np.stack(cols, axis=1)  # (N, 3+casas)

    linhas = [" ".join(w for w in linha if w) for linha in mat.tolist()]
    return f" {SEP_PALAVRA} ".join(linhas) + f" {FIM_PALAVRA}"


def texto_char_por_char(x_window: np.ndarray,
                        formato: Optional[FormatoTexto] = None) -> Tuple[str, float]:
    """Passos 1-4 completos, sem paginacao.

    Args:
        x_window: (L,) ou (L, 1) janela real.
        formato: FormatoTexto com as variantes de escrita (casas,
            encoding). None = formato atual do processo (padrao: 6 casas
            por extenso = texto historico, byte a byte).

    Returns:
        (texto, max_abs): documento char-by-char + max_abs local da janela.
    """
    if formato is not None:
        definir_formato(formato)
    fmt = formato_atual()
    x_norm, max_abs = _sancar_e_normalizar(x_window)
    scaled = _escalar_para_inteiros(x_norm, casas=fmt.casas,
                                    arredondamento=fmt.arredondamento)
    if not np.all(np.isfinite(scaled)):
        texto = UNK_PALAVRA
    else:
        texto = _montar_texto(scaled, casas=fmt.casas, encoding=fmt.encoding,
                              zeros=fmt.zeros)
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


# --------------------------------------------------------------------- #
#  Decomposicao TEXTUAL (orientacao: 2 frases por janela)                #
# --------------------------------------------------------------------- #
def media_movel_janela(x_window: np.ndarray, kernel: int = 25) -> np.ndarray:
    """Tendencia por MA convolucional com padding replicado nas bordas.

    Espelho numpy EXATO de _MediaMovel (modelos_dual.py): mesma semantica
    de janela deslizante, media sobre `kernel` pontos, bordas replicando
    o primeiro/ultimo valor. Vetorizada via cumsum (O(L)).

    Args:
        x_window: (L,) ou (L, 1) valores REAIS da janela.
        kernel: tamanho da janela da media movel (impar).

    Returns:
        trend: (L,) tendencia local multi-escala.
    """
    x = np.asarray(x_window, dtype=np.float64).ravel()
    pad = (kernel - 1) // 2
    xp = np.concatenate([
        np.full(pad, x[0], dtype=np.float64), x,
        np.full(pad, x[-1], dtype=np.float64),
    ])
    c = np.cumsum(np.concatenate([[0.0], xp]))
    return (c[kernel:] - c[:-kernel]) / kernel


def frases_texto_decomposto(
    x_window: np.ndarray, kernel: int = 25,
) -> Tuple[str, str, float]:
    """Decomposicao da janela em DUAS frases char-by-char independentes.

    Pipeline (orientacao Mauricio):
        1. trend = MA-k25(janela); osc = janela - trend;
        2. cada componente vira sua PROPIA frase char-by-char (max_abs
           LOCAL da componente, mesmo contrato de sempre);
        3. cabecalhos "Tendencia:" / "Residuos:" prefixam as frases;
        4. o alvo y continua sendo a serie ORIGINAL nao-decomposta -
           quem aprende a recombinar sao o LLM congelado + fusion.

    Args:
        x_window: (L,) ou (L, 1) janela real.
        kernel: tamanho da media movel.

    Returns:
        (frase_trend, frase_osc, max_abs_original): as duas frases (ainda
        SEM paginar) e o max_abs da janela ORIGINAL (contrato de decode).
    """
    x = np.asarray(x_window, dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.where(x == 0.0, 0.0, x)
    trend = media_movel_janela(x, kernel)
    osc = x - trend
    frase_t, _ = texto_char_por_char(trend)
    frase_o, _ = texto_char_por_char(osc)
    _, max_abs_orig = _sancar_e_normalizar(x)
    return f"Tendencia: {frase_t}", f"Residuos: {frase_o}", max_abs_orig


# --------------------------------------------------------------------- #
#  Passo 2 OFICIAL: STL real (LOESS) sobre a janela de contexto          #
# --------------------------------------------------------------------- #
MARCADOR_TENDENCIA: str = "Tendencia:"
MARCADOR_SAISONALIDADE: str = "Sazonalidade:"
MARCADOR_RESIDUOS: str = "Residuos:"
STL_PERIODO_PADRAO: int = 24  # ETTh1 = amostragem horaria -> ciclo diario


# --------------------------------------------------------------------- #
#  Deteccao dinamica de periodos MSTL a partir da coluna de datas         #
# --------------------------------------------------------------------- #
def detectar_periodos_mstl(csv_path: str) -> Tuple[int, int, dict]:
    """Detecta periodos MSTL a partir da coluna de datas do CSV.

    Le a diferenca temporal entre as duas primeiras linhas da coluna
    de datas e calcula os periodos diario e semanal em numero de passos.

    Args:
        csv_path: caminho para o CSV (ex: dataset/ETT-small/ETTh1.csv).

    Returns:
        (periodo_diario, periodo_semanal, info_dict) onde info_dict contem
        metadados para logging (minutos_do_timestep, passos_por_hora, etc).
    """
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=5)
    dates = pd.to_datetime(df.iloc[:, 0])
    diff = dates.iloc[1] - dates.iloc[0]
    minutos = diff.total_seconds() / 60.0
    passos_por_hora = 60.0 / minutos
    periodo_diario = int(passos_por_hora * 24)
    periodo_semanal = int(periodo_diario * 7)
    info = {
        "minutos_timestep": minutos,
        "passos_por_hora": passos_por_hora,
        "periodo_diario": periodo_diario,
        "periodo_semanal": periodo_semanal,
        "periodos_tupla": (periodo_diario, periodo_semanal),
    }
    return periodo_diario, periodo_semanal, info


def decompor_stl(
    x_window: np.ndarray,
    periodo: int = STL_PERIODO_PADRAO,
    periodos: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decomposicao STL/MSTL (Tendencia, Sazonalidade, Residuo) na janela.

    Passo 2 do fluxo oficial. O STL/MSTL (LOESS, statsmodels) e fitado
    EXCLUSIVAMENTE com os pontos da janela de contexto que termina ANTES
    do horizonte de previsao: os 96 passos futuros (y) nunca entram no
    fit -> sem leakage.

    Se `periodos` for fornecido (tupla de 2 inteiros), usa MSTL com
    multiplos periodos (diario + semanal). Caso contrario, usa STL
    classico com periodo unico.

    Args:
        x_window: (L,) ou (L, 1) janela REAL de contexto.
        periodo: sazonalidade dominante (24 para ETTh1 horario) - usado
            apenas quando periodos=None (STL classico).
        periodos: tupla (periodo_diario, periodo_semanal) para MSTL.
            Se fornecido, MSTL e usado em vez de STL.

    Returns:
        (tendencia, sazonalidade, residuo): tres arrays (L,) float64.
        No caso MSTL, sazonalidade = soma de todas as sazonalidades.
    """
    x = np.asarray(x_window, dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.where(x == 0.0, 0.0, x)
    L = x.size

    if periodos is not None:
        from statsmodels.tsa.seasonal import MSTL

        periodos_validos = [
            int(min(max(p, 2), max(L // 2, 2)))
            for p in periodos
        ]
        # Remove periodos duplicados e mantem ordem
        vistos = set()
        periodos_filtrados = []
        for p in periodos_validos:
            if p not in vistos and p < L:
                vistos.add(p)
                periodos_filtrados.append(p)
        if not periodos_filtrados:
            periodos_filtrados = [2]
        res = MSTL(x, periods=periodos_filtrados, stl_kwargs={"robust": True}).fit()
        # MSTL retorna seasonal como DataFrame/array com coluna por periodo
        seasonal_arr = np.asarray(res.seasonal, dtype=np.float64)
        if seasonal_arr.ndim == 2:
            sazonal = seasonal_arr.sum(axis=1)
        else:
            sazonal = seasonal_arr
    else:
        from statsmodels.tsa.seasonal import STL

        periodo = int(min(max(periodo, 2), max(L // 2, 2)))
        res = STL(x, period=periodo, robust=True).fit()
        sazonal = np.asarray(res.seasonal, dtype=np.float64)

    return (
        np.asarray(res.trend, dtype=np.float64),
        sazonal,
        np.asarray(res.resid, dtype=np.float64),
    )


def frases_texto_stl(
    x_window: np.ndarray,
    periodo: int = STL_PERIODO_PADRAO,
    periodos: Optional[Tuple[int, int]] = None,
) -> Tuple[str, str, str, Tuple[float, float, float]]:
    """Passos 2-4 OFICIAIS com as TRES componentes (T/S/R).

    Cada componente e normalizada pelo PROPRIO max_abs (passo 3, valor
    GUARDADO na tupla final), truncada em 6 casas e vira uma frase
    char-by-char independente (passo 4), prefixada por um marcador
    estavel para o BPE do GPT-2 ("Tendencia:"/"Sazonalidade:"/"Residuos:"

    Args:
        x_window: (L,) ou (L, 1) janela REAL de contexto.
        periodo: periodo sazonal do STL (default 24).
        periodos: tupla (periodo_diario, periodo_semanal) para MSTL.
            Se fornecido, MSTL e usado em vez de STL.

    Returns:
        (frase_t, frase_s, frase_r, max_abss): as tres frases (ainda SEM
        paginar) e o max_abs DE CADA componente (contrato do passo 3).
    """
    trend, sazonal, residuo = decompor_stl(x_window, periodo, periodos=periodos)
    frase_t, m_t = texto_char_por_char(trend)
    frase_s, m_s = texto_char_por_char(sazonal)
    frase_r, m_r = texto_char_por_char(residuo)
    return (
        f"{MARCADOR_TENDENCIA} {frase_t}",
        f"{MARCADOR_SAISONALIDADE} {frase_s}",
        f"{MARCADOR_RESIDUOS} {frase_r}",
        (float(m_t), float(m_s), float(m_r)),
    )


# --------------------------------------------------------------------- #
#  CEEMDAN: decomposicao adaptativa sem periodos fixos                   #
# --------------------------------------------------------------------- #

def decompor_ceemdan(
    x_window: np.ndarray,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CEEMDAN puro -> (tendencia, sazonalidade, residuos).

    CEEMDAN extrai IMFs em ordem de maior -> menor frequencia:
      - cIMFs[0] = maior frequencia (ruido) -> Residuo
      - cIMFs[1:-1] = medias frequencias (oscilacoes) -> Sazonalidade
      - cIMFs[-1] = menor frequencia (trend lento) -> Tendencia

    Interface identica a decompor_stl() para uso intercambiavel.
    """
    from .ceemdan.decomposicao import decompor_ceemdan as _ceemdan
    return _ceemdan(x_window, trials=trials, epsilon=epsilon,
                     max_imf=max_imf, seed=seed)


def decompor_hibrido_mstl_ceemdan(
    x_window: np.ndarray,
    periodos: Tuple[int, int],
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MSTL para T/S + CEEMDAN refina o residuo R."""
    from .ceemdan.decomposicao import decompor_hibrido_mstl_ceemdan as _hib
    return _hib(x_window, periodos, trials=trials, epsilon=epsilon,
                max_imf=max_imf, seed=seed)


def decompor_paralelo(
    x_window: np.ndarray,
    periodo: int = 24,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MSTL(1 periodo) para tendencia + CEEMDAN para sazonalidade."""
    from .ceemdan.decomposicao import decompor_paralelo as _par
    return _par(x_window, periodo=periodo, trials=trials,
                 epsilon=epsilon, max_imf=max_imf, seed=seed)


def decompor_vmd(
    x_window: np.ndarray,
    k: int = 7,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    init: int = 0,
    trend_ma: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VMD puro -> (tendencia, sazonalidade, residuos).

    VMD extrai K modos e os ordena por frequencia central crescente:
      - modes[0] = menor frequencia (trend lento)   -> Tendencia (MA ~L/7)
      - modes[1:-1] = medias frequencias            -> Sazonalidade
      - residuo = x - T - S (inclui erro de fidelidade; T+S+R = x)

    Interface identica a decompor_stl()/decompor_ceemdan() para uso
    intercambiavel. Custo ~dezenas de ms (FFT, deterministico) contra
    ~2200 ms do CEEMDAN trials=100 em CPU.

    init: 0 = omegas em 0 (default, melhor separacao T/S) | 1 = uniformes.
    trend_ma: None = MA automatico ~L/7 | 0 = sem MA | N = kernel fixo.
    """
    from .vmd.decomposicao import decompor_vmd as _vmd
    return _vmd(x_window, k=k, alpha=alpha, tau=tau, tol=tol, init=init,
                trend_ma=trend_ma)


def frases_texto_ceemdan(
    x_window: np.ndarray,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[str, str, str, Tuple[float, float, float]]:
    """Decompoe x_window via CEEMDAN e retorna frases char-by-char.

    Misma interface que frases_texto_stl().
    """
    trend, sazonal, residuo = decompor_ceemdan(x_window, trials=trials,
                                                epsilon=epsilon,
                                                max_imf=max_imf, seed=seed)
    frase_t, m_t = texto_char_por_char(trend)
    frase_s, m_s = texto_char_por_char(sazonal)
    frase_r, m_r = texto_char_por_char(residuo)
    return (
        f"{MARCADOR_TENDENCIA} {frase_t}",
        f"{MARCADOR_SAISONALIDADE} {frase_s}",
        f"{MARCADOR_RESIDUOS} {frase_r}",
        (float(m_t), float(m_s), float(m_r)),
    )


def frases_texto_hibrido(
    x_window: np.ndarray,
    periodos: Tuple[int, int],
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[str, str, str, Tuple[float, float, float]]:
    """Decompoe via MSTL+CEEMDAN hibrido e retorna frases char-by-char."""
    trend, sazonal, residuo = decompor_hibrido_mstl_ceemdan(
        x_window, periodos, trials=trials, epsilon=epsilon,
        max_imf=max_imf, seed=seed,
    )
    frase_t, m_t = texto_char_por_char(trend)
    frase_s, m_s = texto_char_por_char(sazonal)
    frase_r, m_r = texto_char_por_char(residuo)
    return (
        f"{MARCADOR_TENDENCIA} {frase_t}",
        f"{MARCADOR_SAISONALIDADE} {frase_s}",
        f"{MARCADOR_RESIDUOS} {frase_r}",
        (float(m_t), float(m_s), float(m_r)),
    )


def frases_texto_paralelo(
    x_window: np.ndarray,
    periodo: int = 24,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[str, str, str, Tuple[float, float, float]]:
    """Decompoe via MSTL(paralelo) e retorna frases char-by-char."""
    trend, sazonal, residuo = decompor_paralelo(
        x_window, periodo=periodo, trials=trials, epsilon=epsilon,
        max_imf=max_imf, seed=seed,
    )
    frase_t, m_t = texto_char_por_char(trend)
    frase_s, m_s = texto_char_por_char(sazonal)
    frase_r, m_r = texto_char_por_char(residuo)
    return (
        f"{MARCADOR_TENDENCIA} {frase_t}",
        f"{MARCADOR_SAISONALIDADE} {frase_s}",
        f"{MARCADOR_RESIDUOS} {frase_r}",
        (float(m_t), float(m_s), float(m_r)),
    )


def frases_texto_vmd(
    x_window: np.ndarray,
    k: int = 7,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    init: int = 0,
    trend_ma: Optional[int] = None,
) -> Tuple[str, str, str, Tuple[float, float, float]]:
    """Decompoe x_window via VMD e retorna frases char-by-char.

    Misma interface que frases_texto_ceemdan().
    """
    trend, sazonal, residuo = decompor_vmd(x_window, k=k, alpha=alpha,
                                            tau=tau, tol=tol, init=init,
                                            trend_ma=trend_ma)
    frase_t, m_t = texto_char_por_char(trend)
    frase_s, m_s = texto_char_por_char(sazonal)
    frase_r, m_r = texto_char_por_char(residuo)
    return (
        f"{MARCADOR_TENDENCIA} {frase_t}",
        f"{MARCADOR_SAISONALIDADE} {frase_s}",
        f"{MARCADOR_RESIDUOS} {frase_r}",
        (float(m_t), float(m_s), float(m_r)),
    )


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

    # ---- encode/paginar DECOMPOSTO (2 frases) -------------------------- #
    def encode_decomposto(
        self, x_window: np.ndarray, kernel_ma: int = 25,
    ) -> Tuple[str, str, float]:
        """Janela -> (frase_trend, frase_osc, max_abs_original)."""
        return frases_texto_decomposto(x_window, kernel=kernel_ma)

    def paginar_decomposto(
        self, x_window: np.ndarray, kernel_ma: int = 25,
        divisao: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Pagina as DUAS frases dividindo o orcamento de contexto.

        Cada frase recebe um bloco fixo de paginas do total n_paginas_max
        (default: metade superior para a tendencia, inferior para o
        residuo - ex. 7 paginas -> 4 + 3). As frases sao paginadas na
        estrategia "janela" e apenas o PREFIXO de paginas atribuido a cada
        uma entra no documento final; assim tendência E resíduo aparecem
        sempre, nenhum deles e descartado por inteiro.

        Returns:
            Mesmo contrato de paginar(): ids (896,), lens (7,), etc.
        """
        b = self.block_max
        if divisao is None:
            k_t = self.n_paginas_max // 2 + self.n_paginas_max % 2
            k_r = self.n_paginas_max // 2
        else:
            k_t, k_r = divisao
            if k_t + k_r != self.n_paginas_max:
                raise ValueError(
                    f"divisao {k_t}+{k_r} != n_paginas_max "
                    f"{self.n_paginas_max}"
                )
        frase_t, frase_o, max_abs_orig = self.encode_decomposto(
            x_window, kernel_ma=kernel_ma,
        )
        r_t = self.paginador.paginar("janela", frase_t)
        m_t = self.paginador.para_modelo(r_t)
        r_o = self.paginador.paginar("janela", frase_o)
        m_o = self.paginador.para_modelo(r_o)

        ids = np.concatenate([m_t["ids"][:k_t * b], m_o["ids"][:k_r * b]])
        lens = np.concatenate([m_t["lens"][:k_t], m_o["lens"][:k_r]])
        return {
            "ids": ids,
            "lens": lens,
            "n_paginas": int((lens > 0).sum()),
            "paginas_descartadas": (
                int(m_t["paginas_descartadas"]) + max(r_t["n_paginas"] - k_t, 0)
                + int(m_o["paginas_descartadas"]) + max(r_o["n_paginas"] - k_r, 0)
            ),
            "tokens_originais": (
                r_t.get("tokens_originais", 0) + r_o.get("tokens_originais", 0)
            ),
            "tokens_retidos": int(lens.sum()),
            "max_abs": max_abs_orig,
        }

    # ---- encode/paginar STL OFICIAL (3 frases T/S/R) ------------------- #
    def encode_stl(
        self,
        x_window: np.ndarray,
        periodo: int = STL_PERIODO_PADRAO,
        periodos: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, str, str, Tuple[float, float, float]]:
        """Janela -> (frase_tendencia, frase_sazonalidade, frase_residuo,
        max_abss) - fluxo oficial dos passos 2-4."""
        return frases_texto_stl(x_window, periodo, periodos=periodos)

    def paginar_stl(
        self,
        x_window: np.ndarray,
        periodo: int = STL_PERIODO_PADRAO,
        divisao: Optional[Tuple[int, int, int]] = None,
        periodos: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Passo 5: pagina as TRES frases dividindo o orcamento de contexto.

        Cada frase recebe um bloco fixo de paginas do total n_paginas_max
        (default para 7 paginas: 3 tendencia + 2 sazonalidade + 2 residuo;
        regra geral: teto(n/3), piso do resto). Cada frase entra no
        documento com seu marcador ("Tendencia:", "Sazonalidade:",
        "Residuos:") na estrategia "janela"; apenas o PREFIXO de paginas
        atribuido a cada uma entra no documento final.

        Args:
            periodos: tupla (periodo_diario, periodo_semanal) para MSTL.
                Se fornecido, MSTL e usado em vez de STL.

        Returns:
            Mesmo contrato de paginar(), com max_abs = array (3,) contendo
            o max_abs DE CADA componente (passo 3).
        """
        b = self.block_max
        n = self.n_paginas_max
        if divisao is None:
            k_t = n // 3 + (1 if n % 3 else 0)
            k_s = (n - k_t) // 2
            k_r = n - k_t - k_s
            divisao = (k_t, k_s, k_r)
        if sum(divisao) != n:
            raise ValueError(
                f"divisao {divisao} soma {sum(divisao)} != n_paginas_max {n}"
            )
        frase_t, frase_s, frase_r, max_abss = self.encode_stl(
            x_window, periodo=periodo, periodos=periodos,
        )
        ids_parts, lens_parts, descartes, tokens_orig = [], [], 0, 0
        for frase, k in zip((frase_t, frase_s, frase_r), divisao):
            r = self.paginador.paginar("janela", frase)
            m = self.paginador.para_modelo(r)
            ids_parts.append(m["ids"][:k * b])
            lens_parts.append(m["lens"][:k])
            descartes += (
                int(m["paginas_descartadas"]) + max(r["n_paginas"] - k, 0)
            )
            tokens_orig += int(r.get("tokens_originais", 0))
        lens = np.concatenate(lens_parts)
        return {
            "ids": np.concatenate(ids_parts),
            "lens": lens,
            "n_paginas": int((lens > 0).sum()),
            "paginas_descartadas": descartes,
            "tokens_originais": tokens_orig,
            "tokens_retidos": int(lens.sum()),
            "max_abs": np.asarray(max_abss, dtype=np.float32),
        }

    # ---- encode/paginar CEEMDAN (3 frases T/S/R) --------------------- #
    def encode_ceemdan(
        self,
        x_window: np.ndarray,
        trials: int = 100,
        epsilon: float = 0.005,
        max_imf: int = 5,
        seed: int = 42,
    ) -> Tuple[str, str, str, Tuple[float, float, float]]:
        """Janela -> (frase_t, frase_s, frase_r, max_abss) via CEEMDAN."""
        return frases_texto_ceemdan(x_window, trials=trials,
                                     epsilon=epsilon, max_imf=max_imf,
                                     seed=seed)

    def paginar_ceemdan(
        self,
        x_window: np.ndarray,
        divisao: Optional[Tuple[int, int, int]] = None,
        trials: int = 100,
        epsilon: float = 0.005,
        max_imf: int = 5,
        seed: int = 42,
    ) -> Dict:
        """Pagina as 3 frases CEEMDAN dividindo o orcamento de contexto."""
        b = self.block_max
        n = self.n_paginas_max
        if divisao is None:
            k_t = n // 3 + (1 if n % 3 else 0)
            k_s = (n - k_t) // 2
            k_r = n - k_t - k_s
            divisao = (k_t, k_s, k_r)
        frase_t, frase_s, frase_r, max_abss = self.encode_ceemdan(
            x_window, trials=trials, epsilon=epsilon, max_imf=max_imf,
            seed=seed,
        )
        ids_parts, lens_parts, descartes, tokens_orig = [], [], 0, 0
        for frase, k in zip((frase_t, frase_s, frase_r), divisao):
            r = self.paginador.paginar("janela", frase)
            m = self.paginador.para_modelo(r)
            ids_parts.append(m["ids"][:k * b])
            lens_parts.append(m["lens"][:k])
            descartes += (
                int(m["paginas_descartadas"]) + max(r["n_paginas"] - k, 0)
            )
            tokens_orig += int(r.get("tokens_originais", 0))
        lens = np.concatenate(lens_parts)
        return {
            "ids": np.concatenate(ids_parts),
            "lens": lens,
            "n_paginas": int((lens > 0).sum()),
            "paginas_descartadas": descartes,
            "tokens_originais": tokens_orig,
            "tokens_retidos": int(lens.sum()),
            "max_abs": np.asarray(max_abss, dtype=np.float32),
        }

    # ---- encode/paginar VMD (3 frases T/S/R) -------------------------- #
    def encode_vmd(
        self,
        x_window: np.ndarray,
        k: int = 7,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-7,
        init: int = 0,
        trend_ma: Optional[int] = None,
    ) -> Tuple[str, str, str, Tuple[float, float, float]]:
        """Janela -> (frase_t, frase_s, frase_r, max_abss) via VMD."""
        return frases_texto_vmd(x_window, k=k, alpha=alpha, tau=tau,
                                tol=tol, init=init, trend_ma=trend_ma)

    def paginar_vmd(
        self,
        x_window: np.ndarray,
        divisao: Optional[Tuple[int, int, int]] = None,
        k: int = 7,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-7,
        init: int = 0,
        trend_ma: Optional[int] = None,
    ) -> Dict:
        """Pagina as 3 frases VMD dividindo o orcamento de contexto."""
        b = self.block_max
        n = self.n_paginas_max
        if divisao is None:
            k_t = n // 3 + (1 if n % 3 else 0)
            k_s = (n - k_t) // 2
            k_r = n - k_t - k_s
            divisao = (k_t, k_s, k_r)
        frase_t, frase_s, frase_r, max_abss = self.encode_vmd(
            x_window, k=k, alpha=alpha, tau=tau, tol=tol, init=init,
            trend_ma=trend_ma,
        )
        ids_parts, lens_parts, descartes, tokens_orig = [], [], 0, 0
        for frase, kpg in zip((frase_t, frase_s, frase_r), divisao):
            r = self.paginador.paginar("janela", frase)
            m = self.paginador.para_modelo(r)
            ids_parts.append(m["ids"][:kpg * b])
            lens_parts.append(m["lens"][:kpg])
            descartes += (
                int(m["paginas_descartadas"]) + max(r["n_paginas"] - kpg, 0)
            )
            tokens_orig += int(r.get("tokens_originais", 0))
        lens = np.concatenate(lens_parts)
        return {
            "ids": np.concatenate(ids_parts),
            "lens": lens,
            "n_paginas": int((lens > 0).sum()),
            "paginas_descartadas": descartes,
            "tokens_originais": tokens_orig,
            "tokens_retidos": int(lens.sum()),
            "max_abs": np.asarray(max_abss, dtype=np.float32),
        }

    # ---- encode/paginar HIBRIDO MSTL+CEEMDAN ------------------------- #
    def encode_hibrido(
        self,
        x_window: np.ndarray,
        periodos: Tuple[int, int],
        trials: int = 100,
        epsilon: float = 0.005,
        max_imf: int = 5,
        seed: int = 42,
    ) -> Tuple[str, str, str, Tuple[float, float, float]]:
        """Janela -> (frase_t, frase_s, frase_r, max_abss) via MSTL+CEEMDAN."""
        return frases_texto_hibrido(x_window, periodos, trials=trials,
                                     epsilon=epsilon, max_imf=max_imf,
                                     seed=seed)

    def paginar_hibrido(
        self,
        x_window: np.ndarray,
        periodos: Tuple[int, int],
        divisao: Optional[Tuple[int, int, int]] = None,
        trials: int = 100,
        epsilon: float = 0.005,
        max_imf: int = 5,
        seed: int = 42,
    ) -> Dict:
        """Pagina as 3 frases MSTL+CEEMDAN."""
        b = self.block_max
        n = self.n_paginas_max
        if divisao is None:
            k_t = n // 3 + (1 if n % 3 else 0)
            k_s = (n - k_t) // 2
            k_r = n - k_t - k_s
            divisao = (k_t, k_s, k_r)
        frase_t, frase_s, frase_r, max_abss = self.encode_hibrido(
            x_window, periodos, trials=trials, epsilon=epsilon,
            max_imf=max_imf, seed=seed,
        )
        ids_parts, lens_parts, descartes, tokens_orig = [], [], 0, 0
        for frase, k in zip((frase_t, frase_s, frase_r), divisao):
            r = self.paginador.paginar("janela", frase)
            m = self.paginador.para_modelo(r)
            ids_parts.append(m["ids"][:k * b])
            lens_parts.append(m["lens"][:k])
            descartes += (
                int(m["paginas_descartadas"]) + max(r["n_paginas"] - k, 0)
            )
            tokens_orig += int(r.get("tokens_originais", 0))
        lens = np.concatenate(lens_parts)
        return {
            "ids": np.concatenate(ids_parts),
            "lens": lens,
            "n_paginas": int((lens > 0).sum()),
            "paginas_descartadas": descartes,
            "tokens_originais": tokens_orig,
            "tokens_retidos": int(lens.sum()),
            "max_abs": np.asarray(max_abss, dtype=np.float32),
        }

    # ---- encode/paginar PARALELO (MSTL p/ T, CEEMDAN p/ S) ----------- #
    def encode_paralelo(
        self,
        x_window: np.ndarray,
        periodo: int = 24,
        trials: int = 100,
        epsilon: float = 0.005,
        max_imf: int = 5,
        seed: int = 42,
    ) -> Tuple[str, str, str, Tuple[float, float, float]]:
        """Janela -> (frase_t, frase_s, frase_r, max_abss) via paralelo."""
        return frases_texto_paralelo(x_window, periodo=periodo,
                                      trials=trials, epsilon=epsilon,
                                      max_imf=max_imf, seed=seed)

    def paginar_paralelo(
        self,
        x_window: np.ndarray,
        periodo: int = 24,
        divisao: Optional[Tuple[int, int, int]] = None,
        trials: int = 100,
        epsilon: float = 0.005,
        max_imf: int = 5,
        seed: int = 42,
    ) -> Dict:
        """Pagina as 3 frases paralelas (MSTL T + CEEMDAN S)."""
        b = self.block_max
        n = self.n_paginas_max
        if divisao is None:
            k_t = n // 3 + (1 if n % 3 else 0)
            k_s = (n - k_t) // 2
            k_r = n - k_t - k_s
            divisao = (k_t, k_s, k_r)
        frase_t, frase_s, frase_r, max_abss = self.encode_paralelo(
            x_window, periodo=periodo, trials=trials, epsilon=epsilon,
            max_imf=max_imf, seed=seed,
        )
        ids_parts, lens_parts, descartes, tokens_orig = [], [], 0, 0
        for frase, k in zip((frase_t, frase_s, frase_r), divisao):
            r = self.paginador.paginar("janela", frase)
            m = self.paginador.para_modelo(r)
            ids_parts.append(m["ids"][:k * b])
            lens_parts.append(m["lens"][:k])
            descartes += (
                int(m["paginas_descartadas"]) + max(r["n_paginas"] - k, 0)
            )
            tokens_orig += int(r.get("tokens_originais", 0))
        lens = np.concatenate(lens_parts)
        return {
            "ids": np.concatenate(ids_parts),
            "lens": lens,
            "n_paginas": int((lens > 0).sum()),
            "paginas_descartadas": descartes,
            "tokens_originais": tokens_orig,
            "tokens_retidos": int(lens.sum()),
            "max_abs": np.asarray(max_abss, dtype=np.float32),
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
        m_chk = (
            max_abs if isinstance(max_abs, torch.Tensor)
            else torch.as_tensor(np.asarray(max_abs))
        )
        if m_chk.dim() >= 2 and m_chk.shape[-2] > 1:
            raise ValueError(
                "decode(max_abs) NAO se aplica ao contrato STL (max_abs por "
                "componente, shape (B,3,1)): o passo 7 oficial desnormaliza "
                "APENAS pelo RevIN interno do AutoTimes; multiplicar a "
                "predicao pelo max_abs do texto causaria dupla "
                "desnormalizacao."
            )
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


def _processar_janela(
    args_tuple: Tuple,
) -> Dict:
    """Processa UMA janela: decomposicao + texto + tokenizacao + paginacao.

    Funcao auxiliar para executar em paralelo via joblib. Cada worker
    cria seu proprio pipeline (evita problemas de serializacao).

    Args:
        args_tuple: (xw, yw, tokenizer_dir, block_max, n_paginas_max,
                     overlap_frac, decomp, kernel_ma, periodos_mstl,
                     ceemdan_trials, ceemdan_epsilon, ceemdan_max_imf,
                     ceemdan_seed, vmd_k, vmd_alpha, vmd_tau, vmd_tol,
                     vmd_init, vmd_ma, formato) = 20 campos; `formato` e um
                     FormatoTexto congelado (data/casas/encoding) definido
                     como estado do processo ANTES de escrever o texto.

    Returns:
        Dict com xs, ys, ids, lens, max_abs, tokens_orig, tokens_ret,
        n_paginas, paginas_desc.
    """
    (xw, yw, tok_dir, block_max, n_paginas_max, overlap_frac,
     decomp, kernel_ma, periodos_mstl, ceemdan_trials, ceemdan_epsilon,
     ceemdan_max_imf, ceemdan_seed, vmd_k, vmd_alpha, vmd_tau,
     vmd_tol, vmd_init, vmd_ma, formato) = args_tuple

    # Formato do texto desta janela (imutavel; padrao = escrita atual).
    definir_formato(formato)

    pipeline = CharByCharPipeline(
        tok_dir, block_max=block_max,
        n_paginas_max=n_paginas_max, overlap_frac=overlap_frac,
    )

    if decomp == "stl":
        res = pipeline.paginar_stl(xw)
    elif decomp == "mstl":
        res = pipeline.paginar_stl(xw, periodos=periodos_mstl)
    elif decomp == "ceemdan":
        res = pipeline.paginar_ceemdan(xw, trials=ceemdan_trials,
                                        epsilon=ceemdan_epsilon,
                                        max_imf=ceemdan_max_imf,
                                        seed=ceemdan_seed)
    elif decomp == "mstl+ceemdan":
        res = pipeline.paginar_hibrido(xw, periodos=periodos_mstl,
                                        trials=ceemdan_trials,
                                        epsilon=ceemdan_epsilon,
                                        max_imf=ceemdan_max_imf,
                                        seed=ceemdan_seed)
    elif decomp == "paralelo":
        res = pipeline.paginar_paralelo(xw, trials=ceemdan_trials,
                                         epsilon=ceemdan_epsilon,
                                         max_imf=ceemdan_max_imf,
                                         seed=ceemdan_seed)
    elif decomp == "vmd":
        res = pipeline.paginar_vmd(xw, k=vmd_k, alpha=vmd_alpha,
                                    tau=vmd_tau, tol=vmd_tol, init=vmd_init,
                                    trend_ma=vmd_ma)
    elif decomp == "ma":
        res = pipeline.paginar_decomposto(xw, kernel_ma=kernel_ma)
    else:
        res = pipeline.paginar(xw)

    return {
        "x": np.asarray(xw, dtype=np.float32),
        "y": np.asarray(yw, dtype=np.float32),
        "ids": res["ids"],
        "lens": res["lens"],
        "max_abs": np.atleast_1d(res["max_abs"]),
        "tokens_originais": res["tokens_originais"],
        "tokens_retidos": res["tokens_retidos"],
        "n_paginas": res["n_paginas"],
        "paginas_descartadas": res["paginas_descartadas"],
    }


def construir_dataset_char_by_char(
    pool: List[Tuple[np.ndarray, np.ndarray]],
    tokenizer=None,
    block_max: int = BLOCK_MAX,
    n_paginas_max: int = N_PAGINAS_MAX,
    overlap_frac: float = OVERLAP,
    return_stats: bool = False,
    decomp: str = "none",
    kernel_ma: int = 25,
    periodos_mstl: Optional[Tuple[int, int]] = None,
    ceemdan_trials: int = 100,
    ceemdan_epsilon: float = 0.005,
    ceemdan_max_imf: int = 5,
    ceemdan_seed: int = 42,
    ceemdan_backend: str = "pyemd",
    vmd_k: int = 7,
    vmd_alpha: float = 2000.0,
    vmd_tau: float = 0.0,
    vmd_tol: float = 1e-7,
    vmd_init: int = 0,
    vmd_ma: Optional[int] = None,
    formato: Optional[FormatoTexto] = None,
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
        decomp: 'none' = documento unico da serie original; 'stl' =
            FLUXO OFICIAL: tres frases STL reais (Tendencia/Sazonalidade/
            Residuo, LOESS periodo 24) dividindo o contexto (7 paginas ->
            3+2+2), cada componente normalizada pelo PROPRIO max_abs
            (contrato max_abs vira (B,3,1)); 'mstl' = MSTL com periodos
            dinamicos (diario + semanal) detectados da coluna de datas;
            'ma' = ablation legada com duas frases (Tendencia/Residuos
            MA-k25, divisao 4+3). O alvo y e a serie ORIGINAL nos tres
            casos; 'ceemdan' = decomposicao adaptativa trials=100 (o mesmo
            contrato de 3 frases); 'vmd' = VMD (K modos agrupados em
            T/S/R, FFT + <=500 iteracoes, dezenas de ms por janela).
        kernel_ma: kernel da media movel quando decomp='ma'.
        periodos_mstl: tupla (periodo_diario, periodo_semanal) para MSTL.
            Necessario quando decomp='mstl'.
        ceemdan_max_imf: numero maximo de IMFs CEEMDAN (-1 = automatico).
        ceemdan_backend: 'pyemd' (CPU) ou 'hip' (GPU AMD via DLL).
        vmd_k/vmd_alpha/vmd_tau/vmd_tol/vmd_init/vmd_ma: parametros do VMD
            quando decomp='vmd' (K=7 modos agrupados em T/S/R, alpha=2000
            restricao de banda, init=0 = omegas em 0, deterministico ->
            cache reproduzivel; vmd_ma=None = MA automatico ~L/7 na
            tendencia | 0 = sem MA | N = kernel fixo).
        formato: FormatoTexto congelado (data/casas/encoding) com as
            VARIANTES de escrita do Char-by-Char - `casas` (6, 4, 3 ou 2
            decimais por extenso, padrao 6) e `encoding` ('extenso'
            [padrao] ou 'digitos'); `data` so afeta o Pseudo-JSON. Padrao
            (None) = escrita ATUAL byte a byte. Apenas o texto muda:
            paginacao 3+2+2, orçamento de 896 tokens, normalizacao
            local, max_abs e o alvo y continuam identicos.

    Returns:
        TensorDataset(x_numerico, ids, lens, y, max_abs) ou
        (TensorDataset, stats).
    """
    if decomp not in ("none", "ma", "stl", "mstl", "ceemdan", "mstl+ceemdan", "paralelo", "vmd"):
        raise ValueError(
            f"decomp invalida: {decomp!r} (use 'none', 'stl', 'mstl', 'ma',"
            " 'ceemdan', 'mstl+ceemdan', 'paralelo' ou 'vmd')"
        )
    if decomp == "mstl" and periodos_mstl is None:
        raise ValueError("decomp='mstl' exige periodos_mstl=(diario, semanal)")

    # Resolve tokenizer path (string serializavel pelo joblib)
    if tokenizer is None:
        tok_dir = str(caminhos.descobrir_tokenizador())
    elif isinstance(tokenizer, (str, Path)):
        tok_dir = str(tokenizer)
    else:
        tok_dir = getattr(tokenizer, "_name_or_path", None)
        if tok_dir is None:
            tok_dir = str(caminhos.descobrir_tokenizador())

    # Monta argumentos para cada janela (20 campos; o formato viaja junto
    # e e aplicado por _processar_janela antes de escrever o texto).
    fmt = resolver_formato(formato)
    args_list = [
        (xw, yw, tok_dir, block_max, n_paginas_max, overlap_frac,
         decomp, kernel_ma, periodos_mstl, ceemdan_trials, ceemdan_epsilon,
         ceemdan_max_imf, ceemdan_seed, vmd_k, vmd_alpha, vmd_tau, vmd_tol,
         vmd_init, vmd_ma, fmt)
        for xw, yw in pool
    ]

    # Paralelizacao: processa todas as janelas em todos os nucleos.
    # Backend hip: SERIAL (DLL HIP carregada por processo; workers loky
    # teriam de recarregar a GPU e serializariam o mesmo recurso).
    if ceemdan_backend == "hip":
        resultados = [_processar_janela(a) for a in args_list]
    else:
        n_jobs = -1  # todos os nucleos disponiveis
        resultados = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_processar_janela)(a) for a in args_list
        )

    # Agrupa resultados
    xs = [r["x"] for r in resultados]
    ys = [r["y"] for r in resultados]
    ids_linhas = [r["ids"] for r in resultados]
    lens_linhas = [r["lens"] for r in resultados]
    max_abs_linhas = [r["max_abs"] for r in resultados]
    stats = {
        "tokens_originais": [r["tokens_originais"] for r in resultados],
        "tokens_retidos": [r["tokens_retidos"] for r in resultados],
        "paginas": [r["n_paginas"] for r in resultados],
        "paginas_descartadas": [r["paginas_descartadas"] for r in resultados],
        "max_abs": [m for r in resultados for m in np.atleast_1d(r["max_abs"]).tolist()],
    }
    arr_m = np.stack(max_abs_linhas)  # (B,) no contrato simples; (B,3) no STL
    if arr_m.ndim == 1:
        tensores_max_abs = torch.from_numpy(arr_m.reshape(-1, 1, 1).astype(np.float32))
    else:
        tensores_max_abs = torch.from_numpy(arr_m.reshape(-1, arr_m.shape[1], 1).astype(np.float32))
    ds = TensorDataset(
        torch.from_numpy(np.stack(xs)),
        torch.tensor(np.stack(ids_linhas), dtype=torch.int64),
        torch.tensor(np.stack(lens_linhas), dtype=torch.int64),
        torch.from_numpy(np.stack(ys)),
        tensores_max_abs,
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