"""decomposicao - Decomposição VMD (Variational Mode Decomposition).

VMD (Dragomiretskiy & Zosso, 2014) via vmdpy - tradução Python do
toolbox MATLAB oficial. Alternativa ao CEEMDAN com custo muito menor:
FFT + no maximo 500 iteracoes (dezenas de ms por janela) contra ~2000 ms
do CEEMDAN trials=100 em CPU.

Mesma interface do ceemdan/decomposicao.py:
  decompor_vmd(x, ...) -> (tendencia, sazonalidade, residuo)

Convencao identica ao CEEMDAN (decomposicao.py:131-136): os modos sao
ordenados por frequencia central crescente ->
  - modes[0]  = menor frequencia -> Tendencia (com suavizacao MA ~L/7)
  - middle    = medias frequencias -> Sazonalidade (soma)
  - residuo   = x - T - S (inclui erro de fidelidade do VMD; T+S+R
    reconstruem x EXATAMENTE, como no CEEMDAN)

Variantes validadas em 20 janelas reais (ETTh1, L=672) contra CEEMDAN:
  metrica        CEEMDAN      VMD (init=0, K=7, MA=L//7)
  S% (energia   97.18±2.91    97.89±3.32   <- mais sinal previsivel em S
      diaria/                                            (24h/168h)
      semanal
      em S)
  T% (vazamento 2.04±2.66     1.79±3.30    <- menos contaminacao do trend
      p/ trend)
  R% (perdido   0.78±1.38     0.32±0.33    <- menos sinal previsivel
      no residuo)                                       jogado fora
  curv trend     0.0006        0.0005       <- trend tao liso quanto

init=0 (omegas todos em 0, convergem livremente) e deterministico (sem
RNG) -> cache reproduzivel; init=1 e o init uniforme original.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

# Aviso de fallback emitido apenas 1x por processo (evita 94k linhas).
_AVISO_FALLBACK_EMITIDO = False

# Franela de suavizacao da tendencia como fracao da janela (~L/7).
_FRACAO_TREND = 7


def _sanitizar(x: np.ndarray) -> np.ndarray:
    """Remove NaN/Inf e retorna array float64 1D."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _fallback(x: np.ndarray, motivo: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Degradacao controlada: sinal integral como tendencia, resto zerado."""
    global _AVISO_FALLBACK_EMITIDO
    if not _AVISO_FALLBACK_EMITIDO:
        warnings.warn(f"[vmd] fallback ativo ({motivo}); sinal vai como tendencia")
        _AVISO_FALLBACK_EMITIDO = True
    z = np.zeros_like(x)
    return x.copy(), z, z


def suavizar_tendencia(x: np.ndarray, kernel: int) -> np.ndarray:
    """Media movel central com padding replicado (bordas sem artefato).

    Kernel impar; < 3 devolve x intacto. Vetorizada via cumsum (O(L)).
    """
    if kernel is None or int(kernel) < 3:
        return np.asarray(x, dtype=np.float64)
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    x = np.asarray(x, dtype=np.float64)
    pad = k // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    c = np.cumsum(np.concatenate([[0.0], xp]))
    return (c[k:] - c[:-k]) / k


def kernel_trend_auto(L: int) -> int:
    """Kernel impar ~L/7 (para L=672 -> 97; escala junto com a janela)."""
    return max(3, (int(L) // _FRACAO_TREND) | 1)


def extrair_modes_vmd(
    x: np.ndarray,
    k: int = 7,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    init: int = 0,
) -> np.ndarray:
    """Modos VMD (K, L) ordenados por frequencia central CRESCENTE.

    Args:
        x: janela bruta (L,) ou (L, 1).
        k: numero de modos (default 7; K=5 = max_imf do CEEMDAN).
        alpha: restricao de banda (balanco fidelidade x banda).
        tau: passo de subida dual (0 = tolerancia a ruido, default).
        tol: criterio de convergencia (1e-7 = default do toolbox).
        init: 0 = omegas todos em 0 (default, melhor S% e deterministico),
            1 = omegas uniformes (init original), 2 = aleatorio (NAO usar:
            quebra a reprodutibilidade do cache).

    Returns:
        (K, L) float64, linha 0 = menor frequencia, ultima = maior.
    """
    x = _sanitizar(x)
    L = x.size

    if L < 10:
        raise ValueError(f"sinal curto demais para VMD (L={L})")

    k = max(1, int(k))
    init = int(init) if int(init) in (0, 1) else 0

    # vmdpy descarta o ultimo ponto se L e impar -> replica a borda para
    # preservar o comprimento original (truncate depois).
    impar = L % 2 == 1
    f = np.append(x, x[-1]) if impar else x

    from vmdpy import VMD

    # init 0/1 nao usam RNG -> deterministico -> cache estavel.
    u, _u_hat, omega = VMD(f, alpha, tau, k, 0, init, tol)
    u = np.asarray(u, dtype=np.float64)

    if impar:
        u = u[:, :L]

    # omega: (n_iter, K) - frequencia central final de cada modo.
    omega_final = np.asarray(omega, dtype=np.float64)[-1]
    ordem = np.argsort(omega_final)  # crescente: baixa -> alta frequencia
    return u[ordem, :]


def decompor_vmd(
    x_window: np.ndarray,
    k: int = 7,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    init: int = 0,
    trend_ma: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VMD puro -> (tendencia, sazonalidade, residuo).

    VMD recupera K modos em torno de frequencias centrais; apos ordenar
    por frequencia crescente:
      - modes[0]  = menor freq -> Tendencia (suavizada por MA ~L/7)
      - modes[1:-1] = medias freq -> Sazonalidade (oscilacoes)
      - residuo = x - T - S (ultimo modo + erro de fidelidade do VMD;
        garante reconstrucao exata, como no CEEMDAN)

    A suavizacao da tendencia tira o vazamento diario do trend (o modo de
    menor frequencia do VMD ainda puxa um pouco do ciclo de 24h) e deixa a
    curvatura do trend equivalente ou melhor que a do CEEMDAN.

    Args:
        x_window: janela bruta (L,) ou (L, 1).
        k: numero de modos (default 7).
        alpha: restricao de banda (default 2000; >=3000 joga a
            sazonalidade para o residuo - nao usar).
        tau: tolerancia a ruido (default 0 = sem imposicao estrita).
        tol: convergencia (default 1e-7).
        init: 0 = omegas em 0 (default, deterministico) | 1 = uniformes.
        trend_ma: kernel do MA da tendencia. None (default) = automatico
            ~L/7 | 0 = sem suavizacao | N impar = kernel fixo.

    Returns:
        (tendencia, sazonalidade, residuo): tres arrays (L,) float64.
    """
    x = _sanitizar(x_window)
    L = x.size

    if L < 10:
        return _fallback(x, f"L={L} < 10")

    try:
        modes = extrair_modes_vmd(x, k=k, alpha=alpha, tau=tau, tol=tol,
                                  init=init)
    except (ImportError, ModuleNotFoundError):
        # Dependencia ausente = erro de ambiente, nao de dado: não pode
        # virar fallback silencioso (quebraria o cache por engano).
        raise
    except Exception as exc:  # pragma: no cover - protege a rodada longa
        return _fallback(x, f"{type(exc).__name__}: {exc}")

    n_modes = modes.shape[0]

    if trend_ma is None:
        kern = kernel_trend_auto(L)
    else:
        kern = int(trend_ma)
    tendencia = suavizar_tendencia(modes[0], kern)

    if n_modes > 2:
        sazonalidade = np.sum(modes[1:-1], axis=0)
    else:
        sazonalidade = np.zeros_like(x)

    # Residuo = ultimo modo + erro de fidelidade do VMD (a soma dos modos
    # aproxima x apenas dentro do termo de fidelidade) + o que o MA tirou
    # do trend. Assim T+S+R reconstruem x EXATAMENTE, mesma convencao do
    # CEEMDAN (extrair_cIMFs devolve os IMFs + residuo final ate x).
    residuo = x - tendencia - sazonalidade

    return tendencia, sazonalidade, residuo
