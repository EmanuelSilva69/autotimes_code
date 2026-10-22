"""decomposicao - Funções de decomposição temporal via CEEMDAN.

4 estratégias:
  1. decompor_ceemdan: CEEMDAN puro (alternativa ao MSTL)
  2. decompor_hibrido_mstl_ceemdan: MSTL para T/S + CEEMDAN refina R
  3. decompor_paralelo: MSTL para T + CEEMDAN para S
  4. extrair_periodos_ceemdan: CEEMDAN descobre períodos → lista

Todas retornam (tendencia, sazonalidade, residuo) — mesma interface
que decompor_stl() do texto_char_by_char.py.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


def _sanitizar(x: np.ndarray) -> np.ndarray:
    """Remove NaN/Inf e retorna array float64 1D."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# Cache global (por processo worker) do EMD dos ruídos CEEMDAN.
# Com seed fixa e janelas do mesmo tamanho, all_noises é idêntico entre
# chamadas -> _decompose_noise retorna sempre o mesmo resultado (~51% do
# tempo total). Cacheia para tornar bit-idêntico e ~2x mais rápido.
_NOISE_EMD_CACHE: dict = {}


def _criar_ceemdan(trials: int = 100, epsilon: float = 0.005,
                    seed: int = 42):
    """Instancia CEEMDAN com parâmetros padrão (com cache de ruído)."""
    from PyEMD import CEEMDAN
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=False)
    ceemdan.noise_seed(seed)

    orig_decompose = ceemdan._decompose_noise

    def _decompose_cached():
        key = (trials, seed, int(ceemdan.all_noises.shape[1]),
               ceemdan.noise_scale, ceemdan.noise_kind)
        hit = _NOISE_EMD_CACHE.get(key)
        if hit is None:
            hit = orig_decompose()
            _NOISE_EMD_CACHE[key] = hit  # lista fica intacta (so' del[:] na copia)
        # Copias: ceemdan() faz `del self.all_noise_EMD[:]` ao final
        return [np.asarray(a) for a in hit]

    ceemdan._decompose_noise = _decompose_cached
    return ceemdan


# Backend de extracao de cIMFs: "pyemd" (CPU) ou "hip" (GPU AMD via DLL
# CEEMDAN-GPU/ceemdan_gpu.dll). Selecionavel por set_ceemdan_backend()
# (CLI --ceemdan-backend) ou pela variavel de ambiente CEEMDAN_BACKEND.
CEEMDAN_BACKEND: str = os.environ.get("CEEMDAN_BACKEND", "pyemd")


def set_ceemdan_backend(backend: str) -> None:
    """Seleciona o backend de decomposicao: 'pyemd' (CPU) ou 'hip' (GPU)."""
    global CEEMDAN_BACKEND
    if backend not in ("pyemd", "hip"):
        raise ValueError(f"backend CEEMDAN desconhecido: {backend!r}")
    CEEMDAN_BACKEND = backend


def extrair_cIMFs(
    x: np.ndarray,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """cIMFs (n_imfs, L) no backend ativo (CEEMDAN_BACKEND).

    Convencao identica em ambos: linha 0 = IMF1 (alta frequencia),
    ultima linha = tendencia/residuo final.
    """
    if CEEMDAN_BACKEND == "hip":
        from .ceemdan_gpu import ceemdan_gpu
        # a DLL nao suporta max_imf=-1 (auto): usa o default 5
        n_imf = int(max_imf) if max_imf and int(max_imf) > 0 else 5
        return ceemdan_gpu(x, trials=trials, epsilon=epsilon,
                           max_imf=n_imf, seed=seed)
    ceemdan = _criar_ceemdan(trials=trials, epsilon=epsilon, seed=seed)
    return ceemdan(np.asarray(x, dtype=np.float64), max_imf=max_imf)


def decompor_ceemdan(
    x_window: np.ndarray,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CEEMDAN puro → (tendencia, sazonalidade, residuos).

    CEEMDAN extrai IMFs em ordem de maior → menor frequência:
      - cIMFs[0] = maior frequência (ruído) → Resíduo
      - cIMFs[1:-1] = médias frequências (oscilações) → Sazonalidade
      - cIMFs[-1] = menor frequência (trend lento) → Tendência

    Args:
        x_window: janela bruta (L,) ou (L, 1).
        trials: número de trials CEEMDAN (mais = mais estável).
        epsilon: amplitude do ruído adaptativo.
        max_imf: número máximo de IMFs (-1 = automático, default 5).
        seed: semente para reprodutibilidade.

    Returns:
        (tendencia, sazonalidade, residuo): três arrays (L,) float64.
    """
    x = _sanitizar(x_window)
    L = x.size

    if L < 10:
        return x.copy(), np.zeros_like(x), np.zeros_like(x)

    cIMFs = extrair_cIMFs(x, trials=trials, epsilon=epsilon,
                          max_imf=max_imf, seed=seed)  # (n_imfs, L)

    n_imfs = cIMFs.shape[0]

    if n_imfs <= 1:
        return x.copy(), np.zeros_like(x), np.zeros_like(x)

    residuo = cIMFs[0]
    if n_imfs > 2:
        sazonalidade = np.sum(cIMFs[1:-1], axis=0)
    else:
        sazonalidade = np.zeros_like(x)
    tendencia = cIMFs[-1]

    return tendencia, sazonalidade, residuo


def decompor_hibrido_mstl_ceemdan(
    x_window: np.ndarray,
    periodos: Tuple[int, int],
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MSTL para T/S + CEEMDAN refina o resíduo R.

    Fluxo:
      1. MSTL: sinal → T_mstl + S_mstl + R_mstl
      2. CEEMDAN: R_mstl → IMF1 + IMF2 + ... + Resíduo_final
      3. Retorna: (T_mstl, S_mstl, R_refinado)

    Args:
        x_window: janela bruta (L,) ou (L, 1).
        periodos: tupla (periodo_diario, periodo_semanal) para MSTL.
        trials: número de trials CEEMDAN.
        epsilon: amplitude do ruído adaptativo.
        max_imf: número máximo de IMFs (-1 = automático, default 5).
        seed: semente para reprodutibilidade.

    Returns:
        (tendencia, sazonalidade, residuo): três arrays (L,) float64.
    """
    from ..texto_char_by_char import decompor_stl

    x = _sanitizar(x_window)

    # Passo 1: MSTL tradicional
    t_mstl, s_mstl, r_mstl = decompor_stl(x, periodos=periodos)

    # Passo 2: CEEMDAN no resíduo do MSTL (uma unica chamada)
    ceemdan = _criar_ceemdan(trials=trials, epsilon=epsilon, seed=seed)
    cIMFs_r = ceemdan(r_mstl, max_imf=max_imf)  # (n_imfs, L)

    if cIMFs_r.shape[0] <= 2:
        # Resíduo já é simples, não precisa refinar
        return t_mstl, s_mstl, r_mstl

    # IMFs intermediárias = ruído estruturado
    # (última IMF = tendência residual, descartada junto com o resíduo)
    r_ruido = np.sum(cIMFs_r[:-1], axis=0)

    return t_mstl, s_mstl, r_ruido


def decompor_paralelo(
    x_window: np.ndarray,
    periodo: int = 24,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MSTL(1 período) para tendência + CEEMDAN para sazonalidade.

    Fluxo:
      1. MSTL clássico (1 período): sinal → T_mstl + S_mstl + R_mstl
      2. CEEMDAN no sinal original → IMFs
      3. Sazonalidade = soma das IMFs de média frequência do CEEMDAN
      4. Resíduo = original - T_mstl - S_ceemdan

    Args:
        x_window: janela bruta (L,) ou (L, 1).
        periodo: período sazonal para MSTL clássico.
        trials: número de trials CEEMDAN.
        epsilon: amplitude do ruído adaptativo.
        max_imf: número máximo de IMFs (-1 = automático, default 5).
        seed: semente para reprodutibilidade.

    Returns:
        (tendencia, sazonalidade, residuo): três arrays (L,) float64.
    """
    from ..texto_char_by_char import decompor_stl

    x = _sanitizar(x_window)

    # Passo 1: MSTL para tendência
    t_mstl, _, _ = decompor_stl(x, periodo=periodo)

    # Passo 2: CEEMDAN para sazonalidade
    ceemdan = _criar_ceemdan(trials=trials, epsilon=epsilon, seed=seed)
    cIMFs = ceemdan(x, max_imf=max_imf)  # (n_imfs, L)

    n_imfs = cIMFs.shape[0]
    if n_imfs <= 2:
        s_ceemdan = np.zeros_like(x)
    else:
        s_ceemdan = np.sum(cIMFs[1:-1], axis=0)

    # Passo residual
    residuo = x - t_mstl - s_ceemdan

    return t_mstl, s_ceemdan, residuo


def extrair_periodos_ceemdan(
    x_serie: np.ndarray,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
    min_periodo: int = 6,
    max_periodo: int = 336,
) -> List[int]:
    """Extrai períodos dominantes das IMFs do CEEMDAN via FFT.

    Para cada IMF, calcula a FFT e encontra a frequência dominante.
    Converte para período (passos) e filtra por faixa razoável.

    Args:
        x_serie: série temporal completa (L,) — tipicamente treino.
        trials: número de trials CEEMDAN.
        epsilon: amplitude do ruído adaptativo.
        max_imf: número máximo de IMFs (-1 = automático, default 5).
        seed: semente para reprodutibilidade.
        min_periodo: período mínimo aceitável (default 6 = 6h).
        max_periodo: período máximo aceitável (default 336 = 14 dias).

    Returns:
        Lista de períodos dominantes, ordenada e sem duplicatas.
    """
    x = _sanitizar(x_serie)

    if x.size < 20:
        return []

    ceemdan = _criar_ceemdan(trials=trials, epsilon=epsilon, seed=seed)
    cIMFs = ceemdan(x, max_imf=max_imf)  # (n_imfs, L)

    periodos = []
    L = x.size

    for i in range(cIMFs.shape[0]):
        imf = cIMFs[i]
        # FFT
        fft = np.fft.rfft(imf)
        freqs = np.fft.rfftfreq(L)
        # Ignorar DC (freq=0) e altas frequências (> Nyquist/2)
        mask = (freqs > 1.0 / max_periodo) & (freqs < 0.5)
        if not mask.any():
            continue
        fft_mag = np.abs(fft)
        dom_idx = np.argmax(fft_mag[mask])
        dom_freq = freqs[mask][dom_idx]
        periodo = int(round(1.0 / dom_freq))
        if min_periodo <= periodo <= max_periodo:
            periodos.append(periodo)

    # Deduplicar e ordenar
    return sorted(set(periodos))


def _contar_imfs(x: np.ndarray, trials: int = 100,
                  epsilon: float = 0.005, max_imf: int = 5,
                  seed: int = 42) -> int:
    """Conta quantas IMFs o CEEMDAN extrai de um sinal (auxiliar)."""
    if x.size < 10:
        return 0
    ceemdan = _criar_ceemdan(trials=trials, epsilon=epsilon, seed=seed)
    cIMFs = ceemdan(x, max_imf=max_imf)
    return cIMFs.shape[0]
