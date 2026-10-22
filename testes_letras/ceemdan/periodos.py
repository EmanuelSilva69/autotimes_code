"""periodos - Utilidades para análise de períodos via CEEMDAN."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def analisar_imfs(
    x_serie: np.ndarray,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
) -> Dict:
    """Analisa as IMFs extraídas pelo CEEMDAN.

    Retorna informações detalhadas sobre cada IMF:
    - período dominante (via FFT)
    - energia (soma dos quadrados)
    - proporção da energia total

    Args:
        x_serie: série temporal (L,).
        trials: número de trials CEEMDAN.
        epsilon: amplitude do ruído adaptativo.
        max_imf: número máximo de IMFs (-1 = automático, default 5).
        seed: semente.

    Returns:
        dict com chaves:
          - n_imfs: número de IMFs
          - imfs: array (n_imfs, L)
          - periodos: lista de períodos dominantes por IMF
          - energias: lista de energias por IMF
          - props: proporção da energia total por IMF
    """
    from PyEMD import CEEMDAN

    x = np.asarray(x_serie, dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=False)
    ceemdan.noise_seed(seed)
    cIMFs = ceemdan(x, max_imf=max_imf)

    n_imfs, L = cIMFs.shape
    periodos = []
    energias = []
    energia_total = np.sum(x ** 2) + 1e-12

    for i in range(n_imfs):
        imf = cIMFs[i]
        # Período dominante via FFT
        fft = np.fft.rfft(imf)
        freqs = np.fft.rfftfreq(L)
        mask = (freqs > 0) & (freqs < 0.5)
        if mask.any():
            dom_freq = freqs[mask][np.argmax(np.abs(fft[mask]))]
            periodo = int(round(1.0 / dom_freq)) if dom_freq > 0 else 0
        else:
            periodo = 0
        periodos.append(periodo)
        # Energia
        e = float(np.sum(imf ** 2))
        energias.append(e)

    props = [e / energia_total for e in energias]

    return {
        "n_imfs": n_imfs,
        "imfs": cIMFs,
        "periodos": periodos,
        "energias": energias,
        "props": props,
    }


def formatar_periodos_ceemdan(periodos: List[int]) -> str:
    """Formata lista de períodos como string para CLI."""
    return ",".join(str(p) for p in periodos)
