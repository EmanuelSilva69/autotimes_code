"""graficos - Visualização de diagnóstico da decomposição CEEMDAN."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_ceemdan(
    x_serie: np.ndarray,
    imfs: np.ndarray,
    titulo: str = "Decomposição CEEMDAN",
    out_dir: Optional[Path] = None,
    nome: str = "ceemdan_diagnostico.png",
) -> Optional[Path]:
    """Plota sinal original + IMFs extraídas pelo CEEMDAN.

    Args:
        x_serie: série original (L,).
        imfs: array (n_imfs, L) com as IMFs.
        titulo: título do gráfico.
        out_dir: diretório de saída (None = não salva).
        nome: nome do arquivo PNG.

    Returns:
        Caminho do PNG salvo, ou None se não salvou.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    n_imfs = imfs.shape[0]
    L = x_serie.size

    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(14, 2.5 * (n_imfs + 1)),
                              sharex=True)
    if n_imfs + 1 == 1:
        axes = [axes]

    # Sinal original
    axes[0].plot(range(L), x_serie, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("Original", fontsize=9)
    axes[0].set_title(titulo, fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # IMFs
    cores = plt.cm.Set3(np.linspace(0, 1, n_imfs))
    for i in range(n_imfs):
        axes[i + 1].plot(range(L), imfs[i], color=cores[i], linewidth=0.6)
        axes[i + 1].set_ylabel(f"IMF {i + 1}", fontsize=9)
        axes[i + 1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Tempo (passos)", fontsize=9)
    plt.tight_layout()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / nome
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    plt.close(fig)
    return None


def plot_comparacao_ceemdan_vs_mstl(
    x_serie: np.ndarray,
    t_ceemdan: np.ndarray,
    s_ceemdan: np.ndarray,
    r_ceemdan: np.ndarray,
    t_mstl: np.ndarray,
    s_mstl: np.ndarray,
    r_mstl: np.ndarray,
    titulo: str = "CEEMDAN vs MSTL",
    out_dir: Optional[Path] = None,
    nome: str = "ceemdan_vs_mstl.png",
) -> Optional[Path]:
    """Plota comparação lado a lado: CEEMDAN vs MSTL."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    L = x_serie.size
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Original
    axes[0].plot(range(L), x_serie, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("Original")
    axes[0].set_title(titulo, fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Tendência
    axes[1].plot(range(L), t_ceemdan, color="#e74c3c", linewidth=0.8,
                 label="CEEMDAN", alpha=0.8)
    axes[1].plot(range(L), t_mstl, color="#3498db", linewidth=0.8,
                 label="MSTL", alpha=0.8)
    axes[1].set_ylabel("Tendência")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Sazonalidade
    axes[2].plot(range(L), s_ceemdan, color="#e74c3c", linewidth=0.6,
                 label="CEEMDAN", alpha=0.8)
    axes[2].plot(range(L), s_mstl, color="#3498db", linewidth=0.6,
                 label="MSTL", alpha=0.8)
    axes[2].set_ylabel("Sazonalidade")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Resíduo
    axes[3].plot(range(L), r_ceemdan, color="#e74c3c", linewidth=0.6,
                 label="CEEMDAN", alpha=0.8)
    axes[3].plot(range(L), r_mstl, color="#3498db", linewidth=0.6,
                 label="MSTL", alpha=0.8)
    axes[3].set_ylabel("Resíduo")
    axes[3].set_xlabel("Tempo (passos)")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / nome
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    plt.close(fig)
    return None
