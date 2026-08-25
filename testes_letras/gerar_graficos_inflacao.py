"""
gerar_graficos_inflacao - Inflacao de tokens ES/PT vs Ingles (GPT-2).

Consome APENAS resultados/benchmark_etth1_extenso.json ("tokens_por_extenso"):
mesmo conteudo numerico (2.000 pontos HUFL) traduzido para ES/PT/EN, nos
niveis verboso / curto / keywords.

Gera em testes_letras/apresentacao/:
    inflacao_tokens.png - Inflacao de tokens vs Ingles (x1 = ingles) por nivel.

Estetica: limpa, para projetor (16:9, 300 DPI), sem titulo grande.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import caminhos

NIVEIS: List[str] = ["verboso", "curto", "keywords"]
NIVEIS_PT: Dict[str, str] = {
    "verboso": "Verboso",
    "curto": "Curto",
    "keywords": "Keywords",
}
CORES_IDIOMA: Dict[str, str] = {
    "ES": "#D4AF37",   # dourado
    "PT": "#2ca02c",   # verde
}
COR_EN: str = "#1f77b4"  # azul (baseline)


def _carregar() -> Dict:
    p = caminhos.RESULTADOS_DIR / "benchmark_etth1_extenso.json"
    if not p.exists():
        raise FileNotFoundError(f"JSON nao encontrado: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _dir_saida() -> Path:
    d = caminhos.BASE_DIR / "apresentacao"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 20,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 2.2,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 1.2,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _inflacao_por_nivel(tokens: Dict) -> Dict[str, Dict[str, float]]:
    """Para cada nivel: inflacao ES/EN e PT/EN (tokens por ponto)."""
    por_nivel: Dict[str, Dict[str, float]] = {n: {} for n in NIVEIS}
    for chave, info in tokens.items():
        nivel = info["nivel"]
        if nivel not in por_nivel:
            continue
        idi = info["idioma"].upper()
        por_nivel[nivel][idi] = float(info["tokens_por_ponto"])
    out: Dict[str, Dict[str, float]] = {}
    for nivel, dados in por_nivel.items():
        en = dados["EN"]
        out[nivel] = {
            "ES": dados["ES"] / en,
            "PT": dados["PT"] / en,
            "abs_ES": dados["ES"],
            "abs_PT": dados["PT"],
            "abs_EN": en,
        }
    return out


def chart_inflacao(tokens: Dict) -> Path:
    _estilo()
    infl = _inflacao_por_nivel(tokens)
    for n in NIVEIS:
        print(f"  {n:>10s}: ES x{infl[n]['ES']:.2f} "
              f"(abs {infl[n]['abs_ES']:.1f}) | PT x{infl[n]['PT']:.2f} "
              f"(abs {infl[n]['abs_PT']:.1f}) | EN abs {infl[n]['abs_EN']:.1f}")

    x = np.arange(len(NIVEIS))
    w = 0.34
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.axhline(1.0, color=COR_EN, linewidth=2.5, linestyle="--",
               zorder=2, label="Ingles (baseline x1)")
    for i, idi in enumerate(["ES", "PT"]):
        vals = [infl[n][idi] for n in NIVEIS]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w,
                      color=CORES_IDIOMA[idi], edgecolor="black",
                      linewidth=1.8, zorder=3, label=f"{idi} vs Ingles")
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 0.5) * w, v + 0.055, f"x{v:.2f}",
                    ha="center", fontsize=24, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([NIVEIS_PT[n] for n in NIVEIS])
    ax.set_ylim(0, 2.6)
    ax.set_ylabel("Inflacao de tokens vs Ingles (mesma janela, GPT-2)")

    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    p = _dir_saida() / "inflacao_tokens.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    tokens = _carregar()["tokens_por_extenso"]
    print("Inflacao de tokens (tokens/ponto, 2.000 pontos HUFL):")
    p = chart_inflacao(tokens)
    print(f"[ok] {p}")


if __name__ == "__main__":
    main()