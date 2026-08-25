"""
gerar_graficos_convergencia - Convergencia com Bandas de Confianca.

Consome APENAS resultados/estruturas_avancadas_stats.json (21 cenarios,
curvas por epoca: test_mse_media_por_epoca / test_mse_desvio_por_epoca)
e gera em testes_letras/apresentacao/:

    convergencia_21_idiomas.png - Visao Global: 21 curvas, cores
                                  agrupadas por idioma (bandas ±1 dp)
    convergencia_top3.png       - O Podio: os 3 cenarios com menor
                                  Test MSE final, linhas grossas

Estetica: estilo artigo cientifico - fundo branco, grid cinza claro
(alpha=0.3), banda de confianca (fill_between) na cor da linha,
legenda com nome do cenario + resultado final.

SAIDA: testes_letras/apresentacao/convergencia_*.png (300 DPI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from . import caminhos

CAMPEAO: str = "pseudo_json_es__janela"
ORDEM_IDIOMAS: List[str] = ["ES", "EN", "PT"]

# Paleta por idioma (chave -> hex): dourado/verde | azul | cinza/vermelho.
_PALETA_IDIOMA: Dict[str, List[str]] = {
    "ES": [mcolors.to_hex(c) for c in
           plt.cm.YlGn(np.linspace(0.28, 0.88, 7))],       # verde -> dourado
    "EN": [mcolors.to_hex(c) for c in
           plt.cm.Blues(np.linspace(0.30, 0.90, 8))],      # azul claro -> escuro
    "PT": ["#D3D3D3", "#A9A9A9", "#8C8C8C", "#696969",
           "#E74C3C", "#C0392B"],                           # cinzas + vermelhos
}

CORES_PODIO: Dict[str, str] = {
    "1": "#FFB300",   # 1o: laranja vibrante (campeao)
    "2": "#1f77b4",   # 2o: azul
    "3": "#2ca02c",   # 3o: verde
}


def _carregar() -> Dict:
    p = caminhos.RESULTADOS_DIR / "estruturas_avancadas_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"JSON nao encontrado: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _dir_saida() -> Path:
    d = caminhos.BASE_DIR / "apresentacao"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _estilo() -> None:
    """Estilo artigo cientifico (fundo branco, grid leve, sans legivel)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.5,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _cenarios(estat: Dict) -> List[Dict]:
    """Lista plana com chaves normalizadas + curvas por epoca."""
    linhas = []
    for chave, c in estat["cenarios"].items():
        curvas = c.get("test_mse_media_por_epoca") or []
        curvas_dp = c.get("test_mse_desvio_por_epoca") or []
        linhas.append({
            "chave": chave,
            "idioma": c["idioma"].upper(),
            "mse": float(c["test_mse_media"]),
            "dp": float(c["test_mse_desvio"]),
            "curva": [float(v) for v in curvas],
            "curva_dp": [float(v) for v in curvas_dp],
        })
    return linhas


def _cores_por_idioma(cenarios: List[Dict]) -> Dict[str, str]:
    """Atribui uma cor unica por cenario: gradiente por idioma, ordenado
    pelo Test MSE final (o melhor do idioma recebe o tom mais saturado)."""
    cores: Dict[str, str] = {}
    for idi in ORDEM_IDIOMAS:
        grupo = sorted((c for c in cenarios if c["idioma"] == idi),
                       key=lambda c: c["mse"])
        tons = _PALETA_IDIOMA[idi]
        n = len(grupo)
        for i, c in enumerate(grupo):
            cor = tons[min(int(round(i * (len(tons) - 1) / max(n - 1, 1))),
                           len(tons) - 1)]
            cores[c["chave"]] = cor
    return cores


def _plotar_curva(
    ax, cenario: Dict, cor: str, lw: float, ms: float, alpha_banda: float,
    com_legenda: bool,
) -> None:
    """Linha com marcadores circulares + banda sombreada media ± 1 dp."""
    if not cenario["curva"]:
        return
    ep = np.arange(1, len(cenario["curva"]) + 1)
    med = np.asarray(cenario["curva"])
    dp = np.asarray(cenario["curva_dp"])
    ax.plot(ep, med, "o-", color=cor, linewidth=lw, markersize=ms,
            markeredgecolor="black", markeredgewidth=0.8,
            label=f"{cenario['chave']} (Test MSE "
                  f"{cenario['mse']:.2f} ± {cenario['dp']:.2f})"
            if com_legenda else None,
            zorder=4)
    ax.fill_between(ep, med - dp, med + dp, color=cor, alpha=alpha_banda,
                    linewidth=0, zorder=2)


# --------------------------------------------------------------------- #
#  1 - Visao Global: 21 cenarios (cores por idioma, bandas translucidas) #
# --------------------------------------------------------------------- #
def chart_convergencia_21(cenarios: List[Dict]) -> Path:
    _estilo()
    cores = _cores_por_idioma(cenarios)
    ordem = sorted(cenarios, key=lambda c: c["mse"], reverse=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    for c in ordem:  # piores primeiro: os melhores ficam por cima.
        _plotar_curva(ax, c, cores[c["chave"]], lw=1.4, ms=5.5,
                      alpha_banda=0.15, com_legenda=True)

    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val MSE no horizonte (media das 3 seeds)")
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.grid(alpha=0.3)

    ytop = max(np.max(np.asarray(c["curva"]) + np.asarray(c["curva_dp"]))
               for c in cenarios if c["curva"])
    ax.set_ylim(6.4, ytop + 1.9)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], fontsize=11, ncols=2,
                    loc="upper right", frameon=True, framealpha=0.9,
                    edgecolor="0.6", borderpad=0.6, columnspacing=1.2,
                    handlelength=1.6, handletextpad=0.6)
    leg.set_zorder(10)
    fig.tight_layout()
    p = _dir_saida() / "convergencia_21_idiomas.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  2 - O Podio: Top 3 (linhas grossas, bandas destacadas)                #
# --------------------------------------------------------------------- #
def chart_convergencia_top3(cenarios: List[Dict]) -> Path:
    _estilo()
    top3 = sorted(cenarios, key=lambda c: c["mse"])[:3]

    fig, ax = plt.subplots(figsize=(16, 9))
    for pos, c in enumerate(top3, start=1):
        _plotar_curva(ax, c, CORES_PODIO[str(pos)], lw=2.6, ms=10,
                      alpha_banda=0.20, com_legenda=True)

    campeao = next(c for c in top3 if c["chave"] == CAMPEAO)
    ep_ch = np.argmin(campeao["curva"]) + 1
    ax.annotate(f"{campeao['chave']}\nTest MSE {campeao['mse']:.3f}",
                xy=(ep_ch, campeao["curva"][ep_ch - 1]),
                xytext=(ep_ch + 0.6, min(campeao["curva"]) - 1.1),
                fontsize=18, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="black", lw=1.2))

    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val MSE no horizonte (media das 3 seeds)")
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlim(0.5, 11.2)
    ax.grid(alpha=0.3)

    ax.legend(fontsize=18, loc="upper right", frameon=True)
    fig.tight_layout()
    p = _dir_saida() / "convergencia_top3.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def _resumo(cenarios: List[Dict]) -> None:
    print("=" * 80)
    print("CONVERGENCIA COM BANDAS - 21 cenarios (escala raw)")
    print("=" * 80)
    for c in sorted(cenarios, key=lambda x: x["mse"])[:3]:
        print(f"  podio-{sorted(cenarios, key=lambda x: x['mse'])[:3].index(c) + 1}: "
              f"{c['chave']:<32s} final={c['mse']:.3f} ± {c['dp']:.3f} "
              f"({len(c['curva'])} epocas)")


def main() -> None:
    cenarios = _cenarios(_carregar())
    _resumo(cenarios)
    print("\nGraficos de convergencia gerados:")
    for p in (
        chart_convergencia_21(cenarios),
        chart_convergencia_top3(cenarios),
    ):
        print(f"[ok] {p}")


if __name__ == "__main__":
    main()