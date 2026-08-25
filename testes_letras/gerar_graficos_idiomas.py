"""
gerar_graficos_idiomas - Aprofundamento linguistico: EN vs ES vs PT.

Consome APENAS resultados/estruturas_avancadas_stats.json (21 cenarios,
escala raw do harness) e gera em testes_letras/comparacao/:

    08_boxplot_idiomas.png        - distribuicao do Test MSE por idioma
                                    (boxplot + pontos individuais)
    09_agregado_idioma_barras.png - media global por idioma (error bars
                                    pretas explicitas + baseline LN)
    10_template_vs_idioma.png     - interacao formato x idioma (barras
                                    agrupadas, media das 2 estrategias)

Paleta semantica fixa (consistente nos 3 paineis):
    ES = Dourado #D4AF37 | EN = Azul #1f77b4 | PT = Laranja #ff7f0e

SAIDA: testes_letras/comparacao/*.png (300 DPI, fonte serif, estilo IEEE).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import caminhos

# --------------------------------------------------------------------- #
#  Constantes                                                           #
# --------------------------------------------------------------------- #
BARREIRA_LN: float = 7.2445            # baseline LN (es_keywords)
COR_REF: str = "#d62728"               # vermelho: referencias
CAMPEAO: str = "pseudo_json_es__janela"
BASELINE_KEY: str = "es_keywords__keywords"

ORDEM_IDIOMAS: List[str] = ["ES", "EN", "PT"]
CORES_IDIOMA: Dict[str, str] = {
    "ES": "#D4AF37",                   # dourado (cor do campeao)
    "EN": "#1f77b4",                   # azul
    "PT": "#ff7f0e",                   # laranja
}
# Ordem de exibicao e a CHAVE crua correspondente no JSON.
ORDEM_TIPOS: List[Tuple[str, str]] = [
    ("Pseudo-JSON", "pseudo_json"),
    ("Estrutural", "estrutural"),
    ("Relacional", "relacional"),
    ("Temporal Delta", "temporal_delta"),
]


def _carregar() -> Dict:
    p = caminhos.RESULTADOS_DIR / "estruturas_avancadas_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"JSON nao encontrado: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _dir_saida() -> Path:
    d = caminhos.BASE_DIR / "comparacao"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _estilo() -> None:
    sns.set_theme(style="whitegrid", context="paper",
                  font_scale=1.05, rc={"font.family": "serif"})


def _cenarios(estat: Dict) -> List[Dict]:
    """Lista plana dos cenarios com campos normalizados."""
    linhas = []
    for chave, c in estat["cenarios"].items():
        linhas.append({
            "chave": chave,
            "tipo": c["tipo"],
            "idioma": c["idioma"].upper(),
            "estrategia": c["estrategia"],
            "mse": float(c["test_mse_media"]),
            "dp": float(c["test_mse_desvio"]),
            "mae": float(c["test_mae_media"]),
            "e_baseline": chave == BASELINE_KEY,
        })
    return linhas


# --------------------------------------------------------------------- #
#  08 - Boxplot: distribuicao por idioma + pontos individuais            #
# --------------------------------------------------------------------- #
def chart_boxplot(cenarios: List[Dict]) -> Path:
    _estilo()
    y = [c["mse"] for c in cenarios]
    x = [c["idioma"] for c in cenarios]
    champ = next(c for c in cenarios if c["chave"] == CAMPEAO)
    base = next(c for c in cenarios if c["chave"] == BASELINE_KEY)

    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    sns.boxplot(
        x=x, y=y, hue=x, palette=CORES_IDIOMA, dodge=False, width=0.5,
        linewidth=1.2, fliersize=0, ax=ax, legend=False,
        boxprops=dict(alpha=0.92),
    )
    sns.stripplot(
        x=x, y=y, hue=x, palette=CORES_IDIOMA, dodge=False, jitter=0.16,
        size=6.2, edgecolor="black", linewidth=0.7, alpha=0.95, ax=ax,
        legend=False,
    )

    # Campeao em destaque (estrela) + baseline LN (quadrado aberto).
    i_ch = ORDEM_IDIOMAS.index(champ["idioma"])
    ax.scatter([i_ch], [champ["mse"]], marker="*", s=360, color="white",
               edgecolor="black", linewidth=1.5, zorder=7,
               label=f"Campeao {CAMPEAO} = {champ['mse']:.3f}")
    ax.annotate(
        "campeao: pseudo_json_es | janela",
        xy=(i_ch, champ["mse"]), xytext=(0.34, 6.62),
        fontsize=9.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", lw=0.8),
    )
    i_bs = ORDEM_IDIOMAS.index(base["idioma"])
    ax.scatter([i_bs], [base["mse"]], marker="s", s=130, facecolor="none",
               edgecolor=COR_REF, linewidth=1.9, zorder=7,
               label=f"Baseline LN (es_keywords) = {base['mse']:.3f}")

    ax.axhline(BARREIRA_LN, color=COR_REF, ls="--", lw=1.3, zorder=1)
    ax.text(2.44, BARREIRA_LN + 0.015, f"Baseline LN = {BARREIRA_LN:.2f}",
            fontsize=8.8, color=COR_REF, ha="right")

    rotulos = [
        f"{i}\n(n = {sum(1 for c in cenarios if c['idioma'] == i)} cenarios)"
        for i in ORDEM_IDIOMAS
    ]
    ax.set_xticks(range(len(ORDEM_IDIOMAS)))
    ax.set_xticklabels(rotulos, fontsize=11)
    ax.set_ylim(6.35, 8.05)
    ax.set_ylabel("Test MSE raw (media 3 seeds; menor = melhor)", fontsize=11)
    ax.set_title("Distribuicao do Test MSE por Idioma — 21 cenarios\n"
                 "ES concentra 3 dos 5 melhores cenarios (incl. o campeao); "
                 "a vitoria do ES nao e outlier",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8.6, loc="upper right", frameon=True)

    fig.tight_layout()
    p = _dir_saida() / "08_boxplot_idiomas.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  09 - Media global agregada por idioma                                 #
# --------------------------------------------------------------------- #
def chart_agregado(cenarios: List[Dict]) -> Path:
    _estilo()
    medias, desvios, ns = [], [], []
    for idi in ORDEM_IDIOMAS:
        vals = [c["mse"] for c in cenarios if c["idioma"] == idi]
        medias.append(float(np.mean(vals)))
        desvios.append(float(np.std(vals, ddof=1)))
        ns.append(len(vals))

    x = np.arange(len(ORDEM_IDIOMAS))
    cores = [CORES_IDIOMA[i] for i in ORDEM_IDIOMAS]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    bars = ax.bar(x, medias, yerr=desvios, width=0.5, color=cores,
                  edgecolor="black", linewidth=0.7, capsize=6, zorder=3,
                  error_kw={"elinewidth": 1.4, "capthick": 1.4,
                            "ecolor": "black"})
    for xi, m, d, n in zip(x, medias, desvios, ns):
        ax.text(xi, m + d + 0.02, f"{m:.3f}\n± {d:.3f}", ha="center",
                fontsize=9.5, fontweight="bold")

    ax.axhline(BARREIRA_LN, color=COR_REF, ls="--", lw=1.5, zorder=2)
    ax.text(2.40, BARREIRA_LN + 0.012,
            f"Baseline LN (es_keywords) = {BARREIRA_LN:.2f}",
            fontsize=9, color=COR_REF, ha="right")

    rotulos = [
        f"{i}\n(n = {n} cenarios)" for i, n in zip(ORDEM_IDIOMAS, ns)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=11.5)
    ax.set_ylabel("Test MSE raw medio (todos os cenarios do idioma)",
                  fontsize=11)
    ax.set_ylim(0, 8.0)
    ax.set_title("Test MSE Medio Agregado por Idioma\n"
                 "(independente da estrategia de paginacao; ES inclui a "
                 "baseline LN 7,245)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    p = _dir_saida() / "09_agregado_idioma_barras.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  10 - Interacao: formato x idioma (barras agrupadas)                   #
# --------------------------------------------------------------------- #
def chart_template_x_idioma(cenarios: List[Dict]) -> Path:
    _estilo()
    largura, gap = 0.26, 0.18

    def _agregar(tipo_key: str, idi: str) -> Tuple[float, float, int]:
        vals = [c["mse"] for c in cenarios
                if c["tipo"] == tipo_key and c["idioma"] == idi
                and not c["e_baseline"]]
        if not vals:
            return None, None, 0
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals))

    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    x_centro = np.arange(len(ORDEM_TIPOS))
    offset = {j: (i - 1) * (largura + gap / 2)
              for i, j in enumerate(ORDEM_IDIOMAS)}

    for idi in ORDEM_IDIOMAS:
        for gi, (_, tipo_key) in enumerate(ORDEM_TIPOS):
            m, d, n = _agregar(tipo_key, idi)
            if m is None:
                continue
            xi = x_centro[gi] + offset[idi]
            ax.bar(xi, m, width=largura, yerr=d, color=CORES_IDIOMA[idi],
                   edgecolor="black", linewidth=0.6, capsize=3.5, zorder=3,
                   error_kw={"elinewidth": 1.1, "capthick": 1.1,
                             "ecolor": "black"})
            ax.text(xi, m + (d or 0) + 0.015, f"{m:.3f}", ha="center",
                    fontsize=8.4, fontweight="bold", color="0.12")

    ax.axhline(BARREIRA_LN, color=COR_REF, ls="--", lw=1.5, zorder=2)
    ax.text(3.55, BARREIRA_LN + 0.01, f"Baseline LN = {BARREIRA_LN:.2f}",
            fontsize=9, color=COR_REF, ha="right")

    ax.set_xticks(x_centro)
    ax.set_xticklabels([nome for nome, _ in ORDEM_TIPOS], fontsize=11.5)
    ax.set_xlim(-0.55, len(ORDEM_TIPOS) - 0.45)
    ax.set_ylim(6.7, 8.0)
    ax.set_ylabel("Test MSE raw (media das estrategias janela/keywords)",
                  fontsize=11)
    ax.set_title("Interacao Idioma x Estrutura de Prompting\n"
                 "ES vence em Pseudo-JSON e Estrutural; PT vence em "
                 "Relacional; EN sozinho em Temporal Delta",
                 fontsize=12.5, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=CORES_IDIOMA[i],
                             edgecolor="black", label=i)
               for i in ORDEM_IDIOMAS]
    ax.legend(handles=handles, fontsize=10, loc="lower right", frameon=True)
    fig.tight_layout()
    p = _dir_saida() / "10_template_vs_idioma.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def _resumo_console(cenarios: List[Dict]) -> None:
    print("=" * 84)
    print("ANALISE LINGUISTICA - EN vs ES vs PT (21 cenarios, escala raw)")
    print("=" * 84)
    for idi in ORDEM_IDIOMAS:
        vals = sorted(c["mse"] for c in cenarios if c["idioma"] == idi)
        med = float(np.mean(vals))
        dp = float(np.std(vals, ddof=1))
        print(f"  {idi:<3s} n={len(vals):>2d} | media={med:.3f} "
              f"dp={dp:.3f} | min={min(vals):.3f} mediana="
              f"{float(np.median(vals)):.3f} max={max(vals):.3f} | "
              f"{[f'{v:.3f}' for v in vals]}")
    print("-" * 84)
    top5 = sorted(cenarios, key=lambda c: c["mse"])[:5]
    for c in top5:
        print(f"  top-{top5.index(c) + 1}: {c['chave']:<32s} {c['mse']:.3f}")
    print("  agregados por (tipo, idioma):")
    for nome, tipo_key in ORDEM_TIPOS:
        for idi in ORDEM_IDIOMAS:
            vals = [c["mse"] for c in cenarios
                    if c["tipo"] == tipo_key and c["idioma"] == idi
                    and not c["e_baseline"]]
            if vals:
                print(f"    {nome:<16s} {idi}: {np.mean(vals):.3f} "
                      f"± {np.std(vals, ddof=1):.3f} (n={len(vals)})")


def main() -> None:
    cenarios = _cenarios(_carregar())
    _resumo_console(cenarios)
    print("\nPaineis linguisticos gerados:")
    for p in (
        chart_boxplot(cenarios),
        chart_agregado(cenarios),
        chart_template_x_idioma(cenarios),
    ):
        print(f"[ok] {p}")


if __name__ == "__main__":
    main()