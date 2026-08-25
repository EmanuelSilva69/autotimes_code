"""
gerar_graficos_slides - Graficos otimizados para PROJETOR (16:9), LIMPOS.

Sem legendas, sem titulos grandes, sem anotacoes: apenas o dado.

Narrativa obrigatoria da apresentacao:
    1-3) A jornada exploratoria NAO menciona o baseline numerico (0.3539).
         O publico so ve o baseline numerico no slide 4 (o climax).
    4)   Apples-to-Apples final: campeao textual (0.3669 z) vs baseline
         numerico (0.3539 z).

Fontes de dados (somente leitura):
    resultados/estruturas_avancadas_stats.json   (21 cenarios, escala raw)
    resultados/veredito_final_apples_to_apples.json (veredito final z-score)

SAIDAS: testes_letras/apresentacao/
    slide_01_idiomas.png   - Paradoxo dos Idiomas (boxplot + pontos)
    slide_02_ranking.png   - Top 4 + Baseline LN (barras horizontais)
    slide_03_escala.png    - raw -> z-score (dois paineis)
    slide_04_veredito.png  - Apples-to-Apples (barras lado a lado)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import caminhos

# --------------------------------------------------------------------- #
#  Constantes                                                           #
# --------------------------------------------------------------------- #
VAR_HUFL: float = 33.7877            # var_treino(HUFL): fator raw -> z
CAMPEAO: str = "pseudo_json_es__janela"
BASELINE_LN_KEY: str = "es_keywords__keywords"
ORDEM_IDIOMAS: List[str] = ["ES", "EN", "PT"]
CORES_IDIOMA: Dict[str, str] = {
    "ES": "#D4AF37",                   # dourado
    "EN": "#1f77b4",                   # azul
    "PT": "#2ca02c",                   # verde
}
COR_CAMPEAO: str = "#FFB300"          # laranja vibrante (destaque)
COR_NEUTRA: str = "#9AA7B8"           # aco claro (demais top-4)
COR_BASELINE: str = "#6b7280"         # cinza (baseline textual LN)
COR_NUM: str = "#1f77b4"              # azul (baseline numerico, slide 4)


def _carregar_json(nome: str) -> Dict:
    p = caminhos.RESULTADOS_DIR / nome
    if not p.exists():
        raise FileNotFoundError(f"JSON nao encontrado: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _dir_saida() -> Path:
    d = caminhos.BASE_DIR / "apresentacao"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _estilo_slides() -> None:
    """Estetica de projetor: limpa, fontes grandes, sem titulos/legendas."""
    sns.set_theme(style="white", context="paper", rc={
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
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


def _cenarios(estat: Dict) -> List[Dict]:
    linhas = []
    for chave, c in estat["cenarios"].items():
        linhas.append({
            "chave": chave,
            "tipo": c["tipo"],
            "idioma": c["idioma"].upper(),
            "estrategia": c["estrategia"],
            "mse": float(c["test_mse_media"]),
            "dp": float(c["test_mse_desvio"]),
            "e_baseline_ln": chave == BASELINE_LN_KEY,
        })
    return linhas


def _rotulo_cenario(c: Dict) -> str:
    if c["e_baseline_ln"]:
        return "Baseline LN"
    nome_tipo = {
        "pseudo_json": "Pseudo-JSON",
        "estrutural": "Estrutural",
        "relacional": "Relacional",
        "temporal_delta": "Temporal Delta",
    }[c["tipo"]]
    return f"{nome_tipo} {c['idioma']}"


# --------------------------------------------------------------------- #
#  1 - Paradoxo dos Idiomas (sem titulo/legenda/notas)                   #
# --------------------------------------------------------------------- #
def slide_01_idiomas(cenarios: List[Dict]) -> Path:
    _estilo_slides()
    x = [c["idioma"] for c in cenarios]
    y = [c["mse"] for c in cenarios]

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.boxplot(x=x, y=y, hue=x, palette=CORES_IDIOMA, dodge=False,
                width=0.45, linewidth=2.4, fliersize=0, ax=ax, legend=False,
                boxprops=dict(alpha=0.95))
    sns.stripplot(x=x, y=y, hue=x, palette=CORES_IDIOMA, dodge=False,
                  jitter=0.15, size=11, edgecolor="black", linewidth=1.1,
                  ax=ax, legend=False)

    ax.set_xticks(range(len(ORDEM_IDIOMAS)))
    ax.set_xticklabels([
        f"{i} (n={sum(1 for c in cenarios if c['idioma'] == i)})"
        for i in ORDEM_IDIOMAS
    ])
    ax.set_ylim(6.7, 7.85)
    ax.set_ylabel("Test MSE raw (menor = melhor)")

    fig.tight_layout()
    p = _dir_saida() / "slide_01_idiomas.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  2 - Top 4 + Baseline LN (sem titulo/arrow/textos)                     #
# --------------------------------------------------------------------- #
def slide_02_ranking(cenarios: List[Dict]) -> Path:
    _estilo_slides()
    sel = sorted(cenarios, key=lambda c: c["mse"])[:4]
    sel.append(next(c for c in cenarios if c["chave"] == BASELINE_LN_KEY))
    sel = sorted(sel, key=lambda c: c["mse"], reverse=True)  # campeao no topo

    rotulos = [_rotulo_cenario(c) for c in sel]
    mses = [c["mse"] for c in sel]
    dps = [c["dp"] for c in sel]
    cores = [COR_CAMPEAO if c["chave"] == CAMPEAO
             else COR_BASELINE if c["e_baseline_ln"]
             else COR_NEUTRA for c in sel]

    fig, ax = plt.subplots(figsize=(16, 9))
    y = np.arange(len(sel))
    ax.barh(y, mses, xerr=dps, height=0.62, color=cores,
            edgecolor="black", linewidth=1.6, capsize=8, zorder=3,
            error_kw={"elinewidth": 2.4, "capthick": 2.4, "ecolor": "black"})
    for yi, m, dp in zip(y, mses, dps):
        ax.text(m + dp + 0.06, yi, f"{m:.3f}", va="center", fontsize=24,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(rotulos)
    ax.set_xlim(0, 8.7)
    ax.set_ylim(-0.6, len(sel) - 0.4)
    ax.set_xlabel("Test MSE raw (menor = melhor)")

    fig.tight_layout()
    p = _dir_saida() / "slide_02_ranking.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  3 - Os 21 testes na escala raw (linha + error bars)                   #
# --------------------------------------------------------------------- #
def slide_03_escala(cenarios: List[Dict]) -> Path:
    _estilo_slides()
    ordem = sorted(cenarios, key=lambda c: c["mse"])
    xs = np.arange(1, len(ordem) + 1)
    med = np.array([c["mse"] for c in ordem])
    dp = np.array([c["dp"] for c in ordem])
    cores = [COR_CAMPEAO if c["chave"] == CAMPEAO else "#9aa3ad"
             for c in ordem]
    champ = next(c for c in ordem if c["chave"] == CAMPEAO)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(xs, med, "-", color="0.45", linewidth=2.2, alpha=0.55, zorder=2)
    for x, m, d, co in zip(xs, med, dp, cores):
        ax.errorbar([x], [m], yerr=[d], fmt="o", color=co, markersize=14,
                    markerfacecolor=co, markeredgecolor="black",
                    markeredgewidth=1.6, elinewidth=2.4, capsize=5,
                    capthick=2.4, zorder=4)

    ax.annotate(f"campeao: {champ['mse']:.3f} raw",
                xy=(xs[0], champ["mse"]),
                xytext=(3.5, champ["mse"] + 0.5),
                fontsize=20, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="black", lw=1.2))

    ax.set_xlabel("21 cenarios (ranking por Test MSE raw)")
    ax.set_ylabel("Test MSE raw (menor = melhor)")
    ax.set_xticks(range(1, 22, 2))
    ax.set_ylim(6.6, 8.2)

    fig.tight_layout()
    p = _dir_saida() / "slide_03_escala.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  4 - Apples-to-Apples (sem titulo grande; destaque discreto)           #
# --------------------------------------------------------------------- #
def slide_04_veredito(veredito: Dict) -> Path:
    _estilo_slides()
    base_z = float(veredito["baseline"]["medido_z"])
    txt = veredito["campeao_textual"]
    txt_z = float(txt["test_mse_z_media"])
    txt_dp = float(txt["test_mse_z_desvio"])
    delta = float(txt["delta_vs_baseline_pct"])
    seeds = txt["por_seed"]

    rotulos = ["Baseline Numerico", "Campeao Textual"]
    medias = [base_z, txt_z]
    desvios = [0.0, txt_dp]
    cores = [COR_NUM, COR_CAMPEAO]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(range(2), medias, yerr=desvios, width=0.4, color=cores,
           edgecolor="black", linewidth=2.2, capsize=12, zorder=3,
           error_kw={"elinewidth": 3.0, "capthick": 3.0, "ecolor": "black"})
    for i, (m, d) in enumerate(zip(medias, desvios)):
        ax.text(i, m + d + 0.015, f"{m:.4f}" + (f"\n± {d:.4f}" if d else ""),
                ha="center", fontsize=28, fontweight="bold")

    for s in seeds.values():
        ax.scatter(1, s["test_mse_z"], s=260, marker="o", facecolor="white",
                   edgecolor="black", linewidth=2.6, zorder=6)

    ax.set_xticks(range(2))
    ax.set_xticklabels(rotulos)
    ax.set_ylim(0, 0.58)
    ax.set_ylabel("Test MSE z-score (menor = melhor)")

    ax.text(1.0, 0.52, f"{delta:+.2f}%", ha="center", fontsize=42,
            fontweight="bold", color="#c62828")

    fig.tight_layout()
    p = _dir_saida() / "slide_04_veredito.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    estat = _carregar_json("estruturas_avancadas_stats.json")
    veredito = _carregar_json("veredito_final_apples_to_apples.json")
    cenarios = _cenarios(estat)

    print("Paineis de slides (limpos) gerados:")
    for p in (
        slide_01_idiomas(cenarios),
        slide_02_ranking(cenarios),
        slide_03_escala(cenarios),
        slide_04_veredito(veredito),
    ):
        print(f"[ok] {p}")


if __name__ == "__main__":
    main()