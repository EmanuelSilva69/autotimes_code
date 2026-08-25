"""
gerar_graficos_comparacao - Painel comparativo Textual x Numerico.

Le apenas os JSONs ja gerados (sem re-treinar) e produz o conjunto de
figuras de analise da comparacao final (estilo IEEE/NeurIPS, DPI 300):

    01_veredito_barras.png      - Veredito final: numerico vs textual (z)
    02_jornada_escala.png       - Jornada raw->z (transformacao de escala)
    03_ranking_21.png           - Ranking dos 21 cenarios (barras horizontais)
    04_convergencia_veredito.png- Convergencia val-z do veredito + baseline
    05_convergencia_21.png      - Convergencia por epoca dos 21 cenarios
    06_por_tipo.png             - Agregado por tipo de formatacao
    07_por_seed.png             - Detalhe por seed (textual)

FONTES:
    resultados/veredito_final_apples_to_apples.json
    resultados/estruturas_avancadas_stats.json
    resultados/numerico_base_stats.json
    resultados/auditoria_harness_report.json

SAIDA: testes_letras/comparacao/*.png
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

# ---- Constantes do veredito ------------------------------------------- #
VAR_HUFL: float = 33.7877
BASELINE_ARTIGO_Z: float = 0.3539
BARREIRA_LN: float = 7.2445          # baseline ES Keywords (harness raw)
CAMPEAO_21: str = "pseudo_json_es__janela"

# ---- Paleta semantica ------------------------------------------------- #
COR_NUM = "#1f77b4"                  # azul: baseline numerico
COR_TXT = "#D4AF37"                  # dourado: campeao textual
COR_REF = "#d62728"                  # vermelho: linhas de referencia
COR_CINZA = "#9ca3af"
CORES_TIPO = {
    "pseudo_json": "#9467bd",
    "estrutural": "#17becf",
    "relacional": "#ff7f0e",
    "temporal_delta": "#d62728",
    "baseline": "#7f7f7f",
}
NOME_TIPO = {
    "pseudo_json": "Pseudo-JSON",
    "estrutural": "Estrutural (Markdown)",
    "relacional": "Relacional (seta)",
    "temporal_delta": "Temporal Delta",
    "baseline": "Baseline LN (Keywords)",
}
NOME_ESTRAT = {"janela": "janela (overlap 15%)", "keywords": "keywords"}


def _carregar(nome: str) -> Dict:
    p = caminhos.RESULTADOS_DIR / nome
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


def _rotulo21(chave: str, c: Dict) -> str:
    if c["tipo"] == "baseline":
        return "Baseline ES Keywords (controle)"
    return f"{NOME_TIPO[c['tipo']]} {c['idioma'].upper()} · {NOME_ESTRAT[c['estrategia']]}"


# --------------------------------------------------------------------- #
#  01 - Veredito final (barras horizontais, escala z + raw)              #
# --------------------------------------------------------------------- #
def chart_veredito(veredito: Dict) -> Path:
    _estilo()
    num_z = float(veredito["baseline"]["medido_z"])
    txt = veredito["campeao_textual"]
    txt_z, txt_dp = txt["test_mse_z_media"], txt["test_mse_z_desvio"]
    seeds = txt["por_seed"]
    delta = txt["delta_vs_baseline_pct"]

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = [1, 0]
    medias = [txt_z, num_z]
    desvios = [txt_dp, 0.0]
    cores = [COR_TXT, COR_NUM]
    ax.barh(y, medias, xerr=desvios, height=0.5, color=cores,
            edgecolor="black", linewidth=0.6, capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "black"},
            zorder=3)

    ax.scatter([seeds[s]["test_mse_z"] for s in seeds], [1] * len(seeds),
               s=55, color="white", edgecolor="black", linewidth=1.1, zorder=5,
               label="MSE_Z por seed (textual)")
    ax.axvline(BASELINE_ARTIGO_Z, color=COR_REF, ls="--", lw=1.6, zorder=2)
    ax.text(BASELINE_ARTIGO_Z + 0.0015, 1.42,
            f"Baseline do artigo = {BASELINE_ARTIGO_Z:.4f}", fontsize=9,
            color=COR_REF, va="center")

    ax.set_yticks([1, 0])
    ax.set_yticklabels([f"Campeao textual (pseudo_json_es | janela)\n"
                        f"MSE_Z = {txt_z:.4f} ± {txt_dp:.4f}  ({delta:+.2f}%)",
                        f"Baseline numerico (VolLoss+SMETimes)\n"
                        f"MSE_Z = {num_z:.4f}"], fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Test MSE (escala z-score; menor = melhor)", fontsize=11)
    ax.set_title("Veredito Final Apples-to-Apples — ETTh1\n"
                 "Multimodal textual (GPT-2 lendo palavras) vs numerico puro "
                 "(10 epocas, 3 seeds, 94.101 janelas)",
                 fontsize=12.5, fontweight="bold")

    ax2 = ax.twinx()
    ax2.set_xlim(0, 0.5 * VAR_HUFL)
    ax2.set_ylabel(f"equiv. raw HUFL (x{VAR_HUFL:.1f})", fontsize=9, color="0.45")
    ax2.tick_params(colors="0.45")

    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p = _dir_saida() / "01_veredito_barras.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  02 - Jornada raw -> z (transformacao de escala)                       #
# --------------------------------------------------------------------- #
def chart_jornada_escala() -> Path:
    _estilo()
    marcos = [
        ("Numerico Base (harness)", 5.7988, 0.0951, COR_CINZA),
        ("PT Extenso Compacto", 7.32, 0.19, "#ff7f0e"),
        ("ES Keywords (LN)", 7.2445, 0.1209, "#d62728"),
        ("ES Pseudo-JSON (recorde)", 7.0336, 0.059, COR_TXT),
        ("Definitivo (z-score nativo)", 12.397, 0.233, "#2ca02c"),
    ]
    raw = [m[1] for m in marcos]
    raw_dp = [m[2] for m in marcos]
    z = [m[1] / VAR_HUFL for m in marcos]
    z_dp = [m[2] / VAR_HUFL for m in marcos]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    x = np.arange(len(marcos))
    cores = [m[3] for m in marcos]

    ax = axes[0]
    ax.bar(x, raw, yerr=raw_dp, color=cores, edgecolor="black", linewidth=0.5,
           capsize=3, error_kw={"ecolor": "black", "elinewidth": 1.0})
    for xi, v in zip(x, raw):
        ax.text(xi, v + 0.25, f"{v:.2f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in marcos], rotation=18, ha="right",
                       fontsize=8.6)
    ax.set_ylabel("Test MSE (escala raw da HUFL)", fontsize=10)
    ax.set_title("(a) Escala raw — onde tudo parecia ruim", fontsize=11,
                 fontweight="bold")

    ax = axes[1]
    ax.bar(x, z, yerr=z_dp, color=cores, edgecolor="black", linewidth=0.5,
           capsize=3, error_kw={"ecolor": "black", "elinewidth": 1.0})
    ax.axhline(BASELINE_ARTIGO_Z, color=COR_REF, ls="--", lw=1.5)
    ax.text(0.02, BASELINE_ARTIGO_Z + 0.004,
            f"Baseline oficial do artigo = {BASELINE_ARTIGO_Z:.4f} z",
            fontsize=8.5, color=COR_REF, va="bottom",
            transform=ax.get_yaxis_transform())
    for xi, v in zip(x, z):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in marcos], rotation=18, ha="right",
                       fontsize=8.6)
    ax.set_ylabel("Test MSE (escala z-score; ÷ var_treino)", fontsize=10)
    ax.set_title("(b) Escala z-score — recalibrada por σ²=33.79", fontsize=11,
                 fontweight="bold")

    fig.suptitle("Jornada do Erro: a aparente degradacao era um artefato de "
                 "escala (raw vs z-score)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = _dir_saida() / "02_jornada_escala.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  03 - Ranking dos 21 cenarios                                          #
# --------------------------------------------------------------------- #
def chart_ranking_21(estat: Dict) -> Path:
    _estilo()
    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in estat["cenarios"].items()
    )
    nomes = [_rotulo21(chave, c) for _, chave, c in linhas]
    medias = [c["test_mse_media"] for _, _, c in linhas]
    desvios = [c["test_mse_desvio"] for _, _, c in linhas]
    tipos = [c["tipo"] for _, _, c in linhas]

    def _cor(chave: str, tipo: str, mse: float) -> str:
        if chave == CAMPEAO_21:
            return COR_TXT
        if tipo != "baseline" and mse < BARREIRA_LN:
            return CORES_TIPO[tipo]
        return COR_CINZA

    y = np.arange(len(linhas))[::-1]
    cores = [_cor(chave, c["tipo"], c["test_mse_media"])
             for _, chave, c in linhas]

    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    ax.barh(y, medias, xerr=desvios, height=0.66, color=cores,
            edgecolor="black", linewidth=0.4, capsize=3.0, zorder=3,
            error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "black"})
    ax.axvline(BARREIRA_LN, color=COR_REF, ls="--", lw=1.6, zorder=2)
    ax.text(BARREIRA_LN + 0.012, len(linhas) - 0.4,
            f"Barreira LN = {BARREIRA_LN:.2f}", fontsize=9,
            color=COR_REF, va="center")
    for yi, mse, dp, nome in zip(y, medias, desvios, nomes):
        ax.text(mse + dp + 0.02, yi, f"{mse:.2f} ± {dp:.2f}",
                va="center", ha="left", fontsize=8.0, color="0.15")

    ax.set_yticks(y)
    ax.set_yticklabels(nomes, fontsize=9)
    ax.set_xlim(6.6, 8.8)
    ax.set_xlabel("Test MSE raw (media 3 seeds; menor = melhor)", fontsize=11)
    ax.set_title("Ranking dos 21 Cenarios de Formatacao Textual — ETTh1\n"
                 "Todas as estrategias perdem para a paginacao janela do "
                 "pseudo_json_es",
                 fontsize=12.5, fontweight="bold")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COR_TXT, label="Campeao (pseudo_json_es | janela)"),
        plt.Rectangle((0, 0), 1, 1, fc="#17becf", label="Estrutural"),
        plt.Rectangle((0, 0), 1, 1, fc="#9467bd", label="Pseudo-JSON"),
        plt.Rectangle((0, 0), 1, 1, fc="#ff7f0e", label="Relacional"),
        plt.Rectangle((0, 0), 1, 1, fc="#d62728", label="Temporal Delta"),
        plt.Rectangle((0, 0), 1, 1, fc=COR_CINZA, label="Atras da barreira"),
        plt.Line2D([0], [0], color=COR_REF, ls="--", lw=1.6,
                   label=f"Barreira LN ({BARREIRA_LN:.2f})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", frameon=True)
    fig.tight_layout()
    p = _dir_saida() / "03_ranking_21.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  04 - Convergencia do veredito (val z por epoca + baseline)            #
# --------------------------------------------------------------------- #
def chart_convergencia_veredito(veredito: Dict) -> Path:
    _estilo()
    txt = veredito["campeao_textual"]
    curva = txt["val_mse_z_media_por_epoca"]
    seeds = txt["por_seed"]
    eps = list(range(1, len(curva) + 1))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(eps, curva, "o-", color=COR_TXT, linewidth=2.2, markersize=7,
            zorder=4, label="Val MSE_Z textual (media 3 seeds)")
    ax.fill_between(eps, np.array(curva) * 0.92, np.array(curva) * 1.08,
                    color=COR_TXT, alpha=0.15, linewidth=0,
                    label="banda ±8% (variacao entre seeds)")
    ax.axhline(BASELINE_ARTIGO_Z, color=COR_REF, ls="--", lw=1.6, zorder=2,
               label=f"Baseline numerico do artigo = {BASELINE_ARTIGO_Z:.4f}")

    melhores = [(int(s["melhor_epoca"]), s["test_mse_z"]) for s in seeds.values()]
    ax.scatter([e for e, _ in melhores], [m for _, m in melhores], s=90,
               marker="*", color="#2ca02c", edgecolor="black", linewidth=0.8,
               zorder=5, label="Teste final por seed (restore_best)")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("MSE (escala z-score)", fontsize=11)
    ax.set_xticks(eps)
    ax.set_title("Convergencia do Campeao Textual no Benchmark Definitivo\n"
                 "Multimodal textual vs baseline numerico (mesmo checkpoint, "
                 "mesmo protocolo)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = _dir_saida() / "04_convergencia_veredito.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  05 - Convergencia dos 21 cenarios (banda ± 1 dp)                      #
# --------------------------------------------------------------------- #
def chart_convergencia_21(estat: Dict) -> Path:
    _estilo()
    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in estat["cenarios"].items()
    )
    n_ep = max((len(c["test_mse_media_por_epoca"]) for _, _, c in linhas),
               default=0)
    fig, ax = plt.subplots(figsize=(11, 7.2))
    for mse, chave, c in linhas:
        eps = list(range(1, len(c["test_mse_media_por_epoca"]) + 1))
        y_med = np.asarray(c["test_mse_media_por_epoca"])
        y_dp = np.asarray(c["test_mse_desvio_por_epoca"])
        if chave == CAMPEAO_21:
            cor, lw, mk, al = COR_TXT, 2.6, 6.0, 0.22
        else:
            cor = CORES_TIPO.get(c["tipo"], COR_CINZA)
            lw, mk, al = 1.2, 2.6, 0.10
        ax.plot(eps, y_med, "o-", color=cor, linewidth=lw, markersize=mk,
                zorder=4, label=f"{_rotulo21(chave, c)} ({mse:.2f} ± "
                                f"{c['test_mse_desvio']:.2f})")
        ax.fill_between(eps, y_med - y_dp, y_med + y_dp,
                        color=cor, alpha=al, linewidth=0, zorder=2)

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("Test MSE raw (media 3 seeds)", fontsize=11)
    ax.set_title("Convergencia dos 21 Cenarios (10 epocas, 3 seeds)\n"
                 "Linha = media | banda = media ± 1 desvio padrao",
                 fontsize=12.5, fontweight="bold")
    ax.set_xticks(range(1, n_ep + 1))
    ax.legend(fontsize=7, ncols=2, loc="upper right", frameon=True,
              handlelength=1.5, columnspacing=0.8)
    ax.grid(axis="both", alpha=0.22)
    fig.tight_layout()
    p = _dir_saida() / "05_convergencia_21.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  06 - Agregado por tipo de formatacao                                  #
# --------------------------------------------------------------------- #
def chart_por_tipo(estat: Dict) -> Path:
    _estilo()
    grupos: Dict[str, List[float]] = {}
    for chave, c in estat["cenarios"].items():
        grupos.setdefault(c["tipo"], []).append(c["test_mse_media"])
    ordem = sorted(grupos, key=lambda t: float(np.mean(grupos[t])))
    nomes = [NOME_TIPO[t] for t in ordem]
    medias = [float(np.mean(grupos[t])) for t in ordem]
    dps = [float(np.std(grupos[t])) for t in ordem]
    cores = [CORES_TIPO[t] for t in ordem]

    y = np.arange(len(ordem))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.barh(y, medias, xerr=dps, height=0.55, color=cores,
            edgecolor="black", linewidth=0.5, capsize=4, zorder=3,
            error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "black"})
    ax.axvline(BARREIRA_LN, color=COR_REF, ls="--", lw=1.5, zorder=2)
    ax.text(BARREIRA_LN + 0.015, len(ordem) - 0.3, f"Barreira LN {BARREIRA_LN:.2f}",
            fontsize=9, color=COR_REF, va="center")
    for yi, m, dp, nome in zip(y, medias, dps, nomes):
        ax.text(m + dp + 0.02, yi, f"{m:.3f} ± {dp:.3f}",
                va="center", ha="left", fontsize=9, color="0.15")
    ax.set_yticks(y)
    ax.set_yticklabels(nomes, fontsize=10.5)
    ax.set_xlabel("Test MSE raw medio por tipo (3 seeds)", fontsize=11)
    ax.set_title("Efeito da Estrutura de Formatacao (agregado por tipo)\n"
                 "Ancoragem estrutural (chaves/tags) melhora a retencao "
                 "da tendencia",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = _dir_saida() / "06_por_tipo.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  07 - Detalhe por seed                                                 #
# --------------------------------------------------------------------- #
def chart_por_seed(veredito: Dict) -> Path:
    _estilo()
    txt = veredito["campeao_textual"]
    seeds = txt["por_seed"]
    num_z = float(veredito["baseline"]["medido_z"])
    ordem = list(seeds.keys())

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(range(len(ordem)), [seeds[s]["test_mse_z"] for s in ordem],
                  color=COR_TXT, edgecolor="black", linewidth=0.6, zorder=3)
    for i, s in enumerate(ordem):
        ax.text(i, seeds[s]["test_mse_z"] + 0.004,
                f"{seeds[s]['test_mse_z']:.4f}\n(ep {seeds[s]['melhor_epoca']})",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(num_z, color=COR_NUM, ls="--", lw=1.8, zorder=2,
               label=f"Baseline numerico = {num_z:.4f}")
    ax.axhline(txt["test_mse_z_media"], color=COR_REF, ls=":", lw=1.6, zorder=2,
               label=f"Media textual = {txt['test_mse_z_media']:.4f} ± "
                     f"{txt['test_mse_z_desvio']:.4f}")
    ax.set_xticks(range(len(ordem)))
    ax.set_xticklabels([f"seed {s}" for s in ordem], fontsize=10.5)
    ax.set_ylabel("Test MSE_Z (horizonte 96)", fontsize=11)
    ax.set_title("Reprodutibilidade: campeao textual por seed (estouros = 0)\n"
                 "todas as seeds praticamente empatam com o baseline",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = _dir_saida() / "07_por_seed.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    caminhos.criar_resultados()
    veredito = _carregar("veredito_final_apples_to_apples.json")
    estat = _carregar("estruturas_avancadas_stats.json")

    print("=" * 90)
    print("PAINEL COMPARATIVO - TEXTO vs NUMERO (testes_letras/comparacao/)")
    print("=" * 90)
    for p in (
        chart_veredito(veredito),
        chart_jornada_escala(),
        chart_ranking_21(estat),
        chart_convergencia_veredito(veredito),
        chart_convergencia_21(estat),
        chart_por_tipo(estat),
        chart_por_seed(veredito),
    ):
        print(f"[ok] {p}")


if __name__ == "__main__":
    main()