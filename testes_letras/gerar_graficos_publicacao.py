"""
gerar_graficos_publicacao - Storytelling visual (estilo NeurIPS/IEEE) do
estudo de formatacoes estruturadas sobre o ETTh1 (AutoTimes + GPT-2).

Autonomo: consome apenas os JSONs ja gerados pelos benchmarks (sem re-treinar).

FONTES DE DADOS:
    resultados/estruturas_avancadas_stats.json (21 cenarios completos)
    resultados/campeao_absoluto_stats.json      (validacao ouro do top-3)
    resultados/numerico_base_stats.json         (marco inicial da jornada)
    resultados/top3_validacao_stats.json        (marco PT Extenso Compacto)

GRAFICO 1 - formatacoes_21_errorbars.png:
    Barras HORIZONTAIS com todos os 21 cenarios (ranking por Test MSE).
    - xerr = desvio padrao das 3 seeds (error bars PRETAS explicitas);
    - linha vertical tracejada VERMELHA em 7.24 (barreira ES Keywords);
    - cores semanticas: azul = perfurou a barreira (< 7.24), cinza = ficou
      atras, dourado = grande campeao (pseudo_json_es__janela);
    - DPI=300.

GRAFICO 2 - jornada_evolucao_mse.png:
    Evolucao do erro pelos 4 marcos da pesquisa, RECALIBRADA para o espaco
    z-score do artigo (mse_z = mse_raw / var_treino(HUFL)=33.79, fator
    confirmado empiricamente pela auditoria):
      1) Numerico Base  0.3539 z (protocolo oficial, Run-8 reproduzido)
      2) PT Extenso     0.2166 z (harness de exploracao, convertido)
      3) ES Keywords    0.2143 z (idem)
      4) ES Pseudo-JSON 0.2082 z (idem - recorde textual, recalibrado)
    Escala dupla (z-score + equiv. raw HUFL), linha tracejada VERMELHA na
    baseline oficial do artigo (0.3539 z), error bars pretas; DPI=300.

SAIDAS (pasta grafico_erro):
    resultados/graficos/grafico_erro/formatacoes_21_errorbars.png
    resultados/graficos/grafico_erro/jornada_evolucao_mse.png
    resultados/graficos/grafico_erro/dados_figuras.json
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

BARREIRA_LN: float = 7.24          # limiar da baseline ES Keywords
CAMPEAO: str = "pseudo_json_es__janela"

# Recalibracao (ver auditoria_harness.py): o harness antigo media o MSE na
# escala RAW da HUFL, enquanto o artigo mede em z-score global
# (StandardScaler fit no treino). Conversao: mse_z = mse_raw / var_treino.
# var_treino(HUFL) = 33.79 foi confirmado empiricamente (razao 33.79 no
# eval do proprio checkpoint em ambas as escalas).
_VAR_HUFL_DEFAULT: float = 33.7877
BASELINE_OFICIAL_Z: float = 0.3539   # Run-8 VolLoss+SMETimes (artigo)

_COR_CAMPEAO = "#D4AF37"           # dourado
_COR_PERFUROU = "#1f6fd6"          # azul
_COR_ATRAS = "#9ca3af"             # cinza


def _carregar(nome: str) -> Dict:
    p = caminhos.RESULTADOS_DIR / nome
    if not p.exists():
        raise FileNotFoundError(f"JSON de dados nao encontrado: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _dir_saida() -> Path:
    d = caminhos.GRAFICOS_DIR / "grafico_erro"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt(media: float, desvio: float) -> str:
    return f"{media:.2f} ± {desvio:.2f}"


# --------------------------------------------------------------------- #
#  GRAFICO 1 - Ranking das 21 formatacoes (errorbars horizontais)        #
# --------------------------------------------------------------------- #
def _rotulo_cenario(chave: str, c: Dict) -> str:
    """Rotulo legivel para o eixo Y (ex.: 'Pseudo-JSON ES  ·  janela')."""
    if c["tipo"] == "baseline":
        return "Baseline ES Keywords  (controle)"
    return f"{c['template_nome']}  ·  {c['estrategia']}"


def gerar_ranking_21(estat: Dict) -> Path:
    """Barras horizontais ranqueadas (todas as 21) com xerr e linha 7.24."""
    sns.set_theme(style="whitegrid", context="paper",
                  font_scale=1.05, rc={"font.family": "serif"})

    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in estat["cenarios"].items()
    )
    nomes = [_rotulo_cenario(chave, c) for _, chave, c in linhas]
    medias = [c["test_mse_media"] for _, _, c in linhas]
    desvios = [c["test_mse_desvio"] for _, _, c in linhas]
    tipos = [c["tipo"] for _, _, c in linhas]

    def _cor(tipo: str, mse: float, chave: str) -> str:
        if chave == CAMPEAO:
            return _COR_CAMPEAO
        if tipo != "baseline" and mse < BARREIRA_LN:
            return _COR_PERFUROU
        return _COR_ATRAS

    y = np.arange(len(linhas))[::-1]
    cores = [_cor(c["tipo"], c["test_mse_media"], chave)
             for _, chave, c in linhas]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(y, medias, xerr=desvios, height=0.66, color=cores,
            edgecolor="black", linewidth=0.4, capsize=3.0, zorder=3,
            error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "black"})

    ax.axvline(BARREIRA_LN, color="#d62728", ls="--", lw=1.6, zorder=2)
    ax.text(BARREIRA_LN + 0.008, len(linhas) - 0.45,
            f"Baseline ES Keywords = {BARREIRA_LN:.2f}", fontsize=9,
            color="#d62728", va="center", ha="left")

    for yi, mse, desvio, nome in zip(y, medias, desvios, nomes):
        ax.text(mse + desvio + 0.012, yi, _fmt(mse, desvio),
                va="center", ha="left", fontsize=8.2, color="0.15")

    ax.set_yticks(y)
    ax.set_yticklabels(nomes, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Test MSE (media das 3 seeds; menor = melhor)", fontsize=11)
    ax.set_title("Ranking das 21 Formatacoes sobre o ETTh1 (AutoTimes + GPT-2)\n"
                 "Barras horizontais = media ± desvio padrao (10 epocas x 3 seeds)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(6.6, 8.55)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=_COR_CAMPEAO, label="Campeao (pseudo-JSON ES)"),
        plt.Rectangle((0, 0), 1, 1, fc=_COR_PERFUROU, label="Perfurou a barreira (< 7.24)"),
        plt.Rectangle((0, 0), 1, 1, fc=_COR_ATRAS, label="Ficou atras da barreira"),
        plt.Line2D([0], [0], color="#d62728", ls="--", lw=1.6,
                   label=f"Barreira da LN ({BARREIRA_LN:.2f})"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right", frameon=True)
    fig.tight_layout()
    p = _dir_saida() / "formatacoes_21_errorbars.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  GRAFICO 2 - Jornada da evolucao do erro                               #
# --------------------------------------------------------------------- #
def _var_hufl() -> float:
    """Variancia do treino da HUFL (fator de conversao raw->z).

    Tenta o relatorio da auditoria (valor empirico) e cai para o default.
    """
    p = caminhos.RESULTADOS_DIR / "auditoria_harness_report.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                rep = json.load(f)
            var = rep["stats_standardscaler"]["var_treino"]["HUFL"]
            if var and var > 0:
                return float(var)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return _VAR_HUFL_DEFAULT


def _marcos_jornada() -> List[Dict]:
    """4 marcos da jornada, TODOS convertidos para o espaco z-score.

    Marco 1 (Numerico Base): vem do novo benchmark_numerico_base.py que
    replica o protocolo oficial (z-score, dataset completo, split temporal
    disjunto). Se o JSON ainda for o antigo (raw, pool 192), converte por
    var_treino(HUFL) como fallback.

    Marcos 2-4 (textuais): medidos no harness de exploracao (escala raw,
    HUFL, janela curta) - convertidos por var_treino(HUFL). A conversao
    remove apenas o efeito de escala; volume/leakage seguem explicitados
    nas notas do grafico.
    """
    var = _var_hufl()
    num = _carregar("numerico_base_stats.json")["modelos"]["smetimes"]
    top3 = _carregar("top3_validacao_stats.json")["cenarios"]
    campeao = _carregar("campeao_absoluto_stats.json")["cenarios"]

    # Marco 1: preferir o protocolo oficial (z-score); fallback p/ raw.
    if num.get("modo") == "eval_only" and num.get("test_mse_media_z"):
        num_z = float(num["test_mse_media_z"])
        num_dp_z = float(num.get("test_mse_desvio", 0.0))
        num_sub = "SMETimes, sem texto | protocolo oficial (z-score)"
    else:
        num_z = float(num["test_mse_media"]) / var
        num_dp_z = float(num.get("test_mse_desvio", 0.0)) / var
        num_sub = "SMETimes, sem texto | harness antigo (convertido)"
    num_raw = num_z * var

    def _textual(mse_raw: float, dp_raw: float, titulo: str, sub: str) -> Dict:
        return {
            "titulo": titulo,
            "subtitulo": sub,
            "mse_z": mse_raw / var,
            "dp_z": dp_raw / var,
            "mse_raw": mse_raw,
            "dp_raw": dp_raw,
        }

    return [
        {
            "titulo": "Numerico Base",
            "subtitulo": num_sub,
            "mse_z": num_z, "dp_z": num_dp_z,
            "mse_raw": num_raw, "dp_raw": num_dp_z * var,
        },
        _textual(top3["pt_curto"]["test_mse_media"],
                 top3["pt_curto"]["test_mse_desvio"],
                 "PT Extenso Compacto", "pt_curto | janela"),
        _textual(campeao["es_keywords__keywords"]["test_mse_media"],
                 campeao["es_keywords__keywords"]["test_mse_desvio"],
                 "ES Keywords", "Language Tuning"),
        _textual(campeao["pseudo_json_es__janela"]["test_mse_media"],
                 campeao["pseudo_json_es__janela"]["test_mse_desvio"],
                 "ES Pseudo-JSON", "Structural Tuning"),
    ]


def gerar_jornada(marcos: List[Dict], var_hufl: float) -> Path:
    """Line/step plot dos 4 marcos com escala dupla (z-score + raw).

    Eixo esquerdo: Test MSE z-score (mesma unidade do artigo).
    Eixo direito:  escala raw da HUFL (equivalente por var_treino).
    Linha tracejada VERMELHA = baseline oficial do artigo (0.3539 z).
    """
    sns.set_theme(style="whitegrid", context="paper",
                  font_scale=1.0, rc={"font.family": "serif"})

    x = list(range(len(marcos)))
    medias = [m["mse_z"] for m in marcos]
    desvios = [m["dp_z"] for m in marcos]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.errorbar(x, medias, yerr=desvios, fmt="o-", color="#1f77b4",
                ecolor="black", elinewidth=1.6, capsize=5, capthick=1.6,
                linewidth=2.0, markersize=8, zorder=4,
                label="Test MSE z-score (media das 3 seeds)")

    cores_marca = ["#6c757d", "#ff7f0e", "#d62728", "#2ca02c"]
    for xi, marco, cor in zip(x, marcos, cores_marca):
        media_z, desvio_z = marco["mse_z"], marco["dp_z"]
        ax.scatter([xi], [media_z], s=110, color=cor, zorder=5,
                   edgecolor="black", linewidth=0.6)
        ax.annotate(f"{media_z:.3f}\n± {desvio_z:.3f}",
                    (xi, media_z + desvio_z + 0.06),
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.annotate(marco["subtitulo"], (xi, media_z - desvio_z - 0.09),
                    ha="center", va="top", fontsize=7.8, color="0.25")

    # Baseline oficial do artigo (0.3539 z, Run-8 VolLoss+SMETimes).
    ax.axhline(BASELINE_OFICIAL_Z, color="#d62728", ls="--", lw=1.6,
               alpha=0.9, zorder=2)
    ax.text(0.02, BASELINE_OFICIAL_Z + 0.012,
            f"Baseline oficial (artigo) = {BASELINE_OFICIAL_Z:.4f} z",
            fontsize=8, color="#d62728", va="bottom", transform=ax.get_yaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([m["titulo"] for m in marcos], fontsize=10)
    ax.set_xlim(-0.45, 3.55)
    ax.set_ylim(0.0, max(medias) * 1.35 + max(desvios))
    ax.set_ylabel("Test MSE (escala z-score; menor = melhor)", fontsize=10)

    # Eixo secundario: escala raw da HUFL (mse_raw = mse_z * var).
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] * var_hufl, ax.get_ylim()[1] * var_hufl)
    ax2.set_ylabel(f"equiv. escala raw HUFL (x {var_hufl:.1f})", fontsize=9,
                   color="0.45")
    ax2.tick_params(colors="0.45")

    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p = _dir_saida() / "jornada_evolucao_mse.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  GRAFICO 3 - Convergencia dos 21 cenarios c/ banda de confianca        #
# --------------------------------------------------------------------- #
_CORES_TIPO_21 = {
    "estrutural": "#17becf",
    "pseudo_json": "#9467bd",
    "relacional": "#ff7f0e",
    "temporal_delta": "#d62728",
    "baseline": "#7f7f7f",
}


def gerar_convergencia_21(estat: Dict) -> Path:
    """Test MSE por epoca (media das 3 seeds) + banda media ± 1 dp (fill_between).

    Todas as 21 formatacoes: uma linha solida com marcadores por cenario e
    area sombreada semi-transparente (alpha=0.18) representando a variancia
    entre seeds. Cores agrupadas por tipo de template; campeao em dourado.
    """
    sns.set_theme(style="whitegrid", context="paper",
                  font_scale=1.0, rc={"font.family": "serif"})

    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in estat["cenarios"].items()
    )
    n_ep = max((len(c["test_mse_media_por_epoca"]) for _, _, c in linhas),
               default=0)

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, (mse, chave, c) in enumerate(linhas):
        eps = list(range(1, len(c["test_mse_media_por_epoca"]) + 1))
        y_med = c["test_mse_media_por_epoca"]
        y_dp = c["test_mse_desvio_por_epoca"]
        if chave == CAMPEAO:
            cor = _COR_CAMPEAO
            lw, mk, alpha = 2.4, 5.5, 0.22
        else:
            cor = _CORES_TIPO_21.get(c["tipo"], f"C{i % 10}")
            lw, mk, alpha = 1.3, 2.8, 0.13
        ax.plot(eps, y_med, "o-", color=cor, linewidth=lw, markersize=mk,
                zorder=4, label=f"{_rotulo_cenario(chave, c)} "
                               f"(Test MSE {mse:.2f} ± "
                               f"{c['test_mse_desvio']:.2f})")
        ax.fill_between(eps, np.array(y_med) - np.array(y_dp),
                        np.array(y_med) + np.array(y_dp),
                        color=cor, alpha=alpha, linewidth=0, zorder=2)

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("Test MSE (media das 3 seeds)", fontsize=11)
    ax.set_title("Test MSE por Epoca das 21 Formatacoes\n"
                 "Linha = media | banda sombreada = media ± 1 desvio padrao",
                 fontsize=12.5, fontweight="bold")
    ax.set_xticks(range(1, n_ep + 1))
    ax.set_xlim(0.8, n_ep + 0.2)
    ax.set_ylim(6.4, 11.2)
    ax.grid(axis="both", alpha=0.25, zorder=0)
    ax.legend(fontsize=7.5, ncols=2, loc="upper left", frameon=True,
              handlelength=1.6, columnspacing=0.9)
    fig.tight_layout()
    p = _dir_saida() / "convergencia_21_errorband.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    caminhos.criar_resultados()
    estat = _carregar("estruturas_avancadas_stats.json")
    var_hufl = _var_hufl()

    marcos = _marcos_jornada()
    print("=" * 90)
    print("JORNADA DA EVOLUCAO DO ERRO (escala z-score, var_hufl=%.2f)" % var_hufl)
    print("=" * 90)
    for m in marcos:
        print(f"  {m['titulo']:<24s} {m['subtitulo']:<46s} "
              f"{_fmt(m['mse_z'], m['dp_z'])} z | "
              f"{_fmt(m['mse_raw'], m['dp_raw'])} raw")

    dados: Dict = {
        "ranking_21": {
            chave: {
                "nome": _rotulo_cenario(chave, c),
                "tipo": c["tipo"],
                "estrategia": c["estrategia"],
                "mse": c["test_mse_media"],
                "dp": c["test_mse_desvio"],
                "mae": c["test_mae_media"],
            }
            for chave, c in estat["cenarios"].items()
        },
        "barreira_ln": BARREIRA_LN,
        "campeao": CAMPEAO,
        "baseline_oficial_z": BASELINE_OFICIAL_Z,
        "var_treino_hufl": var_hufl,
        "nota_recalibracao": (
            "Marcos textuais convertidos de raw->z por var_treino(HUFL); "
            "remove apenas o efeito de escala (volume/leakage seguem "
            "limitados ao harness de exploracao)."
        ),
        "jornada": marcos,
    }

    p1 = gerar_ranking_21(estat)
    p2 = gerar_jornada(marcos, var_hufl)
    p3 = gerar_convergencia_21(estat)

    out = _dir_saida() / "dados_figuras.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] ranking 21: {p1}")
    print(f"[ok] jornada:    {p2}")
    print(f"[ok] convergencia 21: {p3}")
    print(f"[ok] dados:      {out}")


if __name__ == "__main__":
    main()