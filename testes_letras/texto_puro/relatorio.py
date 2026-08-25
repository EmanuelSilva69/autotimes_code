"""
relatorio - Persistencia, tabelas e graficos do teste de texto puro.

Recebe o dict consolidado (paginacao + inflacao + estabilidade) e:
    1) salva resumo_texto_puro.json;
    2) imprime tabelas comparativas no console;
    3) gera PNGs (paginas por estrategia, tokens por estrategia,
       inflacao PT vs EN, curvas de estabilidade da loss).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .. import caminhos
from .paginacao import ESTRATEGIAS

_IDIOMAS = ["pt", "en"]


def salvar_resumo(resumo: Dict, path: Optional[Path] = None) -> Path:
    path = path or caminhos.RESULTADOS_DIR / "resumo_texto_puro.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    return path


def imprimir_paginacao(pag: Dict) -> None:
    print("\n" + "=" * 100)
    print("PAGINACAO (chunking) - media por documento")
    print("=" * 100)
    hdr = (f"{'Estrategia':<30s} {'Idioma':>6s} {'tokens orig':>10s} "
           f"{'tokens ret':>10s} {'paginas':>8s} {'descarte':>10s} "
           f"{'tempo/doc(s)':>12s}")
    print(hdr)
    print("-" * 100)
    for est, est_data in pag.items():
        for lingua, s in est_data.items():
            print(
                f"{ESTRATEGIAS[est]['nome']:<30s} {lingua:>6s} "
                f"{s['tokens_originais_med']:>10.1f} {s['tokens_retidos_med']:>10.1f} "
                f"{s['paginas_med']:>8.2f} {s['tokens_descartados_med']:>10.1f} "
                f"{s['tempo_tokenizacao_doc_s']:>12.4f}"
            )
    print("=" * 100)


def imprimir_inflacao(inf: Dict) -> None:
    print("\n" + "=" * 100)
    print(f"INFLACAO DE TOKENS PT vs EN (taxa media = x{inf['taxa_media_inflacao']})")
    print("=" * 100)
    print(f"{'Documento':<12s} {'tokens PT':>10s} {'tokens EN':>10s} {'inflacao':>10s}")
    print("-" * 100)
    for l in inf["por_documento"]:
        print(f"{l['documento']:<12s} {l['tokens_pt']:>10d} {l['tokens_en']:>10d} "
              f"x{l['taxa_inflacao']:>9.2f}")
    print(f"TOTAL          {inf['tokens_totais_pt']:>10d} {inf['tokens_totais_en']:>10d} "
          f"x{inf['taxa_media_inflacao']:>9.2f}")
    print("=" * 100)


def imprimir_estabilidade(est: Dict, modelos_ordem: List[str]) -> None:
    print("\n" + "=" * 100)
    print("ESTABILIDADE DA LOSS (texto puro paginado na arquitetura aprimorada)")
    print("=" * 100)
    hdr = (f"{'Modelo/Estrategia':<34s} {'Idioma':>6s} {'best_val':>10s} "
           f"{'test_mse':>10s} {'estouros':>8s} {'ms/batch':>9s}")
    print(hdr)
    print("-" * 100)
    for mk in modelos_ordem:
        for estr in ESTRATEGIAS:
            for lingua in _IDIOMAS:
                chave = f"{mk}__{estr}__{lingua}"
                if chave not in est:
                    continue
                r = est[chave]
                print(
                    f"  {ESTRATEGIAS[estr]['nome'] + ' (' + mk + ')':<32s} {lingua:>6s} "
                    f"{r['melhor_val_mse_pred']:>10.4f} {r['teste_final_mse_pred']:>10.4f} "
                    f"{r['batches_estourados_total']:>8d} "
                    f"{r['tempo_batch_ms_medio']:>9.2f}"
                )
    print("=" * 100)


def gerar_graficos(pag: Dict, inf: Dict, est: Dict, modelos_ordem: List[str]) -> List[Path]:
    """Graficos PNG do teste de texto puro (matplotlib, backend Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arquivos: List[Path] = []
    estrategias = [e for e in ESTRATEGIAS if e in pag]
    labels = [ESTRATEGIAS[e]["nome"] for e in estrategias]

    # 1) Paginas necessarias por estrategia/idioma.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(estrategias))
    largura = 0.35
    for i, lingua in enumerate(_IDIOMAS):
        vals = [pag[e][lingua]["paginas_med"] for e in estrategias]
        ax.bar([v + i * largura for v in x], vals, largura, label=f"idioma {lingua}")
        for v, val in zip(x, vals):
            ax.text(v + i * largura, val + 0.05, f"{val:.1f}", ha="center", fontsize=8)
    ax.set_xticks([v + largura / 2 for v in x])
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Paginas por documento (media)")
    ax.set_title("Paginacao: paginas necessarias para cobrir o documento")
    ax.legend()
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "texto_puro_paginas.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    arquivos.append(p)

    # 2) Tokens originais vs retidos por estrategia (pt).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    orig = [pag[e]["pt"]["tokens_originais_med"] for e in estrategias]
    ret = [pag[e]["pt"]["tokens_retidos_med"] for e in estrategias]
    ax.bar([v - 0.18 for v in x], orig, 0.36, label="tokens originais", alpha=0.7)
    ax.bar([v + 0.18 for v in x], ret, 0.36, label="tokens paginados/retidos", alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Tokens por documento (media)")
    ax.set_title("Compressao: tokens originais vs retidos por estrategia (PT)")
    ax.legend()
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "texto_puro_tokens.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    arquivos.append(p)

    # 3) Inflacao de tokens por documento.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    docs = [l["documento"] for l in inf["por_documento"]]
    taxas = [l["taxa_inflacao"] for l in inf["por_documento"]]
    ax.bar(docs, taxas, color="coral", alpha=0.85)
    ax.axhline(inf["taxa_media_inflacao"], color="navy", ls="--",
               label=f"media x{inf['taxa_media_inflacao']}")
    ax.set_ylabel("tokens PT / tokens EN")
    ax.set_title("Inflacao de tokens do GPT-2: mesmo conteudo em PT vs EN")
    ax.legend()
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "texto_puro_inflacao.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    arquivos.append(p)

    # 4) Curvas de estabilidade (loss de validacao por estrategia/idioma).
    for mk in modelos_ordem:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for est_ in estrategias:
            for lingua in _IDIOMAS:
                chave = f"{mk}__{est_}__{lingua}"
                if chave not in est:
                    continue
                hist = est[chave]["historico"]
                ax.plot(hist["epoch"], hist["val_mse_pred"], marker="o", ms=4,
                        label=f"{ESTRATEGIAS[est_]['nome']} ({lingua})")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("MSE no horizonte (val)")
        ax.set_title(f"Estabilidade da loss com texto puro paginado - {mk}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = caminhos.GRAFICOS_DIR / f"texto_puro_estabilidade_{mk}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        arquivos.append(p)
    return arquivos
