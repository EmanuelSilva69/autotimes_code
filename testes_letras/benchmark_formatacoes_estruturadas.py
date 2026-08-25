"""
benchmark_formatacoes_estruturadas - cruzamento massivo de formatações
estruturadas (delimitadores) x estrategias de paginacao, sobre o campeao
ES Keywords (Test MSE 7.24 +/- 0.12, VolLoss + SMETimes).

MOTIVACAO:
O GPT-2 foi extensamente pre-treinado em codigo (HTML, JSON) e Markdown.
A hipotese: formatações que misturam linguagem natural por extenso com
delimitadores estruturais EXPLICITOS ([ ] | -> chave/valor) ancoram melhor
a atencao causal, isolando a "chave temporal" do "valor matematico" do que
uma frase continua. Este benchmark esgota a via Data-Centric (prompt).

CENARIOS (10 templates estruturados x 2 estrategias = 20 + baseline):
    templates (novo conversor texto_puro/estruturado.py):
        estrutural_en/pt/es       [ Time : ... | Val : ... ]
        pseudo_json_en/pt/es      key time ... key value ...
        relacional_en/pt/es       July ten fourteen hours -> value
        temporal_delta_en         Start July ten. T fourteen: value.
    estrategias (as 2 melhores): janela (overlap) e keywords (stopwords)
    baseline: es_keywords | keywords (7.24 a bater, reexecutado no mesmo
    harness para comparacao justa e visualizacao da barreira).

RIGOR (identico aos benchmarks de publicacao):
    - 3 seeds (2021, 42, 1234) x 10 epocas por cenario;
    - isolamento estrito: seed_everything + montar_modelo DENTRO do loop
      mais interno (a cada iteracao seed/cenario) + empty_cache por ciclo;
    - agregacao: media +/- desvio padrao (MSE e MAE) por cenario.

SAIDAS:
    resultados/estruturas_avancadas_stats.json
    resultados/graficos/formatacoes_estruturadas_ranking.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .modelos import CKPTS, montar_modelo
from .run_etth1_texto import (
    _FEAT_PADRAO,
    carregar_etth1,
    construir_dataset,
    pool_janelas_etth1,
    salvar_benchmark,
)
from .texto_puro import estruturado, extenso
from .texto_puro.paginacao import ESTRATEGIAS, TestadorPaginacao
from .treino import build_loaders, rodar_convergencia

# Baseline a bater (mesmo cenario do campeao publicado).
BASELINE_TEMPLATE: str = "es_keywords"
BASELINE_ESTRATEGIA: str = "keywords"

# As 2 melhores estrategias de paginacao (janela c/ overlap + keywords).
ESTRATEGIAS_A_TESTAR: Tuple[str, ...] = ("janela", "keywords")

_CORES_TIPO: Dict[str, str] = {
    "estrutural": "#17becf",
    "pseudo_json": "#9467bd",
    "relacional": "#ff7f0e",
    "temporal_delta": "#d62728",
    "baseline": "#7f7f7f",
}


def _templates_e_estrategias() -> List[Tuple[str, str]]:
    """Todos os cruzamentos: (template_id, estrategia)."""
    return [
        (tid, est)
        for tid in estruturado.IDS_TEMPLATES_ESTRUTURADOS
        for est in ESTRATEGIAS_A_TESTAR
    ]


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark de formatacoes estruturadas (10 templates x "
                    "2 estrategias) vs baseline ES Keywords 7.24"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="2021,42,1234")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=128)
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int, default=7)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--linhas", type=int, default=2000)
    parser.add_argument("--pool-janelas", dest="pool_janelas", type=int, default=192)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=None)
    parser.add_argument("--cenarios", type=str, default=None,
                        help="Filtro: 'estrutural,pseudo_json,relacional,"
                             "temporal_delta,baseline' (ex.: 'baseline,"
                             "relacional' para testes rapidos)")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Pipeline de dados: documento estruturado + paginacao                  #
# --------------------------------------------------------------------- #
def _gerar_documento(tid: str, dados: Dict) -> str:
    """Documento completo (2000 linhas) no template estruturado ou baseline."""
    if tid == BASELINE_TEMPLATE:
        return extenso.serie_para_texto_extenso(
            dados["dt_list"], dados["values"], _FEAT_PADRAO,
            0, dados["n_linhas"], tid,
        )
    return estruturado.serie_para_texto_estruturado(
        dados["dt_list"], dados["values"], _FEAT_PADRAO,
        0, dados["n_linhas"], tid,
    )


def _idioma_template(tid: str) -> str:
    if tid == BASELINE_TEMPLATE:
        return extenso.TEMPLATES_EXTENSO[tid]["idioma"]
    return estruturado.TEMPLATES_ESTRUTURADOS[tid]["idioma"]


def preparar_documentos(tokenizer, args, cenarios: List[Tuple[str, str]]) -> Dict:
    """Gera/pagina o documento de cada cenario (10 templates + baseline)."""
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][_FEAT_PADRAO]
    print(f"[csv] {dados_etth1['csv']} | {dados_etth1['n_linhas_total']} linhas "
          f"-> {dados_etth1['n_linhas']} linhas (coluna '{feat_name}')")

    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    paginados: Dict[str, Dict] = {}
    for tid, est in cenarios:
        doc = _gerar_documento(tid, dados_etth1)
        lingua = _idioma_template(tid)
        n_tok = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        r = paginador.paginar(est, doc, lingua)
        r.update(paginador.para_modelo(r))
        chave = f"{tid}__{est}"
        paginados[chave] = r
        print(f"  [{tid} | {est}] {n_tok} tokens (orig) -> "
              f"{r['tokens_retidos']} (ret) | {r['n_paginas']} paginas | "
              f"{r['paginas_descartadas']} descartadas")
    return {"dados": dados_etth1, "paginados": paginados}


# --------------------------------------------------------------------- #
#  Rotulos para relatorio e grafico                                      #
# --------------------------------------------------------------------- #
def _rotulo_curto(tid: str) -> str:
    if tid == BASELINE_TEMPLATE:
        return "Baseline ES Keywords"
    t = estruturado.TEMPLATES_ESTRUTURADOS[tid]
    return f"{t['nome']}"


def _tipo(tid: str) -> str:
    if tid == BASELINE_TEMPLATE:
        return "baseline"
    return estruturado.TEMPLATES_ESTRUTURADOS[tid]["tipo"]


# --------------------------------------------------------------------- #
#  Relatorio final                                                       #
# --------------------------------------------------------------------- #
def _fmt(media: float, desvio: float) -> str:
    return f"{media:.2f} ± {desvio:.2f}"


def imprimir_tabela(resumo: Dict) -> None:
    print("\n" + "=" * 100)
    print("RANKING DE FORMATACOES ESTRUTURADAS (10 epocas x 3 seeds)")
    print("=" * 100)
    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in resumo["cenarios"].items()
    )
    print(f"{'#':>2s} {'Cenario':<14s} {'Estrategia':<10s} {'Test MSE':>16s} "
          f"{'Test MAE':>16s} {'best_val':>14s} {'estouros':>8s}")
    print("-" * 100)
    for i, (mse, chave, c) in enumerate(linhas, 1):
        print(f"{i:>2d} {chave:<14s} {c['estrategia']:<10s} "
              f"{_fmt(mse, c['test_mse_desvio']):>16s} "
              f"{_fmt(c['test_mae_media'], c['test_mae_desvio']):>16s} "
              f"{_fmt(c['melhor_val_mse_media'], c['melhor_val_mse_desvio']):>14s} "
              f"{c['estouros_total']:>8d}")
    print("-" * 100)
    print("Formato: media ± desvio padrao sobre as 3 seeds (menor MSE vence).")
    print("=" * 100)


def salvar_resumo(resumo: Dict) -> Path:
    return salvar_benchmark(resumo, nome="estruturas_avancadas_stats.json")


def gerar_grafico(resumo: Dict, baseline_mse: float) -> Path:
    """Barras ranqueadas (MSE medio + error bars) contra a linha baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in resumo["cenarios"].items()
    )
    nomes = [f"{chave}" for _, chave, _ in linhas]
    medias = [c["test_mse_media"] for _, _, c in linhas]
    desvios = [c["test_mse_desvio"] for _, _, c in linhas]
    tipos = [_tipo(chave.split("__")[0]) for _, chave, _ in linhas]
    cores = [_CORES_TIPO[t] for t in tipos]

    fig, ax = plt.subplots(figsize=(16, 8))
    barras = ax.bar(nomes, medias, yerr=desvios, capsize=3.5, width=0.7,
                    color=cores, edgecolor="black", linewidth=0.5, zorder=3)
    ax.axhline(baseline_mse, color="#7f7f7f", ls="--", lw=1.8, zorder=1,
               label=f"Baseline ES Keywords = {baseline_mse:.2f}")
    for b, m, d in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, m + d + 0.05,
                f"{m:.2f}\n±{d:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(baseline_mse - 0.2, color="#7f7f7f", lw=0.6, alpha=0.0)  # respiro
    ax.set_ylim(min(medias) - 1.0, max(m + d for m, d in zip(medias, desvios)) + 1.1)
    ax.set_ylabel("Test MSE (media das 3 seeds)")
    ax.set_xlabel("Cenario: template__estrategia (menor = melhor)")
    ax.set_title("Formatacoes Estruturadas vs Baseline ES Keywords 7.24\n"
                 "(barras = media ± dp sobre 3 seeds; linha tracejada = barreira a bater)")
    handles = [plt.Line2D([0], [0], color="#7f7f7f", ls="--", lw=1.8,
                          label=f"Baseline ES Keywords = {baseline_mse:.2f}")]
    for t in ("estrutural", "pseudo_json", "relacional", "temporal_delta"):
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=_CORES_TIPO[t],
                                     label=t))
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "formatacoes_estruturadas_ranking.png"
    fig.savefig(p, dpi=115)
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    caminhos.criar_resultados()
    rb = caminhos.importar_run_benchmarks()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Nenhuma seed valida em --seeds")
    rb.seed_everything(seeds[0])

    device = rb.get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")

    # ---- Tokenizador -------------------------------------------------- #
    try:
        if args.tokenizer_path and (Path(args.tokenizer_path) / "tokenizer.json").exists():
            tok_dir = Path(args.tokenizer_path)
        else:
            tok_dir = caminhos.descobrir_tokenizador_recursivo(caminhos.ROOT_DIR)
    except FileNotFoundError as e:
        print(f"[erro] tokenizador nao encontrado: {e}")
        raise SystemExit(1)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,})")

    # ---- Cenarios (filtro opcional) ----------------------------------- #
    cenarios = _templates_e_estrategias()
    if args.cenarios:
        pedidos = {p.strip() for p in args.cenarios.split(",") if p.strip()}
        if "baseline" in pedidos:
            cenarios = [(BASELINE_TEMPLATE, BASELINE_ESTRATEGIA)]
            pedidos.discard("baseline")
            if pedidos:
                cenarios += [(tid, est) for tid, est in _templates_e_estrategias()
                             if _tipo(tid) in pedidos]
        else:
            cenarios = [(tid, est) for tid, est in cenarios
                        if _tipo(tid) in pedidos]
    if not cenarios:
        raise SystemExit("Nenhum cenario apos o filtro --cenarios")
    if (BASELINE_TEMPLATE, BASELINE_ESTRATEGIA) not in cenarios:
        cenarios = [(BASELINE_TEMPLATE, BASELINE_ESTRATEGIA)] + cenarios
    print(f"[config] {len(cenarios)} cenarios x {len(seeds)} seeds x "
          f"{args.epochs} epocas | seeds={seeds}")

    # ---- Documentos estruturados + paginacao -------------------------- #
    print("\n" + "=" * 100)
    print("PIPELINE DE DADOS - FORMATACOES ESTRUTURADAS + BASELINE")
    print("=" * 100)
    prep = preparar_documentos(tokenizer, args, cenarios)
    dados_etth1 = prep["dados"]
    paginados = prep["paginados"]

    # ---- Pool + datasets (fixos; o shuffle do loader segue a seed) ---- #
    pool = pool_janelas_etth1(dados_etth1, args.pool_janelas)
    print(f"[pool] {len(pool)} janelas numericas de "
          f"{dados_etth1['feat_names'][_FEAT_PADRAO]}")

    datasets: Dict[str, Tuple] = {}
    n_amostras = len(pool)
    n_tr, n_va = int(n_amostras * 0.6), int(n_amostras * 0.2)
    for tid, est in cenarios:
        chave = f"{tid}__{est}"
        ds = construir_dataset(paginados[chave], pool)
        tr = TensorDataset(*[t[:n_tr] for t in ds.tensors])
        va = TensorDataset(*[t[n_tr:n_tr + n_va] for t in ds.tensors])
        te = TensorDataset(*[t[n_tr + n_va:] for t in ds.tensors])
        datasets[chave] = build_loaders(tr, va, te, args.batch_size)
        print(f"  [{chave}] {len(ds)} amostras "
              f"(train={len(tr)} val={len(va)} test={len(te)})")

    # ---- Checkpoints -------------------------------------------------- #
    ckpts_sel: Dict[str, str] = {}
    for tok in args.checkpoints.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in CKPTS:
            ckpts_sel[tok] = str(CKPTS[tok])
        else:
            p = Path(tok)
            if not p.exists():
                raise SystemExit(f"Checkpoint nao encontrado: {p}")
            ckpts_sel[p.stem] = str(p)

    # ================================================================= #
    #  VALIDACAO - 10 epocas x 3 seeds (isolamento estrito por seed)     #
    # ================================================================= #
    print("\n" + "=" * 100)
    print(f"VALIDACAO - {args.epochs} EPOCAS x {len(seeds)} SEEDS "
          f"x {len(cenarios)} CENARIOS (isolamento estrito)")
    print("=" * 100)
    resumo: Dict = {
        "config": {
            "baseline": f"{BASELINE_TEMPLATE}__{BASELINE_ESTRATEGIA}",
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "linhas": args.linhas,
            "pool_janelas": len(pool),
            "block_max": args.block_max,
            "n_paginas_max": args.n_paginas_max,
            "overlap": args.overlap,
            "templates_estruturados": len(estruturado.TEMPLATES_ESTRUTURADOS),
            "estrategias": list(ESTRATEGIAS_A_TESTAR),
        },
        "cenarios": {},
    }

    for chave, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")
        for tid, est in cenarios:
            ck = f"{tid}__{est}"
            r_pag = paginados[ck]
            tr_ld, va_ld, te_ld = datasets[ck]
            por_seed: Dict[str, Dict] = {}
            mse_list, mae_list, bestval_list, epoca_list, estouros_list = (
                [], [], [], [], [])
            val_por_epoca: List[List[float]] = []
            test_por_epoca: List[List[float]] = []
            for seed in seeds:
                # ---- ISOLAMENTO ESTRITO: dentro do loop mais interno ---- #
                rb.seed_everything(seed)
                model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
                print(f"  [mount seed={seed}] {ck} | treinaveis="
                      f"{info['params_treinaveis']:,}")
                t0 = time.perf_counter()
                result = rodar_convergencia(
                    model, tr_ld, va_ld, te_ld,
                    epochs=args.epochs, lr=args.lr, device=device,
                    use_amp=use_amp, seed=seed,
                )
                mse = result["teste_final_mse_pred"]
                mae = result["teste_final_mae_pred"]
                bv = result["melhor_val_mse_pred"]
                be = result["melhor_epoca"]
                estouros = result["batches_estourados_total"]
                por_seed[str(seed)] = {
                    "test_mse": round(mse, 6), "test_mae": round(mae, 6),
                    "melhor_val_mse": round(bv, 6), "melhor_epoca": be,
                    "estouros": estouros,
                    "val_mse_por_epoca": [round(v, 6) for v in result["historico"]["val_mse_pred"]],
                    "test_mse_por_epoca": [round(v, 6) for v in result["historico"]["test_mse_pred"]],
                }
                mse_list.append(mse)
                mae_list.append(mae)
                bestval_list.append(bv)
                epoca_list.append(be or args.epochs)
                estouros_list.append(estouros)
                val_por_epoca.append(result["historico"]["val_mse_pred"])
                test_por_epoca.append(result["historico"]["test_mse_pred"])
                print(f"    [seed={seed}] test_mse={mse:.4f} | test_mae={mae:.4f} | "
                      f"best_val={bv:.4f} (ep {be}) | estouros={estouros} | "
                      f"{time.perf_counter() - t0:.1f}s")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            n_ep = max((len(v) for v in val_por_epoca), default=0)
            val_media = [float(np.mean([v[e] for v in val_por_epoca]))
                         for e in range(n_ep)]
            val_desvio = [float(np.std([v[e] for v in val_por_epoca]))
                          for e in range(n_ep)]
            test_media = [float(np.mean([v[e] for v in test_por_epoca]))
                          for e in range(n_ep)]
            test_desvio = [float(np.std([v[e] for v in test_por_epoca]))
                           for e in range(n_ep)]
            resumo["cenarios"][ck] = {
                "template": tid,
                "template_nome": _rotulo_curto(tid),
                "tipo": _tipo(tid),
                "idioma": _idioma_template(tid),
                "estrategia": est,
                "estrategia_nome": ESTRATEGIAS[est]["nome"],
                "test_mse_media": round(float(np.mean(mse_list)), 4),
                "test_mse_desvio": round(float(np.std(mse_list)), 4),
                "test_mae_media": round(float(np.mean(mae_list)), 4),
                "test_mae_desvio": round(float(np.std(mae_list)), 4),
                "melhor_val_mse_media": round(float(np.mean(bestval_list)), 4),
                "melhor_val_mse_desvio": round(float(np.std(bestval_list)), 4),
                "melhor_epoca_media": round(float(np.mean(epoca_list)), 2),
                "estouros_total": sum(estouros_list),
                "val_mse_media": [round(x, 6) for x in val_media],
                "val_mse_desvio": [round(x, 6) for x in val_desvio],
                "test_mse_media_por_epoca": [round(x, 6) for x in test_media],
                "test_mse_desvio_por_epoca": [round(x, 6) for x in test_desvio],
                "tokens_originais": r_pag["tokens_originais"],
                "tokens_retidos": r_pag["tokens_retidos"],
                "paginas": r_pag["n_paginas"],
                "por_seed": por_seed,
            }
            c = resumo["cenarios"][ck]
            print(f"  [ok] {ck}: MSE {c['test_mse_media']:.2f} ± "
                  f"{c['test_mse_desvio']:.2f} | MAE {c['test_mae_media']:.2f} ± "
                  f"{c['test_mae_desvio']:.2f} | estouros={c['estouros_total']}")

    # ---- Relatorio final ---------------------------------------------- #
    imprimir_tabela(resumo)
    out = salvar_resumo(resumo)
    print(f"[ok] JSON: {out}")
    if args.plots:
        baseline_mse = resumo["cenarios"][f"{BASELINE_TEMPLATE}__"
                                          f"{BASELINE_ESTRATEGIA}"]["test_mse_media"]
        p = gerar_grafico(resumo, baseline_mse)
        print(f"[ok] grafico: {p}")
    print("Benchmark de formatacoes estruturadas concluido.")


if __name__ == "__main__":
    main()
