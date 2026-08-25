"""
validacao_campeao_absoluto - VALIDACAO OURO do top-3 de formatos do ETTh1
(VolLoss + SMETimes), 10 epocas x 3 seeds com isolamento estrito.

MOTIVACAO:
O ranking exploratorio (estruturas_avancadas_stats.json) elegeu:
    #1  pseudo_json_es__janela      (7.03 +/- 0.06)  - CAMPEAO
    #2  temporal_delta_en__keywords (7.11 +/- 0.10)  - vice
    #3  es_keywords__keywords       (7.24 +/- 0.12)  - baseline (controle)
Este script RE-EXECUTA exatamente esses 3 cenarios (nada alem deles, por
seguranca de vazamento de selecao) e consolida os numeros finais que irao
para a analise de lancamento: media +/- desvio de Test MSE/MAE, best_val,
melhor epoca e estabilidade por seed.

RIGOR (identico aos demais benchmarks de publicacao):
    - 3 seeds (2021, 42, 1234) x 10 epocas por cenario;
    - isolamento estrito: seed_everything + montar_modelo DENTRO do loop
      mais interno (a cada iteracao seed/cenario) + empty_cache por ciclo;
    - baseline executado no MESMO harness (comparacao justa).

SAIDAS:
    resultados/campeao_absoluto_stats.json
    resultados/graficos/campeao_absoluto_validacao.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from .texto_puro.paginacao import TestadorPaginacao
from .treino import build_loaders, rodar_convergencia

# ---- Top-3 selado (da bateria de 21 cenarios) -------------------------- #
CAMPEOES: Tuple[Tuple[str, str], ...] = (
    ("pseudo_json_es", "janela"),        # #1 campeao (7.03 +/- 0.06)
    ("temporal_delta_en", "keywords"),   # #2 vice      (7.11 +/- 0.10)
)
BASELINE_TEMPLATE: str = "es_keywords"   # #3 controle   (7.24 +/- 0.12)
BASELINE_ESTRATEGIA: str = "keywords"

_COR_POR_TIPO: Dict[str, str] = {
    "pseudo_json": "#9467bd",
    "temporal_delta": "#d62728",
    "baseline": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validacao ouro do top-3: pseudo_json_es|janela, "
                    "temporal_delta_en|keywords, es_keywords|keywords"
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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    return parser.parse_args()


def _gerar_documento(tid: str, dados: Dict) -> str:
    """Documento completo (2000 linhas) no template do cenario."""
    if tid == BASELINE_TEMPLATE:
        return extenso.serie_para_texto_extenso(
            dados["dt_list"], dados["values"], _FEAT_PADRAO,
            0, dados["n_linhas"], tid,
        )
    return estruturado.serie_para_texto_estruturado(
        dados["dt_list"], dados["values"], _FEAT_PADRAO,
        0, dados["n_linhas"], tid,
    )


def _idioma(tid: str) -> str:
    if tid == BASELINE_TEMPLATE:
        return extenso.TEMPLATES_EXTENSO[tid]["idioma"]
    return estruturado.TEMPLATES_ESTRUTURADOS[tid]["idioma"]


def _tipo(tid: str) -> str:
    if tid == BASELINE_TEMPLATE:
        return "baseline"
    return estruturado.TEMPLATES_ESTRUTURADOS[tid]["tipo"]


def _rotulo(tid: str, est: str) -> str:
    if tid == BASELINE_TEMPLATE:
        return "Baseline ES Keywords (LN)"
    t = estruturado.TEMPLATES_ESTRUTURADOS[tid]
    est_nome = "janela" if est == "janela" else "keywords"
    return f"{t['nome']} ({est_nome})"


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

    cenarios: List[Tuple[str, str]] = list(CAMPEOES) + [
        (BASELINE_TEMPLATE, BASELINE_ESTRATEGIA)
    ]
    print(f"[config] {len(cenarios)} cenarios (top-3 selado) x {len(seeds)} seeds x "
          f"{args.epochs} epocas | seeds={seeds}")

    # ---- Documentos + paginacao --------------------------------------- #
    print("\n" + "=" * 100)
    print("PIPELINE DE DADOS - TOP-3 (CAMPEOA + VICE + BASELINE)")
    print("=" * 100)
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
        r = paginador.paginar(est, doc, _idioma(tid))
        r.update(paginador.para_modelo(r))
        chave = f"{tid}__{est}"
        paginados[chave] = r
        print(f"  [{tid} | {est}] {r['tokens_originais']} tokens (orig) -> "
              f"{r['tokens_retidos']} (ret) | {r['n_paginas']} paginas | "
              f"{r['paginas_descartadas']} descartadas")

    # ---- Pool + datasets (fixos; shuffle do loader segue a seed) ------ #
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

    # ---- Checkpoint --------------------------------------------------- #
    ckpt_sel: Dict[str, str] = {}
    for tok in args.checkpoints.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in CKPTS:
            ckpt_sel[tok] = str(CKPTS[tok])
        else:
            p = Path(tok)
            if not p.exists():
                raise SystemExit(f"Checkpoint nao encontrado: {p}")
            ckpt_sel[p.stem] = str(p)

    # ================================================================== #
    #  VALIDACAO OURO - 10 epocas x 3 seeds (isolamento estrito)          #
    # ================================================================== #
    print("\n" + "=" * 100)
    print(f"VALIDACAO OURO - {args.epochs} EPOCAS x {len(seeds)} SEEDS "
          f"x {len(cenarios)} CENARIOS (isolamento estrito)")
    print("=" * 100)
    resumo: Dict = {
        "config": {
            "top3": [f"{tid}__{est}" for tid, est in CAMPEOES],
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
        },
        "cenarios": {},
    }

    for chave, ckpt_path in ckpt_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")
        for tid, est in cenarios:
            ck = f"{tid}__{est}"
            r_pag = paginados[ck]
            tr_ld, va_ld, te_ld = datasets[ck]
            por_seed: Dict[str, Dict] = {}
            mse_list, mae_list, bestval_list, epoca_list, estouros_list = (
                [], [], [], [], [])
            val_por_epoca: List[List[float]] = []
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
            resumo["cenarios"][ck] = {
                "template": tid,
                "template_nome": _rotulo(tid, est),
                "tipo": _tipo(tid),
                "idioma": _idioma(tid),
                "estrategia": est,
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
    print("\n" + "=" * 100)
    print("TOP-3 FINAL (media ± dp sobre as 3 seeds)")
    print("=" * 100)
    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in resumo["cenarios"].items()
    )
    print(f"{'#':>2s} {'Cenario':<30s} {'Test MSE':>16s} {'Test MAE':>16s} "
          f"{'best_val':>14s} {'estouros':>8s}")
    print("-" * 100)
    for i, (mse, chave, c) in enumerate(linhas, 1):
        print(f"{i:>2d} {c['template_nome']:<30s} "
              f"{c['test_mse_media']:.2f} ± {c['test_mse_desvio']:.2f}  "
              f"{c['test_mae_media']:.2f} ± {c['test_mae_desvio']:.2f}  "
              f"{c['melhor_val_mse_media']:.2f} ± {c['melhor_val_mse_desvio']:.2f}  "
              f"{c['estouros_total']:>8d}")
    print("-" * 100)
    print("=" * 100)

    out = salvar_benchmark(resumo, nome="campeao_absoluto_stats.json")
    print(f"[ok] JSON: {out}")

    if args.plots:
        _gerar_grafico(resumo)
    print("Validacao ouro do campeao concluida.")


def _gerar_grafico(resumo: Dict) -> Path:
    """Barras ranqueadas do top-3 com error bars e linha da barreira 7.24."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    linhas = sorted(
        (c["test_mse_media"], chave, c) for chave, c in resumo["cenarios"].items()
    )
    nomes = [c["template_nome"] for _, _, c in linhas]
    medias = [c["test_mse_media"] for _, _, c in linhas]
    desvios = [c["test_mse_desvio"] for _, _, c in linhas]
    tipos = [c["tipo"] for _, _, c in linhas]
    cores = [_COR_POR_TIPO[t] for t in tipos]
    melhor = linhas[0]

    fig, ax = plt.subplots(figsize=(11, 6))
    barras = ax.bar(nomes, medias, yerr=desvios, capsize=6, width=0.55,
                    color=cores, edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhline(7.24, color="#7f7f7f", ls="--", lw=1.8, zorder=1,
               label="Baseline ES Keywords = 7.24 ± 0.12")
    for b, m, d in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, m + d + 0.04,
                f"{m:.2f}\n±{d:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(min(medias) - 0.8,
                max(m + d for m, d in zip(medias, desvios)) + 1.0)
    ax.set_ylabel("Test MSE (media das 3 seeds)")
    ax.set_title("Validacao Ouro - Top-3 de Formatos (10 epocas x 3 seeds)\n"
                 f"Campeao: {melhor[1]} = {melhor[2]['test_mse_media']:.2f} ± "
                 f"{melhor[2]['test_mse_desvio']:.2f}")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "campeao_absoluto_validacao.png"
    fig.savefig(p, dpi=115)
    plt.close(fig)
    print(f"[ok] grafico: {p}")
    return p


if __name__ == "__main__":
    main()