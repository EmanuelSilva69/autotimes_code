"""
run_quinta_noite - Experimento CHAR-BY-CHAR (madrugada) no ETTh1 real.

Executa a bateria completa da nova hipotese de serializacao (digito a
digito, normalizacao por max_abs local da janela) com o MESMO protocolo do
campeao pseudo_json_es__janela (estruturas_avancadas_stats.json):

    1) Pool de 192 janelas univariadas da coluna HUFL (linhas 0..2000);
    2) 3 seeds (2021, 42, 1234) x 10 epocas, checkpoint smetimes
       (VolLoss + SMETimes), backbone congelado, fusion textual treinado
       do zero, Adam lr=1e-3 + CosineAnnealingLR, MSE puro;
    3) Isolamento estrito: seed_everything + montar_modelo DENTRO do loop
       de seeds (a cada iteracao) + empty_cache por ciclo;
    4) Contrato de 5 tensores (x, ids, lens, y_real, max_abs) adaptado
       graciosamente em treino.py (treinar_epoca/avaliar/rodar_convergencia).

ATENCAO - DECODE (escala):
    O AutoTimesModel textual ja devolve a reconstrucao DENORMALIZADA pelo
    RevIN interno (escala real da janela). Multiplicar pela max_abs de novo
    (--decode) INFLARIA as metricas por ~max_abs^2 (~100-1000x). Por isso o
    decode vem DESLIGADO por padrao: as metricas sao calculadas no espaco
    real nativo (mesmo espaco do campeao 7.0336). Use --decode somente se
    rodar uma variante que preveja no espaco normalizado por max_abs.

SAIDAS (testes_letras/quinta_noite/, criada automaticamente):
    char_by_char_stats.json        (mesmo schema de estruturas_avancadas)
    training_log.txt               (log linha a linha por seed/epoca)
    convergencia_char_vs_json.png  (Val MSE por epoca + bandas de desvio)
    veredito_char_by_char.png      (Test MSE bruto: Char vs Pseudo-JSON vs Baseline LN)

USO (na raiz do projeto):
    python -m testes_letras.run_quinta_noite
    python -m testes_letras.run_quinta_noite --epochs 1 --seeds 2021 --pool-janelas 24
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .modelos import CKPTS, montar_modelo
from .run_etth1_texto import carregar_etth1, pool_janelas_etth1
from .texto_char_by_char import (
    BLOCK_MAX,
    N_PAGINAS_MAX,
    OVERLAP,
    CharByCharPipeline,
    construir_dataset_char_by_char,
)
from .treino import build_loaders, rodar_convergencia

CENARIO: str = "char_by_char__janela"
NOME_CENARIO: str = "Char-by-Char"
OUT_DIR: Path = caminhos.BASE_DIR / "quinta_noite"
JSON_OUT: Path = OUT_DIR / "char_by_char_stats.json"
LOG_OUT: Path = OUT_DIR / "training_log.txt"
JSON_CAMPEAO: Path = caminhos.RESULTADOS_DIR / "estruturas_avancadas_stats.json"
CAMPEAO_CHAVE: str = "pseudo_json_es__janela"
BASELINE_LN_CHAVE: str = "es_keywords__keywords"


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimento Char-by-Char (10 epocas x 3 seeds) no ETTh1"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="2021,42,1234")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size",
                        type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=BLOCK_MAX)
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int,
                        default=N_PAGINAS_MAX)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument("--linhas", type=int, default=2000)
    parser.add_argument("--pool-janelas", dest="pool_janelas", type=int, default=192)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--decode", action="store_true",
                        help="Aplica decode(max_abs) nas metricas (somente "
                             "para variante que preveja no espaco normalizado)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Dados: pool + dataset de 5 tensores (texto por janela)               #
# --------------------------------------------------------------------- #
def preparar_dados(args: argparse.Namespace, tokenizer):
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][0]
    print(f"[csv] {dados_etth1['csv']} | {dados_etth1['n_linhas_total']} linhas "
          f"-> {dados_etth1['n_linhas']} linhas (coluna '{feat_name}')")

    pool = pool_janelas_etth1(dados_etth1, args.pool_janelas)
    print(f"[pool] {len(pool)} janelas numericas de {feat_name}")

    ds, stats = construir_dataset_char_by_char(
        pool, tokenizer,
        block_max=args.block_max,
        n_paginas_max=args.n_paginas_max,
        overlap_frac=args.overlap,
        return_stats=True,
    )
    print(f"[texto] doc/janela: ~{stats['tokens_originais_media']:.0f} tokens "
          f"originais | {stats['paginas_media']:.0f} paginas "
          f"(descartadas ~{stats['paginas_descartadas_media']:.0f}) | "
          f"max_abs={stats['max_abs_media']:.3f} ± {stats['max_abs_desvio']:.3f}")
    print(f"[dataset] {len(ds)} amostras | tensores: "
          f"{[tuple(t.shape) for t in ds.tensors]}")

    n = len(ds)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)
    tr = TensorDataset(*[t[:n_tr] for t in ds.tensors])
    va = TensorDataset(*[t[n_tr:n_tr + n_va] for t in ds.tensors])
    te = TensorDataset(*[t[n_tr + n_va:] for t in ds.tensors])
    loaders = build_loaders(tr, va, te, args.batch_size)
    print(f"  [split] train={len(tr)} val={len(va)} test={len(te)}")
    return dados_etth1, pool, stats, loaders


# --------------------------------------------------------------------- #
#  Relatorio: JSON + tabela + log                                       #
# --------------------------------------------------------------------- #
def _abrir_log() -> None:
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("# Char-by-Char - training log\n")
        f.write(f"# inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def _log_linha(linha: str) -> None:
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def _fmt(media: float, desvio: float) -> str:
    return f"{media:.2f} ± {desvio:.2f}"


def imprimir_tabela(resumo: Dict) -> None:
    print("\n" + "=" * 100)
    print("VEREDITO CHAR-BY-CHAR (10 epocas x 3 seeds) - escala REAL")
    print("=" * 100)
    c = resumo["cenarios"][CENARIO]
    print(f"{'Cenario':<24s} {'Test MSE':>16s} {'Test MAE':>16s} "
          f"{'best_val':>14s} {'estouros':>8s}")
    print("-" * 100)
    print(f"{CENARIO:<24s} {_fmt(c['test_mse_media'], c['test_mse_desvio']):>16s} "
          f"{_fmt(c['test_mae_media'], c['test_mae_desvio']):>16s} "
          f"{_fmt(c['melhor_val_mse_media'], c['melhor_val_mse_desvio']):>14s} "
          f"{c['estouros_total']:>8d}")
    print("-" * 100)
    if JSON_CAMPEAO.exists():
        antigo = json.load(open(JSON_CAMPEAO, encoding="utf-8"))
        for chave, rotulo in ((CAMPEAO_CHAVE, "Pseudo-JSON ES"),
                              (BASELINE_LN_CHAVE, "Baseline LN")):
            a = antigo["cenarios"].get(chave)
            if a:
                print(f"{rotulo:<24s} {_fmt(a['test_mse_media'], a['test_mse_desvio']):>16s}")
    print("Formato: media ± desvio padrao sobre as 3 seeds (menor MSE vence).")
    print("=" * 100)


# --------------------------------------------------------------------- #
#  Graficos comparativos (estilo publicacao)                            #
# --------------------------------------------------------------------- #
def gerar_graficos(resumo: Dict, out_dir: Path) -> List[Path]:
    """Convergencia Char vs Pseudo-JSON + veredito de barras (Test MSE)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not JSON_CAMPEAO.exists():
        print(f"[graficos] campeao nao encontrado ({JSON_CAMPEAO}) -> pulando")
        return []
    antigo = json.load(open(JSON_CAMPEAO, encoding="utf-8"))
    novo = resumo["cenarios"][CENARIO]
    campeao = antigo["cenarios"].get(CAMPEAO_CHAVE)
    baseline_ln = antigo["cenarios"].get(BASELINE_LN_CHAVE)
    if campeao is None:
        print(f"[graficos] cenario {CAMPEAO_CHAVE} ausente -> pulando")
        return []

    sns.set_theme(style="whitegrid", font_scale=1.25)
    saidas: List[Path] = []

    # ---- Grafico 1: convergencia (Val MSE por epoca + bandas) --------- #
    fig, ax = plt.subplots(figsize=(11, 6.5))
    m = np.asarray(novo["val_mse_media"])
    d = np.asarray(novo["val_mse_desvio"])
    m2 = np.asarray(campeao["val_mse_media"])
    d2 = np.asarray(campeao["val_mse_desvio"])
    ep1 = np.arange(1, len(m) + 1)
    ep2 = np.arange(1, len(m2) + 1)

    ax.plot(ep1, m, "-o", color="#d62728", lw=2.4, ms=5, zorder=4,
            label=f"Char-by-Char (Test MSE {novo['test_mse_media']:.3f})")
    ax.fill_between(ep1, m - d, m + d, color="#d62728", alpha=0.15, zorder=2)

    ax.plot(ep2, m2, "-s", color="#9467bd", lw=2.2, ms=5, zorder=3,
            label=f"Pseudo-JSON ES (Test MSE {campeao['test_mse_media']:.3f})")
    ax.fill_between(ep2, m2 - d2, m2 + d2, color="#9467bd", alpha=0.15, zorder=2)

    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val MSE no horizonte (media das 3 seeds)")
    ax.set_title("Convergencia: Char-by-Char vs Pseudo-JSON ES\n"
                 "(linhas = media, bandas = desvio padrao sobre 3 seeds)")
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    p1 = out_dir / "convergencia_char_vs_json.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saidas.append(p1)
    print(f"[graficos] {p1}")

    # ---- Grafico 2: veredito (barras Test MSE bruto) ------------------- #
    fig, ax = plt.subplots(figsize=(9, 6.5))
    nomes = ["Char-by-Char", "Pseudo-JSON ES", "Baseline LN"]
    medias = [novo["test_mse_media"], campeao["test_mse_media"]]
    desvios = [novo["test_mse_desvio"], campeao["test_mse_desvio"]]
    if baseline_ln is not None:
        medias.append(baseline_ln["test_mse_media"])
        desvios.append(baseline_ln["test_mse_desvio"])
    cores = ["#d62728", "#9467bd", "#7f7f7f"]

    barras = ax.bar(nomes, medias, yerr=desvios, capsize=6, width=0.55,
                    color=cores, edgecolor="black", linewidth=0.8, zorder=3)
    for b, m, dv in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, m + dv + max(medias) * 0.02,
                f"{m:.3f}\n± {dv:.3f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylim(0, max(m + d for m, d in zip(medias, desvios)) * 1.22)
    ax.set_ylabel("Test MSE bruto (escala real, 3 seeds)")
    ax.set_title("Veredito Char-by-Char: Test MSE final\n"
                 "(menor = melhor; barras = media ± desvio padrao)")
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    p2 = out_dir / "veredito_char_by_char.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saidas.append(p2)
    print(f"[graficos] {p2}")
    return saidas


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _abrir_log()
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
    from transformers import AutoTokenizer

    tok_dir = caminhos.descobrir_tokenizador()
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,})")

    # ---- Decode (escala) ---------------------------------------------- #
    decode_fn = None
    if args.decode:
        pl = CharByCharPipeline(tokenizer, block_max=args.block_max,
                                n_paginas_max=args.n_paginas_max,
                                overlap_frac=args.overlap)
        decode_fn = pl.decode
        print("[decode] ATIVO: metricas apos multiplicar pred por max_abs "
              "(somente se a variante prever no espaco normalizado)")
    else:
        print("[decode] OFF (padrao): o modelo ja devolve a reconstrucao "
              "denormalizada pelo RevIN interno (escala real). --decode "
              "INFLARIA as metricas por ~max_abs^2.")

    # ---- Dados -------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("PIPELINE DE DADOS - CHAR-BY-CHAR (texto por janela)")
    print("=" * 100)
    _, pool, stats, (tr_ld, va_ld, te_ld) = preparar_dados(args, tokenizer)

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
          f"x {len(ckpts_sel)} MODELOS (isolamento estrito)")
    print("=" * 100)
    resumo: Dict = {
        "config": {
            "experimento": "char_by_char",
            "template": "char_by_char",
            "estrategia": "janela",
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "linhas": args.linhas,
            "pool_janelas": len(pool),
            "block_max": args.block_max,
            "n_paginas_max": args.n_paginas_max,
            "overlap": args.overlap,
            "decode_aplicado": bool(args.decode),
            "amp": use_amp,
            "referencia_campeao_json": str(JSON_CAMPEAO),
            "referencia_campeao_chave": CAMPEAO_CHAVE,
            "referencia_baseline_ln_chave": BASELINE_LN_CHAVE,
        },
        "cenarios": {},
    }

    for chave_ckpt, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave_ckpt} ({Path(ckpt_path).name}) ---")
        por_seed: Dict[str, Dict] = {}
        mse_list, mae_list, bestval_list, epoca_list, estouros_list = (
            [], [], [], [], [])
        val_por_epoca: List[List[float]] = []
        test_por_epoca: List[List[float]] = []
        for seed in seeds:
            # ---- ISOLAMENTO ESTRITO: dentro do loop mais interno ---- #
            rb.seed_everything(seed)
            model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
            print(f"  [mount seed={seed}] treinaveis={info['params_treinaveis']:,}")
            t0 = time.perf_counter()
            result = rodar_convergencia(
                model, tr_ld, va_ld, te_ld,
                epochs=args.epochs, lr=args.lr, device=device,
                use_amp=use_amp, seed=seed, decode_fn=decode_fn,
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
                "val_mse_por_epoca": [round(v, 6)
                                      for v in result["historico"]["val_mse_pred"]],
                "test_mse_por_epoca": [round(v, 6)
                                       for v in result["historico"]["test_mse_pred"]],
            }
            mse_list.append(mse)
            mae_list.append(mae)
            bestval_list.append(bv)
            epoca_list.append(be or args.epochs)
            estouros_list.append(estouros)
            val_por_epoca.append(result["historico"]["val_mse_pred"])
            test_por_epoca.append(result["historico"]["test_mse_pred"])
            _log_linha(
                f"[{chave_ckpt} seed={seed}] test_mse={mse:.6f} "
                f"test_mae={mae:.6f} best_val={bv:.6f} (ep {be}) "
                f"estouros={estouros} | {time.perf_counter() - t0:.1f}s"
            )
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
        resumo["cenarios"][CENARIO] = {
            "template": "char_by_char",
            "template_nome": NOME_CENARIO,
            "tipo": "char_by_char",
            "idioma": "pt",
            "estrategia": "janela",
            "estrategia_nome": "Janela Deslizante + Overlap",
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
            "tokens_originais": round(stats["tokens_originais_media"], 1),
            "tokens_retidos": round(stats["tokens_retidos_media"], 1),
            "paginas": round(stats["paginas_media"], 1),
            "paginas_descartadas": round(stats["paginas_descartadas_media"], 1),
            "max_abs_media": round(stats["max_abs_media"], 6),
            "max_abs_desvio": round(stats["max_abs_desvio"], 6),
            "decode_aplicado": bool(args.decode),
            "por_seed": por_seed,
        }
        c = resumo["cenarios"][CENARIO]
        _log_linha(
            f"[ok] {CENARIO}: MSE {c['test_mse_media']:.4f} ± "
            f"{c['test_mse_desvio']:.4f} | MAE {c['test_mae_media']:.4f} ± "
            f"{c['test_mae_desvio']:.4f} | estouros={c['estouros_total']}"
        )
        print(f"  [ok] {CENARIO}: MSE {c['test_mse_media']:.4f} ± "
              f"{c['test_mse_desvio']:.4f} | MAE {c['test_mae_media']:.4f} ± "
              f"{c['test_mae_desvio']:.4f} | estouros={c['estouros_total']}")

    # ---- Relatorio final ---------------------------------------------- #
    imprimir_tabela(resumo)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"[ok] JSON: {JSON_OUT}")
    if args.plots:
        gerar_graficos(resumo, OUT_DIR)
    print("Experimento Char-by-Char concluido.")


if __name__ == "__main__":
    main()