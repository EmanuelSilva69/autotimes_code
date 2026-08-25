"""
validacao_final_top3 - validacao final (nivel publicacao) dos 3 formatos
textuais campeoes para previsao de series temporais por extenso.

Cenarios (formato 100% por extenso, sem algarismos):
    1. es_keywords | keywords  (Compressao por Palavras-Chave)
    2. en_verboso  | keywords  (Compressao por Palavras-Chave)
    3. pt_curto    | janela    (Janela Deslizante + Overlap)

Rigor estatistico:
    - 10 epocas (convergencia completa, sem minimo local precoce);
    - 3 seeds (2021, 42, 1234) por cenario;
    - isolamento estrito ZERO-CONTAMINACAO: seed_everything + montar_modelo
      ocorrem DENTRO do loop mais interno (a cada iteracao seed/cenario);
    - torch.cuda.empty_cache() ao fim de cada ciclo (sem fragmentacao);
    - agregacao: media +/- desvio padrao do Test MSE e Test MAE finais.

Saidas:
    resultados/top3_validacao_stats.json
    resultados/graficos/top3_convergencia_errorbar.png
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
from .texto_puro import extenso
from .texto_puro.paginacao import ESTRATEGIAS, TestadorPaginacao
from .treino import build_loaders, rodar_convergencia

# Ordem de exibicao/execucao dos 3 campeoes (formato por extenso).
CENARIOS: Dict[str, str] = {
    "es_keywords": "keywords",  # ES Keywords | Compressao
    "en_verboso": "keywords",   # EN Verboso  | Compressao
    "pt_curto": "janela",       # PT Curto    | Janela + Overlap
}

_CORES: Dict[str, str] = {
    "es": "#31a354",
    "en": "#d95f0e",
    "pt": "#2c7fb8",
}


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validacao final top-3 (es_keywords/en_verboso/pt_curto): "
                    "10 epocas x 3 seeds, media +/- desvio padrao"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes",
                        help="Modelos: 'smetimes', 'amd' ou caminhos .pt")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epocas por (cenario, seed) - convergencia total")
    parser.add_argument("--seeds", type=str, default="2021,42,1234",
                        help="Seeds por cenario (separadas por virgula)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=128,
                        help="Limite de tokens por pagina")
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int, default=7,
                        help="Paginas por janela do modelo (seq_len // token_len)")
    parser.add_argument("--overlap", type=float, default=0.15,
                        help="Sobreposicao da janela deslizante")
    parser.add_argument("--linhas", type=int, default=2000,
                        help="Subconjunto de linhas do ETTh1 convertido para texto")
    parser.add_argument("--pool-janelas", dest="pool_janelas", type=int, default=192,
                        help="Janelas numericas amostradas na mesma regiao")
    parser.add_argument("--csv", type=str, default=None,
                        help="Caminho do CSV (default: dataset/ETT-small/ETTh1.csv)")
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=None,
                        help="Override do diretorio do tokenizador")
    parser.add_argument("--amp", action="store_true",
                        help="Liga AMP fp16 (instavel com fusion aleatorio; default FP32+TF32)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Pipeline: documentos por extenso + paginacao                          #
# --------------------------------------------------------------------- #
def preparar_documentos(tokenizer, args) -> Dict[str, Dict]:
    """Carrega o ETTh1 e gera/pagina o documento de cada cenario.

    Returns:
        {"dados": dados_etth1, "paginados": {f"{est}__{tid}": r}}
    """
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][_FEAT_PADRAO]
    print(f"[csv] {dados_etth1['csv']} | {dados_etth1['n_linhas_total']} linhas "
          f"-> subconjunto de {dados_etth1['n_linhas']} linhas (coluna '{feat_name}')")

    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    paginados: Dict[str, Dict] = {}
    for tid, est in CENARIOS.items():
        doc = extenso.serie_para_texto_extenso(
            dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
            0, dados_etth1["n_linhas"], tid,
        )
        lingua = extenso.TEMPLATES_EXTENSO[tid]["idioma"]
        n_tok = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        r = paginador.paginar(est, doc, lingua)
        r.update(paginador.para_modelo(r))
        paginados[f"{est}__{tid}"] = r
        print(f"  [{tid} | {est}] {n_tok} tokens (orig) -> "
              f"{r['tokens_retidos']} (ret) | {r['n_paginas']} paginas | "
              f"{r['paginas_descartadas']} descartadas")
    return {"dados": dados_etth1, "paginados": paginados}


# --------------------------------------------------------------------- #
#  Relatorio final                                                       #
# --------------------------------------------------------------------- #
def _fmt(media: float, desvio: float) -> str:
    return f"{media:.2f} ± {desvio:.2f}"


def imprimir_tabela(resumo: Dict) -> None:
    print("\n" + "=" * 100)
    print("VALIDACAO FINAL TOP-3 (10 epocas x 3 seeds | isolamento estrito)")
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
    return salvar_benchmark(resumo, nome="top3_validacao_stats.json")


def gerar_grafico(resumo: Dict) -> Path:
    """Curvas de validacao (media das seeds) + banda de desvio padrao."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caminhos.criar_resultados()
    fig, ax = plt.subplots(figsize=(11, 6))
    epocas = np.arange(1, len(list(resumo["cenarios"].values())[0]["val_mse_media"]) + 1)
    for chave, c in resumo["cenarios"].items():
        tid = c["template"]
        cor = _CORES[extenso.TEMPLATES_EXTENSO[tid]["idioma"]]
        media = np.asarray(c["val_mse_media"])
        desvio = np.asarray(c["val_mse_desvio"])
        ax.plot(epocas, media, "-o", color=cor, lw=2, ms=4,
                label=f"{chave} (Test MSE {c['test_mse_media']:.2f} ± "
                      f"{c['test_mse_desvio']:.2f})")
        ax.fill_between(epocas, media - desvio, media + desvio,
                        color=cor, alpha=0.18, linewidth=0)
    ax.set_xticks(epocas)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val MSE no horizonte (media das 3 seeds)")
    ax.set_title("Validacao final top-3 por extenso: convergencia "
                 "com banda de confianca (media ± 1 dp)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "top3_convergencia_errorbar.png"
    fig.savefig(p, dpi=110)
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

    # ---- Tokenizador: busca dinamica recursiva ------------------------ #
    tok_dir: Optional[Path] = None
    try:
        if args.tokenizer_path and (Path(args.tokenizer_path) / "tokenizer.json").exists():
            tok_dir = Path(args.tokenizer_path)
        else:
            tok_dir = caminhos.descobrir_tokenizador_recursivo(caminhos.ROOT_DIR)
    except FileNotFoundError as e:
        print(f"[erro] tokenizador nao encontrado: {e}")
        raise SystemExit(1)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    except Exception as e:  # noqa: BLE001
        print(f"[erro] falha ao carregar o tokenizador de {tok_dir}: {e}")
        raise SystemExit(1)
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,})")
    print(f"[config] {len(CENARIOS)} cenarios x {len(seeds)} seeds x "
          f"{args.epochs} epocas | seeds={seeds}")

    # ---- Documentos por extenso + paginacao --------------------------- #
    print("\n" + "=" * 100)
    print("PIPELINE DE DADOS - DOCUMENTOS POR EXTENSO DOS 3 CAMPEOES")
    print("=" * 100)
    prep = preparar_documentos(tokenizer, args)
    dados_etth1 = prep["dados"]
    paginados = prep["paginados"]

    # ---- Pool + datasets (fixos; o shuffle do loader segue a seed) ---- #
    pool = pool_janelas_etth1(dados_etth1, args.pool_janelas)
    print(f"[pool] {len(pool)} janelas numericas de "
          f"{dados_etth1['feat_names'][_FEAT_PADRAO]}")

    datasets: Dict[str, Tuple] = {}
    n_amostras = len(pool)
    n_tr, n_va = int(n_amostras * 0.6), int(n_amostras * 0.2)
    for tid, est in CENARIOS.items():
        chave = f"{est}__{tid}"
        ds = construir_dataset(paginados[chave], pool)
        tr = TensorDataset(*[t[:n_tr] for t in ds.tensors])
        va = TensorDataset(*[t[n_tr:n_tr + n_va] for t in ds.tensors])
        te = TensorDataset(*[t[n_tr + n_va:] for t in ds.tensors])
        datasets[chave] = build_loaders(tr, va, te, args.batch_size)
        print(f"  [{tid} | {est}] {len(ds)} amostras "
              f"(train={len(tr)} val={len(va)} test={len(te)})")

    # ---- Checkpoints (mesma regra dos demais benchmarks) -------------- #
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
    print(f"VALIDACAO - {args.epochs} EPOCAS x {len(seeds)} SEEDS POR CENARIO")
    print("=" * 100)
    resumo: Dict = {
        "config": {
            "cenarios": list(CENARIOS.items()),
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

    for chave, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")
        for tid, est in CENARIOS.items():
            r_pag = paginados[f"{est}__{tid}"]
            tr_ld, va_ld, te_ld = datasets[f"{est}__{tid}"]
            por_seed: Dict[str, Dict] = {}
            mse_list, mae_list, bestval_list, epoca_list, estouros_list = [], [], [], [], []
            val_por_epoca: List[List[float]] = []
            for seed in seeds:
                # ---- ISOLAMENTO ESTRITO: dentro do loop mais interno ---- #
                rb.seed_everything(seed)
                model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
                print(f"  [mount seed={seed}] {tid} | {est} | "
                      f"flags={info['flags']} | treinaveis={info['params_treinaveis']:,}")
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
                # ---- fim do ciclo: libera VRAM (sem fragmentacao) ------ #
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            n_ep = max((len(v) for v in val_por_epoca), default=0)
            val_media = [float(np.mean([v[e] for v in val_por_epoca])) for e in range(n_ep)]
            val_desvio = [float(np.std([v[e] for v in val_por_epoca])) for e in range(n_ep)]
            resumo["cenarios"][tid] = {
                "template": tid,
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
                "tokens_originais": r_pag["tokens_originais"],
                "tokens_retidos": r_pag["tokens_retidos"],
                "paginas": r_pag["n_paginas"],
                "por_seed": por_seed,
            }
            c = resumo["cenarios"][tid]
            print(f"  [ok] {tid} | {est}: MSE {c['test_mse_media']:.2f} ± "
                  f"{c['test_mse_desvio']:.2f} | MAE {c['test_mae_media']:.2f} ± "
                  f"{c['test_mae_desvio']:.2f} | estouros={c['estouros_total']}")

    # ---- Relatorio final ---------------------------------------------- #
    imprimir_tabela(resumo)
    out = salvar_resumo(resumo)
    print(f"[ok] JSON: {out}")
    arquivos: List[Path] = []
    if args.plots:
        p = gerar_grafico(resumo)
        print(f"[ok] grafico: {p}")
        arquivos.append(p)
    print("Validacao final top-3 concluida.")


if __name__ == "__main__":
    main()
