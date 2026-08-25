"""
benchmark_f3_f4_ablation - sinergia Soft Prompt (F3) + CoT causal (F4).

O benchmark das 4 fases arquiteturais concluiu que apenas F3 e F4 superam a
baseline (7.24 +/- 0.12 -> 7.12/7.14), mas que a configuracao "combinada"
(TODAS as 4 fases) anula os ganhos (7.33), provando interferencia mutua
(vocab expansion e LoRA sao os agentes ruidosos). Este script:

    1) RETROSPECTIVA - le o resultados/benchmark_4fases_stats.json (F1, F2,
       F3, F4, baseline, combinada) e gera otimizacoes_arquiteturais_comparativo.png:
       barras de Test MSE medio com error bars (dp entre seeds) + linha
       horizontal da baseline, evidenciando a sobreposicao das bandas.

    2) ABLACAO FOCADA - treina ESTRITAMENTE F3 (soft prompt continuo de N
       vetores, prependidos aos ids de cada janela) + F4 (frase causal
       "Tendencia alcista/bajista. " por janela), SEM o codigo de vocab
       expansion nem LoRA. Mesmo rigor metodologico (3 seeds 2021/42/1234,
       10 epocas, isolamento estrito: seed_everything + montar_modelo no
       loop mais interno + torch.cuda.empty_cache() por ciclo).

    3) AVALIACAO - salva f3_f4_sinergia_stats.json (media +/- dp do novo
       cenario) e gera ablacao_vencedores_errorbar.png com os campeoes
       limpos: baseline, F3 isolada, F4 isolada e F3+F4 combinada.

O objetivo e verificar se o soft prompt extrai ainda mais valor da dica
causal do CoT sem sofrer interferencia das outras fases.

SAIDAS:
    resultados/f3_f4_sinergia_stats.json
    resultados/graficos/otimizacoes_arquiteturais_comparativo.png
    resultados/graficos/ablacao_vencedores_errorbar.png
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
from . import pipeline_otimizacao_extenso as _pipeline
from .modelos import CKPTS, montar_modelo
from .pipeline_otimizacao_extenso import (
    ESTRATEGIA_CAMPEA,
    FRASE_ALCISTA,
    FRASE_BAJISTA,
    TEMPLATE_CAMPEAO,
    construir_dataset_cot,
    instalar_soft_prompt,
)
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

# Cenario novo desta ablacao (F3 + F4, sem vocab/lora).
NOME_SINERGIA: str = "F3_F4_Sinergia"
JSON_4FASES: Path = caminhos.RESULTADOS_DIR / "benchmark_4fases_stats.json"

# Cores consistentes com o pipeline das 4 fases.
_CORES: Dict[str, str] = {
    "baseline": "#7f7f7f",
    "F1_VocabExpansao": "#9467bd",
    "F2_LoRA": "#d62728",
    "F3_SoftPrompt": "#17becf",
    "F4_CoT": "#ff7f0e",
    "combinada": "#2ca02c",
    NOME_SINERGIA: "#ff007f",
}

_RB = None  # type: ignore


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablacao de sinergia F3 (soft prompt) + F4 (CoT causal) "
                    "sobre o campeao ES Keywords"
    )
    parser.add_argument("--soft-prompt-n", dest="soft_prompt_n", type=int, default=10,
                        help="Numero de vetores continuos do soft prompt (F3)")
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
    parser.add_argument("--amp", action="store_true",
                        help="AMP fp16 (instavel com fusion aleatorio; default FP32+TF32)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  1) RETROSPECTIVA - grafico comparativo das 4 fases (dados antigos)    #
# --------------------------------------------------------------------- #
def _resumo_json_4fases() -> Dict:
    """Carrega o JSON do benchmark das 4 fases arquiteturais."""
    if not JSON_4FASES.exists():
        raise SystemExit(
            f"[erro] {JSON_4FASES} nao encontrado. Rode primeiro "
            f"pipeline_otimizacao_extenso.py"
        )
    import json

    with open(JSON_4FASES, encoding="utf-8") as f:
        return json.load(f)


def gerar_comparativo(resumo: Dict) -> Path:
    """Barras de Test MSE medio (com error bars) das 6 configs + linha baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cenarios = resumo["cenarios"]
    nomes = sorted(cenarios, key=lambda n: cenarios[n]["test_mse_media"])
    medias = [cenarios[n]["test_mse_media"] for n in nomes]
    desvios = [cenarios[n]["test_mse_desvio"] for n in nomes]
    baseline_mse = cenarios["baseline"]["test_mse_media"]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    barras = ax.bar(
        nomes, medias, yerr=desvios, capsize=5, width=0.62,
        color=[_CORES.get(n, "#333333") for n in nomes],
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    ax.axhline(baseline_mse, color="#7f7f7f", ls="--", lw=1.8, zorder=1,
               label=f"baseline = {baseline_mse:.2f}")
    for b, m, d in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, m + d + 0.04,
                f"{m:.2f}\n±{d:.2f}", ha="center", va="bottom", fontsize=8.5)
    ymax = max(m + d for m, d in zip(medias, desvios)) + 0.6
    ax.set_ylim(min(medias) - 0.8, ymax)
    ax.set_ylabel("Test MSE (media das 3 seeds)")
    ax.set_xlabel("Configuracao arquitetural")
    ax.set_title("Otimizacoes arquiteturais vs Baseline ES Keywords\n"
                 "(barra de erro = dp entre seeds; linha tracejada = baseline)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "otimizacoes_arquiteturais_comparativo.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  2) ABLACAO FOCADA - dados (somente variante base; sem vocab/lora)     #
# --------------------------------------------------------------------- #
def preparar_dados(
    args: argparse.Namespace, tok_dir: Path, dados_etth1: Dict,
) -> Dict:
    """Tokenizador base + datasets (sem CoT e com CoT) - F3/F4 apenas.

    Nao constroi o tokenizador expandido (FASE 1) nem envolve o modelo em
    peft (FASE 2): os dois agentes ruidosos ficam FORA desta ablacao.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 50256

    doc = extenso.serie_para_texto_extenso(
        dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
        0, dados_etth1["n_linhas"], TEMPLATE_CAMPEAO,
    )
    paginador = TestadorPaginacao(
        tok, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    r = paginador.paginar(ESTRATEGIA_CAMPEA, doc, "es")
    r.update(paginador.para_modelo(r))
    print(f"  [base] doc paginado: {r['tokens_originais']} tokens -> "
          f"{r['tokens_retidos']} retidos | {r['n_paginas']} paginas")

    pool = pool_janelas_etth1(dados_etth1, args.pool_janelas)
    n = len(pool)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)

    def _split(ds: TensorDataset) -> Tuple:
        tr = TensorDataset(*[t[:n_tr] for t in ds.tensors])
        va = TensorDataset(*[t[n_tr:n_tr + n_va] for t in ds.tensors])
        te = TensorDataset(*[t[n_tr + n_va:] for t in ds.tensors])
        return build_loaders(tr, va, te, args.batch_size)

    return {
        "doc": doc,
        "tok": tok,
        "paginados": r,
        "loaders": {
            "sem_cot": _split(construir_dataset(r, pool)),
            "com_cot": _split(construir_dataset_cot(r, pool, tok)),
        },
        "n_amostras": n,
        "pool": pool,
    }


# --------------------------------------------------------------------- #
#  3) AVALIACAO - treino do cenario F3 + F4 (rigor estrito)              #
# --------------------------------------------------------------------- #
def rodar_sinergia(
    args: argparse.Namespace,
    prep: Dict,
    ckpt_path: str,
    device: torch.device,
    use_amp: bool,
) -> Tuple[Dict, List[float], List[float]]:
    """Treina F3 (soft prompt) + F4 (CoT) sob isolamento estrito por ciclo.

    Returns:
        (cenario_dict, val_mse_media, val_mse_desvio) - schema identico ao
        do pipeline de otimizacao, pronto para plots/JSON.
    """
    rb = _RB
    tr_ld, va_ld, te_ld = prep["loaders"]["com_cot"]
    mse_list: List[float] = []
    mae_list: List[float] = []
    bestval_list: List[float] = []
    epoca_list: List[int] = []
    estouros_list: List[int] = []
    val_por_epoca: List[List[float]] = []
    por_seed: Dict[str, Dict] = {}
    params_extra = 0

    for seed in args.seeds_list:
        rb.seed_everything(seed)
        model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
        sp_info = instalar_soft_prompt(model, args.soft_prompt_n, device)
        n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params_extra == 0:
            params_extra = n_treinaveis - info["params_treinaveis"]
        print(f"  [mount seed={seed}] {NOME_SINERGIA} | "
              f"treinaveis={n_treinaveis:,} (extra={params_extra:,}) | {sp_info}")
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
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_ep = max((len(v) for v in val_por_epoca), default=0)
    val_media = [float(np.mean([v[e] for v in val_por_epoca]))
                 for e in range(n_ep)]
    val_desvio = [float(np.std([v[e] for v in val_por_epoca]))
                  for e in range(n_ep)]
    r_pag = prep["paginados"]
    cenario: Dict = {
        "flags": {"softprompt": True, "cot": True},
        "test_mse_media": round(float(np.mean(mse_list)), 4),
        "test_mse_desvio": round(float(np.std(mse_list)), 4),
        "test_mae_media": round(float(np.mean(mae_list)), 4),
        "test_mae_desvio": round(float(np.std(mae_list)), 4),
        "melhor_val_mse_media": round(float(np.mean(bestval_list)), 4),
        "melhor_val_mse_desvio": round(float(np.std(bestval_list)), 4),
        "melhor_epoca_media": round(float(np.mean(epoca_list)), 2),
        "estouros_total": sum(estouros_list),
        "params_extra": params_extra,
        "val_mse_media": [round(x, 6) for x in val_media],
        "val_mse_desvio": [round(x, 6) for x in val_desvio],
        "tokens_originais": r_pag["tokens_originais"],
        "tokens_retidos": r_pag["tokens_retidos"],
        "paginas": r_pag["n_paginas"],
        "por_seed": por_seed,
    }
    print(f"  [ok] {NOME_SINERGIA}: MSE {cenario['test_mse_media']:.2f} ± "
          f"{cenario['test_mse_desvio']:.2f} | MAE {cenario['test_mae_media']:.2f} ± "
          f"{cenario['test_mae_desvio']:.2f} | estouros={cenario['estouros_total']}")
    return cenario, val_media, val_desvio


# --------------------------------------------------------------------- #
#  Grafico final: campeoes limpos (baseline, F3, F4, F3+F4)              #
# --------------------------------------------------------------------- #
def gerar_abracao_vencedores(resumo_4fases: Dict, cenario_sinergia: Dict) -> Path:
    """Barras com error bars dos 4 campeoes limpos (F3/F4 do JSON antigo)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nomes = ["baseline", "F3_SoftPrompt", "F4_CoT", NOME_SINERGIA]
    medias: List[float] = []
    desvios: List[float] = []
    for n in nomes:
        if n == NOME_SINERGIA:
            medias.append(cenario_sinergia["test_mse_media"])
            desvios.append(cenario_sinergia["test_mse_desvio"])
        else:
            c = resumo_4fases["cenarios"][n]
            medias.append(c["test_mse_media"])
            desvios.append(c["test_mse_desvio"])

    fig, ax = plt.subplots(figsize=(9, 6))
    barras = ax.bar(
        nomes, medias, yerr=desvios, capsize=6, width=0.55,
        color=[_CORES.get(n, "#333333") for n in nomes],
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    for b, m, d in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, m + d + 0.04,
                f"{m:.2f}\n±{d:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(min(medias) - 0.8, max(medias) + max(desvios) + 0.7)
    ax.set_ylabel("Test MSE (media das 3 seeds)")
    ax.set_xlabel("Campeoes limpos")
    ax.set_title("Ablacao de sinergia: F3 (Soft Prompt) x F4 (CoT causal)\n"
                 "F3+F4 combinadas = objetivo de maxima extracao da dica causal")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "ablacao_vencedores_errorbar.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _fmt(m: float, d: float) -> str:
    return f"{m:.2f} ± {d:.2f}"


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    global _RB

    args = parse_args()
    caminhos.criar_resultados()
    _RB = caminhos.importar_run_benchmarks()
    rb = _RB
    _pipeline._RB = _RB

    args.seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not args.seeds_list:
        raise SystemExit("Nenhuma seed valida em --seeds")
    rb.seed_everything(args.seeds_list[0])

    device = rb.get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")

    # ---- 1) RETROSPECTIVA: grafico comparativo do benchmark antigo ---- #
    print("=" * 100)
    print("1) RETROSPECTIVA - benchmark das 4 fases (dados ja treinados)")
    print("=" * 100)
    resumo_4fases = _resumo_json_4fases()
    cenarios = resumo_4fases["cenarios"]
    for nome, c in cenarios.items():
        print(f"  {nome:<16s} MSE {_fmt(c['test_mse_media'], c['test_mse_desvio'])} "
              f"| MAE {_fmt(c['test_mae_media'], c['test_mae_desvio'])}")
    if args.plots:
        p1 = gerar_comparativo(resumo_4fases)
        print(f"[ok] grafico comparativo: {p1}")

    # ---- Tokenizador -------------------------------------------------- #
    try:
        if args.tokenizer_path and (Path(args.tokenizer_path) / "tokenizer.json").exists():
            tok_dir = Path(args.tokenizer_path)
        else:
            tok_dir = caminhos.descobrir_tokenizador_recursivo(caminhos.ROOT_DIR)
    except FileNotFoundError as e:
        print(f"[erro] tokenizador nao encontrado: {e}")
        raise SystemExit(1)
    print(f"[tokenizer] carregado de: {tok_dir}")

    # ---- 2) ABLACAO FOCADA: F3 + F4 (sem vocab, sem LoRA) ------------- #
    print("\n" + "=" * 100)
    print(f"2) ABLACAO FOCADA - {NOME_SINERGIA} "
          f"({len(args.seeds_list)} SEEDS x {args.epochs} EPOCAS, "
          f"isolamento estrito, SEM vocab/LoRA)")
    print("=" * 100)
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][_FEAT_PADRAO]
    print(f"[csv] {dados_etth1['csv']} | {dados_etth1['n_linhas_total']} linhas "
          f"-> {dados_etth1['n_linhas']} linhas (coluna '{feat_name}')")
    prep = preparar_dados(args, tok_dir, dados_etth1)
    print(f"[pool] {len(prep['pool'])} janelas numericas de {feat_name}")

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

    for chave, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")
        cenario, val_media, val_desvio = rodar_sinergia(
            args, prep, ckpt_path, device, use_amp,
        )

    # ---- 3) AVALIACAO FINAL ------------------------------------------ #
    resumo = {
        "config": {
            "template": TEMPLATE_CAMPEAO,
            "estrategia": ESTRATEGIA_CAMPEA,
            "seeds": args.seeds_list,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "linhas": args.linhas,
            "pool_janelas": prep["n_amostras"],
            "block_max": args.block_max,
            "n_paginas_max": args.n_paginas_max,
            "overlap": args.overlap,
            "soft_prompt_n": args.soft_prompt_n,
            "frases_cot": {"alcista": FRASE_ALCISTA, "bajista": FRASE_BAJISTA},
            "fases_ativas": ["F3_SoftPrompt", "F4_CoT"],
            "fases_eliminadas": ["F1_VocabExpansao", "F2_LoRA"],
        },
        "cenarios": {NOME_SINERGIA: cenario},
    }
    out = salvar_benchmark(resumo, nome="f3_f4_sinergia_stats.json")
    print(f"[ok] JSON: {out}")

    print("\n" + "=" * 100)
    print("CAMPEOES LIMPOS (baseline/F3/F4 do benchmark antigo + sinergia nova)")
    print("=" * 100)
    # (baseline/F3/F4 vêm do JSON antigo; sinergia é o cenario novo)
    for nome in ["baseline", "F3_SoftPrompt", "F4_CoT"]:
        c = resumo_4fases["cenarios"][nome]
        print(f"  {nome:<16s} MSE {_fmt(c['test_mse_media'], c['test_mse_desvio'])} "
              f"| MAE {_fmt(c['test_mae_media'], c['test_mae_desvio'])} "
              f"| best_val {_fmt(c['melhor_val_mse_media'], c['melhor_val_mse_desvio'])}")
    c = cenario
    print(f"  {NOME_SINERGIA:<16s} MSE {_fmt(c['test_mse_media'], c['test_mse_desvio'])} "
          f"| MAE {_fmt(c['test_mae_media'], c['test_mae_desvio'])} "
          f"| best_val {_fmt(c['melhor_val_mse_media'], c['melhor_val_mse_desvio'])}")
    print("=" * 100)

    if args.plots:
        p2 = gerar_abracao_vencedores(resumo_4fases, cenario)
        print(f"[ok] grafico vencedores: {p2}")
    print("Ablacao F3+F4 concluida.")


if __name__ == "__main__":
    main()
