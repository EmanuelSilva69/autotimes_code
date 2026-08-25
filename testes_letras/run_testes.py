"""
run_testes - Orquestrador do Teste de Compressao de Prompt / Paginacao.

Executa, para os 2 modelos (SMETimes+AMD e AMD-only) e os 4 templates
(pt_verboso, pt_curto, en_curto, json):
    1) MEDICAO DE TOKENS: tokens por janela/bloco, pontos retidos por
       bloco, taxa de truncamento e tempo de tokenizacao;
    2) LOOP DE CONVERGENCIA: poucas epocas (default 5) com loss MSE/MAE
       por epoca e tempo por batch, avaliando qual template converge
       mais rapido e estavel;
    3) RELATORIO: JSONs em resultados/ + tabelas no console + graficos
       PNG (matplotlib) em resultados/graficos/.

Uso (dentro de testes_letras):
    python run_testes.py                        # tudo (2 modelos x 4 templates)
    python run_testes.py --templates en_curto,json
    python run_testes.py --checkpoints amd      # so o baseline otimizado
    python run_testes.py --epochs 5 --batch-size 16 --max-train-windows 192
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import caminhos
from .dados import (
    BLOCK_MAX,
    SEQ_LEN,
    TOKEN_LEN,
    TestadorTokenizacao,
    carregar_dados,
    construir_datasets,
    salvar_stats_tokenizacao,
)
from .modelos import CKPTS, montar_modelo
from .templates import IDS_TEMPLATES, TEMPLATES
from .treino import avaliar, build_loaders, rodar_convergencia


# --------------------------------------------------------------------- #
#  CLI                                                                 #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testador de Compressao de Prompt (tokenizacao + paginacao)"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes,amd",
                        help="Modelos a testar: 'smetimes' (VolLoss+SMETimes+AMD), "
                             "'amd' (so AMD) ou caminhos .pt separados por virgula")
    parser.add_argument("--templates", type=str, default="all",
                        help="Templates: all ou lista (pt_verboso,pt_curto,en_curto,json)")
    parser.add_argument("--epochs", type=int, default=5, help="Epocas do loop de convergencia")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam lr (bateria 0.001)")
    parser.add_argument("--max-train-windows", dest="max_train_windows", type=int, default=192,
                        help="Cap de janelas por feature no treino (rapidez)")
    parser.add_argument("--max-val-windows", dest="max_val_windows", type=int, default=64)
    parser.add_argument("--max-test-windows", dest="max_test_windows", type=int, default=64)
    parser.add_argument("--amp", action="store_true",
                        help="Liga AMP (autocast fp16 + GradScaler). ATENCAO: com o "
                             "fusion aleatorio, o backward fp16 estoura (NaN) nas "
                             "primeiras epocas; o default fp32+TF32 e estavel.")
    parser.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Nao gera graficos PNG")
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=None,
                        help="Override do diretorio do tokenizador local")
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Graficos (opcional)                                                  #
# --------------------------------------------------------------------- #
def gerar_graficos(
    stats_tok: Dict[str, Dict],
    convergencias: Dict[str, Dict],
    modelos_ordem: List[str],
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arquivos: List[Path] = []
    ids = list(stats_tok.keys())
    nomes = [TEMPLATES[i]["nome"] for i in ids]
    media_jan = [stats_tok[i]["tokens_por_janela"]["media"] for i in ids]
    pontos = [stats_tok[i]["pontos_por_bloco"] for i in ids]

    # 1) Tokens por janela + densidade.
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    cores = plt.cm.viridis(np.linspace(0.2, 0.9, len(ids)))
    b1 = ax1.bar(nomes, media_jan, color=cores, alpha=0.85)
    ax1.set_ylabel("Tokens medios por janela (672 passos)")
    ax1.set_title("Compressao de Prompt: tokens por janela (menor = melhor)")
    ax2 = ax1.twinx()
    ax2.plot(nomes, pontos, "o--", color="crimson", label="pontos por bloco")
    ax2.set_ylabel("Pontos temporais por bloco")
    for r, v in zip(b1, media_jan):
        ax1.text(r.get_x() + r.get_width() / 2, v + max(media_jan) * 0.02,
                 f"{v:.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    p1 = caminhos.GRAFICOS_DIR / "tokens_por_template.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    arquivos.append(p1)

    # 2) Curvas de convergencia por modelo.
    for mk in modelos_ordem:
        if not any(k.startswith(mk + "__") for k in convergencias):
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for tid in ids:
            chave = f"{mk}__{tid}"
            if chave not in convergencias:
                continue
            hist = convergencias[chave]["historico"]
            ax.plot(hist["epoch"], hist["val_mse_pred"],
                    marker="o", ms=4, label=TEMPLATES[tid]["nome"])
        ax.set_xlabel("Epoca")
        ax.set_ylabel("MSE no horizonte (val, pred_len=96)")
        ax.set_title(f"Convergencia por template - {mk}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = caminhos.GRAFICOS_DIR / f"convergencia_{mk}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        arquivos.append(p)
    return arquivos


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    caminhos.criar_resultados()
    rb = caminhos.importar_run_benchmarks()
    rb.seed_everything(args.seed)

    device = rb.get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    # Otimizacoes AMD da bateria (AutoTimes_Gpt2.py): cudnn benchmark +
    # matmul precision TF32 (no ROCm). Com o fusion aleatorio do modo
    # textual, AMP fp16 estoura em NaN no backward; por isso o default e
    # FP32 (estavel) mantendo as otimizacoes de matmul/conv do ROCm.
    import torch

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")
    if use_amp:
        print("[amp] mixed precision ATIVO (autocast fp16 + GradScaler)")
    else:
        print("[precisao] FP32 + TF32 (ROCm) | AMP desligado (estabilidade do fusion)")

    # ---- Tokenizador local (explicito) --------------------------------- #
    tok_dir = caminhos.descobrir_tokenizador(args.tokenizer_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,}, "
          f"pad={tokenizer.pad_token_id})")

    # ---- Dados ---------------------------------------------------------- #
    dados = carregar_dados()
    print(f"[dados] {dados['data_path']} | {dados['n_total']} pontos, "
          f"{dados['n_vars']} variaveis | seq={SEQ_LEN} token_len={TOKEN_LEN} "
          f"| splits b1s={dados['b1s']} b2s={dados['b2s']}")

    # ---- Selecao de templates ------------------------------------------- #
    if args.templates.strip().lower() == "all":
        ids_sel = list(IDS_TEMPLATES)
    else:
        ids_sel = [t.strip() for t in args.templates.split(",") if t.strip()]
        for t in ids_sel:
            if t not in TEMPLATES:
                raise SystemExit(f"Template desconhecido: {t}. Validos: {IDS_TEMPLATES}")

    # ---- Selecao de checkpoints ----------------------------------------- #
    ckpts_sel: Dict[str, Path] = {}
    for tok in args.checkpoints.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in CKPTS:
            ckpts_sel[tok] = CKPTS[tok]
        else:
            p = Path(tok)
            if not p.exists():
                raise SystemExit(f"Checkpoint nao encontrado: {p}")
            ckpts_sel[p.stem] = p

    caps = {"train": args.max_train_windows, "val": args.max_val_windows,
            "test": args.max_test_windows}

    print("\n" + "=" * 100)
    print("FASE 1 - TESTADOR DE TOKENIZACAO E PAGINACAO (compressao de prompt)")
    print("=" * 100)
    testador = TestadorTokenizacao(tokenizer, block_max=BLOCK_MAX)
    datasets: Dict[str, Dict] = {}
    stats_tok: Dict[str, Dict] = {}

    for tid in ids_sel:
        text_step = testador.escolher_step(tid, dados)
        t0 = time.perf_counter()
        train_ds, val_ds, test_ds = construir_datasets(
            tid, tokenizer, dados, text_step, caps=caps
        )
        stats = testador.medir_split(tid, dados, text_step,
                                     dados["b1s"][0], dados["b2s"][0],
                                     stride=1, max_janelas=len(train_ds))
        stats["tempo_build_s"] = round(time.perf_counter() - t0, 2)
        stats["n_janelas_train"] = len(train_ds)
        stats["n_janelas_val"] = len(val_ds)
        stats["n_janelas_test"] = len(test_ds)
        stats_tok[tid] = stats
        datasets[tid] = {"train": train_ds, "val": val_ds, "test": test_ds}
        m = stats["tokens_por_janela"]["media"]
        print(f"  [{TEMPLATES[tid]['nome']:<14s}] step={text_step:>2d} "
              f"| {m:6.1f} tokens/janela | {stats['pontos_por_bloco']:>3d} pts/bloco "
              f"| trunc={stats['taxa_truncamento']:.2%} | {len(train_ds)} janelas train "
              f"({stats['tempo_build_s']}s)")

    # Tabela de compressao relativa ao baseline (T1).
    base = stats_tok[ids_sel[0]]["tokens_por_janela"]["media"] if ids_sel else 0.0
    for tid in ids_sel:
        stats_tok[tid]["compressao_vs_t1"] = round(
            base / max(stats_tok[tid]["tokens_por_janela"]["media"], 1e-9), 3
        )
    salvar_stats_tokenizacao(stats_tok)
    print("\nCompressao relativa ao T1 (tokens/janela):")
    for tid in ids_sel:
        s = stats_tok[tid]
        print(f"  {TEMPLATES[tid]['nome']:<14s} {s['tokens_por_janela']['media']:8.1f} "
              f"tokens -> x{s['compressao_vs_t1']:.2f} de compressao")

    print("\n" + "=" * 100)
    print("FASE 2 - LOOP DE VALIDACAO DE CONVERGENCIA")
    print("=" * 100)
    convergencias: Dict[str, Dict] = {}
    modelos_ordem: List[str] = []

    for chave, ckpt_path in ckpts_sel.items():
        modelos_ordem.append(chave)
        print(f"\n--- Modelo: {chave} ({ckpt_path.name}) ---")
        model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
        print(f"  flags={info['flags']} | treinaveis={info['params_treinaveis']:,}")

        for tid in ids_sel:
            print(f"\n  [template] {TEMPLATES[tid]['nome']} "
                  f"({TEMPLATES[tid]['lingua']}) - {TEMPLATES[tid]['desc']}")
            train_loader, val_loader, test_loader = build_loaders(
                datasets[tid]["train"], datasets[tid]["val"], datasets[tid]["test"],
                args.batch_size,
            )
            result = rodar_convergencia(
                model, train_loader, val_loader, test_loader,
                epochs=args.epochs, lr=args.lr, device=device, use_amp=use_amp,
                seed=args.seed,
            )
            result["template"] = tid
            result["tokens_por_janela"] = stats_tok[tid]["tokens_por_janela"]["media"]
            result["tokens_por_bloco"] = stats_tok[tid]["tokens_por_bloco"]["media"]
            result["compressao_vs_t1"] = stats_tok[tid]["compressao_vs_t1"]
            result["tempo_batch_ms_medio"] = round(
                sum(result["historico"]["tempo_batch_ms"]) / max(args.epochs, 1), 2
            )
            convergencias[f"{chave}__{tid}"] = result
            print(f"  [ok] melhor val MSE_pred={result['melhor_val_mse_pred']:.6f} "
                  f"(ep {result['melhor_epoca']}) | "
                  f"test MSE_pred={result['teste_final_mse_pred']:.6f} "
                  f"MAE_pred={result['teste_final_mae_pred']:.6f} | "
                  f"{result['tempo_batch_ms_medio']} ms/batch")

    # ---- Relatorio final ------------------------------------------------ #
    resumo = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "tokenizador": str(tok_dir),
        "stats_tokenizacao": stats_tok,
        "convergencias": convergencias,
    }
    out_json = caminhos.RESULTADOS_DIR / "resumo_final.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    if args.plots:
        gerar_graficos(stats_tok, convergencias, modelos_ordem)

    # Tabela resumo no console.
    print("\n" + "=" * 100)
    print("RESUMO FINAL - CONVERGENCIA (MSE no horizonte 96, validacao)")
    print("=" * 100)
    hdr = f"{'Modelo/Template':<28s} {'best_val':>10s} {'ep':>3s} {'test_mse':>10s} {'ms/batch':>9s} {'tok/win':>9s}"
    print(hdr)
    print("-" * 100)
    for mk in modelos_ordem:
        for tid in ids_sel:
            c = convergencias[f"{mk}__{tid}"]
            print(
                f"  {TEMPLATES[tid]['nome'] + ' (' + mk + ')':<28s} "
                f"{c['melhor_val_mse_pred']:>10.5f} {c['melhor_epoca']:>3d} "
                f"{c['teste_final_mse_pred']:>10.5f} "
                f"{c['tempo_batch_ms_medio']:>9.2f} {c['tokens_por_janela']:>9.1f}"
            )
    print("=" * 100)
    print(f"[ok] JSON: {out_json}")
    for g in (caminhos.GRAFICOS_DIR.glob("*.png") if args.plots else []):
        print(f"[ok] grafico: {g}")
    print("Teste de compressao de prompt concluido.")


if __name__ == "__main__":
    main()
