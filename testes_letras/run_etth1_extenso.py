"""
run_etth1_extenso - Benchmark de NLP pura: serie ETTh1 100% POR EXTENSO.

Elimina TODOS os algarismos da conversao textual do ETTh1 (numeros e
timestamps viram palavras - ex.: "dez", "catorze", "vinte e nove virgula
cinco") para medir a capacidade pura de compreensao de tendencias do GPT-2.

    FASE A - INFLACAO DE TOKENS POR IDIOMA (absoluta):
        9 templates (3 idiomas x 3 verbosidades - verboso, curto, keywords)
        sao gerados e tokenizados. Loga-se a taxa de inflacao absoluta de
        cada idioma com os numeros por extenso (o GPT-2 lida melhor com
        "twenty" do que com "vinte"? quantos tokens por ponto?).

    FASE B - PAGINACAO (chunking), 3 estrategias:
        truncamento (hard truncation) | janela deslizante c/ overlap |
        compressao por palavras-chave - sobre cada um dos 9 templates.

    FASE C - CONVERGENCIA (VolLoss + SMETimes):
        (3 estrategias x 9 templates = 27 cenarios) x 5-6 epocas. O Test MSE
        final de todos os cenarios e ranqueado para descobrir qual
        linguagem/formato permite a melhor reconstrucao matematica a partir
        de palavras.

Saidas:
    resultados/benchmark_etth1_extenso.json
    resultados/graficos/etth1_extenso_mse.png     (barras horizontais, rank)
    resultados/graficos/etth1_extenso_tokens.png  (inflacao de tokens)

Uso (na raiz do projeto):
    python -m testes_letras.run_etth1_extenso
    python -m testes_letras.run_etth1_extenso --epochs 5 --templates pt_curto,en_curto
    python -m testes_letras.run_etth1_extenso --estrategias janela,keywords
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
from .dados import LABEL_LEN, PRED_LEN, SEQ_LEN
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

_IDIOMAS: List[str] = ["pt", "en", "es"]
_NIVEIS: List[str] = ["verboso", "curto", "keywords"]


# --------------------------------------------------------------------- #
#  CLI                                                                 #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark de NLP pura: serie ETTh1 100% por extenso (PT/EN/ES)"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes",
                        help="Modelos: 'smetimes', 'amd' ou caminhos .pt")
    parser.add_argument("--estrategias", type=str, default="truncamento,janela,keywords",
                        help="Estrategias de paginacao (separadas por virgula)")
    parser.add_argument("--templates", type=str, default="all",
                        help="Templates por extenso: all ou lista "
                             "(pt_verboso,pt_curto,en_curto,...)")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Epocas de convergencia (recomendado 5-6)")
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
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Relatorio (tabelas + JSON + graficos)                                #
# --------------------------------------------------------------------- #
def imprimir_inflacao(tokens: Dict[str, Dict]) -> None:
    print("\n" + "=" * 100)
    print("INFLACAO DE TOKENS ABSOLUTA (serie 100% por extenso, por idioma)")
    print("=" * 100)
    hdr = (f"{'Template':<16s} {'Idioma':>6s} {'Nivel':>9s} {'tokens':>10s} "
           f"{'tokens/ponto':>12s} {'tokens/caract':>13s}")
    print(hdr)
    print("-" * 100)
    for tid, s in tokens.items():
        print(
            f"{tid:<16s} {s['idioma']:>6s} {s['nivel']:>9s} {s['tokens']:>10d} "
            f"{s['tokens_por_ponto']:>12.2f} {s['tokens_por_caractere']:>13.4f}"
        )
    # Media por idioma (3 niveis).
    print("-" * 100)
    for idioma in _IDIOMAS:
        vals = [tokens[t]["tokens_por_ponto"] for t in tokens if tokens[t]["idioma"] == idioma]
        if vals:
            print(f"{'MEDIA':<16s} {idioma:>6s} {'todos':>9s} {'-':>10s} "
                  f"{float(np.mean(vals)):>12.2f} {'(tokens/ponto)':>13s}")
    print("=" * 100)


def imprimir_tabela(resultados: Dict) -> None:
    print("\n" + "=" * 100)
    print("CONVERGENCIA 100% POR EXTENSO (ranqueada pelo Test MSE final)")
    print("=" * 100)
    linhas = sorted(
        (
            (r["teste_final_mse_pred"], tid, est, r["melhor_val_mse_pred"],
             r["batches_estourados_total"])
            for tid in extenso.TEMPLATES_EXTENSO
            for est, r in resultados.get(tid, {}).items()
        ),
        key=lambda t: t[0],
    )
    print(f"{'#':>3s} {'Estrategia':<14s} {'Template':<14s} {'test_mse':>10s} "
          f"{'best_val':>10s} {'estouros':>8s}")
    print("-" * 100)
    for i, (mse, tid, est, bv, estouros) in enumerate(linhas, 1):
        print(f"{i:>3d} {est:<14s} {tid:<14s} {mse:>10.4f} {bv:>10.4f} {estouros:>8d}")
    print("=" * 100)


def salvar_resumo(
    resumo: Dict, tokens: Dict, resultados: Dict, pag: Dict
) -> Path:
    resumo["tokens_por_extenso"] = tokens
    resumo["paginacao"] = pag
    resumo["resultados"] = {
        tid: {est: r for est, r in res.items()} for tid, res in resultados.items()
    }
    return salvar_benchmark(resumo, nome="benchmark_etth1_extenso.json")


def gerar_graficos(resultados: Dict, tokens: Dict) -> List[Path]:
    """PNGs: barras horizontais do Test MSE (rank) + inflacao de tokens."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caminhos.criar_resultados()
    arquivos: List[Path] = []

    # 1) etth1_extenso_mse.png - ranking horizontal do Test MSE final.
    linhas = sorted(
        (
            (r["teste_final_mse_pred"], tid, est)
            for tid in extenso.TEMPLATES_EXTENSO
            for est, r in resultados.get(tid, {}).items()
        ),
        key=lambda t: t[0],
    )
    cores = {
        "pt": "#2c7fb8", "en": "#d95f0e", "es": "#31a354",
    }
    fig, ax = plt.subplots(figsize=(10, 11))
    y = np.arange(len(linhas))
    rotulos = [f"{tid} | {est}" for _, tid, est in linhas]
    vals = [t[0] for t in linhas]
    idiomas = [extenso.TEMPLATES_EXTENSO[tid]["idioma"] for _, tid, _ in linhas]
    ax.barh(y, vals, color=[cores[i] for i in idiomas], alpha=0.9)
    for yi, v in zip(y, vals):
        ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(rotulos, fontsize=7)
    ax.invert_yaxis()  # melhor (menor MSE) no topo
    ax.set_xlabel("Test MSE no horizonte (final)")
    ax.set_title("ETTh1 por extenso: rank da reconstrucao matematica "
                 "(cores: pt azul | en laranja | es verde)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=f"{i}") for i, c in cores.items()],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "etth1_extenso_mse.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    arquivos.append(p)

    # 2) etth1_extenso_tokens.png - inflacao de tokens por ponto (template).
    tids = list(tokens.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [tokens[t]["tokens_por_ponto"] for t in tids]
    cores_b = [cores[extenso.TEMPLATES_EXTENSO[t]["idioma"]] for t in tids]
    ax.bar(tids, vals, color=cores_b, alpha=0.9)
    for xi, v in zip(range(len(tids)), vals):
        ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(tids)))
    ax.set_xticklabels(tids, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Tokens por ponto (media)")
    ax.set_title("Inflacao de tokens absoluta: serie por extenso em PT/EN/ES")
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "etth1_extenso_tokens.png"
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
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")

    # ---- Tokenizador: busca dinamica recursiva (sem hardcode) -------- #
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

    # ---- Validacao de argumentos -------------------------------------- #
    estrategias = [e.strip() for e in args.estrategias.split(",") if e.strip()]
    for e in estrategias:
        if e not in ESTRATEGIAS:
            raise SystemExit(f"Estrategia desconhecida: {e}. Validas: {list(ESTRATEGIAS)}")
    if args.templates.strip().lower() == "all":
        templates = list(extenso.TEMPLATES_EXTENSO.keys())
    else:
        templates = [t.strip() for t in args.templates.split(",") if t.strip()]
        for t in templates:
            if t not in extenso.TEMPLATES_EXTENSO:
                raise SystemExit(f"Template desconhecido: {t}. Validos: "
                                 f"{extenso.IDS_TEMPLATES_EXTENSO}")
    print(f"[config] {len(templates)} templates x {len(estrategias)} estrategias "
          f"x {args.epochs} epocas")

    # ================================================================= #
    # FASE A - CONVERSAO 100% POR EXTENSO + INFLACAO DE TOKENS          #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE A - CONVERSAO POR EXTENSO (PT/EN/ES) E INFLACAO DE TOKENS")
    print("=" * 100)
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][_FEAT_PADRAO]
    print(f"[csv] {dados_etth1['csv']} | {dados_etth1['n_linhas_total']} linhas "
          f"-> subconjunto de {dados_etth1['n_linhas']} linhas (coluna '{feat_name}')")

    # Exemplo do primeiro ponto em cada template (auditoria da conversao).
    dt0 = dados_etth1["dt_list"][0]
    v0 = round(float(dados_etth1["values"][0, _FEAT_PADRAO]), 1)
    for tid in templates:
        print(f"  [{tid}] {extenso.formatar_ponto(tid, dt0, v0)}")

    textos: Dict[str, str] = {}
    tokens_info: Dict[str, Dict] = {}
    for tid in templates:
        doc = extenso.serie_para_texto_extenso(
            dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
            0, dados_etth1["n_linhas"], tid,
        )
        n_tok = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        textos[tid] = doc
        tokens_info[tid] = {
            "idioma": extenso.TEMPLATES_EXTENSO[tid]["idioma"],
            "nivel": extenso.TEMPLATES_EXTENSO[tid]["nivel"],
            "tokens": n_tok,
            "caracteres": len(doc),
            "n_pontos": dados_etth1["n_linhas"],
            "tokens_por_ponto": round(n_tok / dados_etth1["n_linhas"], 3),
            "tokens_por_caractere": round(n_tok / max(len(doc), 1), 4),
        }
    imprimir_inflacao(tokens_info)

    # ================================================================= #
    # FASE B - PAGINACAO (3 estrategias x 9 templates)                  #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE B - PAGINACAO (chunking) DOS TEMPLATES POR EXTENSO")
    print("=" * 100)
    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    paginados: Dict[str, Dict] = {}  # (estrategia__template) -> resultado
    pag: Dict = {e: {} for e in estrategias}
    for est in estrategias:
        for tid in templates:
            lingua = extenso.TEMPLATES_EXTENSO[tid]["idioma"]
            r = paginador.paginar(est, textos[tid], lingua)
            r.update(paginador.para_modelo(r))
            paginados[f"{est}__{tid}"] = r
            pag[est][tid] = {
                "tokens_originais": r["tokens_originais"],
                "tokens_retidos": r["tokens_retidos"],
                "tokens_descartados": r["tokens_descartados"],
                "paginas": r["n_paginas"],
                "paginas_descartadas": int(r.get("paginas_descartadas", 0)),
                "tempo_tokenizacao_s": r["tempo_tokenizacao_s"],
            }
            print(
                f"  [{ESTRATEGIAS[est]['nome']:<28s}] {tid:<14s}: "
                f"{r['tokens_originais']:,} -> {r['tokens_retidos']:,} tokens | "
                f"{r['n_paginas']} paginas (descartadas: {r.get('paginas_descartadas', 0)})"
            )

    # ================================================================= #
    # FASE C - CONVERGENCIA (VolLoss + SMETimes, 27 cenarios)           #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE C - CONVERGENCIA 100% POR EXTENSO (arquitetura VolLoss + SMETimes)")
    print("=" * 100)
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

    pool = pool_janelas_etth1(dados_etth1, args.pool_janelas)
    print(f"[pool] {len(pool)} janelas numericas de {feat_name} "
          f"(linhas 0..{dados_etth1['n_linhas']})")

    resultados: Dict[str, Dict] = {}
    n_amostras = len(pool)
    n_tr, n_va = int(n_amostras * 0.6), int(n_amostras * 0.2)

    for chave, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")

        for tid in templates:
            for est in estrategias:
                # Modelo REMONTADO do checkpoint a cada cenario (isolamento:
                # sem heranca de treinos anteriores entre cenarios).
                rb.seed_everything(args.seed)
                model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
                print(f"  [mount] {tid} | {est} | "
                      f"flags={info['flags']} | treinaveis={info['params_treinaveis']:,}")
                r_pag = paginados[f"{est}__{tid}"]
                ds = construir_dataset(r_pag, pool)
                tr, va, te = (
                    TensorDataset(*[t[:n_tr] for t in ds.tensors]),
                    TensorDataset(*[t[n_tr:n_tr + n_va] for t in ds.tensors]),
                    TensorDataset(*[t[n_tr + n_va:] for t in ds.tensors]),
                )
                train_loader, val_loader, test_loader = build_loaders(
                    tr, va, te, args.batch_size
                )
                print(f"\n  [{tid} | {est}] {len(ds)} amostras "
                      f"(train={len(tr)} val={len(va)} test={len(te)})")
                t0 = time.perf_counter()
                result = rodar_convergencia(
                    model, train_loader, val_loader, test_loader,
                    epochs=args.epochs, lr=args.lr, device=device,
                    use_amp=use_amp, seed=args.seed,
                )
                tempos = result["historico"]["tempo_batch_ms"]
                if len(tempos) > 2:
                    tempos = tempos[1:]  # exclui ep1 (warmup kernels ROCm)
                result["tempo_batch_ms_medio"] = round(
                    sum(tempos) / max(len(tempos), 1), 2
                )
                result["tokens_originais"] = r_pag["tokens_originais"]
                result["tokens_retidos"] = r_pag["tokens_retidos"]
                result["paginas"] = r_pag["n_paginas"]
                resultados.setdefault(tid, {})[est] = result
                print(
                    f"  [ok] test_mse={result['teste_final_mse_pred']:.4f} | "
                    f"best_val={result['melhor_val_mse_pred']:.4f} | "
                    f"estouros={result['batches_estourados_total']}/"
                    f"{result['n_batches_total']} | "
                    f"{result['tempo_batch_ms_medio']} ms/batch "
                    f"({time.perf_counter() - t0:.1f}s)"
                )

    # ---- Relatorio final ---------------------------------------------- #
    resumo = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "tokenizador": str(tok_dir),
        "dataset": {
            "csv": dados_etth1["csv"],
            "n_linhas_total": dados_etth1["n_linhas_total"],
            "n_linhas_usadas": dados_etth1["n_linhas"],
            "coluna": feat_name,
        },
    }
    out_json = salvar_resumo(resumo, tokens_info, resultados, pag)
    imprimir_tabela(resultados)

    if args.plots:
        for g in gerar_graficos(resultados, tokens_info):
            print(f"[ok] grafico: {g}")
    print(f"[ok] JSON: {out_json}")
    print("Benchmark por extenso do ETTh1 concluido.")


if __name__ == "__main__":
    main()
