"""
otimizar_ingles_extenso - Diagnostico e sintonia fina do Ingles por extenso.

Problema: o EN e 2x mais eficiente em tokens no GPT-2 (12.6 vs 26.5
tokens/ponto), mas o PT Compacto venceu o benchmark (Test MSE 6.75). Este
script investiga POR QUE a eficiencia nao virou vitoria e tenta quebrar a
barreira de 6.75 com 3 hipoteses:

    H1 - DENSIDADE DE CONTEXTO (lookback x2):
        O EN ocupa metade dos tokens por ponto; a mesma pagina de 128 tokens
        ja carrega ~2x pontos de historico. Os templates en_curto_x2 e
        en_keywords_x2 injetam DOIS pontos por frase, dobrando ainda mais o
        lookback dentro do MESMO orcamento absoluto de tokens da pagina PT
        vitoriosa. Loga-se quantos pontos o modelo enxerga nas 7 paginas.

    H2 - AUDITORIA LEXICAL/SINTATICA DA MATEMATICA (token drop):
        Log comparativo de tokenizacao e perplexidade do GPT-2 sobre os
        marcadores decimais: "point" (EN) vs "virgula" (PT) vs "coma" (ES),
        e do filtro de stopwords da estrategia keywords (remove o "point"?
        junta casas decimais de forma caotica?).

    H3 - ABLACAO DO MARCADOR DECIMAL ("sweet spot"):
        Duas variacoes hibridas EN rapidas:
            en_math_explicit : "July ten fourteen hours value twenty nine
                                point five"
            en_pseudo_digit  : "July ten fourteen hours value twenty nine
                                . five"   (ponto literal ancorando a atencao)

Pipeline: arquitetura VolLoss+SMETimes, 6 epocas por cenario, comparacao
focada com a baseline PT Keywords (6.75) em grafico de barras horizontais.

Saidas:
    resultados/otimizar_ingles_extenso.json
    resultados/graficos/etth1_ingles_otimizado_mse.png   (vs baseline 6.75)
    resultados/graficos/etth1_ingles_otimizado_densidade.png

Uso (na raiz do projeto):
    python -m testes_letras.otimizar_ingles_extenso
    python -m testes_letras.otimizar_ingles_extenso --epochs 5
    python -m testes_letras.otimizar_ingles_extenso --no-auditoria
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
from .run_etth1_texto import (
    _FEAT_PADRAO,
    carregar_etth1,
    construir_dataset,
    pool_janelas_etth1,
)
from .texto_puro import en_otimizado, extenso
from .texto_puro.paginacao import (
    ESTRATEGIAS,
    STOPWORDS_EN,
    TestadorPaginacao,
    _normalizar,
)
from .treino import build_loaders, rodar_convergencia

# Cenarios do benchmark de convergencia:
#   (template_id, [estrategias]) - "pt_keywords" e a baseline a ser batida.
_CENARIOS: Dict[str, List[str]] = {
    "pt_keywords": ["truncamento", "janela", "keywords"],       # baseline 6.75
    "en_curto": ["truncamento", "janela", "keywords"],          # H1 base
    "en_keywords": ["truncamento", "janela", "keywords"],       # H1 base
    "en_curto_x2": ["truncamento", "janela", "keywords"],       # H1 x2
    "en_keywords_x2": ["truncamento", "janela", "keywords"],    # H1 x2
    "en_math_explicit": ["truncamento", "janela"],              # H3
    "en_pseudo_digit": ["truncamento", "janela"],               # H3
}

# Frases decimais comparativas (mesmo valor 29.5 em cada idioma/formato).
_FRASES_DECIMAIS: Dict[str, str] = {
    "pt_virgula": "vinte e nove virgula cinco",
    "en_point": "twenty nine point five",
    "en_pseudo": "twenty nine . five",
    "es_coma": "veintinueve coma cinco",
}
_FRASES_PONTO_COMPLETO: Dict[str, str] = {
    "pt": "No dia dez de julho, as catorze horas, o valor registrado foi "
          "vinte e nove virgula cinco.",
    "en": "On July tenth, at fourteen hours, the recorded value was "
          "twenty nine point five.",
    "es": "El diez de julio, a las catorce horas, el valor registrado fue "
          "veintinueve coma cinco.",
}


# --------------------------------------------------------------------- #
#  CLI                                                                 #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostico e otimizacao do Ingles por extenso (ETTh1)"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=128)
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int, default=7)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--linhas", type=int, default=2000)
    parser.add_argument("--pool-janelas", dest="pool_janelas", type=int, default=192)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=None)
    parser.add_argument("--no-auditoria", dest="auditoria", action="store_false", default=True,
                        help="Pula a auditoria lexical/perplexidade (H2)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  H2 - Auditoria lexical/sintatica da matematica                       #
# --------------------------------------------------------------------- #
def _auditoria_tokenizacao(tokenizer) -> List[Dict]:
    """Tokenizacao (ids + subwords) das frases decimais por idioma."""
    linhas: List[Dict] = []
    for nome, frase in _FRASES_DECIMAIS.items():
        ids = tokenizer(frase, add_special_tokens=False)["input_ids"]
        sub = [tokenizer.convert_ids_to_tokens(i) for i in ids]
        linhas.append({
            "frase": frase,
            "tokens": ids,
            "subwords": sub,
            "n_tokens": len(ids),
        })
        print(f"  [{nome:<12s}] {len(ids):>3d} tokens: "
              + " | ".join(f"{s}" for s in sub))
    return linhas


def _auditoria_stopwords_en() -> Dict:
    """O filtro da estrategia keywords remove palavras da matematica EN?"""
    frase = "july tenth fourteen hours value twenty nine point five"
    removidas = [p for p in frase.split() if _normalizar(p) in STOPWORDS_EN]
    mantidas = [p for p in frase.split() if _normalizar(p) not in STOPWORDS_EN]
    print(f"  [stopwords EN] frase: '{frase}'")
    print(f"  [stopwords EN] removidas: {removidas if removidas else 'NENHUMA'}")
    print(f"  [stopwords EN] mantidas: {mantidas}")
    return {"frase": frase, "removidas": removidas, "mantidas": mantidas}


@torch.no_grad()
def _auditoria_perplexidade(tokenizer, tok_dir: Path, device) -> Dict:
    """Perplexidade GPT-2 (LM head) das frases decimais e pontos completos.

    PPL = exp(mean NLL). Quanto MENOR, mais 'nativo' o GPT-2 processa a
    frase (marcador decimal bem ancorado no vocabulario).
    """
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(str(tok_dir))
    lm.eval().to(device)

    def _ppl(frase: str) -> float:
        ids = tokenizer(frase, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = ids.to(device)
        out = lm(ids, labels=ids)
        nll = float(out.loss.item())
        return float(np.exp(nll))

    tabela: Dict = {}
    for nome, frase in {**_FRASES_DECIMAIS, **{f"ponto_{k}": v for k, v in
                                               _FRASES_PONTO_COMPLETO.items()}}.items():
        tabela[nome] = {"frase": frase, "perplexidade": _ppl(frase)}
    print(f"\n  [perplexidade GPT-2] (menor = mais nativo):")
    for nome, d in tabela.items():
        print(f"    {nome:<16s} ppl={d['perplexidade']:8.2f} | {d['frase'][:60]}")
    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tabela


# --------------------------------------------------------------------- #
#  H1 - Medicao de densidade (pontos no contexto de 7 paginas)          #
# --------------------------------------------------------------------- #
def _densidade(
    tokenizer, paginador: TestadorPaginacao, textos: Dict[str, str],
    tokens_por_ponto: Dict[str, float],
) -> Dict:
    """Loga tokens/ponto e pontos visiveis nas 7 paginas por template."""
    dens: Dict = {}
    print(f"\n{'Template':<16s} {'tok/ponto':>9s} {'pontos no contexto':>19s} "
          f"{'paginas':>8s}")
    print("-" * 70)
    for tid, texto in textos.items():
        n_tok_doc = len(tokenizer(texto, add_special_tokens=False)["input_ids"])
        dens[tid] = {"tokens_documento": n_tok_doc}
        for est in ESTRATEGIAS:
            lingua = "pt" if tid == "pt_keywords" else "en"
            r = paginador.paginar(est, texto, lingua)
            primeira_paginas = r["paginas"][: paginador.n_paginas_max]
            tok_ctx = sum(len(p) for p in primeira_paginas)
            pontos_ctx = tok_ctx / max(tokens_por_ponto[tid], 1e-9)
            dens[tid][f"{est}_tokens_contexto"] = tok_ctx
            dens[tid][f"{est}_pontos_contexto"] = round(pontos_ctx, 1)
            if est == "truncamento":
                print(f"  {tid:<14s} {tokens_por_ponto[tid]:>9.2f} "
                      f"{pontos_ctx:>19.1f} {r['n_paginas']:>8d}")
    return dens


# --------------------------------------------------------------------- #
#  Saidas (JSON + graficos)                                             #
# --------------------------------------------------------------------- #
def salvar_resumo(resumo: Dict, dens: Dict, auditoria: Dict, resultados: Dict) -> Path:
    resumo["densidade"] = dens
    resumo["auditoria"] = auditoria
    resumo["resultados"] = {
        tid: {est: r for est, r in res.items()} for tid, res in resultados.items()
    }
    out = caminhos.RESULTADOS_DIR / "otimizar_ingles_extenso.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    return out


def imprimir_tabela(resultados: Dict) -> None:
    print("\n" + "=" * 100)
    print("OTIMIZACAO DO INGLES POR EXTENSO (Test MSE final, vs baseline 6.75)")
    print("=" * 100)
    linhas = sorted(
        (
            (r["teste_final_mse_pred"], tid, est, r["melhor_val_mse_pred"],
             r["batches_estourados_total"])
            for tid, res in resultados.items()
            for est, r in res.items()
        ),
        key=lambda t: t[0],
    )
    print(f"{'#':>3s} {'Estrategia':<13s} {'Template':<16s} {'test_mse':>10s} "
          f"{'best_val':>10s} {'estouros':>8s} {'vs 6.75':>9s}")
    print("-" * 100)
    for i, (mse, tid, est, bv, estouros) in enumerate(linhas, 1):
        delta = mse - 6.7512
        marca = f"{delta:+.2f}"
        print(f"{i:>3d} {est:<13s} {tid:<16s} {mse:>10.4f} {bv:>10.4f} "
              f"{estouros:>8d} {marca:>9s}")
    print("=" * 100)


def gerar_graficos(resultados: Dict, dens: Dict) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caminhos.criar_resultados()
    arquivos: List[Path] = []

    # 1) Barras horizontais: todos os cenarios vs baseline PT (6.75).
    linhas = sorted(
        (
            (r["teste_final_mse_pred"], tid, est)
            for tid, res in resultados.items()
            for est, r in res.items()
        ),
        key=lambda t: t[0],
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(linhas))
    rotulos = [f"{tid} | {est}" for _, tid, est in linhas]
    vals = [t[0] for t in linhas]
    cores = ["#d95f0e" if tid.startswith("pt") else "#2c7fb8" for _, tid, _ in linhas]
    ax.barh(y, vals, color=cores, alpha=0.9)
    for yi, v in zip(y, vals):
        ax.text(v + 0.04, yi, f"{v:.2f}", va="center", fontsize=7)
    ax.axvline(6.7512, color="crimson", ls="--", lw=1.5,
               label="Baseline PT Keywords (6.75) - a bater")
    ax.set_yticks(y)
    ax.set_yticklabels(rotulos, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Test MSE no horizonte (final)")
    ax.set_title("Otimizacao do Ingles por extenso: rank vs baseline PT 6.75 "
                 "(laranja=PT | azul=EN)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "etth1_ingles_otimizado_mse.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    arquivos.append(p)

    # 2) Densidade: pontos de historico visiveis nas 7 paginas (H1).
    tids = list(dens.keys())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals_trunc = [dens[t]["truncamento_pontos_contexto"] for t in tids]
    vals_janela = [dens[t]["janela_pontos_contexto"] for t in tids]
    x = np.arange(len(tids))
    ax.bar(x - 0.18, vals_trunc, 0.36, label="truncamento (1 pagina)",
           alpha=0.85)
    ax.bar(x + 0.18, vals_janela, 0.36, label="janela (7 paginas)", alpha=0.85)
    for xi, (a, b) in enumerate(zip(vals_trunc, vals_janela)):
        ax.text(xi - 0.18, a + 1, f"{a:.0f}", ha="center", fontsize=7)
        ax.text(xi + 0.18, b + 1, f"{b:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tids, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Pontos de historico no contexto do modelo")
    ax.set_title("H1 - Densidade: quantos pontos o modelo enxerga "
                 "(mesmo orcamento absoluto de tokens)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "etth1_ingles_otimizado_densidade.png"
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

    # ================================================================= #
    # FASE A - AUDITORIA LEXICAL/SINTATICA (H2)                         #
    # ================================================================= #
    auditoria: Dict = {}
    if args.auditoria:
        print("\n" + "=" * 100)
        print("FASE A - AUDITORIA LEXICAL DA MATEMATICA (token drop / H2)")
        print("=" * 100)
        print("  Tokenizacao das frases decimais (29.5) por idioma:")
        auditoria["tokenizacao"] = _auditoria_tokenizacao(tokenizer)
        auditoria["stopwords_en"] = _auditoria_stopwords_en()
        auditoria["perplexidade"] = _auditoria_perplexidade(tokenizer, tok_dir, device)

    # ================================================================= #
    # FASE B - DENSIDADE DE CONTEXTO (H1)                               #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE B - DENSIDADE DE CONTEXTO: LOOKBACK POR TEMPLATE (H1)")
    print("=" * 100)
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][_FEAT_PADRAO]
    print(f"[csv] {dados_etth1['csv']} | subconjunto de {dados_etth1['n_linhas']} "
          f"linhas (coluna '{feat_name}')")

    textos: Dict[str, str] = {}
    tokens_por_ponto: Dict[str, float] = {}
    for tid in _CENARIOS:
        if tid in extenso.TEMPLATES_EXTENSO:
            doc = extenso.serie_para_texto_extenso(
                dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
                0, dados_etth1["n_linhas"], tid,
            )
        else:
            spec = en_otimizado.TEMPLATES_EN_OTIMIZADO[tid]
            if spec["x2"]:
                doc = en_otimizado.serie_para_texto_x2(
                    dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
                    0, dados_etth1["n_linhas"], spec["fn"],
                )
            else:
                doc = en_otimizado.serie_para_texto_simples(
                    dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
                    0, dados_etth1["n_linhas"], spec["fn"],
                )
        textos[tid] = doc
        n_tok = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        tokens_por_ponto[tid] = n_tok / dados_etth1["n_linhas"]

    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    dens = _densidade(tokenizer, paginador, textos, tokens_por_ponto)
    for tid in en_otimizado.TEMPLATES_EN_OTIMIZADO:
        if tid not in textos:
            continue
        spec = en_otimizado.TEMPLATES_EN_OTIMIZADO[tid]
        ex = (
            en_otimizado.serie_para_texto_simples(
                dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
                0, 1, spec["fn"],
            ) if not spec["x2"] else
            en_otimizado.serie_para_texto_x2(
                dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
                0, 2, spec["fn"],
            )
        )
        print(f"  [exemplo {tid}] {ex}")
    print(f"  [tok/ponto] pt_keywords={tokens_por_ponto['pt_keywords']:.2f} | "
          f"en_curto={tokens_por_ponto['en_curto']:.2f} | "
          f"en_keywords={tokens_por_ponto['en_keywords']:.2f} | "
          f"en_curto_x2={tokens_por_ponto['en_curto_x2']:.2f} | "
          f"en_keywords_x2={tokens_por_ponto['en_keywords_x2']:.2f}")

    # ================================================================= #
    # FASE C - CONVERGENCIA (VolLoss + SMETimes)                        #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE C - CONVERGENCIA DOS CENARIOS DE OTIMIZACAO (6 epocas)")
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
    print(f"[pool] {len(pool)} janelas numericas de {feat_name}")

    paginados: Dict[str, Dict] = {}
    for tid in _CENARIOS:
        lingua = "pt" if tid == "pt_keywords" else "en"
        for est in _CENARIOS[tid]:
            r = paginador.paginar(est, textos[tid], lingua)
            r.update(paginador.para_modelo(r))
            paginados[f"{est}__{tid}"] = r

    resultados: Dict[str, Dict] = {}
    n_amostras = len(pool)
    n_tr, n_va = int(n_amostras * 0.6), int(n_amostras * 0.2)

    for chave, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")
        for tid, ests in _CENARIOS.items():
            for est in ests:
                # Modelo REMONTADO do checkpoint a cada cenario (isolamento:
                # pesos do checkpoint + fusion(seed) identicos; sem heranca
                # de treinos anteriores contaminando a comparacao).
                rb.seed_everything(args.seed)
                model, info = montar_modelo(ckpt_path, device, use_amp=False)
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
                    use_amp=False, seed=args.seed,
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
            "n_linhas": dados_etth1["n_linhas"],
            "coluna": feat_name,
        },
        "baseline_a_bater": {"template": "pt_keywords", "test_mse": 6.7512},
        "tokens_por_ponto": tokens_por_ponto,
    }
    out_json = salvar_resumo(resumo, dens, auditoria, resultados)
    imprimir_tabela(resultados)

    if args.plots:
        for g in gerar_graficos(resultados, dens):
            print(f"[ok] grafico: {g}")
    print(f"[ok] JSON: {out_json}")
    print("Diagnostico e otimizacao do ingles por extenso concluido.")


if __name__ == "__main__":
    main()
