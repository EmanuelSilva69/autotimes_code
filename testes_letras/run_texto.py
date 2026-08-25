"""
run_texto - Orquestrador do Teste de Tokenizacao e Paginacao de TEXTOS PUROS.

Extende o testador testes_letras para o dominio textual puro (logs extensos,
descricoes, noticias), seguindo as diretrizes:

    FASE A - INFLACAO DE TOKENS (PT vs EN):
        mesmos documentos em pt-BR e ingles -> taxa de inflacao do GPT-2.

    FASE B - PAGINACAO (chunking), 3 estrategias:
        truncamento (hard truncation) | janela deslizante c/ overlap |
        compressao por palavras-chave (stopwords). Reporta tokens por
        documento, paginas necessarias e tempo de tokenizacao.

    FASE C - ESTABILIDADE DA LOSS:
        as paginas alimentam o ramo textual da MESMA arquitetura aprimorada
        (VolLoss + SMETimes) sobre janelas numericas reais do ETTh1; mede-se
        convergencia (MSE/MAE), tempo por batch e batches estourados.

Uso (na raiz do projeto):
    python -m testes_letras.run_texto                                  # tudo
    python -m testes_letras.run_texto --estrategias janela,keywords
    python -m testes_letras.run_texto --checkpoints smetimes --idiomas pt,en
    python -m testes_letras.run_texto --texto-pt meu_log.txt --texto-en log_en.txt
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .dados import LABEL_LEN, PRED_LEN, SEQ_LEN
from .modelos import CKPTS, montar_modelo
from .treino import build_loaders, rodar_convergencia
from .texto_puro import amostras, idioma, relatorio
from .texto_puro.paginacao import ESTRATEGIAS, TestadorPaginacao

_IDIOMAS_DISPONIVEIS = ["pt", "en"]


# --------------------------------------------------------------------- #
#  CLI                                                                 #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testador de Tokenizacao e Paginacao de textos puros (PT vs EN)"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes",
                        help="Modelos: 'smetimes' (VolLoss+SMETimes), 'amd' ou caminhos .pt")
    parser.add_argument("--estrategias", type=str, default="truncamento,janela,keywords",
                        help="Estrategias de paginacao (separadas por virgula)")
    parser.add_argument("--idiomas", type=str, default="pt,en",
                        help="Idiomas a testar: 'pt,en' ou apenas 'pt'")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=128,
                        help="Limite de tokens por pagina (contexto do bloco)")
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int, default=7,
                        help="Paginas por janela (seq_len // token_len = 7)")
    parser.add_argument("--overlap", type=float, default=0.15,
                        help="Sobreposicao da janela deslizante (10% a 20%)")
    parser.add_argument("--amostras-por-doc", dest="amostras_por_doc", type=int, default=6,
                        help="Quantas janelas numericas cada documento alimenta")
    parser.add_argument("--pool-janelas", dest="pool_janelas", type=int, default=96,
                        help="Tamanho do pool de janelas numericas do ETTh1")
    parser.add_argument("--texto-pt", dest="texto_pt", type=str, default=None,
                        help="Arquivo com documentos PT (1 por linha); default: built-in")
    parser.add_argument("--texto-en", dest="texto_en", type=str, default=None,
                        help="Arquivo com documentos EN (1 por linha)")
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=None,
                        help="Override do diretorio do tokenizador local")
    parser.add_argument("--amp", action="store_true",
                        help="Liga AMP fp16 (instavel com fusion aleatorio; default FP32+TF32)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Pool de janelas numericas reais (feat 0 do ETTh1)                    #
# --------------------------------------------------------------------- #
def _pool_janelas_numericas(dados: Dict, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """n janelas (x, y) univariadas da regiao de treino, espacadas por stride."""
    values = dados["values"]
    b1, b2 = dados["b1s"][0], dados["b2s"][0]
    tot = len(values[b1:b2]) - SEQ_LEN - PRED_LEN + 1
    stride = max(1, tot // max(n, 1))
    pool: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, tot, stride):
        if len(pool) >= n:
            break
        s_end = b1 + i + SEQ_LEN
        r_begin = s_end - LABEL_LEN
        r_end = r_begin + LABEL_LEN + PRED_LEN
        pool.append((
            values[b1 + i: b1 + i + SEQ_LEN, 0:1],
            values[r_begin:r_end, 0:1],
        ))
    return pool


def construir_dataset(
    resultados_paginacao: List[Dict],
    pool: List[Tuple[np.ndarray, np.ndarray]],
    amostras_por_doc: int,
) -> TensorDataset:
    """Monta o dataset de estabilidade para uma (estrategia, idioma).

    Cada documento pago alimenta `amostras_por_doc` janelas numericas
    distintas do pool (o ramo numerico fornece o alvo de reconstrucao;
    o ramo textual recebe as paginas do texto puro).
    """
    xs, ids_all, lens_all, ys = [], [], [], []
    n_docs = max(len(resultados_paginacao), 1)
    n_pool = len(pool)
    for d, r in enumerate(resultados_paginacao):
        for k in range(amostras_por_doc):
            win = (d * n_pool // n_docs + k * 3) % n_pool
            xw, yw = pool[win]
            xs.append(xw)
            ys.append(yw)
            ids_all.append(r["ids"])
            lens_all.append(r["lens"])
    return TensorDataset(
        torch.from_numpy(np.stack(xs)).float(),
        torch.tensor(ids_all, dtype=torch.int64),
        torch.tensor(lens_all, dtype=torch.int64),
        torch.from_numpy(np.stack(ys)).float(),
    )


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

    # ---- Tokenizador local (explicito: Testes_eficiencia primeiro) ---- #
    tok_dir = caminhos.descobrir_tokenizador(args.tokenizer_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,})")

    # ---- Amostras e config -------------------------------------------- #
    pares = amostras.carregar_pares(args.texto_pt, args.texto_en)
    estrategias = [e.strip() for e in args.estrategias.split(",") if e.strip()]
    for e in estrategias:
        if e not in ESTRATEGIAS:
            raise SystemExit(f"Estrategia desconhecida: {e}. Validas: {list(ESTRATEGIAS)}")
    idiomas = [i.strip() for i in args.idiomas.split(",") if i.strip()]
    for i in idiomas:
        if i not in _IDIOMAS_DISPONIVEIS:
            raise SystemExit(f"Idioma desconhecido: {i}")
    print(f"[amostras] {len(pares)} documentos (PT+EN equivalentes) | "
          f"estrategias={estrategias} | idiomas={idiomas}")

    # ================================================================= #
    # FASE A - INFLACAO DE TOKENS PT vs EN                              #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE A - AVALIACAO DE IDIOMA: INFLACAO DE TOKENS (PT vs EN)")
    print("=" * 100)
    inflacao = idioma.medir_inflacao_tokenizacao(pares, tokenizer)
    relatorio.imprimir_inflacao(inflacao)

    # ================================================================= #
    # FASE B - PAGINACAO (chunking)                                     #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE B - TESTADOR DE PAGINACAO (chunking strategies)")
    print("=" * 100)
    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    pag: Dict = {e: {} for e in estrategias}
    paginados: Dict[str, List[Dict]] = {}  # (estrategia__idioma) -> por documento

    for est in estrategias:
        for lingua in idiomas:
            res_docs: List[Dict] = []
            for nome, pt, en in pares:
                texto = pt if lingua == "pt" else en
                r = paginador.paginar(est, texto, lingua)
                r["documento"] = nome
                r.update(paginador.para_modelo(r))  # ids/lens prontos p/ o modelo
                res_docs.append(r)
            paginados[f"{est}__{lingua}"] = res_docs
            agregado = {
                "n_documentos": len(res_docs),
                "tokens_originais_med": float(np.mean([r["tokens_originais"] for r in res_docs])),
                "tokens_retidos_med": float(np.mean([r["tokens_retidos"] for r in res_docs])),
                "tokens_descartados_med": float(np.mean([r["tokens_descartados"] for r in res_docs])),
                "paginas_med": float(np.mean([r["n_paginas"] for r in res_docs])),
                "n_docs_excedem_contexto": sum(
                    1 for r in res_docs if r["n_paginas"] > args.n_paginas_max
                ),
                "tempo_tokenizacao_doc_s": float(
                    np.mean([r["tempo_tokenizacao_s"] for r in res_docs])
                ),
                "sobreposicao_frac": res_docs[0]["sobreposicao_frac"],
            }
            pag[est][lingua] = agregado
            print(
                f"  [{ESTRATEGIAS[est]['nome']:<28s}] {lingua}: "
                f"{agregado['tokens_originais_med']:.0f} -> "
                f"{agregado['tokens_retidos_med']:.0f} tokens | "
                f"{agregado['paginas_med']:.1f} paginas | "
                f"{agregado['n_docs_excedem_contexto']} docs excedem o contexto"
            )
    relatorio.imprimir_paginacao(pag)

    # ================================================================= #
    # FASE C - ESTABILIDADE DA LOSS (arquitetura VolLoss + SMETimes)    #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE C - ESTABILIDADE DA LOSS (texto puro na arquitetura aprimorada)")
    print("=" * 100)
    dados_numericos = None
    pool = None
    ckpts_sel: Dict[str, str] = {}
    for tok in args.checkpoints.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in CKPTS:
            ckpts_sel[tok] = str(CKPTS[tok])
        else:
            from pathlib import Path

            p = Path(tok)
            if not p.exists():
                raise SystemExit(f"Checkpoint nao encontrado: {p}")
            ckpts_sel[p.stem] = str(p)

    estabilidade: Dict[str, Dict] = {}
    modelos_ordem: List[str] = []

    for chave, ckpt_path in ckpts_sel.items():
        modelos_ordem.append(chave)
        from pathlib import Path as _Path

        print(f"\n--- Modelo: {chave} ({_Path(ckpt_path).name}) ---")

        # Pool de janelas numericas do ETTh1 (1x por modelo).
        if pool is None:
            from .dados import carregar_dados

            dados_numericos = carregar_dados()
            pool = _pool_janelas_numericas(dados_numericos, args.pool_janelas)
            print(f"  [pool] {len(pool)} janelas numericas (feat 0, regiao de treino)")

        for est in estrategias:
            for lingua in idiomas:
                # Modelo REMONTADO do checkpoint a cada cenario (isolamento:
                # sem heranca de treinos anteriores entre cenarios).
                rb.seed_everything(args.seed)
                model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
                print(f"  [mount] {ESTRATEGIAS[est]['nome']} ({lingua}) | "
                      f"flags={info['flags']} | treinaveis={info['params_treinaveis']:,}")
                chave_combo = f"{chave}__{est}__{lingua}"
                res_docs = paginados[f"{est}__{lingua}"]
                ds = construir_dataset(res_docs, pool, args.amostras_por_doc)
                n = len(ds)
                n_tr, n_va = int(n * 0.6), int(n * 0.2)
                tr, va, te = (
                    TensorDataset(*[t[:n_tr] for t in ds.tensors]),
                    TensorDataset(*[t[n_tr:n_tr + n_va] for t in ds.tensors]),
                    TensorDataset(*[t[n_tr + n_va:] for t in ds.tensors]),
                )
                train_loader, val_loader, test_loader = build_loaders(
                    tr, va, te, args.batch_size
                )
                print(f"\n  [{ESTRATEGIAS[est]['nome']} ({lingua})] "
                      f"{len(ds)} amostras (train={len(tr)} val={len(va)} test={len(te)})")
                t0 = time.perf_counter()
                result = rodar_convergencia(
                    model, train_loader, val_loader, test_loader,
                    epochs=args.epochs, lr=args.lr, device=device,
                    use_amp=use_amp, seed=args.seed,
                )
                result["estrategia"] = est
                result["idioma"] = lingua
                tempos = result["historico"]["tempo_batch_ms"]
                if len(tempos) > 2:
                    tempos = tempos[1:]  # exclui ep1 (warmup kernels ROCm)
                result["tempo_batch_ms_medio"] = round(sum(tempos) / max(len(tempos), 1), 2)
                result["tokens_originais_med"] = pag[est][lingua]["tokens_originais_med"]
                result["paginas_med"] = pag[est][lingua]["paginas_med"]
                estabilidade[chave_combo] = result
                print(
                    f"  [ok] best_val={result['melhor_val_mse_pred']:.4f} | "
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
        "inflacao_tokens": inflacao,
        "paginacao": pag,
        "estabilidade": estabilidade,
    }
    out_json = relatorio.salvar_resumo(resumo)
    relatorio.imprimir_estabilidade(estabilidade, modelos_ordem)

    if args.plots:
        gerados = relatorio.gerar_graficos(pag, inflacao, estabilidade, modelos_ordem)
        for g in gerados:
            print(f"[ok] grafico: {g}")
    print(f"[ok] JSON: {out_json}")
    print("Teste de tokenizacao e paginacao de texto puro concluido.")


if __name__ == "__main__":
    main()
