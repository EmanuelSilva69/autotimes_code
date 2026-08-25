"""
run_etth1_texto - Benchmark de paginacao textual sobre o dataset REAL ETTh1.

Aplica as logicas de paginacao validadas em texto puro (Truncamento Direto,
Janela + Overlap e Compressao por Palavras-Chave) ao texto gerado a partir
das series temporais reais do AutoTimes (dataset/ETT-small/ETTh1.csv):

    FASE A - CONVERSAO DA SERIE PARA TEXTO:
        Um subconjunto significativo de linhas do CSV e convertido em dois
        formatos (mesma serie, dois idiomas):
            PT Completo: "Dia 10 de Julho, as 14 horas, o valor era 29.5..."
            EN Curto   : "Jul 10, 14:00, value 29.5..."
        Mede-se quantos tokens cada formato gera no GPT-2 (inflacao PT/EN).

    FASE B - PAGINACAO (chunking), 3 estrategias:
        O documento textual completo alimenta o TestadorPaginacao:
        truncamento (hard truncation) | janela deslizante c/ overlap |
        compressao por palavras-chave. Reporta tokens retidos, paginas,
        descarte e paginas descartadas (alem do contexto do modelo).

    FASE C - CONVERGENCIA NO MODELO REAL (VolLoss + SMETimes):
        Para cada (estrategia, idioma), as paginas alimentam o ramo textual
        da arquitetura aprimorada junto com janelas numericas reais da MESMA
        regiao do CSV. Roda 5-6 epocas (Adam + CosineAnnealing) e compara a
        retencao da tendencia original dos numeros via MSE/MAE final.

Saidas:
    resultados/benchmark_etth1_texto.json   (todos os numeros)
    resultados/graficos/etth1_texto_convergencia.png
    resultados/graficos/etth1_texto_mse_final.png

Uso (na raiz do projeto):
    python -m testes_letras.run_etth1_texto
    python -m testes_letras.run_etth1_texto --linhas 3000 --epochs 6
    python -m testes_letras.run_etth1_texto --estrategias janela,keywords
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .dados import LABEL_LEN, PRED_LEN, SEQ_LEN
from .modelos import CKPTS, montar_modelo
from .treino import build_loaders, rodar_convergencia
from .texto_puro.paginacao import ESTRATEGIAS, TestadorPaginacao

_MESES_PT: List[str] = [
    "Janeiro", "Fevereiro", "Marco", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
]
_MESES_EN: List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_FEAT_PADRAO: int = 0


# --------------------------------------------------------------------- #
#  Conversao serie temporal -> texto (PT Completo / EN Curto)           #
# --------------------------------------------------------------------- #
def formatar_pt_completo(dt: datetime, valor: float) -> str:
    """Ponto da serie em PT por extenso: 'Dia 10 de Julho, as 14 horas,
    o valor era 29.5'."""
    mes = _MESES_PT[dt.month - 1]
    hora = f"as {dt.hour} horas" if dt.minute == 0 else f"as {dt.hour} horas e {dt.minute} minutos"
    return f"Dia {dt.day} de {mes}, {hora}, o valor era {valor:.1f}"


def formatar_en_curto(dt: datetime, valor: float) -> str:
    """Ponto da serie em ingles compacto: 'Jul 10, 14:00, value 29.5'."""
    mes = _MESES_EN[dt.month - 1]
    return f"{mes} {dt.day:02d}, {dt.hour:02d}:{dt.minute:02d}, value {valor:.1f}"


def serie_para_texto(
    dt_list: List[datetime],
    values: np.ndarray,
    feat: int,
    inicio: int,
    fim: int,
    lingua: str,
    feat_name: str,
) -> str:
    """Converte as linhas [inicio:fim] da serie em UM documento textual.

    Args:
        dt_list: datetimes do CSV (tamanho total).
        values: valores numericos (n_total, n_vars).
        feat: coluna de interesse (default 0, igual ao pipeline).
        inicio/fim: intervalo de linhas do subconjunto extraido.
        lingua: 'pt' (Completo) ou 'en' (Curto).
        feat_name: nome da coluna (cabecalho do documento).

    Returns:
        Documento textual da serie (uma frase por ponto, separadas por '; ').
    """
    fmt = formatar_pt_completo if lingua == "pt" else formatar_en_curto
    partes: List[str] = []
    for i in range(inicio, min(fim, values.shape[0])):
        partes.append(fmt(dt_list[i], float(values[i, feat])))
    return f"Serie {feat_name}: " + "; ".join(partes) + "."


# --------------------------------------------------------------------- #
#  Leitura do ETTh1 + pool de janelas numericas da mesma regiao         #
# --------------------------------------------------------------------- #
def carregar_etth1(
    csv_path: Optional[str] = None, linhas: int = 2000
) -> Dict:
    """Carrega o CSV real e extrai o subconjunto significativo (train).

    Args:
        csv_path: caminho do CSV (default dataset/ETT-small/ETTh1.csv).
        linhas: quantas linhas iniciais (regiao de treino) sao usadas.

    Returns:
        Dict com dt_list (subconjunto), values (completo), feat_names,
        linhas_usadas, csv, n_linhas_total.

    Raises:
        FileNotFoundError: CSV ausente.
        ValueError: subconjunto vazio ou maior que o total.
    """
    csv = Path(csv_path or caminhos.DATA_DIR / caminhos.DATA_PATH)
    if not csv.exists():
        raise FileNotFoundError(
            f"Dataset ETTh1 nao encontrado: {csv} (use --csv para apontar outro)"
        )
    df = pd.read_csv(csv)
    n_total = len(df)
    if not 0 < linhas <= n_total:
        raise ValueError(f"linhas={linhas} invalido para CSV com {n_total} linhas")

    dt_full = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])).to_pydatetime().tolist()
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feat_names = list(df.columns[1:])
    return {
        "dt_list": dt_full[:linhas],
        "values": values,
        "feat_names": feat_names,
        "n_linhas": linhas,
        "n_linhas_total": n_total,
        "csv": str(csv),
    }


def pool_janelas_etth1(
    dados: Dict, n: int, feat: int = _FEAT_PADRAO, inicio: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """n janelas (x, y) univariadas amostradas dentro do subconjunto.

    As janelas vêm da MESMA regiao que originou o texto (inicio .. linhas),
    para que o contexto textual pague a serie que o ramo numerico ve.
    """
    values = dados["values"]
    fim = dados["n_linhas"]
    tot = max(fim - inicio - SEQ_LEN - PRED_LEN + 1, 0)
    if tot <= 0:
        raise ValueError(
            f"Subconjunto ({fim} linhas) menor que a janela necessaria "
            f"(seq {SEQ_LEN} + pred {PRED_LEN})"
        )
    stride = max(1, tot // max(n, 1))
    pool: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, tot, stride):
        if len(pool) >= n:
            break
        s_end = inicio + i + SEQ_LEN
        r_begin = s_end - LABEL_LEN
        r_end = r_begin + LABEL_LEN + PRED_LEN
        pool.append((
            values[inicio + i: inicio + i + SEQ_LEN, feat: feat + 1],
            values[r_begin: r_end, feat: feat + 1],
        ))
    return pool


def construir_dataset(
    resultado_paginacao: Dict,
    pool: List[Tuple[np.ndarray, np.ndarray]],
) -> TensorDataset:
    """Dataset textual (x, ids, lens, y) para uma (estrategia, idioma).

    O MESMO documento paginado acompanha todas as janelas numericas do pool
    (o contexto textual e fixo; o alvo vem do ramo numerico real).
    """
    ids = resultado_paginacao["ids"]
    lens = resultado_paginacao["lens"]
    xs, ys = [], []
    for xw, yw in pool:
        xs.append(xw)
        ys.append(yw)
    return TensorDataset(
        torch.from_numpy(np.stack(xs)).float(),
        torch.tensor([ids] * len(pool), dtype=torch.int64),
        torch.tensor([lens] * len(pool), dtype=torch.int64),
        torch.from_numpy(np.stack(ys)).float(),
    )


# --------------------------------------------------------------------- #
#  Relatorio (JSON + tabela + graficos)                                 #
# --------------------------------------------------------------------- #
def salvar_benchmark(resumo: Dict, nome: str = "benchmark_etth1_texto.json") -> Path:
    """Salva o JSON de um benchmark ETTh1 (nome default do benchmark texto)."""
    out = caminhos.RESULTADOS_DIR / nome
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    return out


def imprimir_tabela(resultados: Dict, modelos_ordem: List[str]) -> None:
    print("\n" + "=" * 100)
    print("CONVERGENCIA NO ETTh1 REAL (texto paginado da propria serie)")
    print("=" * 100)
    hdr = (f"{'Modelo/Estrategia':<36s} {'Idioma':>6s} {'best_val':>10s} "
           f"{'test_mse':>10s} {'test_mae':>10s} {'estouros':>8s} {'ms/batch':>9s}")
    print(hdr)
    print("-" * 100)
    for mk in modelos_ordem:
        for estr in ESTRATEGIAS:
            for lingua in ("pt", "en"):
                chave = f"{mk}__{estr}__{lingua}"
                if chave not in resultados:
                    continue
                r = resultados[chave]
                print(
                    f"  {ESTRATEGIAS[estr]['nome'] + ' (' + mk + ')':<34s} {lingua:>6s} "
                    f"{r['melhor_val_mse_pred']:>10.4f} {r['teste_final_mse_pred']:>10.4f} "
                    f"{r['teste_final_mae_pred']:>10.4f} {r['batches_estourados_total']:>8d} "
                    f"{r['tempo_batch_ms_medio']:>9.2f}"
                )
    print("=" * 100)


def gerar_graficos(resultados: Dict, modelos_ordem: List[str]) -> List[Path]:
    """PNGs: curvas de convergencia por estrategia + MSE final (barras)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caminhos.criar_resultados()
    arquivos: List[Path] = []
    estrategias = [e for e in ESTRATEGIAS if any(
        f"{m}__{e}__" in k for m in modelos_ordem for k in resultados
    )]
    labels = [ESTRATEGIAS[e]["nome"] for e in estrategias]

    # 1) Convergencia: val_mse_pred por epoca, uma linha por combo.
    fig, ax = plt.subplots(figsize=(9, 5))
    for mk in modelos_ordem:
        for estr in estrategias:
            for lingua in ("pt", "en"):
                chave = f"{mk}__{estr}__{lingua}"
                if chave not in resultados:
                    continue
                hist = resultados[chave]["historico"]
                ax.plot(hist["epoch"], hist["val_mse_pred"], marker="o", ms=4,
                        label=f"{ESTRATEGIAS[estr]['nome']} ({lingua})")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE no horizonte (validacao)")
    ax.set_title(f"Convergencia no ETTh1 real (texto paginado da serie) - {mk}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "etth1_texto_convergencia.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    arquivos.append(p)

    # 2) MSE final de teste por estrategia/idioma (retencao da tendencia).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(estrategias))
    largura = 0.35
    for i, lingua in enumerate(("pt", "en")):
        vals = []
        for estr in estrategias:
            chave = f"{modelos_ordem[0]}__{estr}__{lingua}"
            vals.append(resultados[chave]["teste_final_mse_pred"] if chave in resultados else 0.0)
        ax.bar(x + i * largura, vals, largura, label=f"teste {lingua}")
        for xi, val in zip(x, vals):
            ax.text(xi + i * largura, val + 0.3, f"{val:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x + largura / 2)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("MSE no horizonte (teste final)")
    ax.set_title("ETTh1 real: retencao da tendencia apos conversao e paginacao")
    ax.legend()
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "etth1_texto_mse_final.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    arquivos.append(p)
    return arquivos


# --------------------------------------------------------------------- #
#  CLI                                                                 #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark de paginacao textual sobre o dataset real ETTh1"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes",
                        help="Modelos: 'smetimes', 'amd' ou caminhos .pt")
    parser.add_argument("--estrategias", type=str, default="truncamento,janela,keywords",
                        help="Estrategias de paginacao (separadas por virgula)")
    parser.add_argument("--idiomas", type=str, default="pt,en",
                        help="Formatos da serie: 'pt' (Completo) e/ou 'en' (Curto)")
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
    except Exception as e:  # noqa: BLE001 - caminho pode ser invalido
        print(f"[erro] falha ao carregar o tokenizador de {tok_dir}: {e}")
        raise SystemExit(1)
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,})")

    # ---- Validacao de argumentos -------------------------------------- #
    estrategias = [e.strip() for e in args.estrategias.split(",") if e.strip()]
    for e in estrategias:
        if e not in ESTRATEGIAS:
            raise SystemExit(f"Estrategia desconhecida: {e}. Validas: {list(ESTRATEGIAS)}")
    idiomas = [i.strip() for i in args.idiomas.split(",") if i.strip()]
    for i in idiomas:
        if i not in ("pt", "en"):
            raise SystemExit(f"Idioma/Formato desconhecido: {i} (use 'pt' e/ou 'en')")

    # ================================================================= #
    # FASE A - CONVERSAO DA SERIE REAL PARA TEXTO                       #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE A - CONVERSAO DA SERIE ETTh1 PARA TEXTO (PT Completo / EN Curto)")
    print("=" * 100)
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][_FEAT_PADRAO]
    print(f"[csv] {dados_etth1['csv']} | {dados_etth1['n_linhas_total']} linhas "
          f"-> subconjunto de {dados_etth1['n_linhas']} linhas "
          f"(coluna '{feat_name}')")

    textos: Dict[str, str] = {}
    tokens_formato: Dict[str, int] = {}
    for lingua in idiomas:
        doc = serie_para_texto(
            dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
            0, dados_etth1["n_linhas"], lingua, feat_name,
        )
        n_tok = len(tokenizer(doc, add_special_tokens=False)["input_ids"])
        textos[lingua] = doc
        tokens_formato[lingua] = n_tok
        print(f"  [{lingua}] {n_tok:,} tokens "
              f"({len(doc):,} caracteres) | ex.: "
              f"{serie_para_texto(dados_etth1['dt_list'], dados_etth1['values'], _FEAT_PADRAO, 5, 6, lingua, feat_name)}")
    if len(idiomas) == 2 and tokens_formato["en"] > 0:
        print(f"  [inflacao] PT Completo / EN Curto = "
              f"x{tokens_formato['pt'] / tokens_formato['en']:.2f}")

    # ================================================================= #
    # FASE B - PAGINACAO (chunking) sobre o texto da serie               #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE B - PAGINACAO (chunking) do texto da serie ETTh1")
    print("=" * 100)
    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    paginados: Dict[str, Dict] = {}
    pag: Dict = {e: {} for e in estrategias}
    for est in estrategias:
        for lingua in idiomas:
            r = paginador.paginar(est, textos[lingua], lingua)
            r.update(paginador.para_modelo(r))  # ids/lens prontos p/ o modelo
            paginados[f"{est}__{lingua}"] = r
            pag[est][lingua] = {
                "n_documentos": 1,
                "tokens_originais_med": float(r["tokens_originais"]),
                "tokens_retidos_med": float(r["tokens_retidos"]),
                "tokens_descartados_med": float(r["tokens_descartados"]),
                "paginas_med": float(r["n_paginas"]),
                "paginas_descartadas": int(r.get("paginas_descartadas", 0)),
                "sobreposicao_frac": r["sobreposicao_frac"],
                "tempo_tokenizacao_doc_s": r["tempo_tokenizacao_s"],
            }
            print(
                f"  [{ESTRATEGIAS[est]['nome']:<28s}] {lingua}: "
                f"{r['tokens_originais']:,} -> {r['tokens_retidos']:,} tokens | "
                f"{r['n_paginas']} paginas (descartadas: {r.get('paginas_descartadas', 0)})"
            )

    print("\n" + "=" * 100)
    print("PAGINACAO (chunking) - documento unico do ETTh1")
    print("=" * 100)
    hdr = (f"{'Estrategia':<30s} {'Formato':>7s} {'tokens orig':>10s} "
           f"{'tokens ret':>10s} {'paginas':>8s} {'descartadas':>12s}")
    print(hdr)
    print("-" * 100)
    for est in estrategias:
        for lingua in idiomas:
            s = pag[est][lingua]
            print(
                f"{ESTRATEGIAS[est]['nome']:<30s} {lingua:>7s} "
                f"{s['tokens_originais_med']:>10.0f} {s['tokens_retidos_med']:>10.0f} "
                f"{s['paginas_med']:>8.0f} {s['paginas_descartadas']:>12d}"
            )
    print("=" * 100)

    # ================================================================= #
    # FASE C - CONVERGENCIA no modelo real (VolLoss + SMETimes)         #
    # ================================================================= #
    print("\n" + "=" * 100)
    print("FASE C - CONVERGENCIA NO ETTh1 REAL (arquitetura VolLoss + SMETimes)")
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
    modelos_ordem: List[str] = []

    for chave, ckpt_path in ckpts_sel.items():
        modelos_ordem.append(chave)
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")

        for est in estrategias:
            for lingua in idiomas:
                # Modelo REMONTADO do checkpoint a cada cenario (isolamento:
                # sem heranca de treinos anteriores entre cenarios).
                rb.seed_everything(args.seed)
                model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
                print(f"  [mount] {ESTRATEGIAS[est]['nome']} ({lingua}) | "
                      f"flags={info['flags']} | treinaveis={info['params_treinaveis']:,}")
                chave_combo = f"{chave}__{est}__{lingua}"
                r_pag = paginados[f"{est}__{lingua}"]
                ds = construir_dataset(r_pag, pool)
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
                result["tempo_batch_ms_medio"] = round(
                    sum(tempos) / max(len(tempos), 1), 2
                )
                result["tokens_originais"] = r_pag["tokens_originais"]
                result["tokens_retidos"] = r_pag["tokens_retidos"]
                result["paginas"] = r_pag["n_paginas"]
                resultados[chave_combo] = result
                print(
                    f"  [ok] best_val={result['melhor_val_mse_pred']:.4f} | "
                    f"teste_mse={result['teste_final_mse_pred']:.4f} | "
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
        "tokens_por_formato": tokens_formato,
        "paginacao": pag,
        "resultados": resultados,
    }
    out_json = salvar_benchmark(resumo)
    imprimir_tabela(resultados, modelos_ordem)

    if args.plots:
        for g in gerar_graficos(resultados, modelos_ordem):
            print(f"[ok] grafico: {g}")
    print(f"[ok] JSON: {out_json}")
    print("Benchmark de paginacao textual sobre o ETTh1 real concluido.")


if __name__ == "__main__":
    main()
