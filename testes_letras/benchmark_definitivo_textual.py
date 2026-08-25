"""
benchmark_definitivo_textual - O BENCHMARK FINAL (Apples-to-Apples).

Roda o CAMPEAO ABSOLUTO (pseudo_json_es + paginacao janela) nas MESMAS
condicoes exatas do baseline numerico do artigo (0.3539 z):

    1) ESCALA (Z-score NATIVO):
       StandardScaler fitado APENAS no split de treino e aplicado a todo o
       dataset; janelas numericas, targets e TEXTO sao gerados sobre os
       valores z-score. Loss e MSE/MAE no horizonte sao calculados
       nativamente nesse espaco - exatamente como o 0.3539 foi obtido.

    2) SPLITS TEMPORAIS ESTRITOS (sem leakage):
       bounds oficiais do AutoTimes/ETTh1 (b1s/b2s via _compute_split_bounds):
       train [0,8640) | val [8640,11520) | test [11520,14400). Todas as
       janelas univariadas por feature com stride=1:
       train=55.111 | val=19.495 | test=19.495.

    3) TEXTO POR JANELA (escala real, fiel ao formato do campeao):
       Cada janela vira UM documento pseudo-json ES (ponto a ponto, 1 casa
       decimal) paginado com a estrategia "janela" (7 paginas x 128 tokens,
       overlap 15%). Como o modelo so consome as primeiras 7 paginas
       (~896 tokens, ~65-90 pontos), o documento e gerado com folga de 160
       pontos - identico ao prefixo efetivo do documento completo.

    4) OTIMIZACAO PARA LARGA ESCALA:
       - Multiprocessing: ProcessPoolExecutor com 1 tarefa por (split, feat);
       - Cache em disco dos tensores paginados (chave = hash da config);
       - O numero de janelas do harness (192) vira 94.101 reais.
       - Batch-size auto-ajustado a VRAM (tenta 256, reduz ate 32).

    5) RODADA FINAL:
       Arquitetura VolLoss+SMETimes (checkpoint Run-8, fusion textual
       aleatorio treinado do zero, backbone congelado), 10 epocas, lr=1e-3,
       3 seeds (2021, 42, 1234), restore_best (melhor val por epoca), MSE.

SAIDA: resultados/veredito_final_apples_to_apples.json
       Test MSE_Z / Test MAE_Z finais vs baseline 0.3539 e vs o recorde
       recalibrado do harness (7.03 -> 0.2082 z, NAO apples-to-apples).

EXECUCAO (smoke test primeiro):
    python -m testes_letras.benchmark_definitivo_textual --stride 60 --epochs 1 --seeds 2021
    python -m testes_letras.benchmark_definitivo_textual
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .modelos import CKPTS, montar_modelo
from .run_etth1_texto import salvar_benchmark
from .treino import build_loaders, rodar_convergencia

SEQ_LEN: int = 672
LABEL_LEN: int = 576
PRED_LEN: int = 96
TOKEN_LEN: int = SEQ_LEN - LABEL_LEN  # 96
BLOCK_MAX: int = 128
N_PAGINAS_MAX: int = 7
OVERLAP: float = 0.15
PONTOS_DO_DOCUMENTO: int = 100      # folga do prefixo efetivo (~896 tokens)

TEMPLATE_ID: str = "pseudo_json_es"
ESTRATEGIA: str = "janela"

BASELINE_ARTIGO_Z: float = 0.3539     # Run-8 VolLoss+SMETimes (numerico)
VAR_HUFL: float = 33.7877             # var_treino da HUFL (fator raw->z)
RECORDE_HARNESS_RAW: float = 7.0336   # pseudo_json_es__janela (pool 192)
RECORDE_HARNESS_Z: float = RECORDE_HARNESS_RAW / VAR_HUFL  # ~0.2082

CACHE_DIR: Path = caminhos.BASE_DIR / "cache_definitivo"

# --------------------------------------------------------------------- #
#  Estado dos workers (multiprocessing)                                 #
# --------------------------------------------------------------------- #
_W: Dict = {}


def _init_worker(
    dt_list, values_z, feat_names, tokenizer_dir, block_max: int,
    n_paginas_max: int, overlap: float, template_id: str, idioma: str,
) -> None:
    """Carrega tokenizador + paginador UMA vez por processo."""
    from transformers import AutoTokenizer

    from .texto_puro import estruturado
    from .texto_puro.paginacao import TestadorPaginacao

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 50256
    _W["dt"] = dt_list
    _W["values"] = values_z
    _W["feat_names"] = feat_names
    _W["paginador"] = TestadorPaginacao(
        tok, block_max=block_max, overlap_frac=overlap,
        n_paginas_max=n_paginas_max,
    )
    _W["template_id"] = template_id
    _W["idioma"] = idioma
    _W["frag_tempo"] = {}  # cache (mes, dia, hora) -> tempo por extenso


def _tarefa_split_feat(args: Tuple) -> Tuple[str, int, np.ndarray, np.ndarray]:
    """Gera ids/lens paginados de TODAS as janelas de um (split, feat)."""
    from .texto_puro import estruturado
    from .texto_puro.extenso import numero_por_extenso

    split_name, b1, b2, feat, stride = args
    tot = b2 - b1 - SEQ_LEN - PRED_LEN + 1
    n = max((tot + stride - 1) // stride, 0)
    if n <= 0:
        return split_name, feat, None, None

    ids_arr = np.zeros((n, N_PAGINAS_MAX * BLOCK_MAX), dtype=np.int64)
    lens_arr = np.zeros((n, N_PAGINAS_MAX), dtype=np.int64)
    pags = _W["paginador"]
    valores = _W["values"]
    n_total = valores.shape[0]

    for i, s_begin in enumerate(range(0, tot, stride)):
        start = b1 + s_begin
        fim = min(start + PONTOS_DO_DOCUMENTO, n_total)
        partes: List[str] = []
        for idx in range(start, fim):
            dt = _W["dt"][idx]
            valor = round(float(valores[idx, feat]), 1)
            chave = (dt.month, dt.day, dt.hour)
            tk = _W["frag_tempo"].get(chave)
            if tk is None:
                tk = estruturado._tempo_keywords(dt, _W["idioma"])
                _W["frag_tempo"][chave] = tk
            partes.append(
                f"{tk} clave valor {numero_por_extenso(valor, _W['idioma'])}"
            )
        doc = " ".join(partes) + "."
        r = pags.paginar(ESTRATEGIA, doc, _W["idioma"])
        m = pags.para_modelo(r)
        ids_arr[i] = m["ids"]
        lens_arr[i] = m["lens"]

    return split_name, feat, ids_arr, lens_arr


# --------------------------------------------------------------------- #
#  Dados: z-score nativo + splits estritos                               #
# --------------------------------------------------------------------- #
def carregar_dados_z(csv_path: Path) -> Dict:
    """Le o ETTh1 e aplica o StandardScaler fitado no split de treino.

    Retorna values_z (z-score global), dt_list, feat_names, bounds dos
    splits (estritos) e stats do scaler (media/std por feature).
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    rb = caminhos.importar_run_benchmarks()
    df = pd.read_csv(csv_path)
    dt_list = (
        pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])).to_pydatetime().tolist()
    )
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feat_names = list(df.columns[1:])
    n_total, n_vars = values.shape
    b1s, b2s = rb._compute_split_bounds(
        csv_path.name.lower(), n_total, SEQ_LEN, PRED_LEN
    )
    scaler = StandardScaler().fit(values[b1s[0]: b2s[0]])
    values_z = scaler.transform(values).astype(np.float32)
    var_treino = {f: float(s ** 2) for f, s in zip(feat_names, scaler.scale_)}
    return {
        "dt_list": dt_list,
        "values_z": values_z,
        "values_raw": values,
        "feat_names": feat_names,
        "n_total": n_total,
        "n_vars": n_vars,
        "b1s": b1s,
        "b2s": b2s,
        "bounds": {
            "train": [b1s[0], b2s[0]],
            "val": [b1s[1], b2s[1]],
            "test": [b1s[2], b2s[2]],
        },
        "scaler": {"media": scaler.mean_.tolist(), "std": scaler.scale_.tolist()},
        "var_treino": var_treino,
        "csv": str(csv_path),
    }


def _chave_cache(
    dados: Dict, stride: int, tokenizer_dir: str, block_max: int,
    n_paginas_max: int, overlap: float, template_id: str, estrategia: str,
    n_max: Optional[int],
) -> str:
    h = hashlib.sha1()
    for s in (
        dados["csv"], str(template_id), estrategia, str(block_max),
        str(n_paginas_max), str(overlap), str(SEQ_LEN), str(LABEL_LEN),
        str(PRED_LEN), str(stride), str(dados["bounds"]), tokenizer_dir,
        str(Path(dados["csv"]).stat().st_mtime_ns), str(n_max),
    ):
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:16]


def montar_datasets(
    dados: Dict, tokenizer_dir: str, stride: int = 1,
    n_max: Optional[int] = None, workers: int = 0,
    forcar: bool = False,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, Dict]:
    """Gera (com cache + multiprocessing) os TensorDatasets textuais z-score.

    Cada sample: x (672,1) z | ids (7*128) tokens GPT-2 | lens (7) | y (672,1) z.
    """
    chave = _chave_cache(
        dados, stride, tokenizer_dir, BLOCK_MAX, N_PAGINAS_MAX, OVERLAP,
        TEMPLATE_ID, ESTRATEGIA, n_max,
    )
    cache_dir = CACHE_DIR / chave
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = [("train", 0), ("val", 1), ("test", 2)]

    if not forcar and (cache_dir / "train.pt").exists():
        print(f"[cache] carregando tensores paginados de {cache_dir}")
        out = []
        for nome, _ in splits:
            d = torch.load(cache_dir / f"{nome}.pt", weights_only=False)
            out.append(TensorDataset(d["x"], d["ids"], d["lens"], d["y"]))
        return out[0], out[1], out[2], {
            "cache_dir": cache_dir, "chave": chave, "de_cache": True,
        }

    print(f"[dados] gerando {TEMPLATE_ID} | {ESTRATEGIA} | stride={stride}"
          f" (workers={workers or os.cpu_count()})")
    b1s, b2s = dados["b1s"], dados["b2s"]
    tarefas: List[Tuple] = []
    for nome, si in splits:
        for feat in range(dados["n_vars"]):
            tarefas.append((nome, b1s[si], b2s[si], feat, stride))

    t0 = time.perf_counter()
    resultados: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {
        "train": {}, "val": {}, "test": {}
    }
    n_workers = workers or min(os.cpu_count(), 16)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
            dados["dt_list"], dados["values_z"], dados["feat_names"],
            tokenizer_dir, BLOCK_MAX, N_PAGINAS_MAX, OVERLAP,
            TEMPLATE_ID, "es",
        ),
    ) as ex:
        futuras = [ex.submit(_tarefa_split_feat, t) for t in tarefas]
        for i, f in enumerate(
            concurrent.futures.as_completed(futuras), start=1
        ):
            nome, feat, ids_arr, lens_arr = f.result()
            if ids_arr is not None:
                resultados[nome][feat] = (ids_arr, lens_arr)
            if i % 7 == 0 or i == len(futuras):
                print(f"    [{i}/{len(futuras)}] tarefas concluidas "
                      f"({time.perf_counter() - t0:.0f}s)")

    print(f"    geracao textual: {time.perf_counter() - t0:.0f}s")

    # ---- Montagem dos tensores + cache em disco ----------------------- #
    datasets: List[TensorDataset] = []
    total_janelas = 0
    for nome, _ in splits:
        ids_parts, lens_parts, xs_parts, ys_parts = [], [], [], []
        n_jan = 0
        for feat in sorted(resultados[nome]):
            ids_arr, lens_arr = resultados[nome][feat]
            if ids_arr is None:
                continue
            # x/y recortados direto dos valores z (mesma ordem de janelas).
            tot = dados["b2s"][{"train": 0, "val": 1, "test": 2}[nome]] - \
                dados["b1s"][{"train": 0, "val": 1, "test": 2}[nome]] - \
                SEQ_LEN - PRED_LEN + 1
            if n_max:
                tot = min(tot, n_max)
            vals = dados["values_z"]
            b1 = dados["b1s"][{"train": 0, "val": 1, "test": 2}[nome]]
            xs = np.zeros((ids_arr.shape[0], SEQ_LEN, 1), dtype=np.float32)
            ys = np.zeros((ids_arr.shape[0], SEQ_LEN, 1), dtype=np.float32)
            for i, s_begin in enumerate(range(0, tot, stride)):
                if n_max and i >= n_max:
                    break
                start = b1 + s_begin
                s_end = start + SEQ_LEN
                r_begin = s_end - LABEL_LEN
                r_end = r_begin + LABEL_LEN + PRED_LEN
                xs[i] = vals[start:s_end, feat: feat + 1]
                ys[i] = vals[r_begin:r_end, feat: feat + 1]
            ids_parts.append(torch.from_numpy(ids_arr))
            lens_parts.append(torch.from_numpy(lens_arr))
            xs_parts.append(torch.from_numpy(xs))
            ys_parts.append(torch.from_numpy(ys))
            n_jan += ids_arr.shape[0]
        ids_all = torch.cat(ids_parts)
        lens_all = torch.cat(lens_parts)
        xs_all = torch.cat(xs_parts)
        ys_all = torch.cat(ys_parts)
        if n_max:
            ids_all, lens_all = ids_all[:n_max], lens_all[:n_max]
            xs_all, ys_all = xs_all[:n_max], ys_all[:n_max]
        ds = TensorDataset(xs_all, ids_all, lens_all, ys_all)
        torch.save(
            {"x": xs_all, "ids": ids_all, "lens": lens_all, "y": ys_all},
            cache_dir / f"{nome}.pt",
        )
        datasets.append(ds)
        total_janelas += len(ds)
        print(f"    [{nome}] {len(ds):,} janelas "
              f"(ids {tuple(ids_all.shape)} | {ids_all.element_size() * ids_all.nelement() / 1e6:.0f} MB)")

    print(f"[cache] salvo em {cache_dir} ({total_janelas:,} janelas no total)")
    return datasets[0], datasets[1], datasets[2], {
        "cache_dir": cache_dir, "chave": chave, "de_cache": False,
    }


# --------------------------------------------------------------------- #
#  Batch-size automatico (VRAM) + aval de paridade numerica              #
# --------------------------------------------------------------------- #
def _batch_fit(model, train_loader, device, pedido: int) -> int:
    """Encontra o maior batch (de pedido ate 32) sem OOM (1 passo de treino)."""
    rb = caminhos.importar_run_benchmarks()
    rb.seed_everything(2021)
    criterion = torch.nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    for bs in (pedido, pedido // 2, 64, 32):
        if bs < 1:
            break
        try:
            batch = next(iter(train_loader))
            xs, y = batch[:-1], batch[-1]
            x = xs[0][:bs].to(device)
            sargs = [t[:bs].to(device) for t in xs[1:]]
            y = y[:bs].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, *sargs)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            del pred, loss
            torch.cuda.empty_cache()
            print(f"[batch] probe OK: batch_size={bs}")
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f"[batch] OOM no batch {bs} -> tentando menor")
            torch.cuda.empty_cache()
    raise SystemExit("OOM mesmo com batch=32. Libere VRAM e tente de novo.")


@torch.no_grad()
def avaliar_numerico(model, loader, device) -> Dict:
    """Eval do checkpoint em modo numerico puro (paridade com o artigo)."""
    preds, trues = [], []
    for batch in loader:
        xs, y = batch[:-1], batch[-1]
        x = xs[0].to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        preds.append(pred[:, -PRED_LEN:, :].float().cpu().numpy())
        trues.append(y[:, -PRED_LEN:, :].float().cpu().numpy())
    p = np.concatenate(preds, 0)
    t = np.concatenate(trues, 0)
    return {
        "mse_z": float(np.mean((p - t) ** 2)),
        "mae_z": float(np.mean(np.abs(p - t))),
        "n_janelas": int(p.shape[0]),
    }


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--checkpoint", type=str, default="smetimes")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=str, default="2021,42,1234")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size",
                   type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stride", type=int, default=1,
                   help="subamostragem das janelas (1 = todas; debug: 60)")
    p.add_argument("--n-max", dest="n_max", type=int, default=None,
                   help="limite de janelas por split (debug)")
    p.add_argument("--workers", type=int, default=0, help="0 = auto")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--forcar-cache", dest="forcar_cache", action="store_true")
    p.add_argument("--sem-paridade", dest="sem_paridade", action="store_true",
                   help="pula o eval de paridade do baseline numerico")
    p.add_argument("--so-gerar", dest="so_gerar", action="store_true",
                   help="apenas gera/carrega o cache textual e sai")
    return p.parse_args()


def _carregar_veredito() -> Dict:
    """Le o veredito existente (para retomar seeds ja concluidas)."""
    p = caminhos.RESULTADOS_DIR / "veredito_final_apples_to_apples.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _fingerprint(args: argparse.Namespace) -> str:
    """Hash da config que afeta os resultados (invalida seeds antigas)."""
    h = hashlib.sha1()
    for s in (
        TEMPLATE_ID, ESTRATEGIA, args.checkpoint, str(args.stride),
        str(args.n_max), str(args.epochs), str(args.lr),
        str(BLOCK_MAX), str(N_PAGINAS_MAX), str(OVERLAP), str(SEQ_LEN),
        str(LABEL_LEN), str(PRED_LEN), caminhos.DATA_PATH,
    ):
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> None:
    args = parse_args()
    caminhos.criar_resultados()
    rb = caminhos.importar_run_benchmarks()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Nenhuma seed valida em --seeds")
    rb.seed_everything(seeds[0])

    device = rb.get_device()
    use_amp = False  # perfil estavel do modo textual: FP32 + TF32 (ROCm)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")

    # ---- Tokenizador -------------------------------------------------- #
    tok_dir = caminhos.descobrir_tokenizador_recursivo(caminhos.ROOT_DIR)
    print(f"[tokenizer] {tok_dir}")

    # ---- Dados (z-score nativo + splits estritos) --------------------- #
    csv_path = Path(args.csv) if args.csv else caminhos.DATA_DIR / caminhos.DATA_PATH
    dados = carregar_dados_z(csv_path)
    print(f"[dados] {dados['csv']} | {dados['n_total']:,} linhas | "
          f"bounds={dados['bounds']} | z-score fit no treino")

    train_ds, val_ds, test_ds, info_cache = montar_datasets(
        dados, str(tok_dir), stride=args.stride, n_max=args.n_max,
        workers=args.workers, forcar=args.forcar_cache,
    )
    print(f"[datasets] train={len(train_ds):,} val={len(val_ds):,} "
          f"test={len(test_ds):,}")
    if args.so_gerar:
        print("[so-gerar] datasets prontos; encerrando "
              "(rode sem --so-gerar para treinar).")
        return

    # ---- Modelo + batch size ------------------------------------------- #
    ckpt_path = CKPTS[args.checkpoint]
    print(f"\n[checkpoint] {ckpt_path.name}")
    model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
    print(f"  parametros: total={info['params_total']:,} "
          f"treinaveis={info['params_treinaveis']:,}")

    batch_size = args.batch_size
    tr_ld, va_ld, te_ld = build_loaders(train_ds, val_ds, test_ds, batch_size)
    if device.type == "cuda" and batch_size > 32:
        try:
            batch_size = _batch_fit(model, tr_ld, device, batch_size)
            tr_ld, va_ld, te_ld = build_loaders(
                train_ds, val_ds, test_ds, batch_size
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"[batch] probe falhou ({e}); usando batch=32")
            batch_size = 32
    print(f"[batch] final: {batch_size}")

    # ---- Paridade: baseline numerico (mesmo checkpoint, mesmo teste) --- #
    resumo: Dict = _carregar_veredito()
    fp = _fingerprint(args)
    if resumo.get("fingerprint") != fp:
        print(f"[resumo] fingerprint da config mudou ({resumo.get('fingerprint')} "
              f"-> {fp}); descartando seeds antigas")
        resumo = {}
    resumo["fingerprint"] = fp
    resumo["config"] = {
        "template": TEMPLATE_ID,
        "estrategia": ESTRATEGIA,
        "checkpoint": args.checkpoint,
        "escala": "z-score nativo (StandardScaler fit no treino)",
        "splits": dados["bounds"],
        "janelas": {"train": len(train_ds), "val": len(val_ds),
                    "test": len(test_ds)},
        "epochs": args.epochs,
        "seeds": seeds,
        "batch_size": batch_size,
        "lr": args.lr,
        "block_max": BLOCK_MAX,
        "n_paginas_max": N_PAGINAS_MAX,
        "overlap": OVERLAP,
        "stride": args.stride,
        "cache": str(info_cache["cache_dir"]),
        "var_treino_hufl": VAR_HUFL,
    }
    resumo.setdefault("baseline", {"artigo_z": BASELINE_ARTIGO_Z})
    resumo["recorde_harness"] = {
        "mse_raw": RECORDE_HARNESS_RAW,
        "mse_z_equiv": RECORDE_HARNESS_Z,
        "aviso": ("7.03 medido no harness antigo (pool 192, escala raw, "
                  "leakage 60/20/20); 0.2082 z remove so a escala - NAO "
                  "e apples-to-apples."),
    }
    resumo.setdefault("campeao_textual", {})
    resumo["campeao_textual"].setdefault("por_seed", {})

    if not args.sem_paridade and "medido_z" not in resumo["baseline"]:
        from .benchmark_numerico_base import montar_modelo_numerico

        rb.seed_everything(seeds[0])
        model_num = montar_modelo_numerico(ckpt_path, device, use_amp=False)
        t0 = time.perf_counter()
        ev = avaliar_numerico(model_num, te_ld, device)
        print(f"[paridade] baseline numerico no MESMO test set: "
              f"MSE_Z={ev['mse_z']:.4f} MAE_Z={ev['mae_z']:.4f} "
              f"({ev['n_janelas']:,} janelas, {time.perf_counter() - t0:.0f}s)")
        print(f"[paridade] vs artigo 0.3539: "
              f"{100 * (ev['mse_z'] - BASELINE_ARTIGO_Z) / BASELINE_ARTIGO_Z:+.1f}%")
        resumo["baseline"]["medido_z"] = round(ev["mse_z"], 6)
        resumo["baseline"]["medido_mae_z"] = round(ev["mae_z"], 6)
        del model_num
        if device.type == "cuda":
            torch.cuda.empty_cache()
        salvar_benchmark(resumo, nome="veredito_final_apples_to_apples.json")

    # ---- Rodada final: 10 epocas x 3 seeds (retomavel por seed) --------- #
    por_seed = resumo["campeao_textual"]["por_seed"]
    mse_list, mae_list, estouros_list = [], [], []
    val_por_epoca: List[List[float]] = []
    seeds_pendentes = [s for s in seeds if str(s) not in por_seed]
    if not seeds_pendentes:
        print("[seed] todas as seeds ja concluidas - retomando agregacao")
    for seed in seeds_pendentes:
        rb.seed_everything(seed)
        model, _ = montar_modelo(ckpt_path, device, use_amp=use_amp)
        t0 = time.perf_counter()
        result = rodar_convergencia(
            model, tr_ld, va_ld, te_ld,
            epochs=args.epochs, lr=args.lr, device=device,
            use_amp=use_amp, seed=seed, restore_best=True,
        )
        mse = result["teste_final_mse_pred"]
        mae = result["teste_final_mae_pred"]
        por_seed[str(seed)] = {
            "test_mse_z": round(mse, 6),
            "test_mae_z": round(mae, 6),
            "melhor_val_mse_z": round(result["melhor_val_mse_pred"], 6),
            "melhor_epoca": result["melhor_epoca"],
            "estouros": result["batches_estourados_total"],
        }
        val_por_epoca.append(result["historico"]["val_mse_pred"])
        print(f"  [seed={seed}] test_mse_z={mse:.4f} test_mae_z={mae:.4f} "
              f"(melhor_val={result['melhor_val_mse_pred']:.4f} ep"
              f"{result['melhor_epoca']}) | {time.perf_counter() - t0:.0f}s")
        salvar_benchmark(resumo, nome="veredito_final_apples_to_apples.json")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for s_key, r in por_seed.items():
        mse_list.append(r["test_mse_z"])
        mae_list.append(r["test_mae_z"])
        estouros_list.append(r["estouros"])
    mse_media = float(np.mean(mse_list))
    mae_media = float(np.mean(mae_list))
    baseline_z = float(resumo["baseline"].get("medido_z", BASELINE_ARTIGO_Z))
    delta = 100.0 * (mse_media - baseline_z) / baseline_z

    resumo["campeao_textual"].update({
        "test_mse_z_media": round(mse_media, 6),
        "test_mse_z_desvio": round(float(np.std(mse_list)), 6),
        "test_mae_z_media": round(mae_media, 6),
        "test_mae_z_desvio": round(float(np.std(mae_list)), 6),
        "test_mse_raw_equiv_hufl": round(mse_media * VAR_HUFL, 6),
        "test_mae_raw_equiv_hufl": round(mae_media * np.sqrt(VAR_HUFL), 6),
        "por_seed": por_seed,
        "estouros_total": int(sum(estouros_list)),
        "delta_vs_baseline_pct": round(delta, 2),
    })
    if len(val_por_epoca) == len(seeds):
        resumo["campeao_textual"]["val_mse_z_media_por_epoca"] = [
            round(float(np.mean([v[e] for v in val_por_epoca])), 6)
            for e in range(min(len(v) for v in val_por_epoca))
        ]

    print("\n" + "=" * 100)
    print("VEREDITO FINAL - APPLES-TO-APPLES (ETTh1, z-score nativo)")
    print("=" * 100)
    print(f"  Baseline numerico (Run-8 VolLoss+SMETimes): "
          f"MSE_Z={baseline_z:.4f}")
    print(f"  Campeao textual (pseudo_json_es | janela):  "
          f"MSE_Z={mse_media:.4f} +/- {resumo['campeao_textual']['test_mse_z_desvio']:.4f}"
          f"  MAE_Z={mae_media:.4f}")
    print(f"  Delta vs baseline: {delta:+.2f}%")
    print(f"  Equiv. raw HUFL (x{round(VAR_HUFL, 2)}): "
          f"{resumo['campeao_textual']['test_mse_raw_equiv_hufl']:.4f}")
    print(f"  Rec. harness (referencia, nao-apples): 7.03 raw -> 0.2082 z")
    print("=" * 100)

    out = salvar_benchmark(resumo, nome="veredito_final_apples_to_apples.json")
    print(f"[ok] JSON: {out}")
    print("Benchmark definitivo concluido.")


if __name__ == "__main__":
    main()
