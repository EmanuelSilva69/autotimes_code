"""
run_apples_to_apples_trio - CONFRONTO FINAL "Apples-to-Apples" em Z-SCORE.

Compara as 3 abordagens sob o PROTOCOLO OFICIAL do AutoTimes (z-score nativo
global ajustado ESTRITAMENTE no treino) com splits canonicos de linhas e
CHANNEL INDEPENDENCE (as 7 features do ETTh1 empilhadas univariadas):

    borders EXATOS do Dataset_ETT_hour (run_benchmarks._compute_split_bounds):
        b1s = [0, 7968, 10848]  ->  b2s = [8640, 11520, 14400]
        (val/test comecam 672 linhas antes do corte do split, como o original)
    janelas por feat: train 7.873 | val 2.785 | test 2.785 (stride 1)
    TOTAL: 7 features x 12.443 = 94.101 janelas (55111 / 19495 / 19495)
    = EXATAMENTE o volume do baseline numerico 0.3539 (que e a media
    multivariada das 7 colunas, nao o erro isolado do HUFL).

OS 3 MODELOS:
    1) Baseline Numerico (Run-8 VolLoss+SMETimes): puramente matematico.
       0.3539 ja cravado (resultados/numerico_base_stats.json) -> carregado
       do log; --re-numerico forca uma re-medicao eval-only nas janelas
       canonicas de HUFL (mesmo harness rb.evaluate, last_only=True).
    2) Pseudo-JSON ES (campeao textual, paginacao janela c/ overlap):
       UM documento por feature (template pseudo_json_es sobre as linhas de
       treino [0, 8640) da COLUNA daquela feature, em z-score), paginado
       (7 blocos x 128), repetido nas janelas da propria feature.
    3) Char-by-Char (serializacao digito a digito com max_abs local):
       documento POR JANELA gerado a partir da janela em z-score da sua
       feature (encode aplica MaxAbs local); contrato de 5 tensores
       (x, ids, lens, y, max_abs).

RIGOR:
    - 3 seeds (2021, 42, 1234) x 10 epocas, restore_best por validacao,
      batch 16, lr 1e-3, isolamento estrito (montar_modelo DENTRO do seed);
    - metrica SEMPRE no espaco Z-SCORE (o modelo devolve a reconstrucao na
      escala de entrada z; o RevIN interno ja denormalizou);
    - cache de tensores no disco (quinta_noite/cache_trio/): o pool z-score
      e os textos/paginacao sao gerados UMA vez e reutilizados nas 3 seeds
      e nas re-execucoes.

ATENCAO - DECODE do char-by-char (--decode):
    O AutoTimesModel textual ja devolve a reconstrucao denormalizada pelo
    RevIN interno para a escala de ENTRADA (aqui: z-score). O encode() do
    char-by-char normaliza o TEXTO pelo max_abs local, mas a ESCALA DA SAIDA
    e definida pelo ramo numerico (RevIN), nao pelo texto. Aplicar decode()
    (pred * max_abs) deslocaria as predicoes z para fora do espaco alvo e
    INFLARIA o MSE por ~max_abs^2 (fator ~4-16x). O decode vem DESLIGADO por
    padrao (flag --decode liga apenas se voce quiser medir essa variante).

SAIDAS (testes_letras/quinta_noite/):
    veredito_trio.json              (consolidado das 3 abordagens)
    convergencia_trio.png           (Val MSE z por epoca + bandas + linha 0.3539)
    barras_veredito_trio.png        (Test MSE z final, error bars pretas)

USO:
    python -m testes_letras.run_apples_to_apples_trio
    python -m testes_letras.run_apples_to_apples_trio --epochs 1 --seeds 2021
    python -m testes_letras.run_apples_to_apples_trio --charts-only
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suprime warnings de SDPA experimental no AMD/ROCm
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import caminhos
from .dados import LABEL_LEN, PRED_LEN, SEQ_LEN
from .modelos import CKPTS, carregar_checkpoint, montar_modelo
from .modelos_dual import montar_modelo_dual
from .run_etth1_texto import _FEAT_PADRAO, carregar_etth1
from .texto_char_by_char import (
    BLOCK_MAX,
    N_PAGINAS_MAX,
    OVERLAP,
    CharByCharPipeline,
    construir_dataset_char_by_char,
)
from .texto_puro import estruturado
from .texto_puro.paginacao import TestadorPaginacao
from .treino import build_loaders, rodar_convergencia

# Splits canonicos de LINHAS (borders EXATOS do Dataset_ETT_hour: o val/test
# comeca SEQ_LEN linhas antes do corte, como no data_provider original).
N_TREINO: int = 8640
N_VAL: int = 11520
N_TESTE: int = 14400
B1S: Tuple[int, int, int] = (0, N_TREINO - SEQ_LEN, N_VAL - SEQ_LEN)
B2S: Tuple[int, int, int] = (N_TREINO, N_VAL, N_TESTE)

N_FEATS: int = 7  # HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (Channel Independence)

TEMPLATE_PSEUDO_JSON: str = "pseudo_json_es"
ESTRATEGIA_PAGINACAO: str = "janela"

OUT_DIR: Path = caminhos.BASE_DIR / "quinta_noite"
CACHE_DIR: Path = OUT_DIR / "cache_trio"
JSON_OUT: Path = OUT_DIR / "veredito_trio.json"
JSON_NUMERICO_LOG: Path = caminhos.RESULTADOS_DIR / "numerico_base_stats.json"

_CORES: Dict[str, str] = {
    "numerico": "#7f7f7f",
    "pseudo_json_es": "#9467bd",
    "char_by_char": "#d62728",
}


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confronto final Apples-to-Apples (z-score canonicas): "
                    "Numerico vs Pseudo-JSON ES vs Char-by-Char"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="2021,42,1234")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=BLOCK_MAX)
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int, default=N_PAGINAS_MAX)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument("--linhas", type=int, default=N_TESTE,
                        help="Linhas do CSV a carregar (>= 14400 p/ splits canonicos)")
    parser.add_argument("--feats", type=str, default="all",
                        help="Features do ETTh1 (indices 0-6 separados por "
                             "virgula, ex: '0,1') ou 'all' (7 colunas)")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--decode", action="store_true",
                        help="APLICA decode(max_abs) nas metricas do char-by-char "
                             "(INFLARIA o MSE por ~max_abs^2; ver docstring)")
    parser.add_argument("--re-numerico", dest="re_numerico", action="store_true",
                        help="Re-mede o baseline numerico (eval-only, HUFL canonico) "
                             "em vez de carregar 0.3539 do log")
    parser.add_argument("--charts-only", dest="charts_only", action="store_true",
                        help="Somente regenera os graficos a partir do JSON existente")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--force-cache", dest="force_cache", action="store_true",
                        help="Regenera os tensores cacheados (pool/textos)")
    parser.add_argument("--variante-arquitetura", dest="variante",
                        choices=["original", "dual"], default="original",
                        help="Arquitetura do modelo numerico: 'original' "
                             "(AutoTimesModel) ou 'dual' (AutoTimesDualModel "
                             "com trend+osc)")
    parser.add_argument("--dual-fusao", dest="dual_fusao",
                        choices=["glu", "soma"], default="glu",
                        help="Combinacao trend/osc da variante dual: "
                             "'glu' (gate enviesado, +1.18M params) ou "
                             "'soma' (adicao simples, params = original). "
                             "Ignorado se --variante-arquitetura original.")
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Pool z-score canonico: 7 features (Channel Independence) + borders    #
#  oficiais do Dataset_ETT_hour. Scaler fitado APENAS no treino [0, 8640)#
# --------------------------------------------------------------------- #
def montar_pool_z(args: argparse.Namespace):
    """Janelas z-score das features selecionadas, construcao oficial.

    Borders b1s=[0, 7968, 10848] / b2s=[8640, 11520, 14400] (val/test com
    overlap de SEQ_LEN linhas, como o data_provider original). Scaler
    ajustado em values[0:8640, :] (todas as colunas, por-coluna, fiel ao
    StandardScaler do run_benchmarks) e aplicado a TODAS as linhas.

    Returns:
        (dados, scaler, splits, z, b1s, b2s) onde splits =
        [(x, y, feat_ids)] com x/y em (n, 672, 1) no espaco z-score e
        feat_ids = posicao da feature na selecao (0..len(feats)-1).
    """
    dados = carregar_etth1(args.csv, args.linhas)
    if args.linhas < N_TESTE:
        raise SystemExit(f"--linhas {args.linhas} < 14400: splits canonicos incompletos")
    values = dados["values"]
    feats = [int(f) for f in args.feats_list]

    scaler = StandardScaler().fit(values[0:B2S[0], :])
    z = scaler.transform(values).astype(np.float32)

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for b1, b2 in zip(B1S, B2S):
        tot = len(z[b1:b2]) - SEQ_LEN - PRED_LEN + 1
        xs, ys, fs = [], [], []
        for fi, f in enumerate(feats):
            for s in range(tot):
                r_begin = s + SEQ_LEN - LABEL_LEN
                xs.append(z[b1 + s: b1 + s + SEQ_LEN, f: f + 1])
                ys.append(z[b1 + r_begin: b1 + r_begin + LABEL_LEN + PRED_LEN, f: f + 1])
                fs.append(fi)
        splits.append((np.stack(xs), np.stack(ys), np.asarray(fs, dtype=np.int64)))
    return dados, scaler, splits, z, B1S, B2S


def _cache_pool_path(args: argparse.Namespace) -> Path:
    feats = "all" if len(args.feats_list) == N_FEATS else "-".join(map(str, args.feats_list))
    return CACHE_DIR / f"pool_z_v2_linhas{args.linhas}_feats{feats}.npz"


def carregar_pool_cache(args: argparse.Namespace):
    """Pool z-score (Channel Independence) com cache em disco (npz)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_pool_path(args)
    if p.exists() and not args.force_cache:
        d = np.load(p)
        splits = [(d["train_x"], d["train_y"], d["train_feats"]),
                  (d["val_x"], d["val_y"], d["val_feats"]),
                  (d["test_x"], d["test_y"], d["test_feats"])]
        scaler = StandardScaler()
        scaler.mean_ = d["scaler_mean"]
        scaler.scale_ = d["scaler_scale"]
        print(f"[cache] pool z-score carregado: {p}")
        return splits, scaler, d["z"], B1S, B2S
    dados, scaler, splits, z, b1s, b2s = montar_pool_z(args)
    np.savez(
        p,
        train_x=splits[0][0], train_y=splits[0][1], train_feats=splits[0][2],
        val_x=splits[1][0], val_y=splits[1][1], val_feats=splits[1][2],
        test_x=splits[2][0], test_y=splits[2][1], test_feats=splits[2][2],
        z=z,
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
    )
    print(f"[cache] pool z-score salvo: {p}")
    return splits, scaler, z, b1s, b2s


def _pool_lista(split: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    x, y, _ = split
    return [(x[i], y[i]) for i in range(len(x))]


# --------------------------------------------------------------------- #
#  Pseudo-JSON ES: UM documento por feature (treino [0, 8640) em z)      #
# --------------------------------------------------------------------- #
def _gerar_pseudo_json(args: argparse.Namespace, tokenizer, dados, z) -> Dict:
    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    lingua = estruturado.TEMPLATES_ESTRUTURADOS[TEMPLATE_PSEUDO_JSON]["idioma"]
    ids_pf, lens_pf, infos = [], [], []
    for fi, f in enumerate(args.feats_list):
        doc = estruturado.serie_para_texto_estruturado(
            dados["dt_list"], z, int(f), 0, N_TREINO, TEMPLATE_PSEUDO_JSON,
        )
        r = paginador.paginar(ESTRATEGIA_PAGINACAO, doc, lingua)
        r.update(paginador.para_modelo(r))
        ids_pf.append(np.asarray(r["ids"]))
        lens_pf.append(np.asarray(r["lens"]))
        infos.append((int(r["tokens_originais"]), int(r["tokens_retidos"]),
                      int(r["n_paginas"]), int(r.get("paginas_descartadas", 0))))
        print(f"  [pseudo-json feat {f}] {infos[-1][0]:,} tokens orig -> "
              f"{infos[-1][1]:,} ret | {infos[-1][2]} paginas")
    return {
        "ids_per_feat": np.stack(ids_pf),
        "lens_per_feat": np.stack(lens_pf),
        "tokens_originais": infos[0][0], "tokens_retidos": infos[0][1],
        "n_paginas": infos[0][2], "paginas_descartadas": infos[0][3],
    }


def carregar_pseudo_json(args: argparse.Namespace, tokenizer, dados, z) -> Dict:
    """ids/lens do documento Pseudo-JSON por feature (cache leve em npz)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chave = hashlib.md5(
        f"{TEMPLATE_PSEUDO_JSON}_v2_b{args.block_max}_p{args.n_paginas_max}"
        f"_o{args.overlap}_feats{'-'.join(map(str, args.feats_list))}".encode()
    ).hexdigest()[:12]
    p = CACHE_DIR / f"pseudo_json_{chave}.npz"
    if p.exists() and not args.force_cache:
        d = np.load(p)
        print(f"[cache] pseudo-json ids/lens (por feature): {p}")
        return {
            "ids_per_feat": d["ids_per_feat"], "lens_per_feat": d["lens_per_feat"],
            "tokens_originais": int(d["tokens_originais"]),
            "tokens_retidos": int(d["tokens_retidos"]),
            "n_paginas": int(d["n_paginas"]),
            "paginas_descartadas": int(d["paginas_descartadas"]),
        }
    r = _gerar_pseudo_json(args, tokenizer, dados, z)
    np.savez(
        p,
        ids_per_feat=r["ids_per_feat"], lens_per_feat=r["lens_per_feat"],
        tokens_originais=r["tokens_originais"], tokens_retidos=r["tokens_retidos"],
        n_paginas=r["n_paginas"], paginas_descartadas=r["paginas_descartadas"],
    )
    print(f"[cache] pseudo-json ids/lens (por feature) salvo: {p}")
    return r


def datasets_pseudo_json(args: argparse.Namespace, splits, prep: Dict) -> List[TensorDataset]:
    """Datasets de 4 tensores (x_z, ids, lens, y_z) por split.

    Cada janela recebe o documento paginado da SUA feature (Channel
    Independence): ids/lens roteados por feat_ids.
    """
    ids_pf = torch.from_numpy(prep["ids_per_feat"])
    lens_pf = torch.from_numpy(prep["lens_per_feat"])
    dss: List[TensorDataset] = []
    for x, y, fs in splits:
        dss.append(TensorDataset(
            torch.from_numpy(x),
            ids_pf[fs].to(torch.int64),
            lens_pf[fs].to(torch.int64),
            torch.from_numpy(y),
        ))
    return dss


# --------------------------------------------------------------------- #
#  Char-by-Char: documento POR JANELA (5 tensores, max_abs local)        #
# --------------------------------------------------------------------- #
def carregar_char_by_char(args: argparse.Namespace, tokenizer, splits) -> Tuple[List[TensorDataset], Dict]:
    """Tensores char-by-char (7 features) com cache em npz (geracao 1x)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chave = hashlib.md5(
        f"char_by_char_v2_b{args.block_max}_p{args.n_paginas_max}"
        f"_o{args.overlap}_feats{'-'.join(map(str, args.feats_list))}".encode()
    ).hexdigest()[:12]
    p = CACHE_DIR / f"char_by_char_{chave}.npz"
    if p.exists() and not args.force_cache:
        d = np.load(p)
        print(f"[cache] char-by-char tensores: {p}")
        dss = []
        for a in ("train", "val", "test"):
            i, l, m = d[f"{a}_ids"], d[f"{a}_lens"], d[f"{a}_max_abs"]
            x, y, _ = splits[("train", "val", "test").index(a)]
            dss.append(TensorDataset(
                torch.from_numpy(x),
                torch.tensor(i, dtype=torch.int64),
                torch.tensor(l, dtype=torch.int64),
                torch.from_numpy(y),
                torch.from_numpy(m),
            ))
        return dss, {}
    # ---- gera do zero (uma unica vez; as 3 seeds reutilizam) ---------- #
    t0 = time.perf_counter()
    dss, stats = [], {}
    guard: Dict[str, np.ndarray] = {}
    for rotulo, split in zip(("train", "val", "test"), splits):
        ds, agg = construir_dataset_char_by_char(
            _pool_lista(split), tokenizer,
            block_max=args.block_max,
            n_paginas_max=args.n_paginas_max,
            overlap_frac=args.overlap,
            return_stats=True,
        )
        guard[f"{rotulo}_ids"] = ds.tensors[1].numpy()
        guard[f"{rotulo}_lens"] = ds.tensors[2].numpy()
        guard[f"{rotulo}_max_abs"] = ds.tensors[4].numpy()
        dss.append(ds)
        stats[rotulo] = agg
        print(f"  [char-by-char {rotulo}] {len(ds)} janelas | max_abs="
              f"{agg['max_abs_media']:.3f} ± {agg['max_abs_desvio']:.3f} | "
              f"{time.perf_counter() - t0:.1f}s")
    try:
        np.savez(p, **guard)
        print(f"[cache] char-by-char tensores salvo: {p}")
    except Exception as e:  # pragma: no cover - cache e so otimizacao
        print(f"[cache] aviso: nao deu para salvar char-by-char ({e})")
    return dss, stats


# --------------------------------------------------------------------- #
#  Baseline numerico: carrega do log (0.3539) ou re-mede eval-only       #
# --------------------------------------------------------------------- #
def montar_modelo_numerico(ckpt_path: Path, device: torch.device, use_amp: bool):
    """AutoTimesModel SEM ramo textual (idêntico ao checkpoint original)."""
    rb = caminhos.importar_run_benchmarks()
    ckpt = carregar_checkpoint(ckpt_path)
    flags, args = ckpt["flags"], ckpt["args"]
    torch_dtype = torch.float16 if (use_amp and device.type == "cuda") else torch.float32
    model = rb.AutoTimesModel(
        seq_len=args["seq_len"], pred_len=args["pred_len"], n_vars=args["n_vars"],
        label_len=args["label_len"], mlp_hidden_dim=args["mlp_hidden_dim"],
        mlp_hidden_layers=args["mlp_hidden_layers"], mlp_activation=args["mlp_activation"],
        dropout=args["dropout"], gpt2_dir=Path(args["gpt2_dir"]),
        torch_dtype=torch_dtype,
        use_robust=bool(flags["use_robust"]), use_overlap=bool(flags["use_overlap"]),
        use_vol_gate=bool(flags["use_vol_gate"]), unfreeze_layers=0,
        textual=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, flags


def avaliar_numerico(splits, ckpt_path: Path, device: torch.device,
                     use_amp: bool, batch_size: int = 256) -> Dict:
    """Eval-only (protocolo oficial: MSE/MAE no horizonte pred_len, z-score)."""
    rb = caminhos.importar_run_benchmarks()
    _, _, (x_te, y_te, _) = splits
    ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model, flags = montar_modelo_numerico(ckpt_path, device, use_amp)
    pred_len = 96
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(device, non_blocking=True)
            y_b = batch[1].to(device, non_blocking=True)
            pred = model(x_b)
            preds.append(pred[:, -pred_len:, :].float().cpu().numpy())
            trues.append(y_b[:, -pred_len:, :].float().cpu().numpy())
    p = np.concatenate(preds, 0)
    t = np.concatenate(trues, 0)
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    return {"test_mse": mse, "test_mae": mae, "n_janelas_test": len(ds)}


def carregar_numerico_log() -> Optional[Dict]:
    """0.3539 cravado (resultados/numerico_base_stats.json)."""
    if not JSON_NUMERICO_LOG.exists():
        return None
    d = json.load(open(JSON_NUMERICO_LOG, encoding="utf-8"))
    m = d.get("modelos", {}).get("smetimes")
    if not m or m.get("test_mse_media") is None:
        return None
    return {
        "test_mse_media": float(m["test_mse_media"]),
        "test_mae_media": float(m.get("test_mae_media")),
        "n_janelas_test": int(m.get("n_janelas_test", 0)),
        "referencia_artigo": float(d.get("paridade", {}).get("referencia_artigo", {})
                                   .get("mse_run8_zscore", 0.3539)),
    }


# --------------------------------------------------------------------- #
#  Bateria textual (3 seeds x 10 epocas, restore_best por validacao)     #
# --------------------------------------------------------------------- #
def rodar_bateria_textual(
    args: argparse.Namespace, nome: str, loaders,
    ckpt_path: Path, device: torch.device, use_amp: bool, seeds: List[int],
    decode_fn=None, model_fn=None,
) -> Dict:
    rb = caminhos.importar_run_benchmarks()
    if model_fn is None:
        model_fn = montar_modelo
    por_seed: Dict[str, Dict] = {}
    mse_list, mae_list, bestval_list, epoca_list, estouros_list = [], [], [], [], []
    val_por_epoca: List[List[float]] = []
    test_por_epoca: List[List[float]] = []
    tr_ld, va_ld, te_ld = loaders
    print(f"\n--- {nome} ---")
    for seed in seeds:
        rb.seed_everything(seed)
        model, info = model_fn(ckpt_path, device, use_amp=use_amp)
        print(f"  [mount seed={seed}] treinaveis={info['params_treinaveis']:,}")
        t0 = time.perf_counter()
        result = rodar_convergencia(
            model, tr_ld, va_ld, te_ld,
            epochs=args.epochs, lr=args.lr, device=device,
            use_amp=use_amp, seed=seed, restore_best=True, decode_fn=decode_fn,
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
        test_por_epoca.append(result["historico"]["test_mse_pred"])
        print(f"    [seed={seed}] test_mse={mse:.4f} | test_mae={mae:.4f} | "
              f"best_val={bv:.4f} (ep {be}) | estouros={estouros} | "
              f"{time.perf_counter() - t0:.1f}s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_ep = max((len(v) for v in val_por_epoca), default=0)
    return {
        "test_mse_media": round(float(np.mean(mse_list)), 4),
        "test_mse_desvio": round(float(np.std(mse_list)), 4),
        "test_mae_media": round(float(np.mean(mae_list)), 4),
        "test_mae_desvio": round(float(np.std(mae_list)), 4),
        "melhor_val_mse_media": round(float(np.mean(bestval_list)), 4),
        "melhor_val_mse_desvio": round(float(np.std(bestval_list)), 4),
        "melhor_epoca_media": round(float(np.mean(epoca_list)), 2),
        "estouros_total": sum(estouros_list),
        "val_mse_media": [round(float(np.mean([v[e] for v in val_por_epoca])), 6) for e in range(n_ep)],
        "val_mse_desvio": [round(float(np.std([v[e] for v in val_por_epoca])), 6) for e in range(n_ep)],
        "test_mse_media_por_epoca": [round(float(np.mean([v[e] for v in test_por_epoca])), 6) for e in range(n_ep)],
        "test_mse_desvio_por_epoca": [round(float(np.std([v[e] for v in test_por_epoca])), 6) for e in range(n_ep)],
        "por_seed": por_seed,
    }


# --------------------------------------------------------------------- #
#  Relatorio + graficos                                                  #
# --------------------------------------------------------------------- #
def imprimir_tabela(resumo: Dict) -> None:
    print("\n" + "=" * 100)
    print("VEREDITO FINAL APPLES-TO-APPLES (z-score, 10 epocas x 3 seeds)")
    print("=" * 100)
    m = resumo["modelos"]
    print(f"{'Modelo':<16s} {'Test MSE':>14s} {'Test MAE':>14s} {'best_val':>12s} "
          f"{'estouros':>8s}")
    print("-" * 100)
    for chave, rotulo in (("numerico", "Numerico"),
                          ("pseudo_json_es", "Pseudo-JSON ES"),
                          ("char_by_char", "Char-by-Char")):
        c = m[chave]
        if chave == "numerico":
            print(f"{rotulo:<16s} {c['test_mse_media']:>14.4f} "
                  f"{c['test_mae_media']:>14.4f} {'-':>12s} {'-':>8s}")
        else:
            print(f"{rotulo:<16s} {c['test_mse_media']:>14.4f} ± {c['test_mse_desvio']:>8.4f} "
                  f"{c['test_mae_media']:>7.4f} ± {c['test_mae_desvio']:>7.4f} "
                  f"{c['melhor_val_mse_media']:>12.4f} {c['estouros_total']:>8d}")
    print("-" * 100)
    print("Formato: media ± desvio padrao sobre 3 seeds (menor MSE vence).")
    print("=" * 100)


def gerar_graficos(resumo: Dict, variante: str = "original") -> List[Path]:
    """G1 convergencia + G2 barras (figsize 16x9, sns.despine, fontes grandes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.35)
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 17,
                         "axes.labelsize": 15, "legend.fontsize": 13,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    _labels = {"original": ("", ""), "dual": ("_dual", " [Dual GLU]"),
               "dual_soma": ("_dual_soma", " [Dual SOMA]")}
    suffix, title_var = _labels.get(variante, ("", ""))
    m = resumo["modelos"]
    numerico = m["numerico"]["test_mse_media"]
    saidas: List[Path] = []

    # ---- G1: Convergencia de Titãs ------------------------------------ #
    fig, ax = plt.subplots(figsize=(16, 9))
    for chave, rotulo in (("pseudo_json_es", "Pseudo-JSON ES"),
                          ("char_by_char", "Char-by-Char")):
        c = m[chave]
        med = np.asarray(c["val_mse_media"])
        dev = np.asarray(c["val_mse_desvio"])
        ep = np.arange(1, len(med) + 1)
        ax.plot(ep, med, "-o", color=_CORES[chave], lw=3, ms=8, zorder=4,
                label=f"{rotulo} (Test MSE {c['test_mse_media']:.4f} ± "
                      f"{c['test_mse_desvio']:.4f})")
        ax.fill_between(ep, med - dev, med + dev, color=_CORES[chave],
                        alpha=0.18, zorder=2)
    ax.axhline(numerico, color="red", ls="--", lw=2.5, zorder=1,
               label=f"Baseline Numerico (Run-8) = {numerico:.4f}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val MSE no horizonte (z-score)")
    ax.set_title(f"A Convergencia de Titãs{title_var}\n"
                 "Val MSE z-score por epoca (media ± 1 desvio padrao das 3 seeds)")
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(axis="y", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    p1 = OUT_DIR / f"convergencia_trio{suffix}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saidas.append(p1)
    print(f"[graficos] {p1}")

    # ---- G2: Veredito Final Z-Score ------------------------------------ #
    fig, ax = plt.subplots(figsize=(16, 9))
    nomes = ["Baseline Numerico\n(Run-8 SMETimes)", "Pseudo-JSON ES", "Char-by-Char"]
    medias = [numerico,
              m["pseudo_json_es"]["test_mse_media"],
              m["char_by_char"]["test_mse_media"]]
    desvios = [0.0,
               m["pseudo_json_es"]["test_mse_desvio"],
               m["char_by_char"]["test_mse_desvio"]]
    cores = [_CORES["numerico"], _CORES["pseudo_json_es"], _CORES["char_by_char"]]
    barras = ax.bar(nomes, medias, yerr=desvios, capsize=8, width=0.55,
                    color=cores, edgecolor="black", linewidth=1.2, zorder=3,
                    error_kw=dict(ecolor="black", elinewidth=2.2, capsize=9))
    for b, med, dev in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, med + dev + max(medias) * 0.03,
                f"{med:.4f}", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(m + d for m, d in zip(medias, desvios)) * 1.28)
    ax.set_ylabel("Test MSE z-score (media das 3 seeds)")
    ax.set_title("O Veredito Final Z-Score\n"
                 "Test MSE no horizonte (menor = melhor; barras de erro = "
                 "desvio padrao das seeds)")
    ax.grid(axis="y", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    p2 = OUT_DIR / f"barras_veredito_trio{suffix}.png"
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rb = caminhos.importar_run_benchmarks()

    # JSON de saida varia conforme a variante arquitetural
    global JSON_OUT
    if args.variante == "dual":
        sufixo = "dual" + ("" if args.dual_fusao == "glu" else "_soma")
        JSON_OUT = OUT_DIR / f"veredito_trio_{sufixo}.json"
    else:
        sufixo = ""
        JSON_OUT = OUT_DIR / "veredito_trio.json"

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Nenhuma seed valida em --seeds")
    rb.seed_everything(seeds[0])

    if args.feats.strip().lower() == "all":
        args.feats_list = list(range(N_FEATS))
    else:
        args.feats_list = [int(f.strip()) for f in args.feats.split(",") if f.strip()]
        if not args.feats_list:
            raise SystemExit("Nenhuma feature valida em --feats")

    device = rb.get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")

    # ---- Charts-only -------------------------------------------------- #
    if args.charts_only:
        if not JSON_OUT.exists():
            raise SystemExit(f"JSON nao encontrado: {JSON_OUT}")
        resumo = json.load(open(JSON_OUT, encoding="utf-8"))
        imprimir_tabela(resumo)
        for p in gerar_graficos(resumo):
            print(f"[ok] grafico: {p}")
        return

    # ---- Tokenizador -------------------------------------------------- #
    from transformers import AutoTokenizer

    tok_dir = caminhos.descobrir_tokenizador()
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,})")

    # ---- Pool z-score canonico (1x, cache em disco) ------------------- #
    print("\n" + "=" * 100)
    print(f"POOL Z-SCORE CANONICO (Channel Independence, {len(args.feats_list)} "
          f"features: {args.feats_list})")
    print(f"borders de linhas: train [0,8640) | val [7968,11520) | test "
          f"[10848,14400) (b1s={B1S} b2s={B2S})")
    print("=" * 100)
    splits, scaler, z, b1s, b2s = carregar_pool_cache(args)
    n_janelas = [len(x) for x, _, _ in splits]
    print(f"[pool] janelas z-score: train={n_janelas[0]} val={n_janelas[1]} "
          f"test={n_janelas[2]} (total={sum(n_janelas)})")
    print(f"[zscore] StandardScaler ajustado nas linhas [0, {B2S[0]}) "
          f"(todas as colunas, por coluna)")

    # ---- Decode (escala) ---------------------------------------------- #
    decode_fn = None
    if args.decode:
        pl = CharByCharPipeline(tokenizer, block_max=args.block_max,
                                n_paginas_max=args.n_paginas_max,
                                overlap_frac=args.overlap)
        decode_fn = pl.decode
        print("[decode] ATIVO: metricas do char-by-char apos pred * max_abs "
              "(variante que INFLARIA o MSE por ~max_abs^2 - ver docstring)")
    else:
        print("[decode] OFF (padrao): o modelo ja devolve a reconstrucao na "
              "escala de entrada (z-score, RevIN interno denormaliza). "
              "--decode deslocaria as metricas para fora do espaco z.")

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
    ckpt_path = Path(next(iter(ckpts_sel.values())))

    # ---- Modelo 1: baseline numerico ---------------------------------- #
    print("\n" + "=" * 100)
    print("MODELO 1 - BASELINE NUMERICO (Run-8 VolLoss + SMETimes)")
    print("=" * 100)
    numerico: Dict
    if args.re_numerico:
        mses, maes = [], []
        for seed in seeds:
            rb.seed_everything(seed)
            r = avaliar_numerico(splits, ckpt_path, device, use_amp)
            mses.append(r["test_mse"]); maes.append(r["test_mae"])
            print(f"  [seed={seed}] eval-only: test_mse={r['test_mse']:.6f} "
                  f"test_mae={r['test_mae']:.6f}")
        numerico = {
            "fonte": "eval_only_canonico_7feats",
            "test_mse_media": round(float(np.mean(mses)), 6),
            "test_mse_desvio": round(float(np.std(mses)), 6),
            "test_mae_media": round(float(np.mean(maes)), 6),
            "test_mae_desvio": round(float(np.std(maes)), 6),
            "n_janelas_test": int(n_janelas[2]),
        }
    else:
        log = carregar_numerico_log()
        if log is None:
            print("[numerico] log nao encontrado -> re-medindo (eval-only)")
            log = {"test_mse_media": 0.0, "test_mae_media": 0.0,
                   "n_janelas_test": 0, "referencia_artigo": 0.3539}
            args.re_numerico = True
            mses, maes = [], []
            for seed in seeds:
                r = avaliar_numerico(splits, ckpt_path, device, use_amp)
                mses.append(r["test_mse"]); maes.append(r["test_mae"])
            log = {"test_mse_media": float(np.mean(mses)),
                   "test_mae_media": float(np.mean(maes)),
                   "n_janelas_test": int(n_janelas[2]),
                   "referencia_artigo": 0.3539}
        numerico = {
            "fonte": "log_numerico_base_stats.json",
            "test_mse_media": log["test_mse_media"],
            "test_mse_desvio": 0.0,
            "test_mae_media": log["test_mae_media"],
            "test_mae_desvio": 0.0,
            "n_janelas_test": int(log["n_janelas_test"]),
            "referencia_artigo": log["referencia_artigo"],
        }
        print(f"[numerico] carregado do log: test_mse={numerico['test_mse_media']:.4f} "
              f"| test_mae={numerico['test_mae_media']:.4f} | "
              f"n_janelas_test={numerico['n_janelas_test']}")

    # ---- Dados dos modelos textuais (textos 1x, cache em disco) ------- #
    print("\n" + "=" * 100)
    print("TEXTOS - PSEUDO-JSON ES (documento fixo do treino) e CHAR-BY-CHAR "
          "(documento por janela)")
    print("=" * 100)
    dados = carregar_etth1(args.csv, args.linhas)
    prep_pj = carregar_pseudo_json(args, tokenizer, dados, z)
    print(f"[pseudo-json ES] doc do treino: {prep_pj['tokens_originais']:,} tokens "
          f"originais -> {prep_pj['tokens_retidos']:,} retidos | "
          f"{prep_pj['n_paginas']} paginas")
    ds_pj = datasets_pseudo_json(args, splits, prep_pj)
    ds_cb, stats_cb = carregar_char_by_char(args, tokenizer, splits)

    loaders_pj = build_loaders(ds_pj[0], ds_pj[1], ds_pj[2], args.batch_size)
    loaders_cb = build_loaders(ds_cb[0], ds_cb[1], ds_cb[2], args.batch_size)

    # ================================================================= #
    #  BATERIA - 10 epocas x 3 seeds por modelo textual                  #
    # ================================================================= #
    print("\n" + "=" * 100)
    print(f"BATERIA TEXTUAL - {args.epochs} EPOCAS x {len(seeds)} SEEDS "
          f"(isolamento estrito, restore_best por validacao)")
    is_dual = args.variante == "dual"
    if is_dual:
        desc = ("trend+osc GLU gate" if args.dual_fusao == "glu"
                else "trend+osc soma (ablation gating)")
        print(f"  [variante] AutoTimesDualModel ({desc})")
    else:
        print("  [variante] AutoTimesModel original")
    print("=" * 100)
    if is_dual:
        model_fn = functools.partial(montar_modelo_dual, fusao=args.dual_fusao)
    else:
        model_fn = montar_modelo
    res_pj = rodar_bateria_textual(args, "Pseudo-JSON ES", loaders_pj,
                                   ckpt_path, device, use_amp, seeds,
                                   model_fn=model_fn)
    res_cb = rodar_bateria_textual(args, "Char-by-Char", loaders_cb,
                                   ckpt_path, device, use_amp, seeds,
                                   decode_fn=decode_fn, model_fn=model_fn)

    # ---- Consolidacao ------------------------------------------------- #
    resumo: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocolo": "apples_to_apples_trio",
        "escala": "zscore",
        "config": {
            "variante_arquitetura": args.variante,
            "dual_fusao": args.dual_fusao if args.variante == "dual" else None,
            "feats": args.feats_list,
            "n_features": len(args.feats_list),
            "splits_linhas": {"train": [B1S[0], B2S[0]],
                              "val": [B1S[1], B2S[1]],
                              "test": [B1S[2], B2S[2]]},
            "janelas_por_split": dict(zip(("train", "val", "test"), n_janelas)),
            "scaler_fit": f"[0, {B2S[0]}) todas as colunas",
            "scaler_mean_por_feat": [round(float(m), 6) for m in scaler.mean_],
            "scaler_std_por_feat": [round(float(s), 6) for s in scaler.scale_],
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "block_max": args.block_max,
            "n_paginas_max": args.n_paginas_max,
            "overlap": args.overlap,
            "checkpoint": str(ckpt_path),
            "restore_best": True,
            "decode_aplicado": bool(args.decode),
            "amp": use_amp,
            "linhas_csv": args.linhas,
        },
        "modelos": {
            "numerico": numerico,
            "pseudo_json_es": dict(res_pj, **{
                "template": TEMPLATE_PSEUDO_JSON,
                "estrategia": ESTRATEGIA_PAGINACAO,
                "tokens_originais": prep_pj["tokens_originais"],
                "tokens_retidos": prep_pj["tokens_retidos"],
                "paginas": prep_pj["n_paginas"],
                "paginas_descartadas": prep_pj["paginas_descartadas"],
            }),
            "char_by_char": dict(res_cb, **{
                "max_abs_media": round(float(np.mean([s["max_abs_media"] for s in stats_cb.values()])), 6) if stats_cb else None,
                "max_abs_desvio": round(float(np.mean([s["max_abs_desvio"] for s in stats_cb.values()])), 6) if stats_cb else None,
                "paginas_media": round(float(np.mean([s["paginas_media"] for s in stats_cb.values()])), 2) if stats_cb else None,
                "paginas_descartadas_media": round(float(np.mean([s["paginas_descartadas_media"] for s in stats_cb.values()])), 2) if stats_cb else None,
                "decode_aplicado": bool(args.decode),
            }),
        },
    }
    imprimir_tabela(resumo)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"[ok] JSON: {JSON_OUT}")
    if args.plots:
        chave_grafico = sufixo if args.variante == "dual" else "original"
        for p in gerar_graficos(resumo, variante=chave_grafico):
            print(f"[ok] grafico: {p}")
    print("Confronto Apples-to-Apples concluido.")


if __name__ == "__main__":
    main()