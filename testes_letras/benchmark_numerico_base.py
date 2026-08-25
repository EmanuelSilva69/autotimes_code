"""
benchmark_numerico_base - Numerico Base para a Jornada do Erro.

CORRIGIDO (apos auditoria_harness.py): o harness original calculava o MSE na
escala RAW (sem o StandardScaler do pipeline original), com 192 janelas de
2000 linhas e split com leakage temporal - por isso reportava 5.80 contra o
0.3539 do ablation (que avalia no espaco z-score global, dataset completo,
split temporal disjunto).

DOIS PROTOCOLOS:

  1) --protocolo original (DEFAULT, paridade com o artigo):
       - StandardScaler fitado no split de treino (12 meses), como o
         load_real_split do run_benchmarks.py (linha 657);
       - dataset COMPLETO (17.420 linhas), stride=1 -> train=55.111 janelas;
       - split temporal disjunto: train [0,8640) val [8640,11520)
         test [11520,14400);
       - batch=256, cosine, restauracao do melhor checkpoint (restore_best);
       - MSE reportado no espaco z-score (mesmo do artigo: 0.3539 p/ Run-8).
       - --vars all (todas as 7 features, paridade total) ou --vars hufl
         (somente a coluna 0, comparavel aos marcos textuais);

  2) --protocolo harness (compatibilidade):
       - escala raw, pool de 192 janelas de 2000 linhas, split 60/20/20
         (somente para comparacao dentro do MESMO harness dos marcos
         textuais; NAO publicavel como baseline absoluta).

SEMPRE reporta as duas escalas: test_mse_media (escala do protocolo),
test_mse_media_z e test_mse_media_raw (quando aplicavel, via var_treino).

--eval-only (recomendado p/ validacao rapida): sem treinar, avalia o
checkpoint carregado no test set do protocolo original e compara com o
0.3539 oficial do ablation (Run-8 VolLoss+SMETimes). Deve reproduzir
aproximadamente o valor do artigo, provando que o harness esta correto.

SAIDAS:
    resultados/numerico_base_stats.json
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .modelos import CKPTS, carregar_checkpoint
from .run_etth1_texto import (
    _FEAT_PADRAO,
    carregar_etth1,
    pool_janelas_etth1,
    salvar_benchmark,
)
from .treino import build_loaders, rodar_convergencia

SEQ_LEN: int = 672
LABEL_LEN: int = 576
PRED_LEN: int = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerico Base (protocolo original z-score OU harness raw)"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes")
    parser.add_argument("--protocolo", type=str, default="original",
                        choices=["original", "harness"],
                        help="'original' = z-score + dataset completo + split "
                             "temporal disjunto (paridade com o artigo); "
                             "'harness' = raw + pool 192 (comparacao textual)")
    parser.add_argument("--vars", type=str, default="all", choices=["all", "hufl"],
                        help="'all' = 7 features (paridade total com o "
                             "0.3539); 'hufl' = somente coluna 0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="2021")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size",
                        type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--linhas", type=int, default=2000)
    parser.add_argument("--pool-janelas", dest="pool_janelas", type=int, default=192)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval-only", dest="eval_only", action="store_true",
                        help="Avalia o checkpoint diretamente no test set do "
                             "protocolo original (sem treinar) e compara com "
                             "o 0.3539 do ablation")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    return parser.parse_args()


def montar_modelo_numerico(
    ckpt_path: Path,
    device: torch.device,
    use_amp: bool = False,
) -> torch.nn.Module:
    """AutoTimesModel com textual=False (forward numerico puro), SEM fusion."""
    rb = caminhos.importar_run_benchmarks()
    ckpt = carregar_checkpoint(ckpt_path)
    flags: Dict = ckpt["flags"]
    args: Dict = ckpt["args"]
    torch_dtype = torch.float16 if (use_amp and device.type == "cuda") else torch.float32

    model = rb.AutoTimesModel(
        seq_len=args["seq_len"],
        pred_len=args["pred_len"],
        n_vars=args["n_vars"],
        label_len=args["label_len"],
        mlp_hidden_dim=args["mlp_hidden_dim"],
        mlp_hidden_layers=args["mlp_hidden_layers"],
        mlp_activation=args["mlp_activation"],
        dropout=args["dropout"],
        gpt2_dir=Path(args["gpt2_dir"]),
        torch_dtype=torch_dtype,
        use_robust=bool(flags["use_robust"]),
        use_overlap=bool(flags["use_overlap"]),
        use_vol_gate=bool(flags["use_vol_gate"]),
        unfreeze_layers=0,
        textual=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


# --------------------------------------------------------------------- #
#  Protocolo ORIGINAL (z-score + dataset completo + split disjunto)      #
# --------------------------------------------------------------------- #
def carregar_protocolo_original(
    rb, args: argparse.Namespace, device: torch.device, ckpt_path: Path,
) -> Dict:
    """Splits no padrao do ablation (load_real_split) + stats do scaler.

    Retorna dict com train/val/test (TensorDataset x,y z-score), var_treino
    por feature e flags do protocolo.
    """
    n_vars, train_ds, val_ds, test_ds = rb.load_real_split(
        str(caminhos.DATA_DIR), caminhos.DATA_PATH,
        SEQ_LEN, LABEL_LEN, PRED_LEN, 1,
    )

    # Stats do StandardScaler (mesmos bounds do load_real_split).
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(caminhos.DATA_DIR / caminhos.DATA_PATH)
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feat_names = list(df.columns[1:])
    n_total = len(df)
    month = 720
    b1s = [0, 12 * month - SEQ_LEN, 16 * month - SEQ_LEN]
    b2s = [12 * month, 16 * month, 20 * month]
    b1s = [max(0, min(b, n_total - SEQ_LEN - PRED_LEN)) for b in b1s]
    b2s = [max(0, min(b, n_total)) for b in b2s]
    scaler = StandardScaler().fit(values[b1s[0]:b2s[0]])
    var_treino = {f: float(s ** 2) for f, s in zip(feat_names, scaler.scale_)}

    def _filtrar_feat(ds: TensorDataset, feat: int) -> TensorDataset:
        """Mantem apenas janelas da feature `feat` (as janelas por feature
        sao contiguas no dataset: as primeiras tot_len janelas = feat 0)."""
        tot = (b2s[0] - b1s[0]) - SEQ_LEN - PRED_LEN + 1
        n = len(ds)
        per_feat = n // n_vars
        sl = slice(feat * per_feat, (feat + 1) * per_feat)
        return TensorDataset(ds.tensors[0][sl], ds.tensors[1][sl])

    if args.vars == "hufl":
        train_ds = _filtrar_feat(train_ds, _FEAT_PADRAO)
        val_ds = _filtrar_feat(val_ds, _FEAT_PADRAO)
        test_ds = _filtrar_feat(test_ds, _FEAT_PADRAO)

    return {
        "train": train_ds, "val": val_ds, "test": test_ds,
        "var_treino": var_treino,
        "escala": "zscore",
        "nome_escala": "z-score global (StandardScaler fit no treino)",
        "fator_hufl": var_treino[feat_names[_FEAT_PADRAO]],
        "feat_names": feat_names,
    }


def _split_harness(args: argparse.Namespace) -> Dict:
    """Pool raw (mesmo harness dos marcos textuais)."""
    dados = carregar_etth1(args.csv, args.linhas)
    pool = pool_janelas_etth1(dados, args.pool_janelas)
    n_amostras = len(pool)
    n_tr, n_va = int(n_amostras * 0.6), int(n_amostras * 0.2)
    xs = torch.from_numpy(np.stack([x for x, _ in pool])).float()
    ys = torch.from_numpy(np.stack([y for _, y in pool])).float()
    tr = TensorDataset(xs[:n_tr], ys[:n_tr])
    va = TensorDataset(xs[n_tr:n_tr + n_va], ys[n_tr:n_tr + n_va])
    te = TensorDataset(xs[n_tr + n_va:], ys[n_tr + n_va:])
    var_hufl = _var_treino_hufl()
    return {
        "train": tr, "val": va, "test": te,
        "escala": "raw",
        "nome_escala": "escala raw (sem StandardScaler)",
        "var_treino": {"HUFL": var_hufl},
        "fator_hufl": var_hufl,
        "feat_names": dados["feat_names"],
        "n_pool": n_amostras,
    }


def _var_treino_hufl() -> float:
    """Variancia do treino da HUFL (fator raw->z), com fallback default."""
    from .auditoria_harness import stats_standardscaler_train

    try:
        stats = stats_standardscaler_train(
            caminhos.DATA_DIR / caminhos.DATA_PATH)
        var = stats["var_treino"]["HUFL"]
        if var and var > 0:
            return float(var)
    except Exception:
        pass
    return 33.7877


# --------------------------------------------------------------------- #
#  Avaliacao direta do checkpoint (--eval-only)                          #
# --------------------------------------------------------------------- #
@torch.no_grad()
def avaliar_loader(model, loader, device, pred_len: int = PRED_LEN) -> Dict:
    """MSE/MAE do modelo (modo eval) sobre um loader completo."""
    model.eval()
    preds_p, trues_p = [], []
    for batch in loader:
        xs, y = batch[:-1], batch[-1]
        x = xs[0].to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        preds_p.append(pred[:, -pred_len:, :].float().cpu().numpy())
        trues_p.append(y[:, -pred_len:, :].float().cpu().numpy())
    p = np.concatenate(preds_p, 0)
    t = np.concatenate(trues_p, 0)
    return {
        "mse_pred": float(np.mean((p - t) ** 2)),
        "mae_pred": float(np.mean(np.abs(p - t))),
        "n_janelas": int(p.shape[0]),
    }


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

    # ---- Dados -------------------------------------------------------- #
    if args.protocolo == "original":
        dados_p = carregar_protocolo_original(rb, args, device, next(iter(ckpts_sel.values())))
        print(f"\n[protocolo ORIGINAL] {dados_p['nome_escala']}")
        print(f"  train={len(dados_p['train']):,} val={len(dados_p['val']):,} "
              f"test={len(dados_p['test']):,} janelas "
              f"(vars={args.vars}) | batch={args.batch_size}")
    else:
        dados_p = _split_harness(args)
        print(f"\n[protocolo HARNESS] {dados_p['nome_escala']} "
              f"(pool={dados_p['n_pool']})")
        print(f"  train={len(dados_p['train'])} val={len(dados_p['val'])} "
              f"test={len(dados_p['test'])} | batch={args.batch_size}")

    tr_ld, va_ld, te_ld = build_loaders(
        dados_p["train"], dados_p["val"], dados_p["test"], args.batch_size
    )

    resumo: Dict = {
        "config": {
            "modo": "numerico_puro",
            "protocolo": args.protocolo,
            "vars": args.vars,
            "escala": dados_p["escala"],
            "escala_nome": dados_p["nome_escala"],
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "eval_only": args.eval_only,
        },
        "escala": {
            "reportada": dados_p["escala"],
            "var_treino": {k: round(v, 4) for k, v in dados_p["var_treino"].items()},
            "fator_hufl_raw_para_z": 1.0 / dados_p["fator_hufl"]
            if dados_p["fator_hufl"] else None,
        },
        "paridade": {
            "train": len(dados_p["train"]), "val": len(dados_p["val"]),
            "test": len(dados_p["test"]),
            "referencia_artigo": {
                "train": 55111, "val": 19495, "test": 19495,
                "mse_run8_zscore": 0.3539,
            },
        },
        "modelos": {},
    }

    for chave, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")

        # ---- EVAL-ONLY: checkpoint direto no test do protocolo original -#
        if args.eval_only:
            rb.seed_everything(seeds[0])
            model = montar_modelo_numerico(ckpt_path, device, use_amp=use_amp)
            t0 = time.perf_counter()
            ev = avaliar_loader(model, te_ld, device)
            print(f"  [eval-only] checkpoint direto no test set "
                  f"({ev['n_janelas']:,} janelas): "
                  f"MSE_z={ev['mse_pred']:.4f} MAE_z={ev['mae_pred']:.4f} "
                  f"({time.perf_counter() - t0:.1f}s)")
            print(f"  [comparacao] ablation Run-8: 0.3539 | delta="
                  f"{100 * (ev['mse_pred'] - 0.3539) / 0.3539:+.1f}%")
            resumo["modelos"][chave] = {
                "modo": "eval_only",
                "test_mse_media": round(ev["mse_pred"], 6),
                "test_mae_media": round(ev["mae_pred"], 6),
                "test_mse_media_z": round(ev["mse_pred"], 6),
                "test_mae_media_z": round(ev["mae_pred"], 6),
                "test_mse_media_raw": round(
                    ev["mse_pred"] * dados_p["fator_hufl"], 6)
                if dados_p["fator_hufl"] and args.vars == "hufl" else None,
                "n_janelas_test": ev["n_janelas"],
                "referencia_artigo": 0.3539,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # ---- Treino de convergencia (seeds x epocas) ------------------ #
        por_seed: Dict[str, Dict] = {}
        mse_list, mae_list, bestval_list, epoca_list, estouros_list = (
            [], [], [], [], [])
        val_por_epoca: List[List[float]] = []
        for seed in seeds:
            rb.seed_everything(seed)
            model = montar_modelo_numerico(ckpt_path, device, use_amp=use_amp)
            n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [mount seed={seed}] treinaveis={n_treinaveis:,}")
            t0 = time.perf_counter()
            result = rodar_convergencia(
                model, tr_ld, va_ld, te_ld,
                epochs=args.epochs, lr=args.lr, device=device,
                use_amp=use_amp, seed=seed, restore_best=True,
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

        mse_media = float(np.mean(mse_list))
        mae_media = float(np.mean(mae_list))
        fator = dados_p["fator_hufl"]
        if dados_p["escala"] == "zscore" and args.vars == "hufl":
            mse_z, mse_raw = mse_media, mse_media * fator
        elif dados_p["escala"] == "raw" and fator is not None:
            mse_raw, mse_z = mse_media, mse_media / fator
        else:
            mse_z, mse_raw = mse_media, None

        resumo["modelos"][chave] = {
            "test_mse_media": round(mse_media, 4),
            "test_mse_desvio": round(float(np.std(mse_list)), 4),
            "test_mae_media": round(mae_media, 4),
            "test_mae_desvio": round(float(np.std(mae_list)), 4),
            "test_mse_media_z": round(mse_z, 4),
            "test_mse_media_raw": round(mse_raw, 4) if mse_raw else None,
            "melhor_val_mse_media": round(float(np.mean(bestval_list)), 4),
            "melhor_val_mse_desvio": round(float(np.std(bestval_list)), 4),
            "melhor_epoca_media": round(float(np.mean(epoca_list)), 2),
            "estouros_total": sum(estouros_list),
            "val_mse_media": [round(x, 6) for x in val_media],
            "val_mse_desvio": [round(x, 6) for x in val_desvio],
            "por_seed": por_seed,
            "referencia_artigo": 0.3539,
        }
        c = resumo["modelos"][chave]
        print(f"  [ok] {chave}: MSE({dados_p['escala']}) {mse_media:.4f} ± "
              f"{c['test_mse_desvio']:.4f}"
              + (f" | equiv. raw {mse_raw:.4f}" if mse_raw else "")
              + (f" | equiv. z {mse_z:.4f}" if mse_raw is not None and dados_p['escala'] == 'raw' else "")
              + f" | estouros={c['estouros_total']}")

    out = salvar_benchmark(resumo, nome="numerico_base_stats.json")
    print(f"[ok] JSON: {out}")
    print("Numerico Base concluido.")


if __name__ == "__main__":
    main()