"""
auditoria_harness - Diagnostico da discrepancia 5.80 (harness) vs 0.3539
(ablation original) no benchmark numerico base do ETTh1.

RESUMO DO VEREDITO (verificado no codigo-fonte):

    1) ESCALA (CAUSA PRINCIPAL):
       - Ablation original  (Testes_eficiencia/run_benchmarks.py:657):
         `StandardScaler().fit(values[train])` -> TODO o dataset entra no
         espaco z-score GLOBAL antes do janelamento. O MSE 0.3539 (Run-8
         VolLoss+SMETimes) e calculado NESSE espaco.
       - Harness testes_letras (run_etth1_texto.py:118 / pool_janelas_etth1):
         carrega o CSV em escala RAW (sem StandardScaler). O MSE 5.80 e
         calculado na escala raw da coluna HUFL (feat 0).
       -> 5.80 e 0.3539 NAO sao comparaveis: estao em espacos de medida
          diferentes. Fator de conversao por feature = variancia do treino
          (sigma^2): HUFL=33.79, OT=84.21, etc.

    2) VOLUME DE DADOS (CAUSA SECUNDARIA):
       - Original: stride=1, TODAS as janelas -> train=55.111, val=19.495,
         test=19.495 (7 features x 7.873/2.785 janelas por split).
       - Harness: 2000 linhas, pool de 192 janelas -> train=115, val=38,
         test=39, somente HUFL.
       -> Underfitting massivo + erro estatistico alto.

    3) LEAKAGE TEMPORAL (CAUSA AGRAVANTE):
       - Original: train [0,8640), val [8640,11520), test [11520,14400) -
         regioes temporais disjuntas (12/4/4 meses).
       - Harness: pool amostrado das linhas 0..2000 e dividido 60/20/20 ->
         o "teste" sai da MESMA regiao temporal do treino (janelas
         sobrepostas da mesma serie) -> MSE otimista e de baixa confianca.

    4) PROTOCOLO:
       - Original: batch=256, cosine, restauracao do MELHOR checkpoint por
         val_loss antes do teste final.
       - Harness: batch=16, sem restauracao do melhor estado.
       -> contribuicao menor, mas incluida por completude.

USO:
    python -m testes_letras.auditoria_harness            # so diagnostico
    python -m testes_letras.auditoria_harness --eval-batches 8
        # avalia o checkpoint smetimes nas DUAS escalas (mesmas janelas) e
        # mede empiricamente a razao mse_z / mse_raw (deve ~= 1/sigma^2).

SAIDAS:
    resultados/auditoria_harness_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import caminhos
from .run_etth1_texto import _FEAT_PADRAO, carregar_etth1, pool_janelas_etth1

SEQ_LEN: int = 672
LABEL_LEN: int = 576
PRED_LEN: int = 96


def stats_standardscaler_train(csv_path: Path) -> Dict:
    """Replica EXATAMENTE o scaler do load_real_split (run_benchmarks:657).

    StandardScaler ajustado em values[b1s[0]:b2s[0]] (treino de 12 meses),
    com os mesmos bounds de _compute_split_bounds('etth1', ...).
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(csv_path)
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feat_names = list(df.columns[1:])
    n_total, n_vars = values.shape
    month = 720
    b1s = [0, 12 * month - SEQ_LEN, 16 * month - SEQ_LEN]
    b2s = [12 * month, 16 * month, 20 * month]
    b1s = [max(0, min(b, n_total - SEQ_LEN - PRED_LEN)) for b in b1s]
    b2s = [max(0, min(b, n_total)) for b in b2s]

    scaler = StandardScaler().fit(values[b1s[0]:b2s[0]])
    var_train = scaler.scale_ ** 2
    return {
        "feat_names": feat_names,
        "n_vars": n_vars,
        "n_total": n_total,
        "bounds": {"train": [b1s[0], b2s[0]], "val": [b1s[1], b2s[1]],
                   "test": [b1s[2], b2s[2]]},
        "media_treino": {f: float(m) for f, m in zip(feat_names, scaler.mean_)},
        "std_treino": {f: float(s) for f, s in zip(feat_names, scaler.scale_)},
        "var_treino": {f: float(v) for f, v in zip(feat_names, var_train)},
        "janelas_por_feature": {
            "train": (b2s[0] - b1s[0]) - SEQ_LEN - PRED_LEN + 1,
            "val": (b2s[1] - b1s[1]) - SEQ_LEN - PRED_LEN + 1,
            "test": (b2s[2] - b1s[2]) - SEQ_LEN - PRED_LEN + 1,
        },
    }


def batch_original(rb, data_dir: str, data_path: str, batch: int = 16):
    """1 batch do loader ORIGINAL (load_real_split, stride=1, z-score)."""
    _, tr, _, _ = rb.load_real_split(
        data_dir, data_path, SEQ_LEN, LABEL_LEN, PRED_LEN, 1
    )
    x, y = tr[:batch]
    return np.asarray(x), np.asarray(y)


def batch_harness(csv_path: Path, pool: int = 192):
    """1 batch do harness (pool_janelas_etth1, escala raw)."""
    dados = carregar_etth1(str(csv_path), 2000)
    pool_ = pool_janelas_etth1(dados, pool)
    xs = np.stack([x for x, _ in pool_])
    ys = np.stack([y for _, y in pool_])
    return xs[:16], ys[:16]


def resumo_tensor(nome: str, x: np.ndarray, y: np.ndarray) -> Dict:
    return {
        "tensor": nome,
        "shape": list(x.shape),
        "media_x": float(x.mean()),
        "std_x": float(x.std()),
        "var_x": float(x.var()),
        "media_y": float(y.mean()),
        "std_y": float(y.std()),
        "var_y": float(y.var()),
    }


def avaliar_escala(
    rb, model, x_np: np.ndarray, y_np: np.ndarray, device: torch.device,
    desc: str,
) -> Dict:
    """MSE/MAE do checkpoint sobre um lote (x,y) na escala dada."""
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    with torch.no_grad():
        pred = model(x)
    p = pred[:, -PRED_LEN:, :].float().cpu().numpy()
    t = y[:, -PRED_LEN:, :].float().cpu().numpy()
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    print(f"  [eval:{desc}] MSE(horizonte 96) = {mse:.6f} | MAE = {mae:.6f}")
    return {"escala": desc, "mse_pred": mse, "mae_pred": mae}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--eval-batches", dest="eval_batches", type=int, default=0,
                    help=">0: avalia o checkpoint smetimes em N janelas de "
                         "cada escala (mesmas janelas) e mede a razao empirica")
    ap.add_argument("--pool", type=int, default=192,
                    help="janelas do harness para comparacao de batch")
    args = ap.parse_args()

    caminhos.criar_resultados()
    rb = caminhos.importar_run_benchmarks()
    device = rb.get_device()
    csv_path = Path(args.csv) if args.csv else caminhos.DATA_DIR / caminhos.DATA_PATH

    print("=" * 100)
    print("AUDITORIA DO HARNESS - escala / normalizacao / paridade de dados")
    print("=" * 100)

    # ---- 1) Escala: stats do StandardScaler original ------------------ #
    stats = stats_standardscaler_train(csv_path)
    print(f"\n[1] StandardScaler ORIGINAL (fit no treino {stats['bounds']['train']}):")
    for f in stats["feat_names"]:
        print(f"    {f:<6s} media={stats['media_treino'][f]:>8.4f} "
              f"std={stats['std_treino'][f]:>8.4f} var={stats['var_treino'][f]:>8.2f}")
    var_feat0 = stats["var_treino"][stats["feat_names"][_FEAT_PADRAO]]
    print(f"    -> fator de conversao (raw->z) da coluna usada pelo harness "
          f"({stats['feat_names'][_FEAT_PADRAO]}): 1/var = "
          f"{1.0 / var_feat0:.6f}")

    # ---- 2) 1 batch de cada pipeline ----------------------------------- #
    print(f"\n[2] Comparacao de escala - 1 batch de cada pipeline:")
    x_orig, y_orig = batch_original(rb, str(caminhos.DATA_DIR), caminhos.DATA_PATH)
    x_har, y_har = batch_harness(csv_path, args.pool)
    r_orig = resumo_tensor("original (load_real_split, z-score)", x_orig, y_orig)
    r_har = resumo_tensor("harness (pool_janelas_etth1, raw)", x_har, y_har)
    for r in (r_orig, r_har):
        print(f"    [{r['tensor']}] media_x={r['media_x']:.4f} "
              f"std_x={r['std_x']:.4f} | media_y={r['media_y']:.4f} "
              f"std_y={r['std_y']:.4f}")

    # ---- 3) Paridade de dados ------------------------------------------ #
    jf = stats["janelas_por_feature"]
    jf_tot = {k: v * stats["n_vars"] for k, v in jf.items()}
    print(f"\n[3] Paridade de dados:")
    print(f"    ORIGINAL (stride=1): train={jf_tot['train']:,} "
          f"val={jf_tot['val']:,} test={jf_tot['test']:,} "
          f"(janelas univariadas, {stats['n_vars']} features)")
    print(f"    HARNESS (2000 linhas, pool=192): train=115 val=38 test=39 "
          f"(somente {stats['feat_names'][_FEAT_PADRAO]})")
    print(f"    Reducao de dados de treino: "
          f"x{jf_tot['train'] / 115:.0f}")
    print(f"    Leakage temporal: ORIGINAL test=[11520,14400) (regiao "
          f"distinta); HARNESS test dentro das mesmas linhas 0..2000 "
          f"do treino (janelas sobrepostas) -> MSE otimista.")

    # ---- 4) Verificacao empirica do fator de conversao ---------------- #
    report: Dict = {
        "veredito": {
            "mse_harness_raw": 5.7988,        # numerico_base_stats.json
            "mse_ablation_zscore": 0.3539,    # Run-8 VolLoss+SMETimes
            "comparaveis": False,
            "causa_principal": "escala (raw vs StandardScaler z-score)",
            "causa_secundaria": "volume (115 vs 55.111 janelas de treino)",
            "causa_agravante": "leakage temporal no split 60/20/20 do pool",
        },
        "stats_standardscaler": stats,
        "batch_original": r_orig,
        "batch_harness": r_har,
        "paridade": {
            "janelas_original": jf_tot,
            "janelas_harness": {"train": 115, "val": 38, "test": 39},
            "leakage_temporal": (
                "harness: teste dentro de 0..2000 (mesma regiao do treino); "
                "original: teste em [11520,14400)"
            ),
        },
        "conversao": {
            "formula": "mse_z = mse_raw / var_treino(feature)",
            f"var_treino_{stats['feat_names'][_FEAT_PADRAO]}": var_feat0,
            "mse_raw_5.80_equiv_z": round(5.7988 / var_feat0, 6),
            "mse_raw_7.03_textual_equiv_z": round(7.0336 / var_feat0, 6),
            "ressalva": (
                "a conversao por variancia so remove o efeito de escala; "
                "o volume de dados e o leakage continuam viciando o 5.80/7.03. "
                "A base publicavel exige o protocolo original (z-score, "
                "dataset completo, split temporal disjunto)."
            ),
        },
    }

    # ---- 5) (opcional) eval do checkpoint nas duas escalas -------------- #
    if args.eval_batches > 0:
        from .modelos import CKPTS
        from .benchmark_numerico_base import montar_modelo_numerico

        print(f"\n[5] Eval empirico do checkpoint (n={args.eval_batches} janelas "
              f"por escala, mesmas janelas):")

        model = montar_modelo_numerico(CKPTS["smetimes"], device)
        # Janelas raw do harness (mesma regiao), e o equivalente z-score:
        dados = carregar_etth1(str(csv_path), 2000)
        pool_ = pool_janelas_etth1(dados, args.pool)[:args.eval_batches]
        xs = np.stack([x for x, _ in pool_]).astype(np.float32)
        ys = np.stack([y for _, y in pool_]).astype(np.float32)

        eval_raw = avaliar_escala(rb, model, xs, ys, device, "raw")
        xz = (xs - stats["media_treino"][stats["feat_names"][_FEAT_PADRAO]]) / \
            stats["std_treino"][stats["feat_names"][_FEAT_PADRAO]]
        yz = (ys - stats["media_treino"][stats["feat_names"][_FEAT_PADRAO]]) / \
            stats["std_treino"][stats["feat_names"][_FEAT_PADRAO]]
        eval_z = avaliar_escala(rb, model, xz, yz, device, "zscore")
        razao = eval_raw["mse_pred"] / max(eval_z["mse_pred"], 1e-12)
        report["eval_empirico"] = {
            "raw": eval_raw, "zscore": eval_z,
            "razao_mse_raw_sobre_z": round(razao, 4),
            "var_treino_esperada": round(var_feat0, 4),
        }
        print(f"    razao empirica mse_raw/mse_z = {razao:.2f} "
              f"(var_treino esperada = {var_feat0:.2f})")

    out = caminhos.RESULTADOS_DIR / "auditoria_harness_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[ok] relatorio: {out}")
    print("\nCONCLUSAO: o 5.80 NAO pode ser comparado ao 0.3539 diretamente. "
          "Corrija o benchmark numerico base para o protocolo original "
          "(StandardScaler z-score + dataset completo + split temporal "
          "disjunto) antes de publicar qualquer grafico.")


if __name__ == "__main__":
    main()