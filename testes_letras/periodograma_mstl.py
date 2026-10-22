"""periodograma_mstl - Análise espectral para detecção de períodos MSTL.

Script isolado que:
  1. Lê o ETTh1.csv (apenas treino: [0, 8640))
  2. Calcula o periodograma (FFT) para cada feature
  3. Detecta os N picos de maior potência (filtrando artefatos)
  4. Gera gráfico PNG com periodograma e picos marcados
  5. Imprime tabela consolidada + ranqueamento de combinações MSTL

Uso:
    python -m testes_letras.periodograma_mstl
    python -m testes_letras.periodograma_mstl --csv caminho/para/csv
    python -m testes_letras.periodograma_mstl --n-peaks 20 --feat 0
    python -m testes_letras.periodograma_mstl --todas-features --n-peaks 20
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram, find_peaks

from . import caminhos

# --------------------------------------------------------------------- #
#  Constantes                                                             #
# --------------------------------------------------------------------- #
RECENTE_DIR: Path = caminhos.BASE_DIR / "recente"
N_TREINO: int = 8640

# Períodos "conhecidos" de interesse para MSTL (em passos, timestep=60min)
PERIODOS_CONHECIDOS = {
    6:   "6h",
    8:   "8h",
    12:  "12h (meia-dia)",
    24:  "24h (diário)",
    168: "168h (semanal)",
    720: "720h (30 dias)",
}


# --------------------------------------------------------------------- #
#  Deteccao de periodos via periodograma                                  #
# --------------------------------------------------------------------- #
def detectar_periodograma(
    csv_path: str,
    n_peaks: int = 20,
    feat_idx: int = 0,
    n_treino: int = N_TREINO,
    max_periodo_frac: float = 0.5,
) -> Dict:
    """Detecta períodos dominantes via periodograma (FFT).

    Args:
        csv_path: caminho para o CSV.
        n_peaks: número de picos a retornar.
        feat_idx: índice da feature a analisar (0=HUFL).
        n_treino: quantas linhas usar (apenas treino, sem leakage).
        max_periodo_frac: fração máxima do período em relação ao dataset
                          (ex: 0.5 = descarta períodos > 50% do dataset).

    Returns:
        dict com chaves: freqs, potencia, picos, fs, n_samples, feat_name, etc.
    """
    df = pd.read_csv(csv_path)
    feat_name = df.columns[feat_idx + 1]
    valores = df.iloc[:n_treino, feat_idx + 1].to_numpy(dtype=np.float64)

    dates = pd.to_datetime(df.iloc[:, 0])
    minutos = (dates.iloc[1] - dates.iloc[0]).total_seconds() / 60.0
    fs = 1.0 / minutos  # frequência de amostragem (por minuto)

    n_samples = len(valores)

    # Periodograma com detrend linear
    freqs, potencia = periodogram(valores, fs=fs, scaling="spectrum", detrend="linear")

    # Filtrar: frequências > 0, < Nyquist, e período < max_periodo_frac * N
    periodo_max_passos = int(n_samples * max_periodo_frac)
    periodo_min_passos = 2  # pelo menos 2 passos

    # Converter frequências para períodos em passos
    with np.errstate(divide="ignore", invalid="ignore"):
        periodos_passos = np.where(freqs > 0, 1.0 / (freqs * minutos), np.inf)

    mascara = (
        (freqs > 0)
        & (freqs < fs / 2)
        & (periodos_passos >= periodo_min_passos)
        & (periodos_passos <= periodo_max_passos)
    )
    freqs_f = freqs[mascara]
    pot_f = potencia[mascara]
    periodos_f = periodos_passos[mascara]

    # Detectar picos com find_peaks
    if len(pot_f) > 0:
        # prominence: pelo menos 2% da potência máxima
        prom_threshold = np.max(pot_f) * 0.02
        indices_picos, props = find_peaks(
            pot_f,
            distance=3,
            prominence=prom_threshold,
        )

        # Se não encontrar suficiente, usar os N maiores
        if len(indices_picos) < n_peaks:
            indices_picos = np.argsort(pot_f)[-n_peaks * 3:][::-1]

        # Ordenar por potência (decrescente)
        pot_picos = pot_f[indices_picos]
        ordem = np.argsort(pot_picos)[::-1][:n_peaks]
        indices_picos = indices_picos[ordem]
        pot_picos = pot_f[indices_picos]
    else:
        indices_picos = np.array([], dtype=int)
        pot_picos = np.array([])

    # Montar lista de picos
    picos = []
    for rank, (idx, pot) in enumerate(zip(indices_picos, pot_picos), 1):
        freq_pico = freqs_f[idx]
        periodo_minutos = 1.0 / freq_pico
        periodo_passos = int(round(periodo_minutos / minutos))
        horas = periodo_minutos / 60.0
        if horas >= 24:
            horas_str = f"{horas / 24:.1f} dias"
        else:
            horas_str = f"{horas:.1f} h"

        # Marcar se é período "conhecido"
        conhecido = ""
        for p_ref, nome in PERIODOS_CONHECIDOS.items():
            if abs(periodo_passos - p_ref) <= 2:
                conhecido = f" [{nome}]"
                break

        picos.append({
            "periodo_passos": periodo_passos,
            "potencia": float(pot),
            "rank": rank,
            "horas_str": horas_str,
            "horas": horas,
            "freq": freq_pico,
            "conhecido": conhecido,
        })

    return {
        "freqs": freqs,
        "potencia": potencia,
        "freqs_f": freqs_f,
        "pot_f": pot_f,
        "periodos_f": periodos_f,
        "picos": picos,
        "fs": fs,
        "n_samples": n_samples,
        "feat_name": feat_name,
        "minutos_timestep": minutos,
        "periodo_max_passos": periodo_max_passos,
    }


# --------------------------------------------------------------------- #
#  Gerar grafico                                                         #
# --------------------------------------------------------------------- #
def plotar_periodograma(resultado: Dict, save_path: Path) -> Path:
    """Gera gráfico do periodograma com picos marcados."""
    freqs = resultado["freqs"]
    potencia = resultado["potencia"]
    picos = resultado["picos"]
    feat_name = resultado["feat_name"]
    n_samples = resultado["n_samples"]
    minutos = resultado["minutos_timestep"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    # --- Gráfico 1: Periodograma completo (eixo log) ---
    ax = axes[0]
    ax.semilogy(freqs * 60, potencia, color="#2c3e50", linewidth=0.8, alpha=0.9)
    ax.set_title(
        f"Periodograma — {feat_name} (ETTh1, treino: {n_samples} passos, "
        f"timestep={minutos:.0f}min, detrend=linear)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Frequência (ciclos/hora)", fontsize=11)
    ax.set_ylabel("Potência Espectral (log)", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    # Marcar top 10 picos
    cores = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db",
             "#9b59b6", "#1abc9c", "#e91e63", "#ff9800", "#795548"]
    for i, pico in enumerate(picos[:10]):
        cor = cores[i % len(cores)]
        freq_hora = pico["freq"] * 60
        ax.axvline(x=freq_hora, color=cor, linestyle="--", linewidth=1.2, alpha=0.7)
        label = f"P{pico['rank']}: {pico['periodo_passos']}p ({pico['horas_str']})"
        ax.annotate(
            label,
            xy=(freq_hora, pico["potencia"]),
            xytext=(10, 10 + (i % 3) * 12),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=cor,
            arrowprops=dict(arrowstyle="->", color=cor, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=cor, alpha=0.85),
        )

    # --- Gráfico 2: Zoom no domínio do período (eixo linear) ---
    ax2 = axes[1]
    if len(resultado["freqs_f"]) > 0:
        periodos = 1.0 / (resultado["freqs_f"] * minutos)
        ax2.plot(periodos, resultado["pot_f"], color="#8e44ad", linewidth=0.8, alpha=0.9)
        ax2.set_xlabel("Período (passos)", fontsize=11)
        ax2.set_ylabel("Potência", fontsize=11)
        ax2.set_xlim(0, min(n_samples // 2, 500))
        ax2.grid(True, alpha=0.3)

        for i, pico in enumerate(picos[:10]):
            cor = cores[i % len(cores)]
            ax2.axvline(x=pico["periodo_passos"], color=cor, linestyle="--", linewidth=1.2, alpha=0.7)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------- #
#  Tabela formatada por feature                                           #
# --------------------------------------------------------------------- #
def imprimir_tabela(resultado: Dict) -> None:
    """Imprime tabela formatada com os picos detectados."""
    picos = resultado["picos"]
    feat_name = resultado["feat_name"]
    n_samples = resultado["n_samples"]
    minutos = resultado["minutos_timestep"]

    print()
    print("=" * 100)
    print(f"  PERIODOGRAMA — Feature: {feat_name} | "
          f"Amostras: {n_samples} | Timestep: {minutos:.0f}min | "
          f"Período máx: {resultado['periodo_max_passos']} passos")
    print("=" * 100)
    print()

    if not picos:
        print("  Nenhum pico significativo detectado.")
        print()
        return

    print(f"  {'Rank':<6} {'Período':<12} {'Horas':<14} {'Dias':<8} {'Potência':<14} {'Força':<8} {'Ref'}")
    print(f"  {'─'*6} {'─'*12} {'─'*14} {'─'*8} {'─'*14} {'─'*8} {'─'*10}")

    pot_max = picos[0]["potencia"] if picos else 1

    for pico in picos:
        forca = (pico["potencia"] / pot_max * 100) if pot_max > 0 else 0
        dias = pico["horas"] / 24.0
        print(f"  {pico['rank']:<6} "
              f"{pico['periodo_passos']:<12} "
              f"{pico['horas_str']:<14} "
              f"{dias:<8.2f} "
              f"{pico['potencia']:<14.2e} "
              f"{forca:<8.1f}% "
              f"{pico['conhecido']}")

    print()


# --------------------------------------------------------------------- #
#  Consolidacao e ranqueamento de combinacoes                             #
# --------------------------------------------------------------------- #
def imprimir_tabela_consolidada(resultados: List[Dict]) -> None:
    """Imprime tabela consolidada com os top períodos de cada feature (apenas < 336 passos)."""
    LIMITE_JANELA_MSTL = 336
    n_top = 8

    print()
    print("=" * 160)
    print(f"  CONSOLIDAÇÃO — Top {n_top} Períodos por Feature (filtrados: < {LIMITE_JANELA_MSTL} passos = janela MSTL)")
    print("=" * 160)
    print()

    header = f"  {'Feature':<8}"
    for i in range(1, n_top + 1):
        header += f" {'P' + str(i) + ' (passos)':<12} {'P' + str(i) + ' (horas)':<12}"
    print(header)
    print(f"  {'─'*8}" + f" {'─'*12} {'─'*12}" * n_top)

    todos_periodos_filtrados = []
    for res in resultados:
        row = f"  {res['feat_name']:<8}"
        feat_periodos = []
        count = 0
        for pico in res["picos"]:
            if pico["periodo_passos"] >= LIMITE_JANELA_MSTL:
                continue
            if count >= n_top:
                break
            row += f" {pico['periodo_passos']:<12} {pico['horas_str']:<12}"
            feat_periodos.append(pico['periodo_passos'])
            count += 1
        for _ in range(n_top - count):
            row += f" {'—':<12} {'—':<12}"
        print(row)
        todos_periodos_filtrados.append(feat_periodos)

    print()

    contagem = Counter()
    for feat_p in todos_periodos_filtrados:
        for p in feat_p:
            contagem[p] += 1

    print("  Períodos mais frequentes (apenas < 336 passos):")
    print(f"  {'Período':<12} {'Horas':<14} {'Contagem':<12} {'Gráfico'}")
    print(f"  {'─'*12} {'─'*14} {'─'*12} {'─'*30}")
    for periodo, count in contagem.most_common(15):
        horas = periodo * resultados[0]["minutos_timestep"] / 60.0
        if horas >= 24:
            horas_str = f"{horas / 24:.1f} dias"
        else:
            horas_str = f"{horas:.1f} h"
        barra = "█" * count
        print(f"  {periodo:<12} {horas_str:<14} {count}/7 feat   {barra}")

    print()


def ranquear_combinacoes(resultados: List[Dict], top_n: int = 10) -> None:
    """Ranqueia combinações de 2 e 3 períodos para MSTL.

    Critérios:
      1. Frequência: quantas features têm pelo menos 1 período da combinação no top-10
      2. Potência total: soma das potências normalizadas dos períodos
      3. Cobertura: quantas features têm TODOS os períodos da combinação

    IMPORTANTE: Filtra períodos > 336 passos (metade da janela de 672 do AutoTimes).
    """
    LIMITE_JANELA_MSTL = 336  # 672 / 2
    minutos = resultados[0]["minutos_timestep"]

    # Coletar todos os períodos únicos (top 10 de cada feature) — FILTRADOS
    todos_periodos = []
    for res in resultados:
        for pico in res["picos"][:10]:
            if pico["periodo_passos"] < LIMITE_JANELA_MSTL:
                todos_periodos.append(pico["periodo_passos"])

    # Períodos únicos ordenados por frequência
    periodos_unicos = sorted(set(todos_periodos), key=lambda p: -todos_periodos.count(p))

    # Top 15 períodos únicos para testar combinações
    candidatos = periodos_unicos[:15]

    combinacoes_testar = []

    # Combinações de 2
    for p1, p2 in combinations(candidatos, 2):
        combinacoes_testar.append((p1, p2))

    # Combinações de 3
    for p1, p2, p3 in combinations(candidatos[:10], 3):
        combinacoes_testar.append((p1, p2, p3))

    # Avaliar cada combinação
    resultados_comb = []
    for combo in combinacoes_testar:
        combo_set = set(combo)

        # 1. Frequência: quantas features têm pelo menos 1 período no top-10
        freq_count = 0
        for res in resultados:
            feat_periodos = set(p["periodo_passos"] for p in res["picos"][:10])
            if combo_set & feat_periodos:  # interseção não vazia
                freq_count += 1

        # 2. Potência total normalizada
        pot_total = 0.0
        for res in resultados:
            feat_periodos_top = {p["periodo_passos"]: p["potencia"] for p in res["picos"][:10]}
            pot_max_feat = max(feat_periodos_top.values()) if feat_periodos_top else 1
            for p in combo:
                if p in feat_periodos_top:
                    pot_total += feat_periodos_top[p] / pot_max_feat

        # 3. Cobertura: quantas features têm TODOS os períodos
        cover_count = 0
        for res in resultados:
            feat_periodos = set(p["periodo_passos"] for p in res["picos"][:10])
            if combo_set.issubset(feat_periodos):
                cover_count += 1

        # Score composto: frequência * 2 + potência * 1 + cobertura * 3
        score = freq_count * 2.0 + pot_total * 1.0 + cover_count * 3.0

        # Converter para horárs legíveis
        labels = []
        for p in sorted(combo):
            h = p * minutos / 60.0
            if h >= 24:
                labels.append(f"{p}p ({h/24:.0f}d)")
            else:
                labels.append(f"{p}p ({h:.0f}h)")

        resultados_comb.append({
            "periodos": combo,
            "labels": labels,
            "freq_count": freq_count,
            "pot_total": pot_total,
            "cover_count": cover_count,
            "score": score,
        })

    # Ordenar por score
    resultados_comb.sort(key=lambda x: -x["score"])

    # Imprimir tabela
    print()
    print("=" * 130)
    print(f"  RANQUEAMENTO DE COMBINAÇÕES MSTL — Top {top_n}")
    print("=" * 130)
    print()
    print(f"  {'Rank':<6} {'Combinação':<40} {'Freq':<8} {'Potência':<10} {'Cobertura':<10} {'Score':<8}")
    print(f"  {'─'*6} {'─'*40} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")

    for i, rc in enumerate(resultados_comb[:top_n], 1):
        combo_str = " + ".join(rc["labels"])
        print(f"  {i:<6} {combo_str:<40} "
              f"{rc['freq_count']}/7   {rc['pot_total']:<10.2f} "
              f"{rc['cover_count']}/7   {rc['score']:<8.1f}")

    print()

    # Comandos sugeridos
    print("  Comandos para rodar:")
    print()
    for i, rc in enumerate(resultados_comb[:5], 1):
        periodos_str = ",".join(str(p) for p in rc["periodos"])
        print(f"  # Opção {i} (score={rc['score']:.1f})")
        print(f"  python -m testes_letras.run_apples_to_apples_trio \\")
        print(f"    --texto-decomp mstl --mstl-periods \"{periodos_str}\" \\")
        print(f"    --epochs 10 --seeds 2021,42,1234")
        print()

    print("=" * 130)


def sugerir_periodos_inteligente(resultados: List[Dict]) -> None:
    """Sugere os 3 melhores períodos para MSTL, filtrando artefatos e duplicatas.

    Regras:
      1. Período < 336 passos (metade da janela de 672 do AutoTimes)
      2. Períodos distintos: diferença >= 15% entre si
      3. Ranqueia por potência total across features
    """
    LIMITE_JANELA_MSTL = 336
    minutos = resultados[0]["minutos_timestep"]

    print()
    print("=" * 100)
    print("  SUGESTÃO INTELIGENTE — Top 3 Períodos para MSTL")
    print(f"  (filtra: período >= {LIMITE_JANELA_MSTL} passos | duplicatas com diff < 15%)")
    print("=" * 100)
    print()

    # Coletar TODOS os picos de todas as features, ordenados por potência
    todos_picos = []
    for res in resultados:
        for pico in res["picos"]:
            todos_picos.append({
                "periodo": pico["periodo_passos"],
                "potencia": pico["potencia"],
                "feat": res["feat_name"],
                "horas_str": pico["horas_str"],
            })

    # Ordenar por potência (decrescente)
    todos_picos.sort(key=lambda x: -x["potencia"])

    # Filtrar e selecionar top 3 distintos
    periodos_selecionados = []
    for pico in todos_picos:
        per = pico["periodo"]

        # 1. Ignorar se for maior que a janela suportada pelo modelo
        if per >= LIMITE_JANELA_MSTL:
            continue

        # 2. Ignorar se for muito próximo de um período já escolhido
        is_distinct = True
        for selecionado in periodos_selecionados:
            if abs(per - selecionado) / selecionado < 0.15:
                is_distinct = False
                break

        if is_distinct:
            periodos_selecionados.append(per)

        if len(periodos_selecionados) == 3:
            break

    # Calcular potência total para cada período selecionado
    print(f"  {'Rank':<6} {'Período':<12} {'Horas':<14} {'Potência Total':<16} {'Features'}")
    print(f"  {'─'*6} {'─'*12} {'─'*14} {'─'*16} {'─'*30}")

    for rank, per in enumerate(periodos_selecionados, 1):
        # Somar potência normalizada across features
        pot_total = 0.0
        feats_com = []
        for res in resultados:
            for pico in res["picos"]:
                if pico["periodo_passos"] == per:
                    pot_max = max(p["potencia"] for p in res["picos"]) if res["picos"] else 1
                    pot_total += pico["potencia"] / pot_max
                    feats_com.append(res["feat_name"])
                    break

        horas = per * minutos / 60.0
        if horas >= 24:
            horas_str = f"{horas / 24:.1f} dias"
        else:
            horas_str = f"{horas:.1f} h"

        feats_str = ", ".join(feats_com)
        print(f"  {rank:<6} {per:<12} {horas_str:<14} {pot_total:<16.2f} {feats_str}")

    print()

    # Comando sugerido
    if periodos_selecionados:
        periodos_str = ",".join(str(p) for p in periodos_selecionados)
        print(f"  Comando MSTL sugerido:")
        print(f"    python -m testes_letras.run_apples_to_apples_trio \\")
        print(f"      --texto-decomp mstl --mstl-periods \"{periodos_str}\" \\")
        print(f"      --epochs 10 --seeds 2021,42,1234")
        print()

    print("=" * 100)
# --------------------------------------------------------------------- #
def analisar_todas_features(csv_path: str, n_peaks: int = 20) -> List[Dict]:
    """Analisa o periodograma para todas as 7 features do ETTh1."""
    resultados = []
    for feat_idx in range(7):
        resultado = detectar_periodograma(csv_path, n_peaks=n_peaks, feat_idx=feat_idx)
        resultados.append(resultado)
    return resultados


# --------------------------------------------------------------------- #
#  CLI                                                                   #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análise espectral (periodograma) para detecção de períodos MSTL"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Caminho do CSV (default: ETTh1.csv)")
    parser.add_argument("--n-peaks", dest="n_peaks", type=int, default=20,
                        help="Número de picos a detectar (default: 20)")
    parser.add_argument("--feat", type=int, default=0,
                        help="Índice da feature a analisar (default: 0=HUFL)")
    parser.add_argument("--todas-features", dest="todas_features",
                        action="store_true", default=False,
                        help="Analisa todas as 7 features do ETTh1")
    parser.add_argument("--max-periodo-frac", dest="max_periodo_frac",
                        type=float, default=0.5,
                        help="Fração máx do período vs dataset (default: 0.5)")
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Main                                                                  #
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = str(caminhos.DATA_DIR / caminhos.DATA_PATH)

    print(f"\n[periodograma] CSV: {csv_path}")
    print(f"[periodograma] Máx período: {args.max_periodo_frac*100:.0f}% do dataset "
          f"({int(N_TREINO * args.max_periodo_frac)} passos)")

    if args.todas_features:
        resultados = analisar_todas_features(csv_path, n_peaks=args.n_peaks)
        for res in resultados:
            imprimir_tabela(res)
        imprimir_tabela_consolidada(resultados)
        sugerir_periodos_inteligente(resultados)
        ranquear_combinacoes(resultados, top_n=10)

        # Gráfico da feature principal
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = RECENTE_DIR / f"periodograma_etth1_{ts}.png"
        plotar_periodograma(resultados[0], save_path)
        print(f"[ok] gráfico: {save_path}")
    else:
        resultado = detectar_periodograma(csv_path, n_peaks=args.n_peaks, feat_idx=args.feat)
        imprimir_tabela(resultado)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = RECENTE_DIR / f"periodograma_{resultado['feat_name']}_{ts}.png"
        plotar_periodograma(resultado, save_path)
        print(f"[ok] gráfico: {save_path}")


if __name__ == "__main__":
    main()
