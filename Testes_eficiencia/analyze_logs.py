"""
analyze_logs.py - Visualizacao analitica dos resultados do Ablation Study.

Le log/benchmark_results.json e log/training_history.log e gera em plots/:

    1. comparativo_mse_mae.html/png - barras lado a lado (MSE e MAE) das 7 runs.
    2. convergencia_val_loss.html/png - curvas de convergencia da loss de validacao.
    3. melhorias_relativas.html/png - % de reducao do Test MSE vs Run 0 (baseline).

Usa plotly (interativo, .html) e exporta tambem .png estatico via kaleido.
Se plotly nao estiver instalado, faz fallback para matplotlib (.png) com os
mesmos graficos. Tratamento de excecao por grafico para que uma falha nao
derrube a geracao dos demais.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
PLOTS_DIR = BASE_DIR / "plots"

JSON_PATH = LOG_DIR / "benchmark_results.json"
HISTORY_PATH = LOG_DIR / "training_history.log"


def _latest_log(path_like: Path, pattern: str) -> Path:
    """Devolve o arquivo de log mais recente (mtime) deste padrao.

    Com o novo nome por sessoao (benchmark_results_<ts>.json /
    training_history_<ts>.log), escolhe o mais novo; senao mantem o
    caminho legado para retrocompatibilidade.
    """
    matches = sorted(
        (p for p in LOG_DIR.glob(pattern) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else path_like


JSON_PATH = _latest_log(JSON_PATH, "benchmark_results_*.json")
HISTORY_PATH = _latest_log(HISTORY_PATH, "training_history_*.log")

SHORT_NAMES: Dict[str, str] = {
    "Run 0_Padrao": "Padrao",
    "Run 1_RobustRevIN": "RobustRevIN",
    "Run 2_VolatilityLoss": "VolLoss",
    "Run 3_OverlappingEmbed": "Overlap",
    "Run 4_DiscrimOptimizer": "DiscOptim",
    "Run 5_VolatilityGate": "VolGate",
    "Run 6_Frankenstein": "Frankenstein",
}

# --------------------------------------------------------------------- #
# Carga e validacao de dados                                              #
# --------------------------------------------------------------------- #
def load_results(json_path: Path = JSON_PATH) -> Tuple[Dict, List[Dict]]:
    """Le benchmark_results.json e devolve (meta, runs_ordenadas).

    Args:
        json_path: Caminho do arquivo de resultados.

    Returns:
        (meta, runs): meta global e lista de dicts das runs (ordem de insercao).
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} nao encontrado. Execute run_benchmarks.py primeiro."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs: Dict = data.get("runs", {})
    if not runs:
        raise ValueError("benchmark_results.json sem entradas em 'runs'.")
    return data.get("meta", {}), list(runs.values())


def load_history_rows(log_path: Path = HISTORY_PATH) -> List[Dict]:
    """Le training_history.log e extrai as linhas tabulares por epoca.

    Formato esperado (gerado pelo run_benchmarks.py):
        Run 3_OverlappingEmbed |    1 |    0.123s | ...

    Args:
        log_path: Caminho do arquivo de historico.

    Returns:
        Lista de dicts {run, epoch, time_s, train_loss, val_loss, test_loss}.
    """
    if not log_path.exists():
        return []
    pattern = re.compile(
        r"^(Run\s+\d+_[^|]+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)s\s*\|\s*"
        r"([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*$"
    )
    rows: List[Dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                rows.append(
                    {
                        "run": m.group(1).strip(),
                        "epoch": int(m.group(2)),
                        "time_s": float(m.group(3)),
                        "train_loss": float(m.group(4)),
                        "val_loss": float(m.group(5)),
                        "test_loss": float(m.group(6)),
                    }
                )
    return rows


def short_name(run_name: str) -> str:
    """Nome curto para rotulos dos graficos (remove 'Run N_')."""
    return SHORT_NAMES.get(run_name, run_name.replace("Run ", "").replace("_", " "))
# ---------------------------------------------------------------------
# Backend de plot: plotly (padrao, .html) ou matplotlib (fallback .png)
# ---------------------------------------------------------------------
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_PLOTLY = False
    HAS_MPL = False

try:
    import kaleido  # noqa: F401

    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

PLOT_EXT = "html" if HAS_PLOTLY else "png"
C = {
    "10": "#636EFA", "20": "#00CC47", "30": "#AB63FA",
    "40": "#FF8A5A", "50": "#19D3F3", "60": "#FF6692", "70": "#B6E2A4",
}
PALETTE = ["#636EFA", "#00CC47", "#AB63FA", "#FF8A5A", "#19D3F3", "#FF6692", "#B6E2A4"]


def save_figure(fig, stem: str) -> Path:
    """Salva a figura em plots/ como .html (plotly) e, com kaleido instalado,
    exporta tambem o .png estatico com o mesmo visual."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{stem}.{PLOT_EXT}"
    if HAS_PLOTLY:
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
        print(f"  [OK] {out.name}")
        if HAS_KALEIDO:
            png = PLOTS_DIR / f"{stem}.png"
            fig.write_image(str(png), width=1200, height=800, scale=2)
            print(f"  [OK] {png.name}")
        else:
            print("  [aviso] kaleido ausente; sem export PNG estatico "
                  "(pip install kaleido)")
    elif HAS_MPL:
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  [OK] {out.name}")
    return out
# ---------------------------------------------------------------------
# Grafico 1: Barras comparativas (MSE e MAE) por run
# ---------------------------------------------------------------------
def plot_comparative_mse_mae(runs: List[Dict]) -> Path:
    """Barras lado a lado do Test MSE e Test MAE final de todas as runs.

    Args:
        runs: Lista ordenada de resultados (Run 0 .. Run 6).

    Returns:
        Path da figura gerada.
    """
    names = [short_name(r["name"]) for r in runs]
    mse = [r["final_test_mse"] for r in runs]
    mae = [r["final_test_mae"] for r in runs]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Test MSE", x=names, y=mse, marker_color=C["10"],
                             text=[f"{v:.4f}" for v in mse], textposition="outside"))
        fig.add_trace(go.Bar(name="Test MAE", x=names, y=mae, marker_color=C["20"],
                             text=[f"{v:.4f}" for v in mae], textposition="outside"))
        best = int(np.argmin(mse))
        fig.add_annotation(x=names[best], y=mse[best], text="melhor", showarrow=True,
                           arrowhead=2, ax=0, ay=-40, font=dict(color="green"))
        fig.update_layout(
            title="Benchmark - Test MSE e Test MAE por Run (menor = melhor)",
            xaxis_title="Run", yaxis_title="Erro",
            barmode="group", template="plotly_white", height=480,
            legend=dict(orientation="h", y=1.08),
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        w = 0.38
        ax.bar(x - w / 2, mse, w, label="Test MSE", color=C["10"])
        ax.bar(x + w / 2, mae, w, label="Test MAE", color=C["20"])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18)
        ax.set_ylabel("Erro")
        ax.set_title("Test MSE e Test MAE por Run")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        for xi, m in zip(x - w / 2, mse):
            ax.text(xi, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        for xi, m in zip(x + w / 2, mae):
            ax.text(xi, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
    return save_figure(fig, "comparativo_mse_mae")


# ---------------------------------------------------------------------
# Grafico 2: Curvas de convergencia (val loss) por epoca
# ---------------------------------------------------------------------
def _trend_slope(ys: List[float]) -> float:
    """Inclinacao da regressao linear sobre as ultimas 5 epocas."""
    window = ys[-5:]
    if len(window) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(window)), window, 1)[0])


def plot_convergence(runs: List[Dict]) -> Path:
    """Sobrepoe a val loss por epoca de todas as runs, destacando 3 e 4.

    Tambem avalia se Run 3 (Overlap) ou Run 4 (DiscOptim) previnem
    estagnacao prematura comparando a inclinacao das ultimas epocas.

    Args:
        runs: Resultados (Run 0 primeiro).

    Returns:
        Path da figura.
    """
    if HAS_PLOTLY:
        fig = go.Figure()
        for i, r in enumerate(runs):
            epochs = list(range(1, len(r["val_loss"]) + 1))
            widening = r["name"] in ("Run 3_OverlappingEmbed", "Run 4_DiscrimOptimizer")
            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(
                x=epochs, y=r["val_loss"],
                name=short_name(r["name"]),
                mode="lines+markers",
                line=dict(width=3 if widening else 1.8,
                          dash="dash" if widening else "solid",
                          color=color),
            ))
        fig.update_layout(
            title="Convergencia da Loss de Validacao por Epoca",
            xaxis_title="Epoca", yaxis_title="Val Loss",
            template="plotly_white", height=520,
            legend=dict(font=dict(size=10)),
        )
        # Anotacao de estagnacao para Run 3 e Run 4.
        for target in ("Run 3_OverlappingEmbed", "Run 4_DiscrimOptimizer"):
            r = next((x for x in runs if x["name"] == target), None)
            if r and len(r["val_loss"]) >= 5:
                slope = _trend_slope(r["val_loss"])
                msg = "continua a descer" if slope < 0 else "estagnou (plateau)"
                fig.add_annotation(
                    x=len(r["val_loss"]), y=r["val_loss"][-1],
                    text=f"{short_name(target)}: {msg}",
                    showarrow=True, arrowhead=2, ax=-70, ay=-40,
                    font=dict(size=11, color="black"),
                )
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, r in enumerate(runs):
            epochs = list(range(1, len(r["val_loss"]) + 1))
            widening = r["name"] in ("Run 3_OverlappingEmbed", "Run 4_DiscrimOptimizer")
            ls = "--" if widening else "-"
            lw = 2.6 if widening else 1.8
            ax.plot(epochs, r["val_loss"], label=short_name(r["name"]), ls=ls, lw=lw)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Val Loss")
        ax.set_title("Convergencia da Loss de Validacao por Run")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
    return save_figure(fig, "convergencia_val_loss")
# ---------------------------------------------------------------------
# Grafico 3: Melhoria relativa (% de reducao do Test MSE vs baseline)
# ---------------------------------------------------------------------
def plot_relative_improvement(runs: List[Dict]) -> Path:
    """Barras da variacao percentual do Test MSE de cada run vs Run 0.

    Valores negativos = melhoria (reducao do erro); positivos = regressao.

    Args:
        runs: Resultados ordenados (Run 0 primeiro).

    Returns:
        Path da figura.
    """
    base_mse = runs[0]["final_test_mse"]
    names = [short_name(r["name"]) for r in runs[1:]]
    deltas = [(r["final_test_mse"] - base_mse) / base_mse * 100.0 for r in runs[1:]]
    colors = ["#2CA02C" if d <= 0 else "#D62728" for d in deltas]

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names, y=deltas, marker_color=colors,
            text=[f"{d:+.2f}%" for d in deltas], textposition="outside",
            hovertemplate="%{x}<br>delta MSE: %{y:.2f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="baseline (Run 0)")
        fig.update_layout(
            title=f"Melhoria Relativa do Test MSE vs Baseline (Run 0 = {base_mse:.4f})",
            xaxis_title="Run", yaxis_title="Delta % (negativo = melhoria)",
            template="plotly_white", height=480,
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(names, deltas, color=colors)
        for xi, d in zip(names, deltas):
            ax.text(xi, d, f"{d:+.2f}%", ha="center", va="bottom" if d >= 0 else "top")
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_ylabel("Delta % (negativo = melhoria)")
        ax.set_title("Melhoria Relativa do Test MSE vs Baseline")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
    return save_figure(fig, "melhorias_relativas")


# ---------------------------------------------------------------------
# Analise textual e main
# ---------------------------------------------------------------------
def print_analysis(runs: List[Dict]) -> None:
    """Imprime tabela comparativa + diagnostico de estagnacao."""
    base_mse = runs[0]["final_test_mse"]
    base_mae = runs[0]["final_test_mae"]
    print("\n" + "=" * 88)
    print("ANALISE - Ablation Study AutoTimes (Test MSE / Test MAE)")
    print("=" * 88)
    print(f"{'Run':<30s} {'MSE':>11s} {'MAE':>11s} {'dMSE%':>9s} {'Tendencia final':>20s}")
    print("-" * 88)
    for r in runs:
        d = (r["final_test_mse"] - base_mse) / base_mse * 100.0
        slope = _trend_slope(r["val_loss"])
        trend = "em queda" if slope < 0 else "plateau"
        print(f"{short_name(r['name']):<30s} {r['final_test_mse']:>11.6f} "
              f"{r['final_test_mae']:>11.6f} {d:>9.2f}% {trend:>20s}")
    print("-" * 88)

    improvements = [r for r in runs[1:] if r["final_test_mse"] < base_mse]
    print(f"\nMelhor isolado: {short_name(min(runs[1:], key=lambda r: r['final_test_mse'])['name'])} "
          f"| Melhor geral: {short_name(min(runs, key=lambda r: r['final_test_mse'])['name'])}")
    print(f"Runs melhores que o baseline: {len(improvements)}/{len(runs) - 1}")

    for target in ("Run 3_OverlappingEmbed", "Run 4_DiscrimOptimizer"):
        r = next((x for x in runs if x["name"] == target), None)
        if r and len(r["val_loss"]) >= 5:
            slope = _trend_slope(r["val_loss"])
            print(f"[diagnostico] {target}: inclinacao das ultimas 5 epocas = "
                  f"{slope:+.2e} -> {'previne estagnacao' if slope < 0 else 'estagnado'}")


def main() -> None:
    try:
        meta, runs = load_results()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERRO] {exc}")
        return

    print(f"Resultados carregados: {len(runs)} runs | meta: {meta}")

    steps = [
        ("Graf 1 - Barras MSE/MAE", lambda: plot_comparative_mse_mae(runs)),
        ("Graf 2 - Convergencia val loss", lambda: plot_convergence(runs)),
        ("Graf 3 - Melhoria relativa", lambda: plot_relative_improvement(runs)),
    ]
    for label, fn in steps:
        print(f"\n[{label}]")
        try:
            fn()
        except Exception as exc:  # falha de um grafico nao derruba os demais
            print(f"  [ERRO no {label}]: {exc!r}")

    print_analysis(runs)
    print(f"\nGraficos salvos em: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
