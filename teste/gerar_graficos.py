import os
import re
import csv
import glob
import warnings
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "benchmark_results.csv"
LOGS_DIR = BASE_DIR / "logs"
OUT_DIR = BASE_DIR / "graficos"

SUB_DIRS = ["comparativos", "iteracoes", "individuais", "geral"]
COLORS = px.colors.qualitative.Plotly
METRIC_NAMES = {"test_mse": "MSE", "test_mae": "MAE", "test_rmse": "RMSE"}


def ensure_dirs():
    for d in SUB_DIRS:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)


def load_csv():
    if not CSV_PATH.exists():
        print(f"[AVISO] {CSV_PATH} nao encontrado.")
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    for col in ["test_mse", "test_mae", "test_rmse", "total_time_s", "peak_vram_gb", "train_loss", "val_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_log_losses(log_path):
    losses = []
    pat = re.compile(
        r"Epoch:\s*(\d+).*?Train Loss:\s*([0-9e.+-]+)\s+Vali Loss:\s*([0-9e.+-]+)\s+Test Loss:\s*([0-9e.+-]+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            m = pat.search(line)
            if m:
                losses.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss": float(m.group(3)),
                    "test_loss": float(m.group(4)),
                })
    return losses


def load_all_log_losses():
    all_data = []
    for fpath in sorted(LOGS_DIR.glob("opt*_output.txt")):
        opt_id = fpath.stem.replace("_output", "")
        losses = parse_log_losses(fpath)
        for entry in losses:
            entry["opt_id"] = opt_id
            all_data.append(entry)
    return pd.DataFrame(all_data)


def save_html(fig, name, subdir):
    path = OUT_DIR / subdir / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=False)
    print(f"  [OK] {path.name}")


def save_div(fig, name, subdir):
    path = OUT_DIR / subdir / f"{name}.html"
    fig.write_html(str(path))
    print(f"  [OK] {path.name}")


def plot_comparativos(df):
    if df.empty:
        return
    df = df.sort_values("opt_id")
    labels = df["opt_id"].tolist()
    names = df["opt_name"].tolist()
    hover = [f"{o}: {n}" for o, n in zip(labels, names)]

    for col, title in [("test_mse", "MSE"), ("test_mae", "MAE"), ("test_rmse", "RMSE")]:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=df[col], text=df[col].round(4),
                             textposition="outside", marker_color=COLORS[:len(df)],
                             hovertemplate="<b>%{text}</b><extra></extra>"))
        best_idx = df[col].idxmin()
        fig.add_annotation(x=labels[best_idx], y=df[col].iloc[best_idx],
                           text="&#9733;", showarrow=False, yshift=10,
                           font=dict(size=20, color="gold"))
        fig.update_layout(title=f"{title} por Metodo", yaxis_title=title,
                          xaxis_title="Metodo", template="plotly_white",
                          height=450, margin=dict(b=80))
        save_html(fig, f"comparativo_{col}", "comparativos")

    for col, title, unit in [("total_time_s", "Tempo Total de Execucao", "s"),
                               ("peak_vram_gb", "Pico de VRAM", "GB")]:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=df[col], text=df[col].round(2),
                             textposition="outside", marker_color=COLORS[:len(df)],
                             marker=dict(line=dict(width=0.5, color="gray")),
                             hovertemplate="<b>%{text} " + unit + "</b><extra></extra>"))
        fig.update_layout(title=f"{title} por Metodo", yaxis_title=unit,
                          xaxis_title="Metodo", template="plotly_white",
                          height=450, margin=dict(b=80))
        save_html(fig, f"comparativo_{col}", "comparativos")

    fig = make_subplots(rows=2, cols=2, subplot_titles=("MSE", "MAE", "RMSE", "Tempo (s)"),
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    metrics = [("test_mse", "MSE"), ("test_mae", "MAE"), ("test_rmse", "RMSE"), ("total_time_s", "Tempo (s)")]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (col, _), (r, c) in zip(metrics, positions):
        fig.add_trace(go.Bar(x=labels, y=df[col], showlegend=False,
                             marker_color=COLORS[:len(df)]), row=r, col=c)
    fig.update_layout(title="Painel Comparativo", template="plotly_white",
                      height=600, margin=dict(b=40))
    save_html(fig, "painel_comparativo", "comparativos")


def plot_iteracoes(df_log):
    if df_log.empty:
        print("  [SKIP] Nenhum dado de loss por epoca encontrado nos logs.")
        return
    for loss_col, title in [("train_loss", "Train Loss"), ("val_loss", "Val Loss"),
                             ("test_loss", "Test Loss")]:
        fig = go.Figure()
        for opt_id in sorted(df_log["opt_id"].unique()):
            sub = df_log[df_log["opt_id"] == opt_id].sort_values("epoch")
            fig.add_trace(go.Scatter(x=sub["epoch"], y=sub[loss_col], mode="lines+markers",
                                     name=opt_id, line=dict(width=2)))
        fig.update_layout(title=f"Evolucao do {title} por Epoca",
                          xaxis_title="Epoca", yaxis_title=title,
                          template="plotly_white", height=500,
                          legend=dict(font=dict(size=9)))
        save_html(fig, f"evolucao_{loss_col}", "iteracoes")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Train Loss", "Val Loss", "Test Loss"),
                        vertical_spacing=0.06)
    for i, loss_col in enumerate(["train_loss", "val_loss", "test_loss"], 1):
        for opt_id in sorted(df_log["opt_id"].unique()):
            sub = df_log[df_log["opt_id"] == opt_id].sort_values("epoch")
            fig.add_trace(go.Scatter(x=sub["epoch"], y=sub[loss_col],
                                     mode="lines+markers", name=opt_id,
                                     showlegend=(i == 1), line=dict(width=1.5)),
                          row=i, col=1)
    fig.update_layout(title="Convergencia - Todas as Losses", template="plotly_white",
                      height=800, legend=dict(font=dict(size=8)))
    save_html(fig, "convergencia_completa", "iteracoes")

    fig = go.Figure()
    for opt_id in sorted(df_log["opt_id"].unique()):
        sub = df_log[df_log["opt_id"] == opt_id].sort_values("epoch")
        fig.add_trace(go.Scatter(x=sub["epoch"], y=sub["train_loss"] - sub["val_loss"],
                                 mode="lines+markers", name=opt_id, line=dict(width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Diferenca Train - Val Loss (sobreajuste)",
                      xaxis_title="Epoca", yaxis_title="Train - Val",
                      template="plotly_white", height=450)
    save_html(fig, "gap_train_val", "iteracoes")


def plot_individuais(df, df_log):
    if df.empty:
        return
    for _, row in df.iterrows():
        opt_id = row["opt_id"]
        opt_name = row.get("opt_name", opt_id)

        metrics_dict = {
            "Metrica": ["MSE", "MAE", "RMSE", "Tempo (s)", "VRAM (GB)"],
            "Valor": [row.get("test_mse", 0), row.get("test_mae", 0),
                      row.get("test_rmse", 0), row.get("total_time_s", 0),
                      row.get("peak_vram_gb", 0)]
        }
        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics_dict["Metrica"], y=metrics_dict["Valor"],
                             text=[f"{v:.4f}" for v in metrics_dict["Valor"]],
                             textposition="outside",
                             marker_color=px.colors.qualitative.Set2[:5]))
        fig.update_layout(title=f"{opt_id} - {opt_name}", template="plotly_white",
                          height=400, yaxis_title="Valor")
        save_html(fig, f"individual_{opt_id}", "individuais")

        sub_log = df_log[df_log["opt_id"] == opt_id].sort_values("epoch") if not df_log.empty else pd.DataFrame()
        if not sub_log.empty:
            fig = go.Figure()
            for col, label, color in [("train_loss", "Train", "blue"),
                                       ("val_loss", "Val", "orange"),
                                       ("test_loss", "Test", "green")]:
                fig.add_trace(go.Scatter(x=sub_log["epoch"], y=sub_log[col],
                                         mode="lines+markers", name=label,
                                         line=dict(color=color, width=2)))
            fig.update_layout(title=f"{opt_id} - Curvas de Loss", template="plotly_white",
                              xaxis_title="Epoca", yaxis_title="Loss", height=400)
            save_html(fig, f"individual_loss_{opt_id}", "individuais")


def plot_geral(df, df_log):
    if df.empty:
        return

    fig = px.scatter_matrix(df, dimensions=["test_mse", "test_mae", "test_rmse",
                                             "total_time_s", "peak_vram_gb"],
                            color="opt_id", hover_name="opt_name",
                            title="Matriz de Correlacao entre Metricas",
                            labels={c: METRIC_NAMES.get(c, c) for c in
                                    ["test_mse", "test_mae", "test_rmse"]})
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="gray")))
    fig.update_layout(height=800, template="plotly_white")
    save_html(fig, "matriz_correlacao", "geral")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["total_time_s"], y=df["test_mse"],
                             mode="markers+text", text=df["opt_id"],
                             textposition="top center",
                             marker=dict(size=12, color=COLORS[:len(df)],
                                         line=dict(width=1, color="black")),
                             hovertemplate="<b>%{text}</b><br>Tempo: %{x:.1f}s<br>MSE: %{y:.4f}<extra></extra>"))
    fig.update_layout(title="Trade-off: Tempo vs Acuracia (MSE)",
                      xaxis_title="Tempo Total (s)", yaxis_title="MSE",
                      template="plotly_white", height=500)
    save_html(fig, "tradeoff_tempo_mse", "geral")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["peak_vram_gb"], y=df["test_mse"],
                             mode="markers+text", text=df["opt_id"],
                             textposition="top center",
                             marker=dict(size=12, color=COLORS[:len(df)],
                                         line=dict(width=1, color="black")),
                             hovertemplate="<b>%{text}</b><br>VRAM: %{x:.2f}GB<br>MSE: %{y:.4f}<extra></extra>"))
    fig.update_layout(title="Trade-off: VRAM vs Acuracia (MSE)",
                      xaxis_title="Pico de VRAM (GB)", yaxis_title="MSE",
                      template="plotly_white", height=500)
    save_html(fig, "tradeoff_vram_mse", "geral")

    fig = go.Figure()
    metrics_plot = ["test_mse", "test_mae", "test_rmse"]
    for i, (col, name) in enumerate(zip(metrics_plot, ["MSE", "MAE", "RMSE"])):
        normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        fig.add_trace(go.Bar(name=name, x=df["opt_id"], y=normalized,
                             marker_color=px.colors.qualitative.Set1[i],
                             offsetgroup=i))
    fig.update_layout(title="Metricas Normalizadas por Metodo (0=melhor, 1=pior)",
                      barmode="group", template="plotly_white",
                      height=450, yaxis_title="Valor Normalizado")
    save_html(fig, "metricas_normalizadas", "geral")

    if not df_log.empty:
        pivot = df_log.groupby("opt_id")[["train_loss", "val_loss", "test_loss"]].last().reset_index()
        fig = go.Figure()
        for col, name, color in [("train_loss", "Train Final", "blue"),
                                  ("val_loss", "Val Final", "orange"),
                                  ("test_loss", "Test Final", "green")]:
            fig.add_trace(go.Bar(name=name, x=pivot["opt_id"], y=pivot[col],
                                 marker_color=color))
        fig.update_layout(title="Loss Final por Metodo (ultima epoca)",
                          barmode="group", template="plotly_white", height=450)
        save_html(fig, "loss_final_por_metodo", "geral")


def main():
    ensure_dirs()
    print("Carregando dados...")
    df = load_csv()
    df_log = load_all_log_losses()
    print(f"  CSV: {len(df)} linhas")
    print(f"  Logs com losses por epoca: {len(df_log)} entradas")

    print("\n[1/4] Graficos Comparativos...")
    plot_comparativos(df)

    print("\n[2/4] Graficos de Iteracoes (convergencia)...")
    plot_iteracoes(df_log)

    print("\n[3/4] Graficos Individuais...")
    plot_individuais(df, df_log)

    print("\n[4/4] Graficos Gerais...")
    plot_geral(df, df_log)

    print(f"\nTodos os graficos foram gerados em: {OUT_DIR}/")


if __name__ == "__main__":
    main()
