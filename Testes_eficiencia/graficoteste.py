import json
import matplotlib.pyplot as plt
import numpy as np

def plot_tradeoff_from_json(json_file_path):
    # Configuração de estilo profissional/acadêmico
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)

    # Carrega os dados do arquivo JSON fornecido
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Identifica o tipo de ablation (numérico ou textual) pelo meta do JSON
    meta = data.get("meta", {})
    textual = bool(meta.get("textual"))
    modo = "Ablação Textual" if textual else "Ablação Numérica"
    sufixo = "textual" if textual else "numerica"
    ts = str(meta.get("timestamp", "")).replace("-", "").replace(":", "").strip()
    if ts:
        sufixo = f"{sufixo}_{ts}"

    runs = data.get("runs", {})
    
    names = []
    times = []
    mses = []

    # Extrai as informações de cada run estruturadamente
    for run_key, run_info in runs.items():
        name = run_info.get("name", run_key)
        total_time = run_info.get("total_time_s", 0.0)
        final_mse = run_info.get("final_test_mse", 0.0)
        
        if total_time > 0 and final_mse > 0:
            names.append(name)
            times.append(total_time)
            mses.append(final_mse)

    # Paleta de cores distinta para cada ponto
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # Plota os pontos de dispersão (Scatter Plot) sem os textos nas bolinhas
    for i, name in enumerate(names):
        clean_name = name.replace("_", " ")
        ax.scatter(times[i], mses[i], color=colors[i], s=140, alpha=0.9, edgecolor='k', linewidth=1.2, zorder=3, label=clean_name)

    ax.set_title(f"Trade-off: Custo Computacional (Tempo Total) vs. Erro de Predição (Test MSE) - {modo}", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Tempo Total de Treinamento (segundos)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Test MSE (menor = melhor)", fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)

    # Margem limpa nas laterais já que não há textos flutuando perto dos pontos
    ax.margins(x=0.15, y=0.15)

    # Legenda colocada estritamente do lado de fora à direita
    ax.legend(
        title="Configurações (Runs)", 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        fontsize=9, 
        title_fontsize=10,
        framealpha=0.95,
        edgecolor='#bdc3c7'
    )

    # bbox_inches='tight' garante que a legenda externa não seja cortada ao salvar
    plt.tight_layout()
    plt.savefig(f"tradeoff_{sufixo}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gráfico de Trade-off ({modo}) gerado e salvo como 'tradeoff_{sufixo}.png' com sucesso!")

if __name__ == "__main__":
    # Salve o JSON fornecido em um arquivo local (ex: benchmark_results.json) e aponte o caminho aqui
    json_path = "E:\\2026.1\\Viplab\\AutoTimes\\Testes_eficiencia\\log\\benchmark_results_20260806_211841.json"
    plot_tradeoff_from_json(json_path)