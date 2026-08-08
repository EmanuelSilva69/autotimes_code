import re
import matplotlib.pyplot as plt
import os
import glob

# 1. Encontra automaticamente o arquivo de log MAIS RECENTE na pasta 'logs'
list_of_files = glob.glob(r'.\logs\Log-*.txt')

if not list_of_files:
    print("ERRO: Nenhum arquivo de log foi encontrado na pasta '.\\logs\\'.")
    exit()

# Pega o arquivo que foi modificado/criado por último
log_file = max(list_of_files, key=os.path.getctime)
print(f"Lendo automaticamente o arquivo: {log_file}")

# Estrutura de dados expandida para acomodar iterações e learning rate
datasets_data = {}
current_dataset = None

# Expressões regulares atualizadas
regex_dataset = re.compile(r"Dataset:\s*(.+)")
regex_epoch_loss = re.compile(r"Epoch:\s*\d+,\s*Steps:\s*\d+\s*\|\s*Train Loss:\s*([\d.]+)\s*Vali Loss:\s*([\d.]+)\s*Test Loss:\s*([\d.]+)")
regex_iter_loss = re.compile(r"iters:\s*(\d+),\s*epoch:\s*(\d+)\s*\|\s*loss:\s*([\d.]+)")
regex_lr = re.compile(r"lr\s*=\s*([0-e.-]+)")

# Variáveis auxiliares para o eixo X do Learning Rate e Iterações
global_iter_counter = 0 
lr_step_counter = 0

# 2. Tenta abrir o arquivo com UTF-16 (Padrão do PowerShell) ou UTF-8
linhas = []
try:
    with open(log_file, 'r', encoding='utf-16', errors='ignore') as f:
        linhas = f.readlines()
        if not any("Epoch:" in linha or "iters:" in linha for linha in linhas):
            raise ValueError("Falha na leitura UTF-16")
except:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        linhas = f.readlines()

if not linhas:
    print("O arquivo está completamente vazio!")
    exit()

print("Varrendo o texto em busca das métricas granulares...")

# 3. Leitura linha por linha
for line in linhas:
    match_ds = regex_dataset.search(line)
    if match_ds:
        current_dataset = match_ds.group(1).strip()
        if current_dataset not in datasets_data:
            datasets_data[current_dataset] = {
                'epoch_train': [], 'epoch_vali': [], 'epoch_test': [],
                'iter_step': [], 'iter_loss': [],
                'lr_step': [], 'lr_value': []
            }
            global_iter_counter = 0
            lr_step_counter = 0
        continue

    if current_dataset:
        # Captura as perdas finais de cada Época
        match_epoch_loss = regex_epoch_loss.search(line)
        if match_epoch_loss:
            datasets_data[current_dataset]['epoch_train'].append(float(match_epoch_loss.group(1)))
            datasets_data[current_dataset]['epoch_vali'].append(float(match_epoch_loss.group(2)))
            datasets_data[current_dataset]['epoch_test'].append(float(match_epoch_loss.group(3)))
            continue

        # Captura a perda por Iteração
        match_iter_loss = regex_iter_loss.search(line)
        if match_iter_loss:
            global_iter_counter += 100 # Como o log printa a cada 100 iterações
            loss_val = float(match_iter_loss.group(3))
            datasets_data[current_dataset]['iter_step'].append(global_iter_counter)
            datasets_data[current_dataset]['iter_loss'].append(loss_val)
            continue
            
        # Captura as atualizações de Learning Rate
        match_lr = regex_lr.search(line)
        if match_lr:
            lr_step_counter += 1
            lr_val = float(match_lr.group(1))
            datasets_data[current_dataset]['lr_step'].append(lr_step_counter)
            datasets_data[current_dataset]['lr_value'].append(lr_val)

# 4. Geração dos gráficos
output_dir = "graficos_treinamento"
os.makedirs(output_dir, exist_ok=True)

graficos_gerados = 0

for ds_name, data in datasets_data.items():
    if not data['epoch_train']:
        continue  # Se não houver dados de época, pula para o próximo

    clean_filename = ds_name.split()[0].replace('/', '_').replace('\\', '_')
    
    # ---------------------------------------------------------
    # Gráfico 1: Perda por Época (Original)
    # ---------------------------------------------------------
    epochs = range(1, len(data['epoch_train']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['epoch_train'], marker='o', linewidth=2, label='Train Loss', color='#1f77b4')
    plt.plot(epochs, data['epoch_vali'], marker='s', linewidth=2, label='Validation Loss', color='#ff7f0e')
    plt.plot(epochs, data['epoch_test'], marker='^', linewidth=2, label='Test Loss', color='#2ca02c')
    plt.title(f'Curva de Aprendizado (Épocas) - {ds_name}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Épocas (Epochs)', fontsize=14)
    plt.ylabel('Erro (Loss)', fontsize=14)
    plt.xticks(epochs)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_epoch = os.path.join(output_dir, f"Loss_Epoch_{clean_filename}.png")
    plt.savefig(save_path_epoch, dpi=300, bbox_inches='tight')
    plt.close()
    graficos_gerados += 1

    # ---------------------------------------------------------
    # Gráfico 2: Perda Granular por Iteração
    # ---------------------------------------------------------
    if data['iter_loss']:
        plt.figure(figsize=(12, 5))
        # Linha fina para mostrar a alta frequência de variação do gradiente
        plt.plot(data['iter_step'], data['iter_loss'], linewidth=1, color='#d62728', alpha=0.8)
        plt.title(f'Variação do Erro por Iteração - {ds_name}', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Iterações Globais (Steps)', fontsize=14)
        plt.ylabel('Train Loss', fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        
        save_path_iter = os.path.join(output_dir, f"Loss_Iter_{clean_filename}.png")
        plt.savefig(save_path_iter, dpi=300, bbox_inches='tight')
        plt.close()
        graficos_gerados += 1

    # ---------------------------------------------------------
    # Gráfico 3: Curva de Learning Rate
    # ---------------------------------------------------------
    if data['lr_value']:
        plt.figure(figsize=(10, 5))
        # Gráfico em degraus/curva mostrando o scheduler Cosine
        plt.plot(data['lr_step'], data['lr_value'], marker='x', markersize=6, linewidth=2, color='#9467bd')
        plt.title(f'Decaimento do Learning Rate - {ds_name}', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Atualizações (Epochs/Steps)', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Força notação científica no eixo Y para LRs pequenos
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path_lr = os.path.join(output_dir, f"LearningRate_{clean_filename}.png")
        plt.savefig(save_path_lr, dpi=300, bbox_inches='tight')
        plt.close()
        graficos_gerados += 1
        
    print(f"[OK] Pacote de gráficos gerado para: {ds_name}")

if graficos_gerados == 0:
    print("\nAVISO: O Python leu o arquivo, mas não encontrou nenhuma métrica para desenhar.")
else:
    print(f"\nFeito! {graficos_gerados} gráficos exportados para a pasta '{output_dir}'.")