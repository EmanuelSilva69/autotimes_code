import os
import urllib.request

# URL direta correta para o repositório original do dataset ETT
ETT_BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
BASE_DIR = "./dataset"

# Focando nos datasets ETT para rodar o experimento inicial
DATASETS = {
    "ETT-small": ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"]
}

def download_datasets():
    print("Iniciando o download da família ETT para o AutoTimes...\n")

    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    for folder, files in DATASETS.items():
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for file in files:
            file_url = f"{ETT_BASE_URL}{file}"
            file_path = os.path.join(folder_path, file)

            if os.path.exists(file_path):
                print(f"[*] Já existe: {file_path} (Pulando)")
                continue

            print(f"[-] Baixando {file} para {folder_path}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
                print(f"[+] Concluído: {file}")
            except Exception as e:
                print(f"[!] Erro ao baixar {file}. Erro: {e}")

    print("\n[OK] Downloads processados! A estrutura está pronta para o teste Zero-Shot.")

if __name__ == "__main__":
    download_datasets()