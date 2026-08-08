$env:PYTHONUNBUFFERED=1
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
$env:HSA_OVERRIDE_GFX_VERSION="12.0.0"

# ==============================================================================
# SISTEMA DE LOGS E AMBIENTE
# ==============================================================================
$LogDir = ".\logs"
if (-not (Test-Path -Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$Timestamp = Get-Date -Format "dd-MM-yyyy_HH-mm-ss"
$LogFile = "$LogDir\ComparativoModelos-$Timestamp.txt"
$ErroFile = "$LogDir\ComparativoErros-$Timestamp.txt"
$StartTime = Get-Date

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "AutoTimes - Bateria de Comparacao de Modelos (Completa)" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan

# ==============================================================================
# MENU 1: ESCOLHA DO DATASET
# ==============================================================================
Write-Host "`nSelecione o Dataset para ancorar a comparacao:" -ForegroundColor Yellow
Write-Host "[1] ETTh1 | [2] Electricity | [3] Weather | [4] Traffic"
$ds_opcao = Read-Host "Digite a opcao (1-4) [Padrao: 1]"

switch ($ds_opcao) {
    '2' { $TargetDataset = @{ Name="Electricity"; Root="./dataset/electricity/"; Data="electricity.csv"; DataType="custom"; ModelId="ECL_672_96" } }
    '3' { $TargetDataset = @{ Name="Weather"; Root="./dataset/weather/"; Data="weather.csv"; DataType="custom"; ModelId="weather_672_96" } }
    '4' { $TargetDataset = @{ Name="Traffic"; Root="./dataset/traffic/"; Data="traffic.csv"; DataType="custom"; ModelId="traffic_672_96" } }
    Default { $TargetDataset = @{ Name="ETTh1"; Root="./dataset/ETT-small/"; Data="ETTh1.csv"; DataType="ETTh1"; ModelId="ETTh1_672_96" } }
}

# ==============================================================================
# MENU 2: SELECAO DE MODELOS
# ==============================================================================
$TodosModelos = @(
    @{ Id=1;  Name="GPT-2 Original (124M)";  Dir="gpt2";         ModelClass="AutoTimes_Gpt2" }
    @{ Id=2;  Name="OPT Original (350M)";    Dir="opt";     ModelClass="AutoTimes_Opt_1b" }
    @{ Id=3;  Name="LLaMA (7B)";             Dir="llama";        ModelClass="AutoTimes_Llama" }
    @{ Id=4;  Name="Qwen 2.5 (0.5B)";        Dir="qwen-0.5b";    ModelClass="AutoTimes_Qwen" } 
    @{ Id=5;  Name="Qwen 2.5 (1.5B)";        Dir="qwen-1.5b";    ModelClass="AutoTimes_Qwen" }
    @{ Id=6;  Name="Qwen-Mini (200M)";       Dir="qwen-200m";    ModelClass="AutoTimes_Qwen" } 
    @{ Id=7;  Name="Qwen-Mini (500M)";       Dir="qwen-500m";    ModelClass="AutoTimes_Qwen" } 
    @{ Id=8;  Name="Qwen-Mini (1.2B)";       Dir="qwen-1.2b";    ModelClass="AutoTimes_Qwen" } 
    @{ Id=9;  Name="LLaMA 3.1-Mini (212M)";  Dir="llama-212m";   ModelClass="AutoTimes_Qwen" }
    @{ Id=10; Name="Pythia (410M)";          Dir="pythia-410m";  ModelClass="AutoTimes_Gpt2" }
    @{ Id=11; Name="SmolLM2 (360M)";         Dir="smollm-360m";  ModelClass="AutoTimes_Llama" }
)

Write-Host "`n--- Selecao de Modelos ---" -ForegroundColor Cyan
foreach ($mod in $TodosModelos) { Write-Host "[$($mod.Id)] $($mod.Name)" }
$mod_opcao = Read-Host "Escolha os modelos (Ex: 1,2,7,9,11) [Padrao: 1,2]"

$ModelosSelecionados = @()
if ([string]::IsNullOrWhiteSpace($mod_opcao)) { $mod_opcao = "1,2" }
foreach ($op in ($mod_opcao -split ',')) {
    $m = $TodosModelos | Where-Object { $_.Id -eq $op.Trim() }
    if ($m) { $ModelosSelecionados += $m }
}

# ==============================================================================
# MENU 3: CONFIGURACOES GERAIS
# ==============================================================================
Write-Host "`n--- Configuracoes Gerais ---" -ForegroundColor Cyan
$BatchSize = Read-Host "Batch Size [Padrao: 256]"
if ([string]::IsNullOrWhiteSpace($BatchSize)) { $BatchSize = "256" }
$LrValue = Read-Host "Learning Rate [Padrao: 0.001]"
if ([string]::IsNullOrWhiteSpace($LrValue)) { $LrValue = "0.001" }
$TrainEpochs = Read-Host "Epocas Maximas [Padrao: 10]"
if ([string]::IsNullOrWhiteSpace($TrainEpochs)) { $TrainEpochs = "10" }

# ==============================================================================
# MENU 4: CONFIGURACOES AVANCADAS (Ablacao)
# ==============================================================================
Write-Host "`n--- Configuracoes Avancadas ---" -ForegroundColor Cyan
Write-Host "MLP Hidden Dim: [1] 256 | [2] 512 | [3] 768 | [4] 1024 | [5] Manual"
$m_in = Read-Host "Escolha o Mlp Hidden Dim"
$MlpDim = switch($m_in) { '2' {512}; '3' {768}; '4' {1024}; '5' {Read-Host "Digite o valor manual"}; Default {256} }

# ==============================================================================
# EXECUCAO DA BATERIA
# ==============================================================================
"Bateria de Comparacao no Dataset $($TargetDataset.Name)" | Out-File $LogFile
"Inicio: $StartTime`n" | Out-File $LogFile -Append

foreach ($modelo in $ModelosSelecionados) {
    Write-Host "`n[TREINO] Iniciando: $($modelo.Name) | BS: $BatchSize | MLP: $MlpDim" -ForegroundColor Green
    
    # Lógica de Override Inteligente para o OPT
    $CurrentMlpDim = $MlpDim
    if ($modelo.Name -like "*OPT*") {
        $confirm = Read-Host "Modelo OPT detetado. Manter MLP Dim $MlpDim? (S/N) [Padrao: S]"
        if ($confirm -eq 'N') { $CurrentMlpDim = Read-Host "Digite o valor manual para OPT" }
    }

    $ArgList = @(
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model", $modelo.ModelClass,
        "--seq_len", "672", "--label_len", "576", "--token_len", "96",
        "--test_seq_len", "672", "--test_label_len", "576", "--test_pred_len", "96",
        "--batch_size", "$BatchSize", "--train_epochs", "$TrainEpochs",
        "--num_workers", "4", "--use_amp", "--llm_ckp_dir", $modelo.Dir,
        "--cosine", "--tmax", "10", "--des", "Exp", "--gpu", "0", "--drop_last",
        "--root_path", $TargetDataset.Root, "--data_path", $TargetDataset.Data,
        "--model_id", $TargetDataset.ModelId, "--data", $TargetDataset.DataType,
        "--learning_rate", "$LrValue", "--mlp_hidden_dim", "$CurrentMlpDim"
    )

    try {
        & python -u run.py @ArgList 2>&1 | ForEach-Object {
            $_ | Out-File $LogFile -Append
        }
    } catch {
        $errMsg = "ERRO em $($modelo.Name): $_"
        Write-Host $errMsg -ForegroundColor Red
        $errMsg | Out-File $ErroFile -Append
    }
    [System.GC]::Collect()
    Start-Sleep -Seconds 2
}

Write-Host "`n[CONCLUIDO] Comparacao finalizada. Logs: $LogFile" -ForegroundColor Yellow