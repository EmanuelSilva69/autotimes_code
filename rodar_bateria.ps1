$env:PYTHONUNBUFFERED=1
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
$env:HSA_OVERRIDE_GFX_VERSION="12.0.0"

# ==============================================================================
# SISTEMA DE LOGS DINAMICOS
# ==============================================================================
$LogDir = ".\logs"
if (-not (Test-Path -Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Timestamp = Get-Date -Format "dd-MM-yyyy_HH-mm-ss"
$LogFile = "$LogDir\Log-$Timestamp.txt"
$ErroFile = "$LogDir\LogErros-$Timestamp.txt"
$StartTime = Get-Date

# ==============================================================================
# MENU 1: ESCOLHA DO MODELO LLM (Generality)
# ==============================================================================
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Bem-vindo ao AutoTimes - Bateria de Testes Completa" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Selecione o modelo LLM base:" -ForegroundColor Yellow
Write-Host "[1] GPT-2 (Padrao do Artigo)"
Write-Host "[2] LLaMA (LLaMA-7B)"
Write-Host "[3] OPT"
$llm_opcao = Read-Host "Digite a opcao desejada (1-3) [Padrao: 1]"

switch ($llm_opcao) {
    '2' { $ModelName = "AutoTimes_Llama"; $LlmDir = "llama" }
    '3' { $ModelName = "AutoTimes_Opt_1b";   $LlmDir = "opt" }
    Default { $ModelName = "AutoTimes_Gpt2";  $LlmDir = "gpt2" }
}

# ==============================================================================
# MENU 2: FUNCIONALIDADES EXTRAS DO ARTIGO (Tasks & Context)
# ==============================================================================
Write-Host "`n--- Tarefas (Tasks) e Zero-Shot ---" -ForegroundColor Cyan
Write-Host "[1] Long-Term Forecasting (Padrao)"
Write-Host "[2] Short-Term Forecasting"
Write-Host "[3] Zero-Shot Forecasting (Requer treino previo em outro dataset)"
Write-Host "[4] In-Context Forecasting"
$task_opcao = Read-Host "Escolha a tarefa principal (1-4) [Padrao: 1]"

switch ($task_opcao) {
    '2' { $TaskName = "short_term_forecast" }
    '3' { $TaskName = "zero_shot_forecast" }
    '4' { $TaskName = "in_context_forecast" }
    Default { $TaskName = "long_term_forecast" }
}

Write-Host "`n--- Textual Embeddings (Metadados de Data/Hora) ---" -ForegroundColor Cyan
Write-Host "Deseja gerar os embeddings textuais dos timestamps antes de treinar?"
Write-Host "Isso executa o preprocess.py para enriquecer a previsao (recomendado para In-Context)."
$pre_opcao = Read-Host "Gerar Embeddings? (S/N) [Padrao: N]"

# ==============================================================================
# MENU 3: MODO DE CONFIGURACAO (Eficiencia)
# ==============================================================================
Write-Host "`nSelecione o perfil de configuracao de hardware:" -ForegroundColor Yellow
Write-Host "[1] Original (Fiel ao Artigo - 10 epocas, Batch 256)"
Write-Host "[2] Acelerado (num_workers, batch_size, early stopping)"
$modo_opcao = Read-Host "Digite a opcao desejada (1-2) [Padrao: 1]"

# Valores Originais (Padrao)
$NumWorkers = 4
$BatchSize = 256
$TrainEpochs = 10

if ($modo_opcao -eq '2') {
    Write-Host "`n--- Configuracoes de Aceleracao ---" -ForegroundColor Cyan
    
    $in_workers = Read-Host "1. Otimizacao de Leitura (num_workers - Ex: 0, 4 ou 8) [Padrao: 4]"
    if (-not [string]::IsNullOrWhiteSpace($in_workers)) { $NumWorkers = $in_workers }

    $in_batch = Read-Host "2. Tamanho do Lote (Batch Size - Ex: 512, 1024) [Padrao: 256]"
    if (-not [string]::IsNullOrWhiteSpace($in_batch)) { $BatchSize = $in_batch }

    $in_epochs = Read-Host "3. Reduzir Epocas Maximas (Ex: 3 ou 5) [Padrao: 10]"
    if (-not [string]::IsNullOrWhiteSpace($in_epochs)) { $TrainEpochs = $in_epochs }
}

# ==============================================================================
# MENU 4: MODO DE OTIMIZACAO (Eficiencia ROCm)
# ==============================================================================
Write-Host "`n--- Modo de Otimizacao ---" -ForegroundColor Cyan
Write-Host "[1] Original (lento, tributo ao artigo - fp32, sem AMP/TF32/benchmark)"
Write-Host "[2] Otimizada (opt1 + ROCm - Mixed Precision, TF32, cudnn benchmark) [Padrao]"
$otimizacao_opcao = Read-Host "Digite a opcao desejada (1-2) [Padrao: 2]"

$ModoOtimizacao = "optimized"
$UsarAmp = $true
if ($otimizacao_opcao -eq '1') {
    $ModoOtimizacao = "original"
    $UsarAmp = $false
}

# ==============================================================================
# MENU 5: SMETIMES (Injecao de Features Estatisticas / opt9)
# ==============================================================================
Write-Host "`n--- SMETimes (Features Estatisticas - opt9) ---" -ForegroundColor Cyan
Write-Host "Injeta media, variancia, tendencia e range (max-min) por token via AdaptiveFusionGate."
Write-Host "[1] Nao usar [Padrao]"
Write-Host "[2] Usar SMETimes (apenas com GPT-2)"
$smetimes_opcao = Read-Host "Digite a opcao desejada (1-2) [Padrao: 1]"

$UsarSMETimes = $false
if ($smetimes_opcao -eq '2') {
    if ($ModelName -eq 'AutoTimes_Gpt2') {
        $UsarSMETimes = $true
        $ModelName = "AutoTimes_Gpt2_SMETimes"
        Write-Host "[SMETIMES] Modelo alterado para AutoTimes_Gpt2_SMETimes" -ForegroundColor Green
    } else {
        Write-Host "[AVISO] SMETimes so e suportado com GPT-2. Usando $ModelName sem SMETimes." -ForegroundColor Red
    }
}

# ==============================================================================
# MENU 6: ESCOLHA DE DATASETS
# ==============================================================================
$TodosDatasets = @(
    @{ Id = 1; Name = "ETTh1 (Transformadores Eletricos)";    Root = "./dataset/ETT-small/";    Data = "ETTh1.csv";       DataType = "ETTh1";  ModelId = "ETTh1_672_96";    LR = "0.001" }
    @{ Id = 2; Name = "Electricity (Consumo de Eletricidade)"; Root = "./dataset/electricity/"; Data = "electricity.csv";  DataType = "custom"; ModelId = "ECL_672_96";       LR = "0.001" }
    @{ Id = 3; Name = "Weather (Clima)";                       Root = "./dataset/weather/";     Data = "weather.csv";      DataType = "custom"; ModelId = "weather_672_96";    LR = "0.001" }
    @{ Id = 4; Name = "Traffic (Transito)";                    Root = "./dataset/traffic/";     Data = "traffic.csv";      DataType = "custom"; ModelId = "traffic_672_96";    LR = "0.001" }
)

Write-Host "`n--- Selecao de Datasets ---" -ForegroundColor Cyan
Write-Host "[0] Todos os Datasets"
foreach ($ds in $TodosDatasets) {
    Write-Host "[$($ds.Id)] $($ds.Name)"
}
$ds_opcao = Read-Host "Escolha os datasets a executar (Ex: 0 para todos, ou 1, 4) [Padrao: 0]"

$Experiments = @()
if ($ds_opcao -eq '0' -or [string]::IsNullOrWhiteSpace($ds_opcao)) {
    $Experiments = $TodosDatasets
    $DsSelecionados = "Todos"
} else {
    # Permite escolher múltiplos datasets separados por vírgula (ex: "1,3")
    $opcoes_escolhidas = $ds_opcao -split ',' | ForEach-Object { $_.Trim() }
    foreach ($op in $opcoes_escolhidas) {
        $found = $TodosDatasets | Where-Object { $_.Id -eq $op }
        if ($found) { $Experiments += $found }
    }
    $DsSelecionados = ($Experiments.Name) -join ", "
}

if ($Experiments.Count -eq 0) {
    Write-Host "Nenhum dataset valido selecionado. A encerrar..." -ForegroundColor Red
    exit
}


Write-Host "`n[RESUMO] LLM: $ModelName | Task: $TaskName | Batch: $BatchSize | Epochs: $TrainEpochs | Workers: $NumWorkers" -ForegroundColor Green
Write-Host "Modo: $(if ($ModoOtimizacao -eq 'original') {'Original (lento/tributo)'} else {'Otimizada (opt1 + ROCm)'}) | SMETimes: $(if ($UsarSMETimes) {'SIM'} else {'NAO'})" -ForegroundColor Green
Write-Host "Datasets: $DsSelecionados" -ForegroundColor Green
Write-Host "Salvando logs em: $LogFile" -ForegroundColor DarkGray
Start-Sleep -Seconds 3

# ==============================================================================
# INICIO DO SCRIPT DE EXECUCAO
# ==============================================================================
"Bateria de Testes - AutoTimes ($ModelName - $TaskName)" | Out-File $LogFile
"Inicio: $StartTime`n" | Out-File $LogFile -Append
"Inicio: $StartTime" | Out-File $ErroFile

# Executa o Pré-processamento se solicitado
if ($pre_opcao -match '^[sS]') {
    Write-Host "`n==================================================" -ForegroundColor Magenta
    Write-Host "Iniciando Geracao de Embeddings Textuais" -ForegroundColor Yellow
    "Geracao de Embeddings Textuais" | Out-File $LogFile -Append
    foreach ($exp in $Experiments) {
        $PreCmd = "python ./preprocess.py --gpu 0 --dataset $($exp.DataType)"
        Write-Host "[PRE-PROCESS] $PreCmd" -ForegroundColor Gray
        & python ./preprocess.py --gpu 0 --dataset $exp.DataType 2>&1 | Out-File $LogFile -Append
    }
    Write-Host "==================================================`n" -ForegroundColor Magenta
}

# Argumentos base dinamicos
$CommonArgs = @(
    "--task_name", $TaskName,
    "--is_training", "1",
    "--model", $ModelName,
    "--seq_len", "672",
    "--label_len", "576",
    "--token_len", "96",
    "--test_seq_len", "672",
    "--test_label_len", "576",
    "--test_pred_len", "96",
    "--batch_size", "$BatchSize",
    "--train_epochs", "$TrainEpochs",
    "--num_workers", "$NumWorkers",
    "--mode", $ModoOtimizacao,
    "--llm_ckp_dir", $LlmDir,
    "--cosine",
    "--tmax", "10",
    "--des", "Exp",
    "--gpu", "0",
    "--drop_last",
    "--mlp_hidden_dim", "256",
    "--mlp_hidden_layers", "2"
)

if ($UsarAmp) { $CommonArgs += "--use_amp" }
if ($UsarSMETimes) { $CommonArgs += @("--n_stat_feats", "4") }

foreach ($exp in $Experiments) {
    Write-Host "`n==================================================" -ForegroundColor Cyan
    Write-Host "Iniciando: $($exp.Name)" -ForegroundColor Green
    Write-Host "$(Get-Date)" -ForegroundColor Gray
    Write-Host "==================================================" -ForegroundColor Cyan

    "`n==================================================" | Out-File $LogFile -Append
    "Dataset: $($exp.Name)" | Out-File $LogFile -Append
    "Inicio: $(Get-Date)" | Out-File $LogFile -Append
    "==================================================" | Out-File $LogFile -Append

    $ArgList = $CommonArgs + @(
        "--root_path", $exp.Root,
        "--data_path", $exp.Data,
        "--model_id", $exp.ModelId,
        "--data", $exp.DataType,
        "--learning_rate", $exp.LR
    )

    Write-Host "[COMANDO] python run.py $($ArgList -join ' ')" -ForegroundColor Gray

    try {
        & python -u run.py @ArgList 2>&1 | ForEach-Object {
            $_
            $_ | Out-File $LogFile -Append
        }
        Write-Host "[OK] $($exp.Name) concluido" -ForegroundColor Green
    } catch {
        $errMsg = "ERRO em $($exp.Name): $_"
        Write-Host $errMsg -ForegroundColor Red
        $errMsg | Out-File $ErroFile -Append
        $errMsg | Out-File $LogFile -Append
    }

    "Fim: $(Get-Date)" | Out-File $LogFile -Append
    [System.GC]::Collect()
    Start-Sleep -Seconds 3
}

$EndTime = Get-Date
$Duration = $EndTime - $StartTime
Write-Host "`n[CONCLUIDO] Todos os testes finalizaram!" -ForegroundColor Yellow
Write-Host "Duracao total: $($Duration.Hours)h $($Duration.Minutes)m $($Duration.Seconds)s" -ForegroundColor Yellow
Write-Host "Resultados salvos em: $LogFile" -ForegroundColor Green
if (Test-Path $ErroFile) { Write-Host "Erros salvos em: $ErroFile" -ForegroundColor Yellow }
"`nFim: $EndTime" | Out-File $LogFile -Append
"Duracao: $($Duration.Hours)h $($Duration.Minutes)m $($Duration.Seconds)s" | Out-File $LogFile -Append