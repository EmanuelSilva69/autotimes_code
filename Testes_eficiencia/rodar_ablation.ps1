<#
.SYNOPSIS
  Executa todo o Ablation Study do AutoTimes de uma vez (benchmark + graficos).

.DESCRIPTION
  Orquestra run_benchmarks.py + analyze_logs.py com configuracao flexivel:
    - menu interativo pergunta batch size, stride da janela, epocas,
      early stopping (+ patience) e dataset, com valores padrao via ENTER;
    - liga/desliga cada melhoria isolada e o Frankenstein (config file);
    - combos customizados ("robust+overlap+vol", "vol+gate", ...);
    - epocas, janelas, backbone, learning rates e beta da perda.
  Configuracoes espelhadas na bateria principal (rodar_bateria.ps1):
    batch 256, lr 0.001, CosineAnnealingLR (tmax=10) e AMP (mixed precision).
  Interativo por padrao; use -NoMenu para usar as configs do bloco no topo.

.PARAMETER NoMenu
  Usa as configuracoes abaixo sem abrir o menu interativo.
.PARAMETER Quick
  Passa --quick ao run_benchmarks (smoke test rapido).
.PARAMETER CleanLogs
  Apaga os arquivos de log (todas as sessoes: training_history_*.log e
  benchmark_results_*.json) antes de rodar.
.PARAMETER SkipPlots
  Nao gera os graficos ao final.
.PARAMETER TesteOtimizado
  Perfil "Teste Otimizado": combo 'otimizado' (todas as melhorias) com
  stride=1 (todas as janelas), descongelamento das ultimas camadas do
  GPT-2 (lr_llm=1e-6) e loss hibrida volatil + Huber/L1.
.PARAMETER TextoPuro
  Perfil "Texto Puro": representa cada bloco de 96h por um TEXTO descritivo
  (pt-BR, por extenso: data/hora + valores) embeddado pelo wte do GPT-2,
  sem tensores numericos no fluxo do LLM. Passa --textual/--text-step/
  --block-max; a serializacao fica em cache (dataset/<nome>_text.csv.

.PARAMETER Sanity
  Valida somente sem treinar: descongelamento parcial (lr_llm=1e-6), perda
  hibrida VolLoss+L1 e o formato textual (CSV-texto + tokenizacao + forward/
  backward). Roda rapido e sai.

.EXAMPLE
  .\rodar_ablation.ps1 -Sanity
  .\rodar_ablation.ps1 -Sanity -TextoPuro
  .\rodar_ablation.ps1 -NoMenu
  .\rodar_ablation.ps1 -Quick -NoMenu
  .\rodar_ablation.ps1 -TesteOtimizado
  .\rodar_ablation.ps1 -TextoPuro
  .\rodar_ablation.ps1
#>

param(
    [switch]$NoMenu,
    [switch]$Quick,
    [switch]$CleanLogs,
    [switch]$Sanity,
    [switch]$SkipPlots,
    [switch]$TesteOtimizado,
    [switch]$TextoPuro
)

$ErrorActionPreference = "Stop"

$env:PYTHONUNBUFFERED=1
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1"
$env:HSA_OVERRIDE_GFX_VERSION="12.0.0"

# ===================================================================== #
#  CONFIGURACAO - edite aqui (o menu sobrescreve em runtime no modo menu) #
# ===================================================================== #
$PythonCmd = $null      # vazio: tenta venv autotimes_env e depois 'python'
$BenchDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $BenchDir

# Dados
$DataDir = Join-Path $ProjectRoot "dataset\ETT-small"
$DataPath = "ETTh1.csv"    # ou ETTh2 / ETTm1 / ETTm2
$SeqLen = 672
$PredLen = 96
$LabelLen = 576           # contexto label (igual bateria: token_len = label_len/6 = 96)

# Treino e otimizacao (espelhado na bateria rodar_bateria.ps1: lr 0.001,
# batch 256, CosineAnnealingLR tmax=10 e AMP)
$Epochs = 10
$BatchSize = 256
$WindowStride = 1           # 1 = TODAS as janelas (igual ao loader original); >1 subamostra
$LRBase = 1e-3
$LRLLm = 2e-5
$LRNew = 1e-3
$WeightDecay = 0.0
$T0 = 5                     # T_0 do CosineAnnealingWarmRestarts
$TMult = 2
$TMax = 10                  # T_max do CosineAnnealingLR (igual bateria)
$Beta = 0.5                 # VolatilityAwareLogCoshLoss

# Otimizacao (igual bateria modo "Otimizada"): AMP + cosine
$UseAmp = $true
$UseCosine = $true

# Early stopping (menu pergunta; -NoMenu usa estes defaults)
$EarlyStop = $true
$Patience = 3

# Backbone: GPT-2 base REAL (transformers) do diretorio local do projeto
$Gpt2Dir = Join-Path $ProjectRoot "gpt2"

# Tokenizer/Detokenizer MLP (igual bateria AutoTimes_Gpt2: 256x2, tanh)
$MLPHiddenDim = 256
$MLPHiddenLayers = 2
$MLPActivation = "tanh"

# Toggles dos testes isolados (Run 0 baseline sempre roda)
$RobustOn = $true       # Run 1
$VolLossOn = $true      # Run 2
$OverlapOn = $true     # Run 3
$DiscOn = $true         # Run 4
$GateOn = $true         # Run 5
$FrankOn = $true        # Run 6

# Combos adicionais (tokens: robust, vol, overlap, disc, gate, unfreeze, hybrid;
# 'otimizado' = todas). -TesteOtimizado substitui este valor no runtime.
$Combos = @("robust+overlap+vol", "vol+gate")

# Perfil "Teste Otimizado" (-TesteOtimizado): descongelamento parcial do
# GPT-2 (lr_llm muito baixa) + loss hibrida (volatil + L1/Huber) + stride=1.
$UnfreezeLayers = 2       # camadas do GPT-2 a descongelar (das ultimas p/ tras)
$LRLLmOtimizado = 1e-6    # LR do backbone no perfil otimizado
$HybridRobust = "huber"   # termo robusto da loss hibrida: huber | l1
$HybridLambda = 0.5       # peso do termo robusto
$HybridDelta = 1.0        # delta do Huber

# Perfil "Texto Puro" (-TextoPuro): serialize os dados como texto (pt-BR)
# e embeda via wte do GPT-2. text_step=24 amostra 4 valores/96h (~100
# tokens por bloco); reduzir se o texto truncar (max_length=block_max).
$TextStep = 24             # pontos amostrados por bloco no texto descritivo
$BlockMax = 128            # max tokens por bloco textual (pad ao final)

# Pos-processamento
$DoPlots = $true
$OpenPlots = $false

# ===================================================================== #
#  Logica (nao precisa editar)                                          #
# ===================================================================== #
$DataFiles = @("ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv")
# ------------------------------------------------------------------ #
#  Funcoes de execucao                                                 #
# ------------------------------------------------------------------ #
function Get-Python {
    if ($PythonCmd) { return $PythonCmd }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $venvPy = Join-Path $ProjectRoot "autotimes_env\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) { return $venvPy }
    return "python"
}

function Get-RunFilter {
    $f = @()
    if ($RobustOn)   { $f += "1" }
    if ($VolLossOn)  { $f += "2" }
    if ($OverlapOn)  { $f += "3" }
    if ($DiscOn)     { $f += "4" }
    if ($GateOn)     { $f += "5" }
    if ($FrankOn)    { $f += "6" }
    return ($f -join ",")   # Run 0 (baseline) entra automatico no py
}

function Invoke-Bench {
    $py = Get-Python
    $filter = Get-RunFilter
    $combos = ($Combos | Where-Object { $_ }) -join ","

    Write-Host ""
    Write-Host "=================== ABLATION STUDY ===================" -ForegroundColor Cyan
    Write-Host "  Python   : $py"
    Write-Host "  Dataset  : $DataPath  em  $DataDir"
    Write-Host "  Runs     : filtro=$filter   combos=$combos"
    Write-Host "  Epocas   : $Epochs   batch=$BatchSize   stride=$WindowStride"
    Write-Host "  Early stop: $(if ($EarlyStop) { "SIM" } else { "NAO" })  patience=$Patience"
    Write-Host "  Seq/pred/label : $SeqLen / $PredLen / $LabelLen    backbone=gpt2-base real em $Gpt2Dir"
    Write-Host "  MLP tokenizer  : dim=$MLPHiddenDim  camadas=$MLPHiddenLayers  act=$MLPActivation"
    Write-Host "  Unfreeze GPT-2 : ultimas $UnfreezeLayers camadas (so uso nas runs unfreeze/hybrid; lr_llm=$LRLLm)"
    Write-Host "  Loss hibrida   : robust=$HybridRobust  lambda=$HybridLambda  delta=$HybridDelta"
    if ($TextoPuro) {
        Write-Host "  Representacao  : TEXTO PURO (pt-BR, por extenso) - wte GPT-2, sem tensores numericos"
        Write-Host "    text_step=$TextStep  block_max=$BlockMax  (cache em text_cache/)"
    }
    Write-Host "  LR base/llm/new : $LRBase / $LRLLm / $LRNew   WD=$WeightDecay"
    Write-Host "  AMP=$UseAmp  Cosine=$UseCosine (tmax=$TMax)  T0=$T0  TMult=$TMult  Beta=$Beta"
    Write-Host "=======================================================" -ForegroundColor Cyan

    $benchArgs = @(
        "--data-dir",      $DataDir,
        "--data-path",     $DataPath,
        "--seq-len",       $SeqLen,
        "--label-len",     $LabelLen,
        "--pred-len",      $PredLen,
        "--epochs",        $Epochs,
        "--batch_size",    $BatchSize,
        "--stride",        $WindowStride,
        "--gpt2-dir",      $Gpt2Dir,
        "--mlp-hidden-dim",   $MLPHiddenDim,
        "--mlp-hidden-layers", $MLPHiddenLayers,
        "--mlp-activation",   $MLPActivation,
        "--lr-base",       $LRBase,
        "--lr-llm",        $LRLLm,
        "--lr-new",        $LRNew,
        "--weight-decay",  $WeightDecay,
        "--t0",            $T0,
        "--t-mult",        $TMult,
        "--beta",          $Beta,
        "--unfreeze-layers", $UnfreezeLayers,
        "--hybrid-robust",   $HybridRobust,
        "--hybrid-lambda",   $HybridLambda,
        "--hybrid-delta",    $HybridDelta
    )
    if ($TextoPuro) {
        $benchArgs += @("--textual", "--text-step", $TextStep, "--block-max", $BlockMax)
    }
    if ($UseAmp) { $benchArgs += "--use-amp" }
    if ($UseCosine) { $benchArgs += @("--cosine", "--tmax", $TMax) }
    if ($EarlyStop) {
        $benchArgs += @("--early_stopping", "--patience", $Patience)
    }
    if ($filter) { $benchArgs += @("--run-filter", $filter) }
    if ($combos) { $benchArgs += @("--combos", $combos) }
    if ($Quick) { $benchArgs += "--quick" }
    if ($Sanity) { $benchArgs += "--sanity" }

    Push-Location $BenchDir
    try {
        & $py run_benchmarks.py @benchArgs
        if ($LASTEXITCODE -ne 0) { throw "run_benchmarks.py falhou (exit $LASTEXITCODE)" }
    }
    finally { Pop-Location }
}

function Invoke-Analyze {
    if (-not $DoPlots) { return }
    $py = Get-Python
    Push-Location $BenchDir
    try {
        Write-Host ""
        Write-Host "=========== ANALISE E GRAFICOS ===========" -ForegroundColor Cyan
        & $py analyze_logs.py
        if ($LASTEXITCODE -ne 0) { throw "analyze_logs.py falhou (exit $LASTEXITCODE)" }
    }
    finally { Pop-Location }
    if ($OpenPlots) {
        $pDir = Join-Path $BenchDir "plots"
        if (Test-Path -LiteralPath $pDir) { Start-Process explorer.exe $pDir }
    }
}

function Reset-Logs {
    if (-not $CleanLogs) { return }
    Remove-Item (Join-Path $BenchDir "log\training_history*.log") -Force -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $BenchDir "log\benchmark_results*.json") -Force -ErrorAction SilentlyContinue
    Write-Host "[logs] historico anterior apagado (todos os arquivos de log/sessao)" -ForegroundColor Yellow
}
# ------------------------------------------------------------------ #
#  Menu interativo de configuracao                                    #
# ------------------------------------------------------------------ #
function Ask-Default {
    param([string]$Label, [string]$Default, [string]$Hint = "")
    $prompt = "  $Label"
    if ($Hint) { $prompt += "   [$Hint]" }
    $answer = Read-Host $prompt
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim()
}

function Invoke-Menu {
    Clear-Host
    Write-Host ""
    Write-Host "=== CONFIGURAÇÃO DO ABLATION STUDY ===" -ForegroundColor Cyan
    Write-Host "  Pressione apenas ENTER para usar o valor padrão." -ForegroundColor DarkGray
    Write-Host ""

    # 1) Batch size
    $b = Ask-Default "Batch size" "256" "dica: 256 = padrao da bateria, 512/1024 em GPUs grandes"

    # 2) Stride da janela deslizante
    $s = Ask-Default "Stride da janela deslizante" "$WindowStride" "dica: 1 = dataset completo, 8 pula dados"

    # 3) Epocas
    $e = Ask-Default "Épocas" "$Epochs" ""

    # 4) Early Stopping (e patience)
    $esRaw = Ask-Default "Ativar Early Stopping? [S/N]" "S" "S padrao"
    $esOn = ($esRaw -match "^(s|S)")
    if ($esOn) {
        $pa = Ask-Default "Patience (epocas sem melhora na val_loss)" "$Patience" "ex: 3"
    } else {
        $pa = "$Patience"
    }

    # 5) AMP / mixed precision (igual bateria modo Otimizada)
    $ampRaw = Ask-Default "Usar AMP / mixed precision? [S/N]" "S" "S padrao (bateria Otimizada)"

    # Conversao com fallback para os defaults (entradas invalidas nao quebram)
    $tmp = 0
    if ([int]::TryParse($b, [ref]$tmp))  { $script:BatchSize   = $tmp } else { $script:BatchSize = 256 }
    if ([int]::TryParse($s, [ref]$tmp))  { $script:WindowStride = $tmp } else { $script:WindowStride = 1 }
    if ([int]::TryParse($e, [ref]$tmp))  { $script:Epochs       = $tmp } else { $script:Epochs = 10 }
    if ([int]::TryParse($pa, [ref]$tmp)) { $script:Patience     = $tmp } else { $script:Patience = 3 }
    $script:EarlyStop = $esOn
    $script:UseAmp = ($ampRaw -match "^(s|S)")

    # 5) Dataset (opcional, mantem o configurado se invalido)
    $d = Ask-Default "Dataset [ETTh1|ETTh2|ETTm1|ETTm2]" $DataPath ""
    if ($DataFiles -contains $d) { $script:DataPath = $d }
    else { Write-Host "  [!] '$d' invalido; mantendo $DataPath" -ForegroundColor DarkYellow }

    # 5b) Representacao: Numerico (ETTh1.csv) vs Texto Puro (ETTh1_text.csv)
    $modoRaw = Ask-Default "Representacao: [N]umerico / [T]exto Puro" "N" "N padrao (Python normal)"
    $script:TextoPuro = ($modoRaw -match "^(t|T)")

    # 6) Combos extras (opcional)
    $c = Ask-Default "Combos extras (ex: 'vol+gate' ou Enter p/ manter)" ($Combos -join ",") ""
    if ($c -match "\S") { $script:Combos = @($c -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }

    Write-Host ""
    Write-Host ("  Resumo -> batch={0}  stride={1}  epocas={2}  early_stop={3}  patience={4}  amp={5}" -f `
        $script:BatchSize, $script:WindowStride, $script:Epochs, $(if ($script:EarlyStop) { "SIM" } else { "NAO" }), $script:Patience, $(if ($script:UseAmp) { "SIM" } else { "NAO" })) -ForegroundColor Green
    Write-Host ("            dataset={0}  representacao={1}  combos={2}" -f $script:DataPath, $(if ($script:TextoPuro) {"TEXTO PURO"} else {"NUMERICO (normal)"}), ($script:Combos -join ",")) -ForegroundColor Green
    Write-Host ""
    $go = Read-Host "  Rodar agora? [R]odar / [X]ancelar"
    return ($go -match "^(r|R)")
}

# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
Reset-Logs

# Perfil "Teste Otimizado": todas as melhorias num combo unico, com
# stride=1 (100% das janelas), descongelamento parcial do GPT-2 (lr_llm
# muito baixa) e loss hibrida graças aos --hybrid-* acima.
if ($TesteOtimizado) {
    $NoMenu = $true
    $WindowStride = 1
    $LRLLm = $LRLLmOtimizado
    $Combos = @("otimizado")
    Write-Host ("[perfil] Teste Otimizado: stride={0}, unfreeze ultimas {1} camadas do GPT-2, " -f $WindowStride, $UnfreezeLayers) -ForegroundColor Cyan
    Write-Host ("         lr_llm={0}  loss hibrida=vol+{1} (lambda={2}, delta={3})" -f $LRLLm, $HybridRobust, $HybridLambda, $HybridDelta) -ForegroundColor Cyan
}

# Perfil "Texto Puro": representa cada bloco de 96h por um TEXTO descritivo
# (pt-BR, por extenso: data/hora + valores), embeddado pelo wte do GPT-2 --
# sem tensores numericos no fluxo do LLM. A serializacao/tokenizacao fica
# em cache (text_cache\) e a bateria completa roda sobre essa representacao.
if ($TextoPuro) {
    if ($TesteOtimizado) {
        Write-Host "[perfil] -TesteOtimizado e -TextoPuro sao mutuamente exclusivos; usando TextoPuro." -ForegroundColor DarkYellow
    }
    $NoMenu = $true
    $WindowStride = 1
    $Combos = @()          # bateria completa textual = Runs 0..6 (numericas)
    Write-Host ("[perfil] Texto Puro: texto descritivo (por extenso) por bloco de 96h,") -ForegroundColor Cyan
    Write-Host ("         embeddado pelo wte do GPT-2, sem tensores numericos no LLM." -f $null) -ForegroundColor Cyan
    Write-Host ("         text_step={0}  block_max={1}  (cache: dataset/<nome>_text.csv)" -f $TextStep, $BlockMax) -ForegroundColor Cyan
}

# Perfil "Sanity": validacao rapida (unfreeze lr_llm=1e-6, VolLoss+L1, modo
# textual ponta a ponta) sem treinar a bateria. Combina com -TextoPuro.
if ($Sanity) {
    $NoMenu = $true
    Write-Host "[perfil] VALIDACAO (--sanity): checa descongelamento parcial (lr_llm=$LRLLm)," -ForegroundColor Cyan
    Write-Host ("         loss hibrida VolLoss+L1 (robust=$HybridRobust, lambda=$HybridLambda" -f $null) -ForegroundColor Cyan
    Write-Host ("         textual=$($(if ($TextoPuro) { 'SIM' } else { 'NAO' })))" -f $null) -ForegroundColor Cyan
}

if ($NoMenu) {
    $doRun = $true
} else {
    $doRun = Invoke-Menu
}

if ($doRun) {
    Invoke-Bench
    Invoke-Analyze
    Write-Host ""
    Write-Host "=== Ablation Study concluido ===" -ForegroundColor Green
} else {
    Write-Host "Encerrado sem rodar."
}