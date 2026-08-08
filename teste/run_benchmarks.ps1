param(
    [string[]]$Opts,
    [switch]$Quick,
    [string]$Data,
    [string]$DataPath,
    [string]$RootPath,
    [string]$ModelDir,
    [string]$DockerContainer,
    [string]$RocmGfxVersion,
    [int]$Epochs,
    [int]$BatchSize,
    [int]$SeqLen,
    [int]$LabelLen,
    [int]$TokenLen,
    [int]$TestPredLen,
    [switch]$DryRun,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Continue"
$TestDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $TestDir

$CondaPy = "C:\Users\Emanuel\miniconda3\envs\autotimes\python.exe"

# ============================================================
# Interactive Configuration
# ============================================================
function Ask-Param {
    param([string]$Name, $Default, [string]$Prompt)
    $val = Get-Variable -Name $Name -ErrorAction SilentlyContinue
    if ($val -and $val.Value -ne $null -and $val.Value -ne "") {
        return $val.Value
    }
    $resp = Read-Host "$Prompt [padrão: $Default]"
    if ([string]::IsNullOrWhiteSpace($resp)) { return $Default }
    return $resp
}

function Ask-ParamInt {
    param([string]$Name, [int]$Default, [string]$Prompt)
    $val = Get-Variable -Name $Name -ErrorAction SilentlyContinue
    if ($val -and $val.Value -ne $null -and $val.Value -ne 0 -and $val.Value -is [int]) {
        return $val.Value
    }
    $resp = Read-Host "$Prompt [padrão: $Default]"
    if ([string]::IsNullOrWhiteSpace($resp)) { return $Default }
    $r = 0
    if ([int]::TryParse($resp, [ref]$r)) { return $r }
    return $Default
}

function Ask-ParamSwitch {
    param([string]$Name, [bool]$Default, [string]$Prompt)
    $val = Get-Variable -Name $Name -ErrorAction SilentlyContinue
    if ($val -and $val.Value -ne $null) { return $val.Value }
    $resp = Read-Host "$Prompt (s/n) [padrão: $(if($Default){'s'}else{'n'})]"
    if ([string]::IsNullOrWhiteSpace($resp)) { return $Default }
    return $resp.ToLower().StartsWith('s')
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  CONFIGURAÇÃO INTERATIVA" -ForegroundColor Cyan
Write-Host "  (Enter = manter padrão)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$Data = Ask-Param "Data" "ETTh1" "  Dataset"
$DataPath = Ask-Param "DataPath" "ETTh1.csv" "  Arquivo de dados"
$RootPath = Ask-Param "RootPath" "../dataset/ETT-small/" "  Caminho dos dados"
$ModelDir = Ask-Param "ModelDir" "gpt2" "  Modelo (nome HF ou path local)"
$Epochs = Ask-ParamInt "Epochs" 10 "  Épocas de treino"
$BatchSize = Ask-ParamInt "BatchSize" 32 "  Batch size"
$SeqLen = Ask-ParamInt "SeqLen" 672 "  Seq len"
$LabelLen = Ask-ParamInt "LabelLen" 576 "  Label len"
$TokenLen = Ask-ParamInt "TokenLen" 96 "  Token len"
$TestPredLen = Ask-ParamInt "TestPredLen" 96 "  Test pred len"
$RocmGfxVersion = Ask-Param "RocmGfxVersion" "11.0.0" "  ROCm GFX version"
$Quick = Ask-ParamSwitch "Quick" $false "  Modo rápido (3 épocas)"
$DryRun = Ask-ParamSwitch "DryRun" $false "  Dry-run (só mostra comandos)"
$SmokeTest = Ask-ParamSwitch "SmokeTest" $false "  Smoke test (1 batch por época)"

Write-Host "`nSeleção de otimizações:" -ForegroundColor Yellow
$allOpts = "base","opt1","opt2","opt3","opt4","opt5","opt6","opt7","opt8","opt9","opt10","opt11","opt12","opt13"
$optNamesMap = @{
    "base"="Baseline (original)"; "opt1"="AMD ROCm"; "opt2"="MLA"; "opt3"="MoE"; "opt4"="GRPO"; "opt5"="RAG Temporal"
    "opt6"="Mamba SSM"; "opt7"="MTP"; "opt8"="FinGPT"; "opt9"="SMETimes"
    "opt10"="Combo Eficiência"; "opt11"="Combo Preditor"; "opt12"="Ultimate"
    "opt13"="Combo SLM Contextual"
}
if (-not $Opts -or $Opts.Count -eq 0) {
    Write-Host "  Opções disponíveis:" -ForegroundColor Gray
    foreach ($o in $allOpts) { Write-Host "    $o - $($optNamesMap[$o])" }
    $resp = Read-Host "  Quais rodar? (separado por vírgula, ex: opt1,opt10,opt12) [padrão: TODAS]"
    if ([string]::IsNullOrWhiteSpace($resp)) { $Opts = $allOpts }
    else { $Opts = $resp -split ',' | ForEach-Object { $_.Trim() } }
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  CONFIGURAÇÃO FINAL" -ForegroundColor Green
Write-Host "  Data: $Data | Epochs: $Epochs | Batch: $BatchSize" -ForegroundColor Green
Write-Host "  SeqLen: $SeqLen | TokenLen: $TokenLen | TestPredLen: $TestPredLen" -ForegroundColor Green
Write-Host "  ROCm GFX: $RocmGfxVersion" -ForegroundColor Green
Write-Host "  Otimizações: $($Opts.Count) ($($Opts -join ', '))" -ForegroundColor Green
if ($Quick) { Write-Host "  Modo rápido: SIM" -ForegroundColor Yellow }
if ($DryRun) { Write-Host "  Dry-run: SIM" -ForegroundColor Yellow }
if ($SmokeTest) { Write-Host "  Smoke test: SIM (1 batch/época)" -ForegroundColor Yellow }
Write-Host "========================================`n" -ForegroundColor Green



$confirm = Read-Host "Iniciar? (s/n) [s]"
if (-not [string]::IsNullOrWhiteSpace($confirm) -and -not $confirm.ToLower().StartsWith('s')) {
    Write-Host "Cancelado." -ForegroundColor Red; exit
}

# ============================================================
# ROCm Environment Setup
# ============================================================
$env:HSA_OVERRIDE_GFX_VERSION = $RocmGfxVersion
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
$env:MIOPEN_FIND_MODE = "1"

Write-Host "ROCm Environment:" -ForegroundColor Blue
Write-Host "  HSA_OVERRIDE_GFX_VERSION = $RocmGfxVersion" -ForegroundColor Blue
Write-Host "  MIOPEN_FIND_MODE = 1 (força tuning)" -ForegroundColor Blue
Write-Host ""

# ============================================================
# GPU Memory Cleanup Function
# ============================================================
function Clear-GpuMemory {
    Write-Host "  [CLEANUP] A limpar memória GPU..." -ForegroundColor Yellow
    if ($DockerContainer) {
        docker exec $DockerContainer python -c "
import torch, gc, sys
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
print('  GPU memory cleaned.')
" 2>$null | ForEach-Object { Write-Host $_ }
    } else {
        & $CondaPy -c "
import torch, gc
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
" 2>$null
    }
    Start-Sleep -Seconds 3
}

# ============================================================
# Docker Helper
# ============================================================
function Get-PythonCmd {
    if ($DockerContainer) { return "docker exec $DockerContainer python" }
    return $CondaPy
}

$PythonCmd = Get-PythonCmd

$OptMap = @{
    "base"  = "run_base.py"
    "opt1"  = "run_opt1.py"; "opt2"  = "run_opt2.py"; "opt3"  = "run_opt3.py"
    "opt4"  = "run_opt4.py"; "opt5"  = "run_opt5.py"; "opt6"  = "run_opt6.py"
    "opt7"  = "run_opt7.py"; "opt8"  = "run_opt8.py"; "opt9"  = "run_opt9.py"
    "opt10" = "run_opt10.py"; "opt11" = "run_opt11.py"; "opt12" = "run_opt12.py"
    "opt13" = "run_opt13.py"
}

$OptNames = @{
    "base"="Base_Original"
    "opt1"="Opt_1_AMD_ROCm"; "opt2"="Opt_2_MLA"; "opt3"="Opt_3_MoE"
    "opt4"="Opt_4_GRPO"; "opt5"="Opt_5_RAG_Temporal"; "opt6"="Opt_6_Mamba_SSM"
    "opt7"="Opt_7_MTP"; "opt8"="Opt_8_FinGPT"; "opt9"="Opt_9_SMETimes"
    "opt10"="Opt_10_Combo_Eficiencia"; "opt11"="Opt_11_Combo_Preditor"
    "opt12"="Opt_12_Ultimate"; "opt13"="Opt_13_Combo_SLM_Contextual"
}

$BaseArgs = @(
    "--task_name=long_term_forecast", "--is_training=1", "--model_id=test",
    "--model=AutoTimes_Gpt2", "--data=$Data", "--root_path=$RootPath",
    "--data_path=$DataPath", "--llm_ckp_dir=$ModelDir", "--train_epochs=$Epochs",
    "--batch_size=$BatchSize", "--seq_len=$SeqLen", "--label_len=$LabelLen",
    "--token_len=$TokenLen", "--test_pred_len=$TestPredLen", "--use_amp"
)

if ($Quick) { $BaseArgs += @("--train_epochs=3", "--patience=2") }

function Run-Optimization {
    param([string]$OptId, [string]$Script)
    $Name = $OptNames[$OptId]
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $OptId - $Name" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan

    $OutputLog = "logs/${OptId}_output.txt"
    $ErrorLog  = "logs/${OptId}_errors.txt"
    if (-not (Test-Path -LiteralPath "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

    # -- Build argument list (no duplicates) --
    # Remove --train_epochs from BaseArgs so smoke-test can set it cleanly
    $CleanBase = $BaseArgs | Where-Object { $_ -notlike "--train_epochs=*" }
    $PyArgs = @($Script) + $CleanBase

    # Safety: only pass --train_epochs once (no duplicates)
    if ($SmokeTest) {
        $PyArgs += "--train_epochs=1"
    } elseif ($Quick) {
        $PyArgs += "--train_epochs=3"
    } else {
        $PyArgs += "--train_epochs=$Epochs"
    }

    $PyArgs += @("--opt_id=$OptId", "--opt_name=$Name")
    if ($SmokeTest) { $PyArgs += "--smoke_test=1" }

    # Reset CSV only on first opt (idx is in parent scope)
    if ($idx -eq 1) { $PyArgs += "--reset_csv" }

    if ($DryRun) {
        Write-Host "[DRY-RUN] $CondaPy $($PyArgs -join ' ')" -ForegroundColor Yellow
        return
    }

    # -- Execute via & operator + Tee-Object (exibe em tempo real + salva em arquivo) --
    $StartTime = Get-Date
    $exitCode = -1

    try {
        if ($DockerContainer) {
            $fullCmd = "docker exec $DockerContainer python $($PyArgs -join ' ')"
            Write-Host "[EXEC] $fullCmd" -ForegroundColor DarkGray
            & docker exec $DockerContainer python $PyArgs 2>&1 | Tee-Object -FilePath $OutputLog
            $exitCode = $LASTEXITCODE
        } else {
            & $CondaPy $PyArgs 2>&1 | Tee-Object -FilePath $OutputLog
            $exitCode = $LASTEXITCODE
        }

        $Elapsed = (Get-Date) - $StartTime

        if ($exitCode -eq 0) {
            Write-Host "[OK] $OptId concluído em $($Elapsed.TotalMinutes.ToString('F2')) min" -ForegroundColor Green
            Get-Content -LiteralPath $OutputLog -Tail 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        } else {
            Write-Host "[FAIL] $OptId — exit code $exitCode ($($Elapsed.TotalSeconds.ToString('F1'))s)" -ForegroundColor Red
            Write-Host "  --- LOG (completo) ---" -ForegroundColor Yellow
            Get-Content -LiteralPath $OutputLog | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            Write-Host "  --- FIM DO ERRO ---" -ForegroundColor Yellow
        }
    } catch {
        Write-Host ("[EXCEPTION] ${OptId}: $_") -ForegroundColor Red
        $exitCode = -999
    }

    Clear-GpuMemory
    return $exitCode
}

# ============================================================
# Main Execution
# ============================================================
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  AutoTimes - Benchmark Runner" -ForegroundColor Magenta
Write-Host "  Backbone: GPT-2 (124M)" -ForegroundColor Magenta
Write-Host "  Data: $Data | Epochs: $Epochs | Batch: $BatchSize" -ForegroundColor Magenta
Write-Host "  GPU ROCm: GFX=$RocmGfxVersion" -ForegroundColor Magenta
if ($DockerContainer) { Write-Host "  Docker: $DockerContainer" -ForegroundColor Magenta }
Write-Host "  Optimizações: $($Opts.Count)" -ForegroundColor Magenta
if ($DryRun) { Write-Host "  *** DRY-RUN MODE ***" -ForegroundColor Yellow }
Write-Host "========================================`n" -ForegroundColor Magenta

if (-not $DryRun) {
    Write-Host "Diagnóstico GPU:" -ForegroundColor Blue
    if ($DockerContainer) {
        docker exec $DockerContainer python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  ROCm available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
" 2>$null
    } else {
        & $CondaPy -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  ROCm available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
" 2>$null
    }
    Write-Host ""
}

$idx = 0
foreach ($optId in $Opts) {
    $idx++
    if (-not $OptMap.ContainsKey($optId)) { Write-Host "[WARN] Opt '$optId' desconhecida, ignorando" -ForegroundColor Yellow; continue }
    if (-not (Test-Path -LiteralPath $OptMap[$optId])) { Write-Host "[WARN] $($OptMap[$optId]) não encontrado, ignorando" -ForegroundColor Yellow; continue }
    Write-Host "[$idx/$($Opts.Count)]" -ForegroundColor Cyan -NoNewline
    Run-Optimization -OptId $optId -Script $OptMap[$optId]
}

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  Benchmarking concluído!" -ForegroundColor Magenta
Write-Host "  Resultados: benchmark_results.csv" -ForegroundColor Magenta
Write-Host "  Logs: logs/" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

if (-not $DryRun -and (Test-Path -LiteralPath "benchmark_results.csv")) {
    Write-Host "Resumo:" -ForegroundColor Blue
    Import-Csv -Path "benchmark_results.csv" | Select-Object opt_id, opt_name, total_time_s, test_mse, test_mae, peak_vram_gb | Format-Table -AutoSize
}
