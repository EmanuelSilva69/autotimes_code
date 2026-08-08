param(
    [int]$StartAt = 0,
    [int]$EndAt = 15,
    [switch]$Single
)

$CondaPy = "C:\Users\Emanuel\miniconda3\envs\autotimes\python.exe"
$TestDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $TestDir

$Opts = @(
    "base", "opt1", "opt2", "opt3", "opt4", "opt5", "opt6", "opt7",
    "opt8", "opt9", "opt10", "opt11", "opt12", "opt13"
)

$ScriptMap = @{
    "base"  = "run_base.py";   "opt1"  = "run_opt1.py";  "opt2"  = "run_opt2.py"
    "opt3"  = "run_opt3.py";   "opt4"  = "run_opt4.py";  "opt5"  = "run_opt5.py"
    "opt6"  = "run_opt6.py";   "opt7"  = "run_opt7.py";  "opt8"  = "run_opt8.py"
    "opt9"  = "run_opt9.py";   "opt10" = "run_opt10.py"; "opt11" = "run_opt11.py"
    "opt12" = "run_opt12.py";  "opt13" = "run_opt13.py"
}

$NameMap = @{
    "base"="Baseline Original"; "opt1"="AMD ROCm"; "opt2"="MLA"; "opt3"="MoE"; "opt4"="GRPO"
    "opt5"="RAG Temporal"; "opt6"="Mamba SSM"; "opt7"="MTP"; "opt8"="FinGPT"; "opt9"="SMETimes"
    "opt10"="Combo Eficiencia"; "opt11"="Combo Preditor"; "opt12"="Ultimate"
    "opt13"="Combo SLM Contextual"
}

if (-not (Test-Path -LiteralPath "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

if (Test-Path -LiteralPath "benchmark_results.csv") { Remove-Item -LiteralPath "benchmark_results.csv" -Force }

$BaseArgs = @(
    "--task_name=long_term_forecast", "--is_training=1", "--model_id=test",
    "--model=AutoTimes_Gpt2", "--data=ETTh1", "--root_path=../dataset/ETT-small/",
    "--data_path=ETTh1.csv", "--llm_ckp_dir=gpt2", "--train_epochs=1",
    "--batch_size=32", "--seq_len=672", "--label_len=576",
    "--token_len=96", "--test_pred_len=96", "--use_amp"
)

$idx = 0
$Results = @()

foreach ($optId in $Opts) {
    $idx++
    if ($idx -lt $StartAt -or $idx -gt $EndAt) { continue }

    $Script = $ScriptMap[$optId]
    $Name = $NameMap[$optId]
    $OutputLog = "logs/${optId}_output.txt"
    $ErrorLog = "logs/${optId}_errors.txt"

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  [$idx/14] $optId - $Name" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $PyArgs = @($Script) + $BaseArgs + @("--opt_id=$optId", "--opt_name=$Name", "--smoke_test=1")
    if ($idx -eq 1) { $PyArgs += @("--reset_csv") }

    $StartTime = Get-Date
    try {
        & $CondaPy $PyArgs 2>&1 | Tee-Object -FilePath $OutputLog
        $exitCode = $LASTEXITCODE
        $Elapsed = (Get-Date) - $StartTime
        if ($exitCode -eq 0) {
            Write-Host "  [PASS] $($Elapsed.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
            $Results += [PSCustomObject]@{ Opt = $optId; Name = $Name; Status = "PASS"; Time_s = [math]::Round($Elapsed.TotalSeconds, 1) }
            Get-Content -LiteralPath $OutputLog -Tail 3 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        } else {
            Write-Host "  [FAIL] Exit code $exitCode" -ForegroundColor Red
            $Results += [PSCustomObject]@{ Opt = $optId; Name = $Name; Status = "FAIL"; Time_s = [math]::Round($Elapsed.TotalSeconds, 1) }
            Write-Host "  --- LOG (completo) ---" -ForegroundColor Red
            Get-Content -LiteralPath $OutputLog | ForEach-Object { Write-Host "  | $_" -ForegroundColor Red }
        }
    } catch {
        Write-Host "  [EXCEPTION] $_" -ForegroundColor Red
        $Results += [PSCustomObject]@{ Opt = $optId; Name = $Name; Status = "EXCEPTION"; Time_s = 0 }
    }

    # GPU memory cleanup
    & $CondaPy -c "import torch,gc; torch.cuda.empty_cache(); gc.collect()" 2>$null
    Start-Sleep -Seconds 1
}

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  SMOKE TEST RESULTS" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
$Results | Format-Table -AutoSize

$failCount = ($Results | Where-Object { $_.Status -ne "PASS" }).Count
if ($failCount -eq 0) {
    Write-Host "`n  *** ALL 14 PASSED - GREEN LIGHT ***" -ForegroundColor Green
} else {
    Write-Host "`n  *** $failCount FAILURES DETECTED ***" -ForegroundColor Red
}

if (Test-Path -LiteralPath "benchmark_results.csv") {
    Write-Host "`nbenchmark_results.csv:" -ForegroundColor Blue
    Import-Csv -Path "benchmark_results.csv" | Select-Object run_uuid, opt_id, opt_name, total_time_s, peak_vram_gb, test_mse, test_mae | Format-Table -AutoSize
}