#!/usr/bin/env bash
# ============================================================
# run_benchmarks.sh — AutoTimes Benchmark Runner (NVIDIA CUDA)
# Uso no Kaggle: bash run_benchmarks.sh
# ============================================================
set -euo pipefail

# ============================================================
# CONFIG — Edite aqui antes de rodar
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Dataset
DATA="ETTh1"
DATA_PATH="ETTh1.csv"
ROOT_PATH="./dataset/ETT-small"

# Modelo (nome HuggingFace ou path local)
LLM_DIR="gpt2"

# Hiperparâmetros
EPOCHS=10
BATCH_SIZE=256
SEQ_LEN=672
LABEL_LEN=576
TOKEN_LEN=96
TEST_PRED_LEN=96

# Smoke test (1 batch/época, desligado por padrão)
SMOKE_TEST=0

# Quais otimizações rodar
OPTS=(
    "base"
    "opt4"  "opt5"  "opt6"  "opt7"
    "opt8"  "opt9"  "opt10" "opt11"
    "opt12" "opt13"
)

# ============================================================
# Mapas
# ============================================================
declare -A OPT_MAP
OPT_MAP["base"]="run_base.py"
OPT_MAP["opt1"]="run_opt1.py";  OPT_MAP["opt2"]="run_opt2.py"
OPT_MAP["opt3"]="run_opt3.py";  OPT_MAP["opt4"]="run_opt4.py"
OPT_MAP["opt5"]="run_opt5.py";  OPT_MAP["opt6"]="run_opt6.py"
OPT_MAP["opt7"]="run_opt7.py";  OPT_MAP["opt8"]="run_opt8.py"
OPT_MAP["opt9"]="run_opt9.py";  OPT_MAP["opt10"]="run_opt10.py"
OPT_MAP["opt11"]="run_opt11.py"; OPT_MAP["opt12"]="run_opt12.py"
OPT_MAP["opt13"]="run_opt13.py"

declare -A OPT_NAMES
OPT_NAMES["base"]="Base_Original"
OPT_NAMES["opt1"]="Opt_1_ROCm";        OPT_NAMES["opt2"]="Opt_2_MLA"
OPT_NAMES["opt3"]="Opt_3_MoE";         OPT_NAMES["opt4"]="Opt_4_GRPO"
OPT_NAMES["opt5"]="Opt_5_RAG_Temporal"; OPT_NAMES["opt6"]="Opt_6_Mamba_SSM"
OPT_NAMES["opt7"]="Opt_7_MTP";         OPT_NAMES["opt8"]="Opt_8_FinGPT"
OPT_NAMES["opt9"]="Opt_9_SMETimes"
OPT_NAMES["opt10"]="Opt_10_Combo_Eficiencia"
OPT_NAMES["opt11"]="Opt_11_Combo_Preditor"
OPT_NAMES["opt12"]="Opt_12_Ultimate"
OPT_NAMES["opt13"]="Opt_13_Combo_SLM_Contextual"

# ============================================================
# Funções
# ============================================================
cleanup_gpu() {
    python -c "
import torch, gc
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()
" && sleep 2
}

run_optimization() {
    local opt_id="$1"
    local script="$2"
    local name="${OPT_NAMES[$opt_id]}"

    echo ""; echo "========================================"
    echo "  $opt_id - $name"
    echo "========================================"; echo ""

    local log_dir="logs"
    mkdir -p "$log_dir"
    local output_log="${log_dir}/${opt_id}_output.txt"

    local py_args=("$script"
        "--task_name=long_term_forecast" "--is_training=1"
        "--model_id=test" "--model=AutoTimes_Gpt2"
        "--data=${DATA}" "--root_path=${ROOT_PATH}"
        "--data_path=${DATA_PATH}" "--llm_ckp_dir=${LLM_DIR}"
        "--batch_size=${BATCH_SIZE}" "--seq_len=${SEQ_LEN}"
        "--label_len=${LABEL_LEN}" "--token_len=${TOKEN_LEN}"
        "--test_pred_len=${TEST_PRED_LEN}" "--use_amp"
        "--opt_id=${opt_id}" "--opt_name=${name}"
    )

    if [ "$SMOKE_TEST" = "1" ]; then
        py_args+=("--train_epochs=1" "--smoke_test=1")
    else
        py_args+=("--train_epochs=${EPOCHS}")
    fi

    if [ "${FIRST_OPT:-0}" = "1" ]; then
        py_args+=("--reset_csv")
    fi

    local start_time; start_time=$(date +%s)
    echo "[EXEC] python ${py_args[*]}"
    python "${py_args[@]}" 2>&1 | tee "$output_log"
    local exit_code=${PIPESTATUS[0]}
    local end_time; end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [ "$exit_code" -eq 0 ]; then
        echo ""; echo "[OK] $opt_id em $((elapsed / 60)) min $((elapsed % 60)) s"
        tail -n 5 "$output_log" | sed 's/^/  /'
    else
        echo ""; echo "[FAIL] $opt_id — exit $exit_code (${elapsed}s)"
        echo "  --- LOG ---"; cat "$output_log" | sed 's/^/  /'
        echo "  --- FIM ---"
    fi

    cleanup_gpu
    return "$exit_code"
}

# ============================================================
# Diagnóstico GPU
# ============================================================
echo "========================================"
echo "  AutoTimes — Benchmark Runner (Kaggle)"
echo "  Backbone: GPT-2 (124M)"
echo "  Data: $DATA | Epochs: $EPOCHS | Batch: $BATCH_SIZE"
echo "  Otimizações: ${#OPTS[@]}"
echo "========================================"; echo ""

echo "Diagnóstico GPU:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  Modelo: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
echo ""

# ============================================================
# Execução
# ============================================================
FIRST_OPT=1; total=${#OPTS[@]}; idx=0

for opt_id in "${OPTS[@]}"; do
    idx=$((idx + 1))
    if [ -z "${OPT_MAP[$opt_id]:-}" ]; then
        echo "[WARN] '$opt_id' desconhecida"; continue
    fi
    if [ ! -f "${OPT_MAP[$opt_id]}" ]; then
        echo "[WARN] ${OPT_MAP[$opt_id]} não encontrado"; continue
    fi
    echo "[$idx/$total]"
    run_optimization "$opt_id" "${OPT_MAP[$opt_id]}"
    FIRST_OPT=0
done

# ============================================================
# Resumo
# ============================================================
echo ""; echo "========================================"
echo "  Benchmarking concluído!"
echo "  Resultados: benchmark_results.csv"
echo "  Logs: logs/"
echo "========================================"; echo ""

if [ -f "benchmark_results.csv" ]; then
    echo "Resumo:"; echo ""
    python -c "
import csv
with open('benchmark_results.csv') as f:
    rows = list(csv.DictReader(f))
    if not rows: exit()
    print(f"{'opt_id':<10} {'opt_name':<28} {'time_s':<10} {'mse':<12} {'mae':<12} {'vram':<8}")
    print('-' * 80)
    for r in rows:
        print(f'{r[\"opt_id\"]:<10} {r[\"opt_name\"]:<28} {r[\"total_time_s\"]:<10} {r[\"test_mse\"]:<12} {r[\"test_mae\"]:<12} {r[\"peak_vram_gb\"]:<8}')
"
fi
