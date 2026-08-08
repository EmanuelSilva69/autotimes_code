# AutoTimes – Test Suite de 15 Otimizações

Este diretório contém uma cópia isolada do **AutoTimes** (base GPT-2, 768 dim) com **15 variações arquiteturais, de treino e híbridas**, prontas para benchmark automatizado.

---

## Estrutura

```
teste/
├── models/               # Modelos base (opt1–opt9)
├── data_provider/        # Data loaders (copiado do original)
├── exp/                  # Experiment runners
├── layers/               # MLP, atenção, etc.
├── utils/                # Métricas, tools, early stopping
├── benchmark_logger.py   # Logging CSV
├── run_benchmarks.ps1    # Script de benchmark (ROCm + Docker)
├── run_opt{1..15}.py     # Cada otimização é um executável independente
└── README_teste.md
```

---

## As 16 Configurações (1 Baseline + 15 Otimizações)

### Baseline

| Opt | Nome | Descrição |
|-----|------|-----------|
| base | **Original** | Código original do AutoTimes, sem modificações |

### Individuais (opt1–opt9)

| Opt | Nome | Descrição |
|-----|------|-----------|
| 1 | **AMD ROCm** | TF32, `torch.compile`, autocast, `non_blocking`, in-place norm |
| 2 | **MLA** | Multi-head Latent Attention com RoPE + compressão KV (384→768) |
| 3 | **MoE** | Mixture-of-Experts com router top-2 + expert compartilhada |
| 4 | **GRPO** | Perda contrastiva em grupo (substitui MSE padrão) |
| 5 | **RAG Temporal** | Memory bank com retrieval por cosseno para contexto histórico |
| 6 | **Mamba SSM** | Bloco State Space Model substituindo 2 camadas GPT-2 |
| 7 | **MTP** | Decodificadores paralelos para horizons [1,2,4] |
| 8 | **FinGPT** | Adaptador cross-attention tempo→sentimento financeiro |
| 9 | **SMETimes** | Fusão de features estatísticas (média, var, slope, range) com gate |

### Combos (opt10–opt15)

| Opt | Nome | Combinação |
|-----|------|------------|
| 10 | **Combo Eficiência** | AMD + MLA + Mamba (mínimo custo) |
| 11 | **Combo Preditor** | RAG + MoE + GRPO (máxima acurácia) |
| 12 | **Ultimate** | Todas as 9 otimizações simultâneas |
| 13 | **Combo SLM Contextual** | SMETimes + RAG + FinGPT |
| 14 | **Combo Perda Adaptativa** | GRPO + MTP + MoE |
| 15 | **Combo Atenção Pura** | MLA + Mamba + MTP |

---

## Como Usar

### Benchmark completo (Windows nativo)

```powershell
cd teste
.\run_benchmarks.ps1 -Epochs 10 -BatchSize 32
```

### Benchmark específico

```powershell
.\run_benchmarks.ps1 -Opts @("opt1","opt10","opt12") -Epochs 5 -Quick
```

### Benchmark no Docker (ROCm/WSL2)

```powershell
.\run_benchmarks.ps1 -DockerContainer "autotime" -Epochs 10
```

### Executar um único opt manualmente

```powershell
cd teste
python run_opt10.py --data ETTm1 --train_epochs 10 --batch_size 32
```

---

## Logging

Resultados em `benchmark_results.csv`:

| Coluna | Descrição |
|--------|-----------|
| opt_id | Identificador (opt1–opt15) |
| opt_name | Nome legível |
| timestamp | Data/hora |
| dataset | Ex.: ETTm1 |
| base_model | AutoTimes_Gpt2 |
| total_time_s | Tempo total (treino + teste) |
| avg_latency_s | Latência média por batch no teste |
| peak_vram_gb | Pico de VRAM |
| test_mse/mae/rmse | Métricas de forecast |

---

## Requisitos

- Python 3.9+
- PyTorch 2.x (ROCm ou CUDA)
- Transformers (GPT-2)
- GPU com ≥8 GB VRAM

---

## Notas

- GPT-2 foi escolhido sobre Llama por ser SLM (menos VRAM, mais rápido).
- Todos os modelos herdam `GPT2Model` do HuggingFace.
- ROCm via WSL2/Docker: usar `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- Para CPU: o benchmark usa o device disponível; performance será limitada.
