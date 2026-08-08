# Otimizações Aplicadas — AutoTimes

## Visão Geral

Foram implementadas 9 otimizações individuais (opt1–opt9) sobre o baseline AutoTimes com backbone GPT-2, e 6 combinações (opt10–opt15). Cada otimização é independente e acumulativa.

---

## Otimizações Individuais

### opt1 — AMD ROCm (Performance)
**Arquivo:** `run_opt1.py` (usa `models/autotimes_opt1.py`)
**Tipo:** Somente runtime, sem alteração arquitetural

- `torch.set_float32_matmul_precision('high')` — ativa TF32 em GPUs compatíveis
- `torch.backends.cudnn.benchmark = True` — otimiza kernels cuDNN
- `torch.float16` via `use_amp` — mixed precision training
- `low_cpu_mem_usage=True` no carregamento do modelo
- `device_map` para alocação otimizada na GPU

---

### opt2 — Multi-head Latent Attention (MLA)
**Arquivo:** `run_opt2.py` (usa `models/autotimes_opt2.py`)

Substitui a self-attention original do GPT-2 por MLA com compressão latente de KV:
- Projeta key/value para um espaço latente menor (`hidden_dim // 2`)
- Expande de volta para o espaço original antes da atenção
- Adiciona Rotary Position Embedding (RoPE)
- Reduz memória de KV cache em ~50%

---

### opt3 — Mixture of Experts (MoE)
**Arquivo:** `run_opt3.py` (usa `models/autotimes_opt3.py`)

Wrappeia o tokenizer/detokenizer MLP com uma camada MoE:
- 8 experts com top-2 routing
- 1 expert compartilhado (sempre ativo)
- Ruído durante treino para balanceamento de carga
- Clamping de logits e proteção contra NaN

---

### opt4 — Group Relative Policy Optimization (GRPO)
**Arquivo:** `run_opt4.py` (usa `models/autotimes_opt4.py`)
**Tipo:** Somente loss function

Substitui MSE padrão por loss contrastiva por grupos:
- `group_size` forward passes por batch
- Vantagem contrastiva: `(mean_loss - loss) / (std + eps)`
- Pesos da loss: `(1 - alpha * advantage.detach())`
- Incentiva consistência entre predições do mesmo grupo

---

### opt5 — RAG Temporal
**Arquivo:** `run_opt5.py` (usa `models/autotimes_opt5.py`)

Banco de memória que armazena embeddings históricos (key-value):
- Até `max_memory_size=1000` entradas
- Retrieval por similaridade de cosseno (top-3)
- Condiciona a predição atual com contextos passados similares
- Evicção FIFO quando a memória atinge o limite

---

### opt6 — Mamba State Space Model (SSM)
**Arquivo:** `run_opt6.py` (usa `models/autotimes_opt6.py`)

Substitui 2 camadas de atenção do GPT-2 por blocos Mamba SSM:
- Convolução 1D depthwise com SiLU
- Discretização ODE-style com atualização de estado por timestep
- Projeções `A_log`, `dt_proj`, `B/C` para o seletivo SSM
- Conexão residual com gating

---

### opt7 — Multi-Task Prediction (MTP)
**Arquivo:** `run_opt7.py` (usa `models/autotimes_opt7.py`)

Decodificador paralelo para múltiplos horizontes:
- Cabeças para horizons [1, 2, 4]
- Cada cabeça prediz `h * token_len` passos
- Loss combinada: MSE principal + losses auxiliares ponderadas
- Shared encoder com múltiplos decodificadores

---

### opt8 — FinGPT (Cross-Attention com Sentimento)
**Arquivo:** `run_opt8.py` (usa `models/autotimes_opt8.py`)

Adiciona atenção cruzada entre embeddings temporais e features de mercado:
- Projeta marcas temporais (timestamp features) para o espaço do modelo
- Cross-attention `MultiheadAttention` entre hidden states e marcas projetadas
- Residual + LayerNorm
- Simula incorporação de sentimento financeiro

---

### opt9 — SMETimes (Fusão de Features Estatísticas)
**Arquivo:** `run_opt9.py` (usa `models/autotimes_opt9.py`)

Extrai 4 features estatísticas por token chunk:
- Média, variância, tendência (slope por mínimos quadrados), range (max-min)
- Projeta features para o espaço do modelo via `AdaptiveFusionGate`
- Gate aprendido: `sigmoid(linear(cat(times_embeds, stats_proj)))`
- Mistura adaptativa entre embeddings originais e estatísticos

---

## Combinações

### opt10 — Combo Eficiência
`MLA + Mamba + ROCm (opt2 + opt6 + opt1)`

Foco em mínimo custo computacional. Combina atenção latente com SSM para reduzir custo por token.

### opt11 — Combo Preditor
`MoE + RAG + GRPO (opt3 + opt5 + opt4)`

Foco em máxima acurácia. Combina expert routing, retrieval de contexto e loss contrastiva.

### opt12 — Ultimate
`Todas as 9 otimizações simultâneas`

Maior e mais complexo. Inclui MLA, MoE, Mamba, RAG, MTP, FinGPT, SMETimes, GRPO e ROCm.

### opt13 — Combo SLM Contextual
`SMETimes + RAG + FinGPT (opt9 + opt5 + opt8)`

Foco em contexto rico: features estatísticas fundidas com retrieval de memória e atenção cruzada.


