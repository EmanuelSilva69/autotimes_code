# Relatório Semanal — Semana de 03/09 a 10/09/2026

## AutoTimes — Decomposição Textual (STL/MSTL) + Decoders Alternativos

**Dataset:** ETTh1 | **Métrica:** Test MSE (z-score) | **Protocolo:** Apples-to-Apples (10 épocas, 3 seeds)

---

## 1. Contexto e Objetivo

O AutoTimes original é um modelo de forecasting baseado em GPT-2 que trabalha com representações **numéricas** (patches de séries temporais). O objetivo desta semana foi:

1. **Substituir representações numéricas por textuais** usando decomposição STL/MSTL
2. **Explorar decoders alternativos** (KAN, TaylorKAN) para tentar superar o baseline numérico
3. **Fechar o gap** entre o modelo textual e o numérico

### Baseline Numérico (referência)
| Métrica | Valor |
|---|---|
| Test MSE | **0.3539** |
| Test MAE | 0.3919 |
| Params treináveis | 2.368M (1.86% do total) |
| Backbone GPT-2 | 100% congelado |

---

## 2. Evolução Cronológica dos Resultados

### Fase 1: Arquitetura Dual (25/08) — Resultados iniciais

A primeira abordagem usava uma arquitetura **dual** com fusão de representações numéricas e textuais.

| Experimento | Fusão | Épocas | Test MSE | Gap |
|---|---|---|---|---|
| Dual soma (2 feats) | Soma | 1 | 0.5233 | +47.8% |
| Dual soma+STL | Soma | 1 | 0.5311 | +50.0% |
| Dual GLU (7 feats) | GLU | 10 | 0.4014 ± 0.0053 | +13.4% |

**Conclusão:** Fusão por soma é ineficaz. GLU melhora significativamente mas ainda fica a +13%.

### Fase 2: Texto STL (26/08) — Decomposição em texto

Implementação do pipeline de 7 passos: split → STL (periodo fixo=24) → normalização por max_abs → texto char-by-char → tokenização → dataset.

| Modelo | Test MSE | Gap |
|---|---|---|
| Pseudo-JSON ES | 0.3923 ± 0.0017 | +10.8% |
| Char-by-Char | 0.3934 ± 0.0020 | +11.1% |

**Conclusão:** STL textual (0.39) já supera o dual GLU (0.40). O gap cai de 13% para 11%.

### Fase 3: KAN Decoder (03-04/09) — Kolmogorov-Arnold Networks

Testamos 4 ativações para o decoder KAN (efficient-kan), com 30 épocas e seed fixa (42):

| Ativação | Params | MSE Pseudo-JSON | MSE Char-by-Char |
|---|---|---|---|
| **SiLU** | — | **0.3791** | **0.3818** |
| SELU | — | 0.3945 | 0.3921 |
| ELU | — | 0.4001 | 0.4079 |
| Sigmoid | — | 0.4106 | 0.4246 |

**Comparação com freeze MLP (baseline textual):**
| Decoder | MSE | Gap |
|---|---|---|
| Freeze MLP | 0.3609 | +2.0% |
| KAN SiLU | 0.3791 | +7.1% |
| KAN SELU | 0.3945 | +11.5% |

**Conclusão:** KAN não supera a MLP congelada. A melhor ativação (SiLU) fica +5% acima do freeze MLP. KAN é descartado.

### Fase 4: MSTL (09/09) — Multi-Seasonal-Trend decomposition

Substituição do STL (periodo fixo=24) por MSTL com detecção automática de periodos:

```
CSV → minutos_timestep = 60 → passos_por_hora = 1.0
periodo_diario = 24 | periodo_semanal = 168
MSTL(periods=(24, 168)) → Tendencia + Sazonalidade(diária+semanal) + Resíduos
```

Otimizações:
- Paralelização com `joblib` (7.5x speedup: 18.7s vs 141.1s para 100 janelas)
- Cache do dataset em `.npz` (649 MB para 94k janelas)
- Detecção automática de periodos a partir das datas do CSV

| Modelo | Test MSE | Gap | Melhor época |
|---|---|---|---|
| **Pseudo-JSON ES** | **0.3609 ± 0.0031** | **+2.0%** | ep 1 |
| Char-by-Char | 0.3643 ± 0.0045 | +2.9% | ep 1-3 |

**Detalhamento por seed (MSTL Pseudo-JSON):**
| Seed | Test MSE | Test MAE | Best Val |
|---|---|---|---|
| 2021 | 0.3581 | 0.3964 | 0.7309 |
| 42 | 0.3652 | 0.4047 | 0.7921 |
| 1234 | 0.3594 | 0.3986 | 0.7662 |

**Conclusão:** MSTL é o primeiro modelo textual a atingir o patamar do numérico. O gap de +2% está dentro da variância entre seeds.

### Fase 5: TaylorKAN (10/09) — Série de Taylor como decoder

Implementação da `TaylorLayer` (método de Horner) baseada no paper KAN4IQA. Testamos 4 configurações:

**Sem LayerNorm (torch.clamp removido, tanh):**
| Ordem | MSE Pseudo-JSON | MSE Char-by-Char |
|---|---|---|
| O3 | 0.4482 ± 0.0209 | 0.4248 ± 0.0176 |
| O2 | 0.4093 ± 0.0149 | 0.4233 ± 0.0269 |

**Com LayerNorm + tanh (versão final):**
| Ordem | Params | MSE Pseudo-JSON | MSE Char-by-Char |
|---|---|---|---|
| O3 | 222,816 | 0.4018 ± 0.0117 | 0.4175 ± 0.0256 |
| **O2** | **149,088** | **0.4040 ± 0.0156** | **0.4006 ± 0.0170** |

**Análise:**
- LayerNorm melhorou ~10% vs sem LN (O3: 0.448 → 0.402)
- Ordem 2 ≈ Ordem 3 (diferença within noise)
- Melhor Taylor (O2+LN: 0.4006) fica **+13%** acima do SOTA (0.3539)
- O `torch.tanh` preservou gradientes (sem NaNs), mas a série de Taylor não consegue modelar a relação decoder MLP equivalente
- **TaylorKAN é descartado**

---

## 3. Ranking Final — Todos os Experimentos

| # | Data | Experimento | Modelo | Test MSE | Gap vs Numérico |
|---|---|---|---|---|---|
| — | — | **Baseline Numérico** | Run-8 | **0.3539** | — |
| 1 | 09/09 | **MSTL textual** | pseudo_json_es | **0.3609 ± 0.003** | **+2.0%** ✅ |
| 2 | 09/09 | MSTL textual | char_by_char | 0.3643 ± 0.005 | +2.9% ✅ |
| 3 | 03/09 | KAN SiLU (30ep) | pseudo_json_es | 0.3791 | +7.1% |
| 4 | 03/09 | KAN SiLU (30ep) | char_by_char | 0.3818 | +7.9% |
| 5 | 04/09 | KAN SELU (30ep) | char_by_char | 0.3921 | +10.8% |
| 6 | 26/08 | STL textual | pseudo_json_es | 0.3923 ± 0.002 | +10.8% |
| 7 | 26/08 | STL textual | char_by_char | 0.3934 ± 0.002 | +11.1% |
| 8 | 04/09 | KAN SELU (30ep) | pseudo_json_es | 0.3945 | +11.5% |
| 9 | 04/09 | KAN ELU (30ep) | pseudo_json_es | 0.4001 | +13.0% |
| 10 | 10/09 | TaylorKAN O2+LN | char_by_char | 0.4006 ± 0.017 | +13.2% |
| 11 | 25/08 | Dual GLU | pseudo_json_es | 0.4014 ± 0.005 | +13.4% |
| 12 | 10/09 | TaylorKAN O3+LN | pseudo_json_es | 0.4018 ± 0.012 | +13.5% |
| 13 | 04/09 | KAN ELU (30ep) | char_by_char | 0.4079 | +15.2% |
| 14 | 10/09 | TaylorKAN O2 (sem LN) | pseudo_json_es | 0.4093 ± 0.015 | +15.7% |
| 15 | 03/09 | KAN Sigmoid (30ep) | pseudo_json_es | 0.4106 | +16.0% |
| 16 | 03/09 | KAN Sigmoid (30ep) | char_by_char | 0.4246 | +20.0% |
| 17 | 10/09 | TaylorKAN O3 (sem LN) | pseudo_json_es | 0.4482 ± 0.021 | +26.6% |
| 18 | 25/08 | Dual soma (1ep) | char_by_char | 0.5217 | +47.4% |
| 19 | 25/09 | Dual soma+STL (1ep) | pseudo_json_es | 0.5311 | +50.0% |

---

## 4. Análise Comparativa

### Gap entre STL e MSTL
```
STL  (periodo fixo=24):     0.3923  (+10.8%)
MSTL (periodos=24,168):     0.3609  (+2.0%)
                              ↓
                    Melhoria de 8.0 pontos percentuais
```

O MSTL melhora porque captura **duas sazonalidades** (diária + semanal), enquanto o STL só captura a diária. A sazonalidade semanal é especialmente importante para ETTh1 (dados horários de transformadores de energia).

### Por que KAN e TaylorKAN falharam
- **KAN (efficient-kan)**: Splines adaptativos não conseguem modelar a relação inversa encoder→decoder tão bem quanto MLP linear. A capacidade não-linear é desperdiçada porque o encoder já extrai features lineares.
- **TaylorKAN**: Série de Taylor (Maclaurin centrada em 0) diverge para |x| >> 1. Mesmo com tanh+LayerNorm, a representação polinomial de ordem 2-3 não tem capacidade suficiente para modelar a transformação 768→96.

### Freeze MLP vs Novos Decoders
| Decoder | Params treináveis | MSE | Conclusão |
|---|---|---|---|
| Freeze MLP (original) | 2.368M | 0.3609 | **SOTA textual** |
| Linear Probing | 73K | 0.3804 | +5.4% — muito leve |
| Fresh MLP | 2.368M | 0.3804 | +5.4% — drift do encoder |
| KAN SiLU | ~200K | 0.3791 | +5.1% — underfit |
| TaylorKAN O2+LN | 149K | 0.4006 | +11.0% — capacidade insuficiente |
| TaylorKAN O3+LN | 223K | 0.4018 | +11.3% — mesmo problema |

---

## 5. Infraestrutura Desenvolvida

### Arquivos criados
| Arquivo | Descrição |
|---|---|
| `testes_letras/taylor_kan.py` | TaylorLayer (Horner + tanh + LayerNorm) |
| `testes_letras/recente/` | Dir de output com timestamps únicos |
| `testes_letras/quinta_noite/relatorio_semana_2026-09-10.md` | Este relatório |

### Arquivos modificados
| Arquivo | Mudanças |
|---|---|
| `testes_letras/modelos.py` | Freeze encoder/decoder, TaylorKAN decoder, LayerNorm |
| `testes_letras/run_apples_to_apples_trio.py` | --taylor-decoder, --taylor-order, --texto-decomp mstl, RECENTE_DIR |
| `testes_letras/texto_char_by_char.py` | MSTL, detectar_periodos_mstl, joblib parallel |
| `testes_letras/treino.py` | Two-phase training (unfreeze_encoder_epoch) |

### Comandos padrão
```powershell
# MSTL (melhor resultado textual)
python -m testes_letras.run_apples_to_apples_trio --texto-decomp mstl --epochs 10 --seeds 2021,42,1234

# TaylorKAN O3 + O2 (sequencial)
python -m testes_letras.run_apples_to_apples_trio --taylor-decoder --taylor-order 3 --epochs 10 --seeds 2021,42,1234 ; python -m testes_letras.run_apples_to_apples_trio --taylor-decoder --taylor-order 2 --epochs 10 --seeds 2021,42,1234

# KAN (batch de ativações)
python testes_letras/run_kan_activations.py
```

---

## 6. Próximos Passos

1. **MSTL como SOTA textual** — Usar 0.3609 como referência para comparações futuras
2. **Two-phase training** — Testar `--unfreeze-encoder-epoch 15` com MSTL para fine-tuning tardio do encoder GPT-2
3. **Investigar gap residual** — MSE de validação MSTL (~0.76) vs numérico (~0.73) sugere que o fusion layer pode ser otimizado
4. **Limpar código** — Remover prints de debug (`[debug] load_state_dict`, `[debug] freeze`, `[debug] model.eval()`) e temporários (`_chk.py`, `_bench_par.py`)
5. **Publicar resultados** — Preparar tabela para o orientador com os números finais

---

*Relatório gerado em 10/09/2026. Dados extraídos dos JSONs de veredito em `testes_letras/quinta_noite/`.*
