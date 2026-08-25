# Relatório de Auditoria de Paridade e Integridade (Anti-Alucinação)

**Escopo:** `testes_letras/benchmark_definitivo_textual.py` + dependências (`treino.py`, `modelos.py`, `benchmark_numerico_base.py`, `run_etth1_texto.py`, `auditoria_harness.py`) **vs** arquitetura original (raiz): `data_provider/data_loader.py`, `Testes_eficiencia/run_benchmarks.py` (réplica fiel da bateria), `exp/exp_long_term_forecasting.py`, `utils/metrics.py` e checkpoint Run-8 VolLoss+SMETimes.

**Método:** leitura cruzada linha a linha + verificação empírica dos JSONs de resultados (`veredito_final_apples_to_apples.json`, `numerico_base_stats.json`, `auditoria_harness_report.json`) + inspeção das flags/args gravados no checkpoint Run-8.

---

## A. Paridade de Dados — StandardScaler e Splits

### A.1 Normalização Z-Score — **PASS**

| Fonte | Comportamento |
|---|---|
| Original `Dataset_ETT_hour` (`data_provider/data_loader.py:38-53`) | `StandardScaler().fit(train_data.values)` (treino **somente**) → `transform(df_data.values)` (dataset inteiro) |
| `run_benchmarks.load_real_split` (`run_benchmarks.py:655-658`) | `scaler = StandardScaler().fit(values[b1s[0]:b2s[0]])` → `data = scaler.transform(values)` |
| Nosso definitivo (`benchmark_definitivo_textual.py:180-181`) | `scaler = StandardScaler().fit(values[b1s[0]: b2s[0]])` → `values_z = scaler.transform(values)` |
| Nosso protocolo original (`benchmark_numerico_base.py:160`) | Idêntico (mesma fórmula) |

Idênticos: média/desvio do treino `[0, 8640)`, transformação aplicada a **todo** o dataset (17.420 linhas), valores salvos em `float32`. O `StandardScaler` do sklearn usa `ddof=0` em ambos. **Empírico:** `auditoria_harness_report.json` → `var_treino.HUFL = 33,7881`; razão medida `mse_raw/mse_z = 33,7886` no mesmo checkpoint (erro < 0,002%).

### A.2 Limites dos Splits — **PASS** (com ressalva de terminologia)

`_compute_split_bounds` (`run_benchmarks.py:526-540`) para ETTh1 (month = 30×24):

```
b1s = [0, 12·720−672, 16·720−672] = [0, 7968, 10848]
b2s = [12·720, 16·720, 20·720]    = [8640, 11520, 14400]
```

Os três scripts (original, `carregar_protocolo_original`, `carregar_dados_z`) usam **exatamente** esses limites — confirmado também no JSON de auditoria (`bounds: {train: [0,8640), val: [7968,11520), test: [10848,14400)}`).

**Ressalva de precisão científica:** o deslocamento `−seq_len` (protocolo canônico do Time-Series-Library, herdado pelo AutoTimes original) faz com que as janelas de validação **comecem** em 7968 — dentro da região temporal de treino `[0,8640)` — e as de teste comecem em 10848 — dentro da região de validação. Isto **não é vazamento adicional**: é o protocolo oficial que produziu o 0,3539, aplicado **identicamente** às duas condições (numérica e textual). No artigo, deve-se dizer "splits cronológicos estritos no protocolo canônico" e não "zero sobreposição absoluta".

Janelas: 55.111 / 19.495 / 19.495 (stride=1), construção `x=[start, start+672)`, `y=[start+96, start+768)` com `r_begin = s_end − 576` — **idêntica** em `run_benchmarks.py:660-673` e `benchmark_definitivo_textual.py:301-309`.

**Prova empírica final:** `numerico_base_stats.json` → eval-only do Run-8 no test set de 19.495 janelas: **MSE_Z = 0,353926** (MAE 0,39191) = **+0,0%** vs artigo (0,3539). O harness está em paridade total com o protocolo do ablation.

---

## B. Integridade da Arquitetura — Projeção de Erro na Fase C

### B.1 Dimensões — **PASS**

Fluxo numérico (`run_benchmarks.py:406-449`) e multimodal (`_forward_multimodal`, `:451-520`) produzem a mesma saída: **(B, seq_len, 1) = (B, 672, 1)**, desnormalizada pelo RevIN **apenas** no ramo numérico. As métricas são calculadas sobre `pred[:, -96:, :]` vs `y[:, -96:, :]` (`treino.py:140-141`, `benchmark_definitivo_textual.py:381-382`), no mesmo espaço e dimensionalidade do protocolo `evaluate(last_only=True)` da bateria (`run_benchmarks.py:1014-1015`) e do `utils/metrics.py:34-41` (MSE/MAE médios sobre todas as entradas).

### B.2 Alinhamento token↔patch — **PASS**

Flags do checkpoint Run-8 (verificadas nos metadados do `.pt`): `use_overlap=false`, `use_vol_gate=true`, `use_robust=false`, `use_vol_loss=true`, `unfreeze_layers=0`. Consequências para o modo textual:

- `n_blocks_text = 672/96 = 7` **=** `n_patches = 7` → **nenhuma interpolação** é acionada (`modelos.py` → `run_benchmarks.py:496-500`); a fusão `cat([txt, num]) → Linear(2D→D) → LayerNorm` alinha 7×7 diretamente.
- `gate` (AdaptiveFusion/SMETimes) ativo **antes** da fusão, idêntico ao caminho numérico (`:502-503`).
- `StandardRevIN(affine=False)` = z-score da bateria.

### B.3 Ressalva (WARN) — função de perda no re-treino

O checkpoint Run-8 foi **treinado** com `VolatilityAwareLogCoshLoss` (flag `use_vol_loss=true`); nosso loop de convergência textual treina com **MSE puro** (`treino.py:192`). A comparação permanece válida porque (i) o baseline é medido **eval-only** com o checkpoint como treinado na bateria e (ii) o protocolo de avaliação é idêntico para ambas as condições. Para o artigo: declarar explicitamente a diferença de perda de treino.

---

## C. Otimizações AMD ROCm e Estabilidade Numérica — **PASS**

- `benchmark_definitivo_textual.py:452-454`: `use_amp = False` (hardcoded, comentado), `torch.backends.cudnn.benchmark = True`, `torch.set_float32_matmul_precision("high")` → **TF32 nas GEMMs do ROCm**, sem autocast nem GradScaler.
- `modelos.py:74`: `torch_dtype = float32` — GPT-2 armazenado e computado em FP32 (a bateria usava FP16 somente sob AMP; o modo textual com `fusion` aleatório estoura em NaN no backward fp16 — documentado no docstring).
- Trava de segurança **idêntica à bateria** (`treino.py:85-100` vs `run_benchmarks.py:924-949`): `clip_grad_norm(max_norm=1.0, error_if_nonfinite=False)` + verificação `isfinite(loss) and isfinite(total_norm)` → batch pulado e contabilizado em `batches_estourados_total`.
- **Empírico:** veredito final → `estouros = 0` nas 3 seeds × 10 épocas (216 batches por época no smoke; 7.860 batches/época na rodada completa).
- Probe de batch (`_batch_fit`) executa 1 passo com lr=1e-4 num modelo descartável; a cada seed o modelo é **remontado do checkpoint** (`benchmark_definitivo_textual.py:563`) → sem contaminação de estado.

---

## D. Correção de Alucinação — o "85,7% de vazamento temporal"

**VEREDITO DO AUDITOR: o número 85,7% NÃO é uma medida de vazamento. Ele é `label_len/seq_len = 576/672 = 85,71%`, a fração de contexto da janela — propriedade de projeto da arquitetura AutoTimes, não uma falha.** Citar "vazamento temporal de 85,7% (overlap de label_len)" no artigo seria um erro de reporte.

O vazamento **real** do harness antigo (quantificado em `auditoria_harness_report.json`):

| Fator | Medida real |
|---|---|
| Sobreposição de região | 100% — val/test extraídos da **mesma** faixa temporal de treino (pool `[0, 2000)` do CSV, 60/20/20 = 115/38/39) |
| Sobreposição entre janelas | stride 6 → janelas consecutivas sobrepõem **666/672 = 99,1%** dos pontos (`run_etth1_texto.py:174`) |
| Volume de treino | 115 vs 55.111 janelas (**×479** menos) |
| Escala | raw vs z-score (fator 33,7886, validado empiricamente) |

Ou seja: os 5,80 raw / 7,03 raw do harness eram comparáveis **entre si** (ranking interno), mas **não publicáveis** como baseline absoluto — e o 0,2082 z (7,03 ÷ 33,79) remove apenas o efeito de escala, **não** o vazamento e o volume. Os relatórios finais (`relatorio_tecnico.md`, `relatorio_jornada_oficial.md`) já tratam o tema com essa precisão.

---

## E. Integridade de Cache, Retomada e Rastreabilidade — **PASS**

- `_chave_cache` (`benchmark_definitivo_textual.py:203-216`): inclui CSV + `st_mtime_ns` + bounds + stride + config de paginação → cache imune a dados/formatos antigos.
- `_fingerprint` (`:429-439`): invalida seeds antigas quando qualquer parâmetro que afeta o resultado muda (inclui `lr`, `epochs`, `stride`, `n_max`, template, checkpoint, caminho do CSV).
- Veredito retomável por seed com gravação JSON após cada seed.
- **Ressalva menor:** o fingerprint **não** inclui `batch_size` (auto-ajustado por VRAM). Risco prático baixo (o probe escolhe deterministicamente o maior batch estável), mas, para rigor, o campo `batch_size` fica gravado no `config` do veredito para rastreabilidade.

---

## Verdicto Consolidado

| Item | Veredito |
|---|---|
| A.1 Z-Score (fit no treino, aplicado ao todo) | ✅ PASS (paridade linha a linha + prova empírica 0,353926 = +0,0%) |
| A.2 Splits cronológicos (b1s/b2s) | ✅ PASS (protocolo canônico idêntico nas duas condições; terminologia deve ser precisa) |
| B Fase C — projeção de erro nas mesmas dimensões | ✅ PASS (saída (B,672,1), horizonte [-96:], sem interpolação; WARN: perda MSE vs VolLoss no re-treino) |
| C ROCm — FP32+TF32, sem AMP, skip-batch | ✅ PASS (estouros = 0 medidos; trava idêntica à bateria) |
| D "85,7%" | ⚠️ **Corrigido**: é label_len/seq_len; vazamento real = mesma região + stride 6 (99,1%) + volume ×479 + escala |
| E Cache/fingerprint/retomada | ✅ PASS (ressalva: batch_size fora do fingerprint) |

**Transparência para o artigo (limitações a declarar):** (1) perda de treino do modo textual difere da bateria (MSE vs VolLoss); (2) splits canônicos com offset `−seq_len` (sobreposição marginal de região val/test com treino — protocolo oficial); (3) benchmark em um único dataset (ETTh1, horizonte 96); (4) 3 seeds apenas (dp = 0,0069, < 2% do valor); (5) números do harness antigo são referência interna, não publicáveis.