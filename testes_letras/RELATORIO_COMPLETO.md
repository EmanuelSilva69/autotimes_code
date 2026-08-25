# Relatório Completo — AutoTimes Multimodal: GPT-2 em Palavras (ETTh1)

> **Este é o documento-mestre.** Ele explica o que foi feito, onde cada artefato está e
> como reproduzir tudo. Detalhes profundos ficam nos relatórios dedicados:
> - `comparacao/relatorio_jornada_oficial.md` — metodologia e resultados (base do artigo);
> - `comparacao/relatorio_tecnico.md` — números, tabelas e figuras IEEE;
> - `comparacao/relatorio_auditoria_paridade.md` — prova anti-alucinação da paridade com o baseline.

---

## 1. O que é o projeto

O **AutoTimes** (autorregressivo, um único backbone Transformer para forecasting
multivariado) usa `PatchTST` embeddings e **RevIN** — avaliação canônica em **z-score**.
Neste trabalho estendemos o pipeline para aceitar **entrada textual**: cada ponto da série
vira um par (tempo, valor) **por extenso** em PT/ES/EN, tokenizado pelo **GPT-2 local
(`gpt2/`) e congelado**, paginado em 7 blocos × 128 tokens e fundido ao ramo numérico.

**Pergunta central:** um LLM lendo apenas palavras consegue preservar a capacidade
preditiva do pipeline numérico puro? E o idioma/forma do prompt importa?

**Hardware/ambiente:** AMD RX 9060 XT 17 GB (ROCm), conda `autotimes`,
`C:\Users\Emanuel\miniconda3\envs\autotimes\python.exe`.

---

## 2. Linha do tempo do trabalho

### Fase 1 — Diagnóstico inicial (por que 5,80 ≠ 0,3539?)
O harness de exploração media o Run-8 na escala **bruta** da HUFL. Auditoria
(`auditoria_harness.py`) encontrou 3 raízes:

1. **Escala:** baseline oficial é avaliado em **z-score** (MSE homogêneo de grau 2 →
   `MSE_z = MSE_raw / σ²_treino`, σ²(HUFL) = 33,7881; razão medida 33,7886, erro < 0,002%).
2. **Vazamento temporal:** pool de 192 janelas da mesma região `[0,2000)` (60/20/20),
   stride 6 → overlap entre janelas de **99,1%** e região idêntica à de treino.
3. **Volume:** 115 janelas vs 55.111 do protocolo original (**×479** menos).

Correção: `benchmark_numerico_base.py --eval-only` no protocolo original reproduziu
**0,353926 = +0,0%** vs o artigo (19.495 janelas de teste).

### Fase 2 — Exploração multilíngue (por extenso)
- Dígitos → palavras por extenso (`texto_puro/extenso.py`, num2words, 1 casa decimal,
  zero algarismos). BPE fragmenta dígitos; palavras inteiras evitam a fragmentação.
- **Páginação:** truncamento × **janela + overlap 15%** × keywords → janela vence
  (zero descarte, continuidade semântica).
- **Inflação de tokens medida** (`benchmark_etth1_extenso.json`): mesmo conteúdo, PT
  custa **×2,1** os tokens do EN, ES **×1,8–1,9** (detalhe na §4.3).

### Fase 3 — Salto estrutural (Structural Prompting)
Bateria de **21 cenários × 3 seeds × 10 épocas** (`benchmark_formatacoes_estruturadas.py`):
10 templates estruturados (pseudo-JSON, estrutural/Markdown, relacional, temporal-delta)
× 2 paginação (janela, keywords) + baseline language-tuning. Campeão do harness:
**`pseudo_json_es__janela` = 7,034 ± 0,059 raw**.

### Fase 4 — Apples-to-Apples (veredito final)
Campeão re-executado no **protocolo definitivo** (`benchmark_definitivo_textual.py`):
z-score nativo, splits canônicos `[0,8640)/[7968,11520)/[10848,14400)`, 94.101 janelas,
cache em disco (211 s), 3 seeds, FP32+TF32, 0 estouros. Paridade interna re-medida
(baseline = 0,353926) antes de cada treino textual. **Resultado:** campeão textual
**0,3669 ± 0,0069 z** → **+3,67%** vs baseline numérico.

### Fase 5 — Auditoria anti-alucinação
Paridade linha-a-linha (`relatorio_auditoria_paridade.md`) + correção do **"85,7%"**:
`576/672 = 85,71%` é `label_len/seq_len` (fração de contexto por projeto), **não** leakage.

### Fase 6 — Figuras e relatórios finais
Painel IEEE (`comparacao/`), relatório técnico, jornada oficial, auditoria, e os
recursos de apresentação (`apresentacao/`).

---

## 3. Inventário de artefatos

### 3.1 Scripts (`testes_letras/`)

| Script | Papel |
|---|---|
| `auditoria_harness.py` | Mede raízes de vazamento/escala do harness antigo |
| `benchmark_numerico_base.py` | Baseline numérico no protocolo original (eval-only Run-8) |
| `benchmark_etth1_texto.py` (via `run_etth1_texto.py`) | Primeira comparação PT × EN |
| `benchmark_formatacoes_estruturadas.py` | Bateria 21 cenários estruturados |
| `benchmark_definitivo_textual.py` | Apples-to-Apples final (3 seeds) |
| `benchmark_f3_f4_ablation.py`, `f3_f4_sinergia` | Ablação das fases 3/4 do fusor |
| `validacao_final_top3.py`, `validacao_campeao_absoluto.py` | Validações de campeões |
| `treino.py`, `modelos.py`, `dados.py`, `templates.py` | Loop de treino, arquitetura multimodal, tokenização/paginação, templates |
| `run_testes.py`, `run_texto.py`, `run_etth1_extenso.py` | Baterias históricas de teste |
| `otimizar_ingles_extenso.py`, `pipeline_otimizacao_extenso.py` | Expansão de vocabulário EN |
| `gerar_graficos_comparacao.py` | Painel IEEE (7 figuras, `comparacao/`) |
| `gerar_graficos_idiomas.py` | Análise linguística (3 figuras, `comparacao/`) |
| `gerar_graficos_convergencia.py` | Convergência com bandas (2 figuras, `apresentacao/`) |
| `gerar_graficos_slides.py` | 4 slides limpos (16:9, `apresentacao/`) |
| `gerar_graficos_inflacao.py` | Inflação ES/PT vs EN (`apresentacao/`) |
| `gerar_graficos_publicacao.py` | Figuras para publicação (IEEE) |

### 3.2 Dados de resultado (`testes_letras/resultados/*.json`)

| JSON | Conteúdo |
|---|---|
| `veredito_final_apples_to_apples.json` | **Resultado final**: baseline 0,3539 (medido 0,353926), campeão 0,366909 ± 0,006912, delta +3,67%, por seed, estouros 0 |
| `estruturas_avancadas_stats.json` | 21 cenários: médias/desvios + curvas por época |
| `numerico_base_stats.json` | Eval-only do Run-8 no test set real (0,353926) |
| `auditoria_harness_report.json` | Provas: var_treino, razão 33,7886, bounds, vazamento |
| `benchmark_etth1_extenso.json` | Tokens/ponto por idioma (PT/ES/EN) e nível (verboso/curto/keywords) |
| `benchmark_etth1_texto.json` | PT 39.262 vs EN 24.006 tokens (×1,64) |
| `resumo_tokenizacao.json`, `resumo_texto_puro.json` | Testes históricos de tokenização/inflação PT×EN |
| `top3_validacao_stats.json`, `campeao_absoluto_stats.json` | Validações de campeões |
| `benchmark_4fases_stats.json`, `f3_f4_sinergia_stats.json`, `otimizar_ingles_extenso.json`, `resumo_final.json` | Benchmarks históricos |

### 3.3 Figuras

**Painel IEEE — `comparacao/`** (via `gerar_graficos_comparacao.py` + `_idiomas.py`):

| Figura | Conteúdo |
|---|---|
| `01_veredito_barras.png` | Baseline numérico vs campeão textual (z-score) |
| `02_jornada_escala.png` | 5,80 → 7,03 → 0,208 → 0,367 (jornada de escalas) |
| `03_ranking_21.png` | Ranking raw dos 21 cenários |
| `04_convergencia_veredito.png` / `05_convergencia_21.png` | Convergência por época (veredito / 21 cenários) |
| `06_por_tipo.png` | Agregação por tipo de template |
| `07_por_seed.png` | Campeão por seed |
| `08_boxplot_idiomas.png` | Boxplot por idioma |
| `09_agregado_idioma_barras.png` | Média por idioma |
| `10_template_vs_idioma.png` | Interação template × idioma |

**Apresentação — `apresentacao/`:**

| Arquivo | Conteúdo |
|---|---|
| `slide_01_idiomas.png` | Paradoxo dos idiomas (boxplot, sem baseline numérico) |
| `slide_02_ranking.png` | Top 4 + baseline LN (raw, sem baseline numérico) |
| `slide_03_escala.png` | Linha + error bars dos 21 cenários em MSE raw |
| `slide_04_veredito.png` | Clímax: 0,3539 vs 0,3669 + "+3,67%" |
| `convergencia_21_idiomas.png` | 21 curvas com bandas (±1 dp), cores por idioma |
| `convergencia_top3.png` | Pódio de convergência (linhas grossas) |
| `inflacao_tokens.png` | Inflação de tokens ES/PT vs EN (×1,8–2,1) |

### 3.4 Relatórios (`comparacao/`)

- `relatorio_tecnico.md` — 6 seções acadêmicas, 5 tabelas com todos os números reais.
- `relatorio_jornada_oficial.md` — 4 fases + 6 ressalvas de transparência (§5).
- `relatorio_auditoria_paridade.md` — paridade A–E com citações `arquivo:linha`.

---

## 4. Números-chave

### 4.1 Veredito final (apples-to-apples, z-score nativo)

| Métrica | Baseline numérico (Run-8) | Campeão textual | Δ |
|---|---|---|---|
| **Test MSE_Z** | **0,3539** (medido 0,353926) | **0,3669 ± 0,0069** | **+3,67%** |
| Test MAE_Z | 0,3919 | 0,4026 ± 0,0045 | +2,73% |
| Equiv. raw HUFL | 11,96 | 12,40 | — |

Seeds: 2021 → 0,357134; 42 → 0,371784; 1234 → 0,371808. Estouros: 0.
**Interpretação:** GPT-2-base congelado lendo palavras preserva **96,3%** da capacidade
preditiva do ramo numérico da mesma arquitetura.

### 4.2 Ranking do harness (referência interna, escala raw)
Campeão `pseudo_json_es__janela` 7,034 ± 0,059; temporal_delta_en 7,105 ± 0,098;
estrutural_es 7,120 ± 0,075; baseline LN `es_keywords` 7,245. Harness = ranking
interno **não publicável** (região compartilhada, ×479 volume, escala raw).

### 4.3 Inflação de tokens (mesmo conteúdo, 2.000 pontos, GPT-2)

| Nível | ES | PT | EN (absoluto) |
|---|---|---|---|
| Verboso | ×1,82 (30,7 tok/pt) | ×2,08 (35,1) | 16,9 |
| Curto | ×1,91 (22,7) | ×2,11 (25,1) | 11,9 |
| Keywords | ×1,83 (16,7) | ×2,09 (19,1) | 9,1 |

Paradoxo: EN é o mais barato em tokens e ainda assim ES/PT vencem no ranking — a
**forma do prompt** (estrutura chave-valor, regularidade morfológica) importa mais
que a densidade.

---

## 5. Como reproduzir

```powershell
# Conda env
& "C:\Users\Emanuel\miniconda3\envs\autotimes\python.exe" -m testes_letras.<script>

# Figuras IEEE (comparacao/)
gerar_graficos_comparacao      # 01..07
gerar_graficos_idiomas         # 08..10

# Apresentação (apresentacao/, 16:9 @ 300 DPI)
gerar_graficos_slides          # slide_01..04
gerar_graficos_convergencia    # convergencia_21_idiomas, convergencia_top3
gerar_graficos_inflacao        # inflacao_tokens
```

Todos os scripts leem **somente** os JSONs de `resultados/` (imutáveis) e são
idempotentes (regeneram as figuras sem re-treinar).

---

## 6. Ressalvas científicas (resumo da §5 da jornada)

1. Re-treino textual usa **MSE puro**; baseline foi treinado com **VolLoss** (eval-only).
2. Splits canônicos com offset `−seq_len` (protocolo oficial, igual nas 2 condições).
3. "85,7%" é `label_len/seq_len`, **não** vazamento; o vazamento do harness era
   região compartilhada + 99,1% de overlap + volume ×479.
4. Contexto textual cobre ~10–13% da janela (896 tokens ≈ 65–90 de 672 pontos).
5. Escopo: 1 dataset (ETTh1), horizonte 96, 3 seeds, 1 LLM congelado.
6. ROCm (FP32+TF32); reprodução em CUDA pode variar no 5º dígito decimal.
