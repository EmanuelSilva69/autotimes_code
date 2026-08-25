# Relatório Oficial da Jornada — AutoTimes Multimodal (ETTh1, GPT-2 em Palavras)

> **Propósito:** base estrutural da metodologia e dos resultados do artigo científico. Toda figura citada está na pasta `comparacao/`; todos os JSONs de origem estão em `testes_letras/resultados/`. Ressalvas de transparência científica no final (§5).

---

## 1. O Diagnóstico Inicial

### 1.1 O problema

O harness de exploração textual reportava Test MSE = **5,80 (raw)** para o checkpoint numérico Run-8 VolLoss+SMETimes — enquanto o ablation original reportava **0,3539 (z-score)** para o mesmo checkpoint. A discrepância não era o texto: era o **pipeline de medição**.

### 1.2 As três raízes encontradas (`auditoria_harness.py`)

1. **Inconsistência de escala (causa principal).** O baseline oficial é avaliado em espaço z-score: o `StandardScaler` é ajustado **apenas** no split de treino `[0, 8640)` e aplicado ao dataset inteiro, e o MSE é calculado nativamente nesse espaço (`data_provider/data_loader.py:38-53`). O harness media na **escala bruta** da HUFL. Como o MSE é homogêneo de grau 2, a conversão é exata: **MSE_z = MSE_raw / σ²_treino**, com σ²(HUFL) = **33,7881** — confirmado empiricamente: o mesmo checkpoint avaliado em ambas as escalas produziu a razão **33,7886** (erro < 0,002%).

2. **Vazamento temporal no split do harness.** O pool de 192 janelas era extraído de **2000 linhas iniciais** (região de treino) e dividido em 60/20/20 (115/38/39). Ou seja: validação e teste **da mesma região temporal do treino**, com janelas consecutivas (stride 6) sobrepondo **666/672 = 99,1%** dos pontos. *Nota de precisão: o "85,7%" que circulou em nossas notas é `label_len/seq_len = 576/672` — a fração de contexto da janela, por projeto, não o vazamento. O vazamento real é a sobreposição de região descrita aqui (ver `relatorio_auditoria_paridade.md`, seção D).*

3. **Volume irrisório.** 115 janelas de treino vs 55.111 do protocolo original — **×479 menos dados**, agravando o erro estatístico de qualquer conclusão.

### 1.3 A correção

- `benchmark_numerico_base.py` passou a suportar o **protocolo original** (z-score nativo + dataset completo + splits canônicos `[0,8640)/[7968,11520)/[10848,14400)`): o modo `--eval-only` avaliou o Run-8 no test set real de 19.495 janelas e reproduziu **0,353926** — **+0,0%** vs o artigo. O harness passou a ser usado apenas como referência interna de ranking.
- Criado o fator de conversão rastreável `var_treino` por feature e o JSON `auditoria_harness_report.json` com toda a prova.
- Definido o protocolo final "apples-to-apples" (§4), eliminando as três raízes.

---

## 2. A Exploração Multilíngue

### 2.1 Dos dígitos às palavras por extenso

A serialização evoluiu em duas fases: (i) **formatos híbridos** (PT Completo / EN Curto) ainda carregavam algarismos ("Dia 10 de Julho, as 14 horas, o valor era 29.5"); (ii) **conversão 100% por extenso** (`texto_puro/extenso.py`, num2words, 1 casa decimal, **zero algarismos**) — cada ponto vira um par (tempo, valor) inteiramente lexical, exigindo que o GPT-2 faça aritmética implícita sobre palavras. A tokenização GPT-2 (BPE byte-level) fragmenta dígitos em múltiplos tokens; palavras inteiras evitam essa fragmentação e preservam a semântica do numeral composto.

### 2.2 Paginação (Fase B): três estratégias medidas

Um documento da série (2.000 pontos) gera ~24-39 mil tokens — muito além do contexto do modelo (7 blocos × 128 = **896 tokens**). Estratégias testadas (`texto_puro/paginacao.py`):

| Estratégia | Mecanismo | Resultado |
|---|---|---|
| **Truncamento** | corte rígido em 128 tokens | descarta o excedente; perde o fim da série |
| **Janela + overlap 15%** | janela deslizante com 15% de sobreposição | **escolhida** — zero descarte, continuidade semântica entre páginas |
| **Keywords** | remoção de stopwords (PT/EN/ES) | compressão agressiva, mas perde conectivos e estrutura |

A janela com overlap preserva a informação e mantém a sintaxe estrutural intacta — decisivo para o §3.

### 2.3 O paradoxo da inflação de tokens

Medição real na Fase A (`benchmark_etth1_texto.json`): o mesmo trecho da série gera **39.262 tokens em PT Completo vs 24.006 em EN Curto — inflação de ×1,64** para o português. A hipótese inicial era que o inglês (língua nativa do GPT-2, menos tokens, maior densidade informacional) dominaria o ranking. **O experimento provou o contrário**: os formatos em **espanhol** venceram. A explicação defendida combina dois fatores: (i) **eficiência ≠ compreensão** — o inglês comprime os numerais em tokens mais longos, mas a sub-tokenização dos números compostos ainda fragmenta o valor; (ii) **regularidade morfológica do espanhol** — palavras mais curtas e fronteiras estáveis reduzem a ambiguidade do numeral por extenso, compensando o custo marginal de tokens. O paradoxo virou hipótese central: para LLMs pré-treinados, a **forma** do prompt importa tanto quanto a densidade da informação.

---

## 3. O Salto Estrutural (Structural Prompting)

### 3.1 A bateria de 21 cenários

Cruzando 10 templates estruturados (estrutural/Markdown, **pseudo-JSON** e relacional/setas em PT/EN/ES + temporal-delta EN) com 2 estratégias de paginação (janela, keywords), mais a baseline de language tuning (ES keywords), executou-se **21 cenários × 3 seeds × 10 épocas** no harness local (`benchmark_formatacoes_estruturadas.py`, `estruturas_avancadas_stats.json`):

| # | Cenário | Test MSE (raw) |
|---|---|---|
| 1 | **pseudo_json_es · janela** | **7,034 ± 0,059** |
| 2 | temporal_delta_en · keywords | 7,105 ± 0,098 |
| 3 | estrutural_es · janela | 7,120 ± 0,075 |
| 4 | relacional_es · keywords | 7,133 ± 0,228 |
| 5 | estrutural_en · janela | 7,151 ± 0,195 |
| … | … (16 restantes) | 7,170 – 7,560 |
| 21 | relacional_es · janela | 7,560 ± 0,537 |

### 3.2 Por que a estrutura funciona

1. **Ancoragem por pares chave-valor.** O pseudo-JSON ES repete a cada ponto a dupla `clave tiempo … clave valor …` — chaves fixas que funcionam como *prompts de atenção*: o GPT-2 aprende a isolar o "valor matemático" do rótulo temporal, ancorando o mecanismo de atenção.
2. **Prior de código/JSON.** A sintaxe nome-valor é a mais compatível com o texto de pré-treinamento do GPT-2 (JSON/código são abundantes no corpus), mais do que narrativas contínuas ou delimitadores `[ … ]`.
3. **Janela > truncamento > keywords.** Manter o documento íntegro com overlap 15% superou descartar e comprimir.
4. **A agregação por tipo confirma** (`06_por_tipo.png`): pseudo-JSON (7,23±0,15) < estrutural (7,21±0,08) < relacional (7,37±0,15) < temporal-delta — o padrão é consistente, não ruído de uma seed.

**Ressalva crítica:** 7,03 era o **melhor do harness** (ranking interno justo: mesmas 192 janelas, mesma escala) — mas o harness não era publicável como baseline absoluto (§1.2). O salto estrutural precisava ser re-validado no protocolo definitivo (§4).

---

## 4. O Benchmark "Apples-to-Apples"

### 4.1 Metodologia definitiva (`benchmark_definitivo_textual.py`)

O campeão `pseudo_json_es · janela` foi re-executado nas **condições exatas do baseline numérico**:

- **Z-score nativo:** `StandardScaler` ajustado apenas em `[0, 8640)` e aplicado ao dataset inteiro; janelas numéricas, targets **e texto** gerados sobre os valores z-score; MSE/MAE calculados nativamente nesse espaço — o mesmo espaço em que o 0,3539 foi obtido.
- **Splits canônicos:** train `[0, 8640)` · val `[7968, 11520)` · test `[10848, 14400)` → **94.101 janelas** reais (55.111 / 19.495 / 19.495), stride=1, 7 variáveis, janelas univariadas por feature (ordem da bateria).
- **Geração em larga escala:** 16 workers (1 tarefa por split×feature), cache em disco (hash da config + mtime do CSV), 94.101 documentos em ~211 s (`cache_definitivo/cf15f677bf6b4ca0`).
- **Treinamento:** 10 épocas × **3 seeds** (2021, 42, 1234), batch 256 (auto-fit à VRAM), lr 1e-3, cosine, `restore_best`, **FP32+TF32 (sem AMP)**, skip-batch com `batches_estourados_total` — **0 estouros**.
- **Paridade interna:** antes do treino textual, o baseline numérico é re-medido no mesmo test set → 0,353926.

### 4.2 Conversão de escala (Raw → Z-score)

Como MSE é homogêneo de grau 2 e o harness media na escala bruta da HUFL: **MSE_z = MSE_raw / σ²_treino** (σ² = 33,7877). O recorde do harness recalibra para 7,03/33,79 = **0,2082 z** — número **otimista demais**: a conversão remove apenas a escala; volume (×479) e leakage do harness permanecem. O número publicável exige o protocolo definitivo.

### 4.3 Resultado final

| Métrica | Baseline numérico (Run-8) | Campeão textual (pseudo_json_es · janela) | Δ |
|---|---|---|---|
| **Test MSE_Z** | **0,3539** (medido 0,353926) | **0,3669 ± 0,0069** | **+3,67%** |
| Test MAE_Z | 0,3919 | 0,4026 ± 0,0045 | +2,73% |
| Equiv. raw HUFL | 11,96 | 12,40 | — |

| Seed | Test MSE_Z | Test MAE_Z | Melhor época | Estouros |
|---|---|---|---|---|
| 2021 | 0,357134 | 0,397474 | 1 | 0 |
| 42 | 0,371784 | 0,405011 | 2 | 0 |
| 1234 | 0,371808 | 0,405238 | 1 | 0 |

**Interpretação.** Um GPT-2-base **congelado**, lendo exclusivamente palavras estruturadas (zero algarismos), preserva **96,3%** da capacidade preditiva do pipeline puramente numérico da mesma arquitetura: o texto não degrada a previsão de forma relevante. O delta de +3,67% é a medida honesta da perda informacional da serialização. Figuras: `01_veredito_barras.png`, `02_jornada_escala.png`, `04_convergencia_veredito.png`, `07_por_seed.png`.

---

## 5. Ressalvas e Limitações (para transparência do artigo)

1. **Perda de treino distinta:** o baseline foi medido eval-only com checkpoint treinado com VolLoss; o modo textual re-treina apenas o módulo de fusão (aleatório, ~2,81 M parâmetros; backbone, encoder/decoder e gate congelados) com **MSE puro**. Avaliação idêntica nas duas condições.
2. **Splits canônicos com offset `−seq_len`:** val/test compartilham marginalmente a região temporal (protocolo oficial do TSLib/AutoTimes, aplicado igualmente às duas condições). Zero vazamento *adicional*.
3. **"85,7%" não é vazamento:** é `label_len/seq_len`; o vazamento real do harness era região compartilhada (100%), overlap de janelas (99,1%) e volume ×479 — e o 0,2082 z derivado do harness **não é publicável**.
4. **Cobertura do contexto textual:** 896 tokens ≈ 65-90 pontos da janela de 672 — a parcela lida pelo LLM é ~10-13%; a diferença residual de +3,67% pode subir com mais contexto.
5. **Escopo:** um dataset (ETTh1), um horizonte (96), 3 seeds (dp 0,0069 < 2% do valor). Generalização (ETTm, outros horizontes, outros LLMs) é trabalho futuro.
6. **Hardware:** resultados em ROCm (AMD RX 9060 XT), FP32+TF32; reprodução em CUDA pode variar no 5º dígito decimal.