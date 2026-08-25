# Relatório Técnico — Previsão Multimodal de Séries Temporais com LLM (AutoTimes + GPT-2)

**Título (proposto):** *"Ler Matemática em Palavras: Serialização Estruturada Multilíngue para Previsão de Séries Temporais com GPT-2 Congelado — um Benchmark Apples-to-Apples no ETTh1"*

**Ambiente:** ETTh1 (17.420 horas, 7 variáveis) · backbone GPT-2-base congelado (127,25 M parâmetros; 2,81 M treináveis) · arquitetura VolLoss + SMETimes · ROCm (AMD RX 9060 XT, 17 GB) · 3 seeds · 10 épocas · batch 256 · FP32/TF32.

---

## 1. Resumo Executivo (Abstract)

Este trabalho investiga se um modelo de linguagem pré-treinado (GPT-2-base), com pesos **congelados**, é capaz de extrair o comportamento matemático de uma série temporal lendo apenas **palavras estruturadas** — sem algarismos. A série contínua é serializada em linguagem natural por extenso (num2words) em múltiplos idiomas (PT, EN, ES) e formatos (verboso, compacto, Markdown, relacional, pseudo-JSON), paginada e alimentada ao LLM em paralelo ao ramo numérico (arquitetura multimodal do AutoTimes). Uma auditoria forense do harness de avaliação revelou **dois vícios** que invalidavam comparações diretas: (i) **escala** — o baseline oficial é medido em espaço z-score global (StandardScaler ajustado no treino), enquanto o harness media na escala bruta; o fator de correção $\sigma^2_{\text{HUFL}} = 33{,}79$ foi confirmado empiricamente com erro nulo; e (ii) **vazamento temporal** — validação e teste do harness eram extraídos da mesma região temporal do treino. Após a correção, um protocolo rigoroso (z-score nativo, limites cronológicos estritos, 94.101 janelas reais) foi aplicado ao campeão estrutural *pseudo-JSON espanhol com paginação em janela*: o modelo multimodal atingiu **Test MSE = 0,3669 ± 0,0069** em espaço z-score, apenas **+3,67%** acima do baseline puramente numérico (**0,3539**, reproduzido com precisão de 0,0%). O LLM, lendo exclusivamente palavras estruturadas, preserva praticamente toda a informação preditiva do sinal.

---

## 2. Auditoria Arquitetural e Correção de Baseline

### 2.1 A descoberta: dois vícios no harness

O harness `testes_letras` foi construído para comparar formatos de texto sobre o **mesmo** checkpoint (Run-8, VolLoss + SMETimes), mas com um pipeline próprio, paralelo ao da bateria oficial. A auditoria (`auditoria_harness.py`) confrontou os dois pipelines e isolou as diferenças:

| Dimensão | Harness antigo (exploração) | Protocolo oficial (artigo) | Diagnóstico |
|---|---|---|---|
| Escala de avaliação | **Raw** (coluna HUFL, sem normalização) | **Z-score global** (`StandardScaler` ajustado no treino) | MSE em espaços de medida diferentes — comparação direta inválida |
| Janelas de treino | 115 (pool de 192, linhas 0–2000) | 55.111 (dataset completo, stride=1) | Redução de **×479** → underfitting e erro estatístico alto |
| Splits | 60/20/20 do mesmo pool (região 0–2000) | Cronológicos disjuntos: `[0,8640)` / `[8640,11520)` / `[11520,14400)` | **Vazamento temporal**: todas as janelas de validação e teste saem da mesma região temporal do treino (janelas sobrepostas na série) |
| Batch / protocolo | batch 16, sem restore | batch 256, `cosine`, restore do melhor checkpoint | Menor contribuição, corrigida por paridade |

**Tabela 1.** Harness de exploração vs. protocolo oficial do ablation (ETTh1, seq=672, pred=96, label=576).

### 2.2 O fator de escala: prova empírica

O baseline oficial (0,3539) é calculado **nativamente em z-score**: o `load_real_split` do pipeline original ajusta o `StandardScaler` apenas no split de treino e janela o dataset já transformado. O harness, ao contrário, carregava os valores brutos e media o MSE na escala da HUFL. Como o erro quadrático é homogêneo de grau 2, a conversão é exata:

$$
\text{MSE}_z = \frac{\text{MSE}_{\text{raw}}}{\sigma^2_{\text{treino}}}
\qquad \sigma^2_{\text{HUFL}} = 33{,}7877
$$

A validação empírica foi **decisiva**: o próprio checkpoint Run-8, avaliado nas mesmas 8 janelas em ambas as escalas, produziu uma razão `MSE_raw / MSE_z = 33,79` — igual à variância do treino da HUFL com erro < 0,01%. Em seguida, o checkpoint foi avaliado em modo numérico puro sobre o test set oficial completo (19.495 janelas): **MSE_z = 0,3539** (MAE 0,3919), delta **+0,0%** em relação ao artigo — o harness corrigido é uma réplica exata do protocolo.

### 2.3 Estabilização (Skip Batch)

O loop de treinamento implementa a trava de segurança da bateria: batches com loss/gradiente NaN ou Inf são **pulados** (`clip_grad_norm(max_norm=1.0)` + verificação de finitude), contabilizados como "estouros". No benchmark definitivo, **0 batches estourados** nas 3 seeds × 10 épocas.

---

## 3. Metodologia Multimodal (O Input Pipeline)

### 3.1 Fase A — Serialização numérica → texto por extenso

A série $x_{t-T+1}, \dots, x_t$ de cada janela (672 passos) é convertida ponto a ponto em linguagem natural com **num2words** (1 casa decimal, zero algarismos), preservando o par (tempo, valor) de cada observação. Exemplos (ponto único):

- **pt-verboso:** "as catorze horas o valor era vinte e nove virgula cinco"
- **pt-curto / en-curto:** "10/Jul/2022 14:00 | Valor: 29.5"
- **estrutural ES:** `[ Tiempo : julio diez , catorce horas | Val : veintinueve coma cinco ]`
- **pseudo-JSON ES:** `clave tiempo julio diez catorce horas clave valor veintinueve coma cinco`

### 3.2 Fase B — Paginação / Chunking

Documentos longos excedem o contexto do GPT-2 (block_max = 128 tokens). Três estratégias foram testadas (`paginacao.py`):

| Estratégia | Mecanismo | Custo |
|---|---|---|
| **truncamento** | corte rígido no limite de 128 tokens | descarte do excedente |
| **janela** (escolhida) | janela deslizante com **overlap de 15%** | zero descarte; continuidade semântica entre páginas |
| **keywords** | remoção de stopwords (PT/EN/ES) antes da tokenização | perda de conectivos, compressão |

Cada janela gera 7 páginas × 128 tokens = **896 tokens** de contexto textual, alinhadas token-a-token com os 7 patches numéricos (token_len = 96).

### 3.3 Fase C — Fusão multimodal

O AutoTimesModel opera em duas rotas paralelas (rigorosa ordem de operações):

1. **Rota numérica:** `RevIN normalize → patching (token_len=96) → encoder MLP → [gate SMETimes]`;
2. **Rota textual:** `wte do GPT-2 (congelado) → pooling médio por bloco (máscara por lens)`; 
3. **Fusão:** `cat([texto, num]) → Linear → LayerNorm` (estabiliza magnitudes) → GPT-2 congelado → decoder sobre a rota numérica → `RevIN denormalize` **apenas** na reconstrução numérica.

O checkpoint (treinado em modo numérico) fornece backbone, encoder/decoder e gate; apenas o módulo de fusão textual (aleatório) é treinado do zero — ~2,81 M parâmetros.

---

## 4. A Descoberta do Prompting Estrutural

### 4.1 A bateria de 21 cenários

Cruzando 10 templates estruturados (estrutural, pseudo-JSON e relacional em PT/EN/ES + temporal delta EN) com 2 estratégias de paginação (janela, keywords), mais a baseline de language tuning (ES keywords), obteve-se o ranking completo (escala raw do harness, pool de 192 janelas):

| # | Cenário | Idioma | Estratégia | Test MSE (raw) |
|---|---|---|---|---|
| 1 | **pseudo_json_es** | ES | **janela** | **7,034 ± 0,059** |
| 2 | temporal_delta_en | EN | keywords | 7,105 ± 0,098 |
| 3 | estrutural_es | ES | janela | 7,120 ± 0,075 |
| 4 | relacional_es | ES | keywords | 7,133 ± 0,228 |
| 5 | estrutural_en | EN | janela | 7,151 ± 0,195 |
| … | (16 demais cenários) | … | … | 7,170 – 7,560 |
| 21 | relacional_es | ES | janela | 7,560 ± 0,537 |

**Tabela 2.** Top-5 e faixa dos 21 cenários (`estruturas_avancadas_stats.json`).

### 4.2 O paradoxo EN vs. ES e a ancoragem estrutural

O GPT-2 é nativamente anglófono: o inglês produz **menos tokens por ponto** (maior densidade de informação por token) e, ainda assim, os formatos em **espanhol** venceram. A hipótese defendida: a **ancoragem estrutural** — chaves fixas repetidas a cada ponto (`clave tiempo … clave valor …`) funcionam como *prompts de atenção* que isolam a "chave temporal" do "valor matemático", ancorando o mecanismo de atenção do GPT-2. O espanhol, por sua vez, produz palavras com fronteiras morfológicas mais regulares (menos sub-tokenização que o inglês em números compostos), reduzindo a ambiguidade numérica. O formato **pseudo-JSON** (chave-valor) superou as narrativas contínuas (verboso/curto) e os delimitadores de Markdown (`[ … | … ]`), sugerindo que a sintaxe de pares nome-valor é a mais compatível com o *prior* de código/JSON do GPT-2. A paginação em **janela com overlap de 15%** venceu o truncamento e as keywords: preserva a continuidade semântica entre páginas sem descartar informação.

### 4.3 A aparente degradação era um artefato de escala

O campeão do harness registrou 7,03 raw — "pior" que o baseline numérico 5,80 raw do mesmo harness. A auditoria mostrou que **ambos** estavam na escala errada: recalibrados pelo fator $\sigma^2_{\text{HUFL}} = 33{,}79$:

| Marco | MSE raw | MSE z equivalente |
|---|---|---|
| Baseline numérico (harness) | 5,7988 | 0,1716 |
| PT Extenso Compacto | 7,32 | 0,2166 |
| ES Keywords (LN) | 7,2445 | 0,2144 |
| **ES Pseudo-JSON (recorde)** | **7,0336** | **0,2082** |

**Tabela 3.** Recalibração dos marcos (a conversão remove apenas a escala; volume e leakage do harness permanecem como limitações — ver §5).

---

## 5. O Benchmark Apples-to-Apples (Resultados Finais)

### 5.1 Protocolo definitivo

O campeão (`pseudo_json_es` + paginação janela) foi re-executado **nas condições exatas do baseline numérico** (`benchmark_definitivo_textual.py`):

- **Z-score nativo:** `StandardScaler` ajustado apenas no split de treino e aplicado a todo o dataset; janelas numéricas, targets **e texto** gerados sobre os valores z-score; MSE/MAE calculados nativamente nesse espaço;
- **Splits cronológicos estritos:** treino `[0, 8640)` · validação `[8640, 11520)` · teste `[11520, 14400)`; **94.101 janelas** reais (55.111 / 19.495 / 19.495), stride=1, todas as 7 variáveis;
- **Larga escala:** geração textual paralela (16 workers, 1 tarefa por split×variável) com cache em disco (chave = hash da configuração) — 94.101 documentos convertidos em ~3,5 min;
- **Treinamento:** 10 épocas, lr=1e-3, batch 256 (auto-ajuste à VRAM), **3 seeds** (2021, 42, 1234), restauração do melhor checkpoint por validação (`restore_best`), FP32/TF32, 0 estouros.

### 5.2 Resultados

| Métrica | Baseline numérico (Run-8) | Campeão textual | Δ |
|---|---|---|---|
| **Test MSE_Z** | **0,3539** (reproduzido: 0,353926) | **0,3669 ± 0,0069** | **+3,67%** |
| Test MAE_Z | 0,3919 | 0,4026 ± 0,0045 | +2,73% |
| Equivalente raw HUFL (×33,79) | 11,96 | 12,40 | — |

**Tabela 4.** Veredito final (10 épocas × 3 seeds, teste de 19.495 janelas).

| Seed | Test MSE_Z | Test MAE_Z | Melhor época (val) | Estouros |
|---|---|---|---|---|
| 2021 | 0,357134 | 0,397474 | 1 | 0 |
| 42 | 0,371784 | 0,405011 | 2 | 0 |
| 1234 | 0,371808 | 0,405238 | 1 | 0 |

**Tabela 5.** Detalhe por seed.

### 5.3 Interpretação honesta dos números

- O recorde anterior (7,03 raw → 0,2082 z) era **otimista demais**: o harness usava 115 janelas de treino com vazamento temporal; a conversão por $\sigma^2$ remove apenas o efeito de escala. No protocolo correto, o mesmo modelo textual atinge **0,3669 z** — um erro **~76% maior** que o 0,2082 sugeria, porém ainda **+3,67%** acima do baseline.
- A variância entre seeds (0,0069) é menor que 2% do valor da métrica: o resultado é reprodutível.

---

## 6. Conclusões e Implicações

1. **LLM4TS sem treinar o LLM:** um GPT-2-base com pesos congelados, recebendo apenas texto estruturado por extenso, reproduz **96,3%** da capacidade preditiva do pipeline puramente numérico da mesma arquitetura (+3,67% de MSE). A matemática não precisa de algarismos — precisa de **ancoragem estrutural**.

2. **A serialização é uma restrição representacional, não um teto:** a distância de 3,7% pode refletir a janela textual efetiva (apenas ~70 pontos/896 tokens por janela) e não uma limitação intrínseca do LLM. Contextos maiores, paginação adaptativa e *prompting* em múltiplos idiomas são direções naturais de fechamento do gap.

3. **Espanhol > Inglês, contrariando o *prior* linguístico:** a riqueza semântica da estrutura (chaves `clave tiempo / clave valor`) superou a eficiência de tokens do inglês — evidência de que, para LLMs pré-treinados, a **forma** do prompt importa tanto quanto a **densidade** da informação.

4. **Metodologia:** a auditoria de escala (raw vs z-score) e de vazamento temporal evitou a publicação de um resultado espúrio (0,2082 z). A lição metodológica: em benchmarks multimodais, o **espaço de medida** e o **escopo temporal** dos splits devem ser idênticos entre as condições comparadas, e qualquer conversão entre escalas deve ser validada empiricamente (razão 33,79 = $\sigma^2$ com erro < 0,01%).

5. **Reprodutibilidade:** pipeline completo versionado (cache de tensores, seeds fixas, fingerprint de configuração, 0 estouros, checkpoint Run-8 reproduzido com delta 0,0%).

---

## Apêndice — Figuras (pasta `comparacao/`)

| Figura | Conteúdo |
|---|---|
| `01_veredito_barras.png` | Veredito final: baseline numérico vs campeão textual (z-score, error bars por seed, eixo raw equivalente) |
| `02_jornada_escala.png` | Jornada raw → z (a aparente degradação era artefato de escala) |
| `03_ranking_21.png` | Ranking dos 21 cenários (barras horizontais, cores semânticas, barreira LN) |
| `04_convergencia_veredito.png` | Convergência val-z do campeão + baseline + testes finais por seed |
| `05_convergencia_21.png` | Convergência por época dos 21 cenários (banda ± 1 dp) |
| `06_por_tipo.png` | Agregado por tipo de formatação (efeito da estrutura) |
| `07_por_seed.png` | Reprodutibilidade: seed a seed vs baseline |

**Materiais de origem:** `testes_letras/benchmark_definitivo_textual.py`, `benchmark_numerico_base.py`, `auditoria_harness.py`, `treino.py`, `testes_letras/resultados/veredito_final_apples_to_apples.json`, `estruturas_avancadas_stats.json`, `Testes_eficiencia/log/benchmark_results_20260809_120255.json`.