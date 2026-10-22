# Metodologia do Pipeline de Dados — AutoTimes Multimodal (VolLoss + SMETimes)

> Documento técnico-acadêmico que descreve, de ponta a ponta, o fluxo de processamento de
> dados da pesquisa: **como a série temporal é transformada em texto** e **como o modelo
> retorna ao espaço numérico para calcular os erros (MSE/MAE)**.
>
> Todo o conteúdo é fundamentado no código real do repositório (`testes_letras/`,
> `Testes_eficiencia/`), com referências `arquivo:linha` para verificação.

---

## Sumário

1. [Contexto e motivação](#1-contexto-e-motivação)
2. [A arquitetura AutoTimes original (fundamentos)](#2-a-arquitetura-autotimes-original-fundamentos)
3. [Modificações estruturais: VolLoss + SMETimes + Fusão Textual](#3-modificações-estruturais-volloss--smetimes--fusão-textual)
4. [Visão geral do pipeline em 3 fases](#4-visão-geral-do-pipeline-em-3-fases)
5. [FASE A — Da série ao texto (Input Pipeline)](#5-fase-a--da-série-ao-texto-input-pipeline)
6. [FASE B — Paginação e tokenização (chunking)](#6-fase-b--paginação-e-tokenização-chunking)
7. [FASE C — Predição no espaço contínuo e cálculo de erro (Output & Loss)](#7-fase-c--predição-no-espaço-contínuo-e-cálculo-de-erro-output--loss)
8. [Perguntas centrais respondidas (FAQ técnica)](#8-perguntas-centrais-respondidas-faq-técnica)
9. [Geometria e shapes (tabelas de referência)](#9-geometria-e-shapes-tabelas-de-referência)
10. [Trade-offs e interpretação dos resultados](#10-trade-offs-e-interpretação-dos-resultados)
11. [Glossário](#11-glossário)
12. [Mapa de arquivos do código](#12-mapa-de-arquivos-do-código)

---

## 1. Contexto e motivação

A previsão de séries temporais (TSF — *Time Series Forecasting*) tradicionalmente opera em
tensores numéricos contínuos. A arquitetura **AutoTimes** (baseada em GPT-2) demonstrou que
um LLM pré-treinado pode ser reaproveitado como **backbone de representação** para séries
temporais, sem geração de texto: a série é *patchificada* (segmentada em janelas de 96 passos),
cada segmento é projetado por um MLP para o espaço de embeddings do GPT-2, e o transformer
processa esses vetores como se fossem "tokens" de uma linguagem aprendida.

Neste projeto, a arquitetura foi estendida em duas direções:

1. **Modo multimodal (textual)**: além da rota numérica, uma descrição em **linguagem
   natural** da série (datas, horas e valores por extenso, em PT/EN/ES) é injetada como
   *conditioning* no espaço latente, por meio da **fusão** dos embeddings textuais com os
   embeddings numéricos dos patches.
2. **Melhorias estruturais** — **VolLoss** (perda Log-Cosh ciente da volatilidade) e
   **SMETimes** (porta adaptativa de fusão com estatísticas por token) — que estabilizam o
   treinamento e modelam explicitamente a volatilidade local da série.

O pipeline demonstrou alta eficácia na previsão de séries como o **ETTh1** usando
formatações estruturadas de texto (campeã: **ES Pseudo-JSON**, Test MSE 7.03 ± 0.06;
**ES Keywords**, 7.24 ± 0.12).

---

## 2. A arquitetura AutoTimes original (fundamentos)

O AutoTimes (Liu et al.) propõe usar o LLM pré-treinado **apenas como um tokenizer
implicitamente aprendido** para séries temporais. O fluxo original (`models/AutoTimes_Gpt2.py`):

```
x_enc (B, 672, 1)
   │  RevIN: z-score com mean/std da janela (detach)
   ▼
x_norm (B, 672, 1)
   │  unfold: patching em segmentos de token_len = 96 (sem overlap)
   ▼
patches (B, 7, 96)
   │  MLP encoder (Linear + tanh + dropout, 256 dims ocultas)
   ▼
times_embeds (B, 7, 768)          ← "tokens" contínuos no espaço do GPT-2
   │
   ▼
GPT-2 (congelado, inputs_embeds)  ← backbone transformer
   │  last_hidden_state
   ▼
outputs (B, 7, 768)
   │  MLP decoder (Segment Projection — inverso do encoder)
   ▼
dec_out (B, 7, 96)
   │  reshape + RevIN denormalize (multiplica por std e soma mean)
   ▼
y_pred (B, 672, 1)                ← reconstrução na escala real
```

Princípios que o AutoTimes estabelece e que o projeto preserva:

| Princípio | Descrição |
|---|---|
| **Token implícito** | O patch de 96 passos é um "token" de série temporal; o MLP é o tokenizer, o MLP inverso é o detokenizer (Segment Projection). |
| **Backbone congelado** | O GPT-2 não é fine-tunado (0 camadas descongeladas); apenas MLPs, fusão e gates treinam. |
| **Sem geração autoregressiva** | Não há LM head nem amostragem sobre o vocabulário; a saída é a reconstrução contínua da janela. |
| **RevIN como normalização** | Z-score por janela antes de entrar e desnormalização na saída (estatística desacoplada do gradiente). |
| **Tarefa de reconstrução com horizonte** | O alvo é a janela completa de 672 passos (label 576 + pred 96), e a métrica de interesse concentra-se nos últimos 96 passos (horizonte). |

---

## 3. Modificações estruturais: VolLoss + SMETimes + Fusão Textual

### 3.1 VolLoss — `VolatilityAwareLogCoshLoss` (`Testes_eficiencia/modules/volatility_loss.py`)

Perda baseada em Log-Cosh, diferenciável e mais robusta a outliers que o MSE:

$$
\mathcal{L}_{\text{logcosh}}(u) = \log(\cosh(u)), \qquad u = y_{pred} - y_{true}
$$

com implementação numericamente estável (evita overflow em $\lvert u \rvert$ grande):

$$
\log(\cosh(u)) = u + \operatorname{softplus}(-2u) - \ln 2 .
$$

A variante ciente da volatilidade multiplica o erro por um peso que decai
exponencialmente com a variância da janela:

$$
w = \exp\left(-\beta \cdot \operatorname{Var}(x_{win})\right), \qquad \beta \ge 0
$$

ou seja, **janelas de alta volatilidade contribuem menos para o gradiente**, evitando que
regimes turbulentos dominem a otimização. A variância é tratada como *feature injetada*
(`variance.detach()`), portanto não recebe gradiente.

> Nota metodológica: o checkpoint `Run-8` foi **treinado com VolLoss** (é a origem do nome do
> modelo `smetimes`). Já os loops de convergência de comparação de templates (`treino.py`)
> usam **MSE puro** (`nn.MSELoss`) para garantir comparabilidade justa entre cenários —
> mesma semente, mesmo otimizador, mesma loss. A VolLoss é o *contexto arquitetural* do
> checkpoint; o MSE/MAE é o critério de avaliação.

### 3.2 SMETimes — `AdaptiveFusion` (`Testes_eficiencia/run_benchmarks.py:263`)

Para cada patch de 96 passos, são computadas **4 estatísticas por token**
(`compute_token_stats`, `run_benchmarks.py:241`):

1. **média** — nível local;
2. **variância** — volatilidade local;
3. **tendência OLS** — inclinação linear (regressão sobre índice temporal centrado);
4. **range** (max − min) — amplitude do regime.

Essas features são projetadas por `stat_proj` e combinadas com o embedding temporal por uma
**porta sigmoidal**:

$$
g = \sigma\big( \text{Linear}([\, \mathbf{t}_{time} \parallel \mathbf{t}_{stat} \,]) \big),
\qquad
\mathbf{t}_{fused} = g \cdot \mathbf{t}_{time} + (1 - g) \cdot \mathbf{t}_{stat}
$$

seguida de LayerNorm. A porta decide, **por token**, o quanto a representação deve confiar na
forma temporal (aprendida) versus nas estatísticas de volatilidade (calculadas). É isso que
permite ao modelo modular a representação conforme o regime local da série.

### 3.3 Fusão Textual (modo multimodal) — `run_benchmarks.py:396-400`

No modo `textual=True`, um módulo `fusion` (que **não existe** no checkpoint numérico e nasce
aleatório) é anexado:

```
cat([txt_tokens, num_tokens]) → Linear(2·768 → 768) → LayerNorm → GPT-2
```

Ele alinha os dois fluxos **antes** do transformer, estabilizando magnitudes e gradientes.

> **Fidelidade dos checkpoints**: os checkpoints `smetimes`/`amd` foram treinados em modo
> **numérico** e não contêm o módulo `fusion`. No modo textual, apenas `fusion` nasce com
> pesos aleatórios e é treinado do zero no loop de convergência; GPT-2 (congelado),
> encoder/decoder MLP, gate SMETimes e RevIN vêm do checkpoint (`testes_letras/modelos.py:14`).

---

## 4. Visão geral do pipeline em 3 fases

```
┌─────────────────────────────────────────────────────────────────────┐
│ FASE A — SERIALIZAÇÃO TEXTUAL                                        │
│   ETTh1.csv ──► recorte (região de treino) ──► num2words ──►        │
│   documento em linguagem natural (PT/EN/ES, verboso/curto/           │
│   keyword/JSON/relacional/estrutural) ──► tokenização BPE GPT-2      │
├─────────────────────────────────────────────────────────────────────┤
│ FASE B — PAGINAÇÃO (chunking)                                        │
│   documento completo ──► 3 estratégias: truncamento | janela +       │
│   overlap | compressão por keywords ──► páginas ≤ block_max ──►      │
│   ids achatados + lens (prontos para o modelo)                       │
├─────────────────────────────────────────────────────────────────────┤
│ FASE C — CONVERGÊNCIA NO MODELO (VolLoss + SMETimes)                 │
│   janelas numéricas da MESMA região + páginas do texto ──►           │
│   forward multimodal ──► Segment Projection ──► predição contínua    │
│   ──► MSE/MAE vs alvo numérico real                                  │
└─────────────────────────────────────────────────────────────────────┘
```

Execução: `python -m testes_letras.run_etth1_texto` (script principal do benchmark
`run_etth1_texto.py`). O mesmo pipeline de texto puro é validado por
`texto_puro/` e os benchmarks de formatações por `benchmark_formatacoes_estruturadas.py`.

---

## 5. FASE A — Da série ao texto (Input Pipeline)

### 5.1 Leitura e recorte do dataset

**Arquivo**: `testes_letras/run_etth1_texto.py:118` (`carregar_etth1`).

1. O CSV `dataset/ETT-small/ETTh1.csv` é lido com `pandas`.
2. A primeira coluna (timestamps) é convertida com `pd.to_datetime` para uma lista de
   `datetime`; as demais colunas (features como `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`,
   `OT`) são convertidas para `np.float32`.
3. Um subconjunto de `L` linhas iniciais (default 2000; a região de treino do split AutoTimes
   é 12 meses = 8640 linhas) é usado como **região de conversão** — ela gera o documento
   textual e, na Fase C, o pool de janelas numéricas da mesma região.

### 5.2 Serialização linguística: `num2words` e conversores por extenso

Dois conversores coexistem e são documentados aqui para clareza:

#### (a) Conversor canônico multi-idioma — `testes_letras/texto_puro/extenso.py:65`

Baseado na biblioteca **`num2words`**, com normalizações por idioma:

| Idioma | Exemplo (29.5) | Normalização aplicada |
|---|---|---|
| **PT** | `vinte e nove virgula cinco` | nativa do num2words (pt-BR) |
| **EN** | `twenty nine point five` | remoção de hífens (`twenty-nine → twenty nine`); dias usam ordinais (`tenth`) |
| **ES** | `veintinueve coma cinco` | `punto → coma` (vírgula decimal do espanhol) |

Timestamps também são verbalizados: meses por idioma, dias por extenso (ordinais no EN),
horas e minutos por extenso. Nenhum algarismo sobrevive no documento.

#### (b) Conversor legado pt-BR — `Testes_eficiencia/modules/textual.py:58`

Implementação por regras próprias (sem `num2words`), utilizada como baseline:
`valor_por_extenso(29.5) → "vinte e nove virgula cinco"`,
`descrever_momento(...) → "dez de julho de dois mil e vinte e dois, as catorze horas"`.
Também oferece geração de dataset textual lookback→forecast
(`gerar_dataset_csv`, `textual.py:230`) e exportação tokenizada em `.pt`
(`tokenizar_para_pt`, `textual.py:305`).

### 5.3 Templates de formatação estruturada

A pesquisa avaliou dezenas de formatos. Os principais grupos:

#### (a) Por verbosidade — `extenso.py` (9 templates = 3 idiomas × 3 níveis)

| Nível | PT | EN | ES |
|---|---|---|---|
| **Verboso** | `No dia dez de julho, as catorze horas, o valor registrado foi vinte e nove virgula cinco.` | `On July tenth, at fourteen hours, the recorded value was twenty nine point five.` | `El diez de julio, a las catorce horas, el valor registrado fue veintinueve coma cinco.` |
| **Curto** | `Dez de julho, catorze horas, valor vinte e nove virgula cinco.` | `July tenth, fourteen hours, value twenty nine point five.` | `Diez de julio, catorce horas, valor veintinueve coma cinco.` |
| **Keywords** | `julho dez catorze horas vinte nove virgula cinco.` | `july tenth fourteen hours twenty nine point five.` | `julio diez catorce horas veintinueve coma cinco.` |

Regras dos templates *keywords*: caixa baixa, mês antes do dia, remoção da conjunção `e`
dos números PT (`vinte e nove → vinte nove`) via regex com boundary de palavra.

#### (b) Estruturados (ancoragem de código do GPT-2) — `texto_puro/estruturado.py`

Exploram o fato de o GPT-2 ter sido pré-treinado em código/Markdown/JSON:

| Template | Formato (exemplo EN) |
|---|---|
| **Estrutural** | `[ Time : July ten , fourteen hours \| Val : twenty nine point five ]` |
| **Pseudo-JSON** | `key time july ten fourteen hours key value twenty nine point five` (PT: `chave tempo ... chave valor ...`; ES: `clave tiempo ... clave valor ...`) |
| **Relacional** | `July ten fourteen hours -> twenty nine point five` (seta causal) |
| **Temporal Delta** | `Start July ten. T fourteen: twenty nine point five. T fifteen: thirty point one.` |

#### (c) Compactos e JSON — `testes_letras/templates.py`

| Template | Formato |
|---|---|
| **T1 PT Verboso** | baseline por extenso (step fixo 24 do pipeline) |
| **T2 PT Curto** | `10/Jul/2022 14:00 \| Valor: 29.5` |
| **T3 EN Curto** | `Jul 10, 2022, 2:00 PM \| Value: 29.5` |
| **T4 JSON** | `[{"time": "2022-07-10T14:00", "val": 29.5}]` |

Os templates compactos têm `text_step` **auto-ajustado** (`escolher_text_step`,
`templates.py:204`): tenta `step=1` (todos os pontos) e aumenta a esparsidade até nenhum
bloco das janelas amostradas ultrapassar `block_max=128` tokens. Assim, maximiza-se a
**informação de tendência por token** — a métrica de "compressão real" do formato.

### 5.4 Tokenização BPE e inflação de tokens

O tokenizador é o **GPT-2 local** (BPE, vocab 50.257; carregado por descoberta recursiva do
diretório — `run_etth1_texto.py:361`). A tokenização usa `add_special_tokens=False` e
`pad_token = eos_token = 50256`.

**Inflação PT/EN** (`texto_puro/idioma.py:19`):

$$
\text{taxa\_inflacao} = \frac{\text{tokens GPT-2 (texto PT)}}{\text{tokens GPT-2 (texto EN equivalente)}}
$$

Como o BPE do GPT-2 é treinado nativamente em inglês, vocábulos portugueses fragmentam-se em
mais sub-tokens (ex.: `vinte e nove virgula cinco` ≈ 6–8 tokens vs `twenty nine point five`
≈ 5 tokens). O benchmark de Fase A reporta a inflação por documento
(`run_etth1_texto.py:406-420`), o que justifica a vantagem dos formatos EN/ES.

> **Resumo Fase A**: série real (timestamps + valores) → documento de linguagem natural em um
> template/idioma escolhido, com **zero algarismos** → vetor de ids BPE do GPT-2 (com métrica
> de inflação).

---

## 6. FASE B — Paginação e tokenização (chunking)

### 6.1 O problema

O documento de uma série longa (ex.: 2000 pontos por extenso) gera muito mais tokens do que a
janela de contexto do GPT-2 e, principalmente, do que o buffer do modelo
(`n_paginas_max × block_max = 7 × 128 = 896` tokens). É necessário **paginar** o documento em
pedaços (páginas) que preservem a continuidade da série.

### 6.2 As três estratégias — `testes_letras/texto_puro/paginacao.py`

A classe `TestadorPaginacao` (construtor: `block_max=128`, `overlap_frac=0.15`,
`n_paginas_max=7`) implementa:

| Estratégia | Mecanismo | Paginação | Descarte | Continuidade |
|---|---|---|---|---|
| **1. Truncamento Direto** (`truncamento_direto`, `:113`) | Mantém os **primeiros** `block_max` tokens do documento | 1 página | `tokens_originais − block_max` | ❌ perde todo o histórico remoto |
| **2. Janela Deslizante + Overlap** (`janela_deslizante`, `:129`) | Janela de `block_max` tokens com passo `block_max − ⌈block_max·overlap⌉` | N páginas | **zero** | ✅ bordas repetidas em 15% preservam o encadeamento semântico entre páginas |
| **3. Compressão por Keywords** (`compressao_keywords`, `:155`) | Remove stopwords **antes** de tokenizar (listas PT/EN/ES, casadas após normalização NFKD + remoção de acentos/pontuação) | N páginas | tokens das stopwords removidas | ⚠️ mantém carga semântica (meses, horas, valores) mas perde conectivos |

O dispatcher `paginar()` cronometra cada estratégia (`tempo_tokenizacao_s`).

### 6.3 Adaptação para o modelo — `para_modelo` (`paginacao.py:207`)

Produz o tensor que o modelo consome:

```
ids  : [p0 | pad…pad] + [p1 | pad…pad] + … + [p6 | pad…pad]   (7 × 128 = 896 ids)
lens : [len(p0), len(p1), …, len(p6)]                          (0 = página vazia)
```

- Cada página é truncada em `block_max` e **padded** até o limite (pad = 50256);
- Páginas além de `n_paginas_max` são **descartadas** e contabilizadas
  (`paginas_descartadas`);
- `lens=0` → o *masked mean pooling* da Fase C zera a contribuição da página.

### 6.4 Semântica do chunking no benchmark

No `run_etth1_texto.py`, o **mesmo documento paginado** acompanha **todas** as janelas
numéricas do pool (`construir_dataset`, `:189`): o contexto textual é fixo por cenário
(estratégia × idioma) e o alvo supervisionado vem exclusivamente do ramo numérico. Isso isola
o efeito da estratégia de paginação na retenção da tendência.

> **Resumo Fase B**: documento → páginas ≤ 128 tokens (por truncamento, janela+overlap ou
> keywords) → 1 tensor plano de 896 ids + vetor de 7 comprimentos, com custo medido em tokens
> retidos/descartados.

---

## 7. FASE C — Predição no espaço contínuo e cálculo de erro (Output & Loss)

### 7.1 O forward multimodal passo a passo — `run_benchmarks.py:451` (`_forward_multimodal`)

#### Passo A — Rota numérica (sempre primeiro, independente do texto)

1. **RevIN normalize** (`StandardRevIN`, `run_benchmarks.py:191`):

   $$
   \mathbf{x}_{norm} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
   \mu = \text{mean}_L(\mathbf{x}) .\text{detach()}, \quad
   \epsilon = 10^{-5}
   $$

2. **Patching**: `unfold` não-sobreposto em `token_len = seq_len − label_len = 96` →
   `patches (B, 7, 96)` (ou overlap com stride 48 → 13 patches).
3. **Encoder MLP** (2 camadas ocultas de 256, tanh, dropout) → `num_tokens (B, 7, 768)`.
4. **SMETimes**: `num_tokens = gate(num_tokens, compute_token_stats(patches))`
   (porta adaptativa de volatilidade).

#### Passo B — Rota textual (wte congelado + masked mean pooling)

1. `emb = gpt2.wte(x_ids)` → `(B, 896, 768)`.
2. Reorganiza por bloco: `(B, 7, 128, 768)`.
3. **Masked mean pooling** por bloco (anula o padding):

   $$
   \mathbf{t}_{txt,i} = \frac{1}{\text{lens}_i} \sum_{j < \text{lens}_i} \mathbf{e}_{i,j},
   \qquad \text{mask}_{i,j} = \mathbb{1}[\, j < \text{lens}_i \,]
   $$

   → `txt_tokens (B, 7, 768)` — **um vetor textual por bloco de 96 passos**, alinhado
   um-a-um com os tokens numéricos.
4. Se `T_txt ≠ T_num` (modo overlap: 7 vs 13), interpolação linear temporal para o mesmo
   comprimento.

#### Passo C — Fusão e GPT-2

1. `fused = fusion(cat([txt_tokens, num_tokens]))` → `(B, 7, 768)`;
2. `latent = gpt2(inputs_embeds=fused).last_hidden_state` → `(B, 7, 768)`;
3. **Decoder MLP** (Segment Projection) → `(B, 7, 96)`;
4. `reshape → (B, 672)`; `view (B, 672, 1)`;
5. **RevIN denormalize** (aplicado **somente** na reconstrução numérica):
   `y_pred = y_hat · std + mean`.

#### Resposta à pergunta central

> **O GPT-2 gera tokens de texto na saída?**
> **Não.** O GPT-2 é usado como **backbone de representação** sobre embeddings contínuos
> (`inputs_embeds`), não como modelo de linguagem: não há LM head, não há softmax sobre o
> vocabulário, não há amostragem autoregressiva e `use_cache=False`. O texto formatado é
> **condicionamento de entrada no espaço latente** — as palavras entram pelo `wte`, são
> agregadas por pooling e fundidas aos tokens numéricos. A previsão é **100% numérica**,
> gerada pela Segment Projection sobre os hidden states.

### 7.2 Comparação com o alvo e cálculo das métricas — `testes_letras/treino.py`

O pool de janelas (`pool_janelas_etth1`, `run_etth1_texto.py:158`) extrai pares:

- `x` = janela de `SEQ_LEN=672` passos (entrada);
- `y` = janela `[s_end − label_len : s_end − label_len + label_len + pred_len]` =
  **672 passos** (label 576 + pred 96) na **escala real** (sem z-score; o RevIN normaliza
  internamente).

O split 60/20/20 (train/val/test) é determinístico e o treino usa **Adam**
(`lr=1e-3`) + **CosineAnnealingLR** (`T_max=epocas`, `eta_min=1e-8`), com
`criterion = nn.MSELoss()`, gradiente clipado em `max_norm=1.0` e contagem de batches
NaN/Inf (`batches_estourados` — métrica de estabilidade).

Métricas reportadas por época e no teste final:

| Métrica | Definição | Onde |
|---|---|---|
| `loss_janela` | MSE sobre a janela completa (672) | `treino.py:148` |
| `mse_janela` / `mae_janela` | MSE e MAE sobre a janela completa | `treino.py:145` |
| **`mse_pred`** / **`mae_pred`** | MSE e MAE **somente no horizonte** — últimos `pred_len=96` passos: `pred[:, −96:, :]` vs `y[:, −96:, :]` | `treino.py:139` |
| `best_val` | menor `val_mse_pred` entre épocas (critério de convergência) | `treino.py:222` |

### 7.3 Por que o modelo **não precisa** converter palavras em números?

Esta é a propriedade arquitetural mais importante do pipeline:

> **O fluxo de erro é 100% numérico e contínuo.** As palavras ("veintinueve", "vinte e nove
> virgula cinco") nunca saem do espaço de embeddings — elas existem apenas como **vetores
> calculados pelo `gpt2.wte`** e **agregados por média ponderada**. A predição é produzida
> pela Segment Projection sobre hidden states e comparada diretamente com o tensor numérico
> real `y`.

Consequências:

1. **Não há parsing**: não é preciso reconstruir `29.0` a partir de "veintinueve" — a loss
   nunca opera sobre texto;
2. **Não há perda por imprecisão de conversão**: `num2words` só afeta a *representação de
   entrada* (o conditioning), nunca o erro medido;
3. **Gradientes contínuos**: o erro propaga pelo decoder → GPT-2 → fusão → encoder e gate,
   todo o caminho em aritmética tensorial diferenciável;
4. **Equivalência de avaliação**: MSE/MAE são calculados exatamente como no pipeline
   puramente numérico — o que torna justa a comparação entre modalidades.

> **Resumo Fase C**: janelas numéricas da mesma região do texto + páginas do documento →
> forward multimodal (RevIN → patching → SMETimes ‖ wte → pooling → fusão → GPT-2 →
> Segment Projection → RevIN denormalize) → predição contínua `(B, 672, 1)` → MSE/MAE vs alvo
> real, com foco no horizonte de 96 passos.

---

## 8. Perguntas centrais respondidas (FAQ técnica)

### P1. O texto gerado na entrada é usado como "prompt" tradicional?
Sim, no sentido de *conditioning*: ele entra no modelo como **embeddings somados/fundidos** ao
fluxo numérico no espaço latente. Não, no sentido de *prompt engineering* clássico: não há
continuidade autoregressiva de tokens de texto.

### P2. Por que o GPT-2 é congelado?
O AutoTimes demonstra que o transformer pré-treinado já codifica correlações temporais úteis;
congelá-lo (0 camadas descongeladas, `unfreeze_layers=0`) reduz drasticamente o número de
parâmetros treináveis e evita *catastrophic forgetting*. Apenas módulos especializados
(encoder/decoder MLP, fusion, gate) aprendem a tarefa.

### P3. Qual o papel do RevIN no ciclo texto→numérico→texto?
O RevIN **isola a dinâmica local** da série (z-score por janela) e é desnormalizado
**somente na reconstrução numérica** (`run_benchmarks.py:520`) — o texto não participa da
desnormalização, desacoplando as duas modalidades no final do fluxo.

### P4. Por que 7 blocos × 128 tokens = 896 tokens de texto?
`n_blocks_text = seq_len // token_len = 672/96 = 7`. Cada bloco textual cobre os 96 passos do
mesmo patch numérico. O buffer de 896 tokens é a forma compacta e fixa de alimentar o modelo
sem variar dinamicamente o shape do batch.

### P5. O overlap de 15% na janela deslizante não duplica informação?
Sim, deliberadamente: a duplicação da borda entre páginas é a "colagem" que preserva a
continuidade semântica da série quando o modelo lê cada página isoladamente. O custo é pago em
tokens; o benefício é zero descarte e menos quebras de contexto.

### P6. As palavras em espanhol ("veintinueve") não são piores para um GPT-2 inglês?
Há um trade-off: por um lado, vocabulário não-inglês fragmenta em sub-tokens (custo);
por outro, os templates ES Pseudo-JSON/Keywords venceram o benchmark — o que sugere que a
**estrutura de ancoragem** (delimitadores `key time`/`key value`, setas `->`) é mais
decisiva para o condicionamento do que a eficiência lexical, e que o GPT-2 mistura
sub-tokens multilingues de forma robusta.

### P7. Qual a diferença entre a VolLoss do checkpoint e o MSE do loop de convergência?
O checkpoint `Run-8` foi otimizado com VolLoss (Log-Cosh ciente de volatilidade). Os loops
de comparação usam MSE puro por comparabilidade. A combinação "arquitetura treinada com
VolLoss + avaliação por MSE/MAE" é o que define o cenário `smetimes` do ablation.

---

## 9. Geometria e shapes (tabelas de referência)

### 9.1 Constantes (`testes_letras/dados.py:33`)

| Constante | Valor | Significado |
|---|---|---|
| `SEQ_LEN` | 672 | Janela de entrada (28 dias horários) |
| `LABEL_LEN` | 576 | Contexto observado (24 dias) |
| `PRED_LEN` | 96 | Horizonte de previsão (4 dias) |
| `TOKEN_LEN` | 96 | Patch = seq − label = 1 dia |
| `BLOCK_MAX` | 128 | Tokens por página textual |
| `n_blocks` | 7 | `seq_len // token_len` = páginas/blocos |
| buffer textual | 896 | `7 × 128` ids |

### 9.2 Shapes no forward (`run_benchmarks.py:451`)

| Tensores | Shape |
|---|---|
| `x_enc` (janela bruta) | `(B, 672, 1)` |
| `patches` (patching) | `(B, 7, 96)` — overlap: `(B, 13, 96)` |
| `num_tokens` (encoder MLP) | `(B, 7, 768)` |
| `stat_feats` (SMETimes) | `(B, 7, 4)` |
| `x_ids` (texto paginado) | `(B, 896)` |
| `emb` (wte) | `(B, 896, 768)` → `(B, 7, 128, 768)` |
| `txt_tokens` (pooling) | `(B, 7, 768)` |
| `fused` | `(B, 7, 768)` |
| `latent` (GPT-2) | `(B, 7, 768)` |
| `dec` (Segment Projection) | `(B, 7, 96)` |
| `y_pred` (desnormalizado) | `(B, 672, 1)` |
| `y` (alvo real) | `(B, 672, 1)` |

---

## 10. Trade-offs e interpretação dos resultados

1. **Token inflation vs. informação**: formatos verbosos maximizam legibilidade mas inflam o
   contexto; formatos compactos/estruturados comprimem mais tendência por token — o
   `text_step` auto-ajustado e o filtro de stopwords automatizam esse equilíbrio.
2. **Página única vs. múltiplas**: truncamento é barato mas perde histórico; janela+overlap
   preserva tudo ao custo de redundância; keywords reduz tokens ao custo de conectivos.
3. **Métrica de estabilidade**: `batches_estourados` mede quantas iterações produzem
   NaN/Inf — texto puro é mais estável que o pipeline numerico sob AMP (fp16 estoura com o
   `fusion` aleatório; por isso o perfil padrão é FP32+TF32).
4. **Fusão textual como regularização**: o texto é o mesmo documento para todas as janelas do
   pool; ele condiciona a representação sem impor o valor exato, o que força o modelo a
   generalizar a tendência local da série.

---

## 11. Glossário

| Termo | Definição |
|---|---|
| **RevIN** | Normalização por janela (z-score) com desnormalização reversível na saída |
| **Patching** | Segmentação da série em patches de `token_len` passos (tokens implícitos) |
| **Segment Projection** | MLP decoder (inverso do encoder) que mapeia hidden states de volta ao espaço de patches |
| **SMETimes** | Porta sigmoidal que funde embeddings temporais com estatísticas por patch (média, var, tendência OLS, range) |
| **VolLoss** | `VolatilityAwareLogCoshLoss`: Log-Cosh ponderado por `exp(−β·var)` |
| **Fusion** | `Linear(2D→D) + LayerNorm` que funde embeddings textuais e numéricos antes do GPT-2 |
| **WTE** | *Word Token Embedding* do GPT-2 (tabela de embeddings congelada) |
| **Masked Mean Pooling** | Agregação por bloco dos embeddings textuais, anulando o padding via máscara |
| **Página (chunk)** | Fatia de até `block_max` tokens do documento textual |
| **Taxa de inflação** | `tokens_pt / tokens_en` para conteúdo equivalente |
| **Horizonte (pred_len)** | Últimos 96 passos da janela — foco da avaliação MSE/MAE |
| **Estouro (batch)** | Iteração descartada por loss/gradiente NaN ou Inf |

---

## 12. Mapa de arquivos do código

| Arquivo | Papel |
|---|---|
| `testes_letras/run_etth1_texto.py` | Script orquestrador das Fases A, B e C sobre o ETTh1 real |
| `testes_letras/texto_puro/paginacao.py` | Testador de paginação (3 estratégias) + `para_modelo` |
| `testes_letras/texto_puro/extenso.py` | Conversão 100% por extenso via `num2words` (PT/EN/ES, 9 templates) |
| `testes_letras/texto_puro/estruturado.py` | Templates estruturados (Estrutural, Pseudo-JSON, Relacional, Temporal Delta) |
| `testes_letras/templates.py` | 4 templates de compressão + auto-ajuste de `text_step` |
| `testes_letras/texto_puro/idioma.py` | Medição da inflação de tokens PT/EN |
| `Testes_eficiencia/modules/textual.py` | Conversor pt-BR legado + gerador de dataset textual (CSV/.pt) |
| `Testes_eficiencia/run_benchmarks.py` | `AutoTimesModel` (forward multimodal, fusion, SMETimes, RevIN) |
| `Testes_eficiencia/modules/volatility_loss.py` | `VolatilityAwareLogCoshLoss` |
| `Testes_eficiencia/modules/volatility_gate.py` | Porta de volatilidade (suporte VolLoss) |
| `Testes_eficiencia/modules/robust_revin.py` | RevIN robusta (ablation) |
| `Testes_eficiencia/modules/overlapping_embed.py` | Embedding de patches com overlap |
| `testes_letras/treino.py` | Loop de convergência: Adam + CosineAnnealing, MSE/MAE por janela e horizonte |
| `testes_letras/modelos.py` | Montagem dos checkpoints `smetimes`/`amd` em modo textual |
| `testes_letras/dados.py` | Constantes (SEQ/LABEL/PRED/TOKEN_LEN), carregamento e split dos dados |
| `models/AutoTimes_Gpt2.py` | Arquitetura original do AutoTimes (referência) |
| `models/AutoTimes_Gpt2_SMETimes.py` | Variante SMETimes da bateria (referência) |

---

*Documento gerado a partir da análise direta do código-fonte do repositório AutoTimes
(modificações VolLoss + SMETimes, ambiente `testes_letras`). Última revisão: agosto/2026.*
