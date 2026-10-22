# Relatório de pesquisa: alterar o embedding e identificar variáveis de confundimento em séries temporais

**Data:** 22/10/2026 · **Contexto:** projeto AutoTimes, change `prompt-entrada-vmd-self-refine` (fechada)
**Objetivo:** (1) mapear como a literatura altera o pathway texto→embedding em forecasting condicionado por texto; (2) mapear como a literatura identifica variáveis de confundimento (confounders) em séries temporais; (3) amarrar os dois achados ao que o **nosso processo experimental** descobriu, para orientar a busca de artigos e os próximos experimentos.

---

## 0. Resumo executivo

1. **Nosso processo esgotou o nível "texto"** — 18 variantes de escrita do prompt, 12 treinadas, **zero vencedoras** em 3 seeds. A literatura de 2024–2026 confirma: mudar *como* o número é escrito no texto muda pouco; o gargalo real é a **interface numérica do embedding** (tokenização BPE do GPT-2 não foi feita para números). Há uma família nova de papers (TempoWAVE, FoNE, xVal, BitTokens, Chronos-quantization) que troca exatamente essa interface — com ganhos reportados.
2. **Nosso experimento `data=nenhuma` foi, na prática, uma sonda de confundidor**: o carimbo `/time` é função determinística do índice — um "shortcut" clássico. A literatura de causal discovery em séries (PCMCI/LPCMCI/J-PCMCI+, tigramite) e de deconfounding para forecasting dá o ferramental formal para esse tipo de pergunta.
3. **Existe um paper que une os dois lados**: CVAformer (2026) trata os componentes dinâmicos da série como **confundidores do alinhamento** com embeddings de LLM e desentrelaça antes de alinhar. É o framework mais próximo do que fazemos.

---

## 1. O que o processo mostrou (base empírica deste relatório)

| referência | test_mse (3 seeds, 10 ép) | nota |
|---|---|---|
| numérico (só z-score) | **0.353926** | melhor modelo |
| CbC padrão (6 casas, extenso) | 0.360573 | melhor textual |
| PJ padrão (data=completa) | 0.362145 | — |
| PJ `data=nenhuma` | 0.3638 ± 0.0050 | venceu no single-seed 2021 (0.356939), perdeu na média |
| alvo | 0.3560 | — |

Fatos estabelecidos:
- **18 variantes de escrita** (5 políticas de data + 3 eixos PJ + 4 eixos CbC + …), 12 treinadas: nenhuma vence o padrão em média de 3 seeds.
- **Custo ≠ qualidade**: `digitos` (−63% tokens) piorou; `pc0` (o mais barato, 10,1 tok/ponto) perdeu para `nenhuma` comum. Cortar tokens só ajuda quando o cortado é redundância (a data).
- **Variância de seed (±0,005) > qualquer efeito de escrita (≤0,002)**.
- Gap texto→numérico ≈ 0,0067 — **não resolvido por nenhuma forma de escrever o mesmo conteúdo**.
- Self-Refine (gate de validação de geração) testado de ponta a ponta; fail-fast e proveniência implementados.

**Interpretação que a literatura sustenta:** o modelo condicionado por texto está sendo limitado por *como os números viram vetores* (interface), não por *qual palavras usamos* (escrita). O próximo degrau legítimo é alterar o embedding — e é exatamente a Parte A abaixo.

---

## 2. Parte A — Como alterar o embedding (texto/número → vetor)

### 2.1 Escada de intervenção (do mais barato ao mais invasivo)

```
Nível 0  mudar o TEXTO                 → já esgotado pelo processo (18 variantes)
Nível 1  mudar a TOKENIZAÇÃO de números (mesmo GPT-2)
Nível 2  mudar a INTERFACE numérica do embedding (número → vetor, antes do transformer)
Nível 3  mudar/trocar o ENCODER de texto (outro LM, projeção treinável)
Nível 4  treinar o embedding DO ZERO (ou continued pretraining)
```

### 2.2 Nível 1 — Tokenização de números mantendo o LM

| paper | id / venue | ideia | por que é relevante aqui |
|---|---|---|---|
| **xVal** | arXiv:2310.02989 (NeurIPS PhysSci W) | token contínuo: `E(token) + valor × E(unidade)` — um número = um token, escala preservada | mapeia direto nosso `pc0`/`inteiro`: mantém precisão sem fragmentar |
| **FoNE** | arXiv:2502.09741 | features de Fourier por número, single-token, 2 dims/dígito | mostra que a ordem numérica pode ser embutida com poucas dims |
| **BitTokens** | arXiv:2510.06824 | número = 1 token via IEEE 754 | alternativa discreta; eficiência extrema |
| **TabGemma** | arXiv:2511.03570 | canoniza números em notação científica assinada + continued pretraining | evidência de que "canonicalizar o formato" ajuda, mas sozinho não basta |
| **When Tokenization is Secretly Output Supervision** | arXiv:2609.01386 (EMNLP 2026) | tokenização = definição do problema (supervisão de saída) | **explica nosso achado CbC**: mudar a granularidade da escrita muda o problema de aprendizado, não só o custo — e comparações entre formatos são comparações entre "tarefas" |
| **Chronos** | arXiv:2403.07815 (ICML 2024) | escala + quantiza a série em bins de vocabulário (4096) e treina T5 com cross-entropy | o caminho "não usar BPE de língua para números" mais maduro e com código aberto |

### 2.3 Nível 2 — Interface numérica do embedding (número → vetor direto)

| paper | id / venue | ideia | destaque |
|---|---|---|---|
| **TempoWAVE** | arXiv:2606.26487 (IJCAI 2026) | **plug-and-play**: substitui as representações de token dos números por embeddings dígito-a-dígito via multi-wavelet | tese literal: *"the numeric interface is the key bottleneck"*; SOTA em 5 benchmarks context-enriched; código + modelo abertos |
| **TimesFM** (Google, 2024) | arXiv:2310.10688 (conhecimento geral, não verificado nesta busca) | patch embedding numérico (janela → vetor), LM decoder-only | família "patch embedding" — similar em espírito ao que o próprio TimesNet faz com a série |
| **FreSia** | arXiv:2610.05726 (Pattern Recognition 2026) | projeta estruturas de frequência (T/S/R!) no espaço semântico do LM como prompt | **muito próximo do AutoTimes**: nossa VMD gera exatamente T/S/R; eles projetam o domínio de frequência em vez de serializar dígitos |
| **SMMD loss** | arXiv:2606.27731 | perda com kernel de distância sobre tokens numéricos (MMD suave) | altera a *supervisão* do número, não o embedding — complementar |

### 2.4 Nível 3 — Trocar/acoplar o encoder de texto

| paper | id / venue | ideia | destaque |
|---|---|---|---|
| **Time-LLM** | arXiv:2310.01728 (ICLR 2024) | reprogramming: patches da série → protótipos de texto → LLM congelado + PaP + projeção | arquétipo clássico do pathway "série vira texto e entra num LM" |
| **Are Language Models Actually Useful for TSF?** | arXiv:2406.16964 (**NeurIPS 2024 Spotlight**) | ablações: remover o LLM ou trocar por uma attention básica **não degrada** (muitas vezes melhora) | **eco direto do nosso resultado**: 18 formas de escrever → mesma MSE. Eles concluem que a estrutura (patching/attention) importa mais que o LM pré-treinado |
| **Random Initialization Can't Catch Up** | arXiv:2506.21570 | transferência de LM só aparece em regime de poucos dados; design (tokenizer, backbone, post-training) importa | delimita QUANDO o embedding pré-treinado paga (low-data) |
| **DP-GPT4MTS** | arXiv:2508.04239 | prompt duplo: embeddings do texto refinados por self-attention + FFN antes da fusão | caminho intermediário: "embutir e então refinar" |
| **LLM-Mixer** | arXiv:2410.11674 | decomposição multiescala + LLM congelado guiado por prompt | parente direto do AutoTimes (decomposição + LM) |
| **Context information > reasoning** | arXiv:2502.05699 | contexto bem dado ≈ melhor prompt de raciocínio; LLMs erram álgebra | consistente com `data=nenhuma` (o carimbo não ajudava) e com nossa dificuldade de ganho via "semântica" |
| **Tang et al. (SIGKDD Explorations)** | arXiv:2402.10835 | LLMs preferem séries com padrão/periodicidade; conhecimento externo + paráfrases ajudam | onde o texto pode ajudar de verdade: **informação externa**, não a forma |

### 2.5 Nível 4 — Do zero / continued pretraining
- **Chronos-T5** (abaixo do 2.2) e **LLMTime** (Gruver et al., NeurIPS 2023, arXiv:**2310.07820**): forecasting zero-shot só com strings de dígitos. Mostram que o LM congelado *consegue* extrair padrão de números serializados — mas com interface dedicada, não BPE de língua. Nota do próprio paper (verificado em 22/10): **"GPT-4 pode ser pior que GPT-3 por como tokeniza números"** — evidência direta da tese da interface numérica.
- **Dissecting Chronos** (arXiv:2603.10071, ICLR 2026 TSALM W): SAEs sobre o Chronos mostram hierarquia causal de features (frequência cedo, change-detection no meio). Ferramenta para auditar *o que* o embedding numérico carrega.

### 2.6 Leitura para o AutoTimes (o que eu tentaria, em ordem)

1. **Quantização tipo Chronos dentro do nosso pipeline** (mais barato e fiel ao espírito do projeto): em vez de "zero ponto nove sete…" (CbC) ou "/t 15.19" (PJ), mapear cada valor z para 1 token de bin (vocab. ~4096) e treinar a tabela de embeddings junto com o TimesNet. Teste de 3 seeds: custo ~1 token/ponto vs 15–28 hoje. *Esse experimento foge de "prompt-only" — é a nova alavanca.*
2. **xVal/FoNE-style mantendo GPT-2**: substituir a embedding dos tokens numéricos por `E(token) + valor·E(unit)` (ou features de Fourier) só na região numérica do documento; GPT-2 segue congelado. Cirúrgico, comparável com o baseline.
3. **Projeção de frequência (FreSia-style)**: em vez de serializar T/S/R em palavras, projetar os espectros/coeficientes de cada componente como pseudo-tokens. Usa a VMD que já temos.
4. **Guarda de literatura**: antes de concluir ganho, lembrar do NeurIPS 2024 "Are LMs actually useful" — toda mudança de interface deve ser comparada contra **uma attention treinável simples** (a ablação deles), senão o ganho pode ser da estrutura, não do LM.

---

## 3. Parte B — Identificação de variáveis de confundimento em séries temporais

### 3.1 Definição operacional (para buscar artigos)

Em séries temporais, **confundidor** é uma variável (observada ou latente) que afeta simultaneamente os preditores e o alvo, criando associação espúria. Autocorrelação e sazonalidade agravam: todo carimbo de tempo é função determinística do índice ⇒ é confundidor/shortcut canônico. Três perguntas práticas da literatura:
1. **Descoberta causal**: quais variáveis são pais causais do alvo (e em que lag)? → PCMCI e familiares.
2. **Confounders não observados**: como ajustar sem vê-los? → fatores latentes, knockoffs, SCEVAE, LPCMCI.
3. **Robustez do preditor**: como prever sem usar correlações espúrias? → backdoor/frontdoor, invariância, deconfounding antes do forecasting.

### 3.2 Família 1 — Descoberta causal constraint-based (com confundidores latentes)

| paper / método | id / venue | ideia | ferramenta |
|---|---|---|---|
| **PCMCI** | Runge et al., *Science Advances* 2019 (citação de memória, âncora clássica) | PC + MCI: testes de independência condicional em painel temporal, controla autocorrelação e multipicidade | **tigramite** (github.com/jakobrunge/tigramite) |
| **LPCMCI** | arXiv:2007.01884 (JMLR) | PCMCI para confundidores latentes: alta recall mesmo com autocorrelação forte | tigramite |
| **Avaliação LPCMCI** | arXiv:2209.03427 | benchmark honesto: LPCMCI vence aleatório, mas longe do ótimo em lags defasados | leitura de expectativas |
| **J-PCMCI+** | arXiv:2306.12896 (Runge group) | vários datasets do mesmo sistema; contextos latentes/temporais viram ajuste | multi-série (ex.: ETTh1 + ETTh2 + ETT) |
| **Causal ancestral graphs** | arXiv:2112.08417 (*Annals of Statistics*) | classe de grafos que representa melhor confundidores latentes em séries | teoria (identificação) |

### 3.3 Família 2 — Granger profundo e métodos paramétricos

| paper | id / venue | ideia |
|---|---|---|
| **Deep Granger com confundimento latente** | arXiv:2202.11286 (ESWA) | dual-decoder RNN, confundidor latente explícito na parametrização |
| **SCEVAE + knockoff interventions** | arXiv:2209.11497 (NeurIPS W) | VAE sequencial + knockoffs para counterfactuals sob confundidor oculto |
| **ORACLE-VARX** | arXiv:2610.05618 (2026) | double/debiased ML (DML) remove efeitos não-lineares do confundidor + lags adaptativos + FDR | 
| **CIC — invisible confounders** | arXiv:2408.05584 | decompõe variáveis observadas em subespaços comum/privado no espaço de delay-embedding; reconstrói confundidores invisíveis |
| **VAR-LiNGAM** | Hyvärinen et al. (conhecimento geral) | não-Gaussianidade orienta arestas; usado como baseline nos papers acima |

### 3.4 Família 3 — Deconfounding aplicado a forecasting (a mais próxima do nosso pipeline)

| paper | id / venue | ideia | leitura para nós |
|---|---|---|---|
| **Deconfounded TSF: A Causal Inference Approach** | arXiv:2410.21328 (Gao et al.) | representa confundidores latentes a partir do histórico e os injeta no preditor | baseline da família "aprender o confundidor e condicionar" |
| **Climate bias correction (factor model)** | arXiv:2408.12063 (IJCAI 2025) | fator latente multi-causa aprendido de observações + simulação | mesma receita, domínio climate |
| **LUCID** | arXiv:2609.31315 (**Bloomberg**, 2026) | detecta o "regime de confundimento" via espectro de Marčenko-Pastur e aplica a estratégia certa; melhora qualquer motor de descoberta | ferramenta moderna + código (github.com/bloomberg/causal-ts) |
| **Causal-Audit** | arXiv:2604.02488 | valida suposições (estacionariedade, proxies de confundimento…) e **recomenda ou se abstém** | como auditar antes de confiar num grafo |
| **Caformer** | arXiv:2403.08572 | backdoor adjustment contra correlação espúria de "ambiente"; 5 tarefas de TS | arquitetura de forecasting com ajuste causal embutido |
| **Casper** | arXiv:2403.11960 (CIKM 2024) | frontdoor adjustment + atenção espaciotemporal causal (imputation) | frontdoor em prática |
| **Invariant Forecasting (smart cities)** | arXiv:2405.05430 | representações invariantes sob domain shift (IRM-style) | família invariância |
| **Causal motion forecasting** | arXiv:2111.14820 (CVPR 2022) | formalismo: variáveis **invariantes**, **confounders de estilo**, **spurious features** — cada grupo com tratamento separado | melhor taxonomia para explicar nosso `data=nenhuma` |

### 3.5 A ponte entre as duas partes (achado importante desta busca)

| paper | id | ideia |
|---|---|---|
| **CVAformer** | arXiv:2606.08262 (2026) | em forecasting com LLM, os **componentes dinâmicos da série atuam como confundidores do alinhamento** com embeddings: influenciam tanto os componentes invariantes quanto o embedding alinhado. Solução: desentrelaça cada variável em componente invariante + dinâmico *antes* do alinhamento e aplica intervenção causal; ainda troca attention causal por não-causal (interação entre variáveis no mesmo passo). |
| **FreSia** | arXiv:2610.05726 | alinha domínio de frequência × semântica do LM (mesma família "interface numérica", vista pelo lado causal/semântico) |
| **Time2time** | arXiv:2509.05801 (NeurIPS W) | intervenção causal em hidden states de TSFMs (Chronos/Toto): conceitos semânticos (regime de mercado) são manipuláveis — evidência de que o embedding numérico carrega conceitos, não só valores |

---

## 4. Como aplicar isso ao processo do AutoTimes

### 4.1 Sonda de confundidor (família B) — perguntas experimentais baratas

1. **Formalizar o `data=nenhuma`**: o carimbo `/time` é função determinística do índice ⇒ rodar um Granger/PCMCI clássico mostrando que `/time` não adiciona informação sobre o alvo dado o índice é a versão "paper" do nosso resultado 3-seeds. Artigo-modelo para citar: Runge et al. 2019 (PCMCI) + a taxonomia do CVPR 2022 (invariantes × spurious).
2. **Descobrir os pais reais de HUFL no ETTh1**: PCMCI+ via `tigramite` (pip, análise offline — não entra no protocolo). Pergunta: os pais causais de HUFL (em lags) explicam o ganho do modelo numérico? Se sim, incluir *só* essas variáveis como contexto textual (em vez das 7 features) testa "informação > densidade" com base causal.
3. **J-PCMCI+ multi-dataset**: ETTh1/2/4 como réplicas do mesmo sistema — contextos observados/latentes. Ajuda a distinguir "padrão real" de "sorte de seed" (o mesmo problema que nos pegou no K=5 e no `nenhuma`).

### 4.2 Alterar o embedding (família A) — desenho de experimento

- **Hipótese de trabalho** (sustentada por TempoWAVE 2026 e pelo nosso esgotamento do nível texto): *a interface BPE→número é o gargalo dos 0,0067.*
- **Protocolo de teste**: manter os mesmos 896 tokens/3+2+2/10 ép/3 seeds; a única mudança é o mapeamento número→vetor; comparar contra (a) baseline textual padrão e (b) a **attention treinável simples** (ablação obrigatória do NeurIPS 2024).
- **Ordem sugerida**: quantização-bin (1) → xVal/FoNE cirúrgico no GPT-2 (2) → projeção de frequência T/S/R (3).
- **Custo**: quantização ~1 tok/ponto (≈56–89 pontos/hora vs 27,8) — mesmo ganho de densidade que `pc0` tentou, mas sem perder precisão (a precisão mora no embedding, não no texto).

### 4.3 Onde ler primeiro (top 6 para caçar)

1. **TempoWAVE** — arXiv:2606.26487 (IJCAI 2026) — a tese da interface numérica.
2. **Are LMs actually useful?** — arXiv:2406.16964 (NeurIPS 2024 Spotlight) — o ceticismo metodológico que confirma nosso resultado.
3. **CVAformer** — arXiv:2606.08262 — a ponte embedding × confundidor.
4. **Chronos** — arXiv:2403.07815 — quantização de série como LM, com código.
5. **LPCMCI** — arXiv:2007.01884 + **tigramite** — descoberta causal com confundidores latentes.
6. **Deconfounded TSF** — arXiv:2410.21328 — confundidor latente como representação condicionante no forecasting.

### 4.4 Consultas de busca sugeridas (para achar mais)

- `numeric interface time series LLM embedding` · `numeric tokenization forecasting` · `value tokenizer time series foundation model`
- `temporal confounder identification` · `causal discovery autocorrelated time series latent confounders` · `deconfounding time series forecasting`
- `invariant risk minimization time series` · `shortcut learning time series forecasting` · `spurious correlation time series environment`
- `Peters Bühlmann invariant prediction` (âncora clássica, de memória) · `Granger causality latent confounder`
- Veis: **NeurIPS/ICLR/UAI workshop "Causal Inference for Time Series Data"**, KDD, *Science Advances* (linha Runge), *Journal of Causal Inference*; datasets/benchmarks: **TimeGraph**, **CausalTime** (usados pelo Causal-Audit).

---

## 5. Referências levantadas (índice com arXiv)

**Embedding/tokenização:** xVal 2310.02989 · FoNE 2502.09741 · BitTokens 2510.06824 · TabGemma 2511.03570 · Tokenization-as-Supervision 2609.01386 · Chronos 2403.07815 · TempoWAVE 2606.26487 · FreSia 2610.05726 · SMMD 2606.27731 · Time-LLM 2310.01728 · Are-LMs-useful 2406.16964 · LM-Transfer 2506.21570 · DP-GPT4MTS 2508.04239 · LLM-Mixer 2410.11674 · Context>Reasoning 2502.05699 · Tang-et-al 2402.10835 · Noise-Injection 2512.20140 · Dissecting-Chronos 2603.10071 · LLMTime **2310.07820** (verificado) · (TimesFM 2310.10688 — de memória)

**Confounders/causal:** LPCMCI 2007.01884 · LPCMCI-eval 2209.03427 · J-PCMCI+ 2306.12896 · Ancestral-graphs 2112.08417 · Deep-Granger 2202.11286 · SCEVAE-knockoffs 2209.11497 · ORACLE-VARX 2610.05618 · CIC 2408.05584 · DSCM 2406.01161 · Causal-ID-TS 2504.20172 · Causal-Audit 2604.02488 · LUCID 2609.31315 · Deconfounded-TSF 2410.21328 · Climate-factor 2408.12063 · Caformer 2403.08572 · Casper 2403.11960 · Invariant-Smart-Cities 2405.05430 · Causal-Motion 2111.14820 · CVAformer 2606.08262 · time2time 2509.05801 · (PCMCI Science Advances 2019, VAR-LiNGAM, IRM/Peters-Bühlmann — de memória)

---
*Busca feita na API do arXiv em 22/10/2026; links `arXiv:NNNN.NNNNN` resolvem para `https://arxiv.org/abs/NNNN.NNNNN`. Itens marcados "de memória" não passaram por verificação nesta rodada.*
