# AUDITORIA FORENSE DE TREINAMENTO — AUTOTIMES COM DADOS TEXTUAIS

Você é um engenheiro sênior de Machine Learning especializado em:

- PyTorch;
- séries temporais;
- modelos fundacionais e embeddings textuais;
- AutoTimes ou arquiteturas semelhantes;
- depuração de autograd;
- estabilidade numérica;
- avaliação experimental;
- auditoria de pipelines de dados;
- desenho e validação de estudos de ablação.

Sua tarefa não é apenas sugerir causas possíveis. Você deverá:

1. inspecionar o log, o código, as configurações e os dados disponíveis;
2. reproduzir o comportamento observado;
3. instrumentar o treinamento;
4. executar testes controlados;
5. localizar as causas com evidências;
6. corrigir os problemas encontrados;
7. executar novamente os experimentos;
8. demonstrar, por resultados antes/depois, se cada correção funcionou.

Não assuma previamente que existe explosão de gradiente.

A conclusão “explosão de gradiente” somente poderá ser emitida se houver evidência direta nos gradientes, como crescimento abrupto de várias ordens de magnitude, gradientes não finitos, instabilidade crescente da loss ou falha correspondente no backward/optimizer.

---

# 1. ARQUIVOS DE ENTRADA

Analise todos os arquivos disponíveis, especialmente:

- `training_history_20260807_234146.log`;
- scripts de treinamento;
- definição do modelo;
- implementação do dataset e dos DataLoaders;
- implementação das losses;
- implementação do `RobustRevIN`;
- implementação do `OverlappingEmbed`;
- implementação do `VolatilityGate`;
- configuração do `DiscrimOptimizer`;
- scripts de avaliação;
- arquivos YAML, JSON ou argumentos CLI;
- código de tokenização e geração/carregamento de embeddings;
- cache de embeddings, se existir;
- amostras ou metadados dos conjuntos de treino, validação e teste;
- `requirements.txt`, `environment.yml`, `pyproject.toml` ou equivalente.

Caso os caminhos não estejam informados, procure-os automaticamente no repositório.

Não interrompa a análise apenas para perguntar onde está um arquivo. Primeiro pesquise pelo nome das classes, flags, argumentos e termos presentes no log.

Se algum artefato essencial realmente não estiver disponível, registre claramente a limitação, mas continue realizando tudo que for possível sem fabricar resultados.

---

# 2. CONTEXTO INICIAL EXTRAÍDO DO LOG

O log apresenta, inicialmente, os seguintes sinais:

1. O baseline inicia aproximadamente com:

   - `TrainLoss = 50.190659`;
   - `ValLoss = 47.277244`;
   - `TestLoss = 24.305275`.

2. Depois de várias épocas, a loss permanece finita e entra em platô aproximadamente em:

   - `TrainLoss ≈ 46.5–46.9`;
   - `ValLoss ≈ 43.7–44.7`;
   - `TestLoss ≈ 21.4–22.7`.

3. Não há, no log fornecido:

   - `NaN`;
   - `Inf`;
   - norma dos gradientes;
   - gradientes por camada;
   - estatísticas de ativações;
   - estatísticas dos embeddings;
   - escala do AMP;
   - magnitude das atualizações dos parâmetros.

4. `Run 1_RobustRevIN` apresenta exatamente os mesmos valores de treino, validação e teste que o baseline durante todas as épocas exibidas.

5. `Run 3_OverlappingEmbed` também apresenta exatamente os mesmos valores que o baseline durante todas as épocas exibidas.

6. `Run 2_VolatilityLoss` muda a escala da função objetivo, portanto seus números não devem ser comparados diretamente com os das outras runs.

7. `Run 4_DiscrimOptimizer` e `Run 5_VolatilityGate` apresentam trajetórias diferentes do baseline, sugerindo que essas configurações afetam o treinamento.

8. O `TestLoss` aparece muito menor que o `ValLoss` durante praticamente todo o treinamento.

9. O conjunto de teste é avaliado em todas as épocas. Verifique se ele está sendo usado direta ou indiretamente para escolher hiperparâmetros, épocas, checkpoints ou variantes.

Esses itens são hipóteses iniciais, não conclusões definitivas. Valide cada um no código e nos experimentos.

---

# 3. REGRAS INEGOCIÁVEIS

## 3.1 Diagnosticar antes de corrigir

Não comece reduzindo arbitrariamente o learning rate ou adicionando gradient clipping.

Primeiro obtenha evidências instrumentais.

Gradient clipping pode ser usado posteriormente como proteção, mas não deve esconder a causa original.

## 3.2 Uma alteração causal por experimento

Não altere simultaneamente:

- learning rate;
- loss;
- normalização;
- clipping;
- arquitetura;
- dados;
- scheduler.

Cada experimento deverá responder a uma hipótese específica.

## 3.3 Comparabilidade

Quando diferentes funções de loss forem usadas, compare os modelos também por métricas comuns calculadas na mesma escala, por exemplo:

- MAE;
- MSE;
- RMSE;
- sMAPE ou MAPE, quando apropriado;
- erro normalizado;
- métricas por horizonte;
- desempenho contra baselines ingênuos.

Não conclua que uma loss de `2.4` é melhor que outra de `21` quando as funções objetivo possuem escalas diferentes.

## 3.4 Uso do teste

O conjunto de teste não deve ser usado para:

- early stopping;
- escolha de checkpoint;
- seleção de hiperparâmetros;
- seleção da melhor ablação;
- decisão de arquitetura.

Use validação durante o desenvolvimento e teste apenas na avaliação final.

## 3.5 Evidência rastreável

Cada conclusão deverá apontar:

- arquivo;
- classe ou função;
- linha ou trecho relevante;
- experimento;
- métrica observada;
- interpretação;
- nível de confiança.

Não produza diagnósticos genéricos sem vinculá-los a evidências.

---

# 4. FASE A — RECONSTRUÇÃO DO PIPELINE

Mapeie o fluxo completo:

```text
fonte dos dados
    ↓
limpeza dos textos
    ↓
tokenização
    ↓
embeddings ou encoder textual
    ↓
alinhamento texto–série–target
    ↓
construção das janelas temporais
    ↓
normalização
    ↓
DataLoader
    ↓
modelo
    ↓
loss
    ↓
backward
    ↓
optimizer
    ↓
scheduler
    ↓
avaliação
    ↓
inverse transform
    ↓
métricas
```

Para cada etapa, registre:

- arquivo e função responsáveis;
- formato e shape dos tensores;
- dtype;
- device;
- escalas esperadas;
- transformações aplicadas;
- diferenças entre treino, validação e teste;
- pontos onde ocorre `detach`;
- pontos onde ocorre cache;
- parâmetros congelados;
- comportamento condicionado por flags.

Crie um diagrama simples do pipeline.

---

# 5. FASE B — REPRODUÇÃO CONTROLADA

Reproduza o baseline com:

- seed fixa;
- configuração registrada;
- versões de Python, PyTorch, CUDA e bibliotecas;
- hash do commit;
- hash ou identificação da configuração;
- número de parâmetros totais;
- número de parâmetros treináveis;
- nomes dos grupos do optimizer;
- learning rate de cada grupo;
- batch size;
- precisão utilizada;
- AMP habilitado ou desabilitado;
- scheduler;
- função de loss;
- estratégia de normalização.

Registre explicitamente:

```text
seed
deterministic_algorithms
cudnn.deterministic
cudnn.benchmark
num_workers
shuffle
drop_last
sampler
```

Execute inicialmente uma reprodução curta, suficiente para verificar se os primeiros valores coincidem com o log.

Se não coincidirem, investigue a diferença antes de prosseguir.

---

# 6. FASE C — AUDITORIA DOS DADOS

## 6.1 Estatísticas dos conjuntos

Para treino, validação e teste, calcule:

- número de amostras;
- número de séries ou entidades;
- intervalo temporal;
- quantidade de janelas;
- média, desvio-padrão, mínimo, máximo e mediana dos targets;
- quantis `p01`, `p05`, `p25`, `p50`, `p75`, `p95`, `p99`;
- amplitude média;
- volatilidade;
- percentual de valores ausentes;
- percentual de zeros;
- percentual de valores constantes;
- quantidade de outliers;
- distribuição dos horizontes;
- distribuição do tamanho das séries.

Compare os três splits.

Verifique se o teste é objetivamente mais fácil, menos volátil ou possui menor amplitude que treino e validação.

## 6.2 Separação temporal e vazamento

Verifique:

- se o split foi realizado antes ou depois da criação das janelas;
- se existem janelas temporais sobrepostas entre os splits;
- se o mesmo período aparece em mais de um split;
- se a mesma entidade apresenta trechos duplicados;
- se existem amostras textuais duplicadas ou quase duplicadas;
- se textos do futuro aparecem em entradas do passado;
- se o texto contém explicitamente valores ou informações do target;
- se há vazamento por normalização;
- se há vazamento por cache;
- se IDs ou timestamps foram alinhados incorretamente.

Para séries temporais, prefira:

```text
split temporal
    antes
da geração das janelas
```

e não o contrário.

## 6.3 Alinhamento entre texto e série

Audite as junções por:

- identificador;
- entidade;
- timestamp;
- janela;
- horizonte;
- ordem dos registros.

Crie um teste manual com pelo menos 10 amostras exibindo:

```text
sample_id
timestamp inicial
timestamp final
texto associado
target esperado
target realmente carregado
```

Verifique possíveis deslocamentos de uma posição, um horizonte ou uma entidade.

---

# 7. FASE D — AUDITORIA DO BANCO TEXTUAL E DOS EMBEDDINGS

Investigue especificamente o componente textual.

## 7.1 Texto bruto

Calcule:

- quantidade de textos vazios;
- textos contendo apenas espaços;
- textos extremamente curtos;
- textos extremamente longos;
- distribuição do número de caracteres;
- distribuição do número de tokens;
- duplicatas exatas;
- duplicatas aproximadas;
- caracteres inválidos;
- problemas de codificação;
- mistura inesperada de idiomas;
- excesso de boilerplate;
- textos constantes associados a targets diferentes;
- textos diferentes associados ao mesmo identificador por erro de join.

## 7.2 Tokenização

Verifique:

- tokenizer efetivamente utilizado;
- versão;
- `max_length`;
- truncamento;
- padding;
- atenção;
- percentual médio de tokens truncados;
- percentual de amostras compostas apenas de padding;
- IDs fora do vocabulário, quando aplicável;
- formato de `attention_mask`;
- se o tokenizer está sendo aplicado igualmente nos três splits.

## 7.3 Embeddings

Antes da primeira camada treinável, registre por batch:

- shape;
- dtype;
- device;
- média;
- desvio-padrão;
- mínimo;
- máximo;
- máximo absoluto;
- norma L2 por amostra;
- quantis das normas;
- percentual de zeros;
- percentual de valores não finitos;
- similaridade de cosseno entre amostras;
- variância entre amostras.

Agregue pelo menos:

```text
p50
p90
p95
p99
máximo
```

Faça isso separadamente para treino, validação e teste.

Investigue:

- embeddings excessivamente grandes;
- embeddings quase constantes;
- embeddings todos iguais;
- embeddings nulos;
- diferenças severas entre os splits;
- embeddings com `NaN` ou `Inf`;
- cache retornando o embedding de outra amostra;
- colisões de chave;
- cache desatualizado;
- embedding salvo em dtype inadequado;
- normalização aplicada duas vezes;
- normalização nunca aplicada;
- encoder textual em `train()` quando deveria estar congelado em `eval()`;
- encoder congelado sem `eval()`, produzindo dropout aleatório;
- encoder que deveria treinar, mas não está no optimizer;
- `detach()` não intencional;
- conversão para NumPy interrompendo o grafo.

## 7.4 Experimentos textuais discriminativos

Mantendo todo o restante idêntico, compare:

1. texto real;
2. embeddings zerados;
3. embeddings constantes;
4. embeddings embaralhados entre amostras;
5. embeddings normalizados por norma L2;
6. modelo sem componente textual;
7. somente componente textual, caso a arquitetura permita.

Use uma métrica comum.

Interpretação esperada:

- desempenho igual com texto real e texto embaralhado: texto provavelmente ignorado;
- desempenho igual com texto real e embedding zero: caminho textual provavelmente inativo;
- queda ao embaralhar: texto contém sinal útil;
- melhora extrema com texto real: verificar possível vazamento;
- instabilidade apenas com embeddings não normalizados: problema de escala provável.

---

# 8. FASE E — INSTRUMENTAÇÃO DOS GRADIENTES

Adicione instrumentação depois de `backward()` e antes de qualquer clipping ou `optimizer.step()`.

Registre:

```text
epoch
batch
loss total
componentes da loss
learning rate por grupo
grad_norm_global_before_clip
grad_norm_global_after_clip
max_abs_grad
min_abs_grad não zero
quantis das normas por camada
número de parâmetros com grad=None
número de parâmetros com gradiente zero
número de gradientes não finitos
AMP scale
parâmetros atualizados
update_norm
weight_norm
update_to_weight_ratio
```

Use hooks ou iteração por `named_parameters()`.

Não grave todos os valores individuais. Registre sumários por módulo, por exemplo:

```text
text_encoder
input_projection
RevIN
embedding_layer
backbone
attention
forecast_head
volatility_gate
```

Utilize temporariamente:

```python
torch.autograd.set_detect_anomaly(True)
```

em uma execução curta de diagnóstico.

Também use:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=float("inf"),
    error_if_nonfinite=True
)
```

apenas para medir a norma global sem aplicar clipping.

Depois da medição, execute um experimento separado com clipping real, por exemplo `max_norm=1.0`, somente se houver justificativa.

---

# 9. CRITÉRIOS DE CLASSIFICAÇÃO DOS GRADIENTES

## 9.1 Explosão confirmada

Classifique como explosão quando houver uma combinação de evidências como:

- gradientes passando abruptamente por várias ordens de magnitude;
- `grad_norm = Inf`;
- gradientes com `NaN`;
- `max_abs_grad` extremo;
- loss tornando-se instável ou não finita;
- falha relacionada no AMP;
- atualização de pesos desproporcional à magnitude dos parâmetros.

Não use apenas “loss alta” como evidência.

## 9.2 Vanishing gradient ou caminho desconectado

Considere essa hipótese quando houver:

- grande quantidade de `grad=None`;
- normas próximas de zero em módulos relevantes;
- pesos que não mudam;
- embeddings que não afetam a saída;
- perda que entra em platô;
- falha no teste de memorização;
- módulo esperado fora dos grupos do optimizer.

## 9.3 Platô de otimização

Considere platô quando:

- gradientes são finitos;
- parâmetros são atualizados;
- as atualizações são pequenas;
- a loss fica estável;
- o modelo consegue memorizar conjunto pequeno, mas não melhora no conjunto completo.

Investigue:

- learning rate;
- scheduler;
- escala dos targets;
- capacidade útil;
- ruído;
- loss;
- baixa informação das features;
- congelamento excessivo;
- mismatch entre encoder e projeção.

## 9.4 Problema de dados ou avaliação

Considere quando:

- splits apresentam escalas muito diferentes;
- normalização é diferente;
- teste é muito mais fácil;
- agregação da loss diverge;
- existe desalinhamento;
- existe vazamento;
- os targets apresentam unidades ou transforms diferentes.

---

# 10. FASE F — AUDITORIA DO OTIMIZADOR

Liste todos os parâmetros:

```text
nome
shape
requires_grad
grupo do optimizer
learning rate
weight decay
gradiente presente?
magnitude da atualização
```

Procure especificamente:

- parâmetros treináveis ausentes do optimizer;
- parâmetros duplicados em grupos;
- learning rate zero;
- scheduler reduzindo o LR precocemente;
- módulos congelados acidentalmente;
- `optimizer.zero_grad()` ausente;
- ordem incorreta de `zero_grad`, `backward`, `step`;
- uso incorreto de gradient accumulation;
- `optimizer.step()` não executado;
- `scaler.step()` ignorado por overflow;
- `scaler.update()` ausente;
- clipping antes de `scaler.unscale_`;
- weight decay excessivo;
- grupos discriminativos configurados incorretamente.

Com AMP, a sequência correta deverá ser conferida:

```python
optimizer.zero_grad(set_to_none=True)

with autocast(...):
    output = model(batch)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)

# medir e, se aplicável, clipar aqui

scaler.step(optimizer)
scaler.update()
```

Execute uma comparação curta com AMP desligado.

---

# 11. FASE G — AUDITORIA DA LOSS E DA AVALIAÇÃO

Verifique:

- mesma função de avaliação entre validação e teste;
- mesma escala;
- mesmo inverse transform;
- mesmo mask;
- mesma redução;
- mesmo denominador;
- mesma ponderação por amostra;
- tratamento igual de batches incompletos;
- diferenças entre `sum` e `mean`;
- regularizadores presentes apenas no treino;
- componentes adicionais da loss;
- critérios distintos por engano.

A média correta deve considerar o número de elementos ou amostras:

```python
total_loss += batch_loss.item() * batch_size
total_items += batch_size
mean_loss = total_loss / total_items
```

Não use:

```python
sum_batch_losses / len(dataset)
```

quando cada `batch_loss` já é uma média.

Para `VolatilityLoss`, registre separadamente:

```text
forecast_loss
volatility_component
regularization_component
total_loss
```

Compare todas as variantes por MAE/RMSE calculados sobre previsões na mesma escala final.

---

# 12. FASE H — VERIFICAÇÃO DAS ABLAÇÕES

Audite especialmente:

```text
use_robust
use_overlap
use_vol_loss
use_disc_optim
use_vol_gate
use_unfreeze
use_hybrid_loss
```

Para cada flag, siga seu caminho:

```text
configuração
    ↓
parser
    ↓
objeto de configuração
    ↓
construtor do modelo
    ↓
atributo interno
    ↓
condicional no forward
    ↓
ativação intermediária
    ↓
saída
    ↓
loss
```

## 12.1 Teste de lote fixo

Carregue:

- os mesmos pesos;
- o mesmo batch;
- mesma seed;
- dropout desabilitado com `eval()`;
- apenas uma flag alterada.

Compare:

```text
input
saída da transformação
embedding
saída do backbone
previsão final
loss
```

Registre:

```text
max_abs_difference
mean_abs_difference
relative_difference
allclose
hash ou checksum
```

Se uma flag deveria modificar o comportamento e a diferença for exatamente zero, classifique como provável `no-op`.

## 12.2 RobustRevIN

Verifique:

- se a instância realmente muda;
- se `use_robust=True` chega ao construtor;
- se a mediana/escala robusta são realmente usadas;
- se logo depois o resultado é sobrescrito;
- se ocorre inverse transform incorreto;
- se a implementação robusta devolve exatamente o mesmo valor da convencional;
- se o módulo está fora do caminho usado no `forward`;
- se o cache foi produzido antes da transformação.

Crie teste unitário com dados contendo outliers, em que RobustRevIN e RevIN convencional necessariamente produzam resultados diferentes.

## 12.3 OverlappingEmbed

Verifique:

- tamanho da janela;
- stride;
- padding;
- número de patches;
- projeção;
- concatenação;
- agregação;
- shape efetivamente usado pelo backbone;
- se a saída overlapping é calculada e depois descartada;
- se a opção é aplicada apenas ao cache antigo;
- se o mesmo embedding pré-calculado está sendo reutilizado;
- se `use_overlap` não chega ao módulo correto.

Crie teste unitário com entrada sintética na qual overlapping e non-overlapping produzam números ou shapes diferentes.

## 12.4 Contadores de execução

Adicione temporariamente contadores ou logs estruturados confirmando:

```text
RobustRevIN branch executed: true/false
OverlappingEmbed branch executed: true/false
VolatilityGate branch executed: true/false
```

Não aceite apenas a configuração impressa no início da run como prova de que o ramo foi executado.

---

# 13. FASE I — TESTE DE MEMORIZAÇÃO DE CONJUNTO PEQUENO

Crie um subconjunto fixo de 16 a 32 amostras.

Execute treinamento com:

- sem augmentation;
- sem shuffle, inicialmente;
- dropout reduzido ou desativado;
- seed fixa;
- modelo em modo correto;
- loss e métricas monitoradas;
- quantidade suficiente de iterações.

A rede possui aproximadamente 125 milhões de parâmetros e deve conseguir reduzir substancialmente o erro nesse pequeno conjunto, salvo restrições arquiteturais explícitas.

Registre:

- loss inicial;
- loss final;
- curva;
- norma dos gradientes;
- magnitude das atualizações;
- previsões e targets de algumas amostras.

Interpretação:

### Se memorizar

O backward e a capacidade básica provavelmente funcionam. Priorize:

- qualidade dos dados;
- generalização;
- normalização;
- distribuição dos splits;
- sinal informativo dos textos;
- hiperparâmetros.

### Se não memorizar

Priorize:

- caminho desconectado;
- parâmetros congelados;
- optimizer incorreto;
- loss incorreta;
- targets desalinhados;
- saturação;
- bug no forward;
- normalização/inverse transform;
- parâmetros que não mudam.

Esse teste é obrigatório.

---

# 14. FASE J — BASELINES SIMPLES

Implemente ou avalie pelo menos:

- previsão do último valor;
- média histórica;
- média móvel;
- modelo linear simples;
- modelo sem texto;
- modelo com texto embaralhado.

Compare as métricas na mesma escala.

Se o modelo complexo não superar um baseline simples, registre isso explicitamente.

Uma loss numericamente alta não representa necessariamente falha se a escala do target também for alta. Calcule métricas relativas e compare com baselines.

---

# 15. EXPERIMENTOS CONTROLADOS MÍNIMOS

Execute, quando aplicável:

| ID | Experimento | Hipótese |
|---|---|---|
| E0 | Baseline reproduzido | Referência |
| E1 | Baseline com instrumentação | Medir gradientes e updates |
| E2 | AMP desligado | Detectar instabilidade de precisão |
| E3 | Tiny-set overfit | Verificar capacidade de aprender |
| E4 | Texto zerado | Verificar se o ramo textual influencia |
| E5 | Texto embaralhado | Verificar sinal ou vazamento |
| E6 | Embeddings normalizados | Verificar problema de escala |
| E7 | RobustRevIN corrigido/validado | Confirmar efeito real |
| E8 | OverlappingEmbed corrigido/validado | Confirmar efeito real |
| E9 | Cálculo unificado de val/test | Verificar inconsistência de avaliação |
| E10 | Clipping após diagnóstico | Avaliar proteção, não mascaramento |
| E11 | LR reduzido isoladamente | Verificar otimização instável |
| E12 | Teste final somente no checkpoint selecionado pela validação | Remover viés de seleção |

Não execute experimentos redundantes. Priorize os de maior poder discriminativo.

---

# 16. CORREÇÕES

Para cada problema confirmado:

1. explique a causa raiz;
2. identifique o arquivo e trecho;
3. implemente a correção;
4. produza um diff;
5. adicione teste unitário ou de regressão;
6. execute o teste;
7. repita o experimento relevante;
8. compare antes/depois;
9. registre efeitos colaterais;
10. informe confiança da conclusão.

Possíveis soluções poderão incluir, somente quando justificadas:

- correção do fluxo das flags;
- correção do cache;
- correção de normalização;
- uso do scaler ajustado apenas no treino;
- normalização dos embeddings;
- LayerNorm ou projeção adequada;
- correção de `detach`;
- inclusão de parâmetros no optimizer;
- correção dos grupos discriminativos;
- correção da ordem AMP/clipping;
- redução de learning rate;
- warm-up;
- gradient clipping;
- correção da agregação da loss;
- correção de masks;
- correção do alinhamento texto–target;
- reconstrução dos splits;
- remoção de vazamento;
- correção do modo `train()`/`eval()`;
- separação entre loss de treino e métricas de comparação;
- avaliação do teste apenas no final.

Não aplique todas essas soluções automaticamente. Aplique apenas as sustentadas pelas evidências.

---

# 17. MATRIZ DE HIPÓTESES

Mantenha uma matriz atualizada neste formato:

| ID | Hipótese | Evidência favorável | Evidência contrária | Teste | Resultado | Status | Confiança |
|---|---|---|---|---|---|---|---|
| H1 | Explosão de gradiente | — | Loss finita no log | Instrumentação | ... | ... | ... |
| H2 | Vanishing/path disconnected | Platô | Ainda sem grad norms | Gradientes por módulo | ... | ... | ... |
| H3 | RobustRevIN no-op | Trajetória idêntica | — | Lote fixo + hooks | ... | ... | ... |
| H4 | OverlappingEmbed no-op | Trajetória idêntica | — | Lote fixo + hooks | ... | ... | ... |
| H5 | Teste mais fácil | TestLoss muito menor | — | Estatísticas dos splits | ... | ... | ... |
| H6 | Loss agregada incorretamente | Val/Test discrepantes | — | Auditoria da avaliação | ... | ... | ... |
| H7 | Escala inconsistente | Loss alta | Valores ainda finitos | Auditoria de scalers | ... | ... | ... |
| H8 | Texto ignorado | Ablações sem efeito | — | Texto zero/shuffle | ... | ... | ... |
| H9 | Embeddings extremos | Banco textual envolvido | Sem estatísticas | Auditoria embeddings | ... | ... | ... |
| H10 | Parâmetros fora do optimizer | Platô | — | Auditoria dos grupos | ... | ... | ... |
| H11 | AMP overflow | CUDA/possível AMP | Sem registros | Rodar sem AMP | ... | ... | ... |
| H12 | Vazamento temporal/textual | TestLoss muito baixo | — | Auditoria dos splits | ... | ... | ... |

Status permitido:

```text
não testada
em investigação
confirmada
parcialmente confirmada
refutada
inconclusiva
```

---

# 18. ARTEFATOS OBRIGATÓRIOS

Produza os seguintes artefatos:

## `01_diagnostico_executivo.md`

Deve responder logo no início:

1. Existe explosão de gradiente?
2. Qual é a principal causa do platô?
3. RobustRevIN está realmente ativo?
4. OverlappingEmbed está realmente ativo?
5. Por que o TestLoss é muito menor?
6. O caminho textual contribui para a previsão?
7. Quais correções foram implementadas?
8. Qual foi a melhora antes/depois?

## `02_matriz_de_evidencias.md`

Inclua a matriz completa de hipóteses, testes e resultados.

## `03_auditoria_dados.md`

Inclua:

- splits;
- estatísticas;
- duplicatas;
- vazamentos;
- alinhamento;
- texto;
- embeddings;
- normalização.

## `04_auditoria_gradientes.md`

Inclua:

- grad norm global;
- normas por módulo;
- gradientes ausentes;
- gradientes não finitos;
- updates;
- AMP;
- gráficos ou tabelas resumidas.

## `05_resultados_experimentos.csv`

Colunas mínimas:

```text
experiment_id
seed
config_hash
epoch
train_loss
val_loss
test_loss_final
mae
rmse
grad_norm_p50
grad_norm_p95
grad_norm_max
max_abs_grad
nonfinite_grads
zero_grad_fraction
learning_rate
update_to_weight_ratio
runtime
notes
```

## `06_correcoes_aplicadas.md`

Para cada correção:

```text
causa
arquivo
função
alteração
teste adicionado
resultado antes
resultado depois
riscos
```

## `07_patch.diff`

Diff completo das alterações realizadas.

## `08_comandos_reproducao.md`

Inclua comandos exatos para:

- instalar o ambiente;
- reproduzir o baseline;
- executar a auditoria;
- rodar os testes;
- executar as ablações;
- gerar o relatório.

---

# 19. ESTRUTURA DA RESPOSTA FINAL

A resposta final deverá seguir esta ordem:

## 1. Veredito

Use uma destas formas:

```text
Explosão de gradiente confirmada.
Explosão de gradiente refutada.
Não foi possível confirmar nem refutar por falta de evidência direta.
```

Explique em no máximo três parágrafos.

## 2. Causas confirmadas

Tabela:

| Prioridade | Causa | Evidência | Impacto | Correção |
|---|---|---|---|---|

## 3. Problemas adicionais encontrados

Inclua problemas metodológicos, de dados, de código e avaliação.

## 4. Correções implementadas

Mostre arquivos modificados e justificativa.

## 5. Resultados antes/depois

Use métricas comparáveis na mesma escala.

## 6. Riscos residuais

Liste apenas riscos ainda não resolvidos.

## 7. Próxima configuração recomendada

Forneça uma configuração final reproduzível, mas somente depois de concluir os testes.

---

# 20. CRITÉRIOS DE ACEITAÇÃO

A investigação somente será considerada concluída quando:

- a norma dos gradientes tiver sido medida;
- os gradientes por módulo tiverem sido analisados;
- os parâmetros treináveis tiverem sido contabilizados;
- os grupos do optimizer tiverem sido auditados;
- a atualização real dos pesos tiver sido medida;
- o teste de memorização de 16–32 amostras tiver sido executado;
- RobustRevIN tiver sido validado por teste unitário e lote fixo;
- OverlappingEmbed tiver sido validado por teste unitário e lote fixo;
- as estatísticas de train/val/test tiverem sido comparadas;
- a agregação da loss tiver sido auditada;
- a normalização tiver sido auditada;
- o caminho textual tiver sido testado com texto real, zero e embaralhado;
- possíveis vazamentos temporais e textuais tiverem sido verificados;
- o teste tiver sido removido da seleção por época;
- as variantes tiverem sido comparadas por métricas comuns;
- todas as correções tiverem testes de regressão;
- o relatório diferenciar fato, hipótese e inferência.

Não encerre a tarefa apresentando apenas recomendações genéricas.

Execute, corrija, teste novamente e apresente evidências.
