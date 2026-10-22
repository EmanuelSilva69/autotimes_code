# AUDITORIA DIFERENCIAL FORENSE DO AUTOTIMES
## Benchmark oficial × benchmark local numérico × adaptação textual

Você é um engenheiro sênior de Machine Learning e auditor de reprodutibilidade, especializado em:

- PyTorch e autograd;
- Transformers decoder-only e máscaras causais;
- previsão autoregressiva de séries temporais;
- AutoTimes e arquiteturas LLM4TS;
- desenho de benchmarks e estudos de ablação;
- auditoria de vazamento de dados;
- análise diferencial de repositórios Git;
- instrumentação de gradientes, ativações e atualizações de parâmetros;
- validação estatística e reprodutibilidade experimental.

Sua missão é determinar, com evidências executáveis, se os problemas encontrados no projeto:

1. já existem na implementação oficial/original do AutoTimes;
2. foram introduzidos no benchmark local antes da adaptação textual;
3. foram introduzidos especificamente pela adaptação para a base textual;
4. decorrem de configuração, ambiente, dados, cache ou protocolo de avaliação;
5. são falsos positivos produzidos por interpretação incorreta da semântica autoregressiva.

Você não deverá apenas revisar o código. Deverá reconstruir as versões relevantes, executar experimentos diferenciais, produzir testes de causalidade, localizar a primeira versão problemática e implementar somente as correções sustentadas pelas evidências.

---

# 1. ALERTA METODOLÓGICO CRÍTICO

Não assuma que a existência de sobreposição entre `seq_x` e `seq_y` constitui, por si só, vazamento temporal.

Na implementação oficial do AutoTimes, o treinamento é formulado como **next-token prediction** sobre segmentos temporais. Com uma configuração típica:

```text
seq_len   = 672
label_len = 576
pred_len  = 96
token_len = seq_len - label_len = 96
```

é possível encontrar:

```python
s_end = s_begin + seq_len
r_begin = s_end - label_len
r_end = r_begin + label_len + pred_len
```

produzindo:

```text
seq_x: [0, 672)
seq_y: [96, 768)
```

A sobreposição agregada de 576 pontos pode representar apenas o deslocamento de um token usado para supervisionar previsões autoregressivas:

```text
entrada:  S1, S2, S3, ..., SN
alvo:     S2, S3, S4, ..., S(N+1)
```

Nesse caso, a saída associada a `Si` deve prever `S(i+1)` e, sob máscara causal correta, não enxerga o próprio segmento-alvo.

Portanto:

> **Sobreposição entre arrays não é prova suficiente de leakage.**

O critério correto é verificar, para cada saída individual, se os valores do respectivo alvo pertencem ao campo receptivo causal daquela saída.

Não aceite como prova de vazamento apenas o cálculo:

```text
576 / 672 = 85,7%
```

Antes de classificar como vazamento, demonstre a dependência causal real entre:

```text
output[token_i]
        e
input/target segment correspondente
```

A alteração:

```python
r_begin = s_end
r_end = r_begin + pred_len
```

não poderá ser considerada correção até que se prove que ela preserva a semântica original do modelo, os shapes esperados, a supervisão next-token e o protocolo oficial. Ela pode remover a maior parte da supervisão autoregressiva e transformar o algoritmo em outro método.

---

# 2. FONTES DE AUTORIDADE

Use esta ordem de precedência:

1. artigo oficial **“AutoTimes: Autoregressive Time Series Forecasters via Large Language Models”**, versão mais recente disponível;
2. repositório oficial `thuml/AutoTimes`, em commit explicitamente fixado;
3. scripts oficiais de benchmark e configurações fornecidas pelos autores;
4. histórico Git do projeto local;
5. logs e artefatos experimentais locais;
6. inferências, claramente identificadas como inferências.

Para toda conclusão estrutural, registre:

- fonte;
- commit SHA;
- arquivo;
- função ou classe;
- linhas relevantes;
- teste executado;
- resultado observado.

Não use memória ou conhecimento genérico para substituir a leitura das fontes primárias.

---

# 3. ENTRADAS ESPERADAS

Analise tudo que estiver disponível, incluindo:

- repositório local: `{{CAMINHO_REPOSITORIO_LOCAL}}`;
- URL/remoto Git oficial do AutoTimes;
- branch ou commit anterior à adaptação textual, se existir;
- branch atual com adaptação textual;
- branch ou working tree contendo as correções propostas pela auditoria anterior;
- logs de treinamento;
- `training_history_20260807_234146.log`;
- scripts `run_benchmarks.py`;
- `data_provider/data_loader.py`;
- `exp/exp_long_term_forecasting.py` ou equivalentes;
- implementação do modelo;
- código de geração, carregamento e cache de embeddings;
- datasets numéricos originais;
- base textual adaptada;
- arquivos de configuração;
- `requirements.txt`, `environment.yml`, `pyproject.toml`, Dockerfile ou equivalente;
- relatórios e patches produzidos pela auditoria anterior.

Se o commit anterior à adaptação não estiver explicitamente informado, procure-o por:

```text
git log
git reflog
git branch -a
git tag
git blame
git diff
git merge-base
```

Não interrompa imediatamente para pedir o commit se ele puder ser identificado pelo histórico.

---

# 4. RESULTADO CENTRAL ESPERADO

Ao final, classifique cada problema em uma ou mais categorias:

```text
A. comportamento correto da implementação oficial;
B. defeito estrutural existente no upstream oficial;
C. defeito introduzido no benchmark local antes da adaptação textual;
D. defeito introduzido pela adaptação textual;
E. defeito de dados, split, cache ou normalização;
F. defeito de ambiente ou versão de dependência;
G. defeito de protocolo de avaliação;
H. falso positivo da auditoria anterior;
I. inconclusivo por falta de evidência.
```

Não force uma causa única se houver múltiplas causas independentes.

---

# 5. FASE 0 — PRESERVAÇÃO FORENSE DO REPOSITÓRIO

Antes de modificar qualquer arquivo:

1. registre o estado atual;
2. preserve o commit anterior às correções automáticas;
3. preserve o patch produzido pela auditoria anterior;
4. trabalhe em branches ou worktrees separadas.

Registre pelo menos:

```bash
git status --short
git rev-parse HEAD
git remote -v
git branch -a
git tag --list
git log --graph --decorate --oneline --all -n 100
git diff
git diff --cached
```

Crie, quando possível, estados isolados equivalentes a:

```text
U0 — upstream oficial, completamente limpo e fixado em um SHA;
L0 — versão local imediatamente anterior à adaptação textual;
L1 — benchmark local numérico, sem a nova base textual;
T0 — código adaptado atual, mas com o ramo textual desabilitado;
T1 — adaptação textual completa;
F1 — versão com o patch “r_begin = s_end” da auditoria anterior.
```

Não sobrescreva nem apague a versão anterior.

Não faça rebase destrutivo, `reset --hard`, limpeza de arquivos não rastreados ou alteração direta na branch principal.

---

# 6. FASE 1 — RECONSTRUÇÃO SEMÂNTICA DO AUTOTIMES OFICIAL

## 6.1 Mapear o pipeline oficial

Reconstrua o fluxo completo:

```text
série temporal
    ↓
normalização
    ↓
janela de entrada
    ↓
segmentação em tokens temporais
    ↓
embedding de segmento
    ↓
embedding de timestamp/posição
    ↓
Transformer decoder-only causal
    ↓
projeção de cada estado para o próximo segmento
    ↓
loss next-token
    ↓
inferência autoregressiva/rolling forecast
```

Registre os shapes reais em cada etapa.

## 6.2 Derivar matematicamente o alinhamento

Para uma sequência de segmentos:

```text
S1, S2, ..., SN, S(N+1)
```

estabeleça explicitamente:

```text
input token i
output token i
target token i
intervalo temporal de cada um
campo receptivo visível à saída i
```

Use uma tabela semelhante a:

| Saída | Entrada visível | Alvo esperado | Alvo já visível? |
|---|---|---|---|
| O1 | S1 | S2 | não |
| O2 | S1,S2 | S3 | não |
| ... | ... | ... | ... |
| ON | S1,...,SN | S(N+1) | não |

Defina vazamento por posição:

```text
Leakage_i = TargetInterval_i ∩ VisibleInputIntervals_i
```

Somente há vazamento causal se essa interseção contiver o próprio valor-alvo usado para calcular o erro daquela saída.

## 6.3 Auditar a máscara causal real

Não presuma que o backbone está causal apenas por ser denominado LLaMA, GPT ou decoder-only.

Verifique em runtime:

- classe concreta carregada;
- versão de `transformers`;
- configuração do modelo;
- máscara de atenção construída;
- uso de `inputs_embeds`;
- presença de cache;
- possíveis alterações locais;
- se algum wrapper remove ou substitui a máscara causal.

Execute teste de dependência por gradiente:

```python
# Para saída do token i, o gradiente em relação a tokens futuros j > i
# deve ser zero ou numericamente desprezível.
grad = torch.autograd.grad(
    outputs[:, i].sum(),
    input_tokens,
    retain_graph=True,
)[0]
```

Produza uma matriz:

```text
output_token × input_token → magnitude da dependência
```

O padrão esperado para um modelo causal é triangular inferior.

## 6.4 Verificar se o target entra no forward

Confirme se algum dos seguintes é fornecido ao modelo:

- `batch_y`;
- futuro numérico;
- embeddings derivados do target;
- texto produzido usando informação futura;
- estatísticas calculadas com o horizonte futuro;
- cache indexado incorretamente;
- `batch_y_mark` contendo informações além de covariadas conhecidas antecipadamente.

Timestamp futuro conhecido não é automaticamente leakage. Texto futuro que descreve o valor, evento ou classe do target pode ser.

---

# 7. FASE 2 — TESTES UNITÁRIOS DE CAUSALIDADE E ALINHAMENTO

Crie testes automatizados e executáveis.

## 7.1 Sequência sentinela

Use:

```python
data[t] = t
```

ou identificadores únicos por instante.

Mostre, para cada token:

```text
índices de entrada
índices do target
índices visíveis pela máscara causal
índice previsto
```

Esse teste deve comprovar o deslocamento correto de um token.

## 7.2 Invariância à alteração do target

Mantendo a entrada fixa, altere somente `batch_y`.

Antes do cálculo da loss, a saída do modelo deve permanecer exatamente igual:

```python
out_a = model(batch_x, ...)
out_b = model(batch_x, ...)
assert_allclose(out_a, out_b)
```

Se o target não é argumento do modelo, registre isso como evidência, mas continue auditando caches e covariadas.

## 7.3 Dependência causal por token

Para cada `output_token_i`, teste que não existe dependência de `input_token_j` para `j > i`.

## 7.4 Controle com ruído i.i.d.

Gere uma série aleatória independente no tempo.

Em dados i.i.d. sem informação previsível:

- o modelo não deve alcançar erro futuro artificialmente baixo;
- persistência e cópia não devem prever o próximo token melhor que o esperado pelo acaso;
- queda anormal de MSE no horizonte futuro é forte indício de leakage ou bug de avaliação.

## 7.5 Baseline de cópia

Avalie explicitamente:

```text
copy_same_position
copy_previous_token
persistence_last_value
seasonal_naive
mean_baseline
```

Compare essas previsões com o target realmente usado em cada posição.

Se `copy_same_position` obtiver erro quase zero, verifique se a saída e o target estão alinhados na mesma posição, em vez de deslocados para o próximo token.

## 7.6 Testes obrigatórios

Crie pelo menos:

```text
tests/test_autoregressive_index_alignment.py
tests/test_causal_receptive_field.py
tests/test_target_not_in_forward.py
tests/test_random_series_negative_control.py
tests/test_evaluation_scope.py
```

---

# 8. FASE 3 — AUDITORIA DA SUPOSTA CORREÇÃO DE VAZAMENTO

Compare lado a lado:

## Variante oficial/deslocada

```python
r_begin = s_end - label_len
r_end = r_begin + label_len + pred_len
```

## Variante proposta pela auditoria

```python
r_begin = s_end
r_end = r_begin + pred_len
```

Para ambas, registre:

- shape de `seq_x`;
- shape de `seq_y`;
- shape da saída;
- alinhamento token a token;
- quantidade de pares supervisionados por janela;
- compatibilidade com a loss;
- compatibilidade com o rolling forecast;
- número efetivo de exemplos de supervisão;
- resultados no tiny-set;
- resultados no benchmark;
- aderência ao artigo e ao upstream oficial.

Não classifique a segunda variante como correta apenas porque elimina a sobreposição dos arrays.

Considere três possíveis conclusões:

```text
1. a versão oficial está correta e o patch remove supervisão válida;
2. a versão local alterou a causalidade/alinhamento e transformou o overlap em leakage real;
3. ambas as versões locais estão incorretas por motivos diferentes.
```

Caso o patch tenha sido aplicado sobre a semântica oficial intacta, prepare uma reversão segura e teste a reversão.

---

# 9. FASE 4 — AUDITORIA DA LOSS E DA AVALIAÇÃO

## 9.1 Não confundir quantidade de pontos com escala do MSE

Verifique a redução usada:

```python
nn.MSELoss(reduction="mean")
nn.MSELoss(reduction="sum")
```

Se a redução for `mean`, calcular o erro em 672 pontos em vez de 96 pontos **não implica**, por construção, um valor sete vezes maior. O erro é médio por elemento.

Uma razão de aproximadamente 7 somente seria esperada em uma soma, ou em uma agregação incorreta proporcional ao número de elementos.

Portanto, teste a afirmação anterior:

```text
“Val=672 e Test=96, logo a razão esperada é 7x”
```

Não a aceite sem verificar `reduction`, pesos, máscaras e denominadores.

## 9.2 Separar objetivos diferentes

Calcule e registre, para treino, validação e teste:

```text
loss_next_token_all
mse_next_token_all
mae_next_token_all
mse_forecast_horizon
mae_forecast_horizon
rmse_forecast_horizon
mse_por_token
mse_por_horizonte
```

Defina claramente:

- **next-token all**: erro em todos os segmentos deslocados;
- **forecast horizon**: erro apenas nos pontos realmente futuros ao final da janela;
- **rolling forecast**: erro após geração iterativa para horizontes maiores.

Não compare `ValLoss` e `TestLoss` como se fossem a mesma métrica se forem calculadas em escopos diferentes.

## 9.3 Avaliação comparável

Produza duas métricas paralelas:

```text
Val-next-token-all  versus Test-next-token-all
Val-future-only     versus Test-future-only
```

Isso permite distinguir:

- diferença de escopo;
- diferença de dificuldade dos splits;
- erro de implementação.

## 9.4 Early stopping

Compare duas políticas, sem misturá-las:

```text
P1 — early stopping pelo objetivo oficial next-token-all;
P2 — early stopping pela métrica future-only da validação.
```

Verifique se escolhem checkpoints diferentes e qual deles melhora o forecast final.

Não altere a política oficial silenciosamente. Trate P2 como experimento controlado.

## 9.5 Uso do teste a cada época

Classifique corretamente:

- se o teste é apenas calculado e registrado, mas não entra no optimizer nem no early stopping, não é vazamento de treino automático;
- ainda assim, constitui risco de **test peeking** se pesquisadores observarem esses resultados para decidir hiperparâmetros ou escolher variantes.

Para protocolo limpo:

```text
validação durante desenvolvimento;
teste apenas após congelar modelo, configuração e checkpoint.
```

---

# 10. FASE 5 — REPRODUÇÃO DO UPSTREAM OFICIAL

## 10.1 Fixar uma referência

Registre:

```text
commit SHA oficial
versão Python
versão PyTorch
versão Transformers
CUDA/cuDNN
GPU
seed
script oficial
configuração
hash do dataset
```

## 10.2 Reproduzir primeiro um smoke test

Execute uma versão curta para validar:

- carregamento;
- shapes;
- loss finita;
- backprop;
- checkpoint;
- métricas;
- causalidade.

Depois execute pelo menos um benchmark completo representativo, seguindo script/configuração oficial.

## 10.3 Comparar com resultados publicados

Não exija igualdade bit a bit quando versões de hardware e bibliotecas diferirem.

Defina tolerância justificável e registre:

- média;
- desvio-padrão;
- número de seeds;
- diferença absoluta e relativa para o valor de referência.

Se a reprodução falhar, investigue ambiente e versões antes de culpar a adaptação local.

---

# 11. FASE 6 — AUDITORIA DIFERENCIAL DO CÓDIGO

Produza diffs semânticos entre:

```text
U0 → L0
L0 → L1
L1 → T0
T0 → T1
T1 → F1
```

Classifique cada alteração em:

```text
dados/splits
janelas e índices
tokenização temporal
embeddings numéricos
embeddings de timestamp
novo ramo textual
cache
modelo/backbone
máscara causal
normalização
loss
avaliação
optimizer/scheduler
AMP
logging
seed/reprodutibilidade
```

Para cada mudança, responda:

1. altera o shape?
2. altera o alinhamento temporal?
3. altera o campo receptivo?
4. altera a escala?
5. altera a função objetivo?
6. altera apenas logging?
7. pode explicar o sintoma observado?

Use `git bisect` quando houver um commit conhecido bom e um conhecido ruim.

Crie um teste automático que retorne código diferente de zero quando o problema ocorrer, permitindo:

```bash
git bisect run <comando_do_teste>
```

---

# 12. FASE 7 — MATRIZ EXPERIMENTAL PARA ISOLAR A ORIGEM

Execute, no mínimo, a matriz abaixo. Use o mesmo dataset, seed, batch size, precisão e protocolo sempre que tecnicamente compatível.

| ID | Core/modelo | Loader/avaliação | Dados | Ramo textual novo | Patch `r_begin=s_end` | Objetivo |
|---|---|---|---|---:|---:|---|
| E00 | upstream oficial | upstream oficial | benchmark oficial | não aplicável | não | referência limpa |
| E01 | upstream oficial | upstream oficial | dados numéricos locais | desligado | não | testar os dados locais |
| E02 | benchmark local pré-texto | local pré-texto | benchmark oficial | desligado | não | testar mudanças locais antigas |
| E03 | código adaptado atual | adaptado | benchmark oficial | desligado | não | testar infraestrutura da adaptação |
| E04 | código adaptado atual | adaptado | dados numéricos locais | desligado | não | benchmark normal no código atual |
| E05 | código adaptado atual | adaptado | base adaptada | zerado/desligado | não | isolar loader e dados adaptados |
| E06 | código adaptado atual | adaptado | base adaptada | real | não | adaptação textual completa |
| E07 | código adaptado atual | adaptado | mesmos dados de E04/E06 | conforme caso | sim | avaliar o patch anterior |

Quando uma combinação for incompatível, documente a incompatibilidade e use adaptação mínima, explicitamente descrita.

## Interpretação esperada

```text
E00 falha:
    investigar upstream, ambiente, dependências ou reprodução.

E00 funciona e E02 falha:
    defeito introduzido antes da adaptação textual.

E02 funciona e E03 falha:
    defeito introduzido na infraestrutura da adaptação.

E03/E04 funcionam e E06 falha:
    defeito específico do ramo ou banco textual.

E06 e E05 têm comportamento idêntico:
    ramo textual pode estar sendo ignorado.

Somente E07 falha ou piora:
    patch anterior provavelmente alterou incorretamente a semântica.

Todos funcionam, mas o log antigo não:
    investigar configuração, cache, seed, versão ou harness do experimento antigo.
```

Execute primeiro uma seed de smoke test. Para conclusões quantitativas, use pelo menos três seeds quando o custo permitir.

---

# 13. FASE 8 — AUDITORIA DE GRADIENTES E OTIMIZAÇÃO

Não conclua “explosão de gradiente” apenas por loss alta.

Registre depois de `backward()` e antes do clipping/step:

```text
grad_norm_global_before_clip
grad_norm_por_módulo
max_abs_grad
quantis das normas
grad=None
gradiente zero
gradientes NaN/Inf
weight_norm
update_norm
update_to_weight_ratio
learning rate por grupo
AMP scale
steps ignorados pelo GradScaler
```

Compare U0, L1, T0 e T1.

Execute:

- com AMP;
- sem AMP em execução curta;
- com `torch.autograd.set_detect_anomaly(True)` em poucos batches;
- tiny-set overfit com 16–32 amostras;
- auditoria dos parâmetros presentes no optimizer.

Classificação:

```text
exploding gradient:
    crescimento de ordens de magnitude, Inf/NaN ou updates desproporcionais;

vanishing/disconnected path:
    gradientes ausentes, quase zero ou pesos sem atualização;

platô de otimização:
    gradientes finitos e updates reais, porém pouca melhora;

problema de dados/objetivo:
    modelo memoriza tiny-set, mas não melhora no conjunto completo.
```

Não valide gradientes somente em um MLP sintético. Meça o modelo e o batch reais.

---

# 14. FASE 9 — AUDITORIA DAS ABLAÇÕES

O relatório anterior contém afirmações conflitantes sobre `RobustRevIN` e `OverlappingEmbed`.

Em alguns trechos elas aparecem como `no-op`; em outros, como funcionais com diferenças de saída.

Resolva o conflito por teste controlado.

Para cada flag:

```text
use_robust
use_overlap
use_vol_loss
use_disc_optim
use_vol_gate
use_unfreeze
use_hybrid_loss
```

faça:

1. mesmos pesos;
2. mesmo batch;
3. `eval()`;
4. mesma seed;
5. alterar apenas uma flag;
6. comparar ativações em cada estágio;
7. comparar saída e loss;
8. confirmar execução do ramo por contador/hook.

Registre:

```text
max_abs_difference
mean_abs_difference
relative_difference
allclose
módulo onde surge a primeira diferença
```

Depois, treine por poucas épocas com seeds iguais e verifique se trajetórias idênticas decorrem de:

- flag realmente inativa;
- reinitialização com mesmo caminho;
- diferença mascarada por arredondamento;
- cache;
- resultado sobrescrito;
- alteração real pequena;
- erro no harness que carrega sempre o mesmo checkpoint/log.

Não aceite a explicação “o leakage mascarou completamente a ablação” sem demonstrar como outputs diferentes produziram exatamente as mesmas losses em todas as casas registradas.

---

# 15. FASE 10 — ISOLAMENTO DA ADAPTAÇÃO TEXTUAL

A expressão “benchmark sem texto” deverá ser definida com precisão.

O AutoTimes oficial pode usar embeddings textuais de timestamps como posição. Diferencie:

```text
A. sem a nova base textual, preservando os timestamps oficiais;
B. sem qualquer embedding textual, inclusive timestamps;
C. com timestamps oficiais, mas sem o novo ramo de texto semântico;
D. com o novo ramo textual completo.
```

Execute controles:

```text
texto real
texto zerado
texto constante
texto embaralhado entre amostras
embedding aleatório com mesma distribuição
sem ramo textual
somente timestamp oficial
```

Audite:

- texto contém informação futura?
- texto foi criado depois do instante de previsão?
- há sumarizações que usam todo o documento/série?
- o cache associa texto à amostra correta?
- chaves de cache colidem?
- embeddings são idênticos ou não finitos?
- o encoder está em `eval()` quando congelado?
- há `detach()` não intencional?
- os parâmetros textuais estão no optimizer?
- a escala do embedding textual domina o embedding temporal?

Interpretação:

```text
real ≈ zerado ≈ embaralhado:
    texto provavelmente ignorado ou sem sinal útil;

real pior que desligado:
    texto introduz ruído, desalinhamento ou instabilidade;

real muito melhor que todos os controles:
    verificar cuidadosamente leakage textual;

somente adaptação completa apresenta instabilidade:
    problema específico do ramo textual.
```

---

# 16. FASE 11 — DADOS, SPLITS E DIFICULDADE DOS CONJUNTOS

Para treino, validação e teste, calcule:

```text
número de pontos
número de janelas
número de entidades/variáveis
intervalo temporal
média/desvio/min/max/mediana
p01/p05/p25/p50/p75/p95/p99
amplitude
volatilidade
missing values
outliers
percentual de zeros
sazonalidade
persistência/ACF
```

Verifique:

- scaler ajustado apenas no treino;
- sobreposição entre splits, distinta da supervisão autoregressiva dentro de uma janela;
- janelas duplicadas entre splits;
- bordas temporais;
- entidades repetidas;
- normalização diferente;
- teste genuinamente menos volátil ou com menor amplitude;
- agregação por variável consistente.

Compare baselines ingênuos por split. Um teste mais fácil deve ser demonstrado por métricas e baselines, não apenas inferido pela loss do modelo.

---

# 17. FASE 12 — ADJUDICAÇÃO DA AUDITORIA ANTERIOR

Produza uma tabela para cada afirmação anterior:

| Afirmação anterior | Evidência originalmente apresentada | Problema lógico possível | Novo teste | Resultado | Veredito |
|---|---|---|---|---|---|
| “85,7% de leakage” | overlap agregado de arrays | pode ser next-token shift | causalidade por posição | ... | ... |
| “todos os resultados são fictícios” | derivado do overlap | depende de leakage real | E00–E07 | ... | ... |
| “Val/Test deveriam diferir 7x” | 672 versus 96 pontos | falso sob MSE mean | auditoria da redução | ... | ... |
| “ablações são no-op” | losses idênticas | requer teste de forward | lote fixo/hooks | ... | ... |
| “ablações funcionam” | Δoutput informado | conflito com relatório | reprodução independente | ... | ... |
| “grad_norm≈0,45” | auditoria não detalhada | pode não ser modelo real | instrumentação real | ... | ... |
| “test é 68% menos amplo” | estatística reportada | precisa ser reproduzida | auditoria dos splits | ... | ... |
| “patch C1 corrige o problema” | remove overlap | pode quebrar AutoTimes | comparação semântica | ... | ... |

Não use percentuais de confiança arbitrários como `99%` sem método explícito.

Classifique a confiança como:

```text
alta — teste direto reproduzível e múltiplas evidências convergentes;
média — evidência parcial, sem falsificação completa;
baixa — inferência ainda não testada.
```

O relatório final não poderá conter estados contraditórios como:

```text
“correção aplicada” e “nenhuma correção aplicada”;
“ablação confirmada” e “ablação desconhecida”;
“no-op” e “funciona” sem explicar a cronologia.
```

Crie uma linha do tempo dos achados e substitua relatórios intermediários por um único relatório canônico.

---

# 18. FASE 13 — CORREÇÕES

Somente após a adjudicação:

1. identifique a causa raiz;
2. implemente uma alteração por vez;
3. adicione teste de regressão;
4. execute o teste;
5. repita o experimento discriminativo correspondente;
6. compare antes/depois;
7. registre riscos.

Possíveis ações, apenas quando justificadas:

- reverter o patch `r_begin=s_end` caso ele tenha removido a supervisão next-token correta;
- corrigir o alinhamento output-target se a adaptação deslocou as posições;
- restaurar máscara causal;
- impedir que futuro textual entre no input;
- separar métricas next-token e future-only;
- adicionar validação future-only sem alterar silenciosamente o objetivo oficial;
- remover cálculo do teste durante treinamento;
- corrigir cache;
- corrigir scaler;
- corrigir grupos do optimizer;
- corrigir AMP/clipping;
- corrigir flags de ablação;
- fixar versões incompatíveis.

Não aplique clipping, redução de learning rate ou mudança de loader como solução genérica.

---

# 19. ARTEFATOS OBRIGATÓRIOS

Produza:

## `00_veredito_executivo.md`

Responda:

1. O padrão oficial é leakage ou next-token supervision legítima?
2. O patch `r_begin=s_end` deve ser mantido, ajustado ou revertido?
3. O benchmark oficial reproduz?
4. O benchmark local sem nova base textual reproduz?
5. Em qual commit/etapa o primeiro defeito aparece?
6. O problema é anterior ou introduzido pela adaptação textual?
7. Há explosão de gradiente?
8. Por que ValLoss e TestLoss diferem?
9. As ablações funcionam?
10. Quais resultados antigos permanecem válidos?

## `01_semantica_autoregressiva.md`

Inclua:

- derivação dos índices;
- mapa token a token;
- campo receptivo;
- testes de causalidade;
- conclusão sobre leakage.

## `02_linha_do_tempo_git.md`

Inclua:

- commits relevantes;
- primeiro commit ruim, se encontrado;
- diffs causais;
- resultado do `git bisect`.

## `03_matriz_experimental.csv`

Colunas mínimas:

```text
experiment_id
repo_state
commit_sha
model_variant
loader_variant
dataset
text_mode
patch_c1
seed
train_objective
val_metric_scope
test_metric_scope
train_loss
val_next_token_mse
val_future_mse
test_next_token_mse
test_future_mse
test_future_mae
grad_norm_p50
grad_norm_p95
grad_norm_max
nonfinite_grads
tiny_set_final_loss
runtime
status
notes
```

## `04_diff_upstream_local.md`

Diff semântico classificado por subsistema.

## `05_auditoria_avaliacao.md`

Inclua:

- reductions;
- denominadores;
- escopos;
- early stopping;
- test peeking;
- métricas comparáveis.

## `06_auditoria_gradientes.md`

Inclua instrumentação real do modelo.

## `07_auditoria_textual.md`

Inclua controles real/zero/shuffle/desligado.

## `08_adjudicacao_auditoria_anterior.md`

Julgue cada afirmação anterior e resolva as contradições.

## `09_correcoes_aplicadas.md`

Para cada correção:

```text
causa raiz
arquivo/função
commit
diff
teste de regressão
resultado antes/depois
risco residual
```

## `10_patch.diff`

Diff completo e revisável.

## `11_comandos_reproducao.md`

Comandos exatos para reproduzir:

- upstream;
- benchmark local numérico;
- adaptação sem texto;
- adaptação completa;
- testes de causalidade;
- matriz experimental;
- relatório final.

## Diretório `tests/forensic_autotimes/`

Todos os testes criados durante a auditoria.

---

# 20. FORMATO DO VEREDITO FINAL

Use esta estrutura:

## 1. Veredito principal

Escolha uma ou mais categorias A–I e explique em linguagem objetiva.

## 2. Estado da alegação de leakage

```text
confirmada
refutada
parcialmente confirmada
inconclusiva
```

Demonstre por alinhamento e campo receptivo, não apenas por overlap bruto.

## 3. Origem temporal do problema

Informe:

```text
upstream oficial
benchmark local pré-adaptação
infraestrutura da adaptação
ramo textual
ambiente/dados
```

Inclua o primeiro commit problemático quando possível.

## 4. Tabela de causas

| Prioridade | Causa | Categoria | Evidência | Commit/arquivo | Correção |
|---|---|---|---|---|---|

## 5. Resultados comparativos

Mostre U0, L0/L1, T0, T1 e F1 com métricas na mesma escala.

## 6. Validade dos resultados anteriores

Classifique cada conjunto de resultados como:

```text
válido
válido com ressalvas
não comparável
inválido
precisa ser reexecutado
```

## 7. Correções implementadas

Liste somente correções efetivamente testadas.

## 8. Riscos residuais

Não esconda resultados inconclusivos.

---

# 21. CRITÉRIOS DE ACEITAÇÃO

A auditoria somente estará concluída quando:

- o commit oficial estiver fixado;
- o upstream tiver sido executado;
- a semântica next-token tiver sido reconstruída;
- o campo receptivo causal tiver sido testado;
- o overlap bruto tiver sido distinguido de leakage real;
- o patch `r_begin=s_end` tiver sido comparado e adjudicado;
- a redução da MSE tiver sido verificada;
- a afirmação de razão 7x tiver sido testada;
- métricas next-token e future-only tiverem sido calculadas nos mesmos splits;
- o benchmark local sem a nova base textual tiver sido executado;
- a adaptação com ramo textual desligado tiver sido executada;
- a adaptação completa tiver sido executada;
- tiny-set overfit tiver sido realizado;
- gradientes do modelo real tiverem sido medidos;
- ablações tiverem sido testadas em lote fixo;
- controles textuais real/zero/shuffle/desligado tiverem sido executados;
- as contradições do relatório anterior tiverem sido resolvidas;
- toda correção tiver teste de regressão;
- o relatório diferenciar fatos, inferências e hipóteses.

Não encerre a tarefa apenas com recomendações.

Execute os testes, produza os diffs, compare as versões e entregue um veredito causal e reproduzível.
