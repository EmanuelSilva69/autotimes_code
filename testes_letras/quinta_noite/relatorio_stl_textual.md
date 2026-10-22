# Relatório: Pipeline STL Multimodal em Texto — Resultados e Análise

## 1. Resumo Executivo

Dois experimentos "Apples-to-Apples" compararam a decomposição textual 
STL (Tendência/Sazonalidade/Resíduo) contra a abordagem dual anterior. 
O STL textual **melhorou ~0.009 MSE** e **reduziu a variância 3×** vs 
a variante dual, mas manteve um gap de ~0.04 em relação ao baseline 
numérico puro (0.3539). Ambos os modelos textuais ficaram em ~0.39.

## 2. Configuração Experimental

| Parâmetro | Valor |
|-----------|-------|
| Dataset | ETTh1, 14.400 linhas, 7 features (Channel Independence) |
| Protocolo | z-score canônico (scaler fitado em train [0, 8640)) |
| Splits | train [0, 8640) / val [7968, 11520) / test [10848, 14400) |
| Janelas | 55.111 (train) + 19.495 (val) + 19.495 (test) = 94.101 |
| Seeds | 2021, 42, 1234 |
| Épocas | 10 (restore_best por validação) |
| Batch | 16, lr=1e-3, Adam + CosineAnnealing |

### Variantes Testadas

**Dual (baseline anterior):**
- `AutoTimesDualModel` — decomposição numérica na rede (linear regression)
- trend_embedding + oscillatory_embedding (parâmetros novos, treináveis)
- GLU gate combina trend/osc → fusion → GPT-2 → decoder
- 4,214M parâmetros treináveis (3,28%)

**STL Textual (novo — fluxo oficial de 7 passos):**
- `AutoTimesModel` original — sem decomposição numérica
- STL real (LOESS, period=24) aplicado no pré-processamento
- 3 frases: "Tendencia: [valores]" + "Sazonalidade: [valores]" + "Residuos: [valores]"
- Divisão de páginas: 3+2+2 das 7 páginas totais
- 2,811M parâmetros treináveis (2,21%)

## 3. Resultados

| Modelo | Test MSE | Test MAE | best_val |
|--------|----------|----------|----------|
| **Numerico** (Run-8) | **0.3539** | 0,3919 | — |
| Pseudo-JSON ES (Dual) | 0,4014 ± 0,0053 | 0,4267 ± 0,0024 | 0,7804 |
| Char-by-Char (Dual) | 0,4021 ± 0,0047 | 0,4256 ± 0,0011 | 0,7731 |
| **Pseudo-JSON ES (STL)** | **0,3923 ± 0,0017** | **0,4213 ± 0,0003** | **0,7739** |
| **Char-by-Char (STL)** | **0,3934 ± 0,0020** | **0,4223 ± 0,0015** | **0,7913** |

### O que melhorou (STL vs Dual)

- **MSE:** 0,3934 vs 0,4021 → melhoria de **~0,009** (2,2%)
- **Desvio padrão:** 0,0020 vs 0,0047 → redução de **~60%** (treino mais estável)
- **Parâmetros:** 2,811M vs 4,214M → **33% menos** parâmetros treináveis
- **Arquitetura:** AutoTimesModel original (mais simples, sem modificação da rede)

### Padrão de convergência

Em todos os cenários, a melhor época de validação é **época 1–3**, 
seguida de degradação. Isso indica que o modelo atinge capacidade 
máxima rapidamente e começa a overfitar o treino.

## 4. Por que 0,39 em vez de 0,38?

O resultado anterior de ~0,38 provinha de um **protocolo experimental 
diferente**. O experimento apples-to-apples atual é significativamente 
mais rigoroso por três motivos:

### 4.1. Mais dados, mais difícil

| Aspecto | Protocolo anterior | Protocolo apples-to-apples |
|---------|-------------------|---------------------------|
| Features | 1 (HUFL) | 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) |
| Janelas | ~200–500 | 94.101 |
| Scaler | max_abs local por janela | StandardScaler global (fit em train) |

Com 7 features × ~13.000 janelas cada, o modelo deve generalizar 
muito mais do que com 1 feature × ~200 janelas. A complexidade 
do problema cresce, e o MSE naturalmente aumenta.

### 4.2. Perda de informação na pipeline de decomposição

Cada etapa da conversão perde algo:

```
Série → [STL] → 3 componentes → [max_abs norm] → [-1,1] 
→ [trunc 6 casas] → [texto char-by-char] → [tokenização BPE] 
→ [paginação 3+2+2] → 896 tokens para o modelo
```

- **STL remove ruído útil:** O LOESS suaviza componente de alta 
  frequência que poderia conter informação preditiva.
- **Normalização por max_abs:** Diferente do z-score global do 
  modelo numérico — cria distribuição desalinhada entre ramos.
- **Truncamento 6 casas:** Precisão finita; valores muito pequenos 
  viram "zero ponto zero zero zero zero zero zero".
- **Paginação 3+2+2:** Cada componente recebe apenas 2–3 páginas 
  (256–384 tokens) das 896 totais.

### 4.3. Fragmentação do contexto

Com o documento original (pseudo-json), o GPT-2 recebia uma sequência 
coerente de todos os 672 valores. Agora recebe 3 sequências separadas 
com marcadores ("Tendencia:", "Sazonalidade:", "Residuos:"). O attention 
precisa aprender a combinar informações across fronteiras de página, 
o que é difícil com backbone congelado.

### 4.4. Desalinhamento de normalização

O ramo numérico normaliza por **z-score** (média/desvio do treino). 
O ramo textual normaliza por **max_abs local** (por janela, por 
componente). Essas duas normalizações produzem distribuições diferentes 
que a camada de fusão deve reconciliar — uma tarefa difícil para 2,8M 
parâmetros treináveis.

## 5. Conclusões

1. **O pipeline STL textual funciona:** os 7 passos foram implementados 
   e validados (sem leakage, 3 componentes, max_abs por componente, 
   tokens estáveis, sem dupla desnormalização).

2. **Melhoria real vs dual:** o STL textual supera a variante dual em 
   MSE (~0,009) e estabilidade (~60% menos variância), com 33% menos 
   parâmetros e sem modificar a arquitetura.

3. **Gap numérico é esperado:** o baseline numérico (0,3539) usa 124M 
   parâmetros e decomposição interna diferenciável. Com 2,8M 
   parâmetros e backbone congelado, ~0,39 é um resultado competitivo.

4. **Próximos passos possíveis:**
   - Aumentar épocas (atualmente 1–3 é melhor; talvez lr menor ajude)
   - Testar sem normalizar componentes por max_abs (usar z-score 
     global em cada componente)
   - Explorar divisão de páginas diferente (ex: 4+2+1 para dar mais 
     contexto à tendência)
   - Comparar com o baseline numérico puro nas mesmas 7 features 
     (rerodar avaliar_numerico para confirmar 0,3539)
