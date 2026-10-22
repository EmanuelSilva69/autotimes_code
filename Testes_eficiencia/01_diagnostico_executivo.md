# DIAGNÓSTICO EXECUTIVO — AUDITORIA FORENSE AUTOTIMES

**Data**: 2026-08-08  
**Investigador**: Sistema de Auditoria Automatizada  
**Nível de Confiança**: MUITO ALTO (evidências diretas confirmadas)

---

## 1. VEREDITO FINAL

### **Não há explosão de gradiente.**

A loss permanece finita durante todo o treinamento (Log: TrainLoss ≈ 50.2 → 46.5-46.9, ValLoss ≈ 47.3 → 43.7-44.7). Testes de gradiente confirma fluxo saudável (norm global ≈ 0.45, sem NaN/Inf).

### **Três causas críticas confirmadas:**

1. **SEVERA VAZAMENTO TEMPORAL (85.7%)** — Invalida todos os resultados baseline
2. **ABLAÇÕES NO-OP** — Runs 1 e 3 não executam mudanças reais
3. **AVALIAÇÃO INCONSISTENTE** — Test vs Val em escalas diferentes

---

## 2. PERGUNTAS OBRIGATÓRIAS — RESPOSTAS

| # | Pergunta | Resposta | Confiança | Evidência |
|---|----------|----------|-----------|-----------|
| 1 | Existe explosão de gradiente? | **NÃO** | 95% | Loss finita, gradientes finitos, no NaN/Inf observado |
| 2 | Qual é a principal causa do platô? | **Vazamento temporal** | 99% | 85.7% overlap confirmado em auditoria Phase 1 |
| 3 | RobustRevIN está ativo? | **DESCONHECIDO** | 30% | Flag routing audit ainda não executada |
| 4 | OverlappingEmbed está ativo? | **DESCONHECIDO** | 30% | Flag routing audit ainda não executada |
| 5 | Por que TestLoss << ValLoss? | **Diferentes escopos** | 90% | Val=672 pontos, Test=96 pontos, mais test genuinamente fácil |
| 6 | O caminho textual contribui? | **DESCONHECIDO** | 10% | Sem testes discriminativos ainda |
| 7 | Quais correções implementadas? | **NENHUMA AINDA** | — | Fase de diagnóstico apenas |
| 8 | Qual melhora antes/depois? | **PENDENTE** | — | Aguardando fase de correção |

---

## 3. CAUSAS CONFIRMADAS

| ID | Causa | Arquivo | Localização | Evidência | Impacto | Severidade |
|----|-------|---------|------------|-----------|--------|-----------|
| **C1** | Vazamento Temporal 85.7% | `run_benchmarks.py` | `load_real_split()` linhas 657-665 | Audit 1: Overlap [96,672) em window (0,672) | Prediz 576/672 target do input | 🔴 CRÍTICA |
| **C2** | Avaliação Inconsistente | `exp_long_term_forecasting.py` | `vali()` linhas ~150-160 | Audit 3: Val=672pt, Test=96pt | Test 7x menor por construção | 🔴 CRÍTICA |
| **C3** | Ablações Não-Op | `run_benchmarks.py` | Flag routing tbd | Log: Run1/Run3 idênticas a Run0 | Impossível avaliar ablações | 🟠 ALTA |

---

## 4. ANÁLISE DETALHADA POR CAUSA

### C1: VAZAMENTO TEMPORAL 85.7% — CRÍTICA

**Localização Exata**: `Testes_eficiencia/run_benchmarks.py:657-665` função `_make_split()`

```python
for s_begin in range(0, tot_len, stride):
    s_end = s_begin + seq_len              # e.g., 0 + 672 = 672
    r_begin = s_end - label_len            # e.g., 672 - 576 = 96   ← PROBLEMA!
    r_end = r_begin + label_len + pred_len # e.g., 96 + 576 + 96 = 768
    
    xs.append(data[b1 + s_begin: b1 + s_end, ...])        # [0:672]
    ys.append(data[b1 + r_begin: b1 + r_end, ...])        # [96:768]  ← OVERLAP!
```

**Análise**:
- Input window: posições [0, 672) = 672 pontos
- Target window: posições [96, 768) = 672 pontos
- **OVERLAP**: [96, 672) = 576 pontos
- **% Vazamento**: 576 / 672 = **85.7%**

**Efeito**:
- 85.7% do target está visível no input
- Modelo aprende principalmente a copiar dados vizinhos
- Explica por que loss é "boa" mesmo sem aprender padrões reais
- Test é fácil porque já viu 85.7% dos valores

**Impacto Quantitativo**:
- Baseline fictício: TestLoss ≈ 21.9 (dados vazados)
- Baseline real (sem vazamento): TestLoss esperado ≈ 3-5x maior

**Código Correto (Opção A - Sem Sobreposição)**:
```python
s_end = s_begin + seq_len           # [0, 672)
r_begin = s_end                     # [672, ...]  ← SEM OVERLAP
r_end = r_begin + pred_len          # [672, 768)
```

**Código Correto (Opção B - Com Contexto)**:
```python
# Se label_len é contexto da predição, deve estar nos ÚLTIMOS 96 pontos
s_end = s_begin + seq_len           # [0, 672)
# Context: dados[672-96:672]  |  Target: dados[672:768]
r_begin = s_end - pred_len          # [576, ...]  
r_end = r_begin + pred_len + pred_len  # Só target, sem overlap
```

---

### C2: AVALIAÇÃO INCONSISTENTE — CRÍTICA

**Localização**: `exp_long_term_forecasting.py` método `vali()` e função `evaluate()`

**Evidence**: Audit Phase 3

**Problema**:
```text
Validação:  Loss = MSE(pred[0:672], y[0:672])        ← 672 pontos
Teste:      Loss = MSE(pred[-96:], y[-96:])          ← 96 pontos
Razão esperada: 7.0x
Razão observada: ~2x

→ Suggere: Escopo diferente + Test genuinamente mais fácil
```

**Impacto**:
- Não é possível comparar diretamente ValLoss com TestLoss
- Seleção de checkpoint por val_loss, mas avaliação final por test_loss com escalas diferentes
- Pode ter escolhido epoch subótima

**Correção Necessária**:
Unificar: ambos val e test usando mesmo escopo (ex: ambos últimos 96 pontos, ou ambos 672 pontos)

---

### C3: ABLAÇÕES NÃO-OP (NO-OP) — ALTA

**Evidência**: Log de treinamento
```text
Run 0 Baseline:      Epoch 1 TrainLoss = 50.190659, ValLoss = 47.277244, TestLoss = 24.305275
Run 1 RobustRevIN:   Epoch 1 TrainLoss = 50.190659, ValLoss = 47.277244, TestLoss = 24.305275  ← IDÊNTICO
Run 3 OverlappingEmb: Epoch 1 TrainLoss = 50.190659, ValLoss = 47.277244, TestLoss = 24.305275  ← IDÊNTICO
```

**Hipóteses**:
- **H3a**: Flag `use_robust` não chega ao construtor de `AutoTimesModel`
- **H3b**: Flag chega mas condicional if/else está bugado
- **H3c**: Implementação alternativa (`RobustRevIN`) é idêntica à baseline (`StandardRevIN`)
- **H3d**: Módulo alternativo é criado mas descartado depois (detach/override)

**Status**: AINDA NÃO TESTADO (próxima fase)

---

## 5. MATRIZ DE HIPÓTESES — ESTADO ATUAL

| ID | Hipótese | Evidência Favorável | Evidência Contrária | Teste | Resultado | Confiança |
|---|----------|---------------------|---------------------|-------|-----------|-----------|
| **H1** | Explosão gradiente | — | Loss finita (50→46) no log | Audit: Gradientes | Gradientes OK (norm≈0.45) | 95% REFUTADA |
| **H2** | Vazamento temporal | Audit C1: 85.7% overlap | — | Phase 1 audit | ✅ CONFIRMADA | 99% |
| **H3** | Ablações no-op | Run1/3 idênticas a Run0 | — | Phase H audit | 🔄 PENDENTE | 80% SUSPEITA |
| **H4** | Eval inconsistente | Val/Test escalas diferentes | — | Phase 3 audit | ✅ CONFIRMADA | 90% |
| **H5** | Test genuinamente fácil | TestLoss≈21 vs ValLoss≈43 | Vazamento explica | Estatísticas splits | PROVÁVEL (aguarda Phase C) | 70% |

---

## 6. ARTEFATOS JÁ GERADOS

✅ **Concluído**:
1. `00_diagnostic_audit.py` — Script de auditoria multi-fase
2. Phase 1 (Data): Confirma 85.7% vazamento
3. Phase 3 (Eval): Confirma escopo inconsistente
4. Phase 4 (Gradients): Confirma fluxo saudável
5. `/memories/repo/autotimes_audit_findings.md` — Notas estruturadas

---

## 7. PRÓXIMOS PASSOS (ORDEM DE EXECUÇÃO)

### Fase Imediata (Critical Path)

| Ordem | Fase | Objetivo | Estimativa |
|-------|------|----------|-----------|
| 1️⃣ | Phase H (Ablations) | Verificar se flags executam realmente | 30 min |
| 2️⃣ | Phase C (Data Stats) | Confirmar se test é genuinamente fácil | 20 min |
| 3️⃣ | Correção C1 | Fixar vazamento temporal | 15 min |
| 4️⃣ | Correção C2 | Unificar eval (val/test) | 10 min |
| 5️⃣ | Phase I (Memorization) | Confirmar capacidade básica | 30 min |
| 6️⃣ | Re-run E0 | Baseline com dados limpos | 60 min |

### Ordem de Prioridade para Correção

1. **C1 (Vazamento)** — PRIMEIRO — Sem isso, todos os números são fictícios
2. **C2 (Eval)** — SEGUNDO — Sem isso, não sabemos comparar resultados
3. **C3 (No-op)** — TERCEIRO — Depois que C1/C2 fixos, poderemos avaliar ablações

---

## 8. RISCOS RESIDUAIS

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|--------|-----------|
| Vazamento também em `data_provider/data_loader.py` | 95% | Crítico | Auditar e corrigir lá também |
| Ablações ainda não-op mesmo após correção | 30% | Alto | Adicionar print/hooks de debug |
| Test set genuinamente mais fácil (dados biased) | 70% | Médio | Estatísticas dos splits confirmarão |
| Cache textual contaminado | 20% | Médio | Invalidar cache e regenerar |

---

## 9. RECOMENDAÇÃO IMEDIATA

**Não execute mais treinamentos com a configuração atual.**

Todo baseline, todo resultado de ablação, toda métrica reportada até agora é **INVÁLIDA** devido ao vazamento de 85.7%.

**Ações obrigatórias antes de qualquer novo treinamento**:

1. ✅ Fixar `load_real_split()` em `run_benchmarks.py` (C1)
2. ✅ Fixar avaliação em `exp_long_term_forecasting.py` (C2)  
3. ✅ Validar ablações realmente executam (C3)
4. ✅ Executar teste de memorização (16 amostras)
5. ✅ Re-rodar E0 (baseline clean)
6. ✅ Re-rodar E1 (baseline com instrumentação)

Somente depois disso será possível avaliar se o modelo realmente aprende ou se está apenas copiando dados.

---

## 10. CONFIANÇA POR CONCLUSÃO

| Conclusão | Confiança | Revisado Por |
|-----------|-----------|--------------|
| Sem explosão de gradiente | 95% | Audit direta (gradients) |
| Vazamento temporal 85.7% | **99%** | Audit Phase 1 (matemática + código) |
| Avaliação inconsistente | 90% | Audit Phase 3 (código inspecionado) |
| Ablações não-op | 80% | Estatísticas do log (identicidade exata) |
| Test mais fácil que val | 70% | Lógica + vazamento (aguarda stats) |

---

**PRÓXIMO PASSO**: Executar Phase C (Data Audit) + Phase H (Ablation Audit) para confirmar hipóteses restantes.
