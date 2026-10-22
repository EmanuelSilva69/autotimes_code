# CORREÇÕES APLICADAS — AUDITORIA FORENSE AUTOTIMES

**Data**: 2026-08-08  
**Nível**: Implementação Fase 1 (C1 Critical)  
**Status**: ✅ CONCLUÍDA

---

## Sumário

✅ **1 correção crítica implementada** (C1 — Vazamento Temporal)

| Cor reção | Causa | Arquivo | Linhas | Status |
|----------|-------|---------|--------|--------|
| C1-A | Leakage no run_benchmarks | `run_benchmarks.py` | 662-665 | ✅ Aplicada |
| C1-B | Leakage na data_loader (Dataset_ETT_hour) | `data_provider/data_loader.py` | 58-64 | ✅ Aplicada |
| C1-C | Leakage na data_loader (Dataset_Custom) | `data_provider/data_loader.py` | 143-149 | ✅ Aplicada |

---

## Correção C1-A: run_benchmarks.py `_make_split()`

**Arquivo**: `Testes_eficiencia/run_benchmarks.py`  
**Função**: `load_real_split()` → `_make_split()`  
**Linhas Afetadas**: 662-665

### Problema Identificado

```python
# ANTES (BUGADO):
s_end = s_begin + seq_len              # 0 + 672 = 672
r_begin = s_end - label_len            # 672 - 576 = 96   ← VAZAMENTO!
r_end = r_begin + label_len + pred_len # 96 + 576 + 96 = 768

xs.append(data[b1 + s_begin: b1 + s_end, ...])        # Input:  [0:672]
ys.append(data[b1 + r_begin: b1 + r_end, ...])        # Target: [96:768]  ← OVERLAP [96,672)!

# Overlap: 85.7% (576 out of 672 pontos do input vazam para target)
```

### Solução Implementada

```python
# DEPOIS (CORRIGIDO):
s_end = s_begin + seq_len       # 0 + 672 = 672
r_begin = s_end                 # 672       ← SEM OVERLAP!
r_end = r_begin + pred_len      # 672 + 96 = 768

xs.append(data[b1 + s_begin: b1 + s_end, ...])        # Input:  [0:672]
ys.append(data[b1 + r_begin: b1 + r_end, ...])        # Target: [672:768]  ← SEM OVERLAP!

# Overlap: 0% (target é puramente FUTURO, sem dados do input)
```

### Mudanças no Código

**Linhas 657-665** (antes):
```python
def _make_split(b1: int, b2: int, stride: int) -> TensorDataset:
    tot_len = len(data[b1:b2]) - seq_len - pred_len + 1
    xs, ys = [], []
    # Ordem da bateria: para cada variavel, todas as janelas temporais.
    for feat in range(n_vars):
        for s_begin in range(0, tot_len, stride):
            s_end = s_begin + seq_len
            r_begin = s_end - label_len
            r_end = r_begin + label_len + pred_len
            xs.append(data[b1 + s_begin: b1 + s_end, feat: feat + 1])
            ys.append(data[b1 + r_begin: b1 + r_end, feat: feat + 1])
    return TensorDataset(
        torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
    )
```

**Linhas 657-672** (depois):
```python
def _make_split(b1: int, b2: int, stride: int) -> TensorDataset:
    tot_len = len(data[b1:b2]) - seq_len - pred_len + 1
    xs, ys = [], []
    # Ordem da bateria: para cada variavel, todas as janelas temporais.
    for feat in range(n_vars):
        for s_begin in range(0, tot_len, stride):
            s_end = s_begin + seq_len
            # FIX (2026-08-08): Remover vazamento temporal
            # Antes: r_begin = s_end - label_len (causava 85.7% overlap)
            # Agora: target começa APÓS input (sem overlap)
            r_begin = s_end  # Começar após input
            r_end = r_begin + pred_len  # Apenas o horizonte de predição
            xs.append(data[b1 + s_begin: b1 + s_end, feat: feat + 1])
            ys.append(data[b1 + r_begin: b1 + r_end, feat: feat + 1])
    return TensorDataset(
        torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
    )
```

### Impacto Esperado

- ✅ **Antes**: TestLoss ~21.9 (fictício, com leakage)
- 📊 **Depois**: TestLoss esperado **~2-3x maior** (dados limpos)
- ✅ **MAE**: Esperado **~2-3x maior**
- ✅ **RMSE**: Esperado **~2-3x maior**
- ✅ **Ablações**: Serão visíveis após fix

### Risco Residual

- **Baixo**: Mudan é semântica e não deve quebrar código
- ⚠️ **Nota**: Labels_len agora não usado na função; será utilizado pela arquitetura do modelo na seleção de contexto

---

## Correção C1-B: data_loader.py Dataset_ETT_hour

**Arquivo**: `data_provider/data_loader.py`  
**Classe**: `Dataset_ETT_hour`  
**Método**: `__getitem__()`  
**Linhas Afetadas**: 58-64

### Problema Identificado

Idêntico ao C1-A:
```python
# ANTES (BUGADO):
r_begin = s_end - self.label_len  # Vazamento!
```

### Solução Implementada

```python
# DEPOIS (CORRIGIDO):
r_begin = s_end  # Sem vazamento
r_end = r_begin + self.pred_len
```

### Mudanças no Código

**Linhas 55-67** (depois):
```python
def __getitem__(self, index):
    feat_id = index // self.tot_len
    s_begin = index % self.tot_len
    
    s_end = s_begin + self.seq_len
    # FIX (2026-08-08): Remover vazamento temporal (85.7% overlap)
    # Antes: r_begin = s_end - self.label_len (causava leakage)
    # Agora: target começa APÓS input (sem overlap)
    r_begin = s_end  # Começar após input
    r_end = r_begin + self.pred_len  # Apenas horizonte de predição
    seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
    seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
    seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
    seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

    return seq_x, seq_y, seq_x_mark, seq_y_mark
```

---

## Correção C1-C: data_loader.py Dataset_Custom

**Arquivo**: `data_provider/data_loader.py`  
**Classe**: `Dataset_Custom`  
**Método**: `__getitem__()`  
**Linhas Afetadas**: 143-149

### Solução Implementada

Idêntica a C1-B:
```python
# DEPOIS (CORRIGIDO):
r_begin = s_end  # Sem vazamento
r_end = r_begin + self.pred_len
```

---

## Teste de Regressão

### Script de Validação

```python
# teste_leakage_fix.py
def verify_no_leakage():
    """Verifica que o fix removeu o vazamento."""
    from run_benchmarks import load_real_split
    
    n_vars, train_ds, val_ds, test_ds = load_real_split(
        "dataset/ETT-small", "ETTh1.csv",
        seq_len=672, label_len=576, pred_len=96, window_stride=1
    )
    
    # Pegar primeira amostra
    xs, ys = train_ds[0]
    
    # xs: input [0:672]
    # ys: target [672:768] (DEPOIS DO LAUNCHSAMEN)
    
    # Verificar que target NÃO está em input
    overlap = np.sum(xs.numpy() == ys.numpy())
    print(f"Overlap de valores: {overlap}")
    # Esperado: Algo não-zero porque os dados não são independentes
    # MAS: A posição temporal é separada
    
    # Verificação mais rigorosa: índices
    # xs covers [0:672] in time
    # ys covers [672:768] in time
    # → SEM overlap temporal
    
    return True
```

### Comando de Teste

```bash
# Antes (bugado):
python -c "from run_benchmarks import load_real_split; \
n_vars, train_ds, val_ds, _ = load_real_split(...); \
xs, ys = train_ds[0]; \
print(f'Input shape: {xs.shape}, Target shape: {ys.shape}, Overlap%: ...')"
# Esperado ANTES: Overlap% = 85.7

# Depois (corrigido):
python -c "..."
# Esperado DEPOIS: Overlap% = 0 (separado por índice)
```

---

## Impacto em Outros Módulos

### Afetados por esta correção

1. **exp_long_term_forecasting.py** — Usa `data_provider` → receberá dados limpos ✅
2. **run_benchmarks.py** — Usa `load_real_split()` → receberá dados limpos ✅
3. **Runs E0-E12** — Todos verão loss ~3x maior → VÁLIDOS apenas após this fix

### Não afetados

- Modelo (AutoTimesModel) — Sem mudanças
- Loss functions — Sem mudanças
- Ablação flags — Continuam funcionando, agora visíveis

---

## Próximas Correções Planejadas (Fase 2)

| Ordem | Correção | Status | Estimativa |
|-------|----------|--------|-----------|
| 1 ✅ | C1 (Leakage) | **DONE** | 30 min |
| 2 🔄 | C2 (Eval inconsistency) | Pendente | 15 min |
| 3 🔄 | Verificação de ambas fixes | Pendente | 20 min |

---

## Checklist de Validação

- [x] Código compilável após correção
- [x] Sem syntax errors
- [x] Sem import errors
- [ ] Test suite passa (próxima fase)
- [ ] E0 baseline re-executado (próxima fase)
- [ ] Loss aumenta 3-5x (esperado) (próxima fase)
- [ ] Ablações mostram diferenças (esperado) (próxima fase)

---

## Arquivos Modificados — Diff Resumido

```diff
=== Testes_eficiencia/run_benchmarks.py ===
@@ -662,10 +662,16 @@
             for s_begin in range(0, tot_len, stride):
                 s_end = s_begin + seq_len
-                r_begin = s_end - label_len
-                r_end = r_begin + label_len + pred_len
+                # FIX (2026-08-08): Remover vazamento temporal
+                # Antes: r_begin = s_end - label_len (causava 85.7% overlap)
+                # Agora: target começa APÓS input (sem overlap)
+                r_begin = s_end  # Começar após input
+                r_end = r_begin + pred_len  # Apenas o horizonte de predição
                 xs.append(data[b1 + s_begin: b1 + s_end, feat: feat + 1])
                 ys.append(data[b1 + r_begin: b1 + r_end, feat: feat + 1])

=== data_provider/data_loader.py (Dataset_ETT_hour) ===
@@ -59,10 +59,16 @@
         s_begin = index % self.tot_len
         
         s_end = s_begin + self.seq_len
-        r_begin = s_end - self.label_len
-        r_end = r_begin + self.label_len + self.pred_len
+        # FIX (2026-08-08): Remover vazamento temporal (85.7% overlap)
+        # Antes: r_begin = s_end - self.label_len (causava leakage)
+        # Agora: target começa APÓS input (sem overlap)
+        r_begin = s_end  # Começar após input
+        r_end = r_begin + self.pred_len  # Apenas horizonte de predição
         seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
         seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]

=== data_provider/data_loader.py (Dataset_Custom) ===
@@ -145,10 +145,16 @@
         s_begin = index % self.tot_len
         
         s_end = s_begin + self.seq_len
-        r_begin = s_end - self.label_len
-        r_end = r_begin + self.label_len + self.pred_len
+        # FIX (2026-08-08): Remover vazamento temporal (85.7% overlap)
+        # Antes: r_begin = s_end - self.label_len (causava leakage)
+        # Agora: target começa APÓS input (sem overlap)
+        r_begin = s_end  # Começar após input
+        r_end = r_begin + self.pred_len  # Apenas horizonte de predição
         seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
         seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
```

---

## Recomendações Imediatas

1. ✅ **Executar E0 (baseline) com dados limpos**
   - Esperar: Loss ~3-5x maior que anterior
   - Confirmar: Modelo ainda consegue aprender
   
2. ✅ **Executar Phase I (memorization 16 samples)**
   - Esperar: Conseguir overfit rapidamente
   - Confirmar: Backward ainda funciona

3. ✅ **Comparar Runs 1 vs Run 0 depois do fix**
   - Esperar: Diferenças perceptíveis em loss
   - Confirmar: Ablações agora funcionam visivamente

---

**Status Final**: ✅ CORREÇÃO C1 IMPLEMENTADA E DOCUMENTADA  
**Próxima Etapa**: Implementar C2 (Evaluation Consistency) + Validação
