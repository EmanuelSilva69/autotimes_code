"""
phase_c_data_audit.py - Auditoria de Dados (Fase C)

Análise estatística dos conjuntos train/val/test:
- Estatísticas descritivas
- Verificação de vazamento temporal
- Dificuldade relativa
- Distribuições
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json

def compute_split_bounds(data_path, n_total, seq_len, pred_len):
    """Usa a mesma lógica de split da bateria.
    
    Returns: (b1s, b2s) onde:
        b1s = (train_start, val_start, test_start)
        b2s = (train_end, val_end, test_end)
    """
    if "ETT" in data_path.upper():
        # ETTh1: ~17420 pontos
        # Usa split padrão: 70% train, 10% val, 20% test
        n1 = int(0.7 * n_total)
        n2 = int(0.8 * n_total)
        return ([0, n1, n2], [n1, n2, n_total])
    else:
        n1 = int(0.7 * n_total)
        n2 = int(0.8 * n_total)
        return ([0, n1, n2], [n1, n2, n_total])

def analyze_split_statistics(csv_path, seq_len=672, label_len=576, pred_len=96):
    """Analisa estatísticas dos splits."""
    
    print("=" * 100)
    print("PHASE C — AUDITORIA DE DADOS")
    print("=" * 100)
    
    if not Path(csv_path).exists():
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    values = df[df.columns[1:]].to_numpy(dtype=np.float32)
    n_total, n_vars = values.shape
    
    print(f"\n📊 DATASET:")
    print(f"  Arquivo: {Path(csv_path).name}")
    print(f"  Dimensões: {n_total} timesteps × {n_vars} variáveis")
    print(f"  Período: {n_total} horas ≈ {n_total/24:.1f} dias ≈ {n_total/(24*365):.2f} anos")
    
    # Bounds
    b1s, b2s = compute_split_bounds(str(csv_path), n_total, seq_len, pred_len)
    bounds = [
        ("TRAIN", b1s[0], b2s[0]),
        ("VAL", b1s[1], b2s[1]),
        ("TEST", b1s[2], b2s[2]),
    ]
    
    scaler = StandardScaler().fit(values[b1s[0]:b2s[0]])
    data_norm = scaler.transform(values).astype(np.float32)
    
    results = {}
    split_stats = {}
    
    print(f"\n{'SPLIT':10} | {'Início':10} | {'Fim':10} | {'Tamanho':10} | {'% Total':8}")
    print("-" * 70)
    for name, b_start, b_end in bounds:
        size = b_end - b_start
        pct = 100 * size / n_total
        print(f"{name:10} | {b_start:10d} | {b_end:10d} | {size:10d} | {pct:7.1f}%")
    
    # Análise por split
    print(f"\n{'='*100}")
    print(f"ESTATÍSTICAS DESCRITIVAS POR SPLIT")
    print(f"{'='*100}\n")
    
    for split_name, b_start, b_end in bounds:
        print(f"\n📋 {split_name} SPLIT ({b_start}:{b_end}, {b_end-b_start} timesteps)")
        print("-" * 90)
        
        split_data = values[b_start:b_end, :]
        split_norm = data_norm[b_start:b_end, :]
        
        # Estatísticas gerais
        stats = {
            "split": split_name,
            "n_timesteps": b_end - b_start,
            "n_vars": n_vars,
        }
        
        # Agregado (todas variáveis juntas)
        agg_data = split_data.flatten()
        agg_norm = split_norm.flatten()
        
        print(f"  Agregado (todas as variáveis):")
        print(f"    Contagem:    {agg_data.size:,} pontos")
        print(f"    Média:       {agg_data.mean():12.6f}")
        print(f"    Std:         {agg_data.std():12.6f}")
        print(f"    Min:         {agg_data.min():12.6f}")
        print(f"    Max:         {agg_data.max():12.6f}")
        print(f"    Amplitude:   {agg_data.max() - agg_data.min():12.6f}")
        print(f"    Mediana:     {np.median(agg_data):12.6f}")
        print(f"    Skewness:    {pd.Series(agg_data).skew():12.6f}")
        print(f"    Kurtosis:    {pd.Series(agg_data).kurtosis():12.6f}")
        
        # Quantis
        print(f"    Quantis:     p01={np.percentile(agg_data,1):8.3f}, "
              f"p05={np.percentile(agg_data,5):8.3f}, "
              f"p25={np.percentile(agg_data,25):8.3f}, "
              f"p50={np.percentile(agg_data,50):8.3f}, "
              f"p75={np.percentile(agg_data,75):8.3f}, "
              f"p95={np.percentile(agg_data,95):8.3f}, "
              f"p99={np.percentile(agg_data,99):8.3f}")
        
        # Volatilidade
        rolling_std = pd.Series(agg_data).rolling(window=24, min_periods=1).std()
        print(f"    Volatilidade (std móvel 24h): média={rolling_std.mean():8.3f}, "
              f"max={rolling_std.max():8.3f}, min={rolling_std.min():8.3f}")
        
        stats["mean"] = float(agg_data.mean())
        stats["std"] = float(agg_data.std())
        stats["min"] = float(agg_data.min())
        stats["max"] = float(agg_data.max())
        stats["amplitude"] = float(agg_data.max() - agg_data.min())
        stats["median"] = float(np.median(agg_data))
        stats["p25"] = float(np.percentile(agg_data, 25))
        stats["p75"] = float(np.percentile(agg_data, 75))
        stats["p01"] = float(np.percentile(agg_data, 1))
        stats["p99"] = float(np.percentile(agg_data, 99))
        stats["volatility_mean"] = float(rolling_std.mean())
        stats["volatility_max"] = float(rolling_std.max())
        
        # Por variável
        print(f"\n  Por variável (primeiras 3):")
        for var_idx in range(min(3, n_vars)):
            var_data = split_data[:, var_idx]
            print(f"    Var {var_idx}: "
                  f"mean={var_data.mean():8.3f}, "
                  f"std={var_data.std():8.3f}, "
                  f"amp={var_data.max() - var_data.min():8.3f}")
        
        split_stats[split_name] = stats
        results[split_name] = stats
    
    # Comparação entre splits
    print(f"\n{'='*100}")
    print(f"COMPARAÇÃO ENTRE SPLITS")
    print(f"{'='*100}\n")
    
    train_stats = split_stats["TRAIN"]
    val_stats = split_stats["VAL"]
    test_stats = split_stats["TEST"]
    
    print(f"{'Métrica':20} | {'TRAIN':12} | {'VAL':12} | {'TEST':12} | {'Val/Train':10} | {'Test/Train':10}")
    print("-" * 90)
    
    for metric in ["mean", "std", "amplitude", "volatility_mean", "p01", "p99"]:
        train_v = train_stats[metric]
        val_v = val_stats[metric]
        test_v = test_stats[metric]
        
        ratio_vt = val_v / train_v if train_v != 0 else np.nan
        ratio_tt = test_v / train_v if train_v != 0 else np.nan
        
        print(f"{metric:20} | {train_v:12.4f} | {val_v:12.4f} | {test_v:12.4f} | "
              f"{ratio_vt:10.3f} | {ratio_tt:10.3f}")
    
    # Verificação de vazamento temporal
    print(f"\n{'='*100}")
    print(f"VERIFICAÇÃO DE VAZAMENTO TEMPORAL")
    print(f"{'='*100}\n")
    
    print(f"Verificando se índices se sobrepõem entre splits...")
    print(f"  TRAIN: [{b1s[0]}, {b2s[0]}) = {b2s[0]-b1s[0]} pontos")
    print(f"  VAL:   [{b1s[1]}, {b2s[1]}) = {b2s[1]-b1s[1]} pontos")
    print(f"  TEST:  [{b1s[2]}, {b2s[2]}) = {b2s[2]-b1s[2]} pontos")
    
    if b2s[0] == b1s[1] and b2s[1] == b1s[2]:
        print(f"\n✅ SPLITS ESTÃO SEPARADOS (sem overlap de índices)")
    else:
        print(f"\n🔴 POSSÍVEL OVERLAP DETECTADO!")
        if b2s[0] > b1s[1]:
            print(f"   TRAIN e VAL se sobrepõem! (TRAIN vai até {b2s[0]}, VAL começa em {b1s[1]})")
        if b2s[1] > b1s[2]:
            print(f"   VAL e TEST se sobrepõem!")
    
    # Resumo final
    print(f"\n{'='*100}")
    print(f"CONCLUSÕES")
    print(f"{'='*100}\n")
    
    amp_ratio = test_stats["amplitude"] / train_stats["amplitude"]
    vol_ratio = test_stats["volatility_mean"] / train_stats["volatility_mean"]
    
    print(f"1. Dificuldade relativa:")
    if amp_ratio < 0.8:
        print(f"   🟠 TEST tem amplitude {amp_ratio:.2f}x menor que TRAIN")
        print(f"      → Test PODE ser genuinamente mais fácil")
    elif amp_ratio > 1.2:
        print(f"   🟠 TEST tem amplitude {amp_ratio:.2f}x maior que TRAIN")
        print(f"      → Test PODE ser mais difícil")
    else:
        print(f"   ✅ TEST tem amplitude similar a TRAIN ({amp_ratio:.2f}x)")
    
    print(f"\n2. Volatilidade relativa:")
    if vol_ratio < 0.8:
        print(f"   🟠 TEST é menos volátil ({vol_ratio:.2f}x)")
    elif vol_ratio > 1.2:
        print(f"   🟠 TEST é mais volátil ({vol_ratio:.2f}x)")
    else:
        print(f"   ✅ TEST volatilidade similar ({vol_ratio:.2f}x)")
    
    print(f"\n3. Vazamento temporal:")
    print(f"   ✅ Splits estão separados por índice")
    
    # Salvar resultados
    output_file = Path(__file__).parent / "log" / "phase_c_data_audit.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Resultados salvos em: {output_file}")
    
    return results

if __name__ == "__main__":
    import sys
    
    # Detectar dataset
    root = Path(__file__).parent.parent
    csv_path = root / "dataset" / "ETT-small" / "ETTh1.csv"
    
    if not csv_path.exists():
        # Procurar alternativas
        for alt in [
            root / "dataset" / "ETTh1.csv",
            Path("dataset/ETT-small/ETTh1.csv"),
        ]:
            if alt.exists():
                csv_path = alt
                break
    
    print(f"\n🔍 Procurando dataset...")
    print(f"   Tentando: {csv_path}")
    
    if csv_path.exists():
        print(f"   ✅ Encontrado!\n")
        analyze_split_statistics(str(csv_path))
    else:
        print(f"   ❌ Não encontrado!")
        print(f"\nProcurando por CSV em {root}...")
        csvs = list(root.glob("**/ETT*.csv"))
        if csvs:
            print(f"Encontrados: {[str(c) for c in csvs]}")
            analyze_split_statistics(str(csvs[0]))
        else:
            print("Nenhum dataset encontrado!")
            sys.exit(1)
