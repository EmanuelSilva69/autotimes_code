"""
phase_h_ablation_verification.py - Verificação de Ablações (Fase H)

Testa se os flags realmente mudam o comportamento do modelo.
Para cada flag, carrega o mesmo batch, mesmo peso, e compara saídas.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))

try:
    from run_benchmarks import AutoTimesModel
except ImportError as e:
    print(f"❌ Não conseguiu importar AutoTimesModel: {e}")
    sys.exit(1)

def test_flag_effect(flag_name: str, flag_value: bool, seq_len=672, pred_len=96, n_vars=7, device="cpu"):
    """
    Testa se um flag muda efetivamente o output do modelo.
    
    Usa seed fixa e batch fixo para comparação bit-a-bit.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Criar batch fixo
    batch_x = torch.randn(2, seq_len, n_vars, dtype=torch.float32)
    
    # Configs
    kwargs = {
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_vars": n_vars,
        "label_len": 576,
        "gpt2_dir": ROOT_DIR / "gpt2",
        "use_robust": False,
        "use_overlap": False,
        "use_vol_gate": False,
        "unfreeze_layers": 0,
    }
    
    # Baseline
    model_baseline = AutoTimesModel(**kwargs)
    model_baseline.eval()
    model_baseline = model_baseline.to(device)
    
    with torch.no_grad():
        out_baseline = model_baseline(batch_x.to(device))
    
    # Com flag
    kwargs_with_flag = kwargs.copy()
    kwargs_with_flag[flag_name] = flag_value
    
    model_with_flag = AutoTimesModel(**kwargs_with_flag)
    model_with_flag.eval()
    model_with_flag = model_with_flag.to(device)
    
    with torch.no_grad():
        out_with_flag = model_with_flag(batch_x.to(device))
    
    # Comparação
    diff_abs = (out_baseline - out_with_flag).abs()
    max_diff = diff_abs.max().item()
    mean_diff = diff_abs.mean().item()
    allclose = torch.allclose(out_baseline, out_with_flag, atol=1e-5)
    
    # Verifica se módulos mudaram
    baseline_type = type(model_baseline.revin).__name__ if hasattr(model_baseline, 'revin') else None
    flag_type = type(model_with_flag.revin).__name__ if hasattr(model_with_flag, 'revin') else None
    
    result = {
        "flag": flag_name,
        "value": flag_value,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "outputs_identical": bool(allclose),
        "baseline_revin_type": baseline_type,
        "flag_revin_type": flag_type,
        "types_differ": baseline_type != flag_type if baseline_type and flag_type else None,
    }
    
    return result

def main():
    print("=" * 100)
    print("PHASE H — VERIFICAÇÃO DE ABLAÇÕES")
    print("=" * 100)
    
    device = "cpu"  # Use CPU para compatibilidade
    
    flags_to_test = [
        ("use_robust", True),
        ("use_overlap", True),
        ("use_vol_gate", True),
        ("unfreeze_layers", 2),  # Unfreeze last 2 layers
    ]
    
    results = {}
    
    print(f"\nTestando cada flag com batch fixo e pesos aleatórios...")
    print(f"Device: {device}")
    print(f"Seed: 42 (fixo)")
    print(f"Batch: (2, 672, 7) - determinístico\n")
    
    for flag_name, flag_value in flags_to_test:
        print(f"Testing: {flag_name}={flag_value}...")
        try:
            result = test_flag_effect(flag_name, flag_value, device=device)
            results[flag_name] = result
            
            print(f"  ✓ Max diff: {result['max_diff']:.2e}")
            print(f"  ✓ Mean diff: {result['mean_diff']:.2e}")
            print(f"  ✓ Outputs identical: {result['outputs_identical']}")
            print(f"  ✓ RevIN types: {result['baseline_revin_type']} → {result['flag_revin_type']}")
            
            if result['outputs_identical']:
                print(f"  🔴 FLAG NO-OP (saída idêntica)")
            else:
                print(f"  ✅ FLAG ATIVO (saída diferente)")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Erro: {e}\n")
            results[flag_name] = {"error": str(e)}
    
    # Resumo
    print("=" * 100)
    print("RESUMO")
    print("=" * 100)
    print(f"\n{'Flag':25} | {'Status':15} | {'Max Diff':15} | {'Tipos Diferem':15}")
    print("-" * 75)
    
    for flag_name, result in results.items():
        if "error" in result:
            status = "❌ ERRO"
            diff = "—"
            types = "—"
        else:
            status = "✅ ATIVO" if not result["outputs_identical"] else "🔴 NO-OP"
            diff = f"{result['max_diff']:.2e}"
            types = "SIM" if result["types_differ"] else "NÃO" if result["types_differ"] is False else "—"
        
        print(f"{flag_name:25} | {status:15} | {diff:15} | {types:15}")
    
    # Salvar
    output_file = BASE_DIR / "log" / "phase_h_ablation_verification.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos em: {output_file}")
    
    # Diagnóstico
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO")
    print("=" * 100)
    
    no_ops = [flag for flag, result in results.items() 
              if "error" not in result and result.get("outputs_identical")]
    active = [flag for flag, result in results.items() 
              if "error" not in result and not result.get("outputs_identical")]
    errors = [flag for flag, result in results.items() if "error" in result]
    
    print(f"\n✅ Flags ATIVOS: {len(active)}")
    for flag in active:
        print(f"   - {flag}")
    
    print(f"\n🔴 Flags NO-OP: {len(no_ops)}")
    for flag in no_ops:
        result = results[flag]
        print(f"   - {flag} (diff={result['max_diff']:.2e})")
    
    if errors:
        print(f"\n❌ Erros: {len(errors)}")
        for flag in errors:
            print(f"   - {flag}")
    
    if no_ops:
        print(f"\n⚠️  ATENÇÃO: {len(no_ops)} flags estão não-op!")
        print(f"   Isso explica por que Run1 (RobustRevIN) e Run3 (OverlappingEmbed) têm resultados idênticos.")
    
    return results

if __name__ == "__main__":
    main()
