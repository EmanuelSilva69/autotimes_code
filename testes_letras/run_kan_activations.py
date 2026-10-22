"""Roda KAN com 3 ativacoes (sigmoid, elu, selu) e compara resultados."""
import subprocess, json, sys, time, shutil, os

ATIVACOES = ["silu", "sigmoid", "elu", "selu"]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEEDS = sys.argv[2] if len(sys.argv) > 2 else "42"
OUT_DIR = "testes_letras/quinta_noite"

resultados = {}

for act in ATIVACOES:
    print(f"\n{'='*80}")
    print(f"  KAN Decoder - ativação: {act.upper()}")
    print(f"{'='*80}\n")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "testes_letras.run_apples_to_apples_trio",
        "--kan-decoder", "--kan-act", act,
        "--epochs", str(EPOCHS), "--seeds", SEEDS,
    ]
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    # Copia o veredito para nome unico antes da proxima execucao
    src = os.path.join(OUT_DIR, "veredito_trio.json")
    dst = os.path.join(OUT_DIR, f"veredito_kan_{act}.json")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        with open(dst) as f:
            data = json.load(f)
        pj = data.get("pseudo_json_es", {})
        cb = data.get("char_by_char", {})
        resultados[act] = {
            "pseudo_json_mse": pj.get("test_mse_media"),
            "char_by_char_mse": cb.get("test_mse_media"),
            "tempo_min": round(elapsed/60, 1),
        }
    print(f"\n>>> {act} concluído em {elapsed/60:.1f} min (exit={proc.returncode})")

# Tabela comparativa final
print(f"\n{'='*80}")
print("  RESUMO COMPARATIVO - KAN Decoder (10 ep, seed 42)")
print(f"{'='*80}")
print(f"{'Ativação':<12} {'Pseudo-JSON MSE':<18} {'Char-by-Char MSE':<18} {'Tempo (min)'}")
print("-"*65)

# Carrega baseline para referencia
freeze_mse = 0.3609
print(f"{'freeze MLP':<12} {freeze_mse:<18.4f} {'0.3651':<18} {'(ref)'}")
print("-"*65)
for act, r in resultados.items():
    pj = r.get("pseudo_json_mse", "-")
    cb = r.get("char_by_char_mse", "-")
    pj_str = f"{pj:.4f}" if isinstance(pj, (int, float)) else str(pj)
    cb_str = f"{cb:.4f}" if isinstance(cb, (int, float)) else str(cb)
    print(f"{act:<12} {pj_str:<18} {cb_str:<18} {r['tempo_min']:.1f}")
