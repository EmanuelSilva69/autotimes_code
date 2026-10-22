"""Golden do FORMATO PADRAO do texto de entrada (byte a byte).

Congela o documento Pseudo-JSON (VMD) e as 3 frases Char-by-Char (VMD) de
uma janela/trecho FIXOS do ETTh1 real, para provar que nenhuma alteracao de
formato mexeu no texto gerado quando nenhuma variante e selecionada.

Uso:
    python -m testes_letras.smoke_texto_padrao              # compara
    python -m testes_letras.smoke_texto_padrao --atualizar  # (re)grava

Saida esperada na comparacao: `OK (identico ao golden)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

GOLDEN_DIR: Path = Path(__file__).resolve().parent / "goldens"

# Entrada FIXA: linhas/feature deterministicas do ETTh1 canonico.
LINHA_INICIAL: int = 1000
JANELA_L: int = 672          # SEQ_LEN (mesma janela do pipeline)
N_PONTOS_PJ: int = 400       # trecho do documento PJ (rápido e fixo)
FEAT: int = 0                # HUFL
N_TREINO: int = 8640         # scaler fitado só no treino (fiel ao runner)

# Parâmetros do VMD = defaults oficiais (VMD_K_PADRAO/VMD_ALPHA_PADRAO).
_VMD = dict(k=7, alpha=2000.0, tau=0.0, tol=1e-7, init=0, trend_ma=None)


def _entradas() -> Dict[str, str]:
    """Documentos dos dois formatos com o padrao atual (sem variante)."""
    # ORDEM IMPORTA neste ambiente: sklearn antes de torch derruba o init do
    # ROCm (WinError 1114) - entao torch primeiro, sklearn depois.
    from .run_etth1_texto import carregar_etth1  # puxa torch

    from sklearn.preprocessing import StandardScaler

    dados = carregar_etth1(None, N_TREINO + 1)
    values = dados["values"]
    z = StandardScaler().fit(values[:N_TREINO]).transform(values)

    # ---- Char-by-Char: janela fixa -> 3 frases VMD --------------------- #
    from .texto_char_by_char import frases_texto_vmd

    xw = z[LINHA_INICIAL: LINHA_INICIAL + JANELA_L, FEAT].astype(np.float64)
    f_t, f_s, f_r, (m_t, m_s, m_r) = frases_texto_vmd(xw, **_VMD)
    cbc = "\n".join([f_t, f_s, f_r])
    cbc += f"\nmax_abs: {m_t!r} {m_s!r} {m_r!r}\n"

    # ---- Pseudo-JSON: trecho fixo do treino -> 1 documento ------------- #
    from .texto_puro import estruturado

    pj = estruturado.serie_para_texto_vmd(
        dados["dt_list"][:N_PONTOS_PJ], z, FEAT, N_PONTOS_PJ, **_VMD
    )
    pj += "\n"

    return {"char_by_char_vmd.txt": cbc, "pseudo_json_vmd.txt": pj}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Golden do formato padrao (byte a byte).",
    )
    ap.add_argument("--atualizar", action="store_true",
                    help="(re)grava os goldens em vez de comparar")
    args = ap.parse_args()

    docs = _entradas()

    if args.atualizar:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        for nome, txt in docs.items():
            (GOLDEN_DIR / nome).write_text(txt, encoding="utf-8")
            print(f"[golden] gravado: {GOLDEN_DIR / nome} ({len(txt):,} chars)")
        print("OK (goldens atualizados)")
        return 0

    falhas = []
    for nome, txt in docs.items():
        p = GOLDEN_DIR / nome
        if not p.exists():
            falhas.append(f"{nome}: golden ausente ({p}) - rode com --atualizar")
            continue
        esperado = p.read_text(encoding="utf-8")
        if esperado != txt:
            n = min(len(esperado), len(txt))
            i = next((j for j in range(n) if esperado[j] != txt[j]), n)
            falhas.append(
                f"{nome}: DIFERE em {i} (golden {len(esperado):,} chars x "
                f"atual {len(txt):,}) | golden: {esperado[i:i+40]!r} x "
                f"atual: {txt[i:i+40]!r}"
            )
        else:
            print(f"[golden] identico: {nome} ({len(txt):,} chars)")

    if falhas:
        print("FALHOU (texto de entrada mudou):")
        for f in falhas:
            print(f"  - {f}")
        return 1

    print("OK (identico ao golden)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
