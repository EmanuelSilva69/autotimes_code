"""Smoke das variantes de DATA no Pseudo-JSON (custo medido no GPT-2 real).

Gera os pontos das 5 politicas para a MESMA janela/trecho fixo, mede
tok/ponto com o tokenizer do repo e confere os invariantes da spec:

  - 'nenhuma' nao tem NENHUM token de data (o ponto comeca em /t);
  - 'completa' e o padrao e continua identico ao formato atual;
  - menos data = menos tokens por ponto (mais horas nos 896 tokens);
  - a escrita e deterministica (mesma entrada -> mesmo texto).

Uso: python -m testes_letras.smoke_pj_data
"""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np


def main() -> int:
    # torch antes de sklearn (WinError 1114 do ROCm neste ambiente).
    from . import caminhos
    from .run_etth1_texto import carregar_etth1  # puxa torch
    from .texto_char_by_char import decompor_vmd
    from .texto_puro import estruturado as est
    from .texto_puro.formato import DATA_PADRAO, FORMATO_PADRAO, POLITICAS_DATA, FormatoTexto

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(caminhos.descobrir_tokenizador()))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def ntok(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    dados = carregar_etth1(None, 1672)
    dt = dados["dt_list"]
    xw = np.asarray(dados["values"][1000:1672, 0], dtype=np.float64)
    t, s, r = decompor_vmd(xw, k=7, alpha=2000.0, tau=0.0, tol=1e-7,
                           init=0, trend_ma=None)

    N = 48
    orc = 7 * 128
    linhas, falhas = [], []

    def doc(politica: str) -> str:
        fmt = FORMATO_PADRAO if politica == DATA_PADRAO else FormatoTexto(data=politica)
        return " ".join(
            est._pseudo_json_mstl(dt[1000 + i], float(t[i]), float(s[i]),
                                  float(r[i]), formato=fmt, idx=i)
            for i in range(N)
        ) + "."

    print("=" * 96)
    print("PSEUDO-JSON | politica de data x custo medido (GPT-2 do repo)")
    print("=" * 96)
    custos = {}
    for pol in POLITICAS_DATA:
        d = doc(pol)
        custos[pol] = ntok(d) / N
        existe_time = "/time" in d
        print(f"  {pol:9s} {custos[pol]:5.1f} tok/ponto -> "
              f"{int(orc // max(1, round(custos[pol]))):3d} pontos/horas nos "
              f"{orc} tokens | /time={'sim' if existe_time else 'nao'}")
        print(f"             ex: {d[:78]!r}")

        # ---- invariantes ------------------------------------------- #
        if pol == "nenhuma":
            if "/time" in d or "/i " in d:
                falhas.append("'nenhuma' ainda tem carimbo temporal")
            if not d.startswith("/t "):
                falhas.append("'nenhuma' nao comeca em '/t'")
            mes = est._MESES_ES[dt[1000].month - 1]
            if mes in d:
                falhas.append(f"'nenhuma' contem o mes {mes!r}")
        if pol == "completa":
            if not d.startswith("/time "):
                falhas.append("'completa' nao comeca em '/time'")
            if est._MESES_ES[dt[1000].month - 1] not in d:
                falhas.append("'completa' perdeu o mes por extenso")
        if pol == "indice":
            if not d.startswith("/i 0 "):
                falhas.append("'indice' nao comeca em '/i 0 '")
        if pol == "hora":
            cabeca = d.split(" /t ")[0]
            if not cabeca.startswith("/time "):
                falhas.append("'hora' nao comeca em '/time '")
            if est._MESES_ES[dt[1000].month - 1] in cabeca:
                falhas.append("'hora' ainda tem o mes")
        if pol == "curta":
            cabeca = d.split(" /t ")[0]
            if est._MESES_ES[dt[1000].month - 1] in cabeca:
                falhas.append("'curta' ainda tem o mes")
            if not cabeca.startswith("/time "):
                falhas.append("'curta' nao comeca em '/time '")
        # determinismo
        if doc(pol) != d:
            falhas.append(f"'{pol}' nao e deterministico")

    if not custos["nenhuma"] < custos["completa"]:
        falhas.append("'nenhuma' nao ficou mais barato que 'completa'")
    if not custos["hora"] < custos["curta"] < custos["completa"]:
        falhas.append("ordem de custo hora < curta < completa violada")

    # padrao = FORMATO_PADRAO (data completa) byte a byte com o formato atual
    primeiro = est._pseudo_json_mstl(dt[1000], float(t[0]), float(s[0]),
                                     float(r[0]))
    if not doc(DATA_PADRAO).startswith(primeiro):
        falhas.append("o padrao nao e o formato atual do ponto")

    print("-" * 96)
    ganho = 100 * (custos["completa"] - custos["nenhuma"]) / custos["completa"]
    print(f"  ganho de 'nenhuma' sobre 'completa': -{ganho:.0f}% tok/ponto")
    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("OK (variantes de data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
