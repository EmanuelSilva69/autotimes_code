"""Smoke dos eixos de ESCRITA do Pseudo-JSON: pj_casas, pj_marcacao, pj_escala.

Tria as combinacoes com a MESMA janela/trecho fixo e mede tok/ponto com o
tokenizer real do repo, confere os invariantes da spec:

  - fidelidade: cada /t /s /r escrito volta ao valor de origem dentro da
    tolerancia do formato (0.5*10**-casas; x100 para escala inteiro);
  - determinismo e ausencia de notacao cientifica/NaN;
  - incompatibilidade validada (inteiro exige pj_casas=2);
  - o padrao (2 casas, marcacao normal, float) segue identico ao atual;
  - a escada de custo pc2 > pc1 > pc0 vale (menos casas = menos tokens).

Tabela final = rankeamento para escolher as variantes de treino A/B.

Uso: python -m testes_letras.smoke_pj_escrita
"""

from __future__ import annotations

import re
import sys

import numpy as np


def main() -> int:
    from . import caminhos
    from .run_etth1_texto import carregar_etth1  # puxa torch (ROCm x sklearn)
    from .texto_char_by_char import decompor_vmd
    from .texto_puro import estruturado as est
    from .texto_puro.formato import (
        FORMATO_PADRAO,
        PJ_CASAS_PADRAO,
        PJ_ESCALA_PADRAO,
        PJ_MARCACAO_PADRAO,
        FormatoTexto,
    )
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
    falhas = []

    def doc(fmt: FormatoTexto) -> str:
        return " ".join(
            est._pseudo_json_mstl(dt[1000 + i], float(t[i]), float(s[i]),
                                  float(r[i]), formato=fmt, idx=i)
            for i in range(N)
        ) + "."

    variantes = [
        ("padrao", FORMATO_PADRAO),
        ("nenhuma", FormatoTexto(data="nenhuma")),
        ("nenhuma+pc1", FormatoTexto(data="nenhuma", pj_casas=1)),
        ("nenhuma+pc0", FormatoTexto(data="nenhuma", pj_casas=0)),
        ("nenhuma+compacta",
         FormatoTexto(data="nenhuma", pj_marcacao="compacta")),
        ("nenhuma+inteiro",
         FormatoTexto(data="nenhuma", pj_escala="inteiro")),
        ("nenhuma+pc1+compacta",
         FormatoTexto(data="nenhuma", pj_casas=1, pj_marcacao="compacta")),
        ("nenhuma+inteiro+compacta",
         FormatoTexto(data="nenhuma", pj_escala="inteiro",
                      pj_marcacao="compacta")),
    ]

    # ---- validacao de incompatibilidade ------------------------------ #
    try:
        FormatoTexto(pj_escala="inteiro", pj_casas=1)
        falhas.append("inteiro + pj_casas=1 nao foi recusado")
    except ValueError:
        pass

    # ---- regex de parse por marcacao --------------------------------- #
    re_norm = re.compile(
        r"/t\s*(-?\d+(?:\.\d+)?)\s*/s\s*(-?\d+(?:\.\d+)?)\s*"
        r"/r\s*(-?\d+(?:\.\d+)?)")

    def valores_do_ponto(ponto: str, fmt: FormatoTexto):
        m = re_norm.search(ponto)
        if not m:
            return None
        vs = [float(g) for g in m.groups()]
        if fmt.pj_escala == "inteiro":
            vs = [v / 100.0 for v in vs]
        return vs

    print("=" * 96)
    print("PSEUDO-JSON | eixos de escrita x custo medido (GPT-2 do repo)")
    print("=" * 96)
    custos = {}
    for rot, fmt in variantes:
        d = doc(fmt)
        custos[rot] = ntok(d) / N
        ok = True

        # determinismo
        if doc(fmt) != d:
            falhas.append(f"{rot}: nao e deterministico")
            ok = False
        # sem notacao cientifica/NaN
        if re.search(r"e[+-]?\d", d.lower()) or "nan" in d.lower():
            falhas.append(f"{rot}: notacao cientifica/NaN no texto")
            ok = False

        # fidelidade dos valores (tolerancia do formato)
        if fmt.pj_escala == "inteiro":
            tol = 0.005 + 1e-9
        else:
            tol = 0.5 * 10 ** (-int(fmt.pj_casas)) + 1e-9
        pontos = d.split(" /") if ("/t" in d and fmt.pj_marcacao == "normal") \
            else None
        # parse robusto: separa pontos pelos marcadores /t (ambas marcacoes)
        fragmentos = re.split(r"(?=/t(?:\s|[-\d]))", d)
        fragmentos = [f for f in fragmentos if re_norm.search(f)]
        if len(fragmentos) != N:
            falhas.append(f"{rot}: {len(fragmentos)} pontos com /t != {N}")
            ok = False
        else:
            for i, frag in enumerate(fragmentos):
                vs = valores_do_ponto(frag, fmt)
                if vs is None:
                    falhas.append(f"{rot}: ponto {i} sem /t /s /r: {frag[:40]!r}")
                    ok = False
                    break
                for valor, orig in zip(vs, (float(t[i]), float(s[i]),
                                            float(r[i]))):
                    if abs(valor - orig) > tol:
                        falhas.append(
                            f"{rot}: ponto {i} escreveu {valor} != {orig} "
                            f"(tol {tol:.4f})")
                        ok = False
                        break
                if not ok:
                    break

        # invariantes especificos
        if fmt.data == "nenhuma" and ("/time" in d or "horas" in d):
            falhas.append(f"{rot}: tem carimbo de data")
            ok = False
        if fmt.pj_marcacao == "compacta" and " /t " in d:
            falhas.append(f"{rot}: marcacao compacta tem ' /t ' com espaco")
            ok = False
        if fmt.pj_escala == "inteiro" and re.search(r"/t\s*-?\d+\.", d):
            falhas.append(f"{rot}: escala inteira tem ponto decimal")
            ok = False

        marca = "OK " if ok else "ERR"
        print(f"  [{marca}] {rot:24s} {custos[rot]:5.1f} tok/ponto -> "
              f"{int(orc // max(1, round(custos[rot]))):3d} pontos/horas "
              f"nos {orc}")
        print(f"         ex: {d[:76]!r}")

    # ---- escada de custo das casas (pc1 pode empatar: o BPE do GPT-2
    #      mergeia "15.2" e "15.19" com o mesmo custo; pc0 e o passo que
    #      garante economia ao tirar o ponto decimal) ----------------- #
    if not custos["nenhuma+pc1"] <= custos["nenhuma"]:
        falhas.append(f"pc1 ficou mais caro que pc2 ({custos['nenhuma+pc1']:.2f}"
                      f" > {custos['nenhuma']:.2f})")
    if not custos["nenhuma+pc0"] < custos["nenhuma+pc1"]:
        falhas.append(f"custo nao caiu de pc1 para pc0 "
                      f"({custos['nenhuma+pc1']:.2f} -> "
                      f"{custos['nenhuma+pc0']:.2f})")

    # ---- padrao byte a byte ------------------------------------------ #
    ponto_padrao = est._pseudo_json_mstl(dt[1000], float(t[0]), float(s[0]),
                                         float(r[0]))
    esperado = (f"/time {est._tempo_keywords_politica(dt[1000], 'es', 'completa')} "
                f"/t {float(t[0]):.2f} /s {float(s[0]):.2f} "
                f"/r {float(r[0]):.2f}")
    if ponto_padrao != esperado:
        falhas.append(f"padrao mudou: {ponto_padrao!r} != {esperado!r}")
        print(f"  [ERR] padrao: {ponto_padrao!r}")
    else:
        print(f"  [OK ] padrao identico ao atual: {ponto_padrao!r}")

    print("-" * 96)
    print("  RANKEAMENTO (menor = mais barato; escolha as top-2 p/ treino):")
    for rot, c in sorted(custos.items(), key=lambda kv: kv[1]):
        print(f"    {c:5.1f} tok/ponto  {int(orc // max(1, round(c))):3d} "
              f"pontos/horas  {rot}")
    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("OK (eixos de escrita do PJ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
