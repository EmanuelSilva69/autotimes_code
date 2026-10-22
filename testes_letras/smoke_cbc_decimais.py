"""Smoke das variantes do CHAR-BY-CHAR: decimais (6/4/3/2) e encoding.

Confere os invariantes da spec na MESMA janela VMD fixa do golden:

  - escrita deterministica, SEM notacao cientifica ('e+'), sem NaN;
  - exatamente `casas` casas decimais escritas por ponto;
  - normalizacao local e max_abs INALTERADOS entre variantes (contrato);
  - menos decimais / digitos crus = menos tokens por ponto (mais pontos
    visiveis nas 7 paginas do orcamento);
  - o padrao (6, extenso) e identico ao formato atual.

Uso: python -m testes_letras.smoke_cbc_decimais
"""

from __future__ import annotations

import re
import sys

import numpy as np

L = 672
I0 = 1000
FEAT = 0


def main() -> int:
    from .run_etth1_texto import carregar_etth1  # puxa torch (ROCm x sklearn)
    from .texto_char_by_char import (
        FIM_PALAVRA,
        MARCADOR_RESIDUOS,
        MARCADOR_SAISONALIDADE,
        MARCADOR_TENDENCIA,
        frases_texto_vmd,
        texto_char_por_char,
    )
    from .texto_puro.formato import (
        DECIMAIS_PADRAO,
        DECIMAIS_VALIDOS,
        ENCODINGS,
        FORMATO_PADRAO,
        FormatoTexto,
        definir_formato,
    )
    from . import caminhos
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(caminhos.descobrir_tokenizador()))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def ntok(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    from sklearn.preprocessing import StandardScaler

    dados = carregar_etth1(None, I0 + L + 1)
    z = StandardScaler().fit(dados["values"][:8640]).transform(dados["values"])
    xw = z[I0: I0 + L, FEAT].astype(np.float64)

    falhas = []
    definir_formato(FORMATO_PADRAO)
    ref0 = texto_char_por_char(xw)[0]
    ref_t, ref_s, ref_r, ref_m = frases_texto_vmd(
        xw, k=7, alpha=2000.0, tau=0.0, tol=1e-7, init=0, trend_ma=None,
    )

    def pontos_de(frase: str) -> list:
        """Frases -> lista de pontos ('fim' removido do ultimo)."""
        partes = [p for p in corpo(frase).split(" sep ") if p]
        partes[-1] = partes[-1].removesuffix(f" {FIM_PALAVRA}")
        return partes

    print("=" * 96)
    print("CHAR-BY-CHAR | casas decimais x custo medido (GPT-2 do repo)")
    print("=" * 96)

    def corpo(frase: str) -> str:
        """Frase sem o marcador 'Tendencia:' (so os pontos)."""
        for marc in (MARCADOR_TENDENCIA, MARCADOR_SAISONALIDADE,
                     MARCADOR_RESIDUOS):
            if frase.startswith(marc + " "):
                return frase[len(marc) + 1:]
        return frase

    custos = {}
    for casas in DECIMAIS_VALIDOS:
        definir_formato(FormatoTexto(casas=casas))
        f_t, f_s, f_r, (m_t, m_s, m_r) = frases_texto_vmd(
            xw, k=7, alpha=2000.0, tau=0.0, tol=1e-7, init=0, trend_ma=None,
        )
        texto = corpo(f_t)
        custos[casas] = ntok(f_t) / L

        # ---- invariantes ------------------------------------------- #
        baixo = texto.lower()
        if "e+" in baixo or "nan" in baixo or "inf" in baixo:
            falhas.append(f"casas={casas}: notacao cientifica/NaN no texto")
        pontos = pontos_de(f_t)
        if len(pontos) != L:
            falhas.append(f"casas={casas}: {len(pontos)} pontos != {L}")
        else:
            for p in pontos[:200]:
                if "ponto" not in p:
                    falhas.append(f"casas={casas}: ponto sem 'ponto': {p!r}")
                    break
                frac = p.split("ponto ", 1)[1]
                if len(frac.split()) != casas:
                    falhas.append(
                        f"casas={casas}: {len(frac.split())} casas escritas "
                        f"!= {casas} em {p!r}"
                    )
                    break
        # contrato max_abs identico ao padrao (so o texto muda)
        if (m_t, m_s, m_r) != ref_m:
            falhas.append(f"casas={casas}: max_abs mudou {ref_m} -> {(m_t, m_s, m_r)}")

        print(f"  {casas} casa(s)  {custos[casas]:5.1f} tok/ponto -> "
              f"{int((3 * 128) // max(1, round(custos[casas]))):3d} pontos na frase T "
              f"| max_abs {m_t:.4f}/{m_s:.4f}/{m_r:.4f}")
        print(f"              ex: {pontos[0][:72]!r}")

    for a, b in zip(DECIMAIS_VALIDOS, DECIMAIS_VALIDOS[1:]):
        if not custos[a] > custos[b]:
            falhas.append(f"custo nao caiu de {a} para {b} casas")

    print("-" * 96)
    print("ENCODING x custo medido (GPT-2 do repo)")
    print("-" * 96)
    custo_enc = {}
    for enc in ENCODINGS:
        definir_formato(FormatoTexto(encoding=enc))
        f_t, _, _, (m_t, _, _) = frases_texto_vmd(
            xw, k=7, alpha=2000.0, tau=0.0, tol=1e-7, init=0, trend_ma=None,
        )
        custo_enc[enc] = ntok(f_t) / L
        pontos = pontos_de(f_t)
        if enc == "digitos":
            expeito = re.compile(rf"-?\d+\.\d{{{DECIMAIS_PADRAO}}}")
            if not all(expeito.fullmatch(p) for p in pontos[:200]):
                ex = next((p for p in pontos[:200]
                           if not expeito.fullmatch(p)), "?")
                falhas.append(f"encoding=digitos: forma inesperada {ex!r}")
            if custo_enc[enc] >= custo_enc.get("extenso", float("inf")):
                falhas.append("encoding=digitos nao ficou mais barato que extenso")
        if m_t != ref_m[0]:
            falhas.append(f"encoding={enc}: max_abs do trend mudou")
        print(f"  {enc:9s} {custo_enc[enc]:5.1f} tok/ponto | ex: {pontos[0][:52]!r}")

    # ---- arredondamento x truncamento (novo eixo) -------------------- #
    print("-" * 96)
    print("ARREDONDAMENTO x ZEROS A DIREITA (novos eixos)")
    print("-" * 96)
    from .texto_char_by_char import _escalar_para_inteiros, _montar_texto

    xs = np.array([0.0005, -0.0005, 0.0004, 0.0015])
    trunc = _escalar_para_inteiros(xs, casas=3, arredondamento="truncar")
    arred = _escalar_para_inteiros(xs, casas=3, arredondamento="arredondar")
    if list(trunc) != [0, 0, 0, 1]:
        falhas.append(f"truncar (3 casas) inesperado: {list(trunc)}")
    if list(arred) != [1, -1, 0, 2]:
        falhas.append(f"arredondar (3 casas) inesperado: {list(arred)}")
    print(f"  0.0005/-0.0005/0.0004/0.0015 @3casas -> truncar {list(trunc)} | "
          f"arredondar {list(arred)}")

    definir_formato(FormatoTexto(arredondamento="arredondar"))
    f_ar, m_ar = texto_char_por_char(xw)
    if texto_char_por_char(xw)[0] != f_ar:
        falhas.append("arredondar nao e deterministico")
    if f_ar == ref0:
        falhas.append("arredondar nao mudou nenhum valor do texto")
    if "e+" in f_ar.lower() or "nan" in f_ar.lower():
        falhas.append("arredondar introduziu notacao cientifica/NaN")

    # zeros=remover: sintetico (casos garantidos) + janela real
    def esc(v):
        return _escalar_para_inteiros(np.asarray(v, dtype=np.float64),
                                      casas=6, arredondamento="truncar")

    sint = _montar_texto(esc([0.5, -0.5, 0.0, 0.123456]), casas=6,
                         encoding="extenso", zeros="remover")
    partes_sint = sint.split(" sep ")
    partes_sint[-1] = partes_sint[-1].removesuffix(f" {FIM_PALAVRA}")
    casos = {
        0: "zero ponto cinco",
        1: "menos zero ponto cinco",
        2: "zero",
        3: "zero ponto um dois tres quatro cinco seis",
    }
    for i, esp in casos.items():
        if partes_sint[i] != esp:
            falhas.append(f"zeros=remover ponto {i}: {partes_sint[i]!r} != {esp!r}")
    sint_dig = _montar_texto(esc([0.5, 0.0]), casas=6,
                             encoding="digitos", zeros="remover")
    if sint_dig != "0.5 sep 0 fim":
        falhas.append(f"zeros=remover digitos: {sint_dig!r}")

    definir_formato(FormatoTexto(zeros="remover"))
    f_nz, m_nz = texto_char_por_char(xw)
    if texto_char_por_char(xw)[0] != f_nz:
        falhas.append("zeros=remover nao e deterministico")
    definir_formato(FORMATO_PADRAO)
    if m_nz != texto_char_por_char(xw)[1]:
        falhas.append("zeros=remover mudou o max_abs (contrato)")
    pts_nz = [p for p in f_nz.split(" sep ")]
    pts_nz[-1] = pts_nz[-1].removesuffix(f" {FIM_PALAVRA}")
    ruins = [p for p in pts_nz if "ponto" in p.split()
             and p.split()[-1] == "zero"]
    if ruins:
        falhas.append(f"zeros=remover deixou 'zero' final: {ruins[0]!r}")

    definir_formato(FORMATO_PADRAO)
    f_manter, _ = texto_char_por_char(xw)
    custo_manter = ntok(f_manter) / L
    definir_formato(FormatoTexto(zeros="remover"))
    f_remover, _ = texto_char_por_char(xw)
    definir_formato(FORMATO_PADRAO)
    custo_remover = ntok(f_remover) / L
    definir_formato(FormatoTexto(arredondamento="arredondar"))
    custo_arred = ntok(texto_char_por_char(xw)[0]) / L
    definir_formato(FORMATO_PADRAO)
    if not custo_remover < custo_manter:
        falhas.append("zeros=remover nao ficou mais barato que manter "
                      f"({custo_remover:.3f} vs {custo_manter:.3f})")
    print(f"  zeros=manter   {custo_manter:5.2f} tok/ponto (padrao)")
    print(f"  zeros=remover  {custo_remover:5.2f} tok/ponto "
          f"(-{100 * (1 - custo_remover / custo_manter):.1f}%)")
    print(f"  arredondar     {custo_arred:5.2f} tok/ponto "
          "(mesmo custo do truncar; so muda o valor)")

    # ---- padrao byte a byte (e nenhum vazamento de formato) --------- #
    definir_formato(FORMATO_PADRAO)
    if texto_char_por_char(xw)[0] != ref0:
        falhas.append("o padrao (6/extenso) nao reproduz o texto atual "
                      "(vazou formato variante?)")

    print("-" * 96)
    print(f"  6 casas -> 3 casas: -{100 * (1 - custos[3] / custos[6]):.0f}% tok/ponto")
    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("OK (variantes do char-by-char)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
