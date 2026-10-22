"""Smoke do BANNER do formato de texto (custo medido) e da flag Self-Refine.

Verifica que a rodada imprime, a partir de UMA janela real tokenizada com o
GPT-2 do repo:

  - `tok/ponto` e `pontos visiveis` no orcamento, lado a lado com a linha
    `[padrao] ...` quando uma variante foi escolhida;
  - `<-- FORA DO PADRAO` quando alguma flag saiu do padrao;
  - `SELF-REFINE: ATIVO` quando a flag esta LIGADA e NADA quando ela
    esta desligada (padrao).

Uso: python -m testes_letras.smoke_banner_formato
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

N_TREINO = 8640


def _args(**kw):
    from .run_apples_to_apples_trio import parse_args

    argv = sys.argv
    sys.argv = ["run_apples_to_apples_trio"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    args.feats_list = list(range(7))
    args.texto_decomp = "vmd"
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _capturar(args, dados, z, tok) -> str:
    from .run_apples_to_apples_trio import _imprimir_banner_formato

    buf = io.StringIO()
    with redirect_stdout(buf):
        _imprimir_banner_formato(args, tok, dados, z)
    return buf.getvalue()


def main() -> int:
    from . import caminhos
    from .run_etth1_texto import carregar_etth1  # puxa torch antes do sklearn

    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(caminhos.descobrir_tokenizador()))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    dados = carregar_etth1(None, N_TREINO + 1673)
    z = StandardScaler().fit(dados["values"][:N_TREINO]).transform(dados["values"])

    falhas = []

    # ---- 1) padrao: so a linha de referencia ------------------------- #
    saida = _capturar(_args(), dados, z, tok)
    print(saida)
    if "FORMATO DO TEXTO DE ENTRADA" not in saida:
        falhas.append("banner nao imprimiu o cabecalho")
    if "tok/ponto" not in saida:
        falhas.append("banner nao imprimiu tok/ponto")
    if "[padrao]" not in saida:
        falhas.append("banner nao imprimiu a linha [padrao]")
    if "[escolhido]" in saida:
        falhas.append("padrao nao devia imprimir [escolhido]")
    if "SELF-REFINE" in saida:
        falhas.append("SELF-REFINE apareceu sem a flag (tem de ser off)")
    for trecho, (lo, hi) in (
        ("Pseudo-JSON", (15.0, 30.0)),
        ("Char-by-Char", (10.0, 20.0)),
    ):
        linha = next((l for l in saida.splitlines() if trecho in l), "")
        try:
            v = float(linha.split("tok/ponto")[0].split()[-1])
        except (ValueError, IndexError):
            falhas.append(f"nao consegui ler tok/ponto de {trecho}: {linha!r}")
            continue
        if not lo <= v <= hi:
            falhas.append(f"{trecho} {v} tok/ponto fora da faixa {lo}-{hi}")
        vis = int(linha.split("->")[1].split()[0])
        if not 0 < vis <= 896:
            falhas.append(f"{trecho}: pontos visiveis {vis} absurdo")

    # ---- 2) variante + self-refine ligados --------------------------- #
    saida2 = _capturar(
        _args(texto_decimais=3, texto_data="nenhuma", self_refine=True),
        dados, z, tok,
    )
    print(saida2)
    if "FORA DO PADRAO" not in saida2:
        falhas.append("variante nao marcou '<-- FORA DO PADRAO'")
    if "decimais=3" not in saida2 or "data=nenhuma" not in saida2:
        falhas.append("rotulo do formato nao listou os eixos fora do padrao")
    if "[padrao]" not in saida2 or "[escolhido]" not in saida2:
        falhas.append("sem variante o banner tem de mostrar padrao x escolhido")
    if "SELF-REFINE: ATIVO" not in saida2:
        falhas.append("flag ligada -> banner nao imprimiu 'SELF-REFINE: ATIVO'")
    if "autotimes_memory.json" not in saida2:
        falhas.append("banner nao mostrou o caminho da memoria")

    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("OK (banner de custo + flag self-refine)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
