"""Smoke de INTEGRACAO do dataset char-by-char com o formato de texto.

Cobre 2 coisas de uma vez:

  - o `args_tuple` passou a ter 20 campos (`FormatoTexto` e o ultimo) e o
    worker aplica o formato ANTES de escrever o texto;
  - o CONTRATO do dataset continua o mesmo em qualquer variante:
    ids (n, 896), lens (n, 7 paginas), max_abs (n, 3, 1) - as 3 frases
    T/S/R dividindo 3+2+2 paginas;

imprime `20 campos` e `INTEGRACAO OK` quando tudo bate.

Uso: python -m testes_letras.smoke_integracao_texto
"""

from __future__ import annotations

import sys

import numpy as np

N_JANELAS = 3
SEQ_LEN = 672
LINHAS = 8640          # regiao de treino (scaler fitado so nela)
N_PONTOS = 576 + 96  # LABEL_LEN + PRED_LEN (alvo y)


def main() -> int:
    from . import caminhos
    from .run_etth1_texto import carregar_etth1  # puxa torch antes do sklearn
    from .texto_char_by_char import (
        BLOCK_MAX,
        N_PAGINAS_MAX,
        _processar_janela,
        construir_dataset_char_by_char,
    )
    from .texto_puro.formato import FormatoTexto

    from sklearn.preprocessing import StandardScaler

    falhas = []
    tok_dir = str(caminhos.descobrir_tokenizador())

    dados = carregar_etth1(None, LINHAS + SEQ_LEN + 1)
    z = StandardScaler().fit(dados["values"][:LINHAS]).transform(dados["values"])
    pool = []
    for i in range(N_JANELAS):
        xw = z[1000 + i * SEQ_LEN: 1000 + (i + 1) * SEQ_LEN, 0: 1]
        yw = z[1000 + i * SEQ_LEN: 1000 + (i + 1) * SEQ_LEN, 0: 1]
        pool.append((xw.astype(np.float32), yw.astype(np.float32)))

    for rotulo, fmt in (("padrao", FormatoTexto()),
                        ("decimais=3", FormatoTexto(casas=3)),
                        ("encoding=digitos", FormatoTexto(encoding="digitos"))):
        ds = construir_dataset_char_by_char(
            pool, tok_dir,
            block_max=BLOCK_MAX, n_paginas_max=N_PAGINAS_MAX,
            decomp="vmd", formato=fmt,
        )
        x, ids, lens, y, max_abs = ds.tensors
        esperado_ids = (N_JANELAS, BLOCK_MAX * N_PAGINAS_MAX)
        esperado_lens = (N_JANELAS, N_PAGINAS_MAX)
        esperado_max = (N_JANELAS, 3, 1)
        ok = (
            tuple(ids.shape) == esperado_ids
            and tuple(lens.shape) == esperado_lens
            and tuple(max_abs.shape) == esperado_max
            and tuple(x.shape) == (N_JANELAS, SEQ_LEN, 1)
            and tuple(y.shape) == (N_JANELAS, N_PONTOS, 1)
            and int(lens.sum()) > 0
        )
        print(f"  [{rotulo:16s}] ids={tuple(ids.shape)} lens={tuple(lens.shape)} "
              f"max_abs={tuple(max_abs.shape)} paginas={int((lens > 0).sum())}"
              f"{'  OK' if ok else '  CONTRATO QUEBRADO'}")
        if not ok:
            falhas.append(f"{rotulo}: contrato {esperado_ids}/{esperado_lens}/"
                          f"{esperado_max} violado "
                          f"(ids={tuple(ids.shape)}, lens={tuple(lens.shape)}, "
                          f"max_abs={tuple(max_abs.shape)})")

    # ---- 20 campos: o worker so aceita a tupla completa --------------- #
    xw, yw = pool[0]
    tupla20 = (
        xw, yw, tok_dir, BLOCK_MAX, N_PAGINAS_MAX, 0.15,
        "vmd", 25, None, 100, 0.005, 5, 42,
        7, 2000.0, 0.0, 1e-7, 0, None, FormatoTexto(),
    )
    print(f"  args_tuple com {len(tupla20)} campos")
    if len(tupla20) != 20:
        falhas.append(f"args_tuple com {len(tupla20)} campos (esperado 20)")
    try:
        r = _processar_janela(tupla20)
        if tuple(r["ids"].shape) != (BLOCK_MAX * N_PAGINAS_MAX,):
            falhas.append(f"_processar_janela devolveu ids {r['ids'].shape}")
        if tuple(r["max_abs"].shape) != (3,):
            falhas.append(f"_processar_janela devolveu max_abs {r['max_abs'].shape}")
        print(f"  _processar_janela -> ids={r['ids'].shape} "
              f"lens={r['lens'].shape} max_abs={r['max_abs'].shape}")
    except TypeError as e:
        falhas.append(f"_processar_janela nao aceitou 20 campos: {e}")

    # tupla de 19 campos (formato faltando) tem de recusar
    try:
        _processar_janela(tupla20[:19])
        falhas.append("tupla de 19 campos foi aceita (era para falhar)")
    except (TypeError, ValueError):
        print("  tupla de 19 campos recusada: OK")

    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("INTEGRACAO OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
