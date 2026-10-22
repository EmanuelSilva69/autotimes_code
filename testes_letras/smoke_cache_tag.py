"""Confere que a TAG DE CACHE do padrao continua a mesma (caches historicos).

A tag so ganha sufixo quando uma variante de texto e selecionada; sem flag,
o caminho calculado tem de bater com um .npz ja existente em
`quinta_noite/cache_trio/`. Se a tag mudasse por engano, o runner regeraria
700 MB de documento e os vereditos anteriores ficariam incomparaveis.

As tags sao POR EIXO (PJ so depende de `data`; Char-by-Char so depende de
`casas`/`encoding`): uma variante de data nao pode forcar a regeneracao do
cache de 650 MB do CbC, nem vice-versa. Este smoke cobre tambem esse
desacoplamento.

Uso: python -m testes_letras.smoke_cache_tag
"""

from __future__ import annotations

import sys


def main() -> int:
    # Parser do runner sem argv propria (defaults puros).
    argv_original = sys.argv
    sys.argv = ["run_apples_to_apples_trio"]
    try:
        from .run_apples_to_apples_trio import (
            CACHE_DIR,
            N_FEATS,
            _caminho_cache_char_by_char,
            _caminho_cache_pseudo_json,
            parse_args,
        )
        args = parse_args()
    finally:
        sys.argv = argv_original

    # Mesmo pre-processingamento do main().
    args.feats_list = (
        list(range(N_FEATS)) if args.feats == "all"
        else [int(f.strip()) for f in args.feats.split(",") if f.strip()]
    )
    args.texto_decomp = "vmd"  # rodada de referencia atual

    pj = _caminho_cache_pseudo_json(args, None)
    cb = _caminho_cache_char_by_char(args, None)
    ok = True

    for nome, p in (("pseudo-json", pj), ("char-by-char", cb)):
        existe = p.exists()
        print(f"[{nome}] {p.name} -> {'EXISTE (reusado)' if existe else 'AUSENTE'}")
        ok = ok and existe

    # ---- o eixo certo muda a tag CERTA (e so ela) --------------------- #
    variacoes = (
        # (rotulo, eixo, valor, caminho_que_muda, caminho_que_NAO_muda)
        ("data=nenhuma", "texto_data", "nenhuma", "pj", "cb"),
        ("decimais=3", "texto_decimais", 3, "cb", "pj"),
        ("encoding=digitos", "texto_encoding", "digitos", "cb", "pj"),
        ("pj_casas=1", "texto_pj_casas", 1, "pj", "cb"),
        ("pj_marcacao=compacta", "texto_pj_marcacao", "compacta", "pj", "cb"),
        ("pj_escala=inteiro", "texto_pj_escala", "inteiro", "pj", "cb"),
        ("arredondamento=arredondar", "texto_arredondamento", "arredondar",
         "cb", "pj"),
        ("zeros=remover", "texto_zeros", "remover", "cb", "pj"),
    )
    funcs = {"pj": _caminho_cache_pseudo_json, "cb": _caminho_cache_char_by_char}
    bases = {"pj": pj, "cb": cb}
    for rot, attr, valor, alvo, outro in variacoes:
        setattr(args, attr, valor)
        p_alvo = funcs[alvo](args, None)
        p_outro = funcs[outro](args, None)
        delattr(args, attr)
        muda = p_alvo != bases[alvo]
        preserva = p_outro == bases[outro]
        print(f"[{rot}] {alvo} -> "
              f"{'tag PROPRIA' if muda else 'COLIDIU COM O PADRAO'} | "
              f"{outro} permanece no cache do padrao: "
              f"{'SIM' if preserva else 'NAO (regeneracao inutil!)'}")
        ok = ok and muda and preserva

    # ---- combinacao: cada cache mantem SO seu eixo --------------------- #
    args.texto_data = "hora"
    args.texto_decimais = 3
    pj_h3 = _caminho_cache_pseudo_json(args, None)
    cb_h3 = _caminho_cache_char_by_char(args, None)
    args.texto_data = "nenhuma"
    pj_n3 = _caminho_cache_pseudo_json(args, None)
    cb_n3 = _caminho_cache_char_by_char(args, None)
    args.texto_data = "hora"
    args.texto_decimais = 6  # volta ao padrao de casas, mantem data=hora
    pj_h6 = _caminho_cache_pseudo_json(args, None)
    cb_h6 = _caminho_cache_char_by_char(args, None)
    delattr(args, "texto_data")
    delattr(args, "texto_decimais")

    casos = (
        ("PJ hora/3 != PJ nenhuma/3 (politicas de data = caches PJ distintos)",
         pj_h3 != pj_n3),
        ("CbC hora/3 == CbC nenhuma/3 (data nao entra no CbC)",
         cb_h3 == cb_n3),
        ("CbC hora/3 != CbC hora/6 (casas decide o CbC)",
         cb_h3 != cb_h6),
        ("PJ hora/6 == PJ hora/3 (casas nao entra no PJ)",
         pj_h6 == pj_h3),
        ("PJ hora/6 != padrao (tag propria no eixo data)",
         pj_h6 != pj),
        ("CbC hora/6 == padrao (data nao contamina o CbC)",
         cb_h6 == cb),
    )
    for rot, cond in casos:
        print(f"[combinado] {rot} -> {'OK' if cond else 'FALHOU'}")
        ok = ok and cond

    if not ok:
        print("FALHOU: a tag de cache do padrao nao bate com os caches "
              "existentes (ou os eixos nao estao desacoplados)")
        return 1

    print("OK (tag do padrao inalterada; eixos desacoplados por documento)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
