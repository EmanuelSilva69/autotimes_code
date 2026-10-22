"""Smoke do --so-modelo: a flag existe e o RELATORIO tolera o modelo ausente.

`--so-modelo pj|cbc` roda so um dos modelos textuais (agiliza os experimentos
A/B de formato de texto: a variante so afeta um dos dois documentos). O modelo
nao rodado fica FORA do JSON e do veredito - entao `imprimir_tabela` e
`gerar_graficos` precisam tolerar a ausencia da chave sem quebrar o fluxo.

Cobre: (a) argparse com default 'ambos' e aceitacao de pj/cbc; (b) a tabela
de veredito com um dos modelos ausente (linha "(nao rodado)"); (c) os dois
graficos gerados com as series presentes. Nao roda treino.

Uso: python -m testes_letras.smoke_so_modelo
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _resumo_sem(*faltando: str) -> dict:
    """resumo minimo no formato do main(), sem os modelos citados."""
    pj = {
        "test_mse_media": 0.3592, "test_mse_desvio": 0.0001,
        "test_mae_media": 0.3953, "test_mae_desvio": 0.0001,
        "melhor_val_mse_media": 0.7321, "melhor_val_mse_desvio": 0.0,
        "melhor_epoca_media": 10.0, "estouros_total": 0,
        "val_mse_media": [0.8, 0.6, 0.5], "val_mse_desvio": [0.0, 0.0, 0.0],
        "test_mse_media_por_epoca": [0.5, 0.4, 0.36],
        "test_mse_desvio_por_epoca": [0.0, 0.0, 0.0],
    }
    cb = dict(pj, test_mse_media=0.3571, test_mae_media=0.3971)
    modelos = {"numerico": {"test_mse_media": 0.3539, "test_mae_media": 0.3919}}
    if "pseudo_json_es" not in faltando:
        modelos["pseudo_json_es"] = pj
    if "char_by_char" not in faltando:
        modelos["char_by_char"] = cb
    return {"modelos": modelos, "config": {"so_modelo": None}}


def main() -> int:
    from .run_apples_to_apples_trio import gerar_graficos, imprimir_tabela, parse_args
    ok = True

    # ---- (a) argparse -------------------------------------------------- #
    argv_original = sys.argv
    casos = ([], ["--so-modelo", "pj"], ["--so-modelo", "cbc"],
             ["--so-modelo", "ambos"])
    esperados = ["ambos", "pj", "cbc", "ambos"]
    try:
        for argv, esperado in zip(casos, esperados):
            sys.argv = ["run_apples_to_apples_trio"] + argv
            args = parse_args()
            bate = args.so_modelo == esperado
            print(f"[argparse] --so-modelo {' '.join(argv[1:]) or '(default)'} "
                  f"-> {args.so_modelo} {'OK' if bate else 'FALHOU'}")
            ok = ok and bate
    finally:
        sys.argv = argv_original

    # ---- (b) tabela com um modelo ausente ------------------------------ #
    for faltando in (("pseudo_json_es",), ("char_by_char",),
                     ("pseudo_json_es", "char_by_char")):
        r = _resumo_sem(*faltando)
        try:
            imprimir_tabela(r)
            print(f"[tabela] sem {'+'.join(faltando)} -> OK")
        except Exception as e:  # noqa: BLE001
            print(f"[tabela] sem {'+'.join(faltando)} -> FALHOU: {e!r}")
            ok = False

    # ---- (c) graficos com um modelo ausente ---------------------------- #
    with tempfile.TemporaryDirectory() as td:
        for faltando in (("pseudo_json_es",), ("char_by_char",)):
            r = _resumo_sem(*faltando)
            r["config"]["so_modelo"] = ("pj" if faltando == ("char_by_char",)
                                        else "cbc")
            try:
                paths = gerar_graficos(r, sufixo_grafico="_smoke_so_modelo",
                                       titulo_extra=" [smoke]", out_dir=Path(td))
                existe = all(p.exists() for p in paths) and len(paths) == 2
                print(f"[graficos] sem {'+'.join(faltando)} -> "
                      f"{len(paths)} PNGs {'OK' if existe else 'FALHOU'}")
                ok = ok and existe
            except Exception as e:  # noqa: BLE001
                print(f"[graficos] sem {'+'.join(faltando)} -> FALHOU: {e!r}")
                ok = False

    if not ok:
        print("FALHOU: --so-modelo quebra argparse, tabela ou graficos")
        return 1
    print("OK (--so-modelo aceito; relatorio e graficos toleram modelo ausente)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
