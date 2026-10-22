"""Smoke da INTEGRACAO da flag --self-refine no runner (task 7.2).

Cobre o que a spec exige no caminho real (Pseudo-JSON), sem treinar nada:

  A. falha injetada -> documento RECUSADO pela validacao, feedback com a
     regra injetado na tentativa seguinte, 2a tentativa aceita, e a licao
     {motivo, regra, contexto} gravada na memoria;
  B. falha persistente -> tentativas esgotadas -> SystemExit explicito com
     a ULTIMA falha na mensagem e o .npz de cache NAO gravado;
  C. flag DESLIGADA -> o gerador e chamado UMA vez, sem validacao, sem
     memoria e sem nenhuma linha de Self-Refine no log.

Uso: python -m testes_letras.smoke_self_refine_runner
"""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

N_ROWS = 2000


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
    args.force_cache = True          # nunca reusa cache neste smoke
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _z_sintetico() -> np.ndarray:
    """Série suave e determinística (VMD converge; reconstrução exata)."""
    rng = np.random.default_rng(2021)
    base = np.sin(np.linspace(0, 40 * np.pi, N_ROWS))
    return (base + 0.01 * rng.standard_normal(N_ROWS)).reshape(-1, 1).repeat(7, 1)


def _doc(bom: bool, block_max: int, n_paginas_max: int) -> dict:
    """Documento PJ fake: 'bom' = contrato ok; 'ruim' = lens 999 > block_max."""
    n_feat = 7
    lens = np.full((n_feat, n_paginas_max), 100, dtype=np.int64)
    if not bom:
        lens[0, 0] = 999  # estoura o orcamento: valida_tensores/orcamento pega
    return {
        "ids_per_feat": np.zeros((n_feat, block_max * n_paginas_max),
                                 dtype=np.int64),
        "lens_per_feat": lens,
        "tokens_originais": 1234, "tokens_retidos": 896,
        "n_paginas": n_paginas_max, "paginas_descartadas": 0,
    }


def main() -> int:
    import testes_letras.run_apples_to_apples_trio as mod
    from .run_apples_to_apples_trio import (
        _caminho_cache_pseudo_json,
        _self_refine_pseudo_json,
        carregar_pseudo_json,
    )

    falhas = []
    raiz = Path(tempfile.mkdtemp(prefix="sr_int_"))
    z = _z_sintetico()

    print("=" * 96)
    print("SELF-REFINE | integracao no runner (Pseudo-JSON)")
    print("=" * 96)

    # ------------------------------------------------------------------ #
    print("  [A] falha injetada -> recusa -> retry -> licao gravada")
    # ------------------------------------------------------------------ #
    args = _args(self_refine=True, self_refine_max_tentativas=3,
                 self_refine_memoria=str(raiz / "memoria.json"))
    original = mod._gerar_pseudo_json
    chamadas = {"n": 0}

    def gerador_com_falha(args_, tok_, dados_, z_, periodos_mstl=None):
        chamadas["n"] += 1
        return _doc(bom=(chamadas["n"] > 1),          # 1a ruim, depois boa
                    block_max=args_.block_max,
                    n_paginas_max=args_.n_paginas_max)

    saida = io.StringIO()
    try:
        mod._gerar_pseudo_json = gerador_com_falha
        with redirect_stdout(saida):
            r = _self_refine_pseudo_json(args, None, None, z)
    finally:
        mod._gerar_pseudo_json = original
    print(saida.getvalue().rstrip())

    if chamadas["n"] != 2:
        falhas.append(f"esperado 2 tentativas (1 falha + 1 aceita), foram "
                      f"{chamadas['n']}")
    if int(r["lens_per_feat"][0, 0]) == 999:
        falhas.append("o documento RECUSADO foi entregue ao chamador")
    if "tentativa seguinte com regra" not in saida.getvalue():
        falhas.append("o feedback da falha nao foi injetado na tentativa "
                      "seguinte")
    mem = raiz / "memoria.json"
    if not mem.exists():
        falhas.append("memoria nao foi gravada no caminho de falha")
    else:
        import json
        licoes = json.loads(mem.read_text(encoding="utf-8"))["entradas"]
        e = licoes[0]
        if not (e["motivo"] and e["regra"] and e["contexto"]):
            falhas.append(f"licao incompleta: {e}")
        else:
            print(f"        memoria: {e['codigo']} | motivo={e['motivo'][:48]!r} "
                  f"| regra={e['regra'][:40]!r}")
            print(f"                  contexto={e['contexto']}")

    # ------------------------------------------------------------------ #
    print("  [B] falha persistente -> aborta e NAO grava cache")
    # ------------------------------------------------------------------ #
    # Variante de texto nao-padrao -> tag PROPRIA de cache: nunca toca nos
    # caches historicos do usuario.
    args_b = _args(self_refine=True, self_refine_max_tentativas=2,
                   texto_data="hora",
                   self_refine_memoria=str(raiz / "memoria_b.json"))
    args_b.feats_list = list(range(7))
    cache_b = _caminho_cache_pseudo_json(args_b, None)
    if cache_b.exists():
        falhas.append(f"o smoke ia sobrescrever um cache existente: {cache_b}")
    chamadas["n"] = 0

    def gerador_sempre_ruim(args_, tok_, dados_, z_, periodos_mstl=None):
        chamadas["n"] += 1
        return _doc(bom=False, block_max=args_.block_max,
                    n_paginas_max=args_.n_paginas_max)

    saida_b = io.StringIO()
    motivo = None
    try:
        mod._gerar_pseudo_json = gerador_sempre_ruim
        with redirect_stdout(saida_b):
            try:
                carregar_pseudo_json(args_b, None, None, z, None)
            except SystemExit as e:
                motivo = str(e)
    finally:
        mod._gerar_pseudo_json = original
    print(saida_b.getvalue().rstrip() or "  (sem stdout)")
    if motivo is None:
        falhas.append("falha persistente nao abortou (devia ser SystemExit)")
    elif "cache NAO gravado" not in motivo or "Ultima falha" not in motivo:
        falhas.append(f"mensagem de aborto incompleta: {motivo[:120]}")
    else:
        print(f"        aborto: {motivo[:100]}...")
    if cache_b.exists():
        falhas.append(f"cache foi gravado apesar do aborto: {cache_b}")
    else:
        print(f"        cache nao gravado: {cache_b.name}")

    # ------------------------------------------------------------------ #
    print("  [C] flag desligada -> comportamento atual intacto")
    # ------------------------------------------------------------------ #
    # Variante nao-padrao de novo: tag propria, sem tocar no cache historico.
    args_c = _args(self_refine=False, texto_data="curta")
    args_c.feats_list = list(range(7))
    cache_c = _caminho_cache_pseudo_json(args_c, None)
    if cache_c.exists():
        falhas.append(f"o smoke ia sobrescrever um cache existente: {cache_c}")
    chamadas["n"] = 0

    def gerador_contador(args_, tok_, dados_, z_, periodos_mstl=None):
        chamadas["n"] += 1
        return _doc(bom=False, block_max=args_.block_max,   # ate ruim: ninguem valida
                    n_paginas_max=args_.n_paginas_max)

    saida_c = io.StringIO()
    try:
        mod._gerar_pseudo_json = gerador_contador
        with redirect_stdout(saida_c):
            carregar_pseudo_json(args_c, None, None, z, None)
    finally:
        mod._gerar_pseudo_json = original
    texto_c = saida_c.getvalue()
    print(texto_c.rstrip())
    if chamadas["n"] != 1:
        falhas.append(f"sem a flag gerador foi chamado {chamadas['n']}x "
                      f"(devia ser 1x, sem retry)")
    if "self-refine" in texto_c.lower():
        falhas.append("sem a flag o log MENCIONA Self-Refine")
    if cache_c.exists():
        cache_c.unlink()   # cache fake do smoke: fora do repo, apagado na hora

    for lixo in raiz.glob("*"):
        try:
            lixo.unlink()
        except OSError:
            pass
    raiz.rmdir()

    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("OK (integracao --self-refine no runner)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
