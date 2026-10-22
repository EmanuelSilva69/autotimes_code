"""Smoke do pacote Self-Refine (validadores, memoria, agente, loop).

Cenarios cobertos (spec self-refine-textual-generation):
  1. validadores : 1 documento VALIDO + 1 falha por verificador (5 no total)
  2. memoria     : grava -> relê -> apaga/corrompe -> relê com aviso, sem excecao
  3. agente      : prompt renderizado tem as 6 tags e os 3 campos de contexto
  4. loop        : falha deterministica -> retry corrige -> memoria registra
                   {motivo, regra, contexto}; com limite=1 aborta com a
                   ULTIMA falha na mensagem e sem gravar "cache"
  5. fail-fast   : falha com os MESMOS codigos da tentativa anterior aborta
                   ja na 2a de 3 geracoes (geracao deterministica: regerar
                   de novo so custa tempo)

Uso: python -m testes_letras.smoke_self_refine
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

L = 672


def _janela_valida():
    """Janela real + decomposicao coerente (T + S + R == x por construcao)."""
    from .texto_char_by_char import decompor_vmd, frases_texto_vmd

    rng = np.random.default_rng(2021)
    x = (np.linspace(-1.0, 1.0, L) + 0.05 * rng.standard_normal(L))
    t, s, r = decompor_vmd(x, k=7, alpha=2000.0, tau=0.0, tol=1e-7,
                           init=0, trend_ma=None)
    f = frases_texto_vmd(x, k=7, alpha=2000.0, tau=0.0, tol=1e-7,
                         init=0, trend_ma=None)
    frases = f[:3]  # (frase_t, frase_s, frase_r); o 4o campo e o max_abs
    return x, t, s, r, frases


def cenario_validadores(falhas: list) -> None:
    from .self_refine.validadores import (
        validar_finitude,
        validar_frases,
        validar_orcamento,
        validar_reconstrucao,
        validar_tensores,
    )

    x, t, s, r, frases = _janela_valida()
    p, b = 7, 128  # n_paginas_max x block_max (orcamento 896)
    ids = np.zeros(b * p, dtype=np.int64)
    lens = np.full(p, 100, dtype=np.float64)
    mx = np.zeros((3, 1), dtype=np.float64)

    print("  [1] validadores")
    # ---- documento valido: os 5 passam -------------------------------- #
    ok = [
        validar_frases(frases),
        validar_finitude(x),
        validar_reconstrucao(x, t, s, r),
        validar_tensores(ids, lens, mx, b, p),
        validar_orcamento(lens, b, p),
    ]
    if any(ok):
        falhas.append(f"documento valido foi recusado: {[str(f) for f in ok if f]}")
    else:
        print("        doc valido aceito pelos 5 verificadores")

    # ---- 1 falha por verificador -------------------------------------- #
    casos = (
        ("FRASE_AUSENTE", lambda: validar_frases(
            [frases[0], frases[1]])),                       # 2 frases
        ("VALOR_NAO_FINITO", lambda: validar_finitude(
            np.array([1.0, np.nan, np.inf]))),
        ("RECONSTRUCAO_FORA", lambda: validar_reconstrucao(
            x, t + 1.0, s, r)),                             # soma errada
        ("TENSOR_CONTRATO", lambda: validar_tensores(
            ids[:10], lens, mx, b, p)),                     # ids truncados
        ("ORCAMENTO_ESTOURADO", lambda: validar_orcamento(
            np.full(p, b + 1), b, p)),                      # 129 > 128
    )
    for codigo, chamar in casos:
        f = chamar()
        if f is None:
            falhas.append(f"{codigo}: o verificador NAO detectou a falha injetada")
        elif f.codigo != codigo:
            falhas.append(f"{codigo}: detectado como {f.codigo} ({f.detalhe})")
        elif not f.regra:
            falhas.append(f"{codigo}: sem regra de correcao associada")
        else:
            print(f"        {codigo} detectado -> regra: {f.regra[:58]}...")


def cenario_memoria(falhas: list) -> None:
    from .self_refine.memoria import Memoria

    print("  [2] memoria episodica")
    raiz = Path(tempfile.mkdtemp(prefix="sr_mem_"))
    try:
        caminho = raiz / "autotimes_memory.json"

        # ausente -> vazia + aviso, sem excecao
        m = Memoria(caminho)
        if m.carregar() != []:
            falhas.append("memoria ausente nao veio vazia")
        else:
            print("        ausente -> vazia (sem excecao)")

        # grava e rele
        m.registrar("RECONSTRUCAO_FORA", "erro 1e-3 > 1e-6",
                    "refaz a decomposicao", {"decomp": "vmd", "feat": 0},
                    registro="<context_analysis>x</context_analysis>")
        m2 = Memoria(caminho)
        licoes = m2.carregar()
        if len(licoes) != 1:
            falhas.append(f"rele 1 entrada, achei {len(licoes)}")
        else:
            e = licoes[0]
            campos = {"motivo": e.motivo, "regra": e.regra,
                      "contexto": e.contexto}
            print(f"        gravou e relei: {json.dumps(campos, ensure_ascii=False)[:78]}")
            if campos["motivo"] != "erro 1e-3 > 1e-6" \
                    or campos["regra"] != "refaz a decomposicao" \
                    or campos["contexto"] != {"decomp": "vmd", "feat": 0}:
                falhas.append(f"campos da licao errados: {campos}")

        # corrompido -> vazia + aviso
        caminho.write_text("{nao e json", encoding="utf-8")
        licoes = Memoria(caminho).carregar()
        if licoes != []:
            falhas.append("arquivo corrompido nao resultou em memoria vazia")
        else:
            print("        corrompido -> vazia (aviso, sem excecao)")

        # escrito pela metade (interrupcao) nao corrompe o que ja la estava
        m3 = Memoria(caminho)
        m3.registrar("VALOR_NAO_FINITO", "NaN", "sanea")
        tmp = caminho.with_suffix(caminho.suffix + ".tmp")
        tmp.write_text("{", encoding="utf-8")   # tmp sujo, nunca publicado
        reaberto = Memoria(caminho).carregar()
        if len(reaberto) != 1:
            falhas.append(f"escrita atomica quebrou o arquivo anterior: "
                          f"{len(reaberto)} entradas")
        else:
            print("        tmp sujo ignorado; arquivo anterior legivel")

        # teto FIFO
        m4 = Memoria(caminho, max_entradas=3)
        for i in range(10):
            m4.registrar("X", f"motivo {i}", "regra")
        n = len(Memoria(caminho, max_entradas=3).carregar())
        if n != 3:
            falhas.append(f"FIFO nao respeitou o teto: {n} entradas")
        else:
            print("        FIFO com teto: 3 entradas (de 10 gravadas)")
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


def cenario_agente(falhas: list) -> None:
    from .self_refine.agente import TAGS, conferir_prompt, montar_prompt
    from .self_refine.validadores import Falha

    print("  [3] agente (prompt de refinamento)")
    falha = Falha("RECONSTRUCAO_FORA", "erro relativo 1e-3 > 1e-6")
    prompt = montar_prompt(
        contexto={"dados": "ETTh1 / HUFL / janela 672 / vmd k=7",
                  "regras": "T+S+R == x (tol 1e-6)",
                  "memoria_episodica": "- RECONSTRUCAO_FORA: ..."},
        tentativa=1, rascunho="Tendencia: ... sep fim", falhas=[falha],
    )
    print(prompt[:200].replace("\n", " ") + " ...")
    problema = conferir_prompt(prompt)
    if problema:
        falhas.append(f"prompt estruturalmente incompleto: {problema}")
    else:
        faltas = [t for t in TAGS if t not in prompt]
        campos = [c for c in ("dados:", "regras:", "memoria_episodica:")
                  if c not in prompt]
        if faltas or campos:
            falhas.append(f"prompt sem tags {faltas} ou campos {campos}")
        else:
            print(f"        {len(TAGS)} tags + 3 campos de contexto presentes")


def cenario_loop(falhas: list) -> None:
    from .self_refine.loop import ErroSelfRefine, rodar_self_refine
    from .self_refine.memoria import Memoria
    from .self_refine.validadores import Falha

    print("  [4] loop (retry + memoria)")
    raiz = Path(tempfile.mkdtemp(prefix="sr_loop_"))
    try:
        caminho = raiz / "autotimes_memory.json"
        ctx = {"decomp": "vmd", "variante": "padrao", "split": "train"}

        def gerar(feedback):
            # falha deterministica na 1a tentativa; o feedback corrige
            return {"doc": "corrigido" if feedback else "com falha"}

        def validar(res):
            if res["doc"] == "com falha":
                return [Falha("RECONSTRUCAO_FORA", "erro 1e-3 > 1e-6")]
            return []

        r = rodar_self_refine(gerar, validar, ctx, rotulo="doc-teste",
                              max_tentativas=3, caminho_memoria=caminho)
        if r.tentativas != 2:
            falhas.append(f"esperado 2 tentativas, foram {r.tentativas}")
        if r.resultado["doc"] != "corrigido":
            falhas.append(f"retry nao corrigiu: {r.resultado}")
        else:
            print(f"        falha na tentativa 1 -> corrigido na 2 "
                  f"({r.tentativas} tentativas)")

        entradas = Memoria(caminho).carregar()
        if not entradas:
            falhas.append("memoria ficou vazia apos falha corrigida")
        else:
            e = entradas[0]
            if not (e.motivo and e.regra and e.contexto == ctx):
                falhas.append(f"licao incompleta: motivo={e.motivo!r} "
                              f"regra={e.regra!r} contexto={e.contexto}")
            else:
                print(f"        memoria: motivo={e.motivo!r} regra={e.regra[:44]!r} "
                      f"contexto={e.contexto}")

        # limite = 1 -> aborta com a ULTIMA falha na mensagem
        caminho2 = raiz / "outra.json"
        try:
            rodar_self_refine(lambda fb: {"doc": "sempre falho"},
                              lambda res: [Falha("TENSOR_CONTRATO",
                                                 "ids (10,) != (896,)")],
                              ctx, rotulo="doc-quebrado", max_tentativas=1,
                              caminho_memoria=caminho2)
            falhas.append("limite=1 nao abortou (devia levantar ErroSelfRefine)")
        except ErroSelfRefine as e:
            msg = str(e)
            if "TENSOR_CONTRATO" not in msg or "cache NAO gravado" not in msg:
                falhas.append(f"mensagem de aborto incompleta: {msg}")
            else:
                print(f"        limite=1 -> aborto explicito: {msg[:88]}...")
        if caminho2.exists() and not Memoria(caminho2).carregar():
            falhas.append("memoria deveria ter a licao do caminho de falha")

        # ---- 5. fail-fast: MESMOS codigos -> aborta antes da 3a -------- #
        chamadas = {"n": 0}

        def gerar_fixo(fb):
            chamadas["n"] += 1
            return {"doc": "sempre falho"}

        def validar_fixo(res):
            return [Falha("ORCAMENTO_ESTOURADO", "129 > 128 (b*p=896)")]

        try:
            rodar_self_refine(gerar_fixo, validar_fixo, ctx,
                              rotulo="doc-fixo", max_tentativas=3,
                              caminho_memoria=raiz / "failfast.json")
            falhas.append("falha repetida nao abortou (devia ErroSelfRefine)")
        except ErroSelfRefine as e:
            if chamadas["n"] != 2:
                falhas.append(f"fail-fast devia parar em 2 geracoes "
                              f"(de 3), fez {chamadas['n']}")
            elif "esgotou 2 tentativa" not in str(e):
                falhas.append(f"mensagem do fail-fast inesperada: {e}")
            else:
                print(f"        falha repetida na 2a -> aborto antecipado em "
                      f"{chamadas['n']} de 3 geracoes")
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


def main() -> int:
    from .texto_char_by_char import decompor_vmd  # noqa: F401  (torch primeiro)

    falhas: list = []
    print("=" * 96)
    print("SELF-REFINE | smoke do pacote (validadores / memoria / agente / loop)")
    print("=" * 96)
    cenario_validadores(falhas)
    cenario_memoria(falhas)
    cenario_agente(falhas)
    cenario_loop(falhas)

    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("OK (self-refine local: 4 modulos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
