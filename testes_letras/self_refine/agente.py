"""Agente do Self-Refine: renderiza o PROMPT DE REFINAMENTO (100% local).

Sem API externa (decisao do usuario + design D9): `montar_prompt` produz o
texto com as 6 tags do protocolo de refinamento e o CONSUMIDOR e o proprio
loop local - o campo `<self_refinement_and_feedback>` vira o feedback
injetado na nova tentativa e gravado na memoria; `<final_result>` e o
candidato validado.

A funcao fica como COSTURA para um backend de LLM no futuro: trocar o
consumidor nao muda o formato do prompt. Nenhuma chamada de rede acontece
aqui (nem agora, nem implicitamente depois).

Uso das tags (obrigatorias, na ordem):
    <context_analysis>              dados + regras + memoria episodica
    <initial_draft>                 o que foi gerado nesta tentativa
    <evaluation>                    resultado da validacao
    <self_reflection_and_feedback>  regra de correcao a injetar
    <refined_output>                o que a nova tentativa tem de produzir
    <final_result>                  candidato aceito (ou vazio se abortou)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .validadores import Falha

TAGS = (
    "<context_analysis>",
    "</context_analysis>",
    "<initial_draft>",
    "</initial_draft>",
    "<evaluation>",
    "</evaluation>",
    "<self_reflection_and_feedback>",
    "</self_reflection_and_feedback>",
    "<refined_output>",
    "</refined_output>",
    "<final_result>",
    "</final_result>",
)

# Campos de contexto preenchiveis exigidos pelo template.
CAMPOS_CONTEXTO = ("dados", "regras", "memoria_episodica")


def _bloco(tag: str, corpo: str) -> str:
    return f"{tag}{corpo.rstrip()}\n{tag.replace('<', '</', 1)}"


def montar_prompt(contexto: Dict[str, Any], tentativa: int,
                  rascunho: Any = None,
                  falhas: Optional[Sequence[Falha]] = None,
                  licoes: Optional[Sequence[Any]] = None,
                  final: Any = None) -> str:
    """Renderiza o prompt de refinamento com as 6 tags e os 3 campos.

    Args:
        contexto: campos de contexto preenchiveis (precisa cobrir
            `dados`, `regras` e `memoria_episodica`; o que faltar e escrito
            como "nao informado" - o template nunca sai incompleto).
        tentativa: numero da tentativa (1 = rascunho inicial).
        rascunho: o documento/texto gerado nesta tentativa.
        falhas: violacoes encontradas (None = nada validado ainda).
        licoes: entradas da memoria episodica ja conhecidas.
        final: candidato validado (preenche <final_result>).

    Returns:
        Prompt completo em portugues, com as tags na ordem do protocolo.
    """
    ctx = {campo: contexto.get(campo, "não informado") for campo
           in CAMPOS_CONTEXTO}

    corpo_ctx = (
        f"  dados: {ctx['dados']}\n"
        f"  regras: {ctx['regras']}\n"
        f"  memoria_episodica: {ctx['memoria_episodica']}\n"
        f"  tentativa: {int(tentativa)}"
    )
    corpo_draft = f"  {repr(rascunho)[:1200]}" if rascunho is not None else \
        "  (sem rascunho nesta tentativa)"

    if falhas:
        corpo_eval = "\n".join(
            f"  - {f.codigo}: {f.detalhe}" for f in falhas
        )
        corpo_fb = "\n".join(
            f"  - {f.regra} (motivo: {f.detalhe})" for f in falhas
        )
        corpo_refined = (
            "  Produzir NOVO documento corrigindo as violacoes acima, "
        ) + "; ".join(f.regra for f in falhas) + "."
    else:
        corpo_eval = "  (validacao ainda nao executada)"
        corpo_fb = "  (sem feedback: nenhuma violacao ate agora)"
        corpo_refined = "  (nada a corrigir)"

    if licoes:
        corpo_ctx += "\n  licoes conhecidas:\n" + "\n".join(
            f"    - {getattr(l, 'codigo', '?')}: "
            f"{getattr(l, 'motivo', '?')} -> {getattr(l, 'regra', '?')}"
            for l in licoes
        )

    corpo_final = (f"  {repr(final)[:1200]}" if final is not None
                   else "  (nenhum candidato aceito; rodada abortada)")

    return "\n".join([
        _bloco("<context_analysis>", corpo_ctx),
        _bloco("<initial_draft>", corpo_draft),
        _bloco("<evaluation>", corpo_eval),
        _bloco("<self_reflection_and_feedback>", corpo_fb),
        _bloco("<refined_output>", corpo_refined),
        _bloco("<final_result>", corpo_final),
    ])


def conferir_prompt(prompt: str) -> List[str]:
    """Problemas estruturais do prompt (usado pelo smoke, nao roda no runner)."""
    faltas = [t for t in TAGS if t not in prompt]
    campos = [c for c in CAMPOS_CONTEXTO if f"{c}:" not in prompt]
    return [f"tag ausente: {t}" for t in faltas] + \
           [f"campo ausente: {c}" for c in campos]
