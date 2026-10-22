"""Loop do Self-Refine: tentativa -> validacao -> feedback -> retry.

100% local (sem API): quem "refina" e o PROPRIO pipeline, corrigido pela
regra associada ao codigo de falha (design D12). O prompt com as 6 tags
(`agente.montar_prompt`) e o registro de cada refinamento e alimenta a
memoria episodica; o consumidor de hoje e este loop, o de amanha pode ser
um LLM (a costura ja existe, nenhuma chamada de rede e feita aqui).

Garantias:
    - `max_tentativas` limita o ciclo; esgotou -> `ErroSelfRefine` com a
      ULTIMA falha na mensagem (aborto explicito, nunca cacheia documento
      invalido);
    - falha com os MESMOS codigos da tentativa anterior aborta ja (com
      geracao deterministica, regenerar de novo so custa tempo);
    - a memoria so e escrita no caminho de FALHA (acerto nao mexe em nada);
    - a memoria e lida no inicio e disponibilizada antes da 1a tentativa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .agente import montar_prompt
from .memoria import Memoria
from .validadores import Falha

MAX_TENTATIVAS_PADRAO: int = 3


class ErroSelfRefine(RuntimeError):
    """Tentativas esgotadas: a ultima falha vem embutida na mensagem."""

    def __init__(self, rotulo: str, falhas: Sequence[Falha],
                 tentativas: int):
        self.rotulo = rotulo
        self.falhas = list(falhas)
        self.tentativas = tentativas
        ultima = self.falhas[-1] if self.falhas else None
        detalhe = str(ultima) if ultima else "(sem detalhe)"
        super().__init__(
            f"Self-Refine esgotou {tentativas} tentativa(s) em {rotulo!r}; "
            f"cache NAO gravado. Ultima falha: {detalhe}"
        )


@dataclass
class ResultadoLoop:
    """O que o loop devolve ao chamador."""

    resultado: Any
    tentativas: int
    falhas: Sequence[Falha] = field(default_factory=tuple)
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    memoria: Optional[Memoria] = None

    @property
    def corrigido(self) -> bool:
        """True quando precisou de mais de 1 tentativa (houve refinamento)."""
        return self.tentativas > 1


def _regra_de(codigo: str) -> str:
    from .validadores import REGRAS
    return REGRAS.get(codigo, "regera o documento")


def corrigir_local(resultado: Any, falhas: Sequence[Falha]):
    """Aplica as correcoes MECANICAS locais (as que nao regeneram nada).

    Duas regras tem correcao fechada em cima do proprio resultado:
        VALOR_NAO_FINITO   -> sanea NaN/Inf por 0.0 e refaz o max_abs
        ORCAMENTO_ESTOURADO -> trunca na cabeca: lens[:n_paginas_max]

    As demais (frase ausente, marcamento, reconstrucao fora) so se corrigem
    regerando a decomposicao - isso o PROXIMO `gerar(feedback)` faz.
    Codigo que nao esta aqui simplesmente nao e corrigido neste passo.
    """
    aplicadas: List[str] = []
    codigos = {f.codigo for f in falhas}

    if "VALOR_NAO_FINITO" in codigos and isinstance(resultado, dict):
        import numpy as np
        novo = dict(resultado)
        for chave, valor in list(novo.items()):
            if isinstance(valor, np.ndarray) and valor.dtype.kind == "f":
                if not bool(np.isfinite(valor).all()):
                    novo[chave] = np.nan_to_num(valor, nan=0.0,
                                                 posinf=0.0, neginf=0.0)
                    aplicadas.append(f"{chave}: NaN/Inf saneado por 0.0")
        resultado = novo

    if "ORCAMENTO_ESTOURADO" in codigos and isinstance(resultado, dict):
        lens = resultado.get("lens")
        n_paginas_max = resultado.get("n_paginas_max")
        import numpy as np
        if isinstance(lens, np.ndarray) and lens.size > (n_paginas_max or 0):
            novo = dict(resultado)
            novo["lens"] = lens[: int(n_paginas_max)]
            resultado = novo
            aplicadas.append(
                f"lens truncado na cabeca ({int(n_paginas_max)} paginas)"
            )

    return resultado, aplicadas


def rodar_self_refine(
    gerar: Callable[[List[Dict[str, Any]]], Any],
    validar: Callable[[Any], List[Falha]],
    contexto: Dict[str, Any],
    rotulo: str = "documento",
    max_tentativas: int = MAX_TENTATIVAS_PADRAO,
    caminho_memoria: Optional[Path] = None,
    corrigir: Optional[Callable] = None,
) -> ResultadoLoop:
    """Roda o ciclo ate o documento passar na validacao (ou abortar).

    Args:
        gerar: `gerar(feedback) -> resultado`; recebe a lista de feedbacks
            (cada item: {codigo, motivo, regra}) da tentativa anterior.
        validar: `validar(resultado) -> List[Falha]` (vazia = valido).
        contexto: campos descritivos (dados, regras, variante...) gravados
            junto de cada licao.
        rotulo: nome curto do documento (aparece no erro e na memoria).
        max_tentativas: teto de ciclos (padrao 3).
        caminho_memoria: arquivo episodico (padrao: nenhum -> so em disco
            se informado; o runner passa `.runtime/autotimes_memory.json`).
        corrigir: `corrigir(resultado, falhas) -> (resultado, aplicadas)`,
            opcional; default = `corrigir_local` (correcoes mecanicas).

    Returns:
        ResultadoLoop com o resultado aceito e as estatisticas do ciclo.

    Raises:
        ErroSelfRefine: todas as tentativas falharam (aborto explicito).
    """
    if corrigir is None:
        corrigir = corrigir_local

    memoria: Optional[Memoria] = None
    licoes: List[Any] = []
    if caminho_memoria is not None:
        memoria = Memoria(Path(caminho_memoria))
        licoes = memoria.consultar(contexto)

    feedback: List[Dict[str, Any]] = []
    ultimas: List[Falha] = []
    codigos_anteriores: Optional[frozenset] = None
    executadas = 0
    max_tentativas = max(1, int(max_tentativas))

    for tentativa in range(1, max_tentativas + 1):
        executadas = tentativa
        resultado = gerar(feedback)
        falhas = list(validar(resultado) or [])

        # ---- correcao mecanica (mesma tentativa, antes de desistir) ---- #
        if falhas and corrigir is not None:
            resultado, aplicadas = corrigir(resultado, falhas)
            if aplicadas:
                falhas = list(validar(resultado) or [])

        if not falhas:
            return ResultadoLoop(
                resultado=resultado, tentativas=tentativa,
                falhas=tuple(ultimas), feedback=feedback, memoria=memoria,
            )

        ultimas = falhas
        codigos = frozenset(f.codigo for f in falhas)

        # ---- aborto antecipado: MESMOS codigos da tentativa anterior ---- #
        # Com geracao deterministica (mesmos parametros), repetir o mesmo
        # codigo significa que regenerar de novo e gastar o tempo da geracao
        # para chegar ao mesmo resultado; a correcao mecanica (que NAO
        # regenera) ja foi tentada acima. Codigos diferentes = a regeneracao
        # mudou algo -> segue o ciclo normal.
        if codigos == codigos_anteriores:
            print(f"[self-refine] falha repetida na tentativa {tentativa} "
                  f"({', '.join(sorted(codigos))}) -- abortando antes de "
                  f"regenerar de novo")
            break
        codigos_anteriores = codigos

        # ---- memoria: so no caminho de falha --------------------------- #
        if memoria is not None:
            prompt = montar_prompt(
                contexto=contexto, tentativa=tentativa, rascunho=resultado,
                falhas=falhas, licoes=licoes, final=None,
            )
            for f in falhas:
                memoria.registrar(
                    codigo=f.codigo, motivo=f.detalhe, regra=f.regra,
                    contexto=contexto, registro=prompt,
                )
        feedback = [f.como_feedback() for f in falhas]

    raise ErroSelfRefine(rotulo, ultimas, executadas)
