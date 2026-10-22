"""Self-Refine LOCAL para a geracao dos documentos textuais (opcional).

Ativado apenas com `--self-refine` no runner (padrao: desligado). Com a
flag ausente nada aqui e importado no fluxo normal: zero validacao, zero
memoria, zero retry, zero mencao no log.

Papeis (design D8):
    validadores.py  checagens puras (contrato T/S/R, finitude, T+S+R,
                    tensores ids/lens/max_abs, orcamento 896/7)
    memoria.py      leitura tolerante + escrita atomica + FIFO em
                    .runtime/autotimes_memory.json (fora do Git)
    agente.py       prompt de refinamento com as 6 tags (registro local,
                    costura para um LLM no futuro - nenhuma API hoje)
    loop.py         tentativa -> validacao -> feedback -> retry -> memoria

Nenhuma dependencia nova e nenhuma chamada de rede.
"""

from __future__ import annotations

from .loop import (  # noqa: F401
    MAX_TENTATIVAS_PADRAO,
    ErroSelfRefine,
    ResultadoLoop,
    rodar_self_refine,
)
from .memoria import Entrada, Memoria  # noqa: F401
from .validadores import Falha, REGRAS  # noqa: F401
