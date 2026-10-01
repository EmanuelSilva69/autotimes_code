"""ceemdan - Decomposição CEEMDAN para o pipeline textual do AutoTimes.

Fornece 4 estratégias de decomposição:
  - decompor_ceemdan: CEEMDAN puro (alternativa ao MSTL)
  - decompor_hibrido_mstl_ceemdan: MSTL + CEEMDAN no resíduo
  - decompor_paralelo: MSTL para tendência + CEEMDAN para sazonalidade
  - extrair_periodos_ceemdan: CEEMDAN descobre períodos → MSTL
"""

from .decomposicao import (
    decompor_ceemdan,
    decompor_hibrido_mstl_ceemdan,
    decompor_paralelo,
    extrair_periodos_ceemdan,
)

__all__ = [
    "decompor_ceemdan",
    "decompor_hibrido_mstl_ceemdan",
    "decompor_paralelo",
    "extrair_periodos_ceemdan",
]
