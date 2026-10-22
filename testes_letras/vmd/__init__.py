"""vmd - Decomposição VMD (Variational Mode Decomposition) do AutoTimes.

Alternativa ao CEEMDAN com custo muito menor (FFT + <=500 iteracoes,
dezenas de ms por janela vs ~2200 ms do CEEMDAN em CPU).

Fornece:
  - decompor_vmd: VMD puro -> (tendencia, sazonalidade, residuo)
  - extrair_modes_vmd: modos VMD ordenados por frequencia central

Mesma interface (tendencia, sazonalidade, residuo) que
ceemdan.decomposicao.decompor_ceemdan - uso intercambiavel no pipeline.
"""

from .decomposicao import decompor_vmd, extrair_modes_vmd

__all__ = ["decompor_vmd", "extrair_modes_vmd"]
