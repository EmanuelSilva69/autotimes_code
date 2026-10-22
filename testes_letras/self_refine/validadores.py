"""Validadores PUROS do documento de entrada (Self-Refine).

Nenhuma I/O, nenhum efeito colateral, sem dependencia alem de numpy: as
checagens rodam em qualquer processo (pai ou worker loky) e devolvem `None`
quando o contrato passa, ou uma `Falha` com o CODIGO e a REGRA de correcao
associada (design D12: retry por regra, nao por sorte).

Contratos cobertos (spec self-refine-textual-generation):
    1. validar_frases         -> 3 frases T/S/R com os marcadores oficiais
    2. validar_finitude       -> nenhum NaN/Inf no que vai ser escrito
    3. validar_reconstrucao   -> T + S + R == serie de origem (tolerancia)
    4. validar_tensores       -> ids/lens/max_abs com as formas do dataset
    5. validar_orcamento      -> n_paginas_max paginas de block_max tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---- Codigos de falha + regra de correcao (fonte unica) ---------------- #
REGRAS: Dict[str, str] = {
    "FRASE_AUSENTE": (
        "regere a decomposicao da janela e escreva as 3 frases na ordem "
        "Tendencia/Sazonalidade/Residuos"
    ),
    "MARCADOR_INVALIDO": (
        "regrave cada frase com o marcador oficial "
        "(Tendencia:/Sazonalidade:/Residuos:)"
    ),
    "VALOR_NAO_FINITO": (
        "sanea NaN/Inf na normalizacao local (max_abs) e reescreva o texto"
    ),
    "RECONSTRUCAO_FORA": (
        "refaz a decomposicao com os mesmos parametros e usa "
        "residuo = x - T - S"
    ),
    "TENSOR_CONTRATO": (
        "refaz a paginacao: n_paginas_max paginas de block_max tokens com as "
        "3 frases"
    ),
    "ORCAMENTO_ESTOURADO": (
        "trunca na cabeca: paginas[:n_paginas_max] (sem regerar o texto)"
    ),
}

# Tolerancia RELATIVA da reconstrucao T+S+R (erro de float64; a decomposicao
# e exata por construcao, entao sobra so ruido numerico).
TOLERANCIA_PADRAO: float = 1e-6


@dataclass(frozen=True)
class Falha:
    """Uma violacao de contrato: o que falhou (codigo) e como corrigir."""

    codigo: str
    detalhe: str
    contexto: Dict[str, Any] = field(default_factory=dict)

    @property
    def regra(self) -> str:
        """Regra de correcao associada ao codigo (feedback injetado no retry)."""
        return REGRAS.get(self.codigo, "regera o documento")

    def como_feedback(self) -> Dict[str, Any]:
        """Formato injetado na nova tentativa e gravado na memoria."""
        return {
            "codigo": self.codigo,
            "motivo": self.detalhe,
            "regra": self.regra,
        }

    def __str__(self) -> str:
        return f"[{self.codigo}] {self.detalhe} -> correcao: {self.regra}"


def _marcadores() -> Sequence[str]:
    """Marcadores oficiais, lidos do modulo que os define (unica fonte)."""
    from ..texto_char_by_char import (  # import tardio: nao pesa o pacote
        MARCADOR_RESIDUOS,
        MARCADOR_SAISONALIDADE,
        MARCADOR_TENDENCIA,
    )
    return (MARCADOR_TENDENCIA, MARCADOR_SAISONALIDADE, MARCADOR_RESIDUOS)


# --------------------------------------------------------------------- #
#  1-2. Texto: marcadores/frases + finitude                              #
# --------------------------------------------------------------------- #
def validar_frases(frases: Sequence[str]) -> Optional[Falha]:
    """Contrato das 3 frases: presentes, na ordem e com marcador."""
    esperados = _marcadores()
    if frases is None or len(frases) != 3:
        return Falha(
            "FRASE_AUSENTE",
            f"esperado 3 frases (T/S/R), recebido "
            f"{0 if frases is None else len(frases)}",
        )
    for i, (frase, marc) in enumerate(zip(frases, esperados)):
        if not frase or not frase.strip():
            return Falha(
                "FRASE_AUSENTE",
                f"frase {i} ({marc}) vazia",
                {"indice": i},
            )
        if not frase.startswith(marc):
            return Falha(
                "MARCADOR_INVALIDO",
                f"frase {i} deveria comecar em {marc!r} e comeca em "
                f"{frase[:24]!r}",
                {"indice": i},
            )
    return None


def validar_finitude(valores: Any) -> Optional[Falha]:
    """Nenhum NaN/Inf em nada que vai ser escrito/normalizado."""
    arr = np.asarray(valores, dtype=np.float64).ravel()
    if arr.size == 0:
        return Falha("VALOR_NAO_FINITO", "sequencia de valores vazia")
    maus = ~np.isfinite(arr)
    if bool(maus.any()):
        k = int(np.argmax(maus))
        return Falha(
            "VALOR_NAO_FINITO",
            f"{int(maus.sum())} valor(es) nao finito(s); o primeiro no "
            f"indice {k} = {arr[k]!r}",
            {"n_invalidos": int(maus.sum()), "indice": k},
        )
    return None


# --------------------------------------------------------------------- #
#  3. Reconstrucao T + S + R                                             #
# --------------------------------------------------------------------- #
def validar_reconstrucao(x_window: np.ndarray, trend: np.ndarray,
                         saz: np.ndarray, resid: np.ndarray,
                         tolerancia: float = TOLERANCIA_PADRAO) -> Optional[Falha]:
    """T + S + R tem de reproduzir a serie de origem (exato por construcao)."""
    x = np.asarray(x_window, dtype=np.float64).ravel()
    soma = (np.asarray(trend, dtype=np.float64).ravel()
            + np.asarray(saz, dtype=np.float64).ravel()
            + np.asarray(resid, dtype=np.float64).ravel())
    if x.shape != soma.shape:
        return Falha(
            "RECONSTRUCAO_FORA",
            f"formas divergentes: origem {x.shape} x T+S+R {soma.shape}",
        )
    escala = max(float(np.max(np.abs(x))), 1e-12)
    erro = float(np.max(np.abs(x - soma))) / escala
    if erro > tolerancia:
        return Falha(
            "RECONSTRUCAO_FORA",
            f"erro relativo {erro:.3e} > tolerancia {tolerancia:.3e}",
            {"erro_relativo": erro, "tolerancia": tolerancia},
        )
    return None


# --------------------------------------------------------------------- #
#  4-5. Dataset: tensores + orcamento                                    #
# --------------------------------------------------------------------- #
def validar_tensores(ids: Any, lens: Any, max_abs: Any,
                     block_max: int, n_paginas_max: int,
                     n_componentes: int = 3) -> Optional[Falha]:
    """Contrato do dataset char-by-char (ids/lens/max_abs)."""
    ids_t = np.asarray(ids)
    lens_t = np.asarray(lens)
    mx = np.asarray(max_abs)
    formas = {
        "ids": tuple(ids_t.shape),
        "lens": tuple(lens_t.shape),
        "max_abs": tuple(mx.shape),
    }
    esperadas = {
        "ids": (block_max * n_paginas_max,),
        "lens": (n_paginas_max,),
        "max_abs": (n_componentes, 1),
    }
    for campo in ("ids", "lens", "max_abs"):
        if formas[campo] != esperadas[campo]:
            return Falha(
                "TENSOR_CONTRATO",
                f"{campo} {formas[campo]} != {esperadas[campo]}",
                {"formas": str(formas)},
            )
    if float(np.min(lens_t)) < 0 or float(np.max(lens_t)) > block_max:
        return Falha(
            "TENSOR_CONTRATO",
            f"lens fora do intervalo [0, {block_max}]: "
            f"[{float(np.min(lens_t)):.0f}, {float(np.max(lens_t)):.0f}]",
        )
    return None


def validar_orcamento(lens: Any, block_max: int,
                      n_paginas_max: int) -> Optional[Falha]:
    """7 paginas x block_max tokens (896 no protocolo oficial)."""
    lens_t = np.asarray(lens)
    if lens_t.size != n_paginas_max:
        return Falha(
            "ORCAMENTO_ESTOURADO",
            f"{lens_t.size} paginas != {n_paginas_max} do orcamento",
        )
    total = int(lens_t.sum())
    limite = block_max * n_paginas_max
    if total > limite:
        return Falha(
            "ORCAMENTO_ESTOURADO",
            f"{total} tokens paginados > orcamento {limite}",
            {"total": total, "limite": limite},
        )
    return None


# --------------------------------------------------------------------- #
#  Conveniencia: valida um documento Char-by-Char completo               #
# --------------------------------------------------------------------- #
def validar_documento_char_by_char(frases: Sequence[str], x_window: np.ndarray,
                                   trend: np.ndarray, saz: np.ndarray,
                                   resid: np.ndarray,
                                   ids: Any = None, lens: Any = None,
                                   max_abs: Any = None,
                                   block_max: int = 128,
                                   n_paginas_max: int = 7) -> List[Falha]:
    """Roda os 5 verificadores e devolve a lista de falhas (vazia = valido)."""
    checagens = [
        validar_frases(frases),
        validar_finitude([np.asarray(v, dtype=np.float64).ravel()
                          for v in (x_window, trend, saz, resid)]),
        validar_reconstrucao(x_window, trend, saz, resid),
    ]
    if ids is not None:
        checagens.append(validar_tensores(ids, lens, max_abs,
                                          block_max, n_paginas_max))
        checagens.append(validar_orcamento(lens, block_max, n_paginas_max))
    return [c for c in checagens if c is not None]
