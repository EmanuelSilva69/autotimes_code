"""
en_otimizado - Templates em ingles para o otimizar_ingles_extenso.py.

H1 - DENSIDADE DE CONTEXTO (lookback x2):
    Os formatos "x2" injetam DOIS pontos de historico por frase (dois
    valores compartilhando um unico datetime), dobrando os pontos por pagina
    dentro do MESMO orcamento de tokens:
        en_curto_x2    : "July ten, fourteen hours: twenty nine point five,
                          thirty point one."
        en_keywords_x2 : "july tenth fourteen hours twenty nine point five
                          thirty point one"

H3 - ABLACAO DO MARCADOR DECIMAL ("sweet spot"):
    Duas variacoes hibridas que mantem o extenso mas isolam a matematica:
        en_math_explicit : "July ten fourteen hours value twenty nine
                            point five"            (extenso puro, cardinal)
        en_pseudo_digit  : "July ten fourteen hours value twenty nine
                            . five"                 (ponto literal ' . '
                            ancorando a fracao no atencao do GPT-2)
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List

from .extenso import _MESES_EN, dia_ordinal_en, numero_por_extenso


def formatar_en_math_explicit(dt: datetime, valor: float) -> str:
    """EN Math-Explicit: 'July ten fourteen hours value twenty nine point five'."""
    mes = _MESES_EN[dt.month - 1]
    return (
        f"{mes} {numero_por_extenso(dt.day, 'en')} "
        f"{numero_por_extenso(dt.hour, 'en')} hours value "
        f"{numero_por_extenso(valor, 'en')}"
    )


def formatar_en_pseudo_digit(dt: datetime, valor: float) -> str:
    """EN Pseudo-Digit: como Math-Explicit, mas 'point' vira '.' literal."""
    mes = _MESES_EN[dt.month - 1]
    valor_txt = numero_por_extenso(valor, "en").replace(" point ", " . ")
    return (
        f"{mes} {numero_por_extenso(dt.day, 'en')} "
        f"{numero_por_extenso(dt.hour, 'en')} hours value {valor_txt}"
    )


def formatar_en_curto_x2(dt: datetime, v1: float, v2: float) -> str:
    """EN Curto com lookback x2: um datetime, DOIS valores consecutivos."""
    mes = _MESES_EN[dt.month - 1]
    return (
        f"{mes} {dia_ordinal_en(dt.day)}, {numero_por_extenso(dt.hour, 'en')} hours: "
        f"{numero_por_extenso(v1, 'en')}, {numero_por_extenso(v2, 'en')}"
    )


def formatar_en_keywords_x2(dt: datetime, v1: float, v2: float) -> str:
    """EN Keywords com lookback x2 (estilo cru, sem pontuacao/stopwords)."""
    mes = _MESES_EN[dt.month - 1].lower()
    return (
        f"{mes} {dia_ordinal_en(dt.day)} {numero_por_extenso(dt.hour, 'en')} hours "
        f"{numero_por_extenso(v1, 'en')} {numero_por_extenso(v2, 'en')}"
    )


TEMPLATES_EN_OTIMIZADO: Dict[str, Dict] = {
    "en_math_explicit": {
        "nome": "EN Math-Explicit", "idioma": "en", "x2": False,
        "fn": formatar_en_math_explicit,
    },
    "en_pseudo_digit": {
        "nome": "EN Pseudo-Digit", "idioma": "en", "x2": False,
        "fn": formatar_en_pseudo_digit,
    },
    "en_curto_x2": {
        "nome": "EN Curto x2 (lookback)", "idioma": "en", "x2": True,
        "fn": formatar_en_curto_x2,
    },
    "en_keywords_x2": {
        "nome": "EN Keywords x2 (lookback)", "idioma": "en", "x2": True,
        "fn": formatar_en_keywords_x2,
    },
}


def serie_para_texto_simples(
    dt_list: List[datetime], values, feat: int, inicio: int, fim: int,
    fn: Callable[[datetime, float], str],
) -> str:
    """Documento de pontos individuais (1 ponto por frase, unido por espaco)."""
    frases: List[str] = []
    for i in range(inicio, min(fim, len(dt_list))):
        valor = round(float(values[i, feat]), 1)
        frases.append(fn(dt_list[i], valor))
    return " ".join(frases) + "."


def serie_para_texto_x2(
    dt_list: List[datetime], values, feat: int, inicio: int, fim: int,
    fn2: Callable[[datetime, float, float], str],
) -> str:
    """Documento com lookback dobrado: 2 pontos consecutivos por frase."""
    frases: List[str] = []
    fim_par = min(fim, len(dt_list)) - 1
    for i in range(inicio, fim_par, 2):
        v1 = round(float(values[i, feat]), 1)
        v2 = round(float(values[i + 1, feat]), 1)
        frases.append(fn2(dt_list[i], v1, v2))
    return " ".join(frases) + "."
