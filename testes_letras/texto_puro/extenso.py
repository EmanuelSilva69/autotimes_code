"""
extenso - Conversao 100% por extenso de numeros/timestamps (PT/EN/ES).

Gera os 9 templates (3 idiomas x 3 verbosidades) para o benchmark de NLP
pura do GPT-2 sobre o ETTh1, eliminando TODOS os algarismos:

    Portugues (PT):
      Verboso  : "No dia dez de julho, as catorze horas, o valor registrado
                  foi vinte e nove virgula cinco."
      Curto    : "Dez de julho, catorze horas, valor vinte e nove virgula
                  cinco."
      Keywords : "julho dez catorze horas vinte nove virgula cinco."
    Ingles (EN):
      Verboso  : "On July tenth, at fourteen hours, the recorded value was
                  twenty nine point five."
      Curto    : "July tenth, fourteen hours, value twenty nine point five."
      Keywords : "july tenth fourteen hours twenty nine point five."
    Espanhol (ES):
      Verboso  : "El diez de julio, a las catorce horas, el valor registrado
                  fue veintinueve coma cinco."
      Curto    : "Diez de julio, catorce horas, valor veintinueve coma cinco."
      Keywords : "julio diez catorce horas veintinueve coma cinco."

A biblioteca `num2words` traduz qualquer valor numerico (inteiro ou float
com 1 casa decimal) estritamente para palavras, com normalizacoes:
    PT: "vinte e nove virgula cinco" (ja correto no num2words);
    EN: "twenty nine point five"     (remove o hifen: twenty-nine -> espaco);
    ES: "veintinueve coma cinco"     (num2words diz "punto" -> "coma").

Os templates "keywords" caem para minusculas, usam mes por extenso antes do
dia e removem a conjuncao "e" dos numeros PT ("vinte e nove" -> "vinte
nove"), como pedido no enunciado.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Dict, List, Tuple

from num2words import num2words

_MESES_PT: List[str] = [
    "janeiro", "fevereiro", "marco", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]
_MESES_EN: List[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MESES_ES: List[str] = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
_MESES_KEYWORDS: Dict[str, List[str]] = {
    "pt": [m.lower() for m in _MESES_PT],
    "en": [m.lower() for m in _MESES_EN],
    "es": [m.lower() for m in _MESES_ES],
}


# --------------------------------------------------------------------- #
#  Conversor universal (num2words + normalizacoes por idioma)           #
# --------------------------------------------------------------------- #
def numero_por_extenso(numero: float, idioma: str) -> str:
    """Traduz um numero (inteiro ou float 1 casa) estritamente para palavras.

    Args:
        numero: valor numerico (horas, dias, valores da serie).
        idioma: 'pt' (BR), 'en' ou 'es'.

    Returns:
        Numero por extenso no idioma, ex.: 29.5 -> "vinte e nove virgula
        cinco" / "twenty nine point five" / "veintinueve coma cinco".

    Raises:
        ValueError: idioma desconhecido.
    """
    if idioma == "pt":
        return num2words(numero, lang="pt")
    if idioma == "en":
        return num2words(numero, lang="en").replace("-", " ")
    if idioma == "es":
        return num2words(numero, lang="es").replace("punto", "coma")
    raise ValueError(f"Idioma desconhecido para por extenso: {idioma}")


def dia_ordinal_en(dia: int) -> str:
    """Dia em ordinal no ingles ('July tenth'): 10 -> 'tenth'."""
    return num2words(dia, lang="en", to="ordinal")


def _remover_conjuncao_pt(texto: str) -> str:
    """Remove a conjuncao 'e' (numero PT keyword): 'vinte e nove' -> 'vinte
    nove'. Usa boundaries de palavra para nao mutilar 'dezesseis' etc."""
    return re.sub(r"\be\b", " ", texto).replace("  ", " ").strip()


# --------------------------------------------------------------------- #
#  Formatadores de ponto (datetime, valor) -> frase por extenso         #
# --------------------------------------------------------------------- #
def _pt_verboso(dt: datetime, valor: float) -> str:
    return (
        f"No dia {numero_por_extenso(dt.day, 'pt')} de {_MESES_PT[dt.month - 1]}, "
        f"as {numero_por_extenso(dt.hour, 'pt')} horas, o valor registrado foi "
        f"{numero_por_extenso(valor, 'pt')}."
    )


def _pt_curto(dt: datetime, valor: float) -> str:
    return (
        f"{numero_por_extenso(dt.day, 'pt').capitalize()} de {_MESES_PT[dt.month - 1]}, "
        f"{numero_por_extenso(dt.hour, 'pt')} horas, valor "
        f"{numero_por_extenso(valor, 'pt')}."
    )


def _pt_keywords(dt: datetime, valor: float) -> str:
    return (
        f"{_MESES_KEYWORDS['pt'][dt.month - 1]} "
        f"{numero_por_extenso(dt.day, 'pt')} "
        f"{numero_por_extenso(dt.hour, 'pt')} horas "
        f"{_remover_conjuncao_pt(numero_por_extenso(valor, 'pt'))}"
    )


def _en_verboso(dt: datetime, valor: float) -> str:
    return (
        f"On {_MESES_EN[dt.month - 1]} {dia_ordinal_en(dt.day)}, "
        f"at {numero_por_extenso(dt.hour, 'en')} hours, the recorded value was "
        f"{numero_por_extenso(valor, 'en')}."
    )


def _en_curto(dt: datetime, valor: float) -> str:
    return (
        f"{_MESES_EN[dt.month - 1]} {dia_ordinal_en(dt.day)}, "
        f"{numero_por_extenso(dt.hour, 'en')} hours, value "
        f"{numero_por_extenso(valor, 'en')}."
    )


def _en_keywords(dt: datetime, valor: float) -> str:
    return (
        f"{_MESES_KEYWORDS['en'][dt.month - 1]} {dia_ordinal_en(dt.day)} "
        f"{numero_por_extenso(dt.hour, 'en')} hours "
        f"{numero_por_extenso(valor, 'en')}"
    )


def _es_verboso(dt: datetime, valor: float) -> str:
    return (
        f"El {numero_por_extenso(dt.day, 'es')} de {_MESES_ES[dt.month - 1]}, "
        f"a las {numero_por_extenso(dt.hour, 'es')} horas, el valor registrado fue "
        f"{numero_por_extenso(valor, 'es')}."
    )


def _es_curto(dt: datetime, valor: float) -> str:
    return (
        f"{numero_por_extenso(dt.day, 'es').capitalize()} de {_MESES_ES[dt.month - 1]}, "
        f"{numero_por_extenso(dt.hour, 'es')} horas, valor "
        f"{numero_por_extenso(valor, 'es')}."
    )


def _es_keywords(dt: datetime, valor: float) -> str:
    return (
        f"{_MESES_KEYWORDS['es'][dt.month - 1]} {numero_por_extenso(dt.day, 'es')} "
        f"{numero_por_extenso(dt.hour, 'es')} horas "
        f"{numero_por_extenso(valor, 'es')}"
    )


# --------------------------------------------------------------------- #
#  Registro dos 9 templates                                             #
# --------------------------------------------------------------------- #
TEMPLATES_EXTENSO: Dict[str, Dict] = {
    "pt_verboso":  {"nome": "PT Verboso",  "idioma": "pt", "nivel": "verboso",
                    "separador": " ", "fn": _pt_verboso},
    "pt_curto":    {"nome": "PT Curto",    "idioma": "pt", "nivel": "curto",
                    "separador": " ", "fn": _pt_curto},
    "pt_keywords": {"nome": "PT Keywords", "idioma": "pt", "nivel": "keywords",
                    "separador": " ", "fn": _pt_keywords},
    "en_verboso":  {"nome": "EN Verboso",  "idioma": "en", "nivel": "verboso",
                    "separador": " ", "fn": _en_verboso},
    "en_curto":    {"nome": "EN Curto",    "idioma": "en", "nivel": "curto",
                    "separador": " ", "fn": _en_curto},
    "en_keywords": {"nome": "EN Keywords", "idioma": "en", "nivel": "keywords",
                    "separador": " ", "fn": _en_keywords},
    "es_verboso":  {"nome": "ES Verboso",  "idioma": "es", "nivel": "verboso",
                    "separador": " ", "fn": _es_verboso},
    "es_curto":    {"nome": "ES Curto",    "idioma": "es", "nivel": "curto",
                    "separador": " ", "fn": _es_curto},
    "es_keywords": {"nome": "ES Keywords", "idioma": "es", "nivel": "keywords",
                    "separador": " ", "fn": _es_keywords},
}

IDS_TEMPLATES_EXTENSO: List[str] = list(TEMPLATES_EXTENSO.keys())


def formatar_ponto(template_id: str, dt: datetime, valor: float) -> str:
    """Aplica o template por extenso a um ponto (datetime, valor)."""
    if template_id not in TEMPLATES_EXTENSO:
        raise ValueError(f"Template por extenso desconhecido: {template_id}. "
                         f"Validos: {IDS_TEMPLATES_EXTENSO}")
    fn: Callable[[datetime, float], str] = TEMPLATES_EXTENSO[template_id]["fn"]
    return fn(dt, valor)


def serie_para_texto_extenso(
    dt_list: List[datetime],
    values,
    feat: int,
    inicio: int,
    fim: int,
    template_id: str,
) -> str:
    """Converte linhas [inicio:fim] da serie em UM documento por extenso.

    Verboso/curto: frases terminadas em ponto, unidas por espaco.
    Keywords: lista crua de palavras-chave, unida por espaco + ponto final.
    """
    pontos: List[str] = []
    for i in range(inicio, min(fim, len(dt_list))):
        valor = round(float(values[i, feat]), 1)  # 1 casa decimal (sem ruido fp32)
        pontos.append(formatar_ponto(template_id, dt_list[i], valor))
    return " ".join(pontos) + "."
