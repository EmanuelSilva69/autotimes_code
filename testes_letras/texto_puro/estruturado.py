"""
estruturado - Formatacoes ESTRUTURADAS de series temporais por extenso.

Apos o teto do prompt engineering classico (frases continuas por extenso,
bandas de confianca sobrepostas) e o encerramento da via arquitetural
(F3+F4 sem sinergia), esta missao testa se a ANCORAGEM ESTRUTURAL do GPT-2
(pré-treinado em codigo/Markdown/JSON) isola melhor a "chave temporal" do
"valor matematico" do que uma frase continua.

Os numeros continuam 100% por extenso (reuso de `extenso.numero_por_extenso`),
mas com delimitadores estruturais rigidos:

    1) ESTRUTURAL (Markdown/Array): chaves rotuladas entre colchetes
       EN: [ Time : July ten , fourteen hours | Val : twenty nine point five ]
       PT: [ Tempo : julho dez , catorze horas | Val : vinte e nove virgula cinco ]
       ES: [ Tiempo : julio diez , catorze horas | Val : veintinueve coma cinco ]

    2) PSEUDO-JSON (chave-valor por extenso):
       EN: key time july ten fourteen hours key value twenty nine point five
       PT: chave tempo julho dez catorze horas chave valor vinte e nove virgula cinco
       ES: clave tiempo julio diez catorce horas clave valor veintinueve coma cinco

    3) RELACIONAL (seta causal):
       EN: July ten fourteen hours -> twenty nine point five
       PT: julho dez catorze horas -> vinte e nove virgula cinco
       ES: julio diez catorce horas -> veintinueve coma cinco

    4) TEMPORAL DELTA (apenas EN, experimento): nao repete o mes/dia; o tempo
       avanca incrementalmente por hora com rotulo T (o mes aparece 1x):
       Start July ten. T fourteen: twenty nine point five. T fifteen: thirty point one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List

from .extenso import _MESES_EN, _MESES_ES, _MESES_PT, numero_por_extenso


# --------------------------------------------------------------------- #
#  Chaves rotuladas (chave temporal / chave de valor) por idioma         #
# --------------------------------------------------------------------- #
# (chave_tempo, chave_valor) - usadas no template pseudo-json.
_CHAVES: Dict[str, Dict[str, str]] = {
    "en": {"tempo": "key time", "valor": "key value"},
    "pt": {"tempo": "chave tempo", "valor": "chave valor"},
    "es": {"tempo": "clave tiempo", "valor": "clave valor"},
}

_MESES: Dict[str, List[str]] = {
    "pt": _MESES_PT,
    "en": _MESES_EN,
    "es": _MESES_ES,
}


def _tempo_keywords(dt: datetime, idioma: str) -> str:
    """'julho dez catorze horas' (mes + dia + hora por extenso)."""
    return (
        f"{_MESES[idioma][dt.month - 1]} "
        f"{numero_por_extenso(dt.day, idioma)} "
        f"{numero_por_extenso(dt.hour, idioma)} horas"
    )


# --------------------------------------------------------------------- #
#  Formatadores de ponto (datetime, valor) -> fragmento estruturado      #
# --------------------------------------------------------------------- #
def _estrutural_en(dt: datetime, valor: float) -> str:
    return (
        f"[ Time : {_MESES_EN[dt.month - 1]} {numero_por_extenso(dt.day, 'en')} , "
        f"{numero_por_extenso(dt.hour, 'en')} hours | "
        f"Val : {numero_por_extenso(valor, 'en')} ]"
    )


def _estrutural_pt(dt: datetime, valor: float) -> str:
    return (
        f"[ Tempo : {_MESES_PT[dt.month - 1]} {numero_por_extenso(dt.day, 'pt')} , "
        f"{numero_por_extenso(dt.hour, 'pt')} horas | "
        f"Val : {numero_por_extenso(valor, 'pt')} ]"
    )


def _estrutural_es(dt: datetime, valor: float) -> str:
    return (
        f"[ Tiempo : {_MESES_ES[dt.month - 1]} {numero_por_extenso(dt.day, 'es')} , "
        f"{numero_por_extenso(dt.hour, 'es')} horas | "
        f"Val : {numero_por_extenso(valor, 'es')} ]"
    )


def _pseudo_json(dt: datetime, valor: float, idioma: str) -> str:
    chaves = _CHAVES[idioma]
    return (
        f"{chaves['tempo']} {_tempo_keywords(dt, idioma)} "
        f"{chaves['valor']} {numero_por_extenso(valor, idioma)}"
    )


def _relacional(dt: datetime, valor: float, idioma: str) -> str:
    tempo = _tempo_keywords(dt, idioma)
    if idioma == "en":
        tempo = tempo[0].upper() + tempo[1:]
    return f"{tempo} -> {numero_por_extenso(valor, idioma)}"


def _temporal_delta(dt: datetime, valor: float) -> str:
    return f"T {numero_por_extenso(dt.hour, 'en')}: {numero_por_extenso(valor, 'en')}"


# --------------------------------------------------------------------- #
#  Registro dos 10 templates estruturados                               #
# --------------------------------------------------------------------- #
TEMPLATES_ESTRUTURADOS: Dict[str, Dict] = {
    "estrutural_en": {"nome": "Estrutural EN", "idioma": "en", "tipo": "estrutural",
                      "exemplo": "[ Time : ... | Val : ... ]", "fn": _estrutural_en},
    "estrutural_pt": {"nome": "Estrutural PT", "idioma": "pt", "tipo": "estrutural",
                      "exemplo": "[ Tempo : ... | Val : ... ]", "fn": _estrutural_pt},
    "estrutural_es": {"nome": "Estrutural ES", "idioma": "es", "tipo": "estrutural",
                      "exemplo": "[ Tiempo : ... | Val : ... ]", "fn": _estrutural_es},
    "pseudo_json_en": {"nome": "Pseudo-JSON EN", "idioma": "en", "tipo": "pseudo_json",
                       "exemplo": "key time ... key value ...",
                       "fn": lambda dt, v: _pseudo_json(dt, v, "en")},
    "pseudo_json_pt": {"nome": "Pseudo-JSON PT", "idioma": "pt", "tipo": "pseudo_json",
                       "exemplo": "chave tempo ... chave valor ...",
                       "fn": lambda dt, v: _pseudo_json(dt, v, "pt")},
    "pseudo_json_es": {"nome": "Pseudo-JSON ES", "idioma": "es", "tipo": "pseudo_json",
                       "exemplo": "clave tiempo ... clave valor ...",
                       "fn": lambda dt, v: _pseudo_json(dt, v, "es")},
    "relacional_en": {"nome": "Relacional EN", "idioma": "en", "tipo": "relacional",
                      "exemplo": "July ten fourteen hours -> value",
                      "fn": lambda dt, v: _relacional(dt, v, "en")},
    "relacional_pt": {"nome": "Relacional PT", "idioma": "pt", "tipo": "relacional",
                      "exemplo": "julho dez catorze horas -> valor",
                      "fn": lambda dt, v: _relacional(dt, v, "pt")},
    "relacional_es": {"nome": "Relacional ES", "idioma": "es", "tipo": "relacional",
                      "exemplo": "julio diez catorce horas -> valor",
                      "fn": lambda dt, v: _relacional(dt, v, "es")},
    "temporal_delta_en": {"nome": "Temporal Delta EN", "idioma": "en", "tipo": "temporal_delta",
                          "exemplo": "Start July ten. T fourteen: value.",
                          "fn": None},  # tratado em serie_para_texto_estruturado
}

IDS_TEMPLATES_ESTRUTURADOS: List[str] = list(TEMPLATES_ESTRUTURADOS.keys())


def formatar_ponto_estruturado(template_id: str, dt: datetime, valor: float) -> str:
    """Aplica o template estruturado a um ponto (datetime, valor)."""
    if template_id not in TEMPLATES_ESTRUTURADOS:
        raise ValueError(
            f"Template estruturado desconhecido: {template_id}. "
            f"Validos: {IDS_TEMPLATES_ESTRUTURADOS}"
        )
    fn: Callable = TEMPLATES_ESTRUTURADOS[template_id]["fn"]
    if fn is None:
        raise ValueError(f"Template {template_id} usa geracao por documento")
    return fn(dt, valor)


def serie_para_texto_estruturado(
    dt_list: List[datetime],
    values,
    feat: int,
    inicio: int,
    fim: int,
    template_id: str,
) -> str:
    """Converte linhas [inicio:fim] da serie em UM documento estruturado.

    Per-ponto: cada linha vira um fragmento estruturado (chaves, seta,
    colchetes). Temporal delta: o mes aparece 1x ("Start July ten.") e o
    tempo avanca por hora com rotulo T - nunca repete a data completa.
    """
    if template_id == "temporal_delta_en":
        partes: List[str] = []
        for i in range(inicio, min(fim, len(dt_list))):
            dt = dt_list[i]
            valor = round(float(values[i, feat]), 1)
            if i == inicio:
                partes.append(
                    f"Start {_MESES_EN[dt.month - 1]} "
                    f"{numero_por_extenso(dt.day, 'en')}."
                )
            else:
                partes.append(_temporal_delta(dt, valor) + ".")
        return " ".join(partes)

    pontos: List[str] = []
    for i in range(inicio, min(fim, len(dt_list))):
        valor = round(float(values[i, feat]), 1)  # 1 casa decimal (sem ruido fp32)
        pontos.append(formatar_ponto_estruturado(template_id, dt_list[i], valor))
    return " ".join(pontos) + "."
