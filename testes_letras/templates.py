"""
templates - 4 formatos de serializacao textual para a MESMA janela numerica.

Problema diagnosticado:
  O GPT-2 foi treinado nativamente em ingles. O formato atual (pt-BR por
  extenso) fragmenta datas e valores ("vinte e nove virgula cinco") em
  muitos sub-tokens, estourando o contexto (block_max=128) e diluindo a
  capacidade do modelo de reter a tendencia temporal (underfitting).

Solucao testada (compressao de prompt / paginacao):
  Para a mesma janela de 672 passos (7 blocos de 96) geramos 4 templates:
    T1 pt_verboso : formato ATUAL por extenso em pt-BR (baseline).
    T2 pt_curto   : datas compactas + valores numericos em pt-BR curto.
    T3 en_curto   : ingles compacto (idioma nativo do GPT-2).
    T4 json       : estrutura JSON/array-like.
  O testador mede tokens/Janela, tokens/bloco, pontos retidos por bloco e
  tempo por batch, e compara a convergencia (MSE/MAE) de cada template.

Este modulo substitui a geracao textual do Testes_eficiencia\\modules\\textual.py
para fins de teste; o arquivo original permanece intacto (baseline).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_MESES_PT_CURTO: List[str] = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez",
]
_MESES_EN: List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _textual_pt():
    """Importa (lazy) o modulo textual do Testes_eficiencia (numeros por extenso)."""
    from . import caminhos
    caminhos.garantir_syspath()
    from modules import textual as _t
    return _t


# --------------------------------------------------------------------- #
#  Formatadores de bloco (lista de (datetime, valor) -> texto)          #
# --------------------------------------------------------------------- #
def formatar_pt_verboso(pts: List[Tuple[datetime, float]], feat_name: str) -> str:
    """Template 1 (baseline): narrativa por extenso em pt-BR.

    Mesma estrutura do gerar_texto_janela do textual.py:
        "Na sessao iniciada em dez de julho de dois mil e vinte e dois,
         as catorze horas, da serie OTELETE, os valores registrados foram:
         as vinte horas o valor era vinte e nove virgula cinco; ..."
    """
    _t = _textual_pt()
    dt0 = pts[0][0]
    cab = (
        f"Na sessao iniciada em {_t.descrever_momento(dt0.year, dt0.month, dt0.day, dt0.hour, dt0.minute)}, "
        f"da serie {feat_name}, os valores registrados foram:"
    )
    frases: List[str] = []
    for dt, v in pts:
        horas = _t.numero_por_extenso(dt.hour)
        tempo = (
            f"as {horas} horas" if dt.minute == 0
            else f"as {horas} horas e {_t.numero_por_extenso(dt.minute)} minutos"
        )
        frases.append(f"{tempo} o valor era {_t.valor_por_extenso(v)}")
    return cab + " " + " ".join(frases) + "."


def formatar_pt_curto(pts: List[Tuple[datetime, float]], feat_name: str) -> str:
    """Template 2: datas e numeros compactos em pt-BR.

        "Serie OTELETE: 10/Jul/2022 14:00 | Valor: 29.5; 10/Jul/2022 15:00 | Valor: 30.1."
    """
    partes: List[str] = []
    for dt, v in pts:
        data = f"{dt.day:02d}/{_MESES_PT_CURTO[dt.month - 1]}/{dt.year} {dt.hour:02d}:{dt.minute:02d}"
        partes.append(f"{data} | Valor: {v:.1f}")
    return f"Serie {feat_name}: " + "; ".join(partes) + "."


def formatar_en_curto(pts: List[Tuple[datetime, float]], feat_name: str) -> str:
    """Template 3: ingles compacto (idioma nativo do GPT-2).

        "Serie OTELETE: Jul 10, 2022, 2:00 PM | Value: 29.5; Jul 10, 2022, 3:00 PM | Value: 30.1."
    """
    partes: List[str] = []
    for dt, v in pts:
        h12 = dt.hour % 12 or 12
        ampm = "AM" if dt.hour < 12 else "PM"
        data = f"{_MESES_EN[dt.month - 1]} {dt.day:02d}, {dt.year}, {h12}:{dt.minute:02d} {ampm}"
        partes.append(f"{data} | Value: {v:.1f}")
    return f"Serie {feat_name}: " + "; ".join(partes) + "."


def formatar_json(pts: List[Tuple[datetime, float]], feat_name: str) -> str:
    """Template 4: estrutura JSON/array-like.

        [{"time": "2022-07-10T14:00", "val": 29.5},
         {"time": "2022-07-10T15:00", "val": 30.1}]
    """
    itens: List[str] = []
    for dt, v in pts:
        itens.append(f'{{"time": "{dt:%Y-%m-%dT%H:%M}", "val": {v:.1f}}}')
    return "[" + ", ".join(itens) + "]"


# --------------------------------------------------------------------- #
#  Registro dos templates                                               #
# --------------------------------------------------------------------- #
TEMPLATES: Dict[str, Dict] = {
    "pt_verboso": {
        "nome": "T1 PT Verboso",
        "lingua": "pt-BR por extenso",
        "desc": "Formato atual (baseline): narrativa verbal, explode tokens.",
        "fn": formatar_pt_verboso,
        "step_fixo": 24,  # baseline do pipeline (4 pontos/bloco)
    },
    "pt_curto": {
        "nome": "T2 PT Curto",
        "lingua": "pt-BR compacto",
        "desc": "Datas compactas + valores numericos (10/Jul/2022 14:00 | Valor: 29.5).",
        "fn": formatar_pt_curto,
        "step_fixo": None,  # auto-ajuste p/ caber em block_max
    },
    "en_curto": {
        "nome": "T3 EN Curto",
        "lingua": "ingles compacto",
        "desc": "Ingles nativo do GPT-2 (Jul 10, 2022, 2:00 PM | Value: 29.5).",
        "fn": formatar_en_curto,
        "step_fixo": None,
    },
    "json": {
        "nome": "T4 JSON",
        "lingua": "JSON/array-like",
        "desc": 'Array JSON ([{"time": "2022-07-10T14:00", "val": 29.5}]).',
        "fn": formatar_json,
        "step_fixo": None,
    },
}

IDS_TEMPLATES: List[str] = list(TEMPLATES.keys())

# Densidade de amostragem candidata dentro de cada bloco de 96 passos.
CANDIDATOS_STEP: List[int] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96]


# --------------------------------------------------------------------- #
#  Geracao de texto por janela                                          #
# --------------------------------------------------------------------- #
def pontos_do_bloco(
    dt_list: List[datetime],
    values: np.ndarray,
    feat: int,
    b_start: int,
    token_len: int,
    text_step: int,
) -> List[Tuple[datetime, float]]:
    """Amostra os (datetime, valor) de um bloco [b_start : b_start+token_len]."""
    pts: List[Tuple[datetime, float]] = []
    for j in range(0, token_len, text_step):
        idx = b_start + j
        if idx >= values.shape[0]:
            break
        pts.append((dt_list[idx], float(values[idx, feat])))
    return pts


def gerar_textos_janela(
    template_id: str,
    dt_list: List[datetime],
    values: np.ndarray,
    feat: int,
    start: int,
    seq_len: int,
    token_len: int,
    feat_name: str,
    text_step: int,
) -> List[str]:
    """Gera os textos dos n_blocks blocos de uma janela (1 texto por bloco).

    Assinatura identica a textual.gerar_texto_janela, porem parametrizada
    pelo template de compressao. n_blocks = seq_len // token_len.
    """
    if template_id not in TEMPLATES:
        raise ValueError(f"Template desconhecido: {template_id}. Validos: {list(TEMPLATES)}")
    fn: Callable = TEMPLATES[template_id]["fn"]
    n_blocks = seq_len // token_len
    textos: List[str] = []
    for b in range(n_blocks):
        b_start = start + b * token_len
        pts = pontos_do_bloco(dt_list, values, feat, b_start, token_len, text_step)
        textos.append(fn(pts, feat_name))
    return textos


# --------------------------------------------------------------------- #
#  Auto-ajuste do text_step (densidade maxima dentro do contexto)       #
# --------------------------------------------------------------------- #
def escolher_text_step(
    template_id: str,
    tokenizer,
    dt_list: List[datetime],
    values: np.ndarray,
    feat_names: List[str],
    seq_len: int,
    token_len: int,
    block_max: int,
    starts_amostra: Optional[Tuple[int, ...]] = (0, 250, 500),
) -> int:
    """Escolhe o text_step mais denso que cabe no contexto (block_max).

    O baseline (pt_verboso) mantem o step fixo do pipeline (24). Os demais
    templates sao auto-ajustados: tenta step=1 (todos os pontos) e aumenta
    a esparsidade ate que NENHUM bloco das janelas amostradas estoure o
    limite. Quanto menor o step, mais pontos temporais por bloco -> mais
    informacao de tendencia por token (compressao real).

    Args:
        template_id: id do template.
        tokenizer: tokenizer GPT-2 local.
        dt_list/values/feat_names: dados numericos carregados.
        seq_len/token_len: geometria das janelas (672/96).
        block_max: limite de tokens por bloco (128).
        starts_amostra: inicios absolutos de janelas usados p/ validar.

    Returns:
        int: text_step escolhido (1 = todos os pontos).
    """
    if TEMPLATES[template_id]["step_fixo"] is not None:
        return int(TEMPLATES[template_id]["step_fixo"])

    for step in CANDIDATOS_STEP:
        ok = True
        for s in starts_amostra:
            for feat in (0, 3):
                textos = gerar_textos_janela(
                    template_id, dt_list, values, feat, s, seq_len, token_len,
                    feat_names[feat], step,
                )
                for txt in textos:
                    n = len(tokenizer(txt, add_special_tokens=False)["input_ids"])
                    if n > block_max:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return step
    return CANDIDATOS_STEP[-1]
