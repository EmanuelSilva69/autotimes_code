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
from .formato import FORMATO_PADRAO, FormatoTexto  # noqa: F401  (formato do texto)


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


def _tempo_keywords_politica(dt: datetime, idioma: str,
                             politica: str = "completa") -> str:
    """Carimbo temporal do ponto conforme a politica de data escolhida.

        completa -> 'julio catorce veintitres horas'  (atual, byte a byte)
        curta    -> 'catorce veintitres horas'        (sem o mes)
        hora     -> 'veintitres horas'                (so a hora)

    'indice' e 'nenhuma' nao passam por aqui (o ponto nem usa /time) -
    ver _pseudo_json_mstl.
    """
    if politica == "completa":
        return _tempo_keywords(dt, idioma)
    hora = f"{numero_por_extenso(dt.hour, idioma)} horas"
    if politica == "curta":
        return f"{numero_por_extenso(dt.day, idioma)} {hora}"
    if politica == "hora":
        return hora
    raise ValueError(
        f"politica de data invalida para o carimbo: {politica!r} "
        "(use completa/curta/hora; 'indice' e 'nenhuma' sao tratados "
        "em _pseudo_json_mstl)"
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


def _formato_pj(formato) -> "FormatoTexto":
    """Aceita FormatoTexto, None (= padrao 'completa') ou so a politica (str)."""
    if formato is None:
        return FORMATO_PADRAO
    if isinstance(formato, FormatoTexto):
        return formato
    if isinstance(formato, str):
        return FormatoTexto(data=formato)
    raise TypeError(f"formato invalido: {type(formato)!r}")


def _pseudo_json_mstl(dt: datetime, trend: float, saz: float, resid: float,
                      formato=None, idx: int = 0) -> str:
    """Template MSTL: /time ... /t ... /s ... /r ... (economiza tokens).

    A politica de data (FormatoTexto.data, padrao 'completa') decide o
    carimbo temporal de TODOS os pontos do documento:

        completa -> /time julio catorce veintitres horas /t ... (atual)
        curta    -> /time catorce veintitres horas /t ...
        hora     -> /time veintitres horas /t ...
        indice   -> /i <idx> /t ...  (posicao do ponto, custo ~3 tokens)
        nenhuma  -> /t ...  (sem nenhum token de data: +~13 tokens/ponto
                    livres no orcamento de 896)

    Args:
        dt: timestamp do ponto (nao usado em 'nenhuma'/'indice').
        trend/saz/resid: componentes do ponto.
        formato: FormatoTexto | politica (str) | None = padrao (2 casas,
            marcacao normal, escala float = texto atual byte a byte).
        idx: indice do ponto no documento (usado em formato='indice').
    """
    fmt = _formato_pj(formato)
    politica = fmt.data
    if politica == "nenhuma":
        cabeca = ""
    elif politica == "indice":
        cabeca = f"/i {int(idx)} " if fmt.pj_marcacao == "normal" \
            else f"/i{int(idx)}"
    else:
        cabeca = f"/time {_tempo_keywords_politica(dt, 'es', politica)} "
    if fmt.pj_escala == "inteiro":
        vals = (f"{int(round(v * 100)):d}" for v in (trend, saz, resid))
    else:
        c = int(fmt.pj_casas)
        vals = (f"{v:.{c}f}" for v in (trend, saz, resid))
    v_t, v_s, v_r = vals
    if fmt.pj_marcacao == "compacta":
        return f"{cabeca}/t{v_t}/s{v_s}/r{v_r}"
    return f"{cabeca}/t {v_t} /s {v_s} /r {v_r}"


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
    "pseudo_json_mstl": {"nome": "Pseudo-JSON MSTL", "idioma": "es", "tipo": "pseudo_json_mstl",
                         "exemplo": "/time julio diez catorce horas /t 12.34 /s -0.56 /r 0.12",
                         "fn": None},  # geração por documento (MSTL precisa da série inteira)
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


def serie_para_texto_mstl(
    dt_list: List[datetime],
    values,
    feat: int,
    n_treino: int,
    periodos: tuple,
    formato=None,
) -> str:
    """Aplica MSTL na série inteira [0:n_treino] e gera documento /time /t /s /r.

    formato: FormatoTexto (ou politica em str) definindo a data dos pontos;
    None = padrao 'completa' (texto historico byte a byte).
    """
    import warnings
    import numpy as np

    from statsmodels.tsa.seasonal import MSTL

    x = np.asarray(values[:n_treino, feat], dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    L = len(x)
    periodos_validos = [int(p) for p in periodos if 2 <= int(p) < L // 2]
    if not periodos_validos:
        periodos_validos = [2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MSTL(x, periods=periodos_validos, stl_kwargs={"robust": True}).fit()

    trend = np.asarray(res.trend, dtype=np.float64)
    seasonal = np.asarray(res.seasonal, dtype=np.float64)
    if seasonal.ndim == 2:
        saz = seasonal.sum(axis=1)
    else:
        saz = seasonal
    resid = np.asarray(res.resid, dtype=np.float64)

    pontos: List[str] = []
    for i in range(min(n_treino, len(dt_list))):
        pontos.append(_pseudo_json_mstl(dt_list[i], trend[i], saz[i], resid[i],
                                        formato=formato, idx=i))
    return " ".join(pontos) + "."


def serie_para_texto_ceemdan(
    dt_list: List[datetime],
    values,
    feat: int,
    n_treino: int,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
    formato=None,
) -> str:
    """Aplica CEEMDAN na série inteira [0:n_treino] e gera documento /time /t /s /r.

    formato: FormatoTexto (ou politica em str) definindo a data dos pontos;
    None = padrao 'completa' (texto historico byte a byte).
    """
    import numpy as np
    from ..ceemdan.decomposicao import extrair_cIMFs

    x = np.asarray(values[:n_treino, feat], dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    cIMFs = extrair_cIMFs(x, trials=trials, epsilon=epsilon,
                          max_imf=max_imf, seed=seed)

    n_imfs = cIMFs.shape[0]
    if n_imfs <= 1:
        trend, saz, resid = x.copy(), np.zeros_like(x), np.zeros_like(x)
    else:
        resid = cIMFs[0]
        saz = np.sum(cIMFs[1:-1], axis=0) if n_imfs > 2 else np.zeros_like(x)
        trend = cIMFs[-1]

    pontos: List[str] = []
    for i in range(min(n_treino, len(dt_list))):
        pontos.append(_pseudo_json_mstl(dt_list[i], trend[i], saz[i], resid[i],
                                        formato=formato, idx=i))
    return " ".join(pontos) + "."


def serie_para_texto_vmd(
    dt_list: List[datetime],
    values,
    feat: int,
    n_treino: int,
    k: int = 7,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    init: int = 0,
    trend_ma=None,
    formato=None,
) -> str:
    """Aplica VMD na série inteira [0:n_treino] e gera documento /time /t /s /r.

    init=0 (omegas em 0, deterministico); trend_ma=None = MA automatico
    ~L/7 na tendencia | 0 = sem MA | N = kernel fixo.

    formato: FormatoTexto (ou politica em str) definindo a data dos pontos;
    None = padrao 'completa' (texto historico byte a byte).
    """
    import numpy as np
    from ..vmd.decomposicao import decompor_vmd

    x_raw = np.asarray(values[:n_treino, feat], dtype=np.float64).ravel()
    try:
        trend, saz, resid = decompor_vmd(x_raw, k=k, alpha=alpha,
                                         tau=tau, tol=tol, init=init,
                                         trend_ma=trend_ma)
    except (ImportError, ModuleNotFoundError):
        raise
    except Exception:
        x = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
        trend, saz, resid = x.copy(), np.zeros_like(x), np.zeros_like(x)

    pontos: List[str] = []
    for i in range(min(n_treino, len(dt_list))):
        pontos.append(_pseudo_json_mstl(dt_list[i], trend[i], saz[i], resid[i],
                                        formato=formato, idx=i))
    return " ".join(pontos) + "."


def serie_para_texto_hibrido(
    dt_list: List[datetime],
    values,
    feat: int,
    n_treino: int,
    periodos: tuple,
    trials: int = 100,
    epsilon: float = 0.005,
    max_imf: int = 5,
    seed: int = 42,
    formato=None,
) -> str:
    """Aplica MSTL+CEEMDAN (hibrido) na série [0:n_treino].

    formato: FormatoTexto (ou politica em str) definindo a data dos pontos;
    None = padrao 'completa' (texto historico byte a byte).
    """
    import warnings
    import numpy as np
    from statsmodels.tsa.seasonal import MSTL
    from PyEMD import CEEMDAN

    x = np.asarray(values[:n_treino, feat], dtype=np.float64).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    L = len(x)
    periodos_validos = [int(p) for p in periodos if 2 <= int(p) < L // 2]
    if not periodos_validos:
        periodos_validos = [2]

    # MSTL para T/S
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MSTL(x, periods=periodos_validos, stl_kwargs={"robust": True}).fit()

    trend = np.asarray(res.trend, dtype=np.float64)
    seasonal = np.asarray(res.seasonal, dtype=np.float64)
    if seasonal.ndim == 2:
        saz_mstl = seasonal.sum(axis=1)
    else:
        saz_mstl = seasonal
    resid_mstl = np.asarray(res.resid, dtype=np.float64)

    # CEEMDAN no resíduo
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon, parallel=False)
    ceemdan.noise_seed(seed)
    cIMFs_r = ceemdan(resid_mstl, max_imf=max_imf)

    if cIMFs_r.shape[0] > 2:
        resid = np.sum(cIMFs_r[:-1], axis=0)
    else:
        resid = resid_mstl

    pontos: List[str] = []
    for i in range(min(n_treino, len(dt_list))):
        pontos.append(_pseudo_json_mstl(dt_list[i], trend[i], saz_mstl[i],
                                        resid[i], formato=formato, idx=i))
    return " ".join(pontos) + "."
