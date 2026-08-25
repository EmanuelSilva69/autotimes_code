"""
paginacao - Testador de Paginacao (Chunking Strategies) para textos puros.

Compara 3 estrategias para caber um documento longo no contexto do GPT-2
(block_max tokens por pagina, n_paginas_max paginas por janela):

    1) truncamento  - Hard Truncation: corta no limite exato, descartando
                      o excedente (1 pagina, perda de informacao);
    2) janela       - Sliding Window c/ sobreposicao (ex: 15%): pagina
                      todo o texto sem perder continuidade semantica entre
                      paginas (n paginas, zero descarte);
    3) keywords     - Compressao por palavras-chave: remove stopwords
                      (artigos/preposicoes) ANTES de tokenizar e pagina o
                      texto filtrado (menos tokens, perda de conectivos).

Saida uniforme por estrategia (dict):
    paginas              : lista de listas de token-ids (cada uma <= block_max)
    n_paginas            : paginas geradas
    n_paginas_para_tudo  : paginas teoricas para cobrir o texto inteiro
    tokens_originais     : tokens do documento original
    tokens_retidos       : tokens efetivamente paginados (soma nas paginas)
    tokens_descartados   : tokens perdidos pela estrategia
    sobreposicao_frac    : fracao de overlap (apenas janela)
    tempo_tokenizacao_s  : tempo gasto (texto -> paginas)
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from typing import Dict, List

from transformers import PreTrainedTokenizer

# --------------------------------------------------------------------- #
#  Stopwords (artigos, preposicoes, conjuncoes...) para o filtro        #
# --------------------------------------------------------------------- #
STOPWORDS_PT = frozenset(
    """
    a o os as de do da dos das em no na nos nas para por com sem se que e ou
    mas um uma uns umas ao aos à às pelo pela pelos pelas sobre sob entre
    até desde depois antes como quando onde qual quais quem cujo cuja cujos
    cujas num numa neste nesta nestes nestas também já não mais muito
    """.split()
)
STOPWORDS_EN = frozenset(
    """
    a an the of to in on at for by with and or but if then so from up down
    out over under again further once here there when where why how all any
    both each few more most other some such only own same than too very just
    be is are was were been being do does did doing have has had having
    about into through during before after above below between
    """.split()
)
STOPWORDS_ES = frozenset(
    """
    el la los las de del al a en y o pero ni que como cuando donde quien
    cuales para por con sin un una unos unas entre sobre desde hasta
    durante segun tras ante bajo es fue era son eran estaba estaban ser
    haber ha han habia tiene tienen no si mas muy ya tambien solo aunque
    porque antes despues luego
    """.split()
)

# Estrategias disponiveis e seus rotulos.
ESTRATEGIAS: Dict[str, Dict] = {
    "truncamento": {
        "nome": "Truncamento Direto",
        "desc": "Hard truncation: corta no limite de tokens por bloco.",
    },
    "janela": {
        "nome": "Janela Deslizante + Overlap",
        "desc": "Sliding window com sobreposicao para nao quebrar a continuidade.",
    },
    "keywords": {
        "nome": "Compressao por Palavras-Chave",
        "desc": "Remove stopwords antes de tokenizar (carga semantica principal).",
    },
}


def _normalizar(palavra: str) -> str:
    """Remove acentos e pontuacao para casar com a lista de stopwords."""
    sem_acentos = unicodedata.normalize("NFKD", palavra).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]", "", sem_acentos).lower()


class TestadorPaginacao:
    """Pagina documentos longos e mede o custo de cada estrategia."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        block_max: int = 128,
        overlap_frac: float = 0.15,
        n_paginas_max: int = 7,
    ) -> None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
        self.tokenizer = tokenizer
        self.block_max = int(block_max)
        self.overlap_frac = float(overlap_frac)
        self.n_paginas_max = int(n_paginas_max)
        self.pad = int(tokenizer.pad_token_id)

    # ---- tokenizacao -------------------------------------------------- #
    def _ids(self, texto: str) -> List[int]:
        return self.tokenizer(texto, add_special_tokens=False)["input_ids"]

    # ---- estrategia 1: hard truncation -------------------------------- #
    def truncamento_direto(self, texto: str) -> Dict:
        ids = self._ids(texto)
        n_tudo = max(1, math.ceil(len(ids) / self.block_max))
        pagina = ids[: self.block_max]
        return {
            "paginas": [pagina] if pagina else [],
            "n_paginas": 1 if pagina else 0,
            "n_paginas_para_tudo": n_tudo,
            "tokens_originais": len(ids),
            "tokens_retidos": len(pagina),
            "tokens_descartados": max(len(ids) - self.block_max, 0),
            "sobreposicao_frac": 0.0,
            "tempo_tokenizacao_s": 0.0,
        }

    # ---- estrategia 2: sliding window com overlap --------------------- #
    def janela_deslizante(self, texto: str) -> Dict:
        ids = self._ids(texto)
        s, o = self.block_max, int(round(self.block_max * self.overlap_frac))
        passo = max(s - o, 1)
        paginas: List[List[int]] = []
        if ids:
            for i in range(0, len(ids), passo):
                paginas.append(ids[i:i + s])
                if i + s >= len(ids):
                    break
        n_paginas = len(paginas)
        n_tudo = (
            max(1, math.ceil((len(ids) - s) / passo) + 1) if len(ids) > s else max(1, n_paginas)
        )
        return {
            "paginas": paginas,
            "n_paginas": n_paginas,
            "n_paginas_para_tudo": n_tudo,
            "tokens_originais": len(ids),
            "tokens_retidos": sum(len(p) for p in paginas),
            "tokens_descartados": 0,
            "sobreposicao_frac": self.overlap_frac,
            "tempo_tokenizacao_s": 0.0,
        }

    # ---- estrategia 3: compressao por palavras-chave ------------------ #
    def compressao_keywords(self, texto: str, lingua: str = "pt") -> Dict:
        ids_orig = self._ids(texto)
        if lingua.lower().startswith("pt"):
            sw = STOPWORDS_PT
        elif lingua.lower().startswith("es"):
            sw = STOPWORDS_ES
        else:
            sw = STOPWORDS_EN
        mantidas: List[str] = []
        removidas = 0
        for palavra in texto.split():
            if _normalizar(palavra) in sw:
                removidas += 1
            else:
                mantidas.append(palavra)
        filtrado = " ".join(mantidas)
        ids = self._ids(filtrado)

        paginas: List[List[int]] = []
        if ids:
            for i in range(0, len(ids), self.block_max):
                paginas.append(ids[i:i + self.block_max])
        return {
            "paginas": paginas,
            "n_paginas": len(paginas),
            "n_paginas_para_tudo": max(1, len(paginas)),
            "tokens_originais": len(ids_orig),
            "tokens_retidos": sum(len(p) for p in paginas),
            "tokens_descartados": len(ids_orig) - len(ids),
            "sobreposicao_frac": 0.0,
            "tempo_tokenizacao_s": 0.0,
            "stopwords_removidas": removidas,
        }

    # ---- dispatcher + cronometragem ----------------------------------- #
    def paginar(self, estrategia: str, texto: str, lingua: str = "pt") -> Dict:
        """Aplica a estrategia e mede o tempo de tokenizacao+paginacao."""
        t0 = time.perf_counter()
        if estrategia == "truncamento":
            res = self.truncamento_direto(texto)
        elif estrategia == "janela":
            res = self.janela_deslizante(texto)
        elif estrategia == "keywords":
            res = self.compressao_keywords(texto, lingua)
        else:
            raise ValueError(f"Estrategia desconhecida: {estrategia}. "
                             f"Validas: {list(ESTRATEGIAS)}")
        res["estrategia"] = estrategia
        res["tempo_tokenizacao_s"] = round(time.perf_counter() - t0, 4)
        return res

    # ---- adaptacao para o modelo (ids + lens padronizados) ------------ #
    def para_modelo(self, resultado: Dict) -> Dict:
        """Converte paginas em ids achatados (n_max*block_max) + lens.

        Preenche blocos vazios com pad (lens=0 -> pooling zera a
        contribuicao de texto) e descarta paginas alem do limite do modelo.

        Returns:
            {"ids": List[int], "lens": List[int], "paginas_descartadas": int}
        """
        paginas = resultado.get("paginas", [])[: self.n_paginas_max]
        descartadas = max(resultado.get("n_paginas", 0) - len(paginas), 0)
        ids_flat: List[int] = []
        lens: List[int] = []
        for p in paginas:
            p = p[: self.block_max]
            lens.append(len(p))
            ids_flat.extend(p)
            ids_flat.extend([self.pad] * (self.block_max - len(p)))
        while len(lens) < self.n_paginas_max:
            lens.append(0)
            ids_flat.extend([self.pad] * self.block_max)
        return {
            "ids": ids_flat,
            "lens": lens,
            "paginas_descartadas": descartadas,
        }
