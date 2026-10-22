"""Formato do texto de entrada: variantes de escrita selecionaveis por flag.

Cinco eixos independentes, todos com o PADRAO = escrita atual do projeto
(byte a byte identica, para preservar caches e vereditos historicos):

    Pseudo-JSON (documento por feature):
        data      : completa | curta | hora | indice | nenhuma
        pj_casas  : 2 | 1 | 0       (casas decimais de /t /s /r, padrao 2)
        pj_marcacao: normal | compacta  ("/t 12.34 /s -0.56 /r 0.12" vs
                     "/t12.34/s-0.56/r0.12")
        pj_escala : float | inteiro (inteiro = valores x100 sem ponto decimal)

    Char-by-Char (frase por janela):
        casas         : 6 | 4 | 3 | 2      (padrao 6)
        encoding      : extenso | digitos  (padrao extenso)
        arredondamento: truncar | arredondar (padrao truncar; arredondar =
                        metade para longe do zero, deterministico)
        zeros         : manter | remover   (manter = casas sempre escritas,
                        padrao atual; remover = corta zeros a direita)

`FormatoTexto` e um dataclass CONGELADO: viaja como UM elemento do
`args_tuple` do joblib (pickle por referencia de classe) e nao pode ser
mutado no meio da geracao - a escrita fica deterministica.

O "formato atual" tambem fica disponivel como estado de modulo
(`formato_atual()`), porque todos os decompositores chamam o mesmo ponto
de estrangulamento (`texto_char_por_char`) e o joblib/loky executa cada
janela em processo proprio: `_processar_janela` define o formato na
entrada de cada chamada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---- Eixos e valores padrao (fonte unica: CLI + banner + ajuda) -------- #
# Pseudo-JSON
POLITICAS_DATA = ("completa", "curta", "hora", "indice", "nenhuma")
PJ_CASAS_VALIDOS = (2, 1, 0)
PJ_MARCACOES = ("normal", "compacta")
PJ_ESCALAS = ("float", "inteiro")
# Char-by-Char
DECIMAIS_VALIDOS = (6, 4, 3, 2)
ENCODINGS = ("extenso", "digitos")
ARREDONDAMENTOS = ("truncar", "arredondar")
ZEROS_VALIDOS = ("manter", "remover")

DATA_PADRAO: str = "completa"     # mes + dia + hora por extenso (atual)
PJ_CASAS_PADRAO: int = 2          # ":.2f" atual de /t /s /r
PJ_MARCACAO_PADRAO: str = "normal"    # espacos soltos (atual)
PJ_ESCALA_PADRAO: str = "float"       # ponto decimal (atual)
DECIMAIS_PADRAO: int = 6          # _MAX_DECIMAIS atual
ENCODING_PADRAO: str = "extenso"  # digitos por extenso (atual)
ARREDONDAMENTO_PADRAO: str = "truncar"  # aritmetica de inteiros atual
ZEROS_PADRAO: str = "manter"      # casas sempre escritas (atual)


@dataclass(frozen=True)
class FormatoTexto:
    """Escrita do documento de entrada (imutavel)."""

    data: str = DATA_PADRAO
    casas: int = DECIMAIS_PADRAO
    encoding: str = ENCODING_PADRAO
    pj_casas: int = PJ_CASAS_PADRAO
    pj_marcacao: str = PJ_MARCACAO_PADRAO
    pj_escala: str = PJ_ESCALA_PADRAO
    arredondamento: str = ARREDONDAMENTO_PADRAO
    zeros: str = ZEROS_PADRAO

    def __post_init__(self) -> None:
        if self.data not in POLITICAS_DATA:
            raise ValueError(
                f"politica de data invalida: {self.data!r} "
                f"(validas: {', '.join(POLITICAS_DATA)})"
            )
        if int(self.casas) not in DECIMAIS_VALIDOS:
            raise ValueError(
                f"casas decimais invalidas: {self.casas!r} "
                f"(validas: {', '.join(map(str, DECIMAIS_VALIDOS))})"
            )
        if self.encoding not in ENCODINGS:
            raise ValueError(
                f"encoding invalido: {self.encoding!r} "
                f"(validos: {', '.join(ENCODINGS)})"
            )
        if int(self.pj_casas) not in PJ_CASAS_VALIDOS:
            raise ValueError(
                f"casas decimais do PJ invalidas: {self.pj_casas!r} "
                f"(validas: {', '.join(map(str, PJ_CASAS_VALIDOS))})"
            )
        if self.pj_marcacao not in PJ_MARCACOES:
            raise ValueError(
                f"marcacao do PJ invalida: {self.pj_marcacao!r} "
                f"(validas: {', '.join(PJ_MARCACOES)})"
            )
        if self.pj_escala not in PJ_ESCALAS:
            raise ValueError(
                f"escala do PJ invalida: {self.pj_escala!r} "
                f"(validas: {', '.join(PJ_ESCALAS)})"
            )
        if self.arredondamento not in ARREDONDAMENTOS:
            raise ValueError(
                f"arredondamento invalido: {self.arredondamento!r} "
                f"(validos: {', '.join(ARREDONDAMENTOS)})"
            )
        if self.zeros not in ZEROS_VALIDOS:
            raise ValueError(
                f"politica de zeros invalida: {self.zeros!r} "
                f"(validas: {', '.join(ZEROS_VALIDOS)})"
            )
        if (self.pj_escala == "inteiro"
                and int(self.pj_casas) != PJ_CASAS_PADRAO):
            raise ValueError(
                "pj_escala='inteiro' (valores x100 sem ponto) e incompativel "
                f"com pj_casas={self.pj_casas}: use pj_casas=2 ou float"
            )

    # ---- cache ---------------------------------------------------- #
    @property
    def eh_padrao(self) -> bool:
        """True quando a escrita e exatamente a atual (sem sufixo na tag)."""
        return (
            self.data == DATA_PADRAO
            and int(self.casas) == DECIMAIS_PADRAO
            and self.encoding == ENCODING_PADRAO
            and int(self.pj_casas) == PJ_CASAS_PADRAO
            and self.pj_marcacao == PJ_MARCACAO_PADRAO
            and self.pj_escala == PJ_ESCALA_PADRAO
            and self.arredondamento == ARREDONDAMENTO_PADRAO
            and self.zeros == ZEROS_PADRAO
        )

    def tag_cache(self) -> str:
        """Sufixo combinado dos eixos: VAZIO no padrao (nomes/arquivos)."""
        partes = []
        if self.data != DATA_PADRAO:
            partes.append(f"d{self.data}")
        if int(self.casas) != DECIMAIS_PADRAO:
            partes.append(f"dec{int(self.casas)}")
        if self.encoding != ENCODING_PADRAO:
            partes.append(f"enc{self.encoding}")
        if int(self.pj_casas) != PJ_CASAS_PADRAO:
            partes.append(f"pc{int(self.pj_casas)}")
        if self.pj_marcacao != PJ_MARCACAO_PADRAO:
            partes.append(f"mj{self.pj_marcacao}")
        if self.pj_escala != PJ_ESCALA_PADRAO:
            partes.append("e100")
        if self.arredondamento != ARREDONDAMENTO_PADRAO:
            partes.append(f"ard{self.arredondamento}")
        if self.zeros != ZEROS_PADRAO:
            partes.append(f"nz{self.zeros}")
        return f"_txt{'-'.join(partes)}" if partes else ""

    # Cada cache so depende dos eixos que MUDA aquele documento (PJ usa
    # `data` + eixos pj_*; Char-by-Char usa `casas`/`encoding` +
    # arredondamento/zeros - ver docstring do modulo). Acoplar os eixos
    # faria o runner regenerar 650 MB de CbC sem necessidade.
    def tag_cache_pj(self) -> str:
        """Sufixo do cache do Pseudo-JSON: SO os eixos que mudam esse texto."""
        partes = []
        if self.data != DATA_PADRAO:
            partes.append(f"d{self.data}")
        if int(self.pj_casas) != PJ_CASAS_PADRAO:
            partes.append(f"pc{int(self.pj_casas)}")
        if self.pj_marcacao != PJ_MARCACAO_PADRAO:
            partes.append(f"mj{self.pj_marcacao}")
        if self.pj_escala != PJ_ESCALA_PADRAO:
            partes.append("e100")
        return f"_txt{'-'.join(partes)}" if partes else ""

    def tag_cache_cb(self) -> str:
        """Sufixo do cache do Char-by-Char: SO os eixos que mudam esse texto."""
        partes = []
        if int(self.casas) != DECIMAIS_PADRAO:
            partes.append(f"dec{int(self.casas)}")
        if self.encoding != ENCODING_PADRAO:
            partes.append(f"enc{self.encoding}")
        if self.arredondamento != ARREDONDAMENTO_PADRAO:
            partes.append(f"ard{self.arredondamento}")
        if self.zeros != ZEROS_PADRAO:
            partes.append(f"nz{self.zeros}")
        return f"_txt{'-'.join(partes)}" if partes else ""

    # ---- relatorios ------------------------------------------------ #
    def rotulo(self) -> str:
        """Linha curta para o banner/JSON (mostra o que saiu do padrao)."""
        if self.eh_padrao:
            return ("padrao (data=completa, decimais=6, encoding=extenso, "
                    "pj_casas=2, marcacao=normal, escala=float, "
                    "arredondamento=truncar, zeros=manter)")
        partes = []
        if self.data != DATA_PADRAO:
            partes.append(f"data={self.data}")
        if int(self.casas) != DECIMAIS_PADRAO:
            partes.append(f"decimais={int(self.casas)}")
        if self.encoding != ENCODING_PADRAO:
            partes.append(f"encoding={self.encoding}")
        if int(self.pj_casas) != PJ_CASAS_PADRAO:
            partes.append(f"pj_casas={int(self.pj_casas)}")
        if self.pj_marcacao != PJ_MARCACAO_PADRAO:
            partes.append(f"pj_marcacao={self.pj_marcacao}")
        if self.pj_escala != PJ_ESCALA_PADRAO:
            partes.append(f"pj_escala={self.pj_escala}")
        if self.arredondamento != ARREDONDAMENTO_PADRAO:
            partes.append(f"arredondamento={self.arredondamento}")
        if self.zeros != ZEROS_PADRAO:
            partes.append(f"zeros={self.zeros}")
        return ", ".join(partes)

    def fora_do_padrao(self) -> list:
        """Quais eixos sairam do padrao (para o banner marcar '<-- FORA')."""
        fora = []
        if self.data != DATA_PADRAO:
            fora.append("data")
        if int(self.casas) != DECIMAIS_PADRAO:
            fora.append("decimais")
        if self.encoding != ENCODING_PADRAO:
            fora.append("encoding")
        if int(self.pj_casas) != PJ_CASAS_PADRAO:
            fora.append("pj_decimais")
        if self.pj_marcacao != PJ_MARCACAO_PADRAO:
            fora.append("pj_marcacao")
        if self.pj_escala != PJ_ESCALA_PADRAO:
            fora.append("pj_escala")
        if self.arredondamento != ARREDONDAMENTO_PADRAO:
            fora.append("arredondamento")
        if self.zeros != ZEROS_PADRAO:
            fora.append("zeros")
        return fora


FORMATO_PADRAO: FormatoTexto = FormatoTexto()


def resolver_formato(formato: Optional[FormatoTexto] = None) -> FormatoTexto:
    """Aceita FormatoTexto, None (= estado de modulo) ou dict de campos."""
    if formato is None:
        return _ATUAL
    if isinstance(formato, FormatoTexto):
        return formato
    if isinstance(formato, dict):
        return FormatoTexto(**formato)
    raise TypeError(f"formato invalido: {type(formato)!r}")


# ---- estado de modulo (usado pelo ponto de estrangulamento) ----------- #
_ATUAL: FormatoTexto = FORMATO_PADRAO


def definir_formato(formato: Optional[FormatoTexto] = None) -> FormatoTexto:
    """Fixa o formato do processo (chamado por _processar_janela por janela)."""
    global _ATUAL
    _ATUAL = resolver_formato(formato)
    return _ATUAL


def formato_atual() -> FormatoTexto:
    return _ATUAL
