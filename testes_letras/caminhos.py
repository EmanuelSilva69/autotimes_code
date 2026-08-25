"""
caminhos - Rotas absolutas do projeto + descoberta do tokenizador local.

Centraliza todos os caminhos usados pelo testador para que nenhum script
dependa do diretorio corrente:
    - testes_letras/          (este testador)
    - Testes_eficiencia/      (modulos de melhoria + checkpoints do ablation)
    - gpt2/                   (GPT-2 base local: pesos + tokenizer)
    - dataset/ETT-small/      (CSV numerico ETTh1)
    - resultados/             (JSON, logs e graficos gerados)

O tokenizador e carregado de forma EXPLICITA do caminho pedido
(E:\\2026.1\\Viplab\\AutoTimes\\Testes_eficiencia); se la nao houver
tokenizer.json, a busca recai para Testes_eficiencia\\gpt2 e, por ultimo,
para gpt2/ na raiz do projeto (onde os arquivos tokenizer.json + merges.txt
realmente estao salvos).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------- #
#  Rotas                                                               #
# --------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = BASE_DIR.parent
TESTES_EF_DIR: Path = ROOT_DIR / "Testes_eficiencia"
GPT2_DIR: Path = ROOT_DIR / "gpt2"
DATA_DIR: Path = ROOT_DIR / "dataset" / "ETT-small"
DATA_PATH: str = "ETTh1.csv"

RESULTADOS_DIR: Path = BASE_DIR / "resultados"
GRAFICOS_DIR: Path = RESULTADOS_DIR / "graficos"
LOG_DIR: Path = RESULTADOS_DIR / "logs"

# Checkpoints do Ablation Study (Testes_eficiencia/checkpoints):
#   - smetimes: Run-8 VolLoss + SMETimes + otimizacoes AMD (todas as
#     melhorias, inclui o AdaptiveFusion/porta de volatilidade).
#   - amd:      Run-0 Padrao + otimizacoes AMD (somente o baseline).
CKPT_SMETIMES: Path = (
    TESTES_EF_DIR / "checkpoints" / "Run-8_VolLoss+SMETimes__ETTh1__20260809_131904.pt"
)
CKPT_AMD: Path = (
    TESTES_EF_DIR / "checkpoints" / "Run-0_Padrao__ETTh1__20260809_120942.pt"
)

# Caminho do tokenizador pedido explicitamente pelo enunciado.
TOKENIZADOR_PEDIDO: str = r"E:\2026.1\Viplab\AutoTimes\Testes_eficiencia"


# --------------------------------------------------------------------- #
#  sys.path (raiz + Testes_eficiencia para importar run_benchmarks)     #
# --------------------------------------------------------------------- #
def garantir_syspath() -> None:
    """Garante Testes_eficiencia/, raiz do projeto e testes_letras no path.

    Necessario para `import run_benchmarks`, `import modules` e
    `import utils.metrics` independentemente de onde o script for lancado.
    """
    for p in (str(TESTES_EF_DIR), str(ROOT_DIR), str(BASE_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def importar_run_benchmarks():
    """Importa (1x) o run_benchmarks.py do Testes_eficiencia.

    Reutiliza EXATAMENTE o mesmo codigo de modelo/dados do ablation
    (AutoTimesModel, AdaptiveFusion/SMETimes, splits, seed) para garantir
    fidelidade aos checkpoints treinados.
    """
    garantir_syspath()
    import run_benchmarks  # noqa: F401

    return run_benchmarks


# --------------------------------------------------------------------- #
#  Tokenizador local                                                    #
# --------------------------------------------------------------------- #
def descobrir_tokenizador(caminho_explicito: Optional[str] = None) -> Path:
    """Encontra o diretorio com o tokenizer.json do GPT-2 local.

    Ordem de busca (primeiro que contiver tokenizer.json vence):
      1) caminho_explicito (default = Testes_eficiencia, como pedido);
      2) Testes_eficiencia\\gpt2;
      3) gpt2/ na raiz do projeto.

    Args:
        caminho_explicito: override manual (ex: 'E:\\\\...\\\\gpt2').

    Returns:
        Path do diretorio do tokenizador.

    Raises:
        FileNotFoundError: nenhum candidato contem tokenizer.json.
    """
    candidatos: list = [
        Path(caminho_explicito or TOKENIZADOR_PEDIDO),
        TESTES_EF_DIR / "gpt2",
        GPT2_DIR,
    ]
    for c in candidatos:
        if (c / "tokenizer.json").exists():
            return c
    raise FileNotFoundError(
        "Tokenizador local nao encontrado. Procurei em: "
        + "; ".join(str(c) for c in candidatos)
    )


# Pastas que nunca contem o tokenizador (puladas na busca recursiva).
_DIRS_A_IGNORAR = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
        "node_modules", "site-packages", "dist", "build", ".idea", ".vscode",
        "resultados", "text_cache",
    }
)


def descobrir_tokenizador_recursivo(
    raiz: Optional[Path] = None, profundidade_max: int = 8
) -> Path:
    """Busca recursiva pelo diretorio do tokenizador GPT-2.

    Varre a arvore a partir de `raiz` (default: raiz do projeto) procurando
    pastas que contenham os arquivos do tokenizador (vocab.json, merges.txt
    ou tokenizer.json), sem depender de caminhos "hardcoded".

    Ordem de prioridade dos candidatos encontrados:
        1) tokenizer.json (fast tokenizer completo e inequivoco);
        2) vocab.json + merges.txt (BPE classico, ex.: diretorio do OPT,
           que tem a mesma estrutura porem vocabulario diferente - so e
           escolhido se nao existir nenhum fast tokenizer);
        e, em empate, o diretorio mais raso e o nome alfabetico menor.

    Args:
        raiz: diretorio inicial da varredura (default ROOT_DIR).
        profundidade_max: limite de subniveis percorridos (evita varreduras
            interminaveis em dependencias aninhadas).

    Returns:
        Path do diretorio que contem os arquivos do tokenizador.

    Raises:
        FileNotFoundError: nenhum diretorio com arquivos do tokenizador.
    """
    raiz = Path(raiz or ROOT_DIR).resolve()
    if not raiz.is_dir():
        raise FileNotFoundError(f"Raiz de busca do tokenizador inexistente: {raiz}")

    achados: list = []
    for dirpath, dirnames, filenames in os.walk(raiz):
        dirnames[:] = sorted(d for d in dirnames if d not in _DIRS_A_IGNORAR)
        rel = Path(dirpath).relative_to(raiz)
        if len(rel.parts) > profundidade_max:
            dirnames[:] = []  # nao desce mais fundo
            continue
        nome = str(Path(dirpath)).lower()
        prof = len(rel.parts)
        if "tokenizer.json" in filenames:
            achados.append((0, prof, nome, Path(dirpath)))  # fast tokenizer
        elif "vocab.json" in filenames and "merges.txt" in filenames:
            achados.append((1, prof, nome, Path(dirpath)))  # BPE classico (ex. OPT)

    if not achados:
        raise FileNotFoundError(
            f"Nenhum diretorio com tokenizador (vocab.json/merges.txt/"
            f"tokenizer.json) encontrado sob {raiz}."
        )
    achados.sort(key=lambda t: (t[0], t[1], t[2]))
    return achados[0][3]


def criar_resultados() -> None:
    """Cria os diretorios de saida (resultados/, graficos/, logs/)."""
    for d in (RESULTADOS_DIR, GRAFICOS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
