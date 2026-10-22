"""
run_apples_to_apples_trio - CONFRONTO FINAL "Apples-to-Apples" em Z-SCORE.

Compara as 3 abordagens sob o PROTOCOLO OFICIAL do AutoTimes (z-score nativo
global ajustado ESTRITAMENTE no treino) com splits canonicos de linhas e
CHANNEL INDEPENDENCE (as 7 features do ETTh1 empilhadas univariadas):

    borders EXATOS do Dataset_ETT_hour (run_benchmarks._compute_split_bounds):
        b1s = [0, 7968, 10848]  ->  b2s = [8640, 11520, 14400]
        (val/test comecam 672 linhas antes do corte do split, como o original)
    janelas por feat: train 7.873 | val 2.785 | test 2.785 (stride 1)
    TOTAL: 7 features x 12.443 = 94.101 janelas (55111 / 19495 / 19495)
    = EXATAMENTE o volume do baseline numerico 0.3539 (que e a media
    multivariada das 7 colunas, nao o erro isolado do HUFL).

OS 3 MODELOS:
    1) Baseline Numerico (Run-8 VolLoss+SMETimes): puramente matematico.
       0.3539 ja cravado (resultados/numerico_base_stats.json) -> carregado
       do log; --re-numerico forca uma re-medicao eval-only nas janelas
       canonicas de HUFL (mesmo harness rb.evaluate, last_only=True).
    2) Pseudo-JSON ES (campeao textual, paginacao janela c/ overlap):
       UM documento por feature (template pseudo_json_es sobre as linhas de
       treino [0, 8640) da COLUNA daquela feature, em z-score), paginado
       (7 blocos x 128), repetido nas janelas da propria feature.
    3) Char-by-Char (serializacao digito a digito com max_abs local):
       documento POR JANELA gerado a partir da janela em z-score da sua
       feature (encode aplica MaxAbs local); contrato de 5 tensores
       (x, ids, lens, y, max_abs).

RIGOR:
    - 3 seeds (2021, 42, 1234) x 10 epocas, restore_best por validacao,
      batch 16, lr 1e-3, isolamento estrito (montar_modelo DENTRO do seed);
    - metrica SEMPRE no espaco Z-SCORE (o modelo devolve a reconstrucao na
      escala de entrada z; o RevIN interno ja denormalizou);
    - cache de tensores no disco (quinta_noite/cache_trio/): o pool z-score
      e os textos/paginacao sao gerados UMA vez e reutilizados nas 3 seeds
      e nas re-execucoes.

ATENCAO - DECODE do char-by-char (--decode):
    O AutoTimesModel textual ja devolve a reconstrucao denormalizada pelo
    RevIN interno para a escala de ENTRADA (aqui: z-score). O encode() do
    char-by-char normaliza o TEXTO pelo max_abs local, mas a ESCALA DA SAIDA
    e definida pelo ramo numerico (RevIN), nao pelo texto. Aplicar decode()
    (pred * max_abs) deslocaria as predicoes z para fora do espaco alvo e
    INFLARIA o MSE por ~max_abs^2 (fator ~4-16x). O decode vem DESLIGADO por
    padrao (flag --decode liga apenas se voce quiser medir essa variante).

SAIDAS (testes_letras/quinta_noite/):
    veredito_trio.json              (consolidado das 3 abordagens)
    convergencia_trio.png           (Val MSE z por epoca + bandas + linha 0.3539)
    barras_veredito_trio.png        (Test MSE z final, error bars pretas)

USO:
    python -m testes_letras.run_apples_to_apples_trio
    python -m testes_letras.run_apples_to_apples_trio --epochs 1 --seeds 2021
    python -m testes_letras.run_apples_to_apples_trio --charts-only
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suprime warnings de SDPA experimental no AMD/ROCm
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import caminhos
from .dados import LABEL_LEN, PRED_LEN, SEQ_LEN
from .modelos import CKPTS, carregar_checkpoint, montar_modelo
from .modelos_dual import montar_modelo_dual
from .run_etth1_texto import _FEAT_PADRAO, carregar_etth1
from .texto_char_by_char import (
    BLOCK_MAX,
    N_PAGINAS_MAX,
    OVERLAP,
    CharByCharPipeline,
    construir_dataset_char_by_char,
    detectar_periodos_mstl,
)
from .texto_puro import estruturado
from .texto_puro.formato import (
    ARREDONDAMENTO_PADRAO,
    ARREDONDAMENTOS,
    DATA_PADRAO,
    DECIMAIS_PADRAO,
    DECIMAIS_VALIDOS,
    ENCODINGS,
    ENCODING_PADRAO,
    PJ_CASAS_PADRAO,
    PJ_CASAS_VALIDOS,
    PJ_ESCALA_PADRAO,
    PJ_ESCALAS,
    PJ_MARCACAO_PADRAO,
    PJ_MARCACOES,
    POLITICAS_DATA,
    ZEROS_PADRAO,
    ZEROS_VALIDOS,
    FormatoTexto,
    definir_formato,
)
from .texto_puro.paginacao import TestadorPaginacao
from .treino import build_loaders, rodar_convergencia

# Splits canonicos de LINHAS (borders EXATOS do Dataset_ETT_hour: o val/test
# comeca SEQ_LEN linhas antes do corte, como no data_provider original).
N_TREINO: int = 8640
N_VAL: int = 11520
N_TESTE: int = 14400
B1S: Tuple[int, int, int] = (0, N_TREINO - SEQ_LEN, N_VAL - SEQ_LEN)
B2S: Tuple[int, int, int] = (N_TREINO, N_VAL, N_TESTE)

N_FEATS: int = 7  # HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (Channel Independence)

TEMPLATE_PSEUDO_JSON: str = "pseudo_json_es"
ESTRATEGIA_PAGINACAO: str = "janela"

OUT_DIR: Path = caminhos.BASE_DIR / "quinta_noite"
CACHE_DIR: Path = OUT_DIR / "cache_trio"
CKPT_DIR: Path = OUT_DIR / "checkpoints"
JSON_OUT: Path = OUT_DIR / "veredito_trio.json"
JSON_NUMERICO_LOG: Path = caminhos.RESULTADOS_DIR / "numerico_base_stats.json"
RECENTE_DIR: Path = caminhos.BASE_DIR / "recente"

_CORES: Dict[str, str] = {
    "numerico": "#7f7f7f",
    "pseudo_json_es": "#9467bd",
    "char_by_char": "#d62728",
}

# ---- Defaults do VMD (fonte unica: argparse + banner + ajuda) --------- #
VMD_K_PADRAO: int = 7
VMD_ALPHA_PADRAO: float = 2000.0
VMD_TAU_PADRAO: float = 0.0
VMD_TOL_PADRAO: float = 1e-7
VMD_INIT_PADRAO: int = 0
VMD_MA_PADRAO: int = -1  # -1 = MA automatico ~L/7

# O que o screen espectral mostrou (20 janelas ETTh1, L=672, ref CEEMDAN):
#   S% = % da energia diaria+semanal (24h/168h) que fica no componente
SCREEN_VMD_K: str = ("S%: K=7 97.9 (154 ms/janela, padrao) | K=8 97.9 (246) | "
                     "K=10 98.0 (328) | K=12 98.7 (405) | K=5 96.0 (rodada "
                     "antiga, PJ 0.3615) | K=3 fraco; CEEMDAN=97.6")
SCREEN_VMD_ALPHA: str = ("com init=0 (padrao) 1000..5000 tudo bom (2000=97.87, "
                         "3000=97.88, 5000=98.75); <=500 piora (S%=95.3, "
                         "T%=3.3, R%=1.4); com init=1 alpha>=3000 COLAPSA "
                         "(S%=87+-25)")


def _pct(texto: str) -> str:
    """Escapa '%' para strings de help do argparse (usa '%s' como placeholder)."""
    return texto.replace("%", "%%")


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confronto final Apples-to-Apples (z-score canonicas): "
                    "Numerico vs Pseudo-JSON ES vs Char-by-Char"
    )
    parser.add_argument("--checkpoints", type=str, default="smetimes")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="2021,42,1234")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=BLOCK_MAX)
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int, default=N_PAGINAS_MAX)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument("--linhas", type=int, default=N_TESTE,
                        help="Linhas do CSV a carregar (>= 14400 p/ splits canonicos)")
    parser.add_argument("--feats", type=str, default="all",
                        help="Features do ETTh1 (indices 0-6 separados por "
                             "virgula, ex: '0,1') ou 'all' (7 colunas)")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--decode", action="store_true",
                        help="APLICA decode(max_abs) nas metricas do char-by-char "
                             "(INFLARIA o MSE por ~max_abs^2; ver docstring)")
    parser.add_argument("--re-numerico", dest="re_numerico", action="store_true",
                        help="Re-mede o baseline numerico (eval-only, HUFL canonico) "
                             "em vez de carregar 0.3539 do log")
    parser.add_argument("--charts-only", dest="charts_only", action="store_true",
                        help="Somente regenera os graficos a partir do JSON existente")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--force-cache", dest="force_cache", action="store_true",
                        help="Regenera os tensores cacheados (pool/textos)")
    parser.add_argument("--variante-arquitetura", dest="variante",
                        choices=["original", "dual"], default="original",
                        help="Arquitetura do modelo numerico: 'original' "
                             "(AutoTimesModel) ou 'dual' (AutoTimesDualModel "
                             "com trend+osc)")
    parser.add_argument("--dual-fusao", dest="dual_fusao",
                        choices=["glu", "soma"], default="glu",
                        help="Combinacao trend/osc da variante dual: "
                             "'glu' (gate enviesado, +1.18M params) ou "
                             "'soma' (adicao simples, params = original). "
                             "Ignorado se --variante-arquitetura original.")
    parser.add_argument("--decomp-mode", dest="decomp_mode",
                        choices=["none", "stl_simple"], default="none",
                        help="Extrator de tendencia na variante dual: "
                             "'none' (regressao linear por janela) ou "
                             "'stl_simple' (media movel k=25, padding "
                             "replicado). Ignorado se variante original.")
    parser.add_argument("--texto-decomp", dest="texto_decomp",
                        choices=["none", "stl", "mstl", "ceemdan",
                                 "mstl+ceemdan", "paralelo", "vmd"],
                        default="none",
                        help="Decomposicao TEXTUAL no pre-processamento: "
                             "'stl' = STL LOESS periodo 24; "
                             "'mstl' = MSTL com periodos dinamicos; "
                             "'ceemdan' = CEEMDAN puro (adaptativo); "
                             "'mstl+ceemdan' = MSTL p/ T/S + CEEMDAN no R; "
                             "'paralelo' = MSTL p/ T + CEEMDAN p/ S; "
                             "'vmd' = VMD (K modos agrupados T/S/R, FFT, "
                             "dezenas de ms por janela).")
    parser.add_argument("--mstl-periods", dest="mstl_periods",
                        type=str, default=None,
                        help="Periodos customizados para MSTL (override da "
                             "detecao automatica). Ex: '12,24,168'.")
    parser.add_argument("--ceemdan-trials", dest="ceemdan_trials",
                        type=int, default=100,
                        help="Numero de trials CEEMDAN (default: 100). "
                             "Mais trials = decomposicao mais estavel.")
    parser.add_argument("--ceemdan-epsilon", dest="ceemdan_epsilon",
                        type=float, default=0.005,
                        help="Amplitude do ruido adaptativo CEEMDAN "
                             "(default: 0.005).")
    parser.add_argument("--ceemdan-seed", dest="ceemdan_seed",
                        type=int, default=42,
                        help="Semente para reprodutibilidade do CEEMDAN "
                             "(default: 42).")
    parser.add_argument("--ceemdan-max-imf", dest="ceemdan_max_imf",
                        type=int, default=5,
                        help="Numero maximo de IMFs CEEMDAN (default: 5). "
                             "-1 = automatico.")
    parser.add_argument("--ceemdan-backend", dest="ceemdan_backend",
                        choices=("pyemd", "hip"), default="pyemd",
                        help="Backend da decomposicao CEEMDAN: 'pyemd' "
                             "(CPU, default) ou 'hip' (GPU AMD via DLL "
                             "CEEMDAN-GPU, char-by-char serial).")
    parser.add_argument("--vmd-k", dest="vmd_k", type=int,
                        default=VMD_K_PADRAO,
                        help=f"Numero de modos VMD (padrao: {VMD_K_PADRAO}; "
                             f"K=5 = max_imf do CEEMDAN; modos agrupados em "
                             f"T/S/R). Screen: {_pct(SCREEN_VMD_K)}.")
    parser.add_argument("--vmd-alpha", dest="vmd_alpha", type=float,
                        default=VMD_ALPHA_PADRAO,
                        help=f"Restricao de banda do VMD (padrao: "
                             f"{VMD_ALPHA_PADRAO:g}, balanco fidelidade x "
                             f"banda). Screen: {_pct(SCREEN_VMD_ALPHA)}.")
    parser.add_argument("--vmd-tau", dest="vmd_tau", type=float,
                        default=VMD_TAU_PADRAO,
                        help=f"Passo de subida dual do VMD (padrao: "
                             f"{VMD_TAU_PADRAO:g} = tolerancia a ruido; "
                             f"tau>0 enrigeca o trend no screen).")
    parser.add_argument("--vmd-tol", dest="vmd_tol", type=float,
                        default=VMD_TOL_PADRAO,
                        help=f"Criterio de convergencia do VMD (padrao: "
                             f"{VMD_TOL_PADRAO:g}, toolbox oficial).")
    parser.add_argument("--vmd-init", dest="vmd_init", type=int,
                        choices=(0, 1, 2), default=VMD_INIT_PADRAO,
                        help=f"Inicializacao das frequencias do VMD "
                             f"(padrao: {VMD_INIT_PADRAO} = omegas todos em 0, "
                             f"melhor separacao T/S, deterministico), 1 = "
                             f"omegas uniformes (S%% ~2.8 pontos pior e "
                             f"alpha>=3000 colapsa), 2 = aleatorio (NAO usar: "
                             f"quebra o cache).")
    parser.add_argument("--vmd-ma", dest="vmd_ma", type=int,
                        default=VMD_MA_PADRAO,
                        help=f"Media movel na tendencia do VMD (padrao: "
                             f"{VMD_MA_PADRAO} = automatico ~L/7), 0 = sem "
                             f"suavizacao (trend 20x mais rugoso), N = kernel "
                             f"fixo impar.")
    parser.add_argument("--texto-data", dest="texto_data",
                        choices=list(POLITICAS_DATA), default=DATA_PADRAO,
                        help="Politica de data no Pseudo-JSON (carimbo /time "
                             "de todos os pontos): 'completa' = mes + dia + "
                             f"hora por extenso, 'curta' = dia+hora, 'hora' = "
                             "so a hora, 'indice' = /i N, 'nenhuma' = sem "
                             "data (o ponto comeca em /t -> mais horas "
                             "dentro dos 896 tokens). "
                             f"[padrao: {DATA_PADRAO}]")
    parser.add_argument("--texto-decimais", dest="texto_decimais",
                        type=int, choices=list(DECIMAIS_VALIDOS),
                        default=DECIMAIS_PADRAO,
                        help=_pct(
                            "Casas decimais por extenso no Char-by-Char "
                            f"(6, 4, 3 ou 2). [padrao: {DECIMAIS_PADRAO}] "
                            "3 = ~37% menos tokens por ponto -> mais pontos "
                            "visiveis. So muda o TEXTO: paginacao 3+2+2, "
                            "orçamento e treino continuam os mesmos."))
    parser.add_argument("--texto-encoding", dest="texto_encoding",
                        choices=list(ENCODINGS), default=ENCODING_PADRAO,
                        help="Escrita dos valores numericos no Char-by-Char: "
                             "'extenso' = por extenso (palavras) ou "
                             "'digitos' = algarismos crus; normalizacao "
                             f"local e max_abs inalterados. [padrao: {ENCODING_PADRAO}]")
    parser.add_argument("--texto-pj-casas", dest="texto_pj_casas",
                        type=int, choices=list(PJ_CASAS_VALIDOS),
                        default=PJ_CASAS_PADRAO,
                        help="Casas decimais de /t /s /r no Pseudo-JSON "
                             f"(2, 1 ou 0). [padrao: {PJ_CASAS_PADRAO}] "
                             "1 = menos tokens por ponto (mais horas nos "
                             "896); so muda o TEXTO do PJ.")
    parser.add_argument("--texto-pj-marcacao", dest="texto_pj_marcacao",
                        choices=list(PJ_MARCACOES), default=PJ_MARCACAO_PADRAO,
                        help="Marcadores do Pseudo-JSON: 'normal' = "
                             "'/t 12.34 /s -0.56 /r 0.12' com espacos "
                             "soltos ou 'compacta' = '/t12.34/s-0.56/r0.12'. "
                             f"[padrao: {PJ_MARCACAO_PADRAO}]")
    parser.add_argument("--texto-pj-escala", dest="texto_pj_escala",
                        choices=list(PJ_ESCALAS), default=PJ_ESCALA_PADRAO,
                        help="Escala dos valores do PJ: 'float' = ponto "
                             "decimal (atual) ou 'inteiro' = valores x100 "
                             "como inteiro ('/t 1234 /s -56 /r 12', sem "
                             f"token de ponto). [padrao: {PJ_ESCALA_PADRAO}]")
    parser.add_argument("--texto-arredondamento", dest="texto_arredondamento",
                        choices=list(ARREDONDAMENTOS),
                        default=ARREDONDAMENTO_PADRAO,
                        help="Como os decimais do Char-by-Char chegam ao "
                             "texto: 'truncar' = corta a cauda (historico) "
                             "ou 'arredondar' = metade para longe do zero "
                             f"(elimina o vies do truncamento). "
                             f"[padrao: {ARREDONDAMENTO_PADRAO}]")
    parser.add_argument("--texto-zeros", dest="texto_zeros",
                        choices=list(ZEROS_VALIDOS), default=ZEROS_PADRAO,
                        help="Zeros a direita no Char-by-Char: 'manter' = "
                             "sempre escreve todas as casas (atual) ou "
                             "'remover' = corta o 'zero' final ('menos zero "
                             "ponto cinco' em vez de '... cinco zero zero "
                             f"zero zero zero'). [padrao: {ZEROS_PADRAO}]")
    parser.add_argument("--so-modelo", dest="so_modelo",
                        choices=("ambos", "pj", "cbc"), default="ambos",
                        help="Agiliza experimentos A/B de texto: roda so o "
                             "modelo textual afetado pela variante ('pj' = "
                             "treina so o Pseudo-JSON, 'cbc' = so o "
                             "Char-by-Char, 'ambos' = trio completo como "
                             "sempre). O modelo nao rodado nao entra no "
                             "JSON/veredito. [padrao: ambos]")
    parser.add_argument("--self-refine", dest="self_refine",
                        action="store_true", default=False,
                        help="FLAG OPCIONAL - ativa o ciclo Self-Refine: "
                             "valida cada documento ANTES do cache/treino, "
                             "refaz com o feedback da falha e registra a "
                             "licao em .runtime/autotimes_memory.json. "
                             "Sem a flag o comportamento e identico ao atual. "
                             "[padrao: off]")
    parser.add_argument("--self-refine-max-tentativas",
                        dest="self_refine_max_tentativas", type=int, default=3,
                        help="Tentativas maximas por documento com o "
                             "self-refine ligado; esgotou -> aborta sem "
                             "gravar cache. [padrao: 3]")
    parser.add_argument("--self-refine-memoria", dest="self_refine_memoria",
                        type=str, default=None,
                        help="Caminho da memoria episodica (fora do Git). "
                             "[padrao: .runtime/autotimes_memory.json]")
    parser.add_argument("--linear-probing", dest="linear_probing",
                        action="store_true", default=False,
                        help="Substitui o decoder MLP pre-treinado por uma "
                             "projecao Linear limpa (treinada do zero). "
                             "Remove o domain shift do decoder numerico.")
    parser.add_argument("--replace-decoder", dest="replace_decoder",
                        action="store_true", default=False,
                        help="Substitui o decoder MLP treinado por uma MLP "
                             "NOVA (Linear->GELU->Dropout->Linear) "
                             "inicializada do zero. Nao herda o vies numerico.")
    parser.add_argument("--kan-decoder", dest="kan_decoder",
                        action="store_true", default=False,
                        help="Substitui o decoder MLP por uma KAN "
                             "(Kolmogorov-Arnold Network) do efficient-kan.")
    parser.add_argument("--kan-act", dest="kan_act",
                        choices=["silu", "sigmoid", "elu", "selu"],
                        default="silu",
                        help="Ativacao base da KAN (base_activation).")
    parser.add_argument("--taylor-decoder", dest="taylor_decoder",
                        action="store_true", default=False,
                        help="Substitui o decoder MLP por uma TaylorKAN "
                             "(serie de Taylor, metodo de Horner).")
    parser.add_argument("--taylor-order", dest="taylor_order",
                        type=int, choices=[2, 3], default=3,
                        help="Ordem da serie de Taylor (2 ou 3). Default: 3.")
    parser.add_argument("--taylor-no-tanh", dest="taylor_no_tanh",
                        action="store_true", default=False,
                        help="Remove o tanh clamp da TaylorLayer. "
                             "VALORES CRUOS PODERAO DIVERGIR (explosao polinomial).")
    parser.add_argument("--residual-adapter", dest="residual_adapter",
                        type=str, default="none",
                        choices=["none", "linear", "taylor", "mole"],
                        help="MLP congelado + residual adapter: "
                             "linear (Opcao B), taylor (Opcao A), "
                             "mole (Opcao C, mixture of linear experts). "
                             "Default: none (sem adapter).")
    parser.add_argument("--residual-order", dest="residual_order",
                        type=int, default=2, choices=[2, 3],
                        help="Ordem do Taylor residual (so se adapter=taylor). "
                             "Default: 2.")
    parser.add_argument("--residual-alpha", dest="residual_alpha",
                        type=float, default=0.1,
                        help="Alpha inicial do residuo (escala do adapter). "
                             "Default: 0.1.")
    parser.add_argument("--unfreeze-encoder-epoch",
                        dest="unfreeze_encoder_epoch", type=int, default=0,
                        help="Epoca para descongelar o encoder (two-phase). "
                             "0 = encoder sempre congelado.")
    parser.add_argument("--lr-unfreeze", dest="lr_unfreeze",
                        type=float, default=1e-5,
                        help="LR para o encoder descongelado (two-phase).")
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  Pool z-score canonico: 7 features (Channel Independence) + borders    #
#  oficiais do Dataset_ETT_hour. Scaler fitado APENAS no treino [0, 8640)#
# --------------------------------------------------------------------- #
def montar_pool_z(args: argparse.Namespace):
    """Janelas z-score das features selecionadas, construcao oficial.

    Borders b1s=[0, 7968, 10848] / b2s=[8640, 11520, 14400] (val/test com
    overlap de SEQ_LEN linhas, como o data_provider original). Scaler
    ajustado em values[0:8640, :] (todas as colunas, por-coluna, fiel ao
    StandardScaler do run_benchmarks) e aplicado a TODAS as linhas.

    Returns:
        (dados, scaler, splits, z, b1s, b2s) onde splits =
        [(x, y, feat_ids)] com x/y em (n, 672, 1) no espaco z-score e
        feat_ids = posicao da feature na selecao (0..len(feats)-1).
    """
    dados = carregar_etth1(args.csv, args.linhas)
    if args.linhas < N_TESTE:
        raise SystemExit(f"--linhas {args.linhas} < 14400: splits canonicos incompletos")
    values = dados["values"]
    feats = [int(f) for f in args.feats_list]

    scaler = StandardScaler().fit(values[0:B2S[0], :])
    z = scaler.transform(values).astype(np.float32)

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for b1, b2 in zip(B1S, B2S):
        tot = len(z[b1:b2]) - SEQ_LEN - PRED_LEN + 1
        xs, ys, fs = [], [], []
        for fi, f in enumerate(feats):
            for s in range(tot):
                r_begin = s + SEQ_LEN - LABEL_LEN
                xs.append(z[b1 + s: b1 + s + SEQ_LEN, f: f + 1])
                ys.append(z[b1 + r_begin: b1 + r_begin + LABEL_LEN + PRED_LEN, f: f + 1])
                fs.append(fi)
        splits.append((np.stack(xs), np.stack(ys), np.asarray(fs, dtype=np.int64)))
    return dados, scaler, splits, z, B1S, B2S


def _cache_pool_path(args: argparse.Namespace) -> Path:
    feats = "all" if len(args.feats_list) == N_FEATS else "-".join(map(str, args.feats_list))
    return CACHE_DIR / f"pool_z_v2_linhas{args.linhas}_feats{feats}.npz"


def carregar_pool_cache(args: argparse.Namespace):
    """Pool z-score (Channel Independence) com cache em disco (npz)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_pool_path(args)
    if p.exists() and not args.force_cache:
        d = np.load(p)
        splits = [(d["train_x"], d["train_y"], d["train_feats"]),
                  (d["val_x"], d["val_y"], d["val_feats"]),
                  (d["test_x"], d["test_y"], d["test_feats"])]
        scaler = StandardScaler()
        scaler.mean_ = d["scaler_mean"]
        scaler.scale_ = d["scaler_scale"]
        print(f"[cache] pool z-score carregado: {p}")
        return splits, scaler, d["z"], B1S, B2S
    dados, scaler, splits, z, b1s, b2s = montar_pool_z(args)
    np.savez(
        p,
        train_x=splits[0][0], train_y=splits[0][1], train_feats=splits[0][2],
        val_x=splits[1][0], val_y=splits[1][1], val_feats=splits[1][2],
        test_x=splits[2][0], test_y=splits[2][1], test_feats=splits[2][2],
        z=z,
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
    )
    print(f"[cache] pool z-score salvo: {p}")
    return splits, scaler, z, b1s, b2s


def _pool_lista(split: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    x, y, _ = split
    return [(x[i], y[i]) for i in range(len(x))]


# --------------------------------------------------------------------- #
#  Pseudo-JSON ES: UM documento por feature (treino [0, 8640) em z)      #
# --------------------------------------------------------------------- #
def _vmd_trend_ma(args: argparse.Namespace):
    """CLI --vmd-ma (-1=auto, 0=off, N=fixo) -> API (None=int)."""
    ma = int(getattr(args, "vmd_ma", -1))
    return None if ma < 0 else ma


def _rotulo_vmd_ma(args: argparse.Namespace) -> str:
    return _rotulo_ma_de_valor(int(getattr(args, "vmd_ma", VMD_MA_PADRAO)))


def _rotulo_ma_de_valor(ma: int) -> str:
    return "auto~L/7" if int(ma) < 0 else str(int(ma))


def _formato_de_texto(args: argparse.Namespace) -> FormatoTexto:
    """Formato do texto pedido no CLI (padrao = escrita atual byte a byte)."""
    return FormatoTexto(
        data=getattr(args, "texto_data", DATA_PADRAO),
        casas=int(getattr(args, "texto_decimais", DECIMAIS_PADRAO)),
        encoding=getattr(args, "texto_encoding", ENCODING_PADRAO),
        pj_casas=int(getattr(args, "texto_pj_casas", PJ_CASAS_PADRAO)),
        pj_marcacao=getattr(args, "texto_pj_marcacao", PJ_MARCACAO_PADRAO),
        pj_escala=getattr(args, "texto_pj_escala", PJ_ESCALA_PADRAO),
        arredondamento=getattr(args, "texto_arredondamento",
                               ARREDONDAMENTO_PADRAO),
        zeros=getattr(args, "texto_zeros", ZEROS_PADRAO),
    )


def _caminho_memoria_self_refine(args: argparse.Namespace) -> Path:
    """Caminho da memoria episodica (.runtime/, fora do Git)."""
    p = getattr(args, "self_refine_memoria", None)
    if p:
        return Path(p)
    return caminhos.ROOT_DIR / ".runtime" / "autotimes_memory.json"


def _tok(tokenizer, texto: str) -> int:
    return len(tokenizer.encode(texto, add_special_tokens=False))


def _medir_custos_texto(args: argparse.Namespace, tokenizer, dados, z) -> Dict:
    """Custo REAL (GPT-2 do repo) de 1 ponto, medido em UMA janela real.

    Nao extrapola de regra nenhuma: monta os pontos das duas escritas a
    partir da MESMA janela (672 linhas, feature 0) e tokeniza de verdade.
    Devolve {rotulo: {tok_ponto, visiveis}} para o padrao e para o formato
    escolhido, para o banner mostrar lado a lado.
    """
    from .texto_char_by_char import decompor_vmd, texto_char_por_char
    from .texto_puro import estruturado as est

    L = 672
    i0 = 1000
    xw = np.asarray(z[i0: i0 + L, 0], dtype=np.float64)
    dts = dados["dt_list"][i0: i0 + L]
    t, s, r = decompor_vmd(
        xw, k=getattr(args, "vmd_k", 7),
        alpha=getattr(args, "vmd_alpha", 2000.0),
        tau=getattr(args, "vmd_tau", 0.0), tol=getattr(args, "vmd_tol", 1e-7),
        init=getattr(args, "vmd_init", 0), trend_ma=_vmd_trend_ma(args),
    )
    n_pontos = 48
    orc = int(args.block_max) * int(args.n_paginas_max)  # 896
    pag_t = (int(args.n_paginas_max) // 3 + (1 if int(args.n_paginas_max) % 3 else 0)) * int(args.block_max)

    res: Dict[str, Dict] = {}
    fmt_padrao = _formato_de_texto_arg(args, padrao=True)
    for rotulo, fmt in (("padrao", fmt_padrao),
                        ("escolhido", _formato_de_texto(args))):
        # ---- Pseudo-JSON (paginas inteiras -> orcamento 896) ---------- #
        pontos = " ".join(
            est._pseudo_json_mstl(dts[i], float(t[i]), float(s[i]),
                                  float(r[i]), formato=fmt, idx=i)
            for i in range(n_pontos)
        ) + "."
        tok_pj = _tok(tokenizer, pontos) / n_pontos

        # ---- Char-by-Char (frase T = pag_t tokens) --------------------- #
        definir_formato(fmt)
        frase_t, _ = texto_char_por_char(xw)
        tok_cb = _tok(tokenizer, frase_t) / L

        res[rotulo] = {
            "formato": fmt,
            "pj_tok_ponto": tok_pj,
            "pj_visiveis": int(orc // max(1, round(tok_pj))),
            "cb_tok_ponto": tok_cb,
            "cb_visiveis": int(pag_t // max(1, round(tok_cb))),
        }
    # Restaura o formato ESCOLHIDO no estado do processo (o worker define o
    # dele em _processar_janela; aqui nao ha geracao em curso).
    definir_formato(_formato_de_texto(args))
    return res


def _formato_de_texto_arg(args: argparse.Namespace, padrao: bool = False) -> FormatoTexto:
    """Formato padrao literal (ignora o CLI) - usado como referencia no banner."""
    if padrao:
        return FormatoTexto(data=DATA_PADRAO, casas=DECIMAIS_PADRAO,
                            encoding=ENCODING_PADRAO)
    return _formato_de_texto(args)


def _imprimir_banner_formato(args: argparse.Namespace, tokenizer, dados,
                             z) -> None:
    """Banner do formato do texto: custo medido x padrao, com a flag Self-Refine."""
    fmt = _formato_de_texto(args)
    print(f"\n{'=' * 100}")
    print("FORMATO DO TEXTO DE ENTRADA (variantes por flag; padrao = atual)")
    print("=" * 100)
    try:
        custos = _medir_custos_texto(args, tokenizer, dados, z)
    except Exception as e:  # pragma: no cover - medicao e informativa
        print(f"  [custo] indisponivel ({type(e).__name__}: {e})")
        custos = {}
    fora = fmt.fora_do_padrao()
    print(f"  formato: {fmt.rotulo()}"
          + ("   <-- FORA DO PADRAO" if fora else ""))
    for rot in ("padrao", "escolhido"):
        if rot == "escolhido" and fmt.eh_padrao:
            continue  # no padrao so a linha de referencia interessa
        c = custos.get(rot)
        if not c:
            continue
        marca = "[padrao] " if rot == "padrao" else "[escolhido] "
        print(f"    {marca}Pseudo-JSON : {c['pj_tok_ponto']:.1f} tok/ponto -> "
              f"{c['pj_visiveis']} pontos/horas visiveis nos "
              f"{args.block_max * args.n_paginas_max} tokens")
        print(f"    {marca}Char-by-Char: {c['cb_tok_ponto']:.1f} tok/ponto -> "
              f"{c['cb_visiveis']} pontos visiveis na frase T "
              f"({(args.n_paginas_max // 3 + (1 if args.n_paginas_max % 3 else 0)) * args.block_max} tok)")
    if fora:
        print(f"  fora do padrao nesta rodada: {', '.join(fora)}")
    if getattr(args, "self_refine", False):
        print(f"  SELF-REFINE: ATIVO (max_tentativas="
              f"{args.self_refine_max_tentativas}, memoria="
              f"{_caminho_memoria_self_refine(args)})")
    # A medicao mexeu no estado de modulo; volta para o formato ESCOLHIDO.
    definir_formato(fmt)
    print("=" * 100)


def _config_vmd(args: argparse.Namespace) -> Dict:
    """Config VMD do JSON: valores usados + padroes + o que saiu do padrao."""
    usado = {
        "k": int(args.vmd_k),
        "alpha": float(args.vmd_alpha),
        "tau": float(args.vmd_tau),
        "tol": float(args.vmd_tol),
        "init": int(args.vmd_init),
        "trend_ma": _rotulo_vmd_ma(args),
    }
    padrao = {
        "k": VMD_K_PADRAO,
        "alpha": VMD_ALPHA_PADRAO,
        "tau": VMD_TAU_PADRAO,
        "tol": VMD_TOL_PADRAO,
        "init": VMD_INIT_PADRAO,
        "trend_ma": _rotulo_ma_de_valor(VMD_MA_PADRAO),
    }
    fora = [kk for kk in padrao if usado[kk] != padrao[kk]]
    return {
        "usado": usado,
        "padrao": padrao,
        "fora_do_padrao": fora,
        "screen": {"k": SCREEN_VMD_K, "alpha": SCREEN_VMD_ALPHA},
    }


def _imprimir_banner_vmd(args: argparse.Namespace) -> None:
    """Banner do VMD: valor usado x padrao + o que o screen mostrou + avisos."""
    print(f"\n{'=' * 100}")
    print("VMD - Variational Mode Decomposition (FFT, deterministico)")
    print("=" * 100)
    cfg = _config_vmd(args)
    fora = cfg["fora_do_padrao"]
    print(f"  K (modos): {args.vmd_k}   [padrao: {VMD_K_PADRAO}]"
          + ("   <-- FORA DO PADRAO" if "k" in fora else ""))
    print(f"    screen: {SCREEN_VMD_K}")
    print(f"  Alpha (banda): {args.vmd_alpha:g}   "
          f"[padrao: {VMD_ALPHA_PADRAO:g}]"
          + ("   <-- FORA DO PADRAO" if "alpha" in fora else ""))
    print(f"    screen: {SCREEN_VMD_ALPHA}")
    print(f"  Tau: {args.vmd_tau:g}   [padrao: {VMD_TAU_PADRAO:g}]")
    print(f"  Tol: {args.vmd_tol:g}   [padrao: {VMD_TOL_PADRAO:g}]")
    print(f"  Init: {args.vmd_init}   [padrao: {VMD_INIT_PADRAO}]"
          f"   ({'omegas em 0, sem RNG -> cache reproduzivel' if args.vmd_init == 0 else 'omegas uniformes' if args.vmd_init == 1 else 'ALEATORIO: cache nao reproduz'})")
    print(f"  Trend MA: {_rotulo_vmd_ma(args)}   "
          f"[padrao: {_rotulo_ma_de_valor(VMD_MA_PADRAO)}]")
    if fora:
        print(f"  foras do padrao nesta rodada: {', '.join(fora)}")
    if args.vmd_alpha <= 500:
        print("  !! alpha<=500: bandas estreitas demais -> S% ~95.3 e "
              "R% ~1.4 (o residuo engole sinal previsivel); screen "
              "mediu alpha=2000 -> S% 97.9")
    if args.vmd_alpha >= 3000 and args.vmd_init != 0:
        print(f"  !! alpha>=3000 COM init={args.vmd_init}: o screen colapsou "
              f"(S%=87+-25) -- com init=0 (padrao) alpha 3000 fica 97.88, "
              f"nao ha problema")
    if args.vmd_k <= 3:
        print("  !! K<=3: no screen S% caiu para 47 (K=3, alpha=2000)")
    if args.vmd_k >= 9:
        print(f"  K>=9: screen S%~{98.0 if args.vmd_k <= 10 else 98.7} "
              f"(ganha ~0.1-1.0 ponto sobre K=7) mas o custo sobe "
              f"(~{330 if args.vmd_k <= 10 else 410} vs 154 ms/janela -> "
              f"cache do char-by-char ~{2 if args.vmd_k <= 10 else 3}x mais lento)")
    if args.vmd_init == 2:
        print("  !! init=2 usa RNG: o cache do char-by-char NAO sera "
              "reprodutivel")
    print("=" * 100)


def _gerar_pseudo_json(args: argparse.Namespace, tokenizer, dados, z,
                       periodos_mstl=None) -> Dict:
    paginador = TestadorPaginacao(
        tokenizer, block_max=args.block_max,
        overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
    )
    texto_decomp = getattr(args, "texto_decomp", "none") or "none"
    use_mstl = texto_decomp == "mstl" and periodos_mstl is not None
    use_ceemdan = texto_decomp == "ceemdan"
    use_hibrido = texto_decomp == "mstl+ceemdan" and periodos_mstl is not None
    use_paralelo = texto_decomp == "paralelo"
    use_vmd = texto_decomp == "vmd"
    if use_mstl or use_hibrido or use_paralelo:
        lingua = "es"
        print(f"  [pseudo-json {texto_decomp}] periodos={periodos_mstl}")
    elif use_ceemdan:
        lingua = "es"
        print(f"  [pseudo-json CEEMDAN] trials={args.ceemdan_trials}, eps={args.ceemdan_epsilon}, max_imf={args.ceemdan_max_imf}")
    elif use_vmd:
        lingua = "es"
        print(f"  [pseudo-json VMD] k={args.vmd_k}, alpha={args.vmd_alpha}, "
              f"tau={args.vmd_tau}, tol={args.vmd_tol}, "
              f"init={args.vmd_init}, ma={_rotulo_vmd_ma(args)}")
    else:
        lingua = estruturado.TEMPLATES_ESTRUTURADOS[TEMPLATE_PSEUDO_JSON]["idioma"]
    ids_pf, lens_pf, infos = [], [], []
    fmt = _formato_de_texto(args)
    for fi, f in enumerate(args.feats_list):
        if use_mstl:
            doc = estruturado.serie_para_texto_mstl(
                dados["dt_list"], z, int(f), N_TREINO, periodos_mstl,
                formato=fmt,
            )
        elif use_ceemdan:
            doc = estruturado.serie_para_texto_ceemdan(
                dados["dt_list"], z, int(f), N_TREINO,
                trials=args.ceemdan_trials, epsilon=args.ceemdan_epsilon,
                max_imf=args.ceemdan_max_imf,
                seed=args.ceemdan_seed,
                formato=fmt,
            )
        elif use_vmd:
            doc = estruturado.serie_para_texto_vmd(
                dados["dt_list"], z, int(f), N_TREINO,
                k=args.vmd_k, alpha=args.vmd_alpha,
                tau=args.vmd_tau, tol=args.vmd_tol,
                init=args.vmd_init, trend_ma=_vmd_trend_ma(args),
                formato=fmt,
            )
        elif use_hibrido:
            doc = estruturado.serie_para_texto_hibrido(
                dados["dt_list"], z, int(f), N_TREINO, periodos_mstl,
                trials=args.ceemdan_trials, epsilon=args.ceemdan_epsilon,
                max_imf=args.ceemdan_max_imf,
                seed=args.ceemdan_seed,
                formato=fmt,
            )
        elif use_paralelo:
            from ..ceemdan.decomposicao import decompor_paralelo
            x_feat = np.asarray(z[:N_TREINO, int(f)], dtype=np.float64).ravel()
            x_feat = np.nan_to_num(x_feat, nan=0.0, posinf=0.0, neginf=0.0)
            t_par, s_par, r_par = decompor_paralelo(
                x_feat, trials=args.ceemdan_trials, epsilon=args.ceemdan_epsilon,
                max_imf=args.ceemdan_max_imf,
                seed=args.ceemdan_seed,
            )
            pontos = []
            for i in range(min(N_TREINO, len(dados["dt_list"]))):
                pontos.append(estruturado._pseudo_json_mstl(
                    dados["dt_list"][i], t_par[i], s_par[i], r_par[i],
                    formato=fmt, idx=i,
                ))
            doc = " ".join(pontos) + "."
        else:
            doc = estruturado.serie_para_texto_estruturado(
                dados["dt_list"], z, int(f), 0, N_TREINO, TEMPLATE_PSEUDO_JSON,
            )
        r = paginador.paginar(ESTRATEGIA_PAGINACAO, doc, lingua)
        r.update(paginador.para_modelo(r))
        ids_pf.append(np.asarray(r["ids"]))
        lens_pf.append(np.asarray(r["lens"]))
        infos.append((int(r["tokens_originais"]), int(r["tokens_retidos"]),
                      int(r["n_paginas"]), int(r.get("paginas_descartadas", 0))))
        print(f"  [pseudo-json feat {f}] {infos[-1][0]:,} tokens orig -> "
              f"{infos[-1][1]:,} ret | {infos[-1][2]} paginas")
    return {
        "ids_per_feat": np.stack(ids_pf),
        "lens_per_feat": np.stack(lens_pf),
        "tokens_originais": infos[0][0], "tokens_retidos": infos[0][1],
        "n_paginas": infos[0][2], "paginas_descartadas": infos[0][3],
    }


def _self_refine_contexto(args: argparse.Namespace, rotulo: str) -> Dict:
    """Contexto descritivo gravado junto de cada licao (spec: {motivo, regra, contexto})."""
    fmt = _formato_de_texto(args)
    return {
        "rotulo": rotulo,
        "decomp": getattr(args, "texto_decomp", "none"),
        "formato_texto": fmt.rotulo(),
        "bloco": int(args.block_max),
        "paginas": int(args.n_paginas_max),
        "vmd": (f"k={getattr(args, 'vmd_k', 7)} "
                f"alpha={getattr(args, 'vmd_alpha', 2000.0):g} "
                f"init={getattr(args, 'vmd_init', 0)} "
                f"ma={_rotulo_vmd_ma(args)}")
        if getattr(args, "texto_decomp", "none") == "vmd" else None,
    }


def _validar_texto_janela(args: argparse.Namespace, xw) -> List:
    """Checa o TEXTO gerado numa janela real: frases + finitude + T+S+R.

    Sao os verificadores de texto (marcadores T/S/R, NaN/Inf, reconstrucao)
    rodando em 1 janela real com os MESMOS parametros da rodada: e a mesma
    funcao de escrita que produz o cache, entao uma falha aqui e uma falha
    de escrita. O contrato dos tensores e checado por inteiro em separado
    (_validar_pseudo_json / _validar_char_by_char).
    """
    from .self_refine import validadores as V
    from .texto_char_by_char import decompor_vmd, frases_texto_vmd

    xw = np.asarray(xw, dtype=np.float64).ravel()
    kw = dict(k=getattr(args, "vmd_k", 7), alpha=getattr(args, "vmd_alpha", 2000.0),
              tau=getattr(args, "vmd_tau", 0.0), tol=getattr(args, "vmd_tol", 1e-7),
              init=getattr(args, "vmd_init", 0), trend_ma=_vmd_trend_ma(args))
    frases = frases_texto_vmd(xw, **kw)[:3]
    t, s, r = decompor_vmd(xw, **kw)
    return [f for f in (
        V.validar_frases(frases),
        V.validar_finitude([xw, t, s, r]),
        V.validar_reconstrucao(xw, t, s, r),
    ) if f is not None]


def _validar_pseudo_json(args: argparse.Namespace, r: Dict) -> List:
    """Contrato ids/lens por feature + orcamento 7x128 (antes do cache)."""
    from .self_refine import validadores as V

    falhas = []
    ids = np.asarray(r["ids_per_feat"])
    lens = np.asarray(r["lens_per_feat"])
    b, p = int(args.block_max), int(args.n_paginas_max)
    esperado_ids = (len(ids), b * p)
    esperado_lens = (len(lens), p)
    if ids.shape != esperado_ids or lens.shape != esperado_lens:
        falhas.append(V.Falha(
            "TENSOR_CONTRATO",
            f"ids {ids.shape} != {esperado_ids} ou lens {lens.shape} != "
            f"{esperado_lens}",
        ))
        return falhas
    falhas.extend(f for f in (
        V.validar_finitude(lens),
    ) if f is not None)
    # Orcamento: checado POR FEATURE (cada documento e paginado separado).
    for i in range(lens.shape[0]):
        f = V.validar_orcamento(lens[i], b, p)
        if f is not None:
            falhas.append(f)
            break
    return falhas


def _validar_char_by_char(args: argparse.Namespace, dss) -> List:
    """Contrato ids/lens/max_abs + finitude + orcamento (antes do cache)."""
    from .self_refine import validadores as V

    falhas = []
    b, p = int(args.block_max), int(args.n_paginas_max)
    for rotulo, ds in zip(("train", "val", "test"), dss):
        ids, lens, mx = (ds.tensors[1].numpy(), ds.tensors[2].numpy(),
                         ds.tensors[4].numpy())
        for f in (
            V.validar_finitude(lens),
            V.validar_finitude(mx),
        ):
            if f is not None:
                falhas.append(V.Falha(f.codigo, f"{rotulo}: {f.detalhe}",
                                      {"split": rotulo}))
        if lens.min() < 0 or lens.max() > b:
            falhas.append(V.Falha(
                "TENSOR_CONTRATO",
                f"{rotulo}: lens fora de [0, {b}] "
                f"[{int(lens.min())}, {int(lens.max())}]",
                {"split": rotulo},
            ))
        estouradas = int((lens.sum(axis=1) > b * p).sum())
        if estouradas:
            falhas.append(V.Falha(
                "ORCAMENTO_ESTOURADO",
                f"{rotulo}: {estouradas} janela(s) com mais de "
                f"{b * p} tokens paginados",
                {"split": rotulo, "n_janelas": estouradas},
            ))
    return falhas


def _self_refine_pseudo_json(args: argparse.Namespace, tokenizer, dados, z,
                             periodos_mstl=None) -> Dict:
    """Gera o Pseudo-JSON com validacao antes do cache (flag --self-refine).

    Roda SÓ no processo pai: a geração do documento acontece aqui (checagem
    textual numa janela real + contrato dos tensores por feature), o retry
    regera o documento e a memoria so e escrita no caminho de falha.
    """
    from .self_refine import ErroSelfRefine, rodar_self_refine

    contexto = _self_refine_contexto(args, "pseudo-json")

    def gerar(feedback):
        if feedback:
            for fb in feedback:
                print(f"  [self-refine] tentativa seguinte com regra: "
                      f"{fb['codigo']} -> {fb['regra']}")
        return _gerar_pseudo_json(args, tokenizer, dados, z,
                                  periodos_mstl=periodos_mstl)

    def validar(r):
        xw = np.asarray(z[1000: 1000 + 672, 0], dtype=np.float64)
        return (_validar_texto_janela(args, xw)
                + _validar_pseudo_json(args, r))

    try:
        res = rodar_self_refine(
            gerar, validar, contexto, rotulo="pseudo-json",
            max_tentativas=getattr(args, "self_refine_max_tentativas", 3),
            caminho_memoria=_caminho_memoria_self_refine(args),
        )
    except ErroSelfRefine as e:
        raise SystemExit(f"[self-refine] {e}") from e
    print(f"[self-refine] pseudo-json aceito na tentativa "
          f"{res.tentativas}/{getattr(args, 'self_refine_max_tentativas', 3)}"
          + (f" (houve refinamento; memoria: "
             f"{_caminho_memoria_self_refine(args)})" if res.corrigido else ""))
    return res.resultado


def _self_refine_char_by_char(args: argparse.Namespace, tokenizer, splits,
                              periodos_mstl=None):
    """Gera os tensores char-by-char com validação antes do cache (flag)."""
    from .self_refine import ErroSelfRefine, rodar_self_refine

    contexto = _self_refine_contexto(args, "char-by-char")

    def gerar(feedback):
        if feedback:
            for fb in feedback:
                print(f"  [self-refine] tentativa seguinte com regra: "
                      f"{fb['codigo']} -> {fb['regra']}")
        return _gerar_char_by_char(args, tokenizer, splits, periodos_mstl)

    def validar(res):
        dss, _stats, _guard = res
        xw = np.asarray(splits[0][0][0], dtype=np.float64).ravel()
        return (_validar_texto_janela(args, xw)
                + _validar_char_by_char(args, dss))

    try:
        res = rodar_self_refine(
            gerar, validar, contexto, rotulo="char-by-char",
            max_tentativas=getattr(args, "self_refine_max_tentativas", 3),
            caminho_memoria=_caminho_memoria_self_refine(args),
        )
    except ErroSelfRefine as e:
        raise SystemExit(f"[self-refine] {e}") from e
    print(f"[self-refine] char-by-char aceito na tentativa "
          f"{res.tentativas}/{getattr(args, 'self_refine_max_tentativas', 3)}"
          + (f" (houve refinamento; memoria: "
             f"{_caminho_memoria_self_refine(args)})" if res.corrigido else ""))
    return res.resultado


def _caminho_cache_pseudo_json(
    args: argparse.Namespace, periodos_mstl=None,
) -> Path:
    """Caminho do cache ids/lens do Pseudo-JSON (tag estavel por variante).

    Mantido em funcao UNICA para que a tag possa ser conferida sem gerar o
    documento: o padrao nao leva sufixo de variante de texto e por isso
    reutiliza os caches historicos.
    """
    texto_decomp = getattr(args, "texto_decomp", "none") or "none"
    decomp_tag = f"_d{texto_decomp}"
    if texto_decomp == "mstl" and periodos_mstl is not None:
        decomp_tag += f"_p{periodos_mstl}"
    elif texto_decomp in ("ceemdan", "mstl+ceemdan", "paralelo"):
        decomp_tag += f"_t{args.ceemdan_trials}_e{args.ceemdan_epsilon}_m{args.ceemdan_max_imf}_s{args.ceemdan_seed}"
        if getattr(args, "ceemdan_backend", "pyemd") == "hip":
            decomp_tag += "_bhip"
        if texto_decomp in ("mstl+ceemdan",) and periodos_mstl is not None:
            decomp_tag += f"_p{periodos_mstl}"
    elif texto_decomp == "vmd":
        decomp_tag += (
            f"_k{getattr(args, 'vmd_k', 7)}"
            f"_a{getattr(args, 'vmd_alpha', 2000.0)}"
            f"_tau{getattr(args, 'vmd_tau', 0.0)}"
            f"_tol{getattr(args, 'vmd_tol', 1e-7)}"
            f"_i{getattr(args, 'vmd_init', 0)}"
            f"_ma{_rotulo_vmd_ma(args).replace('~', '').replace('/', 'x')}"
        )
    decomp_tag += _formato_de_texto(args).tag_cache_pj()
    chave = hashlib.md5(
        f"{TEMPLATE_PSEUDO_JSON}_v2_b{args.block_max}_p{args.n_paginas_max}"
        f"_o{args.overlap}_feats{'-'.join(map(str, args.feats_list))}"
        f"{decomp_tag}".encode()
    ).hexdigest()[:12]
    return CACHE_DIR / f"pseudo_json_{chave}.npz"


def carregar_pseudo_json(args: argparse.Namespace, tokenizer, dados, z,
                         periodos_mstl=None) -> Dict:
    """ids/lens do documento Pseudo-JSON por feature (cache leve em npz)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _caminho_cache_pseudo_json(args, periodos_mstl)
    if p.exists() and not args.force_cache:
        d = np.load(p)
        print(f"[cache] pseudo-json ids/lens (por feature): {p}")
        return {
            "ids_per_feat": d["ids_per_feat"], "lens_per_feat": d["lens_per_feat"],
            "tokens_originais": int(d["tokens_originais"]),
            "tokens_retidos": int(d["tokens_retidos"]),
            "n_paginas": int(d["n_paginas"]),
            "paginas_descartadas": int(d["paginas_descartadas"]),
        }
    if getattr(args, "self_refine", False):
        r = _self_refine_pseudo_json(args, tokenizer, dados, z,
                                     periodos_mstl=periodos_mstl)
    else:
        r = _gerar_pseudo_json(args, tokenizer, dados, z, periodos_mstl=periodos_mstl)
    np.savez(
        p,
        ids_per_feat=r["ids_per_feat"], lens_per_feat=r["lens_per_feat"],
        tokens_originais=r["tokens_originais"], tokens_retidos=r["tokens_retidos"],
        n_paginas=r["n_paginas"], paginas_descartadas=r["paginas_descartadas"],
    )
    print(f"[cache] pseudo-json ids/lens (por feature) salvo: {p}")
    return r


def datasets_pseudo_json(args: argparse.Namespace, splits, prep: Dict) -> List[TensorDataset]:
    """Datasets de 4 tensores (x_z, ids, lens, y_z) por split.

    Cada janela recebe o documento paginado da SUA feature (Channel
    Independence): ids/lens roteados por feat_ids.
    """
    ids_pf = torch.from_numpy(prep["ids_per_feat"])
    lens_pf = torch.from_numpy(prep["lens_per_feat"])
    dss: List[TensorDataset] = []
    for x, y, fs in splits:
        dss.append(TensorDataset(
            torch.from_numpy(x),
            ids_pf[fs].to(torch.int64),
            lens_pf[fs].to(torch.int64),
            torch.from_numpy(y),
        ))
    return dss


# --------------------------------------------------------------------- #
#  Char-by-Char: documento POR JANELA (5 tensores, max_abs local)        #
# --------------------------------------------------------------------- #
def _caminho_cache_char_by_char(
    args: argparse.Namespace, periodos_mstl=None,
) -> Path:
    """Caminho do cache de tensores char-by-char (tag estavel por variante).

    Mesma logica de _caminho_cache_pseudo_json: funcao unica para permitir
    conferir a tag sem regerar o dataset; o padrao nao leva sufixo de
    variante de texto.
    """
    texto_decomp = getattr(args, "texto_decomp", "none")
    ceemdan_tag = ""
    if texto_decomp in ("ceemdan", "mstl+ceemdan", "paralelo"):
        ceemdan_tag = (
            f"_ct{getattr(args, 'ceemdan_trials', 100)}"
            f"_ce{getattr(args, 'ceemdan_epsilon', 0.005)}"
            f"_cm{getattr(args, 'ceemdan_max_imf', 5)}"
            f"_cs{getattr(args, 'ceemdan_seed', 42)}"
        )
        if getattr(args, "ceemdan_backend", "pyemd") == "hip":
            ceemdan_tag += "_bhip"
    elif texto_decomp == "vmd":
        ceemdan_tag = (
            f"_vk{getattr(args, 'vmd_k', 7)}"
            f"_va{getattr(args, 'vmd_alpha', 2000.0)}"
            f"_vt{getattr(args, 'vmd_tau', 0.0)}"
            f"_vtol{getattr(args, 'vmd_tol', 1e-7)}"
            f"_vi{getattr(args, 'vmd_init', 0)}"
            f"_vma{_rotulo_vmd_ma(args).replace('~', '').replace('/', 'x')}"
        )
    ceemdan_tag += _formato_de_texto(args).tag_cache_cb()
    chave = hashlib.md5(
        f"char_by_char_v3_b{args.block_max}_p{args.n_paginas_max}"
        f"_o{args.overlap}_feats{'-'.join(map(str, args.feats_list))}"
        f"_td{texto_decomp}"
        f"_mstl{periodos_mstl}"
        f"{ceemdan_tag}".encode()
    ).hexdigest()[:12]
    return CACHE_DIR / f"char_by_char_{chave}.npz"


def carregar_char_by_char(args: argparse.Namespace, tokenizer, splits,
                          periodos_mstl=None, mstl_info=None) -> Tuple[List[TensorDataset], Dict]:
    """Tensores char-by-char (7 features) com cache em npz (geracao 1x)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    texto_decomp = getattr(args, "texto_decomp", "none")
    if periodos_mstl is not None:
        print(f"[mstl] periodos detectados: diario={periodos_mstl[0]} "
              f"({mstl_info['minutos_timestep']:.0f}min/timestep), "
              f"semanal={periodos_mstl[1]} | "
              f"tupla={periodos_mstl}")
    p = _caminho_cache_char_by_char(args, periodos_mstl)
    if p.exists() and not args.force_cache:
        d = np.load(p)
        print(f"[cache] char-by-char tensores: {p}")
        dss = []
        for a in ("train", "val", "test"):
            i, l, m = d[f"{a}_ids"], d[f"{a}_lens"], d[f"{a}_max_abs"]
            x, y, _ = splits[("train", "val", "test").index(a)]
            dss.append(TensorDataset(
                torch.from_numpy(x),
                torch.tensor(i, dtype=torch.int64),
                torch.tensor(l, dtype=torch.int64),
                torch.from_numpy(y),
                torch.from_numpy(m),
            ))
        return dss, {}
    # ---- gera do zero (uma unica vez; as 3 seeds reutilizam) ---------- #
    if getattr(args, "self_refine", False):
        dss, stats, guard = _self_refine_char_by_char(
            args, tokenizer, splits, periodos_mstl=periodos_mstl)
    else:
        dss, stats, guard = _gerar_char_by_char(
            args, tokenizer, splits, periodos_mstl=periodos_mstl)
    try:
        np.savez(p, **guard)
        print(f"[cache] char-by-char tensores salvo: {p}")
    except Exception as e:  # pragma: no cover - cache e so otimizacao
        print(f"[cache] aviso: nao deu para salvar char-by-char ({e})")
    return dss, stats


def _gerar_char_by_char(args: argparse.Namespace, tokenizer, splits,
                        periodos_mstl=None):
    """Gera os tensores dos 3 splits (sem gravar cache). Devolve (dss, stats, guard)."""
    t0 = time.perf_counter()
    dss, stats = [], {}
    guard: Dict[str, np.ndarray] = {}
    texto_decomp = getattr(args, "texto_decomp", "none")
    for rotulo, split in zip(("train", "val", "test"), splits):
        ds, agg = construir_dataset_char_by_char(
            _pool_lista(split), tokenizer,
            block_max=args.block_max,
            n_paginas_max=args.n_paginas_max,
            overlap_frac=args.overlap,
            return_stats=True,
            decomp=texto_decomp,
            periodos_mstl=periodos_mstl,
            ceemdan_trials=getattr(args, "ceemdan_trials", 100),
            ceemdan_epsilon=getattr(args, "ceemdan_epsilon", 0.005),
            ceemdan_max_imf=getattr(args, "ceemdan_max_imf", 5),
            ceemdan_seed=getattr(args, "ceemdan_seed", 42),
            ceemdan_backend=getattr(args, "ceemdan_backend", "pyemd"),
            vmd_k=getattr(args, "vmd_k", 7),
            vmd_alpha=getattr(args, "vmd_alpha", 2000.0),
            vmd_tau=getattr(args, "vmd_tau", 0.0),
            vmd_tol=getattr(args, "vmd_tol", 1e-7),
            vmd_init=getattr(args, "vmd_init", 0),
            vmd_ma=_vmd_trend_ma(args),
            formato=_formato_de_texto(args),
        )
        guard[f"{rotulo}_ids"] = ds.tensors[1].numpy()
        guard[f"{rotulo}_lens"] = ds.tensors[2].numpy()
        guard[f"{rotulo}_max_abs"] = ds.tensors[4].numpy()
        dss.append(ds)
        stats[rotulo] = agg
        print(f"  [char-by-char {rotulo}] {len(ds)} janelas | max_abs="
              f"{agg['max_abs_media']:.3f} ± {agg['max_abs_desvio']:.3f} | "
              f"{time.perf_counter() - t0:.1f}s")
    return dss, stats, guard


# --------------------------------------------------------------------- #
#  Baseline numerico: carrega do log (0.3539) ou re-mede eval-only       #
# --------------------------------------------------------------------- #
def montar_modelo_numerico(ckpt_path: Path, device: torch.device, use_amp: bool):
    """AutoTimesModel SEM ramo textual (idêntico ao checkpoint original)."""
    rb = caminhos.importar_run_benchmarks()
    ckpt = carregar_checkpoint(ckpt_path)
    flags, args = ckpt["flags"], ckpt["args"]
    torch_dtype = torch.float16 if (use_amp and device.type == "cuda") else torch.float32
    model = rb.AutoTimesModel(
        seq_len=args["seq_len"], pred_len=args["pred_len"], n_vars=args["n_vars"],
        label_len=args["label_len"], mlp_hidden_dim=args["mlp_hidden_dim"],
        mlp_hidden_layers=args["mlp_hidden_layers"], mlp_activation=args["mlp_activation"],
        dropout=args["dropout"], gpt2_dir=Path(args["gpt2_dir"]),
        torch_dtype=torch_dtype,
        use_robust=bool(flags["use_robust"]), use_overlap=bool(flags["use_overlap"]),
        use_vol_gate=bool(flags["use_vol_gate"]), unfreeze_layers=0,
        textual=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, flags


def avaliar_numerico(splits, ckpt_path: Path, device: torch.device,
                     use_amp: bool, batch_size: int = 256) -> Dict:
    """Eval-only (protocolo oficial: MSE/MAE no horizonte pred_len, z-score)."""
    rb = caminhos.importar_run_benchmarks()
    _, _, (x_te, y_te, _) = splits
    ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model, flags = montar_modelo_numerico(ckpt_path, device, use_amp)
    pred_len = 96
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(device, non_blocking=True)
            y_b = batch[1].to(device, non_blocking=True)
            pred = model(x_b)
            preds.append(pred[:, -pred_len:, :].float().cpu().numpy())
            trues.append(y_b[:, -pred_len:, :].float().cpu().numpy())
    p = np.concatenate(preds, 0)
    t = np.concatenate(trues, 0)
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    return {"test_mse": mse, "test_mae": mae, "n_janelas_test": len(ds)}


def carregar_numerico_log() -> Optional[Dict]:
    """0.3539 cravado (resultados/numerico_base_stats.json)."""
    if not JSON_NUMERICO_LOG.exists():
        return None
    d = json.load(open(JSON_NUMERICO_LOG, encoding="utf-8"))
    m = d.get("modelos", {}).get("smetimes")
    if not m or m.get("test_mse_media") is None:
        return None
    return {
        "test_mse_media": float(m["test_mse_media"]),
        "test_mae_media": float(m.get("test_mae_media")),
        "n_janelas_test": int(m.get("n_janelas_test", 0)),
        "referencia_artigo": float(d.get("paridade", {}).get("referencia_artigo", {})
                                   .get("mse_run8_zscore", 0.3539)),
    }


# --------------------------------------------------------------------- #
#  Bateria textual (3 seeds x 10 epocas, restore_best por validacao)     #
# --------------------------------------------------------------------- #
def rodar_bateria_textual(
    args: argparse.Namespace, nome: str, loaders,
    ckpt_path: Path, device: torch.device, use_amp: bool, seeds: List[int],
    decode_fn=None, model_fn=None, sufixo: str = "",
) -> Dict:
    rb = caminhos.importar_run_benchmarks()
    if model_fn is None:
        model_fn = montar_modelo
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    por_seed: Dict[str, Dict] = {}
    mse_list, mae_list, bestval_list, epoca_list, estouros_list = [], [], [], [], []
    val_por_epoca: List[List[float]] = []
    test_por_epoca: List[List[float]] = []
    tr_ld, va_ld, te_ld = loaders
    print(f"\n--- {nome} ---")
    for seed in seeds:
        rb.seed_everything(seed)
        model, info = model_fn(ckpt_path, device, use_amp=use_amp)
        print(f"  [mount seed={seed}] treinaveis={info['params_treinaveis']:,}")
        t0 = time.perf_counter()
        result = rodar_convergencia(
            model, tr_ld, va_ld, te_ld,
            epochs=args.epochs, lr=args.lr, device=device,
            use_amp=use_amp, seed=seed, restore_best=True, decode_fn=decode_fn,
            unfreeze_encoder_epoch=args.unfreeze_encoder_epoch,
            lr_unfreeze=args.lr_unfreeze,
        )
        mse = result["teste_final_mse_pred"]
        mae = result["teste_final_mae_pred"]
        bv = result["melhor_val_mse_pred"]
        be = result["melhor_epoca"]
        estouros = result["batches_estourados_total"]

        # Salvar checkpoint do modelo (pesos treinados, para inferencia)
        ckpt_name = f"model_{sufixo}_{_ts}_seed{seed}.pt"
        ckpt_save_path = CKPT_DIR / ckpt_name
        torch.save({
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "test_mse": mse,
            "test_mae": mae,
            "melhor_val_mse": bv,
            "melhor_epoca": be,
            "epochs": args.epochs,
            "lr": args.lr,
            "texto_decomp": args.texto_decomp,
            "sufixo": sufixo,
        }, ckpt_save_path)
        print(f"    [checkpoint] salvo: {ckpt_save_path.name}")

        por_seed[str(seed)] = {
            "test_mse": round(mse, 6), "test_mae": round(mae, 6),
            "melhor_val_mse": round(bv, 6), "melhor_epoca": be,
            "estouros": estouros,
            "checkpoint": ckpt_save_path.name,
            "val_mse_por_epoca": [round(v, 6) for v in result["historico"]["val_mse_pred"]],
            "test_mse_por_epoca": [round(v, 6) for v in result["historico"]["test_mse_pred"]],
        }
        mse_list.append(mse)
        mae_list.append(mae)
        bestval_list.append(bv)
        epoca_list.append(be or args.epochs)
        estouros_list.append(estouros)
        val_por_epoca.append(result["historico"]["val_mse_pred"])
        test_por_epoca.append(result["historico"]["test_mse_pred"])
        print(f"    [seed={seed}] test_mse={mse:.4f} | test_mae={mae:.4f} | "
              f"best_val={bv:.4f} (ep {be}) | estouros={estouros} | "
              f"{time.perf_counter() - t0:.1f}s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_ep = max((len(v) for v in val_por_epoca), default=0)
    return {
        "test_mse_media": round(float(np.mean(mse_list)), 4),
        "test_mse_desvio": round(float(np.std(mse_list)), 4),
        "test_mae_media": round(float(np.mean(mae_list)), 4),
        "test_mae_desvio": round(float(np.std(mae_list)), 4),
        "melhor_val_mse_media": round(float(np.mean(bestval_list)), 4),
        "melhor_val_mse_desvio": round(float(np.std(bestval_list)), 4),
        "melhor_epoca_media": round(float(np.mean(epoca_list)), 2),
        "estouros_total": sum(estouros_list),
        "val_mse_media": [round(float(np.mean([v[e] for v in val_por_epoca])), 6) for e in range(n_ep)],
        "val_mse_desvio": [round(float(np.std([v[e] for v in val_por_epoca])), 6) for e in range(n_ep)],
        "test_mse_media_por_epoca": [round(float(np.mean([v[e] for v in test_por_epoca])), 6) for e in range(n_ep)],
        "test_mse_desvio_por_epoca": [round(float(np.std([v[e] for v in test_por_epoca])), 6) for e in range(n_ep)],
        "por_seed": por_seed,
    }


# --------------------------------------------------------------------- #
#  Relatorio + graficos                                                  #
# --------------------------------------------------------------------- #
def imprimir_tabela(resumo: Dict) -> None:
    print("\n" + "=" * 100)
    print("VEREDITO FINAL APPLES-TO-APPLES (z-score, 10 epocas x 3 seeds)")
    print("=" * 100)
    m = resumo["modelos"]
    print(f"{'Modelo':<16s} {'Test MSE':>14s} {'Test MAE':>14s} {'best_val':>12s} "
          f"{'estouros':>8s}")
    print("-" * 100)
    for chave, rotulo in (("numerico", "Numerico"),
                          ("pseudo_json_es", "Pseudo-JSON ES"),
                          ("char_by_char", "Char-by-Char")):
        c = m.get(chave)
        if c is None:  # --so-modelo pj|cbc: modelo nao rodado nesta rodada
            print(f"{rotulo:<16s} {'(nao rodado)':>14s} "
                  f"{'-':>14s} {'-':>12s} {'-':>8s}")
            continue
        if chave == "numerico":
            print(f"{rotulo:<16s} {c['test_mse_media']:>14.4f} "
                  f"{c['test_mae_media']:>14.4f} {'-':>12s} {'-':>8s}")
        else:
            print(f"{rotulo:<16s} {c['test_mse_media']:>14.4f} ± {c['test_mse_desvio']:>8.4f} "
                  f"{c['test_mae_media']:>7.4f} ± {c['test_mae_desvio']:>7.4f} "
                  f"{c['melhor_val_mse_media']:>12.4f} {c['estouros_total']:>8d}")
    print("-" * 100)
    print("Formato: media ± desvio padrao sobre 3 seeds (menor MSE vence).")
    print("=" * 100)


def _sufixo_titulo_do_resumo(resumo: Dict) -> Tuple[str, str]:
    """Reconstroi sufixo de arquivo/titulo a partir do config do JSON."""
    cfg = resumo.get("config", {})
    if cfg.get("variante_arquitetura") == "dual":
        partes = ["dual"]
        if cfg.get("dual_fusao") not in (None, "glu"):
            partes.append("soma")
        stl = cfg.get("decomp_mode") == "stl_simple"
        if stl:
            partes.append("stl")
        tag = "Dual " + ("GLU" if cfg.get("dual_fusao", "glu") == "glu" else "SOMA")
        if stl:
            tag += " + STL-k25"
        return "_" + "_".join(partes), f" [{tag}]"
    if cfg.get("texto_decomp") == "stl":
        return "_txt_stl", " [Original + Texto STL (Tendencia/Sazonalidade/Residuos, LOESS p24)]"
    if cfg.get("texto_decomp") == "vmd":
        v = cfg.get("vmd") or {}
        u = v.get("usado", v)  # JSON antigos eram planos (sem 'usado')
        extra = []
        if u.get("init") not in (None, 0):
            extra.append(f"init={u['init']}")
        if str(u.get("trend_ma", "auto~L/7")) != "auto~L/7":
            extra.append(f"MA={u['trend_ma']}")
        sufixo_extra = ("_" + "-".join(e.replace("=", "").replace("/", "x")
                                      for e in extra)) if extra else ""
        fora = v.get("fora_do_padrao") or []
        nota_padrao = ""
        if fora:
            pd = v.get("padrao") or {}
            nota_padrao = (" | padrao: "
                           + ", ".join(f"{kk}={pd.get(kk)}" for kk in fora))
        return ("_txt_vmd" + sufixo_extra,
                f" [Original + Texto VMD (K={u.get('k', VMD_K_PADRAO)} modos, "
                f"alpha={u.get('alpha', VMD_ALPHA_PADRAO):g}"
                + (f", {', '.join(extra)}" if extra else "")
                + nota_padrao + ")]")
    return "", ""


def gerar_graficos(resumo: Dict, sufixo_grafico: str = "",
                   titulo_extra: str = "",
                   out_dir: Optional[Path] = None) -> List[Path]:
    """G1 convergencia + G2 barras (figsize 16x9, sns.despine, fontes grandes).

    sufixo_grafico: ex. "_dual_soma_stl" (com underscore inicial).
    titulo_extra:   ex. " [Dual SOMA + STL-k25]".
    out_dir:        diretorio de saida (default: OUT_DIR; use RECENTE_DIR
                    para salvas em testes_letras/recente/).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Gera timestamp unico para nomes de arquivo
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = out_dir or OUT_DIR
    dest.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.35)
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 17,
                         "axes.labelsize": 15, "legend.fontsize": 13,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    suffix, title_var = sufixo_grafico, titulo_extra
    m = resumo["modelos"]
    numerico = m["numerico"]["test_mse_media"]
    saidas: List[Path] = []

    # ---- G1: Convergencia de Titãs ------------------------------------ #
    fig, ax = plt.subplots(figsize=(16, 9))
    for chave, rotulo in (("pseudo_json_es", "Pseudo-JSON ES"),
                          ("char_by_char", "Char-by-Char")):
        c = m.get(chave)
        if c is None:  # --so-modelo: serie ausente nao entra no grafico
            continue
        med = np.asarray(c["val_mse_media"])
        dev = np.asarray(c["val_mse_desvio"])
        ep = np.arange(1, len(med) + 1)
        ax.plot(ep, med, "-o", color=_CORES[chave], lw=3, ms=8, zorder=4,
                label=f"{rotulo} (Test MSE {c['test_mse_media']:.4f} ± "
                      f"{c['test_mse_desvio']:.4f})")
        ax.fill_between(ep, med - dev, med + dev, color=_CORES[chave],
                        alpha=0.18, zorder=2)
    ax.axhline(numerico, color="red", ls="--", lw=2.5, zorder=1,
               label=f"Baseline Numerico (Run-8) = {numerico:.4f}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val MSE no horizonte (z-score)")
    ax.set_title(f"A Convergencia de Titãs{title_var}\n"
                 "Val MSE z-score por epoca (media ± 1 desvio padrao das 3 seeds)")
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(axis="y", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    p1 = dest / f"convergencia_trio{suffix}_{ts}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saidas.append(p1)
    print(f"[graficos] {p1}")

    # ---- G2: Veredito Final Z-Score ------------------------------------ #
    fig, ax = plt.subplots(figsize=(16, 9))
    _itens_g2 = (("numerico", "Baseline Numerico\n(Run-8 SMETimes)"),
                 ("pseudo_json_es", "Pseudo-JSON ES"),
                 ("char_by_char", "Char-by-Char"))
    nomes = [rot for chave, rot in _itens_g2 if chave in m]
    medias = [numerico if chave == "numerico" else m[chave]["test_mse_media"]
              for chave, _ in _itens_g2 if chave in m]
    desvios = [0.0 if chave == "numerico" else m[chave]["test_mse_desvio"]
               for chave, _ in _itens_g2 if chave in m]
    cores = [_CORES[chave] for chave, _ in _itens_g2 if chave in m]
    barras = ax.bar(nomes, medias, yerr=desvios, capsize=8, width=0.55,
                    color=cores, edgecolor="black", linewidth=1.2, zorder=3,
                    error_kw=dict(ecolor="black", elinewidth=2.2, capsize=9))
    for b, med, dev in zip(barras, medias, desvios):
        ax.text(b.get_x() + b.get_width() / 2, med + dev + max(medias) * 0.03,
                f"{med:.4f}", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(m + d for m, d in zip(medias, desvios)) * 1.28)
    ax.set_ylabel("Test MSE z-score (media das 3 seeds)")
    ax.set_title("O Veredito Final Z-Score\n"
                 "Test MSE no horizonte (menor = melhor; barras de erro = "
                 "desvio padrao das seeds)")
    ax.grid(axis="y", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    p2 = dest / f"barras_veredito_trio{suffix}_{ts}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saidas.append(p2)
    print(f"[graficos] {p2}")
    return saidas


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rb = caminhos.importar_run_benchmarks()

    # JSON de saida varia conforme a variante arquitetural (com timestamp)
    global JSON_OUT
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.decomp_mode != "none" and args.variante != "dual":
        raise SystemExit("--decomp-mode so se aplica com "
                         "--variante-arquitetura dual")
    if args.texto_decomp != "none" and args.variante == "dual":
        raise SystemExit("--texto-decomp exige --variante-arquitetura "
                         "original (decomposicao textual, sem dual embeds)")
    from .ceemdan.decomposicao import set_ceemdan_backend
    set_ceemdan_backend(getattr(args, "ceemdan_backend", "pyemd"))
    if (args.ceemdan_backend == "hip"
            and args.texto_decomp not in ("none", "stl", "mstl", "ceemdan")):
        raise SystemExit("--ceemdan-backend hip suporta apenas "
                         "--texto-decomp none/stl/mstl/ceemdan (os demais "
                         "usam outro backend; vmd = vmdpy/CPU puro)")
    decomp_param = "stl_simple" if args.decomp_mode == "stl_simple" else "linear"
    if args.variante == "dual":
        partes = ["dual"]
        if args.dual_fusao != "glu":
            partes.append("soma")
        if args.decomp_mode == "stl_simple":
            partes.append("stl")
        sufixo = "_".join(partes)
    elif args.texto_decomp == "stl":
        sufixo = "txt_stl"
    elif args.texto_decomp == "mstl":
        sufixo = "txt_mstl"
    elif args.texto_decomp == "vmd":
        sufixo = "txt_vmd"
    else:
        sufixo = ""
    # Variantes de texto (--texto-data/decimais/encoding) e --so-modelo entram
    # no nome do JSON: as rodadas A/B ficam reconheciveis e nunca misturam os
    # vereditos. No padrao ambos os trechos sao vazios -> nome historico.
    _var_txt = _formato_de_texto(args).tag_cache()  # ex.: "_txt-dhora"
    if _var_txt:
        sufixo += _var_txt if sufixo else _var_txt.lstrip("_")
    if args.so_modelo != "ambos":
        sufixo += ("_" if sufixo else "") + args.so_modelo
    JSON_OUT = (OUT_DIR / f"veredito_trio_{sufixo}_{_ts}.json" if sufixo
                else OUT_DIR / f"veredito_trio_{_ts}.json")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Nenhuma seed valida em --seeds")
    rb.seed_everything(seeds[0])

    if args.feats.strip().lower() == "all":
        args.feats_list = list(range(N_FEATS))
    else:
        args.feats_list = [int(f.strip()) for f in args.feats.split(",") if f.strip()]
        if not args.feats_list:
            raise SystemExit("Nenhuma feature valida em --feats")

    device = rb.get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")

    # ---- Charts-only -------------------------------------------------- #
    if args.charts_only:
        if not JSON_OUT.exists():
            raise SystemExit(f"JSON nao encontrado: {JSON_OUT}")
        resumo = json.load(open(JSON_OUT, encoding="utf-8"))
        imprimir_tabela(resumo)
        sfx, tit = _sufixo_titulo_do_resumo(resumo)
        for p in gerar_graficos(resumo, sufixo_grafico=sfx, titulo_extra=tit,
                                out_dir=RECENTE_DIR):
            print(f"[ok] grafico: {p}")
        return

    # ---- Tokenizador -------------------------------------------------- #
    from transformers import AutoTokenizer

    tok_dir = caminhos.descobrir_tokenizador()
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    print(f"[tokenizer] carregado de: {tok_dir} (vocab={len(tokenizer):,})")

    # ---- Pool z-score canonico (1x, cache em disco) ------------------- #
    print("\n" + "=" * 100)
    print(f"POOL Z-SCORE CANONICO (Channel Independence, {len(args.feats_list)} "
          f"features: {args.feats_list})")
    print(f"borders de linhas: train [0,8640) | val [7968,11520) | test "
          f"[10848,14400) (b1s={B1S} b2s={B2S})")
    print("=" * 100)
    splits, scaler, z, b1s, b2s = carregar_pool_cache(args)
    n_janelas = [len(x) for x, _, _ in splits]
    print(f"[pool] janelas z-score: train={n_janelas[0]} val={n_janelas[1]} "
          f"test={n_janelas[2]} (total={sum(n_janelas)})")
    print(f"[zscore] StandardScaler ajustado nas linhas [0, {B2S[0]}) "
          f"(todas as colunas, por coluna)")

    # ---- Decode (escala) ---------------------------------------------- #
    decode_fn = None
    if args.decode:
        pl = CharByCharPipeline(tokenizer, block_max=args.block_max,
                                n_paginas_max=args.n_paginas_max,
                                overlap_frac=args.overlap)
        decode_fn = pl.decode
        print("[decode] ATIVO: metricas do char-by-char apos pred * max_abs "
              "(variante que INFLARIA o MSE por ~max_abs^2 - ver docstring)")
    else:
        print("[decode] OFF (padrao): o modelo ja devolve a reconstrucao na "
              "escala de entrada (z-score, RevIN interno denormaliza). "
              "--decode deslocaria as metricas para fora do espaco z.")

    # ---- Checkpoints -------------------------------------------------- #
    ckpts_sel: Dict[str, str] = {}
    for tok in args.checkpoints.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in CKPTS:
            ckpts_sel[tok] = str(CKPTS[tok])
        else:
            p = Path(tok)
            if not p.exists():
                raise SystemExit(f"Checkpoint nao encontrado: {p}")
            ckpts_sel[p.stem] = str(p)
    ckpt_path = Path(next(iter(ckpts_sel.values())))

    # ---- Modelo 1: baseline numerico ---------------------------------- #
    print("\n" + "=" * 100)
    print("MODELO 1 - BASELINE NUMERICO (Run-8 VolLoss + SMETimes)")
    print("=" * 100)
    numerico: Dict
    if args.re_numerico:
        mses, maes = [], []
        for seed in seeds:
            rb.seed_everything(seed)
            r = avaliar_numerico(splits, ckpt_path, device, use_amp)
            mses.append(r["test_mse"]); maes.append(r["test_mae"])
            print(f"  [seed={seed}] eval-only: test_mse={r['test_mse']:.6f} "
                  f"test_mae={r['test_mae']:.6f}")
        numerico = {
            "fonte": "eval_only_canonico_7feats",
            "test_mse_media": round(float(np.mean(mses)), 6),
            "test_mse_desvio": round(float(np.std(mses)), 6),
            "test_mae_media": round(float(np.mean(maes)), 6),
            "test_mae_desvio": round(float(np.std(maes)), 6),
            "n_janelas_test": int(n_janelas[2]),
        }
    else:
        log = carregar_numerico_log()
        if log is None:
            print("[numerico] log nao encontrado -> re-medindo (eval-only)")
            log = {"test_mse_media": 0.0, "test_mae_media": 0.0,
                   "n_janelas_test": 0, "referencia_artigo": 0.3539}
            args.re_numerico = True
            mses, maes = [], []
            for seed in seeds:
                r = avaliar_numerico(splits, ckpt_path, device, use_amp)
                mses.append(r["test_mse"]); maes.append(r["test_mae"])
            log = {"test_mse_media": float(np.mean(mses)),
                   "test_mae_media": float(np.mean(maes)),
                   "n_janelas_test": int(n_janelas[2]),
                   "referencia_artigo": 0.3539}
        numerico = {
            "fonte": "log_numerico_base_stats.json",
            "test_mse_media": log["test_mse_media"],
            "test_mse_desvio": 0.0,
            "test_mae_media": log["test_mae_media"],
            "test_mae_desvio": 0.0,
            "n_janelas_test": int(log["n_janelas_test"]),
            "referencia_artigo": log["referencia_artigo"],
        }
        print(f"[numerico] carregado do log: test_mse={numerico['test_mse_media']:.4f} "
              f"| test_mae={numerico['test_mae_media']:.4f} | "
              f"n_janelas_test={numerico['n_janelas_test']}")

    # ---- Dados dos modelos textuais (textos 1x, cache em disco) ------- #
    print("\n" + "=" * 100)
    print("TEXTOS - PSEUDO-JSON ES (documento fixo do treino) e CHAR-BY-CHAR "
          "(documento por janela)")
    print("=" * 100)
    dados = carregar_etth1(args.csv, args.linhas)

    # Detectar periodos MSTL ANTES do pseudo-json (necessario para o cache)
    periodos_mstl = None
    mstl_info = None
    if args.texto_decomp in ("mstl", "mstl+ceemdan"):
        csv_path = args.csv
        if csv_path is None:
            from . import caminhos as _cam
            csv_path = str(_cam.DATA_DIR / _cam.DATA_PATH)
        _, _, mstl_info = detectar_periodos_mstl(csv_path)
        if args.mstl_periods is not None:
            periodos_mstl = tuple(int(x.strip()) for x in args.mstl_periods.split(","))
            src = "CUSTOM (--mstl-periods)"
        else:
            periodos_mstl = (mstl_info["periodo_diario"], mstl_info["periodo_semanal"])
            src = "AUTO-DETECTADOS"
        print(f"\n{'=' * 100}")
        print(f"MSTL - PERIODOS {src}")
        print(f"{'=' * 100}")
        print(f"  CSV: {csv_path}")
        print(f"  Timestep: {mstl_info['minutos_timestep']:.0f} minutos "
              f"({mstl_info['passos_por_hora']:.1f} passos/hora)")
        print(f"  Periodos: {periodos_mstl}")
        print(f"  Desc: MSTL( periods={periodos_mstl} ) -> T + S + R")
        print(f"  Window: 672 pontos | todos cabem: "
              f"{all(p < 336 for p in periodos_mstl)}")
        print(f"{'=' * 100}")
    elif args.texto_decomp in ("ceemdan", "paralelo"):
        print(f"\n{'=' * 100}")
        print(f"CEEMDAN - decomposicao adaptativa (sem periodos fixos)")
        print(f"{'=' * 100}")
        print(f"  Trials: {args.ceemdan_trials}")
        print(f"  Epsilon: {args.ceemdan_epsilon}")
        print(f"  Max IMF: {args.ceemdan_max_imf}")
        print(f"  Seed: {args.ceemdan_seed}")
        print(f"  Backend: {args.ceemdan_backend}")
        print(f"{'=' * 100}")
    elif args.texto_decomp == "vmd":
        _imprimir_banner_vmd(args)
    _imprimir_banner_formato(args, tokenizer, dados, z)

    roda_pj = args.so_modelo in ("ambos", "pj")
    roda_cb = args.so_modelo in ("ambos", "cbc")
    if not (roda_pj and roda_cb):
        print(f"[so-modelo] roda SOMENTE o modelo textual "
              f"{'Pseudo-JSON' if roda_pj else 'Char-by-Char'} "
              f"({args.so_modelo}); o outro nao e gerado nem treinado")
    prep_pj = None
    stats_cb = None
    loaders_pj = loaders_cb = None
    if roda_pj:
        prep_pj = carregar_pseudo_json(args, tokenizer, dados, z,
                                       periodos_mstl=periodos_mstl)
        tag_pj = args.texto_decomp.upper().replace("+", "_") if args.texto_decomp != "none" else "ES"
        print(f"[pseudo-json {tag_pj}] doc do treino: {prep_pj['tokens_originais']:,} tokens "
              f"originais -> {prep_pj['tokens_retidos']:,} retidos | "
              f"{prep_pj['n_paginas']} paginas")
        ds_pj = datasets_pseudo_json(args, splits, prep_pj)
        loaders_pj = build_loaders(ds_pj[0], ds_pj[1], ds_pj[2], args.batch_size)

    if roda_cb:
        ds_cb, stats_cb = carregar_char_by_char(args, tokenizer, splits,
                                                periodos_mstl=periodos_mstl,
                                                mstl_info=mstl_info)
        loaders_cb = build_loaders(ds_cb[0], ds_cb[1], ds_cb[2], args.batch_size)

    # ================================================================= #
    #  BATERIA - 10 epocas x 3 seeds por modelo textual                  #
    # ================================================================= #
    print("\n" + "=" * 100)
    print(f"BATERIA TEXTUAL - {args.epochs} EPOCAS x {len(seeds)} SEEDS "
          f"(isolamento estrito, restore_best por validacao)")
    is_dual = args.variante == "dual"
    if is_dual:
        desc = ("trend+osc GLU gate" if args.dual_fusao == "glu"
                else "trend+osc soma (ablation gating)")
        desc += (f" | decomp={args.decomp_mode}"
                 if args.decomp_mode != "none" else "")
        print(f"  [variante] AutoTimesDualModel ({desc})")
    else:
        desc = "AutoTimesModel original"
        if args.texto_decomp == "stl":
            desc += " + texto STL (Tendencia/Sazonalidade/Residuos, LOESS p24)"
        elif args.texto_decomp == "mstl":
            desc += " + texto MSTL (Tendencia/Sazonalidade/Residuos, periodos dinamicos)"
        elif args.texto_decomp == "ceemdan":
            desc += f" + texto CEEMDAN puro (trials={args.ceemdan_trials}, eps={args.ceemdan_epsilon}, max_imf={args.ceemdan_max_imf})"
        elif args.texto_decomp == "mstl+ceemdan":
            desc += f" + texto MSTL+CEEMDAN hibrido (trials={args.ceemdan_trials}, max_imf={args.ceemdan_max_imf})"
        elif args.texto_decomp == "paralelo":
            desc += f" + texto Paralelo (MSTL p/ T, CEEMDAN p/ S, trials={args.ceemdan_trials}, max_imf={args.ceemdan_max_imf})"
        elif args.texto_decomp == "vmd":
            desc += (f" + texto VMD (K={args.vmd_k} modos T/S/R, "
                     f"alpha={args.vmd_alpha:g}, tol={args.vmd_tol:g}, "
                     f"init={args.vmd_init}, trend MA={_rotulo_vmd_ma(args)})")
        if args.linear_probing:
            desc += " + Linear Probing (decoder substituido)"
        if args.replace_decoder:
            desc += " + Fresh MLP Decoder (treinada do zero)"
        if args.kan_decoder:
            desc += f" + KAN Decoder (act={args.kan_act})"
        if args.taylor_decoder:
            desc += f" + TaylorKAN Decoder (order={args.taylor_order})"
        if args.residual_adapter != "none":
            adapter_names = {"linear": "Linear", "taylor": "TaylorKAN", "mole": "MoLE"}
            desc += (f" + {adapter_names[args.residual_adapter]} Residual Adapter "
                     f"(MLP frozen, order={args.residual_order}, "
                     f"alpha={args.residual_alpha})")
        if args.unfreeze_encoder_epoch > 0:
            desc += f" + Two-Phase (encoder descongela ep {args.unfreeze_encoder_epoch})"
        print(f"  [variante] {desc}")
    print("=" * 100)
    if is_dual:
        model_fn = functools.partial(
            montar_modelo_dual, fusao=args.dual_fusao, decomp=decomp_param,
        )
    elif args.linear_probing:
        model_fn = functools.partial(montar_modelo, linear_probing=True)
    elif args.replace_decoder:
        model_fn = functools.partial(montar_modelo, replace_decoder=True)
    elif args.kan_decoder:
        model_fn = functools.partial(montar_modelo, kan_decoder=True, kan_act=args.kan_act)
    elif args.taylor_decoder:
        model_fn = functools.partial(montar_modelo, taylor_decoder=True,
                                     taylor_order=args.taylor_order,
                                     taylor_no_tanh=args.taylor_no_tanh)
    elif args.residual_adapter != "none":
        model_fn = functools.partial(
            montar_modelo,
            residual_adapter=args.residual_adapter,
            residual_order=args.residual_order,
            residual_alpha=args.residual_alpha,
        )
    else:
        model_fn = montar_modelo
    res_pj = rodar_bateria_textual(args, "Pseudo-JSON ES", loaders_pj,
                                   ckpt_path, device, use_amp, seeds,
                                   model_fn=model_fn, sufixo=sufixo) if roda_pj else None
    res_cb = rodar_bateria_textual(args, "Char-by-Char", loaders_cb,
                                   ckpt_path, device, use_amp, seeds,
                                   decode_fn=decode_fn, model_fn=model_fn,
                                   sufixo=sufixo) if roda_cb else None

    # ---- Consolidacao ------------------------------------------------- #
    resumo: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocolo": "apples_to_apples_trio",
        "escala": "zscore",
        "config": {
            "variante_arquitetura": args.variante,
            "dual_fusao": args.dual_fusao if args.variante == "dual" else None,
            "decomp_mode": args.decomp_mode if args.variante == "dual" else "none",
            "texto_decomp": args.texto_decomp,
            "vmd": _config_vmd(args) if args.texto_decomp == "vmd" else None,
            "mstl_info": mstl_info if args.texto_decomp == "mstl" else None,
            "feats": args.feats_list,
            "n_features": len(args.feats_list),
            "splits_linhas": {"train": [B1S[0], B2S[0]],
                              "val": [B1S[1], B2S[1]],
                              "test": [B1S[2], B2S[2]]},
            "janelas_por_split": dict(zip(("train", "val", "test"), n_janelas)),
            "scaler_fit": f"[0, {B2S[0]}) todas as colunas",
            "scaler_mean_por_feat": [round(float(m), 6) for m in scaler.mean_],
            "scaler_std_por_feat": [round(float(s), 6) for s in scaler.scale_],
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "block_max": args.block_max,
            "n_paginas_max": args.n_paginas_max,
            "overlap": args.overlap,
            "checkpoint": str(ckpt_path),
            "checkpoint_dir": str(CKPT_DIR),
            "restore_best": True,
            "decode_aplicado": bool(args.decode),
            "amp": use_amp,
            "linhas_csv": args.linhas,
        },
        "modelos": {"numerico": numerico},
    }
    resumo["config"]["so_modelo"] = args.so_modelo
    resumo["config"]["self_refine"] = bool(getattr(args, "self_refine", False))
    if resumo["config"]["self_refine"]:
        resumo["config"]["self_refine_max_tentativas"] = int(
            getattr(args, "self_refine_max_tentativas", 3))
    if roda_pj:
        resumo["modelos"]["pseudo_json_es"] = dict(res_pj, **{
            "template": TEMPLATE_PSEUDO_JSON,
            "estrategia": ESTRATEGIA_PAGINACAO,
            "tokens_originais": prep_pj["tokens_originais"],
            "tokens_retidos": prep_pj["tokens_retidos"],
            "paginas": prep_pj["n_paginas"],
            "paginas_descartadas": prep_pj["paginas_descartadas"],
        })
    if roda_cb:
        resumo["modelos"]["char_by_char"] = dict(res_cb, **{
            "max_abs_media": round(float(np.mean([s["max_abs_media"] for s in stats_cb.values()])), 6) if stats_cb else None,
            "max_abs_desvio": round(float(np.mean([s["max_abs_desvio"] for s in stats_cb.values()])), 6) if stats_cb else None,
            "paginas_media": round(float(np.mean([s["paginas_media"] for s in stats_cb.values()])), 2) if stats_cb else None,
            "paginas_descartadas_media": round(float(np.mean([s["paginas_descartadas_media"] for s in stats_cb.values()])), 2) if stats_cb else None,
            "decode_aplicado": bool(args.decode),
        })
    imprimir_tabela(resumo)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"[ok] JSON: {JSON_OUT}")

    # Copiar JSON para recente/logs/ tambem
    logs_dir = RECENTE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    shutil.copy2(JSON_OUT, logs_dir / JSON_OUT.name)
    print(f"[ok] JSON copiado: {logs_dir / JSON_OUT.name}")
    if args.plots:
        sfx = ("_" + sufixo) if sufixo else ""
        tit = ""
        if args.variante == "dual":
            tag = "Dual " + ("GLU" if args.dual_fusao == "glu" else "SOMA")
            if args.decomp_mode == "stl_simple":
                tag += " + STL-k25"
            tit = f" [{tag}]"
        elif args.texto_decomp == "stl":
            tit = " [Original + Texto STL (Tendencia/Sazonalidade/Residuos, LOESS p24)]"
        elif args.texto_decomp == "mstl":
            tit = " [Original + Texto MSTL (Tendencia/Sazonalidade/Residuos, periodos dinamicos)]"
        elif args.texto_decomp == "vmd":
            sfx, tit = _sufixo_titulo_do_resumo(resumo)
        for p in gerar_graficos(resumo, sufixo_grafico=sfx, titulo_extra=tit,
                                out_dir=RECENTE_DIR):
            print(f"[ok] grafico: {p}")
    print("Confronto Apples-to-Apples concluido.")


if __name__ == "__main__":
    main()