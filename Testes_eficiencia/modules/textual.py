"""
textual - Serializacao textual pura dos dados para o pipeline AutoTimes.

Converte os inputs numericos (datas, horas, valores) das janelas do dataset em
descricoes em linguagem natural por extenso (pt-BR), para alimentar os modulos
de embedding do LLM (gpt2.wte) diretamente - sem misturar tensores numericos
no fluxo.

Fluxo legado (usado pelo run_benchmarks.py --textual):
    gerar_texto_janela => blocos de texto por janela de 672 passos; a
    tokenizacao e cache .pt ficam no run_benchmarks (load_textual_split).

Fluxo novo (gerador de dataset lookback/forecast):
    gerar_dataset_csv(df, lookback=96, forecast=96, step=1, output_file)
        - janela deslizante; por janela gera input_text (historico do
          lookback) e target_text (narrativa dos proximos `forecast` passos)
        - exporta CSV com colunas "input_text" e "target_text"
    tokenizar_para_pt(dataset, tokenizer, output_file)
        - mesma base textual, tokenizada com o GPT-2 tokenizer e salva em .pt

Exemplo da representacao textual gerada:
    "Na sessao iniciada em dez de julho de dois mil e vinte e dois, as
     catorze horas, da serie OTELETE, os valores registrados foram: as
     vinte horas e doze minutos o valor era vinte e nove virgula cinco; ..."
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- #
#  Numeros por extenso (pt-BR)                                           #
# --------------------------------------------------------------------- #
_MESES_PT: List[str] = [
    "janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho",
    "agosto", "setembro", "outubro", "novembro", "dezembro",
]
_UNIDADES: List[str] = [
    "zero", "um", "dois", "tres", "quatro", "cinco", "seis", "sete", "oito",
    "nove", "dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis",
    "dezessete", "dezoito", "dezenove",
]
_DEZENAS: List[str] = [
    "", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta",
    "oitenta", "noventa",
]
_CENTENAS: List[str] = [
    "", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
    "seiscentos", "setecentos", "oitocentos", "novecentos",
]


def numero_por_extenso(n: int) -> str:
    """Inteiro (>=0) por extenso em pt-BR (ate alguns milhoes)."""
    return _numero_por_extenso(int(n))


def _ate_99(n: int) -> str:
    if n < 20:
        return _UNIDADES[n]
    d, r = divmod(n, 10)
    return _DEZENAS[d] if r == 0 else f"{_DEZENAS[d]} e {_UNIDADES[r]}"


def _centenas(n: int) -> str:
    if n == 100:
        return "cem"
    c, r = divmod(n, 100)
    out = _CENTENAS[c]
    return out if r == 0 else f"{out} e {_ate_99(r)}"


def _numero_por_extenso(n: int) -> str:
    if n < 0:
        return f"menos {_numero_por_extenso(-n)}"
    if n < 100:
        return _ate_99(n)
    if n < 1000:
        return _centenas(n)

    mi, resto = divmod(n, 1_000_000)
    mil, uni = divmod(resto, 1000)

    partes: List[str] = []
    if mi:
        partes.append(
            "um milhao" if mi == 1 else
            f"{_centenas(mi) if mi < 1000 else _numero_por_extenso(mi)} milhoes"
        )
    if mil == 1:
        partes.append("mil")
    elif mil:
        partes.append(f"{_ate_99(mil) if mil < 100 else _centenas(mil)} mil")
    if uni:
        partes.append(_ate_99(uni) if uni < 100 else _centenas(uni))

    if not partes:
        return "zero"
    if len(partes) == 1:
        return partes[0]
    return " ".join(partes[:-1]) + f" e {partes[-1]}"


def valor_por_extenso(v: float) -> str:
    """Valor float por extenso (1 casa decimal): 'vinte e nove virgula dois'."""
    s = f"{v:.1f}"
    inteiro, dec = s.split(".")
    out = numero_por_extenso(inteiro)
    if int(dec):
        out = f"{out} virgula {numero_por_extenso(dec)}"
    return out


def descrever_momento(ano: int, mes: int, dia: int, hora: int, minuto: int) -> str:
    """'dez de julho de dois mil e vinte e dois, as quatorze horas' (pt-BR)."""
    data = f"{numero_por_extenso(dia)} de {_MESES_PT[mes - 1]} de {numero_por_extenso(ano)}"
    horas = numero_por_extenso(hora)
    if minuto == 0:
        return f"{data}, as {horas} horas"
    return f"{data}, as {horas} horas e {numero_por_extenso(minuto)} minutos"


# ---------------------------------------------------------------------------
#  Geracao de texto por janela (formato legado usado pelo run_benchmarks)     #
# ---------------------------------------------------------------------------
def _ts_to_dt(ts: np.datetime64) -> datetime:
    return datetime.utcfromtimestamp(
        ts.astype("datetime64[ns]").astype(np.uint64) // 1_000_000_000
    )


def gerar_texto_janela(
    dates: np.ndarray,
    values: np.ndarray,
    feat: int,
    start: int,
    seq_len: int,
    token_len: int,
    feat_name: str,
    text_step: int = 12,
) -> List[str]:
    """Descricao textual de UMA janela, um texto por bloco (token/patch).

    Formato usado pelo ablation (seq_len 672, token_len 96). Nao alterar
    assinatura: consumida por run_benchmarks.build_textoval_split.
    """
    n_blocks = seq_len // token_len
    textos = []
    for b in range(n_blocks):
        b_start = start + b * token_len
        dt0 = _ts_to_dt(dates[b_start])
        partes = [
            f"Na sessao iniciada em {descrever_momento(dt0.year, dt0.month, dt0.day, dt0.hour, dt0.minute)}, "
            f"da serie {feat_name}, os valores registrados foram:"
        ]
        for j in range(0, token_len, text_step):
            idx = b_start + j
            if idx >= values.shape[0]:
                break
            dtj = _ts_to_dt(dates[idx])
            horas = numero_por_extenso(dtj.hour)
            tempo = (f"as {horas} horas" if dtj.minute == 0 else
                     f"as {horas} horas e {numero_por_extenso(dtj.minute)} minutos")
            partes.append(
                f"{tempo} o valor era {valor_por_extenso(float(values[idx, feat]))}"
            )
        partes.append(".")
        textos.append(" ".join(partes))
    return textos


# --------------------------------------------------------------------------- #
#  NOVO: gerador de dataset (janela deslizante lookback -> forecast)          #
# --------------------------------------------------------------------------- #
def _tempos_serie(dates: np.ndarray) -> List[str]:
    """'as h horas' / 'as h horas e m minutos' de cada timestamp (1x, pre-cache)."""
    cache: dict = {}
    tempos: List[str] = []
    for ts in dates:
        dt = _ts_to_dt(ts)
        key = (dt.hour, dt.minute)
        if key not in cache:
            horas = numero_por_extenso(dt.hour)
            cache[key] = (
                f"as {horas} horas" if dt.minute == 0 else
                f"as {horas} horas e {numero_por_extenso(dt.minute)} minutos"
            )
        tempos.append(cache[key])
    return tempos


def _sessoes_serie(dates: np.ndarray) -> List[str]:
    """'Na sessao iniciada em ...' por sample (independe da feature)."""
    sessoes: List[str] = []
    for ts in dates:
        dt = _ts_to_dt(ts)
        sessoes.append(
            f"Na sessao iniciada em "
            f"{descrever_momento(dt.year, dt.month, dt.day, dt.hour, dt.minute)},"
        )
    return sessoes


def _texto_bloco(
    sessoes: List[str],
    tempos: List[str],
    textos_valores: List[str],
    feat_name: str,
    inicio: int,
    fim: int,
    step: int,
    futuro: bool,
) -> str:
    """Narrativa textual de um trecho [inicio:fim] da serie (1 unica string)."""
    tempo = "previstos" if futuro else "registrados"
    cab = f"{sessoes[inicio]} da serie {feat_name}, os valores {tempo} foram:"
    selecao = range(inicio, fim, step)
    if futuro:
        frases = [f"{tempos[i]} o valor sera {textos_valores[i]}" for i in selecao]
    else:
        frases = [f"{tempos[i]} o valor era {textos_valores[i]}" for i in selecao]
    return cab + " " + " ".join(frases) + "."


def gerar_dataset_csv(
    df: pd.DataFrame,
    lookback: int = 96,
    forecast: int = 96,
    step: int = 1,
    output_file: Optional[str] = "dataset.csv",
    text_step: int = 1,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Gera dataset textual por janela deslizante e exporta CSV.

    Para cada coluna numerica e cada janela, produz DOIS textos:
        input_text  : narrativa dos ultimos `lookback` pontos (historico).
        target_text : narrativa dos proximos `forecast` pontos (previsao).

    O CSV final tem exatamente as colunas "input_text" e "target_text".

    Args:
        df: DataFrame com datas na 1a coluna e series numeric nas demais.
        lookback: tamanho do bloco de entrada.
        forecast: tamanho do bloco de saida.
        step: stride da janela deslizante (1 = todas).
        output_file: caminho do CSV gerado; None desativa o disco.
        text_step: amostra cada N pontos dentro do bloco (para caber no
            contexto do GPT-2); 1 = todos os pontos.

    Returns:
        DataFrame (n_janelas * n_features, 2) [input_text, target_text].
    """
    dates = pd.to_datetime(df.iloc[:, 0]).to_numpy().astype("datetime64[ns]")
    if features is not None:
        values = df[features].to_numpy(dtype=np.float64)
        feat_names = list(features)
    else:
        values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        feat_names = list(df.columns[1:])
    n_total, n_vars = values.shape

    lookback = int(lookback)
    forecast = int(forecast)
    if n_total < lookback + forecast:
        raise ValueError(
            f"Serie curta demais: {n_total} < lookback+forecast "
            f"({lookback + forecast})"
        )
    tempos = _tempos_serie(dates)
    sessoes = _sessoes_serie(dates)

    linhas = []
    for f in range(n_vars):
        # valores por extenso da feature inteira, UMA vez (nao por janela)
        textos_valores = [valor_por_extenso(v) for v in values[:, f]]
        for s in range(0, n_total - lookback - forecast + 1, step):
            fim_in = s + lookback
            input_text = _texto_bloco(
                sessoes, tempos, textos_valores, feat_names[f],
                inicio=s, fim=fim_in, step=text_step, futuro=False,
            )
            target_text = _texto_bloco(
                sessoes, tempos, textos_valores, feat_names[f],
                inicio=fim_in, fim=fim_in + forecast, step=text_step, futuro=True,
            )
            linhas.append((input_text, target_text))

    dset = pd.DataFrame(linhas, columns=["input_text", "target_text"])
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        dset.to_csv(output_file, index=False, encoding="utf-8")
        print(f"[textual] {len(dset)} pares => {output_file}")
    return dset


# ---------------------------------------------------------------------------- #
#  Export .pt: mesma seq texto, tokenizada com o GPT-2 tokenizer                #
# ---------------------------------------------------------------------------- #
def tokenizar_para_pt(
    dataset: pd.DataFrame,
    tokenizer,
    output_file: str = "dataset.pt",
    block_max: int = 128,
) -> None:
    """Tokeniza os pares input/target do dataset e salva os tensores em .pt.

    Cada texto eh truncado em block_max e preenchido (pad) ate o limite, como
    o formato texto do run_benchmarks (x:(n, T*B), x_len:(n, T)). Aqui:
        input_ids/input_lens + target_ids/target_lens (n, block_max).

    Args:
        dataset: DataFrame de gerar_dataset_csv.
        tokenizer: tokenizer GPT-2 (pad = eos/50256).
        output_file: caminho do .pt.
        block_max: limite de tokens por texto.
    """
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    pad = int(tokenizer.pad_token_id)

    def _codificar(col: pd.Series) -> Tuple[List[List[int]], List[int]]:
        ids_all: List[List[int]] = []
        lens: List[int] = []
        for txt in col:
            ids = tokenizer(
                str(txt), add_special_tokens=False,
                truncation=True, max_length=block_max)["input_ids"]
            lens.append(len(ids))
            ids_all.append(ids[:block_max] + [pad] * (block_max - len(ids)))
        return ids_all, lens

    input_ids, input_lens = _codificar(dataset["input_text"])
    target_ids, target_lens = _codificar(dataset["target_text"])

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "input_lens": torch.tensor(input_lens, dtype=torch.int64),
            "target_ids": torch.tensor(target_ids, dtype=torch.int64),
            "target_lens": torch.tensor(target_lens, dtype=torch.int64),
        },
        output_file,
    )
    print(f"[textual] {len(dataset)} pares tokenizados => {output_file}")


# --------------------------------------------------------------------------- #
#  CLI: python -m modules.textual <dataset.csv>                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gera dataset textual (CSV e .pt)")
    parser.add_argument("data_path", type=str, help="CSV numerico (ex.: ETTh1.csv)")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--forecast", type=int, default=96)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--text-step", type=int, default=1)
    parser.add_argument("--output", type=str, default="dataset.csv")
    parser.add_argument("--export-pt", type=str, default=None,
                        help="se informado, tokeniza e salva .pt (ex.: dataset.pt)")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--block-max", type=int, default=128)
    args = parser.parse_args()

    df_raw = pd.read_csv(args.data_path)
    pares = gerar_dataset_csv(
        df_raw,
        lookback=args.lookback,
        forecast=args.forecast,
        step=args.step,
        text_step=args.text_step,
        output_file=args.output,
    )
    if args.export_pt:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizar_para_pt(pares, tok, args.export_pt, block_max=args.block_max)