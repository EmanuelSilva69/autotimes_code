"""
dados - carrega o CSV numerico, monta os TensorDatasets textuais por template
e mede a compressao de tokens (Testador de Tokenizacao e Paginacao).

Replica fielmente o split do pipeline (run_benchmarks.load_real_split /
load_textual_split):
    - janelas UNIVARIADAS por variavel (feat_id = index // tot_len);
    - seq_x = data[s_begin : s_begin+seq_len]  (672 passos);
    - seq_y = data[r_begin : r_end] com r_begin = s_end - label_len
      (label 576 + pred 96 = 672 passos, reconstrucao completa);
    - y na escala REAL (sem z-score; o RevIN normaliza dentro do modelo).

A diferenca e que o texto de cada bloco vem do TEMPLATE escolhido
(pt_verboso / pt_curto / en_curto / json), com text_step auto-ajustado
para caber no block_max (128 tokens/bloco).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from . import caminhos
from .templates import TEMPLATES, escolher_text_step, gerar_textos_janela

SEQ_LEN: int = 672
LABEL_LEN: int = 576
PRED_LEN: int = 96
TOKEN_LEN: int = SEQ_LEN - LABEL_LEN
BLOCK_MAX: int = 128


# --------------------------------------------------------------------- #
#  Carregamento do CSV numerico                                         #
# --------------------------------------------------------------------- #
def carregar_dados(
    data_dir: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Dict:
    """Le o CSV numerico e devolve datetimes, valores e bounds dos splits.

    Returns:
        Dict com: dt_list (List[datetime]), values (np.float32), feat_names,
        n_total, n_vars, b1s, b2s (bounds de train/val/test).
    """
    data_dir = Path(data_dir or caminhos.DATA_DIR)
    data_path = data_path or caminhos.DATA_PATH
    csv = data_dir / data_path
    if not csv.exists():
        raise FileNotFoundError(f"Dataset numerico nao encontrado: {csv}")

    rb = caminhos.importar_run_benchmarks()
    df = pd.read_csv(csv)
    dt_list = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])).to_pydatetime().tolist()
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feat_names = list(df.columns[1:])
    n_total, n_vars = values.shape
    b1s, b2s = rb._compute_split_bounds(data_path.lower(), n_total, SEQ_LEN, PRED_LEN)
    return {
        "dt_list": dt_list,
        "values": values,
        "feat_names": feat_names,
        "n_total": n_total,
        "n_vars": n_vars,
        "b1s": b1s,
        "b2s": b2s,
        "data_dir": data_dir,
        "data_path": data_path,
    }


def _stride_para_cap(tot_len: int, cap: int) -> int:
    """Stride do janelamento para ~cap janelas por feature."""
    if tot_len <= cap:
        return 1
    return int(np.ceil(tot_len / cap))


# --------------------------------------------------------------------- #
#  Testador de Tokenizacao e Paginacao (compressao de prompt)           #
# --------------------------------------------------------------------- #
class TestadorTokenizacao:
    """Mede quantos tokens cada template consome e quanta informacao retem.

    Metrica-chave de compressao:
        tokens_por_ponto = tokens_por_janela / (pontos_por_bloco * n_blocos)
    e a taxa de truncamento (blocos que estouram block_max=128).
    """

    def __init__(self, tokenizer, block_max: int = BLOCK_MAX) -> None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
        self.tokenizer = tokenizer
        self.block_max = int(block_max)

    def contar_tokens(self, texto: str) -> int:
        return len(self.tokenizer(texto, add_special_tokens=False)["input_ids"])

    def escolher_step(self, template_id: str, dados: Dict) -> int:
        return escolher_text_step(
            template_id, self.tokenizer, dados["dt_list"], dados["values"],
            dados["feat_names"], SEQ_LEN, TOKEN_LEN, self.block_max,
        )

    def medir_split(
        self,
        template_id: str,
        dados: Dict,
        text_step: int,
        b1: int,
        b2: int,
        stride: int,
        max_janelas: Optional[int] = None,
    ) -> Dict:
        """Estatisticas de tokenizacao sobre as janelas de um split.

        Retorna agregados de tokens por janela/bloco, taxa de truncamento,
        densidade (pontos por bloco) e tempo de tokenizacao.
        """
        values, dt_list, feat_names = dados["values"], dados["dt_list"], dados["feat_names"]
        n_vars = dados["n_vars"]
        tot_len = len(values[b1:b2]) - SEQ_LEN - PRED_LEN + 1

        lens_janela: List[int] = []
        lens_bloco: List[int] = []
        n_truncados = 0
        n_blocos = 0
        n_janelas = 0
        pontos_por_bloco = None
        t0 = time.perf_counter()

        for feat in range(n_vars):
            for s_begin in range(0, tot_len, stride):
                if max_janelas is not None and n_janelas >= max_janelas:
                    break
                start = b1 + s_begin
                textos = gerar_textos_janela(
                    template_id, dt_list, values, feat, start, SEQ_LEN, TOKEN_LEN,
                    feat_names[feat], text_step,
                )
                soma = 0
                for texto in textos:
                    ids = self.tokenizer(
                        texto, add_special_tokens=False,
                        truncation=True, max_length=self.block_max,
                    )["input_ids"]
                    soma += len(ids)
                    lens_bloco.append(len(ids))
                    n_blocos += 1
                    if len(ids) >= self.block_max:
                        n_truncados += 1
                lens_janela.append(soma)
                n_janelas += 1

        # Pontos por bloco = densidade real da amostragem (range step).
        pontos_por_bloco = sum(1 for _ in range(0, TOKEN_LEN, text_step))
        arr_j = np.asarray(lens_janela, dtype=np.float64)
        arr_b = np.asarray(lens_bloco, dtype=np.float64)
        return {
            "template": template_id,
            "nome": TEMPLATES[template_id]["nome"],
            "text_step": text_step,
            "n_janelas": n_janelas,
            "n_blocos": n_blocos,
            "pontos_por_bloco": int(round(TOKEN_LEN / max(text_step, 1))),
            "tokens_por_janela": {
                "media": float(arr_j.mean()) if len(arr_j) else 0.0,
                "mediana": float(np.median(arr_j)) if len(arr_j) else 0.0,
                "max": float(arr_j.max()) if len(arr_j) else 0.0,
            },
            "tokens_por_bloco": {
                "media": float(arr_b.mean()) if len(arr_b) else 0.0,
                "max": float(arr_b.max()) if len(arr_b) else 0.0,
            },
            "tokens_por_ponto": float(arr_j.sum() / max(n_blocos * pontos_por_bloco, 1))
            if len(arr_j) else 0.0,
            "taxa_truncamento": float(n_truncados / max(n_blocos, 1)),
            "tempo_tokenizacao_s": round(time.perf_counter() - t0, 3),
        }


# --------------------------------------------------------------------- #
#  Construcao dos TensorDatasets (x, ids, lens, y)                      #
# --------------------------------------------------------------------- #
def construir_datasets(
    template_id: str,
    tokenizer,
    dados: Dict,
    text_step: int,
    caps: Optional[Dict[str, int]] = None,
    block_max: int = BLOCK_MAX,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Monta (train, val, test) no formato do run_benchmarks textual.

    Cada sample:
        x    : (seq_len, 1) sinal numerico bruto da janela (multimodal);
        ids  : (n_blocks * block_max) tokens GPT-2, 1 bloco por patch;
        lens : (n_blocks) comprimento real de cada bloco (p/ pooling);
        y    : (seq_len, 1) alvo REAL (label+pred), reconstrucao completa.

    Args:
        template_id: id do template (pt_verboso/pt_curto/en_curto/json).
        tokenizer: tokenizer GPT-2 local (pad = eos/50256).
        dados: saida de carregar_dados().
        text_step: amostragem por bloco (ver TestadorTokenizacao.escolher_step).
        caps: limites de janelas por split {train, val, test}.
        block_max: pad de tokens por bloco.

    Returns:
        (train_ds, val_ds, test_ds) TensorDatasets.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
    pad = int(tokenizer.pad_token_id)
    caps = caps or {"train": 192, "val": 64, "test": 64}

    values, dt_list, feat_names = dados["values"], dados["dt_list"], dados["feat_names"]
    n_vars = dados["n_vars"]
    b1s, b2s = dados["b1s"], dados["b2s"]

    def _build_split(b1: int, b2: int, cap: int) -> TensorDataset:
        tot_len = len(values[b1:b2]) - SEQ_LEN - PRED_LEN + 1
        stride = _stride_para_cap(tot_len, cap)
        xs, ids_all, lens_all, ys = [], [], [], []
        n_janelas = 0
        for feat in range(n_vars):
            for s_begin in range(0, tot_len, stride):
                if n_janelas >= cap:
                    break
                start = b1 + s_begin
                textos = gerar_textos_janela(
                    template_id, dt_list, values, feat, start, SEQ_LEN, TOKEN_LEN,
                    feat_names[feat], text_step,
                )
                ids_blocos: List[int] = []
                lens: List[int] = []
                for texto in textos:
                    ids = tokenizer(
                        texto, add_special_tokens=False,
                        truncation=True, max_length=block_max,
                    )["input_ids"]
                    lens.append(len(ids))
                    ids_blocos.extend(ids[:block_max])
                    ids_blocos.extend([pad] * (block_max - len(ids)))
                s_end = start + SEQ_LEN
                r_begin = s_end - LABEL_LEN
                r_end = r_begin + LABEL_LEN + PRED_LEN
                ys.append(values[r_begin:r_end, feat:feat + 1])
                xs.append(values[start:start + SEQ_LEN, feat:feat + 1])
                ids_all.append(ids_blocos)
                lens_all.append(lens)
                n_janelas += 1
        return TensorDataset(
            torch.from_numpy(np.stack(xs)).float(),
            torch.tensor(ids_all, dtype=torch.int64),
            torch.tensor(lens_all, dtype=torch.int64),
            torch.from_numpy(np.stack(ys)).float(),
        )

    train_ds = _build_split(b1s[0], b2s[0], caps["train"])
    val_ds = _build_split(b1s[1], b2s[1], caps["val"])
    test_ds = _build_split(b1s[2], b2s[2], caps["test"])
    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------- #
#  Persistencia dos resultados de tokenizacao                           #
# --------------------------------------------------------------------- #
def salvar_stats_tokenizacao(stats: Dict, path: Optional[Path] = None) -> Path:
    """Salva as estatisticas de tokens de todos os templates em JSON."""
    path = path or caminhos.RESULTADOS_DIR / "resumo_tokenizacao.json"
    caminhos.criar_resultados()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return path
