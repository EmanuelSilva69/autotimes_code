"""
pipeline_otimizacao_extenso - 4 fases arquiteturais para quebrar a barreira
do campeao "ES Keywords" (Test MSE 7.24 +/- 0.12, VolLoss + SMETimes).

O teto do prompt engineering foi atingido (bandas de confianca entre os
melhores idiomas se sobrepoem). Este script intervem na ARQUITETURA e na
forma como o GPT-2 (Decoder-Only) processa a matematica por extenso:

    FASE 1 - EXPANSAO DE VOCABULARIO (cirurgia no tokenizador)
        Palavras unicas do documento ES Keywords (numeros 0-100, meses,
        "coma", etc.) viram tokens ATOMICOS via tokenizer.add_tokens +
        model.gpt2.resize_token_embeddings(). Os numeros deixam de ser
        fragmentados em sub-tokens semantico-inuteis ("ve|int|in|ue|ve").

    FASE 2 - LoRA (biblioteca peft) nas camadas de atencao do GPT-2
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1) injetado em
        c_attn (qkv) de todas as 12 camadas. O backbone continua congelado
        (wte direto preservado); apenas lora_A/lora_B treinam (matrizes de
        baixo posto que ajustam a gramatica interna sem overfitting).

    FASE 3 - SOFT PROMPT continuo (prompt tuning)
        N vetores continuos treinaveis (default 10) concatenados a frente
        dos input_ids de cada janela (o conteudo da cauda do contexto e
        deslocado para manter o orcamento fixo). O gradiente descobre a
        "instrucao latente" ideal - sem palavras humanas.

    FASE 4 - COT CAUSAL (Chain-of-Thought numerico)
        Para cada janela, a derivada primeira do lookback e computada
        (ultimo vs primeiro valor) e a frase "Tendencia alcista. " ou
        "Tendencia bajista. " e prependida ao contexto textual - um vies
        causal explicito que poupa a LLM de deduzir a tendencia dos tokens.

MODULARIDADE (CLI):
    --use_vocab_expansion  --use_lora  --use_soft_prompt  --use_cot
    Cada flag ativa uma fase ISOLADA (ablacao); com 2+ flags, uma
    configuracao "combinada" (todas ativas) e adicionada. O baseline
    (ES Keywords puro) roda SEMPRE.

RIGOR (herdado da validacao final):
    - 3 seeds (2021, 42, 1234) x 10 epocas por configuracao;
    - isolamento estrito: seed_everything + montar_modelo DENTRO do loop
      mais interno (a cada iteracao seed/config); torch.cuda.empty_cache()
      ao fim de cada ciclo;
    - treinamento hibrido: todos os novos parametros (lora_A/B, soft
      prompt, linhas novas do wte) entram no Adam via model.parameters().

SAIDAS:
    resultados/benchmark_4fases_stats.json
    resultados/graficos/otimizacao_4fases_errorbar.png
"""

from __future__ import annotations

import argparse
import math
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from . import caminhos
from .modelos import CKPTS, montar_modelo
from .run_etth1_texto import (
    _FEAT_PADRAO,
    carregar_etth1,
    construir_dataset,
    pool_janelas_etth1,
    salvar_benchmark,
)
from .texto_puro import extenso
from .texto_puro.paginacao import ESTRATEGIAS, TestadorPaginacao
from .treino import build_loaders, rodar_convergencia

# Campeao atual (nivel publicacao): ES Keywords + Compressao.
TEMPLATE_CAMPEAO: str = "es_keywords"
ESTRATEGIA_CAMPEA: str = "keywords"

# Frases causais da FASE 4 (CoT).
FRASE_ALCISTA: str = "Tendencia alcista. "
FRASE_BAJISTA: str = "Tendencia bajista. "

# Modulo run_benchmarks (AutoTimesModel + compute_token_stats) - setado em main().
_RB = None  # type: ignore


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4 fases arquiteturais sobre o campeao ES Keywords "
                    "(vocab expansion | LoRA | soft prompt | CoT causal)"
    )
    # ---- fases (modular / ablacao) ------------------------------------ #
    parser.add_argument("--use_vocab_expansion", action="store_true",
                        help="FASE 1: numeros/meses como tokens atomicos")
    parser.add_argument("--use_lora", action="store_true",
                        help="FASE 2: LoRA (peft, r=8) nas c_attn do GPT-2")
    parser.add_argument("--use_soft_prompt", action="store_true",
                        help="FASE 3: N vetores continuos treinaveis prependidos")
    parser.add_argument("--use_cot", action="store_true",
                        help="FASE 4: frases causais de tendencia por janela")
    # ---- hiperparametros das fases ------------------------------------ #
    parser.add_argument("--lora-r", dest="lora_r", type=int, default=8)
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.1)
    parser.add_argument("--soft-prompt-n", dest="soft_prompt_n", type=int, default=10,
                        help="Numero de vetores continuos do soft prompt")
    # ---- pipeline (mesmos defaults dos benchmarks anteriores) --------- #
    parser.add_argument("--checkpoints", type=str, default="smetimes")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="2021,42,1234")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--block-max", dest="block_max", type=int, default=128)
    parser.add_argument("--n-paginas-max", dest="n_paginas_max", type=int, default=7)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--linhas", type=int, default=2000)
    parser.add_argument("--pool-janelas", dest="pool_janelas", type=int, default=192)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str, default=None)
    parser.add_argument("--amp", action="store_true",
                        help="AMP fp16 (instavel com fusion aleatorio; default FP32+TF32)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    return parser.parse_args()


# --------------------------------------------------------------------- #
#  FASE 1 - Cirurgia no tokenizador (expansao de vocabulario)            #
# --------------------------------------------------------------------- #
def _palavras_unicas(doc: str) -> List[str]:
    """Palavras unicas do documento (numeros, meses, 'coma', etc.)."""
    return sorted({p for p in doc.split() if p})


def expandir_tokenizador(tokenizer, palavras: List[str]) -> Tuple[int, Dict]:
    """Adiciona cada palavra (e sua variante com prefixo de espaco 'Ġ').

    O GPT-2 marca espacos como 'Ġ' no inicio do token; a variante nua so
    casa no inicio do documento, a variante 'Ġpalavra' casa no meio/fim.

    Returns:
        (n_novos, info) - quantos tokens novos foram adicionados e detalhes.
    """
    variantes: List[str] = []
    for w in palavras:
        if w not in variantes:
            variantes.append(w)
        if "Ġ" + w not in variantes:
            variantes.append("Ġ" + w)
    vocab_antes = len(tokenizer)
    n_novos = tokenizer.add_tokens(variantes)
    return n_novos, {
        "palavras_unicas": len(palavras),
        "variantes_candidatas": len(variantes),
        "tokens_adicionados": n_novos,
        "vocab_antes": vocab_antes,
        "vocab_depois": len(tokenizer),
    }


# --------------------------------------------------------------------- #
#  FASE 2 - LoRA (peft) nas c_attn do GPT-2                              #
# --------------------------------------------------------------------- #
def instalar_lora(model, r: int, alpha: int, dropout: float) -> Dict:
    """Envolve model.gpt2 com peft (LoraConfig em c_attn, qkv).

    O wrapper PeftModel preserva o acesso direto a `gpt2.wte` (verificado
    em runtime), entao o pipeline textual existente continua intocado.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["c_attn"],
    )
    model.gpt2 = get_peft_model(model.gpt2, cfg)
    n_lora = sum(1 for n, p in model.gpt2.named_parameters() if "lora" in n)
    n_lora_params = sum(p.numel() for n, p in model.gpt2.named_parameters()
                        if "lora" in n and p.requires_grad)
    return {"camadas_c_attn": n_lora // 2, "params_lora": n_lora_params}


# --------------------------------------------------------------------- #
#  FASE 3 - Soft prompt continuo (vetores aprendidos prependidos)        #
# --------------------------------------------------------------------- #
def _forward_multimodal_soft(
    self, x_enc: torch.Tensor, x_ids: torch.Tensor, x_lens: torch.Tensor,
) -> torch.Tensor:
    """Replica fiel do _forward_multimodal + prepend de vetores continuos.

    Diferenca unica (FASE 3): logo apos `wte(x_ids)`, os n_soft vetores
    treinaveis sao concatenados na frente da sequencia embedada e o mesmo
    numero de posicoes da cauda e descartado (orcamento fixo de tokens).
    """
    b = x_ids.shape[0]
    if x_enc is None:
        raise ValueError("Modo textual exige x_enc numerico (multimodal)")

    # ---------- Passo A: numerica primeiro (sempre) ---------------------
    x_norm = self.revin.normalize(x_enc)
    flat = x_norm.permute(0, 2, 1).reshape(b, self.seq_len)
    if self.use_overlap:
        patches = flat.unfold(-1, self.token_len, self.token_len // 2)
        num_tokens = self.encoder(flat)
    else:
        patches = flat.unfold(-1, self.token_len, self.token_len)
        num_tokens = self.encoder(patches)

    # ---------- Passo B: texto (wte congelado) + SOFT PROMPT ------------
    n_soft = self._soft_prompt.size(0)
    emb = self.gpt2.wte(x_ids)                                  # (B,Tt*block,D)
    sp = self._soft_prompt.unsqueeze(0).expand(b, n_soft, -1)   # (B,n_soft,D)
    emb = torch.cat([sp, emb[:, : emb.size(1) - n_soft, :]], dim=1)
    emb = emb.view(b, self.n_blocks_text, self.block_max, -1)   # (B,Tt,block,D)
    arange = torch.arange(self.block_max, device=emb.device)[None, None, :, None]
    mask = (arange < x_lens[:, :, None, None]).to(emb.dtype)    # (B,Tt,block,1)
    lens = x_lens.clamp(min=1).to(emb.dtype).unsqueeze(-1)
    txt_tokens = (emb * mask).sum(dim=2) / lens                 # (B,Tt,D)

    if txt_tokens.shape[1] != num_tokens.shape[1]:
        txt_tokens = torch.nn.functional.interpolate(
            txt_tokens.permute(0, 2, 1), size=num_tokens.shape[1],
            mode="linear", align_corners=False,
        ).permute(0, 2, 1)

    if self.gate is not None:
        num_tokens = self.gate(num_tokens, _RB.compute_token_stats(patches))

    # ---------- Passo C: fusao + estabilizacao --------------------------
    fused = torch.cat([txt_tokens, num_tokens], dim=-1)         # (B,T,2D)
    fused = self.fusion(fused)                                  # (B,T,D)
    latent = self.gpt2(inputs_embeds=fused.contiguous(), use_cache=False
                       ).last_hidden_state                      # (B,T,D)

    if self.use_overlap:
        rec = self.encoder.deembed(latent, output_len=self.seq_len)
    else:
        dec = self.decoder(latent.to(next(self.decoder.parameters()).dtype))
        rec = dec.reshape(b, self.n_patches * self.token_len)

    rec = rec.view(b, 1, self.seq_len).permute(0, 2, 1)
    return self.revin.denormalize(rec)


def instalar_soft_prompt(model, n_soft: int, device: torch.device) -> Dict:
    """Registra o vetor continuo e substitui o forward do modelo.

    O `nn.Parameter` vira atributo de instancia (entra no optimizer via
    model.parameters()); o metodo de instancia sobrescrito e usado pelo
    forward herdado (resolucao de atributo do Python).
    """
    if n_soft <= 0:
        raise ValueError(f"soft_prompt_n invalido: {n_soft}")
    hidden = model.gpt2.config.n_embd
    model._soft_prompt = nn.Parameter(
        torch.randn(n_soft, hidden, device=device) * 0.02
    )
    model._forward_multimodal = types.MethodType(_forward_multimodal_soft, model)
    return {"soft_prompt_n": n_soft, "params_soft_prompt": n_soft * hidden}


# --------------------------------------------------------------------- #
#  FASE 4 - CoT causal (frase de tendencia por janela)                   #
# --------------------------------------------------------------------- #
def construir_dataset_cot(
    resultado_paginacao: Dict,
    pool: List[Tuple[np.ndarray, np.ndarray]],
    tokenizer,
) -> TensorDataset:
    """Dataset com a frase causal prependida ao contexto de CADA janela.

    ids = frase(alcista/bajista, conforme o sinal da derivada primeira
    do lookback) + ids do documento, truncado no fim para manter o mesmo
    orcamento total de tokens (n_paginas_max * block_max).
    """
    ids_base = resultado_paginacao["ids"]
    lens = resultado_paginacao["lens"]
    n_tot = len(ids_base)
    frase_a = tokenizer(FRASE_ALCISTA, add_special_tokens=False)["input_ids"]
    frase_b = tokenizer(FRASE_BAJISTA, add_special_tokens=False)["input_ids"]
    n_f = max(len(frase_a), len(frase_b))
    frase_a = (frase_a + [tokenizer.pad_token_id] * n_f)[:n_f]
    frase_b = (frase_b + [tokenizer.pad_token_id] * n_f)[:n_f]

    xs, ids_linhas, ys = [], [], []
    for xw, yw in pool:
        tendencia = float(xw[-1, 0]) >= float(xw[0, 0])
        frase = frase_a if tendencia else frase_b
        ids_linhas.append((frase + ids_base)[:n_tot])
        xs.append(xw)
        ys.append(yw)
    return TensorDataset(
        torch.from_numpy(np.stack(xs)).float(),
        torch.tensor(np.stack(ids_linhas), dtype=torch.int64),
        torch.tensor([lens] * len(pool), dtype=torch.int64),
        torch.from_numpy(np.stack(ys)).float(),
    )


# --------------------------------------------------------------------- #
#  Configuracoes (baseline + fases isoladas + combinada)                 #
# --------------------------------------------------------------------- #
_FLAGS = ("vocab", "lora", "softprompt", "cot")
_ARG_FASES = {
    "vocab": "use_vocab_expansion",
    "lora": "use_lora",
    "softprompt": "use_soft_prompt",
    "cot": "use_cot",
}
_NOMES_FASES = {
    "vocab": "F1_VocabExpansao",
    "lora": "F2_LoRA",
    "softprompt": "F3_SoftPrompt",
    "cot": "F4_CoT",
}


def _montar_configs(args: argparse.Namespace) -> List[Tuple[str, Dict[str, bool]]]:
    """[baseline] + cada fase ativa isolada + combinacao completa (2+)."""
    base = {f: False for f in _FLAGS}
    configs: List[Tuple[str, Dict[str, bool]]] = [("baseline", dict(base))]
    ativos = [f for f in _FLAGS if getattr(args, _ARG_FASES[f])]
    for f in ativos:
        cfg = dict(base)
        cfg[f] = True
        configs.append((_NOMES_FASES[f], cfg))
    if len(ativos) > 1:
        configs.append(("combinada", {f: True for f in _FLAGS}))
    return configs


# --------------------------------------------------------------------- #
#  Dados: tokenizadores, documentos e datasets por variante               #
# --------------------------------------------------------------------- #
def preparar_dados(
    args: argparse.Namespace, tok_dir: Path, dados_etth1: Dict,
) -> Dict:
    """Duas variantes (vocab base vs expandido) x (sem/com CoT) -> loaders."""
    from transformers import AutoTokenizer

    def _novo_tokenizer() -> "object":
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id or 50256
        return tok

    doc = extenso.serie_para_texto_extenso(
        dados_etth1["dt_list"], dados_etth1["values"], _FEAT_PADRAO,
        0, dados_etth1["n_linhas"], TEMPLATE_CAMPEAO,
    )

    # --- tokenizador base (paginacao do baseline) ---------------------- #
    tok_base = _novo_tokenizer()
    n_tok_base = len(tok_base(doc, add_special_tokens=False)["input_ids"])

    # --- tokenizador expandido (FASE 1) -------------------------------- #
    palavras = _palavras_unicas(doc)
    tok_vocab = _novo_tokenizer()
    n_novos, vocab_info = expandir_tokenizador(tok_vocab, palavras)
    n_tok_vocab = len(tok_vocab(doc, add_special_tokens=False)["input_ids"])
    vocab_info["doc_tokens_base"] = n_tok_base
    vocab_info["doc_tokens_vocab"] = n_tok_vocab
    vocab_info["doc_reducao_x"] = round(n_tok_base / max(n_tok_vocab, 1), 2)
    print(f"  [FASE 1] {len(palavras)} palavras unicas | {n_novos} tokens "
          f"adicionados (vocab {vocab_info['vocab_antes']} -> "
          f"{vocab_info['vocab_depois']})")
    print(f"  [FASE 1] doc: {n_tok_base} -> {n_tok_vocab} tokens "
          f"(x{vocab_info['doc_reducao_x']} densidade)")

    def _paginados(tok) -> Dict:
        paginador = TestadorPaginacao(
            tok, block_max=args.block_max,
            overlap_frac=args.overlap, n_paginas_max=args.n_paginas_max,
        )
        r = paginador.paginar(ESTRATEGIA_CAMPEA, doc, "es")
        r.update(paginador.para_modelo(r))
        return r

    r_base = _paginados(tok_base)
    r_vocab = _paginados(tok_vocab)
    for nome, r in (("base", r_base), ("vocab", r_vocab)):
        print(f"  [{nome}] doc paginado: {r['tokens_originais']} tokens -> "
              f"{r['tokens_retidos']} retidos | {r['n_paginas']} paginas")

    pool = pool_janelas_etth1(dados_etth1, args.pool_janelas)
    n = len(pool)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)

    def _split(ds: TensorDataset) -> Tuple:
        tr = TensorDataset(*[t[:n_tr] for t in ds.tensors])
        va = TensorDataset(*[t[n_tr:n_tr + n_va] for t in ds.tensors])
        te = TensorDataset(*[t[n_tr + n_va:] for t in ds.tensors])
        return build_loaders(tr, va, te, args.batch_size)

    return {
        "doc": doc,
        "tok_base": tok_base,
        "tok_vocab": tok_vocab,
        "vocab_info": vocab_info,
        "paginados_base": r_base,
        "paginados_vocab": r_vocab,
        "loaders": {
            ("base", False): _split(construir_dataset(r_base, pool)),
            ("vocab", False): _split(construir_dataset(r_vocab, pool)),
            ("base", True): _split(construir_dataset_cot(r_base, pool, tok_base)),
            ("vocab", True): _split(construir_dataset_cot(r_vocab, pool, tok_vocab)),
        },
        "n_amostras": n,
        "pool": pool,
    }


# --------------------------------------------------------------------- #
#  Relatorio final                                                       #
# --------------------------------------------------------------------- #
def _fmt(media: float, desvio: float) -> str:
    return f"{media:.2f} ± {desvio:.2f}"


def imprimir_tabela(resumo: Dict) -> None:
    print("\n" + "=" * 100)
    print("PIPELINE DE OTIMIZACAO ARQUITETURAL (10 epocas x 3 seeds)")
    print("=" * 100)
    linhas = sorted(
        (c["test_mse_media"], nome, c) for nome, c in resumo["cenarios"].items()
    )
    print(f"{'#':>2s} {'Config':<16s} {'Test MSE':>16s} {'Test MAE':>16s} "
          f"{'best_val':>14s} {'estouros':>8s} {'params_extra':>12s}")
    print("-" * 100)
    for i, (mse, nome, c) in enumerate(linhas, 1):
        print(f"{i:>2d} {nome:<16s} "
              f"{_fmt(mse, c['test_mse_desvio']):>16s} "
              f"{_fmt(c['test_mae_media'], c['test_mae_desvio']):>16s} "
              f"{_fmt(c['melhor_val_mse_media'], c['melhor_val_mse_desvio']):>14s} "
              f"{c['estouros_total']:>8d} {c['params_extra']:>12,}")
    print("-" * 100)
    print("Formato: media ± desvio padrao sobre as 3 seeds (menor MSE vence).")
    print("=" * 100)


def salvar_resumo(resumo: Dict) -> Path:
    return salvar_benchmark(resumo, nome="benchmark_4fases_stats.json")


def gerar_grafico(resumo: Dict) -> Path:
    """Curvas de validacao (media das seeds) + banda de desvio padrao."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caminhos.criar_resultados()
    fig, ax = plt.subplots(figsize=(11, 6))
    cores = {
        "baseline": "#7f7f7f",
        "F1_VocabExpansao": "#9467bd",
        "F2_LoRA": "#d62728",
        "F3_SoftPrompt": "#17becf",
        "F4_CoT": "#ff7f0e",
        "combinada": "#2ca02c",
    }
    epocas = np.arange(
        1, len(list(resumo["cenarios"].values())[0]["val_mse_media"]) + 1
    )
    for nome, c in resumo["cenarios"].items():
        cor = cores.get(nome, "#333333")
        media = np.asarray(c["val_mse_media"])
        desvio = np.asarray(c["val_mse_desvio"])
        rotulo = (f"{nome} (Test MSE {c['test_mse_media']:.2f} ± "
                  f"{c['test_mse_desvio']:.2f})")
        if nome == "baseline":
            ax.plot(epocas, media, "-o", color=cor, lw=2.2, ms=4,
                    label=rotulo, zorder=5)
        else:
            ax.plot(epocas, media, "-o", color=cor, lw=2, ms=4, label=rotulo)
        ax.fill_between(epocas, media - desvio, media + desvio,
                        color=cor, alpha=0.18, linewidth=0)
    ax.set_xticks(epocas)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Val MSE no horizonte (media das 3 seeds)")
    ax.set_title("Otimizacao arquitetural sobre ES Keywords: convergencia "
                 "com banda de confianca (media ± 1 dp)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = caminhos.GRAFICOS_DIR / "otimizacao_4fases_errorbar.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# --------------------------------------------------------------------- #
#  Main                                                                 #
# --------------------------------------------------------------------- #
def main() -> None:
    global _RB

    args = parse_args()
    caminhos.criar_resultados()
    _RB = caminhos.importar_run_benchmarks()
    rb = _RB

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Nenhuma seed valida em --seeds")
    rb.seed_everything(seeds[0])

    device = rb.get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")

    # ---- Tokenizador: busca dinamica recursiva ------------------------ #
    tok_dir: Optional[Path] = None
    try:
        if args.tokenizer_path and (Path(args.tokenizer_path) / "tokenizer.json").exists():
            tok_dir = Path(args.tokenizer_path)
        else:
            tok_dir = caminhos.descobrir_tokenizador_recursivo(caminhos.ROOT_DIR)
    except FileNotFoundError as e:
        print(f"[erro] tokenizador nao encontrado: {e}")
        raise SystemExit(1)
    print(f"[tokenizer] carregado de: {tok_dir}")

    # ---- Configuracoes a rodar ---------------------------------------- #
    configs = _montar_configs(args)
    nomes_fases = [f for f in _FLAGS if getattr(args, _ARG_FASES[f])]
    print(f"[config] campeao={TEMPLATE_CAMPEAO}|{ESTRATEGIA_CAMPEA} | "
          f"fases ativas={nomes_fases} | configs={[c[0] for c in configs]} | "
          f"seeds={seeds} x {args.epochs} epocas")

    # ---- Dados (2 variantes de tokenizacao x sem/com CoT) ------------- #
    dados_etth1 = carregar_etth1(args.csv, args.linhas)
    feat_name = dados_etth1["feat_names"][_FEAT_PADRAO]
    print(f"[csv] {dados_etth1['csv']} | {dados_etth1['n_linhas_total']} linhas "
          f"-> {dados_etth1['n_linhas']} linhas (coluna '{feat_name}')")
    prep = preparar_dados(args, tok_dir, dados_etth1)
    pool = prep["pool"]
    print(f"[pool] {len(pool)} janelas numericas de {feat_name}")

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

    # ================================================================= #
    #  VALIDACAO - config x seed (isolamento estrito por ciclo)          #
    # ================================================================= #
    print("\n" + "=" * 100)
    print(f"VALIDACAO - {args.epochs} EPOCAS x {len(seeds)} SEEDS "
          f"x {len(configs)} CONFIGS (isolamento estrito)")
    print("=" * 100)
    resumo: Dict = {
        "config": {
            "template": TEMPLATE_CAMPEAO,
            "estrategia": ESTRATEGIA_CAMPEA,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "linhas": args.linhas,
            "pool_janelas": len(pool),
            "block_max": args.block_max,
            "n_paginas_max": args.n_paginas_max,
            "overlap": args.overlap,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "soft_prompt_n": args.soft_prompt_n,
            "vocab": prep["vocab_info"],
            "frases_cot": {"alcista": FRASE_ALCISTA, "bajista": FRASE_BAJISTA},
        },
        "cenarios": {},
    }

    for chave, ckpt_path in ckpts_sel.items():
        print(f"\n--- Modelo: {chave} ({Path(ckpt_path).name}) ---")
        for nome, fl in configs:
            variante = "vocab" if fl["vocab"] else "base"
            tr_ld, va_ld, te_ld = prep["loaders"][(variante, fl["cot"])]
            por_seed: Dict[str, Dict] = {}
            mse_list, mae_list, bestval_list, epoca_list, estouros_list = (
                [], [], [], [], [])
            val_por_epoca: List[List[float]] = []
            params_extra = 0
            for seed in seeds:
                # ---- ISOLAMENTO ESTRITO: dentro do loop mais interno ---- #
                rb.seed_everything(seed)
                model, info = montar_modelo(ckpt_path, device, use_amp=use_amp)
                extras = []
                if fl["vocab"]:
                    model.gpt2.resize_token_embeddings(
                        len(prep["tok_vocab"]), mean_resizing=False,
                    )
                    extras.append("wte.resize")
                if fl["lora"]:
                    extras.append(instalar_lora(
                        model, args.lora_r, args.lora_alpha, args.lora_dropout))
                if fl["softprompt"]:
                    extras.append(instalar_soft_prompt(model, args.soft_prompt_n, device))
                n_treinaveis = sum(p.numel() for p in model.parameters()
                                   if p.requires_grad)
                if params_extra == 0:
                    params_extra = n_treinaveis - info["params_treinaveis"]
                print(f"  [mount seed={seed}] {nome} | flags={fl} | "
                      f"treinaveis={n_treinaveis:,} "
                      f"(extra={n_treinaveis - info['params_treinaveis']:,})")
                t0 = time.perf_counter()
                result = rodar_convergencia(
                    model, tr_ld, va_ld, te_ld,
                    epochs=args.epochs, lr=args.lr, device=device,
                    use_amp=use_amp, seed=seed,
                )
                mse = result["teste_final_mse_pred"]
                mae = result["teste_final_mae_pred"]
                bv = result["melhor_val_mse_pred"]
                be = result["melhor_epoca"]
                estouros = result["batches_estourados_total"]
                por_seed[str(seed)] = {
                    "test_mse": round(mse, 6), "test_mae": round(mae, 6),
                    "melhor_val_mse": round(bv, 6), "melhor_epoca": be,
                    "estouros": estouros,
                }
                mse_list.append(mse)
                mae_list.append(mae)
                bestval_list.append(bv)
                epoca_list.append(be or args.epochs)
                estouros_list.append(estouros)
                val_por_epoca.append(result["historico"]["val_mse_pred"])
                print(f"    [seed={seed}] test_mse={mse:.4f} | test_mae={mae:.4f} | "
                      f"best_val={bv:.4f} (ep {be}) | estouros={estouros} | "
                      f"{time.perf_counter() - t0:.1f}s")
                # ---- fim do ciclo: libera VRAM -------------------------- #
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            n_ep = max((len(v) for v in val_por_epoca), default=0)
            val_media = [float(np.mean([v[e] for v in val_por_epoca]))
                         for e in range(n_ep)]
            val_desvio = [float(np.std([v[e] for v in val_por_epoca]))
                          for e in range(n_ep)]
            r_pag = prep["paginados_vocab" if fl["vocab"] else "paginados_base"]
            resumo["cenarios"][nome] = {
                "flags": fl,
                "test_mse_media": round(float(np.mean(mse_list)), 4),
                "test_mse_desvio": round(float(np.std(mse_list)), 4),
                "test_mae_media": round(float(np.mean(mae_list)), 4),
                "test_mae_desvio": round(float(np.std(mae_list)), 4),
                "melhor_val_mse_media": round(float(np.mean(bestval_list)), 4),
                "melhor_val_mse_desvio": round(float(np.std(bestval_list)), 4),
                "melhor_epoca_media": round(float(np.mean(epoca_list)), 2),
                "estouros_total": sum(estouros_list),
                "params_extra": params_extra,
                "val_mse_media": [round(x, 6) for x in val_media],
                "val_mse_desvio": [round(x, 6) for x in val_desvio],
                "tokens_originais": r_pag["tokens_originais"],
                "tokens_retidos": r_pag["tokens_retidos"],
                "paginas": r_pag["n_paginas"],
                "por_seed": por_seed,
            }
            c = resumo["cenarios"][nome]
            print(f"  [ok] {nome}: MSE {c['test_mse_media']:.2f} ± "
                  f"{c['test_mse_desvio']:.2f} | MAE {c['test_mae_media']:.2f} ± "
                  f"{c['test_mae_desvio']:.2f} | estouros={c['estouros_total']}")

    # ---- Relatorio final ---------------------------------------------- #
    imprimir_tabela(resumo)
    out = salvar_resumo(resumo)
    print(f"[ok] JSON: {out}")
    if args.plots:
        p = gerar_grafico(resumo)
        print(f"[ok] grafico: {p}")
    print("Pipeline de otimizacao arquitetural concluido.")


if __name__ == "__main__":
    main()
