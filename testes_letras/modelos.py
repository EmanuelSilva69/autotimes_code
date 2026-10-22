"""
modelos - carrega os checkpoints do Ablation Study e reconstroi o
AutoTimesModel em modo TEXTUAL (multimodal numerico+texto), preservando as
melhorias arquiteturais ja treinadas:

    - smetimes: Run-8 VolLoss + SMETimes (AdaptiveFusion/porta de
      volatilidade) + otimizacoes AMD (matmul precision, cudnn benchmark).
    - amd:      Run-0 Padrao + otimizacoes AMD (somente o baseline).

O loop de convergencia roda em FP32 + TF32 (perfil estavel no ROCm); o AMP
fp16 do checkpoint original estoura em NaN no backward por causa do modulo
`fusion` aleatorio do modo textual (ver run_testes.py --amp).

Nota de fidelidade:
    Os checkpoints foram treinados em modo NUMERICO (textual=False) e nao
    contem o modulo `fusion` (cat(texto, num) -> Linear -> LayerNorm) usado
    no modo textual. Esse modulo nasce com pesos aleatorios (4 tensores) e e
    treinado do zero no loop de convergencia; todo o resto (GPT-2 congelado,
    encoder/decoder MLP, gate SMETimes, RevIN) vem do checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from . import caminhos

# Chave -> checkpoint do Ablation Study.
CKPTS: Dict[str, Path] = {
    "smetimes": caminhos.CKPT_SMETIMES,  # VolLoss + SMETimes + AMD
    "amd": caminhos.CKPT_AMD,            # somente otimizacoes AMD
}


def carregar_checkpoint(caminho: Path) -> Dict:
    """torch.load do checkpoint do ablation (CPU, dict completo)."""
    ckpt = torch.load(caminho, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint invalido (sem model_state_dict): {caminho}")
    return ckpt


def montar_modelo(
    ckpt_path: Path,
    device: torch.device,
    use_amp: bool = False,
    gpt2_dir: Optional[Path] = None,
    block_max: int = 128,
    linear_probing: bool = False,
    replace_decoder: bool = False,
    kan_decoder: bool = False,
    kan_act: str = "silu",
    taylor_decoder: bool = False,
    taylor_order: int = 3,
    taylor_no_tanh: bool = False,
    residual_adapter: str = "none",
    residual_order: int = 2,
    residual_alpha: float = 0.1,
) -> Tuple[torch.nn.Module, Dict]:
    """Reconstroi o AutoTimesModel textual a partir do checkpoint de ablacao.

    Args:
        ckpt_path: checkpoint Run-0 (amd) ou Run-8 (smetimes).
        device: cuda (ROCm/AMD) ou cpu.
        use_amp: mixed precision (autocast + GradScaler + GPT-2 fp16).
            Default False: com o fusion aleatorio do modo textual o backward
            fp16 estoura em NaN; FP32 + TF32 e o perfil estavel.
        gpt2_dir: diretorio do GPT-2 base (default: args do checkpoint).
        block_max: pad de tokens por bloco (128, igual ao pipeline).

    Returns:
        (model, info): modelo em device (modo eval) e dict de metadados
        (flags, parametros, missing/unexpected keys do load).
    """
    rb = caminhos.importar_run_benchmarks()
    ckpt = carregar_checkpoint(ckpt_path)
    flags: Dict = ckpt["flags"]
    args: Dict = ckpt["args"]

    # Mesma regra da bateria: backbone congelado + AMP -> GPT-2 em FP16.
    torch_dtype = torch.float16 if (use_amp and device.type == "cuda") else torch.float32

    model = rb.AutoTimesModel(
        seq_len=args["seq_len"],
        pred_len=args["pred_len"],
        n_vars=args["n_vars"],
        label_len=args["label_len"],
        mlp_hidden_dim=args["mlp_hidden_dim"],
        mlp_hidden_layers=args["mlp_hidden_layers"],
        mlp_activation=args["mlp_activation"],
        dropout=args["dropout"],
        gpt2_dir=Path(gpt2_dir or args["gpt2_dir"]),
        torch_dtype=torch_dtype,
        use_robust=bool(flags["use_robust"]),
        use_overlap=bool(flags["use_overlap"]),
        use_vol_gate=bool(flags["use_vol_gate"]),
        unfreeze_layers=0,
        textual=True,
        block_max=block_max,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("  [debug] load_state_dict concluido")

    # ---- Freeze encoder/decoder: preserva a capacidade de reconstrucao
    # numerica do checkpoint. O fusion layer (novo, aleatorio) precisa
    # aprender a produzir latentes COMPATIVEIS com o decoder congelado.
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    print("  [debug] freeze concluido")

    # ---- Linear Probing: substitui o decoder MLP por uma projecao linear
    # limpa, treinada do zero. Remove o domain shift causado pelo decoder
    # pre-treinado em latentes puramente numericas.
    if linear_probing:
        if model.use_overlap:
            raise ValueError("Linear Probing nao suportado com use_overlap=True")
        hidden = model.gpt2.config.n_embd
        token_len = model.token_len
        model.decoder = torch.nn.Linear(hidden, token_len).to(device)
        # Encoder continua congelado; novo decoder e treinavel
        print(f"  [linear-probing] Decoder MLP substituido por Linear({hidden}, {token_len})")

    # ---- Replace Decoder: substitui o decoder MLP treinado por uma MLP
    # NOVA, inicializada do zero (Linear->GELU->Dropout->Linear). Herda a
    # capacidade nao-linear sem o vies numerico do checkpoint.
    if replace_decoder:
        if model.use_overlap:
            raise ValueError("replace_decoder nao suportado com use_overlap=True")
        hidden = model.gpt2.config.n_embd
        token_len = model.token_len
        model.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 3, token_len),
        ).to(device)
        print(f"  [replace-decoder] MLP nova: Linear({hidden},{hidden//3})->GELU->Dropout(0.1)->Linear({hidden//3},{token_len})")

    # ---- KAN Decoder: substitui o decoder MLP por uma Kolmogorov-Arnold
    # Network. Capacidade nao-linear com base em spline adaptativa.
    if kan_decoder:
        if model.use_overlap:
            raise ValueError("kan_decoder nao suportado com use_overlap=True")
        from efficient_kan import KAN as KANLayer
        hidden = model.gpt2.config.n_embd
        token_len = model.token_len
        act_map = {
            'silu': nn.SiLU,
            'sigmoid': nn.Sigmoid,
            'elu': nn.ELU,
            'selu': nn.SELU,
        }
        act_cls = act_map.get(kan_act, nn.SiLU)
        model.decoder = KANLayer([hidden, token_len], base_activation=act_cls).to(device)
        print(f"  [kan-decoder] KAN([{hidden}, {token_len}], act={kan_act})")

    # ---- TaylorKAN Decoder: substitui o decoder MLP por uma camada de
    # serie de Taylor (metodo de Horner). Params: input_dim x out_dim x order.
    # LayerNorm antes normaliza latentes do GPT-2 (media 0, var 1).
    if taylor_decoder:
        if model.use_overlap:
            raise ValueError("taylor_decoder nao suportado com use_overlap=True")
        from testes_letras.taylor_kan import TaylorLayer
        hidden = model.gpt2.config.n_embd
        token_len = model.token_len
        _taylor_use_tanh = not taylor_no_tanh
        model.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            TaylorLayer(hidden, token_len, order=taylor_order, use_tanh=_taylor_use_tanh),
        ).to(device)
        n_params = sum(p.numel() for p in model.decoder.parameters())
        _tanh_str = "tanh=ON" if _taylor_use_tanh else "tanh=OFF"
        print(f"  [taylor-decoder] LayerNorm({hidden}) + TaylorLayer({hidden}, {token_len}, order={taylor_order}, {_tanh_str}) "
              f"| {n_params:,} params")

    # ---- Residual Adapter: MLP congelado + adapter leve treinavel ----
    # Nao substitui o MLP, apenas envolve: y = mlp(x) + alpha * adapter(x)
    if residual_adapter != "none":
        from testes_letras.residual_adapters import (
            LinearResidualAdapter,
            TaylorResidualAdapter,
            MoLEResidualAdapter,
        )
        if model.use_overlap:
            raise ValueError("residual_adapter nao suportado com use_overlap=True")
        hidden = model.gpt2.config.n_embd
        token_len = model.token_len
        original_decoder = model.decoder  # MLP congelado

        adapter_map = {
            "linear": lambda: LinearResidualAdapter(
                original_decoder, hidden, token_len,
                alpha_init=residual_alpha,
            ),
            "taylor": lambda: TaylorResidualAdapter(
                original_decoder, hidden, token_len,
                order=residual_order, alpha_init=residual_alpha,
            ),
            "mole": lambda: MoLEResidualAdapter(
                original_decoder, hidden, token_len,
                alpha_init=residual_alpha,
            ),
        }

        if residual_adapter not in adapter_map:
            raise ValueError(
                f"residual_adapter desconhecido: {residual_adapter}. "
                f"Opcoes: {list(adapter_map.keys())}"
            )

        model.decoder = adapter_map[residual_adapter]().to(device)
        n_total = sum(p.numel() for p in model.decoder.parameters())
        if residual_adapter == "mole":
            n_adapter = (sum(p.numel() for p in model.decoder.expert1.parameters())
                         + sum(p.numel() for p in model.decoder.expert2.parameters())
                         + sum(p.numel() for p in model.decoder.gate.parameters()))
        else:
            n_adapter = sum(p.numel() for p in model.decoder.adapter.parameters())
        n_alpha = 1
        print(f"  [{residual_adapter}-residual] MLP congelado + adapter | "
              f"{n_total:,} params total ({n_adapter:,} adapter + {n_alpha} alpha)")

    model.eval()
    print("  [debug] model.eval() concluido")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    gpt2_congelado = all(not p.requires_grad for p in model.gpt2.parameters())
    print(f"  [model] Parametros Treinaveis: {trainable / 1e6:.3f} milhoes | "
          f"Parametros Congelados: {frozen / 1e6:.3f} milhoes "
          f"({100.0 * trainable / max(total, 1):.2f}% treinaveis)")
    print(f"  [model] backbone GPT-2 100%% congelado: {gpt2_congelado}")
    if trainable > 3.5e6:
        print("  [model] ATENCAO: treinaveis > 3.5 milhoes -> "
              "verificar congelamento do backbone!")
    info = {
        "checkpoint": str(ckpt_path),
        "flags": flags,
        "best_val_loss": ckpt.get("best_val_loss"),
        "best_epoch": ckpt.get("best_epoch"),
        "params_total": total,
        "params_treinaveis": trainable,
        "params_congelados": frozen,
        "gpt2_backbone_congelado": bool(gpt2_congelado),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }
    return model, info
