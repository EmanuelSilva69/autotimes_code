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
    model.eval()

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
