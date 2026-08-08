"""
discriminative_optim - Builder de otimizador discriminativo.

Agrupa os parametros treinaveis do modelo em dois grupos com learning rates
distintas, seguindo a pratica de fine-tuning de modelos baseados em LLM:

    - Grupo "llm": parametros cujo nome contem "gpt2" ou "llm" (backbone
      congelado parcialmente ou nao) -> lr baixa (default 2e-5).
    - Grupo "novos": parametros dos modulos adicionados para o forecasting
      (heads, gates, embeddings customizados, RevIN, encoders/decoders) ->
      lr alta (default 1e-3).

Parametros congelados (requires_grad=False) sao ignorados; grupos vazios sao
descartados automaticamente. Acompanha um scheduler CosineAnnealingWarmRestarts
(restarts periodicos para escapar de minimos locais e evitar estagnacao).

Uso:
    optimizer, groups = build_discriminative_optimizer(model, lr_llm=2e-5, lr_new=1e-3)
    scheduler = build_cosine_warm_scheduler(optimizer, t_0=5, t_mult=2, eta_min=1e-6)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from torch import nn, optim


def build_discriminative_optimizer(
    model: nn.Module,
    lr_llm: float = 2e-5,
    lr_new: float = 1e-3,
    weight_decay: float = 0.0,
    llm_tokens: Tuple[str, ...] = ("gpt2", "llm"),
) -> Tuple[optim.Optimizer, Dict[str, int]]:
    """Constroi um AdamW com grupos de lr discriminativos.

    Args:
        model: Modelo cujos parametros serao agrupados.
        lr_llm: Learning rate do grupo backbone ("gpt2"/"llm").
        lr_new: Learning rate dos novos modulos (heads, gates, embeddings).
        weight_decay: Decaimento de peso do AdamW.
        llm_tokens: Substrings de nome que identificam o backbone LLM.

    Returns:
        (optimizer, resumo): AdamW configurado e dict com contagens
        {"llm": n_params, "novos": n_params} para diagnostico.
    """
    llm_params: List[nn.Parameter] = []
    new_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered: str = name.lower()
        if any(token in lowered for token in llm_tokens):
            llm_params.append(param)
        else:
            new_params.append(param)

    param_groups = []
    if llm_params:
        param_groups.append({"params": llm_params, "lr": lr_llm})
    if new_params:
        param_groups.append({"params": new_params, "lr": lr_new})

    if not param_groups:
        raise ValueError("Nenhum parametro treinavel encontrado no modelo.")

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    summary: Dict[str, int] = {"llm": len(llm_params), "novos": len(new_params)}
    return optimizer, summary


def build_cosine_warm_scheduler(
    optimizer: optim.Optimizer,
    t_0: int = 5,
    t_mult: int = 2,
    eta_min: float = 1e-6,
) -> optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """Scheduler com restarts cossenoidais (CosineAnnealingWarmRestarts).

    Args:
        optimizer: Otimizador (ex: retornado pelo builder acima).
        t_0: Numero de epocas ate o primeiro restart.
        t_mult: Fator de crescimento do periodo apos cada restart.
        eta_min: Learning rate minimo no vale da curva cosseno.

    Returns:
        CosineAnnealingWarmRestarts configurado.
    """
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
    )


if __name__ == "__main__":  # pragma: no cover - smoke test manual
    class _Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt2 = nn.Linear(8, 8)
            self.encoder = nn.Linear(4, 8)
            self.gate = nn.Linear(8, 1)
            self.frozen_head = nn.Linear(2, 2)

        def forward(self, x):
            return self.gate(self.gpt2(self.encoder(x)))

    m = _Dummy()
    m.frozen_head.weight.requires_grad = False
    m.frozen_head.bias.requires_grad = False

    opt, summary = build_discriminative_optimizer(m, lr_llm=2e-5, lr_new=1e-3)
    assert summary == {"llm": 2, "novos": 4}, summary
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs == [2e-5, 1e-3], lrs
    sch = build_cosine_warm_scheduler(opt, t_0=5, t_mult=2)
    assert isinstance(sch, optim.lr_scheduler.CosineAnnealingWarmRestarts)
    print("Smoke test discriminative_optim OK -", summary, lrs)