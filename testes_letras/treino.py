"""
treino - mini-loop de treinamento (epocas curtas) para medir a CONVERGENCIA
de cada template de compressao.

Para cada (checkpoint, template) roda algumas epocas (default 5) com Adam +
CosineAnnealingLR (mesmo esquema da bateria, T_max=epocas) e reporta:
    - loss de treino e validacao por epoca (MSE no horizonte pred_len);
    - MAE no horizonte por epoca;
    - tempo medio por batch (ms) e por epoca;
    - melhor epoca de validacao (criterio de "convergencia mais rapida").

A comparacao entre templates e justa: mesma semente, mesmo otimizador,
mesma loss (MSE pura), mesmos parametros treinaveis (o backbone GPT-2 fica
congelado; apenas encoder/decoder MLP + fusion + gate SMETimes treinam).
"""

from __future__ import annotations

import contextlib
import copy
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import caminhos


def _amp_ctx(device: torch.device, enabled: bool):
    """Contexto de autocast (AMP) para ROCm/CUDA; nullcontext em CPU."""
    if device.type == "cuda" and enabled:
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def build_loaders(
    train_ds, val_ds, test_ds, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoaders simples (in-memory, workers=0 p/ determinismo e rapidez)."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _desempacotar_batch(batch, device):
    """Desempacota batch de 4 (x, ids, lens, y) ou 5 tensores
    (x, ids, lens, y, max_abs), retornando (x, sargs, y, max_abs).

    Nos DOIS contratos o x e batch[0], os argumentos do modelo (ids, lens)
    sao batch[1:3] e o y e batch[3]; o contrato de 5 tensores apenas
    acrescenta o max_abs em batch[4].
    """
    n = len(batch)
    x = batch[0].to(device, non_blocking=True)
    sargs = [t.to(device, non_blocking=True) for t in batch[1:3]] if n >= 4 else []
    y = batch[3].to(device, non_blocking=True) if n >= 4 else batch[-1].to(device, non_blocking=True)
    max_abs = batch[4].to(device, non_blocking=True) if n >= 5 else None
    return x, sargs, y, max_abs


def treinar_epoca(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    scaler,
    device: torch.device,
    amp_on: bool,
    pred_len: int = 96,
) -> Tuple[float, List[float], int]:
    """Treina UMA epoca completa; devolve (loss media, tempos, estouros).

    `estouros` conta batches com loss/gradiente NaN ou Inf (pulados) -
    metrica de ESTABILIDADE: quanto menor, mais estavel o texto puro.
    A loss e calculada APENAS nos ultimos `pred_len` passos (horizonte),
    alinhando o treino ao objetivo real de previsao.
    """
    model.train()
    running, count = 0.0, 0
    tempos: List[float] = []
    estouros: int = 0
    for batch in loader:
        t0 = time.perf_counter()
        x, sargs, y, max_abs = _desempacotar_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with _amp_ctx(device, amp_on):
            pred = model(x, *sargs)
            loss = criterion(pred[:, -pred_len:, :], y[:, -pred_len:, :])
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler is not None:
            scaler.unscale_(optimizer)

        # Trava de seguranca identica a bateria: batch NaN/Inf e pulado.
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=False
        )
        finite = torch.isfinite(loss) and torch.isfinite(total_norm)
        if finite:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        else:
            estouros += 1
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()

        b = x.size(0)
        running += float(loss.item()) * b
        count += b
        tempos.append(time.perf_counter() - t0)
    return running / max(count, 1), tempos, estouros


@torch.no_grad()
def avaliar(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_on: bool,
    pred_len: int = 96,
    decode_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Avalia um loader completo, retornando loss + MSE/MAE (janela e pred).

    Args:
        decode_fn: opcional, aplicado a `pred` (e ao loss) ANTES das
            metricas quando o batch tem o contrato de 5 tensores (max_abs).
            Ex.: CharByCharPipeline.decode para voltar da escala normalizada
            para a escala real. None = metricas no espaco nativo do modelo.

    Returns:
        Dict com loss_janela, mse_janela, mae_janela (reconstrucao total)
        e mse_pred, mae_pred (somente os ultimos `pred_len` passos, o
        horizonte que interessa para a previsao).
    """
    model.eval()
    running, count = 0.0, 0
    preds_j, trues_j, preds_p, trues_p = [], [], [], []
    for batch in loader:
        x, sargs, y, max_abs = _desempacotar_batch(batch, device)
        with _amp_ctx(device, amp_on):
            pred = model(x, *sargs)
            if decode_fn is not None and max_abs is not None:
                pred = decode_fn(pred, max_abs)
            loss = criterion(pred, y)
        running += float(loss.item()) * x.size(0)
        count += x.size(0)
        preds_j.append(pred.float().cpu().numpy())
        trues_j.append(y.float().cpu().numpy())
        preds_p.append(pred[:, -pred_len:, :].float().cpu().numpy())
        trues_p.append(y[:, -pred_len:, :].float().cpu().numpy())

    def _mse_mae(p: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
        return float(np.mean((p - t) ** 2)), float(np.mean(np.abs(p - t)))

    mse_j, mae_j = _mse_mae(np.concatenate(preds_j, 0), np.concatenate(trues_j, 0))
    mse_p, mae_p = _mse_mae(np.concatenate(preds_p, 0), np.concatenate(trues_p, 0))
    return {
        "loss_janela": running / max(count, 1),
        "mse_janela": mse_j,
        "mae_janela": mae_j,
        "mse_pred": mse_p,
        "mae_pred": mae_p,
    }


def rodar_convergencia(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device = None,
    use_amp: bool = True,
    seed: int = 2021,
    restore_best: bool = False,
    decode_fn: Optional[Callable] = None,
    pred_len: int = 96,
    unfreeze_encoder_epoch: int = 0,
    lr_unfreeze: float = 1e-5,
) -> Dict:
    """Mini-loop de convergencia para um (checkpoint, template).

    Args:
        restore_best: se True, restaura o estado do modelo da melhor epoca
            de validacao ANTES da avaliacao final de teste e reporta
            teste_final_* como o MSE/MAE desse estado (protocolo do
            ablation original: melhor checkpoint por val_loss).
        decode_fn: repassado a avaliar() (decode de escala, ver treino.avaliar).

    Returns:
        Dict com historico por epoca (train_loss, val/metricas, tempo por
        batch), melhor epoca de validacao e metricas finais de teste.
    """
    rb = caminhos.importar_run_benchmarks()
    rb.seed_everything(seed)  # mesma semente em todos os (modelo, template)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = bool(use_amp and device.type == "cuda")

    params = [p for p in model.parameters() if p.requires_grad]
    n_opt = sum(p.numel() for p in params)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"  [otimizador] Adam recebe APENAS treinaveis: "
          f"{len(params)} tensores ({n_opt / 1e6:.3f}M de {n_tot / 1e6:.3f}M)")
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-8
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if amp_on else None

    hist: Dict = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_mse": [],
        "val_mae": [], "val_mse_pred": [], "val_mae_pred": [],
        "test_mse_pred": [], "test_mae_pred": [],
        "tempo_epoca_s": [], "tempo_batch_ms": [],
        "batches_estourados_epoca": [],
    }
    best_val = float("inf")
    best_epoch = None
    best_state = None
    # Snapshot barato: clona so treinaveis; congelados por referencia
    # (nao mutam durante o treino -> valores preservados no restore).
    trainable_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    n_batches = max(len(train_loader), 1)

    for ep in range(1, epochs + 1):
        # ---- Two-phase: descongela encoder na epoca definida -----------
        if unfreeze_encoder_epoch > 0 and ep == unfreeze_encoder_epoch:
            n_unfrozen = 0
            for p in model.encoder.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    n_unfrozen += 1
            if n_unfrozen > 0:
                # Reconstrui otimizador com NOVOS parametros (encoder + existentes)
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(params, lr=lr_unfreeze, weight_decay=0.0)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(epochs - ep + 1, 1), eta_min=1e-8
                )
                trainable_keys.update({k for k, p in model.named_parameters() if p.requires_grad})
                n_opt = sum(p.numel() for p in params)
                print(f"  [two-phase] epoca {ep}: encoder descongelado! "
                      f"Novo optimizer com {n_opt / 1e6:.3f}M params (lr={lr_unfreeze})")

        t0 = time.perf_counter()
        tl, tempos, estouros = treinar_epoca(
            model, train_loader, optimizer, criterion, scaler, device, amp_on,
            pred_len=pred_len,
        )
        v = avaliar(model, val_loader, criterion, device, amp_on,
                    pred_len=pred_len, decode_fn=decode_fn)
        te = avaliar(model, test_loader, criterion, device, amp_on,
                     pred_len=pred_len, decode_fn=decode_fn)
        ep_time = time.perf_counter() - t0
        if scheduler is not None:
            scheduler.step()

        hist["epoch"].append(ep)
        hist["train_loss"].append(round(tl, 8))
        hist["val_loss"].append(round(v["loss_janela"], 8))
        hist["val_mse"].append(round(v["mse_janela"], 8))
        hist["val_mae"].append(round(v["mae_janela"], 8))
        hist["val_mse_pred"].append(round(v["mse_pred"], 8))
        hist["val_mae_pred"].append(round(v["mae_pred"], 8))
        hist["test_mse_pred"].append(round(te["mse_pred"], 8))
        hist["test_mae_pred"].append(round(te["mae_pred"], 8))
        hist["tempo_epoca_s"].append(round(ep_time, 4))
        hist["tempo_batch_ms"].append(round(float(np.mean(tempos)) * 1000.0, 2))
        hist["batches_estourados_epoca"].append(estouros)

        if v["mse_pred"] < best_val:
            best_val = v["mse_pred"]
            best_epoch = ep
            if restore_best:
                sd = model.state_dict()
                best_state = {
                    k: (v.clone() if k in trainable_keys else v)
                    for k, v in sd.items()
                }

        print(f"    [ep {ep}/{epochs}] train={tl:.6f} | val_mse_pred={v['mse_pred']:.6f} "
              f"val_mae_pred={v['mae_pred']:.6f} | test_mse_pred={te['mse_pred']:.6f} "
              f"| {ep_time:.1f}s ({hist['tempo_batch_ms'][-1]} ms/batch)")

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        te_best = avaliar(model, test_loader, criterion, device, amp_on,
                          pred_len=pred_len, decode_fn=decode_fn)
        print(f"    [restore_best] estado da epoca {best_epoch} restaurado -> "
              f"test_mse_pred={te_best['mse_pred']:.6f} "
              f"test_mae_pred={te_best['mae_pred']:.6f}")

    return {
        "historico": hist,
        "melhor_epoca": best_epoch,
        "melhor_val_mse_pred": best_val,
        "teste_final_mse_pred": (
            te_best["mse_pred"] if (restore_best and best_state is not None)
            else hist["test_mse_pred"][-1]
        ),
        "teste_final_mae_pred": (
            te_best["mae_pred"] if (restore_best and best_state is not None)
            else hist["test_mae_pred"][-1]
        ),
        "batches_estourados_total": sum(hist["batches_estourados_epoca"]),
        "n_batches_total": n_batches * epochs,
        "tokens_batches_epoca": n_batches,
    }
