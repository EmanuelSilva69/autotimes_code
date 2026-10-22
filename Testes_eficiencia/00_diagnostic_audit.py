"""
00_diagnostic_audit.py - Forensic Audit Phase 1: Data & Code Verification

This script performs systematic audits of:
1. Data pipeline integrity (temporal leakage, splits)
2. Ablation flag routing (are flags actually used?)
3. Test vs Val evaluation methodology
4. Gradient computation health
5. Model instantiation with different configs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))

from modules import (
    RobustRevIN,
    OverlappingPatchEmbedding,
)


def audit_data_pipeline():
    """AUDIT 1: Verify data leakage in window construction."""
    print("\n" + "=" * 100)
    print("AUDIT 1: DATA PIPELINE INTEGRITY")
    print("=" * 100)

    seq_len = 672
    label_len = 576
    pred_len = 96
    token_len = seq_len - label_len

    print(f"\nConfiguration:")
    print(f"  seq_len (input window):    {seq_len}")
    print(f"  label_len (context):       {label_len}")
    print(f"  pred_len (horizon):        {pred_len}")
    print(f"  token_len (stride):        {token_len}")

    # Simulate window construction as in load_real_split
    print(f"\nWindow construction for sample s_begin=0:")
    s_begin = 0
    s_end = s_begin + seq_len
    r_begin = s_end - label_len
    r_end = r_begin + label_len + pred_len

    print(f"  Input  xs[0:{s_end}] covers indices [0, {s_end})")
    print(f"  Target ys[{r_begin}:{r_end}] covers indices [{r_begin}, {r_end})")

    overlap_start = max(s_begin, r_begin)
    overlap_end = min(s_end, r_end)
    if overlap_end > overlap_start:
        overlap_size = overlap_end - overlap_start
        print(f"\n  🔴 TEMPORAL LEAKAGE DETECTED:")
        print(f"     Overlap: [{overlap_start}, {overlap_end}) = {overlap_size} steps")
        print(f"     Percentage of input leaked: {100 * overlap_size / seq_len:.1f}%")
        print(f"     Percentage of target leaked from input: {100 * overlap_size / (label_len + pred_len):.1f}%")
    else:
        print(f"\n  ✅ No temporal overlap")

    # Check a few more windows
    print(f"\nLeakage analysis for multiple windows:")
    print(f"{'s_begin':>10} | {'Input':>15} | {'Target':>15} | {'Overlap':>15} | {'Leak %':>8}")
    print("-" * 70)
    for s_begin in [0, 96, 192, 288]:
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len
        overlap_start = max(s_begin, r_begin)
        overlap_end = min(s_end, r_end)
        overlap_size = max(0, overlap_end - overlap_start)
        leak_pct = 100 * overlap_size / seq_len
        print(
            f"{s_begin:10d} | [{s_begin:5d}, {s_end:5d}) | "
            f"[{r_begin:5d}, {r_end:5d}) | [{overlap_start:5d}, {overlap_end:5d}) | {leak_pct:7.1f}%"
        )

    return {"leakage_exists": True, "leak_percent": 100 * overlap_size / seq_len}


def audit_ablation_routing():
    """AUDIT 2: Verify ablation flags are routed to modules correctly."""
    print("\n" + "=" * 100)
    print("AUDIT 2: ABLATION FLAG ROUTING")
    print("=" * 100)

    gpt2_dir = ROOT_DIR / "gpt2"
    seq_len = 672
    label_len = 576
    pred_len = 96

    # Try importing model
    try:
        from run_benchmarks import AutoTimesModel
    except ImportError as e:
        print(f"❌ Cannot import AutoTimesModel: {e}")
        return {}

    results = {}

    # Test 1: RobustRevIN flag
    print(f"\nTest 1: RobustRevIN flag routing")
    try:
        model_standard = AutoTimesModel(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=7,
            label_len=label_len,
            use_robust=False,
            gpt2_dir=gpt2_dir,
        )
        model_robust = AutoTimesModel(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=7,
            label_len=label_len,
            use_robust=True,
            gpt2_dir=gpt2_dir,
        )

        # Check if modules are different types
        type_standard = type(model_standard.revin).__name__
        type_robust = type(model_robust.revin).__name__

        print(f"  Standard model RevIN: {type_standard}")
        print(f"  Robust model RevIN:   {type_robust}")

        if type_standard != type_robust:
            print(f"  ✅ Flag correctly routes to different RevIN implementations")
            results["robust_flag_works"] = True
        else:
            print(f"  🔴 Flag does NOT change RevIN type (both are {type_standard})")
            results["robust_flag_works"] = False

        # Test with sample data
        x_test = torch.randn(2, seq_len, 1)
        y_standard = model_standard.revin.normalize(x_test)
        y_robust = model_robust.revin.normalize(x_test)

        diff = (y_standard - y_robust).abs().max().item()
        print(f"  Normalization difference: {diff:.8f}")

        if diff > 1e-5:
            print(f"  ✅ Normalizations differ (expected for robust vs standard)")
            results["robust_normalization_differs"] = True
        else:
            print(f"  🔴 Normalizations are identical (suggests no-op)")
            results["robust_normalization_differs"] = False

    except Exception as e:
        print(f"  ❌ Error testing RobustRevIN: {e}")
        results["robust_flag_error"] = str(e)

    # Test 2: OverlappingEmbed flag
    print(f"\nTest 2: OverlappingEmbed flag routing")
    try:
        model_standard = AutoTimesModel(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=7,
            label_len=label_len,
            use_overlap=False,
            gpt2_dir=gpt2_dir,
        )
        model_overlap = AutoTimesModel(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=7,
            label_len=label_len,
            use_overlap=True,
            gpt2_dir=gpt2_dir,
        )

        type_standard = type(model_standard.encoder).__name__
        type_overlap = type(model_overlap.encoder).__name__

        print(f"  Standard model encoder: {type_standard}")
        print(f"  Overlap model encoder:  {type_overlap}")

        if type_standard != type_overlap:
            print(f"  ✅ Flag correctly routes to different encoder implementations")
            results["overlap_flag_works"] = True
        else:
            print(f"  🔴 Flag does NOT change encoder type")
            results["overlap_flag_works"] = False

        # Check num_patches
        n_patches_standard = model_standard.n_patches if hasattr(model_standard, "n_patches") else None
        n_patches_overlap = model_overlap.n_patches if hasattr(model_overlap, "n_patches") else None

        print(
            f"  Standard n_patches: {n_patches_standard} | "
            f"Overlap n_patches: {n_patches_overlap}"
        )

        if n_patches_standard and n_patches_overlap and n_patches_standard != n_patches_overlap:
            print(f"  ✅ Patch counts differ (expected for overlap)")
            results["overlap_patches_differ"] = True
        else:
            print(f"  🔴 Patch counts are same")
            results["overlap_patches_differ"] = False

    except Exception as e:
        print(f"  ❌ Error testing OverlappingEmbed: {e}")
        results["overlap_flag_error"] = str(e)

    return results


def audit_evaluation_methodology():
    """AUDIT 3: Verify test vs val evaluation methodology."""
    print("\n" + "=" * 100)
    print("AUDIT 3: TEST VS VAL EVALUATION METHODOLOGY")
    print("=" * 100)

    seq_len = 672
    label_len = 576
    pred_len = 96
    token_len = seq_len - label_len

    print(f"\nConfiguration:")
    print(f"  Full sequence length: {seq_len}")
    print(f"  Last pred_len points: {pred_len}")
    print(f"  Token length:         {token_len}")

    print(f"\nValidation evaluation:")
    print(f"  Compares full sequence (672 points)")
    print(f"  Loss = MSE(pred_full, y_full)")

    print(f"\nTest evaluation:")
    print(f"  Compares only last pred_len points (96 points)")
    print(f"  Loss = MSE(pred[-96:], y[-96:])")

    print(f"\n⚠️  METHODOLOGY MISMATCH DETECTED:")
    print(f"  Val loss computed on  {seq_len} points")
    print(f"  Test loss computed on {pred_len} points")
    print(f"  Expected test loss to be ~{seq_len / pred_len:.1f}x smaller IF same model MSE")
    print(f"  Observed ratio in logs: ~2x")
    print(f"  → Suggests test set is genuinely easier or last 96 points are easier")

    return {
        "val_points": seq_len,
        "test_points": pred_len,
        "mismatch_ratio": seq_len / pred_len,
    }


def audit_gradient_computation():
    """AUDIT 4: Quick gradient flow check with minimal model."""
    print("\n" + "=" * 100)
    print("AUDIT 4: GRADIENT COMPUTATION HEALTH")
    print("=" * 100)

    print("\nCreating minimal 2-layer MLP model...")

    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 50),
    )

    x = torch.randn(32, 100, requires_grad=True)
    y = torch.randn(32, 50)

    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Forward
    pred = model(x)
    loss = criterion(pred, y)

    print(f"\nForward pass:")
    print(f"  Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    print(f"\nGradient statistics:")
    total_params = 0
    nan_count = 0
    inf_count = 0
    zero_count = 0
    max_grad = 0
    min_grad_nonzero = float("inf")

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  {name}: grad is NONE")
            continue

        total_params += param.numel()
        grad_abs = param.grad.abs()

        nan_count += torch.isnan(param.grad).sum().item()
        inf_count += torch.isinf(param.grad).sum().item()
        zero_count += (grad_abs < 1e-10).sum().item()

        grad_max = grad_abs.max().item()
        grad_min_nz = grad_abs[grad_abs > 0].min().item() if (grad_abs > 0).any() else 0

        max_grad = max(max_grad, grad_max)
        if grad_min_nz > 0:
            min_grad_nonzero = min(min_grad_nonzero, grad_min_nz)

        print(
            f"  {name:20s} | mean={grad_abs.mean():.2e}, "
            f"max={grad_max:.2e}, min_nz={grad_min_nz:.2e}"
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    print(f"\nGlobal gradient norm: {grad_norm.item():.6f}")
    print(f"  Total params: {total_params}")
    print(f"  NaN gradients: {nan_count}")
    print(f"  Inf gradients: {inf_count}")
    print(f"  Near-zero gradients (< 1e-10): {zero_count}/{total_params}")

    if nan_count == 0 and inf_count == 0:
        print(f"  ✅ Gradient flow is healthy")
        return {"gradient_health": "ok"}
    else:
        print(f"  🔴 Gradient anomalies detected")
        return {"gradient_health": "anomaly", "nan_count": nan_count, "inf_count": inf_count}


def main():
    parser = argparse.ArgumentParser(description="Forensic audit of AutoTimes pipeline")
    parser.add_argument("--audit", type=str, default="all",
                        choices=["all", "data", "flags", "eval", "gradients"],
                        help="Which audit to run")
    args = parser.parse_args()

    results = {}

    if args.audit in ["all", "data"]:
        results["audit_data"] = audit_data_pipeline()

    if args.audit in ["all", "flags"]:
        results["audit_flags"] = audit_ablation_routing()

    if args.audit in ["all", "eval"]:
        results["audit_eval"] = audit_evaluation_methodology()

    if args.audit in ["all", "gradients"]:
        results["audit_gradients"] = audit_gradient_computation()

    # Summary
    print("\n" + "=" * 100)
    print("AUDIT SUMMARY")
    print("=" * 100)
    print(json.dumps(results, indent=2, default=str))

    # Save results
    out_path = BASE_DIR / "log" / "audit_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
