#!/usr/bin/env python3
"""
Phase I - Memorization Test: Validates that the model can memorize and learn
on a small subset of 16 samples without any data leakage or architectural issues.

If this test PASSES:
  - Backward pass works correctly
  - Gradient flow is healthy
  - Loss can reduce to near-zero on tiny dataset
  - No numerical instability

If this test FAILS:
  - Indicates fundamental problem with model architecture
  - Suggests issue with training loop, loss computation, or gradient flow
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_benchmarks import AutoTimesModel


class TinyDataset(Dataset):
    """Minimal synthetic dataset for memorization test."""

    def __init__(self, num_samples: int = 16, seq_len: int = 672, pred_len: int = 96, num_vars: int = 7):
        """
        Args:
            num_samples: Number of samples (default 16 for tiny test)
            seq_len: Input sequence length
            pred_len: Prediction length
            num_vars: Number of variables
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_vars = num_vars

        # Generate random but fixed sequences
        np.random.seed(42)
        torch.manual_seed(42)

        # Create X: (num_samples, seq_len, num_vars)
        # Create Y: (num_samples, pred_len, num_vars)
        self.xs = torch.randn(num_samples, seq_len, num_vars, dtype=torch.float32)
        self.ys = torch.randn(num_samples, pred_len, num_vars, dtype=torch.float32)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx: int):
        """Return (x, y) without time marks since model doesn't need them."""
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y


def run_memorization_test(
    num_samples: int = 16,
    seq_len: int = 672,
    pred_len: int = 96,
    label_len: int = 576,
    num_vars: int = 1,  # Changed to 1 for univariate (as model expects)
    hidden_dim: int = 256,
    num_layers: int = 2,
    batch_size: int = 4,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = None,
) -> dict:
    """
    Run memorization test on tiny dataset.
    
    Returns:
        dict with results: initial_loss, final_loss, can_memorize, gradient_health
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print(f"PHASE I - MEMORIZATION TEST")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Dataset: {num_samples} samples, seq_len={seq_len}, pred_len={pred_len}, vars={num_vars}")
    print(f"Model: hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"Training: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"{'-'*80}\n")

    # Create tiny dataset and loader
    dataset = TinyDataset(num_samples=num_samples, seq_len=seq_len, pred_len=pred_len, num_vars=num_vars)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = AutoTimesModel(
        seq_len=seq_len,
        pred_len=pred_len,
        n_vars=num_vars,
        label_len=label_len,
        mlp_hidden_dim=hidden_dim,
        mlp_hidden_layers=num_layers,
        use_robust=False,
        use_overlap=False,
        use_vol_gate=False,
        unfreeze_layers=0,
    )
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    # Loss
    criterion = nn.MSELoss()

    # Training loop
    losses = []
    grad_norms = []
    epoch_times = []

    print("Epoch | Loss       | ΔLoss    | Grad Norm | Status")
    print("-" * 55)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_count = 0
        epoch_start = datetime.now()

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            # Forward
            pred = model(x)
            # Extract prediction for the last pred_len timesteps
            # Model outputs full sequence (B, seq_len, 1), we need last pred_len points
            pred_horizon = pred[:, -pred_len:, :]  # (B, pred_len, 1)
            loss = criterion(pred_horizon, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient norm
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_grad_norm = p.grad.detach().data.norm(2)
                    total_grad_norm += param_grad_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # Check for NaN/Inf
            has_nan = False
            has_inf = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        has_nan = True
                    if torch.isinf(p.grad).any():
                        has_inf = True

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            epoch_count += x.size(0)
            grad_norms.append(total_grad_norm)

            if has_nan or has_inf:
                print(f"WARNING: Epoch {epoch}, Batch {batch_idx}: NaN={has_nan}, Inf={has_inf}")

        avg_epoch_loss = epoch_loss / max(epoch_count, 1)
        losses.append(avg_epoch_loss)
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        epoch_times.append(epoch_time)

        # Compute loss delta
        if len(losses) > 1:
            delta_loss = losses[-2] - losses[-1]
        else:
            delta_loss = 0.0

        avg_grad_norm = np.mean(grad_norms[-len(loader) :])

        # Print progress
        status = "✓" if avg_epoch_loss < 1.0 else "..." if epoch % 5 == 0 else ""
        print(f"{epoch+1:5d} | {avg_epoch_loss:10.6f} | {delta_loss:8.6f} | {avg_grad_norm:9.6f} | {status}")

    print(f"{'-'*55}\n")

    # Analyze results
    initial_loss = losses[0]
    final_loss = losses[-1]
    total_reduction = initial_loss - final_loss
    pct_reduction = (total_reduction / initial_loss * 100) if initial_loss > 0 else 0

    avg_grad_norm_overall = np.mean(grad_norms)
    max_grad_norm = np.max(grad_norms)
    min_grad_norm = np.min(grad_norms)

    # Test criteria
    can_memorize = final_loss < 0.1 * initial_loss  # Loss reduced by >90%
    gradient_health = 0.01 < avg_grad_norm_overall < 10.0  # Healthy range
    no_nan_inf = not any(np.isnan(g) or np.isinf(g) for g in grad_norms)

    print(f"RESULTS:")
    print(f"  Initial Loss:        {initial_loss:12.6f}")
    print(f"  Final Loss:          {final_loss:12.6f}")
    print(f"  Reduction:           {total_reduction:12.6f} ({pct_reduction:6.2f}%)")
    print(f"  Avg Gradient Norm:   {avg_grad_norm_overall:12.6f}")
    print(f"  Max Gradient Norm:   {max_grad_norm:12.6f}")
    print(f"  Min Gradient Norm:   {min_grad_norm:12.6f}")
    print()
    print(f"VALIDATION:")
    print(f"  ✓ Can Memorize (>90% reduction):  {can_memorize}")
    print(f"  ✓ Gradient Health (0.01-10.0):   {gradient_health}")
    print(f"  ✓ No NaN/Inf:                     {no_nan_inf}")
    print()

    success = can_memorize and gradient_health and no_nan_inf
    status_msg = "✅ PASS - Model can learn" if success else "❌ FAIL - Model has issues"
    print(f"OVERALL: {status_msg}")
    print(f"{'='*80}\n")

    return {
        "success": bool(success),
        "can_memorize": bool(can_memorize),
        "gradient_health": bool(gradient_health),
        "no_nan_inf": bool(no_nan_inf),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "pct_reduction": float(pct_reduction),
        "avg_grad_norm": float(avg_grad_norm_overall),
        "max_grad_norm": float(max_grad_norm),
        "min_grad_norm": float(min_grad_norm),
        "losses": [float(x) for x in losses],
        "grad_norms": [float(x) for x in grad_norms],
        "epoch_times": epoch_times,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase I - Memorization Test")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Run test
    results = run_memorization_test(
        num_samples=args.num_samples,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
    )

    # Save results
    output_file = Path(__file__).parent / "log" / "phase_i_memorization_test.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

    # Exit code
    sys.exit(0 if results["success"] else 1)
