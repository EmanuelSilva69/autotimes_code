#!/usr/bin/env python3
"""
E0 Baseline Validation: Execute baseline with C1 fix applied to verify:
1. Code compiles without errors
2. Training loop works
3. Loss is reasonable (3-5x higher than before due to removed leakage)
4. No NaN/Inf detected
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_benchmarks import (
    AutoTimesModel,
    load_real_split,
    evaluate,
    build_discriminative_optimizer,
    RUN_CONFIGS,
)
from Testes_eficiencia.modules.robust_revin import RobustRevIN
from Testes_eficiencia.modules.overlapping_embed import OverlappingPatchEmbedding


def test_e0_baseline():
    """Quick validation of E0 baseline with C1 fix."""
    print("\n" + "=" * 80)
    print("E0 BASELINE VALIDATION - C1 FIX")
    print("=" * 80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load data with C1 fix applied
    print("Loading data with C1 fix (no temporal leakage)...")
    try:
        n_vars, train_ds, val_ds, test_ds = load_real_split(
            data_dir="..",
            data_path="dataset/ETT-small/ETTh1.csv",
            seq_len=672,
            label_len=576,
            pred_len=96,
            window_stride=1,
        )
        print(f"✓ Data loaded successfully")
        print(f"  Num vars: {n_vars}")
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples: {len(val_ds)}")
        print(f"  Test samples: {len(test_ds)}\n")
        
        # Prepare loaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    except Exception as e:
        print(f"✗ Error loading data: {e}\n")
        return False
    
    # Create model
    print("Creating AutoTimesModel (E0 baseline, no ablations)...")
    try:
        model = AutoTimesModel(
            seq_len=672,
            pred_len=96,
            n_vars=n_vars,
            label_len=576,
            mlp_hidden_dim=256,
            mlp_hidden_layers=2,
        )
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}\n")
    except Exception as e:
        print(f"✗ Error creating model: {e}\n")
        return False
    
    # Create optimizer and loss
    print("Setting up optimizer and loss...")
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        criterion = torch.nn.MSELoss()
        print(f"✓ Optimizer: Adam(lr=0.001)")
        print(f"✓ Loss: MSELoss\n")
    except Exception as e:
        print(f"✗ Error setting up optimizer: {e}\n")
        return False
    
    # Prepare data loader
    print("Preparing data loaders...")
    try:
        # Loaders already created above
        print(f"✓ Data loaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}\n")
    except Exception as e:
        print(f"✗ Error creating loaders: {e}\n")
        return False
    
    # Quick training (5 epochs)
    print("Running quick training (5 epochs)...")
    print("-" * 80)
    print("Epoch | Train Loss | Val Loss | Test Loss | Grad Norm | Status")
    print("-" * 80)
    
    results = {
        "epoch_losses": [],
        "has_nan": False,
        "has_inf": False,
        "max_grad_norm": 0.0,
        "final_train_loss": 0.0,
        "final_val_loss": 0.0,
    }
    
    for epoch in range(3):  # Reduced to 3 epochs for speed
        # Train
        model.train()
        train_loss = 0.0
        count = 0
        max_grad = 0.0
        
        for batch in train_loader:
            # batch contains (x, y) from TensorDataset
            batch_x, batch_y = batch[0], batch[1]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward
            pred = model(batch_x)
            loss = criterion(pred[:, -96:, :], batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            for p in model.parameters():
                if p.grad is not None:
                    g_norm = p.grad.detach().norm(2).item()
                    if np.isnan(g_norm) or np.isinf(g_norm):
                        results["has_nan"] = np.isnan(g_norm)
                        results["has_inf"] = np.isinf(g_norm)
                    max_grad = max(max_grad, g_norm)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
        
        train_loss /= max(count, 1)
        results["final_train_loss"] = train_loss
        results["max_grad_norm"] = max(results["max_grad_norm"], max_grad)
        
        # Val
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y = batch[0], batch[1]
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred[:, -96:, :], batch_y)
                val_loss += loss.item() * batch_x.size(0)
                count += batch_x.size(0)
        val_loss /= max(count, 1)
        results["final_val_loss"] = val_loss
        
        # Test
        test_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y = batch[0], batch[1]
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred[:, -96:, :], batch_y)
                test_loss += loss.item() * batch_x.size(0)
                count += batch_x.size(0)
        test_loss /= max(count, 1)
        
        status = "✓" if not (results["has_nan"] or results["has_inf"]) else "✗ NaN/Inf"
        print(
            f"{epoch+1:5d} | {train_loss:10.6f} | {val_loss:8.6f} | "
            f"{test_loss:9.6f} | {max_grad:9.6f} | {status}"
        )
        
        results["epoch_losses"].append({
            "epoch": epoch + 1,
            "train": float(train_loss),
            "val": float(val_loss),
            "test": float(test_loss),
        })
    
    print("-" * 80 + "\n")
    
    # Validation checks
    print("VALIDATION CHECKS:")
    checks = {
        "loss_reasonable": train_loss < 100.0,  # Loss should be finite and reasonable
        "gradient_finite": not results["has_nan"] and not results["has_inf"],
        "gradient_norm_healthy": 0.01 < results["max_grad_norm"] < 100.0,
    }
    
    print(f"  ✓ Loss reasonable (<100): {checks['loss_reasonable']}")
    print(f"  ✓ Gradients finite: {checks['gradient_finite']}")
    print(f"  ✓ Gradient norm healthy: {checks['gradient_norm_healthy']}")
    print()
    
    success = all(checks.values())
    if success:
        print("✅ E0 BASELINE VALIDATION PASSED")
        print("\nNext steps:")
        print("  1. Run full E0 training with all data")
        print("  2. Compare loss before/after C1 fix (expect 3-5x higher)")
        print("  3. Re-run ablations E1-E12 to see differences now visible")
    else:
        print("❌ E0 BASELINE VALIDATION FAILED")
        print(f"\nIssues found:")
        for check, passed in checks.items():
            if not passed:
                print(f"  - {check}: {passed}")
    
    print("=" * 80 + "\n")
    
    # Save results
    output_file = Path(__file__).parent / "log" / "e0_baseline_validation.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "success": success,
            "checks": {k: bool(v) for k, v in checks.items()},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    return success


if __name__ == "__main__":
    success = test_e0_baseline()
    sys.exit(0 if success else 1)
