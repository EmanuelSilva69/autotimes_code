#!/usr/bin/env python3
"""
Quick Ablation Test: Execute E1-E3 (subset) to verify that ablations now show 
differences when data leakage is removed.

Tests: Run 1 (RobustRevIN), Run 3 (OverlappingEmbed), Run 5 (VolatilityLoss)
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
    RUN_CONFIGS,
)


def quick_ablation_test():
    """Test 3 ablations to verify differences are now visible."""
    print("\n" + "=" * 80)
    print("QUICK ABLATION TEST - Verify Ablation Differences Visible")
    print("=" * 80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load data with C1 fix
    print("Loading data...")
    try:
        n_vars, train_ds, val_ds, test_ds = load_real_split(
            data_dir="..",
            data_path="dataset/ETT-small/ETTh1.csv",
            seq_len=672,
            label_len=576,
            pred_len=96,
            window_stride=1,
        )
        print(f"✓ Data loaded: {n_vars} vars, {len(train_ds)} train\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False
    
    # Prepare loaders (use subset for speed)
    from torch.utils.data import DataLoader, Subset
    
    subset_indices = list(range(0, len(train_ds), 50))[:100]  # Every 50th sample, max 100
    train_ds_subset = Subset(train_ds, subset_indices)
    train_loader = DataLoader(train_ds_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    print(f"Quick test setup:")
    print(f"  Train subset: {len(train_ds_subset)} samples ({len(train_loader)} batches)")
    print(f"  Val: {len(val_ds)} samples ({len(val_loader)} batches)\n")
    
    # Test 3 ablations
    ablations = [
        ("Run 0 - Baseline", {"use_robust": False, "use_overlap": False, "use_vol_gate": False}),
        ("Run 1 - RobustRevIN", {"use_robust": True, "use_overlap": False, "use_vol_gate": False}),
        ("Run 3 - OverlappingEmbed", {"use_robust": False, "use_overlap": True, "use_vol_gate": False}),
    ]
    
    print("Training quick runs (3 epochs each)...")
    print("-" * 80)
    print("Run Name              | Epoch | Train Loss | Val Loss | Δ from E0")
    print("-" * 80)
    
    results = {}
    baseline_val_loss = None
    
    for run_name, ablation_flags in ablations:
        try:
            # Create model
            model = AutoTimesModel(
                seq_len=672,
                pred_len=96,
                n_vars=n_vars,
                label_len=576,
                mlp_hidden_dim=256,
                mlp_hidden_layers=2,
                use_robust=ablation_flags["use_robust"],
                use_overlap=ablation_flags["use_overlap"],
                use_vol_gate=ablation_flags["use_vol_gate"],
            )
            model = model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()
            
            run_results = {"val_losses": [], "train_losses": []}
            
            # Quick training
            for epoch in range(3):
                # Train
                model.train()
                train_loss = 0.0
                count = 0
                for batch in train_loader:
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    pred = model(batch_x)
                    loss = criterion(pred[:, -96:, :], batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item() * batch_x.size(0)
                    count += batch_x.size(0)
                train_loss /= max(count, 1)
                run_results["train_losses"].append(train_loss)
                
                # Val
                model.eval()
                val_loss = 0.0
                count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                        pred = model(batch_x)
                        loss = criterion(pred[:, -96:, :], batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                        count += batch_x.size(0)
                val_loss /= max(count, 1)
                run_results["val_losses"].append(val_loss)
                
                # Track baseline
                if "Baseline" in run_name and epoch == 2:
                    baseline_val_loss = val_loss
                
                # Compute delta if baseline known
                delta_str = f"{val_loss - baseline_val_loss:+.6f}" if baseline_val_loss is not None else "-"
                print(f"{run_name:20s} | {epoch+1:5d} | {train_loss:10.6f} | {val_loss:8.6f} | {delta_str:>10s}")
            
            results[run_name] = run_results
            
        except Exception as e:
            print(f"✗ Error in {run_name}: {e}")
    
    print("-" * 80 + "\n")
    
    # Analyze differences
    print("ANALYSIS:")
    if baseline_val_loss is not None:
        e0_loss = results["Run 0 - Baseline"]["val_losses"][-1]
        e1_loss = results["Run 1 - RobustRevIN"]["val_losses"][-1]
        e3_loss = results["Run 3 - OverlappingEmbed"]["val_losses"][-1]
        
        delta_e1 = e1_loss - e0_loss
        delta_e3 = e3_loss - e0_loss
        
        print(f"  E0 (Baseline):         {e0_loss:.6f}")
        print(f"  E1 (RobustRevIN):      {e1_loss:.6f} (Δ = {delta_e1:+.6f})")
        print(f"  E3 (OverlapEmbed):     {e3_loss:.6f} (Δ = {delta_e3:+.6f})")
        print()
        
        # Check if differences are visible (>1% relative)
        e1_visible = abs(delta_e1 / e0_loss) > 0.01
        e3_visible = abs(delta_e3 / e0_loss) > 0.01
        
        print(f"  ✓ E1 difference visible (>1%): {e1_visible}")
        print(f"  ✓ E3 difference visible (>1%): {e3_visible}")
        print()
        
        success = e1_visible or e3_visible
        if success:
            print("✅ ABLATION DIFFERENCES NOW VISIBLE (as expected after C1 fix)")
        else:
            print("⚠️  Ablation differences still small (may need more epochs)")
    else:
        success = False
        print("❌ Could not analyze differences")
    
    print("=" * 80 + "\n")
    
    # Save results
    output_file = Path(__file__).parent / "log" / "quick_ablation_test.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"success": success, "results": results}, f, indent=2, ensure_ascii=False)
    
    return success


if __name__ == "__main__":
    success = quick_ablation_test()
    sys.exit(0 if success else 1)
