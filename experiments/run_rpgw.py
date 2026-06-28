"""
Run RPGW-Net Experiment
========================
Usage:
    python experiments/run_rpgw.py --source 0 --target 3 --shots 5 --snr 5
    python experiments/run_rpgw.py --source 0 --target 3 --gw_type partial
    python experiments/run_rpgw.py --source 0 --target 3 --ablation no_gw
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rpgw.train import train_rpgw


def parse_args():
    parser = argparse.ArgumentParser(description="Train RPGW-Net on CWRU")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--source", type=int, default=0,
                        help="Source domain motor load (0-3)")
    parser.add_argument("--target", type=int, default=3,
                        help="Target domain motor load (0-3)")
    parser.add_argument("--shots", type=int, default=5,
                        help="Few-shot: samples per class in target")
    parser.add_argument("--snr", type=float, default=5.0,
                        help="Target SNR in dB (inf = no noise)")
    parser.add_argument("--gw_type", type=str, default="partial",
                        choices=["vanilla", "entropic", "fused", "partial"],
                        help="GW variant")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_gw", "no_gat", "no_denoise", "mmd_only"],
                        help="Ablation study mode")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    config["experiment"]["source_load"] = args.source
    config["experiment"]["target_load"] = args.target
    config["experiment"]["seed"] = args.seed
    config["train"]["n_shot"] = args.shots
    config["train"]["epochs"] = args.epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["lr"] = args.lr
    config["noise"]["target_snr_db"] = args.snr
    config["noise"]["enabled"] = (args.snr != float("inf"))
    config["model"]["gw"]["type"] = args.gw_type

    # Experiment name
    exp_name = args.name or (
        f"rpgw_{args.gw_type}_s{args.source}t{args.target}"
        f"_shot{args.shots}_snr{args.snr}_{datetime.now():%m%d-%H%M}"
    )
    print(f"Experiment: {exp_name}")
    print(f"  Source: load={args.source}, Target: load={args.target}")
    print(f"  GW type: {args.gw_type}, Shots: {args.shots}, SNR: {args.snr} dB")

    # Handle ablation
    if args.ablation:
        print(f"  Ablation: {args.ablation}")
        if args.ablation == "no_gw":
            config["train"]["loss"]["gw_weight"] = 0.0
        elif args.ablation == "mmd_only":
            config["train"]["loss"]["gw_weight"] = 0.0
            config["train"]["loss"]["mmd_weight"] = 1.0
        elif args.ablation == "no_gat":
            print("  (not implemented yet — use GCN? MLP?)")
        elif args.ablation == "no_denoise":
            config["preprocess"]["denoise"] = False

    # Save dir
    save_dir = os.path.join(config["experiment"]["checkpoint_dir"], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    config["experiment"]["name"] = exp_name

    # Train
    import torch
    torch.manual_seed(args.seed)

    best_acc = train_rpgw(config, save_dir)
    print(f"\n{'='*50}")
    print(f"Experiment: {exp_name}")
    print(f"Best target accuracy: {best_acc:.4f}")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
