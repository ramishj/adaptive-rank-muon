#!/usr/bin/env python3
"""
Main entry point for Adaptive Rank Muon experiments.

Usage:
  python scripts/train.py --config configs/pythia_160m.yaml --optimizer adaptive_rank_muon
  python scripts/train.py --config configs/llama_350m.yaml --optimizer adamw --device cuda:1
  python scripts/train.py --config configs/pythia_70m.yaml --max_steps 5 --batch_size 2  # quick test
"""

import argparse
import os
import sys
import random

import numpy as np
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ArrowPackedDataset, collate_packed
from src.models.factory import create_model
from src.training.trainer import Trainer
from torch.utils.data import DataLoader


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Adaptive Rank Muon Training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--optimizer", choices=["adamw", "adaptive_rank_muon", "muon_fixed", "muon_adaptive"],
                        default=None, help="Override optimizer from config")
    parser.add_argument("--device", default=None, help="Device (default: cuda:0 or cpu)")
    parser.add_argument("--save_dir", default=None, help="Override output directory")
    # CLI overrides for quick experiments
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--bf16", action="store_true", default=None, help="Force bf16 training")
    parser.add_argument("--no_bf16", action="store_true", default=False, help="Force fp32 training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.grad_accum is not None:
        config["training"]["grad_accum"] = args.grad_accum
    if args.bf16:
        config["training"]["bf16"] = True
    if args.no_bf16:
        config["training"]["bf16"] = False

    # Reproducibility
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    # Device
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Resolve optimizer name
    optimizer_name = args.optimizer or config["optimizer"]["name"]

    # Resolve save directory
    save_dir = args.save_dir or config["output"]["save_dir"]
    save_dir = save_dir.replace("{model.name}", config["model"]["name"])
    save_dir = save_dir.replace("{optimizer.name}", optimizer_name)

    print(f"Config: {args.config}")
    print(f"Device: {device}")

    # Dataset
    data_cfg = config["data"]
    dataset = ArrowPackedDataset(data_cfg["path"])
    print(f"Dataset: {len(dataset)} examples from {data_cfg['path']}")

    # num_workers=0 for NFS to avoid temp file cleanup issues on full-disk servers
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=collate_packed,
        drop_last=True,
    )

    # Model
    model = create_model(config, device)

    # Train
    trainer = Trainer(
        model=model,
        loader=loader,
        config=config,
        optimizer_name=optimizer_name,
        device=device,
        save_dir=save_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
