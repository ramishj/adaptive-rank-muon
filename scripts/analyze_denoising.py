#!/usr/bin/env python3
"""
Denoising analysis tools for Adaptive Rank Muon experiments.

Analyses:
  1. Batch ablation: How does gradient noise vary with batch size?
  2. SVD spectrum: Singular value distribution of gradient/momentum matrices
  3. Gradient SNR: Signal-to-noise ratio across layers and training steps

Usage:
  python scripts/analyze_denoising.py --checkpoint results/Pythia-160M_adaptive_rank_muon/checkpoint_step5000.pt \
      --config configs/pythia_160m.yaml --analysis svd_spectrum --device cuda:0
"""

import argparse
import os
import sys

import torch
import yaml
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ArrowPackedDataset, collate_packed
from src.models.factory import create_model
from torch.utils.data import DataLoader


def svd_spectrum(model, loader, device, output_dir, num_batches=10):
    """Compute and save SVD spectrum of gradients for each 2D layer."""
    model.train()
    os.makedirs(output_dir, exist_ok=True)

    # Accumulate gradients over num_batches
    it = iter(loader)
    for _ in range(num_batches):
        try:
            ids = next(it)
        except StopIteration:
            it = iter(loader)
            ids = next(it)
        ids = ids.to(device)
        loss = model(ids, labels=ids).loss
        loss.backward()

    # Extract SVD spectrum per layer
    csv_path = os.path.join(output_dir, "svd_spectrum.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param_name", "shape", "rank_idx", "singular_value", "cumulative_energy"])
        for name, p in model.named_parameters():
            if p.grad is None or p.grad.ndim != 2:
                continue
            G = p.grad.float()
            S = torch.linalg.svdvals(G)
            total_energy = (S ** 2).sum()
            cumulative = torch.cumsum(S ** 2, dim=0) / total_energy
            for i, (sv, ce) in enumerate(zip(S.tolist(), cumulative.tolist())):
                writer.writerow([name, f"{p.shape[0]}x{p.shape[1]}", i, f"{sv:.6e}", f"{ce:.6f}"])

    print(f"SVD spectrum saved to {csv_path}")
    model.zero_grad()


def gradient_snr(model, loader, device, output_dir, num_trials=5, num_batches=4):
    """
    Estimate gradient SNR per layer.
    SNR = ||E[g]|| / ||Var[g]||^0.5, estimated over num_trials independent batches.
    """
    model.train()
    os.makedirs(output_dir, exist_ok=True)

    # Collect per-trial gradients
    grad_samples = {}
    it = iter(loader)
    for trial in range(num_trials):
        model.zero_grad()
        for _ in range(num_batches):
            try:
                ids = next(it)
            except StopIteration:
                it = iter(loader)
                ids = next(it)
            ids = ids.to(device)
            loss = model(ids, labels=ids).loss
            (loss / num_batches).backward()

        for name, p in model.named_parameters():
            if p.grad is None or p.grad.ndim != 2:
                continue
            if name not in grad_samples:
                grad_samples[name] = []
            grad_samples[name].append(p.grad.float().clone())

    # Compute SNR
    csv_path = os.path.join(output_dir, "gradient_snr.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param_name", "shape", "mean_norm", "std_norm", "snr"])
        for name, grads in grad_samples.items():
            stacked = torch.stack(grads)
            mean_g = stacked.mean(dim=0)
            mean_norm = mean_g.norm().item()
            var_norm = stacked.std(dim=0).norm().item()
            snr = mean_norm / (var_norm + 1e-8)
            shape = f"{grads[0].shape[0]}x{grads[0].shape[1]}"
            writer.writerow([name, shape, f"{mean_norm:.6e}", f"{var_norm:.6e}", f"{snr:.4f}"])

    print(f"Gradient SNR saved to {csv_path}")
    model.zero_grad()


def main():
    parser = argparse.ArgumentParser(description="Denoising Analysis")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load (optional)")
    parser.add_argument("--analysis", required=True, choices=["svd_spectrum", "gradient_snr"],
                        help="Analysis to run")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="analysis_output")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Model
    model = create_model(config, device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Data
    data_cfg = config["data"]
    dataset = ArrowPackedDataset(data_cfg["path"])
    loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True,
        collate_fn=collate_packed, drop_last=True,
    )

    if args.analysis == "svd_spectrum":
        svd_spectrum(model, loader, device, args.output_dir)
    elif args.analysis == "gradient_snr":
        gradient_snr(model, loader, device, args.output_dir)


if __name__ == "__main__":
    main()
