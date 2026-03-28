"""
Main training loop with layer-aware optimizer routing.

Design decisions:
  - Layer classification: embed/norm/bias/lm_head → AdamW, 2D weights → Muon variant
  - Embed routing is handled HERE (not in the optimizer) so optimizers stay clean
  - CSV logging: train.csv (step, loss, lr, tok/s) + layers.csv (per-layer rank stats)
  - Checkpointing: model state dict + metadata (no optimizer state to save disk)
"""

import csv
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..optimizers import AdaptiveRankMuon, MuonSimple
from .lr_schedule import get_lr


def is_embed_or_norm(name: str) -> bool:
    """Returns True for params that should use AdamW (embeds, norms, biases, lm_head)."""
    nl = name.lower()
    return any(kw in nl for kw in ("embed", "norm", "lm_head", ".bias"))


def classify_layer(name: str) -> str:
    """Classify a parameter name into layer type for logging."""
    nl = name.lower()
    if "self_attn" in nl or "attention" in nl:
        return "attn"
    if "mlp" in nl:
        return "mlp"
    return "other"


class CombinedOptimizer:
    """Wraps a Muon-family optimizer + AdamW for embed/norm params."""

    def __init__(self, opt_muon, opt_adamw):
        self.opt_muon = opt_muon
        self.opt_adamw = opt_adamw
        self.param_names = opt_muon.param_names

    @property
    def state(self):
        combined = dict(self.opt_muon.state)
        combined.update(self.opt_adamw.state)
        return combined

    def step(self):
        self.opt_muon.step()
        self.opt_adamw.step()

    def zero_grad(self):
        self.opt_muon.zero_grad()
        self.opt_adamw.zero_grad()

    @property
    def param_groups(self):
        return self.opt_muon.param_groups + self.opt_adamw.param_groups


class Trainer:
    """
    Training loop for language model experiments.

    Args:
        model: The language model.
        loader: DataLoader yielding (batch_size, seq_len) tensors.
        config: Full experiment config dict.
        optimizer_name: Override optimizer name from config.
        device: Training device.
        save_dir: Output directory for logs and checkpoints.
    """

    def __init__(self, model: nn.Module, loader: DataLoader, config: dict,
                 optimizer_name: Optional[str] = None, device: torch.device = None,
                 save_dir: str = "results"):
        self.model = model
        self.loader = loader
        self.config = config
        self.device = device or torch.device("cuda:0")
        self.save_dir = save_dir

        train_cfg = config["training"]
        opt_cfg = config["optimizer"]

        self.max_steps = train_cfg["max_steps"]
        self.lr = train_cfg["lr"]
        self.warmup_steps = train_cfg["warmup_steps"]
        self.grad_accum = train_cfg.get("grad_accum", 1)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.use_bf16 = train_cfg.get("bf16", True)
        self.log_freq = train_cfg.get("log_freq", 50)
        self.save_every = train_cfg.get("save_every", 5000)

        self.optimizer_name = optimizer_name or opt_cfg["name"]
        self.optimizer = self._build_optimizer(opt_cfg)

        os.makedirs(save_dir, exist_ok=True)

    def _build_optimizer(self, opt_cfg: dict):
        """Build optimizer with layer-aware param routing."""
        name = self.optimizer_name

        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.config["training"].get("weight_decay", 0.01),
            )

        # Split params: embed/norm/bias → AdamW, rest → Muon variant
        embed_params = []
        muon_params = []
        muon_names = {}
        for pname, p in self.model.named_parameters():
            if is_embed_or_norm(pname):
                embed_params.append(p)
            else:
                muon_params.append(p)
                if p.ndim >= 2:
                    muon_names[id(p)] = pname

        n_embed = sum(p.numel() for p in embed_params)
        n_muon = sum(p.numel() for p in muon_params if p.ndim >= 2)
        print(f"Optimizer routing: embed/norm/bias (AdamW) = {n_embed:,} params")
        print(f"                   2D weights ({name}) = {n_muon:,} params")

        # Build the Muon variant
        if name == "adaptive_rank_muon":
            opt_muon = AdaptiveRankMuon(
                muon_params, lr=self.lr,
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=self.config["training"].get("weight_decay", 0.0),
                ns_iters=opt_cfg.get("ns_iters", 5),
                rms_scale=opt_cfg.get("rms_scale", 0.04),
                rho0=opt_cfg.get("rho0", 0.15),
                rho_min=opt_cfg.get("rho_min", 0.0625),
                rho_max=opt_cfg.get("rho_max", 0.6),
                xi_thresh=opt_cfg.get("xi_thresh", 0.15),
                delta_rho=opt_cfg.get("delta_rho", 0.005),
                beta3=opt_cfg.get("beta3", 0.9),
                adapt_freq=opt_cfg.get("adapt_freq", 10),
            )
        elif name == "muon_fixed":
            opt_muon = MuonSimple(
                muon_params, lr=self.lr,
                momentum=opt_cfg.get("momentum", 0.95),
                rms_scale=opt_cfg.get("rms_scale", 0.04),
                rho_init=opt_cfg.get("rho0", 0.15),
                adaptive=False,
            )
        elif name == "muon_adaptive":
            opt_muon = MuonSimple(
                muon_params, lr=self.lr,
                momentum=opt_cfg.get("momentum", 0.95),
                rms_scale=opt_cfg.get("rms_scale", 0.04),
                rho_init=opt_cfg.get("rho0", 0.15),
                rho_min=opt_cfg.get("rho_min", 0.05),
                rho_max=opt_cfg.get("rho_max", 0.6),
                adaptive=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")

        opt_muon.param_names = muon_names
        opt_adamw = torch.optim.AdamW(embed_params, lr=self.lr, weight_decay=0.01)

        return CombinedOptimizer(opt_muon, opt_adamw)

    def _set_lr(self, lr: float):
        """Set learning rate across all param groups."""
        if isinstance(self.optimizer, CombinedOptimizer):
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def train(self):
        """Run the training loop."""
        is_muon = self.optimizer_name != "adamw"

        # CSV loggers
        train_csv_path = os.path.join(self.save_dir, "train.csv")
        ft = open(train_csv_path, "w", newline="")
        wt = csv.writer(ft)
        wt.writerow(["step", "loss", "lr", "tok_per_sec"])

        fl, wl = None, None
        if is_muon:
            layer_csv_path = os.path.join(self.save_dir, "layers.csv")
            fl = open(layer_csv_path, "w", newline="")
            wl = csv.writer(fl)
            wl.writerow(["step", "param_name", "layer_type", "m", "n", "k", "rho", "xi"])

        it = iter(self.loader)
        t0 = time.time()
        last_log = t0
        total_tokens = 0
        accum_loss = 0.0

        batch_size = self.config["training"]["batch_size"]
        seq_len = self.config["data"]["seq_len"]

        print(f"\n{'='*70}")
        print(f"Training: {self.max_steps} steps | batch={batch_size}x{self.grad_accum} | lr={self.lr}")
        print(f"Optimizer: {self.optimizer_name} | bf16={self.use_bf16}")
        print(f"Output: {self.save_dir}")
        print(f"{'='*70}\n", flush=True)

        for step in range(1, self.max_steps + 1):
            cur_lr = get_lr(step, self.max_steps, self.lr, self.warmup_steps)
            self._set_lr(cur_lr)

            # Gradient accumulation
            for _ in range(self.grad_accum):
                try:
                    ids = next(it)
                except StopIteration:
                    it = iter(self.loader)
                    ids = next(it)
                ids = ids.to(self.device)
                total_tokens += ids.numel()

                if self.use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = self.model(ids, labels=ids).loss
                    if self.grad_accum > 1:
                        loss = loss / self.grad_accum
                    loss.backward()
                else:
                    loss = self.model(ids, labels=ids).loss
                    if self.grad_accum > 1:
                        loss = loss / self.grad_accum
                    loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step == 1:
                print(f"Step 1 — loss={accum_loss:.4f}", flush=True)

            # Logging
            if step % self.log_freq == 0:
                dt = time.time() - last_log
                tok_per_sec = (self.log_freq * self.grad_accum * batch_size * seq_len) / max(dt, 1e-6)
                avg_loss = accum_loss / self.log_freq

                wt.writerow([step, f"{avg_loss:.6f}", f"{cur_lr:.6e}", f"{tok_per_sec:.0f}"])
                ft.flush()

                # Layer stats (every 10x log_freq to reduce overhead)
                if is_muon and wl and step % (self.log_freq * 10) == 0:
                    for p, s in self.optimizer.state.items():
                        if "k" not in s or id(p) not in self.optimizer.param_names:
                            continue
                        name = self.optimizer.param_names[id(p)]
                        wl.writerow([
                            step, name, classify_layer(name),
                            p.shape[0], p.shape[1], s["k"],
                            f"{s.get('rho_moment', s.get('rho', 0)):.4f}",
                            f"{s.get('xi', 0):.4f}",
                        ])
                    fl.flush()

                elapsed = time.time() - t0
                eta_min = (self.max_steps - step) / step * elapsed / 60
                print(
                    f"{step:6d}/{self.max_steps} | loss {avg_loss:.4f} | "
                    f"lr {cur_lr:.2e} | {tok_per_sec:.0f} tok/s | ETA {eta_min:.0f}m",
                    flush=True,
                )
                accum_loss = 0.0
                last_log = time.time()

            # Checkpoint
            if self.save_every > 0 and step % self.save_every == 0:
                self._save_checkpoint(step, total_tokens, accum_loss or loss.item())

        # Final checkpoint
        self._save_checkpoint(self.max_steps, total_tokens, loss.item(), final=True)

        ft.close()
        if fl:
            fl.close()

        elapsed = time.time() - t0
        print(f"\n✅ Done. {self.max_steps} steps in {elapsed/3600:.1f}h")

    def _save_checkpoint(self, step: int, total_tokens: int, loss: float, final: bool = False):
        """Save a model checkpoint."""
        tag = "final" if final else f"step{step}"
        ckpt_path = os.path.join(self.save_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "total_tokens": total_tokens,
            "loss": loss,
            "config": self.config,
        }, ckpt_path)
        sz_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
        print(f"💾 Checkpoint: {ckpt_path} ({sz_mb:.0f} MB)", flush=True)
