"""
Simple Muon optimizer — low-rank SVD truncation + Newton-Schulz.

Key difference from AdaptiveRankMuon:
  - No residual storage (full momentum buffer in float32)
  - Newton-Schulz applied to TRUNCATED M (rank-k approximation), not full M
  - Simpler memory model: just stores M and warm-start basis Q

Used for ablation studies comparing:
  - muon_fixed: Fixed rank ratio throughout training
  - muon_adaptive: Adaptive rank (same xi mechanism) but NS on truncated M

This isolates the effect of NS-on-full-M vs NS-on-truncated-M.
"""

import math
import torch
from .utils import newton_schulz, power_iteration, s_rsi


class MuonSimple(torch.optim.Optimizer):
    """
    Simplified Muon with optional adaptive rank.

    For 2D parameters:
      1. Maintain full momentum buffer M in float32
      2. Compute rank-k truncation via power iteration
      3. Optionally adapt rank based on reconstruction error
      4. Apply Newton-Schulz to the TRUNCATED M (P_k @ R_k^T)
      5. Scale and update parameters

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient.
        weight_decay: L2 weight decay.
        ns_iters: Newton-Schulz iterations.
        rms_scale: RMS scaling factor.
        rho_init: Initial rank ratio.
        rho_min: Minimum rank ratio (adaptive mode).
        rho_max: Maximum rank ratio (adaptive mode).
        adaptive: Enable adaptive rank adjustment.
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0,
                 ns_iters=5, rms_scale=0.04, rho_init=0.15, rho_min=0.05,
                 rho_max=0.60, adaptive=False):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_iters=ns_iters, rms_scale=rms_scale, rho_init=rho_init,
            rho_min=rho_min, rho_max=rho_max, adaptive=adaptive,
        )
        super().__init__(params, defaults)
        self.step_count = 0
        self.param_names: dict[int, str] = {}

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_iters = group["ns_iters"]
            rms_scale = group["rms_scale"]
            rho_init = group["rho_init"]
            rho_min = group["rho_min"]
            rho_max = group["rho_max"]
            adaptive = group["adaptive"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if not torch.isfinite(grad).all():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
                state = self.state[p]

                # Non-2D: plain momentum
                if grad.ndim != 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    state["momentum_buffer"].mul_(mu).add_(grad, alpha=1.0 - mu)
                    p.add_(state["momentum_buffer"], alpha=-lr)
                    if wd > 0:
                        p.add_(p, alpha=-lr * wd)
                    continue

                m, n = grad.shape
                dim = max(m, n)

                # Full momentum buffer (no compression)
                if "M" not in state:
                    state["M"] = torch.zeros_like(grad, dtype=torch.float32)
                    state["rho"] = rho_init

                M = state["M"]
                M.mul_(mu).add_(grad, alpha=1.0 - mu)
                if not torch.isfinite(M).all():
                    M = torch.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)
                    state["M"] = M

                # Low-rank truncation
                k = max(1, int(state["rho"] * min(m, n)))
                if "Q_warm" in state and state["Q_warm"].shape[1] >= k:
                    P_k, R_k = power_iteration(M.float(), state["Q_warm"][:, :k], k)
                else:
                    P_k, R_k = s_rsi(M.float(), k)
                state["Q_warm"] = R_k / (R_k.norm(dim=0, keepdim=True) + 1e-8)
                M_trunc = P_k @ R_k.T

                # Adaptive rank update
                if adaptive and self.step_count > 1000 and self.step_count % 10 == 0:
                    residual_norm = (M.float() - M_trunc).norm("fro")
                    total_norm = M.float().norm("fro") + 1e-8
                    xi = residual_norm / total_norm
                    rho_cur = state["rho"]
                    rho_next = rho_cur + 0.005 if xi > 0.15 else rho_cur - 0.005
                    state["rho"] = max(rho_min, min(rho_max, rho_next))
                    state["xi"] = xi.item()

                state["k"] = k

                # Newton-Schulz on TRUNCATED M (key difference from AdaptiveRankMuon)
                X = newton_schulz(M_trunc, iters=ns_iters)
                if not torch.isfinite(X).all():
                    continue

                scale = math.sqrt(float(dim)) * rms_scale
                p.add_((X * scale).to(p.dtype), alpha=-lr)
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)
