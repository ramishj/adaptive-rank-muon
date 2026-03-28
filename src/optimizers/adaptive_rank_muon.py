"""
Adaptive Rank Muon — GraSP v8 with BF16 storage.

Key design:
  - Per-layer adaptive rank via power iteration + s-RSI warm start
  - BF16 storage of P, R, and residual (momentum decomposed as P @ R^T + residual)
  - Adaptive rho via xi threshold + EMA smoothing (beta3)
  - Newton-Schulz applied to FULL reconstructed momentum M (not truncated)
  - Non-2D parameters: plain momentum update

This is the main optimizer for the Adaptive Rank Muon paper.
The key insight vs MuonSimple is that NS operates on the full M,
while the low-rank decomposition is only for memory-efficient storage.
"""

import math
import torch
from .utils import newton_schulz, power_iteration, s_rsi


class AdaptiveRankMuon(torch.optim.Optimizer):
    """
    Muon optimizer with GraSP-style adaptive rank and BF16 momentum storage.

    For 2D weight matrices:
      1. Reconstruct full momentum M from BF16 low-rank factors + residual
      2. Update M with new gradient: M = mu * M + (1 - mu) * grad
      3. Compute rank-k approximation via power iteration (warm) or s-RSI (cold)
      4. Adaptively adjust rank based on reconstruction error xi
      5. Store decomposition in BF16: P, R, residual = M - P @ R^T
      6. Apply Newton-Schulz to full M, scale, and update parameters

    For non-2D parameters (biases, norms handled externally by trainer):
      Plain momentum update.

    Args:
        params: Parameters to optimize (should be the Muon group, not embeds/norms).
        lr: Learning rate.
        momentum: Momentum coefficient (mu).
        weight_decay: L2 weight decay.
        ns_iters: Newton-Schulz iterations.
        rms_scale: RMS scaling factor for the orthogonal update.
        rho0: Initial rank ratio (fraction of min(m, n)).
        rho_min: Minimum rank ratio.
        rho_max: Maximum rank ratio.
        xi_thresh: Reconstruction error threshold for rank adaptation.
        delta_rho: Step size for rank ratio adjustment.
        beta3: EMA coefficient for rank ratio smoothing.
        adapt_freq: How often to run rank adaptation (in steps).
        fixed_rank: If True, disable adaptive rank (use rho0 throughout).
    """

    def __init__(self, params, lr=5e-4, momentum=0.9, weight_decay=0.0,
                 ns_iters=5, rms_scale=0.04, rho0=0.15, rho_min=0.0625,
                 rho_max=0.6, xi_thresh=0.15, delta_rho=0.005, beta3=0.9,
                 adapt_freq=10, fixed_rank=False):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_iters=ns_iters, rms_scale=rms_scale, rho0=rho0,
            rho_min=rho_min, rho_max=rho_max, xi_thresh=xi_thresh,
            delta_rho=delta_rho, beta3=beta3, adapt_freq=adapt_freq,
            fixed_rank=fixed_rank,
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
            rho0 = group["rho0"]
            rho_min = group["rho_min"]
            rho_max = group["rho_max"]
            xi_thresh = group["xi_thresh"]
            delta_rho = group["delta_rho"]
            beta3 = group["beta3"]
            adapt_freq = group["adapt_freq"]
            fixed_rank = group["fixed_rank"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if not torch.isfinite(grad).all():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)

                state = self.state[p]

                # --- Non-2D parameters: plain momentum ---
                if grad.ndim != 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    state["momentum_buffer"].mul_(mu).add_(grad, alpha=1.0 - mu)
                    p.add_(state["momentum_buffer"], alpha=-lr)
                    if wd > 0:
                        p.add_(p, alpha=-lr * wd)
                    continue

                # --- 2D parameters: full Muon + GraSP ---
                m, n = grad.shape
                dim = max(m, n)

                # Reconstruct momentum from BF16 storage or initialize
                if "M_res_bf16" in state:
                    M_res = state["M_res_bf16"].float()
                    P = state["P_bf16"].float()
                    R = state["R_bf16"].float()
                    M = M_res + P @ R.T
                    Q_warm = R / (R.norm(dim=0, keepdim=True) + 1e-8)
                else:
                    M = torch.zeros_like(grad, dtype=torch.float32)
                    Q_warm = None
                    state["rho"] = rho0
                    state["rho_moment"] = rho0

                # Momentum update
                M.mul_(mu).add_(grad, alpha=1.0 - mu)
                if not torch.isfinite(M).all():
                    M = torch.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)

                # Compute target rank
                if fixed_rank:
                    k = max(1, int(rho0 * min(m, n)))
                else:
                    k = max(1, int(state["rho_moment"] * min(m, n)))

                # Rank-k subspace: warm start (power iteration) or cold start (s-RSI)
                if Q_warm is not None and Q_warm.shape[1] >= k:
                    P_k, R_k = power_iteration(M, Q_warm[:, :k], k)
                else:
                    P_k, R_k = s_rsi(M, k)

                # Adaptive rank adjustment (after warmup, every adapt_freq steps)
                if not fixed_rank and self.step_count > 1000 and (self.step_count % adapt_freq == 0):
                    M_norm = M.norm("fro") + 1e-8
                    if torch.isfinite(M_norm):
                        xi = (M - P_k @ R_k.T).norm("fro") / M_norm
                        rho_cur = state["rho_moment"]
                        rho_next = rho_cur + delta_rho if xi > xi_thresh else rho_cur - delta_rho
                        rho_next = max(rho_min, min(rho_max, rho_next))
                        state["rho_moment"] = beta3 * state["rho_moment"] + (1 - beta3) * rho_next
                        state["rho"] = state["rho_moment"]
                        state["xi"] = xi.item()
                else:
                    if "xi" not in state:
                        M_norm = M.norm("fro") + 1e-8
                        state["xi"] = (M - P_k @ R_k.T).norm("fro").item() / M_norm.item()

                state["k"] = k

                # Store decomposition in BF16
                M_res = M - P_k @ R_k.T
                state["M_res_bf16"] = M_res.bfloat16()
                state["P_bf16"] = P_k.bfloat16()
                state["R_bf16"] = R_k.bfloat16()

                # Newton-Schulz on FULL M (key difference from MuonSimple)
                X = newton_schulz(M, iters=ns_iters)
                if not torch.isfinite(X).all():
                    continue

                scale = math.sqrt(float(dim)) * rms_scale
                p.add_((X * scale).to(p.dtype), alpha=-lr)
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)
