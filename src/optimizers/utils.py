"""
Core numerical routines for Muon-family optimizers.

- newton_schulz: Cubic-convergence Newton-Schulz iteration for polar decomposition
- power_iteration: Warm-started rank-k subspace approximation
- s_rsi: Cold-start randomized subspace iteration (Halko et al.)
"""

import torch


@torch.no_grad()
def newton_schulz(M: torch.Tensor, iters: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute the orthogonal factor of the polar decomposition M = U @ S @ V^T
    via Newton-Schulz iteration. Returns U @ V^T (the closest orthogonal matrix).

    Uses the cubic-convergence variant with coefficients (a, b, c) from
    Björck & Bowie (1971). Operates in float32 for numerical stability.

    Args:
        M: Input matrix (m x n).
        iters: Number of NS iterations (5 is typically sufficient).
        eps: Small constant to avoid division by zero.

    Returns:
        Orthogonal approximation of M, same shape as M.
    """
    M = M.float()
    X = M / (M.norm() + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(iters):
        XTX = X.T @ X
        X = a * X + b * (X @ XTX) + c * (X @ (XTX @ XTX))
        if not torch.isfinite(X).all():
            X = M / (M.norm() + eps)
            break
    return X


@torch.no_grad()
def power_iteration(M: torch.Tensor, Q: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Warm-started rank-k subspace approximation via one step of power iteration.

    Given a warm-start basis Q (from previous step), compute a single QR-based
    refinement of the top-k left/right singular subspace.

    Args:
        M: Input matrix (m x n), float32.
        Q: Warm-start right basis (n x k).
        k: Target rank (used implicitly via Q's shape).

    Returns:
        P: Left basis (m x k), orthonormal columns.
        R: Right factor (n x k), such that M ≈ P @ R^T.
    """
    P = M @ Q
    P, _ = torch.linalg.qr(P)
    R = M.T @ P
    return P, R


@torch.no_grad()
def s_rsi(A: torch.Tensor, k: int, l: int = 2, p: int = 5, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomized Subspace Iteration (s-RSI) for cold-start rank-k approximation.

    When no warm-start basis is available, this provides a good initial
    subspace estimate using Halko et al.'s randomized algorithm.

    Args:
        A: Input matrix (m x n).
        k: Target rank.
        l: Number of power iterations (more = better accuracy, slower).
        p: Oversampling parameter (k + p columns in random sketch).
        eps: Stability constant.

    Returns:
        Q: Left basis (m x k_actual), orthonormal columns.
        U: Right factor (n x k_actual).
    """
    A = A.float()
    m, n = A.shape
    k_actual = min(k, m, n)
    U = torch.randn(n, k_actual + p, device=A.device, dtype=torch.float32)
    U /= (U.norm(dim=0, keepdim=True) + eps)
    for _ in range(l):
        Q = A @ U
        Q, _ = torch.linalg.qr(Q)
        U = A.T @ Q
    return Q[:, :k_actual], U[:, :k_actual]
