"""Linear warmup + cosine decay learning rate schedule."""

import math


def get_lr(step: int, max_steps: int, lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    """
    Compute learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step (1-indexed).
        max_steps: Total training steps.
        lr: Peak learning rate.
        warmup_steps: Number of warmup steps.
        min_lr_ratio: Minimum LR as fraction of peak (default 10%).

    Returns:
        Current learning rate.
    """
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
