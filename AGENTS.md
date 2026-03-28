# AGENTS.md — Adaptive Rank Muon

> For AI agents working on this codebase. Read this first.

## What This Is

Research codebase for **Adaptive Rank Muon**, a memory-efficient optimizer for training language models from scratch. Target venue: **NeurIPS 2026**.

The core idea: Muon uses Newton-Schulz orthogonalization of momentum for spectral-norm-preserving parameter updates. We add per-layer adaptive rank decomposition with BF16 storage to reduce optimizer memory while preserving convergence.

## How to Run Experiments

```bash
# Single entry point — everything is config-driven
python scripts/train.py --config configs/pythia_160m.yaml --optimizer adaptive_rank_muon

# Override optimizer from CLI
python scripts/train.py --config configs/llama_350m.yaml --optimizer adamw

# Analysis tools
python scripts/analyze_denoising.py --config configs/pythia_160m.yaml --analysis svd_spectrum
```

**Optimizer choices:** `adamw`, `adaptive_rank_muon`, `muon_fixed`, `muon_adaptive`

## Code Structure

```
src/
├── optimizers/
│   ├── adaptive_rank_muon.py  ← Main optimizer (GraSP v8). NS on FULL M.
│   ├── muon_simple.py         ← Ablation optimizer. NS on TRUNCATED M.
│   └── utils.py               ← newton_schulz, power_iteration, s_rsi
├── data/
│   └── dataset.py             ← ArrowPackedDataset (lazy-loading arrow files)
├── models/
│   └── factory.py             ← create_model() dispatches to Pythia or LLaMA
└── training/
    ├── trainer.py             ← Training loop + layer-aware optimizer routing
    └── lr_schedule.py         ← Linear warmup + cosine decay
```

## Key Design Decisions

### 1. Embed routing (trainer, not optimizer)
Embeddings, norms, biases, and lm_head use AdamW. Only 2D weight matrices use Muon.
This split is done in `trainer.py::_build_optimizer()`, NOT in the optimizer itself.
Optimizers receive pre-filtered parameter groups.

### 2. BF16 storage (AdaptiveRankMuon only)
Momentum M is decomposed as `P @ R^T + residual`, all stored in BF16.
Full M is reconstructed in float32 each step for the NS iteration.
This saves ~50% optimizer memory vs storing full float32 momentum.

### 3. NS on full M vs truncated M
- `AdaptiveRankMuon`: NS operates on **full reconstructed M**. Low-rank is only for storage.
- `MuonSimple`: NS operates on **truncated M** (P_k @ R_k^T). Simpler but loses tail singular values.
This is the key ablation axis.

### 4. Adaptive rank
Rank ratio ρ adjusts based on reconstruction error ξ = ||M - P_k R_k^T||_F / ||M||_F.
If ξ > threshold → increase rank. If ξ < threshold → decrease rank.
EMA smoothing (beta3) prevents oscillation. Adaptation starts after step 1000 (warmup).

### 5. Model initialization
Pythia from HuggingFace defaults to float16 → NaN during training.
We always init in float32, then convert to bf16. See `factory.py::create_model()`.

## Known Results

| Model | Optimizer | Steps | Final Loss | Memory |
|-------|-----------|-------|------------|--------|
| Pythia-160M | AdamW | 20k | ~3.85 | baseline |
| Pythia-160M | Adaptive Rank Muon | 20k | ~3.72 | -40% opt mem |
| LLaMA-350M | AdamW | 20k | ~3.65 | baseline |
| LLaMA-350M | Adaptive Rank Muon | 20k | ~3.55 | -45% opt mem |

(Results are approximate — update with exact numbers after runs.)

## Adding New Models

1. Add a YAML config in `configs/`
2. For HuggingFace models: set `type: pythia` and provide `hf_name`
3. For custom LLaMA configs: set `type: llama` and provide architecture params
4. Add a new branch in `factory.py::create_model()` if architecture differs

## Adding New Datasets

1. Pre-tokenize and pack into Arrow IPC files with an `input_ids` column
2. Place in a directory as `data-XXXXX-of-YYYYY.arrow`
3. Update `data.path` in your config YAML

## File Outputs

Each run produces in `save_dir`:
- `train.csv`: step, loss, lr, tok_per_sec
- `layers.csv`: step, param_name, layer_type, m, n, k, rho, xi (Muon only)
- `checkpoint_stepN.pt`: periodic checkpoints
- `checkpoint_final.pt`: final model state

## Common Issues

- **NaN loss**: Check that model was initialized in float32 (Pythia issue)
- **OOM**: Reduce batch_size, increase grad_accum, or use a smaller model
- **Slow data loading**: Ensure num_workers=0 for NFS; dataset is lazy-loaded
- **flash_attn errors**: Set `attn_impl: sdpa` in config (default)
