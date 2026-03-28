# Adaptive Rank Muon

**Memory-efficient adaptive-rank momentum for training language models from scratch.**

This repository implements Adaptive Rank Muon (GraSP v8), a Muon-family optimizer that uses per-layer adaptive rank decomposition with BF16 storage to reduce optimizer memory while maintaining or improving convergence over AdamW.

## Key Ideas

1. **Newton-Schulz orthogonalization** of momentum for spectral-norm-preserving updates
2. **Low-rank decomposition** (P, R, residual) stored in BF16 for memory efficiency
3. **Adaptive rank** per layer: rank ratio ρ adjusts based on reconstruction error ξ
4. **NS on full M** (not truncated): the low-rank structure is for storage, not for the update

## Quick Start

```bash
pip install -r requirements.txt

# Train Pythia-160M with Adaptive Rank Muon
python scripts/train.py --config configs/pythia_160m.yaml --optimizer adaptive_rank_muon

# Train LLaMA-350M with AdamW baseline
python scripts/train.py --config configs/llama_350m.yaml --optimizer adamw --device cuda:0

# Override optimizer from any config
python scripts/train.py --config configs/llama_1b.yaml --optimizer muon_fixed
```

## Optimizer Choices

| Name | Description |
|------|-------------|
| `adamw` | Standard AdamW baseline |
| `adaptive_rank_muon` | **Main method.** GraSP v8: BF16 low-rank storage, adaptive rank, NS on full M |
| `muon_fixed` | Fixed-rank Muon: NS on truncated M, no rank adaptation |
| `muon_adaptive` | Adaptive-rank Muon: NS on truncated M, with rank adaptation |

## Project Structure

```
configs/          YAML configs for each model scale
src/
  optimizers/     AdaptiveRankMuon, MuonSimple, newton_schulz, power_iteration, s_rsi
  data/           Arrow-based packed dataset loader
  models/         Model factory (Pythia via HF, LLaMA from config)
  training/       Training loop, LR schedule, CSV logging
scripts/
  train.py        Single entry point
  analyze_denoising.py  SVD spectrum and gradient SNR analysis
```

## Configs

Each YAML config specifies model architecture, training hyperparameters, optimizer settings, and data paths. The optimizer can be overridden via `--optimizer` CLI flag.

Available configs: `pythia_70m`, `pythia_160m`, `llama_350m`, `llama_1b`

## Dataset

Expects pre-tokenized, packed Arrow files (seq_len=2048) in a directory:
```
data_path/
  data-00000-of-00004.arrow
  data-00001-of-00004.arrow
  ...
```

We use OpenWebText tokenized with the Pythia tokenizer, packed to 2048 tokens per example.

## Known Issues

- **Pythia defaults to float16** in HuggingFace → NaN. We force float32 init before bf16 conversion.
- **num_workers=0** for NFS-mounted data (temp file cleanup issues on full disks).
- **flash_attn** may not be installed; defaults to `sdpa` (PyTorch native).
- **LLaMA-350M** may not fit on 2080 Ti (11GB). Use `batch_size=2, grad_accum=4`.

## Citation

Coming soon (NeurIPS 2026 submission).
