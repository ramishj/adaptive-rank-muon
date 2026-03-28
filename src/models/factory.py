"""
Model factory for creating language models from YAML config.

Supports:
  - pythia: EleutherAI Pythia models via AutoModelForCausalLM (from config, not pretrained)
  - llama: LLaMA-style models via LlamaForCausalLM (from config)

Critical notes:
  - Pythia defaults to float16 in HuggingFace → causes NaN. Must force float32 before
    converting to bf16 training dtype.
  - flash_attn may not be installed; default to "sdpa" (PyTorch native scaled dot product).
  - LLaMA-350M may not fit on 2080 Ti (11GB) with bf16 + gradient checkpointing.
    Use gradient accumulation to reduce batch size.
"""

import torch
import torch.nn as nn


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create a causal LM from a config dict.

    Args:
        config: Full experiment config (with 'model' and 'training' keys).
        device: Target device.

    Returns:
        Model on device, in training mode, with gradient checkpointing enabled.
    """
    model_cfg = config["model"]
    training_cfg = config["training"]
    model_type = model_cfg["type"]
    use_bf16 = training_cfg.get("bf16", True)
    attn_impl = training_cfg.get("attn_impl", "sdpa")

    if model_type == "pythia":
        model = _create_pythia(model_cfg, attn_impl)
    elif model_type == "llama":
        model = _create_llama(model_cfg, attn_impl)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # NOTE: Always init in float32 first to avoid NaN (especially Pythia),
    # then convert to bf16 if requested.
    model = model.float()
    if use_bf16:
        model = model.bfloat16()

    model = model.to(device).train()
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_cfg['name']} | {total_params:,} params ({total_params/1e6:.0f}M)")
    print(f"  dtype={'bf16' if use_bf16 else 'fp32'} | attn={attn_impl} | grad_ckpt=True")

    return model


def _create_pythia(model_cfg: dict, attn_impl: str) -> nn.Module:
    """Create a Pythia model from scratch (not pretrained)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    # Use the HuggingFace model name to get the architecture config
    hf_name = model_cfg.get("hf_name", f"EleutherAI/pythia-{model_cfg['name'].lower().replace('pythia-', '')}")
    hf_config = AutoConfig.from_pretrained(hf_name)

    # Override with our config values
    hf_config.use_cache = False
    if hasattr(hf_config, "torch_dtype"):
        hf_config.torch_dtype = torch.float32  # Pythia defaults to fp16 → NaN

    # Set attention implementation
    if hasattr(hf_config, "_attn_implementation"):
        hf_config._attn_implementation = attn_impl

    return AutoModelForCausalLM.from_config(hf_config)


def _create_llama(model_cfg: dict, attn_impl: str) -> nn.Module:
    """Create a LLaMA model from scratch with custom dimensions."""
    from transformers import LlamaConfig, LlamaForCausalLM

    llama_cfg = LlamaConfig(
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg.get("num_key_value_heads", model_cfg["num_attention_heads"]),
        intermediate_size=model_cfg["intermediate_size"],
        max_position_embeddings=model_cfg.get("max_position_embeddings", 2048),
        vocab_size=model_cfg.get("vocab_size", 50304),
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        use_cache=False,
        tie_word_embeddings=False,
        hidden_act="silu",
        _attn_implementation=attn_impl,
    )

    return LlamaForCausalLM(llama_cfg)
