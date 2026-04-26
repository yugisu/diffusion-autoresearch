from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_SSL_CKPT = "checkpoints/dinov3-ssl4eos12-best-r@1=0.615-mvicreg-569ef72.ckpt"
DEFAULT_SSL_LORA_RANK = 16
DEFAULT_SSL_LORA_ALPHA = 32.0
DEFAULT_SSL_LORA_LAST_N_BLOCKS = 4


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear layers."""

    def __init__(self, orig: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.orig = orig
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for p in self.orig.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, orig.in_features))
        self.lora_B = nn.Parameter(torch.zeros(orig.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.orig(x)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base + lora


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0, last_n_blocks: int = 0) -> nn.Module:
    """Apply LoRA to attention projection layers in the ViT backbone."""
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and any(k in name for k in ("query", "key", "value", "qkv", "q_proj", "k_proj", "v_proj"))):
            continue

        if last_n_blocks > 0:
            parts = name.split(".")
            block_indices = [int(p) for p in parts if p.isdigit()]
            if not block_indices:
                continue
            block_idx = block_indices[0]
            total_blocks = sum(1 for n, _ in model.named_modules() if n.endswith(".layernorm_before") or n.endswith(".layer_norm1"))
            if total_blocks == 0:
                total_blocks = 12
            if block_idx < total_blocks - last_n_blocks:
                continue

        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)
        setattr(parent, attr_name, LoRALinear(getattr(parent, attr_name), rank, alpha))
    return model


def merge_lora_backbone(backbone: nn.Module) -> nn.Module:
    """Merge LoRA deltas into base weights and unwrap LoRALinear back to nn.Linear."""
    replacements: dict[str, nn.Linear] = {}
    for name, module in backbone.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        with torch.no_grad():
            delta = (module.lora_B @ module.lora_A) * module.scaling
            module.orig.weight.data += delta
        module.orig.weight.requires_grad = True
        replacements[name] = module.orig

    for name, merged_linear in replacements.items():
        parts = name.split(".")
        parent = backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], merged_linear)

    return backbone


def load_base_backbone(model_name: str) -> nn.Module:
    """Load plain pretrained DINO backbone and unfreeze all params."""
    backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    for p in backbone.parameters():
        p.requires_grad = True
    print("Base backbone loaded from pretrained weights.")
    return backbone


def load_ssl_backbone(
    ckpt_path: str,
    model_name: str,
    lora_rank: int = DEFAULT_SSL_LORA_RANK,
    lora_alpha: float = DEFAULT_SSL_LORA_ALPHA,
    lora_last_n_blocks: int = DEFAULT_SSL_LORA_LAST_N_BLOCKS,
) -> nn.Module:
    """Load SSL checkpoint, merge LoRA, and return fully-unfrozen backbone."""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {ckpt_path}")

    backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    for p in backbone.parameters():
        p.requires_grad = False

    backbone = apply_lora(
        backbone,
        rank=lora_rank,
        alpha=lora_alpha,
        last_n_blocks=lora_last_n_blocks,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    backbone_sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=True)
    if missing:
        print(f"WARNING: missing keys in SSL checkpoint: {missing}")
    if unexpected:
        print(f"WARNING: unexpected keys in SSL checkpoint: {unexpected}")

    backbone = merge_lora_backbone(backbone)
    for p in backbone.parameters():
        p.requires_grad = True

    print(f"SSL backbone loaded and LoRA merged: {ckpt_path}")
    print(f"  SSL epoch: {ckpt.get('epoch', '?')} | SSL step: {ckpt.get('global_step', '?')}")
    return backbone
