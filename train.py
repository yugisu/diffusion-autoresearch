"""
VPR feature extraction script. This is the ONLY file you modify.

Goal : maximize Recall@1 for UAV-to-satellite geo-localization using SD v2.1.
Budget: TIME_BUDGET seconds (12 min) wall-clock per experiment.
Usage : uv run train.py > run.log 2>&1

The fixed evaluation is in prepare.py — do not modify it.
"""

import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from prepare import (
    CHUNK_PIXELS,
    CHUNK_STRIDE,
    FLIGHT_ID,
    MAP_SCALE_FACTOR,
    TIME_BUDGET,
    VISLOC_ROOT,
    SatChunkDataset,
    UAVDataset,
    evaluate_r1,
)

# ---------------------------------------------------------------------------
# Config (edit these freely — this is the only file you modify)
# ---------------------------------------------------------------------------

SD21 = "sd2-community/stable-diffusion-2-1"
BATCH_SIZE = 16     # images per UNet forward pass; reduce if OOM
IMG_SIZE = 256      # resize images to IMG_SIZE × IMG_SIZE before VAE encode
TIMESTEP = 880      # DDPM timestep 0–999. Prior sweep: ~800–950 is best for down_blocks.
PROMPT = "a satellite image"
DEVICE = "cuda"
DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

t_start = time.time()
gc.disable()

# ---- Load SD21 components ----
print("Loading SD21 UNet, VAE, CLIP text encoder...")
unet = UNet2DConditionModel.from_pretrained(SD21, subfolder="unet", torch_dtype=DTYPE).to(DEVICE)
vae = AutoencoderKL.from_pretrained(SD21, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(SD21, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(SD21, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)
scheduler = DDPMScheduler.from_pretrained(SD21, subfolder="scheduler")

unet.eval().requires_grad_(False)
vae.eval().requires_grad_(False)
text_encoder.eval().requires_grad_(False)

try:
    unet.enable_xformers_memory_efficient_attention()
    print("xformers memory-efficient attention enabled.")
except Exception:
    pass

# ---- Encode text prompt once ----
text_inputs = tokenizer(
    PROMPT, return_tensors="pt", padding="max_length",
    max_length=tokenizer.model_max_length, truncation=True,
)
with torch.inference_mode():
    prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]  # (1, 77, 1024)

# ---------------------------------------------------------------------------
# Feature hooks
#
# We hook every Transformer2DModel block in down_blocks (the cross-attention
# down-path of the UNet). Each block outputs a spatial feature map (B, C, H, W).
#
# For IMG_SIZE=256, latent size is 32×32. Feature spatial dims per block:
#   down_blocks[0].attentions[0,1] : (B, 320, 32, 32)
#   down_blocks[1].attentions[0,1] : (B, 640, 16, 16)
#   down_blocks[2].attentions[0,1] : (B, 1280, 8, 8)
# Total after GeM pool + concat: 2*(320+640+1280) = 4480 dims.
# ---------------------------------------------------------------------------

_features: dict[str, torch.Tensor] = {}


def _make_hook(name: str):
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if hasattr(out, "sample"):
            out = out.sample
        _features[name] = out.detach().float()
    return hook


_hooks = []
for _i, _block in enumerate(unet.down_blocks):
    if hasattr(_block, "attentions"):
        for _j, _attn in enumerate(_block.attentions):
            _hooks.append(_attn.register_forward_hook(_make_hook(f"d{_i}_{_j}")))

# ---------------------------------------------------------------------------
# Image transforms (UAV images: 3976×2652 px; sat chunks: CHUNK_PIXELS × CHUNK_PIXELS)
# ---------------------------------------------------------------------------

_img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1] for VAE
])

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def gem_pool(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalised Mean Pooling: (B, C, H, W) → (B, C)."""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), x.shape[-2:]).pow(1.0 / p).flatten(1)


@torch.inference_mode()
def extract_features(dataloader: DataLoader) -> np.ndarray:
    """Run the extraction pipeline over a dataloader, return (N, D) float32 array."""
    all_embs = []
    for batch in dataloader:
        imgs = batch[0].to(DEVICE, dtype=DTYPE)
        B = imgs.shape[0]

        # VAE encode: PIL image → latent
        latents = vae.encode(imgs).latent_dist.sample() * 0.18215

        # Add noise at the fixed timestep
        t = torch.tensor([TIMESTEP] * B, device=DEVICE, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = scheduler.add_noise(latents, noise, t)

        # UNet forward pass — we only care about the hooked intermediate features
        _features.clear()
        pe = prompt_embeds.expand(B, -1, -1)
        unet(noisy, t, encoder_hidden_states=pe)

        # GeM-pool each spatial feature map, concatenate, L2-normalise
        vecs = [gem_pool(_features[k]) for k in sorted(_features)]
        emb = F.normalize(torch.cat(vecs, dim=1).float(), dim=1)
        all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

uav_ds = UAVDataset(VISLOC_ROOT, FLIGHT_ID, transform=_img_transform)
sat_ds = SatChunkDataset(
    VISLOC_ROOT, FLIGHT_ID,
    chunk_pixels=CHUNK_PIXELS, stride_pixels=CHUNK_STRIDE,
    scale_factor=MAP_SCALE_FACTOR, transform=_img_transform,
)
print(f"UAV queries: {len(uav_ds)} | Satellite gallery: {len(sat_ds)} chunks")

uav_loader = DataLoader(uav_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
sat_loader = DataLoader(sat_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------------------------------------------------------------------------
# Extract embeddings
# ---------------------------------------------------------------------------

print("Extracting UAV embeddings...")
uav_embs = extract_features(uav_loader)

print("Extracting satellite embeddings...")
sat_embs = extract_features(sat_loader)

# Geo-metadata for evaluation
uav_coords = np.array([
    (float(uav_ds.records.iloc[i]["lat"]), float(uav_ds.records.iloc[i]["lon"]))
    for i in range(len(uav_ds))
])
chunk_bboxes = sat_ds.chunk_bboxes

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

print("Evaluating...")
metrics = evaluate_r1(uav_embs, sat_embs, uav_coords, chunk_bboxes)

elapsed = time.time() - t_start
print("---")
print(f"R@1:       {metrics['R@1']:.6f}")
print(f"R@5:       {metrics['R@5']:.6f}")
print(f"R@10:      {metrics['R@10']:.6f}")
print(f"elapsed_s: {elapsed:.1f}")
print(f"emb_dim:   {uav_embs.shape[1]}")
