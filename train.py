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
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from sklearn.decomposition import PCA
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
    build_ground_truth,
    evaluate_r1,
)

# ---------------------------------------------------------------------------
# Config (edit these freely — this is the only file you modify)
# ---------------------------------------------------------------------------

SD21 = "sd2-community/stable-diffusion-2-1"
BATCH_SIZE = 8      # reduced for 512×512 (4× more pixels than 256×256)
IMG_SIZE = 512      # higher res; Resize(512)+CenterCrop(512) for non-square UAV images
DDIM_STEPS = 5      # DDIM inversion steps (5 passes per image, ~600s total)
COLLECT = {2, 3, 4} # step indices to collect features from (high-noise end)
PROMPT = ""         # null text: purely visual UNet features
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
scheduler = DDIMScheduler.from_pretrained(SD21, subfolder="scheduler")
scheduler.set_timesteps(DDIM_STEPS)
# scheduler.timesteps (generation order): [801, 601, 401, 201, 1]
# Inversion order (clean → noisy): [1, 201, 401, 601, 801]
_inv_timesteps = list(reversed(scheduler.timesteps.tolist()))
_alphas = scheduler.alphas_cumprod  # (1000,) on CPU
print(f"DDIM inversion timesteps (clean→noisy): {_inv_timesteps}")

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
    transforms.Resize(IMG_SIZE),       # scale short edge to IMG_SIZE (preserve aspect)
    transforms.CenterCrop(IMG_SIZE),   # square crop; UAV images are 3976×2652
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1] for VAE
])

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def pool_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Spatial pyramid: global avg+max + 2×2 grid avg+max → (B, 10C)."""
    B, C, H, W = x.shape
    parts = [
        F.avg_pool2d(x, (H, W)).flatten(1),   # global avg
        F.max_pool2d(x, (H, W)).flatten(1),   # global max
    ]
    h2, w2 = H // 2, W // 2
    for i in range(2):
        for j in range(2):
            tile = x[:, :, i*h2:(i+1)*h2, j*w2:(j+1)*w2]
            parts.append(F.avg_pool2d(tile, tile.shape[-2:]).flatten(1))
            parts.append(F.max_pool2d(tile, tile.shape[-2:]).flatten(1))
    return torch.cat(parts, dim=1)             # (B, 10C)


@torch.inference_mode()
def extract_features(dataloader: DataLoader) -> np.ndarray:
    """DDIM inversion: map each image to noise, collect features at high-noise steps."""
    all_embs = []
    pe_cache = None
    for batch in dataloader:
        imgs = batch[0].to(DEVICE, dtype=DTYPE)
        B = imgs.shape[0]

        z = vae.encode(imgs).latent_dist.sample() * 0.18215
        if pe_cache is None or pe_cache.shape[0] != B:
            pe_cache = prompt_embeds.expand(B, -1, -1)

        collected = []
        for step_idx, t_curr in enumerate(_inv_timesteps):
            t_tensor = torch.tensor([t_curr] * B, device=DEVICE, dtype=torch.long)
            _features.clear()
            noise_pred = unet(z, t_tensor, encoder_hidden_states=pe_cache).sample

            if step_idx in COLLECT:
                vecs = [pool_features(_features[k]) for k in sorted(_features)]
                collected.append(torch.cat(vecs, dim=1).float())

            # DDIM inversion step: z_t → z_{t_next}
            if step_idx < len(_inv_timesteps) - 1:
                t_next = _inv_timesteps[step_idx + 1]
                a_t    = _alphas[t_curr].to(z.dtype)
                a_next = _alphas[t_next].to(z.dtype)
                # predicted x0 from current z and noise estimate
                x0_pred = (z - (1 - a_t).sqrt() * noise_pred) / a_t.sqrt()
                # inversion step: move z to t_next noise level
                z = a_next.sqrt() * x0_pred + (1 - a_next).sqrt() * noise_pred

        emb = F.normalize(torch.cat(collected, dim=1), dim=1)
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
# PCA whitening — fit on combined UAV+sat, remove first N dominant components
# (first components capture sensor/quality domain differences, not location)
# ---------------------------------------------------------------------------

PCA_REMOVE = 16   # number of leading PCA components to discard
PCA_KEEP   = 1024 # total output dims after whitening (testing: 512 may be too aggressive)

print(f"Applying PCA whitening: remove top {PCA_REMOVE}, keep {PCA_KEEP} dims...")
all_embs = np.concatenate([uav_embs, sat_embs], axis=0)
pca = PCA(n_components=PCA_REMOVE + PCA_KEEP, whiten=False)
pca.fit(all_embs)
# drop the first PCA_REMOVE components, keep the next PCA_KEEP
uav_embs = pca.transform(uav_embs)[:, PCA_REMOVE:]
sat_embs = pca.transform(sat_embs)[:, PCA_REMOVE:]

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

print("Evaluating...")
metrics = evaluate_r1(uav_embs, sat_embs, uav_coords, chunk_bboxes)

gt = build_ground_truth(uav_coords, chunk_bboxes)
avg_gt_chunks = sum(len(g) for g in gt) / len(gt)

elapsed = time.time() - t_start
print("---")
print(f"R@1:           {metrics['R@1']:.6f}")
print(f"R@5:           {metrics['R@5']:.6f}")
print(f"R@10:          {metrics['R@10']:.6f}")
print(f"elapsed_s:     {elapsed:.1f}")
print(f"emb_dim:       {uav_embs.shape[1]}")
print(f"avg_gt_chunks: {avg_gt_chunks:.1f}")
