"""
VPR feature extraction script. This is the ONLY file you modify.

Goal : maximize Recall@1 for UAV-to-satellite geo-localization using DiffusionSat.
Budget: TIME_BUDGET seconds (12 min) wall-clock per experiment.
Usage : uv run train.py > run.log 2>&1

The fixed evaluation is in prepare.py — do not modify it.
"""

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusionsat import load_sat_unet, DIFFUSIONSAT_CKPT
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
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------------------------------------------------------
# Config (edit these freely — this is the only file you modify)
# ---------------------------------------------------------------------------

BATCH_SIZE = 16
IMG_SIZE   = 256
DDIM_STEPS = 10
COLLECT    = {7, 8}   # prior CSV best: steps 7+8 of 10 (t≈801, 881)
PROMPT     = "A satellite image"
DEVICE     = "cuda"
DTYPE      = torch.float16

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

t_start = time.time()
gc.disable()

print("Loading DiffusionSat SatUNet, VAE, CLIP text encoder...")
unet         = load_sat_unet(device=DEVICE, dtype=DTYPE)
vae          = AutoencoderKL.from_pretrained(DIFFUSIONSAT_CKPT, subfolder="vae",          torch_dtype=DTYPE).to(DEVICE)
tokenizer    = CLIPTokenizer.from_pretrained(DIFFUSIONSAT_CKPT, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(DIFFUSIONSAT_CKPT, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)
scheduler    = DDIMScheduler.from_pretrained(DIFFUSIONSAT_CKPT, subfolder="scheduler")
scheduler.set_timesteps(DDIM_STEPS)
_inv_timesteps = list(reversed(scheduler.timesteps.tolist()))
_alphas        = scheduler.alphas_cumprod
print(f"DDIM inversion timesteps (clean→noisy): {_inv_timesteps}")

unet.eval().requires_grad_(False)
vae.eval().requires_grad_(False)
text_encoder.eval().requires_grad_(False)

try:
    unet.enable_xformers_memory_efficient_attention()
    print("xformers memory-efficient attention enabled.")
except Exception:
    pass

text_inputs = tokenizer(
    PROMPT, return_tensors="pt", padding="max_length",
    max_length=tokenizer.model_max_length, truncation=True,
)
with torch.inference_mode():
    prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]

# ---------------------------------------------------------------------------
# Feature hooks — down_blocks attention outputs
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
# Image transforms
# ---------------------------------------------------------------------------

_img_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_features(dataloader: DataLoader) -> np.ndarray:
    """DDIM inversion: collect features at high-noise steps."""
    all_embs = []
    pe_cache = None
    for batch in dataloader:
        imgs = batch[0].to(DEVICE, dtype=DTYPE)
        B = imgs.shape[0]

        z = vae.encode(imgs).latent_dist.mode() * 0.18215
        if pe_cache is None or pe_cache.shape[0] != B:
            pe_cache = prompt_embeds.expand(B, -1, -1)

        collected = []
        for step_idx, t_curr in enumerate(_inv_timesteps):
            t_tensor = torch.tensor([t_curr] * B, device=DEVICE, dtype=torch.long)
            _features.clear()
            noise_pred = unet(z, t_tensor, encoder_hidden_states=pe_cache).sample

            if step_idx in COLLECT:
                # avg pool each feature map, concat across blocks
                vecs = [F.avg_pool2d(_features[k], _features[k].shape[-2:]).flatten(1)
                        for k in sorted(_features)]
                collected.append(torch.cat(vecs, dim=1).float())

            if step_idx < len(_inv_timesteps) - 1:
                t_next  = _inv_timesteps[step_idx + 1]
                a_t     = _alphas[t_curr].to(z.dtype)
                a_next  = _alphas[t_next].to(z.dtype)
                x0_pred = (z - (1 - a_t).sqrt() * noise_pred) / a_t.sqrt()
                z       = a_next.sqrt() * x0_pred + (1 - a_next).sqrt() * noise_pred

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
# Extract and evaluate
# ---------------------------------------------------------------------------

print("Extracting UAV embeddings...")
uav_embs = extract_features(uav_loader)

print("Extracting satellite embeddings...")
sat_embs = extract_features(sat_loader)

uav_coords = np.array([
    (float(uav_ds.records.iloc[i]["lat"]), float(uav_ds.records.iloc[i]["lon"]))
    for i in range(len(uav_ds))
])
chunk_bboxes = sat_ds.chunk_bboxes

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
