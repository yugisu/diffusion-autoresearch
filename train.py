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
from sklearn.decomposition import PCA
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

BATCH_SIZE = 8
IMG_SIZE   = 512
DDIM_STEPS = 10
# Original framework: save_timesteps=[8,7] means cur_t = num_steps-1-i,
# so cur_t=8→i=1 (t=1), cur_t=7→i=2 (t=101). Our step_idx maps directly:
# step_idx=0 → t=1, step_idx=1 → t=101.  LOW-noise, not high.
COLLECT    = {0, 1, 2}   # t=1, t=101, t=201 — add third low-noise step
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
        out = out.detach().float()
        # attn1 outputs (B, H*W, C) — reshape to (B, C, H, W)
        if out.dim() == 3:
            B, L, C = out.shape
            H = W = int(L ** 0.5)
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        _features[name] = out
    return hook


# Hook attn1 (self-attention) inside each Transformer2DModel in down_blocks.
# Matches original: layer_idxs={'down_blocks': {'attn1': 'all'}}
_hooks = []
for _i, _block in enumerate(unet.down_blocks):
    if hasattr(_block, "attentions"):
        for _j, _transformer in enumerate(_block.attentions):
            for _k, _tblock in enumerate(_transformer.transformer_blocks):
                _hooks.append(_tblock.attn1.register_forward_hook(_make_hook(f"d{_i}_{_j}_{_k}")))

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

def gem_pool(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """GeM pooling: (B, C, H, W) → (B, C). Matches original PoolConcatEmbedder."""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), x.shape[-2:]).pow(1.0 / p).flatten(1)


@torch.inference_mode()
def extract_features(dataloader: DataLoader) -> np.ndarray:
    """DDIM inversion: collect attn1 features at low-noise steps (t=1, t=101)."""
    all_embs = []
    pe_cache = None
    max_step = max(COLLECT)
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
                vecs = [gem_pool(_features[k]) for k in sorted(_features)]
                collected.append(torch.cat(vecs, dim=1).float())

            if step_idx >= max_step:
                break  # no need to invert further

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

sat_loader = DataLoader(sat_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def _uav_loader(hflip: bool = False) -> DataLoader:
    ops = [transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE)]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    ds = UAVDataset(VISLOC_ROOT, FLIGHT_ID, transform=transforms.Compose(ops))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# ---------------------------------------------------------------------------
# Extract and evaluate
# ---------------------------------------------------------------------------

print("Extracting UAV embeddings (original)...")
uav_embs = extract_features(_uav_loader(hflip=False))
print("Extracting UAV embeddings (h-flip TTA)...")
uav_embs += extract_features(_uav_loader(hflip=True))

print("Extracting satellite embeddings...")
sat_embs = extract_features(sat_loader)

uav_coords = np.array([
    (float(uav_ds.records.iloc[i]["lat"]), float(uav_ds.records.iloc[i]["lon"]))
    for i in range(len(uav_ds))
])
chunk_bboxes = sat_ds.chunk_bboxes

PCA_REMOVE = 16
PCA_KEEP   = 1024
print(f"Applying PCA: remove top {PCA_REMOVE}, keep {PCA_KEEP} dims (whiten=False)...")
_all = np.concatenate([uav_embs, sat_embs], axis=0)
pca = PCA(n_components=PCA_REMOVE + PCA_KEEP, whiten=False)
pca.fit(_all)
uav_embs = pca.transform(uav_embs)[:, PCA_REMOVE:]
sat_embs = pca.transform(sat_embs)[:, PCA_REMOVE:]

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
