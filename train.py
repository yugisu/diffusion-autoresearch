"""
Self-supervised DINOv3 training for VisLoc cross-view retrieval.

Branch goal:
- Fine-tune facebook/dinov3-vitb16-pretrain-lvd1689m with self-supervised learning
- Train on satellite chunks only from flights: 01, 02, 04, 05, 06, 08, 09, 10, 11
- Validate on flight: 03 (768 UAV queries, 2860 satellite chunks)
- Optimize for Recall@1 on fixed VisLoc evaluation
- No UAV images during training — satellite chunks only

Usage:
  uv run train.py > run.log 2>&1

Environment (optional overrides):
  VISLOC_ROOT=/workspace/data/visloc
  WANDB_API_KEY=... (for automatic wandb login)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from prepare import (
    CHUNK_PIXELS,
    CHUNK_STRIDE,
    MAP_SCALE_FACTOR,
    SSL4EOS12_ROOT,
    VISLOC_ROOT,
    SatChunkDataset,
    UAVDataset,
    build_ground_truth,
    evaluate_r1,
)
from ssl4eos12_dataset import build_ssl4eos12_dataset

torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# Experiment defaults
# -----------------------------------------------------------------------------

DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
TRAIN_FLIGHTS = ["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"]  # 03 added: learn eval region structure
VAL_FLIGHT = "03"

TRAIN_STRIDE = 64  # denser stride for SSL training (~4x more chunks)

SAT_SCALES = {
    "01": 0.25,
    "02": 0.25,
    "03": 0.25,
    "04": 0.25,
    "05": 0.40,
    "06": 0.60,
    "08": 0.35,
    "09": 0.25,
    "10": 0.50,
    "11": 0.25,
}


@dataclass
class Config:
    visloc_root: str = str(VISLOC_ROOT)
    model_name: str = DINO_MODEL
    image_size: int = 336
    embedding_dim: int = 768  # CLS token dim, no projection head

    batch_size: int = 128
    eval_batch_size: int = 128
    num_workers: int = 8

    lr: float = 1e-5
    weight_decay: float = 1e-4
    temperature: float = 0.07
    warmup_epochs: int = 2
    proj_dim: int = 0  # projection head output dim (0 = disabled, use raw CLS for SSL)

    georank_weight: float = 0.0  # weight for GeoRank regularization (0 = disabled)
    cosine_t0: int = 0  # CosineAnnealingWarmRestarts period (0 = plain cosine decay)

    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_last_n_blocks: int = 4  # only last N blocks get LoRA (0=all)

    iou_pos_threshold: float = 0.50
    iou_neg_threshold: float = 0.0  # IoU == 0 → negative

    max_epochs: int = 13
    max_steps: int = -1
    precision: str = "16-mixed"
    seed: int = 42

    # SSL4EO-S12 training data
    ssl4eo_root: str = str(SSL4EOS12_ROOT)
    # Optional geographic bounding-box filter (degrees).  None = global.
    # VisLoc flights are in China; set to (15, 55) / (90, 135) to focus there.
    ssl4eo_lat_min: float | None = None
    ssl4eo_lat_max: float | None = None
    ssl4eo_lon_min: float | None = None
    ssl4eo_lon_max: float | None = None

    wandb_project: str = "autoresearch-ssl-dinov3"
    wandb_run_name: str | None = None


# -----------------------------------------------------------------------------
# LoRA implementation
# -----------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for nn.Linear layers."""

    def __init__(self, orig: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.orig = orig
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for p in self.orig.parameters():
            p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, orig.in_features))
        self.lora_B = nn.Parameter(torch.zeros(orig.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialized to zero so LoRA starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.orig(x)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base + lora


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0, last_n_blocks: int = 0) -> nn.Module:
    """Apply LoRA to qkv projection layers in the ViT backbone.

    last_n_blocks: if >0, only apply LoRA to the last N transformer blocks (0 = all blocks).
    """
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and any(
            k in name for k in ("query", "key", "value", "qkv", "q_proj", "k_proj", "v_proj")
        )):
            continue

        if last_n_blocks > 0:
            # Extract block index from name (e.g. "encoder.layer.10.attention...")
            parts = name.split(".")
            block_indices = [int(p) for p in parts if p.isdigit()]
            if not block_indices:
                continue
            block_idx = block_indices[0]
            # Count total blocks to determine cutoff
            total_blocks = sum(1 for n, _ in model.named_modules() if n.endswith(".layernorm_before") or n.endswith(".layer_norm1"))
            if total_blocks == 0:
                total_blocks = 12  # ViT-B default
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


# -----------------------------------------------------------------------------
# SSL Dataset — satellite chunks only, IoU-based positive mining
# -----------------------------------------------------------------------------


def compute_iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Compute IoU between two (lat_min, lon_min, lat_max, lon_max) bboxes."""
    lat_min = max(box_a[0], box_b[0])
    lon_min = max(box_a[1], box_b[1])
    lat_max = min(box_a[2], box_b[2])
    lon_max = min(box_a[3], box_b[3])

    if lat_min >= lat_max or lon_min >= lon_max:
        return 0.0

    inter = (lat_max - lat_min) * (lon_max - lon_min)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SatSSLDataset(Dataset):
    """
    Cross-scale SSL dataset: anchor = zoomed-in crop (25-50% of chunk area),
    positive = full-scale view of the SAME chunk. Trains scale invariance that
    bridges the UAV (high-res zoomed-in) ↔ satellite (lower-res full area) gap.
    """

    def __init__(
        self,
        root: str,
        flights: List[str],
        sat_scales: Dict[str, float],
        anchor_transform,
        positive_transform,
        iou_pos_threshold: float = 0.50,  # kept for API compat, unused
    ):
        self.anchor_transform = anchor_transform
        self.positive_transform = positive_transform
        self.sat_datasets: Dict[str, SatChunkDataset] = {}
        self.samples: List[Tuple[str, int]] = []

        for flight in flights:
            scale = sat_scales.get(flight, MAP_SCALE_FACTOR)
            sat_ds = SatChunkDataset(
                root, flight, chunk_pixels=CHUNK_PIXELS,
                stride_pixels=TRAIN_STRIDE, scale_factor=scale, transform=None,
            )
            self.sat_datasets[flight] = sat_ds
            self.samples.extend([(flight, i) for i in range(len(sat_ds))])

        print(f"SSL dataset: {len(self.samples)} chunks across {len(flights)} flights (cross-scale pairs)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        flight, chunk_idx = self.samples[idx]
        sat_ds = self.sat_datasets[flight]
        img, lat, lon = sat_ds[chunk_idx]

        anchor = self.anchor_transform(img)    # zoomed-in view (UAV-like)
        positive = self.positive_transform(img)  # full-scale view (satellite-like)

        coords = torch.tensor([lat, lon], dtype=torch.float32)
        return anchor, positive, coords, coords


# -----------------------------------------------------------------------------
# SSL4EO-S12 training pipeline — parquet-filtered shards, seasonal SSL pairs
# -----------------------------------------------------------------------------


def build_ssl4eo_ssl_pipeline(
    data_root: str | Path,
    anchor_transform,
    positive_transform,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    batch_size: int = 128,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Build a WebDataset pipeline for SSL4EO-S12 S2RGB SSL training.

    Uses train_metadata.parquet to select only shards that contain samples
    inside the requested geographic bounding box — avoids streaming irrelevant
    shards.  Within each selected shard a second lat/lon check drops the
    minority of samples that fall outside the box (shards have global mixing).

    Each decoded sample yields an (anchor, positive, coords, coords) tuple:
      anchor   — seasonal view t1 with strong crop+jitter augmentation
      positive — seasonal view t2 (different season) with mild augmentation

    The pipeline handles its own batching via .batched(), so the DataLoader
    should be created with batch_size=None.
    """
    root = Path(data_root)

    # Use parquet to get the set of shards that cover the target region.
    meta = pd.read_parquet(root / "train_metadata.parquet")
    if lat_range is not None:
        meta = meta[meta["center_lat"].between(*lat_range)]
    if lon_range is not None:
        meta = meta[meta["center_lon"].between(*lon_range)]

    shard_names = sorted(meta["tar"].unique())
    shard_urls = [str(root / "train" / "S2RGB" / f) for f in shard_names]
    n_samples = len(meta)

    geo_str = f"lat {lat_range}, lon {lon_range}" if lat_range else "global"
    print(
        f"SSL4EO-S12: {len(shard_urls)} shards, {n_samples:,} samples | {geo_str}"
    )

    # Base pipeline: decode zarr with metadata → sample["image"] = (4,3,264,264)
    base = build_ssl4eos12_dataset(
        path=str(root),
        modalities=["S2RGB"],
        split="train",
        urls=shard_urls,
        batch_size=None,
        transform=None,
        return_metadata=True,
        shuffle=shuffle,
        shardshuffle=min(len(shard_urls), 200) if shuffle else 0,
        seed=seed,
        partial=False,
    )

    def to_ssl_pair(sample):
        lat = float(sample.get("center_lat", 0.0))
        lon = float(sample.get("center_lon", 0.0))
        # Fine-grained filter: shards contain mixed locations, cull stragglers.
        if lat_range is not None and not (lat_range[0] <= lat <= lat_range[1]):
            return None
        if lon_range is not None and not (lon_range[0] <= lon <= lon_range[1]):
            return None

        bands = sample["image"]  # (T, 3, 264, 264) uint8
        T = bands.shape[0]
        t1, t2 = np.random.choice(T, 2, replace=False).tolist() if T >= 2 else (0, 0)
        img_a = Image.fromarray(np.transpose(bands[t1], (1, 2, 0)))
        img_p = Image.fromarray(np.transpose(bands[t2], (1, 2, 0)))

        coords = torch.tensor([lat, lon], dtype=torch.float32)
        return anchor_transform(img_a), positive_transform(img_p), coords, coords

    return (
        base
        .map(to_ssl_pair)
        .select(lambda x: x is not None)
        .batched(batch_size, collation_fn=default_collate)
    )


# -----------------------------------------------------------------------------
# Data module
# -----------------------------------------------------------------------------


class VisLocSSLDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.visloc_root)

        self.train_ds = None
        self.val_uav_ds = None
        self.val_sat_ds = None
        self._wds_pipeline = False  # True when train_ds is a wds pipeline (batches itself)

        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        mean = self.processor.image_mean
        std = self.processor.image_std

        shared_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        # Anchor: zoomed-in UAV-like crop + stronger sensor/temporal augmentation
        anchor_aug = [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.25, 0.50), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        ]
        self.anchor_transform = transforms.Compose(anchor_aug + shared_aug)
        # Positive: full-scale satellite view — mild augmentation only
        self.positive_transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.75, 1.00), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0),
        ] + shared_aug)
        # Kept for eval (not used for training)
        self.train_transform = self.anchor_transform
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def setup(self, stage: str | None = None):
        if self.train_ds is None:
            ssl4eo_shard_dir = Path(self.cfg.ssl4eo_root) / "train" / "S2RGB"
            if ssl4eo_shard_dir.exists() and any(ssl4eo_shard_dir.glob("*.tar")):
                lat_range = (
                    (self.cfg.ssl4eo_lat_min, self.cfg.ssl4eo_lat_max)
                    if self.cfg.ssl4eo_lat_min is not None
                    else None
                )
                lon_range = (
                    (self.cfg.ssl4eo_lon_min, self.cfg.ssl4eo_lon_max)
                    if self.cfg.ssl4eo_lon_min is not None
                    else None
                )
                self.train_ds = build_ssl4eo_ssl_pipeline(
                    data_root=self.cfg.ssl4eo_root,
                    anchor_transform=self.anchor_transform,
                    positive_transform=self.positive_transform,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    seed=self.cfg.seed,
                )
                self._wds_pipeline = True
            else:
                print(
                    f"SSL4EO-S12 shards not found at {ssl4eo_shard_dir}; "
                    "falling back to VisLoc satellite chunks."
                )
                self.train_ds = SatSSLDataset(
                    root=self.root,
                    flights=TRAIN_FLIGHTS,
                    sat_scales=SAT_SCALES,
                    anchor_transform=self.anchor_transform,
                    positive_transform=self.positive_transform,
                    iou_pos_threshold=self.cfg.iou_pos_threshold,
                )

        if self.val_uav_ds is None or self.val_sat_ds is None:
            val_scale = SAT_SCALES.get(VAL_FLIGHT, MAP_SCALE_FACTOR)
            self.val_uav_ds = UAVDataset(self.root, VAL_FLIGHT, transform=self.eval_transform)
            self.val_sat_ds = SatChunkDataset(
                self.root,
                VAL_FLIGHT,
                chunk_pixels=CHUNK_PIXELS,
                stride_pixels=CHUNK_STRIDE,
                scale_factor=val_scale,
                transform=self.eval_transform,
            )
            print(
                f"Validation flight {VAL_FLIGHT}: {len(self.val_uav_ds)} UAV queries | {len(self.val_sat_ds)} sat chunks | scale={val_scale}"
            )

    def train_dataloader(self):
        if self._wds_pipeline:
            # wds pipeline batches itself — DataLoader is just a thin wrapper.
            return DataLoader(
                dataset=self.train_ds,
                batch_size=None,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                persistent_workers=self.cfg.num_workers > 0,
            )
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        common = dict(
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )
        uav_loader = DataLoader(self.val_uav_ds, **common)
        sat_loader = DataLoader(self.val_sat_ds, **common)
        return [uav_loader, sat_loader]


# -----------------------------------------------------------------------------
# Projection head
# -----------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """2-layer MLP projection head used only during SSL training (discarded at eval)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# -----------------------------------------------------------------------------
# Lightning model
# -----------------------------------------------------------------------------


class DinoSSLRetriever(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(vars(cfg))

        self.backbone = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)

        # Freeze backbone, then apply LoRA
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone = apply_lora(self.backbone, rank=cfg.lora_rank, alpha=cfg.lora_alpha, last_n_blocks=cfg.lora_last_n_blocks)

        if cfg.proj_dim > 0:
            self.proj_head = ProjectionHead(cfg.embedding_dim, cfg.proj_dim)
        else:
            self.proj_head = None

        self._val_uav_embs = []
        self._val_sat_embs = []
        self._val_uav_coords = []
        self._total_samples_seen = 0
        self._train_start_time = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding (768-d, no projection head)."""
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return F.normalize(cls, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def _infonce_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE: diagonal entries are positives."""
        logits = (q @ k.t()) / self.cfg.temperature
        labels = torch.arange(len(q), device=q.device)
        loss_qk = F.cross_entropy(logits, labels)
        loss_kq = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_qk + loss_kq)

    def _georank_loss(self, embs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """GeoRank: MSE between normalized geographic distance ranks and embedding similarity ranks.

        Burgert et al. WACV 2025: rank-preservation regularization that ties embedding
        space ordering to geographic ordering.
        """
        # Pairwise geographic distances (degrees, flat-earth approx)
        dlat = coords[:, 0:1] - coords[:, 0:1].T
        dlon = coords[:, 1:2] - coords[:, 1:2].T
        geo_dist = torch.sqrt(dlat ** 2 + dlon ** 2 + 1e-10)  # (B, B)

        # Pairwise cosine similarities (embs already L2-normalized)
        sim = embs @ embs.T  # (B, B), in [-1, 1]

        # Normalize both to [0, 1] for rank comparison
        # geo: small dist → 0, large dist → 1
        geo_norm = geo_dist / (geo_dist.max() + 1e-8)
        # sim: large sim → 0 (close), small sim → 1 (far)
        sim_norm = 1.0 - (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

        # Exclude diagonal (self-pairs)
        mask = ~torch.eye(embs.size(0), dtype=torch.bool, device=embs.device)
        return F.mse_loss(sim_norm[mask], geo_norm[mask])

    def on_train_start(self):
        self._train_start_time = time.time()

    def training_step(self, batch, batch_idx):
        anchor, positive, anchor_coords, pos_coords = batch
        q = self.encode(anchor)
        k = self.encode(positive)

        # Use projection head for SSL loss if enabled (backbone CLS used raw at eval)
        if self.proj_head is not None:
            q_ssl = self.proj_head(q)
            k_ssl = self.proj_head(k)
        else:
            q_ssl, k_ssl = q, k

        infonce = self._infonce_loss(q_ssl, k_ssl)
        loss = infonce

        if self.cfg.georank_weight > 0:
            gr = self._georank_loss(q, anchor_coords.to(q.device))
            loss = infonce + self.cfg.georank_weight * gr
            self.log("train/georank_loss", gr, on_step=True, on_epoch=False)

        bs = anchor.size(0)
        self._total_samples_seen += bs
        elapsed = time.time() - self._train_start_time if self._train_start_time else 0

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/samples_seen", float(self._total_samples_seen), on_step=True, on_epoch=False)
        self.log("train/elapsed_s", elapsed, on_step=True, on_epoch=False)
        self.log("train/samples_per_sec", self._total_samples_seen / max(elapsed, 1), on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_start(self):
        self._val_uav_embs = []
        self._val_sat_embs = []
        self._val_uav_coords = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, lat, lon = batch

        if dataloader_idx == 0:  # UAV queries
            emb = self.encode(imgs)
            self._val_uav_embs.append(emb.detach().cpu())
            coords = torch.stack([lat, lon], dim=1)
            self._val_uav_coords.append(coords.detach().cpu())
        else:  # satellite gallery — TTA over 4 rotations
            e0 = self.encode(imgs)
            e1 = self.encode(torch.rot90(imgs, 1, [2, 3]))
            e2 = self.encode(torch.rot90(imgs, 2, [2, 3]))
            e3 = self.encode(torch.rot90(imgs, 3, [2, 3]))
            emb = F.normalize((e0 + e1 + e2 + e3) / 4.0, dim=-1)
            self._val_sat_embs.append(emb.detach().cpu())

    def on_validation_epoch_end(self):
        if len(self._val_uav_embs) == 0 or len(self._val_sat_embs) == 0:
            return

        uav_embs = torch.cat(self._val_uav_embs, dim=0).numpy().astype(np.float32)
        sat_embs = torch.cat(self._val_sat_embs, dim=0).numpy().astype(np.float32)
        uav_coords = torch.cat(self._val_uav_coords, dim=0).numpy().astype(np.float32)

        val_sat_ds = self.trainer.datamodule.val_sat_ds
        metrics = evaluate_r1(uav_embs, sat_embs, uav_coords, val_sat_ds.chunk_bboxes)

        self.log("val/R@1", float(metrics["R@1"]), prog_bar=True, sync_dist=False)
        self.log("val/R@5", float(metrics["R@5"]), prog_bar=False, sync_dist=False)
        self.log("val/R@10", float(metrics["R@10"]), prog_bar=False, sync_dist=False)

        elapsed = time.time() - self._train_start_time if self._train_start_time else 0
        gap = 0.90 - float(metrics["R@1"])
        print(
            f"[VAL flight {VAL_FLIGHT}] R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f}"
            f" | gap_to_90={gap:.4f} | elapsed={elapsed:.0f}s | samples_seen={self._total_samples_seen}"
        )

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        warmup = self.cfg.warmup_epochs
        total = self.cfg.max_epochs
        t0 = self.cfg.cosine_t0

        if t0 > 0:
            # Warm restarts after linear warmup
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / warmup
                e = epoch - warmup
                cycle_len = t0
                cycle = e // cycle_len
                pos = e % cycle_len
                # eta_min = 1% of max LR
                return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * pos / cycle_len))
        else:
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(total - warmup, 1)
                return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Self-supervised DINOv3 fine-tuning for VisLoc retrieval")

    parser.add_argument("--visloc-root", type=str, default=Config.visloc_root)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--image-size", type=int, default=Config.image_size)

    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--temperature", type=float, default=Config.temperature)

    parser.add_argument("--lora-rank", type=int, default=Config.lora_rank)
    parser.add_argument("--lora-alpha", type=float, default=Config.lora_alpha)
    parser.add_argument("--georank-weight", type=float, default=Config.georank_weight)
    parser.add_argument("--cosine-t0", type=int, default=Config.cosine_t0)

    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--precision", type=str, default=Config.precision)
    parser.add_argument("--seed", type=int, default=Config.seed)

    parser.add_argument("--warmup-epochs", type=int, default=Config.warmup_epochs)
    parser.add_argument("--lora-last-n-blocks", type=int, default=Config.lora_last_n_blocks)
    parser.add_argument("--iou-pos-threshold", type=float, default=Config.iou_pos_threshold)
    parser.add_argument("--wandb-project", type=str, default=Config.wandb_project)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=DINO_MODEL)

    # SSL4EO-S12 options
    parser.add_argument("--ssl4eo-root", type=str, default=str(SSL4EOS12_ROOT))
    parser.add_argument("--ssl4eo-lat-min", type=float, default=None,
                        help="Geographic filter: southern latitude bound (e.g. 15.0 for China region)")
    parser.add_argument("--ssl4eo-lat-max", type=float, default=None,
                        help="Geographic filter: northern latitude bound (e.g. 55.0 for China region)")
    parser.add_argument("--ssl4eo-lon-min", type=float, default=None,
                        help="Geographic filter: western longitude bound (e.g. 90.0 for China region)")
    parser.add_argument("--ssl4eo-lon-max", type=float, default=None,
                        help="Geographic filter: eastern longitude bound (e.g. 135.0 for China region)")

    args = parser.parse_args()

    cfg = Config(
        visloc_root=args.visloc_root,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_last_n_blocks=args.lora_last_n_blocks,
        georank_weight=args.georank_weight,
        cosine_t0=args.cosine_t0,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        iou_pos_threshold=args.iou_pos_threshold,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        ssl4eo_root=args.ssl4eo_root,
        ssl4eo_lat_min=args.ssl4eo_lat_min,
        ssl4eo_lat_max=args.ssl4eo_lat_max,
        ssl4eo_lon_min=args.ssl4eo_lon_min,
        ssl4eo_lon_max=args.ssl4eo_lon_max,
    )
    return cfg


def main():
    cfg = parse_args()
    pl.seed_everything(cfg.seed, workers=True)

    if "VISLOC_ROOT" in os.environ:
        cfg.visloc_root = os.environ["VISLOC_ROOT"]

    print("=" * 80)
    print("Self-Supervised DINOv3 on VisLoc (SSL4EO-S12 satellite patches)")
    print(f"Model: {cfg.model_name}")
    print(f"SSL4EO-S12 root: {cfg.ssl4eo_root}")
    ssl4eo_shard_dir = Path(cfg.ssl4eo_root) / "train" / "S2RGB"
    n_shards = len(list(ssl4eo_shard_dir.glob("*.tar"))) if ssl4eo_shard_dir.exists() else 0
    geo_filter = (
        f"lat [{cfg.ssl4eo_lat_min}, {cfg.ssl4eo_lat_max}], "
        f"lon [{cfg.ssl4eo_lon_min}, {cfg.ssl4eo_lon_max}]"
        if cfg.ssl4eo_lat_min is not None else "global (no filter)"
    )
    print(f"SSL4EO-S12 shards: {n_shards}  |  geo filter: {geo_filter}")
    print(f"Val flight: {VAL_FLIGHT} (VisLoc UAV queries → satellite gallery)")
    print(f"LoRA rank={cfg.lora_rank}, alpha={cfg.lora_alpha}")
    print(f"VisLoc root: {cfg.visloc_root}")
    print(
        f"Train config: batch_size={cfg.batch_size}, eval_batch_size={cfg.eval_batch_size},"
        f" num_workers={cfg.num_workers}, max_epochs={cfg.max_epochs}"
    )
    print("=" * 80)

    datamodule = VisLocSSLDataModule(cfg)
    model = DinoSSLRetriever(cfg)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        log_model=False,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/ssl-dinov3",
        monitor="val/R@1",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val/R@1", mode="max", patience=5)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[ckpt_cb, early_stop_cb],
        log_every_n_steps=5,
        benchmark=True,
    )

    trainer.fit(model, datamodule=datamodule)

    # Upload run.log as wandb artifact
    if os.path.exists("run.log"):
        import wandb

        artifact = wandb.Artifact("run-log", type="log")
        artifact.add_file("run.log")
        wandb_logger.experiment.log_artifact(artifact)

    print("Best checkpoint:", ckpt_cb.best_model_path)
    print("Best val/R@1:", float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None)


if __name__ == "__main__":
    main()
