"""
Self-supervised DINOv3 training for VisLoc cross-view retrieval.

Branch goal:
- Fine-tune facebook/dinov3-vitb16-pretrain-lvd1689m with self-supervised learning
- Optimize for Recall@1 on fixed VisLoc evaluation
- No UAV images during training — satellite chunks only

Usage:
  uv run train.py > run.log 2>&1

Environment (optional overrides):
  VISLOC_ROOT=/workspace/data/visloc
  WANDB_API_KEY=... (for automatic wandb login)
"""

from __future__ import annotations
from src.losses import georank_loss
from src.utils import batched_iou

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from pytorch_metric_learning import losses

from prepare import (
    VISLOC_ROOT,
    SatChunkDataset,
    UAVDataset,
    build_ground_truth,
    evaluate_r1,
)

torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# Experiment defaults
# -----------------------------------------------------------------------------

DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VAL_FLIGHT = "03"
TRAIN_FLIGHTS = ["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"]  # val flight included: learn eval region structure

CHUNK_PIXELS = 512
CHUNK_STRIDE = 128
TRAIN_STRIDE = 128  # Produce more chunks for training.

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

TRAIN_SAT_SCALES = {
    "01": 0.20,
    "02": 0.20,
    "03": 0.20,
    "04": 0.20,
    "05": 0.30,
    "06": 0.45,
    "08": 0.25,
    "09": 0.20,
    "10": 0.35,
    "11": 0.20,
}


@dataclass
class Config:
    visloc_root: str = str(VISLOC_ROOT)
    model_name: str = DINO_MODEL
    embedding_dim: int = 768  # CLS token dim
    precision: str = "16-mixed"
    num_workers: int = 8
    seed: int = 42

    eval_batch_size: int = 128

    image_size: int = 336

    max_epochs: int = 7
    warmup_epochs: int = 1
    batch_size: int = 128
    lr: float = 5e-5
    weight_decay: float = 1e-4
    cosine_t0: int = 0  # 0 = plain cosine, no restarts
    llrd_decay: float = 0.6  # per-block LR decay: block[n-1] gets lr*decay, block[n-2] gets lr*decay^2, ...
    gradient_clip_val: float | None = None

    contrastive_temperature: float = 0.1  # contrastive temperature
    georank_weight: float = 0.0  # weight for GeoRank regularization (0 = disabled)
    georank_temperature: float = (
        1.0  # sigmoid slope inside the soft rank - larger -> harder/sharper ranks (less gradient), smaller -> softer ranks
    )

    wandb_project: str = "ssl-dinov3"
    wandb_run_name: str | None = None

    iou_pos_threshold: float = 0.50  # unused
    iou_neg_threshold: float = 0.0  # off-diagonal pairs with IoU > this are masked out of InfoNCE negatives (0 = mask any overlap)


# -----------------------------------------------------------------------------
# SSL Dataset — satellite chunks only, IoU-based positive mining
# -----------------------------------------------------------------------------


class SatSSLDataset(Dataset):
    """
    Cross-scale SSL dataset: anchor = zoomed-in crop (25-50% of chunk area),
    positive = full-scale view of the SAME chunk. Trains scale invariance that
    bridges the UAV (high-res zoomed-in) ↔ satellite (lower-res full area) gap.
    """

    def __init__(
        self,
        root: Path,
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
            scale = sat_scales[flight]
            sat_ds = SatChunkDataset(
                root,
                flight,
                chunk_pixels=CHUNK_PIXELS,
                stride_pixels=TRAIN_STRIDE,
                scale_factor=scale,
                transform=None,
            )
            self.sat_datasets[flight] = sat_ds
            self.samples.extend([(flight, i) for i in range(len(sat_ds))])

        print(f"SSL dataset: {len(self.samples)} chunks across {len(flights)} flights (cross-scale pairs)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        flight, chunk_idx = self.samples[idx]
        sat_ds = self.sat_datasets[flight]
        img, lat, lon = sat_ds[chunk_idx]

        anchor = self.anchor_transform(img)  # zoomed-in view (UAV-like)
        positive = self.positive_transform(img)  # full-scale view (satellite-like)

        coords = torch.tensor([lat, lon], dtype=torch.float32)
        bbox = torch.tensor(sat_ds.chunk_bboxes[chunk_idx], dtype=torch.float32)
        return anchor, positive, coords, coords, bbox


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
        self.positive_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.75, 1.00), ratio=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0),
            ]
            + shared_aug
        )
        # Kept for eval (not used for training)
        self.train_transform = self.anchor_transform
        self.eval_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def setup(self, stage: str | None = None):
        if self.train_ds is None:
            self.train_ds = SatSSLDataset(
                root=self.root,
                flights=TRAIN_FLIGHTS,
                sat_scales=TRAIN_SAT_SCALES,
                anchor_transform=self.anchor_transform,
                positive_transform=self.positive_transform,
                iou_pos_threshold=self.cfg.iou_pos_threshold,
            )

        if self.val_uav_ds is None or self.val_sat_ds is None:
            val_scale = SAT_SCALES[VAL_FLIGHT]
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
# Losses
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Lightning model
# -----------------------------------------------------------------------------


class DinoSSLRetrieverSt1(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(vars(cfg))

        self.backbone = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
        hidden = self.backbone.config.hidden_size

        # Fully freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, cfg.embedding_dim),
            )
        )

        self._ntxent_loss = losses.NTXentLoss(temperature=cfg.contrastive_temperature)
        self._ntxent_loss = losses.SelfSupervisedLoss(self._ntxent_loss)

        self._val_uav_embs = []
        self._val_sat_embs = []
        self._val_uav_coords = []
        self._total_samples_seen = 0
        self._train_start_time = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding."""
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        emb = self.proj(cls)
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def on_train_start(self):
        self._train_start_time = time.time()

    def training_step(self, batch, batch_idx):
        anchor, positive, anchor_coords, pos_coords, bboxes = batch

        q = self.encode(anchor)
        k = self.encode(positive)

        q_ssl, k_ssl = q, k

        infonce = self._ntxent_loss(q_ssl, k_ssl)
        self.log("train/ntxent_loss", infonce, on_step=True, on_epoch=False)

        if self.cfg.georank_weight > 0:
            gr = georank_loss(q, anchor_coords[:, 0], anchor_coords[:, 1], self.cfg.georank_temperature)
            self.log("train/georank_loss", gr, on_step=True, on_epoch=False)

            loss = infonce + self.cfg.georank_weight * gr
        else:
            loss = infonce

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
        print(
            f"[VAL flight {VAL_FLIGHT}] R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f}"
            f" | elapsed={elapsed:.0f}s | samples_seen={self._total_samples_seen}"
        )

    def configure_optimizers(self):
        head_params = list(self.proj.parameters())

        param_groups = [
            {"params": head_params, "lr": self.cfg.lr, "name": "head"},
        ]

        # AdamW optimizer
        optimizer = AdamW(param_groups, weight_decay=self.cfg.weight_decay)

        warmup = self.cfg.warmup_epochs
        t0 = self.cfg.cosine_t0 if self.cfg.cosine_t0 > 0 else max(self.cfg.max_epochs - warmup, 1)

        # Linear warmup + cosine annealing with (optional) restarts.
        warmup_sched = LinearLR(optimizer, start_factor=1.0 / max(warmup, 1), end_factor=1.0, total_iters=warmup)
        cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=t0, eta_min=0)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
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
    parser.add_argument("--gradient-clip-val", type=float, default=Config.gradient_clip_val)

    parser.add_argument("--contrastive-temperature", type=float, default=Config.contrastive_temperature)
    parser.add_argument("--georank-weight", type=float, default=Config.georank_weight)
    parser.add_argument("--georank-temperature", type=float, default=Config.georank_temperature)

    parser.add_argument("--cosine-t0", type=int, default=Config.cosine_t0)
    parser.add_argument("--llrd-decay", type=float, default=Config.llrd_decay)

    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--precision", type=str, default=Config.precision)
    parser.add_argument("--seed", type=int, default=Config.seed)

    parser.add_argument("--warmup-epochs", type=int, default=Config.warmup_epochs)
    parser.add_argument("--iou-pos-threshold", type=float, default=Config.iou_pos_threshold)
    parser.add_argument("--iou-neg-threshold", type=float, default=Config.iou_neg_threshold)
    parser.add_argument("--wandb-project", type=str, default=Config.wandb_project)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=DINO_MODEL)

    args = parser.parse_args()

    cfg = Config(
        visloc_root=args.visloc_root,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        gradient_clip_val=args.gradient_clip_val,
        weight_decay=args.weight_decay,
        contrastive_temperature=args.contrastive_temperature,
        georank_weight=args.georank_weight,
        georank_temperature=args.georank_temperature,
        cosine_t0=args.cosine_t0,
        llrd_decay=args.llrd_decay,
        max_epochs=args.max_epochs,
        precision=args.precision,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        iou_pos_threshold=args.iou_pos_threshold,
        iou_neg_threshold=args.iou_neg_threshold,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    return cfg


def main():
    cfg = parse_args()
    pl.seed_everything(cfg.seed, workers=True)

    if "VISLOC_ROOT" in os.environ:
        cfg.visloc_root = os.environ["VISLOC_ROOT"]

    print("=" * 80)
    print("Self-Supervised DINOv3 on VisLoc (satellite chunks only)")
    print(f"Model: {cfg.model_name}")
    print(f"Train flights: {TRAIN_FLIGHTS}")
    print(f"Val flight: {VAL_FLIGHT}")
    print(f"Satellite scales: {SAT_SCALES}")
    print(f"Training stride: {TRAIN_STRIDE} (eval stride: {CHUNK_STRIDE})")
    print(f"Data root: {cfg.visloc_root}")
    print(
        f"Train config: batch_size={cfg.batch_size}, eval_batch_size={cfg.eval_batch_size},"
        f" num_workers={cfg.num_workers}, max_epochs={cfg.max_epochs}"
    )
    print("=" * 80)

    datamodule = VisLocSSLDataModule(cfg)
    model = DinoSSLRetrieverSt1(cfg)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

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
        filename="best-{epoch:02d}-R@1={val/R@1:.4f}",
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(monitor="val/R@1", mode="max", patience=5)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[ckpt_cb, early_stop_cb, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=5,
        benchmark=True,
        gradient_clip_val=cfg.gradient_clip_val,
        num_sanity_val_steps=0,
    )

    trainer.validate(model, datamodule=datamodule)
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
