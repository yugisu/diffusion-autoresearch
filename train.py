"""
Supervised DINOv3 training for VisLoc cross-view retrieval.

Branch goal:
- Fine-tune facebook/dinov3-vitb16-pretrain-lvd1689m with supervised contrastive learning
- Train on flights: 01, 02, 04, 05, 06, 08, 09, 10, 11
- Validate on flight: 03
- Optimize for Recall@1 on fixed VisLoc evaluation

Usage:
  uv run train.py

Environment (optional overrides):
  VISLOC_ROOT=/workspace/data/visloc
  WANDB_API_KEY=... (for automatic wandb login)
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from prepare import (
    CHUNK_PIXELS,
    CHUNK_STRIDE,
    MAP_SCALE_FACTOR,
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
TRAIN_FLIGHTS = ["01", "02", "04", "05", "06", "08", "09", "10", "11"]
VAL_FLIGHT = "03"

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

DEFAULT_BATCH_SIZE = 128
DEFAULT_EVAL_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 8


@dataclass
class Config:
    visloc_root: str = str(VISLOC_ROOT)
    model_name: str = DINO_MODEL
    image_size: int = 224
    embedding_dim: int = 512

    batch_size: int = DEFAULT_BATCH_SIZE
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS

    lr: float = 1e-5
    weight_decay: float = 1e-4
    temperature: float = 0.07

    max_epochs: int = 20
    max_steps: int = -1
    precision: str = "16-mixed"
    seed: int = 42

    wandb_project: str = "autoresearch-supervised-dinov3"
    wandb_run_name: str | None = None


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


class VisLocTrainPairDataset(Dataset):
    """
    Creates supervised UAV->satellite positive pairs from GPS overlap.
    For each UAV image, positives are chunk indices from build_ground_truth(...).
    """

    def __init__(
        self,
        root: str,
        flights: List[str],
        sat_scales: Dict[str, float],
        uav_transform,
        sat_transform,
    ):
        self.root = root
        self.flights = flights
        self.uav_datasets: Dict[str, UAVDataset] = {}
        self.sat_datasets: Dict[str, SatChunkDataset] = {}
        self.samples: List[tuple[str, int]] = []
        self.gt_per_flight: Dict[str, List[List[int]]] = {}

        for flight in flights:
            scale = sat_scales.get(flight, MAP_SCALE_FACTOR)

            uav_ds = UAVDataset(root, flight, transform=uav_transform)
            sat_ds = SatChunkDataset(
                root,
                flight,
                chunk_pixels=CHUNK_PIXELS,
                stride_pixels=CHUNK_STRIDE,
                scale_factor=scale,
                transform=sat_transform,
            )

            uav_coords = np.array(
                [(float(uav_ds.records.iloc[i]["lat"]), float(uav_ds.records.iloc[i]["lon"])) for i in range(len(uav_ds))]
            )
            gt = build_ground_truth(uav_coords, sat_ds.chunk_bboxes)

            self.uav_datasets[flight] = uav_ds
            self.sat_datasets[flight] = sat_ds
            self.gt_per_flight[flight] = gt

            self.samples.extend([(flight, i) for i in range(len(uav_ds))])

        print(f"Train dataset: {len(self.samples)} UAV samples across {len(flights)} flights (paired with GPS-positive sat chunks).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        flight, uav_idx = self.samples[idx]
        gt_candidates = self.gt_per_flight[flight][uav_idx]
        sat_idx = random.choice(gt_candidates)

        uav_img, _, _ = self.uav_datasets[flight][uav_idx]
        sat_img, _, _ = self.sat_datasets[flight][sat_idx]
        return uav_img, sat_img


class VisLocDataModule(pl.LightningDataModule):
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

        self.train_uav_transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.train_sat_transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def setup(self, stage: str | None = None):
        if self.train_ds is None:
            self.train_ds = VisLocTrainPairDataset(
                root=self.root,
                flights=TRAIN_FLIGHTS,
                sat_scales=SAT_SCALES,
                uav_transform=self.train_uav_transform,
                sat_transform=self.train_sat_transform,
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
        kwargs = dict(
            dataset=self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=True,
        )
        return DataLoader(**kwargs)

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
# Lightning model
# -----------------------------------------------------------------------------


class DinoCrossViewRetriever(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(vars(cfg))

        self.backbone = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
        hidden = self.backbone.config.hidden_size

        # Freeze early blocks (0-7), train only last 4 blocks + norm
        for param in self.backbone.parameters():
            param.requires_grad = False
        encoder_layers = self.backbone.encoder.layer
        for block in encoder_layers[8:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, cfg.embedding_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / cfg.temperature), dtype=torch.float32))

        self._val_uav_embs = []
        self._val_sat_embs = []
        self._val_uav_coords = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        emb = self.proj(cls)
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def _contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100)
        logits = (q @ k.t()) * scale
        labels = torch.arange(q.size(0), device=q.device)
        loss_qk = F.cross_entropy(logits, labels)
        loss_kq = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_qk + loss_kq)

    def training_step(self, batch, batch_idx):
        uav, sat = batch
        q = self.encode(uav)
        k = self.encode(sat)

        loss = self._contrastive_loss(q, k)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=uav.size(0))
        self.log("train/logit_scale", self.logit_scale.exp(), on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def on_validation_epoch_start(self):
        self._val_uav_embs = []
        self._val_sat_embs = []
        self._val_uav_coords = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, lat, lon = batch
        emb = self.encode(imgs)

        if dataloader_idx == 0:
            self._val_uav_embs.append(emb.detach().cpu())
            coords = torch.stack([lat, lon], dim=1)
            self._val_uav_coords.append(coords.detach().cpu())
        else:
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

        gap = 0.90 - float(metrics["R@1"])
        print(
            f"[VAL flight {VAL_FLIGHT}] R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f} | gap_to_90={gap:.4f}"
        )

    def configure_optimizers(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.proj.parameters()) + [self.logit_scale]

        param_groups = [
            {"params": backbone_params, "lr": 5e-5},
            {"params": head_params, "lr": 1e-4},
        ]
        optimizer = AdamW(param_groups, weight_decay=self.cfg.weight_decay)

        if self.trainer.max_steps is not None and self.trainer.max_steps > 0:
            t_max = self.trainer.max_steps
        else:
            train_batches = max(len(self.trainer.datamodule.train_dataloader()), 1)
            t_max = self.trainer.max_epochs * train_batches

        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=5e-5 * 0.05)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Supervised DINOv3 fine-tuning for VisLoc retrieval")

    parser.add_argument("--visloc-root", type=str, default=Config.visloc_root)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)

    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--temperature", type=float, default=Config.temperature)

    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--precision", type=str, default=Config.precision)
    parser.add_argument("--seed", type=int, default=Config.seed)

    parser.add_argument("--wandb-project", type=str, default="autoresearch-supervised-dinov3")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=DINO_MODEL)

    args = parser.parse_args()

    cfg = Config(
        visloc_root=args.visloc_root,
        model_name=args.model_name,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        seed=args.seed,
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
    print("Supervised DINOv3 on VisLoc")
    print(f"Model: {cfg.model_name}")
    print(f"Train flights: {TRAIN_FLIGHTS}")
    print(f"Val flight: {VAL_FLIGHT}")
    print(f"Satellite scales: {SAT_SCALES}")
    print(f"Data root: {cfg.visloc_root}")
    print(
        f"Train config: batch_size={cfg.batch_size}, eval_batch_size={cfg.eval_batch_size}, num_workers={cfg.num_workers}, max_epochs={cfg.max_epochs}"
    )
    print("=" * 80)

    datamodule = VisLocDataModule(cfg)
    model = DinoCrossViewRetriever(cfg)

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        log_model=False,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/supervised-dinov3",
        monitor="val/R@1",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val/R@1", mode="max", patience=6)

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

    print("Best checkpoint:", ckpt_cb.best_model_path)
    print("Best val/R@1:", float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None)


if __name__ == "__main__":
    main()
