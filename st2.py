"""
Stage-2 supervised fine-tuning on top of the best SSL checkpoint.

Backbone: loaded from SSL Exp13 checkpoint (LoRA merged into weights), then fully
          unfrozen with the same LLRD as supervised Exp16.
Training: multi-positive InfoNCE + GPS proximity mask + TwoFlightBatchSampler
          + CosineAnnealingWarmRestarts T_0=10 epochs (2 cycles) + grad clip 1.0 + head lr=2e-5.

exp4 changes vs exp3:
  - Deeper LLRD: 4 backbone tiers (5e-6 / 1e-5 / 1.5e-5 / 2e-5) instead of 3
    Smooths the abrupt 1e-5→2e-5 jump across SSL-adapted blocks 8-11.
  - UAV aug aligned with SSL Exp13 anchor aug: ColorJitter(0.5/0.5/0.3/0.1),
    RandomGrayscale(p=0.1), GaussianBlur(k=9). Eliminates distribution gap between
    what the backbone was adapted to and what SFT trains on.

SSL checkpoint:  checkpoints/dinov3-ssl-best-r@1=0.53-e656447.ckpt
Supervised baseline: R@1 = 73.6% (Exp16, trained from pretrained DINOv3)
Goal: R@1 > 73.6% via SSL head-start on satellite domain

Usage:
  uv run st2.py [--ssl-ckpt PATH] [--wandb-run-name st2-exp1]

Environment (optional overrides):
  VISLOC_ROOT=/workspace/data/visloc
  WANDB_API_KEY=...
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
from train import LoRALinear, apply_lora  # LoRA utilities from SSL training script

torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
SSL_CKPT_DEFAULT = "checkpoints/dinov3-ssl-best-r@1=0.53-e656447.ckpt"

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

# SSL Exp13 LoRA config — must match the checkpoint
SSL_LORA_RANK = 16
SSL_LORA_ALPHA = 32.0
SSL_LORA_LAST_N_BLOCKS = 4

# -----------------------------------------------------------------------------
# LoRA merge utility
# -----------------------------------------------------------------------------


def merge_lora_backbone(backbone: nn.Module) -> nn.Module:
    """
    Merge LoRA deltas (lora_B @ lora_A * scaling) into each base weight in-place,
    then replace every LoRALinear with its unwrapped nn.Linear so the backbone
    looks identical to a standard pretrained model before supervised training.
    """
    replacements: Dict[str, nn.Linear] = {}
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


# -----------------------------------------------------------------------------
# SSL backbone loader
# -----------------------------------------------------------------------------


def load_ssl_backbone(ckpt_path: str, model_name: str = DINO_MODEL) -> nn.Module:
    """
    Instantiate the backbone with LoRA matching SSL Exp13 config, load the SSL
    checkpoint, merge LoRA adapters into base weights, and return a plain backbone
    with all parameters unfrozen and ready for supervised LLRD fine-tuning.
    """
    backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = apply_lora(backbone, rank=SSL_LORA_RANK, alpha=SSL_LORA_ALPHA, last_n_blocks=SSL_LORA_LAST_N_BLOCKS)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    backbone_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
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


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class Config:
    visloc_root: str = str(VISLOC_ROOT)
    model_name: str = DINO_MODEL
    ssl_ckpt: str = SSL_CKPT_DEFAULT
    image_size: int = 336
    embedding_dim: int = 512

    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 8

    weight_decay: float = 1e-4
    temperature: float = 0.07

    max_epochs: int = 20
    max_steps: int = -1
    precision: str = "16-mixed"
    seed: int = 42

    wandb_project: str = "autoresearch-st2-dinov3"
    wandb_run_name: str | None = None


# -----------------------------------------------------------------------------
# Datasets (from train-supervised.py / Exp16)
# -----------------------------------------------------------------------------


class VisLocTrainPairDataset(Dataset):
    """Supervised UAV->satellite positive pairs from GPS overlap."""

    def __init__(
        self,
        root: str,
        flights: List[str],
        sat_scales: Dict[str, float],
        uav_transform,
        sat_transform,
    ):
        self.uav_datasets: Dict[str, UAVDataset] = {}
        self.sat_datasets: Dict[str, SatChunkDataset] = {}
        self.samples: List[tuple] = []
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
            uav_coords = np.array([
                (float(uav_ds.records.iloc[i]["lat"]), float(uav_ds.records.iloc[i]["lon"]))
                for i in range(len(uav_ds))
            ])
            gt = build_ground_truth(uav_coords, sat_ds.chunk_bboxes)
            self.uav_datasets[flight] = uav_ds
            self.sat_datasets[flight] = sat_ds
            self.gt_per_flight[flight] = gt
            self.samples.extend([(flight, i) for i in range(len(uav_ds))])

        print(f"Train dataset: {len(self.samples)} UAV samples across {len(flights)} flights.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        flight, uav_idx = self.samples[idx]
        gt_candidates = self.gt_per_flight[flight][uav_idx]
        sat_idx = random.choice(gt_candidates)
        uav_img, uav_lat, uav_lon = self.uav_datasets[flight][uav_idx]
        sat_img, sat_lat, sat_lon = self.sat_datasets[flight][sat_idx]
        uav_coords = torch.tensor([uav_lat, uav_lon], dtype=torch.float32)
        sat_coords = torch.tensor([sat_lat, sat_lon], dtype=torch.float32)
        return uav_img, sat_img, uav_coords, sat_coords


class TwoFlightBatchSampler(torch.utils.data.Sampler):
    """Each batch draws half from one flight, half from another — geographic hard negatives."""

    def __init__(self, dataset: VisLocTrainPairDataset, batch_size: int):
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.flight_indices: Dict[str, List[int]] = {}
        for idx, (flight, _) in enumerate(dataset.samples):
            self.flight_indices.setdefault(flight, []).append(idx)
        self.flights = list(self.flight_indices.keys())

    def __len__(self) -> int:
        total = sum(len(v) for v in self.flight_indices.values())
        return total // self.batch_size

    def __iter__(self):
        shuffled = {f: idx[:] for f, idx in self.flight_indices.items()}
        for idxs in shuffled.values():
            random.shuffle(idxs)
        pointers = {f: 0 for f in self.flights}
        for _ in range(len(self)):
            chosen = random.sample(self.flights, min(2, len(self.flights)))
            batch: List[int] = []
            for f in chosen:
                ptr = pointers[f]
                idxs = shuffled[f]
                end = ptr + self.half
                if end > len(idxs):
                    random.shuffle(idxs)
                    ptr = 0
                    end = self.half
                batch.extend(idxs[ptr:end])
                pointers[f] = end % len(idxs)
            yield batch


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

        # UAV augmentations — aligned with SSL Exp13 anchor aug to eliminate distribution gap
        self.train_uav_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        # Satellite augmentations from supervised Exp16
        self.train_sat_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def setup(self, stage: str | None = None):
        if self.train_ds is None:
            self.train_ds = VisLocTrainPairDataset(
                root=self.root,
                flights=TRAIN_FLIGHTS,
                sat_scales=SAT_SCALES,
                uav_transform=self.train_uav_transform,
                sat_transform=self.train_sat_transform,
            )
        if self.val_uav_ds is None:
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
                f"Validation flight {VAL_FLIGHT}: {len(self.val_uav_ds)} UAV queries"
                f" | {len(self.val_sat_ds)} sat chunks | scale={val_scale}"
            )

    def train_dataloader(self):
        sampler = TwoFlightBatchSampler(self.train_ds, self.cfg.batch_size)
        return DataLoader(
            dataset=self.train_ds,
            batch_sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self):
        common = dict(
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )
        return [DataLoader(self.val_uav_ds, **common), DataLoader(self.val_sat_ds, **common)]


# -----------------------------------------------------------------------------
# Lightning model
# -----------------------------------------------------------------------------


class DinoCrossViewRetrieverST2(pl.LightningModule):
    """
    Supervised fine-tuning model initialised from the SSL Exp13 checkpoint.

    Backbone init: SSL checkpoint (LoRA merged) — satellite-adapted features.
    Architecture:  backbone CLS → 2-layer MLP projection head → L2-normalised embedding.
    Loss:          multi-positive InfoNCE with GPS proximity mask (100 m threshold).
    Optimizer:     AdamW with LLRD (5e-6 / 1e-5 / 1.5e-5 / 2e-5 / 2e-5 for head).
    Scheduler:     CosineAnnealingWarmRestarts, T_0=10 epochs, step-level.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(vars(cfg))

        self.backbone = load_ssl_backbone(cfg.ssl_ckpt, cfg.model_name)
        hidden = self.backbone.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, cfg.embedding_dim),
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1.0 / cfg.temperature), dtype=torch.float32)
        )

        self._val_uav_embs: list = []
        self._val_sat_embs: list = []
        self._val_uav_coords: list = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        emb = self.proj(cls)
        return F.normalize(emb, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def _multi_pos_infonce(self, q: torch.Tensor, k: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100)
        logits = (q @ k.t()) * scale
        log_probs = F.log_softmax(logits, dim=1)
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss_qk = -(log_probs * pos_mask).sum(dim=1) / n_pos
        log_probs_t = F.log_softmax(logits.t(), dim=1)
        n_pos_t = pos_mask.t().sum(dim=1).clamp(min=1)
        loss_kq = -(log_probs_t * pos_mask.t()).sum(dim=1) / n_pos_t
        return 0.5 * (loss_qk.mean() + loss_kq.mean())

    def _build_pos_mask(self, uav_coords: torch.Tensor, sat_coords: torch.Tensor, threshold_m: float = 100.0) -> torch.Tensor:
        uav_lat = uav_coords[:, 0].cpu().numpy()
        uav_lon = uav_coords[:, 1].cpu().numpy()
        sat_lat = sat_coords[:, 0].cpu().numpy()
        sat_lon = sat_coords[:, 1].cpu().numpy()
        dlat = (sat_lat[None, :] - uav_lat[:, None]) * 111111.0
        mean_lat = np.radians((uav_lat[:, None] + sat_lat[None, :]) / 2.0)
        dlon = (sat_lon[None, :] - uav_lon[:, None]) * 111111.0 * np.cos(mean_lat)
        dist = np.sqrt(dlat**2 + dlon**2)
        mask = torch.tensor(dist < threshold_m, dtype=torch.float32, device=uav_coords.device)
        mask = (mask + torch.eye(len(uav_lat), device=mask.device)).clamp(max=1.0)
        return mask

    def training_step(self, batch, batch_idx):
        uav, sat, uav_coords, sat_coords = batch
        q = self.encode(uav)
        k = self.encode(sat)
        pos_mask = self._build_pos_mask(uav_coords, sat_coords)
        loss = self._multi_pos_infonce(q, k, pos_mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=uav.size(0))
        self.log("train/logit_scale", self.logit_scale.exp(), on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_start(self):
        self._val_uav_embs = []
        self._val_sat_embs = []
        self._val_uav_coords = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, lat, lon = batch
        if dataloader_idx == 0:
            emb = self.encode(imgs)
            self._val_uav_embs.append(emb.detach().cpu())
            self._val_uav_coords.append(torch.stack([lat, lon], dim=1).detach().cpu())
        else:
            e0 = self.encode(imgs)
            e1 = self.encode(torch.rot90(imgs, 1, [2, 3]))
            e2 = self.encode(torch.rot90(imgs, 2, [2, 3]))
            e3 = self.encode(torch.rot90(imgs, 3, [2, 3]))
            emb = F.normalize((e0 + e1 + e2 + e3) / 4.0, dim=-1)
            self._val_sat_embs.append(emb.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_uav_embs or not self._val_sat_embs:
            return
        uav_embs = torch.cat(self._val_uav_embs).numpy().astype(np.float32)
        sat_embs = torch.cat(self._val_sat_embs).numpy().astype(np.float32)
        uav_coords = torch.cat(self._val_uav_coords).numpy().astype(np.float32)
        metrics = evaluate_r1(uav_embs, sat_embs, uav_coords, self.trainer.datamodule.val_sat_ds.chunk_bboxes)
        self.log("val/R@1", float(metrics["R@1"]), prog_bar=True, sync_dist=False)
        self.log("val/R@5", float(metrics["R@5"]), sync_dist=False)
        self.log("val/R@10", float(metrics["R@10"]), sync_dist=False)
        gap = 0.90 - float(metrics["R@1"])
        print(
            f"[VAL flight {VAL_FLIGHT}] R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f}"
            f" R@10={metrics['R@10']:.4f} | gap_to_90={gap:.4f}"
        )

    def configure_optimizers(self):
        early_backbone_params = list(self.backbone.embeddings.parameters())
        early_backbone_params += [p for blk in self.backbone.layer[:4] for p in blk.parameters()]
        mid_backbone_params = [p for blk in self.backbone.layer[4:8] for p in blk.parameters()]
        # Split late backbone into two tiers: 8-9 (SSL-adapted mid-late) at 1.5e-5,
        # 10-11 (SSL-adapted top) at 2e-5 — smooths the 1e-5→2e-5 jump
        mid_late_backbone_params = [p for blk in self.backbone.layer[8:10] for p in blk.parameters()]
        top_backbone_params = [p for blk in self.backbone.layer[10:] for p in blk.parameters()]
        top_backbone_params += list(self.backbone.norm.parameters())
        head_params = list(self.proj.parameters()) + [self.logit_scale]

        param_groups = [
            {"params": early_backbone_params, "lr": 5e-6},
            {"params": mid_backbone_params, "lr": 1e-5},
            {"params": mid_late_backbone_params, "lr": 1.5e-5},
            {"params": top_backbone_params, "lr": 2e-5},
            {"params": head_params, "lr": 2e-5},
        ]
        optimizer = AdamW(param_groups, weight_decay=self.cfg.weight_decay)

        train_batches = max(len(self.trainer.datamodule.train_dataloader()), 1)
        T_0 = 10 * train_batches  # 2 cycles over 20 epochs instead of 4; eliminates destructive late restarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=2e-5 * 0.05)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Stage-2: supervised DINOv3 SFT from SSL checkpoint")
    parser.add_argument("--visloc-root", type=str, default=Config.visloc_root)
    parser.add_argument("--ssl-ckpt", type=str, default=Config.ssl_ckpt)
    parser.add_argument("--model-name", type=str, default=DINO_MODEL)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--embedding-dim", type=int, default=Config.embedding_dim)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--precision", type=str, default=Config.precision)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--wandb-project", type=str, default=Config.wandb_project)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    return Config(
        visloc_root=args.visloc_root,
        model_name=args.model_name,
        ssl_ckpt=args.ssl_ckpt,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def main():
    cfg = parse_args()
    pl.seed_everything(cfg.seed, workers=True)

    if "VISLOC_ROOT" in os.environ:
        cfg.visloc_root = os.environ["VISLOC_ROOT"]

    print("=" * 80)
    print("Stage-2 Supervised DINOv3 SFT from SSL Checkpoint")
    print(f"SSL checkpoint:  {cfg.ssl_ckpt}")
    print(f"Model:           {cfg.model_name}")
    print(f"Train flights:   {TRAIN_FLIGHTS}")
    print(f"Val flight:      {VAL_FLIGHT}")
    print(f"Satellite scales: {SAT_SCALES}")
    print(f"Data root:       {cfg.visloc_root}")
    print(
        f"Train config: batch_size={cfg.batch_size}, eval_batch_size={cfg.eval_batch_size},"
        f" num_workers={cfg.num_workers}, max_epochs={cfg.max_epochs}"
    )
    print("=" * 80)

    datamodule = VisLocDataModule(cfg)
    model = DinoCrossViewRetrieverST2(cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        log_model=False,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/st2-dinov3",
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
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)

    print("Best checkpoint:", ckpt_cb.best_model_path)
    print("Best val/R@1:", float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None)


if __name__ == "__main__":
    main()
