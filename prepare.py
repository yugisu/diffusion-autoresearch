"""
Fixed VPR evaluation harness for diffusion-autoresearch.
Provides dataset classes and the Recall@1/5/10 evaluation. DO NOT MODIFY.

Usage (one-time sanity check): uv run prepare.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge as rasterio_merge
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

VISLOC_ROOT = Path("/workspace/data/visloc")
SSL4EOS12_ROOT = Path("/workspace/data/SSL4EOS12")
HF_HOME = "/workspace/.hugging_face"
FLIGHT_ID = "03"
TIME_BUDGET = 2700  # seconds (45 minutes wall-clock budget per experiment)

# Satellite gallery config — fixed for fair comparison across runs.
# These define the retrieval database: 2860 chunks at 512px with 128-px stride.
CHUNK_PIXELS = 512
CHUNK_STRIDE = 128
MAP_SCALE_FACTOR = 0.25

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

os.environ["HF_HOME"] = HF_HOME

# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


def _read_sat_bounds(root: Path, flight_id: str) -> tuple[float, float, float, float]:
    """Returns (lat_min, lon_min, lat_max, lon_max) from the VisLoc coordinates CSV."""
    df = pd.read_csv(root / "satellite_coordinates_range.csv")
    row = df[df["mapname"].isin([f"satellite{flight_id}.tif", f"{flight_id}.tif"])].iloc[0]
    return (
        float(row["RB_lat_map"]),  # lat_min  (RB = right-bottom corner)
        float(row["LT_lon_map"]),  # lon_min  (LT = left-top corner)
        float(row["LT_lat_map"]),  # lat_max
        float(row["RB_lon_map"]),  # lon_max
    )


class UAVDataset(Dataset):
    """VisLoc UAV drone images. Used as the query set — never for training."""

    def __init__(self, root: Path, flight_id: str, transform=None):
        self.drone_dir = root / flight_id / "drone"
        self.transform = transform
        df = pd.read_csv(root / flight_id / f"{flight_id}.csv")
        self.records = df[["filename", "lat", "lon"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        img = Image.open(self.drone_dir / row["filename"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, float(row["lat"]), float(row["lon"])


class SatChunkDataset(Dataset):
    """Fixed, evenly-tiled satellite chunks. The retrieval gallery."""

    def __init__(
        self,
        root: Path,
        flight_id: str,
        chunk_pixels: int = CHUNK_PIXELS,
        stride_pixels: int = CHUNK_STRIDE,
        scale_factor: float = MAP_SCALE_FACTOR,
        transform=None,
    ):
        self.chunk_pixels = chunk_pixels
        self.scale_factor = scale_factor
        self.transform = transform

        sat_path = root / flight_id / f"satellite{flight_id}.tif"
        tile_paths = sorted((root / flight_id).glob(f"satellite{flight_id}_*.tif"))

        if sat_path.exists():
            with rasterio.open(sat_path) as src:
                h = int(src.height * scale_factor)
                w = int(src.width * scale_factor)
                data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)
        elif tile_paths:
            srcs = [rasterio.open(p) for p in tile_paths]
            native_res = srcs[0].res
            target_res = (native_res[0] / scale_factor, native_res[1] / scale_factor)
            merged, _ = rasterio_merge(srcs, res=target_res, indexes=[1, 2, 3], resampling=Resampling.bilinear)
            for s in srcs:
                s.close()
            data = merged
            h, w = data.shape[1], data.shape[2]
        else:
            raise FileNotFoundError(f"No satellite TIF found for flight {flight_id}")

        self._img = np.transpose(data, (1, 2, 0))  # (H, W, 3) uint8
        self._bounds = _read_sat_bounds(root, flight_id)  # bounds from CSV (authoritative)
        lat_min, lon_min, lat_max, lon_max = self._bounds

        self._chunks: list[tuple[int, int, float, float]] = []
        self._bboxes: list[tuple[float, float, float, float]] = []
        for y in range(0, h - chunk_pixels, stride_pixels):
            for x in range(0, w - chunk_pixels, stride_pixels):
                cx = x + chunk_pixels // 2
                cy = y + chunk_pixels // 2
                lat = lat_max - (cy / h) * (lat_max - lat_min)
                lon = lon_min + (cx / w) * (lon_max - lon_min)
                self._chunks.append((x, y, lat, lon))
                self._bboxes.append((
                    lat_max - ((y + chunk_pixels) / h) * (lat_max - lat_min),  # lat_min of chunk
                    lon_min + (x / w) * (lon_max - lon_min),  # lon_min of chunk
                    lat_max - (y / h) * (lat_max - lat_min),  # lat_max of chunk
                    lon_min + ((x + chunk_pixels) / w) * (lon_max - lon_min),  # lon_max of chunk
                ))

    @property
    def chunk_coords(self) -> list[tuple[float, float]]:
        return [(lat, lon) for _, _, lat, lon in self._chunks]

    @property
    def chunk_bboxes(self) -> list[tuple[float, float, float, float]]:
        return self._bboxes

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int):
        x, y, lat, lon = self._chunks[idx]
        crop = self._img[y : y + self.chunk_pixels, x : x + self.chunk_pixels]
        img = Image.fromarray(crop)
        if self.transform is not None:
            img = self.transform(img)
        return img, lat, lon


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


def _flat_earth_dist_m(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Approximate flat-earth distance in metres from one point to an array of points."""
    dlat = (lats - lat1) * 111_111
    dlon = (lons - lon1) * 111_111 * np.cos(np.radians(lat1))
    return np.sqrt(dlat**2 + dlon**2)


def build_ground_truth(
    uav_coords: np.ndarray,
    chunk_bboxes: list[tuple[float, float, float, float]],
) -> list[list[int]]:
    """For each UAV query, return indices of satellite chunks that contain its GPS point.

    Falls back to the nearest chunk if the GPS point falls outside all bboxes.
    Multiple overlapping chunks are sorted by distance to the GPS point.
    """
    bboxes = np.array(chunk_bboxes)
    lat_mins, lon_mins = bboxes[:, 0], bboxes[:, 1]
    lat_maxs, lon_maxs = bboxes[:, 2], bboxes[:, 3]
    center_lats = (lat_mins + lat_maxs) / 2
    center_lons = (lon_mins + lon_maxs) / 2
    ground_truth = []
    for lat, lon in uav_coords:
        mask = (lat_mins <= lat) & (lat <= lat_maxs) & (lon_mins <= lon) & (lon <= lon_maxs)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            dists = _flat_earth_dist_m(lat, lon, center_lats, center_lons)
            indices = np.array([np.argmin(dists)])
        else:
            dists = _flat_earth_dist_m(lat, lon, center_lats[indices], center_lons[indices])
            indices = indices[np.argsort(dists)]
        ground_truth.append(indices.tolist())
    return ground_truth


def recall_at_k(preds: np.ndarray, ground_truth: list[list[int]], k: int) -> float:
    """Fraction of queries where any ground-truth chunk appears in the top-k predictions."""
    hits = sum(any(p in gt for p in preds[i, :k]) for i, gt in enumerate(ground_truth))
    return hits / len(ground_truth)


def evaluate_r1(
    uav_embs: np.ndarray,
    sat_embs: np.ndarray,
    uav_coords: np.ndarray,
    chunk_bboxes: list[tuple[float, float, float, float]],
) -> dict:
    """
    Compute Recall@1/5/10 for UAV-to-satellite retrieval.

    L2-normalises both embedding sets, computes cosine similarity, ranks the
    satellite gallery for each UAV query, and checks against GPS-based ground truth.

    Args:
        uav_embs:    (N_uav, D) float32 array of UAV query embeddings
        sat_embs:    (N_sat, D) float32 array of satellite gallery embeddings
        uav_coords:  (N_uav, 2) array of (lat, lon) for each UAV image
        chunk_bboxes: list of (lat_min, lon_min, lat_max, lon_max) per gallery chunk

    Returns:
        dict with keys "R@1", "R@5", "R@10"
    """
    uav_n = uav_embs.astype(np.float32)
    sat_n = sat_embs.astype(np.float32)
    uav_n /= np.linalg.norm(uav_n, axis=1, keepdims=True) + 1e-8
    sat_n /= np.linalg.norm(sat_n, axis=1, keepdims=True) + 1e-8
    sim = uav_n @ sat_n.T  # (N_uav, N_sat) cosine similarity
    preds = np.argsort(-sim, axis=1)  # descending rank
    gt = build_ground_truth(uav_coords, chunk_bboxes)
    return {
        "R@1": recall_at_k(preds, gt, 1),
        "R@5": recall_at_k(preds, gt, 5),
        "R@10": recall_at_k(preds, gt, 10),
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"VISLOC_ROOT    : {VISLOC_ROOT}")
    print(f"SSL4EOS12_ROOT : {SSL4EOS12_ROOT}")
    print(f"HF_HOME        : {HF_HOME}")
    print(f"FLIGHT_ID      : {FLIGHT_ID}")
    print(f"TIME_BUDGET    : {TIME_BUDGET}s")
    print()

    print("Loading UAV dataset...")
    uav_ds = UAVDataset(VISLOC_ROOT, FLIGHT_ID)
    print(f"  UAV images   : {len(uav_ds)}")

    print("Loading satellite dataset...")
    sat_ds = SatChunkDataset(VISLOC_ROOT, FLIGHT_ID)
    print(f"  Sat chunks   : {len(sat_ds)}")

    sd21_path = Path(HF_HOME) / "hub" / "models--sd2-community--stable-diffusion-2-1"
    print()
    print(f"SD21 cached  : {sd21_path.exists()}")

    print()
    print("Setup OK — ready to run experiments.")
