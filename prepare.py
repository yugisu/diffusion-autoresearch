"""
Fixed VPR evaluation harness for diffusion-autoresearch.
Provides dataset classes and the Recall@1/5/10 evaluation. DO NOT MODIFY.

Usage (one-time sanity check): uv run prepare.py
"""

from visloc import UAVDataset, SatChunkDataset

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

VISLOC_ROOT = Path("/workspace/data/visloc")
os.environ["HF_HOME"] = "/workspace/.hugging_face"
FLIGHT_ID = "03"
TIME_BUDGET = 2700  # seconds (45 minutes wall-clock budget per experiment)

# Satellite gallery config — fixed for fair comparison across runs.
# These define the retrieval database: 2860 chunks at 512px with 128-px stride.
CHUNK_PIXELS = 512
CHUNK_STRIDE = 128
MAP_SCALE_FACTOR = 0.25

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
    print(f"VISLOC_ROOT : {VISLOC_ROOT}")
    print(f"FLIGHT_ID   : {FLIGHT_ID}")
    print(f"TIME_BUDGET : {TIME_BUDGET}s")
    print()

    print("Loading UAV dataset...")
    uav_ds = UAVDataset(VISLOC_ROOT, FLIGHT_ID)
    print(f"  UAV images   : {len(uav_ds)}")

    print("Loading satellite dataset...")
    sat_ds = SatChunkDataset(
        VISLOC_ROOT,
        FLIGHT_ID,
        chunk_pixels=512,
        stride_pixels=128,
        scale_factor=0.25,
    )
    print(f"  Sat chunks   : {len(sat_ds)}")

    print()
    print("Setup OK — ready to run experiments.")
