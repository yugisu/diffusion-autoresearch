import torch.nn.functional as F
import torch
from typing import Dict, List, Tuple


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(lambda x: torch.deg2rad(x.float()), [lat1, lon1, lat2, lon2])
    dlat = lat2.unsqueeze(0) - lat1.unsqueeze(1)
    dlon = lon2.unsqueeze(0) - lon1.unsqueeze(1)
    a = torch.sin(dlat / 2) ** 2 + (torch.cos(lat1.unsqueeze(1)) * torch.cos(lat2.unsqueeze(0)) * torch.sin(dlon / 2) ** 2)
    return 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0, 1))


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


def batched_iou(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for a batch of (lat_min, lon_min, lat_max, lon_max) bboxes.
    Returns (N, N) float tensor in [0, 1]."""
    lat_min, lon_min, lat_max, lon_max = boxes.unbind(dim=-1)
    inter_lat = (torch.minimum(lat_max[:, None], lat_max[None, :]) - torch.maximum(lat_min[:, None], lat_min[None, :])).clamp(min=0)
    inter_lon = (torch.minimum(lon_max[:, None], lon_max[None, :]) - torch.maximum(lon_min[:, None], lon_min[None, :])).clamp(min=0)
    inter = inter_lat * inter_lon
    area = (lat_max - lat_min) * (lon_max - lon_min)
    union = area[:, None] + area[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))
