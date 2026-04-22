import torch
from src.utils import haversine_distance

import torch.nn.functional as F


def symnce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float,
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE: diagonal entries are positives. If ``negative_mask``
    is provided (bool, (N, N)), True entries are excluded from the softmax
    denominator in both directions. The diagonal must be False."""
    logits = (q @ k.t()) / temperature
    if negative_mask is not None:
        logits = logits.masked_fill(negative_mask, float("-inf"))
    labels = torch.arange(len(q), device=q.device)
    loss_qk = F.cross_entropy(logits, labels)
    loss_kq = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_qk + loss_kq)


def georank_loss(
    embeddings: torch.Tensor,  # (N, D), L2-normalized
    lats: torch.Tensor,  # (N,) degrees
    lons: torch.Tensor,  # (N,) degrees
    regularization_strength: float = 1.0,
) -> torch.Tensor:
    """
    GeoRank regularization term (Burgert et al., 2026, arXiv:2601.02289).

    Minimizes the Spearman-like rank disagreement between pairwise
    embedding distances and pairwise spherical geographic distances.

    For each anchor i, soft-ranks all other samples by:
      - embedding distance (cosine or L2)
      - geographic (haversine) distance
    Then penalizes the MSE between the two rank vectors.
    """
    N = embeddings.size(0)

    emb_dist = 1.0 - embeddings @ embeddings.T  # (N, N)
    geo_dist = haversine_distance(lats, lons, lats, lons)

    # Mask diagonal with large value so self-distance ranks last
    inf = torch.finfo(emb_dist.dtype).max
    eye = torch.eye(N, dtype=torch.bool, device=embeddings.device)
    emb_dist = emb_dist.masked_fill(eye, inf)
    geo_dist = geo_dist.masked_fill(eye, inf)

    # Differentiable soft rank: rank(x)[i] = 1 + Σⱼ σ((x[i]-x[j]) / strength)
    # Applied per-row: diff[i,j,k] = x[i,j] - x[i,k], sum over k → rank of each j within row i
    # Previous version: torchsort's soft_rank fn; didn't work because it doesn't support torch 2.3
    def _soft_rank(x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-1) - x.unsqueeze(-2)
        return 1.0 + torch.sigmoid(diff * regularization_strength).sum(-1)

    e_ranks = _soft_rank(emb_dist)
    g_ranks = _soft_rank(geo_dist)

    # Normalize to [0, 1]
    e_ranks = e_ranks / N
    g_ranks = g_ranks / N

    return F.mse_loss(e_ranks, g_ranks)
