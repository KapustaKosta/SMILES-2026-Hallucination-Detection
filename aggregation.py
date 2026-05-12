"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


SELECTED_LAYERS = (12, 16, 20, 24)
TAIL_TOKENS = 64
MAX_LENGTH = 512

def _safe_std(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.zeros(1, dtype=x.dtype, device=x.device)
    return x.std(unbiased=False).view(1)


def _mst_edge_stats(points: torch.Tensor) -> torch.Tensor:
    """
    Simple 0-dimensional persistent-homology proxy.

    For a point cloud, the death times of connected components in H0
    are the edge lengths of the minimum spanning tree.
    We summarize these edge lengths by mean/std/max/sum.
    """

    n = points.size(0)

    if n <= 1:
        return torch.zeros(4, dtype=points.dtype, device=points.device)

    points = F.normalize(points.float(), dim=1)
    distances = torch.cdist(points, points)

    selected = torch.zeros(n, dtype=torch.bool, device=points.device)
    selected[0] = True

    min_dist = distances[0].clone()
    edges = []

    for _ in range(n - 1):
        min_dist[selected] = float("inf")
        next_idx = torch.argmin(min_dist)

        edge = min_dist[next_idx]
        edges.append(edge)

        selected[next_idx] = True
        min_dist = torch.minimum(min_dist, distances[next_idx])

    edges = torch.stack(edges)

    return torch.cat([
        edges.mean().view(1),
        edges.std(unbiased=False).view(1),
        edges.max().view(1),
        edges.sum().view(1),
    ])


def _spectral_stats(points: torch.Tensor) -> torch.Tensor:
    """
    Geometry of the local token cloud via singular values.

    Features:
    - normalized first singular value
    - normalized sum of first 5 singular values
    - effective rank
    - participation ratio
    """

    if points.size(0) <= 1:
        return torch.zeros(4, dtype=points.dtype, device=points.device)

    centered = points.float() - points.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)

    total = singular_values.sum() + 1e-8
    probs = singular_values / total

    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    effective_rank = torch.exp(entropy)

    participation_ratio = (singular_values.sum() ** 2) / (
        (singular_values ** 2).sum() + 1e-8
    )

    top1_ratio = singular_values[0] / total
    top5_ratio = singular_values[:5].sum() / total

    return torch.stack([
        top1_ratio,
        top5_ratio,
        effective_rank,
        participation_ratio,
    ])


def _token_path_stats(points: torch.Tensor) -> torch.Tensor:
    """
    Geometry of the sequence trajectory inside one layer.
    """

    if points.size(0) <= 1:
        return torch.zeros(6, dtype=points.dtype, device=points.device)

    steps = points[1:] - points[:-1]
    step_lengths = steps.norm(dim=1)

    path_length = step_lengths.sum().view(1)
    mean_step = step_lengths.mean().view(1)
    std_step = _safe_std(step_lengths)
    max_step = step_lengths.max().view(1)

    if steps.size(0) <= 1:
        mean_curvature = torch.zeros(1, dtype=points.dtype, device=points.device)
        std_curvature = torch.zeros(1, dtype=points.dtype, device=points.device)
    else:
        cos_steps = F.cosine_similarity(
            steps[1:],
            steps[:-1],
            dim=1,
        )
        mean_curvature = cos_steps.mean().view(1)
        std_curvature = _safe_std(cos_steps)

    return torch.cat([
        path_length,
        mean_step,
        std_step,
        max_step,
        mean_curvature,
        std_curvature,
    ])


def _cloud_distance_stats(points: torch.Tensor) -> torch.Tensor:
    """
    Basic geometry of token cloud around its centroid.
    """

    center = points.mean(dim=0, keepdim=True)
    distances = (points - center).norm(dim=1)

    return torch.cat([
        distances.mean().view(1),
        _safe_std(distances),
        distances.max().view(1),
    ])


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(1)
    real_positions = real_positions.to(hidden_states.device)

    tail_len = min(TAIL_TOKENS, real_positions.numel())
    tail_positions = real_positions[-tail_len:]

    n_layers = hidden_states.size(0)
    selected_layers = [min(layer_idx, n_layers - 1) for layer_idx in SELECTED_LAYERS]

    vector_features = []
    scalar_features = []

    last_vectors = []
    mean_tail_vectors = []
    std_tail_vectors = []

    for layer_idx in selected_layers:
        layer = hidden_states[layer_idx]

        real_tokens = layer.index_select(0, real_positions)
        tail_tokens = layer.index_select(0, tail_positions)

        last_vec = real_tokens[-1]
        mean_tail = tail_tokens.mean(dim=0)
        std_tail = tail_tokens.std(dim=0, unbiased=False)
        last_minus_tail = last_vec - mean_tail

        vector_features.extend([
            last_vec,
            mean_tail,
            std_tail,
            last_minus_tail,
        ])

        last_vectors.append(last_vec)
        mean_tail_vectors.append(mean_tail)
        std_tail_vectors.append(std_tail)

        scalar_features.append(_cloud_distance_stats(tail_tokens))
        scalar_features.append(_token_path_stats(tail_tokens))
        scalar_features.append(_spectral_stats(tail_tokens))
        scalar_features.append(_mst_edge_stats(tail_tokens))

    seq_len_feature = torch.tensor(
        [real_positions.numel() / MAX_LENGTH],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    scalar_features.append(seq_len_feature)

    for last_vec, mean_tail, std_tail in zip(
        last_vectors,
        mean_tail_vectors,
        std_tail_vectors,
    ):
        scalar_features.append(last_vec.norm().view(1))
        scalar_features.append(mean_tail.norm().view(1))
        scalar_features.append(std_tail.norm().view(1))

        cos_last_tail = F.cosine_similarity(
            last_vec.view(1, -1),
            mean_tail.view(1, -1),
            dim=1,
        )
        scalar_features.append(cos_last_tail)

    for left_vec, right_vec in zip(last_vectors[:-1], last_vectors[1:]):
        cos = F.cosine_similarity(
            left_vec.view(1, -1),
            right_vec.view(1, -1),
            dim=1,
        )
        drift = (right_vec - left_vec).norm().view(1)

        scalar_features.append(cos)
        scalar_features.append(drift)

    layer_path = torch.stack(last_vectors, dim=0)
    scalar_features.append(_token_path_stats(layer_path))
    scalar_features.append(_cloud_distance_stats(layer_path))
    scalar_features.append(_mst_edge_stats(layer_path))

    return torch.cat(vector_features + scalar_features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
