from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from exp.common import ensure_dir, load_stage_feature_bundle


GROUP_ORDER = ("vis", "lang", "sub", "dyn", "spa", "act")
STAGE_ORDER = ("stage1", "stage2", "stage3")
STAGE_LABELS = {"stage1": "Stage I", "stage2": "Stage II", "stage3": "Stage III"}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stage-wise TSV tables for silhouette and log10 inter/intra ratio.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--distance-metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 normalize hidden features before computing pairwise distances.",
    )
    parser.add_argument(
        "--max-points-per-stage",
        type=int,
        default=5000,
        help="Cap token points per stage before pairwise metrics. Set <=0 to disable subsampling.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _stage_hidden_and_labels(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    hidden_chunks = []
    label_chunks = []
    for sample in payload["samples"]:
        hidden = sample["hidden"].numpy().astype(np.float32, copy=False)
        labels = np.asarray(sample["token_groups"], dtype=object)
        hidden_chunks.append(hidden)
        label_chunks.append(labels)
    return np.concatenate(hidden_chunks, axis=0), np.concatenate(label_chunks, axis=0)


def _subsample_hidden_and_labels(
    hidden: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or hidden.shape[0] <= max_points:
        return hidden, labels
    indices = np.sort(rng.choice(hidden.shape[0], size=max_points, replace=False))
    return hidden[indices], labels[indices]


def _safe_log10(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    return float(np.log10(value))


def _pairwise_euclidean(hidden: np.ndarray) -> np.ndarray:
    hidden = hidden.astype(np.float32, copy=False)
    sq_norm = np.sum(hidden * hidden, axis=1, keepdims=True)
    dist_sq = sq_norm + sq_norm.T - 2.0 * (hidden @ hidden.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return np.sqrt(dist_sq, out=dist_sq)


def _l2_normalize(hidden: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if hidden.size == 0:
        return hidden
    hidden = hidden.astype(np.float32, copy=False)
    norms = np.linalg.norm(hidden, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return hidden / norms


def _pairwise_distance(hidden: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return _pairwise_euclidean(hidden)
    if metric == "cosine":
        hidden = _l2_normalize(hidden)
        sim = hidden @ hidden.T
        np.clip(sim, -1.0, 1.0, out=sim)
        return 1.0 - sim
    raise ValueError(f"Unsupported distance metric: {metric}")


def _silhouette_samples(distance_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = labels.shape[0]
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return np.full(n, np.nan, dtype=np.float32)

    label_masks = {label: (labels == label) for label in unique_labels}
    sil = np.full(n, np.nan, dtype=np.float32)

    for idx in range(n):
        own_label = labels[idx]
        own_mask = label_masks[own_label]
        own_count = int(own_mask.sum())
        if own_count <= 1:
            sil[idx] = 0.0
            continue

        a_i = float(distance_matrix[idx, own_mask].sum() / (own_count - 1))

        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = label_masks[other_label]
            if not other_mask.any():
                continue
            other_mean = float(distance_matrix[idx, other_mask].mean())
            if other_mean < b_i:
                b_i = other_mean

        if not np.isfinite(b_i):
            sil[idx] = float("nan")
            continue

        denom = max(a_i, b_i)
        sil[idx] = 0.0 if denom <= 0 else float((b_i - a_i) / denom)

    return sil


def _group_inter_intra_ratio(distance_matrix: np.ndarray, labels: np.ndarray, group: str) -> float:
    group_mask = labels == group
    if group_mask.sum() < 2:
        return float("nan")

    intra_mask = np.outer(group_mask, group_mask)
    np.fill_diagonal(intra_mask, False)
    inter_mask = np.outer(group_mask, ~group_mask)

    intra_vals = distance_matrix[intra_mask]
    inter_vals = distance_matrix[inter_mask]
    if intra_vals.size == 0 or inter_vals.size == 0:
        return float("nan")

    intra = float(intra_vals.mean())
    inter = float(inter_vals.mean())
    if intra <= 0:
        return float("nan")
    return inter / intra


def _all_inter_intra_ratio(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    same_mask = np.equal.outer(labels, labels)
    diff_mask = ~same_mask
    np.fill_diagonal(same_mask, False)

    intra_vals = distance_matrix[same_mask]
    inter_vals = distance_matrix[diff_mask]
    if intra_vals.size == 0 or inter_vals.size == 0:
        return float("nan")

    intra = float(intra_vals.mean())
    inter = float(inter_vals.mean())
    if intra <= 0:
        return float("nan")
    return inter / intra


def _compute_stage_rows(
    *,
    stage_name: str,
    hidden: np.ndarray,
    labels: np.ndarray,
    distance_metric: str,
) -> tuple[dict[str, float | str], dict[str, float | str]]:
    sil_row: dict[str, float | str] = {"stage": STAGE_LABELS[stage_name]}
    iir_row: dict[str, float | str] = {"stage": STAGE_LABELS[stage_name]}

    distance_matrix = _pairwise_distance(hidden, distance_metric)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        sil_row["s.all"] = float("nan")
        sample_sil = None
    else:
        sample_sil = _silhouette_samples(distance_matrix, labels)
        sil_row["s.all"] = float(np.nanmean(sample_sil))
    iir_row["lgIIR.all"] = _safe_log10(_all_inter_intra_ratio(distance_matrix, labels))

    for group in GROUP_ORDER:
        sil_key = f"s.{group}"
        iir_key = f"lgIIR.{group}"
        if sample_sil is None or np.sum(labels == group) == 0:
            sil_row[sil_key] = float("nan")
        else:
            sil_row[sil_key] = float(sample_sil[labels == group].mean())
        iir_row[iir_key] = _safe_log10(_group_inter_intra_ratio(distance_matrix, labels, group))

    return sil_row, iir_row


def main() -> None:
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    rng = np.random.default_rng(args.random_state)

    silhouette_rows = []
    iir_rows = []

    print(
        f"[metrics] start: features_dir={features_dir}, output_dir={output_dir}, "
        f"distance_metric={args.distance_metric}, l2_normalize={args.l2_normalize}, "
        f"max_points_per_stage={args.max_points_per_stage}",
        flush=True,
    )

    for stage_name in STAGE_ORDER:
        print(f"[metrics] loading {stage_name}", flush=True)
        payload = load_stage_feature_bundle(features_dir / f"{stage_name}_features.pt")
        hidden, labels = _stage_hidden_and_labels(payload)
        print(f"[metrics] {stage_name}: raw hidden={hidden.shape}, labels={labels.shape}", flush=True)
        hidden, labels = _subsample_hidden_and_labels(
            hidden,
            labels,
            max_points=args.max_points_per_stage,
            rng=rng,
        )
        print(f"[metrics] {stage_name}: after subsample hidden={hidden.shape}, labels={labels.shape}", flush=True)
        if args.l2_normalize:
            print(f"[metrics] {stage_name}: applying L2 normalize", flush=True)
            hidden = _l2_normalize(hidden)
        sil_row, iir_row = _compute_stage_rows(
            stage_name=stage_name,
            hidden=hidden,
            labels=labels,
            distance_metric=args.distance_metric,
        )
        silhouette_rows.append(sil_row)
        iir_rows.append(iir_row)
        print(f"[metrics] {stage_name}: rows ready", flush=True)

    silhouette_df = pd.DataFrame(
        silhouette_rows,
        columns=["stage", "s.all", "s.vis", "s.lang", "s.sub", "s.dyn", "s.spa", "s.act"],
    )
    iir_df = pd.DataFrame(
        iir_rows,
        columns=[
            "stage",
            "lgIIR.all",
            "lgIIR.vis",
            "lgIIR.lang",
            "lgIIR.sub",
            "lgIIR.dyn",
            "lgIIR.spa",
            "lgIIR.act",
        ],
    )

    silhouette_df.to_csv(output_dir / "silhouette_stage_table.tsv", sep="\t", index=False)
    iir_df.to_csv(output_dir / "log_iir_stage_table.tsv", sep="\t", index=False)
    print(f"[metrics] saved {output_dir / 'silhouette_stage_table.tsv'}", flush=True)
    print(f"[metrics] saved {output_dir / 'log_iir_stage_table.tsv'}", flush=True)


if __name__ == "__main__":
    main()
