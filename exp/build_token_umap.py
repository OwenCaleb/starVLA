from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from exp.common import ensure_dir, load_stage_feature_bundle


def _lazy_import_umap_and_sklearn():
    try:
        import umap  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "build_token_umap.py requires `umap-learn` and `scikit-learn`. "
            "Install them before running this script."
        ) from exc
    return umap, silhouette_score


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stage-wise token UMAP CSVs and separation metrics.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.3)
    parser.add_argument("--umap-metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 normalize hidden features before PCA/UMAP and metrics.",
    )
    parser.add_argument(
        "--max-tokens-per-stage",
        type=int,
        default=0,
        help="Cap tokens kept per stage before UMAP fitting. Set <=0 to disable subsampling.",
    )
    parser.add_argument(
        "--metrics-max-points",
        type=int,
        default=2000,
        help="Cap points used per stage for metrics. Set <=0 to disable subsampling.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="Apply PCA to this dimension before UMAP. Set <=0 to disable PCA.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip separation metrics and export only UMAP CSVs.",
    )
    parser.add_argument(
        "--artifacts-path",
        default=None,
        help="Path to save/load fitted PCA+UMAP artifacts. Defaults to <output-dir>/umap_artifacts.pkl.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit shared PCA+UMAP artifacts and exit without exporting per-stage CSVs.",
    )
    parser.add_argument(
        "--transform-only",
        action="store_true",
        help="Load previously fitted artifacts and export per-stage CSVs without re-fitting.",
    )
    parser.add_argument(
        "--transform-stages",
        nargs="+",
        default=["stage1", "stage2", "stage3"],
        choices=["stage1", "stage2", "stage3"],
        help="Stages to export during transform.",
    )
    return parser


def _records_from_payload(payload: dict, slot_only: bool) -> tuple[list[dict], np.ndarray]:
    rows = []
    hidden_list = []
    allowed = {"sub", "spa", "dyn", "act"} if slot_only else None
    for sample in payload["samples"]:
        hidden = sample["hidden"].numpy()
        for token_id, token_group in zip(sample["token_ids"], sample["token_groups"]):
            if allowed is not None and token_group not in allowed:
                continue
            rows.append(
                {
                    "image_id": sample["image_id"],
                    "token_id": token_id,
                    "token_group": token_group,
                }
            )
            hidden_list.append(hidden[token_id])
    return rows, np.asarray(hidden_list, dtype=np.float32)


def _group_mean_records_from_payload(payload: dict, slot_only: bool) -> tuple[list[dict], np.ndarray]:
    rows = []
    hidden_list = []
    allowed = {"sub", "spa", "dyn", "act"} if slot_only else {"vis", "lang", "sub", "spa", "dyn", "act"}
    for sample in payload["samples"]:
        hidden = sample["hidden"].numpy()
        grouped_indices: dict[str, list[int]] = {}
        for token_id, token_group in zip(sample["token_ids"], sample["token_groups"]):
            if token_group not in allowed:
                continue
            grouped_indices.setdefault(token_group, []).append(int(token_id))

        for token_group, token_ids in grouped_indices.items():
            mean_hidden = hidden[token_ids].mean(axis=0)
            rows.append(
                {
                    "image_id": sample["image_id"],
                    "token_id": -1,
                    "token_group": token_group,
                }
            )
            hidden_list.append(mean_hidden)
    return rows, np.asarray(hidden_list, dtype=np.float32)


def _subsample_rows_and_hidden(
    rows: list[dict],
    hidden: np.ndarray,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[list[dict], np.ndarray]:
    if max_points <= 0 or len(rows) <= max_points:
        return rows, hidden
    indices = np.sort(rng.choice(len(rows), size=max_points, replace=False))
    sampled_rows = [rows[int(i)] for i in indices]
    sampled_hidden = hidden[indices]
    return sampled_rows, sampled_hidden


def _l2_normalize(hidden: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if hidden.size == 0:
        return hidden
    hidden = hidden.astype(np.float32, copy=False)
    norms = np.linalg.norm(hidden, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return hidden / norms


def _should_post_pca_l2_normalize(metric: str) -> bool:
    return metric == "cosine"


def _inter_intra_ratio(hidden: np.ndarray, labels: list[str]) -> float:
    if hidden.shape[0] < 2:
        return float("nan")
    distances = np.linalg.norm(hidden[:, None, :] - hidden[None, :, :], axis=-1)
    same_mask = np.equal.outer(labels, labels)
    diff_mask = ~same_mask
    same_mask[np.eye(same_mask.shape[0], dtype=bool)] = False
    intra = distances[same_mask].mean() if same_mask.any() else np.nan
    inter = distances[diff_mask].mean() if diff_mask.any() else np.nan
    if intra == 0 or np.isnan(intra) or np.isnan(inter):
        return float("nan")
    return float(inter / intra)


def _safe_silhouette_score(hidden: np.ndarray, labels: list[str], silhouette_score, metric: str) -> float:
    if hidden.shape[0] < 2:
        return float("nan")
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(hidden, labels, metric=metric))


def _fit_stage_payloads(
    *,
    features_dir: Path,
    slot_only: bool,
    max_tokens_per_stage: int,
    rng: np.random.Generator,
) -> tuple[dict, dict, np.ndarray]:
    stage_rows = {}
    stage_hidden = {}
    concat_hidden = []
    for stage_name in ("stage1", "stage2", "stage3"):
        payload = load_stage_feature_bundle(features_dir / f"{stage_name}_features.pt")
        rows, hidden = _group_mean_records_from_payload(payload, slot_only=slot_only)
        rows, hidden = _subsample_rows_and_hidden(
            rows,
            hidden,
            max_points=max_tokens_per_stage,
            rng=rng,
        )
        stage_rows[stage_name] = rows
        stage_hidden[stage_name] = hidden
        concat_hidden.append(hidden)
    return stage_rows, stage_hidden, np.concatenate(concat_hidden, axis=0)


def main() -> None:
    args = build_argparser().parse_args()
    if args.fit_only and args.transform_only:
        raise ValueError("`--fit-only` and `--transform-only` cannot be used together.")
    umap_mod, silhouette_score = _lazy_import_umap_and_sklearn()
    pca_cls = None
    if args.pca_dim > 0:
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PCA preprocessing requires `scikit-learn`. Install it before running this script."
            ) from exc
        pca_cls = PCA
    features_dir = Path(args.features_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    artifacts_path = Path(args.artifacts_path) if args.artifacts_path is not None else (output_dir / "umap_artifacts.pkl")
    rng = np.random.default_rng(args.umap_random_state)

    artifacts = {}
    if not args.transform_only:
        print(
            f"[umap] fit start: features_dir={features_dir}, output_dir={output_dir}, "
            f"neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}, "
            f"metric={args.umap_metric}, l2_normalize={args.l2_normalize}, "
            f"pca_dim={args.pca_dim}, max_points={args.max_tokens_per_stage}",
            flush=True,
        )
        for slot_only in (False, True):
            suffix = "_slot_only" if slot_only else ""
            print(f"[umap] fitting shared reducer for suffix='{suffix or 'all'}'", flush=True)
            _stage_rows, _stage_hidden, all_hidden = _fit_stage_payloads(
                features_dir=features_dir,
                slot_only=slot_only,
                max_tokens_per_stage=args.max_tokens_per_stage,
                rng=rng,
            )
            for stage_name in ("stage1", "stage2", "stage3"):
                print(
                    f"[umap] fit payload {stage_name}{suffix}: "
                    f"{len(_stage_rows[stage_name])} points, hidden={_stage_hidden[stage_name].shape}",
                    flush=True,
                )
            print(f"[umap] concatenated{suffix}: hidden={all_hidden.shape}", flush=True)
            if args.l2_normalize:
                print(f"[umap] applying L2 normalize{suffix}", flush=True)
                all_hidden = _l2_normalize(all_hidden)
            pca = None
            if pca_cls is not None and all_hidden.shape[1] > args.pca_dim and all_hidden.shape[0] > args.pca_dim:
                print(f"[umap] applying PCA{suffix}: {all_hidden.shape[1]} -> {args.pca_dim}", flush=True)
                pca = pca_cls(n_components=args.pca_dim, random_state=args.umap_random_state)
                pca.fit(all_hidden)
                all_hidden = pca.transform(all_hidden)
                print(f"[umap] PCA done{suffix}: transformed={all_hidden.shape}", flush=True)
                if args.l2_normalize and _should_post_pca_l2_normalize(args.umap_metric):
                    print(f"[umap] applying post-PCA L2 normalize{suffix} for cosine metric", flush=True)
                    all_hidden = _l2_normalize(all_hidden)

            reducer = umap_mod.UMAP(
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                random_state=args.umap_random_state,
                low_memory=True,
            )
            print(f"[umap] reducer.fit start{suffix}: points={all_hidden.shape[0]}", flush=True)
            reducer.fit(all_hidden)
            print(f"[umap] reducer.fit done{suffix}", flush=True)
            artifacts[suffix] = {"pca": pca, "reducer": reducer}

        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"[umap] saved artifacts -> {artifacts_path}", flush=True)

        if args.fit_only:
            print("[umap] fit-only complete", flush=True)
            return
    else:
        print(f"[umap] transform-only: loading artifacts from {artifacts_path}", flush=True)
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        print("[umap] artifacts loaded", flush=True)

    for slot_only in (False, True):
        suffix = "_slot_only" if slot_only else ""
        stage_names = tuple(args.transform_stages)
        bundle = artifacts[suffix]
        pca = bundle["pca"]
        reducer = bundle["reducer"]
        print(f"[umap] export start for suffix='{suffix or 'all'}'", flush=True)
        for stage_name in stage_names:
            print(f"[umap] loading {stage_name}{suffix}", flush=True)
            payload = load_stage_feature_bundle(features_dir / f"{stage_name}_features.pt")
            rows, hidden = _group_mean_records_from_payload(payload, slot_only=slot_only)
            rows, hidden = _subsample_rows_and_hidden(
                rows,
                hidden,
                max_points=args.max_tokens_per_stage,
                rng=rng,
            )
            if args.l2_normalize:
                print(f"[umap] {stage_name}{suffix}: applying L2 normalize", flush=True)
                hidden = _l2_normalize(hidden)
            print(
                f"[umap] {stage_name}{suffix}: {len(rows)} points before transform, hidden={hidden.shape}",
                flush=True,
            )
            if pca is not None:
                print(f"[umap] {stage_name}{suffix}: applying PCA transform", flush=True)
                hidden = pca.transform(hidden)
                print(f"[umap] {stage_name}{suffix}: PCA transformed={hidden.shape}", flush=True)
                if args.l2_normalize and _should_post_pca_l2_normalize(args.umap_metric):
                    print(f"[umap] {stage_name}{suffix}: applying post-PCA L2 normalize for cosine metric", flush=True)
                    hidden = _l2_normalize(hidden)
            print(f"[umap] {stage_name}{suffix}: reducer.transform start", flush=True)
            embedding = reducer.transform(hidden)
            print(f"[umap] {stage_name}{suffix}: reducer.transform done -> {embedding.shape}", flush=True)
            df = pd.DataFrame(rows)
            df["x"] = embedding[:, 0]
            df["y"] = embedding[:, 1]
            df = df.rename(columns={"image_id": "sample", "token_group": "group"})
            df["sample"] = np.arange(len(df), dtype=int)
            df = df[["sample", "x", "y", "group"]]
            umap_path = output_dir / f"umap_{stage_name}{suffix}.tsv"
            df.to_csv(umap_path, index=False, sep="\t")
            print(f"[umap] saved {umap_path}", flush=True)

            if not args.skip_metrics:
                metrics_rows, metrics_hidden = _subsample_rows_and_hidden(
                    rows,
                    hidden,
                    max_points=args.metrics_max_points,
                    rng=rng,
                )
                metrics_labels = [row["token_group"] for row in metrics_rows]
                print(
                    f"[umap] {stage_name}{suffix}: metrics on {len(metrics_rows)} points",
                    flush=True,
                )
                metrics_df = pd.DataFrame(
                    [
                        {
                            "metric": "silhouette",
                            "value": _safe_silhouette_score(
                                metrics_hidden, metrics_labels, silhouette_score, args.umap_metric
                            ),
                        },
                        {
                            "metric": "inter_intra_ratio",
                            "value": _inter_intra_ratio(metrics_hidden, metrics_labels),
                        },
                    ]
                )
                metrics_path = output_dir / f"token_separation_metrics_{stage_name}{suffix}.csv"
                metrics_df.to_csv(metrics_path, index=False)
                print(f"[umap] saved {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
