from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from exp.common import ensure_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate per-stage token separation metrics into TSV tables.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_stage_metric(metrics_dir: Path, stage_name: str, slot_only: bool) -> dict[str, float]:
    suffix = "_slot_only" if slot_only else ""
    path = metrics_dir / f"token_separation_metrics_{stage_name}{suffix}.csv"
    df = pd.read_csv(path)
    if "metric" not in df.columns or "value" not in df.columns:
        raise ValueError(f"Unexpected metric schema in {path}")
    metric_map = {str(row["metric"]): float(row["value"]) for _, row in df.iterrows()}
    return metric_map


def _build_table(metrics_dir: Path, slot_only: bool) -> pd.DataFrame:
    rows = []
    for stage_name in ("stage1", "stage2", "stage3"):
        metric_map = _load_stage_metric(metrics_dir, stage_name, slot_only=slot_only)
        rows.append(
            {
                "stage": stage_name,
                "silhouette": metric_map.get("silhouette", float("nan")),
                "inter_intra_ratio": metric_map.get("inter_intra_ratio", float("nan")),
            }
        )
    return pd.DataFrame(rows, columns=["stage", "silhouette", "inter_intra_ratio"])


def main() -> None:
    args = build_argparser().parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    all_token_df = _build_table(metrics_dir, slot_only=False)
    slot_only_df = _build_table(metrics_dir, slot_only=True)

    all_token_df.to_csv(output_dir / "token_separation_metrics_all_token.tsv", sep="\t", index=False)
    slot_only_df.to_csv(output_dir / "token_separation_metrics_slot_only.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
