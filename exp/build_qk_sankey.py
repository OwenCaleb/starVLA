from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from exp.common import TOKEN_GROUPS, ensure_dir, load_stage_feature_bundle

SANKEY_GROUP_ORDER = ("vis", "lang", "sub", "dyn", "spa", "act")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build stage-wise group-level Q->K Sankey CSVs.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _aggregate_qk(sample: dict) -> list[dict]:
    attn = sample["final_attention"].numpy()
    token_groups = sample["token_groups"]
    results = []
    for q_group in SANKEY_GROUP_ORDER:
        q_indices = [i for i, group in enumerate(token_groups) if group == q_group]
        if not q_indices:
            continue
        for k_group in SANKEY_GROUP_ORDER:
            k_indices = [i for i, group in enumerate(token_groups) if group == k_group]
            if not k_indices:
                continue
            value = float(attn[q_indices][:, k_indices].mean())
            results.append(
                {
                    "image_id": sample["image_id"],
                    "q_group": q_group,
                    "k_group": k_group,
                    "value": value,
                }
            )
    return results


def main() -> None:
    args = build_argparser().parse_args()
    features_dir = Path(args.features_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    for stage_name in ("stage1", "stage2", "stage3"):
        print(f"[qk] loading {stage_name} from {features_dir / f'{stage_name}_features.pt'}", flush=True)
        payload = load_stage_feature_bundle(features_dir / f"{stage_name}_features.pt")
        rows = []
        total_samples = len(payload["samples"])
        for sample_idx, sample in enumerate(payload["samples"], start=1):
            if sample_idx == 1 or sample_idx == total_samples or sample_idx % 100 == 0:
                print(
                    f"[qk] {stage_name}: aggregating sample {sample_idx}/{total_samples} ({sample['image_id']})",
                    flush=True,
                )
            rows.extend(_aggregate_qk(sample))

        long_df = pd.DataFrame(rows)
        if long_df.empty:
            raise ValueError(f"No QK rows generated for {stage_name}")

        sankey_df = (
            long_df.groupby(["q_group", "k_group"], as_index=False)["value"]
            .mean()
            .assign(
                source=lambda df: "K_" + df["k_group"].astype(str),
                target=lambda df: "Q_" + df["q_group"].astype(str),
            )[["source", "target", "value"]]
        )
        heatmap_df = long_df.groupby(["q_group", "k_group"], as_index=False)["value"].mean()

        sankey_df.to_csv(output_dir / f"sankey_qk_{stage_name}.csv", index=False)
        long_df.to_csv(output_dir / f"qk_flow_{stage_name}_long.csv", index=False)
        heatmap_df.to_csv(output_dir / f"qk_heatmap_{stage_name}.csv", index=False)
        print(
            f"[qk] saved {stage_name}: "
            f"sankey={len(sankey_df)} rows, long={len(long_df)} rows, heatmap={len(heatmap_df)} rows",
            flush=True,
        )


if __name__ == "__main__":
    main()
