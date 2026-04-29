from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from exp.common import (
    analyze_example,
    build_model_from_stage_spec,
    ensure_dir,
    list_episode_samples,
    load_images,
    normalize_images,
    parse_stage_specs_from_args,
    save_stage_feature_bundle,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract stage-wise hidden states and final-layer attentions.")
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--stage3-config", required=True)
    parser.add_argument("--stage3-ckpt", required=True)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional global instruction override. If omitted and input is video/video-dir, use video filename stem.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["stage1", "stage2", "stage3"],
        choices=["stage1", "stage2", "stage3"],
        help="Subset of stages to extract.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of episode samples to process.",
    )
    parser.add_argument(
        "--analysis-progress",
        type=float,
        default=1.0,
        help="Progress value used when resolving stage mask/schedule semantics during analysis.",
    )
    parser.add_argument(
        "--temporal-delta-indices",
        nargs="+",
        type=int,
        default=[0, 19],
        help="Temporal frame offsets used to build explicit `video` input for the dynamic teacher.",
    )
    parser.add_argument(
        "--force-eager-for-attentions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Qwen attention backend to eager during analysis so `output_attentions=True` works.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    intermediate_dir = ensure_dir(output_dir / "intermediate")
    samples = list_episode_samples(Path(args.episode_dir), temporal_delta_indices=args.temporal_delta_indices)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    stage_specs = parse_stage_specs_from_args(args)
    selected_stages = set(args.stages)

    for stage_spec in stage_specs:
        if stage_spec.stage_name not in selected_stages:
            continue
        print(f"[extract] loading {stage_spec.stage_name} from {stage_spec.checkpoint_path}")
        if args.force_eager_for_attentions:
            print(
                f"[extract] {stage_spec.stage_name}: overriding qwenvl.attn_implementation -> eager for attention capture",
                flush=True,
            )
        model = build_model_from_stage_spec(
            stage_spec,
            device=args.device,
            force_eager_for_attentions=args.force_eager_for_attentions,
        )
        image_results = []
        total_samples = len(samples)
        for sample_idx, sample in enumerate(samples, start=1):
            print(
                f"[extract] {stage_spec.stage_name} sample {sample_idx}/{total_samples}: {sample['image_id']}",
                flush=True,
            )
            instruction = sample.get("instruction", args.instruction)
            if not instruction:
                raise ValueError(
                    "No instruction available for current sample. "
                    "Pass `--instruction`, or use video/video-dir mode where filename stem is used."
                )
            if "image_images" in sample:
                images = normalize_images(sample["image_images"])
                temporal_images = normalize_images(sample["video_images"]) if sample.get("video_images") else None
            else:
                images = load_images(sample["image_paths"])
                temporal_images = load_images(sample["video_paths"]) if sample.get("video_paths") else None
            analysis = analyze_example(
                model,
                batch_images=images,
                temporal_images=temporal_images,
                instruction=instruction,
                analysis_progress=args.analysis_progress,
            )
            analysis["image_id"] = sample["image_id"]
            if "instruction" in sample:
                analysis["instruction"] = sample["instruction"]
            if "video_name" in sample:
                analysis["video_name"] = sample["video_name"]
            image_results.append(analysis)

        save_stage_feature_bundle(
            intermediate_dir / f"{stage_spec.stage_name}_features.pt",
            {
                "stage": stage_spec.stage_name,
                "config_path": str(stage_spec.config_path),
                "checkpoint_path": str(stage_spec.checkpoint_path),
                "instruction": args.instruction,
                "analysis_progress": float(args.analysis_progress),
                "samples": image_results,
            },
        )
        print(f"[extract] saved {stage_spec.stage_name} features", flush=True)
        del model


if __name__ == "__main__":
    main()
