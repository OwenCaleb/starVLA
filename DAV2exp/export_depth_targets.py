from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.modules.depth_encoder.encoder import DepthAnythingV2Encoder


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export high-quality / low-quality depth targets and dyn residual targets from a frame directory."
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/DAV2exp/episode",
        help="Directory containing ordered RGB frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/DAV2exp/output",
        help="Directory to save spa/dyn targets.",
    )
    parser.add_argument(
        "--hq-checkpoint-path",
        type=str,
        default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/Depth-Anything-V2-Large",
        help="High-quality Depth-Anything-V2 checkpoint file or directory.",
    )
    parser.add_argument(
        "--lq-checkpoint-path",
        type=str,
        default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/Depth-Anything-V2-Small",
        help="Low-quality Depth-Anything-V2 checkpoint file or directory.",
    )
    parser.add_argument("--hq-encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--lq-encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--hq-input-size", type=int, default=518, help="High-quality inference size.")
    parser.add_argument("--lq-input-size", type=int, default=224, help="Low-quality inference size.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--lq-blur-kernel",
        type=int,
        default=3,
        help="Gaussian kernel size for low-quality depth blur. Use odd number; 0 or 1 disables blur.",
    )
    parser.add_argument(
        "--lq-blur-sigma",
        type=float,
        default=0.6,
        help="Gaussian sigma for low-quality depth blur.",
    )
    parser.add_argument(
        "--dyn-blur-kernel",
        type=int,
        default=3,
        help="Gaussian kernel size for dyn residual smoothing. Use odd number; 0 or 1 disables blur.",
    )
    parser.add_argument(
        "--dyn-blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for dyn residual smoothing.",
    )
    parser.add_argument(
        "--dyn-threshold-mode",
        type=str,
        default="percentile",
        choices=["percentile", "mad"],
        help="Thresholding mode for dyn visualization mask.",
    )
    parser.add_argument(
        "--dyn-threshold-percentile",
        type=float,
        default=90.0,
        help="Percentile used when dyn-threshold-mode=percentile.",
    )
    parser.add_argument(
        "--dyn-mad-scale",
        type=float,
        default=2.5,
        help="Scale for median + scale * MAD threshold when dyn-threshold-mode=mad.",
    )
    parser.add_argument(
        "--dyn-min-area",
        type=int,
        default=50,
        help="Minimum connected component area kept in dyn mask.",
    )
    return parser.parse_args()


def list_frames(episode_dir: Path) -> List[Path]:
    frames = [p for p in sorted(episode_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not frames:
        raise FileNotFoundError(f"No image frames found under {episode_dir}")
    return frames


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def robust_minmax_normalize(array: np.ndarray, lower_q: float = 2.0, upper_q: float = 98.0) -> np.ndarray:
    lo = float(np.percentile(array, lower_q))
    hi = float(np.percentile(array, upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(array))
        hi = float(np.max(array))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    norm = (array.astype(np.float32) - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    norm = robust_minmax_normalize(depth)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def save_bgr(path: Path, bgr: np.ndarray) -> None:
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Failed to save image: {path}")


def maybe_blur_float_map(array: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    kernel = int(kernel)
    if kernel <= 1:
        return array.astype(np.float32, copy=False)
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(array.astype(np.float32), (kernel, kernel), sigmaX=float(sigma), sigmaY=float(sigma))


def fit_affine_depth_alignment(src: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    x = src.reshape(-1).astype(np.float64)
    y = target.reshape(-1).astype(np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = float(solution[0])
    b = float(solution[1])
    return a, b


def threshold_residual_map(residual: np.ndarray, mode: str, percentile: float, mad_scale: float) -> Tuple[np.ndarray, float]:
    if mode == "percentile":
        tau = float(np.percentile(residual, percentile))
    else:
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        tau = median + float(mad_scale) * mad
    mask = residual > tau
    return mask, tau


def refine_binary_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    refined = np.zeros_like(binary)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            refined[labels == label_idx] = 255
    return refined.astype(bool)


def colorize_dyn_residual(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((residual.shape[0], residual.shape[1], 3), dtype=np.uint8)
    if not mask.any():
        return canvas

    values = residual[mask]
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values.astype(np.float32) - lo) / (hi - lo)

    color_values = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    canvas[mask] = color_values.reshape(-1, 3)
    return canvas


def export_spa_branch(
    model: DepthAnythingV2Encoder,
    frames: List[Path],
    branch_name: str,
    output_dir: Path,
    blur_kernel: int,
    blur_sigma: float,
) -> List[np.ndarray]:
    depth_npy_dir = output_dir / branch_name / "depth_npy"
    depth_png_dir = output_dir / branch_name / "depth_png"
    ensure_dirs([depth_npy_dir, depth_png_dir])

    depth_maps: List[np.ndarray] = []
    for index, frame_path in enumerate(frames, start=1):
        print(f"[{branch_name}] {index}/{len(frames)} {frame_path.name}")
        rgb = load_rgb(frame_path)
        depth = model.encode_image(rgb).detach().cpu().numpy().astype(np.float32)
        depth = maybe_blur_float_map(depth, blur_kernel, blur_sigma)
        depth_maps.append(depth)
        stem = frame_path.stem
        np.save(depth_npy_dir / f"{stem}.npy", depth)
        save_bgr(depth_png_dir / f"{stem}.png", colorize_depth(depth))
    return depth_maps


def export_dyn_branch(
    depth_maps: List[np.ndarray],
    frame_names: List[str],
    output_dir: Path,
    blur_kernel: int,
    blur_sigma: float,
    threshold_mode: str,
    threshold_percentile: float,
    mad_scale: float,
    min_area: int,
) -> None:
    residual_npy_dir = output_dir / "dyn" / "residual_npy"
    mask_npy_dir = output_dir / "dyn" / "mask_npy"
    residual_png_dir = output_dir / "dyn" / "residual_png"
    ensure_dirs([residual_npy_dir, mask_npy_dir, residual_png_dir])

    for index in range(len(depth_maps) - 1):
        stem = frame_names[index]
        depth_t = depth_maps[index]
        depth_next = depth_maps[index + 1]

        a, b = fit_affine_depth_alignment(depth_next, depth_t)
        aligned_next = a * depth_next + b
        residual = np.abs(depth_t - aligned_next).astype(np.float32)
        residual = maybe_blur_float_map(residual, blur_kernel, blur_sigma)

        mask, _ = threshold_residual_map(
            residual,
            mode=threshold_mode,
            percentile=threshold_percentile,
            mad_scale=mad_scale,
        )
        mask = refine_binary_mask(mask, min_area=min_area)

        np.save(residual_npy_dir / f"{stem}.npy", residual.astype(np.float32))
        np.save(mask_npy_dir / f"{stem}.npy", mask.astype(np.uint8))
        save_bgr(residual_png_dir / f"{stem}.png", colorize_dyn_residual(residual, mask))


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    frames = list_frames(episode_dir)
    frame_names = [frame.stem for frame in frames]

    device = torch.device(args.device)

    hq_model = DepthAnythingV2Encoder(
        checkpoint_path=args.hq_checkpoint_path,
        encoder=args.hq_encoder,
        input_size=args.hq_input_size,
        freeze=True,
    ).to(device)
    hq_model.eval()

    lq_model = DepthAnythingV2Encoder(
        checkpoint_path=args.lq_checkpoint_path,
        encoder=args.lq_encoder,
        input_size=args.lq_input_size,
        freeze=True,
    ).to(device)
    lq_model.eval()

    hq_depth_maps = export_spa_branch(
        model=hq_model,
        frames=frames,
        branch_name="spa_hq",
        output_dir=output_dir,
        blur_kernel=0,
        blur_sigma=0.0,
    )
    export_spa_branch(
        model=lq_model,
        frames=frames,
        branch_name="spa_lq",
        output_dir=output_dir,
        blur_kernel=args.lq_blur_kernel,
        blur_sigma=args.lq_blur_sigma,
    )
    export_dyn_branch(
        depth_maps=hq_depth_maps,
        frame_names=frame_names,
        output_dir=output_dir,
        blur_kernel=args.dyn_blur_kernel,
        blur_sigma=args.dyn_blur_sigma,
        threshold_mode=args.dyn_threshold_mode,
        threshold_percentile=args.dyn_threshold_percentile,
        mad_scale=args.dyn_mad_scale,
        min_area=args.dyn_min_area,
    )

    print("[done] spa_hq targets:", output_dir / "spa_hq")
    print("[done] spa_lq targets:", output_dir / "spa_lq")
    print("[done] dyn targets:", output_dir / "dyn")
    print("[note] dyn is computed from aligned high-quality depth residuals; last frame has no dyn target.")


if __name__ == "__main__":
    main()
