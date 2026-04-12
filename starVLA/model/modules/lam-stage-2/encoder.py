"""
UniVLA LAM stage-2 encoder adapter for starVLA.

This adapter keeps UniVLA as an external vendor component and only wraps:
- model construction
- checkpoint loading
- frozen/eval mode handling
- minimal vq_encode interface
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import sys
import importlib
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def _find_univla_root() -> Path:
    """Find UniVLA root by walking upward from this file and cwd."""
    candidates = []

    # 1) Walk from current file path upward.
    for p in Path(__file__).resolve().parents:
        candidates.append(p)

    # 2) Also consider cwd hierarchy for direct script execution from other dirs.
    try:
        for p in Path.cwd().resolve().parents:
            candidates.append(p)
        candidates.append(Path.cwd().resolve())
    except Exception:
        pass

    seen = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        uni = base / "UniVLA"
        if uni.is_dir() and (uni / "latent_action_model").is_dir():
            return uni

    raise ModuleNotFoundError(
        "Cannot locate UniVLA root with latent_action_model/. "
        "Expected repository layout containing ./UniVLA/latent_action_model"
    )


def _resolve_checkpoint_path(ckpt_path: str) -> Path:
    """Resolve checkpoint file path.

    Accepts either a direct file path, or a directory containing checkpoint files.
    """
    path = Path(ckpt_path).expanduser().resolve()
    if path.is_file():
        return path

    if path.is_dir():
        patterns = ["*.ckpt", "*.pt", "*.pth", "*.safetensors", "*.bin"]
        matches = []
        for pat in patterns:
            matches.extend(sorted(path.rglob(pat)))
        if not matches:
            raise FileNotFoundError(f"No checkpoint file found under directory: {path}")

        # Prefer files that look like stage-2 or latent-action checkpoints.
        ranked = sorted(
            matches,
            key=lambda p: (
                "stage-2" not in p.as_posix().lower() and "stage2" not in p.as_posix().lower(),
                "latent" not in p.as_posix().lower(),
                len(p.as_posix()),
            ),
        )
        return ranked[0]

    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def _stack_frames(frames) -> torch.Tensor:
    """Stack a list/tuple of 2+ frames into [T, C, H, W]."""
    if isinstance(frames, torch.Tensor):
        if frames.ndim == 4:
            return frames
        if frames.ndim == 3:
            return frames.unsqueeze(0)
        raise ValueError(f"Unsupported temporal tensor shape: {tuple(frames.shape)}")

    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise ValueError("Temporal frames must be a non-empty list/tuple or a tensor")

    stacked = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            stacked.append(transforms.ToTensor()(frame))
        elif isinstance(frame, torch.Tensor):
            stacked.append(frame)
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

    return torch.stack(stacked, dim=0)


def _load_univla_lam_class():
    """Resolve UniVLA latent action model class without modifying vendor code."""
    try:
        mod = importlib.import_module("latent_action_model.genie.modules.lam")
        return getattr(mod, "ControllableDINOLatentActionModel")
    except ImportError:
        univla_root = _find_univla_root()
        if str(univla_root) not in sys.path:
            sys.path.append(str(univla_root))

        mod = importlib.import_module("latent_action_model.genie.modules.lam")
        return getattr(mod, "ControllableDINOLatentActionModel")


class UniVLALAMStage2Encoder(nn.Module):
    """Minimal frozen wrapper for UniVLA stage-2 latent action encoder."""

    def __init__(
        self,
        ckpt_path: str,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        in_dim: int = 3,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        model_cls = _load_univla_lam_class()
        self.model = model_cls(
            in_dim=in_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            patch_size=patch_size,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=0.0,
        )
        self._load_checkpoint(ckpt_path)

        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

        self.image_to_tensor = transforms.ToTensor()

    def _load_checkpoint(self, ckpt_path: str) -> None:
        resolved_ckpt_path = _resolve_checkpoint_path(ckpt_path)
        ckpt = torch.load(str(resolved_ckpt_path), map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        new_ckpt = {}
        for key, value in state_dict.items():
            if key.startswith("lam."):
                new_ckpt[key.replace("lam.", "", 1)] = value
            else:
                new_ckpt[key] = value

        self.model.load_state_dict(new_ckpt, strict=True)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _examples_to_video_tensor(self, examples: List[dict]) -> torch.Tensor:
        """Build [B, T, 3, H, W] tensor from examples.

        Important semantic rule:
        - `image` is treated as multi-view at one time step.
        - temporal teacher input must come from an explicit temporal/video field.
        """
        clips = []
        for sample in examples:
            temporal = None
            for key in ("videos", "video", "temporal_video", "temporal_frames", "lam_video"):
                if key in sample:
                    temporal = sample[key]
                    break

            if temporal is None:
                raise ValueError(
                    "Temporal teacher input is missing. Provide an explicit 'video'/'videos' field "
                    "for time-pair or temporal sequence input. Do not reuse sample['image']; "
                    "that field is reserved for same-timestep multi-view images."
                )

            clip = _stack_frames(temporal)
            clips.append(clip)

        videos = torch.stack(clips, dim=0)
        return videos.to(self.device)

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model.vq_encode(videos)
        return {
            "indices": outputs["indices"],
            "z_q": outputs["z_q"],
            "emb": outputs["emb"],
        }

    @torch.no_grad()
    def encode_examples(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        videos = self._examples_to_video_tensor(examples)
        return self.encode(videos)

    @torch.no_grad()
    def forward(self, examples: Optional[List[dict]] = None, videos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if videos is not None:
            return self.encode(videos.to(self.device))
        if examples is not None:
            return self.encode_examples(examples)
        raise ValueError("Either examples or videos must be provided")


def build_lam_stage2_encoder(cfg) -> UniVLALAMStage2Encoder:
    lam_cfg = cfg.framework.lam_stage2
    num_latents = lam_cfg.get("num_latents", lam_cfg.get("codebook_size", 16))
    return UniVLALAMStage2Encoder(
        ckpt_path=lam_cfg.ckpt_path,
        model_dim=lam_cfg.model_dim,
        latent_dim=lam_cfg.latent_dim,
        num_latents=num_latents,
        patch_size=lam_cfg.patch_size,
        enc_blocks=lam_cfg.enc_blocks,
        dec_blocks=lam_cfg.dec_blocks,
        num_heads=lam_cfg.num_heads,
        in_dim=lam_cfg.get("in_dim", 3),
        freeze=lam_cfg.get("freeze", True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default='/mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/univla-latent-action-model/lam-stage-2.ckpt',
        help="Path to UniVLA lam-stage-2 checkpoint file or directory",
    )
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_latents", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--enc_blocks", type=int, default=12)
    parser.add_argument("--dec_blocks", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--in_dim", type=int, default=3)
    parser.add_argument(
        "--test_mode",
        type=str,
        default="forward",
        choices=["construct", "forward"],
        help="construct: only init+load; forward: run random video forward",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint_path(args.ckpt_path)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    encoder = UniVLALAMStage2Encoder(
        ckpt_path=str(ckpt_path),
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        patch_size=args.patch_size,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        num_heads=args.num_heads,
        in_dim=args.in_dim,
        freeze=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    print("[OK] Constructed UniVLALAMStage2Encoder")
    print(f"[INFO] Device: {device}")

    if args.test_mode == "construct":
        print("[OK] construct test finished")
        raise SystemExit(0)

    videos = torch.rand(
        args.batch_size,
        2,
        args.in_dim,
        args.height,
        args.width,
        device=device,
        dtype=torch.float32,
    )

    outputs = encoder(videos=videos)
    print("[OK] forward test finished")
    print(f"[INFO] indices shape: {tuple(outputs['indices'].shape)}")
    print(f"[INFO] z_q shape: {tuple(outputs['z_q'].shape)}")
    print(f"[INFO] emb shape: {tuple(outputs['emb'].shape)}")
