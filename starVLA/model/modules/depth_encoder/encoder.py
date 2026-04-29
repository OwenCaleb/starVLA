"""
Depth-Anything-V2 encoder adapter for starVLA.

This wrapper keeps Depth-Anything-V2 as an external vendor component and only handles:
- model construction
- checkpoint loading
- frozen/eval mode handling
- minimal feature forward interface for spatial teacher usage
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _find_depth_anything_root() -> Path:
	"""Find Depth-Anything-V2 root by walking upward from this file and cwd."""
	candidates = []

	for p in Path(__file__).resolve().parents:
		candidates.append(p)

	try:
		cwd = Path.cwd().resolve()
		candidates.append(cwd)
		candidates.extend(list(cwd.parents))
	except Exception:
		pass

	seen = set()
	for base in candidates:
		if base in seen:
			continue
		seen.add(base)
		da = base / "Depth-Anything-V2"
		if da.is_dir() and (da / "depth_anything_v2").is_dir():
			return da

	raise ModuleNotFoundError(
		"Cannot locate Depth-Anything-V2 root. "
		"Expected repository layout containing ./Depth-Anything-V2/depth_anything_v2"
	)


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
	"""Resolve depth checkpoint path from a file or a directory."""
	path = Path(checkpoint_path).expanduser().resolve()
	if path.is_file():
		return path

	if path.is_dir():
		patterns = ["*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.bin"]
		matches = []
		for pat in patterns:
			matches.extend(sorted(path.rglob(pat)))

		if not matches:
			raise FileNotFoundError(f"No checkpoint file found under directory: {path}")

		ranked = sorted(
			matches,
			key=lambda p: (
				"depth_anything_v2" not in p.name.lower(),
				"vitl" not in p.name.lower(),
				len(str(p)),
			),
		)
		return ranked[0]

	raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def _load_depth_anything_class():
	"""Resolve DepthAnythingV2 class without modifying vendor code."""
	try:
		mod = importlib.import_module("depth_anything_v2.dpt")
		return getattr(mod, "DepthAnythingV2")
	except ImportError:
		depth_root = _find_depth_anything_root()
		if str(depth_root) not in sys.path:
			sys.path.append(str(depth_root))

		mod = importlib.import_module("depth_anything_v2.dpt")
		return getattr(mod, "DepthAnythingV2")


def _get_model_config(encoder: str) -> Dict:
	configs = {
		"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
		"vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
		"vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
		"vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
	}
	if encoder not in configs:
		raise ValueError(f"Unsupported encoder '{encoder}'. Choose from {list(configs.keys())}")
	return configs[encoder]


def _pil_or_np_to_bgr(image) -> np.ndarray:
	if isinstance(image, Image.Image):
		rgb = np.array(image.convert("RGB"), dtype=np.uint8)
		return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

	if isinstance(image, np.ndarray):
		if image.ndim != 3 or image.shape[2] not in (3, 4):
			raise ValueError(f"Expect image ndarray shape [H, W, C], got {image.shape}")
		if image.shape[2] == 4:
			image = image[:, :, :3]
		if image.dtype != np.uint8:
			image = np.clip(image, 0, 255).astype(np.uint8)
		# assume RGB ndarray by default
		return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	raise TypeError(f"Unsupported image type: {type(image)}")


class DepthAnythingV2Encoder(nn.Module):
	"""Frozen spatial teacher wrapper around Depth-Anything-V2."""

	def __init__(
		self,
		checkpoint_path: str,
		encoder: str = "vitl",
		input_size: int = 518,
		freeze: bool = True,
	) -> None:
		super().__init__()
		model_cls = _load_depth_anything_class()
		model_cfg = _get_model_config(encoder)
		self.model = model_cls(**model_cfg)

		resolved_ckpt = _resolve_checkpoint_path(checkpoint_path)
		state_dict = torch.load(str(resolved_ckpt), map_location="cpu")
		self.model.load_state_dict(state_dict, strict=True)

		self.encoder_name = encoder
		self.input_size = input_size

		if freeze:
			self.model.requires_grad_(False)
			self.model.eval()

	@property
	def device(self) -> torch.device:
		return next(self.model.parameters()).device

	@torch.no_grad()
	def _extract_decoder_input_feature_maps(self, features, patch_h: int, patch_w: int):
		"""Extract multi-scale feature maps that are fed into DPT decoder.

		These are the outputs right after depth_head.projects + resize_layers,
		i.e. the true decoder inputs (before refinenet fusion).
		"""
		head = self.model.depth_head
		out = []
		for i, feat in enumerate(features):
			if head.use_clstoken:
				tokens, cls_token = feat[0], feat[1]
				readout = cls_token.unsqueeze(1).expand_as(tokens)
				tokens = head.readout_projects[i](torch.cat((tokens, readout), -1))
			else:
				tokens = feat[0]

			fmap = tokens.permute(0, 2, 1).reshape((tokens.shape[0], tokens.shape[-1], patch_h, patch_w))
			fmap = head.projects[i](fmap)
			fmap = head.resize_layers[i](fmap)
			out.append(fmap)

		return out

	@torch.no_grad()
	def _batch_decoder_feature_maps(self, per_image_decoder_maps: List[List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
		"""Pad and batch decoder input maps for image-list path.

		Input:
		- per_image_decoder_maps: length N, each item has 4 tensors [C_l, H_l, W_l]
		Output keys:
		- decoder_map_l{1..4}: [N, C_l, Hmax_l, Wmax_l]
		- decoder_map_mask_l{1..4}: [N, Hmax_l, Wmax_l]
		"""
		num_levels = len(per_image_decoder_maps[0])
		outputs = {}

		for li in range(num_levels):
			maps = [item[li] for item in per_image_decoder_maps]
			c = maps[0].shape[0]
			max_h = max(m.shape[1] for m in maps)
			max_w = max(m.shape[2] for m in maps)

			batched = torch.zeros((len(maps), c, max_h, max_w), device=self.device, dtype=maps[0].dtype)
			mask = torch.zeros((len(maps), max_h, max_w), device=self.device, dtype=torch.bool)

			for ni, fmap in enumerate(maps):
				h, w = fmap.shape[1], fmap.shape[2]
				batched[ni, :, :h, :w] = fmap
				mask[ni, :h, :w] = True

			outputs[f"decoder_map_l{li + 1}"] = batched
			outputs[f"decoder_map_mask_l{li + 1}"] = mask

		return outputs

	@torch.no_grad()
	def _infer_depth_and_tokens(self, image, input_size: Optional[int] = None):
		"""Run one pass and return depth map plus encoder tokens.

		Returns:
		- depth_map: [H, W]
		- patch_tokens: [T, C] from last encoder stage
		- cls_tokens: [L, C] class tokens from selected intermediate layers
		- decoder_feature_maps: list of 4 tensors [C_l, H_l, W_l]
		"""
		use_size = input_size or self.input_size
		raw_bgr = _pil_or_np_to_bgr(image)

		x, (h, w) = self.model.image2tensor(raw_bgr, use_size)
		x = x.to(self.device)
		patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

		features = self.model.pretrained.get_intermediate_layers(
			x,
			self.model.intermediate_layer_idx[self.encoder_name],
			return_class_token=True,
		)
		decoder_inputs = self._extract_decoder_input_feature_maps(features, patch_h, patch_w)

		depth = self.model.depth_head(features, patch_h, patch_w)
		depth = torch.relu(depth)
		depth = torch.nn.functional.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]

		patch_tokens = features[-1][0][0]  # [T, C]
		cls_tokens = torch.stack([item[1][0] for item in features], dim=0)  # [L, C]
		decoder_feature_maps = [fmap[0] for fmap in decoder_inputs]

		return depth, patch_tokens, cls_tokens, decoder_feature_maps

	@torch.no_grad()
	def encode_image(self, image, input_size: Optional[int] = None) -> torch.Tensor:
		"""Encode one image and return depth map tensor [H, W] on current device."""
		depth, _, _, _ = self._infer_depth_and_tokens(image, input_size=input_size)
		return depth

	@torch.no_grad()
	def encode_image_features(self, image, input_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
		"""Encode one image and return depth plus encoder feature tokens."""
		depth, patch_tokens, cls_tokens, decoder_feature_maps = self._infer_depth_and_tokens(image, input_size=input_size)
		return {
			"depth_map": depth,
			"patch_tokens": patch_tokens,
			"cls_tokens": cls_tokens,
			"decoder_feature_maps": decoder_feature_maps,
		}

	@torch.no_grad()
	def encode_images(self, images: Sequence, input_size: Optional[int] = None) -> torch.Tensor:
		"""Encode a sequence of images and return [N, H, W]."""
		depth_maps = [self.encode_image(img, input_size=input_size) for img in images]
		return torch.stack(depth_maps, dim=0)

	@torch.no_grad()
	def encode_examples(self, examples: List[dict], input_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
		"""Encode starVLA batch examples.

		Semantics:
		- prefer `sample['image']` as the current-timestep spatial input
		- fall back to `sample['video']` only when the current image is unavailable
		- return per-sample depth stacks and a pooled depth summary
		"""
		def _select_spatial_images(sample: dict):
			images = sample.get("image")
			if isinstance(images, (list, tuple)) and len(images) > 0:
				return list(images)

			video = sample.get("video")
			if video is not None:
				if isinstance(video, (list, tuple)) and len(video) > 0:
					first_frame = video[0]
					return [first_frame] if not isinstance(first_frame, (list, tuple)) else list(first_frame)
				if isinstance(video, torch.Tensor):
					if video.ndim == 4:
						frame = video[0].detach().cpu()
					elif video.ndim == 5:
						frame = video[0, 0].detach().cpu()
					else:
						raise ValueError(f"Unsupported temporal video tensor shape: {tuple(video.shape)}")
					if frame.ndim == 3 and frame.shape[0] in (1, 3):
						frame = frame.permute(1, 2, 0)
					frame_np = frame.numpy()
					return [frame_np]
			raise ValueError("Each sample must provide a non-empty list in sample['video'] or sample['image']")

		batch_depth = []
		batch_tokens = []
		batch_cls = []
		batch_decoder = []
		for sample in examples:
			images = _select_spatial_images(sample)

			per_depth = []
			per_tokens = []
			per_cls = []
			per_decoder = []
			for img in images:
				out = self.encode_image_features(img, input_size=input_size)
				per_depth.append(out["depth_map"])
				per_tokens.append(out["patch_tokens"])
				per_cls.append(out["cls_tokens"])
				per_decoder.append(out["decoder_feature_maps"])

			batch_depth.append(torch.stack(per_depth, dim=0))
			batch_tokens.append(per_tokens)
			batch_cls.append(torch.stack(per_cls, dim=0))
			batch_decoder.append(per_decoder)

		# Variable number of views is possible; pad to max V for batching.
		max_views = max(item.shape[0] for item in batch_depth)
		max_h = max(item.shape[1] for item in batch_depth)
		max_w = max(item.shape[2] for item in batch_depth)

		padded = []
		view_mask = []
		for item in batch_depth:
			v, h, w = item.shape
			pad = torch.zeros((max_views, max_h, max_w), device=item.device, dtype=item.dtype)
			pad[:v, :h, :w] = item
			padded.append(pad)

			mask = torch.zeros((max_views,), device=item.device, dtype=torch.bool)
			mask[:v] = True
			view_mask.append(mask)

		depth_maps = torch.stack(padded, dim=0)  # [B, Vmax, H, W]
		view_mask = torch.stack(view_mask, dim=0)  # [B, Vmax]

		# Simple pooled scaffold summary for downstream adapter/projection.
		denom = view_mask.sum(dim=1, keepdim=True).clamp(min=1).to(depth_maps.dtype)
		pooled_depth = (depth_maps * view_mask[:, :, None, None].to(depth_maps.dtype)).sum(dim=1) / denom[:, None, None]

		# Patch-token batching: [B, Vmax, Tmax, C] + boolean token mask.
		token_dim = batch_tokens[0][0].shape[1]
		max_tokens = max(tok.shape[0] for sample_tokens in batch_tokens for tok in sample_tokens)

		padded_tokens = []
		token_mask = []
		for sample_tokens in batch_tokens:
			v = len(sample_tokens)
			tok_pad = torch.zeros((max_views, max_tokens, token_dim), device=self.device, dtype=sample_tokens[0].dtype)
			mask_pad = torch.zeros((max_views, max_tokens), device=self.device, dtype=torch.bool)

			for vi, tok in enumerate(sample_tokens):
				t = tok.shape[0]
				tok_pad[vi, :t] = tok
				mask_pad[vi, :t] = True

			if v < max_views:
				tok_pad[v:] = 0
				mask_pad[v:] = False

			padded_tokens.append(tok_pad)
			token_mask.append(mask_pad)

		encoder_tokens = torch.stack(padded_tokens, dim=0)  # [B, Vmax, Tmax, C]
		token_mask = torch.stack(token_mask, dim=0)  # [B, Vmax, Tmax]

		# Pool over valid (view, token) entries as a compact spatial scaffold summary.
		valid = token_mask.to(encoder_tokens.dtype)
		num_valid = valid.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
		pooled_tokens = (encoder_tokens * valid[..., None]).sum(dim=(1, 2)) / num_valid.squeeze(-1)

		# Class tokens from selected layers: [B, Vmax, L, C]
		cls_token_dim = batch_cls[0].shape[-1]
		num_cls_layers = batch_cls[0].shape[-2]
		cls_padded = []
		for cls_item in batch_cls:
			v = cls_item.shape[0]
			pad = torch.zeros((max_views, num_cls_layers, cls_token_dim), device=self.device, dtype=cls_item.dtype)
			pad[:v] = cls_item
			cls_padded.append(pad)
		cls_tokens = torch.stack(cls_padded, dim=0)

		# Decoder input feature maps (4 scales): [B, Vmax, C_l, H_l, W_l] + spatial masks.
		decoder_outputs = {}
		num_levels = len(batch_decoder[0][0])
		for li in range(num_levels):
			all_maps = []
			for sample_decoder in batch_decoder:
				for view_maps in sample_decoder:
					all_maps.append(view_maps[li])

			c = all_maps[0].shape[0]
			max_h = max(m.shape[1] for m in all_maps)
			max_w = max(m.shape[2] for m in all_maps)

			maps_pad = torch.zeros((len(examples), max_views, c, max_h, max_w), device=self.device, dtype=all_maps[0].dtype)
			maps_mask = torch.zeros((len(examples), max_views, max_h, max_w), device=self.device, dtype=torch.bool)

			for bi, sample_decoder in enumerate(batch_decoder):
				for vi, view_maps in enumerate(sample_decoder):
					fmap = view_maps[li]
					h, w = fmap.shape[1], fmap.shape[2]
					maps_pad[bi, vi, :, :h, :w] = fmap
					maps_mask[bi, vi, :h, :w] = True

			decoder_outputs[f"decoder_map_l{li + 1}"] = maps_pad
			decoder_outputs[f"decoder_map_mask_l{li + 1}"] = maps_mask

		outputs = {
			"depth_maps": depth_maps,
			"view_mask": view_mask,
			"pooled_depth": pooled_depth,
			"encoder_tokens": encoder_tokens,
			"token_mask": token_mask,
			"pooled_tokens": pooled_tokens,
			"cls_tokens": cls_tokens,
		}
		outputs.update(decoder_outputs)
		return outputs

	@torch.no_grad()
	def forward(self, examples: Optional[List[dict]] = None, images: Optional[Sequence] = None) -> Dict[str, torch.Tensor]:
		if examples is not None:
			return self.encode_examples(examples)
		if images is not None:
			per_image = [self.encode_image_features(img) for img in images]
			depth_maps = torch.stack([item["depth_map"] for item in per_image], dim=0)

			token_dim = per_image[0]["patch_tokens"].shape[1]
			max_tokens = max(item["patch_tokens"].shape[0] for item in per_image)
			padded_tokens = []
			token_mask = []
			cls_tokens = []
			for item in per_image:
				tok = item["patch_tokens"]
				t = tok.shape[0]
				pad = torch.zeros((max_tokens, token_dim), device=tok.device, dtype=tok.dtype)
				mask = torch.zeros((max_tokens,), device=tok.device, dtype=torch.bool)
				pad[:t] = tok
				mask[:t] = True
				padded_tokens.append(pad)
				token_mask.append(mask)
				cls_tokens.append(item["cls_tokens"])

			encoder_tokens = torch.stack(padded_tokens, dim=0)  # [N, Tmax, C]
			token_mask = torch.stack(token_mask, dim=0)  # [N, Tmax]
			cls_tokens = torch.stack(cls_tokens, dim=0)  # [N, L, C]

			valid = token_mask.to(encoder_tokens.dtype)
			num_valid = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
			pooled_tokens = (encoder_tokens * valid[..., None]).sum(dim=1) / num_valid

			decoder_outputs = self._batch_decoder_feature_maps([item["decoder_feature_maps"] for item in per_image])

			outputs = {
				"depth_maps": depth_maps,
				"encoder_tokens": encoder_tokens,
				"token_mask": token_mask,
				"pooled_tokens": pooled_tokens,
				"cls_tokens": cls_tokens,
			}
			outputs.update(decoder_outputs)
			return outputs
		raise ValueError("Either examples or images must be provided")


def build_depth_encoder(cfg) -> DepthAnythingV2Encoder:
	"""Factory helper for starVLA config integration."""
	depth_cfg = cfg.framework.depth_encoder
	return DepthAnythingV2Encoder(
		checkpoint_path=depth_cfg.checkpoint_path,
		encoder=depth_cfg.get("encoder", "vitl"),
		input_size=depth_cfg.get("input_size", 518),
		freeze=depth_cfg.get("freeze", True),
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Depth-Anything-V2 spatial encoder smoke test")
	parser.add_argument("--checkpoint_path", type=str, default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/Depth-Anything-V2-Large", help="Depth-Anything-V2 checkpoint file or dir")
	parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
	parser.add_argument("--input_size", type=int, default=518)
	parser.add_argument("--test_mode", type=str, default="forward", choices=["construct", "forward"])
	parser.add_argument("--num_images", type=int, default=2)
	parser.add_argument("--height", type=int, default=480)
	parser.add_argument("--width", type=int, default=640)
	args = parser.parse_args()

	ckpt = _resolve_checkpoint_path(args.checkpoint_path)
	print(f"[INFO] Using checkpoint: {ckpt}")

	model = DepthAnythingV2Encoder(
		checkpoint_path=str(ckpt),
		encoder=args.encoder,
		input_size=args.input_size,
		freeze=True,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	print("[OK] Constructed DepthAnythingV2Encoder")
	print(f"[INFO] Device: {device}")

	if args.test_mode == "construct":
		print("[OK] construct test finished")
		raise SystemExit(0)

	random_images = [
		np.random.randint(0, 255, size=(args.height, args.width, 3), dtype=np.uint8)
		for _ in range(args.num_images)
	]
	outputs = model(images=random_images)
	print("[OK] forward test finished")
	print(f"[INFO] depth_maps shape: {tuple(outputs['depth_maps'].shape)}")
	print(f"[INFO] encoder_tokens shape: {tuple(outputs['encoder_tokens'].shape)}")
	print(f"[INFO] token_mask shape: {tuple(outputs['token_mask'].shape)}")
	print(f"[INFO] pooled_tokens shape: {tuple(outputs['pooled_tokens'].shape)}")
	print(f"[INFO] cls_tokens shape: {tuple(outputs['cls_tokens'].shape)}")
	print(f"[INFO] decoder_map_l1 shape: {tuple(outputs['decoder_map_l1'].shape)}")
	print(f"[INFO] decoder_map_l2 shape: {tuple(outputs['decoder_map_l2'].shape)}")
	print(f"[INFO] decoder_map_l3 shape: {tuple(outputs['decoder_map_l3'].shape)}")
	print(f"[INFO] decoder_map_l4 shape: {tuple(outputs['decoder_map_l4'].shape)}")
	print(f"[INFO] decoder_map_mask_l1 shape: {tuple(outputs['decoder_map_mask_l1'].shape)}")
 

'''
[OK] forward test finished
[INFO] depth_maps shape: (2, 480, 640)
[INFO] encoder_tokens shape: (2, 1813, 1024)
[INFO] token_mask shape: (2, 1813)
[INFO] pooled_tokens shape: (2, 1024)
[INFO] cls_tokens shape: (2, 4, 1024)
depth_maps     = 最终深度图
encoder_tokens = 空间 patch tokens（最重要）
token_mask     = token 有效掩码
pooled_tokens  = 全局空间摘要
cls_tokens     = 4层全局 cls 特征

Maybe concat([pooled_tokens, flatten(cls_tokens)])
'''
