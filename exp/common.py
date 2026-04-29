from __future__ import annotations

import argparse
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import torch
from PIL import Image
from omegaconf import OmegaConf

from starVLA.model.framework.QwenMyVLA import QwenMyVLA


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TOKEN_GROUPS = ("vis", "lang", "sub", "spa", "dyn", "act")


@dataclass
class StageSpec:
    stage_name: str
    config_path: Path
    checkpoint_path: Path


def parse_stage_specs_from_args(args: argparse.Namespace) -> List[StageSpec]:
    return [
        StageSpec("stage1", Path(args.stage1_config), Path(args.stage1_ckpt)),
        StageSpec("stage2", Path(args.stage2_config), Path(args.stage2_ckpt)),
        StageSpec("stage3", Path(args.stage3_config), Path(args.stage3_ckpt)),
    ]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _instruction_from_video_path(video_path: Path) -> str:
    return video_path.stem.strip()


def _list_video_samples(video_path: Path, temporal_delta_indices: List[int]) -> List[Dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    frames: List[Image.Image] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_idx += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames decoded from video file: {video_path}")

    samples = []
    instruction = _instruction_from_video_path(video_path)
    video_tag = video_path.stem
    for idx in range(len(frames)):
        temporal_frames = []
        for delta in temporal_delta_indices:
            temporal_idx = min(max(idx + int(delta), 0), len(frames) - 1)
            temporal_frames.append(frames[temporal_idx].copy())
        samples.append(
            {
                "image_id": f"{video_tag}__frame_{idx + 1:06d}",
                "image_images": [frames[idx].copy()],
                "video_images": temporal_frames,
                "instruction": instruction,
                "subtask": instruction,
                "video_name": video_path.name,
            }
        )
    return samples


def list_episode_samples(episode_dir: Path, temporal_delta_indices: Optional[List[int]] = None) -> List[Dict]:
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode path not found: {episode_dir}")

    temporal_delta_indices = temporal_delta_indices or [0, 19]

    if episode_dir.is_file():
        if episode_dir.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported episode file type: {episode_dir}")
        return _list_video_samples(episode_dir, temporal_delta_indices)

    video_files = sorted([p for p in episode_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES])
    if video_files:
        samples = []
        for video_path in video_files:
            samples.extend(_list_video_samples(video_path, temporal_delta_indices))
        if not samples:
            raise ValueError(f"No frames decoded from video directory: {episode_dir}")
        return samples

    subdirs = sorted([p for p in episode_dir.iterdir() if p.is_dir()])
    if subdirs:
        samples = []
        for idx, subdir in enumerate(subdirs):
            image_paths = sorted([p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
            if not image_paths:
                continue
            temporal_paths = []
            for delta in temporal_delta_indices:
                temporal_idx = min(max(idx + int(delta), 0), len(subdirs) - 1)
                temporal_subdir = subdirs[temporal_idx]
                temporal_images = sorted([p for p in temporal_subdir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
                if not temporal_images:
                    continue
                temporal_paths.append(temporal_images[0])
            samples.append({"image_id": subdir.name, "image_paths": image_paths, "video_paths": temporal_paths})
        if not samples:
            raise ValueError(f"No image files found in nested episode directory: {episode_dir}")
        return samples

    image_paths = sorted([p for p in episode_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    if not image_paths:
        raise ValueError(f"No supported image files found in episode directory: {episode_dir}")
    samples = []
    for idx, path in enumerate(image_paths):
        temporal_paths = []
        for delta in temporal_delta_indices:
            temporal_idx = min(max(idx + int(delta), 0), len(image_paths) - 1)
            temporal_paths.append(image_paths[temporal_idx])
        samples.append({"image_id": path.stem, "image_paths": [path], "video_paths": temporal_paths})
    return samples


def load_images(image_paths: Iterable[Path]) -> List[Image.Image]:
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((224, 224))
        images.append(img)
    return images


def normalize_images(images: Iterable[Image.Image]) -> List[Image.Image]:
    normalized = []
    for img in images:
        normalized.append(img.convert("RGB").resize((224, 224)))
    return normalized


def build_model_from_stage_spec(
    stage_spec: StageSpec,
    device: Optional[str] = None,
    *,
    force_eager_for_attentions: bool = False,
) -> QwenMyVLA:
    cfg = OmegaConf.load(stage_spec.config_path)
    cfg.trainer.pretrained_checkpoint = str(stage_spec.checkpoint_path)
    cfg.trainer.is_resume = False
    cfg.trainer.resume_epoch = None
    cfg.trainer.resume_step = 0
    if force_eager_for_attentions:
        if "qwenvl" not in cfg.framework:
            cfg.framework.qwenvl = {}
        cfg.framework.qwenvl.attn_implementation = "eager"
    model = QwenMyVLA(config=cfg)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def _layout_mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask, got {tuple(mask.shape)}")
    return torch.where(mask)[0]


def build_token_group_labels(
    *,
    input_ids: torch.Tensor,
    sequence_layout_pre_injection: Dict[str, Dict[str, torch.Tensor]],
    action_pos: Optional[torch.Tensor],
) -> List[str]:
    seq_len = int(input_ids.shape[0])
    groups = ["lang"] * seq_len

    image_positions = sequence_layout_pre_injection["image"]["positions"][0]
    for idx in image_positions.tolist():
        if idx >= 0:
            groups[idx] = "vis"

    sub_positions = sequence_layout_pre_injection["subtask"]["positions"][0]
    for idx in sub_positions.tolist():
        if idx >= 0:
            groups[idx] = "sub"

    dyn_positions = sequence_layout_pre_injection["dynamic"]["positions"][0]
    for idx in dyn_positions.tolist():
        if idx >= 0:
            groups[idx] = "dyn"

    spa_positions = sequence_layout_pre_injection["spatial"]["positions"][0]
    for idx in spa_positions.tolist():
        if idx >= 0:
            groups[idx] = "spa"

    if action_pos is not None:
        for idx in action_pos[0].tolist():
            if idx >= 0:
                groups[idx] = "act"

    return groups


@torch.inference_mode()
def analyze_example(
    model: QwenMyVLA,
    *,
    batch_images: List[Image.Image],
    temporal_images: Optional[List[Image.Image]] = None,
    instruction: str,
    analysis_progress: float = 1.0,
) -> Dict:
    examples = [{"image": batch_images, "lang": instruction}]
    if temporal_images is not None and len(temporal_images) > 0:
        examples[0]["video"] = temporal_images
    batch_images_wrapped = [batch_images]
    instructions = [instruction + model._slot_prompt_suffix()]

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=batch_images_wrapped, instructions=instructions)
    input_ids = qwen_inputs["input_ids"]

    dyn_pos = model._gather_positions(input_ids, model.dynamic_token_id, model.k_dyn)
    spa_pos = model._gather_positions(input_ids, model.spatial_token_id, model.k_spa)
    sub_pos = model._gather_positions(input_ids, model.subtask_token_id, model.k_sub)

    enable_action_tokens = bool(getattr(model, "enable_action_tokens", False))
    num_action_tokens = int(getattr(model, "num_action_tokens", 0))
    action_token_id = getattr(model, "action_token_id", None)
    action_pos = None
    action_mask = None
    if enable_action_tokens and action_token_id is not None and num_action_tokens > 0:
        action_pos = model._gather_positions(input_ids, action_token_id, num_action_tokens)
        action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_mask.scatter_(1, action_pos, True)

    sequence_layout_pre_injection = model._build_sequence_layout_pre_injection(
        input_ids=input_ids,
        dyn_pos=dyn_pos,
        spa_pos=spa_pos,
        sub_pos=sub_pos,
        action_mask=action_mask,
    )
    token_type_ids = model._build_token_type_ids(
        input_ids,
        dyn_pos=dyn_pos,
        spa_pos=spa_pos,
        sub_pos=sub_pos,
        action_pos=action_pos,
    )
    self_attention_mask = model._build_self_attention_mask(sequence_layout_pre_injection=sequence_layout_pre_injection)

    batch_size = input_ids.shape[0]
    dynamic_slots = model._build_dynamic_slots(model.dynamic_teacher_encoder(examples=examples))
    spatial_slots = model._build_spatial_slots(model.spatial_teacher_encoder(examples=examples))
    subtask_slots = model._build_subtask_slots(model.subtask_slot_encoder(examples=examples))

    if model.enable_slot_mask:
        inner_mask_ratios_cur, outside_mask_probs_cur = model._resolve_slot_mask_hparams(float(analysis_progress))
        masked = model._apply_slot_masks(
            dynamic_slots,
            spatial_slots,
            subtask_slots,
            inner_mask_ratios=inner_mask_ratios_cur,
            outside_num_masked_probs=outside_mask_probs_cur,
        )
        dynamic_slots = masked["dynamic_slots"]
        spatial_slots = masked["spatial_slots"]
        subtask_slots = masked["subtask_slots"]
        keep_dynamic = masked["keep_dynamic"]
        keep_spatial = masked["keep_spatial"]
        keep_subtask = masked["keep_subtask"]
    else:
        keep_dynamic = torch.ones((dynamic_slots.shape[0], model.k_dyn), dtype=torch.bool, device=dynamic_slots.device)
        keep_spatial = torch.ones((spatial_slots.shape[0], model.k_spa), dtype=torch.bool, device=spatial_slots.device)
        keep_subtask = torch.ones((subtask_slots.shape[0], model.k_sub), dtype=torch.bool, device=subtask_slots.device)

    def inject_slot_hook(_module, _inputs, output):
        batch_idx_dyn = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, model.k_dyn)
        batch_idx_spa = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, model.k_spa)
        batch_idx_sub = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, model.k_sub)
        batch_idx_act = None
        if action_pos is not None:
            batch_idx_act = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, action_pos.shape[1])

        dyn_keep = keep_dynamic.to(device=output.device)
        spa_keep = keep_spatial.to(device=output.device)
        sub_keep = keep_subtask.to(device=output.device)

        dyn_slots_cast = dynamic_slots.to(device=output.device, dtype=output.dtype)
        spa_slots_cast = spatial_slots.to(device=output.device, dtype=output.dtype)
        sub_slots_cast = subtask_slots.to(device=output.device, dtype=output.dtype)

        if dyn_keep.any():
            output[batch_idx_dyn[dyn_keep], dyn_pos[dyn_keep], :] = dyn_slots_cast[dyn_keep]
        if spa_keep.any():
            output[batch_idx_spa[spa_keep], spa_pos[spa_keep], :] = spa_slots_cast[spa_keep]
        if sub_keep.any():
            output[batch_idx_sub[sub_keep], sub_pos[sub_keep], :] = sub_slots_cast[sub_keep]

        action_token_queries = getattr(model, "action_token_queries", None)
        if action_pos is not None and action_token_queries is not None:
            action_queries = action_token_queries.to(device=output.device, dtype=output.dtype)
            action_queries = action_queries.unsqueeze(0).expand(batch_size, -1, -1)
            output[batch_idx_act, action_pos, :] = action_queries
        return output

    embedding_layer = model._resolve_input_embeddings_layer()
    model_device = next(model.parameters()).device
    model._set_typed_ffn_token_type_ids(token_type_ids)
    if model.enable_back_half_typed_attention:
        model._set_back_half_typed_attention_mask(self_attention_mask)
    hook_handle = embedding_layer.register_forward_hook(inject_slot_hook)
    try:
        if model_device.type == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            qwen_outputs = model.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        hook_handle.remove()
        model._set_typed_ffn_token_type_ids(None)
        if model.enable_back_half_typed_attention:
            model._set_back_half_typed_attention_mask(None)

    if not qwen_outputs.attentions:
        raise RuntimeError("Qwen outputs do not contain attentions. Cannot build Sankey analysis.")

    final_hidden = qwen_outputs.hidden_states[-1][0].detach().float().cpu()
    final_attention = qwen_outputs.attentions[-1][0].detach().float().mean(dim=0).cpu()
    token_groups = build_token_group_labels(
        input_ids=input_ids[0].detach().cpu(),
        sequence_layout_pre_injection={
            key: {
                subkey: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for subkey, value in span.items()
            }
            for key, span in sequence_layout_pre_injection.items()
        },
        action_pos=action_pos.detach().cpu() if action_pos is not None else None,
    )

    return {
        "input_ids": input_ids[0].detach().cpu(),
        "token_ids": list(range(int(input_ids.shape[1]))),
        "token_groups": token_groups,
        "hidden": final_hidden,
        "final_attention": final_attention,
    }


def save_stage_feature_bundle(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_stage_feature_bundle(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")
