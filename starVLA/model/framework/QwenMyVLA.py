"""QwenMyVLA framework with direct slot replacement.

Slot semantics (no alpha/beta gating):
- Dynamic slot: z_q -> g_dyn -> [B, K_dyn, H], then directly used as slot tokens
- Spatial slot: encoder_tokens + token_mask -> g_spa -> [B, K_spa, H]
- Subtask slot: text_tokens + text_mask -> g_sub -> [B, K_sub, H]

Injected to Qwen by placeholder characters in prompt and embedding replacement hook.
"""

from __future__ import annotations

import copy
import importlib.util
import math
from contextlib import nullcontext
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency guard
    LoraConfig = None
    get_peft_model = None

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.DiTActionHeader import ActionModel as DiTActionModel
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model as get_fast_action_model
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.modules.projector.QFormer import LayerwiseQFormer
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.QWen3 import IMAGE_TOKEN_INDEX
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


def _load_lam_stage2_builder():
    """Load encoder builder from lam-stage-2 directory.

    The directory name includes '-', so we use a file-based loader.
    """
    module_path = Path(__file__).resolve().parents[1] / "modules" / "lam-stage-2" / "encoder.py"
    if not module_path.exists():
        raise FileNotFoundError(f"LAM stage-2 adapter not found: {module_path}")

    spec = importlib.util.spec_from_file_location("starvla_lam_stage2_encoder", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_lam_stage2_encoder


def _load_depth_builder():
    module_path = Path(__file__).resolve().parents[1] / "modules" / "depth_encoder" / "encoder.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Depth encoder adapter not found: {module_path}")

    spec = importlib.util.spec_from_file_location("starvla_depth_encoder", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_depth_encoder


def _load_subtask_slot_builder():
    module_path = Path(__file__).resolve().parents[1] / "modules" / "qwen3-embedding" / "encoder.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Qwen3-Embedding adapter not found: {module_path}")

    spec = importlib.util.spec_from_file_location("starvla_qwen3_embedding_encoder", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_qwen3_embedding_slot_adapter


class TokenAwareResampler(nn.Module):
    """Lightweight token-aware compressor/resampler.

    Input:  tokens [B, N, C_in], mask [B, N]
    Output: slots [B, K, C_out]
    """

    def __init__(self, in_dim: int, out_dim: int, num_slots: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_slots = num_slots

        self.query = nn.Parameter(torch.randn(num_slots, in_dim) * 0.02)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expect tokens [B, N, C], got {tuple(tokens.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"Expect mask [B, N], got {tuple(mask.shape)}")

        param_dtype = self.query.dtype
        tokens = tokens.to(dtype=param_dtype)
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)

        logits = torch.matmul(query, tokens.transpose(1, 2)) / math.sqrt(self.in_dim)
        logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.matmul(weights, tokens)
        return self.out_proj(pooled)


class TypedResidualFFN(nn.Module):
    """Wrap Qwen gated MLP with three typed residual experts.

    Base path stays identical to the original Qwen FFN:
        base = base_mlp(x)
    Expert path adds one residual expert depending on token type:
        out = base + expert_type(x) on token positions of that type

    Important:
        Each transformer layer owns its own TypedResidualFFN instance,
        so experts are not shared across layers.

    Token type ids:
        0: base
        1: subtask
        2: dynamic
        3: spatial
        4: action
    """

    TYPE_IDS = {"base": 0, "subtask": 1, "dynamic": 2, "spatial": 3, "action": 4}

    def __init__(self, base_mlp: nn.Module, freeze_base_mlp: bool = True) -> None:
        super().__init__()
        self.base_mlp = base_mlp

        self.experts = nn.ModuleDict({
            "subtask": copy.deepcopy(base_mlp),
            "dynamic": copy.deepcopy(base_mlp),
            "spatial": copy.deepcopy(base_mlp),
            "action": copy.deepcopy(base_mlp),
        })
        self.base_mlp.requires_grad_(False)
        self.experts.requires_grad_(True)
        self.token_type_ids: Optional[torch.Tensor] = None

    def set_token_type_ids(self, token_type_ids: Optional[torch.Tensor]) -> None:
        self.token_type_ids = token_type_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_mlp(x)
        if self.token_type_ids is None:
            return base_out

        token_type_ids = self.token_type_ids.to(device=x.device)
        if token_type_ids.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"token_type_ids shape {tuple(token_type_ids.shape)} does not match hidden shape {tuple(x.shape)}"
            )

        output = base_out
        for expert_name, type_id in self.TYPE_IDS.items():
            if expert_name == "base":
                continue

            mask = (token_type_ids == type_id).unsqueeze(-1).to(dtype=base_out.dtype)
            if mask.any():
                expert_out = self.experts[expert_name](x)
                output = output + expert_out * mask

        return output


class BackHalfTypedSelfAttention(nn.Module):
    """Wrap self-attention and merge typed keep-mask."""

    def __init__(self, base_self_attn: nn.Module) -> None:
        super().__init__()
        self.base_self_attn = base_self_attn
        self.typed_keep_mask: Optional[torch.Tensor] = None  # [B, L, L], bool

    def set_typed_keep_mask(self, typed_keep_mask: Optional[torch.Tensor]) -> None:
        self.typed_keep_mask = typed_keep_mask

    @staticmethod
    def _to_additive_bias_from_keep(keep_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        bias = torch.zeros_like(keep_mask, dtype=dtype)
        return bias.masked_fill(~keep_mask, torch.finfo(dtype).min)

    def _merge_attention_mask(
        self,
        base_attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.typed_keep_mask is None:
            return base_attention_mask

        typed_keep = self.typed_keep_mask.to(device=hidden_states.device)
        if typed_keep.ndim != 3:
            raise ValueError(f"typed_keep_mask must be [B, L, L], got {tuple(typed_keep.shape)}")

        typed_bias = self._to_additive_bias_from_keep(typed_keep, dtype=hidden_states.dtype).unsqueeze(1)

        if base_attention_mask is None:
            return typed_bias

        base_mask = base_attention_mask.to(device=hidden_states.device)

        if base_mask.ndim == 2:
            # [B, L] key padding mask -> additive [B,1,L,L], then add typed bias.
            key_keep = base_mask > 0
            pad_keep = key_keep.unsqueeze(1).unsqueeze(1).expand(-1, 1, typed_keep.shape[1], -1)
            pad_bias = self._to_additive_bias_from_keep(pad_keep, dtype=hidden_states.dtype)
            return pad_bias + typed_bias

        if base_mask.ndim == 3:
            # [B, L, L] bool/float -> normalize to [B,1,L,L]
            if base_mask.dtype == torch.bool:
                merged_keep = base_mask & typed_keep
                return self._to_additive_bias_from_keep(merged_keep, dtype=hidden_states.dtype).unsqueeze(1)
            return base_mask.unsqueeze(1) + typed_bias

        if base_mask.ndim == 4:
            if base_mask.dtype == torch.bool:
                return base_mask & typed_keep.unsqueeze(1)
            return base_mask + typed_bias

        # Fallback: keep original mask if shape is unknown.
        return base_attention_mask

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")

        if hidden_states is None:
            return self.base_self_attn(*args, **kwargs)

        merged_attention_mask = self._merge_attention_mask(kwargs.get("attention_mask"), hidden_states)
        if merged_attention_mask is not None:
            kwargs["attention_mask"] = merged_attention_mask

        return self.base_self_attn(*args, **kwargs)


class ResidualMLPBlock(nn.Module):
    """Pure MLP residual block for lightweight slot readout fusion."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class SlotMLPFuser(nn.Module):
    """Lightweight slot fuser without any attention operation."""

    def __init__(self, hidden_size: int, num_blocks: int = 2, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_size=hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep fuser input dtype aligned with module params across train/eval/autocast contexts.
        target_dtype = self.out_norm.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


@FRAMEWORK_REGISTRY.register("QwenMyVLA")
class QwenMyVLA(baseframework):
    """MyVLA framework with dynamic/spatial/subtask slot replacement."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.stage_name = str(getattr(self.config.trainer, "stage_name", "")).lower()
        self._preloaded_from_checkpoint = False

        lora_cfg = self.config.framework.get("lora", {})
        self.enable_lora = bool(lora_cfg.get("enable", False))
        default_lora_rank_by_stage = {"stage1": 4, "stage2": 8, "stage3": 4}
        self.lora_rank = int(lora_cfg.get("rank", default_lora_rank_by_stage.get(self.stage_name, 8)))
        self.lora_alpha = int(lora_cfg.get("alpha", min(self.lora_rank, 16)))
        self.lora_dropout = float(lora_cfg.get("dropout", 0.0))
        self.lora_target_modules = self._normalize_lora_value(lora_cfg.get("target_modules", "all-linear"))
        self.lora_exclude_modules = self._normalize_lora_value(lora_cfg.get("exclude_modules", ["base_mlp"]))
        self.lora_modules_to_save = self._normalize_lora_value(lora_cfg.get("modules_to_save", []))
        self.lora_bias = str(lora_cfg.get("bias", "none"))
        self.lora_init_weights = lora_cfg.get("init_lora_weights", "gaussian")

        required_fields = ["lam_stage2", "depth_encoder", "qwen3_embedding"]
        missing = [name for name in required_fields if not hasattr(self.config.framework, name)]
        if missing:
            raise ValueError(f"Missing framework config fields for QwenMyVLA: {missing}")

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self._attach_typed_ffn_moe()
        self._attach_back_half_typed_attention()

        build_encoder = _load_lam_stage2_builder()
        self.dynamic_teacher_encoder = build_encoder(self.config)

        build_depth_encoder = _load_depth_builder()
        self.spatial_teacher_encoder = build_depth_encoder(self.config)

        build_subtask_slot_encoder = _load_subtask_slot_builder()
        self.subtask_slot_encoder = build_subtask_slot_encoder(self.config)

        k_dyn = int(self.config.framework.get("slot_dynamic", {}).get("num_slots", 4))
        k_spa = int(self.config.framework.get("slot_spatial", {}).get("num_slots", 8))
        k_sub = int(self.config.framework.get("slot_subtask", {}).get("num_slots", 4))

        dyn_in = int(self.config.framework.lam_stage2.get("latent_dim", 128))
        depth_cfg = self.config.framework.get("depth_encoder", {})
        depth_out_channels = depth_cfg.get("out_channels", [256, 512, 1024, 1024])
        if depth_out_channels is None:
            depth_out_channels = []
        depth_out_channels = list(depth_out_channels)
        if len(depth_out_channels) > 0:
            spa_in = int(depth_out_channels[-1])
        else:
            spa_in = int(depth_cfg.get("features", 1024))
        sub_in = int(self.subtask_slot_encoder.hidden_size)

        self.dynamic_projector = TokenAwareResampler(in_dim=dyn_in, out_dim=self.hidden_size, num_slots=k_dyn)
        self.spatial_compressor = TokenAwareResampler(in_dim=spa_in, out_dim=self.hidden_size, num_slots=k_spa)
        self.subtask_compressor = TokenAwareResampler(in_dim=sub_in, out_dim=self.hidden_size, num_slots=k_sub)

        self.k_dyn = k_dyn
        self.k_spa = k_spa
        self.k_sub = k_sub

        slot_placeholders = self.config.framework.get("slot_placeholders", {})
        self.dynamic_placeholder = str(slot_placeholders.get("dynamic", "<slot_dynamic>"))
        self.spatial_placeholder = str(slot_placeholders.get("spatial", "<slot_spatial>"))
        self.subtask_placeholder = str(slot_placeholders.get("subtask", "<slot_subtask>"))
        self.action_placeholder = str(slot_placeholders.get("action", "<slot_action>"))
        self.slot_boundary_tokens = self.config.framework.get(
            "slot_boundary_tokens",
            {
                "dynamic_start": "<SOSD>",
                "dynamic_end": "<EOSD>",
                "spatial_start": "<SOSS>",
                "spatial_end": "<EOSS>",
                "subtask_start": "<SOST>",
                "subtask_end": "<EOST>",
                "action_start": "<SOSA>",
                "action_end": "<EOSA>",
            },
        )
        action_token_cfg = self.config.framework.get("action_tokens", {})
        self.enable_action_tokens = bool(action_token_cfg.get("enable", False))
        self.action_placeholder = str(action_token_cfg.get("placeholder", self.action_placeholder))
        self.num_action_tokens = int(action_token_cfg.get("num_tokens", 0)) if self.enable_action_tokens else 0

        slot_mask_cfg = self.config.framework.get("slot_mask", {})
        self.enable_slot_mask = bool(slot_mask_cfg.get("enable", False))
        self.slot_mask_apply_in_eval = bool(slot_mask_cfg.get("apply_in_eval", False))
        self.inner_mask_ratios = {
            "dynamic": float(slot_mask_cfg.get("inner", {}).get("dynamic", 0.0)),
            "spatial": float(slot_mask_cfg.get("inner", {}).get("spatial", 0.0)),
            "subtask": float(slot_mask_cfg.get("inner", {}).get("subtask", 0.0)),
        }
        self.outside_num_masked_probs = slot_mask_cfg.get("outside", {}).get("num_masked_probs", [1.0, 0.0, 0.0, 0.0])
        self.outside_num_masked_probs = self._normalize_outside_probs(
            self.outside_num_masked_probs,
            "slot_mask.outside.num_masked_probs",
        )

        slot_mask_curriculum_cfg = slot_mask_cfg.get("curriculum", {})
        self.enable_slot_mask_curriculum = bool(slot_mask_curriculum_cfg.get("enable", False))
        self.slot_mask_curriculum_progress_start = float(slot_mask_curriculum_cfg.get("progress_start", 0.0))
        self.slot_mask_curriculum_progress_end = float(slot_mask_curriculum_cfg.get("progress_end", 1.0))
        self.slot_mask_inner_start = {
            "dynamic": float(slot_mask_curriculum_cfg.get("inner_start", {}).get("dynamic", self.inner_mask_ratios["dynamic"])),
            "spatial": float(slot_mask_curriculum_cfg.get("inner_start", {}).get("spatial", self.inner_mask_ratios["spatial"])),
            "subtask": float(slot_mask_curriculum_cfg.get("inner_start", {}).get("subtask", self.inner_mask_ratios["subtask"])),
        }
        self.slot_mask_inner_end = {
            "dynamic": float(slot_mask_curriculum_cfg.get("inner_end", {}).get("dynamic", self.inner_mask_ratios["dynamic"])),
            "spatial": float(slot_mask_curriculum_cfg.get("inner_end", {}).get("spatial", self.inner_mask_ratios["spatial"])),
            "subtask": float(slot_mask_curriculum_cfg.get("inner_end", {}).get("subtask", self.inner_mask_ratios["subtask"])),
        }
        self.slot_mask_outside_start_probs = self._normalize_outside_probs(
            slot_mask_curriculum_cfg.get("outside_start_num_masked_probs", self.outside_num_masked_probs),
            "slot_mask.curriculum.outside_start_num_masked_probs",
        )
        self.slot_mask_outside_end_probs = self._normalize_outside_probs(
            slot_mask_curriculum_cfg.get("outside_end_num_masked_probs", self.outside_num_masked_probs),
            "slot_mask.curriculum.outside_end_num_masked_probs",
        )

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self.dynamic_token_id = self._resolve_single_token_id(tokenizer, self.dynamic_placeholder, "dynamic")
        self.spatial_token_id = self._resolve_single_token_id(tokenizer, self.spatial_placeholder, "spatial")
        self.subtask_token_id = self._resolve_single_token_id(tokenizer, self.subtask_placeholder, "subtask")
        self.action_token_id = None
        self.action_token_queries = None
        if self.enable_action_tokens and self.num_action_tokens > 0:
            self.action_token_id = self._resolve_single_token_id(tokenizer, self.action_placeholder, "action")
            self.action_token_queries = nn.Parameter(torch.randn(self.num_action_tokens, self.hidden_size) * 0.02)
        self.fast_action_token_ids = self._resolve_fast_action_token_ids(tokenizer)
        self.image_token_id = IMAGE_TOKEN_INDEX
        self._moe_token_type_ids = None
        progress_ctrl_cfg = self.config.framework.get("progress_control", {})
        self.enable_progress_step_mapping = bool(progress_ctrl_cfg.get("enable_step_mapping", False))
        self.enable_progress_window_mapping = bool(progress_ctrl_cfg.get("enable_window_mapping", False))
        self.progress_step_start = int(progress_ctrl_cfg.get("step_start", 0))
        self.progress_step_end = progress_ctrl_cfg.get("step_end", None)
        if self.progress_step_end is not None:
            self.progress_step_end = int(self.progress_step_end)
        self.progress_step_windows = self._normalize_step_windows(progress_ctrl_cfg.get("step_windows", []))
        self.progress_window_active_flags = self._normalize_window_flags(
            progress_ctrl_cfg.get("window_active_flags", []),
            expected_length=len(self.progress_step_windows),
        )
        self.attention_visibility_rules = self._resolve_attention_visibility_rules()
        self.enable_back_half_typed_attention = bool(
            self.config.framework.get("attention_mask", {}).get("enable_apply", True)
        )
        if self.enable_back_half_typed_attention and self._is_flash_attention_2_active():
            logger.warning(
                "Disable back-half typed attention mask because flash_attention_2 does not support "
                "the custom [B, L, L] typed visibility mask path reliably."
            )
            self.enable_back_half_typed_attention = False

        slot_fuse_cfg = self.config.framework.get("slot_fuse", {})
        fuse_blocks = int(slot_fuse_cfg.get("num_blocks", 2))
        fuse_mlp_ratio = float(slot_fuse_cfg.get("mlp_ratio", 2.0))
        self.dynamic_slot_fuser = SlotMLPFuser(
            hidden_size=self.hidden_size,
            num_blocks=fuse_blocks,
            mlp_ratio=fuse_mlp_ratio,
        )
        self.spatial_slot_fuser = SlotMLPFuser(
            hidden_size=self.hidden_size,
            num_blocks=fuse_blocks,
            mlp_ratio=fuse_mlp_ratio,
        )
        self.subtask_slot_fuser = SlotMLPFuser(
            hidden_size=self.hidden_size,
            num_blocks=fuse_blocks,
            mlp_ratio=fuse_mlp_ratio,
        )

        align_cfg = self.config.framework.get("slot_align_loss", {})
        self.enable_slot_align_loss = bool(align_cfg.get("enable", True))
        self.slot_align_loss_weights = {
            "dynamic": float(align_cfg.get("dynamic", 1.0)),
            "spatial": float(align_cfg.get("spatial", 1.0)),
            "subtask": float(align_cfg.get("subtask", 1.0)),
        }
        self.slot_align_loss_mse_ratio = float(align_cfg.get("mse_ratio", 1.0))
        self.slot_align_loss_cos_ratio = float(align_cfg.get("cos_ratio", 1.0))
        self.slot_align_masked_target_prediction = bool(align_cfg.get("masked_target_prediction", False))
        self.slot_align_masked_only = bool(align_cfg.get("masked_only", True))
        self.slot_align_unmasked_weight = float(align_cfg.get("unmasked_weight", 0.0))
        self.slot_align_fallback_to_all_when_no_masked = bool(
            align_cfg.get("fallback_to_all_when_no_masked", True)
        )

        self.return_debug_tensors = bool(self.config.framework.get("return_debug_tensors", False))
        self.strict_loss_checks = bool(self.config.framework.get("strict_loss_checks", True))

        action_cfg = self.config.framework.get("action_model", {})
        self.action_dim = int(action_cfg.get("action_dim", 7))
        self.num_actions_chunk = int(action_cfg.get("num_actions_chunk", 16))
        self.action_predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 3),
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_actions_chunk * self.action_dim),
        )

        continuous_cfg = self.config.framework.get("continuous_fm_loss", {})
        self.continuous_fm_loss_weight = float(continuous_cfg.get("weight", 1.0))
        self.continuous_fm_loss_detach = bool(continuous_cfg.get("detach", False))
        self.continuous_fm_objective = str(continuous_cfg.get("objective", "fm")).lower()
        self.continuous_fm_noise_scale = float(continuous_cfg.get("noise_scale", 1.0))
        self.continuous_fm_t_eps = float(continuous_cfg.get("t_eps", 1e-3))
        self.continuous_fm_loss_type = str(continuous_cfg.get("loss_type", "mse")).lower()

        loss_schedule_cfg = self.config.framework.get("loss_weight_schedule", {})
        self.enable_loss_weight_schedule = bool(loss_schedule_cfg.get("enable", False))
        self.loss_weight_schedule_progress_start = float(loss_schedule_cfg.get("progress_start", 0.0))
        self.loss_weight_schedule_progress_end = float(loss_schedule_cfg.get("progress_end", 1.0))
        self.slot_align_weight_start = {
            "dynamic": float(loss_schedule_cfg.get("slot_align_start", {}).get("dynamic", self.slot_align_loss_weights["dynamic"])),
            "spatial": float(loss_schedule_cfg.get("slot_align_start", {}).get("spatial", self.slot_align_loss_weights["spatial"])),
            "subtask": float(loss_schedule_cfg.get("slot_align_start", {}).get("subtask", self.slot_align_loss_weights["subtask"])),
        }
        self.slot_align_weight_end = {
            "dynamic": float(loss_schedule_cfg.get("slot_align_end", {}).get("dynamic", self.slot_align_loss_weights["dynamic"])),
            "spatial": float(loss_schedule_cfg.get("slot_align_end", {}).get("spatial", self.slot_align_loss_weights["spatial"])),
            "subtask": float(loss_schedule_cfg.get("slot_align_end", {}).get("subtask", self.slot_align_loss_weights["subtask"])),
        }
        self.continuous_fm_loss_weight_start = float(
            loss_schedule_cfg.get("continuous_fm_start", self.continuous_fm_loss_weight)
        )
        self.continuous_fm_loss_weight_end = float(
            loss_schedule_cfg.get("continuous_fm_end", self.continuous_fm_loss_weight)
        )

        stage3_cfg = self.config.framework.get("stage3_action_header", {})
        self.enable_stage3_action_header = bool(stage3_cfg.get("enable", False))
        self.stage3_condition_model_type = str(stage3_cfg.get("model_type", "DiT-B"))
        stage3_model_default_dims = {"DiT-S": 384, "DiT-B": 768, "DiT-L": 1024}
        default_condition_hidden_dim = stage3_model_default_dims.get(self.stage3_condition_model_type, 768)
        self.stage3_condition_hidden_dim = int(stage3_cfg.get("condition_hidden_dim", default_condition_hidden_dim))
        self.stage3_condition_num_queries = int(stage3_cfg.get("condition_num_queries", 64))
        self.stage3_condition_num_heads = int(
            stage3_cfg.get("condition_num_heads", max(self.stage3_condition_hidden_dim // 64, 1))
        )
        self.stage3_dino_backbone = str(stage3_cfg.get("dino_backbone", "dinov2_vits14"))
        self.stage3_action_query_num = int(
            stage3_cfg.get(
                "action_query_num",
                action_cfg.get("action_query_num", self.k_sub + self.k_dyn + self.k_spa),
            )
        )
        self.stage3_action_head = None
        self.stage3_dino_encoder = None
        self.stage3_dino_projector = None
        self.stage3_condition_fuser = None
        if self.enable_stage3_action_header:
            self.stage3_dino_encoder = get_dino_model(backone_name=self.stage3_dino_backbone)
            self.stage3_dino_encoder.eval()
            for param in self.stage3_dino_encoder.parameters():
                param.requires_grad = False

            self.stage3_dino_projector = nn.Linear(self.stage3_dino_encoder.num_channels, self.hidden_size)
            self.stage3_condition_fuser = LayerwiseQFormer(
                input_hidden_dim=self.hidden_size,
                output_hidden_dim=self.stage3_condition_hidden_dim,
                num_query_tokens=self.stage3_condition_num_queries,
                num_layers=self._resolve_qwen_num_hidden_states(),
                num_heads=self.stage3_condition_num_heads,
                config=self.config,
            )
            self.stage3_action_head = DiTActionModel(
                action_hidden_dim=self.stage3_condition_hidden_dim,
                model_type=self.stage3_condition_model_type,
                in_channels=self.action_dim,
                future_action_window_size=max(self.num_actions_chunk - 1, 0),
                past_action_window_size=0,
            )

        fast_cfg = self.config.framework.get("fast_action_loss", {})
        self.enable_fast_action_loss = bool(fast_cfg.get("enable", False))
        self.fast_action_loss_weight = float(fast_cfg.get("weight", 1.0))
        self.fast_action_loss_weight_start = float(loss_schedule_cfg.get("fast_action_start", self.fast_action_loss_weight))
        self.fast_action_loss_weight_end = float(loss_schedule_cfg.get("fast_action_end", self.fast_action_loss_weight))
        self.fast_action_loss_detach = bool(fast_cfg.get("detach", False))
        self.fast_action_model = None
        if self.enable_fast_action_loss:
            self.fast_action_model = get_fast_action_model(config=self.config)
            self._configure_fast_action_tokenizer()

        self._maybe_load_pretrained_and_attach_lora()

    def _attach_typed_ffn_moe(self) -> None:
        """Replace all Qwen FFN blocks with typed residual expert wrappers."""
        text_model, layers = self._resolve_text_backbone_and_layers()
        if layers is None:
            raise AttributeError("Qwen text backbone does not expose `layers` for FFN wrapping.")

        self.moe_start_layer = 0
        self.moe_wrapped_layers: List[int] = []
        expert_param_ids = []

        for layer_idx in range(self.moe_start_layer, len(layers)):
            layer = layers[layer_idx]
            if not hasattr(layer, "mlp"):
                raise AttributeError(f"Layer {layer_idx} does not have an mlp module.")
            if isinstance(layer.mlp, TypedResidualFFN):
                continue
            layer.mlp = TypedResidualFFN(layer.mlp, freeze_base_mlp=True)
            self.moe_wrapped_layers.append(layer_idx)

            # Sanity check: each layer should own an independent expert module.
            dyn_first_param = next(layer.mlp.experts["dynamic"].parameters(), None)
            if dyn_first_param is not None:
                expert_param_ids.append(id(dyn_first_param))

        if len(expert_param_ids) != len(set(expert_param_ids)):
            raise RuntimeError("Detected shared FFN experts across layers; expected per-layer independent experts.")

    def _set_typed_ffn_token_type_ids(self, token_type_ids: Optional[torch.Tensor]) -> None:
        """Broadcast token_type_ids to all wrapped FFN layers."""
        self._moe_token_type_ids = token_type_ids
        text_model, layers = self._resolve_text_backbone_and_layers()
        if layers is None:
            return

        for layer_idx in self.moe_wrapped_layers:
            layer = layers[layer_idx]
            assert isinstance(layer.mlp, TypedResidualFFN)
            layer.mlp.set_token_type_ids(token_type_ids)

    def _attach_back_half_typed_attention(self) -> None:
        """Wrap self-attention in all layers to accept typed keep-mask."""
        text_model, layers = self._resolve_text_backbone_and_layers()
        if layers is None:
            raise AttributeError("Qwen text backbone does not expose `layers` for attention wrapping.")

        self.attn_start_layer = 0
        self.attn_wrapped_layers: List[int] = []

        for layer_idx in range(self.attn_start_layer, len(layers)):
            layer = layers[layer_idx]
            if not hasattr(layer, "self_attn"):
                raise AttributeError(f"Layer {layer_idx} does not have a self_attn module.")
            if isinstance(layer.self_attn, BackHalfTypedSelfAttention):
                continue
            layer.self_attn = BackHalfTypedSelfAttention(layer.self_attn)
            self.attn_wrapped_layers.append(layer_idx)

    def _set_back_half_typed_attention_mask(self, typed_keep_mask: Optional[torch.Tensor]) -> None:
        text_model, layers = self._resolve_text_backbone_and_layers()
        if layers is None:
            return

        for layer_idx in getattr(self, "attn_wrapped_layers", []):
            layer = layers[layer_idx]
            if isinstance(layer.self_attn, BackHalfTypedSelfAttention):
                layer.self_attn.set_typed_keep_mask(typed_keep_mask)

    @staticmethod
    def _load_state_dict_from_checkpoint(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint `{checkpoint_path}` does not contain a state dict.")
        return checkpoint

    @staticmethod
    def _state_dict_looks_like_lora(state_dict: Dict[str, torch.Tensor]) -> bool:
        for key in state_dict.keys():
            if "lora_A" in key or "lora_B" in key or "base_model.model" in key:
                return True
        return False

    @staticmethod
    def _normalize_lora_value(value):
        if hasattr(value, "_cfg"):
            value = value._cfg
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        return value

    def _resolve_lora_modules_to_save(self) -> List[str]:
        modules_to_save = self._normalize_lora_value(getattr(self, "lora_modules_to_save", []))
        if modules_to_save is None:
            return []
        if isinstance(modules_to_save, str):
            modules_to_save = [modules_to_save]
        if not isinstance(modules_to_save, (list, tuple, set)):
            raise TypeError(f"Unsupported framework.lora.modules_to_save type: {type(modules_to_save)!r}")
        return [str(name) for name in modules_to_save if str(name)]

    def _configure_fast_action_tokenizer(self) -> None:
        if self.fast_action_model is None:
            return
        fast_tokenizer = getattr(self.fast_action_model, "fast_tokenizer", None)
        if fast_tokenizer is None:
            return

        fast_tokenizer.time_horizon = self.num_actions_chunk
        fast_tokenizer.action_dim = self.action_dim

    @staticmethod
    def _module_name_matches_exclude(module_name: str, exclude_name: str) -> bool:
        if not exclude_name:
            return False
        if module_name == exclude_name:
            return True
        if module_name.startswith(f"{exclude_name}."):
            return True
        return exclude_name in module_name.split(".")

    def _resolve_lora_target_module_names(self) -> List[str]:
        root_model = getattr(self.qwen_vl_interface, "model", None)
        if root_model is None:
            raise AttributeError("Unable to resolve Qwen model for LoRA target discovery.")

        target_modules_value = self._normalize_lora_value(getattr(self, "lora_target_modules", "all-linear"))
        effective_exclude_modules = self._resolve_lora_exclude_module_names()
        if isinstance(target_modules_value, (list, tuple, set)):
            target_suffixes = [str(name) for name in target_modules_value if str(name)]
            if not target_suffixes:
                raise ValueError("framework.lora.target_modules is empty after normalization.")
        elif isinstance(target_modules_value, str) and target_modules_value == "all-linear":
            target_suffixes = None
        elif isinstance(target_modules_value, str):
            target_suffixes = [target_modules_value]
        else:
            raise TypeError(
                f"Unsupported framework.lora.target_modules type: {type(target_modules_value)!r}"
            )

        target_module_names: List[str] = []
        for module_name, module in root_model.named_modules():
            if not module_name or not isinstance(module, nn.Linear):
                continue
            if any(self._module_name_matches_exclude(module_name, exclude_name) for exclude_name in effective_exclude_modules):
                continue
            if target_suffixes is None:
                target_module_names.append(module_name)
                continue

            if any(module_name == suffix or module_name.endswith(f".{suffix}") for suffix in target_suffixes):
                target_module_names.append(module_name)

        if not target_module_names:
            raise ValueError(
                "No LoRA target modules were resolved. Check framework.lora.target_modules and exclude_modules."
            )

        return target_module_names

    def _resolve_lora_exclude_module_names(self) -> List[str]:
        exclude_modules = self._normalize_lora_value(getattr(self, "lora_exclude_modules", ["base_mlp"]))
        if isinstance(exclude_modules, str):
            exclude_modules = [exclude_modules]
        exclude_modules = [str(name) for name in exclude_modules if str(name)]
        default_excludes = ["base_mlp", "lm_head"]
        for default_exclude in default_excludes:
            if default_exclude not in exclude_modules:
                exclude_modules.append(default_exclude)
        return exclude_modules

    def _attach_qwen_lora(self) -> None:
        if not self.enable_lora:
            return
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("PEFT is required for LoRA training but is not installed.")

        effective_exclude_modules = self._resolve_lora_exclude_module_names()
        resolved_target_modules = self._resolve_lora_target_module_names()
        modules_to_save = self._resolve_lora_modules_to_save()
        logger.info(
            "Resolved %d LoRA target modules for Qwen backbone (excluding %s)",
            len(resolved_target_modules),
            effective_exclude_modules,
        )
        if modules_to_save:
            logger.info("LoRA modules_to_save for Qwen backbone: %s", modules_to_save)

        lora_config = LoraConfig(
            r=getattr(self, "lora_rank", 8),
            lora_alpha=getattr(self, "lora_alpha", 8),
            lora_dropout=getattr(self, "lora_dropout", 0.0),
            target_modules=resolved_target_modules,
            exclude_modules=effective_exclude_modules,
            modules_to_save=modules_to_save or None,
            bias=getattr(self, "lora_bias", "none"),
            init_lora_weights=getattr(self, "lora_init_weights", "gaussian"),
        )
        self.qwen_vl_interface.model = get_peft_model(self.qwen_vl_interface.model, lora_config)
        if hasattr(self.qwen_vl_interface.model, "print_trainable_parameters"):
            self.qwen_vl_interface.model.print_trainable_parameters()

    def _maybe_load_pretrained_and_attach_lora(self) -> None:
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        if not self.enable_lora or not pretrained_checkpoint:
            if self.enable_lora:
                self._attach_qwen_lora()
            return

        checkpoint_path = Path(pretrained_checkpoint)
        if not checkpoint_path.exists():
            self._attach_qwen_lora()
            return

        state_dict = self._load_state_dict_from_checkpoint(checkpoint_path)
        checkpoint_is_lora = self._state_dict_looks_like_lora(state_dict)

        if checkpoint_is_lora:
            self._attach_qwen_lora()
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=False)
            self._attach_qwen_lora()

        self._preloaded_from_checkpoint = True

    def _resolve_text_backbone_and_layers(self):
        """Resolve the text backbone module and its transformer layers across Qwen variants."""
        root_model = getattr(self.qwen_vl_interface, "model", None)
        candidates = [
            getattr(root_model, "model", None),
            getattr(getattr(root_model, "model", None), "model", None),
            getattr(getattr(root_model, "model", None), "language_model", None),
            getattr(getattr(getattr(root_model, "model", None), "language_model", None), "model", None),
            getattr(root_model, "language_model", None),
            getattr(getattr(root_model, "language_model", None), "model", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            layers = getattr(candidate, "layers", None)
            if layers is not None:
                return candidate, layers
        return None, None

    def _is_flash_attention_2_active(self) -> bool:
        """Detect whether the loaded Qwen backend is using flash_attention_2."""
        root_model = getattr(self.qwen_vl_interface, "model", None)
        if root_model is None:
            return False

        cfg_candidates = [
            getattr(root_model, "config", None),
            getattr(getattr(root_model, "model", None), "config", None),
            getattr(getattr(root_model, "language_model", None), "config", None),
            getattr(getattr(getattr(root_model, "model", None), "language_model", None), "config", None),
            getattr(getattr(root_model, "config", None), "text_config", None),
        ]

        for cfg in cfg_candidates:
            if cfg is None:
                continue
            attn_impl = getattr(cfg, "_attn_implementation", None)
            if isinstance(attn_impl, str) and attn_impl == "flash_attention_2":
                return True

        return False

    def _resolve_input_embeddings_layer(self) -> nn.Module:
        """Get Qwen input embedding layer with fallback paths for different model wrappers."""
        root_model = getattr(self.qwen_vl_interface, "model", None)
        candidates = [
            getattr(root_model, "model", None),
            getattr(getattr(root_model, "model", None), "model", None),
            getattr(getattr(root_model, "model", None), "language_model", None),
            getattr(getattr(getattr(root_model, "model", None), "language_model", None), "model", None),
            getattr(root_model, "language_model", None),
            getattr(getattr(root_model, "language_model", None), "model", None),
            root_model,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            get_embed = getattr(candidate, "get_input_embeddings", None)
            if callable(get_embed):
                return get_embed()
        raise AttributeError("Unable to resolve Qwen input embeddings layer from qwen_vl_interface.model")

    @staticmethod
    def _resolve_single_token_id(tokenizer, token: str, token_name: str) -> int:
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"Placeholder '{token}' for {token_name} must map to exactly one tokenizer id, got {ids}"
            )
        return int(ids[0])

    @staticmethod
    def _resolve_fast_action_token_ids(tokenizer) -> set[int]:
        """Resolve `<robot_action_*>` token ids if tokenizer has them.

        Returns empty set when vocab APIs are not available or token ids are absent.
        """
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            return set()

        try:
            vocab = get_vocab()
        except Exception:
            return set()

        if not isinstance(vocab, dict):
            return set()

        action_ids: set[int] = set()
        for token, token_id in vocab.items():
            if isinstance(token, str) and token.startswith("<robot_action_") and token.endswith(">"):
                action_ids.add(int(token_id))
        return action_ids

    @staticmethod
    def _normalize_outside_probs(prob_list, field_name: str) -> List[float]:
        if len(prob_list) != 4:
            raise ValueError(f"{field_name} must have 4 entries for masking 0/1/2/3 slot types")
        prob_sum = float(sum(prob_list))
        if prob_sum <= 0:
            raise ValueError(f"{field_name} must have positive sum")
        return [float(p) / prob_sum for p in prob_list]

    @staticmethod
    def _clamp_progress(progress: float, start: float, end: float) -> float:
        if end <= start:
            return 1.0
        ratio = (progress - start) / (end - start)
        return max(0.0, min(1.0, float(ratio)))

    @staticmethod
    def _lerp(start: float, end: float, t: float) -> float:
        return float(start) + (float(end) - float(start)) * float(t)

    def _resolve_training_progress(self, kwargs: Dict) -> float:
        return self._resolve_training_progress_info(kwargs)["progress"]

    def _resolve_training_progress_info(self, kwargs: Dict) -> Dict[str, float | int | bool]:
        global_step = kwargs.get("global_step")
        max_train_steps = kwargs.get("max_train_steps")
        if global_step is None or max_train_steps is None:
            return {
                "progress": 0.0,
                "global_step": 0,
                "max_train_steps": 1,
                "step_start": 0,
                "step_end": 1,
                "uses_step_mapping": False,
                "uses_window_mapping": False,
                "is_before_start": False,
                "is_after_end": False,
                "is_skipped": False,
                "is_finished": False,
                "current_window_index": -1,
                "current_window_enabled": False,
            }

        global_step = int(global_step)
        max_train_steps = max(1, int(max_train_steps))
        if self.enable_progress_window_mapping and self.progress_step_windows:
            windows = self.progress_step_windows
            flags = self.progress_window_active_flags or [True] * len(windows)
            active_windows = [(start, end) for (start, end), enabled in zip(windows, flags) if enabled]
            if not active_windows:
                return {
                    "progress": 0.0,
                    "global_step": global_step,
                    "max_train_steps": max_train_steps,
                    "step_start": windows[0][0],
                    "step_end": windows[-1][1],
                    "uses_step_mapping": False,
                    "uses_window_mapping": True,
                    "is_before_start": True,
                    "is_after_end": False,
                    "is_skipped": True,
                    "is_finished": False,
                    "current_window_index": -1,
                    "current_window_enabled": False,
                }

            active_start = active_windows[0][0]
            active_end = active_windows[-1][1]
            active_total = float(sum(max(1, end - start) for start, end in active_windows))
            if active_total <= 0:
                active_total = 1.0

            if global_step < active_start:
                return {
                    "progress": 0.0,
                    "global_step": global_step,
                    "max_train_steps": max_train_steps,
                    "step_start": windows[0][0],
                    "step_end": windows[-1][1],
                    "uses_step_mapping": False,
                    "uses_window_mapping": True,
                    "is_before_start": True,
                    "is_after_end": False,
                    "is_skipped": True,
                    "is_finished": False,
                    "current_window_index": -1,
                    "current_window_enabled": False,
                }

            accumulated = 0.0
            for window_index, ((start, end), enabled) in enumerate(zip(windows, flags)):
                window_length = max(1, end - start)
                if not enabled:
                    if start <= global_step < end:
                        return {
                            "progress": accumulated / active_total,
                            "global_step": global_step,
                            "max_train_steps": max_train_steps,
                            "step_start": windows[0][0],
                            "step_end": windows[-1][1],
                            "uses_step_mapping": False,
                            "uses_window_mapping": True,
                            "is_before_start": False,
                            "is_after_end": False,
                            "is_skipped": True,
                            "is_finished": False,
                            "current_window_index": window_index,
                            "current_window_enabled": False,
                        }
                    continue

                if global_step < start:
                    return {
                        "progress": accumulated / active_total,
                        "global_step": global_step,
                        "max_train_steps": max_train_steps,
                        "step_start": windows[0][0],
                        "step_end": windows[-1][1],
                        "uses_step_mapping": False,
                        "uses_window_mapping": True,
                        "is_before_start": False,
                        "is_after_end": False,
                        "is_skipped": True,
                        "is_finished": False,
                        "current_window_index": window_index,
                        "current_window_enabled": True,
                    }

                if global_step < end:
                    return {
                        "progress": (accumulated + float(global_step - start)) / active_total,
                        "global_step": global_step,
                        "max_train_steps": max_train_steps,
                        "step_start": windows[0][0],
                        "step_end": windows[-1][1],
                        "uses_step_mapping": False,
                        "uses_window_mapping": True,
                        "is_before_start": False,
                        "is_after_end": False,
                        "is_skipped": False,
                        "is_finished": False,
                        "current_window_index": window_index,
                        "current_window_enabled": True,
                    }

                accumulated += float(window_length)

            return {
                "progress": 1.0,
                "global_step": global_step,
                "max_train_steps": max_train_steps,
                "step_start": windows[0][0],
                "step_end": windows[-1][1],
                "uses_step_mapping": False,
                "uses_window_mapping": True,
                "is_before_start": False,
                "is_after_end": True,
                "is_skipped": False,
                "is_finished": True,
                "current_window_index": len(windows) - 1,
                "current_window_enabled": flags[-1],
            }

        if self.enable_progress_step_mapping:
            step_start = int(self.progress_step_start)
            step_end = int(self.progress_step_end) if self.progress_step_end is not None else max_train_steps
        else:
            step_start = 0
            step_end = max_train_steps

        if step_end <= step_start:
            step_end = step_start + 1

        progress = max(0.0, min(1.0, float(global_step - step_start) / float(step_end - step_start)))
        return {
            "progress": float(progress),
            "global_step": global_step,
            "max_train_steps": max_train_steps,
            "step_start": step_start,
            "step_end": step_end,
            "uses_step_mapping": bool(self.enable_progress_step_mapping),
            "uses_window_mapping": False,
            "is_before_start": bool(global_step < step_start),
            "is_after_end": bool(global_step >= step_end),
            "is_skipped": bool(global_step < step_start),
            "is_finished": bool(global_step >= step_end),
            "current_window_index": -1,
            "current_window_enabled": False,
        }

    @staticmethod
    def _normalize_step_windows(step_windows) -> List[List[int]]:
        if not step_windows:
            return []

        normalized = []
        for window in step_windows:
            if isinstance(window, (str, bytes)):
                raise ValueError(f"progress_control.step_windows must be a list of [start, end] pairs, got: {window}")
            try:
                start, end = window[0], window[1]
            except Exception as exc:
                raise ValueError(
                    f"progress_control.step_windows must be a list of [start, end] pairs, got: {window}"
                ) from exc

            try:
                if len(window) != 2:
                    raise ValueError
            except Exception as exc:
                raise ValueError(
                    f"progress_control.step_windows must be a list of [start, end] pairs, got: {window}"
                ) from exc

            start, end = int(start), int(end)
            if end <= start:
                raise ValueError(f"Invalid progress window [{start}, {end}]; end must be greater than start")
            normalized.append([start, end])
        return normalized

    @staticmethod
    def _normalize_window_flags(window_flags, expected_length: int) -> List[bool]:
        if expected_length <= 0:
            return []
        if not window_flags:
            return [True] * expected_length

        flags = [bool(flag) for flag in window_flags]
        if len(flags) != expected_length:
            raise ValueError(
                f"progress_control.window_active_flags length {len(flags)} does not match step_windows length {expected_length}"
            )
        return flags

    @staticmethod
    def _resolve_schedule_flags(progress: float, start: float, end: float, enable: bool) -> Dict[str, float | bool]:
        if not enable:
            return {"enable": False, "skipped": True, "finished": False, "t": 0.0}

        if end <= start:
            finished = progress >= end
            return {
                "enable": True,
                "skipped": bool(progress < start),
                "finished": bool(finished),
                "t": 1.0 if finished else 0.0,
            }

        if progress < start:
            return {"enable": True, "skipped": True, "finished": False, "t": 0.0}
        if progress >= end:
            return {"enable": True, "skipped": False, "finished": True, "t": 1.0}
        return {
            "enable": True,
            "skipped": False,
            "finished": False,
            "t": max(0.0, min(1.0, float((progress - start) / (end - start)))),
        }

    def _resolve_slot_mask_hparams(self, progress: float) -> tuple[Dict[str, float], List[float]]:
        if not self.enable_slot_mask_curriculum:
            return self.inner_mask_ratios, self.outside_num_masked_probs

        t = self._clamp_progress(
            progress,
            self.slot_mask_curriculum_progress_start,
            self.slot_mask_curriculum_progress_end,
        )
        inner = {
            name: self._lerp(self.slot_mask_inner_start[name], self.slot_mask_inner_end[name], t)
            for name in ["dynamic", "spatial", "subtask"]
        }
        outside = [
            self._lerp(self.slot_mask_outside_start_probs[i], self.slot_mask_outside_end_probs[i], t)
            for i in range(4)
        ]
        outside = self._normalize_outside_probs(outside, "slot_mask curriculum interpolated probs")
        return inner, outside

    def _resolve_loss_weights(self, progress: float) -> Dict[str, float]:
        if not self.enable_loss_weight_schedule:
            return {
                "slot_dynamic": self.slot_align_loss_weights["dynamic"],
                "slot_spatial": self.slot_align_loss_weights["spatial"],
                "slot_subtask": self.slot_align_loss_weights["subtask"],
                "continuous": self.continuous_fm_loss_weight,
                "fast": self.fast_action_loss_weight,
            }

        t = self._clamp_progress(
            progress,
            self.loss_weight_schedule_progress_start,
            self.loss_weight_schedule_progress_end,
        )
        return {
            "slot_dynamic": self._lerp(self.slot_align_weight_start["dynamic"], self.slot_align_weight_end["dynamic"], t),
            "slot_spatial": self._lerp(self.slot_align_weight_start["spatial"], self.slot_align_weight_end["spatial"], t),
            "slot_subtask": self._lerp(self.slot_align_weight_start["subtask"], self.slot_align_weight_end["subtask"], t),
            "continuous": self._lerp(self.continuous_fm_loss_weight_start, self.continuous_fm_loss_weight_end, t),
            "fast": self._lerp(self.fast_action_loss_weight_start, self.fast_action_loss_weight_end, t),
        }

    def _resolve_attention_visibility_rules(self) -> Dict[str, Dict[str, bool]]:
        """Resolve stage-aware typed attention visibility rules.

        Priority order:
        1. attention_mask.stage_visibility[stage_name]
        2. attention_mask.visibility
        3. built-in default rules
        """
        attention_cfg = self.config.framework.get("attention_mask", {})
        stage_visibility = attention_cfg.get("stage_visibility", {})
        if self.stage_name and isinstance(stage_visibility, Mapping) and self.stage_name in stage_visibility:
            return stage_visibility[self.stage_name]

        visibility = attention_cfg.get("visibility")
        if visibility is not None:
            return visibility

        return self._default_attention_visibility_rules()

    @staticmethod
    def _gather_positions(input_ids: torch.Tensor, token_id: int, expected_count: int) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        positions = torch.zeros((batch_size, expected_count), dtype=torch.long, device=input_ids.device)
        for b in range(batch_size):
            pos = torch.where(input_ids[b] == token_id)[0]
            if pos.numel() < expected_count:
                raise ValueError(
                    f"Placeholder token id {token_id} count is {int(pos.numel())}, expected at least {expected_count}"
                )
            positions[b] = pos[-expected_count:]
        return positions

    @staticmethod
    def _flatten_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim == 4 and mask.ndim == 3:
            b, v, t, c = tokens.shape
            return tokens.reshape(b, v * t, c), mask.reshape(b, v * t)
        if tokens.ndim == 3 and mask.ndim == 2:
            return tokens, mask
        raise ValueError(f"Unsupported token/mask shapes: {tuple(tokens.shape)}, {tuple(mask.shape)}")

    def _build_dynamic_slots(self, lam_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_q = lam_outputs["z_q"]
        if z_q.ndim == 4:
            b, t, k, d = z_q.shape
            tokens = z_q.reshape(b, t * k, d)
        elif z_q.ndim == 3:
            tokens = z_q
        else:
            raise ValueError(f"Unsupported z_q shape: {tuple(z_q.shape)}")

        mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        return self.dynamic_projector(tokens, mask)

    def _build_spatial_slots(self, depth_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = depth_outputs["encoder_tokens"]
        mask = depth_outputs["token_mask"]
        flat_tokens, flat_mask = self._flatten_tokens(tokens, mask)
        return self.spatial_compressor(flat_tokens, flat_mask)

    def _build_subtask_slots(self, subtask_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "subtask_slots" in subtask_outputs:
            # Some subtask encoders return native embedding-dim slots (e.g. 1024 for Qwen3-Embedding).
            # Normalize all subtask slots to Qwen hidden size via the shared compressor.
            subtask_slots = subtask_outputs["subtask_slots"]
            if subtask_slots.ndim != 3:
                raise ValueError(f"Expected subtask_slots [B, K, C], got {tuple(subtask_slots.shape)}")
            if subtask_slots.shape[-1] == self.hidden_size and subtask_slots.shape[1] == self.k_sub:
                return subtask_slots
            subtask_mask = torch.ones(
                subtask_slots.shape[:2],
                device=subtask_slots.device,
                dtype=torch.bool,
            )
            return self.subtask_compressor(subtask_slots, subtask_mask)
        tokens = subtask_outputs["text_tokens"]
        mask = subtask_outputs["text_mask"]
        return self.subtask_compressor(tokens, mask)

    def _slot_prompt_suffix(self) -> str:
        dyn_start = str(self.slot_boundary_tokens.get("dynamic_start", "<SOSD>"))
        dyn_end = str(self.slot_boundary_tokens.get("dynamic_end", "<EOSD>"))
        spa_start = str(self.slot_boundary_tokens.get("spatial_start", "<SOSS>"))
        spa_end = str(self.slot_boundary_tokens.get("spatial_end", "<EOSS>"))
        sub_start = str(self.slot_boundary_tokens.get("subtask_start", "<SOST>"))
        sub_end = str(self.slot_boundary_tokens.get("subtask_end", "<EOST>"))
        act_start = str(self.slot_boundary_tokens.get("action_start", "<SOSA>"))
        act_end = str(self.slot_boundary_tokens.get("action_end", "<EOSA>"))

        sub = " ".join([self.subtask_placeholder] * self.k_sub)
        dyn = " ".join([self.dynamic_placeholder] * self.k_dyn)
        spa = " ".join([self.spatial_placeholder] * self.k_spa)
        enable_action_tokens = bool(getattr(self, "enable_action_tokens", False))
        num_action_tokens = int(getattr(self, "num_action_tokens", 0))
        action_placeholder = str(getattr(self, "action_placeholder", "<slot_action>"))
        act = " ".join([action_placeholder] * num_action_tokens) if enable_action_tokens and num_action_tokens > 0 else ""
        # order: [Qwen native multimodal sequence | dynamic block | spatial block | subtask block | action block]
        if act:
            return (
                f" {dyn_start} {dyn} {dyn_end}"
                f" {spa_start} {spa} {spa_end}"
                f" {sub_start} {sub} {sub_end}"
                f" {act_start} {act} {act_end}."
            )
        return f" {dyn_start} {dyn} {dyn_end} {spa_start} {spa} {spa_end} {sub_start} {sub} {sub_end}."

    @staticmethod
    def _positions_to_span(positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert per-sample token positions [B, N] to (start, end, length)."""
        start = positions.min(dim=1).values
        end = positions.max(dim=1).values
        length = torch.full_like(start, positions.shape[1])
        return {
            "start": start,
            "end": end,
            "length": length,
            "positions": positions,
            "mask": positions.ge(0),
        }

    @staticmethod
    def _concat_valid_positions(*position_tensors: Optional[torch.Tensor]) -> torch.Tensor:
        valid_tensors = [positions for positions in position_tensors if positions is not None]
        if not valid_tensors:
            raise ValueError("No position tensors provided for concatenation.")

        first = valid_tensors[0]
        if first.ndim != 2:
            raise ValueError(f"Position tensor must be [B, N], got {tuple(first.shape)}")

        batch_size = first.shape[0]
        device = first.device
        merged_rows: List[torch.Tensor] = []
        max_len = 0

        for batch_idx in range(batch_size):
            pieces = []
            for positions in valid_tensors:
                if positions.ndim != 2:
                    raise ValueError(f"Position tensor must be [B, N], got {tuple(positions.shape)}")
                row = positions[batch_idx]
                row = row[row.ge(0)]
                if row.numel() > 0:
                    pieces.append(row)

            merged = torch.cat(pieces, dim=0) if pieces else torch.empty((0,), dtype=torch.long, device=device)
            merged_rows.append(merged)
            max_len = max(max_len, int(merged.numel()))

        merged_positions = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
        for batch_idx, row in enumerate(merged_rows):
            if row.numel() > 0:
                merged_positions[batch_idx, : row.numel()] = row
        return merged_positions

    def _resolve_qwen_num_hidden_states(self) -> int:
        root_model = getattr(self.qwen_vl_interface, "model", None)
        if root_model is None:
            raise AttributeError("Unable to resolve Qwen model from qwen_vl_interface.")

        candidate_configs = [
            getattr(getattr(root_model, "config", None), "text_config", None),
            getattr(root_model, "config", None),
            getattr(getattr(root_model, "model", None), "config", None),
            getattr(getattr(getattr(root_model, "model", None), "config", None), "text_config", None),
        ]
        for candidate in candidate_configs:
            if candidate is None:
                continue
            num_hidden_layers = getattr(candidate, "num_hidden_layers", None)
            if num_hidden_layers is not None:
                return int(num_hidden_layers) + 1

        text_model = getattr(root_model, "model", None)
        layers = getattr(text_model, "layers", None)
        if layers is not None:
            return len(layers) + 1

        raise AttributeError("Unable to resolve the number of Qwen hidden states for Stage-3 fusion.")

    def _build_sequence_layout_pre_injection(
        self,
        input_ids: torch.Tensor,
        dyn_pos: torch.Tensor,
        spa_pos: torch.Tensor,
        sub_pos: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Build sequence layout records before slot projection injection.

        Records img/text/slot blocks with start/end/length per sample.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        slot_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        slot_mask.scatter_(1, dyn_pos, True)
        slot_mask.scatter_(1, spa_pos, True)
        slot_mask.scatter_(1, sub_pos, True)

        image_mask = input_ids == self.image_token_id
        if action_mask is not None:
            action_mask = action_mask.to(device=device, dtype=torch.bool)
            if action_mask.shape != input_ids.shape:
                raise ValueError(
                    f"action_mask shape {tuple(action_mask.shape)} does not match input_ids {tuple(input_ids.shape)}"
                )
            text_mask = (~image_mask) & (~slot_mask) & (~action_mask)
        else:
            text_mask = (~image_mask) & (~slot_mask)

        def mask_to_span(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
            counts = mask.sum(dim=1).long()
            # For empty groups, set start/end to -1.
            start = torch.full((batch_size,), -1, dtype=torch.long, device=device)
            end = torch.full((batch_size,), -1, dtype=torch.long, device=device)
            positions = []
            for b in range(batch_size):
                pos = torch.where(mask[b])[0]
                if pos.numel() > 0:
                    start[b] = pos.min()
                    end[b] = pos.max()
                positions.append(pos)
            max_len = max((p.numel() for p in positions), default=0)
            pos_pad = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
            for b, pos in enumerate(positions):
                if pos.numel() > 0:
                    pos_pad[b, : pos.numel()] = pos
            return {
                "start": start,
                "end": end,
                "length": counts,
                "positions": pos_pad,
                "mask": counts > 0,
            }

        layout = {
            "text": mask_to_span(text_mask),
            "image": mask_to_span(image_mask),
            "subtask": self._positions_to_span(sub_pos),
            "dynamic": self._positions_to_span(dyn_pos),
            "spatial": self._positions_to_span(spa_pos),
        }
        if action_mask is not None:
            layout["action"] = mask_to_span(action_mask)
        return layout

    @staticmethod
    def _default_attention_visibility_rules() -> Dict[str, Dict[str, bool]]:
        """Default group-wise visibility rules for typed self-attention."""
        return {
            "text": {"text": True, "image": True, "subtask": True, "dynamic": True, "spatial": True, "action": False},
            "image": {"text": True, "image": True, "subtask": False, "dynamic": False, "spatial": True, "action": False},
            "subtask": {"text": True, "image": False, "subtask": True, "dynamic": False, "spatial": False, "action": False},
            "dynamic": {"text": True, "image": False, "subtask": False, "dynamic": True, "spatial": False, "action": False},
            "spatial": {"text": False, "image": True, "subtask": False, "dynamic": False, "spatial": True, "action": False},
            "action": {"text": False, "image": False, "subtask": True, "dynamic": True, "spatial": True, "action": True},
        }

    @staticmethod
    def _span_mask(span: Dict[str, torch.Tensor], seq_len: int) -> torch.Tensor:
        positions = span["positions"]
        if positions.ndim != 2:
            raise ValueError(f"span positions must be [B, N], got {tuple(positions.shape)}")
        batch_size = positions.shape[0]
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=positions.device)
        valid = positions.ge(0)
        if valid.any():
            batch_idx = torch.arange(batch_size, device=positions.device).unsqueeze(1).expand_as(positions)
            mask[batch_idx[valid], positions[valid]] = True
        return mask

    @staticmethod
    def _build_typed_attention_mask_from_layout(
        sequence_layout_pre_injection: Dict[str, Dict[str, torch.Tensor]],
        visibility_rules: Optional[Dict[str, Dict[str, bool]]] = None,
        causal: bool = True,
        return_bias: bool = False,
    ) -> torch.Tensor:
        """Build a typed self-attention keep-mask from recorded layout spans.

        Returns:
            bool tensor [B, L, L] where True means query token may attend to key token.
        """
        required_groups = ["text", "image", "subtask", "dynamic", "spatial"]
        missing = [group for group in required_groups if group not in sequence_layout_pre_injection]
        if missing:
            raise KeyError(f"sequence_layout_pre_injection is missing groups: {missing}")

        if visibility_rules is None:
            visibility_rules = QwenMyVLA._default_attention_visibility_rules()

        all_groups = list(required_groups)
        if "action" in sequence_layout_pre_injection:
            all_groups.append("action")

        sample_group = sequence_layout_pre_injection["text"]
        batch_size = int(sample_group["positions"].shape[0])
        seq_len = 0
        for group_name in all_groups:
            positions = sequence_layout_pre_injection[group_name]["positions"]
            if positions.numel() == 0:
                continue
            valid_positions = positions[positions.ge(0)]
            if valid_positions.numel() > 0:
                seq_len = max(seq_len, int(valid_positions.max().item()) + 1)

        device = sample_group["positions"].device
        keep_mask = torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)

        if causal:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
            keep_mask &= causal_mask.unsqueeze(0)

        group_masks = {
            group: QwenMyVLA._span_mask(sequence_layout_pre_injection[group], seq_len=seq_len)
            for group in all_groups
        }

        for query_group in all_groups:
            query_mask = group_masks[query_group]
            if not query_mask.any():
                continue
            query_mask_3d = query_mask.unsqueeze(-1)

            for key_group in all_groups:
                if visibility_rules.get(query_group, {}).get(key_group, False):
                    continue
                key_mask = group_masks[key_group].unsqueeze(1)
                keep_mask = keep_mask & ~(query_mask_3d & key_mask)

        if not return_bias:
            return keep_mask

        bias = torch.zeros_like(keep_mask, dtype=torch.float32)
        bias = bias.masked_fill(~keep_mask, torch.finfo(bias.dtype).min)
        return bias

    def _build_self_attention_mask(
        self,
        sequence_layout_pre_injection: Dict[str, Dict[str, torch.Tensor]],
        causal: bool = True,
    ) -> torch.Tensor:
        return self._build_typed_attention_mask_from_layout(
            sequence_layout_pre_injection=sequence_layout_pre_injection,
            visibility_rules=self.attention_visibility_rules,
            causal=causal,
            return_bias=False,
        )

    def _build_token_type_ids(
        self,
        input_ids: torch.Tensor,
        dyn_pos: torch.Tensor,
        spa_pos: torch.Tensor,
        sub_pos: torch.Tensor,
        action_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build token-type ids for routing in the last half layers.

        0 = base tokens
        1 = subtask block tokens
        2 = dynamic block tokens
        3 = spatial block tokens
        """
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids.scatter_(1, sub_pos, TypedResidualFFN.TYPE_IDS["subtask"])
        token_type_ids.scatter_(1, dyn_pos, TypedResidualFFN.TYPE_IDS["dynamic"])
        token_type_ids.scatter_(1, spa_pos, TypedResidualFFN.TYPE_IDS["spatial"])
        if action_pos is not None and action_pos.numel() > 0:
            token_type_ids.scatter_(1, action_pos, TypedResidualFFN.TYPE_IDS["action"])
        return token_type_ids

    def _build_fast_action_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build token types for fast-action CE branch.

        In fast branch we mark `<robot_action_*>` tokens as action type so
        action expert can be activated on those positions.
        """
        token_type_ids = torch.zeros_like(input_ids)
        if not self.fast_action_token_ids:
            return token_type_ids

        device = input_ids.device
        action_token_ids = torch.tensor(sorted(self.fast_action_token_ids), device=device, dtype=input_ids.dtype)
        action_mask = (input_ids.unsqueeze(-1) == action_token_ids.view(1, 1, -1)).any(dim=-1)
        token_type_ids[action_mask] = TypedResidualFFN.TYPE_IDS["action"]
        return token_type_ids

    @staticmethod
    def _compute_supervised_fast_token_metrics(
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute token-level FAST supervision stats aligned with causal LM loss.

        Assumes `labels` already contain only FAST action-token supervision and
        all non-action/template positions are IGNORE_INDEX.
        """
        if logits is None or labels is None:
            return {
                "fast_label_token_count": 0.0,
                "fast_token_correct": 0.0,
                "fast_token_accuracy": 0.0,
            }

        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError(
                f"Expected logits [B, L, V] and labels [B, L], got {tuple(logits.shape)} and {tuple(labels.shape)}"
            )

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return {
                "fast_label_token_count": 0.0,
                "fast_token_correct": 0.0,
                "fast_token_accuracy": 0.0,
            }

        predictions = shift_logits.argmax(dim=-1)
        correct = int(((predictions == shift_labels) & valid_mask).sum().item())
        return {
            "fast_label_token_count": float(valid_count),
            "fast_token_correct": float(correct),
            "fast_token_accuracy": float(correct / valid_count),
        }

    @staticmethod
    def _gather_slot_hidden(last_hidden: torch.Tensor, slot_positions: torch.Tensor) -> torch.Tensor:
        """Gather slot hidden states from [B, L, H] with slot positions [B, K]."""
        if last_hidden.ndim != 3:
            raise ValueError(f"last_hidden must be [B, L, H], got {tuple(last_hidden.shape)}")
        if slot_positions.ndim != 2:
            raise ValueError(f"slot_positions must be [B, K], got {tuple(slot_positions.shape)}")

        batch_size, _, _ = last_hidden.shape
        gather_batch_idx = torch.arange(batch_size, device=last_hidden.device).unsqueeze(1).expand_as(slot_positions)
        return last_hidden[gather_batch_idx, slot_positions, :]

    def _compute_slot_alignment_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute simple alignment loss used by three slot branches.

        pred/target: [B, K, H]
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

        target_cast = target.to(device=pred.device, dtype=pred.dtype)

        # Default: all slots supervised equally.
        weight = torch.ones(pred.shape[:2], device=pred.device, dtype=pred.dtype)

        if keep_mask is not None and self.slot_align_masked_target_prediction:
            keep = keep_mask.to(device=pred.device, dtype=torch.bool)
            if keep.shape != pred.shape[:2]:
                raise ValueError(
                    f"keep_mask shape {tuple(keep.shape)} does not match slot shape {tuple(pred.shape[:2])}"
                )

            masked = ~keep
            if self.slot_align_masked_only:
                weight = masked.to(dtype=pred.dtype)
                if self.slot_align_fallback_to_all_when_no_masked and not masked.any():
                    weight = torch.ones_like(weight)
            else:
                weight = masked.to(dtype=pred.dtype) + self.slot_align_unmasked_weight * keep.to(dtype=pred.dtype)

        mse_per_token = ((pred - target_cast) ** 2).mean(dim=-1)
        cos_per_token = 1.0 - F.cosine_similarity(pred, target_cast, dim=-1)
        denom = weight.sum().clamp_min(1.0)
        mse = (mse_per_token * weight).sum() / denom
        cos = (cos_per_token * weight).sum() / denom
        total = self.slot_align_loss_mse_ratio * mse + self.slot_align_loss_cos_ratio * cos
        return {
            "mse": mse,
            "cos": cos,
            "total": total,
        }

    def _build_predicted_actions(
        self,
        dynamic_fused: torch.Tensor,
        spatial_fused: torch.Tensor,
        subtask_fused: torch.Tensor,
    ) -> torch.Tensor:
        """Build normalized action predictions from fused slot readouts."""
        summary = torch.cat(
            [
                dynamic_fused.mean(dim=1),
                spatial_fused.mean(dim=1),
                subtask_fused.mean(dim=1),
            ],
            dim=-1,
        )
        pred = self.action_predictor(summary)
        pred = pred.view(summary.shape[0], self.num_actions_chunk, self.action_dim)
        return pred

    def _extract_gt_actions(
        self,
        examples: List[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not all("action" in sample and sample["action"] is not None for sample in examples):
            return None

        gt_actions = torch.as_tensor(np.stack([sample["action"] for sample in examples]), device=device, dtype=dtype)
        if gt_actions.ndim == 2:
            gt_actions = gt_actions.unsqueeze(1)
        if gt_actions.ndim != 3:
            raise ValueError(f"GT action must be [B, T, D], got {tuple(gt_actions.shape)}")
        return gt_actions

    def _compute_continuous_fm_loss(
        self,
        predicted_actions: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for the light continuous branch.

        Important:
            This branch does not condition on sampled x_t or timestep t, so it
            cannot implement true flow matching. We therefore use direct
            supervised regression on x1 and reserve actual FM for the Stage-3
            DiT action head.
        """
        objective = self.continuous_fm_objective
        if objective == "l1":
            return F.l1_loss(predicted_actions, gt_actions)

        if objective != "fm":
            raise ValueError(f"Unsupported continuous_fm_loss.objective: {objective}")

        if self.continuous_fm_loss_type == "l1":
            return F.l1_loss(predicted_actions, gt_actions)
        if self.continuous_fm_loss_type == "mse":
            return F.mse_loss(predicted_actions, gt_actions)
        raise ValueError(f"Unsupported continuous_fm_loss.loss_type: {self.continuous_fm_loss_type}")

    def _build_stage3_action_positions(self, dyn_pos: torch.Tensor, spa_pos: torch.Tensor, sub_pos: torch.Tensor) -> torch.Tensor:
        pos = torch.cat([sub_pos, dyn_pos, spa_pos], dim=1)
        cur = pos.shape[1]
        if cur == self.stage3_action_query_num:
            return pos
        if cur > self.stage3_action_query_num:
            return pos[:, : self.stage3_action_query_num]

        pad_count = self.stage3_action_query_num - cur
        pad = pos[:, -1:].expand(-1, pad_count)
        return torch.cat([pos, pad], dim=1)

    def _build_stage3_action_positions_with_optional_action_tokens(
        self,
        dyn_pos: torch.Tensor,
        spa_pos: torch.Tensor,
        sub_pos: torch.Tensor,
        action_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if action_pos is not None and action_pos.numel() > 0:
            pos = action_pos
            cur = pos.shape[1]
            if cur == self.stage3_action_query_num:
                return pos
            if cur > self.stage3_action_query_num:
                return pos[:, : self.stage3_action_query_num]
            pad_count = self.stage3_action_query_num - cur
            pad = pos[:, -1:].expand(-1, pad_count)
            return torch.cat([pos, pad], dim=1)

        return self._build_stage3_action_positions(dyn_pos=dyn_pos, spa_pos=spa_pos, sub_pos=sub_pos)

    def _build_stage3_multilayer_hidden(
        self,
        hidden_states: List[torch.Tensor],
        image_positions: torch.Tensor,
        action_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Build [B, Layers(all), vision_len + action_query_num, H]."""
        selected = hidden_states
        batch_size = image_positions.shape[0]
        device = image_positions.device
        valid_image = image_positions.ge(0)
        vision_hidden_len = int(valid_image.sum(dim=1).max().item())

        if vision_hidden_len <= 0:
            raise ValueError("No image tokens found for Stage-3 action header.")

        image_pos_trim = image_positions[:, :vision_hidden_len].clone()
        image_pos_trim[~valid_image[:, :vision_hidden_len]] = 0

        batch_idx_image = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(image_pos_trim)
        batch_idx_action = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(action_positions)

        all_layers = []
        for layer_hidden in selected:
            vision_states = layer_hidden[batch_idx_image, image_pos_trim, :]
            if (~valid_image[:, :vision_hidden_len]).any():
                vision_states = vision_states.masked_fill((~valid_image[:, :vision_hidden_len]).unsqueeze(-1), 0.0)
            action_states = layer_hidden[batch_idx_action, action_positions, :]
            all_layers.append(torch.cat([vision_states, action_states], dim=1).unsqueeze(1))

        return torch.cat(all_layers, dim=1), vision_hidden_len

    def _build_stage3_m1_condition(
        self,
        hidden_states: List[torch.Tensor],
        batch_images: List,
        sequence_layout_pre_injection: Dict[str, Dict[str, torch.Tensor]],
        action_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.stage3_condition_fuser is None or self.stage3_dino_encoder is None or self.stage3_dino_projector is None:
            raise RuntimeError("Stage-3 M1 condition builder is not initialized.")

        if len(hidden_states) < self.stage3_condition_fuser.num_layers:
            raise ValueError(
                f"Qwen hidden state count {len(hidden_states)} is smaller than the stage-3 fusion depth "
                f"{self.stage3_condition_fuser.num_layers}."
            )

        hidden_states = hidden_states[-self.stage3_condition_fuser.num_layers :]
        text_positions = sequence_layout_pre_injection["text"]["positions"]
        sub_positions = sequence_layout_pre_injection["subtask"]["positions"]
        dyn_positions = sequence_layout_pre_injection["dynamic"]["positions"]
        spa_positions = sequence_layout_pre_injection["spatial"]["positions"]
        merged_positions = self._concat_valid_positions(text_positions, sub_positions, dyn_positions, spa_positions, action_pos)

        valid_mask = merged_positions.ge(0)
        gather_positions = merged_positions.clamp_min(0)
        batch_size = gather_positions.shape[0]
        device = gather_positions.device
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(gather_positions)

        with torch.no_grad():
            dino_inputs = self.stage3_dino_encoder.prepare_dino_input(batch_images)
            dino_tokens = self.stage3_dino_encoder(dino_inputs)

        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with autocast_ctx:
            dino_tokens = self.stage3_dino_projector(dino_tokens.to(device=device, dtype=hidden_states[0].dtype))
            if dino_tokens.shape[0] != batch_size:
                if dino_tokens.shape[0] % batch_size != 0:
                    raise ValueError(
                        f"Stage-3 DINO token batch size {dino_tokens.shape[0]} is not divisible by qwen batch size {batch_size}."
                    )
                num_views = dino_tokens.shape[0] // batch_size
                dino_tokens = dino_tokens.reshape(batch_size, num_views * dino_tokens.shape[1], dino_tokens.shape[2])
            dino_mask = torch.zeros((batch_size, dino_tokens.shape[1]), dtype=torch.bool, device=device)

            fused_layers = []
            for layer_hidden in hidden_states:
                layer_hidden = layer_hidden.to(device=device, dtype=dino_tokens.dtype)
                qwen_tokens = layer_hidden[batch_indices, gather_positions, :]
                if (~valid_mask).any():
                    qwen_tokens = qwen_tokens.masked_fill((~valid_mask).unsqueeze(-1), 0.0)
                fused_layers.append(torch.cat([qwen_tokens, dino_tokens], dim=1))

            fused_attention_mask = torch.cat([~valid_mask, dino_mask], dim=1)
            return self.stage3_condition_fuser(fused_layers, encoder_attention_mask=fused_attention_mask)

    def _sample_stage3_actions(
        self,
        stage3_condition: torch.Tensor,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 5,
    ) -> torch.Tensor:
        if self.stage3_action_head is None:
            raise RuntimeError("Stage-3 action head is not initialized.")

        batch_size = stage3_condition.shape[0]
        device = stage3_condition.device
        model_dtype = next(self.stage3_action_head.net.parameters()).dtype
        stage3_condition = stage3_condition.to(device=device, dtype=model_dtype)
        if hasattr(self.stage3_action_head, "sample_actions"):
            sample_steps = int(num_ddim_steps) if use_ddim and num_ddim_steps is not None else 10
            return self.stage3_action_head.sample_actions(
                condition=stage3_condition,
                cfg_scale=cfg_scale,
                num_steps=sample_steps,
            )
        action_horizon = getattr(self.stage3_action_head, "action_horizon", None)
        if action_horizon is None:
            future_window = int(
                getattr(self.stage3_action_head, "future_action_window_size", self.num_actions_chunk - 1)
            )
            past_window = int(getattr(self.stage3_action_head, "past_action_window_size", 0))
            action_horizon = future_window + past_window + 1
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"Invalid stage-3 action horizon: {action_horizon}")

        noise = torch.randn(
            batch_size,
            action_horizon,
            self.stage3_action_head.in_channels,
            device=device,
        ).to(model_dtype)

        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], dim=0)
            uncondition = self.stage3_action_head.net.z_embedder.uncondition.to(device=device, dtype=model_dtype)
            uncondition_shape = uncondition.shape
            uncondition = uncondition.unsqueeze(0).expand(batch_size, uncondition_shape[0], uncondition_shape[1])
            z = torch.cat([stage3_condition, uncondition], dim=0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.stage3_action_head.net.forward_with_cfg
        else:
            model_kwargs = dict(z=stage3_condition)
            sample_fn = self.stage3_action_head.net.forward

        if use_ddim and num_ddim_steps is not None:
            if self.stage3_action_head.ddim_diffusion is None:
                self.stage3_action_head.create_ddim(ddim_step=num_ddim_steps)
            samples = self.stage3_action_head.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
                eta=0.0,
            )
        else:
            samples = self.stage3_action_head.diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
        return samples

    @staticmethod
    def _map_fast_token_to_vlm_action(tokens: List[int]) -> str:
        return "".join([f"<robot_action_{token}>" for token in tokens])

    def _compute_fast_action_loss(self, examples: List[dict], batch_images: List, instructions: List[str]) -> torch.Tensor:
        if self.fast_action_model is None:
            raise RuntimeError("fast_action_model is not initialized while fast_action_loss is enabled.")
        if not all("action" in sample and sample["action"] is not None for sample in examples):
            return torch.zeros((), device=self.qwen_vl_interface.model.device, dtype=torch.float32)

        actions = [sample["action"] for sample in examples]
        batch_fast_tokens = self.fast_action_model.encoder_action2fastoken(actions)
        vlm_action_tokens = [self._map_fast_token_to_vlm_action(tokens) for tokens in batch_fast_tokens]

        qwen_inputs_fast = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=vlm_action_tokens,
        )

        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
        fast_token_type_ids = self._build_fast_action_token_type_ids(qwen_inputs_fast["input_ids"])
        fast_self_attention_mask = None
        if self.enable_back_half_typed_attention:
            try:
                fast_input_ids = qwen_inputs_fast["input_ids"]
                dyn_pos_fast = self._gather_positions(fast_input_ids, self.dynamic_token_id, self.k_dyn)
                spa_pos_fast = self._gather_positions(fast_input_ids, self.spatial_token_id, self.k_spa)
                sub_pos_fast = self._gather_positions(fast_input_ids, self.subtask_token_id, self.k_sub)
                fast_action_mask = fast_token_type_ids.eq(TypedResidualFFN.TYPE_IDS["action"])
                fast_layout = self._build_sequence_layout_pre_injection(
                    input_ids=fast_input_ids,
                    dyn_pos=dyn_pos_fast,
                    spa_pos=spa_pos_fast,
                    sub_pos=sub_pos_fast,
                    action_mask=fast_action_mask,
                )
                fast_self_attention_mask = self._build_self_attention_mask(
                    sequence_layout_pre_injection=fast_layout,
                )
            except Exception as exc:
                logger.warning("Skip fast-branch typed attention mask build due to layout error: %s", exc)

        self._set_typed_ffn_token_type_ids(fast_token_type_ids)
        if self.enable_back_half_typed_attention and fast_self_attention_mask is not None:
            self._set_back_half_typed_attention_mask(fast_self_attention_mask)
        try:
            with autocast_ctx:
                fast_outputs = self.qwen_vl_interface(
                    **qwen_inputs_fast,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            self._set_typed_ffn_token_type_ids(None)
            if self.enable_back_half_typed_attention:
                self._set_back_half_typed_attention_mask(None)

        fast_loss = fast_outputs.loss
        if fast_loss is None or torch.isnan(fast_loss):
            fast_loss = torch.zeros((), device=qwen_inputs_fast["input_ids"].device, dtype=torch.float32)
        return fast_loss

    @staticmethod
    def _sample_inner_keep_mask(batch_size: int, num_slots: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        if mask_ratio <= 0:
            return torch.ones((batch_size, num_slots), dtype=torch.bool, device=device)
        if mask_ratio >= 1:
            return torch.zeros((batch_size, num_slots), dtype=torch.bool, device=device)

        mask_count = max(1, int(round(num_slots * mask_ratio)))
        keep = torch.ones((batch_size, num_slots), dtype=torch.bool, device=device)
        for b in range(batch_size):
            perm = torch.randperm(num_slots, device=device)
            keep[b, perm[:mask_count]] = False
        return keep

    @staticmethod
    def _sample_outside_keep_mask(
        batch_size: int,
        slot_sizes: Dict[str, int],
        num_masked_probs: List[float],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        slot_names = ["subtask", "dynamic", "spatial"]
        keep = {
            name: torch.ones((batch_size, slot_sizes[name]), dtype=torch.bool, device=device)
            for name in slot_names
        }

        probs = torch.tensor(num_masked_probs, device=device, dtype=torch.float32)
        choices = torch.multinomial(probs, num_samples=batch_size, replacement=True)  # each in {0,1,2,3}

        for b in range(batch_size):
            k = int(choices[b].item())
            if k <= 0:
                continue
            if k >= len(slot_names):
                masked_types = slot_names
            else:
                perm = torch.randperm(len(slot_names), device=device)
                masked_types = [slot_names[i] for i in perm[:k].tolist()]
            for name in masked_types:
                keep[name][b] = False

        return keep

    def _apply_slot_masks(
        self,
        dynamic_slots: torch.Tensor,
        spatial_slots: torch.Tensor,
        subtask_slots: torch.Tensor,
        inner_mask_ratios: Dict[str, float],
        outside_num_masked_probs: List[float],
    ) -> Dict[str, torch.Tensor]:
        batch_size = dynamic_slots.shape[0]
        device = dynamic_slots.device

        keep_dynamic = self._sample_inner_keep_mask(
            batch_size, self.k_dyn, inner_mask_ratios["dynamic"], device
        )
        keep_spatial = self._sample_inner_keep_mask(
            batch_size, self.k_spa, inner_mask_ratios["spatial"], device
        )
        keep_subtask = self._sample_inner_keep_mask(
            batch_size, self.k_sub, inner_mask_ratios["subtask"], device
        )

        outside_keep = self._sample_outside_keep_mask(
            batch_size=batch_size,
            slot_sizes={"subtask": self.k_sub, "dynamic": self.k_dyn, "spatial": self.k_spa},
            num_masked_probs=outside_num_masked_probs,
            device=device,
        )

        keep_dynamic = keep_dynamic & outside_keep["dynamic"]
        keep_spatial = keep_spatial & outside_keep["spatial"]
        keep_subtask = keep_subtask & outside_keep["subtask"]

        masked_dynamic = dynamic_slots * keep_dynamic.to(dynamic_slots.dtype).unsqueeze(-1)
        masked_spatial = spatial_slots * keep_spatial.to(spatial_slots.dtype).unsqueeze(-1)
        masked_subtask = subtask_slots * keep_subtask.to(subtask_slots.dtype).unsqueeze(-1)

        return {
            "dynamic_slots": masked_dynamic,
            "spatial_slots": masked_spatial,
            "subtask_slots": masked_subtask,
            "keep_dynamic": keep_dynamic,
            "keep_spatial": keep_spatial,
            "keep_subtask": keep_subtask,
        }

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        if examples is None:
            raise ValueError("QwenMyVLA.forward expects examples: List[dict]")

        batch_images = [example["image"] for example in examples]
        instructions = [example.get("lang", example.get("instruction", "")) for example in examples]
        actions = [example["action"] for example in examples]
        progress_info = self._resolve_training_progress_info(kwargs)
        training_progress = float(progress_info["progress"])

        lam_outputs = self.dynamic_teacher_encoder(examples=examples)
        depth_outputs = self.spatial_teacher_encoder(examples=examples)
        subtask_outputs = self.subtask_slot_encoder(examples=examples)

        dynamic_slots = self._build_dynamic_slots(lam_outputs)
        spatial_slots = self._build_spatial_slots(depth_outputs)
        subtask_slots = self._build_subtask_slots(subtask_outputs)

        dynamic_slots_gt = dynamic_slots
        spatial_slots_gt = spatial_slots
        subtask_slots_gt = subtask_slots

        use_mask = self.enable_slot_mask and (self.training or self.slot_mask_apply_in_eval)
        if use_mask:
            inner_mask_ratios_cur, outside_mask_probs_cur = self._resolve_slot_mask_hparams(training_progress)
            masked = self._apply_slot_masks(
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
            inner_mask_ratios_cur = self.inner_mask_ratios
            outside_mask_probs_cur = self.outside_num_masked_probs
            keep_dynamic = torch.ones((dynamic_slots.shape[0], self.k_dyn), dtype=torch.bool, device=dynamic_slots.device)
            keep_spatial = torch.ones((spatial_slots.shape[0], self.k_spa), dtype=torch.bool, device=spatial_slots.device)
            keep_subtask = torch.ones((subtask_slots.shape[0], self.k_sub), dtype=torch.bool, device=subtask_slots.device)

        suffix = self._slot_prompt_suffix()
        instructions = [text + suffix for text in instructions]

        use_fast_supervision = self.enable_fast_action_loss and self.training
        vlm_action_tokens = None
        if use_fast_supervision:
            if self.fast_action_model is None:
                raise RuntimeError(
                    "fast_action_loss is enabled but fast_action_model is not initialized."
                )
            if not all("action" in sample and sample["action"] is not None for sample in examples):
                raise RuntimeError(
                    "fast_action_loss is enabled during training but batch examples do not contain GT `action`."
                )
            actions = [sample["action"] for sample in examples]
            batch_fast_tokens = self.fast_action_model.encoder_action2fastoken(actions)
            vlm_action_tokens = [self._map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=vlm_action_tokens if use_fast_supervision else None,
        )
        input_ids = qwen_inputs["input_ids"]

        dyn_pos = self._gather_positions(input_ids, self.dynamic_token_id, self.k_dyn)
        spa_pos = self._gather_positions(input_ids, self.spatial_token_id, self.k_spa)
        sub_pos = self._gather_positions(input_ids, self.subtask_token_id, self.k_sub)
        enable_action_tokens = bool(getattr(self, "enable_action_tokens", False))
        num_action_tokens = int(getattr(self, "num_action_tokens", 0))
        action_token_id = getattr(self, "action_token_id", None)
        action_pos = None
        action_mask = None
        if enable_action_tokens and action_token_id is not None and num_action_tokens > 0:
            action_pos = self._gather_positions(input_ids, action_token_id, num_action_tokens)
            action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            action_mask.scatter_(1, action_pos, True)
        fast_action_token_type_ids = None
        if use_fast_supervision:
            fast_action_token_type_ids = self._build_fast_action_token_type_ids(input_ids)
            fast_action_mask = fast_action_token_type_ids.eq(TypedResidualFFN.TYPE_IDS["action"])
            if action_mask is None:
                action_mask = fast_action_mask
            else:
                action_mask = action_mask | fast_action_mask
        sequence_layout_pre_injection = self._build_sequence_layout_pre_injection(
            input_ids=input_ids,
            dyn_pos=dyn_pos,
            spa_pos=spa_pos,
            sub_pos=sub_pos,
            action_mask=action_mask,
        )
        token_type_ids = self._build_token_type_ids(
            input_ids,
            dyn_pos=dyn_pos,
            spa_pos=spa_pos,
            sub_pos=sub_pos,
            action_pos=action_pos,
        )
        if fast_action_token_type_ids is not None:
            fast_action_mask = fast_action_token_type_ids.eq(TypedResidualFFN.TYPE_IDS["action"])
            token_type_ids[fast_action_mask] = TypedResidualFFN.TYPE_IDS["action"]
        self_attention_mask = self._build_self_attention_mask(sequence_layout_pre_injection=sequence_layout_pre_injection)

        batch_size = input_ids.shape[0]

        self._set_typed_ffn_token_type_ids(token_type_ids)
        if self.enable_back_half_typed_attention:
            self._set_back_half_typed_attention_mask(self_attention_mask)

        def inject_slot_hook(_module, _inputs, output):
            batch_idx_dyn = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_dyn)
            batch_idx_spa = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_spa)
            batch_idx_sub = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_sub)
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

            action_token_queries = getattr(self, "action_token_queries", None)
            if action_pos is not None and action_token_queries is not None:
                action_queries = action_token_queries.to(device=output.device, dtype=output.dtype)
                action_queries = action_queries.unsqueeze(0).expand(batch_size, -1, -1)
                output[batch_idx_act, action_pos, :] = action_queries
            return output

        embedding_layer = self._resolve_input_embeddings_layer()
        hook_handle = embedding_layer.register_forward_hook(inject_slot_hook)
        try:
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
            with autocast_ctx:
                qwen_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()
            self._set_typed_ffn_token_type_ids(None)
            if self.enable_back_half_typed_attention:
                self._set_back_half_typed_attention_mask(None)

        loss_device = qwen_outputs.hidden_states[-1].device
        action_loss = torch.zeros((), device=loss_device, dtype=torch.float32)
        qwen_last_hidden = qwen_outputs.hidden_states[-1]

        qwen_dynamic_slot_hidden = self._gather_slot_hidden(qwen_last_hidden, dyn_pos)
        qwen_spatial_slot_hidden = self._gather_slot_hidden(qwen_last_hidden, spa_pos)
        qwen_subtask_slot_hidden = self._gather_slot_hidden(qwen_last_hidden, sub_pos)

        qwen_dynamic_slot_fused = self.dynamic_slot_fuser(qwen_dynamic_slot_hidden)
        qwen_spatial_slot_fused = self.spatial_slot_fuser(qwen_spatial_slot_hidden)
        qwen_subtask_slot_fused = self.subtask_slot_fuser(qwen_subtask_slot_hidden)
        predicted_actions_light = self._build_predicted_actions(
            dynamic_fused=qwen_dynamic_slot_fused,
            spatial_fused=qwen_spatial_slot_fused,
            subtask_fused=qwen_subtask_slot_fused,
        )

        stage3_condition = None
        if self.enable_stage3_action_header:
            if self.stage3_action_head is None:
                raise RuntimeError("stage3_action_head is not initialized while stage3_action_header is enabled.")

            stage3_action_pos = self._build_stage3_action_positions_with_optional_action_tokens(
                dyn_pos=dyn_pos,
                spa_pos=spa_pos,
                sub_pos=sub_pos,
                action_pos=action_pos,
            )
            stage3_condition = self._build_stage3_m1_condition(
                hidden_states=list(qwen_outputs.hidden_states),
                batch_images=batch_images,
                sequence_layout_pre_injection=sequence_layout_pre_injection,
                action_pos=stage3_action_pos,
            )

        predicted_actions = predicted_actions_light

        gt_actions = self._extract_gt_actions(
            examples=examples,
            device=predicted_actions.device,
            dtype=predicted_actions.dtype,
        )
        if gt_actions is not None:
            if gt_actions.shape[-1] != predicted_actions.shape[-1]:
                raise ValueError(
                    f"Action dim mismatch: gt={gt_actions.shape[-1]}, pred={predicted_actions.shape[-1]}. "
                    "Please check framework.action_model.action_dim and dataset action dimension."
                )

            if self.enable_stage3_action_header and stage3_condition is not None:
                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                action_head_dtype = next(self.stage3_action_head.net.parameters()).dtype
                stage3_condition_repeated = stage3_condition.to(device=gt_actions.device, dtype=action_head_dtype).repeat(
                    repeated_diffusion_steps, 1, 1
                )
                gt_actions_repeated = gt_actions.to(device=gt_actions.device, dtype=action_head_dtype).repeat(
                    repeated_diffusion_steps, 1, 1
                )
                noise_pred, noise, timestep = self.stage3_action_head(gt_actions_repeated, stage3_condition_repeated)
                continuous_fm_loss = self.stage3_action_head.loss(noise_pred, noise)
            else:
                if gt_actions.shape[1] != predicted_actions.shape[1]:
                    target_len = min(gt_actions.shape[1], predicted_actions.shape[1])
                    logger.warning(
                        "Action chunk length mismatch detected (gt=%d, pred=%d). Truncating both to %d for L1 loss.",
                        gt_actions.shape[1],
                        predicted_actions.shape[1],
                        target_len,
                    )
                    gt_actions = gt_actions[:, :target_len, :]
                    predicted_actions = predicted_actions[:, :target_len, :]

                continuous_fm_loss = self._compute_continuous_fm_loss(predicted_actions, gt_actions)
        else:
            continuous_fm_loss = torch.zeros((), device=loss_device, dtype=torch.float32)

        if self.enable_slot_align_loss:
            effective_weights = self._resolve_loss_weights(training_progress)
            align_dynamic = self._compute_slot_alignment_loss(
                qwen_dynamic_slot_fused,
                dynamic_slots_gt,
                keep_mask=keep_dynamic,
            )
            align_spatial = self._compute_slot_alignment_loss(
                qwen_spatial_slot_fused,
                spatial_slots_gt,
                keep_mask=keep_spatial,
            )
            align_subtask = self._compute_slot_alignment_loss(
                qwen_subtask_slot_fused,
                subtask_slots_gt,
                keep_mask=keep_subtask,
            )

            slot_align_loss = (
                effective_weights["slot_dynamic"] * align_dynamic["total"]
                + effective_weights["slot_spatial"] * align_spatial["total"]
                + effective_weights["slot_subtask"] * align_subtask["total"]
            )
        else:
            effective_weights = self._resolve_loss_weights(training_progress)
            zero = torch.zeros((), device=loss_device, dtype=torch.float32)
            align_dynamic = {"mse": zero, "cos": zero, "total": zero}
            align_spatial = {"mse": zero, "cos": zero, "total": zero}
            align_subtask = {"mse": zero, "cos": zero, "total": zero}
            slot_align_loss = zero

        if use_fast_supervision:
            fast_action_loss = qwen_outputs.loss
            if fast_action_loss is None or torch.isnan(fast_action_loss):
                fast_action_loss = torch.zeros((), device=loss_device, dtype=torch.float32)
            fast_action_loss_for_optim = fast_action_loss.detach() if self.fast_action_loss_detach else fast_action_loss
            fast_token_metrics = self._compute_supervised_fast_token_metrics(
                logits=getattr(qwen_outputs, "logits", None),
                labels=qwen_inputs.get("labels"),
            )
        else:
            fast_action_loss = torch.zeros((), device=loss_device, dtype=torch.float32)
            fast_action_loss_for_optim = fast_action_loss
            fast_token_metrics = {
                "fast_label_token_count": 0.0,
                "fast_token_correct": 0.0,
                "fast_token_accuracy": 0.0,
            }

        continuous_fm_loss_for_optim = (
            continuous_fm_loss.detach() if self.continuous_fm_loss_detach else continuous_fm_loss
        )

        action_loss = (
            action_loss
            + slot_align_loss
            + effective_weights["continuous"] * continuous_fm_loss_for_optim
            + effective_weights["fast"] * fast_action_loss_for_optim
        )

        if self.strict_loss_checks and self.training and gt_actions is None and not self.enable_slot_align_loss:
            raise RuntimeError(
                "No training signal found: both GT action supervision and slot alignment loss are disabled."
            )

        outputs = {
            "action_loss": action_loss,
            "slot_align_loss": slot_align_loss.detach(),
            "slot_align_masked_target_prediction": self.slot_align_masked_target_prediction,
            "slot_align_masked_only": self.slot_align_masked_only,
            # continuous_fm_loss: continuous action branch loss (Stage-3 style)
            "continuous_fm_loss": continuous_fm_loss.detach(),
            "continuous_fm_loss_for_optim": continuous_fm_loss_for_optim.detach(),
            "continuous_fm_loss_detach": self.continuous_fm_loss_detach,
            "continuous_fm_objective": self.continuous_fm_objective,
            # backward compatibility alias
            "supervised_action_loss": continuous_fm_loss.detach(),
            "fast_action_loss": fast_action_loss.detach(),
            "fast_action_loss_for_optim": fast_action_loss_for_optim.detach(),
            "fast_action_loss_detach": self.fast_action_loss_detach,
            "fast_label_token_count": fast_token_metrics["fast_label_token_count"],
            "fast_token_correct": fast_token_metrics["fast_token_correct"],
            "fast_token_accuracy": fast_token_metrics["fast_token_accuracy"],
            "predicted_actions": predicted_actions,
            "predicted_actions_light": predicted_actions_light.detach(),
            "stage3_coarse_condition": (
                stage3_condition.mean(dim=1).detach() if stage3_condition is not None else None
            ),
            "stage3_condition": stage3_condition.detach() if stage3_condition is not None else None,
            "use_stage3_action_header": self.enable_stage3_action_header,
            "lam_indices": lam_outputs["indices"],
            "lam_z_q": lam_outputs["z_q"],
            "dynamic_slots": dynamic_slots,
            "spatial_slots": spatial_slots,
            "subtask_slots": subtask_slots,
            "slot_positions": {
                "dynamic": dyn_pos,
                "spatial": spa_pos,
                "subtask": sub_pos,
                "action": action_pos,
            },
            "token_type_ids_for_moe": token_type_ids,
            "sequence_layout_pre_injection": sequence_layout_pre_injection,
            "self_attention_mask": self_attention_mask,
            "slot_keep_mask": {
                "dynamic": keep_dynamic,
                "spatial": keep_spatial,
                "subtask": keep_subtask,
            },
            "training_progress": training_progress,
            "training_progress_state": {
                "uses_step_mapping": bool(progress_info["uses_step_mapping"]),
                "global_step": int(progress_info["global_step"]),
                "max_train_steps": int(progress_info["max_train_steps"]),
                "step_start": int(progress_info["step_start"]),
                "step_end": int(progress_info["step_end"]),
                "is_before_start": bool(progress_info["is_before_start"]),
                "is_after_end": bool(progress_info["is_after_end"]),
            },
            "slot_mask_curriculum_state": self._resolve_schedule_flags(
                progress=training_progress,
                start=float(self.slot_mask_curriculum_progress_start),
                end=float(self.slot_mask_curriculum_progress_end),
                enable=bool(self.enable_slot_mask_curriculum and use_mask),
            ),
            "loss_weight_schedule_state": self._resolve_schedule_flags(
                progress=training_progress,
                start=float(self.loss_weight_schedule_progress_start),
                end=float(self.loss_weight_schedule_progress_end),
                enable=bool(self.enable_loss_weight_schedule),
            ),
            "slot_mask_hparams": {
                "inner": {
                    "dynamic": float(inner_mask_ratios_cur["dynamic"]),
                    "spatial": float(inner_mask_ratios_cur["spatial"]),
                    "subtask": float(inner_mask_ratios_cur["subtask"]),
                },
                "outside_num_masked_probs": [float(x) for x in outside_mask_probs_cur],
            },
            "effective_loss_weights": {
                "slot_dynamic": float(effective_weights["slot_dynamic"]),
                "slot_spatial": float(effective_weights["slot_spatial"]),
                "slot_subtask": float(effective_weights["slot_subtask"]),
                "continuous": float(effective_weights["continuous"]),
                "fast": float(effective_weights["fast"]),
            },
            "slot_align_loss_breakdown": {
                "dynamic": {
                    "mse": align_dynamic["mse"].detach(),
                    "cos": align_dynamic["cos"].detach(),
                    "total": align_dynamic["total"].detach(),
                },
                "spatial": {
                    "mse": align_spatial["mse"].detach(),
                    "cos": align_spatial["cos"].detach(),
                    "total": align_spatial["total"].detach(),
                },
                "subtask": {
                    "mse": align_subtask["mse"].detach(),
                    "cos": align_subtask["cos"].detach(),
                    "total": align_subtask["total"].detach(),
                },
            },
        }

        if self.return_debug_tensors:
            outputs.update(
                {
                    "qwen_last_hidden": qwen_last_hidden,
                    "qwen_slot_hidden": {
                        "dynamic": qwen_dynamic_slot_hidden,
                        "spatial": qwen_spatial_slot_hidden,
                        "subtask": qwen_subtask_slot_hidden,
                    },
                    "qwen_slot_fused": {
                        "dynamic": qwen_dynamic_slot_fused,
                        "spatial": qwen_spatial_slot_fused,
                        "subtask": qwen_subtask_slot_fused,
                    },
                }
            )

        return outputs

    @torch.inference_mode()
    def generate_with_action_routing(
        self,
        examples: List[dict],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        skip_teacher_slots: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressively generate action tokens with dynamic action-type routing."""
        if not self.enable_fast_action_loss or self.fast_action_model is None:
            raise RuntimeError(
                "generate_with_action_routing requires fast_action_loss enabled and fast_action_model initialized."
            )

        batch_size = len(examples)
        batch_images = [example["image"] for example in examples]
        instructions = [example.get("lang", example.get("instruction", "")) for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        suffix = self._slot_prompt_suffix()
        instructions = [text + suffix for text in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=None,
        )
        input_ids = qwen_inputs["input_ids"]

        dyn_pos = self._gather_positions(input_ids, self.dynamic_token_id, self.k_dyn)
        spa_pos = self._gather_positions(input_ids, self.spatial_token_id, self.k_spa)
        sub_pos = self._gather_positions(input_ids, self.subtask_token_id, self.k_sub)
        enable_action_tokens = bool(getattr(self, "enable_action_tokens", False))
        num_action_tokens = int(getattr(self, "num_action_tokens", 0))
        action_token_id = getattr(self, "action_token_id", None)
        action_pos = None
        action_mask = None
        if enable_action_tokens and action_token_id is not None and num_action_tokens > 0:
            action_pos = self._gather_positions(input_ids, action_token_id, num_action_tokens)
            action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            action_mask.scatter_(1, action_pos, True)

        sequence_layout_pre_injection = self._build_sequence_layout_pre_injection(
            input_ids=input_ids,
            dyn_pos=dyn_pos,
            spa_pos=spa_pos,
            sub_pos=sub_pos,
            action_mask=action_mask,
        )
        token_type_ids = self._build_token_type_ids(
            input_ids,
            dyn_pos=dyn_pos,
            spa_pos=spa_pos,
            sub_pos=sub_pos,
            action_pos=action_pos,
        )
        if skip_teacher_slots:
            slot_device = input_ids.device
            dynamic_slots = torch.zeros((batch_size, self.k_dyn, self.hidden_size), device=slot_device)
            spatial_slots = torch.zeros((batch_size, self.k_spa, self.hidden_size), device=slot_device)
            subtask_slots = torch.zeros((batch_size, self.k_sub, self.hidden_size), device=slot_device)
            keep_dynamic = torch.zeros((batch_size, self.k_dyn), dtype=torch.bool, device=slot_device)
            keep_spatial = torch.zeros((batch_size, self.k_spa), dtype=torch.bool, device=slot_device)
            keep_subtask = torch.zeros((batch_size, self.k_sub), dtype=torch.bool, device=slot_device)
        else:
            dynamic_slots = self._build_dynamic_slots(self.dynamic_teacher_encoder(examples=examples))
            spatial_slots = self._build_spatial_slots(self.spatial_teacher_encoder(examples=examples))
            subtask_slots = self._build_subtask_slots(self.subtask_slot_encoder(examples=examples))

            if self.enable_slot_mask and self.slot_mask_apply_in_eval:
                masked = self._apply_slot_masks(
                    dynamic_slots,
                    spatial_slots,
                    subtask_slots,
                    inner_mask_ratios=self.inner_mask_ratios,
                    outside_num_masked_probs=self.outside_num_masked_probs,
                )
                dynamic_slots = masked["dynamic_slots"]
                spatial_slots = masked["spatial_slots"]
                subtask_slots = masked["subtask_slots"]
                keep_dynamic = masked["keep_dynamic"]
                keep_spatial = masked["keep_spatial"]
                keep_subtask = masked["keep_subtask"]
            else:
                keep_dynamic = torch.ones((dynamic_slots.shape[0], self.k_dyn), dtype=torch.bool, device=dynamic_slots.device)
                keep_spatial = torch.ones((spatial_slots.shape[0], self.k_spa), dtype=torch.bool, device=spatial_slots.device)
                keep_subtask = torch.ones((subtask_slots.shape[0], self.k_sub), dtype=torch.bool, device=subtask_slots.device)

        current_input_ids = input_ids.clone()
        current_attention_mask = qwen_inputs["attention_mask"].clone() if "attention_mask" in qwen_inputs else torch.ones_like(input_ids)
        current_token_type_ids = token_type_ids.clone()
        generated_token_ids = []

        action_token_ids_tensor = None
        if self.fast_action_token_ids:
            action_token_ids_tensor = torch.tensor(
                sorted(self.fast_action_token_ids),
                device=current_input_ids.device,
                dtype=current_input_ids.dtype,
            )
        action_started = torch.zeros((batch_size,), dtype=torch.bool, device=current_input_ids.device)
        action_finished = torch.zeros((batch_size,), dtype=torch.bool, device=current_input_ids.device)

        def inject_slot_hook(_module, _inputs, output):
            batch_idx_dyn = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_dyn)
            batch_idx_spa = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_spa)
            batch_idx_sub = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_sub)
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

            action_token_queries = getattr(self, "action_token_queries", None)
            if action_pos is not None and action_token_queries is not None:
                action_queries = action_token_queries.to(device=output.device, dtype=output.dtype)
                action_queries = action_queries.unsqueeze(0).expand(batch_size, -1, -1)
                output[batch_idx_act, action_pos, :] = action_queries
            return output

        embedding_layer = self._resolve_input_embeddings_layer()
        hook_handle = embedding_layer.register_forward_hook(inject_slot_hook)
        try:
            for _ in range(max_new_tokens):
                self._set_typed_ffn_token_type_ids(current_token_type_ids)
                if self.enable_back_half_typed_attention:
                    current_action_mask = current_token_type_ids.eq(TypedResidualFFN.TYPE_IDS["action"])
                    current_layout = self._build_sequence_layout_pre_injection(
                        input_ids=current_input_ids,
                        dyn_pos=dyn_pos,
                        spa_pos=spa_pos,
                        sub_pos=sub_pos,
                        action_mask=current_action_mask,
                    )
                    current_self_attention_mask = self._build_self_attention_mask(current_layout)
                    self._set_back_half_typed_attention_mask(current_self_attention_mask)
                model_inputs = dict(qwen_inputs)
                model_inputs["input_ids"] = current_input_ids
                model_inputs["attention_mask"] = current_attention_mask
                outputs = self.qwen_vl_interface(
                    **model_inputs,
                    output_hidden_states=False,
                    return_dict=True,
                )

                logits = outputs.logits[:, -1, :]
                if action_token_ids_tensor is not None:
                    allowed_mask = torch.zeros_like(logits, dtype=torch.bool)
                    allowed_mask[:, action_token_ids_tensor] = True
                    logits = logits.masked_fill(~allowed_mask, float("-inf"))
                logits = logits / max(temperature, 1e-6)
                probs = torch.nn.functional.softmax(logits, dim=-1)

                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cumsum_probs > top_p
                nucleus_mask[..., 0] = False
                sorted_probs[nucleus_mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                if do_sample:
                    next_token_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
                    next_token_id = sorted_indices.gather(-1, next_token_idx.unsqueeze(-1)).squeeze(-1)
                else:
                    next_token_id = logits.argmax(dim=-1)
                generated_token_ids.append(next_token_id.detach().cpu())

                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones((batch_size, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device),
                    ],
                    dim=-1,
                )

                new_token_type = torch.zeros(
                    (batch_size, 1),
                    dtype=current_token_type_ids.dtype,
                    device=current_token_type_ids.device,
                )
                if action_token_ids_tensor is not None:
                    is_action = (next_token_id.unsqueeze(-1) == action_token_ids_tensor.view(1, -1)).any(dim=-1)
                    new_token_type[is_action] = TypedResidualFFN.TYPE_IDS["action"]
                    action_finished = action_finished | (action_started & (~is_action))
                    action_started = action_started | is_action

                current_token_type_ids = torch.cat([current_token_type_ids, new_token_type], dim=-1)
                if bool(action_finished.all()):
                    break
        finally:
            hook_handle.remove()
            self._set_typed_ffn_token_type_ids(None)
            if self.enable_back_half_typed_attention:
                self._set_back_half_typed_attention_mask(None)

        return {
            "input_ids": current_input_ids,
            "generated_token_ids": (
                torch.stack(generated_token_ids, dim=-1)
                if generated_token_ids
                else torch.empty((batch_size, 0), dtype=torch.long)
            ),
            "token_type_ids": current_token_type_ids,
        }

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        results = []
        for batch_idx in range(generated_ids.size(0)):
            seq = generated_ids[batch_idx]
            is_action = (seq >= act_min) & (seq <= act_max)
            idx = is_action.nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            start = int(idx[0].item())
            end = start
            seq_len = seq.shape[0]
            while end < seq_len and bool(is_action[end]):
                end += 1
            results.append(seq[start:end].tolist())
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[List[int] | None]:
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            batch_fast_token_ids.append([token_id - act_min for token_id in seq])
        return batch_fast_token_ids

    def _trim_fast_token_prefixes_to_valid_actions(
        self,
        batch_fast_token_ids: List[List[int] | None],
    ) -> List[List[int] | None]:
        processor = self.fast_action_model.fast_tokenizer
        bpe_tokenizer = processor.bpe_tokenizer
        target_chars = int(self.num_actions_chunk) * int(self.action_dim)

        trimmed_batch = []
        for seq in batch_fast_token_ids:
            if not seq:
                trimmed_batch.append(None)
                continue

            valid_prefix = None
            for end in range(1, len(seq) + 1):
                decoded_text = bpe_tokenizer.decode(seq[:end])
                char_count = len(decoded_text)
                if char_count == target_chars:
                    valid_prefix = seq[:end]
                    break
                if char_count > target_chars:
                    break

            trimmed_batch.append(valid_prefix)

        return trimmed_batch

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        if examples is None:
            raise ValueError("QwenMyVLA.predict_action expects examples: List[dict]")

        if not self.enable_stage3_action_header:
            if kwargs.get("fast_inference", False):
                generated = self.generate_with_action_routing(
                    examples=examples,
                    max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
                    temperature=float(kwargs.get("temperature", 0.7)),
                    top_p=float(kwargs.get("top_p", 0.9)),
                    do_sample=bool(kwargs.get("do_sample", False)),
                    skip_teacher_slots=bool(kwargs.get("skip_teacher_slots", True)),
                )
                batch_vlm_action_token_ids = self._extract_action_token_ids(generated["generated_token_ids"].to(self.qwen_vl_interface.model.device))
                batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
                batch_fast_action_token_idx = self._trim_fast_token_prefixes_to_valid_actions(batch_fast_action_token_idx)
                if any(seq is None for seq in batch_fast_action_token_idx):
                    raise RuntimeError(
                        "FAST inference did not produce a valid token prefix that decodes to "
                        f"({self.num_actions_chunk}, {self.action_dim}) actions."
                    )
                normalized_actions = self.fast_action_model.fast_tokenizer.decode(batch_fast_action_token_idx)
                return {
                    "normalized_actions": normalized_actions,
                    "generated_token_ids": generated["generated_token_ids"].numpy(),
                }
            outputs = self.forward(examples=examples)
            return {
                "normalized_actions": outputs["predicted_actions"].detach().cpu().numpy(),
            }

        batch_images = [example["image"] for example in examples]
        instructions = [example.get("lang", example.get("instruction", "")) for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [text + self._slot_prompt_suffix() for text in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = qwen_inputs["input_ids"]

        dyn_pos = self._gather_positions(input_ids, self.dynamic_token_id, self.k_dyn)
        spa_pos = self._gather_positions(input_ids, self.spatial_token_id, self.k_spa)
        sub_pos = self._gather_positions(input_ids, self.subtask_token_id, self.k_sub)
        enable_action_tokens = bool(getattr(self, "enable_action_tokens", False))
        num_action_tokens = int(getattr(self, "num_action_tokens", 0))
        action_token_id = getattr(self, "action_token_id", None)
        action_pos = None
        action_mask = None
        if enable_action_tokens and action_token_id is not None and num_action_tokens > 0:
            action_pos = self._gather_positions(input_ids, action_token_id, num_action_tokens)
            action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            action_mask.scatter_(1, action_pos, True)

        sequence_layout_pre_injection = self._build_sequence_layout_pre_injection(
            input_ids=input_ids,
            dyn_pos=dyn_pos,
            spa_pos=spa_pos,
            sub_pos=sub_pos,
            action_mask=action_mask,
        )
        token_type_ids = self._build_token_type_ids(
            input_ids,
            dyn_pos=dyn_pos,
            spa_pos=spa_pos,
            sub_pos=sub_pos,
            action_pos=action_pos,
        )
        self_attention_mask = self._build_self_attention_mask(sequence_layout_pre_injection=sequence_layout_pre_injection)

        batch_size = input_ids.shape[0]

        self._set_typed_ffn_token_type_ids(token_type_ids)
        if self.enable_back_half_typed_attention:
            self._set_back_half_typed_attention_mask(self_attention_mask)

        dynamic_slots = self._build_dynamic_slots(self.dynamic_teacher_encoder(examples=examples))
        spatial_slots = self._build_spatial_slots(self.spatial_teacher_encoder(examples=examples))
        subtask_slots = self._build_subtask_slots(self.subtask_slot_encoder(examples=examples))

        if self.enable_stage3_action_header and self.enable_slot_mask:
            masked = self._apply_slot_masks(
                dynamic_slots,
                spatial_slots,
                subtask_slots,
                inner_mask_ratios=self.inner_mask_ratios,
                outside_num_masked_probs=self.outside_num_masked_probs,
            )
            dynamic_slots = masked["dynamic_slots"]
            spatial_slots = masked["spatial_slots"]
            subtask_slots = masked["subtask_slots"]
            keep_dynamic = masked["keep_dynamic"]
            keep_spatial = masked["keep_spatial"]
            keep_subtask = masked["keep_subtask"]
        else:
            keep_dynamic = torch.ones((dynamic_slots.shape[0], self.k_dyn), dtype=torch.bool, device=dynamic_slots.device)
            keep_spatial = torch.ones((spatial_slots.shape[0], self.k_spa), dtype=torch.bool, device=spatial_slots.device)
            keep_subtask = torch.ones((subtask_slots.shape[0], self.k_sub), dtype=torch.bool, device=subtask_slots.device)

        def inject_slot_hook(_module, _inputs, output):
            batch_idx_dyn = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_dyn)
            batch_idx_spa = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_spa)
            batch_idx_sub = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.k_sub)
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

            action_token_queries = getattr(self, "action_token_queries", None)
            if action_pos is not None and action_token_queries is not None:
                action_queries = action_token_queries.to(device=output.device, dtype=output.dtype)
                action_queries = action_queries.unsqueeze(0).expand(batch_size, -1, -1)
                output[batch_idx_act, action_pos, :] = action_queries
            return output

        embedding_layer = self._resolve_input_embeddings_layer()
        hook_handle = embedding_layer.register_forward_hook(inject_slot_hook)
        try:
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
            with autocast_ctx:
                qwen_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()
            self._set_typed_ffn_token_type_ids(None)
            if self.enable_back_half_typed_attention:
                self._set_back_half_typed_attention_mask(None)

        stage3_action_pos = self._build_stage3_action_positions_with_optional_action_tokens(
            dyn_pos=dyn_pos,
            spa_pos=spa_pos,
            sub_pos=sub_pos,
            action_pos=action_pos,
        )
        stage3_condition = self._build_stage3_m1_condition(
            hidden_states=list(qwen_outputs.hidden_states),
            batch_images=batch_images,
            sequence_layout_pre_injection=sequence_layout_pre_injection,
            action_pos=stage3_action_pos,
        )

        cfg_scale = float(kwargs.get("cfg_scale", 1.5))
        use_ddim = bool(kwargs.get("use_ddim", True))
        num_ddim_steps = kwargs.get("num_ddim_steps", 5)
        normalized_actions = self._sample_stage3_actions(
            stage3_condition=stage3_condition,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
        )

        return {
            "normalized_actions": normalized_actions.detach().cpu().numpy(),
        }

if __name__ == "__main__":
    from starVLA.model.framework.qwen_myvla_smoke import main as smoke_main

    smoke_main()
