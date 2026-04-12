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

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model as get_fast_action_model
from starVLA.model.modules.action_model.VLA_AdapterHeader import get_action_model as get_stage3_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.QWen3 import IMAGE_TOKEN_INDEX
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

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

    def __init__(self, base_mlp: nn.Module) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.base_mlp.requires_grad_(False)

        self.experts = nn.ModuleDict({
            "subtask": copy.deepcopy(base_mlp),
            "dynamic": copy.deepcopy(base_mlp),
            "spatial": copy.deepcopy(base_mlp),
            "action": copy.deepcopy(base_mlp),
        })
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
        spa_in = int(self.config.framework.depth_encoder.get("out_channels", [256, 512, 1024, 1024])[-1])
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
        self.stage_name = str(getattr(self.config.trainer, "stage_name", "")).lower()
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
        self.coarse_action_state_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
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
        self.stage3_action_query_num = int(
            stage3_cfg.get(
                "action_query_num",
                action_cfg.get("action_query_num", self.k_sub + self.k_dyn + self.k_spa),
            )
        )
        self.stage3_action_head = None
        if self.enable_stage3_action_header:
            self.stage3_action_head = get_stage3_action_model(config=self.config)
            if hasattr(self.stage3_action_head, "action_query_num"):
                self.stage3_action_query_num = int(self.stage3_action_head.action_query_num)

        fast_cfg = self.config.framework.get("fast_action_loss", {})
        self.enable_fast_action_loss = bool(fast_cfg.get("enable", False))
        self.fast_action_loss_weight = float(fast_cfg.get("weight", 1.0))
        self.fast_action_loss_weight_start = float(loss_schedule_cfg.get("fast_action_start", self.fast_action_loss_weight))
        self.fast_action_loss_weight_end = float(loss_schedule_cfg.get("fast_action_end", self.fast_action_loss_weight))
        self.fast_action_loss_detach = bool(fast_cfg.get("detach", False))
        self.fast_action_model = None
        if self.enable_fast_action_loss:
            self.fast_action_model = get_fast_action_model(config=self.config)

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
            layer.mlp = TypedResidualFFN(layer.mlp)
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
        global_step = kwargs.get("global_step")
        max_train_steps = kwargs.get("max_train_steps")
        if global_step is None or max_train_steps is None:
            return 0.0
        max_train_steps = max(1, int(max_train_steps))
        return max(0.0, min(1.0, float(global_step) / float(max_train_steps)))

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
        """Compute continuous branch loss with optional flow matching objective.

        objective=l1:
            plain supervised L1 on predicted x1.
        objective=fm:
            conditional flow matching in action space, supervising velocity:
                x_t = (1-t) * x0 + t * x1,   u_t = x1 - x0
            We use model predicted x1_hat to form u_hat = x1_hat - x0.
        """
        objective = self.continuous_fm_objective
        if objective == "l1":
            return F.l1_loss(predicted_actions, gt_actions)

        if objective != "fm":
            raise ValueError(f"Unsupported continuous_fm_loss.objective: {objective}")

        x1 = gt_actions
        x0 = torch.randn_like(x1) * self.continuous_fm_noise_scale

        batch = x1.shape[0]
        t = torch.rand((batch, 1, 1), device=x1.device, dtype=x1.dtype)
        if self.continuous_fm_t_eps > 0:
            t = t.clamp(min=self.continuous_fm_t_eps, max=1.0 - self.continuous_fm_t_eps)

        _xt = (1.0 - t) * x0 + t * x1
        target_u = x1 - x0
        pred_u = predicted_actions - x0

        if self.continuous_fm_loss_type == "l1":
            return F.l1_loss(pred_u, target_u)
        if self.continuous_fm_loss_type == "mse":
            return F.mse_loss(pred_u, target_u)
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

    def _build_stage3_coarse_condition(
        self,
        qwen_last_hidden: torch.Tensor,
        action_pos: Optional[torch.Tensor],
        predicted_actions_light: torch.Tensor,
    ) -> torch.Tensor:
        # Preferred source: hidden states at action-token positions (coarse discrete path).
        if action_pos is not None and action_pos.numel() > 0:
            action_hidden = self._gather_slot_hidden(qwen_last_hidden, action_pos)
            return action_hidden.mean(dim=1)

        # Fallback source: project light coarse action chunks into hidden space.
        projected = self.coarse_action_state_proj(predicted_actions_light)
        return projected.mean(dim=1)

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
        training_progress = self._resolve_training_progress(kwargs)

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

        predicted_actions_stage3 = None
        stage3_coarse_condition = None
        if self.enable_stage3_action_header:
            if self.stage3_action_head is None:
                raise RuntimeError("stage3_action_head is not initialized while stage3_action_header is enabled.")

            stage3_action_pos = self._build_stage3_action_positions_with_optional_action_tokens(
                dyn_pos=dyn_pos,
                spa_pos=spa_pos,
                sub_pos=sub_pos,
                action_pos=action_pos,
            )
            stage3_hidden, stage3_vision_len = self._build_stage3_multilayer_hidden(
                hidden_states=list(qwen_outputs.hidden_states),
                image_positions=sequence_layout_pre_injection["image"]["positions"],
                action_positions=stage3_action_pos,
            )
            self.stage3_action_head = self.stage3_action_head.to(device=stage3_hidden.device, dtype=stage3_hidden.dtype)
            stage3_coarse_condition = self._build_stage3_coarse_condition(
                qwen_last_hidden=qwen_last_hidden,
                action_pos=action_pos,
                predicted_actions_light=predicted_actions_light,
            )
            predicted_actions_stage3 = self.stage3_action_head.predict_action(
                stage3_hidden,
                vision_hidden_len=stage3_vision_len,
                state_projected=stage3_coarse_condition,
                phase="Training" if self.training else "Inference",
            )

        predicted_actions = predicted_actions_stage3 if predicted_actions_stage3 is not None else predicted_actions_light

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

        if self.enable_fast_action_loss:
            fast_action_loss = self._compute_fast_action_loss(
                examples=examples,
                batch_images=batch_images,
                instructions=instructions,
            )
            fast_action_loss_for_optim = fast_action_loss.detach() if self.fast_action_loss_detach else fast_action_loss
        else:
            fast_action_loss = torch.zeros((), device=loss_device, dtype=torch.float32)
            fast_action_loss_for_optim = fast_action_loss

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
            "predicted_actions": predicted_actions,
            "predicted_actions_light": predicted_actions_light.detach(),
            "predicted_actions_stage3": (
                predicted_actions_stage3.detach() if predicted_actions_stage3 is not None else None
            ),
            "stage3_coarse_condition": (
                stage3_coarse_condition.detach() if stage3_coarse_condition is not None else None
            ),
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
    def predict_action(self, examples: List[dict] = None, **kwargs):
        if examples is None:
            raise ValueError("QwenMyVLA.predict_action expects examples: List[dict]")

        outputs = self.forward(examples=examples)
        return {
            "normalized_actions": outputs["predicted_actions"].detach().cpu().numpy(),
        }

if __name__ == "__main__":
    from starVLA.model.framework.qwen_myvla_smoke import main as smoke_main

    smoke_main()
