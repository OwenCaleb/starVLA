"""Smoke tests for QwenMyVLA.

This file intentionally keeps smoke logic out of QwenMyVLA runtime implementation
so the framework file stays focused on train/infer behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.model.framework.QwenMyVLA import (
    BackHalfTypedSelfAttention,
    QwenMyVLA,
    SlotMLPFuser,
    TokenAwareResampler,
    TypedResidualFFN,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def _build_mock_model(*, enable_fast_loss: bool, fast_detach: bool) -> QwenMyVLA:
    hidden_size = 16
    k_dyn, k_spa, k_sub = 2, 3, 2
    action_dim, chunk_len = 7, 4

    model = QwenMyVLA.__new__(QwenMyVLA)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        framework={},
        datasets=SimpleNamespace(vla_data=SimpleNamespace(image_size=None)),
        trainer={},
    )
    model.hidden_size = hidden_size
    model.k_dyn = k_dyn
    model.k_spa = k_spa
    model.k_sub = k_sub
    model.dynamic_placeholder = "<slot_dynamic>"
    model.spatial_placeholder = "<slot_spatial>"
    model.subtask_placeholder = "<slot_subtask>"
    model.action_placeholder = "<slot_action>"
    model.slot_boundary_tokens = {
        "dynamic_start": "<SOSD>",
        "dynamic_end": "<EOSD>",
        "spatial_start": "<SOSS>",
        "spatial_end": "<EOSS>",
        "subtask_start": "<SOST>",
        "subtask_end": "<EOST>",
        "action_start": "<SOSA>",
        "action_end": "<EOSA>",
    }
    model.dynamic_token_id = 10
    model.spatial_token_id = 11
    model.subtask_token_id = 12
    model.image_token_id = 99
    model.attention_visibility_rules = None
    model.enable_back_half_typed_attention = False
    model.attn_wrapped_layers = []
    model._moe_token_type_ids = None
    model.moe_wrapped_layers = []

    model.enable_slot_mask = False
    model.slot_mask_apply_in_eval = False
    model.inner_mask_ratios = {"dynamic": 0.0, "spatial": 0.0, "subtask": 0.0}
    model.outside_num_masked_probs = [1.0, 0.0, 0.0, 0.0]
    model.enable_slot_mask_curriculum = False
    model.slot_mask_curriculum_progress_start = 0.0
    model.slot_mask_curriculum_progress_end = 1.0
    model.slot_mask_inner_start = {"dynamic": 0.0, "spatial": 0.0, "subtask": 0.0}
    model.slot_mask_inner_end = {"dynamic": 0.0, "spatial": 0.0, "subtask": 0.0}
    model.slot_mask_outside_start_probs = [1.0, 0.0, 0.0, 0.0]
    model.slot_mask_outside_end_probs = [1.0, 0.0, 0.0, 0.0]

    model.enable_slot_align_loss = True
    model.slot_align_loss_weights = {"dynamic": 1.0, "spatial": 1.0, "subtask": 1.0}
    model.slot_align_loss_mse_ratio = 1.0
    model.slot_align_loss_cos_ratio = 1.0
    model.slot_align_masked_target_prediction = False
    model.slot_align_masked_only = True
    model.slot_align_unmasked_weight = 0.0
    model.slot_align_fallback_to_all_when_no_masked = True

    model.enable_fast_action_loss = enable_fast_loss
    model.fast_action_loss_weight = 1.0
    model.fast_action_loss_weight_start = 1.0
    model.fast_action_loss_weight_end = 1.0
    model.fast_action_loss_detach = fast_detach
    model.fast_action_token_ids = set()

    class _FakeFastActionModel:
        def __init__(self):
            self.fast_tokenizer = SimpleNamespace(time_horizon=None, action_dim=None)

        def encoder_action2fastoken(self, actions):
            return [[1, 2, 3, 4] for _ in actions]

    model.fast_action_model = _FakeFastActionModel() if enable_fast_loss else None

    model.continuous_fm_loss_weight = 1.0
    model.continuous_fm_loss_weight_start = 1.0
    model.continuous_fm_loss_weight_end = 1.0
    model.continuous_fm_loss_detach = False
    model.continuous_fm_objective = "fm"
    model.continuous_fm_noise_scale = 1.0
    model.continuous_fm_t_eps = 1e-3
    model.continuous_fm_loss_type = "mse"
    model.enable_loss_weight_schedule = False
    model.loss_weight_schedule_progress_start = 0.0
    model.loss_weight_schedule_progress_end = 1.0
    model.slot_align_weight_start = {"dynamic": 1.0, "spatial": 1.0, "subtask": 1.0}
    model.slot_align_weight_end = {"dynamic": 1.0, "spatial": 1.0, "subtask": 1.0}

    model.enable_stage3_action_header = False
    model.stage3_action_query_num = k_sub + k_dyn + k_spa
    model.stage3_action_head = None

    model.return_debug_tensors = False
    model.strict_loss_checks = True

    model.action_dim = action_dim
    model.num_actions_chunk = chunk_len
    model.action_predictor = nn.Sequential(
        nn.LayerNorm(hidden_size * 3),
        nn.Linear(hidden_size * 3, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, chunk_len * action_dim),
    )

    model.dynamic_projector = TokenAwareResampler(in_dim=8, out_dim=hidden_size, num_slots=k_dyn)
    model.spatial_compressor = TokenAwareResampler(in_dim=8, out_dim=hidden_size, num_slots=k_spa)
    model.subtask_compressor = TokenAwareResampler(in_dim=8, out_dim=hidden_size, num_slots=k_sub)

    model.dynamic_slot_fuser = SlotMLPFuser(hidden_size=hidden_size, num_blocks=2, mlp_ratio=2.0)
    model.spatial_slot_fuser = SlotMLPFuser(hidden_size=hidden_size, num_blocks=2, mlp_ratio=2.0)
    model.subtask_slot_fuser = SlotMLPFuser(hidden_size=hidden_size, num_blocks=2, mlp_ratio=2.0)

    class _FakeTokenizer:
        def __call__(self, token: str, add_special_tokens: bool = False):
            mapping = {
                "<slot_dynamic>": [10],
                "<slot_spatial>": [11],
                "<slot_subtask>": [12],
                "<slot_action>": [13],
                "<SOSD>": [14],
                "<EOSD>": [15],
                "<SOSS>": [16],
                "<EOSS>": [17],
                "<SOST>": [18],
                "<EOST>": [19],
                "<SOSA>": [20],
                "<EOSA>": [21],
            }
            return {"input_ids": mapping[token]}

    class _FakeProcessor:
        def __init__(self):
            self.tokenizer = _FakeTokenizer()

    class _FakeTextModel(nn.Module):
        def __init__(self, h: int):
            super().__init__()
            self.embed = nn.Embedding(256, h)
            self.layers = nn.ModuleList([])

        def get_input_embeddings(self):
            return self.embed

    class _FakeOuterModel(nn.Module):
        def __init__(self, h: int):
            super().__init__()
            self.model = _FakeTextModel(h)
            self.config = SimpleNamespace(hidden_size=h)

    class _FakeQwenInterface(nn.Module):
        def __init__(self, h: int, ks: int, kd: int, kp: int):
            super().__init__()
            self.processor = _FakeProcessor()
            self.model = _FakeOuterModel(h)
            self.k_sub = ks
            self.k_dyn = kd
            self.k_spa = kp
            self.image_token_id = 99

        def build_qwenvl_inputs(self, images, instructions, solutions=None):
            del instructions, solutions
            batch_size = len(images)
            seq_len = 24
            input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
            input_ids[:, 3:5] = self.image_token_id
            input_ids[:, 10 : 10 + self.k_sub] = 12
            input_ids[:, 10 + self.k_sub : 10 + self.k_sub + self.k_dyn] = 10
            input_ids[:, 10 + self.k_sub + self.k_dyn : 10 + self.k_sub + self.k_dyn + self.k_spa] = 11
            return {"input_ids": input_ids}

        def forward(self, **kwargs):
            input_ids = kwargs["input_ids"]
            hidden = self.model.model.get_input_embeddings()(input_ids)
            return SimpleNamespace(hidden_states=[hidden, hidden], loss=torch.tensor(0.5, device=hidden.device, dtype=hidden.dtype))

    model.qwen_vl_interface = _FakeQwenInterface(hidden_size, k_sub, k_dyn, k_spa)
    model.dynamic_teacher_encoder = lambda examples: {
        "indices": torch.zeros((len(examples), 1), dtype=torch.long),
        "z_q": torch.randn((len(examples), 4, 8)),
    }
    model.spatial_teacher_encoder = lambda examples: {
        "encoder_tokens": torch.randn((len(examples), 6, 8)),
        "token_mask": torch.ones((len(examples), 6), dtype=torch.bool),
    }
    model.subtask_slot_encoder = lambda examples: {
        "text_tokens": torch.randn((len(examples), 5, 8)),
        "text_mask": torch.ones((len(examples), 5), dtype=torch.bool),
    }

    return model


def train_predict_contract_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=False, fast_detach=False)
    samples = [{"image": [torch.zeros(3, 8, 8)], "lang": "pick up the cup", "action": np.zeros((4, 7), dtype=np.float32)}]

    model.train()
    out_train = model.forward(examples=samples)
    assert "action_loss" in out_train and out_train["action_loss"].ndim == 0

    model.eval()
    out_pred = model.predict_action(examples=samples)
    assert "normalized_actions" in out_pred
    print("[OK] QwenMyVLA train/predict contract smoke passed")


def loss_branch_semantics_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=True, fast_detach=True)
    model._compute_fast_action_loss = lambda examples, batch_images, instructions: torch.tensor(0.25)

    samples = [{"image": [torch.zeros(3, 8, 8)], "lang": "pick up the cup", "action": np.zeros((4, 7), dtype=np.float32)}]
    out = model.forward(examples=samples)

    assert bool(out["continuous_fm_loss_detach"]) is False
    assert bool(out["fast_action_loss_detach"]) is True
    print("[OK] QwenMyVLA loss-branch semantics smoke passed")


def typed_attention_mask_smoke_test() -> None:
    batch_size = 1
    seq_len = 12
    device = torch.device("cpu")

    def span_from_positions(positions):
        pos = torch.tensor([positions], dtype=torch.long, device=device)
        return {
            "start": torch.tensor([positions[0]], dtype=torch.long, device=device),
            "end": torch.tensor([positions[-1]], dtype=torch.long, device=device),
            "length": torch.tensor([len(positions)], dtype=torch.long, device=device),
            "positions": pos,
            "mask": torch.ones((batch_size,), dtype=torch.bool, device=device),
        }

    layout = {
        "text": span_from_positions([0, 1, 2]),
        "image": span_from_positions([3, 4]),
        "subtask": span_from_positions([5, 6]),
        "dynamic": span_from_positions([7, 8]),
        "spatial": span_from_positions([9, 10, 11]),
    }

    mask = QwenMyVLA._build_typed_attention_mask_from_layout(layout, causal=True, return_bias=False)
    assert mask.shape == (batch_size, seq_len, seq_len)
    assert bool(mask[0, 7, 1].item()) is True
    assert bool(mask[0, 7, 4].item()) is False
    print("[OK] Typed attention mask smoke passed")


def all_layer_typed_attention_apply_smoke_test() -> None:
    class _CaptureSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_attention_mask = None

        def forward(self, hidden_states, attention_mask=None, **kwargs):
            del kwargs
            self.last_attention_mask = attention_mask
            return (hidden_states,)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _CaptureSelfAttn()
            self.mlp = nn.Identity()

    class _TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(4)])

    model = QwenMyVLA.__new__(QwenMyVLA)
    nn.Module.__init__(model)
    model.qwen_vl_interface = SimpleNamespace(model=SimpleNamespace(model=_TextModel()))

    model._attach_back_half_typed_attention()
    typed_keep = torch.ones((1, 6, 6), dtype=torch.bool)
    typed_keep[:, :, 4:] = False
    model._set_back_half_typed_attention_mask(typed_keep)

    layers = model.qwen_vl_interface.model.model.layers
    hidden = torch.randn(1, 6, 8)
    base_pad_mask = torch.ones((1, 6), dtype=torch.long)
    _ = layers[0].self_attn(hidden, attention_mask=base_pad_mask)

    got = layers[0].self_attn.base_self_attn.last_attention_mask
    assert got is not None and got.ndim == 4
    assert float(got[0, 0, 0, 4].item()) < -1e20
    print("[OK] QwenMyVLA all-layer typed attention apply smoke passed")


def attn_implementation_guard_smoke_test() -> None:
    def _build_model(attn_impl: str) -> QwenMyVLA:
        model = QwenMyVLA.__new__(QwenMyVLA)
        nn.Module.__init__(model)
        text_cfg = SimpleNamespace(_attn_implementation=attn_impl)
        root_cfg = SimpleNamespace(_attn_implementation=attn_impl, text_config=text_cfg)
        model.qwen_vl_interface = SimpleNamespace(model=SimpleNamespace(config=root_cfg))
        return model

    flash_model = _build_model("flash_attention_2")
    assert flash_model._is_flash_attention_2_active() is True

    eager_model = _build_model("eager")
    assert eager_model._is_flash_attention_2_active() is False

    sdpa_model = _build_model("sdpa")
    assert sdpa_model._is_flash_attention_2_active() is False

    enable_back_half_typed_attention = True
    if enable_back_half_typed_attention and flash_model._is_flash_attention_2_active():
        enable_back_half_typed_attention = False
    assert enable_back_half_typed_attention is False

    enable_back_half_typed_attention = True
    if enable_back_half_typed_attention and sdpa_model._is_flash_attention_2_active():
        enable_back_half_typed_attention = False
    assert enable_back_half_typed_attention is True
    print("[OK] Attention implementation guard smoke passed")


def masked_target_prediction_slot_align_smoke_test() -> None:
    model = QwenMyVLA.__new__(QwenMyVLA)
    nn.Module.__init__(model)
    model.slot_align_loss_mse_ratio = 1.0
    model.slot_align_loss_cos_ratio = 0.0
    model.slot_align_masked_target_prediction = True
    model.slot_align_masked_only = True
    model.slot_align_unmasked_weight = 0.0
    model.slot_align_fallback_to_all_when_no_masked = True

    target = torch.zeros((1, 2, 3), dtype=torch.float32)
    pred = target.clone()
    pred[:, 0, :] = 5.0
    keep_mask = torch.tensor([[True, False]], dtype=torch.bool)

    out = model._compute_slot_alignment_loss(pred, target, keep_mask=keep_mask)
    assert float(out["mse"].item()) == 0.0
    print("[OK] QwenMyVLA masked target prediction slot-align smoke passed")


def main_forward_action_group_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=False, fast_detach=False)
    model.enable_action_tokens = True
    model.action_placeholder = "<slot_action>"
    model.num_action_tokens = 2
    model.action_token_id = 13
    model.action_token_queries = nn.Parameter(torch.randn(model.num_action_tokens, model.hidden_size) * 0.02)

    orig_build_inputs = model.qwen_vl_interface.build_qwenvl_inputs

    def _build_inputs_with_action(*args, **kwargs):
        out = orig_build_inputs(*args, **kwargs)
        input_ids = out["input_ids"]
        input_ids[:, 17:19] = model.action_token_id
        return out

    model.qwen_vl_interface.build_qwenvl_inputs = _build_inputs_with_action

    samples = [{"image": [torch.zeros(3, 8, 8)], "lang": "pick up the cup", "action": np.zeros((4, 7), dtype=np.float32)}]
    out = model.forward(examples=samples)

    action_pos = out["slot_positions"]["action"]
    assert action_pos is not None and action_pos.shape[1] == model.num_action_tokens

    action_type_id = int(TypedResidualFFN.TYPE_IDS["action"])
    token_types = out["token_type_ids_for_moe"]
    assert int((token_types == action_type_id).sum().item()) >= model.num_action_tokens

    action_span = out["sequence_layout_pre_injection"].get("action", None)
    assert action_span is not None
    assert int(action_span["length"][0].item()) == model.num_action_tokens
    print("[OK] QwenMyVLA main forward action-group smoke passed")


def freeze_policy_smoke_test() -> None:
    class _DummySelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            del kwargs
            return (self.proj(hidden_states),)

    class _DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = BackHalfTypedSelfAttention(_DummySelfAttn())
            self.mlp = TypedResidualFFN(nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)))

    class _DummyQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_DummyLayer()])
            self.lm_head = nn.Linear(8, 32)

    class _DummyWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _DummyQwen()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen_vl_interface = _DummyWrapper()
            self.fast_action_model = nn.Linear(8, 8)
            self.action_predictor = nn.Linear(8, 8)
            self.action_token_queries = nn.Parameter(torch.randn(2, 8))
            self.keep_trainable = nn.Linear(8, 8)

    model = _Model()
    freeze_spec = (
        r"param_regex:\.self_attn\.,"
        r"param_regex:\.mlp\.experts\.action\.,"
        "qwen_vl_interface.model.lm_head,"
        "fast_action_model,"
        "action_predictor,"
        r"param_regex:^action_token_queries$"
    )
    TrainerUtils.freeze_backbones(model, freeze_modules=freeze_spec)

    for name, param in model.named_parameters():
        if ".self_attn." in name:
            assert param.requires_grad is False
        if ".mlp.experts.action." in name:
            assert param.requires_grad is False
        if name.startswith("qwen_vl_interface.model.lm_head"):
            assert param.requires_grad is False
        if name.startswith("fast_action_model"):
            assert param.requires_grad is False
        if name.startswith("action_predictor"):
            assert param.requires_grad is False
        if name == "action_token_queries":
            assert param.requires_grad is False

    # A control parameter outside freeze specs should remain trainable.
    assert model.keep_trainable.weight.requires_grad is True
    print("[OK] Freeze policy smoke passed")


def typed_ffn_stage_trainability_smoke_test() -> None:
    class _DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def _build_wrapped_layer(stage_name: str, enable_stage3_action_header: bool, train_stages=None):
        model = QwenMyVLA.__new__(QwenMyVLA)
        nn.Module.__init__(model)
        model.stage_name = stage_name
        if train_stages is None:
            train_stages = ["stage2", "stage3"]
        model.config = SimpleNamespace(
            framework={
                "typed_ffn_train": {"train_stages": train_stages},
                "stage3_action_header": {"enable": enable_stage3_action_header},
            }
        )
        model.typed_ffn_train_stages = {str(x).lower() for x in train_stages}
        model.moe_wrapped_layers = []

        layer = _DummyLayer()
        model._resolve_text_backbone_and_layers = lambda: (object(), [layer])
        model._attach_typed_ffn_moe()
        return layer

    stage1_layer = _build_wrapped_layer("stage1", False)
    assert isinstance(stage1_layer.mlp, TypedResidualFFN)
    assert stage1_layer.mlp.base_mlp[0].weight.requires_grad is False
    assert stage1_layer.mlp.experts["dynamic"][0].weight.requires_grad is True

    stage2_layer = _build_wrapped_layer("stage2", False)
    assert isinstance(stage2_layer.mlp, TypedResidualFFN)
    assert stage2_layer.mlp.base_mlp[0].weight.requires_grad is False
    assert stage2_layer.mlp.experts["dynamic"][0].weight.requires_grad is True

    stage3_layer = _build_wrapped_layer("stage3", True)
    assert isinstance(stage3_layer.mlp, TypedResidualFFN)
    assert stage3_layer.mlp.base_mlp[0].weight.requires_grad is False
    assert stage3_layer.mlp.experts["dynamic"][0].weight.requires_grad is True

    print("[OK] TypedResidualFFN stage trainability smoke passed")


def lora_target_resolution_smoke_test() -> None:
    class _DummyQwenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_mlp = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
            self.lm_head = nn.Linear(4, 4)
            self.experts = nn.ModuleDict({
                "dynamic": nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)),
                "spatial": nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)),
            })
            self.attn = nn.Linear(4, 4)

        def forward(self, x):
            x = self.attn(x)
            x = self.base_mlp(x)
            return self.experts["dynamic"](x)

    model = QwenMyVLA.__new__(QwenMyVLA)
    nn.Module.__init__(model)
    model.enable_lora = True
    model.lora_rank = 2
    model.lora_alpha = 2
    model.lora_dropout = 0.0
    model.lora_target_modules = "all-linear"
    model.lora_exclude_modules = ["base_mlp"]
    model.lora_bias = "none"
    model.lora_init_weights = "gaussian"
    model.qwen_vl_interface = SimpleNamespace(model=_DummyQwenBackbone())

    resolved_targets = model._resolve_lora_target_module_names()
    assert any("attn" == name for name in resolved_targets)
    assert any("experts.dynamic" in name for name in resolved_targets)
    assert not any("base_mlp" in name for name in resolved_targets)
    assert not any("lm_head" in name for name in resolved_targets)

    model._attach_qwen_lora()
    wrapped_model = model.qwen_vl_interface.model

    lora_module_names = [name for name, module in wrapped_model.named_modules() if hasattr(module, "lora_A")]
    assert any("attn" in name for name in lora_module_names)
    assert any("experts.dynamic" in name for name in lora_module_names)
    assert not any("base_mlp" in name for name in lora_module_names)
    assert not any("lm_head" in name for name in lora_module_names)

    print("[OK] LoRA target resolution smoke passed")


def fast_action_tokenizer_init_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=True, fast_detach=False)
    assert model.fast_action_model is not None
    assert model.fast_action_model.fast_tokenizer.time_horizon is None
    assert model.fast_action_model.fast_tokenizer.action_dim is None

    model._configure_fast_action_tokenizer()

    assert model.fast_action_model.fast_tokenizer.time_horizon == model.num_actions_chunk
    assert model.fast_action_model.fast_tokenizer.action_dim == model.action_dim
    print("[OK] Fast action tokenizer init smoke passed")


def lora_modules_to_save_smoke_test() -> None:
    class _DummyQwenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_mlp = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
            self.lm_head = nn.Linear(4, 4)
            self.experts = nn.ModuleDict({
                "dynamic": nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)),
                "spatial": nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)),
            })
            self.attn = nn.Linear(4, 4)

        def forward(self, x):
            x = self.attn(x)
            x = self.base_mlp(x)
            return self.experts["dynamic"](x)

    model = QwenMyVLA.__new__(QwenMyVLA)
    nn.Module.__init__(model)
    model.enable_lora = True
    model.lora_rank = 2
    model.lora_alpha = 2
    model.lora_dropout = 0.0
    model.lora_target_modules = "all-linear"
    model.lora_exclude_modules = ["base_mlp"]
    model.lora_modules_to_save = ["lm_head"]
    model.lora_bias = "none"
    model.lora_init_weights = "gaussian"
    model.qwen_vl_interface = SimpleNamespace(model=_DummyQwenBackbone())

    resolved_modules_to_save = model._resolve_lora_modules_to_save()
    assert resolved_modules_to_save == ["lm_head"]

    model._attach_qwen_lora()
    wrapped_model = model.qwen_vl_interface.model

    lm_head_saved_params = [
        name for name, param in wrapped_model.named_parameters() if "lm_head" in name and param.requires_grad
    ]
    assert any("modules_to_save" in name for name in lm_head_saved_params)
    print("[OK] LoRA modules_to_save smoke passed")


def slot_mask_config_smoke_test(config_yaml: str) -> None:
    cfg = OmegaConf.load(config_yaml)

    k_dyn = int(cfg.framework.slot_dynamic.num_slots)
    k_spa = int(cfg.framework.slot_spatial.num_slots)
    k_sub = int(cfg.framework.slot_subtask.num_slots)

    slot_mask_cfg = cfg.framework.get("slot_mask", {})
    outside_probs = list(slot_mask_cfg.get("outside", {}).get("num_masked_probs", [1.0, 0.0, 0.0, 0.0]))
    prob_sum = sum(float(x) for x in outside_probs)
    if prob_sum <= 0:
        raise ValueError("slot_mask.outside.num_masked_probs must have positive sum")

    keep = QwenMyVLA._sample_outside_keep_mask(
        batch_size=2,
        slot_sizes={"subtask": k_sub, "dynamic": k_dyn, "spatial": k_spa},
        num_masked_probs=[float(x) / prob_sum for x in outside_probs],
        device=torch.device("cpu"),
    )
    assert keep["subtask"].shape[1] == k_sub
    print("[OK] QwenMyVLA config-driven slot mask smoke passed")


def stage2_attention_visibility_semantics_smoke_test() -> None:
    stage2_cfg = Path("/mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla_stage2.yaml")
    if not stage2_cfg.exists():
        raise FileNotFoundError(f"stage2 config not found: {stage2_cfg}")

    cfg = OmegaConf.load(str(stage2_cfg))
    v = cfg.framework.attention_mask.stage_visibility.stage2

    # image only reads image
    assert bool(v.image.image) is True
    assert bool(v.image.text) is False
    assert bool(v.image.subtask) is False
    assert bool(v.image.dynamic) is False
    assert bool(v.image.spatial) is False

    # instruction reads image + instruction
    assert bool(v.text.text) is True
    assert bool(v.text.image) is True
    assert bool(v.text.subtask) is False
    assert bool(v.text.dynamic) is False
    assert bool(v.text.spatial) is False

    # slot groups are mutually visible and can read image+instruction
    for row_name in ["subtask", "dynamic", "spatial"]:
        row = v[row_name]
        assert bool(row.text) is True
        assert bool(row.image) is True
        assert bool(row.subtask) is True
        assert bool(row.dynamic) is True
        assert bool(row.spatial) is True

    # action cannot read image/instruction directly; only structured slots + past action
    assert bool(v.action.text) is False
    assert bool(v.action.image) is False
    assert bool(v.action.subtask) is True
    assert bool(v.action.dynamic) is True
    assert bool(v.action.spatial) is True
    assert bool(v.action.action) is True
    print("[OK] Stage2 attention visibility semantics smoke passed")


def subtask_adapter_primary_path_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=False, fast_detach=False)
    adapter_slots = torch.randn((1, model.k_sub, model.hidden_size), dtype=torch.float32)
    subtask_outputs = {
        "text_tokens": torch.randn((1, 5, model.hidden_size), dtype=torch.float32),
        "text_mask": torch.ones((1, 5), dtype=torch.bool),
        "subtask_slots": adapter_slots,
    }

    built = model._build_subtask_slots(subtask_outputs)
    assert torch.allclose(built, adapter_slots)
    print("[OK] Subtask adapter primary-path smoke passed")


def stage3_m1_condition_and_sampling_smoke_test() -> None:
    class _FakeDinoEncoder(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.num_channels = hidden_size

        def prepare_dino_input(self, img_list):
            batch_size = len(img_list)
            views = len(img_list[0]) if batch_size > 0 else 1
            device = next(self.parameters()).device
            return torch.zeros((batch_size * views, 3, 8, 8), device=device)

        def forward(self, tensor):
            return torch.ones((tensor.shape[0], 5, self.num_channels), device=tensor.device)

    class _FakeConditionFuser(nn.Module):
        def __init__(self, output_dim: int, num_layers: int = 2, num_queries: int = 64):
            super().__init__()
            self.num_layers = num_layers
            self.num_queries = num_queries
            self.output_dim = output_dim

        def forward(self, hidden_states_list, encoder_attention_mask=None):
            del encoder_attention_mask
            batch_size = hidden_states_list[0].shape[0]
            device = hidden_states_list[0].device
            return torch.zeros((batch_size, self.num_queries, self.output_dim), device=device)

    class _FakeStage3Net(nn.Module):
        def __init__(self, condition_dim: int):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.z_embedder = SimpleNamespace(uncondition=torch.zeros(64, condition_dim))

        def forward(self, x, t, z):
            del t, z
            return torch.zeros_like(x)

        def forward_with_cfg(self, x, t, z, cfg_scale):
            del cfg_scale
            return self.forward(x, t, z)

    class _FakeDiffusion:
        def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False):
            del clip_denoised, denoised_fn, cond_fn, progress
            x = noise if noise is not None else torch.zeros(shape, device=device)
            t = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
            return model(x, t, **(model_kwargs or {}))

        def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
            del eta
            return self.p_sample_loop(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
            )

    class _FakeStage3Head(nn.Module):
        def __init__(self, action_dim: int, condition_dim: int, chunk_len: int):
            super().__init__()
            self.in_channels = action_dim
            self.future_action_window_size = max(chunk_len - 1, 0)
            self.past_action_window_size = 0
            self.net = _FakeStage3Net(condition_dim)
            self.diffusion = _FakeDiffusion()
            self.ddim_diffusion = None

        def forward(self, gt_action, condition, **kwargs):
            del condition, kwargs
            noise = torch.randn_like(gt_action)
            return noise, noise, torch.zeros((gt_action.shape[0],), dtype=torch.long, device=gt_action.device)

        def loss(self, noise_pred, noise):
            return ((noise_pred - noise) ** 2).mean()

        def create_ddim(self, ddim_step=10):
            del ddim_step
            self.ddim_diffusion = _FakeDiffusion()
            return self.ddim_diffusion

    model = _build_mock_model(enable_fast_loss=False, fast_detach=False)
    model.enable_stage3_action_header = True
    model.enable_slot_mask = True
    model.slot_mask_apply_in_eval = False
    model.inner_mask_ratios = {"dynamic": 1.0, "spatial": 1.0, "subtask": 1.0}
    model.outside_num_masked_probs = [0.0, 0.0, 0.0, 1.0]
    model.stage3_dino_encoder = _FakeDinoEncoder(model.hidden_size)
    model.stage3_dino_projector = nn.Identity()
    model.stage3_condition_fuser = _FakeConditionFuser(output_dim=8, num_layers=2, num_queries=64)
    model.stage3_action_head = _FakeStage3Head(action_dim=model.action_dim, condition_dim=8, chunk_len=model.num_actions_chunk)

    class _CaptureEmbedding(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.embedding = nn.Embedding(256, hidden_size)
            with torch.no_grad():
                weight = torch.arange(256 * hidden_size, dtype=torch.float32).reshape(256, hidden_size)
                self.embedding.weight.copy_(weight / 1000.0)
            self.last_output = None

        def forward(self, input_ids):
            output = self.embedding(input_ids)
            self.last_output = output
            return output

    capture_embedding = _CaptureEmbedding(model.hidden_size)
    model.qwen_vl_interface.model.model.embed = capture_embedding

    samples = [{"image": [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)], "lang": "pick up the cup", "action": np.zeros((4, 7), dtype=np.float32)}]
    model.train()
    out_train = model.forward(examples=samples)
    assert "action_loss" in out_train and out_train["action_loss"].ndim == 0

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=[samples[0]["image"]], instructions=[samples[0]["lang"]])
    input_ids = qwen_inputs["input_ids"]
    dyn_pos = model._gather_positions(input_ids, model.dynamic_token_id, model.k_dyn)
    spa_pos = model._gather_positions(input_ids, model.spatial_token_id, model.k_spa)
    sub_pos = model._gather_positions(input_ids, model.subtask_token_id, model.k_sub)
    baseline_embeddings = capture_embedding(input_ids).detach().clone()

    model.eval()
    out_pred = model.predict_action(examples=samples, cfg_scale=1.0, use_ddim=True, num_ddim_steps=2)
    assert out_pred["normalized_actions"].shape == (1, model.num_actions_chunk, model.action_dim)
    masked_embeddings = capture_embedding.last_output.detach().to(dtype=baseline_embeddings.dtype)
    assert torch.allclose(masked_embeddings[0, dyn_pos[0]], baseline_embeddings[0, dyn_pos[0]])
    assert torch.allclose(masked_embeddings[0, spa_pos[0]], baseline_embeddings[0, spa_pos[0]])
    assert torch.allclose(masked_embeddings[0, sub_pos[0]], baseline_embeddings[0, sub_pos[0]])
    print("[OK] Stage3 M1-style condition and sampling smoke passed")


def teacher_encoder_single_call_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=False, fast_detach=False)
    call_counts = {"dynamic": 0, "spatial": 0, "subtask": 0}

    def _dynamic_teacher(examples):
        call_counts["dynamic"] += 1
        return {
            "indices": torch.zeros((len(examples), 1), dtype=torch.long),
            "z_q": torch.randn((len(examples), 4, 8)),
        }

    def _spatial_teacher(examples):
        call_counts["spatial"] += 1
        return {
            "encoder_tokens": torch.randn((len(examples), 6, 8)),
            "token_mask": torch.ones((len(examples), 6), dtype=torch.bool),
        }

    def _subtask_teacher(examples):
        call_counts["subtask"] += 1
        return {
            "text_tokens": torch.randn((len(examples), 5, 8)),
            "text_mask": torch.ones((len(examples), 5), dtype=torch.bool),
        }

    model.dynamic_teacher_encoder = _dynamic_teacher
    model.spatial_teacher_encoder = _spatial_teacher
    model.subtask_slot_encoder = _subtask_teacher

    samples = [{"image": [torch.zeros(3, 8, 8)], "lang": "pick up the cup", "action": np.zeros((4, 7), dtype=np.float32)}]
    model.train()
    _ = model.forward(examples=samples)

    assert call_counts == {"dynamic": 1, "spatial": 1, "subtask": 1}
    print("[OK] Teacher encoder single-call smoke passed")


def curriculum_schedule_interpolation_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=True, fast_detach=False)
    model.enable_slot_mask_curriculum = True
    model.slot_mask_curriculum_progress_start = 0.0
    model.slot_mask_curriculum_progress_end = 1.0
    model.slot_mask_inner_start = {"dynamic": 0.1, "spatial": 0.2, "subtask": 0.3}
    model.slot_mask_inner_end = {"dynamic": 0.4, "spatial": 0.5, "subtask": 0.6}
    model.slot_mask_outside_start_probs = [0.7, 0.2, 0.1, 0.0]
    model.slot_mask_outside_end_probs = [0.1, 0.2, 0.3, 0.4]

    model.enable_loss_weight_schedule = True
    model.loss_weight_schedule_progress_start = 0.0
    model.loss_weight_schedule_progress_end = 1.0
    model.slot_align_weight_start = {"dynamic": 1.0, "spatial": 0.9, "subtask": 0.8}
    model.slot_align_weight_end = {"dynamic": 0.4, "spatial": 0.3, "subtask": 0.2}
    model.fast_action_loss_weight_start = 0.5
    model.fast_action_loss_weight_end = 1.0
    model.continuous_fm_loss_weight_start = 0.1
    model.continuous_fm_loss_weight_end = 0.3

    inner0, outside0 = model._resolve_slot_mask_hparams(0.0)
    inner1, outside1 = model._resolve_slot_mask_hparams(1.0)
    assert abs(float(inner0["dynamic"]) - 0.1) < 1e-6
    assert abs(float(inner1["dynamic"]) - 0.4) < 1e-6
    assert abs(float(outside0[0]) - 0.7) < 1e-6
    assert abs(float(outside1[-1]) - 0.4) < 1e-6

    w0 = model._resolve_loss_weights(0.0)
    w1 = model._resolve_loss_weights(1.0)
    assert abs(float(w0["slot_dynamic"]) - 1.0) < 1e-6
    assert abs(float(w1["slot_dynamic"]) - 0.4) < 1e-6
    assert abs(float(w0["fast"]) - 0.5) < 1e-6
    assert abs(float(w1["fast"]) - 1.0) < 1e-6
    print("[OK] Curriculum/schedule interpolation smoke passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="QwenMyVLA smoke tests")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla.yaml",
        help="Path to yaml config used by config-driven mask smoke test",
    )
    parser.add_argument("--with_slot_mask_config", action="store_true", help="Run config-driven slot-mask smoke")
    args = parser.parse_args()

    train_predict_contract_smoke_test()
    loss_branch_semantics_smoke_test()
    typed_attention_mask_smoke_test()
    all_layer_typed_attention_apply_smoke_test()
    attn_implementation_guard_smoke_test()
    masked_target_prediction_slot_align_smoke_test()
    main_forward_action_group_smoke_test()
    freeze_policy_smoke_test()
    stage2_attention_visibility_semantics_smoke_test()
    subtask_adapter_primary_path_smoke_test()
    stage3_m1_condition_and_sampling_smoke_test()
    teacher_encoder_single_call_smoke_test()
    typed_ffn_stage_trainability_smoke_test()
    fast_action_tokenizer_init_smoke_test()
    lora_target_resolution_smoke_test()
    lora_modules_to_save_smoke_test()
    curriculum_schedule_interpolation_smoke_test()
    if args.with_slot_mask_config:
        slot_mask_config_smoke_test(args.config_yaml)


if __name__ == "__main__":
    main()
