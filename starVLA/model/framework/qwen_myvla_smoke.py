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
    model.config = SimpleNamespace(framework={})
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
    model.fast_action_model = object() if enable_fast_loss else None

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
    model.coarse_action_state_proj = nn.Sequential(
        nn.LayerNorm(action_dim),
        nn.Linear(action_dim, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
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


def stage3_coarse_condition_smoke_test() -> None:
    model = _build_mock_model(enable_fast_loss=False, fast_detach=False)
    qwen_last_hidden = torch.randn((1, 24, model.hidden_size), dtype=torch.float32)
    predicted_actions_light = torch.randn((1, model.num_actions_chunk, model.action_dim), dtype=torch.float32)

    action_pos = torch.tensor([[17, 18]], dtype=torch.long)
    cond_from_action = model._build_stage3_coarse_condition(
        qwen_last_hidden=qwen_last_hidden,
        action_pos=action_pos,
        predicted_actions_light=predicted_actions_light,
    )
    expected = qwen_last_hidden[:, 17:19, :].mean(dim=1)
    assert torch.allclose(cond_from_action, expected)

    cond_fallback = model._build_stage3_coarse_condition(
        qwen_last_hidden=qwen_last_hidden,
        action_pos=None,
        predicted_actions_light=predicted_actions_light,
    )
    assert cond_fallback.shape == (1, model.hidden_size)
    print("[OK] Stage3 coarse-condition smoke passed")


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
    masked_target_prediction_slot_align_smoke_test()
    main_forward_action_group_smoke_test()
    freeze_policy_smoke_test()
    stage2_attention_visibility_semantics_smoke_test()
    subtask_adapter_primary_path_smoke_test()
    stage3_coarse_condition_smoke_test()
    curriculum_schedule_interpolation_smoke_test()
    if args.with_slot_mask_config:
        slot_mask_config_smoke_test(args.config_yaml)


if __name__ == "__main__":
    main()
