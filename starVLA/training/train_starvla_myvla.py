# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA MyVLA trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def build_accelerator(cfg):
    """Create an Accelerator that matches the current run mode.

    Debug or single-process runs should not require DeepSpeed/MPI.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if getattr(cfg, "is_debug", False) or world_size == 1:
        accelerator = Accelerator()
    else:
        deepspeed_plugin = DeepSpeedPlugin()
        accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

    accelerator.print(accelerator.state)
    return accelerator


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _maybe_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    _maybe_barrier()
    return vla_train_dataloader


def resolve_training_horizon(cfg, vla_train_dataloader: DataLoader) -> None:
    """Resolve max_train_steps from either explicit steps or epoch-based control.

    Default behavior stays step-based for backward compatibility.
    If `trainer.use_epoch_control=true`, we derive max_train_steps from:
        ceil(epochs * ceil(len(dataloader) / gradient_accumulation_steps)).
    """
    grad_accum = max(1, int(getattr(cfg.trainer, "gradient_accumulation_steps", 1)))
    dataloader_len = max(1, int(len(vla_train_dataloader)))
    steps_per_epoch = max(1, int(math.ceil(dataloader_len / grad_accum)))

    use_epoch_control = bool(getattr(cfg.trainer, "use_epoch_control", False))
    if use_epoch_control:
        epochs = float(getattr(cfg.trainer, "epochs", 1))
        if epochs <= 0:
            raise ValueError("trainer.epochs must be > 0 when trainer.use_epoch_control=true")
        derived_steps = max(1, int(math.ceil(epochs * steps_per_epoch)))
        cfg.trainer.max_train_steps = derived_steps
        logger.info(
            "Epoch control enabled: epochs=%s, steps_per_epoch=%d, max_train_steps=%d",
            epochs,
            steps_per_epoch,
            derived_steps,
        )
    else:
        logger.info(
            "Step control enabled: max_train_steps=%d (steps_per_epoch=%d, epochs field is informational)",
            int(cfg.trainer.max_train_steps),
            steps_per_epoch,
        )


def validate_stage_configuration(cfg) -> None:
    """Validate that a stage YAML matches the expected high-level switches.

    This does not change training behavior. It only warns when a stage file is
    likely being used with inconsistent flags.
    """
    def _warn(message: str) -> None:
        print(f"[stage-config-warning] {message}")

    stage_name = getattr(cfg.trainer, "stage_name", None)
    if stage_name is None:
        _warn("No trainer.stage_name found in config; stage-specific validation is skipped.")
        return

    stage_name = str(stage_name).lower()
    framework = cfg.framework

    def _flag(path: str, default=None):
        node = cfg
        for key in path.split("."):
            if not hasattr(node, key):
                return default
            node = getattr(node, key)
        return node

    checks = {
        "stage1": [
            ("framework.fast_action_loss.enable", False),
            ("framework.stage3_action_header.enable", False),
            ("framework.slot_mask.enable", False),
        ],
        "stage2": [
            ("framework.fast_action_loss.enable", True),
            ("framework.slot_mask.enable", True),
            ("framework.stage3_action_header.enable", False),
        ],
        "stage3": [
            ("framework.fast_action_loss.enable", True),
            ("framework.stage3_action_header.enable", True),
        ],
    }

    if stage_name not in checks:
        _warn(f"Unknown trainer.stage_name `{stage_name}`. Expected one of stage1/stage2/stage3.")
        return

    for path, expected in checks[stage_name]:
        actual = _flag(path)
        if actual != expected:
            _warn(f"Stage config mismatch: {path}={actual} but stage `{stage_name}` expects {expected}")

    def _is_mapping_like(value) -> bool:
        if value is None:
            return False
        if isinstance(value, Mapping):
            return True
        keys = getattr(value, "keys", None)
        return callable(keys)

    stage_visibility = getattr(getattr(cfg.framework, "attention_mask", None), "stage_visibility", {})
    if _is_mapping_like(stage_visibility):
        if stage_name not in stage_visibility:
            _warn(
                f"attention_mask.stage_visibility does not define `{stage_name}`; model will fall back to attention_mask.visibility."
            )
    else:
        _warn("attention_mask.stage_visibility is missing or malformed; model will use fallback visibility rules.")

    action_query_num = getattr(getattr(framework, "action_model", None), "action_query_num", None)
    if stage_name == "stage3" and action_query_num is None:
        _warn("Stage3 config should define framework.action_model.action_query_num for action head reuse.")

    action_tokens_cfg = framework.get("action_tokens", {}) if hasattr(framework, "get") else {}
    action_tokens_enable = bool(action_tokens_cfg.get("enable", False))
    action_tokens_num = int(action_tokens_cfg.get("num_tokens", 0))
    action_tokens_placeholder = str(action_tokens_cfg.get("placeholder", "")).strip()

    if stage_name in {"stage2", "stage3"} and not action_tokens_enable:
        _warn(f"Stage `{stage_name}` expects framework.action_tokens.enable=true for action group masking semantics.")
    if stage_name == "stage1" and action_tokens_enable:
        _warn("Stage `stage1` usually keeps framework.action_tokens.enable=false to avoid unnecessary action placeholders.")

    if action_tokens_enable:
        if action_tokens_num <= 0:
            _warn("framework.action_tokens.enable=true but framework.action_tokens.num_tokens<=0")
        if not action_tokens_placeholder:
            _warn("framework.action_tokens.enable=true but framework.action_tokens.placeholder is empty")
        if stage_name == "stage3" and action_query_num is not None and int(action_query_num) != action_tokens_num:
            _warn(
                "Stage `stage3` recommends framework.action_tokens.num_tokens == "
                "framework.action_model.action_query_num for stable action query alignment."
            )

    if stage_name in {"stage2", "stage3"} and _is_mapping_like(stage_visibility) and stage_name in stage_visibility:
        stage_rule = stage_visibility[stage_name]
        if _is_mapping_like(stage_rule):
            if "action" not in stage_rule:
                _warn(f"attention_mask.stage_visibility.{stage_name} is missing root `action` row")
            else:
                action_row = stage_rule["action"]
                if not _is_mapping_like(action_row):
                    _warn(f"attention_mask.stage_visibility.{stage_name}.action should be a mapping")
                elif "action" not in action_row:
                    _warn(f"attention_mask.stage_visibility.{stage_name}.action is missing `action` column")

    freeze_modules = str(getattr(cfg.trainer, "freeze_modules", "") or "").strip()
    if stage_name in {"stage1", "stage2"} and not freeze_modules:
        _warn(f"Stage `{stage_name}` expects non-empty trainer.freeze_modules for explicit freeze policy.")
    if stage_name == "stage1":
        if "param_regex:\\.self_attn\\." not in freeze_modules:
            _warn("Stage `stage1` recommends freezing self-attention via `param_regex:\\.self_attn\\.`")
        if "param_regex:\\.mlp\\.experts\\.action\\." not in freeze_modules:
            _warn("Stage `stage1` recommends freezing Action-FFN via `param_regex:\\.mlp\\.experts\\.action\\.`")
    if stage_name == "stage2" and "action_predictor" not in freeze_modules:
        _warn("Stage `stage2` recommends freezing continuous action head via `action_predictor`")

    save_mode = str(getattr(cfg.trainer, "save_mode", "step")).lower()
    if save_mode not in {"step", "epoch"}:
        _warn(f"trainer.save_mode={save_mode} is invalid. Expected `step` or `epoch`.")
    elif save_mode == "epoch":
        save_interval_epochs = int(getattr(cfg.trainer, "save_interval_epochs", 1))
        if save_interval_epochs <= 0:
            _warn("trainer.save_interval_epochs must be > 0 when trainer.save_mode=epoch")
    else:
        save_interval_steps = int(getattr(cfg.trainer, "save_interval_steps", getattr(cfg.trainer, "save_interval", 0)))
        if save_interval_steps <= 0:
            _warn("trainer.save_interval/save_interval_steps must be > 0 when trainer.save_mode=step")


def validate_action_tokenizer_configuration(cfg) -> bool:
    """Validate placeholder tokenization against the configured tokenizer.

    This check is lightweight and exits before model/framework construction.
    """

    def _warn(message: str) -> None:
        print(f"[action-tokenizer-warning] {message}")

    framework = cfg.framework
    qwenvl_cfg = getattr(framework, "qwenvl", None)
    base_vlm = getattr(qwenvl_cfg, "base_vlm", None)
    if not base_vlm:
        _warn("framework.qwenvl.base_vlm is missing; tokenizer validation is skipped.")
        return False

    base_vlm_str = str(base_vlm)
    base_vlm_path = Path(base_vlm_str).expanduser().resolve()
    if base_vlm_str.startswith(".") and not base_vlm_path.exists():
        _warn(
            "Configured local model path does not exist for tokenizer validation: "
            f"{base_vlm_path}"
        )
        return False

    if base_vlm_path.exists() and base_vlm_path.is_dir():
        has_processor_files = any(
            (base_vlm_path / name).exists()
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
            )
        )
        if not has_processor_files:
            _warn(
                "Local model directory exists but tokenizer/processor files are missing; "
                f"download may still be in progress: {base_vlm_path}"
            )
            return False

    load_target = str(base_vlm_path) if base_vlm_path.exists() else base_vlm_str

    try:
        processor = AutoProcessor.from_pretrained(load_target, trust_remote_code=True)
        tokenizer = processor.tokenizer
    except Exception as exc:
        _warn(f"Failed to load tokenizer from `{load_target}`: {exc}")
        return False

    slot_placeholders = framework.get("slot_placeholders", {}) if hasattr(framework, "get") else {}
    slot_boundary_tokens = framework.get("slot_boundary_tokens", {}) if hasattr(framework, "get") else {}
    action_tokens_cfg = framework.get("action_tokens", {}) if hasattr(framework, "get") else {}

    checks = [
        ("dynamic", str(slot_placeholders.get("dynamic", "<slot_dynamic>"))),
        ("spatial", str(slot_placeholders.get("spatial", "<slot_spatial>"))),
        ("subtask", str(slot_placeholders.get("subtask", "<slot_subtask>"))),
        ("action", str(slot_placeholders.get("action", action_tokens_cfg.get("placeholder", "<slot_action>")))),
        ("dynamic_start", str(slot_boundary_tokens.get("dynamic_start", "<SOSD>"))),
        ("dynamic_end", str(slot_boundary_tokens.get("dynamic_end", "<EOSD>"))),
        ("spatial_start", str(slot_boundary_tokens.get("spatial_start", "<SOSS>"))),
        ("spatial_end", str(slot_boundary_tokens.get("spatial_end", "<EOSS>"))),
        ("subtask_start", str(slot_boundary_tokens.get("subtask_start", "<SOST>"))),
        ("subtask_end", str(slot_boundary_tokens.get("subtask_end", "<EOST>"))),
        ("action_start", str(slot_boundary_tokens.get("action_start", "<SOSA>"))),
        ("action_end", str(slot_boundary_tokens.get("action_end", "<EOSA>"))),
    ]

    ok = True
    token_ids = {}
    for name, token in checks:
        ids = tokenizer(token, add_special_tokens=False).get("input_ids", [])
        if len(ids) != 1:
            _warn(
                f"Placeholder `{name}` token `{token}` maps to {len(ids)} tokens {ids}; expected exactly 1 token."
            )
            ok = False
            continue
        token_ids[name] = int(ids[0])
        print(f"[OK] tokenizer placeholder `{name}` -> id {token_ids[name]}")

    seen = {}
    for name, tid in token_ids.items():
        if tid in seen:
            _warn(f"Placeholder id collision: `{name}` and `{seen[tid]}` both map to token id {tid}")
            ok = False
        else:
            seen[tid] = name

    return ok


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        grad_accum = max(1, int(getattr(self.config.trainer, "gradient_accumulation_steps", 1)))
        self.steps_per_epoch = max(1, int(math.ceil(len(self.vla_train_dataloader) / grad_accum)))

        self.save_mode = str(getattr(self.config.trainer, "save_mode", "step")).lower()
        if self.save_mode not in {"step", "epoch"}:
            raise ValueError(f"Invalid trainer.save_mode={self.save_mode}. Expected `step` or `epoch`.")

        # Backward compatibility: keep old `save_interval` behavior when save_mode=step.
        if self.save_mode == "epoch":
            save_interval_epochs = int(getattr(self.config.trainer, "save_interval_epochs", 1))
            if save_interval_epochs <= 0:
                raise ValueError("trainer.save_interval_epochs must be > 0 when save_mode=epoch")
            self.save_interval_steps = max(1, save_interval_epochs * self.steps_per_epoch)
        else:
            save_interval_steps = int(
                getattr(
                    self.config.trainer,
                    "save_interval_steps",
                    getattr(self.config.trainer, "save_interval", 5000),
                )
            )
            if save_interval_steps <= 0:
                raise ValueError("trainer.save_interval/save_interval_steps must be > 0 when save_mode=step")
            self.save_interval_steps = save_interval_steps

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="myvla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and _is_rank0():
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(self._format_metrics_for_log(metrics))

    def _format_metrics_for_log(self, metrics: dict) -> str:
        """Render a compact, human-readable training summary."""
        def _fmt(key: str, digits: int = 4) -> str:
            value = metrics.get(key, None)
            if value is None:
                return None
            if isinstance(value, (int, float, np.floating)):
                return f"{key}={float(value):.{digits}f}"
            return f"{key}={value}"

        header_keys = ["total_loss", "training_progress", "learning_rate", "epoch", "mse_score"]
        loss_keys = [
            "slot_align_loss",
            "slot_dynamic_total",
            "slot_spatial_total",
            "slot_subtask_total",
            "continuous_fm_loss",
            "fast_action_loss",
            "supervised_action_loss",
        ]
        weight_keys = ["w_slot_dynamic", "w_slot_spatial", "w_slot_subtask", "w_continuous", "w_fast"]
        timing_keys = ["data_time", "model_time"]

        lines = [f"Step {self.completed_steps}"]

        header_parts = [part for key in header_keys if (part := _fmt(key)) is not None]
        if header_parts:
            lines.append("  " + " | ".join(header_parts))

        slot_parts = [
            part
            for key in (
                "slot_dynamic_mse",
                "slot_dynamic_cos",
                "slot_spatial_mse",
                "slot_spatial_cos",
                "slot_subtask_mse",
                "slot_subtask_cos",
            )
            if (part := _fmt(key)) is not None
        ]
        loss_parts = [part for key in loss_keys if (part := _fmt(key)) is not None]
        if loss_parts:
            lines.append("  losses: " + " | ".join(loss_parts))
        if slot_parts:
            lines.append("  slots:  " + " | ".join(slot_parts))

        weight_parts = [part for key in weight_keys if (part := _fmt(key, digits=3)) is not None]
        if weight_parts:
            lines.append("  weights: " + " | ".join(weight_parts))

        timing_parts = [part for key in timing_keys if (part := _fmt(key, digits=3)) is not None]
        if timing_parts:
            lines.append("  timing:  " + " | ".join(timing_parts))

        return "\n".join(lines)

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.save_interval_steps == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        _maybe_barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(
                f"  Save mode = {self.save_mode}, save every {self.save_interval_steps} optimization steps"
            )

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            # self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    global_step=self.completed_steps,
                    max_train_steps=self.config.trainer.max_train_steps,
                )
                total_loss = output_dict["action_loss"]

            self.accelerator.backward(total_loss)

            # 只有在真正的优化步（即累积满了）才做这些
            if self.accelerator.sync_gradients:
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)  # 更省显存/更快
                
            # # gradient clipping
            # if self.config.trainer.gradient_clipping is not None:
            #     self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # # optimizer step
            # self.optimizer.step()
            # self.lr_scheduler.step()

        metrics = {
            "total_loss": total_loss.item(),
        }

        if isinstance(output_dict, dict):
            if "training_progress" in output_dict:
                metrics["training_progress"] = float(output_dict["training_progress"])

            if "slot_align_loss" in output_dict:
                metrics["slot_align_loss"] = float(output_dict["slot_align_loss"].item())
            if "continuous_fm_loss" in output_dict:
                metrics["continuous_fm_loss"] = float(output_dict["continuous_fm_loss"].item())
            if "fast_action_loss" in output_dict:
                metrics["fast_action_loss"] = float(output_dict["fast_action_loss"].item())
            if "supervised_action_loss" in output_dict:
                metrics["supervised_action_loss"] = float(output_dict["supervised_action_loss"].item())

            slot_breakdown = output_dict.get("slot_align_loss_breakdown")
            if isinstance(slot_breakdown, dict):
                for slot_name, slot_stats in slot_breakdown.items():
                    if not isinstance(slot_stats, dict):
                        continue
                    for stat_name in ("mse", "cos", "total"):
                        if stat_name in slot_stats:
                            metrics[f"slot_{slot_name}_{stat_name}"] = float(slot_stats[stat_name].item())

            effective_weights = output_dict.get("effective_loss_weights")
            if isinstance(effective_weights, dict):
                if "slot_dynamic" in effective_weights:
                    metrics["w_slot_dynamic"] = float(effective_weights["slot_dynamic"])
                if "slot_spatial" in effective_weights:
                    metrics["w_slot_spatial"] = float(effective_weights["slot_spatial"])
                if "slot_subtask" in effective_weights:
                    metrics["w_slot_subtask"] = float(effective_weights["slot_subtask"])
                if "continuous" in effective_weights:
                    metrics["w_continuous"] = float(effective_weights["continuous"])
                if "fast" in effective_weights:
                    metrics["w_fast"] = float(effective_weights["fast"])

            if "continuous_fm_loss_for_optim" in output_dict:
                metrics["continuous_fm_loss_for_optim"] = float(output_dict["continuous_fm_loss_for_optim"].item())
            if "fast_action_loss_for_optim" in output_dict:
                metrics["fast_action_loss_for_optim"] = float(output_dict["fast_action_loss_for_optim"].item())

        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    cfg = wrap_config(cfg) # 保存下来，并且让后续访问都能被记录（如果是AccessTrackedConfig的话）
    validate_stage_configuration(cfg)

    accelerator = build_accelerator(cfg)
    logger.info("MyVLA Training :: Warming Up")
    logger.info("✅ Configuration wrapped for access tracking")
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    resolve_training_horizon(cfg=cfg, vla_train_dataloader=vla_train_dataloader)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla_stage1.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--validate-stage-config",
        action="store_true",
        help="Validate stage YAML switches and exit without training",
    )
    parser.add_argument(
        "--validate-action-tokenizer",
        action="store_true",
        help="Validate placeholder tokenization against configured tokenizer and exit",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if args.validate_stage_config:
        validate_stage_configuration(cfg)
        print("[OK] stage configuration validation finished")

    if args.validate_action_tokenizer:
        token_ok = validate_action_tokenizer_configuration(cfg)
        if token_ok:
            print("[OK] action/tokenizer placeholder validation finished")
        else:
            raise SystemExit(1)

    if args.validate_stage_config or args.validate_action_tokenizer:
        raise SystemExit(0)

    if cfg.is_debug and _is_rank0():
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
