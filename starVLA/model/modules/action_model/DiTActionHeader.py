# Copyright 2025 CogACT. All rights reserved.
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [add global config ].
"""
DiT-based flow matching action prediction head.

Provides:
  - Size presets (S/B/L) for transformer-based temporal action backbone
  - ActionModel: standard flow matching training and Euler integration sampling
"""

from starVLA.model.modules.action_model.DiT_modules.models import DiT

import torch
from torch import nn


# Create model sizes of ActionModels
def DiT_S(**kwargs):  # TODO move to config for reproducibility
    """
    Small DiT variant.

    Args:
        **kwargs: Passed through to DiT constructor.

    Returns:
        DiT: Initialized small model.
    """
    return DiT(depth=6, token_size=384, num_heads=4, **kwargs)


def DiT_B(**kwargs):
    """
    Base DiT variant.

    Args:
        **kwargs: Passed through to DiT constructor.

    Returns:
        DiT: Initialized base model.
    """
    return DiT(depth=12, token_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    """
    Large DiT variant.

    Args:
        **kwargs: Passed through to DiT constructor.

    Returns:
        DiT: Initialized large model.
    """
    return DiT(depth=24, token_size=1024, num_heads=16, **kwargs)


# Model size
DiT_models = {"DiT-S": DiT_S, "DiT-B": DiT_B, "DiT-L": DiT_L}


# Create ActionModel
class ActionModel(nn.Module):
    """
    DiT temporal action head trained with standard flow matching.

    Components:
        - DiT transformer backbone (token-wise velocity predictor)

    Responsibilities:
        - Forward: sample interpolation time and predict velocity field
        - loss(): MSE on velocity prediction
        - sample_actions(): Euler integration from noise to action trajectory
    """

    def __init__(
        self,
        action_hidden_dim,
        model_type,
        in_channels,
        future_action_window_size,
        past_action_window_size,
        diffusion_steps=100,
        noise_schedule="squaredcos_cap_v2",
        t_eps=1.0e-3,
    ):
        """
        Initialize diffusion model and backbone.

        Args:
            action_hidden_dim: Hidden size of conditioning tokens (QFormer output dim).
            model_type: One of {'DiT-S','DiT-B','DiT-L'}.
            in_channels: Action dimensionality (per timestep).
            future_action_window_size: Number of future steps modeled.
            past_action_window_size: Number of past steps possibly encoded (for context).
            diffusion_steps: Used as the number of timestep buckets for conditioning.
            noise_schedule: Retained for backward-compatible config surface.
            t_eps: Clamp to avoid degeneracy near t=1 during training.
        """
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        self.diffusion_steps = diffusion_steps
        self.num_timestep_buckets = diffusion_steps
        self.t_eps = float(t_eps)
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.action_horizon = future_action_window_size + past_action_window_size + 1
        self.token_size = action_hidden_dim  # QFormer output size
        self.net = DiT_models[model_type](
            in_channels=in_channels,
            class_dropout_prob=0.1,
            learn_sigma=False,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )

    def forward(self, gt_action, condition, **kwargs):
        """
        Perform one flow matching training step.

        Args:
            gt_action: Ground truth action tensor [B, T, C].
            condition: Conditioning tokens [B, L, D].
            **kwargs: Ignored (reserved).

        Returns:
            tuple:
                pred_velocity: Predicted velocity tensor.
                target_velocity: Target velocity tensor.
                timestep: Discrete timesteps used for conditioning.
        """
        del kwargs
        noise = torch.randn_like(gt_action)
        t_cont = torch.rand((gt_action.size(0),), device=gt_action.device, dtype=gt_action.dtype)
        if self.t_eps > 0:
            t_cont = t_cont.clamp(min=self.t_eps, max=1.0 - self.t_eps)
        t_broadcast = t_cont[:, None, None]
        noisy_trajectory = (1.0 - t_broadcast) * noise + t_broadcast * gt_action
        target_velocity = gt_action - noise
        timestep = (t_cont * (self.num_timestep_buckets - 1)).long()

        pred_velocity = self.net(noisy_trajectory, timestep, condition)
        assert pred_velocity.shape == target_velocity.shape == gt_action.shape

        return pred_velocity, target_velocity, timestep

    def loss(self, pred_velocity, target_velocity):
        """
        Compute MSE velocity prediction loss.

        Args:
            pred_velocity: Predicted velocity tensor.
            target_velocity: Target velocity tensor.

        Returns:
            torch.Tensor: Scalar loss.
        """
        return ((pred_velocity - target_velocity) ** 2).mean()

    @torch.no_grad()
    def sample_actions(self, condition, cfg_scale: float = 1.0, num_steps: int = 10):
        """Sample actions with Euler integration under the learned velocity field."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        batch_size = condition.shape[0]
        device = condition.device
        model_dtype = next(self.net.parameters()).dtype
        condition = condition.to(device=device, dtype=model_dtype)

        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.in_channels,
            device=device,
            dtype=model_dtype,
        )
        dt = 1.0 / float(num_steps)
        using_cfg = cfg_scale > 1.0 and hasattr(self.net, "forward_with_cfg") and hasattr(self.net, "z_embedder")

        for step_idx in range(num_steps):
            t_cont = float(step_idx) / float(num_steps)
            timestep = torch.full(
                (batch_size,),
                int(t_cont * (self.num_timestep_buckets - 1)),
                device=device,
                dtype=torch.long,
            )
            if using_cfg:
                uncondition = self.net.z_embedder.uncondition.to(device=device, dtype=model_dtype)
                uncondition = uncondition.unsqueeze(0).expand(batch_size, -1, -1)
                model_input = torch.cat([actions, actions], dim=0)
                z = torch.cat([condition, uncondition], dim=0)
                pred_velocity = self.net.forward_with_cfg(model_input, timestep.repeat(2), z, cfg_scale)
                pred_velocity, _ = pred_velocity.chunk(2, dim=0)
            else:
                pred_velocity = self.net(actions, timestep, condition)
            actions = actions + dt * pred_velocity

        return actions


def get_action_model(model_typ="DiT-B", config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        model_typ: (Unused override; model type inferred from config).
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        ActionModel: Initialized diffusion action head.
    """
    action_model_cfg = config.framework.action_model

    model_type = action_model_cfg.action_model_type
    action_hidden_dim = action_model_cfg.action_hidden_dim
    action_dim = action_model_cfg.action_dim
    future_action_window_size = action_model_cfg.future_action_window_size
    past_action_window_size = action_model_cfg.past_action_window_size

    return ActionModel(
        model_type=model_type,  # Model type, e.g., 'DiT-B'
        action_hidden_dim=action_hidden_dim,  # Hidden size of action tokens
        in_channels=action_dim,  # Input channel size
        future_action_window_size=future_action_window_size,  # Future action window size
        past_action_window_size=past_action_window_size,  # Past action window size
    )
