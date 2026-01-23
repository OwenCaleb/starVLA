# model_server/adapters/starvla_adapter.py
from __future__ import annotations
from typing import Any, Dict, Optional
import torch

from .vla_adapter import VLAAdapter


class StarVLAAdapter(VLAAdapter):
    """
    用这个替代原来 server_policy.py 里写死的 StarVLA 加载+cuda+bf16 逻辑
    """

    # ===== CONFIG (所有可配置参数集中在这里) =====
    DEFAULT_DEVICE = "cuda"
    # ==========================================

    def __init__(
        self,
        ckpt_path: str,
        use_bf16: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.use_bf16 = use_bf16
        self.device = device or self.DEFAULT_DEVICE

    def load(self) -> "StarVLAAdapter":
        # 这里的 import 放在 load() 内，避免“部署层 import 就必须装 StarVLA”
        from starVLA.model.framework.base_framework import baseframework

        vla = baseframework.from_pretrained(self.ckpt_path)

        if self.use_bf16:
            vla = vla.to(torch.bfloat16)

        vla = vla.to(self.device)
        vla.eval()

        self.policy = vla
        return self

    @torch.no_grad()
    def _predict_action_impl(self, **kwargs) -> Dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("StarVLAAdapter not loaded. Call load() first.")
        return self.policy.predict_action(**kwargs)
