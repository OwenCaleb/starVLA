# model_server/adapters/vla_adapter.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class VLAAdapter(ABC):
    """
    部署层统一依赖的接口：
      - self.policy: 底层模型对象
      - load(): 负责加载并就绪（cuda / dtype / eval 等）
      - predict_action(**kwargs): 统一推理入口（保持旧命名）
      
    msg = {
    "type": "infer",                 # 协议字段：告诉 server 这是推理请求
    "request_id": "xxx",             # 协议字段：链路追踪
    "batch_images": [[image_np]],    # 模型字段：给 policy 的输入
    "instructions": ["..."],         # 模型字段：给 policy 的输入
    }
    """
    # 你现在旧协议里最可能污染模型的字段
    # PROTOCOL_KEYS = {"type"}  # 如果你不想模型看到 request_id，也加上它
    PROTOCOL_KEYS = {}  # 如果你不想模型看到 request_id，也加上它
    
    def __init__(self):
        self.policy = None  # 具体子类 load 后填充

    def _sanitize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(kwargs)
        for k in self.PROTOCOL_KEYS:
            clean.pop(k, None)
        return clean
    
    @abstractmethod
    def load(self) -> "VLAAdapter":
        """加载模型并就绪，返回 self 方便链式调用。"""
        raise NotImplementedError

    @abstractmethod
    def _predict_action_impl(self, **kwargs) -> Dict[str, Any]:
        """子类只实现真正的模型推理逻辑。"""
        raise NotImplementedError

    def predict_action(self, **kwargs) -> Dict[str, Any]:
        """统一入口：先清洗，再调用子类实现。"""
        clean = self._sanitize_kwargs(kwargs)
        return self._predict_action_impl(**clean)

    def reset(self, **kwargs) -> Dict[str, Any]:
        """可选：某些策略需要 reset（比如有状态 policy）。默认无操作。"""
        return {"ok": True}

    def close(self) -> None:
        """可选：释放资源。默认无操作。"""
        return
