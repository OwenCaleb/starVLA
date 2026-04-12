"""
Qwen3-Embedding encoder adapter for starVLA.

This wrapper keeps the language encoder as a frozen external component and only handles:
- model / tokenizer loading
- subtask text encoding
- minimal smoke test for non-autoregressive subtask representation

Slot construction for VLM routing is intentionally kept separate from the pure encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


TextLike = Union[str, Sequence[str]]


def _normalize_text(text: TextLike) -> str:
	if isinstance(text, str):
		return text.strip()
	return "\n".join(str(item).strip() for item in text if str(item).strip())


def _resolve_model_path(model_path: str) -> str:
	path = Path(model_path).expanduser()
	if path.exists():
		return str(path.resolve())
	return model_path


def _pick_subtask_text(sample: dict) -> str:
	for key in ("subtask", "subtask_text", "subtasks", "instruction", "lang", "text", "prompt"):
		if key in sample and sample[key] is not None:
			return _normalize_text(sample[key])
	raise ValueError("Each sample must contain a subtask text field such as 'subtask' or 'instruction'.")


class Qwen3EmbeddingSubtaskEncoder(nn.Module):
	"""Frozen text encoder for subtask descriptions.

	This module is intentionally pure: it returns text tokens and pooled text embeddings,
	but does not construct VLM slots.
	"""

	def __init__(
		self,
		model_path: str,
		max_length: int = 128,
		pooling: str = "mean",
		freeze: bool = True,
		trust_remote_code: bool = True,
	) -> None:
		super().__init__()
		resolved_model_path = _resolve_model_path(model_path)

		self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=trust_remote_code)
		self.model = AutoModel.from_pretrained(
			resolved_model_path,
			trust_remote_code=trust_remote_code,
			dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
		)

		hidden_size = getattr(self.model.config, "hidden_size", None)
		if hidden_size is None:
			hidden_size = getattr(self.model.config, "dim", None)
		if hidden_size is None and hasattr(self.model.config, "text_config"):
			hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
		if hidden_size is None:
			raise ValueError("Cannot infer hidden size from Qwen3-Embedding config.")

		self.hidden_size = int(hidden_size)
		self.max_length = int(max_length)
		self.pooling = pooling
		self.freeze_backbone = bool(freeze)

		if self.freeze_backbone:
			self.model.requires_grad_(False)
			self.model.eval()

	@property
	def device(self) -> torch.device:
		return next(self.model.parameters()).device

	def _encode_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		batch = self.tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		batch = {key: value.to(self.device) for key, value in batch.items()}

		if self.freeze_backbone:
			with torch.no_grad():
				outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
		else:
			outputs = self.model(**batch, output_hidden_states=True, return_dict=True)

		hidden_states = getattr(outputs, "last_hidden_state", None)
		if hidden_states is None:
			hidden_states = outputs.hidden_states[-1]

		attention_mask = batch.get("attention_mask")
		if attention_mask is None:
			attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long)

		if self.pooling == "mean":
			mask = attention_mask.to(hidden_states.dtype)
			pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
		elif self.pooling == "last":
			lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
			pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), lengths]
		else:
			raise ValueError(f"Unsupported pooling mode: {self.pooling}")

		return hidden_states, attention_mask.bool(), pooled

	def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
		hidden_states, attention_mask, pooled = self._encode_texts(texts)
		return {
			"text_tokens": hidden_states,
			"text_mask": attention_mask,
			"pooled_subtask": pooled,
		}

	def encode_examples(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
		texts = [_pick_subtask_text(sample) for sample in examples]
		return self.encode_texts(texts)

	def forward(
		self,
		examples: Optional[List[dict]] = None,
		texts: Optional[List[str]] = None,
	) -> Dict[str, torch.Tensor]:
		if examples is not None:
			return self.encode_examples(examples)
		if texts is not None:
			return self.encode_texts(texts)
		raise ValueError("Either examples or texts must be provided")


class Qwen3EmbeddingSubtaskSlotAdapter(nn.Module):
	"""VLM-facing subtask slot adapter built on top of the pure text encoder.

	This adapter is the only place where slot queries and projection live.
	"""

	def __init__(self, text_encoder: Qwen3EmbeddingSubtaskEncoder, num_subtask_slots: int = 4) -> None:
		super().__init__()
		self.text_encoder = text_encoder
		self.num_subtask_slots = int(num_subtask_slots)
		self.hidden_size = int(text_encoder.hidden_size)
		self.subtask_queries = nn.Parameter(torch.randn(self.num_subtask_slots, self.hidden_size) * 0.02)
		self.subtask_projector = nn.Sequential(
			nn.LayerNorm(self.hidden_size),
			nn.Linear(self.hidden_size, self.hidden_size),
			nn.GELU(),
			nn.Linear(self.hidden_size, self.hidden_size),
		)

	@property
	def device(self) -> torch.device:
		return self.text_encoder.device

	def _build_slots(self, pooled_embeddings: torch.Tensor) -> torch.Tensor:
		projector_dtype = next(self.subtask_projector.parameters()).dtype
		pooled_embeddings = pooled_embeddings.to(dtype=projector_dtype)
		subtask_queries = self.subtask_queries.to(dtype=projector_dtype)
		projected = self.subtask_projector(pooled_embeddings)
		return subtask_queries.unsqueeze(0) + projected.unsqueeze(1)

	def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
		text_outputs = self.text_encoder.encode_texts(texts)
		text_outputs["subtask_slots"] = self._build_slots(text_outputs["pooled_subtask"])
		return text_outputs

	def encode_examples(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
		text_outputs = self.text_encoder.encode_examples(examples)
		text_outputs["subtask_slots"] = self._build_slots(text_outputs["pooled_subtask"])
		return text_outputs

	def forward(
		self,
		examples: Optional[List[dict]] = None,
		texts: Optional[List[str]] = None,
	) -> Dict[str, torch.Tensor]:
		if examples is not None:
			return self.encode_examples(examples)
		if texts is not None:
			return self.encode_texts(texts)
		raise ValueError("Either examples or texts must be provided")


def build_qwen3_embedding_encoder(cfg) -> Qwen3EmbeddingSubtaskEncoder:
	qwen_cfg = cfg.framework.qwen3_embedding
	return Qwen3EmbeddingSubtaskEncoder(
		model_path=qwen_cfg.model_path,
		max_length=qwen_cfg.get("max_length", 128),
		pooling=qwen_cfg.get("pooling", "mean"),
		freeze=qwen_cfg.get("freeze", True),
		trust_remote_code=qwen_cfg.get("trust_remote_code", True),
	)


def build_qwen3_embedding_slot_adapter(cfg) -> Qwen3EmbeddingSubtaskSlotAdapter:
	qwen_cfg = cfg.framework.qwen3_embedding
	text_encoder = build_qwen3_embedding_encoder(cfg)
	return Qwen3EmbeddingSubtaskSlotAdapter(
		text_encoder=text_encoder,
		num_subtask_slots=qwen_cfg.get("num_subtask_slots", 4),
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Qwen3-Embedding subtask encoder smoke test")
	parser.add_argument("--model_path", type=str, default="/mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/Qwen3-Embedding-0.6B", help="Local path or HF id for Qwen3-Embedding")
	parser.add_argument("--max_length", type=int, default=128)
	parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
	parser.add_argument("--test_mode", type=str, default="forward", choices=["construct", "forward"])
	args = parser.parse_args()

	encoder = Qwen3EmbeddingSubtaskEncoder(
		model_path=args.model_path,
		max_length=args.max_length,
		pooling=args.pooling,
		freeze=True,
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder = encoder.to(device)
	print("[OK] Constructed Qwen3EmbeddingSubtaskEncoder")
	print(f"[INFO] Device: {device}")

	if args.test_mode == "construct":
		print("[OK] construct test finished")
		raise SystemExit(0)

	outputs = encoder(texts=["pick up the red cube", "open the drawer and place the block inside"])
	print("[OK] forward test finished")
	print(f"[INFO] text_tokens shape: {tuple(outputs['text_tokens'].shape)}")
	print(f"[INFO] text_mask shape: {tuple(outputs['text_mask'].shape)}")
	print(f"[INFO] pooled_subtask shape: {tuple(outputs['pooled_subtask'].shape)}")
	print("[INFO] subtask_slots are provided by Qwen3EmbeddingSubtaskSlotAdapter, not the pure encoder")
