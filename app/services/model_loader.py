# SPDX-License-Identifier: Apache-2.0
"""Model loader for PhoBERT Multi-Task (Civic Classification + Urgency + Embeddings)."""

import json
import logging
from typing import Any

from app.core.config import DEVICE, MULTITASK_DIR

logger = logging.getLogger("opencity.ai.model")

_model = None
_tokenizer = None
_meta: dict[str, Any] = {}
_is_multitask: bool = False


def _define_multitask_class():
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class PhoBertMultiTask(nn.Module):
        """1 Shared PhoBERT Backbone -> 3 Heads: Civic 12, Urgency 3, Semantic Embeddings."""

        def __init__(self, config, num_categories: int = 12, num_urgencies: int = 3):
            super().__init__()
            self.roberta = AutoModel.from_config(config, add_pooling_layer=False)
            hidden_size = config.hidden_size
            dropout = getattr(config, "hidden_dropout_prob", 0.1)
            self.category_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_categories),
            )
            self.urgency_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_urgencies),
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            sequence_output = outputs.last_hidden_state
            cls_output = sequence_output[:, 0, :]
            category_logits = self.category_classifier(cls_output)
            urgency_logits = self.urgency_classifier(cls_output)

            # Head 3: Mean-pooled sentence embeddings with L2 normalization
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            sentence_embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            return {
                "category_logits": category_logits,
                "urgency_logits": urgency_logits,
                "sentence_embeddings": sentence_embeddings,
            }

    return PhoBertMultiTask


def load_model():
    """Load PhoBERT Multi-Task weights and metadata onto the target device."""
    global _model, _tokenizer, _meta, _is_multitask

    if _model is not None:
        return _model, _tokenizer, _meta, _is_multitask

    weights_file = MULTITASK_DIR / "model.safetensors"
    config_file = MULTITASK_DIR / "config.json"

    if weights_file.exists() and config_file.exists():
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoTokenizer

            PhoBertMultiTask = _define_multitask_class()
            config = AutoConfig.from_pretrained(str(MULTITASK_DIR))
            _tokenizer = AutoTokenizer.from_pretrained(str(MULTITASK_DIR))
            model = PhoBertMultiTask(config, num_categories=12, num_urgencies=3)
            state_dict = load_file(str(weights_file))
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            _model = model
            _is_multitask = True

            meta_file = MULTITASK_DIR / "phobert_multitask_metadata.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    _meta = json.load(f)

            logger.info("PhoBERT Multi-Task loaded successfully on device: %s", DEVICE)
        except Exception as exc:
            logger.warning("PhoBERT Multi-Task could not be loaded (%s). Running in fallback mode.", exc)
            _model = None
            _is_multitask = False
    else:
        logger.info("No weights found at %s. Running in fallback mode.", MULTITASK_DIR)
        _model = None
        _is_multitask = False

    return _model, _tokenizer, _meta, _is_multitask


# Eager initial load
load_model()


def get_ai_engine():
    """Retrieve the cached model instance and metadata."""
    return load_model()
