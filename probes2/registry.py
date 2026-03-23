from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification
from transformers.modeling_layers import GenericForSequenceClassification
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def _normalize_config(config: Any) -> None:
    """Promote hidden_size from text_config for multimodal models (e.g. Qwen3.5)."""
    text_cfg = getattr(config, "text_config", None)
    if not hasattr(config, "hidden_size") and text_cfg is not None:
        config.hidden_size = text_cfg.hidden_size


class GenericForSequenceClassificationWDropout(GenericForSequenceClassification):
    def __init__(self, config: Any) -> None:
        _normalize_config(config)
        super().__init__(config)
        self.head_dropout = nn.Dropout(float(getattr(config, "head_dropout", 0.0)))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        transformer_outputs: BaseModelOutputWithPast = getattr(
            self, self.base_model_prefix
        )(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = self.head_dropout(transformer_outputs.last_hidden_state)
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(
                logits.device, torch.int32
            )
            token_indices = torch.arange(
                input_ids.shape[-1], device=logits.device, dtype=torch.int32
            )
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), last_non_pad_token
        ]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=pooled_logits,
                config=self.config,
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GptOssForSequenceClassification(
    GenericForSequenceClassificationWDropout,
    GptOssPreTrainedModel,
):
    config_class = GptOssConfig
    _keep_in_fp32_modules = ["score"]

    def __init__(self, config: GptOssConfig, **kwargs: Any) -> None:
        num_labels = kwargs.pop("num_labels", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"{self.__class__.__name__} received unexpected constructor kwargs: {unexpected}"
            )
        if num_labels is not None and getattr(config, "num_labels", None) != num_labels:
            config = deepcopy(config)
            config.num_labels = num_labels
        super().__init__(config)


AutoModelForSequenceClassification.register(
    GptOssConfig,
    GptOssForSequenceClassification,
    exist_ok=True,
)

_REGISTRY_IMPORTS = {
    "Qwen3ForCausalLM": (
        "transformers.models.qwen3.modeling_qwen3",
        "Qwen3PreTrainedModel",
    ),
    "Qwen3VLForConditionalGeneration": (
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "Qwen3VLPreTrainedModel",
    ),
    "Qwen3_5ForConditionalGeneration": (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "Qwen3_5PreTrainedModel",
    ),
}


def _build_probe_class(pretrained_cls: type) -> type:
    return type(
        f"{pretrained_cls.__name__.removesuffix('PreTrainedModel')}ForSequenceClassification",
        (GenericForSequenceClassificationWDropout, pretrained_cls),
        {},
    )


# Auto-register all _REGISTRY_IMPORTS entries with AutoModelForSequenceClassification
for _arch, (_mod_name, _cls_name) in _REGISTRY_IMPORTS.items():
    try:
        _mod = importlib.import_module(_mod_name)
        _pretrained_cls = getattr(_mod, _cls_name)
        _probe_cls = _build_probe_class(_pretrained_cls)
        _config_cls = _pretrained_cls.config_class
        AutoModelForSequenceClassification.register(
            _config_cls, _probe_cls, exist_ok=True
        )
    except (ImportError, AttributeError):
        pass


def get_registered_probe_class(config: Any) -> type | None:
    for architecture in getattr(config, "architectures", None) or []:
        if str(architecture) in {
            "GptOssForCausalLM",
            "GptOssForSequenceClassification",
        }:
            return GptOssForSequenceClassification
        module_info = _REGISTRY_IMPORTS.get(str(architecture))
        if module_info is None:
            continue
        module_name, class_name = module_info
        try:
            module = importlib.import_module(module_name)
            pretrained_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        return _build_probe_class(pretrained_cls)
    return None
