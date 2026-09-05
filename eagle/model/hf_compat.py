"""Feature-based adapters for the vendored Qwen3 model's Transformers APIs.

Keep EAGLE's attention, cache and generation code independent of HF API churn.
No packages are installed and no Transformers globals are patched here.
"""

from functools import wraps
from inspect import signature

try:
    from typing import Unpack
except ImportError:  # Python 3.10; typing_extensions is a torch dependency.
    from typing_extensions import Unpack

import torch
from transformers import utils as _utils
from transformers import masking_utils as _masking
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import generic as _generic


def _utils_symbol(name):
    return getattr(_utils, name, None) or getattr(_generic, name, None)


# In 4.53, LossKwargs describes num_items_in_batch. Later Transformers moves
# that same loss-normalization field into the broader TransformersKwargs
# TypedDict, used by upstream Qwen3.forward. Both are optional-key TypedDicts,
# so combining them with FlashAttentionKwargs preserves runtime typing.
LossKwargs = _utils_symbol("LossKwargs")
if LossKwargs is None:
    LossKwargs = _utils_symbol("TransformersKwargs")
    if LossKwargs is None or "num_items_in_batch" not in LossKwargs.__annotations__:
        raise ImportError("Transformers exposes neither LossKwargs nor compatible TransformersKwargs")


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    pass


def _docstring_noop(obj=None, **kwargs):
    # Documentation only: support both @auto_docstring and @auto_docstring(...).
    return obj if obj is not None else lambda decorated: decorated


auto_docstring = _utils_symbol("auto_docstring") or _docstring_noop


def _can_return_tuple(func):
    # This decorator affects outputs, so it must NOT fall back to a no-op.
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        requested = kwargs.pop("return_dict", return_dict)
        if requested is not None:
            return_dict = requested
        output = func(self, *args, **kwargs)
        return output.to_tuple() if not return_dict and not isinstance(output, tuple) else output

    return wrapper


can_return_tuple = _utils_symbol("can_return_tuple") or _can_return_tuple


def _adapt_mask_factory(factory):
    parameters = signature(factory).parameters

    @wraps(factory)
    def create_mask(**kwargs):
        # HF explicitly passes through prepared 4D masks. EAGLE has already
        # applied its causal/prefix/tree mask and uses its own KVCache objects.
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            return attention_mask
        if "input_embeds" not in parameters:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if "cache_position" not in parameters:
            kwargs.pop("cache_position", None)
        return factory(**kwargs)

    return create_mask


create_causal_mask = _adapt_mask_factory(_masking.create_causal_mask)
create_sliding_window_causal_mask = _adapt_mask_factory(_masking.create_sliding_window_causal_mask)


def qwen3_rope_init(config):
    """Resolve the installed HF implementation without changing RoPE math."""
    parameters = getattr(config, "rope_parameters", None)
    if parameters is None:
        parameters = getattr(config, "rope_scaling", None) or {}
    rope_type = parameters.get("rope_type", parameters.get("type", "default"))
    if rope_type == "default" and "default" not in ROPE_INIT_FUNCTIONS:
        # Recent HF moved default RoPE from the registry to each model class.
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        return rope_type, Qwen3RotaryEmbedding.compute_default_rope_parameters
    return rope_type, ROPE_INIT_FUNCTIONS[rope_type]


def causal_lm_tied_weights_keys():
    # New HF loaders expand an explicit target->source map; old loaders expect
    # a list of target patterns. This does not enable tying when config disables it.
    if hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
        return {"lm_head.weight": "model.embed_tokens.weight"}
    return ["lm_head.weight"]


def initialize_qwen3_rope(module):
    """Restore non-checkpoint RoPE buffers after HF's meta-device loading."""
    inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
    module.inv_freq.copy_(inv_freq)
    module.original_inv_freq.copy_(inv_freq)
