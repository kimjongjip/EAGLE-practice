from types import SimpleNamespace
import unittest

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from eagle.model import hf_compat


class CompatibilityTests(unittest.TestCase):
    def test_legacy_drafter_rope_settings_survive_hf_normalization(self):
        from eagle.model.configs import EConfig

        for kind in ("linear", "dynamic"):
            with self.subTest(kind=kind):
                scaling = {"type": kind, "factor": 2.0}
                config = EConfig(rope_scaling=scaling, rope_theta=1000000.0)
                self.assertEqual(scaling, {"type": kind, "factor": 2.0})
                self.assertEqual(config.rope_scaling, scaling)
                self.assertEqual(config.rope_theta, 1000000.0)

    def test_return_tuple_fallback_preserves_config_and_explicit_override(self):
        class Model:
            config = SimpleNamespace(return_dict=False)

            @hf_compat._can_return_tuple
            def forward(self):
                return BaseModelOutputWithPast(last_hidden_state=torch.ones(1))

        model = Model()
        self.assertIsInstance(model.forward(), tuple)
        self.assertIsInstance(model.forward(return_dict=None), tuple)
        self.assertIsInstance(model.forward(return_dict=True), BaseModelOutputWithPast)
        model.config.return_dict = True
        self.assertIsInstance(model.forward(return_dict=False), tuple)

    def test_docstring_fallback_supports_both_decorator_forms(self):
        def function():
            return 42
        self.assertIs(hf_compat._docstring_noop(function), function)
        self.assertIs(hf_compat._docstring_noop(custom_intro="docs")(function), function)

    def test_prepared_tree_mask_is_passed_through_unchanged(self):
        mask = torch.randn(1, 1, 4, 7)
        for factory in (hf_compat.create_causal_mask, hf_compat.create_sliding_window_causal_mask):
            self.assertIs(factory(attention_mask=mask, past_key_values=object()), mask)

    def test_modern_mask_argument_adaptation(self):
        def modern(config, inputs_embeds, attention_mask, past_key_values):
            return inputs_embeds, past_key_values
        adapted = hf_compat._adapt_mask_factory(modern)
        embeddings = torch.ones(1, 1, 2)
        actual, cache = adapted(
            config=None, input_embeds=embeddings, attention_mask=None,
            past_key_values="cache", cache_position=None,
        )
        self.assertIs(actual, embeddings)
        self.assertEqual(cache, "cache")
