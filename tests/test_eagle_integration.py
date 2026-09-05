import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

from eagle.model import student_tree
from eagle.model.utils import (
    initialize_tree, tree_decoding, evaluate_posterior, update_inference_inputs,
)
from eagle.model.kv_cache import initialize_past_key_values
from tests.tiny_models import load_tiny_eagle, write_tiny_checkpoints
from tests.tree_reference import build_tree_mask_and_positions as reference


class EagleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.paths = write_tiny_checkpoints(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.model = load_tiny_eagle(self.paths)
        self.prompt = torch.tensor([[1, 4, 7]])

    def test_loading_and_generation_work_without_patch(self):
        self.assertEqual(self.model.ea_layer.midlayer.self_attn.head_dim, 12)
        self.assertEqual(self.model.ea_layer.midlayer.self_attn.rotary_emb.base, 1000000.0)
        self.assertIsNone(self.model.ea_layer.config.rope_scaling)
        output = self.model.eagenerate(self.prompt, max_new_tokens=8, max_length=64)
        self.assertGreater(output.shape[1], self.prompt.shape[1])
        self.assertTrue(torch.equal(output[:, :self.prompt.shape[1]], self.prompt))

    def test_runtime_patch_reaches_target_and_every_generation_round(self):
        calls, verifications = [], []

        def student(parents, total_tokens, device=None):
            mask, positions = reference(parents, total_tokens, device)
            calls.append((list(parents), mask, positions))
            return mask, positions

        def observe_target(module, args, kwargs):
            positions = kwargs.get("position_ids")
            if positions is not None:
                parents, mask, relative = calls[-1]
                self.assertIs(module.tree_mask, mask)
                prefix_length = kwargs["past_key_values"][0][0].shape[2]
                self.assertTrue(torch.equal(positions[0], relative + prefix_length))
                self.assertEqual(kwargs["input_ids"].shape[1], len(parents) + 1)
                verifications.append(prefix_length)

        handle = self.model.base_model.model.register_forward_pre_hook(observe_target, with_kwargs=True)
        try:
            with patch.object(student_tree, "build_tree_mask_and_positions", student):
                output = self.model.eagenerate(self.prompt, max_new_tokens=8, max_length=64)
        finally:
            handle.remove()
        self.assertGreaterEqual(len(verifications), 2)
        self.assertEqual(len(calls), len(verifications) + 1)
        self.assertGreater(output.shape[1], self.prompt.shape[1])
        self.assertTrue(torch.equal(output[:, :3], self.prompt))

        # Reassign after loading and after generation; the module lookup must
        # immediately pick up the new function, without reload or file edits.
        with patch.object(student_tree, "build_tree_mask_and_positions", side_effect=RuntimeError("replacement reached")):
            with self.assertRaisesRegex(RuntimeError, "replacement reached"):
                self.model.eagenerate(self.prompt, max_new_tokens=8, max_length=64)

    @torch.no_grad()
    def test_real_tree_paths_verification_posterior_and_next_round(self):
        model = self.model
        cache, storage, lengths = initialize_past_key_values(model.base_model, max_length=64)
        recorded = []

        def student(parents, count, device=None):
            recorded.append(list(parents))
            return reference(parents, count, device)

        with patch.object(student_tree, "build_tree_mask_and_positions", student):
            draft, retrieve, mask, positions, _, _, _ = initialize_tree(self.prompt, model, cache, None)
            parents = recorded[0]
            for path in retrieve.tolist():
                path = [node for node in path if node >= 0]
                self.assertEqual(path[0], 0)
                for parent, child in zip(path, path[1:]):
                    self.assertEqual(parents[child - 1], parent)
                self.assertEqual(set(torch.where(mask[0, 0, path[-1]])[0].tolist()), set(path))

            model.base_model.model.tree_mask = mask
            logits, hidden, _ = tree_decoding(model, draft, cache, positions, self.prompt, retrieve)
            # Tree verification must equal an independent sequential forward
            # for each real candidate path, including its prefix and RoPE positions.
            model.base_model.model.tree_mask = None
            for row, path in enumerate(retrieve.tolist()):
                path = [node for node in path if node >= 0]
                sequence = torch.cat((self.prompt, draft[:, path]), dim=1)
                expected = model.base_model(sequence, use_cache=False).logits[:, 3:]
                torch.testing.assert_close(logits[row, :len(path)], expected[0], atol=1e-5, rtol=1e-5)

            padded = torch.cat((draft, torch.tensor([[-1]])), dim=1)
            candidates = padded[0, retrieve]
            best, accepted, probabilities = evaluate_posterior(logits, candidates, None)
            result = update_inference_inputs(
                self.prompt, candidates, best, accepted, retrieve, None, 0,
                storage, lengths, model, hidden, probabilities,
            )
            self.assertEqual(len(recorded), 2)
            self.assertEqual(result[0].shape[1], 3 + accepted.item() + 1)
            self.assertTrue(torch.all(lengths == result[0].shape[1]))

    def test_eagle_greedy_matches_official_autoregressive_loop(self):
        speculative = self.model.eagenerate(self.prompt, max_new_tokens=8, max_length=64)
        baseline = self.model.naivegenerate(self.prompt, max_new_tokens=8, max_length=64)
        # Upstream EAGLE checks the limit after accepting an entire branch.
        length = min(speculative.shape[1], baseline.shape[1])
        self.assertTrue(torch.equal(speculative[:, :length], baseline[:, :length]))

    @torch.no_grad()
    def test_target_logits_match_installed_huggingface_qwen3(self):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        base = self.model.base_model
        upstream = Qwen3ForCausalLM(copy.deepcopy(base.config)).eval()
        upstream.load_state_dict(base.state_dict(), strict=True)
        actual = base(self.prompt, use_cache=False).logits
        expected = upstream(self.prompt, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
