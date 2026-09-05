import random
import unittest

import torch

from eagle.model import student_tree
from tests.tree_reference import build_tree_mask_and_positions


class TreeMaskTests(unittest.TestCase):
    def test_student_default_is_todo(self):
        with self.assertRaisesRegex(NotImplementedError, "Implement EAGLE"):
            student_tree.build_tree_mask_and_positions([0], 1)

    def test_branching_example_and_no_sibling_leakage(self):
        mask, positions = build_tree_mask_and_positions([0, 0, 1], 3)
        expected = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
        ], dtype=torch.float32)[None, None]
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(torch.equal(positions, torch.tensor([0, 1, 1, 2])))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(positions.dtype, torch.int64)
        self.assertEqual(mask[0, 0, 1, 2].item(), 0)
        self.assertEqual(mask[0, 0, 2, 1].item(), 0)
        self.assertEqual(mask[0, 0, 3, 2].item(), 0)

    def test_root_only_chain_and_wide_tree(self):
        for parents, depths in [([], [0]), ([0, 1, 2], [0, 1, 2, 3]), ([0, 0, 0], [0, 1, 1, 1])]:
            with self.subTest(parents=parents):
                mask, positions = build_tree_mask_and_positions(parents, len(parents))
                self.assertEqual(mask.shape, (1, 1, len(parents) + 1, len(parents) + 1))
                self.assertEqual(positions.tolist(), depths)

    def test_random_trees_against_parent_walk(self):
        rng = random.Random(0)
        for count in (1, 7, 31, 59):
            parents = [rng.randrange(i + 1) for i in range(count)]
            mask, positions = build_tree_mask_and_positions(parents, count)
            for node in range(count + 1):
                ancestors = {node}
                current = node
                while current:
                    current = parents[current - 1]
                    ancestors.add(current)
                self.assertEqual(set(torch.where(mask[0, 0, node])[0].tolist()), ancestors)
                self.assertEqual(positions[node].item(), len(ancestors) - 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_device_argument(self):
        cpu = build_tree_mask_and_positions([0, 0, 1], 3, device="cpu")
        gpu = build_tree_mask_and_positions([0, 0, 1], 3, device="cuda")
        for expected, actual in zip(cpu, gpu):
            self.assertEqual(actual.device.type, "cuda")
            self.assertTrue(torch.equal(expected, actual.cpu()))
