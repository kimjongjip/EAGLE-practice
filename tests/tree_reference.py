"""Regression fixture: the original cnets.py tree construction.

Never imported by core inference; used to check the default tree builder.
"""

import torch


def build_tree_mask_and_positions(mask_index_list, total_tokens, device=None):
    tree_mask = torch.eye(total_tokens + 1, device=device).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
    tree_position_ids = torch.sum(tree_mask, dim=1) - 1
    return tree_mask.float()[None, None], tree_position_ids
