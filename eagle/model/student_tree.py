import torch


def build_tree_mask_and_positions(mask_index_list, total_tokens, device=None):
    """Build EAGLE tree attention mask and relative position IDs."""

    # Each node can attend to itself.
    tree_mask = torch.eye(
        total_tokens + 1,
        dtype=torch.bool,
        device=device
    )

    # Every node can attend to the root(anchor).
    tree_mask[:, 0] = True

    # A child can attend to everything its parent can attend to.
    for i, parent in enumerate(mask_index_list):
        tree_mask[i + 1] |= tree_mask[parent]

    # Number of visible ancestors corresponds to tree depth.
    tree_position_ids = tree_mask.sum(dim=1).long() - 1

    # EAGLE expects [1, 1, N, N] float visibility mask.
    tree_mask = tree_mask.float()[None, None]

    return tree_mask, tree_position_ids
