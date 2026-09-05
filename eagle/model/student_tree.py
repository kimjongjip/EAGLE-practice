"""Student exercise used by the real EAGLE-3 drafter at generation time."""


def build_tree_mask_and_positions(mask_index_list, total_tokens, device=None):
    """Build ancestor visibility and relative tree positions.

    Args:
        mask_index_list: Parent tree indices, with length ``total_tokens``.
            Node ``i + 1`` has parent ``mask_index_list[i]``. The anchor/root
            is node 0. Parents precede their children: ``0 <= parent <= i``.
        total_tokens: Number of draft nodes EXCLUDING the root/anchor.
        device: Output device (None means the default torch device).

    Returns:
        tree_mask: float32 0/1 tensor, shape [1, 1, total_tokens + 1,
            total_tokens + 1]. Each row sees itself, the root and its ancestors;
            it cannot see siblings or other branches. This is a visibility
            mask, not the additive 0/-inf mask used inside attention.
        tree_position_ids: int64 tensor, shape [total_tokens + 1]. The root
            has position 0; other nodes have their depth as position.

    In Colab, implement a function with this signature and assign it to
    ``eagle.model.student_tree.build_tree_mask_and_positions`` before calling
    ``eagle_model.eagenerate(...)``. No reload or source-file edit is needed.
    """
    raise NotImplementedError(
        "Implement EAGLE tree attention mask and tree position IDs. "
        "Assign your function to student_tree.build_tree_mask_and_positions "
        "before calling eagenerate()."
    )
