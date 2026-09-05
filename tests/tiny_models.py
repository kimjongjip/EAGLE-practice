"""Small random checkpoints using the actual Qwen3 and EAGLE-3 classes.

Only dimensions/weights are small. No drafter, candidate tree, verification,
posterior or generation loop is mocked or replaced.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from eagle.model.cnets import Model
from eagle.model.configs import EConfig
from eagle.model.ea_model import EaModel
from eagle.model.modeling_qwen3_kv import Qwen3Config, Qwen3ForCausalLM


def write_tiny_checkpoints(directory):
    torch.manual_seed(7)
    base_path = Path(directory) / "base"
    draft_path = Path(directory) / "draft"
    base_path.mkdir()
    draft_path.mkdir()
    config = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=31, pad_token_id=0,
        attn_implementation="eager",
    )
    base = Qwen3ForCausalLM(config).eval()
    base.save_pretrained(base_path, max_shard_size="32KB")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0")),
        unk_token="t0", bos_token="t1", eos_token="t31", pad_token="t0",
    )
    tokenizer.save_pretrained(base_path)
    draft_config = EConfig(
        vocab_size=32, draft_vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        # Like Qwen3-4B's drafter, attention width differs from hidden_size.
        head_dim=12, max_position_embeddings=128, rope_theta=1000000.0,
    )
    draft_config.to_json_file(draft_path / "config.json")
    draft = Model(draft_config, total_tokens=6, depth=2, top_k=2).eval()
    draft.d2t.copy_(torch.arange(16))  # draft token i corresponds to target 2*i
    draft.t2d[::2] = True
    from safetensors.torch import save_file

    save_file(draft.state_dict(), draft_path / "model.safetensors")
    return str(base_path), str(draft_path)


def load_tiny_eagle(paths):
    model = EaModel.from_pretrained(
        base_model_path=paths[0], ea_model_path=paths[1],
        total_token=6, depth=2, top_k=2,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    # Keep the random fixture running for several verification rounds.
    model.tokenizer.eos_token_id = None
    return model
