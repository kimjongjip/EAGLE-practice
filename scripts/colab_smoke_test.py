"""Import check by default; --full downloads and runs Qwen3-4B + EAGLE-3.

No installation commands. Full inference uses the built-in tree builder:
    python scripts/colab_smoke_test.py --full
"""

import argparse
import importlib
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Download the models and generate a short answer on CUDA")
    tree = parser.add_mutually_exclusive_group()
    tree.add_argument("--reference-tree", action="store_true", help="Optional regression against the upstream test fixture")
    tree.add_argument("--student-module", help="Import build_tree_mask_and_positions from this Python module")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--drafter", default="AngelSlim/Qwen3-4B_eagle3")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Explain speculative decoding in one short sentence.")
    args = parser.parse_args()

    import torch
    from eagle.model.ea_model import EaModel
    # Also check the backend: EaModel intentionally loads architectures lazily.
    from eagle.model.modeling_qwen3_kv import Qwen3ForCausalLM

    for package in ("torch", "transformers", "huggingface_hub", "safetensors", "tokenizers"):
        print(f"{package}: {version(package)}", flush=True)
    print("EaModel and Qwen3 backend imports: OK", flush=True)
    if not args.full:
        return
    if not torch.cuda.is_available():
        parser.error("Full smoke test needs a CUDA GPU; select a GPU runtime in Colab.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Explicit single-GPU placement avoids automatic CPU/disk offloading.
    # accelerate is needed by Transformers 4.x for device_map and is in Colab.
    model = EaModel.from_pretrained(
        base_model_path=args.base_model, ea_model_path=args.drafter,
        use_eagle3=True, total_token=16, depth=3, top_k=4,
        torch_dtype=dtype, device_map={"": "cuda:0"},
        attn_implementation="eager",
    ).eval()
    print("Model loading: OK", flush=True)

    from eagle.model import student_tree
    if args.reference_tree:
        from tests.tree_reference import build_tree_mask_and_positions
    elif args.student_module:
        build_tree_mask_and_positions = importlib.import_module(args.student_module).build_tree_mask_and_positions
    else:
        build_tree_mask_and_positions = student_tree.build_tree_mask_and_positions

    original = student_tree.build_tree_mask_and_positions
    calls = 0

    def counted_tree_builder(parents, count, device=None):
        nonlocal calls
        calls += 1
        return build_tree_mask_and_positions(parents, count, device)

    tokenizer = model.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    max_length = max(256, input_ids.shape[1] + args.max_new_tokens + 64)
    try:
        student_tree.build_tree_mask_and_positions = counted_tree_builder
        output = model.eagenerate(
            input_ids, temperature=0.0, max_new_tokens=args.max_new_tokens, max_length=max_length,
        )
    finally:
        student_tree.build_tree_mask_and_positions = original
    assert calls >= 2, "Tree builder must run during initialization and the following round"
    assert output.shape[1] > input_ids.shape[1], "No tokens generated"
    assert torch.equal(output[:, :input_ids.shape[1]], input_ids), "Prompt was changed"
    baseline = model.naivegenerate(
        input_ids, temperature=0.0, max_new_tokens=args.max_new_tokens, max_length=max_length,
    )
    common_length = min(output.shape[1], baseline.shape[1])
    assert torch.equal(output[:, :common_length], baseline[:, :common_length]), "Greedy EAGLE/AR tokens differ"
    generated = output[0, input_ids.shape[1]:]
    print(f"Full inference: OK; generated={generated.numel()}, tree calls={calls}, greedy AR comparison=OK")
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"Peak CUDA allocation: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
