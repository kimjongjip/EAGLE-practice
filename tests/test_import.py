"""Fresh-process imports, with no model downloads or optional app packages."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class ImportTests(unittest.TestCase):
    def test_core_and_qwen3_without_app_dependencies(self):
        code = """
import builtins
import sys
blocked = {'gradio', 'openai', 'anthropic', 'wandb', 'fastchat', 'fschat'}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ImportError('Optional application package is unavailable: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from eagle.model.ea_model import EaModel
from eagle.model.modeling_qwen3_kv import Qwen3ForCausalLM, KwargsForCausalLM
from typing import get_type_hints
assert 'num_items_in_batch' in get_type_hints(KwargsForCausalLM)
assert not KwargsForCausalLM.__required_keys__
assert not any(name.split('.')[0] in blocked for name in sys.modules)
assert 'eagle.model.modeling_llama_kv' not in sys.modules
assert 'eagle.model.modeling_mixtral_kv' not in sys.modules
assert 'eagle.model.modeling_qwen2_kv' not in sys.modules
print('EaModel and Qwen3 import OK')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("EaModel and Qwen3 import OK", result.stdout)
