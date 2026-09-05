# On-Device AI 실습: EAGLE-3 + Qwen3

공식 EAGLE의 drafter와 generation loop를 사용하면서, **실제 draft tree의
attention mask와 position IDs를 만드는 함수 하나**를 학생이 구현하는 실습입니다.
기본 `student_tree.py`는 TODO입니다. Import와 모델 로딩은 가능하며, 학생 함수를
연결하기 전 `eagenerate()`를 실행하면 첫 draft tree 구성 시 `NotImplementedError`가 발생합니다.

## 1. Colab 시작 — 기본 패키지 사용

Colab에서 GPU 런타임을 선택하고 다음 셀부터 시작합니다.
`requirements.txt` 설치나 `pip install -e .`는 필요하지 않습니다.
기존 `requirements.txt`와 `setup.py`는 upstream 기록으로 보존되어 있으며,
수업에서는 이 파일들의 torch/Transformers pin을 적용하지 않습니다.

```python
%cd /content
!git clone -q https://github.com/kimjongjip/EAGLE-practice.git

import sys
sys.path.insert(0, "/content/EAGLE-practice")

from eagle.model.ea_model import EaModel
```

환경 확인은 설치 없이 가능합니다.

```python
import torch
import transformers
from importlib.metadata import version

for package in ("torch", "transformers", "huggingface_hub", "safetensors", "tokenizers", "accelerate"):
    print(package, version(package))
print("CUDA:", torch.cuda.is_available())
```

필수 패키지는 torch, Transformers 및 그 의존성(tokenizers, huggingface_hub,
safetensors 등)입니다. 아래의 `device_map` 로딩에는 Transformers 4.x에서
accelerate도 필요합니다. **확인한 Colab 기본 GPU 패키지 목록에는 이들이 모두
있으므로 기본 경로의 추가 설치는 0개입니다.** 실제 Colab 세션에서의 실행을
직접 검증한 것은 아니며, 런타임별 설치 목록은 달라질 수 있습니다.

2026-09-05에 확인한 [Google 공식 GPU 패키지 목록](https://github.com/googlecolab/backend-info/blob/f14042477cce6487b65eb419661bdf0375610a3f/pip-freeze.gpu.txt)은
torch 2.11.0+cu128, Transformers 5.16.1, huggingface_hub 1.29.0,
safetensors 0.8.0, tokenizers 0.23.1, accelerate 1.14.0을 포함합니다.
이 목록은 Google의 배포 스냅샷이며 모든 실행 중인 세션의 상태를 보장하지 않습니다.

gradio, openai, anthropic, wandb, fschat/fastchat은 이 Qwen3 추론 실습에
필요하지 않습니다. FastChat 대신 tokenizer의 chat template를 사용합니다.
Qwen3의 이 tokenizer 경로에는 sentencepiece를 추가할 필요도 없습니다.

## 2. 모델 로딩

```python
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

eagle_model = EaModel.from_pretrained(
    base_model_path="Qwen/Qwen3-4B",
    ea_model_path="AngelSlim/Qwen3-4B_eagle3",
    use_eagle3=True,
    total_token=16,
    depth=3,
    top_k=4,
    torch_dtype=dtype,
    device_map={"": "cuda:0"},
    attn_implementation="eager",
).eval()
```

이 단계에서는 학생 함수를 호출하지 않습니다. 첫 실행에는 Hugging Face에서
모델을 다운로드합니다. 라이브러리 설치와 모델 다운로드는 별개입니다.
명시적인 단일 GPU 배치를 사용합니다. Colab T4는 float16을, BF16 지원 GPU는
bfloat16을 사용합니다. 이번 검증의 BF16 peak GPU 할당량은 약 8.72 GiB였으며,
실제 메모리 사용량은 GPU, dtype, prefix 길이와 tree 크기에 따라 달라집니다.

`total_token=16`은 기존 구현에서 anchor를 포함한 tree budget입니다.
학생 함수에 전달되는 `total_tokens`는 anchor를 제외하므로 여기서는 15입니다.

## 3. 학생이 구현할 함수

정확한 signature:

```python
def build_tree_mask_and_positions(mask_index_list, total_tokens, device=None):
    ...
```

입력 계약:

- root/anchor의 tree index는 0입니다.
- draft node의 index는 1부터 `total_tokens`까지입니다.
- `mask_index_list[i]`는 node `i + 1`의 parent tree index입니다.
- 부모는 자식보다 앞에 있습니다: `0 <= mask_index_list[i] <= i`.
- 부모 목록은 실제 EAGLE drafter가 후보를 선택한 결과입니다.

출력 계약:

| 출력 | shape | dtype와 의미 |
| --- | --- | --- |
| `tree_mask` | `[1, 1, N+1, N+1]` | float32 0/1 visibility. 행=query, 열=key. 자기 자신, root, ancestor만 1 |
| `tree_position_ids` | `[N+1]` | int64. root=0, 나머지는 depth. 같은 깊이의 sibling은 같은 position |

`N = total_tokens`입니다. Attention 내부의 0/-inf additive mask와 구별하세요.
Prefix 길이를 position에 더하는 작업은 공식 `tree_decoding()`이 수행합니다.
학생 함수에서는 상대적인 depth만 반환합니다.

예를 들어 부모 목록 `[0, 0, 1]`은 root의 자식 1·2와 node 1의 자식 3을 뜻합니다.
기대 mask와 position은 다음과 같습니다.

```text
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 0, 1, 0],
 [1, 1, 0, 1]]

[0, 1, 1, 2]
```

다음 셀에 자신의 구현을 작성하고 연결합니다. 구현을 바꿀 때에는 셀을 다시
실행하면 됩니다. 파일 수정, 모델 재로딩, `importlib.reload()`는 필요하지 않습니다.

```python
import eagle.model.student_tree as student_tree

def my_build_tree_mask_and_positions(mask_index_list, total_tokens, device=None):
    # TODO: 위 계약을 만족하는 두 torch.Tensor를 구현하세요.
    raise NotImplementedError("학생 구현")

student_tree.build_tree_mask_and_positions = my_build_tree_mask_and_positions
```

원래 CPU에서 mask를 구성하던 동작을 유지하기 위해 `cnets.py`는
`device="cpu"`로 호출합니다. 반환된 position은 기존 코드가 target device로
옮깁니다. 자신의 함수에서는 전달된 `device`를 사용해 tensor를 생성하세요.

모델 다운로드 없이 자신의 구현을 점검하는 셀:

```python
mask, positions = student_tree.build_tree_mask_and_positions([0, 0, 1], 3, device="cpu")
expected = torch.tensor([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
], dtype=torch.float32)[None, None]
assert mask.shape == (1, 1, 4, 4)
assert mask.dtype == torch.float32
assert positions.dtype == torch.int64
assert torch.equal(mask, expected)
assert torch.equal(positions, torch.tensor([0, 1, 1, 2]))
assert mask[0, 0, 3, 2] == 0  # 다른 branch가 보이면 안 됩니다.
```

## 4. 실제 EAGLE 생성

학생 구현을 연결한 뒤 실행합니다.

```python
tokenizer = eagle_model.get_tokenizer()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain speculative decoding in one short sentence."}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")

output_ids = eagle_model.eagenerate(
    input_ids,
    temperature=0.0,
    max_new_tokens=8,
    max_length=256,
)
print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True))
```

기존 EAGLE은 branch를 수락한 **뒤** `new_token > max_new_tokens`를 확인합니다.
따라서 8을 지정해도 실제 생성량은 몇 토큰 더 많을 수 있습니다. 이 종료 로직도
보존했습니다. `max_length=256`은 짧은 smoke prompt용이며, 긴 입력에는 prefix와
생성 토큰, draft tree를 수용할 여유가 있는 값을 사용하세요. 기존 모델 객체는
처음 생성한 KV cache를 재사용하므로 더 큰 cache가 필요하면 모델을 다시 로딩하세요.

## 5. 코드 연결과 보존 범위

```text
EaModel.from_pretrained()                 target·drafter 로딩, init_tree()
  → eagenerate()
  → initialize_tree()                    target prefill, anchor 선택
  → ea_layer.topK_genrate()               실제 drafter forward와 top-k 후보 선택
      → draft_tokens / draft_parents / mask_index_list
      → student_tree.build_tree_mask_and_positions()
      → 기존 retrieve_indices 구성
  → base_model.model.tree_mask 설정
  → tree_decoding()                      prefix + relative depth, target 검증, logits 재정렬
  → evaluate_posterior()                 수락할 candidate와 길이 결정
  → update_inference_inputs()            수락 branch의 KV 복사, 다음 topK_genrate()
  → 다음 round
```

`cnets.py`의 부모 목록 계산 다음에 있던 identity/root mask 초기화,
부모 mask 누적, row sum에서 depth 계산, float32·4D 변환을 함수 경계로 추출했습니다.
원래 구현은 강사용 `tests/tree_reference.py`에 보존했습니다. Core inference는
이 reference를 자동으로 import하거나 대신 실행하지 않습니다.

`utils.py`의 `tree_decoding()`, `evaluate_posterior()`, `update_inference_inputs()`,
`retrieve_indices` 알고리즘, generation loop와 `kv_cache.py`는 유지했습니다.
학생 hook은 EAGLE-3의 `cnets.py`에만 적용되며 EAGLE-2의 `cnets1.py`는 그대로입니다.

## 6. 호환성 수정의 근거

`eagle/model/hf_compat.py`에서 symbol과 함수 signature를 확인합니다.
Transformers 버전 문자열로 분기하지 않습니다.

- `LossKwargs`: 있으면 사용하고, 없으면 같은 `num_items_in_batch` 필드를 가진
  최신 `TransformersKwargs` TypedDict를 사용합니다. `FlashAttentionKwargs`와
  결합한 `KwargsForCausalLM`의 import 및 runtime type hints를 테스트합니다.
- `Unpack`: 표준 typing을 사용하고 Python 3.10에서는 torch 의존성인
  typing_extensions를 사용합니다. 이 타입 때문에 processing_utils를 import하지 않습니다.
- `auto_docstring`: 제공되는 decorator를 사용합니다. 없으면 문서 생성에만 no-op을
  적용합니다. `can_return_tuple`은 반환 형식에 영향을 주므로 fallback에서도
  `return_dict` 설정과 `to_tuple()` 동작을 보존합니다.
- HF causal-mask 인자 이름 변경을 흡수합니다. 이미 EAGLE이 만든 4D additive mask는
  HF의 기존 계약대로 그대로 통과시킵니다. EAGLE의 KVCache를 HF Cache로 변환하지 않습니다.
- RoPE 초기화는 설치된 HF의 실제 구현을 호출합니다. 최신 HF의 meta-device 로딩에서
  비영속 RoPE buffer가 초기화되지 않던 문제도 처리합니다. buffer는 checkpoint에 추가되지 않습니다.
- 최신 HF가 요구하는 weight-tying 메타데이터 형식을 사용합니다. 실제 tying 여부는
  기존 `tie_word_embeddings` 설정을 따릅니다.

추가로 `configs.py`는 HF가 정규화하기 전 drafter의 기존 RoPE 설정을 보존합니다.
`cnets.py`는 checkpoint의 명시적 `head_dim`을 따릅니다. 지정된 AngelSlim 가중치는
head dimension 128, Q projection 폭 4096을 사용하지만, 기존 코드는 2560/32=80을
계산해 loading shape가 맞지 않았습니다. checkpoint의 키·가중치나 target Qwen3
아키텍처를 변환하지 않고 해당 설정을 반영했습니다. `head_dim`이 없는 기존 설정은
원래 계산식을 사용합니다.

확인한 공식 API 소스:
[4.53.3 LossKwargs](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/utils/generic.py),
[최신 Qwen3](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3/modeling_qwen3.py),
[최신 mask API](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/masking_utils.py).

## 7. 강사·개발자 검증

테스트 runner는 Python 표준 `unittest`입니다. pytest나 테스트 전용 패키지를
추가로 설치할 필요가 없습니다. 아래 명령은 repository root에서 실행합니다.

```bash
python -m unittest discover -s tests -v
python scripts/colab_smoke_test.py
```

첫 번째 명령은 모델 다운로드 없이 작은 실제 Qwen3/EAGLE 클래스를 사용합니다.
로컬 임시 checkpoint를 만들어 `EaModel.from_pretrained()`부터 실행합니다.
무작위 작은 가중치는 테스트에만 사용하며, 추론 코드의 drafter를 대체하지 않습니다.
테스트는 다음을 확인합니다.

- 예제·chain·wide·무작위 tree의 ancestor visibility, sibling leakage, dtype/device.
- 학생 함수 없이 loading 가능, 실제 generation에서는 TODO 발생.
- 로딩 후 monkey-patch와 재대입, 매 round 호출, target의 mask와 absolute RoPE position.
- 실제 후보 tree의 retrieve 경로, path별 순차 forward와 tree verification logits 일치.
- 공식 posterior·KV 갱신과 다음 round, EAGLE/AR greedy token 일치.
- 설치된 HF Qwen3와 target logits 일치, decorator/typing 호환성.
- UI·평가 패키지 없이 새 Python process에서 core와 Qwen3 import.

실제 4B 가중치로 강사용 reference를 명시적으로 연결하는 smoke test:

```bash
python scripts/colab_smoke_test.py --full --reference-tree
```

기본 `python scripts/colab_smoke_test.py`는 import만 확인하며 다운로드하지 않습니다.
Full test는 CUDA를 사용해 모델을 로딩하고 8 토큰 budget으로 생성한 뒤 기존
`naivegenerate()`와 공통 구간의 greedy token을 비교합니다. 생성 중 reference의
실제 호출 횟수도 확인합니다. 실행 후 원래 TODO 함수를 복원합니다.
별도 Python 모듈에 학생 구현이 있다면 `--student-module 모듈이름`을 사용할 수 있습니다.
어느 테스트에도 패키지 설치 명령은 없습니다.

### 실행 기록 (2026-09-05)

| Transformers | PyTorch | offline unit/integration tests | 4B full smoke |
| --- | --- | --- | --- |
| 4.53.3 | 기존 2.6.0 환경 | 16개 통과 | 성공, 11 tokens, hook 6회, greedy AR 일치 |
| 4.56.1 | 기존 2.6.0 환경 | 16개 통과 | 미실행 |
| 5.16.1 | 기존 2.5.1+cu121 환경 | 16개 통과 | 성공, 11 tokens, hook 6회, greedy AR 일치 |

두 full test는 RTX 3090 한 장, BF16으로 실행했습니다. Peak CUDA allocation은
8.72 GiB였습니다. 사용된 모델 snapshot은 다음과 같습니다.

- Qwen3-4B: `1cfa9a7208912126459214e8b04321603b3df60c`
- AngelSlim drafter: `fd331e59626c8e95c392381a16ee59d518727fbb`

5.16.1에서 BF16 지원 판정을 테스트용으로 False로 설정해 float16 경로도 RTX 3090에서
실행했습니다. 동일하게 11 tokens, hook 6회, greedy AR 일치를 확인했습니다.
이 결과는 T4 하드웨어에서의 실행을 검증한 것은 아닙니다.

4.53.3 검증에는 임시 디렉터리에 Transformers 4.53.3과 호환 tokenizers 0.21.4의
wheel을 풀어 `PYTHONPATH`로 선택했습니다. 기존 Python 환경이나 torch 설치를
변경하지 않았습니다. 이것은 개발용 버전 간 검증이며 수업의 설치 절차가 아닙니다.

### 남은 검증 범위와 문제 발생 시

실제 stock Colab 세션, Colab의 torch 2.11.0+cu128 및 T4 하드웨어 실행은 아직
직접 검증하지 않았습니다. 위 테스트는 로컬 CUDA 환경에서 이루어졌습니다.
성공한 짧은 smoke test는 장문·모든 sampling 설정의 품질이나 속도 향상을 보장하지 않습니다.

패키지가 없다는 오류가 실제 발생하면 위 환경 확인 셀에서 빠진 필수 패키지만
확인하세요. 예를 들어 `device_map`에서 accelerate 부족이 보고될 때만 accelerate를
추가하면 됩니다. UI/평가 의존성을 함께 설치하지 마세요. torch는 Colab 기본
CUDA 빌드를 유지하세요. 예기치 않은 새 Transformers 변경이 확인되고 이 호환성
코드로 해결되지 않을 때에만 검증된 4.53.3 환경으로의 전환을 마지막 fallback으로
고려하세요. Transformers pin이나 torch 재설치는 기본 실습 경로가 아닙니다.
