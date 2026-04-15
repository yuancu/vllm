# Encoder-Only LoRA (BertWithRope) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LoRA adapter support to the `BertWithRope` base class so that GteNewModel, NomicBertModel, SnowflakeGteNewModel, and JinaRobertaModel can serve multi-LoRA embedding requests in vLLM.

**Architecture:** The only code change needed is adding `SupportsLoRA` to `BertWithRope` with the correct `packed_modules_mapping`. The existing LoRA infrastructure (weight mapper, model manager, worker manager, Punica kernels) already handles everything else — the model's `hf_to_vllm_mapper` is already passed through `weights_mapper` to translate HF LoRA weight names (e.g., `o_proj` → `out_proj`) to vLLM module names.

**Tech Stack:** vLLM, PyTorch, PEFT/LoRA adapters in safetensors format

---

### Task 1: Add SupportsLoRA to BertWithRope

**Files:**
- Modify: `vllm/model_executor/models/bert_with_rope.py:44` (imports)
- Modify: `vllm/model_executor/models/bert_with_rope.py:443-446` (class definition)

- [ ] **Step 1: Add SupportsLoRA import**

In `vllm/model_executor/models/bert_with_rope.py`, change line 44 from:

```python
from .interfaces import SupportsCrossEncoding, SupportsQuant
```

to:

```python
from .interfaces import SupportsCrossEncoding, SupportsLoRA, SupportsQuant
```

- [ ] **Step 2: Add SupportsLoRA mixin and attributes to BertWithRope**

Change the class definition (lines 443-446) from:

```python
@support_torch_compile
@default_pooling_type(seq_pooling_type="CLS")
class BertWithRope(nn.Module, SupportsQuant):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})
```

to:

```python
@support_torch_compile
@default_pooling_type(seq_pooling_type="CLS")
class BertWithRope(nn.Module, SupportsLoRA, SupportsQuant):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {}
    embedding_padding_modules: list[str] = []

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})
```

**Why these values:**
- `packed_modules_mapping`: `qkv_proj` is a `QKVParallelLinear` that fuses Q/K/V. `gate_up_proj` is a `MergedColumnParallelLinear` that fuses gate+up (used in GteNewModel's silu MLP). The LoRA system uses this to know how to split/pack LoRA weights for fused layers.
- `embedding_modules = {}`: No embedding LoRA support per design.
- `embedding_padding_modules = []`: No padding needed.

- [ ] **Step 3: Verify the change compiles**

Run:
```bash
.venv/bin/python -c "from vllm.model_executor.models.bert_with_rope import BertWithRope; print('OK:', hasattr(BertWithRope, 'packed_modules_mapping'))"
```
Expected: `OK: True`

- [ ] **Step 4: Commit**

```bash
git add vllm/model_executor/models/bert_with_rope.py
git commit -m "feat(lora): add SupportsLoRA to BertWithRope encoder-only models

Enable LoRA adapter support for all BertWithRope-derived models:
GteNewModel, NomicBertModel, SnowflakeGteNewModel, JinaRobertaModel.

The existing LoRA infrastructure (weights_mapper, model manager, Punica
kernels) handles name translation and weight loading without changes.

Co-authored-by: Claude"
```

---

### Task 2: Write integration test for GteNewModel + LoRA

**Files:**
- Create: `tests/lora/test_encoder_only.py`

This test verifies the full end-to-end flow: load GteNewModel, load a LoRA adapter, produce embeddings, and confirm the adapter changes the output.

- [ ] **Step 1: Create the test file**

Create `tests/lora/test_encoder_only.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for encoder-only models (BertWithRope) with LoRA adapters.

Tests verify that GteNewModel can load and use LoRA adapters for embedding
tasks, producing distinct outputs per adapter.
"""

import numpy as np
import pytest

import vllm
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test

GTE_MODEL = "Alibaba-NLP/gte-multilingual-base"

# S3 adapter paths — downloaded to local in fixture
ADAPTER_S3_PREFIX = (
    "s3://sagemaker-us-east-1-273354668433/embed-adapter/adapters"
)


@pytest.fixture(scope="session")
def gte_lora_adapter_1(tmp_path_factory):
    """Download and extract tenant-1 LoRA adapter from S3."""
    import subprocess
    import tarfile

    tmp_dir = tmp_path_factory.mktemp("gte_lora_1")
    tar_path = tmp_dir / "adapter.tar.gz"
    subprocess.run(
        [
            "aws", "s3", "cp",
            f"{ADAPTER_S3_PREFIX}/tenant-1-r32.tar.gz",
            str(tar_path),
        ],
        check=True,
    )
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)
    return str(tmp_dir)


@pytest.fixture(scope="session")
def gte_lora_adapter_2(tmp_path_factory):
    """Download and extract tenant-2 LoRA adapter from S3."""
    import subprocess
    import tarfile

    tmp_dir = tmp_path_factory.mktemp("gte_lora_2")
    tar_path = tmp_dir / "adapter.tar.gz"
    subprocess.run(
        [
            "aws", "s3", "cp",
            f"{ADAPTER_S3_PREFIX}/tenant-2-r32.tar.gz",
            str(tar_path),
        ],
        check=True,
    )
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)
    return str(tmp_dir)


def create_gte_llm(enable_lora: bool = True, max_loras: int = 4):
    """Create a GteNewModel LLM instance with optional LoRA support."""
    return vllm.LLM(
        model=GTE_MODEL,
        task="embed",
        enable_lora=enable_lora,
        max_loras=max_loras if enable_lora else 1,
        max_lora_rank=32,
        dtype="half",
        enforce_eager=True,
        trust_remote_code=True,
    )


TEST_PROMPTS = [
    "What is machine learning?",
    "The quick brown fox jumps over the lazy dog.",
]


@create_new_process_for_each_test()
def test_gte_lora_basic_inference(gte_lora_adapter_1):
    """Test that GteNewModel can load and run with a LoRA adapter."""
    llm = create_gte_llm(enable_lora=True)

    lora_request = LoRARequest(
        lora_name="tenant-1",
        lora_int_id=1,
        lora_path=gte_lora_adapter_1,
    )

    outputs = llm.embed(TEST_PROMPTS, lora_request=lora_request)

    assert len(outputs) == len(TEST_PROMPTS)
    for output in outputs:
        embedding = output.outputs.embedding
        assert len(embedding) > 0
        assert not all(v == 0.0 for v in embedding)


@create_new_process_for_each_test()
def test_gte_lora_changes_output(gte_lora_adapter_1):
    """Test that LoRA adapter produces different embeddings than base model."""
    llm = create_gte_llm(enable_lora=True)

    # Base model embeddings
    base_outputs = llm.embed(TEST_PROMPTS)

    # LoRA adapter embeddings
    lora_request = LoRARequest(
        lora_name="tenant-1",
        lora_int_id=1,
        lora_path=gte_lora_adapter_1,
    )
    lora_outputs = llm.embed(TEST_PROMPTS, lora_request=lora_request)

    # Embeddings should differ
    for base_out, lora_out in zip(base_outputs, lora_outputs):
        base_emb = np.array(base_out.outputs.embedding)
        lora_emb = np.array(lora_out.outputs.embedding)
        cosine_sim = np.dot(base_emb, lora_emb) / (
            np.linalg.norm(base_emb) * np.linalg.norm(lora_emb)
        )
        # Adapter has small perturbation, so similarity is high but not 1.0
        assert cosine_sim < 1.0, (
            "LoRA adapter should produce different embeddings than base model"
        )


@create_new_process_for_each_test()
def test_gte_multi_lora(gte_lora_adapter_1, gte_lora_adapter_2):
    """Test that different LoRA adapters produce different embeddings."""
    llm = create_gte_llm(enable_lora=True, max_loras=4)

    lora_request_1 = LoRARequest(
        lora_name="tenant-1",
        lora_int_id=1,
        lora_path=gte_lora_adapter_1,
    )
    lora_request_2 = LoRARequest(
        lora_name="tenant-2",
        lora_int_id=2,
        lora_path=gte_lora_adapter_2,
    )

    outputs_1 = llm.embed(TEST_PROMPTS, lora_request=lora_request_1)
    outputs_2 = llm.embed(TEST_PROMPTS, lora_request=lora_request_2)

    # Different adapters should produce different results
    for out_1, out_2 in zip(outputs_1, outputs_2):
        emb_1 = np.array(out_1.outputs.embedding)
        emb_2 = np.array(out_2.outputs.embedding)
        assert not np.allclose(emb_1, emb_2, atol=1e-5), (
            "Different LoRA adapters should produce different embeddings"
        )
```

- [ ] **Step 2: Run the test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest tests/lora/test_encoder_only.py::test_gte_lora_basic_inference -v -x 2>&1 | tail -30
```
Expected: PASS — the model loads with LoRA enabled and produces non-zero embeddings.

- [ ] **Step 3: Run the remaining tests**

Run:
```bash
.venv/bin/python -m pytest tests/lora/test_encoder_only.py -v -x 2>&1 | tail -30
```
Expected: All 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/lora/test_encoder_only.py
git commit -m "test(lora): add integration tests for encoder-only model LoRA

Tests GteNewModel (BertWithRope) with LoRA adapters:
- Basic inference with LoRA adapter
- LoRA produces different embeddings than base model
- Multiple LoRA adapters produce distinct embeddings

Uses real PEFT adapters (rank=32, target: qkv_proj + o_proj)
trained on Alibaba-NLP/gte-multilingual-base.

Co-authored-by: Claude"
```

---

### Task 3: Run linting and fix any issues

**Files:**
- Modify (if needed): `vllm/model_executor/models/bert_with_rope.py`
- Modify (if needed): `tests/lora/test_encoder_only.py`

- [ ] **Step 1: Run pre-commit hooks**

Run:
```bash
pre-commit run --all-files 2>&1 | tail -30
```

- [ ] **Step 2: Fix any issues found**

If ruff or mypy flags issues, fix them.

- [ ] **Step 3: Commit fixes if any**

```bash
git add -u
git commit -m "style: fix linting issues in encoder-only LoRA

Co-authored-by: Claude"
```
