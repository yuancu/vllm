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
ADAPTER_S3_PREFIX = "s3://sagemaker-us-east-1-273354668433/embed-adapter/adapters"


@pytest.fixture(scope="session")
def gte_lora_adapter_1(tmp_path_factory):
    """Download and extract tenant-1 LoRA adapter from S3."""
    import subprocess
    import tarfile

    tmp_dir = tmp_path_factory.mktemp("gte_lora_1")
    tar_path = tmp_dir / "adapter.tar.gz"
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
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
            "aws",
            "s3",
            "cp",
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
