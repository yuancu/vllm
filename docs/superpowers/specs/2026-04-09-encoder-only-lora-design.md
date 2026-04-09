# LoRA Support for BertWithRope (Encoder-Only Models)

**Date:** 2026-04-09
**Scope:** Add LoRA support to `BertWithRope` base class, enabling LoRA for GteNewModel, NomicBertModel, SnowflakeGteNewModel, JinaRobertaModel, and GteNewForSequenceClassification.

## Problem

vLLM supports LoRA for decoder-only and encoder-decoder models, but not for encoder-only models. Users deploying multi-tenant embedding services with models like `Alibaba-NLP/gte-multilingual-base` cannot use vLLM's native multi-LoRA serving and must fall back to frameworks like PEFT + DJL.

## Target Adapter Format

PEFT LoRA adapters trained on GteNewModel with:
- **Target modules:** `qkv_proj`, `o_proj` (HuggingFace naming)
- **Rank:** 32 (arbitrary, any rank should work)
- **Weight names in checkpoint:** `base_model.model.encoder.layer.{i}.attention.{qkv_proj,o_proj}.lora_{A,B}.weight`

## Design

### 1. Add `SupportsLoRA` to `BertWithRope`

In `vllm/model_executor/models/bert_with_rope.py`, add `SupportsLoRA` to `BertWithRope`:

```python
class BertWithRope(nn.Module, SupportsLoRA, SupportsQuant):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {}
    embedding_padding_modules = []
```

**Supported LoRA modules** (auto-discovered from `LinearBase` instances):
- `qkv_proj` — `QKVParallelLinear` → handled by `QKVParallelLinearWithLoRA`
- `out_proj` — `RowParallelLinear` → handled by `RowParallelLinearWithLoRA`
- `gate_up_proj` — `MergedColumnParallelLinear` → handled by `MergedColumnParallelLinearWithLoRA`
- `up_proj` — `ColumnParallelLinear` → handled by `ColumnParallelLinearWithLoRA`
- `down_proj` — `RowParallelLinear` → handled by `RowParallelLinearWithLoRA`

### 2. Handle HF→vLLM Name Translation for LoRA Weights

**The problem:** GteNewModel's `hf_to_vllm_mapper` renames HF modules:
- `attention.o_proj` → `attn.out_proj`
- `attention.qkv_proj` → `attn.qkv_proj`
- `layer` → `layers`

But LoRA adapter checkpoints use HF naming (`o_proj`, not `out_proj`). The LoRA weight loader must translate adapter weight names to match vLLM's internal module names.

**Solution:** In `vllm/lora/lora_model.py`, when loading LoRA weights, apply the model's `hf_to_vllm_mapper` substring replacements to LoRA module names. The model already defines this mapping — we reuse it rather than hardcoding a second mapping.

Specifically, in `LoRAModel.from_local_checkpoint` (or the weight name resolution path), after stripping the `base_model.model.` prefix from LoRA weight names, apply the same `orig_to_new_substr` transformations that the base model uses. This ensures `encoder.layer.0.attention.o_proj` becomes `encoder.layers.0.attn.out_proj`.

### 3. No Runner-Level Changes

The v1 `GPUModelRunner` already:
- Calls `set_active_loras()` in `_prepare_inputs()` before `_pool()`
- The `LoRAModelManager` already has pooling model handling (module name prefix stripping)
- `EncoderOnlyAttention` doesn't affect LoRA — LoRA wraps the linear layers, not the attention computation

### 4. No New LoRA Layer Types

All linear layers in BertWithRope use standard vLLM parallel linear types that already have LoRA-compatible wrappers.

## Files to Change

| File | Change |
|------|--------|
| `vllm/model_executor/models/bert_with_rope.py` | Add `SupportsLoRA` mixin + class attributes |
| `vllm/lora/lora_model.py` | Apply `hf_to_vllm_mapper` name translation to LoRA weight names |

## Out of Scope

- Embedding module LoRA (`word_embeddings`)
- BertModel (bert.py) LoRA support — separate effort
- New LoRA layer types
- MoE layer LoRA for NomicBERT

## Testing

- Unit test: Load GteNewModel with a LoRA adapter, verify embeddings differ from base
- Multi-adapter test: Load 2+ adapters, verify distinct outputs
- Use existing tenant adapters from S3 (`s3://sagemaker-us-east-1-273354668433/embed-adapter/adapters/tenant-{1,2}-r32.tar.gz`)
