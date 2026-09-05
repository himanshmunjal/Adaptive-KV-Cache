"""Manual generation loop driving `AdaptiveKVCache`.

We don't rely on `model.generate()` because the "attn" / "attn_value"
importance modes need the per-step softmax attention weights fed back into
the cache after every forward pass (`cache.record_attention`), which HF's
generation loop has no hook for. The "key_diversity" mode does not need this
and could run under plain `model.generate(past_key_values=cache)`, but we use
the same explicit loop for all modes for consistency and so that
memory/timing can be measured token-by-token by the eval scripts.
"""
from __future__ import annotations

import time

import torch

from .cache import AdaptiveKVCache, AdaptiveKVConfig


@torch.no_grad()
def generate_with_adaptive_cache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    cache_config: AdaptiveKVConfig | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    return_stats: bool = False,
    max_length: int | None = None,
    add_special_tokens: bool = True,
):
    """`add_special_tokens` should be set False when `prompt` is already a
    fully chat-templated string (e.g. from `tokenizer.apply_chat_template(...,
    tokenize=False)`), since that string already contains whatever special
    tokens the chat format needs -- tokenizing it again with
    `add_special_tokens=True` would insert a second, spurious BOS/marker.
    """
    cache_config = cache_config or AdaptiveKVConfig()
    needs_attn = cache_config.importance_mode != "key_diversity"
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    if max_length is not None and inputs["input_ids"].shape[1] > max_length:
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
    inputs = inputs.to(device)
    input_ids = inputs["input_ids"]

    cache = AdaptiveKVCache(model.config.num_hidden_layers, cache_config)

    t0 = time.perf_counter()
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        output_attentions=needs_attn,
        logits_to_keep=1,  # only project the last position through lm_head during
                            # prefill -- without this, HF computes logits for every
                            # prompt position at once (O(seq_len * vocab_size)),
                            # which is the dominant memory cost on long contexts and
                            # was causing CUDA OOMs on long LongBench documents.
    )
    if needs_attn:
        for i, attn in enumerate(out.attentions):
            cache.record_attention(i, attn[0])
    prefill_s = time.perf_counter() - t0

    generated = [input_ids]
    next_token = out.logits[:, -1, :]

    t1 = time.perf_counter()
    n_generated = 0
    for _ in range(max_new_tokens):
        if do_sample:
            probs = torch.softmax(next_token / temperature, dim=-1)
            tok = torch.multinomial(probs, 1)
        else:
            tok = next_token.argmax(dim=-1, keepdim=True)
        generated.append(tok)
        n_generated += 1
        if tokenizer.eos_token_id is not None and tok.item() == tokenizer.eos_token_id:
            break
        out = model(
            input_ids=tok,
            past_key_values=cache,
            use_cache=True,
            output_attentions=needs_attn,
        )
        if needs_attn:
            for i, attn in enumerate(out.attentions):
                cache.record_attention(i, attn[0])
        next_token = out.logits[:, -1, :]
    decode_s = time.perf_counter() - t1

    full_ids = torch.cat(generated, dim=1)[0]
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    generated_text = tokenizer.decode(full_ids[input_ids.shape[1]:], skip_special_tokens=True)

    if not return_stats:
        return text

    stats = {
        "prompt_tokens": input_ids.shape[1],
        "generated_tokens": n_generated,
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
        "tokens_per_second": n_generated / decode_s if decode_s > 0 else float("nan"),
        "cache_bytes": cache.nbytes(),
        "cache_bytes_fp16_equivalent": cache.nbytes_fp16_equivalent(),
        "compression_ratio": cache.compression_ratio(),
        "tier_summary": cache.tier_summary(),
        "generated_text": generated_text,
    }
    return text, stats