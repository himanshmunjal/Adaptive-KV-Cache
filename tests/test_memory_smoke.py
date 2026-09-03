"""Smoke test: adaptive cache uses fewer bytes than an equivalent DynamicCache
on a longer synthetic sequence, using a tiny random-init model (no download).
"""
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from adaptive_kv import AdaptiveKVConfig
from adaptive_kv.generate import generate_with_adaptive_cache


class _DummyTokenizer:
    eos_token_id = None

    class _Batch(dict):
        def to(self, device):
            return {k: v.to(device) for k, v in self.items()}

    def __call__(self, text, return_tensors="pt"):
        n = len(text.split())
        ids = [3 + (i % 250) for i in range(n)]
        return self._Batch(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids.tolist())


def test_adaptive_cache_smaller_than_dynamic_cache():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    tok = _DummyTokenizer()
    prompt = " ".join(["word"] * 300)

    with torch.no_grad():
        inputs = tok(prompt, return_tensors="pt")
        dyn_cache = DynamicCache()
        model(input_ids=inputs["input_ids"], past_key_values=dyn_cache, use_cache=True)
        dyn_bytes = sum(l.keys.numel() + l.values.numel() for l in dyn_cache.layers if l.is_initialized) * 2

    cfg = AdaptiveKVConfig(recent_window=16, realloc_interval=8, min_tokens_to_compress=32,
                            importance_mode="key_diversity", fp16_frac=0.2, int8_frac=0.3)
    _, stats = generate_with_adaptive_cache(model, tok, prompt, max_new_tokens=0,
                                             cache_config=cfg, return_stats=True)

    assert stats["cache_bytes"] < dyn_bytes
    reduction = 1 - stats["cache_bytes"] / dyn_bytes
    print(f"\nDynamicCache bytes: {dyn_bytes}  AdaptiveKVCache bytes: {stats['cache_bytes']}"
          f"  reduction: {reduction * 100:.1f}%")
    assert reduction > 0.1
