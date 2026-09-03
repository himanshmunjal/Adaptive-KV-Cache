import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

from adaptive_kv import AdaptiveKVCache, AdaptiveKVConfig, AdaptiveKVLayer
from adaptive_kv.generate import generate_with_adaptive_cache


def _tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # exercise GQA path too
        max_position_embeddings=512,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class _DummyTokenizer:
    """Byte-level dummy tokenizer so the test doesn't need network access."""

    eos_token_id = 2

    class _Batch(dict):
        def to(self, device):
            return {k: v.to(device) for k, v in self.items()}

    def __call__(self, text, return_tensors="pt"):
        ids = [3 + (b % 250) for b in text.encode("utf-8")][:40]
        return self._Batch(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids.tolist())


def test_cache_grows_and_compresses_key_diversity_mode():
    model = _tiny_model()
    tok = _DummyTokenizer()
    cfg = AdaptiveKVConfig(recent_window=4, realloc_interval=4, min_tokens_to_compress=8,
                            importance_mode="key_diversity")
    text, stats = generate_with_adaptive_cache(
        model, tok, "hello world this is a reasonably long prompt for testing purposes",
        max_new_tokens=40, cache_config=cfg, return_stats=True,
    )
    total_tokens = stats["prompt_tokens"] + stats["generated_tokens"]
    tiers = stats["tier_summary"]
    tiers_total = tiers["fp16"] + tiers["int8"] + tiers["int4"]
    assert tiers_total == total_tokens * model.config.num_hidden_layers
    # with a long enough sequence, some tokens should have been compressed
    assert tiers["int8"] + tiers["int4"] > 0
    assert 0 < stats["compression_ratio"] < 1.0
    assert stats["cache_bytes"] < stats["cache_bytes_fp16_equivalent"]


def test_cache_attn_value_mode_runs():
    model = _tiny_model()
    tok = _DummyTokenizer()
    cfg = AdaptiveKVConfig(recent_window=4, realloc_interval=4, min_tokens_to_compress=8,
                            importance_mode="attn_value")
    text, stats = generate_with_adaptive_cache(
        model, tok, "another test prompt with enough tokens to trigger reallocation logic",
        max_new_tokens=20, cache_config=cfg, return_stats=True,
    )
    assert stats["generated_tokens"] > 0
    assert stats["tier_summary"]["fp16"] > 0


def test_recent_window_is_never_compressed():
    model = _tiny_model()
    tok = _DummyTokenizer()
    cfg = AdaptiveKVConfig(recent_window=50, realloc_interval=2, min_tokens_to_compress=4,
                            importance_mode="key_diversity")
    _, stats = generate_with_adaptive_cache(
        model, tok, "short prompt", max_new_tokens=10, cache_config=cfg, return_stats=True,
    )
    # recent_window (50) exceeds total sequence length, so nothing should be
    # demoted out of FP16 regardless of realloc_interval.
    tiers = stats["tier_summary"]
    assert tiers["int8"] == 0 and tiers["int4"] == 0


def test_sink_tokens_never_compressed():
    model = _tiny_model()
    tok = _DummyTokenizer()
    # tiny recent_window so recency protection can't explain the result, but a
    # sink_tokens large enough to cover the whole (short) sequence.
    cfg = AdaptiveKVConfig(recent_window=0, sink_tokens=50, realloc_interval=2,
                            min_tokens_to_compress=4, importance_mode="key_diversity")
    _, stats = generate_with_adaptive_cache(
        model, tok, "short prompt", max_new_tokens=10, cache_config=cfg, return_stats=True,
    )
    tiers = stats["tier_summary"]
    assert tiers["int8"] == 0 and tiers["int4"] == 0


def test_retier_reuses_unchanged_quantized_bytes():
    """If nothing about a token's importance changes between two retiering
    passes, its INT8/INT4 tier assignment shouldn't change either. Values use
    the byte-reuse optimization in `_rebuild_tier_v` directly. Keys are
    per-channel quantized and always recomputed from full precision in
    `_rebuild_tier_k` (chunk statistics couple neighboring tokens, so
    per-token byte-reuse doesn't apply) -- but since the input tokens and
    their order are unchanged, that recomputation is deterministic and
    reproduces the exact same bytes.
    """
    torch.manual_seed(0)
    cfg = AdaptiveKVConfig(recent_window=0, sink_tokens=0, realloc_interval=1,
                            min_tokens_to_compress=1, fp16_frac=0.0, int8_frac=0.5,
                            importance_mode="key_diversity")
    layer = AdaptiveKVLayer(cfg)
    H, D, T = 1, 8, 12
    k = torch.randn(1, H, T, D)
    v = torch.randn(1, H, T, D)
    layer.update(k, v)  # triggers a retier internally
    assert layer._n_int8() > 0 and layer._n_int4() > 0
    int8_k_before, int4_k_before = layer._int8_k.clone(), layer._int4_k.clone()
    int8_v_before, int4_v_before = layer._int8_v.clone(), layer._int4_v.clone()

    layer._retier()  # importance scores are unchanged, so the ranking is identical

    assert torch.equal(layer._int8_k, int8_k_before)
    assert torch.equal(layer._int8_v, int8_v_before)
    assert torch.equal(layer._int4_k, int4_k_before)
    assert torch.equal(layer._int4_v, int4_v_before)


def test_k_granularity_token_ablation_mode_runs():
    """k_granularity='token' reproduces the original symmetric per-token K/V
    scheme (used as the ablation baseline against the channel-K default)."""
    model = _tiny_model()
    tok = _DummyTokenizer()
    cfg = AdaptiveKVConfig(recent_window=4, realloc_interval=4, min_tokens_to_compress=8,
                            importance_mode="key_diversity", k_granularity="token")
    text, stats = generate_with_adaptive_cache(
        model, tok, "hello world this is a reasonably long prompt for testing purposes",
        max_new_tokens=40, cache_config=cfg, return_stats=True,
    )
    assert stats["tier_summary"]["int8"] + stats["tier_summary"]["int4"] > 0
    assert 0 < stats["compression_ratio"] < 1.0


def test_k_uses_channel_scale_v_uses_token_scale():
    """K's per-tier scale/zero should have shape [H, n_chunks, 1, D] (shared
    across a chunk of tokens, varies per channel); V's should have shape
    [H, n_tier, 1] (one scale/zero per token, shared across channels) --
    the asymmetric granularity this project adds on top of the tiered cache.
    """
    torch.manual_seed(0)
    cfg = AdaptiveKVConfig(recent_window=0, sink_tokens=0, realloc_interval=1,
                            min_tokens_to_compress=1, fp16_frac=0.0, int8_frac=1.0,
                            k_channel_group_int8=4, importance_mode="key_diversity")
    layer = AdaptiveKVLayer(cfg)
    H, D, T = 1, 8, 12
    k = torch.randn(1, H, T, D)
    v = torch.randn(1, H, T, D)
    layer.update(k, v)

    n_int8 = layer._n_int8()
    assert n_int8 > 0
    expected_chunks = (n_int8 + cfg.k_channel_group_int8 - 1) // cfg.k_channel_group_int8
    assert layer._int8_k_scale.shape == (H, expected_chunks, 1, D)
    assert layer._int8_v_scale.shape == (H, n_int8, 1)

    # dequantized K/V still round-trip to something finite and the same shape
    # as what the model actually attends over.
    full_k, full_v = layer._full_kv_dequant()
    assert full_k.shape == (H, T, D) and full_v.shape == (H, T, D)
    assert torch.isfinite(full_k).all() and torch.isfinite(full_v).all()
