"""AdaptiveKVCache: a HuggingFace `transformers`-compatible KV cache that
keeps every token in the sequence (no eviction) but stores each token's
Key/Value pair at one of three precisions -- FP16, INT8 or INT4 -- chosen
dynamically from a running importance score, and re-evaluated periodically
during generation.

This directly implements the "Adaptive Token-Aware KV Cache Compression"
idea: recent + heavy-attended tokens stay FP16, medium-importance tokens are
compressed to INT8, and low-importance tokens are compressed to INT4, with
tokens free to move back up in precision if they become important again
later (unlike hard-eviction methods such as H2O, whose failure mode when a
token is needed again after being dropped is documented in the Spotlight
Attention paper).

Within the INT8/INT4 tiers, Keys and Values are quantized with *different*
granularities: Keys use per-channel, chunked quantization (`quant.py`'s
`*_channel` functions) and Values use per-token quantization, matching the
KIVI/SubKV finding that key caches carry structured per-channel outliers
while value caches are closer to per-token uniform (see `Papers/mpoq.docx`,
`Papers/ABSTRACT.docx`). No prior work in this project's survey combines
that asymmetric K/V granularity with continuous, promotable multi-tier
precision allocation.

Batch size is assumed to be 1 (the common setting for single-sequence
perplexity / LongBench / Needle-in-a-Haystack evaluation); extending the
tiering logic to a batched, per-example policy is a natural but unimplemented
extension (see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .importance import ImportanceTracker
from .quant import (
    dequantize_int4,
    dequantize_int4_channel,
    dequantize_int8,
    dequantize_int8_channel,
    quantize_int4,
    quantize_int4_channel,
    quantize_int8,
    quantize_int8_channel,
    tier_nbytes,
    tier_nbytes_channel,
)

FP16, INT8, INT4 = 16, 8, 4


@dataclass
class AdaptiveKVConfig:
    recent_window: int = 32          # last N tokens are always kept at FP16
    sink_tokens: int = 4             # first N tokens are always kept at FP16 (attention sinks:
                                      # StreamingLLM/AQUA-KV/KIVI all find the first few tokens draw
                                      # disproportionate attention despite often having tiny value
                                      # norm, so `attn_value` scoring alone can undervalue them)
    fp16_frac: float = 0.30          # fraction of *non-recent, non-sink* tokens kept FP16
    int8_frac: float = 0.30          # fraction of *non-recent, non-sink* tokens compressed to INT8
    # remaining (1 - fp16_frac - int8_frac) go to INT4
    realloc_interval: int = 16       # re-run the tiering policy every N appended tokens
    importance_mode: str = "key_diversity"  # {"key_diversity", "attn", "attn_value"}
    decay: float = 0.98
    min_tokens_to_compress: int = 64  # do nothing until the cache is at least this long
    # Chunk size (number of consecutive tokens sharing one per-channel scale/
    # zero) for the K-only per-channel quantization in each tier. Finer
    # (smaller) chunks control error better but cost more scale/zero
    # overhead; INT4 uses a finer chunk than INT8 since it's more error-prone.
    k_channel_group_int8: int = 32
    k_channel_group_int4: int = 16
    # "channel": per-channel, chunked K quantization (this project's addition).
    # "token": fall back to the original symmetric per-token K/V quantization
    # -- kept only so eval scripts can run a controlled ablation against it.
    k_granularity: str = "channel"


class AdaptiveKVLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, config: AdaptiveKVConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = config or AdaptiveKVConfig()
        self.importance = ImportanceTracker(decay=self.cfg.decay, mode=self.cfg.importance_mode)
        self.num_heads = None
        self.head_dim = None
        self.dtype = None
        self.device = None

        # per-tier storage; keys/values stored as [H, n_tier, D] (fp16) or the
        # quantized equivalents below. `pos` tracks the global sequence
        # position of each slot for recency protection.
        self._fp16_k = self._fp16_v = None
        self._int8_k = self._int8_v = None
        self._int8_k_scale = self._int8_k_zero = self._int8_v_scale = self._int8_v_zero = None
        self._int4_k = self._int4_v = None
        self._int4_k_scale = self._int4_k_zero = self._int4_v_scale = self._int4_v_zero = None
        self.pos_fp16 = self.pos_int8 = self.pos_int4 = None

        self._global_len = 0
        self._tokens_since_realloc = 0

    # ------------------------------------------------------------------ #
    # CacheLayerMixin required interface
    # ------------------------------------------------------------------ #
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        assert key_states.shape[0] == 1, "AdaptiveKVCache currently supports batch_size == 1"
        self.dtype, self.device = key_states.dtype, key_states.device
        self.num_heads, self.head_dim = key_states.shape[1], key_states.shape[3]
        empty_kv = lambda: torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device)
        self._fp16_k, self._fp16_v = empty_kv(), empty_kv()
        self.pos_fp16 = torch.empty(0, dtype=torch.long, device=self.device)
        self.pos_int8 = torch.empty(0, dtype=torch.long, device=self.device)
        self.pos_int4 = torch.empty(0, dtype=torch.long, device=self.device)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        k_new = key_states[0]      # [H, Tnew, D]
        v_new = value_states[0]
        t_new = k_new.shape[1]

        new_positions = torch.arange(self._global_len, self._global_len + t_new,
                                      device=self.device, dtype=torch.long)
        self._global_len += t_new
        self._tokens_since_realloc += t_new

        # new tokens always enter at FP16
        self._fp16_k = torch.cat([self._fp16_k, k_new], dim=1)
        self._fp16_v = torch.cat([self._fp16_v, v_new], dim=1)
        self.pos_fp16 = torch.cat([self.pos_fp16, new_positions], dim=0)
        self.importance.register_new_tokens(t_new, self.device, torch.float32)

        if self.cfg.importance_mode == "key_diversity":
            self.importance.update_from_key_diversity(self._full_keys_dequant())

        if (self._tokens_since_realloc >= self.cfg.realloc_interval and
                self.get_seq_length() >= self.cfg.min_tokens_to_compress):
            self._retier()
            self._tokens_since_realloc = 0

        full_k, full_v = self._full_kv_dequant()
        self._last_order_positions = self._concat_positions()
        return full_k.unsqueeze(0), full_v.unsqueeze(0)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return self._fp16_k.shape[1] + self._n_int8() + self._n_int4()

    def get_max_length(self) -> int:
        return -1

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _n_int8(self):
        return 0 if self._int8_k is None else self._int8_k.shape[1]

    def _n_int4(self):
        return 0 if self._int4_k is None else self._int4_k.shape[1]

    def _int8_kv_dequant(self):
        if self._n_int8() == 0:
            empty = torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device)
            return empty, empty
        if self.cfg.k_granularity == "channel":
            k = dequantize_int8_channel(self._int8_k, self._int8_k_scale, self._int8_k_zero,
                                         self.cfg.k_channel_group_int8, self.dtype)
        else:
            k = dequantize_int8(self._int8_k, self._int8_k_scale, self._int8_k_zero, self.dtype)
        v = dequantize_int8(self._int8_v, self._int8_v_scale, self._int8_v_zero, self.dtype)
        return k, v

    def _int4_kv_dequant(self):
        if self._int4_k is None or self._int4_k.shape[1] == 0:
            empty = torch.empty(self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device)
            return empty, empty
        if self.cfg.k_granularity == "channel":
            k = dequantize_int4_channel(self._int4_k, self._int4_k_scale, self._int4_k_zero,
                                         self.cfg.k_channel_group_int4, self.head_dim, self.dtype)
        else:
            k = dequantize_int4(self._int4_k, self._int4_k_scale, self._int4_k_zero, self.head_dim, self.dtype)
        v = dequantize_int4(self._int4_v, self._int4_v_scale, self._int4_v_zero, self.head_dim, self.dtype)
        return k, v

    def _full_keys_dequant(self):
        k8, _ = self._int8_kv_dequant()
        k4, _ = self._int4_kv_dequant()
        return torch.cat([self._fp16_k, k8, k4], dim=1)

    def _full_kv_dequant(self):
        k8, v8 = self._int8_kv_dequant()
        k4, v4 = self._int4_kv_dequant()
        full_k = torch.cat([self._fp16_k, k8, k4], dim=1)
        full_v = torch.cat([self._fp16_v, v8, v4], dim=1)
        return full_k, full_v

    def _concat_positions(self):
        n8 = self._n_int8()
        return torch.cat([self.pos_fp16, self.pos_int8[:n8] if n8 else self.pos_int8, self.pos_int4], dim=0)

    def record_attention(self, attn_weights: torch.Tensor):
        """Optional: feed real softmax attention weights ([H, Tq, N], aligned
        with the concatenation order returned by the last `update()` call)
        for the "attn" / "attn_value" importance modes. Only needed when
        `importance_mode != "key_diversity"`.
        """
        if self.cfg.importance_mode == "key_diversity":
            return
        full_k, full_v = self._full_kv_dequant()
        if self.cfg.importance_mode == "attn_value":
            self.importance.update_from_attention(attn_weights, full_v)
        else:
            self.importance.update_from_attention(attn_weights, None)

    def _retier(self):
        full_k, full_v = self._full_kv_dequant()
        positions = self._concat_positions()
        n = full_k.shape[1]
        if n == 0:
            return
        scores = self.importance.scores.clone()

        # --- decide the new tier for every token -------------------------- #
        protected = (positions >= (self._global_len - self.cfg.recent_window)) | \
                    (positions < self.cfg.sink_tokens)
        order = torch.argsort(scores, descending=True)
        # stable ranking restricted to the *unprotected* subset
        unprotected_order = order[~protected[order]]
        n_rest = unprotected_order.numel()
        n_fp16_extra = int(round(self.cfg.fp16_frac * n_rest))
        n_int8 = int(round(self.cfg.int8_frac * n_rest))

        tier = torch.full((n,), INT4, dtype=torch.long, device=self.device)
        tier[protected] = FP16
        tier[unprotected_order[:n_fp16_extra]] = FP16
        tier[unprotected_order[n_fp16_extra:n_fp16_extra + n_int8]] = INT8
        # rest already INT4

        # --- figure out which tokens are *already* stored at their target
        # tier, so we can reuse their existing quantized bytes instead of
        # dequantizing + requantizing them again (every extra round-trip adds
        # avoidable rounding error). This is safe because no token is ever
        # evicted, so `positions` always covers exactly `0 .. global_len - 1`
        # with no gaps -- a plain lookup table, no eviction bookkeeping needed.
        old_tier_by_pos = torch.zeros(self._global_len, dtype=torch.long, device=self.device)
        old_local_by_pos = torch.zeros(self._global_len, dtype=torch.long, device=self.device)
        old_tier_by_pos[self.pos_fp16] = FP16
        old_local_by_pos[self.pos_fp16] = torch.arange(self.pos_fp16.numel(), device=self.device)
        if self._n_int8():
            old_tier_by_pos[self.pos_int8] = INT8
            old_local_by_pos[self.pos_int8] = torch.arange(self._n_int8(), device=self.device)
        if self._n_int4():
            old_tier_by_pos[self.pos_int4] = INT4
            old_local_by_pos[self.pos_int4] = torch.arange(self._n_int4(), device=self.device)

        old_tier = old_tier_by_pos[positions]
        old_local = old_local_by_pos[positions]

        fp16_mask = tier == FP16
        self._fp16_k = full_k.index_select(1, fp16_mask.nonzero(as_tuple=True)[0])
        self._fp16_v = full_v.index_select(1, fp16_mask.nonzero(as_tuple=True)[0])
        self.pos_fp16 = positions[fp16_mask]

        # Values: always per-token quantization, byte-reuse preserved for
        # tokens whose tier assignment didn't change (see `_rebuild_tier_v`).
        self._int8_v, self._int8_v_scale, self._int8_v_zero, self.pos_int8, int8_order = \
            self._rebuild_tier_v(INT8, tier, old_tier, old_local, positions, full_v,
                                  self._int8_v, self._int8_v_scale, self._int8_v_zero, quantize_int8)
        self._int4_v, self._int4_v_scale, self._int4_v_zero, self.pos_int4, int4_order = \
            self._rebuild_tier_v(INT4, tier, old_tier, old_local, positions, full_v,
                                  self._int4_v, self._int4_v_scale, self._int4_v_zero, quantize_int4)

        if self.cfg.k_granularity == "channel":
            # Per-channel, chunked quantization. Because chunk statistics
            # couple neighboring tokens together, byte-reuse across retiering
            # passes doesn't apply cleanly here (unlike V) -- each tier is
            # fully requantized from full precision every retiering pass, in
            # the same [kept-then-fresh] order V just computed, so
            # K/V/positions/scores all stay aligned.
            self._int8_k, self._int8_k_scale, self._int8_k_zero = self._rebuild_tier_k(
                full_k, int8_order, quantize_int8_channel, self.cfg.k_channel_group_int8)
            self._int4_k, self._int4_k_scale, self._int4_k_zero = self._rebuild_tier_k(
                full_k, int4_order, quantize_int4_channel, self.cfg.k_channel_group_int4)
        else:
            # Ablation baseline: original symmetric per-token K quantization,
            # with the same byte-reuse optimization V uses (`tier`/`old_tier`
            # are shared by K and V, so this reproduces the identical
            # kept-then-fresh order `int8_order`/`int4_order` V just computed).
            self._int8_k, self._int8_k_scale, self._int8_k_zero, _, _ = \
                self._rebuild_tier_v(INT8, tier, old_tier, old_local, positions, full_k,
                                      self._int8_k, self._int8_k_scale, self._int8_k_zero, quantize_int8)
            self._int4_k, self._int4_k_scale, self._int4_k_zero, _, _ = \
                self._rebuild_tier_v(INT4, tier, old_tier, old_local, positions, full_k,
                                      self._int4_k, self._int4_k_scale, self._int4_k_zero, quantize_int4)

        # importance.scores must be reordered to match the *actual* new storage
        # order in each tier, which is [kept-as-is tokens, then freshly (re)quantized
        # tokens] -- not plain ascending index order -- see `_rebuild_tier`.
        self.importance.reorder(torch.cat([
            fp16_mask.nonzero(as_tuple=True)[0],
            int8_order,
            int4_order,
        ]))

    def _rebuild_tier_v(self, tier_id, tier, old_tier, old_local, positions, full_v,
                         old_v, old_v_scale, old_v_zero, quantize_fn):
        """Build the new Value storage for one quantized tier, reusing
        already-quantized bytes for tokens whose tier assignment didn't
        change (see `_retier`). Also returns `order`, the [kept-then-fresh]
        token order this tier now uses -- `_rebuild_tier_k` reproduces the
        same order so K/V/positions/scores stay aligned.
        """
        target_mask = tier == tier_id
        keep_mask = target_mask & (old_tier == tier_id)   # unchanged: reuse stored bytes as-is
        fresh_mask = target_mask & ~keep_mask              # changed tier: (re)quantize from full precision

        # `old_v` is only `None` before this tier has ever been populated, in
        # which case `keep_mask` is necessarily all-False (nothing could have
        # been stored at this tier yet), so there is nothing to gather.
        keep_local = old_local[keep_mask]
        has_keep = old_v is not None and keep_local.numel() > 0

        fresh_idx = fresh_mask.nonzero(as_tuple=True)[0]
        fresh_v_src = full_v.index_select(1, fresh_idx)
        fresh_v, fresh_v_scale, fresh_v_zero = quantize_fn(fresh_v_src)

        if has_keep:
            new_v = torch.cat([old_v.index_select(1, keep_local), fresh_v], dim=1)
            new_v_scale = torch.cat([old_v_scale.index_select(1, keep_local), fresh_v_scale], dim=1)
            new_v_zero = torch.cat([old_v_zero.index_select(1, keep_local), fresh_v_zero], dim=1)
        else:
            new_v, new_v_scale, new_v_zero = fresh_v, fresh_v_scale, fresh_v_zero
        new_pos = torch.cat([positions[keep_mask], positions[fresh_mask]])
        order = torch.cat([keep_mask.nonzero(as_tuple=True)[0], fresh_idx])

        return new_v, new_v_scale, new_v_zero, new_pos, order

    def _rebuild_tier_k(self, full_k, order, quantize_channel_fn, group):
        """Build the new Key storage for one tier: per-channel, chunked
        quantization over exactly the tokens (in the exact order) V's
        `_rebuild_tier_v` just assigned to this tier. Always requantizes from
        full precision -- see the note in `_retier`.
        """
        k_src = full_k.index_select(1, order)
        return quantize_channel_fn(k_src, group)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError("AdaptiveKVCache does not support beam search (batch_size == 1 only).")

    # ------------------------------------------------------------------ #
    # Memory accounting
    # ------------------------------------------------------------------ #
    def nbytes(self) -> int:
        n0, n1, n2 = self._fp16_k.shape[1], self._n_int8(), self.pos_int4.numel()
        k_bytes8 = (tier_nbytes_channel(n1, self.num_heads, self.head_dim, 8, self.cfg.k_channel_group_int8)
                    if self.cfg.k_granularity == "channel" else
                    tier_nbytes(n1, self.num_heads, self.head_dim, 8))
        k_bytes4 = (tier_nbytes_channel(n2, self.num_heads, self.head_dim, 4, self.cfg.k_channel_group_int4)
                    if self.cfg.k_granularity == "channel" else
                    tier_nbytes(n2, self.num_heads, self.head_dim, 4))
        return (
            2 * tier_nbytes(n0, self.num_heads, self.head_dim, 16)  # FP16: K + V, no asymmetry
            + k_bytes8 + tier_nbytes(n1, self.num_heads, self.head_dim, 8)  # INT8: K + V
            + k_bytes4 + tier_nbytes(n2, self.num_heads, self.head_dim, 4)  # INT4: K + V
        )

    def nbytes_fp16_equivalent(self) -> int:
        n = self.get_seq_length()
        return 2 * tier_nbytes(n, self.num_heads, self.head_dim, 16)

    def tier_counts(self) -> dict:
        return {"fp16": self._fp16_k.shape[1], "int8": self._n_int8(), "int4": self.pos_int4.numel()}


class AdaptiveKVCache(Cache):
    """Drop-in replacement for `DynamicCache`. Usage:

        cache = AdaptiveKVCache(num_hidden_layers=model.config.num_hidden_layers,
                                 config=AdaptiveKVConfig(...))
        out = model(input_ids=..., past_key_values=cache, use_cache=True)
    """

    def __init__(self, num_hidden_layers: int, config: AdaptiveKVConfig | None = None):
        cfg = config or AdaptiveKVConfig()
        layers = [AdaptiveKVLayer(cfg) for _ in range(num_hidden_layers)]
        super().__init__(layers=layers)

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers if layer.is_initialized)

    def nbytes_fp16_equivalent(self) -> int:
        return sum(layer.nbytes_fp16_equivalent() for layer in self.layers if layer.is_initialized)

    def compression_ratio(self) -> float:
        fp16 = self.nbytes_fp16_equivalent()
        return 1.0 if fp16 == 0 else self.nbytes() / fp16

    def tier_summary(self) -> dict:
        agg = {"fp16": 0, "int8": 0, "int4": 0}
        for layer in self.layers:
            if not layer.is_initialized:
                continue
            for k, v in layer.tier_counts().items():
                agg[k] += v
        return agg

    def record_attention(self, layer_idx: int, attn_weights: torch.Tensor):
        self.layers[layer_idx].record_attention(attn_weights)
