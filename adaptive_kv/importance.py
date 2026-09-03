"""Token importance scoring for adaptive KV precision allocation.

Combines ideas from three of the surveyed papers:
  * H2O (Zhang et al., 2023)      -- accumulated attention mass identifies
                                      "heavy hitter" tokens.
  * VATP (Guo et al., 2024)       -- attention score alone is not enough:
                                      the L1 norm of a token's value vector
                                      also predicts its contribution to the
                                      attention output, so importance should
                                      be attn_score * ||v||_1.
  * KeyDiff (Park et al., 2025)   -- when attention weights are unavailable
                                      (e.g. under FlashAttention/SDPA kernels
                                      that never materialize the attention
                                      matrix), geometric key dissimilarity is
                                      a cheap, attention-free proxy: tokens
                                      whose key vectors are far (in cosine
                                      distance) from the "average" cached key
                                      tend to receive higher attention.

`ImportanceTracker` maintains one running score per cached token (aggregated
across heads by mean, see README for the "global vs. per-head" tiering
trade-off) with exponential decay, so that a token's importance can both grow
and fade over the course of generation -- unlike H2O's monotonic accumulation,
this lets a token be *promoted back* to a higher precision tier if it becomes
relevant again later (the failure mode of hard-eviction methods that Spotlight
Attention points out).
"""
from __future__ import annotations

import torch


class ImportanceTracker:
    def __init__(self, decay: float = 0.98, mode: str = "attn_value", eps: float = 1e-6):
        """
        Args:
            decay: EMA decay factor applied to existing scores before adding
                the new step's contribution (0 < decay <= 1). decay=1 recovers
                H2O's plain cumulative-sum behaviour.
            mode: one of {"attn", "attn_value", "key_diversity"}.
        """
        assert mode in ("attn", "attn_value", "key_diversity")
        self.decay = decay
        self.mode = mode
        self.eps = eps
        self.scores: torch.Tensor | None = None  # [N]

    def _grow(self, n_new: int, device, dtype):
        add = torch.zeros(n_new, device=device, dtype=dtype)
        self.scores = add if self.scores is None else torch.cat([self.scores, add], dim=0)

    def register_new_tokens(self, n_new: int, device, dtype):
        self._grow(n_new, device, dtype)

    @torch.no_grad()
    def update_from_attention(self, attn_weights: torch.Tensor, value_states: torch.Tensor | None = None):
        """attn_weights: [H, Tq, N] softmax probabilities over the full (already
        updated) key cache for this step. value_states: [H, N, D] full,
        already-dequantized value cache aligned with the last axis of
        attn_weights (only needed for mode == "attn_value").
        """
        # mean over heads and query positions -> [N] mass received by each cached token this step
        mass = attn_weights.mean(dim=(0, 1))
        if self.mode == "attn_value" and value_states is not None:
            v_norm = value_states.norm(p=1, dim=-1).mean(dim=0)  # [N], mean over heads
            v_norm = v_norm / (v_norm.mean() + self.eps)  # normalize so scale is comparable across steps
            mass = mass * v_norm
        self._ema_update(mass)

    @torch.no_grad()
    def update_from_key_diversity(self, key_states: torch.Tensor):
        """Attention-free fallback (KeyDiff-style). key_states: [H, N, D].
        Score = 1 - cosine_similarity(key, mean_key), higher => more distinct
        => empirically more likely to be attended to.
        """
        k = key_states.mean(dim=0)  # [N, D], average over heads
        k_norm = torch.nn.functional.normalize(k, dim=-1)
        anchor = torch.nn.functional.normalize(k.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
        cos_sim = (k_norm * anchor).sum(dim=-1)  # [N]
        score = (1.0 - cos_sim).clamp_min(0.0)
        self._ema_update(score)

    def _ema_update(self, new_mass: torch.Tensor):
        new_mass = new_mass.to(self.scores.dtype)
        self.scores = self.decay * self.scores + (1 - self.decay) * new_mass * new_mass.numel()

    def gather(self, index: torch.Tensor) -> torch.Tensor:
        return self.scores[index]

    def reorder(self, index: torch.Tensor):
        self.scores = self.scores[index]

    def __len__(self):
        return 0 if self.scores is None else self.scores.numel()
