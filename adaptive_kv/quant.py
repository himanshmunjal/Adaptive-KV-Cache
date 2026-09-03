"""Group-wise asymmetric INT8 / INT4 fake-quantization for KV cache tensors.

Two quantization granularities are implemented:

* **Per-token** (`quantize_int8` / `quantize_int4`): the reduction/group axis
  is `head_dim`, i.e. one scale/zero per token per head. This mirrors the
  scheme KIVI/AQUA-KV use for *values*.

* **Per-channel, chunked** (`quantize_int8_channel` / `quantize_int4_channel`):
  the reduction axis is the *token* axis, computed within fixed-size chunks
  of `group` consecutive tokens (so scale/zero has one value per channel per
  chunk, shape `[H, n_chunks, 1, D]`). This mirrors KIVI's and SubKV's
  finding (see `Papers/mpoq.docx`, `Papers/ABSTRACT.docx`) that the **key**
  cache has structured, per-channel outliers (a few fixed channels are
  consistently large across many tokens), so grouping by channel rather than
  by token preserves far more fidelity for K at the same bit-width; the
  **value** cache, in contrast, is closer to per-token uniform, which is why
  it stays on the per-token scheme. This project applies per-channel
  quantization to K and per-token quantization to V within the same tier —
  see `cache.py`.

All per-token quantizers operate on tensors shaped [..., group] where `group`
is the last axis (typically `head_dim`) and return:
    q      -- integer codes (int8 dtype; INT4 codes are packed 2-per-byte)
    scale  -- per-group fp16/fp32 scale, shape [..., 1]
    zero   -- per-group zero point (in original units), shape [..., 1]
"""
from __future__ import annotations

import torch

INT8_QMIN, INT8_QMAX = -128, 127
INT4_QMIN, INT4_QMAX = -8, 7


def _minmax_scale_zero(x: torch.Tensor, qmin: int, qmax: int, eps: float = 1e-8):
    xmax = x.amax(dim=-1, keepdim=True)
    xmin = x.amin(dim=-1, keepdim=True)
    scale = (xmax - xmin).clamp_min(eps) / (qmax - qmin)
    zero = xmin - qmin * scale
    return scale, zero


def quantize_int8(x: torch.Tensor):
    """Per-group asymmetric INT8 quantization. x: [..., group] (fp16/fp32).

    With `zero = xmin - qmin * scale`, the code `q = round((x - zero) / scale)`
    already lands in `[qmin, qmax]` and dequantizes exactly as `q * scale + zero`
    (no extra `qmin` offset is needed on either side).
    """
    scale, zero = _minmax_scale_zero(x, INT8_QMIN, INT8_QMAX)
    q = torch.clamp(torch.round((x - zero) / scale), INT8_QMIN, INT8_QMAX)
    return q.to(torch.int8), scale, zero


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, dtype=torch.float16):
    return (q.to(torch.float32) * scale + zero).to(dtype)


def quantize_int4(x: torch.Tensor):
    """Per-group asymmetric INT4 quantization, packed two codes per uint8 byte
    along the last axis. x: [..., group] with `group` even.
    """
    scale, zero = _minmax_scale_zero(x, INT4_QMIN, INT4_QMAX)
    q = torch.clamp(torch.round((x - zero) / scale), INT4_QMIN, INT4_QMAX)
    q = (q - INT4_QMIN).to(torch.uint8)  # shift to unsigned nibble range [0, 15]
    packed = _pack_nibbles(q)
    return packed, scale, zero


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                     group: int, dtype=torch.float16):
    q = _unpack_nibbles(packed, group).to(torch.float32)  # unsigned nibble in [0, 15]
    return ((q + INT4_QMIN) * scale + zero).to(dtype)


def _pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """q: uint8 tensor [..., group] with values in [0, 15] and even `group`.
    Returns uint8 tensor [..., group // 2].
    """
    assert q.shape[-1] % 2 == 0, "INT4 packing requires an even group size"
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, group: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty(*packed.shape[:-1], group, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _channel_scale_zero(x_chunks: torch.Tensor, qmin: int, qmax: int, eps: float = 1e-8):
    """x_chunks: [H, n_chunks, group, D]. Reduces over the `group` axis (dim=2)
    so scale/zero are shared by every token in the chunk but vary per channel.
    """
    xmax = x_chunks.amax(dim=2, keepdim=True)
    xmin = x_chunks.amin(dim=2, keepdim=True)
    scale = (xmax - xmin).clamp_min(eps) / (qmax - qmin)
    zero = xmin - qmin * scale
    return scale, zero


def _chunk_pad(x: torch.Tensor, group: int):
    """x: [H, N, D] -> ([H, n_chunks, group, D], N). Pads the last chunk (if
    N isn't a multiple of `group`) by repeating the last real token, so the
    padding never introduces a new min/max outlier into that chunk's stats.
    """
    h, n, d = x.shape
    if n == 0:
        return x.reshape(h, 0, group, d), 0
    n_chunks = (n + group - 1) // group
    pad = n_chunks * group - n
    if pad:
        x = torch.cat([x, x[:, -1:, :].expand(h, pad, d)], dim=1)
    return x.reshape(h, n_chunks, group, d), n


def quantize_int8_channel(x: torch.Tensor, group: int):
    """Per-channel asymmetric INT8 quantization, chunked every `group` tokens
    along the token axis. x: [H, N, D]. Returns q [H, N, D] int8, scale/zero
    [H, n_chunks, 1, D].
    """
    h, _, d = x.shape
    xc, n_real = _chunk_pad(x, group)
    scale, zero = _channel_scale_zero(xc, INT8_QMIN, INT8_QMAX)
    q = torch.clamp(torch.round((xc - zero) / scale), INT8_QMIN, INT8_QMAX)
    q = q.reshape(h, -1, d)[:, :n_real].to(torch.int8)
    return q, scale, zero


def dequantize_int8_channel(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                             group: int, dtype=torch.float16):
    h, n, d = q.shape
    n_chunks = scale.shape[1]
    pad = n_chunks * group - n
    qf = q.to(torch.float32)
    if pad:
        qf = torch.cat([qf, qf[:, -1:, :].expand(h, pad, d)], dim=1)
    qf = qf.reshape(h, n_chunks, group, d)
    x = (qf * scale + zero).reshape(h, -1, d)[:, :n]
    return x.to(dtype)


def quantize_int4_channel(x: torch.Tensor, group: int):
    """Per-channel asymmetric INT4 quantization, chunked every `group` tokens,
    packed two codes per byte along the channel axis. x: [H, N, D], `D` even.
    """
    h, _, d = x.shape
    xc, n_real = _chunk_pad(x, group)
    scale, zero = _channel_scale_zero(xc, INT4_QMIN, INT4_QMAX)
    q = torch.clamp(torch.round((xc - zero) / scale), INT4_QMIN, INT4_QMAX)
    q = (q - INT4_QMIN).to(torch.uint8)  # [H, n_chunks, group, D] in [0, 15]
    q = q.reshape(h, -1, d)[:, :n_real]  # [H, N, D]
    packed = _pack_nibbles(q)  # [H, N, D // 2]
    return packed, scale, zero


def dequantize_int4_channel(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                             group: int, head_dim: int, dtype=torch.float16):
    h, n, _ = packed.shape
    q = _unpack_nibbles(packed, head_dim).to(torch.float32)  # [H, N, D] in [0, 15]
    n_chunks = scale.shape[1]
    pad = n_chunks * group - n
    if pad:
        q = torch.cat([q, q[:, -1:, :].expand(h, pad, head_dim)], dim=1)
    q = q.reshape(h, n_chunks, group, head_dim)
    x = ((q + INT4_QMIN) * scale + zero).reshape(h, -1, head_dim)[:, :n]
    return x.to(dtype)


def tier_nbytes_channel(n_tokens: int, num_heads: int, head_dim: int, bits: int, group: int) -> int:
    """Bytes for a channel-quantized (K) tier: data bytes are the same as the
    per-token scheme, but scale+zero overhead is paid once per chunk of
    `group` tokens per channel (shape `[H, n_chunks, 1, D]`) instead of once
    per token, so it scales with `head_dim / group` rather than being
    independent of `head_dim`.
    """
    if n_tokens <= 0:
        return 0
    n_chunks = (n_tokens + group - 1) // group
    overhead = n_chunks * num_heads * head_dim * 4  # 2 fp16 scalars (scale, zero) = 4 bytes
    if bits == 8:
        data = n_tokens * num_heads * head_dim * 1
    elif bits == 4:
        data = n_tokens * num_heads * (head_dim // 2)
    else:
        raise ValueError(f"Unsupported bit-width {bits}")
    return data + overhead


def tier_nbytes(n_tokens: int, num_heads: int, head_dim: int, bits: int) -> int:
    """Bytes needed to store `n_tokens` cached vectors (one of K or V) at a
    given effective bit-width, including per-token/per-head scale+zero
    overhead (2 fp16 scalars = 4 bytes) for quantized tiers.
    """
    if n_tokens <= 0:
        return 0
    if bits == 16:
        return n_tokens * num_heads * head_dim * 2
    if bits == 8:
        return n_tokens * num_heads * (head_dim * 1 + 4)
    if bits == 4:
        return n_tokens * num_heads * (head_dim // 2 + 4)
    raise ValueError(f"Unsupported bit-width {bits}")
