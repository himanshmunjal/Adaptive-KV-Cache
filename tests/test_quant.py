import torch

from adaptive_kv.quant import (
    dequantize_int4,
    dequantize_int4_channel,
    dequantize_int8,
    dequantize_int8_channel,
    quantize_int4,
    quantize_int4_channel,
    quantize_int8,
    quantize_int8_channel,
)


def test_int8_roundtrip_error_bounded():
    torch.manual_seed(0)
    x = torch.randn(4, 20, 64, dtype=torch.float32)
    q, scale, zero = quantize_int8(x)
    x_hat = dequantize_int8(q, scale, zero, dtype=torch.float32)
    assert q.dtype == torch.int8
    max_err = (x - x_hat).abs().max().item()
    span = (x.amax(-1) - x.amin(-1)).max().item()
    assert max_err < span / 200  # well within one INT8 quantization step


def test_int4_roundtrip_error_bounded():
    torch.manual_seed(0)
    x = torch.randn(4, 20, 64, dtype=torch.float32)
    packed, scale, zero = quantize_int4(x)
    assert packed.shape[-1] == x.shape[-1] // 2
    x_hat = dequantize_int4(packed, scale, zero, group=64, dtype=torch.float32)
    max_err = (x - x_hat).abs().max().item()
    span = (x.amax(-1) - x.amin(-1)).max().item()
    assert max_err < span / 12  # coarser (4-bit = 16 levels) but still bounded


def test_int8_channel_roundtrip_error_bounded():
    torch.manual_seed(0)
    h, n, d, group = 3, 37, 16, 8  # n not a multiple of group -> exercises padding
    x = torch.randn(h, n, d, dtype=torch.float32)
    q, scale, zero = quantize_int8_channel(x, group)
    assert q.shape == (h, n, d)
    n_chunks = (n + group - 1) // group
    assert scale.shape == (h, n_chunks, 1, d)
    x_hat = dequantize_int8_channel(q, scale, zero, group, dtype=torch.float32)
    assert x_hat.shape == x.shape
    # per-channel span (max over the whole tensor for a simple, loose bound)
    span = (x.amax() - x.amin()).item()
    assert (x - x_hat).abs().max().item() < span / 100


def test_int4_channel_roundtrip_error_bounded():
    torch.manual_seed(0)
    h, n, d, group = 2, 21, 16, 5  # neither n nor d/group aligns cleanly
    x = torch.randn(h, n, d, dtype=torch.float32)
    packed, scale, zero = quantize_int4_channel(x, group)
    assert packed.shape == (h, n, d // 2)
    x_hat = dequantize_int4_channel(packed, scale, zero, group, d, dtype=torch.float32)
    assert x_hat.shape == x.shape
    span = (x.amax() - x.amin()).item()
    assert (x - x_hat).abs().max().item() < span / 6


def test_channel_quant_chunk_locality():
    """A change confined to one chunk's tokens must not perturb another
    chunk's scale/zero (that's the whole point of chunking by channel)."""
    torch.manual_seed(2)
    h, n, d, group = 1, 8, 4, 4  # exactly 2 chunks of 4 tokens each
    x = torch.randn(h, n, d, dtype=torch.float32)
    _, scale_before, zero_before = quantize_int8_channel(x, group)

    x2 = x.clone()
    x2[:, 0, :] = 1000.0  # blow up a value in chunk 0 only
    _, scale_after, zero_after = quantize_int8_channel(x2, group)

    assert torch.equal(scale_before[:, 1], scale_after[:, 1])
    assert torch.equal(zero_before[:, 1], zero_after[:, 1])
    assert not torch.equal(scale_before[:, 0], scale_after[:, 0])


def test_channel_quant_tier_smaller_than_group():
    """Tier length below the chunk size must not crash (single partial chunk)."""
    torch.manual_seed(3)
    h, n, d, group = 2, 3, 8, 16
    x = torch.randn(h, n, d, dtype=torch.float32)
    q, scale, zero = quantize_int8_channel(x, group)
    assert q.shape == (h, n, d)
    assert scale.shape == (h, 1, 1, d)
    x_hat = dequantize_int8_channel(q, scale, zero, group, dtype=torch.float32)
    assert x_hat.shape == x.shape


def test_int4_pack_unpack_exact_codes():
    torch.manual_seed(1)
    x = torch.randn(2, 5, 8, dtype=torch.float32) * 3
    packed, scale, zero = quantize_int4(x)
    x_hat1 = dequantize_int4(packed, scale, zero, group=8, dtype=torch.float32)
    # Re-quantizing the dequantized value should reproduce the same codes
    # (idempotency check -- guards against nibble packing bugs).
    packed2, scale2, zero2 = quantize_int4(x_hat1)
    x_hat2 = dequantize_int4(packed2, scale2, zero2, group=8, dtype=torch.float32)
    assert torch.allclose(x_hat1, x_hat2, atol=1e-4)
