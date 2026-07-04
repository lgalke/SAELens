import pytest
import torch

from sae_lens.saes.quantization import TERNARY_BITS, quantize


def test_quantize_none_is_full_precision_identity():
    x = torch.randn(5, 7)
    assert torch.equal(quantize(x, None, dim=-1), x)


def test_quantize_ternary_outputs_only_three_levels_per_row():
    x = torch.randn(4, 10)
    x_q = quantize(x, TERNARY_BITS, signed=True, dim=1)

    scale = x.abs().mean(dim=1, keepdim=True)
    normalized = x_q / scale
    # every entry must be (numerically) one of -1, 0, +1 once rescaled
    assert torch.all(
        (normalized.isclose(torch.tensor(-1.0)))
        | (normalized.isclose(torch.tensor(0.0), atol=1e-6))
        | (normalized.isclose(torch.tensor(1.0)))
    )


@pytest.mark.parametrize("b", [2, 3, 4])
def test_quantize_signed_b_bit_has_at_most_2b_minus_1_levels(b: int):
    x = torch.randn(1000)
    x_q = quantize(x, b, signed=True, dim=0)
    qmax = 2 ** (b - 1) - 1
    assert x_q.unique().numel() <= 2 * qmax + 1


@pytest.mark.parametrize("b", [2, 3, 4])
def test_quantize_unsigned_b_bit_stays_non_negative(b: int):
    x = torch.randn(1000).relu()
    x_q = quantize(x, b, signed=False, dim=0)
    assert torch.all(x_q >= 0)


def test_quantize_straight_through_gradient_matches_upstream_grad():
    x = torch.randn(6, 9, requires_grad=True)
    y = quantize(x, TERNARY_BITS, signed=True, dim=1)
    y.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


def test_quantize_is_homogeneous_under_positive_row_scaling():
    """Quantizing c*x for a per-row positive scalar c equals c*quantize(x), since the
    scale used internally is itself derived from x along the same reduction axis. This
    is what makes decoder-norm folding (which rescales W_dec rows by a positive scalar)
    compatible with quantization without needing to re-derive the scale post-fold.
    """
    x = torch.randn(5, 8)
    c = torch.rand(5, 1) + 0.1  # positive per-row scalars

    q_x = quantize(x, TERNARY_BITS, signed=True, dim=1)
    q_cx = quantize(c * x, TERNARY_BITS, signed=True, dim=1)

    torch.testing.assert_close(q_cx, c * q_x)


def test_quantize_dead_row_does_not_produce_nan():
    x = torch.zeros(3, 5)
    x_q = quantize(x, TERNARY_BITS, signed=True, dim=1)
    assert torch.all(x_q == 0)
    assert not torch.isnan(x_q).any()


def test_quantize_invalid_bits_raises():
    with pytest.raises(ValueError):
        quantize(torch.randn(2, 2), 1.0, dim=-1)
