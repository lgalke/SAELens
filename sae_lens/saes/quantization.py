"""Straight-through-estimator (STE) fake-quantization primitives.

Used by :class:`~sae_lens.saes.discrete_sae.DiscreteSAE` to quantize decoder columns
(feature directions) and/or codes (feature magnitudes) during training, while allowing
gradients to flow through the quantization step as if it were the identity function.

Precision is specified in bits via a single ``bits`` argument:

- ``None`` -- full precision (identity, no quantization).
- ``TERNARY_BITS`` (1.58) -- ternary quantization, i.e. BitNet b1.58: values are rounded
  to one of ``{-1, 0, +1}`` times a per-row/column scale.
- an integer ``b >= 2`` -- symmetric ``b``-bit quantization.
"""

import torch

TERNARY_BITS = 1.58
"""Sentinel value for ``bits`` denoting ternary ({-1, 0, +1}) quantization."""

_EPS = 1e-8


def quantize(
    x: torch.Tensor,
    bits: float | None,
    *,
    signed: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """
    Fake-quantize ``x`` to the given bit-width with a straight-through gradient estimator.

    The scale is derived from ``x`` per-slice along ``dim`` (absmean for ternary, absmax
    otherwise) and is treated as a constant for gradient purposes, matching standard
    quantization-aware training practice.

    Args:
        x: Tensor to quantize.
        bits: `None` for full precision, `TERNARY_BITS` for ternary, or an integer >= 2
            for symmetric b-bit quantization.
        signed: Whether the quantization levels are symmetric around zero (for weights)
            or non-negative (for codes, which are already >= 0 after ReLU/TopK).
        dim: Dimension along which the scale is computed (e.g. the dictionary dimension
            for decoder columns, or the batch dimension for codes).

    Returns:
        Tensor of the same shape and dtype as `x`, with values snapped to the
        quantization grid on the forward pass and gradient passed straight through.
    """
    if bits is None:
        return x
    if bits == TERNARY_BITS:
        x_q = _ternary_levels(x, dim)
    else:
        b = int(bits)
        if b < 2:
            raise ValueError(
                f"bits must be None, {TERNARY_BITS}, or an integer >= 2, got {bits}"
            )
        x_q = _uniform_levels(x, b, signed=signed, dim=dim)
    # Straight-through estimator: forward value is x_q, gradient flows as if identity.
    return x + (x_q - x).detach()


def _ternary_levels(x: torch.Tensor, dim: int) -> torch.Tensor:
    scale = x.detach().abs().mean(dim=dim, keepdim=True).clamp_min(_EPS)
    return torch.clamp(torch.round(x / scale), -1, 1) * scale


def _uniform_levels(x: torch.Tensor, b: int, *, signed: bool, dim: int) -> torch.Tensor:
    amax = x.detach().abs().max(dim=dim, keepdim=True).values.clamp_min(_EPS)
    if signed:
        qmax = 2 ** (b - 1) - 1
        qmin = -qmax
    else:
        qmin = 0
        qmax = 2**b - 1
    scale = amax / qmax
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale
