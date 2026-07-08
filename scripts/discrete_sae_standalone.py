"""Standalone DiscreteSAE script (no SAELens imports).

This reproduces the core behavior of SAELens DiscreteSAE in one file:
- TopK ReLU encoder activations
- Optional code quantization (non-negative)
- Optional decoder quantization (signed)
- Straight-through estimator (STE) for quantization-aware training

Usage:
    python scripts/discrete_sae_standalone.py --steps 500 --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn

TERNARY_BITS = 1.58
_EPS = 1e-8


def quantize(
    x: torch.Tensor,
    bits: float | None,
    *,
    signed: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """Fake-quantize ``x`` with an STE backward pass."""
    if bits is None:
        return x

    if bits == TERNARY_BITS:
        x_q = _ternary_levels(x, dim=dim)
    else:
        b = int(bits)
        if b < 2:
            raise ValueError(
                f"bits must be None, {TERNARY_BITS}, or an integer >= 2, got {bits}"
            )
        x_q = _uniform_levels(x, b=b, signed=signed, dim=dim)

    # Forward uses quantized value; backward is identity.
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


def topk_relu(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep TopK entries per row, apply ReLU, zero the rest."""
    topk_values, topk_indices = torch.topk(x, k=k, dim=-1, sorted=False)
    values = topk_values.relu()
    out = torch.zeros_like(x)
    out.scatter_(-1, topk_indices, values)
    return out


@dataclass
class DiscreteSAEConfig:
    d_in: int
    d_sae: int
    k: int
    decoder_bits: float | None = None
    code_bits: float | None = None
    rescale_acts_by_decoder_norm: bool = True
    decoder_init_norm: float | None = 0.1
    device: str = "cpu"
    dtype: str = "float32"


class DiscreteSAE(nn.Module):
    """Single-file TopK + quantized decoder/code SAE."""

    def __init__(self, cfg: DiscreteSAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        torch_dtype = getattr(torch, cfg.dtype)
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae, dtype=torch_dtype))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, dtype=torch_dtype))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in, dtype=torch_dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, dtype=torch_dtype))

        self.reset_parameters()
        self.to(cfg.device)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)

        if self.cfg.decoder_init_norm is not None:
            with torch.no_grad():
                norms = self.W_dec.norm(dim=-1, keepdim=True).clamp_min(_EPS)
                self.W_dec.mul_(self.cfg.decoder_init_norm / norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden_pre = x @ self.W_enc + self.b_enc
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        feature_acts = topk_relu(hidden_pre, self.cfg.k)

        # DiscreteSAE behavior: quantize only non-negative codes after TopK+ReLU.
        return quantize(feature_acts, self.cfg.code_bits, signed=False, dim=0)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        # Quantize decoder columns/rows with STE during training.
        W_dec_q = quantize(self.W_dec, self.cfg.decoder_bits, signed=True, dim=1)

        if self.cfg.rescale_acts_by_decoder_norm:
            feature_acts = feature_acts / self.W_dec.norm(dim=-1).clamp_min(_EPS)

        return feature_acts @ W_dec_q + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def metrics(self, x: torch.Tensor) -> dict[str, float]:
        x_hat, z = self(x)
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean().item()
        explained_variance = 1.0 - (mse / max(var, _EPS))
        l0 = (z != 0).sum(dim=-1).float().mean().item()
        return {
            "mse": mse,
            "explained_variance": explained_variance,
            "l0": l0,
        }


def sample_sparse_data(
    batch_size: int,
    d_in: int,
    n_components: int,
    k_active: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Simple synthetic activations for quick standalone training."""
    basis = torch.randn(n_components, d_in, device=device, dtype=dtype)
    basis = basis / basis.norm(dim=-1, keepdim=True).clamp_min(_EPS)

    idx = torch.randint(0, n_components, (batch_size, k_active), device=device)
    coeff = torch.randn(batch_size, k_active, device=device, dtype=dtype).relu()

    x = torch.zeros(batch_size, d_in, device=device, dtype=dtype)
    for i in range(k_active):
        x = x + coeff[:, i : i + 1] * basis[idx[:, i]]

    x = x + 0.01 * torch.randn_like(x)
    return x


def train(
    model: DiscreteSAE,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    n_components: int,
    k_active: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dtype = model.W_enc.dtype
    for step in range(1, steps + 1):
        x = sample_sparse_data(
            batch_size=batch_size,
            d_in=model.cfg.d_in,
            n_components=n_components,
            k_active=k_active,
            device=model.cfg.device,
            dtype=dtype,
        )

        x_hat, _ = model(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            m = model.metrics(x)
            print(
                f"step={step:5d} loss={loss.item():.6f} "
                f"mse={m['mse']:.6f} ev={m['explained_variance']:.4f} l0={m['l0']:.2f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d_in", type=int, default=64)
    parser.add_argument("--d_sae", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--decoder_bits", type=float, default=TERNARY_BITS)
    parser.add_argument("--code_bits", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_components", type=int, default=512)
    parser.add_argument("--k_active", type=int, default=16)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DiscreteSAEConfig(
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        decoder_bits=args.decoder_bits,
        code_bits=args.code_bits,
        device=args.device,
        dtype=args.dtype,
    )

    model = DiscreteSAE(cfg)
    train(
        model,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_components=args.n_components,
        k_active=args.k_active,
    )


if __name__ == "__main__":
    main()
