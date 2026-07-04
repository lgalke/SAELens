"""Covariance-matched null activations, for the E3 confound check in PLAN-BitSAE.md.

The claim that LLM feature geometry is "discrete" (tolerates low-bit SAE dictionaries)
is only interesting if it does *not* also hold for activations with the same
second-order statistics but no real feature structure. `GaussianNullGenerator` and
`shuffled_null_sampler` produce such nulls from a batch of real activations, to be run
through the identical precision sweep as the real data.
"""

from collections.abc import Callable

import torch
from torch.distributions import MultivariateNormal


class GaussianNullGenerator:
    """Samples i.i.d. Gaussian activations matched to the empirical mean/covariance of
    a batch of real activations.
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        Args:
            mean: Empirical mean, shape (d_in,).
            cov: Empirical covariance, shape (d_in, d_in).
        """
        self.mean = mean
        self.cov = cov
        self._dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

    @classmethod
    def fit(cls, activations: torch.Tensor) -> "GaussianNullGenerator":
        """Fit a `GaussianNullGenerator` to the empirical mean/covariance of `activations`.

        Args:
            activations: Real activations of shape (n_samples, d_in).
        """
        mean = activations.mean(dim=0)
        cov = torch.cov(activations.T)
        return cls(mean, cov)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw a batch of shape (batch_size, d_in)."""
        return self._dist.sample((batch_size,))


def shuffled_null_sampler(
    activations: torch.Tensor,
) -> Callable[[int], torch.Tensor]:
    """Build a sampler that returns batches of `activations` with each feature
    dimension independently permuted across the sample axis.

    This destroys cross-feature covariance while exactly preserving each feature's
    marginal distribution (unlike the Gaussian null, which only matches the first two
    moments) -- the "shuffled/phase-randomized" variant from PLAN-BitSAE.md.

    Args:
        activations: Real activations of shape (n_samples, d_in), used as the pool to
            sample from and permute.
    """
    n_samples, d_in = activations.shape

    def sample(batch_size: int) -> torch.Tensor:
        row_idx = torch.randint(n_samples, (batch_size, d_in))
        col_idx = torch.arange(d_in).expand(batch_size, d_in)
        return activations[row_idx, col_idx]

    return sample


def batch_iterator(sample_fn: Callable[[int], torch.Tensor], batch_size: int):
    """Wrap a `sample_fn(batch_size) -> Tensor` callable (e.g. `GaussianNullGenerator.sample`
    or a `shuffled_null_sampler` closure) into an infinite iterator of batches, matching the
    iterator protocol expected by `SAETrainer`'s `data_provider`.
    """
    while True:
        yield sample_fn(batch_size)
