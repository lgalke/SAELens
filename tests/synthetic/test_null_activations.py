import torch

from sae_lens.synthetic.null_activations import (
    GaussianNullGenerator,
    batch_iterator,
    shuffled_null_sampler,
)


def _correlated_activations(n_samples: int, d_in: int) -> torch.Tensor:
    """Activations with real cross-feature covariance, for null-fitting tests."""
    mixing = torch.randn(d_in, d_in)
    return torch.randn(n_samples, d_in) @ mixing


def test_GaussianNullGenerator_fit_matches_empirical_mean_and_covariance():
    real = _correlated_activations(n_samples=20_000, d_in=6)
    gen = GaussianNullGenerator.fit(real)

    samples = gen.sample(200_000)

    torch.testing.assert_close(samples.mean(dim=0), real.mean(dim=0), atol=0.05, rtol=0)
    torch.testing.assert_close(
        torch.cov(samples.T), torch.cov(real.T), atol=0.1, rtol=0.05
    )


def test_GaussianNullGenerator_destroys_non_gaussian_structure():
    """A null fit to non-Gaussian (e.g. sparse, heavy-tailed) real activations still
    produces exactly Gaussian samples -- it only matches the first two moments, which is
    the entire point of the confound check (E3 in PLAN-BitSAE.md): any precision
    tolerance the null shares with the real data must come from second-order statistics
    alone, not from replicating higher-order structure.
    """
    real = torch.rand(20_000, 4).pow(8)  # heavily right-skewed, non-Gaussian
    gen = GaussianNullGenerator.fit(real)
    samples = gen.sample(50_000)

    # skewness of the real (skewed) data should differ sharply from the null's (~0)
    def skewness(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=0)
        return (centered**3).mean(dim=0) / centered.std(dim=0).pow(3)

    real_skew = skewness(real)
    null_skew = skewness(samples)
    assert torch.all((real_skew - null_skew).abs() > 1.0)


def test_shuffled_null_sampler_preserves_marginals_but_breaks_covariance():
    real = _correlated_activations(n_samples=20_000, d_in=5)
    sampler = shuffled_null_sampler(real)
    shuffled = sampler(200_000)

    # marginal per-feature distribution is exactly preserved (same pool, just reshuffled)
    torch.testing.assert_close(
        shuffled.mean(dim=0), real.mean(dim=0), atol=0.05, rtol=0
    )
    torch.testing.assert_close(shuffled.std(dim=0), real.std(dim=0), atol=0.05, rtol=0)

    # but cross-feature covariance is destroyed: off-diagonal entries should collapse
    # towards zero while the real data's off-diagonal entries are substantial
    real_cov = torch.cov(real.T)
    shuffled_cov = torch.cov(shuffled.T)
    off_diag_mask = ~torch.eye(5, dtype=torch.bool)

    real_off_diag_scale = real_cov[off_diag_mask].abs().mean()
    shuffled_off_diag_scale = shuffled_cov[off_diag_mask].abs().mean()
    assert real_off_diag_scale > 10 * shuffled_off_diag_scale


def test_batch_iterator_yields_correctly_shaped_batches_indefinitely():
    real = _correlated_activations(n_samples=1000, d_in=3)
    gen = GaussianNullGenerator.fit(real)
    it = batch_iterator(gen.sample, batch_size=16)

    for _ in range(5):
        batch = next(it)
        assert batch.shape == (16, 3)
