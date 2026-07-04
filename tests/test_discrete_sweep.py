"""Fast, deterministic tests for the precision-sweep harness (PLAN-BitSAE.md E1-E3
mechanics): matched feature count, quantization actually affecting reconstruction, and
the full real+null pipeline wiring together end to end.

The actual scientific claim -- that real activations tolerate a low-bit decoder better
than a covariance-matched null -- is a real (and open) empirical question that depends
on an SAE actually recovering a sparse dictionary's structure via training, which needs
a real compute budget (GPU, thousands of steps) to answer reliably. That experiment
lives in scripts/discrete_sae_synthetic_e3_experiment.py (synthetic ground truth) and
scripts/discrete_sae_precision_sweep.py (real LLM activations), not here: forcing a
statistically significant real-vs-null gap out of a few hundred CPU training steps
would make this test either flaky or dependent on tuned-to-pass hyperparameters, neither
of which verifies real behavior.
"""

import torch

from sae_lens.config import SAETrainerConfig
from sae_lens.discrete_sweep import compute_reconstruction_metrics, run_precision_sweep
from sae_lens.saes.discrete_sae import DiscreteTrainingSAE, DiscreteTrainingSAEConfig
from sae_lens.saes.quantization import TERNARY_BITS
from sae_lens.synthetic.null_activations import GaussianNullGenerator, batch_iterator


def _correlated_activation_provider(d_in: int, batch_size: int):
    mixing = torch.randn(d_in, d_in)

    def sample(n: int) -> torch.Tensor:
        return torch.randn(n, d_in) @ mixing

    return batch_iterator(sample, batch_size)


def _trainer_cfg(batch_size: int, steps: int) -> SAETrainerConfig:
    return SAETrainerConfig(
        total_training_samples=batch_size * steps,
        train_batch_size_samples=batch_size,
        lr=1e-3,
        lr_end=1e-3,
        device="cpu",
    )


def test_compute_reconstruction_metrics_matches_manual_calculation():
    cfg = DiscreteTrainingSAEConfig(d_in=8, d_sae=16, k=3)
    sae = DiscreteTrainingSAE(cfg)
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    activations = torch.randn(50, 8)
    metrics = compute_reconstruction_metrics(sae, activations)

    feature_acts = sae.encode(activations)
    sae_out = sae.decode(feature_acts)
    expected_mse = (activations - sae_out).pow(2).sum(dim=-1).mean().item()

    assert metrics["mse"] == expected_mse
    assert metrics["l0"] <= 3


def test_run_precision_sweep_holds_width_and_l0_fixed_across_the_grid():
    d_in, d_sae, k, batch_size = 8, 24, 4, 32
    provider = _correlated_activation_provider(d_in, batch_size)
    base_cfg = DiscreteTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k, decoder_init_norm=0.1
    )

    cells = run_precision_sweep(
        provider,
        base_cfg,
        decoder_bits_grid=[TERNARY_BITS, 2, None],
        code_bits_grid=[None, 3],
        trainer_cfg=_trainer_cfg(batch_size, steps=5),
        n_eval_batches=2,
    )

    assert len(cells) == 3 * 2
    for cell in cells:
        assert cell.reconstruction["l0"] <= k
        assert cell.causal_fidelity is None  # no causal_eval_fn was passed


def test_run_precision_sweep_decoder_bits_changes_reconstruction():
    d_in, d_sae, k, batch_size = 8, 24, 4, 32
    provider = _correlated_activation_provider(d_in, batch_size)
    base_cfg = DiscreteTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k, decoder_init_norm=0.1
    )

    cells = run_precision_sweep(
        provider,
        base_cfg,
        decoder_bits_grid=[TERNARY_BITS, None],
        code_bits_grid=[None],
        trainer_cfg=_trainer_cfg(batch_size, steps=5),
        n_eval_batches=2,
    )
    mse_by_bits = {cell.decoder_bits: cell.reconstruction["mse"] for cell in cells}
    assert mse_by_bits[TERNARY_BITS] != mse_by_bits[None]


def test_run_precision_sweep_supports_a_causal_eval_callback():
    d_in, d_sae, k, batch_size = 8, 16, 3, 32
    provider = _correlated_activation_provider(d_in, batch_size)
    base_cfg = DiscreteTrainingSAEConfig(d_in=d_in, d_sae=d_sae, k=k)

    def fake_causal_eval(sae: DiscreteTrainingSAE) -> dict[str, float]:
        return {"kl_div_score": 1.0 if sae.cfg.decoder_bits is None else 0.5}

    cells = run_precision_sweep(
        provider,
        base_cfg,
        decoder_bits_grid=[TERNARY_BITS, None],
        code_bits_grid=[None],
        trainer_cfg=_trainer_cfg(batch_size, steps=3),
        n_eval_batches=2,
        causal_eval_fn=fake_causal_eval,
    )
    causal_by_bits = {cell.decoder_bits: cell.causal_fidelity for cell in cells}
    assert causal_by_bits[None] == {"kl_div_score": 1.0}
    assert causal_by_bits[TERNARY_BITS] == {"kl_div_score": 0.5}


def test_real_and_null_activation_providers_both_run_through_the_full_pipeline():
    """End-to-end wiring smoke test: a covariance-matched null generator built from
    "real" activations can be run through the exact same sweep harness as the real
    activations, producing well-formed reconstruction metrics for both. This is the
    E3 confound-check plumbing from PLAN-BitSAE.md; whether the two conditions'
    numbers actually *differ* the way the discreteness hypothesis predicts is the
    open scientific question the scripts/ experiments are for, not this test.
    """
    d_in, d_sae, k, batch_size = 8, 16, 3, 32
    real_provider = _correlated_activation_provider(d_in, batch_size)
    real_fit_batch = torch.cat([next(real_provider) for _ in range(20)], dim=0)
    null_provider = batch_iterator(
        GaussianNullGenerator.fit(real_fit_batch).sample, batch_size
    )

    base_cfg = DiscreteTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k, decoder_init_norm=0.1
    )
    trainer_cfg = _trainer_cfg(batch_size, steps=5)

    for provider in (real_provider, null_provider):
        cells = run_precision_sweep(
            provider,
            base_cfg,
            [TERNARY_BITS, None],
            [None],
            trainer_cfg,
            n_eval_batches=2,
        )
        assert len(cells) == 2
        for cell in cells:
            assert torch.isfinite(torch.tensor(cell.reconstruction["mse"]))
            assert cell.reconstruction["l0"] <= k
