"""2-D precision sweep harness for DiscreteSAE (PLAN-BitSAE.md experiments E1-E3).

Model-agnostic: everything here operates on batches of activations from an
`activation_provider` iterator. Whether those activations come from a real LLM's
residual stream (via `ActivationsStore`) or from a covariance-matched null generator
(`sae_lens.synthetic.null_activations`) is entirely the caller's concern -- this module
never imports a model.

Causal fidelity (KL / CE recovered) *does* require a real model and is therefore left to
an optional `causal_eval_fn` callback the caller wires up using `sae_lens.evals`; nulls
have no causal fidelity by construction (see PLAN-BitSAE.md's E3 discussion), so pass
`causal_eval_fn=None` for null runs.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace

import torch

from sae_lens.config import SAETrainerConfig
from sae_lens.evals import ExplainedVarianceCalculator
from sae_lens.saes.discrete_sae import DiscreteTrainingSAE, DiscreteTrainingSAEConfig
from sae_lens.training.sae_trainer import SAETrainer


@dataclass
class SweepCell:
    """Results for one (decoder_bits, code_bits) cell of the precision sweep."""

    decoder_bits: float | None
    code_bits: float | None
    reconstruction: dict[str, float]
    causal_fidelity: dict[str, float] | None = None


@torch.no_grad()
def compute_reconstruction_metrics(
    sae: DiscreteTrainingSAE, activations: torch.Tensor
) -> dict[str, float]:
    """
    Model-agnostic reconstruction metrics for a batch of activations: `mse`, `l0`, and
    `explained_variance` (reusing `ExplainedVarianceCalculator`, the same streaming
    calculator `sae_lens.evals.get_sparsity_and_variance_metrics` uses). Needs only the
    SAE and a batch of activations, so it works identically for real and null
    activations.
    """
    activations = activations.to(sae.device)
    feature_acts = sae.encode(activations)
    sae_out = sae.decode(feature_acts)

    calc = ExplainedVarianceCalculator()
    calc.add_batch(sae_output=sae_out, hidden_acts=activations)

    l0 = (feature_acts != 0).sum(dim=-1).float().mean().item()
    mse = (activations - sae_out).pow(2).sum(dim=-1).mean().item()
    return {"mse": mse, "l0": l0, "explained_variance": calc.compute()}


def run_precision_sweep(
    activation_provider: Iterator[torch.Tensor],
    base_sae_cfg: DiscreteTrainingSAEConfig,
    decoder_bits_grid: Sequence[float | None],
    code_bits_grid: Sequence[float | None],
    trainer_cfg: SAETrainerConfig,
    *,
    n_eval_batches: int = 8,
    causal_eval_fn: Callable[[DiscreteTrainingSAE], dict[str, float]] | None = None,
) -> list[SweepCell]:
    """
    Train a fresh `DiscreteTrainingSAE` for every (decoder_bits, code_bits) cell of the
    sweep, holding `d_sae`/`k` (and every other field of `base_sae_cfg`) fixed so that
    only precision varies -- the matched-feature-count confound from PLAN-BitSAE.md is
    enforced by construction, since every cell shares the same `base_sae_cfg` besides
    the two precision fields.

    Args:
        activation_provider: Infinite iterator of activation batches, shape
            (batch_size, d_in). Consumed for training and for the eval batches at the
            end of each cell, so it must be able to yield more batches than
            `trainer_cfg.total_training_steps + n_eval_batches`.
        base_sae_cfg: Template config; `decoder_bits`/`code_bits` are overridden per
            cell via `dataclasses.replace`, every other field is held fixed.
        decoder_bits_grid: Decoder (dictionary) precisions to sweep, e.g.
            `[TERNARY_BITS, 2, 3, 4, None]`.
        code_bits_grid: Code precisions to sweep, same convention.
        trainer_cfg: Trainer config, shared across all cells.
        n_eval_batches: Number of batches drawn from `activation_provider` after
            training to compute reconstruction metrics for each cell.
        causal_eval_fn: Optional callback computing causal-fidelity metrics (e.g. via
            `sae_lens.evals.get_downstream_reconstruction_metrics`) for a trained SAE.
            Pass `None` for null runs, which have no causal fidelity to measure.

    Returns:
        One `SweepCell` per grid point, in row-major (decoder_bits, code_bits) order.
    """
    cells = []
    for decoder_bits in decoder_bits_grid:
        for code_bits in code_bits_grid:
            cfg = replace(base_sae_cfg, decoder_bits=decoder_bits, code_bits=code_bits)
            sae = DiscreteTrainingSAE(cfg)
            trainer = SAETrainer(
                cfg=trainer_cfg, sae=sae, data_provider=activation_provider
            )
            sae = trainer.fit()

            eval_acts = torch.cat(
                [next(activation_provider) for _ in range(n_eval_batches)], dim=0
            )
            reconstruction = compute_reconstruction_metrics(sae, eval_acts)
            causal_fidelity = (
                causal_eval_fn(sae) if causal_eval_fn is not None else None
            )
            cells.append(
                SweepCell(
                    decoder_bits=decoder_bits,
                    code_bits=code_bits,
                    reconstruction=reconstruction,
                    causal_fidelity=causal_fidelity,
                )
            )
    return cells
