"""E3 confound check (PLAN-BitSAE.md) on synthetic data: does a DiscreteSAE tolerate a
low-bit decoder better on activations with real sparse+discrete structure than on a
covariance-matched Gaussian null with no such structure?

"Real" activations here are synthetic-but-structured: sparse combinations of a ternary
ground-truth dictionary (standing in for the hypothesis that real LLM feature
directions are close to a low-bit lattice). Recovering that structure via sparse
dictionary learning (a TopK SAE trained from a random initialization) is a real
optimization problem that needs many steps to converge, especially once the dictionary
is overcomplete (num_ground_truth_features > d_in) -- unlike the fast unit tests in
tests/, this script is meant to be run with a real compute budget (GPU, thousands of
steps) rather than in CI.

Usage:
    python scripts/discrete_sae_synthetic_e3_experiment.py --steps 5000 --device cuda
"""

import argparse

import torch

from sae_lens.config import SAETrainerConfig
from sae_lens.discrete_sweep import SweepCell, run_precision_sweep
from sae_lens.saes.discrete_sae import DiscreteTrainingSAEConfig
from sae_lens.saes.quantization import TERNARY_BITS
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.synthetic.null_activations import GaussianNullGenerator, batch_iterator


def ternary_dictionary_activation_provider(
    num_ground_truth_features: int,
    d_in: int,
    k_active: int,
    batch_size: int,
    device: str,
):
    """Infinite iterator of activation batches, each a sparse combination of
    `k_active` (on average) ground-truth features whose directions are exactly ternary
    vectors in {-1, 0, +1}^d_in.
    """
    feature_dict = FeatureDictionary(
        num_features=num_ground_truth_features,
        hidden_dim=d_in,
        initializer=None,
        device=device,
    )
    with torch.no_grad():
        feature_dict.feature_vectors.data = torch.randint(
            -1, 2, feature_dict.feature_vectors.shape, device=device
        ).float()
    activation_generator = ActivationGenerator(
        num_features=num_ground_truth_features,
        firing_probabilities=k_active / num_ground_truth_features,
        mean_firing_magnitudes=3.0,
        std_firing_magnitudes=0.0,
        device=device,
    )

    def sample(n: int) -> torch.Tensor:
        return feature_dict(activation_generator.sample(n))

    return batch_iterator(sample, batch_size)


def reconstruction_gap(cells: list[SweepCell]) -> float:
    """Drop in explained variance between full precision and ternary decoder."""
    by_bits = {
        cell.decoder_bits: cell.reconstruction["explained_variance"] for cell in cells
    }
    return by_bits[None] - by_bits[TERNARY_BITS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d_in", type=int, default=64)
    parser.add_argument("--d_sae", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--num_ground_truth_features", type=int, default=512)
    parser.add_argument("--k_active", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    real_provider = ternary_dictionary_activation_provider(
        args.num_ground_truth_features,
        args.d_in,
        args.k_active,
        args.batch_size,
        args.device,
    )
    real_fit_batch = torch.cat([next(real_provider) for _ in range(50)], dim=0)
    null_provider = batch_iterator(
        GaussianNullGenerator.fit(real_fit_batch).sample, args.batch_size
    )

    base_cfg = DiscreteTrainingSAEConfig(
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        decoder_init_norm=0.1,
        rescale_acts_by_decoder_norm=True,
        device=args.device,
    )
    trainer_cfg = SAETrainerConfig(
        total_training_samples=args.batch_size * args.steps,
        train_batch_size_samples=args.batch_size,
        lr=args.lr,
        lr_end=args.lr / 10,
        lr_scheduler_name="cosineannealing",
        device=args.device,
    )

    decoder_bits_grid = [TERNARY_BITS, None]

    print("Training on real (ternary-dictionary) activations...")
    real_cells = run_precision_sweep(
        real_provider, base_cfg, decoder_bits_grid, [None], trainer_cfg
    )
    print("Training on covariance-matched Gaussian null...")
    null_cells = run_precision_sweep(
        null_provider, base_cfg, decoder_bits_grid, [None], trainer_cfg
    )

    print("\n=== Results ===")
    for label, cells in [("real", real_cells), ("null", null_cells)]:
        for cell in cells:
            print(f"{label} decoder_bits={cell.decoder_bits}: {cell.reconstruction}")

    real_gap = reconstruction_gap(real_cells)
    null_gap = reconstruction_gap(null_cells)
    print("\nExplained-variance gap (full precision - ternary):")
    print(f"  real: {real_gap:.4f}")
    print(f"  null: {null_gap:.4f}")
    print(
        "Claim supported (real tolerates ternary better than null)"
        if real_gap < null_gap
        else "Claim NOT supported at these settings"
    )


if __name__ == "__main__":
    main()
