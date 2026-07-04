"""Real-model precision sweep (PLAN-BitSAE.md E1/E2/E3), run against an actual LLM's
residual stream.

Pipeline: cache one layer's residual-stream activations for a real LLM to disk (via the
existing `CacheActivationsRunner`) -> feed them through the DiscreteSAE precision sweep
(`sae_lens.discrete_sweep.run_precision_sweep`) -> for the real-activation run only,
also measure causal fidelity (KL / CE recovered) by wiring `sae_lens.evals` in as
`causal_eval_fn`. The null run (covariance-matched Gaussian) has no underlying model to
splice into, so it only gets reconstruction metrics, per PLAN-BitSAE.md's resolution of
the null-vs-causal-fidelity question.

This is meant to be run with a real compute budget (GPU, a real dataset, thousands of
training steps per cell), not in CI -- see tests/test_discrete_sweep.py for the fast,
deterministic unit tests of the same harness.

Usage:
    python scripts/discrete_sae_precision_sweep.py \\
        --model_name gpt2 --hook_name blocks.6.hook_resid_pre --d_in 768 \\
        --dataset_path NeelNanda/c4-tokenized-2b --device cuda
"""

import argparse

import torch

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig, SAETrainerConfig
from sae_lens.discrete_sweep import run_precision_sweep
from sae_lens.evals import get_downstream_reconstruction_metrics
from sae_lens.load_model import load_model
from sae_lens.saes.discrete_sae import DiscreteTrainingSAE, DiscreteTrainingSAEConfig
from sae_lens.saes.quantization import TERNARY_BITS
from sae_lens.saes.sae import SAEMetadata
from sae_lens.synthetic.null_activations import GaussianNullGenerator, batch_iterator
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--hook_name", type=str, default="blocks.6.hook_resid_pre")
    parser.add_argument("--d_in", type=int, default=768)
    parser.add_argument("--dataset_path", type=str, default="NeelNanda/c4-tokenized-2b")
    parser.add_argument("--context_size", type=int, default=128)
    parser.add_argument("--cache_training_tokens", type=int, default=2_000_000)
    parser.add_argument("--model_batch_size", type=int, default=16)
    parser.add_argument("--d_sae", type=int, default=768 * 8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_causal_eval_batches", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    cache_cfg = CacheActivationsRunnerConfig(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        model_batch_size=args.model_batch_size,
        hook_name=args.hook_name,
        d_in=args.d_in,
        training_tokens=args.cache_training_tokens,
        context_size=args.context_size,
        device=args.device,
    )
    print(f"Caching activations to {cache_cfg.new_cached_activations_path} ...")
    CacheActivationsRunner(cache_cfg).run()

    model = load_model(
        cache_cfg.model_class_name, cache_cfg.model_name, device=args.device
    )
    real_store = ActivationsStore.from_cache_activations(model, cache_cfg)
    real_provider = real_store  # ActivationsStore is itself an iterator of batches

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
        metadata=SAEMetadata(hook_name=args.hook_name, model_name=args.model_name),
    )
    trainer_cfg = SAETrainerConfig(
        total_training_samples=args.batch_size * args.steps,
        train_batch_size_samples=args.batch_size,
        lr=args.lr,
        lr_end=args.lr / 10,
        lr_scheduler_name="cosineannealing",
        device=args.device,
    )

    activation_scaler = ActivationScaler()

    def causal_eval_fn(sae: DiscreteTrainingSAE) -> dict[str, float]:
        return get_downstream_reconstruction_metrics(
            sae,
            model,
            real_store,
            activation_scaler,
            compute_kl=True,
            compute_ce_loss=True,
            n_batches=args.n_causal_eval_batches,
            eval_batch_size_prompts=args.model_batch_size,
        )

    decoder_bits_grid = [TERNARY_BITS, 2, 3, 4, None]

    print("Running sweep on real activations (with causal fidelity)...")
    real_cells = run_precision_sweep(
        real_provider,
        base_cfg,
        decoder_bits_grid,
        [None],
        trainer_cfg,
        causal_eval_fn=causal_eval_fn,
    )
    print("Running sweep on covariance-matched null (reconstruction only)...")
    null_cells = run_precision_sweep(
        null_provider, base_cfg, decoder_bits_grid, [None], trainer_cfg
    )

    print("\n=== Results ===")
    for label, cells in [("real", real_cells), ("null", null_cells)]:
        for cell in cells:
            print(
                f"{label} decoder_bits={cell.decoder_bits}: "
                f"reconstruction={cell.reconstruction} causal={cell.causal_fidelity}"
            )


if __name__ == "__main__":
    main()
