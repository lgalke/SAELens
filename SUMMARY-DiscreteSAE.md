# DiscreteSAE — Implementation Summary & Runbook

Implements the foundation for `PLAN-BitSAE.md`: a quantization-aware SAE (BitNet-style
STE decoder/code precision) inside SAELens, plus the covariance-matched null generator
and 2-D precision-sweep harness needed for experiments E0–E3. This document summarizes
what was built and gives copy-pasteable commands for running the test suite and the
two experiment scripts.

## What was built

### New library code (`sae_lens/`)

| File | Purpose |
|---|---|
| `sae_lens/saes/quantization.py` | STE fake-quantization primitive `quantize(x, bits, *, signed, dim)`. `bits=None` → full precision, `bits=TERNARY_BITS` (1.58) → ternary `{-1,0,+1}·scale`, integer `b>=2` → symmetric b-bit. Scale is self-referential (absmean for ternary, absmax otherwise) and provably homogeneous under positive per-row scaling — compatible with decoder-norm folding. |
| `sae_lens/saes/discrete_sae.py` | `DiscreteSAEConfig`/`DiscreteSAE` and `DiscreteTrainingSAEConfig`/`DiscreteTrainingSAE` — a TopK SAE subclass with independently-toggleable `decoder_bits` (dictionary/feature-direction precision) and `code_bits` (feature-magnitude precision). Registered as architecture `"discrete"`. Width (`d_sae`) and sparsity (`k`) are untouched by precision — the *matched-feature-count confound* from the plan is enforced by construction, not by convention. |
| `sae_lens/synthetic/null_activations.py` | `GaussianNullGenerator` (covariance-matched Gaussian null) and `shuffled_null_sampler` (marginal-preserving, covariance-destroying null) — the E3 confound-check generators. `batch_iterator()` wraps either into an infinite iterator of batches. |
| `sae_lens/discrete_sweep.py` | Model-agnostic `run_precision_sweep()`: trains a fresh `DiscreteTrainingSAE` per `(decoder_bits, code_bits)` grid cell (width/k held fixed), reports reconstruction metrics (`mse`, `l0`, `explained_variance`, via the existing `ExplainedVarianceCalculator`) always, and causal-fidelity metrics (KL/CE recovered) only if a `causal_eval_fn` callback is supplied — nulls have no causal fidelity by construction, so they get `None`. |

### Scripts (`scripts/`) — for running on real compute, not CI

| File | Purpose |
|---|---|
| `scripts/discrete_sae_synthetic_e3_experiment.py` | E3 confound check on **synthetic** data: sparse combinations of a ternary ground-truth dictionary (standing in for "LLM features are near a low-bit lattice") vs. a covariance-matched Gaussian null. Recovering the ground-truth structure via sparse dictionary learning needs a real step budget (thousands of steps), so this is a script, not a unit test. |
| `scripts/discrete_sae_precision_sweep.py` | The real experiment: caches one layer's residual stream from a real LLM (reusing the existing `CacheActivationsRunner`), runs the full decoder-bits × code-bits sweep on it with causal fidelity (KL/CE recovered via `sae_lens.evals`), and runs the same sweep on a covariance-matched null (reconstruction only). |

### Tests (all passing; 872 pre-existing tests unaffected)

- `tests/saes/test_quantization.py` — STE correctness, ternary/b-bit level counts, straight-through gradient, homogeneity property, dead-row safety.
- `tests/saes/test_discrete_sae.py` — matches plain TopK when bits are `None`; decoder is genuinely ternary; code quantization reduces distinct levels; width/sparsity invariant across precisions; train↔inference exact parity; loss decreases when trained.
- `tests/synthetic/test_null_activations.py` — Gaussian null matches empirical mean/cov at large N; shuffled null preserves marginals but destroys covariance.
- `tests/test_discrete_sweep.py` — sweep harness mechanics: matched feature count across the grid, decoder bits actually changing reconstruction, the causal-fidelity callback wiring, and the full real+null pipeline running end to end. (Deliberately does **not** assert a real-vs-null statistical gap — that's an open empirical question requiring a real compute budget, not something to force into a fast/flaky CI test. See the scripts above for that.)
- `tests/helpers.py` / `tests/saes/test_sae.py` — added `build_discrete_*` config builders and dispatch-dict entries so `"discrete"` is automatically covered by the existing cross-architecture parametrized tests (config round-trip, activation-norm folding, decoder-norm folding, checkpoint save/load).

### Design decisions worth remembering

- **Base architecture: TopK.** Fixed `k` across every precision cell gives the cleanest matched comparison. Quantizers are standalone functions, so a ReLU/L1 variant is a small future addition, not built now.
- **No decoder-norm folding at save time.** `decode()` always recomputes the ternary/b-bit scale from the live `W_dec`, so training and inference are numerically identical without folding. Skips TopK's `_fold_norm_topk` at save.
- **Null vs. causal fidelity.** Synthetic null activations have no LLM behind them, so they can only be evaluated on reconstruction. The real-vs-null gap is measured on reconstruction; causal fidelity (KL/CE recovered) is a real-activations-only headline. This resolves the apparent tension in the plan's proposed headline figure.
- **Encoder always full precision** — per the plan, discreteness is a claim about the SAE's dictionary/codes, not the encoder.

## Running the test suite

The sandbox this was built in had no `poetry` executable, so tests were run against a
plain venv built from `pyproject.toml`'s dependencies. If you have `poetry` available,
the normal project workflow applies:

```bash
poetry install
poetry run pytest tests/saes/test_quantization.py tests/saes/test_discrete_sae.py \
    tests/synthetic/test_null_activations.py tests/test_discrete_sweep.py -v

# full suite + lint + types
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run pyright
```

Without poetry, in any Python 3.10+ environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # main deps
pip install pytest pytest-cov pytest-randomly ruff==0.7.4 pyright  # dev tools (pin ruff to match pyproject.toml)
pip install eai-sparsify dictionary-learning kaleido  # only needed for unrelated pre-existing tests

pytest tests/saes/test_quantization.py tests/saes/test_discrete_sae.py \
    tests/synthetic/test_null_activations.py tests/test_discrete_sweep.py -v
```

## Running the experiments

### 1. Synthetic E3 confound check (no LLM required, cheap to start on CPU, scale up on GPU)

Sanity-check wiring first (seconds, not meaningful results):

```bash
python scripts/discrete_sae_synthetic_e3_experiment.py \
    --d_in 8 --d_sae 16 --k 3 \
    --num_ground_truth_features 16 --k_active 3 \
    --batch_size 32 --steps 20 --device cpu
```

Real run, on a GPU, with enough steps to actually let the SAE recover the sparse
dictionary structure (this is the part that needs real compute — a few hundred CPU
steps is not enough to distinguish real from null):

```bash
python scripts/discrete_sae_synthetic_e3_experiment.py \
    --d_in 64 \
    --d_sae 256 \
    --k 16 \
    --num_ground_truth_features 512 \
    --k_active 16 \
    --batch_size 256 \
    --steps 20000 \
    --lr 1e-3 \
    --device cuda
```

All flags (defaults shown):

| Flag | Default | Meaning |
|---|---|---|
| `--d_in` | 64 | Activation dimensionality |
| `--d_sae` | 256 | SAE dictionary size (fixed across the precision sweep) |
| `--k` | 16 | SAE TopK sparsity (fixed across the precision sweep) |
| `--num_ground_truth_features` | 512 | Size of the ternary ground-truth dictionary (overcomplete if > `d_in`) |
| `--k_active` | 16 | Average number of ground-truth features active per sample |
| `--batch_size` | 256 | Training batch size |
| `--steps` | 5000 | Training steps per sweep cell (4 cells run per invocation: {ternary, full} × {real, null}) |
| `--lr` | 1e-3 | Learning rate (cosine-annealed to `lr/10`) |
| `--device` | `cuda` if available else `cpu` | Torch device |

Output: prints per-cell reconstruction metrics (`mse`, `l0`, `explained_variance`) for
both the real and null runs, then the explained-variance gap between full precision and
ternary for each, and whether the gap is smaller for real activations (the claim).

### 2. Real-LLM precision sweep (E1/E2/E3 on an actual model's residual stream)

Requires a GPU for any non-trivial model/dataset, and a HuggingFace dataset accessible
to the environment (defaults to `NeelNanda/c4-tokenized-2b`).

```bash
python scripts/discrete_sae_precision_sweep.py \
    --model_name gpt2 \
    --hook_name blocks.6.hook_resid_pre \
    --d_in 768 \
    --dataset_path NeelNanda/c4-tokenized-2b \
    --context_size 128 \
    --cache_training_tokens 2000000 \
    --model_batch_size 16 \
    --d_sae 6144 \
    --k 32 \
    --batch_size 1024 \
    --steps 5000 \
    --lr 3e-4 \
    --n_causal_eval_batches 10 \
    --device cuda
```

All flags (defaults shown):

| Flag | Default | Meaning |
|---|---|---|
| `--model_name` | `gpt2` | TransformerLens model name |
| `--hook_name` | `blocks.6.hook_resid_pre` | Residual-stream hook to cache/train on |
| `--d_in` | 768 | Must match the model's residual-stream width at that hook |
| `--dataset_path` | `NeelNanda/c4-tokenized-2b` | HF dataset for activation caching |
| `--context_size` | 128 | Sequence length used when caching |
| `--cache_training_tokens` | 2,000,000 | Total tokens to cache to disk (one-time cost, reused across the whole sweep) |
| `--model_batch_size` | 16 | Batch size for the LLM forward pass during caching/eval |
| `--d_sae` | `d_in * 8` (6144 for gpt2) | SAE dictionary size (fixed across the sweep) |
| `--k` | 32 | SAE TopK sparsity (fixed across the sweep) |
| `--batch_size` | 1024 | SAE training batch size (tokens) |
| `--steps` | 5000 | Training steps per sweep cell (10 cells: `{ternary, 2, 3, 4, full}` × `{real, null}`) |
| `--lr` | 3e-4 | Learning rate (cosine-annealed to `lr/10`) |
| `--n_causal_eval_batches` | 10 | Prompt batches used for the KL/CE-recovered causal-fidelity eval (real activations only) |
| `--device` | `cuda` if available else `cpu` | Torch device |

Output: caches activations to `activations/{dataset}/{model}/{hook_name}` (skipped on
reruns if the cache already exists), then prints per-cell reconstruction metrics for
both real and null runs, plus causal-fidelity metrics (`kl_div_score`, `ce_loss_score`,
etc.) for the real run only.

**Before a long/expensive run**, sanity-check wiring on a tiny slice first, e.g.:

```bash
python scripts/discrete_sae_precision_sweep.py \
    --model_name tiny-stories-1M --hook_name blocks.0.hook_resid_pre --d_in 64 \
    --dataset_path NeelNanda/c4-10k --cache_training_tokens 5000 \
    --d_sae 128 --k 8 --batch_size 32 --steps 20 --n_causal_eval_batches 2 --device cpu
```
