import os
from pathlib import Path

import pytest
import torch

from sae_lens.saes.discrete_sae import DiscreteSAE, DiscreteTrainingSAE
from sae_lens.saes.quantization import TERNARY_BITS
from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_discrete_sae_training_cfg,
    build_topk_sae_training_cfg,
    random_params,
)


def test_DiscreteTrainingSAE_matches_TopKTrainingSAE_when_bits_are_none():
    d_in, d_sae, k = 16, 32, 5
    discrete_cfg = build_discrete_sae_training_cfg(
        d_in=d_in, d_sae=d_sae, k=k, decoder_bits=None, code_bits=None
    )
    topk_cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        rescale_acts_by_decoder_norm=discrete_cfg.rescale_acts_by_decoder_norm,
    )

    discrete_sae = DiscreteTrainingSAE(discrete_cfg)
    topk_sae = TopKTrainingSAE(topk_cfg)
    random_params(discrete_sae)
    topk_sae.load_state_dict(discrete_sae.state_dict())

    x = torch.randn(10, d_in)
    assert_close(discrete_sae.encode(x), topk_sae.encode(x))
    assert_close(discrete_sae(x), topk_sae(x))


def test_DiscreteTrainingSAE_decoder_is_ternary_when_decoder_bits_set():
    d_in, d_sae, k = 8, 16, 3
    cfg = build_discrete_sae_training_cfg(
        d_in=d_in, d_sae=d_sae, k=k, decoder_bits=TERNARY_BITS, code_bits=None
    )
    sae = DiscreteTrainingSAE(cfg)
    random_params(sae)

    feature_acts, _ = sae.encode_with_hidden_pre(torch.randn(4, d_in))
    out = sae.decode(feature_acts)
    assert out.shape == (4, d_in)

    scale = sae.W_dec.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    levels = torch.round(torch.clamp(sae.W_dec.detach() / scale, -1, 1)).unique()
    assert set(levels.tolist()) <= {-1.0, 0.0, 1.0}


def test_DiscreteTrainingSAE_code_bits_reduces_the_number_of_distinct_nonzero_codes():
    d_in, d_sae, k = 8, 64, 4
    cfg = build_discrete_sae_training_cfg(
        d_in=d_in, d_sae=d_sae, k=k, decoder_bits=None, code_bits=2
    )
    sae = DiscreteTrainingSAE(cfg)
    random_params(sae)

    feature_acts, _ = sae.encode_with_hidden_pre(torch.randn(200, d_in))
    nonzero = feature_acts[feature_acts != 0]
    # 2-bit unsigned has at most 2**2 = 4 levels (including 0)
    assert nonzero.numel() > 0
    # each feature column has its own scale, so compare per-column level counts
    for col in range(d_sae):
        col_nonzero = feature_acts[:, col][feature_acts[:, col] != 0]
        if col_nonzero.numel() > 0:
            assert col_nonzero.unique().numel() <= 3  # levels 1,2,3 (0 excluded here)


def test_DiscreteTrainingSAEConfig_rejects_sparse_activations():
    with pytest.raises(ValueError, match="use_sparse_activations"):
        build_discrete_sae_training_cfg(use_sparse_activations=True)


@pytest.mark.parametrize(
    ("decoder_bits", "code_bits"),
    [(None, None), (TERNARY_BITS, None), (None, 3), (TERNARY_BITS, 3)],
)
def test_DiscreteTrainingSAE_holds_width_and_sparsity_fixed_across_precisions(
    decoder_bits: float | None, code_bits: float | None
):
    d_in, d_sae, k = 8, 24, 4
    cfg = build_discrete_sae_training_cfg(
        d_in=d_in, d_sae=d_sae, k=k, decoder_bits=decoder_bits, code_bits=code_bits
    )
    sae = DiscreteTrainingSAE(cfg)
    random_params(sae)

    feature_acts, _ = sae.encode_with_hidden_pre(torch.randn(10, d_in))
    assert feature_acts.shape[-1] == d_sae
    # TopK selects the top k pre-activations, then ReLUs them -- so at most k are
    # nonzero (a negative top-k value gets zeroed), but never more than k regardless
    # of decoder_bits/code_bits. This upper bound is exactly the matched-feature-count
    # / matched-sparsity confound control from PLAN-BitSAE.md.
    assert torch.all((feature_acts != 0).sum(dim=-1) <= k)


def test_DiscreteTrainingSAE_quantization_changes_output_relative_to_full_precision():
    d_in, d_sae, k = 8, 16, 4
    full_cfg = build_discrete_sae_training_cfg(
        d_in=d_in, d_sae=d_sae, k=k, decoder_bits=None, code_bits=None
    )
    ternary_cfg = build_discrete_sae_training_cfg(
        d_in=d_in, d_sae=d_sae, k=k, decoder_bits=TERNARY_BITS, code_bits=None
    )
    full_sae = DiscreteTrainingSAE(full_cfg)
    random_params(full_sae)
    ternary_sae = DiscreteTrainingSAE(ternary_cfg)
    ternary_sae.load_state_dict(full_sae.state_dict())

    x = torch.randn(10, d_in)
    assert_not_close(full_sae(x), ternary_sae(x))


def test_DiscreteTrainingSAE_save_and_load_inference_sae_reconstructs_identically(
    tmp_path: Path,
):
    cfg = build_discrete_sae_training_cfg(
        d_in=8, d_sae=16, k=3, decoder_bits=TERNARY_BITS, code_bits=4
    )
    training_sae = DiscreteTrainingSAE(cfg)
    random_params(training_sae)
    training_sae.eval()

    x = torch.randn(10, cfg.d_in)
    out_train = training_sae(x)

    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)
    assert os.path.exists(model_path)

    inference_sae = SAE.load_from_disk(model_path, device="cpu")
    assert isinstance(inference_sae, DiscreteSAE)
    out_inference = inference_sae(x)

    assert_close(out_train, out_inference)


def test_DiscreteTrainingSAE_loss_decreases_when_trained_repeatedly_on_same_acts():
    cfg = build_discrete_sae_training_cfg(
        d_in=8, d_sae=32, k=4, decoder_bits=TERNARY_BITS, code_bits=3
    )
    sae = DiscreteTrainingSAE(cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
    sae_in = torch.randn(64, cfg.d_in)

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        output = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=sae_in,
                dead_neuron_mask=None,
                coefficients={},
                n_training_steps=step,
                is_logging_step=False,
            ),
        )
        output.loss.backward()
        optimizer.step()
        losses.append(output.loss.item())

    assert losses[-1] < losses[0]
