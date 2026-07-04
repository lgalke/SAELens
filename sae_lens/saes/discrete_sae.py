"""DiscreteSAE: a TopK SAE with independently-toggleable quantization of the decoder
(feature directions) and/or codes (feature magnitudes), trained with a straight-through
estimator (STE).

This is the QAT object described in PLAN-BitSAE.md: the encoder stays full precision
(it is not part of the discreteness claim), while `decoder_bits` and `code_bits` let a
2-D bits-in-dictionary x bits-in-code sweep be run at fixed dictionary size (`d_sae`)
and fixed sparsity (`k`), isolating precision from width/sparsity confounds.
"""

from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import override

from sae_lens.saes.quantization import quantize
from sae_lens.saes.sae import TrainingSAE
from sae_lens.saes.topk_sae import (
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)


@dataclass
class DiscreteSAEConfig(TopKSAEConfig):
    """
    Configuration class for DiscreteSAE inference.

    Args:
        decoder_bits (float | None): Bit-width for decoder columns (feature directions).
            `None` for full precision, `1.58` for ternary ({-1, 0, +1}), or an integer
            >= 2 for symmetric b-bit quantization.
        code_bits (float | None): Bit-width for codes (feature magnitudes), applied to
            the nonzero entries selected by TopK. Same convention as `decoder_bits`.
    """

    decoder_bits: float | None = None
    code_bits: float | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "discrete"


class DiscreteSAE(TopKSAE):
    """Inference-only DiscreteSAE: applies decoder/code quantization at inference time too."""

    cfg: DiscreteSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: DiscreteSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return quantize(feature_acts, self.cfg.code_bits, signed=False, dim=0)

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return _discrete_decode(self, feature_acts)


@dataclass
class DiscreteTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training a DiscreteTrainingSAE.

    Args:
        decoder_bits (float | None): Bit-width for decoder columns (feature directions).
            `None` for full precision, `1.58` for ternary, or an integer >= 2 for
            symmetric b-bit quantization. This is the primary axis of the precision
            sweep (see PLAN-BitSAE.md).
        code_bits (float | None): Bit-width for codes (feature magnitudes), applied to
            the nonzero entries selected by TopK. Secondary axis of the sweep.
    """

    decoder_bits: float | None = None
    code_bits: float | None = None

    @override
    @classmethod
    def architecture(cls) -> str:
        return "discrete"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.use_sparse_activations:
            raise ValueError(
                "DiscreteSAE does not support use_sparse_activations=True: code "
                "quantization is only implemented for dense activations."
            )


class DiscreteTrainingSAE(TopKTrainingSAE):
    """
    TopK SAE variant with STE quantization of the decoder columns and/or codes.

    Width (`d_sae`) and sparsity (`k`) are controlled entirely by the inherited TopK
    config and are unaffected by `decoder_bits`/`code_bits`, so sweeping precision does
    not change the dictionary size or L0 -- the matched-feature-count confound from
    PLAN-BitSAE.md is enforced by construction.
    """

    cfg: DiscreteTrainingSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: DiscreteTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        feature_acts = quantize(feature_acts, self.cfg.code_bits, signed=False, dim=0)
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return _discrete_decode(self, feature_acts)

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        # Skip TopKTrainingSAE's decoder-norm folding: folding is a nonlinear rescaling
        # of W_dec that is not needed here, since decode() always recomputes the
        # rescale norm from the (unquantized) W_dec at call time, both at train and
        # inference time. Falling through to the base TrainingSAE implementation keeps
        # W_dec as-is so the inference SAE re-quantizes identically to training.
        TrainingSAE.process_state_dict_for_saving_inference(self, state_dict)


def _discrete_decode(
    sae: "DiscreteSAE | DiscreteTrainingSAE", feature_acts: torch.Tensor
) -> torch.Tensor:
    """Shared decode logic for DiscreteSAE and DiscreteTrainingSAE.

    The rescale-by-decoder-norm bookkeeping (if enabled) always uses the full-precision
    `W_dec`, matching what `encode`/`encode_with_hidden_pre` used to select the topk
    activations. Only the matmul itself uses the quantized decoder.
    """
    W_dec_q = quantize(sae.W_dec, sae.cfg.decoder_bits, signed=True, dim=1)
    if sae.cfg.rescale_acts_by_decoder_norm:
        feature_acts = feature_acts / sae.W_dec.norm(dim=-1)
    sae_out_pre = feature_acts @ W_dec_q + sae.b_dec
    sae_out_pre = sae.hook_sae_recons(sae_out_pre)
    sae_out_pre = sae.run_time_activation_norm_fn_out(sae_out_pre)
    return sae.reshape_fn_out(sae_out_pre, sae.d_head)
