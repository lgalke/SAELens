# PLAN.md — Quantization-Aware SAEs as a Probe of Feature Discreteness

## One-line summary

Impose quantization-aware training (QAT) on a sparse autoencoder (SAE) trained on a **full-precision, untouched** LLM's residual stream, and use the lowest bit-precision at which reconstruction and *causal* fidelity survive as an **estimate of the intrinsic precision of the model's feature geometry**. The scientific claim: LLM feature bases are approximately discrete — meaningful directions live near a low-bit lattice — and this precision floor is a measurable, model-agnostic quantity.

## What this is NOT

- Not about interpreting already-quantized LLMs. The base model is always full-precision; quantization is a constraint we impose on the *SAE*, not inherited from the model.
- Not an efficiency/deployment project. Low-bit is the *instrument*, not the payoff. Do not optimize for speed or memory.
- Not "we built a quantized SAE." The deliverable is a claim about the *model's* geometry, defended against the confounds below.

## Core design

Base LLM is frozen and full-precision throughout. We only ever read the residual stream. The SAE is the QAT object. Encoder-side precision is irrelevant to the claim — **keep the encoder full-precision** unless an experiment specifically says otherwise.

Two knobs carry interpretability meaning:

1. **Dictionary precision** (decoder columns = feature *directions*). This is the centerpiece: the claim is about directions, so ternarizing the decoder is the primary axis.
2. **Code precision** (sparse latent activations = feature *magnitudes*). Secondary axis: given ternarized directions, how few levels does activation need?

The experimental spine is a **2-D sweep**: bits-in-dictionary × bits-in-code, with reconstruction and intervention-fidelity surfaces over it. The corner we're betting exists: ternary dictionary + few-level codes + negligible loss in causal footprint.

## Implementation notes

- Quantize via straight-through estimator (STE). Ternary = {−1, 0, +1} with a learned or absmean-derived scale per decoder column; generalize to b-bit for the sweep.
- Baseline SAE: standard architecture (start with a plain ReLU/L1 or TopK SAE — pick one and hold it fixed). Reuse an existing SAE codebase rather than writing from scratch; wrap the decoder/codes with quantization layers.
- Residual-stream activations only. Cache activations to disk for a fixed layer first; parameterize layer/depth later.
- **Everything must be model-agnostic**: the only model-specific object is the activation cache. The SAE + quant + eval pipeline takes activations in and knows nothing about which LLM produced them.

## The two confounds to preempt (build these in from the start — not later)

### 1. Matched feature count
Ternarizing the decoder must NOT be allowed to silently buy back fidelity via width. Hold the dictionary size (number of features) **fixed** across all precisions. The result of interest is low-bit fidelity *at matched feature count*. If ternary needs 3× the features to hold reconstruction, that's precision-for-width trading, not discreteness — and the "intrinsic precision" number is confounded. Report the frontier with width pinned.

### 2. Covariance-matched null
Train the same-width SAE, same quantization sweep, on **synthetic activations with the same second-order statistics** as the real ones (Gaussian matched to the empirical mean/covariance of the residual-stream activations; also try a shuffled/phase-randomized variant). If the null tolerates ternarization just as well as real activations, then low-bit sufficiency is a fact about overcomplete dictionaries, not about LLM features. **The claim requires real activations to sit at a measurably different (lower) precision floor than the null.** This comparison is what turns "we built a quantized SAE" into "LLM feature geometry is measurably discrete." It is not optional.

## The estimator fork (decide before building the sweep)

The "precision floor" can be defined by:

- **Reconstruction** (MSE / variance-explained), or
- **Causal fidelity** (ablate or set a code, measure downstream logit effect).

These can dissociate: a decoder can round-trip activations well while its columns become causally mushier, or vice versa. **Default to causal fidelity as the headline estimator** — reconstruction is gameable by the overcomplete basis — but measure and report both. Settle this explicitly before building, because it determines what "the feature geometry needs N bits" actually asserts.

Causal-fidelity metric (concrete): for a feature, intervene on its code (ablate to 0, or set to its quantized on-value), run the model forward, measure KL / logit-diff on the affected tokens vs. the full-precision-SAE intervention. "Fidelity preserved" = the ternary intervention reproduces the full-precision intervention's causal footprint, not just its magnitude.

## Experiments (in order)

**E0 — QAT-as-regularizer sanity (optional but cheap, do first).**
Before quantizing the SAE at all: train a normal full-precision SAE on the real activations, establish the reconstruction/L0 and causal-fidelity baselines. These are the reference points every quantized config is compared against.

**E1 — Dictionary precision sweep (primary).**
Sweep decoder bits {ternary, 2-bit, 3-bit, 4-bit, full}, codes full-precision, width fixed. Reconstruction + causal fidelity vs. bits. Locate the precision floor.

**E2 — Code precision sweep (secondary).**
Fix decoder at the E1 floor, sweep code levels. Find the joint corner (ternary dict + few-level codes).

**E3 — Null comparison (decisive).**
Repeat E1 (and the relevant E2 slice) on covariance-matched Gaussian + shuffled nulls, same width. The headline result is the **gap** between the real-activation precision floor and the null's floor.

**E4 — Scale/depth curve (the strongest version of the claim).**
Run the identical protocol across a model family and/or across depths within one model. Report the precision floor as a **curve**: does feature discreteness increase with scale / with depth? "Discreteness increases with scale" is a far more compelling result than a single-model existence proof, and it speaks to the superposition-decongestion hypothesis (does a tighter activation budget decongest superposition toward a more axis-aligned code?).

## Headline figure

Bits-per-feature (x) vs. causal-fidelity (y), one curve for real activations and one for the null, ideally faceted across scale/depth. The story lives or dies on: real activations tolerate far lower precision than the null, at matched feature count, with the causal footprint preserved.

## Connections (context, not requirements)

- A ternary dictionary with few-level codes is operationally a **natural-language-addressable switchboard** — the discreteness claim and steerability (CAVES-style) are the same result viewed twice. Steering = setting a code to its quantized on-value.
- Feeds the superposition-scaling question directly via E4.

## Deferred / explicitly out of scope

- Encoder quantization (no interpretability value for the claim; revisit only for a matched-precision on-device probe, which is a different paper).
- Any efficiency benchmarking.
- Automated feature *labeling* — useful downstream, but the precision-floor claim stands without it.

## First tasks for the coding agent

1. Set up activation caching for one LLM, one fixed layer, residual stream. Keep the LLM-facing code isolated behind a thin interface.
2. Stand up a baseline full-precision SAE on that cache (E0). Report reconstruction, L0, and a first causal-fidelity number.
3. Add STE ternary/​b-bit wrappers around (a) decoder columns and (b) codes, toggleable independently, width held fixed.
4. Build the null-activation generator (Gaussian matched to empirical mean/cov; plus a shuffled variant).
5. Wire the 2-D sweep harness and the two metric surfaces (reconstruction, causal fidelity). Everything downstream of the activation cache must be model-agnostic.
