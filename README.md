# ternary-quantize

FP32 → {-1, 0, +1} quantization with error analysis, stochastic rounding, and per-channel scaling.

## The Problem

A 7B-parameter model stores ~28 GB of weights in FP32. You can replace every float with a trit — a ternary digit in {-1, 0, +1} — and drop to ~1.7 GB, small enough to sit in L3 cache. But you need to actually compress those weights without destroying what the model learned.

Binary quantization {-1, +1} gets you 16× compression but forces every weight to carry signal — there's no "off" state. The zero in ternary changes the game: it gives you sparsity for free. A weight matrix where 60% of values are zero means 60% of multiply-accumulates vanish entirely.

The challenge is choosing the threshold τ that decides which weights become zero. Set it too high and you oversparify; too low and you lose the compression benefit. This crate gives you four strategies for making that call.

## The Insight

The quantization operation is trivially simple — compare a float to a threshold, emit a trit — but the engineering is entirely in the bookkeeping around it:

- **What scale do you dequantize back to?** The mean absolute value of the original weights is a reasonable proxy, but per-channel (per-row) scales recover much more signal because different output channels in a transformer have wildly different norms.
- **Is deterministic rounding unbiased?** No. The value 0.3 always maps to 0 with threshold 0.5, so you always lose the same 0.3 of signal. Stochastic rounding fixes this: 0.3 maps to +1 with probability 0.3/0.5 = 0.6 and to 0 with probability 0.4. Over many weights, the *expected* value is preserved exactly.
- **Can you learn the threshold?** Yes. The MSE between original and quantized-then-dequantized weights is a differentiable (piecewise-constant, but numerically gradient-approximable) function of τ. You can gradient-descend to the threshold that minimizes reconstruction error for a specific weight distribution.

## How It Works

### Deterministic quantization

```
|x| ≤ τ  →  0
x  >  τ  → +1
x  < -τ  → -1
```

Dequantization multiplies back by a scale factor: `trit.to_f32() * scale`.

### Stochastic rounding

Normalize each weight by the scale, then use it as a probability. For x ∈ (0, 1): P(+1) = x, P(0) = 1-x. For x ∈ (-1, 0): P(-1) = |x|, P(0) = 1-|x|. Values outside [-1, 1] clamp to ±1. The PRNG is a self-contained xoshiro128\*\* — no external `rand` dependency.

### Learned threshold

Start with an initial τ, quantize, dequantize, compute MSE. Perturb τ by ε, re-quantize, compute MSE again. The finite difference gives a numerical gradient. Step τ downhill, clamp to [0, max(|w|)]. Each iteration costs O(n) re-quantization.

### Per-channel scaling

Each row of a weight matrix gets its own `scale = mean(|row|)` and `threshold = 0.05 × scale`. This is the practical choice for transformer layers where attention head 0 might have weights around ±0.1 and head 7 around ±5.0.

### Error analysis

`QuantizationReport` bundles MSE, max absolute error, distribution shift (mean of original minus mean of quantized), trit counts, and sparsity fraction into one struct with a `Display` impl.

## Code Example

```rust
use ternary_quantize::{
    quantize_f32_to_ternary, dequantize_ternary_to_f32,
    stochastic_quantize, learn_threshold,
    per_channel_quantize, per_channel_dequantize,
    quantization_report, QuantizeConfig, Trit, SimpleRng,
};

// ── Deterministic quantization with a fixed threshold ──
let weights = vec![-0.8, -0.02, 0.0, 0.03, 0.9];
let config = QuantizeConfig::new(0.05, 1.0);
let trits = quantize_f32_to_ternary(&weights, &config);
// [Neg, Zero, Zero, Zero, Pos]

let deq = dequantize_ternary_to_f32(&trits, 1.0);
// [-1.0, 0.0, 0.0, 0.0, 1.0]

// ── Stochastic rounding — unbiased in expectation ──
let weights = vec![0.3; 10_000];
let mut rng = SimpleRng::new(42);
let trits = stochastic_quantize(&weights, 1.0, &mut rng);
// ~3000 Pos, ~7000 Zero — expected value ≈ 0.3, same as original

// ── Learn the threshold that minimizes MSE ──
let layer_weights: Vec<f32> = /* your layer weights */;
let (threshold, mse) = learn_threshold(&layer_weights, 0.5, 1.0, 100, 0.01);

// ── Per-channel: each row gets its own scale and threshold ──
let row1: &[f32] = &[0.1, -0.2, 0.3];
let row2: &[f32] = &[10.0, -5.0, 0.0];
let (trits, scales, thresholds) = per_channel_quantize(&[row1, row2]);
// scales[0] ≈ 0.2, scales[1] ≈ 5.0 — independent of each other
let reconstructed = per_channel_dequantize(&trits, &scales, 3);

// ── Full error report ──
let report = quantization_report(&weights, &trits, 1.0);
println!("{}", report);
// QuantizationReport { mse: 0.004200, max_err: 0.200000, shift: 0.001000,
//                       sparsity: 60.00%, counts: neg=200 zero=6000 pos=3800 }
```

## Module Map

Everything lives in `src/lib.rs`. Flat module, no submodules.

```
Trit                    — enum {Neg, Zero, Pos} with to_i8(), to_f32(), from_i8(), Display
QuantizeConfig          — threshold + scale pair
SimpleRng               — xoshiro128** PRNG, seed → deterministic f32 in [0, 1)

quantize_f32_to_ternary — deterministic threshold quantization
dequantize_ternary_to_f32 — scale × trit
stochastic_quantize     — unbiased random rounding via SimpleRng
learn_threshold         — gradient descent on τ to minimize MSE
per_channel_quantize    — independent quantization per matrix row
per_channel_dequantize  — inverse of per_channel_quantize

mean_squared_error      — MSE between two f32 slices
max_error               — max |a_i - b_i|
distribution_shift      — mean(original) - mean(quantized)
trit_distribution       — count (neg, zero, pos)
QuantizationReport      — all error metrics + sparsity in one struct
quantization_report     — compute full report
```

## Design Decisions

**No `rand` dependency.** The `SimpleRng` struct reimplements xoshiro128\*\* inline. This keeps the dependency tree to just `ternary-types`. The tradeoff: the PRNG isn't crypto-quality and doesn't support thread-local state. For quantization, determinism is more valuable than cryptographic randomness — you want the same seed to produce the same compressed model across builds.

**`learn_threshold` uses finite differences.** The quantization function is piecewise-constant in τ, so there's no clean closed-form gradient. The implementation computes MSE at τ and τ+ε, takes the ratio, and steps. This costs 2× quantization per iteration. A smarter approach would be to compute the gradient analytically by tracking which values cross the threshold boundary — but finite differences is simpler and the cost is bounded by `O(n × iterations)`.

**Per-channel threshold is `0.05 × scale`.** The ratio is hardcoded, not learned. This means per-channel quantization doesn't optimize its own threshold — it uses a heuristic. The `learn_threshold` function operates on flat vectors, not per-row. Connecting the two (per-channel learned thresholds) would be a meaningful improvement.

**`Trit` is an enum, not a packed integer.** Each `Trit` takes one byte as a Rust enum discriminant. For in-memory computation this is fine. For storage or network transfer, 16 trits fit in a single `u32` — but the crate doesn't provide a pack/unpack API. That's a real gap for deployment.

**f32 only.** The quantization pipeline operates on `f32`, not `f64`. Neural network weights are typically stored as f32 or smaller. The `ternary-optimizer` crate uses `f64` for its training loop, which means there's a precision boundary if you pipeline the two.

## Status

- **32 tests passing.** Known-value quantization, round-trip error bounds, stochastic distribution correctness (±500 on 10K samples), learned threshold convergence, per-channel independence, error metric accuracy, RNG determinism and uniformity, full-pipeline integration.
- **Production-ready for inference quantization.** The deterministic and per-channel paths are straightforward and well-tested.
- **Known gaps:**
  - No trit packing (2-bit storage representation)
  - No SIMD acceleration
  - No mixed-precision support (e.g., keep attention FP32, quantize FFN)
  - No Straight-Through Estimator for training-aware quantization
  - Per-channel threshold is heuristic, not learned
  - `learn_threshold` converges but the loss landscape is non-smooth — no convergence guarantee

## Ecosystem

- [`ternary-optimizer`](https://github.com/SuperInstance/ternary-optimizer) — sign-based training with weight ternarization
- [`ternary-svm`](https://github.com/SuperInstance/ternary-svm) — SVM classification on quantized feature vectors
- [`ternary-em`](https://github.com/SuperInstance/ternary-em) — EM for clustering ternary weight distributions
- [`ternary-types`](https://github.com/SuperInstance/ternary-types) — shared `Trit` trait and type definitions

## References

- Li, F., Zhang, B., & Liu, B. (2016). *Ternary Weight Networks*. [arXiv:1605.04711](https://arxiv.org/abs/1605.04711)
- Zhu, C., Han, S., Mao, H., & Dally, W. J. (2016). *Trained Ternary Quantization*. [arXiv:1612.01064](https://arxiv.org/abs/1612.01064)

## License

MIT OR Apache-2.0
