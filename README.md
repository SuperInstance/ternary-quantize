# ternary-quantize

**Ternary quantization and dequantization for neural network weights.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

---

## Overview

`ternary-quantize` implements quantization of full-precision (f32) neural network weights into ternary values {−1, 0, +1}, along with the inverse dequantization, stochastic rounding, learned-threshold optimization, per-channel quantization, and comprehensive error metrics.

Ternary Weight Networks (TWNs, [Li et al. 2016]) constrain weights to three values, reducing model size by ~16× and replacing multiplications with sign operations. This crate provides the quantization algorithms needed to train and deploy such networks.

## Quick Start

### Deterministic Quantization

```rust
use ternary_quantize::{quantize_f32_to_ternary, dequantize_ternary_to_f32, QuantizeConfig, Trit};

let weights = vec![-0.8, -0.02, 0.0, 0.03, 0.9];
let config = QuantizeConfig::new(0.05, 1.0); // threshold=0.05, scale=1.0

let trits = quantize_f32_to_ternary(&weights, &config);
assert_eq!(trits, vec![Trit::Neg, Trit::Zero, Trit::Zero, Trit::Zero, Trit::Pos]);

let deq = dequantize_ternary_to_f32(&trits, 1.0);
assert_eq!(deq, vec![-1.0, 0.0, 0.0, 0.0, 1.0]);
```

### Stochastic Quantization

Reduces systematic bias by randomly rounding to the nearest ternary value with probability proportional to distance:

```rust
use ternary_quantize::{stochastic_quantize, SimpleRng};

let weights = vec![0.3, 0.3, 0.3, 0.3, 0.3];
let mut rng = SimpleRng::new(42);

let trits = stochastic_quantize(&weights, 1.0, &mut rng);
// ~30% will be +1, ~70% will be 0 — unbiased in expectation
```

### Learned Threshold

Iteratively optimize the quantization threshold to minimize MSE:

```rust
use ternary_quantize::learn_threshold;

let weights: Vec<f32> = /* your layer weights */;
let (threshold, mse) = learn_threshold(&weights, 0.5, 1.0, 100, 0.01);
// threshold is now optimized for this weight distribution
```

### Per-Channel Quantization

Each output channel (row of a weight matrix) gets its own scale and threshold:

```rust
use ternary_quantize::per_channel_quantize;

let row1: &[f32] = &[0.1, -0.2, 0.3];
let row2: &[f32] = &[10.0, -5.0, 0.0];

let (trits, scales, thresholds) = per_channel_quantize(&[row1, row2]);
// scales[0] ≈ 0.2, scales[1] ≈ 5.0 — independent per channel
```

### Error Metrics

```rust
use ternary_quantize::{quantization_report, QuantizeConfig};

let original = vec![0.5, -0.3, 0.0, 0.8, -0.9];
let config = QuantizeConfig::new(0.1, 1.0);
let trits = quantize_f32_to_ternary(&original, &config);
let report = quantization_report(&original, &trits, 1.0);

println!("{}", report);
// QuantizationReport { mse: 0.028000, max_err: 0.300000, shift: 0.000000,
//                       sparsity: 20.00%, counts: neg=2 zero=1 pos=2 }
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Trit` | A ternary value: `Neg` (-1), `Zero` (0), `Pos` (+1) |
| `QuantizeConfig` | Threshold + scale for deterministic quantization |
| `QuantizationReport` | MSE, max error, distribution shift, sparsity, trit counts |

### Functions

| Function | Description |
|----------|-------------|
| `quantize_f32_to_ternary` | Deterministic threshold-based quantization |
| `dequantize_ternary_to_f32` | Expand trits back to f32 with scale |
| `stochastic_quantize` | Random rounding for unbiased quantization |
| `learn_threshold` | Gradient-descent threshold optimization |
| `per_channel_quantize` | Independent quantization per matrix row |
| `per_channel_dequantize` | Inverse of per-channel quantize |
| `mean_squared_error` | MSE between two f32 slices |
| `max_error` | Max absolute error between two f32 slices |
| `distribution_shift` | Mean difference: original − quantized |
| `trit_distribution` | Count of {neg, zero, pos} in a trit slice |
| `quantization_report` | Full error analysis in one call |

### `SimpleRng`

A deterministic xoshiro128** PRNG for reproducible stochastic quantization. No external dependencies.

## Quantization Theory

### Deterministic Threshold Quantization

For a weight w and threshold τ:

```
q(w) = { +1  if w > τ
        {  0  if |w| ≤ τ
        { -1  if w < -τ
```

Dequantization: `ŵ = q(w) × s` where s is a learned or computed scale factor.

### Stochastic Rounding

For w ∈ [0, s]:

```
P(q = +1) = w/s
P(q =  0) = 1 - w/s
```

This ensures `E[q × s] = w` — the quantization is **unbiased in expectation**.

### Per-Channel Scaling

For a weight matrix W ∈ ℝ^(m×n), each output channel i gets its own scale:

```
s_i = (1/|W_i|) Σ|w_ij|

q_ij = ternary(w_ij / s_i)
ŵ_ij = q_ij × s_i
```

This preserves the relative magnitude of each channel — critical for transformer attention layers where different heads have vastly different norms.

## Performance

All operations are pure-Rust, no SIMD, no GPU. For production inference:
- Quantization is a one-time cost (post-training)
- Dequantization can be fused into the kernel (ternary matmul doesn't need explicit dequant)
- Error metrics are for analysis, not hot-path code

## Testing

```bash
cargo test
```

32 tests covering: known-value quantization, round-trip error bounds, stochastic distribution correctness, learned threshold convergence, per-channel independence, error metric accuracy, RNG determinism and uniformity, full-pipeline integration tests.

## References

- Li, F., Zhang, B., & Liu, B. (2016). *Ternary Weight Networks*. arXiv:1605.04711
- Zhu, C., Han, S., Mao, H., & Dally, W. J. (2016). *Trained Ternary Quantization*. arXiv:1612.01064
- Courbariaux, M., et al. (2015). *BinaryConnect: Deep Neural Networks with Ternary Weights*

## License

Dual-licensed under MIT or Apache-2.0.
