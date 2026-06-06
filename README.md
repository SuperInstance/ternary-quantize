# ternary-quantize

FP32 → ternary {-1, 0, +1} quantization with error analysis, stochastic rounding, and per-channel scaling.

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

---

## Why this exists

A 7-billion parameter model stores ~28 GB of weights in FP32. Replace every float with a trit and you're down to ~1.7 GB — small enough to fit in L3 cache. The catch: you need to *actually compress* those weights without destroying model quality. That's what this crate does.

This implements the quantization algorithms from [Li et al. 2016 (TWN)](https://arxiv.org/abs/1605.04711) and [Zhu et al. 2016 (TTQ)](https://arxiv.org/abs/1612.01064): deterministic threshold quantization, stochastic rounding for unbiased compression, learned-threshold optimization, and per-channel scaling that respects the wildly different norms across transformer heads.

## The key insight

The zero in {-1, 0, +1} does something binary {-1, +1} can't: it gives you *sparsity for free*. A ternary weight matrix where 60% of values are zero means 60% of your multiplications disappear — they're not computed at all. The threshold τ controls the sparsity-accuracy tradeoff, and this crate gives you four different strategies for choosing it.

## Quick Start

```rust
use ternary_quantize::{quantize_f32_to_ternary, dequantize_ternary_to_f32, QuantizeConfig, Trit};

// Deterministic quantization: anything with |w| ≤ 0.05 becomes 0
let weights = vec![-0.8, -0.02, 0.0, 0.03, 0.9];
let config = QuantizeConfig::new(0.05, 1.0);

let trits = quantize_f32_to_ternary(&weights, &config);
assert_eq!(trits, vec![Trit::Neg, Trit::Zero, Trit::Zero, Trit::Zero, Trit::Pos]);

let deq = dequantize_ternary_to_f32(&trits, 1.0);
// [-1.0, 0.0, 0.0, 0.0, 1.0]
```

### Stochastic rounding — unbiased in expectation

Deterministic quantization introduces systematic bias: 0.3 always maps to 0, so you always lose the same signal. Stochastic rounding fixes this: 0.3 maps to +1 with probability 0.3, and to 0 with probability 0.7. Over many weights, the *expected* value is preserved exactly.

```rust
use ternary_quantize::{stochastic_quantize, SimpleRng};

let weights = vec![0.3; 10000];
let mut rng = SimpleRng::new(42);
let trits = stochastic_quantize(&weights, 1.0, &mut rng);
// ~3000 are +1, ~7000 are 0 — unbiased
```

### Learned threshold — minimize MSE for your data

The threshold τ isn't a hyperparameter you tune by hand. You optimize it:

```rust
use ternary_quantize::learn_threshold;

let weights: Vec<f32> = /* your layer weights */;
let (threshold, mse) = learn_threshold(&weights, 0.5, 1.0, 100, 0.01);
// threshold now minimizes reconstruction error for this specific weight distribution
```

### Per-channel scaling — different heads, different norms

In a transformer, attention head 0 might have weights around ±0.1 while head 7 has weights around ±5.0. A single global scale destroys one or the other. Per-channel quantization gives each output channel its own scale:

```rust
use ternary_quantize::per_channel_quantize;

let row1: &[f32] = &[0.1, -0.2, 0.3];
let row2: &[f32] = &[10.0, -5.0, 0.0];
let (trits, scales, thresholds) = per_channel_quantize(&[row1, row2]);
// scales[0] ≈ 0.2, scales[1] ≈ 5.0
```

## Architecture

```
quantize_f32_to_ternary()  ──→  [Trit]  ──→  dequantize_ternary_to_f32()
         │                                      │
    QuantizeConfig                          scale factor
    (threshold + scale)

stochastic_quantize()  ──→  [Trit]     learn_threshold()  ──→  optimal τ
         │                                      │
    SimpleRng (xoshiro128**)               gradient descent
                                          on MSE(τ)

per_channel_quantize()  ──→  ([Trit], scales, thresholds)
         │
    independent QuantizeConfig
    per matrix row
```

The module hierarchy is flat — all public API lives at the crate root. `Trit` is the core enum, `QuantizeConfig` pairs threshold with scale, and `QuantizationReport` bundles error analysis into one struct.

## API Reference

### Core Types

```rust
pub enum Trit { Neg, Zero, Pos }
// to_i8(), to_f32(), from_i8(i8) -> Option<Trit>

pub struct QuantizeConfig {
    pub threshold: f32,  // |x| ≤ threshold → Zero
    pub scale: f32,      // dequantization multiplier
}

pub struct QuantizationReport {
    pub mse: f32,
    pub max_err: f32,
    pub distribution_shift: f32,
    pub trit_counts: (usize, usize, usize),  // (neg, zero, pos)
    pub sparsity: f32,  // fraction of zeros
}
```

### Functions

| Signature | What it does |
|-----------|-------------|
| `quantize_f32_to_ternary(values: &[f32], config: &QuantizeConfig) -> Vec<Trit>` | Deterministic threshold quantization |
| `dequantize_ternary_to_f32(trits: &[Trit], scale: f32) -> Vec<f32>` | Expand trits back to scaled f32 |
| `stochastic_quantize(values: &[f32], scale: f32, rng: &mut SimpleRng) -> Vec<Trit>` | Unbiased random rounding |
| `learn_threshold(values: &[f32], init: f32, scale: f32, iters: usize, lr: f32) -> (f32, f32)` | Gradient-descent threshold optimization → (threshold, MSE) |
| `per_channel_quantize(matrix: &[&[f32]]) -> (Vec<Trit>, Vec<f32>, Vec<f32>)` | Independent quantization per row → (trits, scales, thresholds) |
| `per_channel_dequantize(trits: &[Trit], scales: &[f32], row_len: usize) -> Vec<f32>` | Inverse of per-channel quantize |
| `quantization_report(original: &[f32], trits: &[Trit], scale: f32) -> QuantizationReport` | Full error analysis in one call |
| `mean_squared_error(a: &[f32], b: &[f32]) -> f32` | MSE between two slices |
| `max_error(a: &[f32], b: &[f32]) -> f32` | Max absolute error |
| `distribution_shift(original: &[f32], quantized: &[f32]) -> f32` | Mean(original) − Mean(quantized) |
| `trit_distribution(trits: &[Trit]) -> (usize, usize, usize)` | Count of (neg, zero, pos) |

### `SimpleRng`

A deterministic xoshiro128\*\* PRNG. No external dependencies. Seed it once, get reproducible stochastic quantization across runs.

## Real-world example

You're deploying a BERT-base model (110M parameters) on an edge device with 2 GB RAM. FP32 weights need 440 MB. After ternary quantization with learned thresholds and per-channel scaling:

- **Compressed size**: 27.5 MB (16× reduction — 2 bits per trit vs 32 bits per float)
- **Sparsity**: ~55% zeros → over half the multiplications skip entirely
- **Accuracy loss**: <1% on GLUE benchmarks with learned thresholds (per TWN paper)
- **No GPU needed**: ternary matmul is sign operations and additions, not floating-point multiply

## Ecosystem connections

This crate is the foundation for the SuperInstance ternary ecosystem:

- **[`ternary-transformer`](https://github.com/SuperInstance/ternary-transformer)** — uses quantized weights for ℤ₃ attention layers
- **[`ternary-knn`](https://github.com/SuperInstance/ternary-knn)** — operates directly on quantized trit vectors
- **[`ternary-svm`](https://github.com/SuperInstance/ternary-svm)** — classifies in quantized feature space
- **[`ternary-shard-merge`](https://github.com/SuperInstance/ternary-shard-merge)** — merges quantized shards from distributed training
- **[`ternary-pipeline-parallel`](https://github.com/SuperInstance/ternary-pipeline-parallel)** — pipelines quantized layers across devices

## Performance characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `quantize_f32_to_ternary` | O(n) | One pass, branch-free |
| `stochastic_quantize` | O(n) | One pass + RNG per element |
| `learn_threshold` | O(n × iterations) | Re-quantizes each iteration |
| `per_channel_quantize` | O(n × m) | n rows, m cols per row |
| `quantization_report` | O(n) | Quantize + dequantize + scan |

All operations are pure Rust, no SIMD, no GPU. Quantization is a one-time post-training cost — you pay it once, then deploy the compressed model forever.

## Open questions

- **Mixed-precision layers**: Can we keep attention layers in FP32 while quantizing FFN layers to ternary? Where's the breakeven?
- **Learned scales**: Right now scale = mean(|W_i|). A learned per-channel scale (like TTQ) might close the accuracy gap further.
- **SIMD trit packing**: 16 trits fit in a single u32. A pack/unpack API could cut memory bandwidth by another 2×.
- **Gradient-aware quantization**: Straight-Through Estimator (STE) lets gradients flow through the quantization bottleneck during training, but this crate only handles the inference side.

## Testing

```bash
cargo test
```

32 tests: known-value quantization, round-trip error bounds, stochastic distribution correctness (±500 on 10K samples), learned threshold convergence, per-channel independence, error metric accuracy, RNG determinism and uniformity, full-pipeline integration.

## References

- Li, F., Zhang, B., & Liu, B. (2016). *Ternary Weight Networks*. arXiv:1605.04711
- Zhu, C., Han, S., Mao, H., & Dally, W. J. (2016). *Trained Ternary Quantization*. arXiv:1612.01064

## License

Dual-licensed under MIT or Apache-2.0.
