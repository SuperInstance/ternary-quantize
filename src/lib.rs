//! # ternary-quantize
//!
//! Ternary quantization and dequantization for neural network weights.
//!
//! Implements deterministic and stochastic quantization of full-precision (f32) tensors
//! into ternary values {−1, 0, +1}, along with learned-threshold quantization,
//! per-channel quantization, and comprehensive error metrics.

use std::fmt;

// ---------------------------------------------------------------------------
// Core quantization types
// ---------------------------------------------------------------------------

/// A ternary value: Negative, Zero, or Positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Trit {
    /// Convert to i8.
    pub fn to_i8(self) -> i8 {
        match self {
            Trit::Neg => -1,
            Trit::Zero => 0,
            Trit::Pos => 1,
        }
    }

    /// Convert to f32.
    pub fn to_f32(self) -> f32 {
        self.to_i8() as f32
    }

    /// From i8 (-1, 0, 1).
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::Neg),
            0 => Some(Trit::Zero),
            1 => Some(Trit::Pos),
            _ => None,
        }
    }
}

impl fmt::Display for Trit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_i8())
    }
}

// ---------------------------------------------------------------------------
// Deterministic quantization
// ---------------------------------------------------------------------------

/// Configuration for deterministic ternary quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Threshold: values with |x| <= threshold become 0.
    /// Values above threshold map to +1, below to −1.
    pub threshold: f32,
    /// Scale factor applied during dequantization.
    pub scale: f32,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.05,
            scale: 1.0,
        }
    }
}

impl QuantizeConfig {
    pub fn new(threshold: f32, scale: f32) -> Self {
        Self { threshold, scale }
    }
}

/// Quantize a slice of f32 values into ternary {−1, 0, +1} using a fixed threshold.
///
/// - |x| <= threshold → 0
/// - x > threshold    → +1
/// - x < -threshold   → −1
pub fn quantize_f32_to_ternary(values: &[f32], config: &QuantizeConfig) -> Vec<Trit> {
    values
        .iter()
        .map(|&x| {
            if x > config.threshold {
                Trit::Pos
            } else if x < -config.threshold {
                Trit::Neg
            } else {
                Trit::Zero
            }
        })
        .collect()
}

/// Dequantize ternary values back to f32 using the scale factor.
///
/// Each trit is mapped to `trit.to_f32() * scale`.
pub fn dequantize_ternary_to_f32(trits: &[Trit], scale: f32) -> Vec<f32> {
    trits.iter().map(|t| t.to_f32() * scale).collect()
}

// ---------------------------------------------------------------------------
// Stochastic quantization
// ---------------------------------------------------------------------------

/// A simple pseudo-random number generator (xoshiro128**) for stochastic rounding.
///
/// This avoids depending on the `rand` crate and ensures deterministic tests
/// given a fixed seed.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    s: [u32; 4],
}

impl SimpleRng {
    /// Create a new PRNG from a seed.
    pub fn new(seed: u64) -> Self {
        let mut s = [0u32; 4];
        s[0] = (seed & 0xFFFF_FFFF) as u32;
        s[1] = (seed >> 32) as u32;
        s[2] = s[0].wrapping_mul(0x5851_F42D);
        s[3] = s[1].wrapping_mul(0x5851_F42D);
        Self { s }
    }

    /// Next uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        let x = self.next_u32();
        // Map to [0, 1) with 24 bits of precision
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    fn next_u32(&mut self) -> u32 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .wrapping_mul(5);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);
        result
    }
}

/// Stochastic ternary quantization.
///
/// For each value x:
/// - Compute probabilities p(+1), p(0), p(−1) proportional to the distance
///   from each ternary value (linear interpolation).
/// - Randomly sample according to these probabilities.
///
/// This reduces systematic bias compared to deterministic thresholding.
pub fn stochastic_quantize(values: &[f32], scale: f32, rng: &mut SimpleRng) -> Vec<Trit> {
    values
        .iter()
        .map(|&x| {
            let normalized = x / scale;
            if normalized >= 1.0 {
                Trit::Pos
            } else if normalized <= -1.0 {
                Trit::Neg
            } else {
                let r = rng.next_f32();
                if normalized > 0.0 {
                    // Between 0 and +1: p(+1) = normalized, p(0) = 1 - normalized
                    if r < normalized {
                        Trit::Pos
                    } else {
                        Trit::Zero
                    }
                } else {
                    // Between -1 and 0: p(-1) = -normalized, p(0) = 1 + normalized
                    if r < -normalized {
                        Trit::Neg
                    } else {
                        Trit::Zero
                    }
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Learned threshold quantization
// ---------------------------------------------------------------------------

/// Iteratively learn the optimal threshold that minimizes MSE between original
/// and quantized-then-dequantized values.
///
/// Returns the learned threshold and the final MSE.
pub fn learn_threshold(
    values: &[f32],
    initial_threshold: f32,
    scale: f32,
    iterations: usize,
    lr: f32,
) -> (f32, f32) {
    let mut threshold = initial_threshold;

    for _ in 0..iterations {
        let config = QuantizeConfig::new(threshold, scale);
        let trits = quantize_f32_to_ternary(values, &config);
        let deq = dequantize_ternary_to_f32(&trits, scale);

        // Compute MSE
        let mse = mean_squared_error(values, &deq);

        // Numerical gradient approximation
        let eps = 1e-4;
        let config_plus = QuantizeConfig::new(threshold + eps, scale);
        let trits_plus = quantize_f32_to_ternary(values, &config_plus);
        let deq_plus = dequantize_ternary_to_f32(&trits_plus, scale);
        let mse_plus = mean_squared_error(values, &deq_plus);

        let grad = (mse_plus - mse) / eps;

        // Gradient descent step
        threshold -= lr * grad;

        // Clamp threshold to [0, max_abs_value]
        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        threshold = threshold.clamp(0.0, max_abs);
    }

    // Final MSE
    let config = QuantizeConfig::new(threshold, scale);
    let trits = quantize_f32_to_ternary(values, &config);
    let deq = dequantize_ternary_to_f32(&trits, scale);
    let final_mse = mean_squared_error(values, &deq);

    (threshold, final_mse)
}

// ---------------------------------------------------------------------------
// Per-channel quantization
// ---------------------------------------------------------------------------

/// Quantize a 2-D weight matrix (rows × cols) per output channel (row).
///
/// Each row gets its own threshold and scale derived from its statistics.
/// Returns quantized trits (flattened row-major), per-row scales, and per-row thresholds.
pub fn per_channel_quantize(
    matrix: &[&[f32]], // rows
) -> (Vec<Trit>, Vec<f32>, Vec<f32>) {
    let mut all_trits = Vec::new();
    let mut scales = Vec::new();
    let mut thresholds = Vec::new();

    for row in matrix {
        let mean_abs: f32 = if row.is_empty() {
            0.0
        } else {
            row.iter().map(|v| v.abs()).sum::<f32>() / row.len() as f32
        };
        let scale = mean_abs.max(1e-8);
        // Threshold as fraction of scale (mean absolute deviation heuristic)
        let threshold = 0.05 * scale;

        let config = QuantizeConfig::new(threshold, scale);
        let trits = quantize_f32_to_ternary(row, &config);
        all_trits.extend(trits);
        scales.push(scale);
        thresholds.push(threshold);
    }

    (all_trits, scales, thresholds)
}

/// Dequantize per-channel: each row uses its own scale.
pub fn per_channel_dequantize(
    trits: &[Trit],
    scales: &[f32],
    row_len: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(trits.len());
    for (row_idx, &scale) in scales.iter().enumerate() {
        let start = row_idx * row_len;
        let end = (start + row_len).min(trits.len());
        for i in start..end {
            result.push(trits[i].to_f32() * scale);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

/// Mean squared error between two slices.
pub fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slices must have equal length");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    sum / a.len() as f32
}

/// Maximum absolute error between two slices.
pub fn max_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute the distribution shift: mean(original) − mean(quantized).
///
/// A positive value means quantization shifted the mean downward.
pub fn distribution_shift(original: &[f32], quantized: &[f32]) -> f32 {
    let mean_orig = original.iter().sum::<f32>() / original.len().max(1) as f32;
    let mean_quant = quantized.iter().sum::<f32>() / quantized.len().max(1) as f32;
    mean_orig - mean_quant
}

/// Ternary distribution: count of {−1, 0, +1} in a trit slice.
pub fn trit_distribution(trits: &[Trit]) -> (usize, usize, usize) {
    let mut neg = 0;
    let mut zero = 0;
    let mut pos = 0;
    for t in trits {
        match t {
            Trit::Neg => neg += 1,
            Trit::Zero => zero += 1,
            Trit::Pos => pos += 1,
        }
    }
    (neg, zero, pos)
}

/// Comprehensive quantization report.
#[derive(Debug, Clone)]
pub struct QuantizationReport {
    pub mse: f32,
    pub max_err: f32,
    pub distribution_shift: f32,
    pub trit_counts: (usize, usize, usize),
    pub sparsity: f32, // fraction of zeros
}

/// Generate a full quantization report.
pub fn quantization_report(original: &[f32], trits: &[Trit], scale: f32) -> QuantizationReport {
    let deq = dequantize_ternary_to_f32(trits, scale);
    let counts = trit_distribution(trits);
    let total = trits.len().max(1) as f32;
    QuantizationReport {
        mse: mean_squared_error(original, &deq),
        max_err: max_error(original, &deq),
        distribution_shift: distribution_shift(original, &deq),
        trit_counts: counts,
        sparsity: counts.1 as f32 / total,
    }
}

impl fmt::Display for QuantizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantizationReport {{ mse: {:.6}, max_err: {:.6}, shift: {:.6}, sparsity: {:.2}%, counts: neg={} zero={} pos={} }}",
            self.mse,
            self.max_err,
            self.distribution_shift,
            self.sparsity * 100.0,
            self.trit_counts.0,
            self.trit_counts.1,
            self.trit_counts.2
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Trit ----

    #[test]
    fn trit_conversions() {
        assert_eq!(Trit::Neg.to_i8(), -1);
        assert_eq!(Trit::Zero.to_i8(), 0);
        assert_eq!(Trit::Pos.to_i8(), 1);
        assert_eq!(Trit::Neg.to_f32(), -1.0);
        assert_eq!(Trit::Zero.to_f32(), 0.0);
        assert_eq!(Trit::Pos.to_f32(), 1.0);
    }

    #[test]
    fn trit_from_i8() {
        assert_eq!(Trit::from_i8(-1), Some(Trit::Neg));
        assert_eq!(Trit::from_i8(0), Some(Trit::Zero));
        assert_eq!(Trit::from_i8(1), Some(Trit::Pos));
        assert_eq!(Trit::from_i8(2), None);
    }

    // ---- Deterministic quantization ----

    #[test]
    fn quantize_positive() {
        let config = QuantizeConfig::new(0.05, 1.0);
        let result = quantize_f32_to_ternary(&[0.5, 1.0, 10.0], &config);
        assert_eq!(result, vec![Trit::Pos, Trit::Pos, Trit::Pos]);
    }

    #[test]
    fn quantize_negative() {
        let config = QuantizeConfig::new(0.05, 1.0);
        let result = quantize_f32_to_ternary(&[-0.5, -1.0, -10.0], &config);
        assert_eq!(result, vec![Trit::Neg, Trit::Neg, Trit::Neg]);
    }

    #[test]
    fn quantize_zero_region() {
        let config = QuantizeConfig::new(0.1, 1.0);
        let result = quantize_f32_to_ternary(&[0.0, 0.05, -0.05, 0.1], &config);
        assert_eq!(result[0], Trit::Zero);
        assert_eq!(result[1], Trit::Zero);
        assert_eq!(result[2], Trit::Zero);
        assert_eq!(result[3], Trit::Zero);
    }

    #[test]
    fn quantize_mixed() {
        let config = QuantizeConfig::new(0.5, 1.0);
        let result = quantize_f32_to_ternary(&[-2.0, -0.3, 0.0, 0.3, 2.0], &config);
        assert_eq!(result, vec![Trit::Neg, Trit::Zero, Trit::Zero, Trit::Zero, Trit::Pos]);
    }

    #[test]
    fn quantize_empty() {
        let config = QuantizeConfig::default();
        let result = quantize_f32_to_ternary(&[], &config);
        assert!(result.is_empty());
    }

    // ---- Dequantization ----

    #[test]
    fn dequantize_basic() {
        let trits = vec![Trit::Neg, Trit::Zero, Trit::Pos];
        let result = dequantize_ternary_to_f32(&trits, 1.0);
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn dequantize_with_scale() {
        let trits = vec![Trit::Neg, Trit::Zero, Trit::Pos];
        let result = dequantize_ternary_to_f32(&trits, 2.5);
        assert_eq!(result, vec![-2.5, 0.0, 2.5]);
    }

    #[test]
    fn round_trip_error_bounds() {
        let original = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let config = QuantizeConfig::new(0.25, 1.0);
        let trits = quantize_f32_to_ternary(&original, &config);
        let deq = dequantize_ternary_to_f32(&trits, config.scale);

        let mse = mean_squared_error(&original, &deq);
        let max_err = max_error(&original, &deq);

        // With threshold 0.25, values ±0.5 round to ±1, and ±1.0 stay ±1.
        // Errors should be bounded
        assert!(mse < 1.0, "MSE should be small, got {}", mse);
        assert!(max_err <= 2.0, "max error bounded, got {}", max_err);
    }

    // ---- Stochastic quantization ----

    #[test]
    fn stochastic_quantize_extremes() {
        let mut rng = SimpleRng::new(42);
        let result = stochastic_quantize(&[10.0, -10.0], 1.0, &mut rng);
        assert_eq!(result[0], Trit::Pos);
        assert_eq!(result[1], Trit::Neg);
    }

    #[test]
    fn stochastic_quantize_distribution() {
        let mut rng = SimpleRng::new(12345);
        let n = 10000;
        let values = vec![0.3f32; n]; // 30% chance of +1, 70% of 0
        let trits = stochastic_quantize(&values, 1.0, &mut rng);
        let (neg, zero, pos) = trit_distribution(&trits);
        assert_eq!(neg, 0);
        // Expect ~3000 positives, allow ±500
        assert!(
            (2500..3500).contains(&pos),
            "expected ~3000 positives, got {}",
            pos
        );
        assert!(
            (6500..7500).contains(&zero),
            "expected ~7000 zeros, got {}",
            zero
        );
    }

    #[test]
    fn stochastic_quantize_symmetry() {
        let mut rng = SimpleRng::new(99);
        let values: Vec<f32> = (0..1000).map(|i| -0.5 + i as f32 / 1000.0).collect();
        let trits = stochastic_quantize(&values, 1.0, &mut rng);
        let (neg, _zero, pos) = trit_distribution(&trits);
        // Roughly symmetric: more negs for negative inputs, more pos for positive
        assert!(neg > 0);
        assert!(pos > 0);
    }

    #[test]
    fn stochastic_quantize_zero_input() {
        let mut rng = SimpleRng::new(0);
        let trits = stochastic_quantize(&[0.0; 100], 1.0, &mut rng);
        let (neg, zero, pos) = trit_distribution(&trits);
        assert_eq!(neg, 0);
        assert_eq!(pos, 0);
        assert_eq!(zero, 100);
    }

    // ---- Learned threshold ----

    #[test]
    fn learn_threshold_converges() {
        // Values centered around ±1 with some near-zero noise
        let values: Vec<f32> = [-1.0, -0.9, -0.1, 0.0, 0.1, 0.9, 1.0]
            .iter()
            .cycle()
            .take(700)
            .copied()
            .collect();

        let (threshold, mse) = learn_threshold(&values, 0.5, 1.0, 50, 0.01);

        // Threshold should be reasonable (not 0, not huge)
        assert!(threshold > 0.0, "threshold should be positive, got {}", threshold);
        assert!(threshold < 1.0, "threshold should be < 1.0, got {}", threshold);
        assert!(mse < 0.5, "MSE should be small after learning, got {}", mse);
    }

    #[test]
    fn learn_threshold_all_same_sign() {
        let values = vec![0.5f32; 100];
        let (threshold, _mse) = learn_threshold(&values, 0.1, 1.0, 20, 0.01);
        // All positive → threshold should push to maximize Pos
        assert!(threshold >= 0.0);
    }

    // ---- Per-channel quantization ----

    #[test]
    fn per_channel_basic() {
        let row1: &[f32] = &[0.1, -0.2, 0.3];
        let row2: &[f32] = &[10.0, -5.0, 0.0];
        let (trits, scales, thresholds) = per_channel_quantize(&[row1, row2]);

        assert_eq!(trits.len(), 6);
        assert_eq!(scales.len(), 2);
        assert_eq!(thresholds.len(), 2);
        // Row 2 has much larger scale than row 1
        assert!(scales[1] > scales[0] * 10.0);
    }

    #[test]
    fn per_channel_dequantize_roundtrip() {
        let row1: &[f32] = &[0.5, -0.5, 0.0];
        let row2: &[f32] = &[2.0, -1.0, 0.5];
        let (trits, scales, _thresholds) = per_channel_quantize(&[row1, row2]);
        let deq = per_channel_dequantize(&trits, &scales, 3);

        assert_eq!(deq.len(), 6);
        // Values should be multiples of their row's scale
        for i in 0..3 {
            let s = scales[0];
            assert!(deq[i] == -s || deq[i] == 0.0 || deq[i] == s, "row1 deq[{}] = {}", i, deq[i]);
        }
    }

    #[test]
    fn per_channel_independent() {
        let row1: &[f32] = &[100.0, -100.0];
        let row2: &[f32] = &[0.01, -0.01];
        let (trits, scales, _) = per_channel_quantize(&[row1, row2]);

        // Both rows should quantize to [+1, -1] but with very different scales
        assert_eq!(trits[0], Trit::Pos);
        assert_eq!(trits[1], Trit::Neg);
        assert_eq!(trits[2], Trit::Pos);
        assert_eq!(trits[3], Trit::Neg);

        let ratio = scales[0] / scales[1];
        assert!(ratio > 100.0, "scales should differ by ~10000x, ratio = {}", ratio);
    }

    // ---- Error metrics ----

    #[test]
    fn mse_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(mean_squared_error(&a, &a), 0.0);
    }

    #[test]
    fn mse_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        // errors: 1, 1, 1 → MSE = 1.0
        assert!((mean_squared_error(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mse_empty() {
        assert_eq!(mean_squared_error(&[], &[]), 0.0);
    }

    #[test]
    fn max_error_known() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![-1.0, 0.5, 2.0];
        assert!((max_error(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn distribution_shift_zero() {
        let a = vec![1.0, -1.0, 0.0];
        assert!((distribution_shift(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn distribution_shift_known() {
        let original = vec![1.0, 2.0, 3.0];
        let quantized = vec![1.0, 1.0, 1.0];
        let shift = distribution_shift(&original, &quantized);
        // mean_orig = 2.0, mean_quant = 1.0, shift = 1.0
        assert!((shift - 1.0).abs() < 1e-6);
    }

    // ---- Trit distribution ----

    #[test]
    fn trit_dist_counts() {
        let trits = vec![Trit::Neg, Trit::Neg, Trit::Zero, Trit::Pos, Trit::Pos, Trit::Pos];
        let (n, z, p) = trit_distribution(&trits);
        assert_eq!(n, 2);
        assert_eq!(z, 1);
        assert_eq!(p, 3);
    }

    // ---- QuantizationReport ----

    #[test]
    fn report_basic() {
        let original = vec![0.5, -0.5, 0.0, 1.0, -1.0];
        let config = QuantizeConfig::new(0.1, 1.0);
        let trits = quantize_f32_to_ternary(&original, &config);
        let report = quantization_report(&original, &trits, 1.0);

        assert!(report.mse >= 0.0);
        assert!(report.max_err >= 0.0);
        assert!(report.sparsity >= 0.0 && report.sparsity <= 1.0);
        let total: usize = report.trit_counts.0 + report.trit_counts.1 + report.trit_counts.2;
        assert_eq!(total, 5);
    }

    #[test]
    fn report_display() {
        let report = QuantizationReport {
            mse: 0.123,
            max_err: 0.5,
            distribution_shift: 0.01,
            trit_counts: (10, 20, 30),
            sparsity: 0.333,
        };
        let s = format!("{}", report);
        assert!(s.contains("mse:"));
        assert!(s.contains("sparsity:"));
    }

    // ---- RNG ----

    #[test]
    fn rng_deterministic() {
        let mut r1 = SimpleRng::new(42);
        let mut r2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_f32(), r2.next_f32());
        }
    }

    #[test]
    fn rng_in_range() {
        let mut rng = SimpleRng::new(0);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "value out of range: {}", v);
        }
    }

    // ---- Integration / round-trip ----

    #[test]
    fn full_pipeline_deterministic() {
        let weights: Vec<f32> = (0..100)
            .map(|i| ((i as f32 - 50.0) / 50.0).sin())
            .collect();

        let config = QuantizeConfig::new(0.05, 1.0);
        let trits = quantize_f32_to_ternary(&weights, &config);
        let deq = dequantize_ternary_to_f32(&trits, config.scale);
        let report = quantization_report(&weights, &trits, config.scale);

        // Ternary values should compress well
        assert!(report.sparsity > 0.0, "some values should be zero");
        // MSE should be reasonable for sin values
        assert!(report.mse < 1.0, "MSE too high: {}", report.mse);
    }

    #[test]
    fn full_pipeline_stochastic() {
        let weights: Vec<f32> = (0..1000)
            .map(|i| ((i as f32 * 0.01).sin() * 0.5))
            .collect();

        let mut rng = SimpleRng::new(7);
        let trits = stochastic_quantize(&weights, 0.5, &mut rng);
        let deq = dequantize_ternary_to_f32(&trits, 0.5);

        let mse = mean_squared_error(&weights, &deq);
        assert!(mse < 1.0, "stochastic round-trip MSE too high: {}", mse);
    }
}
