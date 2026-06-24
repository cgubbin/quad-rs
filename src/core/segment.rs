use nalgebra::ComplexField;
use num_traits::Float;
use std::ops::Range;

use crate::IntegrationOutput;

/// Result of applying an integration rule to one interval.
///
/// A `Segment` stores the interval, the local integral estimate, the scalar
/// local error estimate, and optionally the quadrature samples used to produce
/// the estimate.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Segment<I, O, F> {
    /// Interval in the input domain.
    pub range: Range<I>,

    /// Estimated integral over `range`.
    pub result: O,

    /// Scalar local error estimate used by the adaptive controller.
    pub error: F,

    /// Optional quadrature samples used to compute `result`.
    pub samples: Option<QuadratureSamples<I, O>>,
}

/// Inner data for a segment, containing the resolved values.
///
/// This is useful for situations where we want both the integrated quantity, and
/// visibility over the integrand.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuadratureSamples<I, O> {
    /// Evaluation points in the input domain, ordered from left to right on the
    /// reference rule.
    pub points: Vec<I>,

    /// Quadrature weights mapped to the segment.
    ///
    /// These include the segment half-length. For complex contours, the weights
    /// are complex and encode the contour direction.
    pub weights: Vec<I>,

    /// Integrand values at `points`.
    pub values: Vec<O>,
}

impl<I, O> QuadratureSamples<I, O> {
    /// Constructs a complete set of Gauss–Kronrod quadrature samples for a segment.
    ///
    /// The inputs are assumed to originate from the symmetric evaluation of a
    /// Gauss–Kronrod rule on a single interval.
    ///
    /// # Assumptions
    ///
    /// The following invariants must hold:
    ///
    /// - `output_left.len() == output_right.len()`
    /// - `xgk.len() == bare_weights.len()`
    /// - `output_left.len() + 1 == xgk.len()`
    ///
    /// The values in `output_left` and `output_right` correspond to evaluations of
    /// the integrand at the symmetric Gauss–Kronrod abscissae
    ///
    /// ```text
    /// center - half_length * xgk[j]
    /// center + half_length * xgk[j]
    /// ```
    ///
    /// respectively, for `j = 0 .. xgk.len() - 2`.
    ///
    /// The final entry of `xgk` is assumed to be the central Kronrod node:
    ///
    /// ```text
    /// xgk[xgk.len() - 1] == 0
    /// ```
    ///
    /// and `output_center` is assumed to be the corresponding integrand value
    /// evaluated at
    ///
    /// ```text
    /// center
    /// ```
    ///
    /// The entries of `xgk` are expected to be ordered by decreasing magnitude:
    ///
    /// ```text
    /// xgk[0] > xgk[1] > ... > xgk[n - 2] > xgk[n - 1] == 0
    /// ```
    ///
    /// which is the ordering used internally by the integrator.
    ///
    /// The entries of `bare_weights` are expected to follow the same ordering as
    /// `xgk`, with the final weight corresponding to the central Kronrod node.
    ///
    /// # Behaviour
    ///
    /// The constructor reconstructs the full set of quadrature samples by combining
    /// the negative-side samples (`output_left`), the central sample
    /// (`output_center`), and the positive-side samples (`output_right`) into a
    /// single ordered representation.
    ///
    /// The resulting sample set contains
    ///
    /// ```text
    /// 2 * xgk.len() - 1
    /// ```
    ///
    /// points, weights and values.
    ///
    /// # Notes
    ///
    /// The supplied weights are assumed to be reference-interval
    /// Gauss–Kronrod weights defined on `[-1, 1]`. They are scaled by the segment
    /// mapping before being stored.
    pub(crate) fn from_gauss_kronrod_samples<F>(
        output_left: &[O],
        output_center: O,
        output_right: &[O],
        reference_weights: &[F],
        center: I,
        half_length: I,
        xgk: &[F],
    ) -> Self
    where
        I: ComplexField<RealField = F> + Copy,
        O: Clone,
        F: Float,
    {
        debug_assert_eq!(output_left.len(), output_right.len());
        debug_assert_eq!(output_left.len() + 1, xgk.len());
        debug_assert_eq!(xgk.len(), reference_weights.len());
        debug_assert!(xgk.last().is_some_and(|x| *x == F::zero()));

        let n = reference_weights.len();

        let mut points = Vec::with_capacity(2 * n - 1);
        let mut weights = Vec::with_capacity(2 * n - 1);
        let mut values = Vec::with_capacity(2 * n - 1);

        for j in 0..n - 1 {
            let abscissa = half_length.scale(xgk[j]);

            points.push(center - abscissa);
            weights.push(half_length * I::from_real(reference_weights[j]));
            values.push(output_left[j].clone());
        }

        points.push(center);
        weights.push(half_length * I::from_real(reference_weights[n - 1]));
        values.push(output_center);

        for j in (0..n - 1).rev() {
            let abscissa = half_length.scale(xgk[j]);

            points.push(center + abscissa);
            weights.push(half_length * I::from_real(reference_weights[j]));
            values.push(output_right[j].clone());
        }

        Self {
            points,
            weights,
            values,
        }
    }

    pub(crate) fn integral(&self) -> O
    where
        I: ComplexField + Copy,
        O: IntegrationOutput<Scalar = I>,
    {
        self.weights
            .iter()
            .zip(&self.values)
            .fold(O::default(), |acc, (&w, v)| acc.add(&v.mul(&w)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex;

    const TOL: f64 = 1e-12;

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < TOL,
            "expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    fn assert_complex_close(a: Complex<f64>, b: Complex<f64>) {
        assert_close(a.re, b.re);
        assert_close(a.im, b.im);
    }

    #[test]
    fn segment_samples_have_expected_lengths() {
        let output_left = vec![1.0, 2.0];
        let output_center = 3.0;
        let output_right = vec![4.0, 5.0];

        let xgk = vec![0.75, 0.25, 0.0];
        let weights = vec![0.2, 0.3, 0.4];

        let samples = QuadratureSamples::from_gauss_kronrod_samples(
            &output_left,
            output_center,
            &output_right,
            &weights,
            0.0,
            2.0,
            &xgk,
        );

        assert_eq!(samples.points.len(), 5);
        assert_eq!(samples.weights.len(), 5);
        assert_eq!(samples.values.len(), 5);
    }

    #[test]
    fn segment_samples_are_ordered_from_left_to_right() {
        let output_left = vec![1.0, 2.0];
        let output_center = 3.0;
        let output_right = vec![4.0, 5.0];

        let xgk = vec![0.75, 0.25, 0.0];
        let weights = vec![0.2, 0.3, 0.4];

        let samples = QuadratureSamples::from_gauss_kronrod_samples(
            &output_left,
            output_center,
            &output_right,
            &weights,
            10.0,
            2.0,
            &xgk,
        );

        let expected_points = vec![8.5, 9.5, 10.0, 10.5, 11.5];

        for (actual, expected) in samples.points.iter().zip(expected_points) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn segment_samples_preserve_value_order() {
        let output_left = vec![1.0, 2.0];
        let output_center = 3.0;
        let output_right = vec![4.0, 5.0];

        let xgk = vec![0.75, 0.25, 0.0];
        let weights = vec![0.2, 0.3, 0.4];

        let samples = QuadratureSamples::from_gauss_kronrod_samples(
            &output_left,
            output_center,
            &output_right,
            &weights,
            0.0,
            1.0,
            &xgk,
        );

        assert_eq!(samples.values, vec![1.0, 2.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn segment_samples_scale_real_weights_by_half_length() {
        let output_left = vec![1.0, 2.0];
        let output_center = 3.0;
        let output_right = vec![4.0, 5.0];

        let xgk = vec![0.75, 0.25, 0.0];
        let weights = vec![0.2, 0.3, 0.4];

        let samples = QuadratureSamples::from_gauss_kronrod_samples(
            &output_left,
            output_center,
            &output_right,
            &weights,
            0.0,
            2.0,
            &xgk,
        );

        let expected_weights = vec![0.4, 0.6, 0.8, 0.6, 0.4];

        for (actual, expected) in samples.weights.iter().zip(expected_weights) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn segment_samples_support_complex_contour_weights() {
        let output_left = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let output_center = Complex::new(3.0, 0.0);
        let output_right = vec![Complex::new(4.0, 0.0), Complex::new(5.0, 0.0)];

        let xgk = vec![0.75, 0.25, 0.0];
        let weights = vec![0.2, 0.3, 0.4];

        let center = Complex::new(1.0, 1.0);
        let half_length = Complex::new(0.0, 2.0);

        let samples = QuadratureSamples::from_gauss_kronrod_samples(
            &output_left,
            output_center,
            &output_right,
            &weights,
            center,
            half_length,
            &xgk,
        );

        assert_complex_close(samples.points[0], Complex::new(1.0, -0.5));
        assert_complex_close(samples.points[1], Complex::new(1.0, 0.5));
        assert_complex_close(samples.points[2], Complex::new(1.0, 1.0));
        assert_complex_close(samples.points[3], Complex::new(1.0, 1.5));
        assert_complex_close(samples.points[4], Complex::new(1.0, 2.5));

        assert_complex_close(samples.weights[0], Complex::new(0.0, 0.4));
        assert_complex_close(samples.weights[1], Complex::new(0.0, 0.6));
        assert_complex_close(samples.weights[2], Complex::new(0.0, 0.8));
        assert_complex_close(samples.weights[3], Complex::new(0.0, 0.6));
        assert_complex_close(samples.weights[4], Complex::new(0.0, 0.4));
    }
}
